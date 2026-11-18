"""Stopping a battery cancels the runs it started, not only its client.

The client is an ordinary HTTP client and its runs are the server's, in their
own process groups: killing the client's group ends none of them. These tests
stand in a sleeping process for the client, register the runs its ledger names
as runs the server holds, and check what a stop does to each.

Run with: cd server && ./venv/bin/pytest tests/test_benchmark_stop.py
"""

from __future__ import annotations

import os
import secrets
import signal
import subprocess
import sys
import threading
import time

os.environ.setdefault("API_TOKEN", "test-token")

import pytest
from fastapi.testclient import TestClient

import benchmark_client
import benchmark_jobs
import main
from config import settings
from wire import runs

client = TestClient(main.app)


def _new_id() -> str:
    return secrets.token_urlsafe(24)


def _sleeper() -> subprocess.Popen:
    """A process group leader that does nothing, the way a client or a tool is."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                            start_new_session=True)


def _until(condition, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


@pytest.fixture
def admin(monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_TOKEN", "operator-secret")
    return {"X-Admin-Token": "operator-secret"}


@pytest.fixture
def registered():
    """Register runs on demand; discard them whatever the test does."""
    made = []

    def make(phase=None) -> str:
        run_id = runs.register(_new_id())
        made.append(run_id)
        if phase is not None:
            runs.append(run_id, phase)
        return run_id

    yield make
    for run_id in made:
        runs.discard(run_id)


@pytest.fixture
def battery(monkeypatch, tmp_path):
    """A running battery whose client is a sleeping process, and its ledger."""
    processes = []

    def make(lines) -> dict:
        process = _sleeper()
        processes.append(process)
        ledger = tmp_path / "battery.runs"
        ledger.write_text("".join(f"{kind} {run_id}\n" for kind, run_id in lines))
        job = {"preset": "fake", "label": None, "total_runs": len(lines),
               "summary": "preset-fake.json", "started_at": time.time(),
               "state": "running", "exit_code": None, "cancelled_runs": None,
               "_process": process, "_plan_path": str(tmp_path / "plan.json"),
               "_runs_path": str(ledger)}
        monkeypatch.setattr(benchmark_jobs, "_current", job)
        return job

    yield make
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_a_stop_cancels_every_run_still_in_flight_and_leaves_finished_ones_alone(
        admin, registered, battery):
    running = registered(runs.PHASE_RUNNING)
    queued = registered(runs.PHASE_QUEUED_GPU)
    finished = registered(runs.PHASE_DONE)
    answered = registered(runs.PHASE_RUNNING)
    gone = _new_id()  # finished and discarded with its request
    # The running one has a tool process: the stop must signal it, not only
    # mark it, since that is what frees the card now rather than at the next poll.
    tool = _sleeper()
    runs.set_pgid(running, tool.pid)
    job = battery([("start", running), ("start", queued), ("start", finished),
                   ("start", answered), ("end", answered), ("start", gone)])
    try:
        response = client.delete("/benchmark/run", headers=admin)
        assert response.status_code == 200, response.text
        assert response.json() == {"stopped": True, "cancelled_runs": 2}

        assert runs.is_cancelled(running)
        assert runs.is_cancelled(queued)
        assert not runs.is_cancelled(finished)
        assert not runs.is_cancelled(answered)
        assert _until(lambda: tool.poll() is not None), "the run's process group was not signalled"
        assert job["_process"].poll() is not None, "the client is still alive"
        assert not os.path.exists(job["_runs_path"])
        assert benchmark_jobs.current()["cancelled_runs"] == 2
    finally:
        if tool.poll() is None:
            os.killpg(tool.pid, signal.SIGKILL)
        tool.wait()


def test_a_second_stop_finds_nothing_running(admin, battery):
    battery([])
    assert client.delete("/benchmark/run", headers=admin).json() == {
        "stopped": True, "cancelled_runs": 0}
    assert client.delete("/benchmark/run", headers=admin).json() == {
        "stopped": False, "cancelled_runs": 0}


def test_a_run_that_registers_after_the_stop_is_cancelled_when_it_arrives(
        admin, battery, monkeypatch):
    # The client was killed with its POST already sent: the server registers
    # the run a moment after the stop has read the ledger.
    monkeypatch.setattr(benchmark_jobs, "LATE_SWEEP_SECONDS", 5.0)
    late = _new_id()
    battery([("start", late)])
    try:
        assert client.delete("/benchmark/run", headers=admin).json()["cancelled_runs"] == 0
        runs.register(late)
        assert _until(lambda: runs.is_cancelled(late)), "a late run slipped through"
    finally:
        runs.discard(late)


def test_a_client_killed_from_outside_still_has_its_runs_cancelled(registered, battery, tmp_path):
    run = registered(runs.PHASE_RUNNING)
    job = battery([("start", run)])
    benchmark_jobs._watch(job, job["_process"], job["_plan_path"], str(tmp_path / "inputs"))
    os.killpg(job["_process"].pid, signal.SIGKILL)
    assert _until(lambda: runs.is_cancelled(run))
    assert _until(lambda: job["cancelled_runs"] == 1)


def test_the_client_records_a_run_before_posting_it_and_its_end_after(monkeypatch, tmp_path):
    ledger = tmp_path / "battery.runs"
    monkeypatch.setattr(benchmark_client, "_ledger", str(ledger))
    monkeypatch.setattr(benchmark_client, "_watch", lambda *args: None)
    seen = {}

    def fake_request(url, token, data=None, method=None, timeout=None, headers=None):
        seen["ledger_at_post"] = ledger.read_text()
        seen["run_id"] = headers[benchmark_client.RUN_ID_HEADER]
        return 200, b"ok", {}

    monkeypatch.setattr(benchmark_client, "_request", fake_request)
    results = []
    benchmark_client._one_run("http://server", "token",
                              {"index": 0, "tool": "Tool", "params": {}},
                              0, time.time(), results, threading.Lock())
    run_id = seen["run_id"]
    assert seen["ledger_at_post"] == f"start {run_id}\n"
    assert ledger.read_text() == f"start {run_id}\nend {run_id}\n"
    assert benchmark_jobs._recorded(str(ledger)) == []


def test_a_post_that_got_no_answer_stays_on_the_ledger(monkeypatch, tmp_path):
    ledger = tmp_path / "battery.runs"
    monkeypatch.setattr(benchmark_client, "_ledger", str(ledger))
    monkeypatch.setattr(benchmark_client, "_watch", lambda *args: None)

    def timed_out(*args, **kwargs):
        raise TimeoutError()

    monkeypatch.setattr(benchmark_client, "_request", timed_out)
    results = []
    benchmark_client._one_run("http://server", "token",
                              {"index": 0, "tool": "Tool", "params": {}},
                              0, time.time(), results, threading.Lock())
    assert benchmark_jobs._recorded(str(ledger)) == [results[0]["run_id"]]
