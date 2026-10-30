"""One workstation cannot fill the queue for every other one.

Every workstation holds the same API token, so admission alone cannot tell
"six clinicians at once" from "one client sending sixty": both are sixty runs
waiting for the card, and the five other workstations wait behind them. A run
in flight -- registered and not yet ended -- counts against the address that
sent it, past `MAX_RUNS_PER_CLIENT` the next one is a 429, and a paused run
counts for nothing: it waits for a person and holds no machine.

The addresses are documentation addresses (RFC 5737).
"""

import io
import os
import secrets
import threading
import time

import pytest
from fastapi.testclient import TestClient

import benchmark_client
import benchmark_jobs
import registry
from base import ArgSpec, Tool
from config import settings
from main import app
from wire import runs

AUTH = {"Authorization": f"Bearer {settings.API_TOKEN}"}
FIRST, SECOND = "192.0.2.10", "192.0.2.20"


def _client(address):
    return TestClient(app, client=(address, 50000))


def _new_id() -> str:
    return secrets.token_urlsafe(24)


@pytest.fixture(autouse=True)
def _own_registry(tmp_path, monkeypatch):
    """A registry of this test's own, holding nothing but what it puts there."""
    monkeypatch.setattr(settings, "TEMP_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "MAX_RUNS_PER_CLIENT", 2)


def _in_flight(address, count=2, phase=runs.PHASE_RUNNING):
    made = []
    for _ in range(count):
        identifier = runs.register(_new_id(), tool="AMASSS", client=address)
        runs.append(identifier, phase)
        made.append(identifier)
    return made


def _post(address, run_id=None, **kwargs):
    headers = dict(AUTH, **({"X-Run-Id": run_id} if run_id else {}))
    headers.update(kwargs.pop("headers", {}))
    return _client(address).post("/run/Test_Tool", headers=headers,
                                 data={"text_1": "a", "text_2": "b"}, **kwargs)


def test_the_run_past_the_cap_is_refused_and_nothing_is_staged(tmp_path):
    _in_flight(FIRST)
    refused = _new_id()
    before = set(os.listdir(tmp_path))

    response = _post(FIRST, refused, files={"text_3": ("x.nii.gz", io.BytesIO(b"scan"))})

    assert response.status_code == 429
    assert "2 run(s) in flight" in response.json()["detail"]
    # Refused before it was registered: nothing to watch, nothing on disk.
    assert _client(FIRST).get(f"/runs/{refused}", headers=AUTH).status_code == 404
    assert set(os.listdir(tmp_path)) == before


def test_another_workstation_is_not_held_by_the_first_one():
    _in_flight(FIRST)
    assert _post(SECOND, _new_id()).status_code == 200


@pytest.mark.parametrize("ending", [runs.PHASE_DONE, runs.PHASE_FAILED, runs.PHASE_CANCELLED])
def test_a_run_that_ended_frees_its_slot(ending):
    first, _second = _in_flight(FIRST)
    assert _post(FIRST, _new_id()).status_code == 429
    runs.finish(first, ending)
    assert _post(FIRST, _new_id()).status_code == 200


def test_a_run_waiting_for_the_card_still_counts():
    _in_flight(FIRST, phase=runs.PHASE_QUEUED_GPU)
    assert _post(FIRST, _new_id()).status_code == 429


def test_a_paused_run_does_not_count(tmp_path):
    first, _second = _in_flight(FIRST)
    job = tmp_path / "job"
    job.mkdir()
    runs.pause(first, str(job), "ASO")
    assert _post(FIRST, _new_id()).status_code == 200


def test_zero_is_no_cap(monkeypatch):
    monkeypatch.setattr(settings, "MAX_RUNS_PER_CLIENT", 0)
    _in_flight(FIRST, count=5)
    assert _post(FIRST, _new_id()).status_code == 200


def test_a_run_sent_without_an_id_counts_while_it_runs(monkeypatch):
    """No `X-Run-Id`, no registry entry: counted in this process instead, for
    exactly as long as the request lasts."""
    monkeypatch.setattr(settings, "MAX_RUNS_PER_CLIENT", 1)
    started, release = threading.Event(), threading.Event()

    class _Holding(Tool):
        name = "Holding_Probe"
        arguments = {"text": ArgSpec(type=str)}
        output_kind = "text"

        def run(self, text):
            started.set()
            # Bounded, so a refusal that never comes cannot hang the suite.
            release.wait(timeout=20)
            return text

    monkeypatch.setitem(registry.TOOLS, _Holding.name, _Holding())
    answers = []
    with TestClient(app, client=(FIRST, 50000)) as shared:
        holder = threading.Thread(target=lambda: answers.append(shared.post(
            f"/run/{_Holding.name}", headers=AUTH, data={"text": "x"})))
        holder.start()
        try:
            assert started.wait(timeout=20)
            refused = shared.post("/run/Test_Tool", headers=AUTH,
                                  data={"text_1": "a", "text_2": "b"})
        finally:
            release.set()
            holder.join(timeout=20)
        assert refused.status_code == 429
        assert answers and answers[0].status_code == 200
        # And the slot is free again once that request has ended.
        assert shared.post("/run/Test_Tool", headers=AUTH,
                           data={"text_1": "a", "text_2": "b"}).status_code == 200


class _Running:
    def poll(self):
        return None


def test_a_battery_launched_by_an_operator_is_not_capped(monkeypatch):
    """A benchmark battery runs from this server's own loopback, several runs
    at a time, because measuring concurrency is what it is for."""
    key = secrets.token_urlsafe(32)
    monkeypatch.setattr(benchmark_jobs, "_current", {"_key": key, "_process": _Running()})
    _in_flight(FIRST)
    battery = {benchmark_jobs.KEY_HEADER: key}
    assert _post(FIRST, _new_id(), headers=battery).status_code == 200
    # A key of its own making buys nothing.
    assert _post(FIRST, _new_id(), headers={benchmark_jobs.KEY_HEADER: "guessed"}).status_code == 429


def test_a_finished_battery_s_key_buys_nothing(monkeypatch):
    class _Ended:
        def poll(self):
            return 0

    key = secrets.token_urlsafe(32)
    monkeypatch.setattr(benchmark_jobs, "_current", {"_key": key, "_process": _Ended()})
    _in_flight(FIRST)
    assert _post(FIRST, _new_id(), headers={benchmark_jobs.KEY_HEADER: key}).status_code == 429


def test_the_battery_and_the_server_agree_on_the_key_they_share():
    """Two files that cannot import each other: the battery runs as a script."""
    assert benchmark_client.BENCHMARK_KEY_HEADER == benchmark_jobs.KEY_HEADER
    assert benchmark_client.BENCHMARK_KEY_ENV == benchmark_jobs.KEY_ENV


def test_the_count_is_taken_off_the_event_loop(monkeypatch):
    """Counting lists every run on disk; on the event loop that held every
    other request -- uploads, the event streams, /status -- for as long."""
    import main

    loop_thread, listed_on = [], []
    real_run_tool, real_active = main._run_tool, runs.active

    async def run_tool(*args, **kwargs):
        loop_thread.append(threading.get_ident())
        return await real_run_tool(*args, **kwargs)

    def active(*args, **kwargs):
        listed_on.append(threading.get_ident())
        return real_active(*args, **kwargs)

    monkeypatch.setattr(main, "_run_tool", run_tool)
    monkeypatch.setattr(runs, "active", active)
    assert _post(FIRST, _new_id()).status_code == 200
    assert loop_thread and listed_on
    assert [thread for thread in listed_on if thread in loop_thread] == []


def test_two_runs_counted_at_the_same_instant_cannot_both_pass_the_cap(monkeypatch):
    """The count and the registration are one step: two requests arriving
    together must not both count the same free slot and both register."""
    import main

    monkeypatch.setattr(settings, "MAX_RUNS_PER_CLIENT", 1)
    together = threading.Barrier(2)
    real_count = main._runs_in_flight

    def counted_together(address):
        # Holds each count back until the other has counted too, between the
        # count and the registration -- so two counts that CAN overlap both
        # see the slot free. Times out, and goes on, when the second is held
        # back, as it must be.
        counted = real_count(address)
        try:
            together.wait(timeout=1)
        except threading.BrokenBarrierError:
            pass
        return counted

    release = threading.Event()

    class _Holding(Tool):
        name = "Holding_Race_Probe"
        arguments = {"text": ArgSpec(type=str)}
        output_kind = "text"

        def run(self, text):
            release.wait(timeout=20)
            return text

    monkeypatch.setattr(main, "_runs_in_flight", counted_together)
    monkeypatch.setitem(registry.TOOLS, _Holding.name, _Holding())
    answers = []
    with TestClient(app, client=(FIRST, 50000)) as shared:
        def post():
            answers.append(shared.post(f"/run/{_Holding.name}",
                                       headers=dict(AUTH, **{"X-Run-Id": _new_id()}),
                                       data={"text": "x"}).status_code)

        senders = [threading.Thread(target=post) for _ in range(2)]
        for sender in senders:
            sender.start()
        deadline = time.monotonic() + 10
        while not answers and time.monotonic() < deadline:
            time.sleep(0.02)
        release.set()
        for sender in senders:
            sender.join(timeout=20)
    assert sorted(answers) == [200, 429]


@pytest.mark.parametrize("with_id", [False, True], ids=["unwatched", "watched"])
def test_a_run_cancelled_while_it_claims_its_slot_keeps_none(monkeypatch, with_id):
    """A client that hangs up while its count is being taken: the claim runs on
    in its thread, so it must give the slot back rather than hold it forever."""
    import anyio
    from fastapi import BackgroundTasks
    from starlette.requests import Request

    import main

    monkeypatch.setattr(settings, "MAX_RUNS_PER_CLIENT", 1)
    address = "192.0.2.30"
    real_claim = main._claimed_run

    def slow_claim(*args, **kwargs):
        time.sleep(0.3)
        return real_claim(*args, **kwargs)

    monkeypatch.setattr(main, "_claimed_run", slow_claim)
    headers = [(b"authorization", AUTH["Authorization"].encode())]
    if with_id:
        headers.append((b"x-run-id", _new_id().encode()))
    scope = {"type": "http", "method": "POST", "path": "/run/Test_Tool", "headers": headers,
             "query_string": b"", "client": (address, 50000), "server": ("testserver", 80),
             "scheme": "http", "app": app}

    async def never_sends():
        await anyio.sleep(30)
        return {"type": "http.disconnect"}

    async def hang_up_early():
        with anyio.move_on_after(0.1):
            await main.run_tool("Test_Tool", Request(scope, never_sends), BackgroundTasks())
        await anyio.sleep(0.5)  # the shielded claim finishing in its thread

    anyio.run(hang_up_early)
    assert main._unwatched_runs.get(address) is None
    assert main._runs_in_flight(address) == 0
