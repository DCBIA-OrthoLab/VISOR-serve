"""Starting a preset battery, and knowing where it got to.

The battery itself runs in `benchmark_client.py`, in a process of its own. This
is the half that lives in the server: it writes the plan, spawns that process,
keeps track of the one job at a time this server will run, and puts the summary
where `/benchmarks/view` already looks.

Three decisions, and each is a thing that would go wrong without it.

**One battery at a time, server-wide.** Two batteries would measure each other:
the second one's runs would queue behind the first's, and both would report
admission delays that came from the benchmark rather than from the deployment.
A second request is refused with what is already running, not queued.

**The summary lands in `SADT_BENCHMARK_DIR`.** That is the directory the
campaign viewer reads, so a battery pressed from the launcher appears in the
same page, with the same Gantt, as a campaign run from `benchmarks/`. Named
`preset-*` so the two are never confused for each other -- a preset measures
dispatch and admission, a campaign measures the protocol too, and reading one
as the other is the mistake this naming exists to prevent.

**A stop cancels the runs, not only the client.** The client is an ordinary
HTTP client, and the tool runs it asked for are the SERVER's: they live in
their own process groups, and a client that goes away never stops a run. So
killing the client's process group, which is all a stop used to do, left every
run it had started holding the card to the end. The client now appends each
run id to a ledger BEFORE posting the run; a stop kills the client first, so
nothing more can be posted, then cancels every recorded run that has not
finished through `dispatch.cancel_run` -- exactly what `DELETE /runs/{id}`
does. A queued run notices the marker while it waits for admission; a nested
call dies with its root's process group. The same settlement runs whenever
the client exits, however it exits, so a battery that crashed or was
SIGKILLed from outside leaves nothing running either.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Optional

from execution import dispatch
from wire import runs

logger = logging.getLogger("inference_server.benchmark")

# How long a battery may run before it is stopped. Generous -- `each-tool-solo`
# over twenty tools legitimately takes half an hour -- and finite, because a
# forgotten battery holding the card is worse than a truncated measurement.
MAX_SECONDS = 2 * 3600
_KILL_GRACE_SECONDS = 10.0
# How long a stop keeps looking for a recorded run the server does not know
# yet: a POST the client had already sent when it was killed is still on its
# way in, and registers within milliseconds of arriving.
LATE_SWEEP_SECONDS = 10.0
_LATE_SWEEP_POLL = 0.25

_lock = threading.Lock()
_current: Optional[dict] = None
# Serialises settling a battery's runs: a stop and the watcher both settle,
# and only the first one does the work.
_settle_lock = threading.Lock()


class BatteryError(RuntimeError):
    """A battery that cannot be started, with the reason for a caller."""


def _summary_name(preset_id: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    safe = "".join(c for c in preset_id if c.isalnum() or c in "-_")[:40]
    return f"preset-{safe}-{stamp}.json"


def current() -> Optional[dict]:
    """What is running, or the last thing that ran. Never the plan itself."""
    with _lock:
        if _current is None:
            return None
        job = dict(_current)
    process = job.get("_process")
    job = {key: value for key, value in job.items() if not key.startswith("_")}
    if process is not None and job.get("state") == "running":
        code = process.poll()
        if code is not None:
            job["state"] = "done" if code == 0 else "failed"
            job["exit_code"] = code
    job["elapsed"] = round(time.time() - job["started_at"], 1)
    return job


def _finished(job: dict) -> bool:
    process = job.get("_process")
    return process is None or process.poll() is not None


def start(plan: dict, summary_dir: str, base_url: str, token: str,
          scratch_dir: str) -> dict:
    """Spawn the battery. Raises `BatteryError` if one is already running."""
    global _current
    with _lock:
        if _current is not None and not _finished(_current):
            raise BatteryError(
                f"A battery is already running ({_current['preset']}, started "
                f"{int(time.time() - _current['started_at'])}s ago). Two at "
                "once would measure each other."
            )

        os.makedirs(summary_dir, exist_ok=True)
        os.makedirs(scratch_dir, exist_ok=True)
        name = _summary_name(plan["preset"])
        out_path = os.path.join(summary_dir, name)
        plan_path = os.path.join(scratch_dir, name.replace(".json", ".plan.json"))
        # The ledger of run ids the client starts (see the module docstring).
        # Created empty here so a stop always finds a file to read.
        runs_path = os.path.join(scratch_dir, name.replace(".json", ".runs"))
        with open(runs_path, "w", encoding="utf-8"):
            pass
        # Where the battery packs a bench FOLDER into the archive it uploads.
        # One per battery, removed with the plan whatever way the battery ends:
        # what lands in it is a copy of a clinical case.
        plan = dict(plan, scratch=os.path.join(scratch_dir, name.replace(".json", "-inputs")))
        with open(plan_path, "w", encoding="utf-8") as handle:
            json.dump(plan, handle)

        environment = dict(os.environ)
        environment["API_TOKEN"] = token
        client = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "benchmark_client.py")
        process = subprocess.Popen(
            [sys.executable, client, "--plan", plan_path,
             "--out", out_path, "--base", base_url, "--runs", runs_path],
            env=environment,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            # Its own process group, so stopping the battery stops the whole
            # tree rather than orphaning whatever it started.
            start_new_session=True,
        )
        _current = {
            "preset": plan["preset"],
            "label": plan.get("label"),
            "total_runs": plan.get("total_runs"),
            "summary": name,
            "started_at": time.time(),
            "state": "running",
            "exit_code": None,
            "cancelled_runs": None,
            "_process": process,
            "_plan_path": plan_path,
            "_runs_path": runs_path,
        }
        logger.info("benchmark battery=%s runs=%d pid=%d",
                    plan["preset"], plan.get("total_runs", 0), process.pid)
        job = _current

    _watch(job, process, plan_path, plan["scratch"])
    return {key: value for key, value in job.items() if not key.startswith("_")}


def _watch(job: dict, process: subprocess.Popen, plan_path: str,
           inputs_dir: str) -> None:
    """Enforce the ceiling, and clean up after the child, off the event loop."""
    def wait() -> None:
        try:
            process.wait(timeout=MAX_SECONDS)
        except subprocess.TimeoutExpired:
            logger.warning("benchmark battery exceeded %ds; stopping it", MAX_SECONDS)
            stop()
        finally:
            # Whatever way the client ended -- finished, crashed, killed from
            # outside -- nothing it started may outlive it.
            try:
                _settle(job)
            except Exception:  # noqa: BLE001 - the cleanup below must still run
                logger.exception("could not settle a battery's runs")
            try:
                os.remove(plan_path)
            except OSError:
                pass
            shutil.rmtree(inputs_dir, ignore_errors=True)
    threading.Thread(target=wait, daemon=True).start()


def _recorded(path: str) -> list:
    """The ids the client started and never saw answered, in order."""
    started, ended = [], set()
    try:
        with open(path, encoding="ascii", errors="replace") as handle:
            for line in handle:
                kind, _, run_id = line.strip().partition(" ")
                if kind == "start" and run_id and run_id not in started:
                    started.append(run_id)
                elif kind == "end":
                    ended.add(run_id)
    except OSError:
        return []
    return [run_id for run_id in started if run_id not in ended]


def _cancel(run_id: str) -> Optional[bool]:
    """Cancel one recorded run unless it is over.

    True if it was cancelled, False if it was left alone (finished, or an id
    the registry refuses), None if the server does not know it -- gone with
    its finished request, or not arrived yet.
    """
    try:
        state = runs.snapshot(run_id)["state"]
    except runs.RunError as exc:
        return None if exc.status_code == 404 else False
    if state in runs.TERMINAL_STATES:
        return False
    try:
        dispatch.cancel_run(run_id)
    except runs.RunError as exc:
        return None if exc.status_code == 404 else False
    return True


def _sweep_late(pending: list) -> None:
    """Cancel recorded runs that register only after the stop, for a while."""
    deadline = time.time() + LATE_SWEEP_SECONDS
    while pending and time.time() < deadline:
        time.sleep(_LATE_SWEEP_POLL)
        for run_id in list(pending):
            outcome = _cancel(run_id)
            if outcome is not None:
                pending.remove(run_id)
                if outcome:
                    logger.info("benchmark battery: cancelled a run that arrived after the stop")


def _settle(job: dict) -> int:
    """Cancel every run the battery recorded that is still in flight; once.

    Only ever called once the client is dead, so the ledger can no longer
    grow. Returns how many runs were cancelled.
    """
    with _settle_lock:
        if job.get("cancelled_runs") is not None:
            return job["cancelled_runs"]
        path = job.get("_runs_path")
        cancelled, unknown = 0, []
        for run_id in _recorded(path) if path else []:
            outcome = _cancel(run_id)
            if outcome:
                cancelled += 1
            elif outcome is None:
                unknown.append(run_id)
        if path:
            try:
                os.remove(path)
            except OSError:
                pass
        job["cancelled_runs"] = cancelled
    if cancelled:
        logger.info("benchmark battery=%s cancelled %d run(s) still in flight",
                    job.get("preset"), cancelled)
    if unknown:
        threading.Thread(target=_sweep_late, args=(unknown,), daemon=True).start()
    return cancelled


def stop() -> dict:
    """Stop the running battery and every run it started that is in flight.

    The client is killed FIRST, with its whole process group, and waited for:
    once it is dead the ledger cannot grow, so no run can be posted after the
    cancellation has read it. `{"stopped": False, ...}` if nothing was running.
    """
    with _lock:
        job = _current
        process = job.get("_process") if job else None
        if process is None or process.poll() is not None:
            return {"stopped": False, "cancelled_runs": 0}
        pid = process.pid
    try:
        group = os.getpgid(pid)
    except OSError:
        group = None
    # The same guard dispatch.py carries: killpg(0) would signal this server.
    if group is not None and group > 1:
        try:
            os.killpg(group, signal.SIGTERM)
        except OSError:
            pass
    try:
        process.wait(timeout=_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        if group is not None and group > 1:
            try:
                os.killpg(group, signal.SIGKILL)
            except OSError:
                pass
        try:
            process.wait(timeout=_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            logger.error("benchmark battery did not die on SIGKILL; its runs are cancelled anyway")
    return {"stopped": True, "cancelled_runs": _settle(job)}
