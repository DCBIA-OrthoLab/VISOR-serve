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

**The child is killed with its process group.** A battery is a tree -- the
client, and whatever the server spawned on its behalf -- and a stop that
signalled only the client would leave the runs going with nobody watching.
`start_new_session` plus `killpg` is the same mechanism `dispatch.py` uses to
cancel a run, for the same reason.
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

logger = logging.getLogger("inference_server.benchmark")

# How long a battery may run before it is stopped. Generous -- `each-tool-solo`
# over twenty tools legitimately takes half an hour -- and finite, because a
# forgotten battery holding the card is worse than a truncated measurement.
MAX_SECONDS = 2 * 3600
_KILL_GRACE_SECONDS = 10.0

_lock = threading.Lock()
_current: Optional[dict] = None


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
    process = job.pop("_process", None)
    job.pop("_plan_path", None)
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
             "--out", out_path, "--base", base_url],
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
            "_process": process,
            "_plan_path": plan_path,
        }
        logger.info("benchmark battery=%s runs=%d pid=%d",
                    plan["preset"], plan.get("total_runs", 0), process.pid)
        job = dict(_current)

    _watch(process, plan_path, plan["scratch"])
    job.pop("_process", None)
    job.pop("_plan_path", None)
    return job


def _watch(process: subprocess.Popen, plan_path: str, inputs_dir: str) -> None:
    """Enforce the ceiling, and clean up after the child, off the event loop."""
    def wait() -> None:
        try:
            process.wait(timeout=MAX_SECONDS)
        except subprocess.TimeoutExpired:
            logger.warning("benchmark battery exceeded %ds; stopping it", MAX_SECONDS)
            stop()
        finally:
            try:
                os.remove(plan_path)
            except OSError:
                pass
            shutil.rmtree(inputs_dir, ignore_errors=True)
    threading.Thread(target=wait, daemon=True).start()


def stop() -> bool:
    """Signal the running battery's whole process group. False if none is."""
    with _lock:
        job = _current
        process = job.get("_process") if job else None
        if process is None or process.poll() is not None:
            return False
        pid = process.pid
    try:
        group = os.getpgid(pid)
    except OSError:
        return False
    # The same guard dispatch.py carries: killpg(0) would signal this server.
    if group <= 1:
        return False
    try:
        os.killpg(group, signal.SIGTERM)
    except OSError:
        return False
    try:
        process.wait(timeout=_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(group, signal.SIGKILL)
        except OSError:
            pass
    return True
