"""Runs a preset plan against this server, from OUTSIDE it.

    python benchmark_client.py --plan <plan.json> --out <summary.json>

A subprocess, and that is the whole reason this file exists rather than a
function. `tool.invoke` runs in a worker thread capped by MAX_CONCURRENT_TOOLS,
and admission caps it again: a battery executed inside the server would occupy
the slots it is trying to measure, and six concurrent runs launched from the
event loop would deadlock against their own limit. So the server spawns this,
it opens ordinary HTTP connections like any other client, and what it measures
is what a client would see.

Standard library only, for the same reason `runner.py` is: it has to run under
whatever interpreter the server happens to be, and a benchmark that needs its
own dependency tree is a benchmark nobody runs.

**It uploads nothing.** Every input in a plan is a name the server already
hosts, so a run is a small form POST. That is deliberate and it bounds what
these numbers mean: they cover dispatch, admission and the tools, and they say
nothing about the transfer path. `benchmarks/` B2 is the instrument for that,
and quietly re-measuring it worse here would be the worst of both.

**Timings come from the server's own event stream.** The client records wall
clock, but the phase breakdown -- staging, queued for the card, running,
packaging -- is read back from `GET /runs/{id}` afterwards, which is the same
source the dashboard draws. A client-side guess at where the seconds went would
disagree with the server about the one thing the server actually knows.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

RUN_ID_HEADER = "X-Run-Id"
# Long: a cohort segmentation legitimately takes an hour, and a benchmark that
# times out before the tool does would record a failure that did not happen.
REQUEST_TIMEOUT = 3 * 3600
POLL_TIMEOUT = 30


def _request(url: str, token: str, data=None, method=None, timeout=POLL_TIMEOUT,
             headers=None):
    body = None
    if data is not None:
        body = urllib.parse.urlencode(data, doseq=True).encode("utf-8")
    request = urllib.request.Request(url, data=body, method=method)
    request.add_header("Authorization", "Bearer " + token)
    if body is not None:
        request.add_header("Content-Type", "application/x-www-form-urlencoded")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read(), dict(response.headers)


def _watch(base: str, token: str, run_id: str, sink: dict,
           stop: threading.Event) -> None:
    """Poll a run's snapshot WHILE it runs, keeping the longest event list seen.

    It has to be during, not after. `POST /run` discards the run's directory as
    the response goes out, so a client that asks afterwards is answered 404 --
    which is correct behaviour and was silently costing every span until it was
    measured. Polling is enough here and a stream is not needed: the phases are
    seconds apart and nothing is shown live.
    """
    url = f"{base}/runs/{urllib.parse.quote(run_id)}"
    while not stop.is_set():
        try:
            status, body, _ = _request(url, token, timeout=10)
            if status == 200:
                events = (json.loads(body) or {}).get("events") or []
                if len(events) > len(sink.get("events") or ()):
                    sink["events"] = events
        except Exception:  # noqa: BLE001 - a missed poll costs one sample
            pass
        stop.wait(0.25)


def _spans_from(events: list, origin: float) -> list:
    """The phases the SERVER recorded, as spans relative to the arm's origin."""
    spans, current = [], None
    for event in events:
        phase, at = event.get("phase"), event.get("at")
        if not phase or at is None:
            continue
        if current is None or current["phase"] != phase:
            if current is not None:
                current["end"] = round(at - origin, 3)
                spans.append(current)
            current = {"phase": phase, "start": round(at - origin, 3), "end": None}
    if current is not None:
        current["end"] = round(time.time() - origin, 3)
        spans.append(current)
    return spans


def _phase_of(event: dict) -> Optional[str]:
    return event.get("phase")


def _one_run(base: str, token: str, item: dict, client: int, origin: float,
             results: list, lock: threading.Lock) -> None:
    run_id = secrets.token_urlsafe(24)
    started = time.time()
    record = {
        "client": client, "index": item["index"], "tool": item["tool"],
        "run_id": run_id, "started": round(started - origin, 3),
        "status": "ok", "error": None, "seconds": None, "spans": [],
        "nested": [],
        # The params ARE shown: every one of them is a name this server hosts,
        # chosen by the preset rather than sent by a clinician, so none of it is
        # anybody's data. A preset that ever took an uploaded file would have to
        # stop reporting this.
        "invocation": {"params": dict(item["params"]), "inputs": [], "produced": {}},
    }
    sink, stop = {}, threading.Event()
    watcher = threading.Thread(target=_watch,
                               args=(base, token, run_id, sink, stop), daemon=True)
    watcher.start()
    try:
        status, body, headers = _request(
            f"{base}/run/{urllib.parse.quote(item['tool'])}",
            token, data=item["params"], method="POST",
            timeout=REQUEST_TIMEOUT, headers={RUN_ID_HEADER: run_id})
        record["seconds"] = round(time.time() - started, 2)
        record["invocation"]["produced"] = {"bytes": len(body)}
        if status >= 400:
            record["status"] = "error"
            record["error"] = f"HTTP {status}"
    except urllib.error.HTTPError as exc:
        record["seconds"] = round(time.time() - started, 2)
        record["status"] = "error"
        # The status and the tool's own exception TYPE, never the body: a 422
        # message is written by a tool and can name the file it refused.
        record["error"] = f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 - one failed run is a result
        record["seconds"] = round(time.time() - started, 2)
        record["status"] = "error"
        record["error"] = type(exc).__name__
    finally:
        stop.set()
        watcher.join(timeout=2)
    record["spans"] = _spans_from(sink.get("events") or [], origin)
    with lock:
        results.append(record)


def run_arm(base: str, token: str, arm: dict, origin: float) -> dict:
    """Execute one arm and summarise it the way `/benchmarks/view` reads."""
    results, lock = [], threading.Lock()
    started = time.time()
    width = max(1, int(arm.get("concurrency", 1)))
    pending = list(arm["runs"])

    while pending:
        batch, pending = pending[:width], pending[width:]
        threads = []
        for slot, item in enumerate(batch):
            thread = threading.Thread(target=_one_run,
                                      args=(base, token, item, slot, origin,
                                            results, lock), daemon=True)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

    results.sort(key=lambda row: row["index"])
    durations = [row["seconds"] for row in results
                 if row["status"] == "ok" and row["seconds"] is not None]
    ok = sum(1 for row in results if row["status"] == "ok")
    phases = {}
    for row in results:
        for span in row["spans"]:
            length = (span["end"] or 0) - span["start"]
            phases.setdefault(span["phase"], []).append(max(0.0, length))
    return {
        "arm": arm["arm"],
        "label": arm["arm"],
        "clients": width,
        "detached": False,
        "tools": arm["tools"],
        "setup": {"clients": width, "shape": arm["shape"],
                  "runs": len(arm["runs"]), "uploads": 0},
        "origin": origin,
        "ok": ok,
        "total": len(results),
        "wall_seconds": round(time.time() - started, 2),
        "fastest_seconds": round(min(durations), 2) if durations else None,
        "slowest_seconds": round(max(durations), 2) if durations else None,
        "median_seconds": round(statistics.median(durations), 2) if durations else None,
        "phases": {name: round(statistics.mean(values), 2)
                   for name, values in sorted(phases.items())},
        "phase_detail": {
            name: {"mean": round(statistics.mean(values), 2),
                   "max": round(max(values), 2), "runs": len(values)}
            for name, values in sorted(phases.items())
        },
        "runs": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    token = os.environ.get("API_TOKEN") or ""
    if not token:
        print("API_TOKEN is not set in this process's environment.")
        return 2
    with open(args.plan, encoding="utf-8") as handle:
        plan = json.load(handle)

    origin = time.time()
    arms = []
    for arm in plan["arms"]:
        arms.append(run_arm(args.base, token, arm, origin))
        # Written after every arm, not once at the end: a battery interrupted
        # halfway should still be readable, and a long one is exactly the one
        # somebody stops.
        _write(args.out, plan, arms, origin, done=False)
    _write(args.out, plan, arms, origin, done=True)
    return 0


def _write(path: str, plan: dict, arms: list, origin: float, done: bool) -> None:
    payload = {
        "generated_at": time.time(),
        "source_preset": plan.get("preset"),
        "label": plan.get("label"),
        "about": plan.get("about"),
        "finished": done,
        "origin": origin,
        "arms": arms,
        "coverage": [],
        "out_of_scope": [],
        "hardware": plan.get("hardware", ""),
    }
    temporary = path + ".part"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
