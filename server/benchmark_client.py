"""Runs a preset plan against this server, from OUTSIDE it.

    python benchmark_client.py --plan <plan.json> --out <summary.json>

A subprocess, and that is the whole reason this file exists rather than a
function. `tool.invoke` runs in a worker thread that admission gates: a battery
executed inside the server would occupy the room it is trying to measure, and
concurrent runs launched from the event loop would deadlock against their own
admission. So the server spawns this,
it opens ordinary HTTP connections like any other client, and what it measures
is what a client would see.

Standard library only, for the same reason `runner.py` is: it has to run under
whatever interpreter the server happens to be, and a benchmark that needs its
own dependency tree is a benchmark nobody runs.

**A preset uploads nothing.** Every input in a preset is a name the server
already hosts, so a run is a small form POST. That is deliberate and it bounds
what these numbers mean: they cover dispatch, admission and the tools, and they
say nothing about the transfer path. `benchmarks/` B2 is the instrument for
that, and quietly re-measuring it worse here would be the worst of both.

**A custom battery may take a bench input**, a case staged under
`DATA/<tool>/bench/`, and that one IS uploaded -- from this machine to itself,
through the same chunked `POST /uploads` a workstation uses. Not because the
transfer is worth measuring here, but because no other door to a bench case
exists, on purpose: `/run` resolves test files by name for any API token, and a
clinical case must not become one. The upload is drawn as its own `transfer`
phase, before the server's, so it never blurs the tool's own time; and the
summary records the input's shape -- files, bytes -- never its name.

**Timings come from the server's own event stream.** The client records wall
clock, but the phase breakdown -- staging, queued for the card, running,
packaging -- is read back from `GET /runs/{id}` afterwards, which is the same
source the dashboard draws. A client-side guess at where the seconds went would
disagree with the server about the one thing the server actually knows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import statistics
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

RUN_ID_HEADER = "X-Run-Id"
# Long: a cohort segmentation legitimately takes an hour, and a benchmark that
# times out before the tool does would record a failure that did not happen.
REQUEST_TIMEOUT = 3 * 3600
POLL_TIMEOUT = 30
# Parts of a bench upload. Loopback, so the size is about request count rather
# than congestion: 32 MB parts, four at a time.
UPLOAD_CHUNK = 32 * 1024 * 1024
UPLOAD_WORKERS = 4
UPLOADS_FIELD = "__uploads__"

# The ledger of the runs this battery has started, one line each, which is how
# the server finds them when the battery is stopped. The runs are the
# SERVER's: killing this process ends none of them, so a stop that did not
# know their ids would leave them holding the card to the end. Set by `--runs`.
LEDGER_START = "start"
LEDGER_END = "end"
_ledger: Optional[str] = None


def _record(kind: str, run_id: str) -> None:
    """Append one line to the ledger: `start` before the run's POST leaves,
    `end` once the server has answered it.

    One `write` on an O_APPEND descriptor, so concurrent runs never interleave
    a line and a line is on the kernel's side of the file the moment this
    returns: a battery SIGKILLed right after has still recorded it. Not fsynced
    -- what this guards against is the process dying, not the machine.
    """
    if not _ledger:
        return
    if kind == LEDGER_END:
        # Best effort: a missing end line costs a stop one needless look at
        # a run that is already over, never a run left going.
        try:
            _write_line(kind, run_id)
        except OSError:
            pass
        return
    _write_line(kind, run_id)


def _write_line(kind: str, run_id: str) -> None:
    descriptor = os.open(_ledger, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(descriptor, f"{kind} {run_id}\n".encode("ascii"))
    finally:
        os.close(descriptor)


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


def _send(url: str, token: str, body: bytes, method: str, content_type: str,
          headers=None):
    request = urllib.request.Request(url, data=body, method=method)
    request.add_header("Authorization", "Bearer " + token)
    request.add_header("Content-Type", content_type)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return response.status, response.read()


# A bench folder is packed ONCE per battery, whatever the number of runs that
# send it: the archive is the same bytes every time, and packing a cohort per
# run would put the packing on the clock of every run after the first.
_packed_lock = threading.Lock()
_packed = {}


def _shape(path: str) -> dict:
    """What an input IS, without saying what it is called."""
    if os.path.isdir(path):
        files, size = 0, 0
        for root, _dirs, names in os.walk(path):
            for name in names:
                try:
                    size += os.path.getsize(os.path.join(root, name))
                    files += 1
                except OSError:
                    continue
        return {"kind": "folder", "files": files, "bytes": size}
    return {"kind": "file", "files": 1, "bytes": os.path.getsize(path)}


def _pack(path: str, scratch: str) -> tuple:
    """(file to send, name to send it under, shape) for one bench entry.

    A file goes as itself, under its own name: several tools read a token out
    of an input's name (a jaw, a timepoint), and renaming it would change the
    run being measured. A folder goes as a stored zip with the folder as its
    single root, which the server strips when it unpacks a `.zip` for a path.
    """
    with _packed_lock:
        if path in _packed:
            return _packed[path]
        shape = _shape(path)
        if shape["kind"] == "file":
            packed = (path, os.path.basename(path), shape)
        else:
            os.makedirs(scratch, exist_ok=True)
            archive = os.path.join(scratch, f"{len(_packed)}.zip")
            root = os.path.basename(os.path.normpath(path))
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED, allowZip64=True) as handle:
                for folder, _dirs, names in os.walk(path):
                    for name in sorted(names):
                        full = os.path.join(folder, name)
                        handle.write(full, os.path.join(root, os.path.relpath(full, path)))
            packed = (archive, root + ".zip", shape)
        _packed[path] = packed
        return packed


def _upload(base: str, token: str, path: str, filename: str) -> str:
    """One file through `POST /uploads` and its parts; the upload id."""
    size = os.path.getsize(path)
    _status, body = _send(f"{base}/uploads", token,
                          json.dumps({"filename": filename, "size": size,
                                      "chunk_size": UPLOAD_CHUNK}).encode("utf-8"),
                          "POST", "application/json")
    session = json.loads(body)
    chunk, count = int(session["chunk_size"]), int(session["part_count"])

    def part(index: int) -> None:
        with open(path, "rb") as handle:
            data = os.pread(handle.fileno(), chunk, index * chunk)
        _send(f"{base}/uploads/{urllib.parse.quote(session['upload_id'])}/parts/{index}",
              token, data, "PUT", "application/octet-stream",
              headers={"X-Part-SHA256": hashlib.sha256(data).hexdigest()})

    with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as pool:
        list(pool.map(part, range(count)))
    return session["upload_id"]


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
             results: list, lock: threading.Lock, scratch: str = "") -> None:
    run_id = secrets.token_urlsafe(24)
    started = time.time()
    record = {
        "client": client, "index": item["index"], "tool": item["tool"],
        "config": item.get("config"),
        "run_id": run_id, "started": round(started - origin, 3),
        "status": "ok", "error": None, "seconds": None, "spans": [],
        "nested": [],
        # The params ARE shown: every one of them is a name this server hosts,
        # chosen by the preset rather than sent by a clinician, so none of it is
        # anybody's data. A preset that ever took an uploaded file would have to
        # stop reporting this.
        "invocation": {"params": dict(item["params"]), "inputs": [], "produced": {}},
    }
    form = dict(item["params"])
    transfer = None
    if item.get("uploads"):
        # Before the run exists: what the server records starts when the POST
        # arrives, so the upload is drawn as its own span ahead of it.
        references = {}
        try:
            for argument, path in sorted(item["uploads"].items()):
                packed, filename, shape = _pack(path, scratch or ".")
                references[argument] = _upload(base, token, packed, filename)
                record["invocation"]["inputs"].append(
                    {"argument": argument, "name": "bench input", "kind": shape["kind"],
                     "files": shape["files"], "bytes": shape["bytes"]})
        except urllib.error.HTTPError as exc:
            record.update(status="error", error=f"upload: HTTP {exc.code}",
                          seconds=round(time.time() - started, 2))
        except Exception as exc:  # noqa: BLE001 - one failed run is a result
            record.update(status="error", error=f"upload: {type(exc).__name__}",
                          seconds=round(time.time() - started, 2))
        transfer = {"phase": "transfer", "start": round(started - origin, 3),
                    "end": round(time.time() - origin, 3)}
        record["transfer_seconds"] = round(time.time() - started, 2)
        if record["status"] != "ok":
            record["spans"] = [transfer]
            with lock:
                results.append(record)
            return
        form[UPLOADS_FIELD] = json.dumps(references)
    # Recorded before the POST, never after: a battery stopped between the two
    # would otherwise leave a run on the server that no stop could name. And a
    # run that could not be recorded is not sent, for the same reason.
    try:
        _record(LEDGER_START, run_id)
    except OSError as exc:
        record.update(status="error", error=f"ledger: {type(exc).__name__}", seconds=0.0)
        record["spans"] = [transfer] if transfer else []
        with lock:
            results.append(record)
        return
    posted = time.time()

    sink, stop = {}, threading.Event()
    watcher = threading.Thread(target=_watch,
                               args=(base, token, run_id, sink, stop), daemon=True)
    watcher.start()
    try:
        status, body, headers = _request(
            f"{base}/run/{urllib.parse.quote(item['tool'])}",
            token, data=form, method="POST",
            timeout=REQUEST_TIMEOUT, headers={RUN_ID_HEADER: run_id})
        _record(LEDGER_END, run_id)
        record["seconds"] = round(time.time() - posted, 2)
        record["invocation"]["produced"] = {"bytes": len(body)}
        if status >= 400:
            record["status"] = "error"
            record["error"] = f"HTTP {status}"
    except urllib.error.HTTPError as exc:
        # The server answered, so the run is over. Any OTHER exception -- a
        # timeout, a reset -- says nothing about the run, which may still be
        # going, and leaves it on the ledger for a stop to cancel.
        _record(LEDGER_END, run_id)
        record["seconds"] = round(time.time() - posted, 2)
        record["status"] = "error"
        # The status and the tool's own exception TYPE, never the body: a 422
        # message is written by a tool and can name the file it refused.
        record["error"] = f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 - one failed run is a result
        record["seconds"] = round(time.time() - posted, 2)
        record["status"] = "error"
        record["error"] = type(exc).__name__
    finally:
        stop.set()
        watcher.join(timeout=2)
    record["spans"] = ([transfer] if transfer else []) + \
        _spans_from(sink.get("events") or [], origin)
    with lock:
        results.append(record)


def _compare(results: list) -> list:
    """Per configuration, in the order they first ran: what each side cost.

    `running` is the server's own compute phase, which is what an A/B of two
    configurations of one tool is asking about; `seconds` adds the queue and
    the packaging around it, which is what a client waits for.
    """
    order, rows = [], {}
    for row in results:
        label = row.get("config")
        if label is None:
            continue
        if label not in rows:
            order.append(label)
            rows[label] = []
        rows[label].append(row)
    if len(order) < 2:
        return []

    def stats(values: list) -> dict:
        if not values:
            return {"n": 0}
        return {"n": len(values), "mean": round(statistics.mean(values), 2),
                "sd": round(statistics.stdev(values), 2) if len(values) > 1 else None,
                "median": round(statistics.median(values), 2),
                "min": round(min(values), 2), "max": round(max(values), 2)}

    table = []
    for label in order:
        group = rows[label]
        ok = [row for row in group if row["status"] == "ok"]
        running = []
        for row in ok:
            spent = sum(max(0.0, (span["end"] or 0) - span["start"])
                        for span in row["spans"] if span["phase"] == "running")
            if spent:
                running.append(spent)
        table.append({"config": label, "tool": group[0]["tool"], "runs": len(group),
                      "ok": len(ok),
                      "seconds": stats([row["seconds"] for row in ok if row["seconds"] is not None]),
                      "running": stats(running)})
    return table


def run_arm(base: str, token: str, arm: dict, origin: float, scratch: str = "") -> dict:
    """Execute one arm and summarise it the way `/benchmarks/view` reads.

    A rolling pool: `concurrency` runs in flight, the next one starting as soon
    as one ends, each start no earlier than `stagger` seconds after the one
    before it. With as many runs as the width, which is every preset, that is
    exactly the batch it always was.
    """
    results, lock = [], threading.Lock()
    started = time.time()
    width = max(1, int(arm.get("concurrency", 1)))
    stagger = max(0.0, float(arm.get("stagger") or 0))
    free = threading.Semaphore(width)
    threads = []

    def one(item: dict, client: int) -> None:
        try:
            _one_run(base, token, item, client, origin, results, lock, scratch)
        finally:
            free.release()

    for position, item in enumerate(arm["runs"]):
        if stagger:
            delay = started + position * stagger - time.time()
            if delay > 0:
                time.sleep(delay)
        free.acquire()
        thread = threading.Thread(target=one, args=(item, position % width), daemon=True)
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
    uploads = sum(len(item.get("uploads") or ()) for item in arm["runs"])
    setup = {"clients": width, "shape": arm["shape"],
             "runs": len(arm["runs"]), "uploads": uploads}
    if stagger:
        setup["stagger"] = stagger
    return {
        "arm": arm["arm"],
        "label": arm.get("label") or arm["arm"],
        "clients": width,
        "detached": False,
        "tools": arm["tools"],
        "setup": setup,
        "compare": _compare(results),
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
    parser.add_argument("--runs", default=None,
                        help="file to append the id of every run started to")
    args = parser.parse_args()

    global _ledger
    _ledger = args.runs

    token = os.environ.get("API_TOKEN") or ""
    if not token:
        print("API_TOKEN is not set in this process's environment.")
        return 2
    with open(args.plan, encoding="utf-8") as handle:
        plan = json.load(handle)

    origin = time.time()
    arms = []
    for arm in plan["arms"]:
        arms.append(run_arm(args.base, token, arm, origin, plan.get("scratch") or ""))
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
        # What each configuration of a custom battery was, by its letter: the
        # tool and the values typed, and only the NAMES of the arguments a
        # bench input filled.
        "configs": plan.get("configs") or [],
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
