"""What this machine is doing right now, for `GET /server-debug`.

`resources.py` answers what this deployment is ALLOWED to use -- a budget,
resolved once at startup from configuration and detection. This answers what is
being used, sampled per request, which is a different question and the one a
reader has when a run is slow.

Three things it deliberately does not do:

* **It samples, it does not poll.** A CPU percentage is a delta between two
  readings of `/proc/stat`, so the first call after startup has nothing to
  subtract from and honestly answers `None` rather than inventing a number from
  the machine's uptime. The page polls on a timer and the second reading is the
  first useful one.
* **It never blocks to be accurate.** No `sleep` to widen a sampling window: a
  telemetry endpoint that holds a worker thread for a second is a telemetry
  endpoint that changes what it measures.
* **It reports an unavailable reading as absent, never as zero.** A container
  without `/proc`, a host with no card, a directory that does not exist: each
  answers `None`, and the page says "unavailable" instead of drawing an empty
  bar that reads as "idle".

The queue log is in this module rather than in `admission.py` because it is an
observation of admission, not part of it: nothing here is read by the decision,
and losing it costs a page a table rather than costing a run its room. It is
in-process and bounded -- it resets when the server restarts, and two uvicorn
workers keep two of them. Both are stated on the page, because a leaderboard
that silently covers one worker of four is worse than no leaderboard.
"""

from __future__ import annotations

import collections
import os
import threading
import time
from typing import Optional

_PROC_STAT = "/proc/stat"
_PROC_MEMINFO = "/proc/meminfo"

# Enough to see a working day's shape at the rate runs actually arrive, and
# small enough that the whole structure stays a rounding error against one
# CBCT. A deque discards from the far end on its own, so nothing here grows.
QUEUE_LOG_SIZE = 512

# Below this, a run did not queue: it asked for room and got it. Recording
# those as waits would bury the real ones under a thousand zeroes, and the
# distinction the page exists to draw is exactly "waited" against "ran".
WAITED_SECONDS = 0.05


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------

_cpu_lock = threading.Lock()
_cpu_previous = None  # (busy jiffies, total jiffies)


def _cpu_jiffies() -> Optional[tuple]:
    """(busy, total) from the aggregate line of /proc/stat."""
    try:
        with open(_PROC_STAT, encoding="utf-8") as handle:
            first = handle.readline()
    except OSError:
        return None
    fields = first.split()
    if not fields or fields[0] != "cpu":
        return None
    try:
        values = [int(value) for value in fields[1:]]
    except ValueError:
        return None
    if len(values) < 4:
        return None
    total = sum(values)
    # user, nice, system, irq, softirq, steal -- everything except idle and
    # iowait, which are the two the kernel counts as "not computing".
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return total - idle, total


def cpu_percent() -> Optional[float]:
    """Busy share of the machine since the previous call, 0..100.

    `None` on the first call and whenever `/proc/stat` cannot be read: there is
    no interval to divide by, and a made-up figure on a debug page is worse
    than a blank one.
    """
    sample = _cpu_jiffies()
    if sample is None:
        return None
    global _cpu_previous
    with _cpu_lock:
        previous = _cpu_previous
        _cpu_previous = sample
    if previous is None:
        return None
    busy = sample[0] - previous[0]
    total = sample[1] - previous[1]
    if total <= 0:
        # Two calls inside one jiffy. Not an error, just nothing to say yet.
        return None
    return round(max(0.0, min(100.0, 100.0 * busy / total)), 1)


# ---------------------------------------------------------------------------
# Host memory
# ---------------------------------------------------------------------------

def ram() -> Optional[dict]:
    """{total, available, used} in bytes, from /proc/meminfo.

    `MemAvailable`, not `MemFree`: the kernel's own estimate of what a new
    allocation could actually get, which counts reclaimable page cache. On a
    server that has just written a 2 GB result, `MemFree` reads as almost
    nothing while the machine is not under memory pressure at all.
    """
    wanted = {"MemTotal:": None, "MemAvailable:": None}
    try:
        with open(_PROC_MEMINFO, encoding="utf-8") as handle:
            for line in handle:
                key = line.split(None, 1)[0] if line.split() else ""
                if key in wanted and wanted[key] is None:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            wanted[key] = int(parts[1]) * 1024
                        except ValueError:
                            return None
                if all(value is not None for value in wanted.values()):
                    break
    except OSError:
        return None
    total, available = wanted["MemTotal:"], wanted["MemAvailable:"]
    if not total or available is None:
        return None
    return {"total": total, "available": available, "used": max(0, total - available)}


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------

def _tree_bytes(path: str, budget: int = 40000) -> Optional[int]:
    """Bytes held under `path`, or None if it does not exist.

    `budget` bounds the walk. TEMP_DIR holds uploads, results and run
    directories, and an unpacked cohort is tens of thousands of files -- a
    debug page must not turn into a recursive stat of a whole dataset while a
    run is trying to write into it. Past the budget the figure is what was
    counted so far, and the caller is told it is partial.
    """
    if not os.path.isdir(path):
        return None
    total = 0
    seen = 0
    for root, _, files in os.walk(path, onerror=lambda exc: None):
        for name in files:
            seen += 1
            if seen > budget:
                return total
            try:
                stat = os.stat(os.path.join(root, name))
            except OSError:
                continue
            # st_blocks, not st_size: a sparse upload blob is truncated to its
            # full length before a byte of it arrives, so st_size would report
            # a 2 GB transfer the moment it was opened.
            total += stat.st_blocks * 512
    return total


def disk(paths: dict) -> dict:
    """Free space on the filesystem holding each path, and what it holds.

    Keyed by the caller's own label so the page can name them: this module has
    no opinion on which directories matter to a deployment.
    """
    report = {}
    for label, path in paths.items():
        entry = {"path": path, "exists": os.path.isdir(path)}
        try:
            stat = os.statvfs(path)
            entry["free"] = stat.f_bavail * stat.f_frsize
            entry["total"] = stat.f_blocks * stat.f_frsize
        except OSError:
            entry["free"] = None
            entry["total"] = None
        entry["used_here"] = _tree_bytes(path)
        report[label] = entry
    return report


# ---------------------------------------------------------------------------
# In-flight requests
# ---------------------------------------------------------------------------

_inflight_lock = threading.Lock()
_inflight = 0
_inflight_peak = 0


def request_started() -> None:
    global _inflight, _inflight_peak
    with _inflight_lock:
        _inflight += 1
        _inflight_peak = max(_inflight_peak, _inflight)


def request_finished() -> None:
    global _inflight
    with _inflight_lock:
        _inflight = max(0, _inflight - 1)


def inflight() -> dict:
    """How many requests are being served right now, and the most ever at once.

    This is the closest honest answer to "how many clients are connected".
    HTTP has no session: a client that uploaded a cohort and is now waiting
    inside a four-hour POST is one in-flight request, and a client that has
    gone away without closing its socket is none. The page says "requests",
    not "clients", for that reason.
    """
    with _inflight_lock:
        return {"now": _inflight, "peak": _inflight_peak}


# ---------------------------------------------------------------------------
# The queue
# ---------------------------------------------------------------------------

_queue_lock = threading.Lock()
_queue_log = collections.deque(maxlen=QUEUE_LOG_SIZE)


def record_admission(tool: str, waited: float, channels: int = 1) -> None:
    """One run got its room. Never raises: this is an observation.

    Called after the wait, so `waited` is the whole time between asking and
    being admitted -- which for a run that did not queue is microseconds, and
    is kept anyway: "how many ran straight through" is half of what the
    leaderboard below means.
    """
    try:
        entry = {
            "tool": str(tool),
            "waited": round(max(0.0, float(waited)), 2),
            "channels": int(channels) if channels else 1,
            "at": time.time(),
        }
    except (TypeError, ValueError):
        return
    with _queue_lock:
        _queue_log.append(entry)


def queue_history() -> dict:
    """The admissions this process has seen, in order, and ranked by waiting.

    `order` is the order of passage the reader asked for -- newest last, as a
    log reads. `ranked` answers "which tool spends the most time waiting",
    which is total wait rather than mean: a tool that queues for four seconds
    a hundred times costs this server more than one that queued once for a
    minute, and the mean would say the opposite.
    """
    with _queue_lock:
        entries = list(_queue_log)
    per_tool = {}
    for entry in entries:
        row = per_tool.setdefault(entry["tool"], {
            "tool": entry["tool"], "runs": 0, "queued": 0,
            "total_wait": 0.0, "worst_wait": 0.0,
        })
        row["runs"] += 1
        row["total_wait"] += entry["waited"]
        row["worst_wait"] = max(row["worst_wait"], entry["waited"])
        if entry["waited"] >= WAITED_SECONDS:
            row["queued"] += 1
    ranked = sorted(per_tool.values(), key=lambda row: row["total_wait"], reverse=True)
    for row in ranked:
        row["total_wait"] = round(row["total_wait"], 2)
        row["worst_wait"] = round(row["worst_wait"], 2)
        row["mean_wait"] = round(row["total_wait"] / row["runs"], 2) if row["runs"] else 0.0
    return {
        "order": entries[-60:],
        "ranked": ranked,
        "capacity": QUEUE_LOG_SIZE,
        "held": len(entries),
    }


# ---------------------------------------------------------------------------
# The run ledger
# ---------------------------------------------------------------------------
#
# One record per run this process has seen, from admission to outcome. It is
# what the page's history list and its per-run detail are drawn from, and what
# the per-tool uptime is summed out of.
#
# In this module rather than in `wire/runs.py` for the same reason the queue log
# is: `runs.py` holds the state a run NEEDS -- its events, its process group,
# whether it was cancelled -- on disk, because a `DELETE` may be served by a
# different worker than the `POST`. This holds what an operator would LIKE to
# know afterwards, and losing it costs a page a table rather than costing a run
# anything. So it is in memory, bounded, and says so on the page.
#
# **No argument VALUE is ever recorded here, only names.** A value is a path,
# and a path is a patient's file name. The same rule that keeps progress
# messages off this page keeps parameters off it: which knobs a caller set is
# an operational fact, what they set them to is clinical data.

LEDGER_SIZE = 240

_ledger_lock = threading.Lock()
_ledger = collections.OrderedDict()  # run_id -> record


def _ledger_record(run_id: str) -> Optional[dict]:
    """The record for `run_id`, created on first sight. Caller holds the lock."""
    if not run_id:
        return None
    record = _ledger.get(run_id)
    if record is None:
        record = {
            "run_id": run_id, "tool": None, "client": None,
            "started_at": time.time(),
            "ended_at": None, "seconds": None, "outcome": None, "phase": None,
            "waited": None, "channels": None, "cpus": None,
            "ram_bytes": None, "vram_bytes": None,
            "files": None, "input_bytes": None, "arguments": None,
            "inputs": None, "settings": None,
        }
        _ledger[run_id] = record
        while len(_ledger) > LEDGER_SIZE:
            _ledger.popitem(last=False)
    else:
        _ledger.move_to_end(run_id)
    return record


def record_run_start(run_id: str, tool: str, client: str = None) -> None:
    """A run exists and is about to be staged. Never raises.

    `client` is the peer the server saw, which is what makes a run attributable
    to a workstation -- the one thing a single shared API token otherwise makes
    impossible.

    Guarded rather than merely claimed: this is called from `runs.register`,
    which is on the path of every run, and an observation that could take a
    cohort down would be an absurd trade for a row on a page.
    """
    try:
        with _ledger_lock:
            record = _ledger_record(run_id)
            if record is not None:
                record["tool"] = str(tool)[:100] if tool else None
                if client:
                    record["client"] = str(client)[:64]
    except Exception:  # noqa: BLE001 - telemetry must never fail a run
        pass


# How many files a single input is walked over before its size is reported as
# partial. Deliberately smaller than the disk sweep's: this runs on the request
# path, microseconds before the tool is invoked, and an unpacked cohort can be
# tens of thousands of files.
_INPUT_WALK_BUDGET = 5000


def describe_inputs(args: dict, specs: dict) -> dict:
    """{inputs, settings} for one run: what arrived, and what was asked of it.

    The whole point is the line it draws, because "show the arguments" is two
    different questions wearing one name:

    * **An uploaded path is never named.** Its name is the patient's. What is
      reported is its SHAPE -- how many files, how many bytes, which extensions
      -- which is what makes two runs comparable without identifying either.
    * **A hosted name IS reported.** A `server_selectable` argument resolves to
      something this deployment staged itself, and `GET /tools/{tool}/data`
      already lists those names to anyone holding the same token. Hiding it
      here would protect nothing and cost the reader the one fact that says
      which bundle a result came from.
    * **A setting is reported in full.** `landmarks=["Ba","S","N"]`,
      `device="cuda"`, `structures=["MAND","MAX"]`: every one of them names an
      entry in a catalogue the tool publishes, not anything about a person. And
      they are exactly what two runs have to be compared on.
    """
    inputs, settings = [], {}
    for name, value in sorted(args.items()):
        spec = specs.get(name) or {}
        hosted = spec.get("server_selectable")
        text = str(value)
        looks_like_path = bool(text) and os.path.exists(text)
        if not looks_like_path:
            if not hosted:
                settings[name] = _plain(value)
            continue
        entry = {"argument": name, "kind": getattr(value, "kind", None)}
        if hosted:
            # The hosted name, which is this deployment's own and already
            # public through the data listing.
            entry["hosted"] = os.path.basename(text.rstrip("/"))
        shape = _shape_of(text)
        entry.update(shape)
        inputs.append(entry)
    return {"inputs": inputs, "settings": settings}


def _plain(value):
    """A setting reduced to what JSON can carry, bounded."""
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return sorted(str(item)[:60] for item in value)[:40]
    if isinstance(value, dict):
        return sorted(str(key)[:60] for key, on in value.items() if on)[:40]
    return str(value)[:120]


def _shape_of(path: str) -> dict:
    """How many files, how big, and of what kind -- never a name."""
    if os.path.isfile(path):
        try:
            return {"files": 1, "bytes": os.path.getsize(path),
                    "extensions": _extensions([path])}
        except OSError:
            return {"files": 1, "bytes": None, "extensions": []}
    total, seen, names = 0, 0, []
    partial = False
    for root, _, files in os.walk(path, onerror=lambda exc: None):
        for name in files:
            seen += 1
            if seen > _INPUT_WALK_BUDGET:
                partial = True
                break
            names.append(name)
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
        if partial:
            break
    shape = {"files": seen, "bytes": total, "extensions": _extensions(names)}
    if partial:
        shape["partial"] = True
    return shape


def _extensions(names) -> list:
    """The distinct suffixes present. `.nii.gz` counts as one, not as `.gz`."""
    found = set()
    for name in names:
        base = os.path.basename(name).lower()
        for double in (".nii.gz", ".nrrd.gz", ".gipl.gz", ".tar.gz"):
            if base.endswith(double):
                found.add(double)
                break
        else:
            suffix = os.path.splitext(base)[1]
            if suffix:
                found.add(suffix)
    return sorted(found)[:8]


def record_run_inputs(run_id: str, files: int, input_bytes: int,
                      arguments=None, detail=None) -> None:
    """What the caller sent: how much of it, and which knobs were named.

    `arguments` is a sequence of argument NAMES. Values are deliberately not
    accepted by this function, so a caller cannot pass one by accident.
    """
    with _ledger_lock:
        record = _ledger_record(run_id)
        if record is None:
            return
        try:
            record["files"] = int(files)
            record["input_bytes"] = int(input_bytes)
        except (TypeError, ValueError):
            pass
        if arguments is not None:
            record["arguments"] = sorted({str(name)[:60] for name in arguments})
        if detail is not None:
            record["inputs"] = detail.get("inputs") or []
            record["settings"] = detail.get("settings") or {}



def record_run_grant(run_id: str, channels=None, cpus=None, ram_bytes=None,
                     vram_bytes=None, waited=None) -> None:
    """What admission let this run hold, once it had been let in."""
    with _ledger_lock:
        record = _ledger_record(run_id)
        if record is None:
            return
        for key, value in (("channels", channels), ("cpus", cpus),
                           ("ram_bytes", ram_bytes), ("vram_bytes", vram_bytes),
                           ("waited", waited)):
            if value is None:
                continue
            try:
                record[key] = round(float(value), 2) if key == "waited" else int(value)
            except (TypeError, ValueError):
                pass


def record_run_end(run_id: str, outcome: str, phase: str = "") -> None:
    """How it ended, and therefore how long it occupied this server.

    Guarded for the same reason `record_run_start` is: `runs.finish` is the one
    funnel every ending passes through, and a run that completed must not be
    reported as failed because the bookkeeping after it threw.
    """
    try:
        with _ledger_lock:
            record = _ledger.get(run_id)
            if record is None:
                return
            record["ended_at"] = time.time()
            record["outcome"] = str(outcome)[:40]
            record["phase"] = str(phase)[:40] if phase else None
            record["seconds"] = round(
                max(0.0, record["ended_at"] - record["started_at"]), 2)
    except Exception:  # noqa: BLE001 - telemetry must never fail a run
        pass


def run_ledger(limit: int = 80) -> list:
    """The runs this process has seen, newest first."""
    with _ledger_lock:
        records = list(_ledger.values())
    records.reverse()
    return [dict(record) for record in records[:limit]]


def tool_uptime() -> list:
    """How long each tool has occupied this server, busiest first.

    The figure a reader wants for "which tool is this machine actually for" --
    and the one a capacity decision is made on later, which is why the counts
    that produced it travel beside it rather than being collapsed into a mean.

    A run still going contributes the time it has taken SO FAR, and is counted
    separately in `running`, so a long cohort in flight is visible as work in
    progress rather than silently missing until it ends.
    """
    now = time.time()
    with _ledger_lock:
        records = list(_ledger.values())
    per_tool = {}
    for record in records:
        name = record["tool"] or "(unnamed)"
        row = per_tool.setdefault(name, {
            "tool": name, "runs": 0, "running": 0, "ok": 0, "failed": 0,
            "seconds": 0.0, "longest": 0.0, "last_seen": 0.0,
        })
        row["runs"] += 1
        if record["ended_at"] is None:
            row["running"] += 1
            elapsed = max(0.0, now - record["started_at"])
        else:
            elapsed = record["seconds"] or 0.0
            if record["outcome"] == "done":
                row["ok"] += 1
            else:
                row["failed"] += 1
        row["seconds"] += elapsed
        row["longest"] = max(row["longest"], elapsed)
        row["last_seen"] = max(row["last_seen"],
                               record["ended_at"] or record["started_at"])
    rows = sorted(per_tool.values(), key=lambda row: row["seconds"], reverse=True)
    for row in rows:
        row["seconds"] = round(row["seconds"], 1)
        row["longest"] = round(row["longest"], 1)
        row["mean"] = round(row["seconds"] / row["runs"], 1) if row["runs"] else 0.0
    return rows


def reset() -> None:
    """Forget everything sampled so far. For tests."""
    global _cpu_previous, _inflight, _inflight_peak
    with _cpu_lock:
        _cpu_previous = None
    with _inflight_lock:
        _inflight = 0
        _inflight_peak = 0
    with _queue_lock:
        _queue_log.clear()
    with _ledger_lock:
        _ledger.clear()
