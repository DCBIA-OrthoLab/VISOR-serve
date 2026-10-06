"""What a run is doing while it is doing it, and how a client stops it.

`POST /run/{tool}` blocks for as long as the tool takes, which for a cohort is
hours. Until now the only thing a client could show for that was an elapsed
timer: a multi-minute wait for the GPU queue and a multi-hour segmentation look
identical from the outside, and there was no way at all to call one off. This
module is the state that makes both possible -- one directory per run, holding
the events written so far, the process group to signal, and whether the client
has asked for the run to stop.

It is `transfer.py`'s discipline applied to a second kind of ephemeral state,
and deliberately so:

- The id is matched against a regex BEFORE any path is built from it, so
  `../../etc` is never looked up, not even to be reported as missing.
- State lives on disk, not in a module global. The POST and the `DELETE` that
  cancels it may legitimately be served by different `uvicorn --workers`, which
  is exactly why the child's process group id goes in a file: `os.killpg`
  reaches across processes of the same user, an in-process handle does not.
- `touch()` on every append and every read makes RUN_TTL_SECONDS an IDLE
  timeout, so a run still reporting progress is never reaped under itself.
- The same `_reaper_loop` that sweeps transfers sweeps these.

Two properties are load-bearing enough to state plainly.

**Appends are lock-free.** Every writer opens the file `O_APPEND` and writes
one record with a single `write()` of under 4096 bytes, which POSIX guarantees
is atomic below PIPE_BUF -- so the server process, the tool's process, and
every supervised child below it can append to one file with nothing between
them. A reader only ever parses lines terminated by a newline, so a record
caught mid-flight is read on the next poll rather than half-parsed.

**`seq` is the line number, assigned when the record is READ.** Nothing writes
it. It cannot be written: a value monotonic per run would have to be counted
first, and between the count and the write a supervised child could append its
own. The file's order IS the sequence, every reader derives the same numbering
from it, and the client gets exactly the monotonic key the contract promises.
"""

import contextvars
import errno
import json
import logging
import math
import os
import re
import shutil
import time
from typing import List, Optional

import redact
import telemetry
from config import settings
from wire import _scratch

logger = logging.getLogger("inference_server.runs")


EVENTS_FILE = "events.jsonl"
# The tool a run is for, written once at registration. Not in the event stream
# because the events are a tool's own output and this is the server's fact, and
# because a run that failed before emitting anything would otherwise be a bare
# id nobody could place.
META_FILE = "meta.json"
PGID_FILE = "pgid"
CANCEL_FILE = "cancel"

# The environment variable naming events.jsonl for the tool process. Absolute,
# set only when the run has a directory, and inherited by every supervised
# child -- which is the whole implementation of chain progress.
PROGRESS_FILE_ENV = "SADT_PROGRESS_FILE"

# The closed vocabulary. A client translates these into the clinician's
# language, so a phase added here is a client release: keep it small.
PHASE_RECEIVED = "received"
PHASE_STAGING = "staging"
PHASE_QUEUED_GPU = "queued_gpu"
PHASE_RUNNING = "running"
PHASE_PACKAGING = "packaging"
PHASE_DONE = "done"
PHASE_FAILED = "failed"
PHASE_CANCELLED = "cancelled"
PHASE_PAUSED = "paused"

STATE_PENDING = "pending"
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_FAILED = "failed"
STATE_CANCELLED = "cancelled"
# A run that stopped where it was asked to, and can be told to carry on.
# NOT terminal: the client is expected to come back, and the reaper's idle
# timeout is what bounds how long that is worth waiting for.
STATE_PAUSED = "paused"

TERMINAL_STATES = (STATE_DONE, STATE_FAILED, STATE_CANCELLED)

# One rule, in one place: a caller names a phase and the state follows from it.
# Two fields travel because the client needs both -- `state` decides whether to
# keep watching, `phase` decides what to say -- but only one of them is ever
# chosen.
_STATE_OF_PHASE = {
    PHASE_RECEIVED: STATE_PENDING,
    PHASE_STAGING: STATE_PENDING,
    PHASE_QUEUED_GPU: STATE_PENDING,
    PHASE_RUNNING: STATE_RUNNING,
    PHASE_PACKAGING: STATE_RUNNING,
    PHASE_DONE: STATE_DONE,
    PHASE_FAILED: STATE_FAILED,
    PHASE_CANCELLED: STATE_CANCELLED,
    PHASE_PAUSED: STATE_PAUSED,
}

# A progress message is free text written by a tool, so it is bounded here
# rather than trusted. 200 characters is what a panel can show; the cap also
# keeps a record comfortably under PIPE_BUF, which is what makes concurrent
# appends atomic.
MAX_MESSAGE_CHARS = 200
# What a tool folder may be called, which is what a nested marker may name.
# Deliberately narrow: see `_clean_tool_name`.
_TOOL_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")

# One file, two kinds of writer: this server, and the tool process (plus every
# supervised level below it). The server marks its own records, and only a
# marked one is allowed to name a phase -- a tool's line is a running tool
# whatever it says.
#
# This is not a defence against a hostile tool: a tool runs on this machine with
# that file open for append, and could write the marker too. It is a defence
# against the realistic accident -- a tool author reading the wire shape in
# RUN_PROGRESS.md and writing `"phase": "done"` when its own work finishes,
# which would end every watcher's stream while the run had hours left.
_SOURCE_KEY = "src"
_SERVER_SOURCE = "server"

# A record longer than this cannot be appended atomically, so it is not
# appended at all. Unreachable with a truncated message -- 200 characters of
# four-byte UTF-8 is 800 bytes -- and here because "unreachable" is not
# "impossible" when the writer is in another repository.
MAX_RECORD_BYTES = 4096

# The deepest a supervised chain can nest (runner.MAX_SUPERVISOR_DEPTH). A
# `depth` outside this is not a depth, and the record is read as the root's.
_MAX_DEPTH = 10

# Which supervised call a record belongs to (runner.PROGRESS_CALL_ENV): absent
# for the root, "1" for its first `sup.run`, "1.2" for that child's second.
_CALL_ID = re.compile(r"[0-9]{1,4}(\.[0-9]{1,4}){0,9}")

# A log line, as opposed to a progress record: `{"kind": "log", "level",
# "audience", "message"}`, written by `sup.log` (or a tool's own helper).
#
# **Never delivered to a reader that did not ask for it.** The Slicer client
# released before this existed treats every event as progress -- it would put
# a log line in place of the progress message, blank the bar on its null
# fraction, and show an operator's line to the clinician. So a reader is given
# the audiences it asked for and nothing else, and the client's own stream asks
# for `user` at most; `admin` lines reach only the operator page, redacted.
LOG_KIND = "log"
LOG_AUDIENCE_USER = "user"
LOG_AUDIENCE_ADMIN = "admin"
LOG_AUDIENCES = (LOG_AUDIENCE_USER, LOG_AUDIENCE_ADMIN)
LOG_LEVELS = ("debug", "info", "warning", "error")
# Per reader, apart from MAX_RUN_EVENTS: a chatty log must not be able to use
# up the progress budget and leave a client's bar frozen.
MAX_RUN_LOGS = 500
# What a run's ledger record keeps of its log once the run is gone, redacted:
# the latest warnings and errors, and as many of the latest info lines.
_KEPT_LOG_LEVELS = ("warning", "error")
_MAX_KEPT_LOGS = 20

# The run this request belongs to, for the code between the endpoint and the
# subprocess. A ContextVar rather than a parameter threaded through
# `Tool.invoke`: that signature is the one every tool and both dispatch paths
# agree on, and a run id is request scope, not tool input. `anyio.to_thread`
# copies the context into the worker thread, so `dispatch` reads the same value
# the endpoint set -- the mechanism `file_utils._scratch_dirs` already relies on
# for exactly this reason.
CURRENT_RUN: contextvars.ContextVar = contextvars.ContextVar("sadt_run_id", default=None)


class RunError(Exception):
    """Anything the CLIENT got wrong about a run id: malformed, already used,
    unknown. main.py maps `status_code` straight through, so the message
    reaches the caller intact -- it is never a server bug."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _runs_root() -> str:
    return os.path.join(settings.TEMP_DIR, "runs")


def run_directory(run_id: str) -> str:
    """The run's directory, or raise. The regex runs FIRST: a malformed id is
    refused before `os.path.join` is given it."""
    if not _scratch.is_valid_id(run_id or ""):
        raise RunError("Malformed run id.", status_code=400)
    path = os.path.join(_runs_root(), run_id)
    if not os.path.isdir(path):
        raise RunError(f"Unknown or expired run '{run_id}'.", status_code=404)
    return path


def register(run_id: str, tool: Optional[str] = None,
             client: Optional[str] = None, batch: Optional[dict] = None) -> str:
    """Claim an id and open its directory. Returns the id.

    Called as the FIRST thing `POST /run` does, before `await request.form()`,
    and that ordering is the feature: parsing the form IS the multipart upload,
    minutes of it for a CBCT, and the client opens its event stream the instant
    it has minted the id. Registering afterwards would answer that watcher a
    404 -- which the client is told to read as "an older server, stop watching"
    -- on precisely the long uploads this exists for.
    """
    if not _scratch.is_valid_id(run_id or ""):
        raise RunError("Malformed run id.", status_code=400)
    root = _runs_root()
    os.makedirs(root, exist_ok=True)
    directory = os.path.join(root, run_id)
    try:
        # exist_ok=False: an id already in flight, or finished and not yet
        # reaped, must not be silently joined. Two runs sharing a directory
        # would interleave their events and let either cancel the other.
        os.makedirs(directory)
    except FileExistsError:
        raise RunError(f"Run id '{run_id}' is already in use.", status_code=409)
    # Created empty so the tool's O_APPEND open finds a file rather than
    # failing: nothing on that side is allowed to create it, since a stale
    # SADT_PROGRESS_FILE would otherwise litter whatever it points at.
    with open(os.path.join(directory, EVENTS_FILE), "wb"):
        pass
    if tool or client or batch:
        # Best effort: a run whose name could not be written is still a run.
        try:
            with open(os.path.join(directory, META_FILE), "w", encoding="utf-8") as handle:
                json.dump({"tool": str(tool)[:100] if tool else None,
                           "client": str(client)[:64] if client else None,
                           "batch": batch,
                           "at": time.time()}, handle)
        except OSError:
            pass
    telemetry.record_run_start(run_id, tool, client, batch=batch)
    reap_expired()
    return run_id


# A batch as the client names it: which cohort split this run is one part of.
# The id is minted by the client per Apply and means nothing outside it.
_BATCH_ID = re.compile(r"[A-Za-z0-9_-]{8,64}")
_MAX_BATCHES = 10000


def parse_batch(batch_id, index, total) -> Optional[dict]:
    """`{id, index, total}` from the three headers, or None if any is absent
    or implausible. Never raises: a run with a malformed batch is a run."""
    if not batch_id or not _BATCH_ID.fullmatch(str(batch_id)):
        return None
    try:
        index, total = int(index), int(total)
    except (TypeError, ValueError):
        return None
    if not (1 <= index <= total <= _MAX_BATCHES):
        return None
    return {"id": str(batch_id), "index": index, "total": total}


def meta(run_id: str) -> dict:
    """What `register` wrote for this run, or {}."""
    try:
        with open(os.path.join(run_directory(run_id), META_FILE), encoding="utf-8") as handle:
            return json.load(handle) or {}
    except (RunError, OSError, ValueError):
        return {}


def discard(run_id: str) -> None:
    """Drop a run's directory. Best effort, and never an error for an id that
    is already gone: it is called from cleanup paths and from a background
    task, where raising would be worse than the leak the reaper would catch."""
    if not _scratch.is_valid_id(run_id or ""):
        return
    # What a paused run was holding on to goes with it. This is the only
    # place that knows: the request that staged those directories handed
    # them over rather than deleting them, precisely so a resume could use
    # them, and nothing else is told when that resume finally ends.
    for directory in (paused_at(run_id) or {}).get("keep") or ():
        shutil.rmtree(directory, ignore_errors=True)
    shutil.rmtree(os.path.join(_runs_root(), run_id), ignore_errors=True)


# ----------------------------------------------------------------------
# Expiry
# ----------------------------------------------------------------------

# Called on every append and every read, which is what makes RUN_TTL_SECONDS an
# idle timeout rather than an age limit: a cohort reporting progress for six
# hours is never reaped under itself, while one whose client vanished expires
# minutes later.
touch = _scratch.touch


def _kept_by(directory: str) -> list:
    """Every directory a paused run is holding, read off its own record.

    By PATH rather than by run id, because this is what the reaper has: it is
    deleting a directory whose id it is about to forget. Both halves are
    returned -- the job directory the resume would have read, and the inputs
    the request staged for it -- since nothing else in this server is told
    they exist.
    """
    try:
        with open(os.path.join(directory, PAUSED_FILE), encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return []
    if not isinstance(record, dict):
        return []
    held = [record.get("job_dir")]
    held.extend(record.get("keep") or ())
    return [str(entry) for entry in held if entry]


def _ttl_of(directory: str) -> float:
    """How long this run may sit idle: longer while somebody is reading it."""
    if os.path.exists(os.path.join(directory, PAUSED_FILE)):
        return settings.PAUSED_RUN_TTL_SECONDS
    return settings.RUN_TTL_SECONDS


def reap_expired(now: Optional[float] = None) -> int:
    """Delete run directories that have gone idle, and what they were holding.

    The normal path removes a run with its request, so this is the safety net
    for the one that never got there: a client that vanished mid-POST, a worker
    killed between the terminal event and the cleanup. It matters because a
    progress message is written by a tool and can name a file.

    **It has to release what a paused run kept, and this is the only place
    that can.** A run stopped at a checkpoint hands its job directory and its
    staged inputs to `pause()` precisely so the request does NOT delete them,
    and the only record of where they are is the file inside the run
    directory. Deleting that directory first -- which is what a generic
    sweep does -- left a staged cohort of patient data on disk with nothing
    left in the server that knew its name. Measured: the run gone, the
    cohort still there, and no second sweep that would ever find it.

    A paused run is also given longer to be idle, because what it is waiting
    for is a person reading a cohort and nothing touches it meanwhile.
    """
    deadline_now = time.time() if now is None else now
    removed = 0
    root = _runs_root()
    try:
        entries = os.listdir(root)
    except FileNotFoundError:
        return 0
    for entry in entries:
        directory = os.path.join(root, entry)
        try:
            if os.path.getmtime(directory) > deadline_now - _ttl_of(directory):
                continue
        except OSError:
            # Gone between listdir and getmtime: another worker won.
            continue
        for held in _kept_by(directory):
            shutil.rmtree(held, ignore_errors=True)
        # ignore_errors: every uvicorn worker runs its own reaper, so losing
        # the race to delete the same directory is expected.
        shutil.rmtree(directory, ignore_errors=True)
        removed += 1
    if removed:
        logger.info("run reaper removed %d expired directory(ies)", removed)
    return removed


# ----------------------------------------------------------------------
# Writing an event
# ----------------------------------------------------------------------

# What a run may report about its own cost, and nothing else may ride this
# field. Each is a plain number the server itself measured.
_MEASURED_FIELDS = ("vram_bytes", "ram_bytes", "cpu_cores", "channels")


def _clean_measured(measured) -> Optional[dict]:
    """A run's own cost, reduced to four numbers, or None.

    Whitelisted rather than passed through: this rides an event the client
    reads and a browser renders, so the shape is stated here instead of being
    whatever a caller happened to build. A field that is not a finite number
    is dropped, and an object with nothing left is None rather than `{}` --
    "not measured" and "measured as empty" must not render the same.
    """
    if not isinstance(measured, dict):
        return None
    kept = {}
    for name in _MEASURED_FIELDS:
        value = measured.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isnan(value) or math.isinf(value) or value < 0:
            continue
        kept[name] = value
    return kept or None


def _clean_tool_name(name) -> str:
    """A nested call's tool name, or "" -- and it is checked, not merely bounded.

    This field travels further than a message does: the benchmark payload drops
    messages precisely because a tool writes them and a tool's free text may
    name a patient's file, while it keeps this so a chain's bar can say WHICH
    tool the parent called. That only holds while the field cannot be free
    text, so anything that is not a plain identifier is dropped rather than
    truncated -- a truncated file name is still a file name.

    The pattern is what a tool folder may be called (`registry` discovers
    `<TOOLS_DIR>/<name>/`), so a real name always survives it.
    """
    text = "" if name is None else str(name)
    return text if _TOOL_NAME.fullmatch(text) else ""


def _clean_message(message) -> str:
    """A message is free text from a tool. Bounded, and stripped of the one
    character that would corrupt the file: a newline inside a record would be
    read as the end of it, and everything after as a separate malformed one."""
    text = "" if message is None else str(message)
    text = text.replace("\r", " ").replace("\n", " ")
    return text[:MAX_MESSAGE_CHARS]


def _clean_fraction(fraction) -> Optional[float]:
    """0.0..1.0 or None. `null` is a first-class answer here -- the contract
    says a fraction is never fabricated, so anything unusable becomes "unknown"
    rather than a number a progress bar would believe."""
    if fraction is None:
        return None
    try:
        value = float(fraction)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return min(1.0, max(0.0, value))


def _clean_span(span) -> Optional[list]:
    """`[start, end]` within 0..1, start <= end, or None."""
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return None
    start, end = _clean_fraction(span[0]), _clean_fraction(span[1])
    if start is None or end is None or start > end:
        return None
    return [start, end]


def _append_record(directory: str, record: dict) -> None:
    """One record, one `write()`, `O_APPEND`. See the module docstring for why
    that is the whole locking strategy."""
    payload = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
    if len(payload) > MAX_RECORD_BYTES:
        logger.debug("dropping an oversized run event (%d bytes)", len(payload))
        return
    path = os.path.join(directory, EVENTS_FILE)
    try:
        handle = os.open(path, os.O_WRONLY | os.O_APPEND)
    except OSError as exc:
        # The run was reaped, or discarded by the request that owned it. Losing
        # progress is never a reason to fail a run that is otherwise fine.
        if exc.errno not in (errno.ENOENT, errno.EACCES):
            raise
        return
    try:
        os.write(handle, payload)
    finally:
        os.close(handle)
    touch(directory)


def append(run_id: str, phase: str, fraction=None, message: str = "",
           depth: int = 0, result=None, measured=None) -> None:
    """Record one server-side phase. Silent for a run that no longer exists.

    `result` rides the TERMINAL event of a detached run: the response to the
    POST was a 202 minutes earlier, so the event stream is the only place left
    to hand the client its `result_ref`.

    `measured` is what THIS run actually cost -- its VRAM and RSS peaks, the
    channels it opened and the cores it burned. The figures existed already:
    `runner.py` has measured every run since the subprocess path landed and
    `execution/costs.py` folds them into a per-TOOL high-water mark. What was
    missing was per-RUN attribution: with six runs on one card, a trace of the
    whole card lines a peak up with the runs that could have caused it and
    attributes it to none of them. They are numbers, never patient data, which
    is why they may travel at all.
    """
    try:
        directory = run_directory(run_id)
    except RunError:
        return
    record = {
        _SOURCE_KEY: _SERVER_SOURCE,
        "at": time.time(),
        "state": _STATE_OF_PHASE.get(phase, STATE_RUNNING),
        "phase": phase,
        "fraction": _clean_fraction(fraction),
        "message": _clean_message(message),
        "depth": depth,
    }
    if result is not None:
        record["result"] = result
    if measured is not None:
        record["measured"] = measured
    _append_record(directory, record)


def emit(phase: str, fraction=None, message: str = "", depth: int = 0,
         measured=None) -> None:
    """`append` for the run this request belongs to, and a no-op when there is
    none.

    Every call site in main.py and dispatch.py goes through this, so a client
    that sent no `X-Run-Id` costs exactly one ContextVar read per phase and the
    two paths never fork.
    """
    run_id = CURRENT_RUN.get()
    if run_id is None:
        return
    append(run_id, phase, fraction, message, depth, measured=measured)


def finish(run_id: str, phase: str, message: str = "", result=None,
           failure: Optional[dict] = None) -> None:
    """The terminal event. Written before the directory is discarded, so a
    watcher that polls once more sees how the run ended.

    `failure` is the operator's diagnosis of a failed run -- which tool, where
    in the chain, why -- already redacted by the caller. It goes to the ledger
    only, never on the event stream.
    """
    append(run_id, phase, None, message, result=result)
    # The ledger is closed HERE rather than at each of the five call sites that
    # end a run, because this is the one funnel all of them pass through: a
    # sixth ending added later is recorded without anybody remembering to. The
    # phase is all that travels -- never `message`, which on a failure carries
    # a tool's own words and can name the file it died on.
    #
    # The timeline is taken now because this is the last moment the events
    # exist: the directory is discarded right after, and the ledger is what
    # the operator page draws a finished run from.
    #
    # The warnings and errors a tool logged are kept too, REDACTED, so "why did
    # this run fail" still has an answer after the directory is gone. Never
    # the whole log, and never a line as the tool wrote it.
    shape, kept = None, None
    try:
        events = EventReader(run_directory(run_id), logs=LOG_AUDIENCES).read()
        shape = timeline(events)
        kept = notable_logs(events)
    except Exception:  # noqa: BLE001 - the ledger must not fail a run's ending
        pass
    telemetry.record_run_end(run_id, phase, phase, timeline=shape,
                             failure=failure, logs=kept)


def notable_logs(events: List[dict]) -> List[dict]:
    """What a run's history keeps of its log, redacted, in the order written:
    the latest warnings and errors, and the latest info lines beside them --
    apart, so a chatty tool's info cannot push its one error out."""
    serious, chatter = [], []
    for event in events:
        if event.get("kind") != LOG_KIND or event.get("level") == "debug":
            continue
        (serious if event.get("level") in _KEPT_LOG_LEVELS else chatter).append(event)
    chosen = {id(event) for event in serious[-_MAX_KEPT_LOGS:] + chatter[-_MAX_KEPT_LOGS:]}
    kept = []
    for event in events:
        if id(event) not in chosen:
            continue
        line = {"at": event.get("at"), "level": event["level"],
                "audience": event.get("audience"), "depth": event.get("depth", 0),
                "message": redact.scrub(event.get("message"))}
        if event.get("source"):
            line["source"] = event["source"]
        kept.append(line)
    return kept


# ----------------------------------------------------------------------
# Reading events
# ----------------------------------------------------------------------

def _normalised(raw: dict, seq: int, seen_at: float) -> dict:
    """One parsed line as the wire event the client is promised.

    A line may have been written by a TOOL -- third-party code in another
    repository, appending `{"fraction": ..., "message": ...}` to a path it was
    handed in the environment. So nothing here is trusted:

    - `seq` is the line number, never the record's. A tool cannot forge an
      ordering, and two readers of the same file derive the same one.
    - `state` and `phase` are the server's vocabulary, and only a record this
      server wrote may name one. A tool's line is a running tool, whatever it
      claims; only the server declares a run done, failed or cancelled.
    - `at` and `depth` are facts only the writer knows, so they are read when
      present and sane -- `at` falls back to the moment the line was first
      seen, which the 0.25s poll keeps within a quarter second of the truth,
      and an implausible `depth` reads as the root's.
    """
    phase = raw.get("phase")
    if raw.get(_SOURCE_KEY) == _SERVER_SOURCE and phase in _STATE_OF_PHASE:
        state = _STATE_OF_PHASE[phase]
    else:
        state, phase = STATE_RUNNING, PHASE_RUNNING

    at = raw.get("at")
    try:
        at = float(at)
        if math.isnan(at) or math.isinf(at) or at <= 0:
            at = seen_at
    except (TypeError, ValueError):
        at = seen_at

    depth = raw.get("depth", 0)
    try:
        depth = int(depth)
    except (TypeError, ValueError):
        depth = 0
    if depth < 0 or depth > _MAX_DEPTH:
        depth = 0

    call = raw.get("call")
    call = call if isinstance(call, str) and _CALL_ID.fullmatch(call) else None

    if raw.get("kind") == LOG_KIND and raw.get(_SOURCE_KEY) != _SERVER_SOURCE:
        # A log line. `state` and `phase` ride along as a running tool's, so
        # that anything reading `event["state"]` -- the terminal check above
        # all -- reads a log line as harmless; it carries no fraction, because
        # it says nothing about how far along the run is.
        level = str(raw.get("level") or "").lower()
        audience = raw.get("audience")
        event = {
            "seq": seq,
            "at": at,
            "kind": LOG_KIND,
            "state": STATE_RUNNING,
            "phase": PHASE_RUNNING,
            "level": level if level in LOG_LEVELS else "info",
            # Unmarked is the operator's: a line nobody said the clinician
            # should read is not shown to the clinician.
            "audience": audience if audience in LOG_AUDIENCES else LOG_AUDIENCE_ADMIN,
            "message": _clean_message(raw.get("message")),
            "depth": depth,
        }
        if call:
            event["call"] = call
        return event

    event = {
        "seq": seq,
        "at": at,
        "state": state,
        "phase": phase,
        "fraction": _clean_fraction(raw.get("fraction")),
        "message": _clean_message(raw.get("message")),
        "depth": depth,
    }
    # The supervisor's marker for a nested call, at the CHILD's depth. Unlike
    # a phase this MAY come from a tool's own process -- the supervisor runs
    # inside one -- so it is guarded by what it is allowed to look like
    # instead of by who wrote it.
    nested = _clean_tool_name(raw.get("tool"))
    if nested:
        event["tool"] = nested
        if raw.get("edge") in ("open", "close"):
            event["edge"] = raw["edge"]
        span = _clean_span(raw.get("span"))
        if span:
            event["span"] = span
    if call:
        event["call"] = call
    # Only ever from the server, and for the same reason a phase is: a TOOL
    # appends to this same file, and a tool able to write its own `result`
    # could hand the client a pointer to somebody else's bytes.
    if raw.get(_SOURCE_KEY) == _SERVER_SOURCE and isinstance(raw.get("result"), dict):
        event["result"] = raw["result"]
    # Server-sourced for the same reason, and for a sharper one: a tool able to
    # write its own cost could tell the budget it is free.
    if raw.get(_SOURCE_KEY) == _SERVER_SOURCE:
        measured = _clean_measured(raw.get("measured"))
        if measured:
            event["measured"] = measured
    return event


def _parent_call(call: str) -> str:
    return call.rsplit(".", 1)[0] if "." in call else ""


class _Call:
    __slots__ = ("parent", "tool", "span", "depth", "local", "opened", "done", "children")

    def __init__(self, parent):
        self.parent = parent
        self.tool = None
        self.span = None
        self.depth = 0
        self.local = None
        self.opened = False
        self.done = False
        self.children = []


class _ProgressTree:
    """One run's supervised calls, and the bar they add up to.

    A caller that hands a call a span -- `sup.run(..., _progress=(0.2, 0.6))`
    -- is saying that call fills 0.2..0.6 of ITS bar, so a child at half way
    puts the caller at 0.4. Applied at every level, a leaf three calls down
    moves the root's bar by exactly its share, and the clinician sees one bar
    that only ever moves forward instead of each tool's own 0..1 in turn.

    A level's position is the furthest of what it said itself and where its
    weighted calls have got it; a finished call counts as its whole span. A
    call given no span weighs nothing on its caller, and its records keep the
    fraction they were written with -- so a chain nobody weighted reads
    exactly as it did before this existed.

    Records say whose they are with `call`. One that does not -- a tool's own
    `progress.py`, written before it learned to -- is attributed by where it
    falls: to the one call open when it was written, since its caller is
    blocked in `sup.run` meanwhile. Two calls open at once make that a guess,
    and a guess is not made: the record keeps its own fraction.
    """

    def __init__(self):
        self.calls = {"": _Call(None)}
        self.calls[""].opened = True

    def _node(self, call: str) -> _Call:
        node = self.calls.get(call)
        if node is None:
            parent = _parent_call(call)
            parent_node = self._node(parent)
            node = _Call(parent)
            node.depth = min(_MAX_DEPTH, call.count(".") + 1)
            parent_node.children.append(call)
            self.calls[call] = node
        return node

    def _open(self) -> List[str]:
        return [call for call, node in self.calls.items()
                if call and node.opened and not node.done]

    def _attributed(self, raw: dict) -> Optional[str]:
        """The call an untagged record belongs to, or None when it is a guess."""
        open_calls = self._open()
        if "depth" in raw:
            depth = raw.get("depth")
            if depth == 0:
                return ""
            here = [call for call in open_calls if self.calls[call].depth == depth]
            return here[0] if len(here) == 1 else None
        if not open_calls:
            return ""
        depths = [self.calls[call].depth for call in open_calls]
        if len(set(depths)) != len(depths):
            return None  # siblings running at once: whose it is is unknowable
        return max(open_calls, key=lambda call: self.calls[call].depth)

    def _position(self, call: str) -> Optional[float]:
        """How far along `call` is on its own 0..1, its weighted calls included."""
        node = self.calls[call]
        if node.done:
            return 1.0
        found = [] if node.local is None else [node.local]
        for child_id in node.children:
            child = self.calls[child_id]
            if child.span is None or not child.opened:
                continue
            start, end = child.span
            found.append(start + (end - start) * (self._position(child_id) or 0.0))
        return max(found) if found else None

    def _weighted(self, call: str) -> bool:
        return any(self.calls[child].span is not None for child in self.calls[call].children)

    def feed(self, event: dict, raw: dict) -> dict:
        """Fold one tool record into the tree; return it with the run's fraction."""
        if raw.get(_SOURCE_KEY) == _SERVER_SOURCE:
            return event
        call = event.get("call")
        if event.get("tool"):
            if call is None:
                return event  # a marker from before calls had ids
            node = self._node(call)
            edge = event.get("edge")
            if edge == "open" or (edge is None and not node.opened):
                node.opened, node.tool = True, event["tool"]
                node.span = tuple(event["span"]) if event.get("span") else None
                node.depth = event.get("depth", node.depth)
            else:
                node.done = True
        else:
            if call is None:
                call = self._attributed(raw)
                if call is None:
                    return event
                if call and "depth" not in raw:
                    event["depth"] = self.calls[call].depth
            node = self._node(call)
            if event.get("fraction") is not None:
                node.local = event["fraction"]
        # Up the weighted links: the outermost call this record's position
        # reaches is the scale its fraction is reported on.
        top = call
        while top and self.calls[top].span is not None:
            top = self.calls[top].parent
        if top == call and not self._weighted(call):
            return event  # nothing weighted anywhere above or below: as written
        value = self._position(top)
        if value is None:
            return event
        if event.get("fraction") is not None and event["fraction"] != value:
            event["own_fraction"] = event["fraction"]
        event["fraction"] = round(value, 4)
        return event

    def tool_of(self, call: Optional[str]) -> Optional[str]:
        node = self.calls.get(call or "")
        return node.tool if node is not None else None


class EventReader:
    """Reads `events.jsonl` forward, once for a snapshot or repeatedly for a
    stream.

    One class for both so "what counts as an event" is written once: the
    snapshot is a single `read()`, the SSE endpoint is `read()` on a timer
    until `finished`. It holds a byte offset and whatever tail of a line has
    not been terminated yet, which is what makes tailing a file three
    processes are appending to safe -- a record caught mid-`write()` is
    completed on the next poll instead of being parsed in half.
    """

    def __init__(self, directory: str, keep_alive: bool = True, logs=()):
        # `keep_alive=False` reads WITHOUT stamping the directory, and exists
        # for the operator listing. The TTL is an idle timeout so that a run
        # still reporting progress is never reaped under itself -- that is
        # about the RUN's own activity, not about somebody looking at it. A
        # dashboard polling every two seconds would otherwise keep every
        # abandoned run alive for as long as the tab stayed open, which is
        # exactly the leak the idle timeout exists to close.
        self.directory = directory
        self._keep_alive = keep_alive
        self.finished = False
        self._offset = 0
        self._partial = b""
        self._seq = 0
        self._delivered = 0
        self._truncated = False
        # Which log audiences this reader hands out: none unless asked. See
        # LOG_KIND for why that default is the only safe one.
        self._logs = tuple(audience for audience in logs if audience in LOG_AUDIENCES)
        self._logs_delivered = 0
        self._tree = _ProgressTree()

    @property
    def path(self) -> str:
        return os.path.join(self.directory, EVENTS_FILE)

    def read(self) -> List[dict]:
        """Every event appended since the last call, oldest first."""
        try:
            with open(self.path, "rb") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
                self._offset = handle.tell()
        except OSError:
            # The directory was reaped or discarded under us. The stream ends
            # rather than erroring: the run is over either way.
            self.finished = True
            return []

        if self._keep_alive:
            touch(self.directory)
        buffer = self._partial + chunk
        lines = buffer.split(b"\n")
        self._partial = lines.pop()

        seen_at = time.time()
        events = []
        for line in lines:
            if not line.strip():
                self._seq += 1
                continue
            try:
                raw = json.loads(line.decode("utf-8", errors="replace"))
            except ValueError:
                # A malformed line costs that line. It is written by another
                # repository's code, so it is data, not a reason to fail.
                self._seq += 1
                continue
            if not isinstance(raw, dict):
                self._seq += 1
                continue
            event = _normalised(raw, self._seq, seen_at)
            self._seq += 1
            if event.get("kind") == LOG_KIND:
                if event["audience"] in self._logs and self._logs_delivered < MAX_RUN_LOGS:
                    self._logs_delivered += 1
                    source = self._tree.tool_of(event.get("call"))
                    if source:
                        event["source"] = source
                    events.append(event)
                continue
            event = self._tree.feed(event, raw)
            capped = self._capped(event)
            if capped is not None:
                events.append(capped)
                if event["state"] in TERMINAL_STATES:
                    self.finished = True
                    break
        return events

    def _capped(self, event: dict) -> Optional[dict]:
        """MAX_RUN_EVENTS, applied where the events are handed out.

        The cap cannot live at the write side: a tool appends to the file from
        its own process, and a supervised chain from several at once, so no
        writer knows how many records the run already holds. It lives here, and
        a terminal event is never dropped -- a stream that could not end would
        leave every watcher hanging on a run that finished half an hour ago.
        """
        if self._delivered < settings.MAX_RUN_EVENTS:
            self._delivered += 1
            return event
        if event["state"] in TERMINAL_STATES:
            return event
        if self._truncated:
            return None
        self._truncated = True
        return dict(
            event,
            fraction=None,
            message=(
                f"Progress messages past {settings.MAX_RUN_EVENTS} are not "
                f"reported; the run continues."
            ),
        )


def read_events(run_id: str, logs=()) -> List[dict]:
    """Every event so far, oldest first. `404` for an unknown id.

    `logs` names the log audiences to include; none by default.
    """
    return EventReader(run_directory(run_id), logs=logs).read()


# What a timeline keeps at most. A run that reports progress every second for
# hours is still one span; these bound the pathological cases, a tool calling
# a sibling thousands of times or a phase flapping, so a ledger record stays
# a few kilobytes.
_MAX_SPANS = 60
_MAX_NESTED = 200
_TERMINAL_PHASES = (PHASE_DONE, PHASE_FAILED, PHASE_CANCELLED)


def timeline(events: List[dict]) -> dict:
    """A run's shape over time, built from its events: `{spans, nested, chain,
    measured}`.

    * `spans` -- the server's phases at the root, `[{phase, start, end}]`, the
      bars a Gantt draws. A terminal phase closes the last span and is not a
      span itself; an open span (a live run) has `end: None`.
    * `nested` -- every call a tool made to another one through the
      supervisor, `[{tool, depth, start, end}]`. The supervisor brackets each
      call with two identical markers at the child's depth, so they pair by
      `(tool, depth)` in file order; an unclosed one is still running.
    * `chain` -- the calls open right now, outermost first: what the root is
      waiting on at this instant, `["ASO", "ALI_CBCT"]` under an AREG.
    * `measured` -- the run's own peaks, from its last `measured` event.

    Only the phase, the time, the depth and the tool name are read. A message
    is a tool's free text and can name a patient's file, so nothing here is
    built from one.
    """
    spans, nested, open_calls, measured = [], [], {}, None
    for event in events:
        if event.get("kind") == LOG_KIND:
            continue
        at = event.get("at")
        if event.get("measured"):
            measured = event["measured"]
        tool = event.get("tool")
        if tool:
            key = (tool, event.get("depth", 0))
            started = open_calls.pop(key, None)
            if started is None:
                open_calls[key] = at
            elif len(nested) < _MAX_NESTED:
                nested.append({"tool": tool, "depth": key[1], "start": started, "end": at})
            continue
        if event.get("depth", 0) != 0:
            continue
        phase = event.get("phase")
        if spans and spans[-1]["end"] is None:
            if spans[-1]["phase"] == phase:
                continue
            spans[-1]["end"] = at
        if phase in _TERMINAL_PHASES or len(spans) >= _MAX_SPANS:
            continue
        spans.append({"phase": phase, "start": at, "end": None})
    still_open = sorted(open_calls.items(), key=lambda item: (item[0][1], item[1]))
    for (tool, depth), started in still_open:
        if len(nested) < _MAX_NESTED:
            nested.append({"tool": tool, "depth": depth, "start": started, "end": None})
    nested.sort(key=lambda call: (call["start"] or 0, call["depth"]))
    return {
        "spans": spans,
        "nested": nested,
        "chain": [tool for (tool, _depth), _started in still_open],
        "measured": measured,
    }


def snapshot(run_id: str, logs=()) -> dict:
    """The whole run in one object: how it stands now, and how it got there.

    Exists for tests, for debugging, and for a client that cannot hold a
    streaming connection open. The Slicer client watches the event stream
    instead, so nothing here is on the hot path.
    """
    directory = run_directory(run_id)
    events = EventReader(directory, logs=logs).read()
    try:
        started_at = os.path.getctime(directory)
    except OSError:
        started_at = time.time()
    # Where the run STANDS is its latest progress, never a log line.
    progress = [event for event in events if event.get("kind") != LOG_KIND]
    latest = progress[-1] if progress else None
    return {
        "run_id": run_id,
        "state": latest["state"] if latest else STATE_PENDING,
        "phase": latest["phase"] if latest else PHASE_RECEIVED,
        "fraction": latest["fraction"] if latest else None,
        "message": latest["message"] if latest else "",
        "depth": latest["depth"] if latest else 0,
        "started_at": progress[0]["at"] if progress else started_at,
        "updated_at": latest["at"] if latest else started_at,
        "events": events,
    }


# ----------------------------------------------------------------------
# Cancellation
# ----------------------------------------------------------------------

def set_pgid(run_id: str, pgid: int) -> None:
    """Record the tool's process group, once, immediately after `Popen`.

    On disk rather than in a module global because the `DELETE` that cancels
    this run may be served by a different `uvicorn --workers` process, which
    holds no handle on the child. `os.killpg` reaches it anyway -- same user,
    and `start_new_session` made the child a group leader -- so the group id is
    the only thing that has to travel, and a file is how it travels.
    """
    try:
        directory = run_directory(run_id)
    except RunError:
        return
    try:
        with open(os.path.join(directory, PGID_FILE), "w", encoding="utf-8") as handle:
            handle.write(str(int(pgid)))
    except OSError as exc:
        # Losing this costs the immediate kill, not the cancellation: dispatch
        # still polls the marker and stops at the next check.
        logger.warning("could not record the process group for a run: %s", exc)


def get_pgid(run_id: str) -> Optional[int]:
    try:
        directory = run_directory(run_id)
    except RunError:
        return None
    try:
        with open(os.path.join(directory, PGID_FILE), encoding="utf-8") as handle:
            pgid = int(handle.read().strip())
    except (OSError, ValueError):
        return None
    # 0 means "the caller's own process group" to killpg, which is this server.
    # A truncated write must not be able to take the API down on a cancel.
    return pgid if pgid > 1 else None


# Where a paused run's job directory is recorded, beside its events. One
# line, because that is all a resume needs to find everything else.
PAUSED_FILE = "paused.json"


def pause(run_id: str, job_dir: str, stopped_after: str, keep=()) -> None:
    """Record that this run stopped, where its work is, and what it still needs.

    The job directory is what a resume reads: the memo each supervised call
    left behind, and the outputs a reader is about to correct. Written here
    rather than held in a module global for the same reason the process group
    is -- the POST that resumes may be served by a different `uvicorn
    --workers` process, which holds nothing.

    `keep` is the staged INPUTS. A request deletes everything it staged once
    its response has streamed, which is right for confidential imaging and
    wrong for exactly this case: the resume runs the tool again and hands it
    the same inputs, so deleting them left the run unable to carry on.

    It never showed up in testing because a hosted test file resolves to a
    path in `DATA/` that no request owns; every UPLOAD, which is every
    clinical use, lost its input the moment the run paused.
    """
    try:
        directory = run_directory(run_id)
    except RunError:
        return
    try:
        with open(os.path.join(directory, PAUSED_FILE), "w", encoding="utf-8") as handle:
            json.dump({"job_dir": job_dir, "stopped_after": stopped_after,
                       "keep": [str(entry) for entry in keep or ()]}, handle)
    except OSError as exc:
        logger.warning("could not record the pause for run %s: %s", run_id, exc)
        return
    touch(directory)
    append(run_id, phase=PHASE_PAUSED)


def keep_while_paused(run_id, directories) -> None:
    """Add `directories` to what a paused run is holding on to.

    Called after the pause was recorded, because what the REQUEST staged is
    only fully known once its response is being built. Silent for a run that
    is not paused: a finished run has nothing to hold.
    """
    record = paused_at(run_id) if run_id else None
    if record is None:
        return
    keep = list(record.get("keep") or ())
    keep.extend(str(entry) for entry in directories or () if entry and entry not in keep)
    record["keep"] = keep
    try:
        directory = run_directory(run_id)
        with open(os.path.join(directory, PAUSED_FILE), "w", encoding="utf-8") as handle:
            json.dump(record, handle)
    except (RunError, OSError) as exc:
        logger.warning("Could not record what run %s is keeping: %s", run_id, exc)


def paused_at(run_id: str) -> Optional[dict]:
    """`{"job_dir": ..., "stopped_after": ...}` for a paused run, or None."""
    try:
        directory = run_directory(run_id)
    except RunError:
        return None
    try:
        with open(os.path.join(directory, PAUSED_FILE), encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return None
    job_dir = record.get("job_dir")
    if not job_dir or not os.path.isdir(job_dir):
        # The directory went with the reaper, or the server was restarted on
        # a fresh TEMP_DIR. A resume cannot be offered for work that is gone.
        return None
    return record


def clear_pause(run_id: str) -> None:
    """Forget the pause, once the run has been told to carry on."""
    try:
        directory = run_directory(run_id)
    except RunError:
        return
    try:
        os.remove(os.path.join(directory, PAUSED_FILE))
    except OSError:
        pass


def request_cancel(run_id: str) -> Optional[int]:
    """Write the cancel marker; hand back the process group to signal, if one
    has been recorded yet.

    Idempotent: a second `DELETE` is a marker that already exists, which is the
    same answer. Returning the pgid rather than signalling here keeps the kill
    discipline in one place -- `execution/dispatch.py` owns TERM-then-KILL, and
    a second implementation of it would be a second thing to get wrong.
    """
    directory = run_directory(run_id)
    marker = os.path.join(directory, CANCEL_FILE)
    try:
        os.close(os.open(marker, os.O_CREAT | os.O_WRONLY))
    except OSError as exc:
        raise RunError(f"Could not cancel run '{run_id}': {exc}", status_code=500)
    touch(directory)
    return get_pgid(run_id)


def is_cancelled(run_id: Optional[str]) -> bool:
    """Has the client asked for this run to stop?

    Polled by dispatch every RUN_CANCEL_POLL_SECONDS, so it is one `stat` and
    nothing else. A malformed or unknown id is not cancelled -- there is
    nothing to cancel.
    """
    if not _scratch.is_valid_id(run_id or ""):
        return False
    return os.path.exists(os.path.join(_runs_root(), run_id, CANCEL_FILE))


def progress_file(run_id: Optional[str]) -> Optional[str]:
    """The absolute path a tool appends its own progress to, or None.

    None whenever there is no run directory, which is what keeps
    SADT_PROGRESS_FILE out of the environment of a run nobody is watching.
    """
    if not run_id:
        return None
    try:
        directory = run_directory(run_id)
    except RunError:
        return None
    return os.path.abspath(os.path.join(directory, EVENTS_FILE))


def active(limit: int = 500, with_chain: bool = False) -> list:
    """Every run the registry still holds, newest first, without its events.

    For an operator looking at a live server: what is on it, how far along, and
    how long it has been there. `snapshot` answers that for ONE run to whoever
    holds its id; this answers it for all of them, so it is deliberately
    narrower than `snapshot` is.

    **No message, ever.** A progress message is free text written by a tool and
    can name the file it is working on, which is a patient's. `snapshot`
    carries it because a caller holding a run id is the client that started
    that run; a listing is read by anyone holding the shared API token, which
    on this deployment is every workstation. The phase says what it is doing;
    the message would say whose data it is doing it to.
    """
    root = _runs_root()
    try:
        names = os.listdir(root)
    except OSError:
        return []
    found = []
    for name in names:
        if not _scratch.is_valid_id(name):
            continue
        directory = os.path.join(root, name)
        events_path = os.path.join(directory, EVENTS_FILE)
        try:
            started_at = os.path.getctime(directory)
            updated_at = os.path.getmtime(events_path)
        except OSError:
            continue
        latest = None
        events = []
        try:
            # keep_alive=False: LOOKING at a run is not the run being alive.
            # Reading stamps the directory, and the TTL is an idle timeout, so
            # an operator page polling every couple of seconds would hold every
            # abandoned run open for as long as the tab stayed on screen.
            events = EventReader(directory, keep_alive=False).read()
            latest = events[-1] if events else None
        except Exception:  # noqa: BLE001 - a listing must not fail on one bad run
            pass
        tool = None
        client = None
        batch = None
        try:
            with open(os.path.join(directory, META_FILE), encoding="utf-8") as handle:
                meta = json.load(handle) or {}
            tool = meta.get("tool")
            client = meta.get("client")
            batch = meta.get("batch")
            # The registration stamp, not the directory's ctime. ctime is the
            # INODE CHANGE time: every `touch` moves it, so a run that reported
            # progress -- or that anything stat'd and stamped -- claimed to have
            # started at that moment, and an elapsed column read zero for
            # everything. `at` is written once and never rewritten.
            recorded = meta.get("at")
            if isinstance(recorded, (int, float)) and recorded > 0:
                started_at = float(recorded)
        except (OSError, ValueError):
            pass
        entry = {
            "run_id": name,
            "tool": tool,
            "client": client,
            "state": latest["state"] if latest else STATE_PENDING,
            "phase": latest["phase"] if latest else PHASE_RECEIVED,
            "fraction": latest["fraction"] if latest else None,
            "depth": latest["depth"] if latest else 0,
            "started_at": started_at,
            "updated_at": updated_at,
        }
        if batch:
            entry["batch"] = batch
        if with_chain:
            # Tool names only, which `_clean_tool_name` already restricted to
            # identifiers: the operator page may say a run is inside ALI_CBCT
            # without anything a tool wrote reaching it.
            entry["chain"] = timeline(events)["chain"]
        found.append(entry)
    found.sort(key=lambda entry: entry["started_at"], reverse=True)
    return found[:limit]
