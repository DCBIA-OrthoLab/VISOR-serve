"""Raw data is append-only. This module is the only thing that writes it.

One JSONL record per INDIVIDUAL run, into

    results/raw/<campaign>-<UTC timestamp>.jsonl

created with O_EXCL so a file is never reopened and never overwritten. Every
summary under results/summary/ is derived from these files and may be deleted
and regenerated at will; nothing regenerates a raw file.

A failed run is a record like any other, with `status: "failed"` and its error.
Dropping it would make a campaign that half worked look like a campaign that
worked, which is the failure mode this whole file exists to prevent.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from . import __version__
from .provenance import utc_now

RAW_SUBDIR = os.path.join("results", "raw")
SUMMARY_SUBDIR = os.path.join("results", "summary")

STATUS_OK = "ok"
STATUS_FAILED = "failed"

# The phase names campaigns are allowed to use. Fixed here rather than left to
# each campaign, because the summaries add them up and a typo would silently
# create a new column that sums to nothing.
PHASES = (
    # A remote call, decomposed the way B2 needs it.
    "pack",
    "upload",
    "server_exec",
    "download",
    "unpack",
    # The local path.
    "job_setup",
    "interpreter_start",
    "import_stack",
    "compute",
    "collect",
    # Whatever the named phases did not account for. Always derived, never timed.
    "other",
)


@dataclass
class RunRecord:
    """One run of one tool through one path. This is the unit of raw data.

    `phases` holds wall-clock seconds per phase and is not required to be
    complete: `total_seconds` is measured independently, and `other` is what
    the named phases did not account for. Deriving `other` rather than timing it
    is deliberate -- it can only be non-negative if the phases really are
    disjoint, so a negative `other` in a summary is a bug report about the
    instrumentation rather than a plausible number.
    """

    campaign: str
    tool: str
    path: str
    repetition: int
    started_at: str
    finished_at: str
    total_seconds: float
    status: str
    phases: dict = field(default_factory=dict)
    # Set on the runs a campaign protocol discards (run 1 of B1). Kept in the
    # raw file, excluded from the summary -- discarding at write time would
    # make "we discarded the warm-up" unverifiable.
    warmup: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    # Anything campaign-specific: payload size, concurrency level, VRAM peak,
    # the parity report. Free-form on purpose; the summaries read named keys
    # out of it and ignore the rest.
    extra: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    harness_version: str = __version__

    def as_json_line(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)


class PhaseTimer:
    """Accumulates named phase durations for one run.

    A context manager per phase rather than start/stop calls, so a phase whose
    body raises is still recorded -- the duration up to the exception is real,
    and a partial decomposition is worth more than none when diagnosing a
    failure.
    """

    def __init__(self) -> None:
        self.phases: dict = {}
        self._started = time.monotonic()

    @contextmanager
    def phase(self, name: str):
        if name not in PHASES:
            raise ValueError(f"Unknown phase {name!r}. Add it to recording.PHASES first.")
        started = time.monotonic()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + (time.monotonic() - started)

    def add(self, name: str, seconds: float) -> None:
        """Record a phase timed somewhere else -- inside a worker thread, or by
        the server and read back off the wire."""
        if name not in PHASES:
            raise ValueError(f"Unknown phase {name!r}. Add it to recording.PHASES first.")
        self.phases[name] = self.phases.get(name, 0.0) + seconds

    @property
    def total(self) -> float:
        return time.monotonic() - self._started

    def finalize(self, total: Optional[float] = None) -> dict:
        """The phase map with `other` filled in.

        Clamped at zero and reported as-is: see the note on RunRecord.phases.
        """
        total_seconds = self.total if total is None else total
        named = sum(value for key, value in self.phases.items() if key != "other")
        phases = dict(self.phases)
        phases["other"] = total_seconds - named
        return phases


class Recorder:
    """Appends records to one raw file, and never to a second one.

    The file is created O_EXCL: two harness invocations in the same second
    would otherwise share a name, and the second would append into the first's
    data under a different provenance. It is flushed after every record so a
    campaign killed halfway still leaves every run it completed.
    """

    def __init__(self, root: str, campaign: str, timestamp: Optional[str] = None) -> None:
        stamp = timestamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        directory = os.path.join(root, RAW_SUBDIR)
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, f"{campaign}-{stamp}.jsonl")
        # "x" rather than "a": a name collision is an error to see, not a merge
        # to perform.
        self._handle = open(self.path, "x", encoding="utf-8")
        self.count = 0
        self.failures = 0

    def write(self, record: RunRecord) -> None:
        self._handle.write(record.as_json_line() + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.count += 1
        if record.status == STATUS_FAILED:
            self.failures += 1

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *_exception) -> None:
        self.close()


def read_raw(path: str) -> list:
    """Every record in one raw file.

    A truncated last line -- the harness was killed mid-write -- is skipped
    with its own marker rather than raising, so a partial campaign can still be
    summarised. It cannot happen through Recorder (which fsyncs whole lines),
    only through a copy interrupted in flight.
    """
    records: list = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                records.append({"_unreadable_line": number, "_raw": line[:200]})
    return records


def load_records(root: str, campaign: Optional[str] = None) -> list:
    """Every record of every raw file, optionally for one campaign only."""
    directory = os.path.join(root, RAW_SUBDIR)
    if not os.path.isdir(directory):
        return []
    records: list = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".jsonl"):
            continue
        if campaign and not name.startswith(f"{campaign}-"):
            continue
        for record in read_raw(os.path.join(directory, name)):
            record.setdefault("_source_file", name)
            records.append(record)
    return records


def failed(status: str, error: BaseException) -> dict:
    """The two error fields, from an exception. Kept here so every campaign
    records a failure the same way."""
    return {
        "status": status,
        "error_type": type(error).__name__,
        "error_message": str(error)[:4000],
    }


def now_fields(started_monotonic: float, started_wall: str) -> dict:
    return {
        "started_at": started_wall,
        "finished_at": utc_now(),
        "total_seconds": time.monotonic() - started_monotonic,
    }


def new_record(**fields: Any) -> RunRecord:
    return RunRecord(**fields)
