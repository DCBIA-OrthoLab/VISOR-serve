#!/usr/bin/env python3
"""The half of a tool run that happens inside the tool's own interpreter.

    /tools/<name>/.venv/bin/python /opt/sadt/runner.py --job /jobs/<id>/job.json

Read the job file, import the tool, call its run(), write result.json. That is
the whole job -- plus one thing: a tool declaring `*, sup` is handed a
**supervisor**, and can call another tool through it. That call re-enters this
same file with the sibling's interpreter, so chaining and nesting are one
recursion rather than a feature.

Three constraints shape this file, and all three are load-bearing:

- **Standard library only.** It runs inside a TOOL's virtualenv, which contains
  whatever that tool needs and nothing of the server's. It cannot import
  fastapi, pydantic, or anything else from server/.
- **Python 3.9 through 3.13.** Each tool pins its own interpreter, so this file
  is executed by all of them. No match statements, no `X | Y` annotations
  outside the `from __future__` below.
- **It ships with the SERVER, not with the tools, and is injected by path.** It
  is deliberately not a package installed into each venv: runner and server are
  then always the same version, and there is no cross-repo version skew to
  negotiate.

On failure it exits non-zero, prints the traceback to stderr, and writes the
exception's CLASS NAME into the result file -- `{"error": {"type": ...}}`.
There is no shared exception type to catch, because there is no shared package,
so the name is what tells the server whether the caller sent something wrong
(422) or the tool itself broke (500). A successful result is written whole and
replaced into place, so a half-written one is never read as a result.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import os
import subprocess
import threading
import shutil
import sys
import time
from time import monotonic
import traceback
import typing
from pathlib import Path

# Where the tool's code lives inside its folder, and the environment variable
# that overrides the folder itself (tests and dev checkouts, where the venv is
# not necessarily next to the sources).
SRC_DIR_NAME = "src"
TOOL_DIR_ENV = "SADT_TOOL_DIR"

RESULT_FILE = "result.json"
JOB_FILE = "job.json"

# Where a VRAM figure came from, written into result.json beside the figure
# itself. The two sources are not equally trustworthy and only the SERVER can
# tell which one it may believe -- it alone knows whether this run had the
# machine to itself -- so the runner states the provenance and decides nothing.
VRAM_SOURCE_KEY = "vram_source"
VRAM_FROM_TORCH = "torch"   # this process's own allocator, exact
VRAM_FROM_CARD = "card"     # the whole card's growth, contaminated by neighbours
VRAM_FROM_NONE = "none"     # no figure at all

# The supervisor: how a tool calls ANOTHER tool. Keyword-only and unannotated,
# which is what `describe.py` reads to keep it out of the published schema -- a
# client never sends one, because it is not data.
SUPERVISOR_ARGUMENT = "sup"

# The read-only DATA root, injected the same way and for the same reason: a tool
# that needs its OWN model bundle can find it, without the path being a value
# anyone passes in. Declared exactly as `sup` is -- keyword-only, unannotated --
# so `describe.py` keeps it out of the schema and no client can name it.
#
# It exists because of supervised calls. The server resolves a hosted-model
# argument on the way in, but a supervised call is not a request and never
# passes that way, which used to leave the CALLER naming its neighbour's bundle
# -- ASO composing a path into ALI's data folder. Now ALI answers that itself,
# from its own name, and ASO passes only the landmarks it wants.
DATA_ROOT_ARGUMENT = "data_root"

# Where a tool writes. Over HTTP no client can name it: the server strips it from
# the published schema and fills it with the job's own `output/`. A SUPERVISED
# call is not a request but has the same property -- where a callee writes is the
# infrastructure's business, not the caller's -- so the supervisor drops any
# `output_dir` a caller passed and the callee's own runner fills it in the same
# way, from its own job directory.
#
# It is injected here rather than passed down because a tool that declares no
# `output_dir` would fail on `run(**params)` if one arrived anyway. Only the
# runner that imported the tool can see whether it takes one.
OUTPUT_DIR_ARGUMENT = "output_dir"
OUTPUT_DIR_NAME = "output"

# Set by a caller that wants what the tools BELOW this one produced, not only
# this tool's own results. Declared by any tool that makes supervised calls; the
# tool itself does nothing with it, because collecting is the same operation for
# every chain and is done once, here, after run() returns.
KEEP_INTERMEDIATE_ARGUMENT = "keep_intermediate"

# Where the collected outputs land inside the caller's own output directory.
INTERMEDIATE_DIRNAME = "intermediate"

# The supervisor's own subtree of a job directory: one numbered folder per
# nested call, which is what gives the collected results a name a reader can
# match to the order the chain ran in.
SUP_DIRNAME = "sup"

# "every step", for the spellings that cannot name them: a direct Python call
# passing True, where listing the chain would mean knowing it. Matched by
# IDENTITY, so its contents never stand for a tool -- but non-empty, because an
# empty set is the answer for "keep nothing" and the two must not be confused.
_ALL_STEPS = frozenset({"*"})

# A backstop, and only that. The real cycle protection is the CHAIN check below,
# which refuses a tool already running above the call and names it; this catches
# the other shape -- a chain that grows without ever repeating -- and nothing
# else. That is why the number can be generous: the deepest real chain is
# AREG_IOSCBCT -> ASO -> ALI_CBCT, three levels, and depth costs almost nothing.
#
# Measured: an orchestrating tool holds ~12 MB (AREG_IOSCBCT imports no torch),
# a leaf ~500 MB, and a chain is SEQUENTIAL -- each sup.run() waits for its
# child -- so one heavy process lives at a time per chain whatever the depth.
# What multiplies memory is MAX_CONCURRENT_TOOLS, not this.
SUPERVISOR_DEPTH_ENV = "SADT_SUPERVISOR_DEPTH"
MAX_SUPERVISOR_DEPTH = 10

# When the whole job must be finished by, as a monotonic-clock deadline, passed
# down so every level can give its child only the time it has left. See
# _remaining_seconds.
SUPERVISOR_DEADLINE_ENV = "SADT_SUPERVISOR_DEADLINE"

# The file a tool appends its progress to, absolute, set by the server only when
# the run has a directory to hold it. Inherited by every supervised child, so a
# chain writes one file and `depth` is what tells the levels apart -- which is
# the whole implementation of chain progress.
#
# It is NOT the recommended way for a tool to report progress. A tool appends to
# this path itself, with its own ~15-line helper; only the four tools that
# already take a supervisor reach it through sup.progress(), and they keep
# working unchanged. See RUN_PROGRESS.md.
PROGRESS_FILE_ENV = "SADT_PROGRESS_FILE"

# One record, one write(), and under PIPE_BUF so the append is atomic against
# every other level of the chain. 200 characters of message cannot approach it;
# the check exists because the message comes from a tool.
MAX_PROGRESS_MESSAGE = 200
MAX_PROGRESS_RECORD_BYTES = 4096

# The disk backstop, and the only one this side can enforce: a tool's process
# cannot know how many events the run already holds -- several processes of one
# chain append to the same file -- so it bounds the BYTES it can add rather than
# the count. The server applies the real MAX_RUN_EVENTS cap when it reads.
MAX_PROGRESS_FILE_BYTES = 8 * 1024 * 1024

# The tools already on the stack, innermost last, as one comma-separated value.
# It travels in the environment rather than in job.json because it belongs to
# the CALL, not to the job: the same tool run directly and run as a child reads
# a different chain, and job.json is what a caller writes.
SUPERVISOR_CHAIN_ENV = "SADT_SUPERVISOR_CHAIN"

# The tool's package under src/, when there is exactly one. SADT-VISOR names it
# `sadt_<tool>`, but the rule is the one its own describe.py uses -- the single
# importable package -- rather than the prefix, so a tool that names its
# package something else still loads.
_FALLBACK_PREFIX = "sadt_"


# Exception names a tool raises to answer the CALLER rather than to report a
# crash. The server maps these to 422 with the message passed through, so a
# supervised child raising one has to reach it under its own name -- see
# _supervised. Kept in step with main.TOOL_ERROR_STATUS by a test.
CALLER_FACING_ERRORS = ("ToolInputError", "ValueError", "FileNotFoundError",
                        "ToolUnavailableError")


class RunnerError(Exception):
    """Anything that stops this script before the tool's run() is reached."""


def _tool_dir() -> str:
    """The tool's folder: /tools/<name>/, holding src/ and .venv/.

    Derived from the interpreter we are running in rather than from the job
    file, because that is the one thing the invocation already fixes: the
    server picked this venv precisely because it is the tool's. Deriving it
    keeps job.json to the four fields the contract declares.
    """
    override = os.environ.get(TOOL_DIR_ENV)
    if override:
        return os.path.abspath(override)

    if sys.prefix == sys.base_prefix:
        raise RunnerError(
            "This interpreter is not a virtualenv, so the tool folder cannot be derived "
            "from it. Run the tool's own /tools/<name>/.venv/bin/python, or set "
            f"{TOOL_DIR_ENV}."
        )
    # <tool dir>/.venv/bin/python -> sys.prefix is <tool dir>/.venv
    return os.path.dirname(os.path.abspath(sys.prefix))


def _package_under(src_dir: str):
    """The one importable package under src/, or None.

    Same rule as SADT-VISOR's own describe.py: exactly one directory holding an
    __init__.py. Deriving it rather than hardcoding `sadt_<tool>` means the
    runner and the schema generator agree on what they load, which is the only
    way the schema can describe what actually runs.
    """
    candidates = sorted(
        entry
        for entry in os.listdir(src_dir)
        if os.path.isfile(os.path.join(src_dir, entry, "__init__.py"))
    )
    return candidates[0] if len(candidates) == 1 else None


def _import_tool(tool_name: str, src_dir: str):
    """Import the module defining run(), from the tool's src/.

    `uv sync` installs the tool into its own virtualenv, so the package is
    importable without any of this; src/ goes on the path first anyway, so the
    runner also works against a checkout that was never synced.
    """
    if not os.path.isdir(src_dir):
        raise RunnerError(f"Tool '{tool_name}' has no '{SRC_DIR_NAME}/' directory at {src_dir}")

    sys.path.insert(0, src_dir)
    package = _package_under(src_dir)
    candidates = [name for name in (package, f"{_FALLBACK_PREFIX}{tool_name}", tool_name, "tool") if name]

    tried = []
    for candidate in dict.fromkeys(candidates):
        if not candidate.isidentifier():
            tried.append(f"{candidate} (not a valid module name)")
            continue
        try:
            module = importlib.import_module(candidate)
        except ImportError as exc:
            # Only a MISSING module moves on to the next candidate. An
            # ImportError raised from inside the module (a missing dependency
            # of the tool itself) is the answer, not a reason to keep looking.
            if getattr(exc, "name", None) != candidate:
                raise
            tried.append(f"{candidate} ({exc})")
            continue
        if package is not None:
            _reject_module_from_elsewhere(module, candidate, src_dir)
        return module

    raise RunnerError(
        f"Tool '{tool_name}': no module defining run() found in {src_dir}. Tried: "
        + ", ".join(tried)
    )


def _reject_module_from_elsewhere(module, name: str, src_dir: str) -> None:
    """Refuse a module that came from anywhere but the tool's src/.

    Without this, a tool whose name collides with a standard-library module
    imports the standard library one and fails much further down, with an error
    naming neither.
    """
    origin = getattr(module, "__file__", None)
    if origin is None or os.path.commonpath(
        [os.path.abspath(src_dir), os.path.abspath(origin)]
    ) != os.path.abspath(src_dir):
        raise RunnerError(
            f"Module '{name}' was imported from {origin!r}, outside the tool's "
            f"{SRC_DIR_NAME}/ directory. Rename the tool's module so it does not collide."
        )


def _run_function(module, tool_name: str):
    run = getattr(module, "run", None)
    if not callable(run):
        raise RunnerError(
            f"Tool '{tool_name}': module '{module.__name__}' defines no callable run()."
        )
    return run


def _coerce(value, annotation):
    """Turn a JSON value into what run()'s annotation asks for.

    Only paths need it: JSON has no path type, so they travel as strings and a
    tool annotating `scans: Path` would receive a `str` and fail on the first
    `.glob()`. Everything else the schema allows -- str, int, float, bool and
    lists of them -- arrives as the right type already.

    Deliberately identical to SADT-VISOR's testkit driver
    (`testkit/src/sadt_testkit/_driver.py`): every tool's integration tests run
    against that one, so a difference here is a suite that passes while
    production fails.

    An empty string is ABSENCE and stays a string. `Path("")` is
    `PosixPath(".")` -- the current directory, and truthy -- so coercing the
    "not supplied" default of an optional path hands the tool a real directory
    to walk. `ASO` read an unset `landmarks=""` as a supplied landmark folder
    and walked its whole checkout. Calling `run()` directly in Python keeps the
    `""` the signature declares, so coercing it is also what makes the two call
    paths disagree.
    """
    if annotation is Path:
        return Path(value) if value != "" else value
    if typing.get_origin(annotation) is list and typing.get_args(annotation) == (Path,):
        return [Path(item) for item in value]
    return value


def _call_arguments(run, params: dict) -> dict:
    try:
        hints = typing.get_type_hints(run)
    except Exception:  # noqa: BLE001 - an unresolvable annotation is not fatal
        hints = {}
    return {name: _coerce(value, hints.get(name)) for name, value in params.items()}


def _jsonable(value):
    """json.dump fallback: a returned Path is a path, anything else is a bug.

    Narrow on purpose. `default=str` would serialize any object at all as its
    repr, so a tool returning something the server cannot use would look like a
    successful run right up until the server tried to open it.
    """
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError(
        f"run() returned a {type(value).__name__}, which cannot be written to "
        f"{RESULT_FILE}. Return JSON-serializable values (paths as strings)."
    )


class _TreeSampler:
    """High-water mark of what this run's whole PROCESS GROUP holds.

    The two figures it replaces were both measured on the runner process alone,
    and both were wrong the moment a tool did its work in children:

    - `torch.cuda.max_memory_reserved()` counts what THIS interpreter
      allocated. ALI_CBCT searches a scan's landmarks in parallel processes, so
      the runner allocates nothing on the card and reported no VRAM at all --
      not a low figure, an absent one.
    - `RUSAGE_CHILDREN.ru_maxrss` is the largest single child ever seen, never
      the sum of several alive at once. Four workers holding 1.5 GiB each
      reported 2.38 GiB against 6.9 GiB actually resident.

    Both errors are in the direction that ends in an out-of-memory: the server
    grants a width, measures almost nothing, and admits the next runs against
    that. A tool being honest about what it costs is no use if the instrument
    in front of it is not.

    Matched by process GROUP rather than by walking parent links. `dispatch`
    starts a tool with `start_new_session=True`, so every descendant shares the
    group -- including a grandchild a tool shelled out to, which a parent-link
    walk finds only if it is still attached.

    Sampling, not accounting, so it can miss a spike shorter than its interval.
    That is the accepted trade: the alternative is a counter inside every tool,
    in twenty-two virtualenvs this server does not import.
    """

    INTERVAL_SECONDS = 2.0

    def __init__(self):
        self.peak_rss = 0
        self.peak_vram = 0
        # Cores are a RATE, so they need two readings and an interval. The mark
        # is (when, ticks) from the previous sample.
        self.peak_cores = 0.0
        self._cpu_mark = None
        self._ticks_per_second = float(
            os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        )
        # What the card already held when this run started, so the fallback
        # measures this run's growth rather than the host's history.
        self._card_baseline = None
        # Once the per-process listing has answered even once, it is the better
        # figure and the fallback is never mixed into it.
        self._per_process_worked = False
        self._stop = threading.Event()
        self._thread = None

    # -- one sample ----------------------------------------------------
    def _group(self) -> set:
        """Every pid in this run's process group."""
        try:
            mine = os.getpgrp()
        except OSError:
            return set()
        found = set()
        try:
            entries = os.listdir("/proc")
        except OSError:
            return found  # not Linux, or /proc not mounted
        for entry in entries:
            if not entry.isdigit():
                continue
            try:
                with open("/proc/%s/stat" % entry, "rb") as handle:
                    fields = handle.read().rsplit(b")", 1)[-1].split()
                # After the comm field: state, ppid, pgrp, ...
                if int(fields[2]) == mine:
                    found.add(int(entry))
            except (OSError, IndexError, ValueError):
                continue  # it exited between the listing and the read
        return found

    def _cpu_ticks(self, pids) -> int:
        """Total CPU time this group has burned, in clock ticks.

        `utime + stime` per process, summed. A DIFFERENCE of two of these over
        a wall-clock interval is how many cores the group was actually keeping
        busy -- the number the cost table has never had, and whose absence is
        why a run reserving ten cores could occupy fifty.
        """
        total = 0
        for pid in pids:
            try:
                with open("/proc/%d/stat" % pid, "rb") as handle:
                    fields = handle.read().rsplit(b")", 1)[-1].split()
                # After the comm field: state, ppid, pgrp, session, tty_nr,
                # tpgid, flags, minflt, cminflt, majflt, cmajflt, utime, stime.
                total += int(fields[11]) + int(fields[12])
            except (OSError, IndexError, ValueError):
                continue
        return total

    def _rss(self, pids) -> int:
        page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
        total = 0
        for pid in pids:
            try:
                with open("/proc/%d/statm" % pid, "rb") as handle:
                    total += int(handle.read().split()[1]) * page
            except (OSError, IndexError, ValueError):
                continue
        return total

    def _card_used(self):
        """What the whole card holds, or None. The fallback's raw material."""
        try:
            done = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if done.returncode != 0:
            return None
        try:
            line = done.stdout.decode("utf-8", "replace").strip().splitlines()[0]
            return int(line.strip()) * 1024 * 1024
        except (IndexError, ValueError):
            return None

    def _vram(self, pids) -> int:
        """Per-process card memory, from nvidia-smi's compute-app listing.

        Per PROCESS, not per card: the card's total would include whatever else
        the host is running, which is exactly what a cost table must not learn.

        Returns 0 when the listing is empty, which is a real case rather than a
        theoretical one -- this deployment's driver answers
        `--query-compute-apps` with nothing at all, on the host as well as in
        the container. `_fallback_vram` is what covers it.
        """
        try:
            done = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                 "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return 0
        if done.returncode != 0:
            return 0
        total = 0
        for line in done.stdout.decode("utf-8", "replace").splitlines():
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                if int(parts[0].strip()) in pids:
                    total += int(parts[1].strip()) * 1024 * 1024
            except ValueError:
                continue
        return total

    def _sample_cores(self, pids) -> None:
        """How many cores this group kept busy since the last sample.

        The high-water mark, not the average: what a reservation has to hold is
        the busiest the run ever got, the same rule the memory figures follow.

        The first sample only establishes the baseline -- a rate needs two
        readings -- and a very short interval is ignored, because dividing a
        tick count by a few milliseconds turns rounding into a number like
        forty cores on a machine that has four.
        """
        now = monotonic()
        ticks = self._cpu_ticks(pids)
        if self._cpu_mark is not None:
            elapsed = now - self._cpu_mark[0]
            if elapsed >= 0.5:
                burned = (ticks - self._cpu_mark[1]) / self._ticks_per_second
                self.peak_cores = max(self.peak_cores, burned / elapsed)
            elif elapsed > 0:
                return  # too short to divide by; keep the older mark
        self._cpu_mark = (now, ticks)

    def _fallback_vram(self) -> int:
        """What the CARD grew by since this run started, when nothing better exists.

        Used only where the per-process listing is unavailable -- which is this
        deployment, whose driver answers it with nothing. It is an UPPER bound
        and it is contaminated: another job allocating at the same time is
        counted here too.

        That direction is chosen deliberately. Reporting nothing lets a tool
        whose work happens in children be admitted as if it used no card at
        all, and the server then stacks runs onto a card it believes is empty.
        Reporting too much makes it reserve more than it needs, which costs
        parallelism and nothing else. Between an out-of-memory in somebody's
        cohort and a slower queue, the queue wins.

        Clamped at zero, because another job FREEING memory during this run
        would otherwise make the delta negative and read as "cost nothing" --
        the one outcome this exists to prevent.
        """
        used = self._card_used()
        if used is None or self._card_baseline is None:
            return 0
        return max(0, used - self._card_baseline)

    def sample(self) -> None:
        pids = self._group()
        if not pids:
            return
        self._sample_cores(pids)
        self.peak_rss = max(self.peak_rss, self._rss(pids))
        measured = self._vram(pids)
        if measured:
            self.peak_vram = max(self.peak_vram, measured)
            self._per_process_worked = True
        elif not self._per_process_worked:
            self.peak_vram = max(self.peak_vram, self._fallback_vram())

    # -- the thread ----------------------------------------------------
    def start(self) -> "_TreeSampler":
        def loop():
            while not self._stop.wait(self.INTERVAL_SECONDS):
                try:
                    self.sample()
                except Exception:  # noqa: BLE001 - instrumentation, never fatal
                    pass

        try:
            self._card_baseline = self._card_used()
            self.sample()  # a run shorter than one interval still gets a figure
            self._thread = threading.Thread(target=loop, name="sadt-sampler",
                                            daemon=True)
            self._thread.start()
        except Exception:  # noqa: BLE001
            self._thread = None
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(self.INTERVAL_SECONDS * 2)
        try:
            self.sample()  # the tail, after the tool has finished allocating
        except Exception:  # noqa: BLE001
            pass


_sampler = _TreeSampler()


def _touched_torch() -> bool:
    """Did this interpreter import torch at all?

    `sys.modules` and nothing else -- the same test `_peak_vram_bytes` makes,
    and for the same reason: importing torch to ask would cost seconds and a
    CUDA context on every run of every tool that has no use for either.

    A tool whose GPU work happens in CHILD processes still imports torch here
    (ALI_CBCT builds its Environment through monai before any worker starts),
    which is what keeps the fallback available to the case it exists for.
    """
    return sys.modules.get("torch") is not None


def _peak_vram_bytes():
    """Peak CUDA memory this process allocated, or None.

    `sys.modules.get` rather than an import: a tool that never touched torch
    stays untouched, and importing it here would cost seconds and a CUDA
    context on every tabular run. If torch is not in sys.modules, the tool did
    not use it, by definition.

    Every failure is swallowed on purpose. This is instrumentation -- it exists
    so a real VRAM budget can be set later from measurements instead of
    guesses -- and instrumentation must never be able to fail a run that
    otherwise succeeded.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        # RESERVED, not allocated. The caching allocator keeps what it has taken
        # from the driver instead of giving it back, so the card is short of
        # `reserved` while `allocated` reports only what the tool was holding at
        # the peak -- and under-estimating is the direction that ends in an
        # out-of-memory in the middle of somebody's cohort. `allocated` stays
        # beside it: it is the tool's own high-water mark, and the difference
        # between the two is how much the allocator is sitting on.
        return max(
            int(torch.cuda.max_memory_reserved()),
            int(torch.cuda.max_memory_allocated()),
        )
    except Exception:  # noqa: BLE001 - see above
        return None


def _peak_rss_bytes():
    """Peak resident memory of this process and its children, or None.

    Host RAM was never measured, and on the machine this was written for it is
    the resource that binds BEFORE the card: 28 physical cores and 125 GB
    against a 48 GiB GPU whose heaviest tool peaks at 2.19 GiB. Admitting on
    VRAM alone would let far more jobs in than the host can hold.

    `resource` is standard library and present in every tool venv by
    construction, so this costs no dependency anywhere. RUSAGE_CHILDREN is
    included because a supervised call is a child process: without it an
    orchestrator that spends its life inside `sup.run` reports its own twelve
    megabytes and nothing else.
    """
    try:
        import resource as _resource

        peak_kb = max(
            _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss,
            _resource.getrusage(_resource.RUSAGE_CHILDREN).ru_maxrss,
        )
        # Linux reports kilobytes; macOS reports bytes. Only Linux runs here,
        # but a benchmark harness on a laptop should not record a 1000x figure.
        return int(peak_kb) if sys.platform == "darwin" else int(peak_kb) * 1024
    except Exception:  # noqa: BLE001 - instrumentation never fails a run
        return None


# The worst VRAM a supervised child was measured to hold, folded up as each
# call returns. A module global because one runner process IS one job: there is
# no second chain in here to confuse it with.
#
# Only VRAM. Host memory needs no folding -- RUSAGE_CHILDREN already reports the
# high-water mark across every child this process waited for, so adding the
# children's own figures would count them twice.
_child_vram_bytes = 0
# And where THAT figure came from. A chain is only as trustworthy as the
# weakest reading in it: ASO's own interpreter allocates nothing, so its whole
# VRAM figure is ALI's, and reporting the sum as "torch" because the addition
# happened in this process would hand the server a card reading wearing the
# label it trusts unconditionally. ASO is exactly the tool that reached
# 32.43 GiB per channel in the table this guard exists to fix.
_child_vram_source = VRAM_FROM_NONE


def _fold_child_measurement(payload: dict) -> None:
    """Keep the worst VRAM any child of this process reported, and its source."""
    global _child_vram_bytes, _child_vram_source
    try:
        reported = int(payload.get("peak_vram_bytes") or 0)
    except (TypeError, ValueError):
        return
    if reported > _child_vram_bytes:
        _child_vram_bytes = reported
        _child_vram_source = str(payload.get(VRAM_SOURCE_KEY) or VRAM_FROM_NONE)


def _width_reached() -> int:
    """The NARROWEST width this process reported, from its own progress records.

    A tool says how wide it is by putting `width` on the progress records it
    already writes. Only records at THIS process's depth count -- a chain
    writes to one file, and a nested call's width belongs to the nested call,
    which reports its own.

    **The smallest, not the largest, and that is the whole subtlety.** A peak
    and a width are not measured at the same instant. AMASSS reads its cohort
    four at a time, then runs nnUNet one structure at a time, then assembles
    four at a time -- and its VRAM peak is in the SERIAL middle. Taking the
    largest width would divide an inference-sized peak by four and teach the
    table that a channel costs a quarter of what it does, which is the same
    defect as reading the grant, one step further in.

    The smallest is right because it is safe: dividing a peak by a width that
    was not in force when it happened under-reserves, and dividing by a
    narrower one over-reserves. A tool whose heavy phase is serial says so by
    reporting `width 1` around it, and then this is exact rather than merely
    conservative.

    **What the width is FOR has changed, and a tool that under-declares now
    pays for it twice.** `costs.py` no longer divides a peak by this number:
    it fits `fixed + marginal x width` across the widths a tool has been seen
    at, so this is a coordinate rather than a divisor, and a tool that reports
    `width 1` on every run gives the fit one column of points and no way to
    separate its intercept from its slope. It is still the SAFE answer -- the
    whole peak is then treated as one channel's -- but a tool whose heavy
    phase really is two channels wide should now say two, and get the honest
    line instead of the conservative one.

    A record with no width is IGNORED, not read as one: a tool writes progress
    before it knows how wide it will be, and counting those would pin every
    tool at one for ever.

    Falls back to ONE when nothing was reported at all -- never to what the
    server granted. Reading a grant as a measurement is the defect this exists
    to close.
    """
    path = os.environ.get(PROGRESS_FILE_ENV)
    if not path:
        return 1
    mine = None
    try:
        depth = int(os.environ.get(SUPERVISOR_DEPTH_ENV, "0") or 0)
    except (TypeError, ValueError):
        depth = 0
    try:
        with open(path, "rb") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue  # a half-written line is one record, never the run
                if not isinstance(record, dict):
                    continue
                if "width" not in record:
                    continue  # said nothing, rather than said one
                if int(record.get("depth") or 0) != depth:
                    continue
                try:
                    width = int(record["width"])
                except (TypeError, ValueError):
                    continue
                if width < 1:
                    continue
                mine = width if mine is None else min(mine, width)
    except OSError:
        return 1
    return 1 if mine is None else mine


def _measurements() -> dict:
    """What this process cost, for the server's cost table.

    Written on the failure path as well as the success one, and that is not
    symmetry for its own sake: an out-of-memory is the single most informative
    run a budget could learn from, and it was the one run that recorded nothing.

    **A chain reports what the chain cost.** Each level already measured itself
    into its own result.json and nobody ever read them, so a supervised run was
    invisible to any budget -- which is exactly the hole CLAUDE.md names. The
    fold is `own + worst child` rather than a sum because `sup.run` blocks:
    children are strictly sequential, so only one of them is ever resident, and
    the parent's own peak is added because an orchestrator may still be holding
    something while its child runs. That makes it an upper bound, and exact for
    the orchestrators here, which import no torch at all.
    """
    measured = {}
    # How widely this run actually spread. Reported because the bytes above
    # mean nothing without it: a tool measured at four channels and one
    # measured at one are not the same measurement, and a table that stored
    # them as if they were would under-reserve the next wide run.
    #
    # What the tool OPENED, not what the server granted. A tool opens
    # `min(granted, items)` -- every one of them does -- so a run granted five
    # channels with one scan opens one, costs what one costs, and recording it
    # as five made the table believe a channel was a fifth as expensive as it
    # is. Observed on this deployment: AMASSS at 2.41 GiB went into the table
    # as 0.48 GiB per channel.
    measured["channels"] = _width_reached()
    _sampler.stop()
    # torch's own counter FIRST, and the sampler only where it has nothing.
    #
    # They are not two views of one number. torch's is exact and belongs to
    # this process; the sampler's fallback is the card's growth, which counts
    # whatever else the host allocated while this ran. Taking the larger let
    # that contamination win: Batch_Dental_Seg, which really costs 4.77 GiB and
    # whose nnUNet runs in THIS interpreter, went into the table at 38.40 GiB
    # per channel -- more than the whole budget, so nothing could ever be
    # admitted beside it.
    #
    # The fallback exists for the opposite case and only that one: a tool whose
    # GPU work happens in children, where torch here sees nothing at all.
    vram = _peak_vram_bytes()
    own = (vram or 0) + _child_vram_bytes
    # The fallback is the CARD's growth, so it counts whatever else the host
    # allocated while this ran. Offered only to a run that could plausibly have
    # allocated on the card at all -- meaning torch is loaded in this
    # interpreter -- because a tool that never imported it provably did not.
    #
    # Without that gate a tabular tool picks up a neighbour's allocation and
    # the server reserves card memory for something that never touches the
    # card. It also made the test that pins this FLAKY: passing on an idle
    # machine, failing whenever anything else was running.
    chain_vram = own if own else (_sampler.peak_vram if _touched_torch() else 0)
    if chain_vram:
        measured["peak_vram_bytes"] = chain_vram
    # WHICH of the two produced it, said out loud. A source, not a confidence
    # score: the server decides what to do with it, and it is the only side
    # that can, because trusting the card's growth depends on whether anything
    # else was allocating at the time -- which this process cannot see.
    #
    # Measured on this deployment, ALI_CBCT alone on an idle machine: 0.62 G at
    # one channel, 7.80 G at seven, 16.74 G at fifteen -- 1.12 G per channel,
    # flat. The same tool learned through a CONTENDED card reading went into
    # the table at 2.2 G and, earlier the same day, 5.66 G; ASO reached 32.43 G
    # per channel. The card figure is not noisy, it is systematically too
    # large, because it is everybody's allocation attributed to each of them.
    source = (VRAM_FROM_TORCH if own else
              (VRAM_FROM_CARD if chain_vram else VRAM_FROM_NONE))
    # A chain reports the weakest source it is built from. `own` is this
    # process's torch figure PLUS the worst child's, and a child that had to
    # read the card contaminates the sum it is part of.
    if source == VRAM_FROM_TORCH and _child_vram_bytes and \
            _child_vram_source == VRAM_FROM_CARD:
        source = VRAM_FROM_CARD
    measured[VRAM_SOURCE_KEY] = source
    # How many cores the run actually kept busy. Nothing measured this before,
    # and its absence is why a run that reserved ten cores could occupy fifty:
    # ALI_CBCT at eight channels filled 51.7 of 56 while admission believed it
    # had admitted a ten-core job.
    if _sampler.peak_cores > 0:
        measured["peak_cpu_cores"] = round(_sampler.peak_cores, 2)
    rss = _peak_rss_bytes()
    rss = max(rss or 0, _sampler.peak_rss) or None
    if rss is not None:
        measured["peak_rss_bytes"] = rss
    return measured


def _write_result(job_dir: str, result) -> None:
    """Write {"result": ...} atomically.

    Serialized in full BEFORE anything touches the disk, so an unserializable
    result is a clean failure rather than a truncated result.json; and replaced
    into place, so the server never reads one being written.
    """
    body = {"result": result}
    body.update(_measurements())
    payload = json.dumps(body, default=_jsonable)
    final_path = os.path.join(job_dir, RESULT_FILE)
    temp_path = final_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        handle.write(payload)
    os.replace(temp_path, final_path)


def _write_error(job_dir: str, exc: Exception) -> None:
    """Record WHICH failure happened, next to where a result would have gone.

    There is no shared exception type -- there is no shared package -- so the
    server maps the class NAME: ToolInputError/ValueError/FileNotFoundError are
    the caller's problem (422, message passed through), ToolUnavailableError is
    the deployment's (503), anything else is opaque (500). The exit code stays
    non-zero either way, so a caller that reads nothing but that still sees a
    failure.
    """
    payload = {"error": {"type": type(exc).__name__, "message": str(exc)}}
    # The measurements travel with the failure too. A run that died of an
    # out-of-memory is the most informative thing a budget can learn from, and
    # it used to be the one run that recorded nothing at all.
    payload.update(_measurements())
    try:
        with open(os.path.join(job_dir, RESULT_FILE), "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except OSError:
        pass  # the traceback on stderr is still the record


def _configure_logging() -> None:
    """The runner owns handlers, levels and formatting.

    Tools attach none -- a library that configures logging takes the decision
    away from whatever runs it -- so without this their log records go nowhere
    at all.
    """
    logging.basicConfig(
        level=os.environ.get("SADT_LOG_LEVEL", "INFO").upper(),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def _append_progress(fraction, message, depth: int, tool: str = "") -> None:
    """Append one progress record to SADT_PROGRESS_FILE, if there is one.

    `tool` is set only by the supervisor, to mark a nested call opening and
    closing at the CHILD's depth -- see `Supervisor._run_nested`.

    Best effort in the strictest sense: nothing here may reach the tool. A run
    that cannot report its progress is a run, and turning a failed `write` into
    a failed cohort would be an absurd trade.

    Three things make the append safe with several processes of one chain
    writing to the same file, and all three are load-bearing:

    - `O_APPEND`, never a seek. Two writers cannot land on the same offset.
    - ONE `write()` of under PIPE_BUF, which POSIX makes atomic, so a record
      never interleaves with another level's.
    - No `O_CREAT`. The server creates the file when it registers the run; a
      stale variable inherited from somewhere else must not cause this to
      litter whatever it points at.

    The file's SIZE is the only cap this side can apply. How many events a run
    already holds is not knowable from here -- a chain appends from several
    processes -- so the count is capped by the server when it reads, and the
    bytes are capped here, which is what actually protects the disk.
    """
    path = os.environ.get(PROGRESS_FILE_ENV)
    if not path:
        return
    try:
        try:
            value = float(fraction)
            if value != value or value in (float("inf"), float("-inf")):
                value = None
            else:
                value = min(1.0, max(0.0, value))
        except (TypeError, ValueError):
            value = None
        text = "" if message is None else str(message)
        # `at` and `depth`, and nothing else the server would only override:
        # a tool's line says how far along it is, never what phase the RUN is
        # in. These two are facts only this side knows -- the depth of a
        # supervised level, and when the line was actually written rather than
        # when a poll happened to notice it.
        record = {
            "at": time.time(),
            "fraction": value,
            # A newline inside a record would be read as the end of it, and a
            # message is free text written by a tool.
            "message": text.replace("\r", " ").replace("\n", " ")[:MAX_PROGRESS_MESSAGE],
            "depth": depth,
        }
        # Only on a supervisor's nested marker. It is a TOOL NAME, so it is
        # emitted as its own field rather than inside `message`: a message is
        # free text a tool wrote and may name a patient's file, which is why
        # the benchmark payload drops messages entirely. A name checked
        # against a strict identifier on the way out (`wire/runs`) carries the
        # one fact the drawing needs and cannot carry anything else.
        if tool:
            record["tool"] = str(tool)[:MAX_PROGRESS_MESSAGE]
        payload = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
        if len(payload) > MAX_PROGRESS_RECORD_BYTES:
            return
        handle = os.open(path, os.O_WRONLY | os.O_APPEND)
        try:
            if os.fstat(handle).st_size + len(payload) > MAX_PROGRESS_FILE_BYTES:
                return
            os.write(handle, payload)
        finally:
            os.close(handle)
    except Exception:  # noqa: BLE001 - progress must never break a run
        pass


# ---------------------------------------------------------------------------
# The supervisor
# ---------------------------------------------------------------------------

def _takes_data_root(run) -> bool:
    """Does this run() ask for the DATA root? Same shape rule as `sup`."""
    return _declares_injected(run, DATA_ROOT_ARGUMENT)


def _takes_supervisor(run) -> bool:
    """Does this run() ask for a supervisor?

    The same rule `describe.py` uses to keep it out of the schema: a parameter
    named `sup`, keyword-only, and UNANNOTATED. Unannotated is the marker rather
    than an accident -- every other parameter must be annotated, so there is
    nothing else this shape could be, and a tool cannot grow a supervisor by
    forgetting a type.
    """
    return _declares_injected(run, SUPERVISOR_ARGUMENT)


# How many of its own items this run may process at once, and how much room it
# holds while it does. Both are set by the server at dispatch (execution/
# concurrency.py); a tool run outside a server finds neither and stays serial.
#
# `CHANNELS_ENV` is a count and `CHANNEL_BUDGET_ENV` is ROOM, `"<vram>,<ram>"`
# in bytes. The pair is what makes a level's width its own: a top-level run is
# told its width, because admission reserved against it; a nested one is told
# only the room, and works its width out from what IT was measured to cost.
CHANNELS_ENV = "SADT_CHANNELS"
CHANNEL_BUDGET_ENV = "SADT_CHANNEL_BUDGET"
# Where the server keeps what each tool was measured to cost, and which
# argument each tool's width is counted from. Both are server knowledge a
# nested level needs and cannot import: this file ships WITH the server and is
# injected by path, so reading them is the same repository at the same version,
# but it executes inside a tool's virtualenv on any interpreter from 3.9 to
# 3.13 -- hence a path and `json`, never `from execution import costs`.
COST_TABLE_ENV = "SADT_COST_TABLE"
WIDTH_AXIS_ENV = "SADT_WIDTH_AXIS"
MAX_CHANNELS_ENV = "SADT_MAX_CHANNELS"
CHANNEL_ARGUMENTS = ("num_workers", "batch_size")


# How wide this process may be running at any one instant -- the divisor the
# supervisor hands a child its share by. One number with one meaning, from two
# sources with a stated precedence rather than two opinions about it.
#
# `_WIDTH_GRANTED` is what the run was HANDED: the number the server wrote into
# `num_workers`, or the caller's own, or the tool's declared default. It is the
# only answer available for a tool that has not been migrated, which is most of
# the catalogue -- such a tool opens what it was handed and tells nobody.
#
# `_WIDTH_ASKED` is what `sup.channels()` ANSWERED, and None until it is called
# at all. Calling it is the tool saying it decides its own width, so from that
# moment the answer REPLACES the grant rather than joining it in a maximum: a
# tool granted four that asks for one is running at one, and dividing its
# child's share by the four it ignored would be the defect this whole design
# removed (an unused grant penalising the child) wearing a new name.
#
# Across calls the running MAXIMUM is kept, because a tool has phases: one that
# opens one channel for its serial phase and six for its batch may call
# `sup.run` from inside either, so the divisor has to be the widest it has been
# allowed to be, not the latest. Conservative in the safe direction -- a child
# gets less room than it might have, never more than was reserved.
_WIDTH_GRANTED = 1
_WIDTH_ASKED = None


def _record_width(channels) -> int:
    """Keep the widest width `sup.channels()` has answered, and return it."""
    global _WIDTH_ASKED
    try:
        value = max(1, int(channels))
    except (TypeError, ValueError):
        return _width_running_at()
    _WIDTH_ASKED = value if _WIDTH_ASKED is None else max(_WIDTH_ASKED, value)
    return value


def _record_granted_width(channels) -> None:
    """Keep the widest width this process was HANDED, asked for or not."""
    global _WIDTH_GRANTED
    try:
        _WIDTH_GRANTED = max(_WIDTH_GRANTED, max(1, int(channels)))
    except (TypeError, ValueError):
        pass


def _width_running_at() -> int:
    """The width a child's share is divided by. See the two globals above."""
    return max(1, int(_WIDTH_GRANTED if _WIDTH_ASKED is None else _WIDTH_ASKED))


def _grant_channels(run, params: dict, tool_name=None) -> None:
    """Fill in how many channels this run may open, if it did not say.

    **The migration path, and the one a new tool should not take.** A tool that
    calls `sup.channels(wanted)` gets the same number out of the same budget,
    from the side that knows how many items there are -- see
    `_Supervisor.channels`. This one has to keep working meanwhile: every tool
    in the catalogue declaring `num_workers` or `batch_size` is served through
    it today, and flipping them all at once is how a migration breaks a tool
    nobody was looking at. It is removable when no served tool declaring a
    channel argument leaves `sup.channels()` uncalled -- at which point
    `_channels_for`, `_affordable_here`, `_items_here` and `WIDTH_AXIS_ENV` go
    with it, since nothing else reads them.

    In the ENVIRONMENT rather than in job.json because the two answer different
    questions: job.json is what the caller asked for, and this is what the
    machine granted -- decided after admission, which is after the job file was
    written. A caller who named a number keeps it, the way every other
    server-provided argument works here.

    A tool declaring neither argument is left alone: a capability nobody
    declared is not one the server may assume. Opening several channels inside
    a loop that shares a temporary file corrupts its own outputs, silently, and
    the declaration is the tool saying it does not.

    **A width of one is not written**, and that is deliberate rather than an
    optimisation. `CHANNEL_ARGUMENTS` are the tool's own arguments and carry
    the tool's own defaults -- CLIC's `batch_size = 4` is four slices in one
    forward pass, Crown_Seg's `num_workers = 2` is two meshes -- so writing a
    one over them would quarter one tool and halve the other on every run the
    machine happened to admit at a single channel. Whether the server should
    override a default it did not choose is a real question, and it is not this
    one: nothing here has ever written a one, and starting to is its own change
    with its own measurements.
    """
    for name in CHANNEL_ARGUMENTS:
        if not _takes(run, name):
            continue
        if params.get(name) in (None, ""):
            channels = _channels_for(tool_name, params)
            if channels and channels > 1:
                params[name] = channels
        # Recorded whoever decided it, and INCLUDING the value nobody wrote.
        # A tool given six channels and calling `sup.run` from inside each of
        # them has six children alive at once, whether it asked for the six,
        # was handed them, or is simply carrying its own default -- and the
        # child's share has to be divided by the same number either way. The
        # default is the case that would otherwise be missed: CLIC declares
        # `batch_size = 4` and the server deliberately does not write a one
        # over it, so a run admitted at one channel is still four wide here.
        _record_granted_width(_effective_width(run, name, params))
        return


def _effective_width(run, name: str, params: dict) -> int:
    """The value `run()` will actually see for its channel argument.

    The server's grant and the caller's own number both arrive in `params`; a
    run where neither wrote anything gets the tool's declared default, which is
    a width like any other.
    """
    value = params.get(name)
    if value in (None, ""):
        try:
            value = inspect.signature(run).parameters[name].default
        except (TypeError, ValueError, KeyError):
            return 1
        if value is inspect.Parameter.empty:
            return 1
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _channels_for(tool_name, params: dict):
    """How wide this process may run, or None when nothing decided.

    Two answers, and which one applies says who decided. A run the SERVER
    admitted is told its width outright, because the reservation is held
    against that number and this process may not revise it. A nested run was
    admitted as part of its parent and never saw admission at all, so it is
    told only the ROOM it may spend and works the width out here -- its own
    measured cost against that room, bounded by the items its own request
    carries. That is deliberately the same arithmetic `concurrency.ceiling`
    does at the top, and deliberately NOT a number its caller computed: a tool
    asking for seven landmarks should get the same width whoever asked.
    """
    named = os.environ.get(CHANNELS_ENV)
    if named:
        try:
            return max(1, int(named))
        except (TypeError, ValueError):
            return None
    budget = _channel_budget()
    if budget is None:
        # No server around this run at all -- `scripts/run_tool.py`, or a tool
        # called straight from Python. It must not find a width it cannot
        # account for.
        return None
    return _affordable_here(tool_name, budget, params)


def _channel_budget():
    """The room this process holds, as `(vram, ram)` bytes, or None if unset."""
    raw = os.environ.get(CHANNEL_BUDGET_ENV)
    if not raw:
        return None
    parts = str(raw).split(",")
    try:
        room = tuple(max(0, int(part.strip() or 0)) for part in parts[:2])
    except (TypeError, ValueError):
        return None
    return room if len(room) == 2 else None


def _channels_affordable(tool_name, budget) -> int:
    """How many channels of this tool the room it holds could pay for.

    The room left once the tool's FIXED cost is paid, divided by what a
    further channel adds, on whichever of the card and host memory runs out
    first -- the same arithmetic `costs.Cost.channels_within` does on the
    server side, and deliberately the same at every depth: that identity is
    what makes a tool's width its own property rather than its caller's. A
    deliberate second implementation, not a shared one, for the reason this
    whole file is: it executes inside a tool's virtualenv and imports nothing
    of the server's.

    A tool nothing has measured gets ONE and spends the whole budget on it,
    which is the rule admission applies to an unmeasured tool at the top: an
    unknown cost is reserved as if it were everything. A tool measured at only
    one width gets the same answer it always got -- see `_measured_cost`.
    """
    cost = _measured_cost(tool_name)
    if cost is None:
        return 1
    vram_one, vram_marginal, ram_one, ram_marginal, widths = cost
    room = []
    for held, one, marginal in ((budget[0], vram_one, vram_marginal),
                                (budget[1], ram_one, ram_marginal)):
        if not held or not one:
            continue
        if held < one:
            return 1  # not even one channel's worth of room
        if not marginal:
            continue  # measured to cost nothing per channel: no bound here
        if widths >= 2:
            room.append(1 + (held - one) // marginal)
        else:
            room.append(held // marginal)
    return max(1, min(room)) if room else 1


def _affordable_here(tool_name, budget, params: dict) -> int:
    """This tool's own width, from its own cost against the room it holds.

    The migration path's half of `_Supervisor.channels`: it has to COUNT the
    request, because an unmigrated tool never says how many items it has.
    """
    width = _channels_affordable(tool_name, budget)
    # Never wider than the request has things to do. Giving a tool two workers
    # for one item spends a process launch -- ~3 s for ALI_CBCT -- to do one
    # item's work, and then the run is MEASURED at a width it never reached.
    items = _items_here(tool_name, params)
    if items:
        width = min(width, items)
    return _capped_here(width)


def _measured_cost(tool_name):
    """This tool's fitted cost: one channel, and what each one after it adds.

    Read straight off the table's flat keys, which `costs.record` maintains as
    exactly the model `costs.cost_of` hands admission -- `vram_bytes` is what
    ONE channel costs, the tool's fixed part included, and
    `vram_marginal_bytes` is what a second channel adds on top of it. None
    means nothing has measured this tool, which is not the same as a tool
    measured at zero.

    A table written before the intercept existed carries no marginal key, and
    the one-channel figure stands in for it -- which is precisely the purely
    proportional model that table was written under, so an old file keeps
    answering exactly as it did. `widths` below two says the same thing about
    a fresh entry: one point cannot be split, so nothing pretends it was.
    """
    path = os.environ.get(COST_TABLE_ENV)
    if not path or not tool_name:
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            table = json.load(handle)
    except (OSError, ValueError):
        return None
    entry = table.get(tool_name) if isinstance(table, dict) else None
    if not isinstance(entry, dict):
        return None

    def figure(key, fallback):
        # `or 0` would turn a measured zero -- a tool whose channels cost
        # nothing extra -- back into the whole one-channel cost.
        value = entry.get(key)
        return fallback if value is None else max(0, int(value))

    try:
        vram_one = figure("vram_bytes", 0)
        ram_one = figure("ram_bytes", 0)
        return (vram_one, figure("vram_marginal_bytes", vram_one),
                ram_one, figure("ram_marginal_bytes", ram_one),
                max(1, figure("widths", 1)))
    except (TypeError, ValueError):
        return None


def _items_here(tool_name, params: dict):
    """How many things this request holds, or None where there is no counting it.

    A deliberate second implementation of `concurrency._count`, not a shared
    one: this file runs inside a tool's virtualenv and imports nothing of the
    server's. Which ARGUMENT to count is not duplicated -- the server sends its
    own answer down in `WIDTH_AXIS_ENV`, so both levels bound a width by one
    rule.
    """
    raw = os.environ.get(WIDTH_AXIS_ENV)
    if not raw or not tool_name:
        return None
    try:
        axes = json.loads(raw)
    except ValueError:
        return None
    axis = axes.get(tool_name) if isinstance(axes, dict) else None
    if not isinstance(axis, str) or not axis:
        return None
    value = params.get(axis)
    if value is None:
        return None
    # A multichoice arrives as the COMPLETE {option: ticked} mapping, so an
    # unticked option is present and False rather than missing.
    if isinstance(value, dict):
        return sum(1 for ticked in value.values() if ticked) or None
    if isinstance(value, (list, tuple, set)):
        return len(value) or None
    text = str(getattr(value, "path", value) or "")
    if not text:
        return None
    if os.path.isdir(text):
        found = 0
        for _root, _dirs, files in os.walk(text):
            found += len(files)
        return found or None
    if os.path.exists(text):
        return 1
    return None


def _capped_here(channels: int) -> int:
    """The deployment's backstop, if it set one. 0 or unset means no cap.

    Read from the environment under the name the server's own setting already
    has, so there is nothing extra to export and nothing that can disagree.
    """
    try:
        declared = int(os.environ.get(MAX_CHANNELS_ENV) or 0)
    except (TypeError, ValueError):
        declared = 0
    if declared > 0:
        return max(1, min(int(channels), declared))
    return max(1, int(channels))


def _child_channel_budget():
    """The room this process may hand to ONE tool it calls.

    What it holds, divided by the channels it is running at -- because each of
    those channels may be inside a `sup.run` at the same instant, so each has to
    be affordable on its own. `_width_running_at()` is that divisor: what
    `sup.channels()` answered where the tool asked, and what it was handed
    where it did not.

    **It used to divide by `CHANNELS_ENV` and the room was already divided.**
    `concurrency.granted` wrote one channel's worth into the environment and
    this divided it by the grant again, so a top-level run handed its child a
    share `channels` times too small -- and at depth 2 and below, where
    `CHANNELS_ENV` is deliberately absent, it divided by nothing at all and a
    grandchild inherited its parent's whole room. Both are gone: the room that
    arrives is undivided, and this is the one place it is divided.

    The floor of one in `_record_width` is what stops a deep chain from
    dividing its way to nothing; a budget of zero bytes is harmless on its own,
    since the width computed from it floors at one channel.
    """
    budget = _channel_budget()
    if budget is None:
        return None
    share = _width_running_at()
    return tuple(held // share for held in budget)


def _takes(run, name: str) -> bool:
    """Whether run() has a parameter of that name, injected or not."""
    try:
        return name in inspect.signature(run).parameters
    except (TypeError, ValueError):
        return False


def _wanted_steps(value) -> set:
    """Which steps of the chain a caller asked to keep.

    A multichoice arrives as `{tool: ticked}` -- every declared option, so an
    unticked one is present and False rather than missing. The other spellings
    are here because this runner is also driven by `scripts/run_tool.py`, where
    a developer writes a list, and by a direct Python call, where `True` is the
    obvious way to say "all of them".
    """
    if value is True:
        return _ALL_STEPS
    if isinstance(value, dict):
        return {str(name) for name, on in value.items() if on}
    if isinstance(value, (list, tuple, set)):
        return {str(name) for name in value}
    return set()


def _collect_supervised_outputs(job_dir: str, output_dir, wanted: set) -> list:
    """Move what the WANTED supervised calls produced under `output_dir`.

    One folder per kept call, named as the supervisor named it -- `01_ALI_CBCT`
    -- so a reader can match a result to the order the chain ran in, and two
    calls to the same tool stay apart.

    What is collected is only the callee's OWN output directory. A chain's work
    dir also holds the caller's inputs unpacked, its reference bundle, its
    conversions; returning those would send a patient's own scans back to them
    and double an archive to do it. What a supervised tool produced is the part
    nobody can otherwise see.

    Moved, not copied: both sides are in the same job directory, so it is a
    rename, and the job directory is removed either way.
    """
    root = os.path.join(job_dir, SUP_DIRNAME)
    if not wanted or not output_dir or not os.path.isdir(root):
        return []

    collected = []
    for entry in sorted(os.listdir(root)):
        # "01_ALI_CBCT" -> "ALI_CBCT". The number is the call's position, which
        # is what keeps two calls to one tool from colliding.
        _, _, tool = entry.partition("_")
        if wanted is not _ALL_STEPS and tool not in wanted:
            continue
        produced = os.path.join(root, entry, OUTPUT_DIR_NAME)
        if not os.path.isdir(produced) or not os.listdir(produced):
            # A call that wrote nowhere. Nothing to say about it: the run
            # succeeded, and an empty folder would read as a lost result.
            continue
        destination = os.path.join(str(output_dir), INTERMEDIATE_DIRNAME, entry)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.move(produced, destination)
        collected.append(destination)
    return collected


def _declares_injected(run, name: str) -> bool:
    """A keyword-only, UNANNOTATED parameter of that name.

    Unannotated is the marker rather than an accident: every other parameter
    must be annotated, so there is nothing else this shape could be, and a tool
    cannot grow one of these by forgetting a type.
    """
    try:
        parameter = inspect.signature(run).parameters.get(name)
        hints = typing.get_type_hints(run)
    except Exception:  # noqa: BLE001 - an unresolvable annotation is not fatal
        return False
    return (
        parameter is not None
        and parameter.kind is parameter.KEYWORD_ONLY
        and name not in hints
    )


class _Supervisor:
    """What a tool receives as `sup`: six members, duck-typed, nothing shared.

    A tool never imports this class. It cannot -- the tool's venv holds none of
    the server -- and that is the point: the same object shape is produced by
    `SADT-VISOR`'s `scripts/run_tool.py` and faked in its tests, and a tool
    cannot tell the three apart.

    `run()` re-enters THIS file with the sibling's interpreter, so the callee
    gets its own venv, its own dependency set and its own supervisor. Nesting
    (`AREG -> ASO -> ALI`) is that same recursion and needs no special case.

    **It does not go back through the server.** A nested call is a subprocess of
    the parent, so it never queues for a slot the parent is already holding --
    which is exactly the deadlock the in-process version had, where four
    concurrent ASO runs each waited on a fifth slot. The cost is that nested
    work is invisible to MAX_GPU_JOBS: a supervised chain can put two tools on
    the card at once. Chains are serial today (ASO waits for ALI before it
    registers), so the peak is one tool at a time per chain, but a deployment
    running several supervised jobs concurrently has to size for that.
    """

    @staticmethod
    def _deadline():
        """When this whole job must be done, or None when nothing said.

        A monotonic timestamp shared by every level: `time.monotonic()` counts
        from an arbitrary origin that is constant per BOOT, not per process, so
        a parent and a child on the same machine read the same clock. Passing an
        absolute instant rather than a duration is what makes it survive the
        hop -- a duration would restart at every level and a five-deep chain
        would get five times the budget.
        """
        raw = os.environ.get(SUPERVISOR_DEADLINE_ENV)
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def _remaining_seconds(self):
        """What is left of the job's budget, or None when there is no deadline.

        Refuses to start a child with no time rather than starting one that will
        be killed mid-run: a tool that never began is a clean 422-shaped failure
        naming the chain, where a SIGKILL two levels down is a 500 with nothing
        in it.
        """
        deadline = self._deadline()
        if deadline is None:
            return None
        left = deadline - time.monotonic()
        if left <= 0:
            raise RunnerError(
                "The job's time budget is exhausted ({}), so '{}' was not started. "
                "Raise this tool's timeout_seconds in deployment.toml -- an "
                "orchestrating tool's budget has to cover its whole chain, not "
                "its own work.".format(" -> ".join(self._chain), "the next tool")
            )
        return left

    def __init__(self, tools_dir: str, job_dir: str, depth: int, chain=(), job_id=None,
                 root=None, data_dir=None, tool=None):
        self._tools_dir = tools_dir
        self._job_dir = job_dir
        self._depth = depth
        # Whose costs `channels()` reads out of the table. The chain's last
        # entry is the same name; it is passed explicitly all the same, so a
        # supervisor built for a tool that refused to say cannot silently
        # answer with a neighbour's figures.
        self._tool = tool or (chain[-1] if chain else None)
        # Innermost last. `chain` is what refuses a cycle by NAME; `depth` is
        # only a backstop for a chain that grows without repeating.
        self._chain = tuple(chain)
        self._job_id = job_id
        # The job at the top of this tree. Carried so every record in a
        # supervised run can be attributed to the request that started it --
        # traceability, and what a VRAM budget would later be applied to. It is
        # NOT what makes nesting deadlock-free: a nested call is a subprocess of
        # its parent and never re-enters the server's admission queue at all,
        # so there is no queue it could wait in.
        self._root = root or job_id
        self._calls = 0
        self.out = Path(job_dir) / "output"
        # Removed with the job directory, by whoever owns it. The tool is held
        # to writing only under `output/`, so its scratch sits beside it rather
        # than inside what the caller keeps.
        self.tmp = Path(job_dir) / "tmp"
        self.tmp.mkdir(parents=True, exist_ok=True)
        # The read-only DATA root, for building a NEIGHBOUR's model path:
        #
        #     sup.run("ALI_CBCT", model=sup.datapath / "ALI" / "models", ...)
        #
        # A tool's own arguments are resolved by the server before it ever runs,
        # and this does not change that. What it answers is the one thing the
        # server cannot: a supervised call is not a request, so it never passes
        # through the admission path where a hosted-model argument is filled in,
        # and the caller has to name the bundle itself.
        #
        # A NAME is all the caller needs -- a DATA folder is named after the tool
        # it belongs to (harmonised 2026-09-10). None on a deployment that
        # publishes no root, which a tool must handle rather than assume.
        self.datapath = Path(data_dir) if data_dir else None

    # -- the frozen interface ----------------------------------------------

    def run(self, tool: str, **params):
        """Run another tool, blocking, and return what its run() returned."""
        # Checked BEFORE the depth cap, because it is the more precise answer to
        # the same question. A cycle hits the depth limit eventually anyway, but
        # only after starting four processes and whatever they each loaded, and
        # the message it produces names a number rather than the mistake.
        if tool in self._chain:
            raise RunnerError(
                "Supervised call cycle: {} -> {}. A tool cannot call one that is "
                "already running above it.".format(" -> ".join(self._chain), tool)
            )
        if self._depth >= MAX_SUPERVISOR_DEPTH:
            raise RunnerError(
                "Supervised calls nested more than {} deep ({} -> {}).".format(
                    MAX_SUPERVISOR_DEPTH, " -> ".join(self._chain), tool
                )
            )

        interpreter = self._interpreter(tool)
        # Where a callee writes is not the caller's to choose, exactly as it is
        # not an HTTP client's: the server strips `output_dir` from the
        # published schema and fills it with the job's own output/. Dropping it
        # here gives a supervised call the same property, and it is what lets
        # `keep_intermediate` find what a chain produced -- a callee pointed at
        # the caller's scratch directory has its results deleted with it, before
        # anything could collect them.
        params.pop(OUTPUT_DIR_ARGUMENT, None)
        self._calls += 1
        nested_dir = os.path.join(self._job_dir, "sup", f"{self._calls:02d}_{tool}")
        os.makedirs(os.path.join(nested_dir, "output"), exist_ok=True)

        child_id = f"{os.path.basename(self._job_dir)}.{self._calls}"
        job_path = os.path.join(nested_dir, JOB_FILE)
        with open(job_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "job_id": child_id,
                    "tool": tool,
                    "job_dir": nested_dir,
                    "params": params,
                    "parent": self._job_id,
                    "root": self._root,
                    # Inherited, or a tool two levels down could not reach a
                    # neighbour's models while its parent could.
                    "data_dir": str(self.datapath) if self.datapath else None,
                },
                handle,
                default=_jsonable,
            )

        self.log(f"running '{tool}'")
        environment = dict(os.environ)
        environment[SUPERVISOR_DEPTH_ENV] = str(self._depth + 1)
        environment[SUPERVISOR_CHAIN_ENV] = ",".join(self._chain + (tool,))
        # What the child may SPEND -- not how wide it may run. A nested call
        # never re-enters the server's admission queue (it is a subprocess of
        # this one), so nothing between the two levels would divide anything,
        # and a parent running six channels whose child opens eight is
        # forty-eight on a machine that admitted one job. What is handed down
        # is therefore this level's room divided by this level's channels, and
        # the child divides again for its own. The multiplication is impossible
        # because the BYTES were divided on the way down: level k spends at most
        # R / (C0 x ... x Ck-1) times Ck channels, which telescopes back to R.
        #
        # `CHANNELS_ENV` is REMOVED rather than set. Setting it told the child
        # its width, computed here out of numbers that were never about the
        # child: on 2026-09-18 the same ALI_CBCT asking for seven landmarks got
        # 7 channels standalone, 5 under an ASO holding 28 cores and 7 under an
        # ASO holding 31. Removed, the child finds only the room and sizes
        # itself from its own measured cost -- the same arithmetic a top-level
        # run gets, so a tool's width is its own property again. It also has to
        # be removed explicitly: the environment is a copy of this process's,
        # and inheriting the PARENT's width would be worse than either.
        #
        # Removing it is also what makes the child's `sup.channels()` answer
        # from its room rather than from a reservation it was never part of:
        # the cap by `CHANNELS_ENV` in `channels()` belongs to the level
        # admission actually reserved for, and that is this one.
        budget = _child_channel_budget()
        if budget is None:
            environment.pop(CHANNEL_BUDGET_ENV, None)
        else:
            environment[CHANNEL_BUDGET_ENV] = "{},{}".format(*budget)
        environment.pop(CHANNELS_ENV, None)
        # Raises when the budget is already spent, before starting anything.
        remaining = self._remaining_seconds()
        # SADT_TOOL_DIR points at the PARENT's folder; the callee derives its
        # own from its interpreter, and inheriting ours would send it to the
        # wrong sources.
        environment.pop(TOOL_DIR_ENV, None)

        # The nested call declares itself, at the CHILD's depth.
        #
        # Without this a chain is invisible to anything reading the run's
        # events. ASO drives ALI_CBCT for most of its wall clock and the only
        # trace of it was the log line above -- a depth-0 MESSAGE, which the
        # benchmark payload drops on purpose. Measured on the chain arms: 56
        # events across seven runs, every one of them at depth 0, on a tool
        # whose nested call is two thirds of its duration.
        #
        # Two markers rather than one record carrying a duration, because the
        # close has to be written on the failure path too: a nested call that
        # raised still occupied the parent for as long as it ran, and a bar
        # that only appears when a chain succeeds is a bar that lies about the
        # runs worth looking at. The child's own records land between them at
        # the same depth, so the span is right whether or not it reports
        # anything of its own.
        #
        # Opened HERE rather than beside the log line above: everything
        # between them can raise -- `_remaining_seconds` does, when the job's
        # time is already spent -- and an open with no close is a span with no
        # end.
        _append_progress(None, "", self._depth + 1, tool=tool)
        try:
            return self._nested_subprocess(
                tool, interpreter, job_path, nested_dir, environment, remaining
            )
        finally:
            _append_progress(None, "", self._depth + 1, tool=tool)

    def _nested_subprocess(self, tool, interpreter, job_path, nested_dir,
                           environment, remaining):
        """Run one nested level and return its result, or raise its error.

        Split out of `_run_nested` so the nested markers around it are one
        `try`/`finally` rather than a close repeated on every exit path.
        """
        # NO start_new_session here, and that is deliberate. Without it a nested
        # child inherits its parent's process group, so the SIGTERM the server
        # sends to the ROOT's group on a timeout reaches every level at once.
        # Giving each level its own session would look tidier and would detach
        # the grandchildren from the group the server kills -- exactly the
        # orphaned-worker-holding-VRAM failure that killpg was added to prevent.
        #
        # The timeout below is therefore the polite path, not the only one: it
        # lets a level fail with a message naming the chain instead of being
        # killed silently, and the group kill remains the backstop.
        try:
            completed = subprocess.run(
                [interpreter, os.path.abspath(__file__), "--job", job_path],
                # Not captured: a nested tool's log is the only sign of life
                # during an hour-long run, and it is already on stderr where the
                # server collects it. Its result never travels on stdout anyway.
                cwd=nested_dir,
                env=environment,
                timeout=remaining,
            )
        except subprocess.TimeoutExpired:
            raise RunnerError(
                "Supervised tool '{}' ran out of the job's remaining time after "
                "{:.0f}s. The chain is {} -> {}; raise the ROOT tool's "
                "timeout_seconds, which has to cover the whole chain.".format(
                    tool, remaining or 0, " -> ".join(self._chain), tool
                )
            )
        if completed.returncode != 0:
            # Its own result file is where the useful half is. The child's
            # traceback goes to stderr, which whatever runs the PARENT may be
            # capturing and trimming -- so "see above" is a promise this cannot
            # keep, and the message has to carry the reason itself.
            kind, message = self._failure_parts(tool, nested_dir)

            # A child's answer to the CALLER is re-raised under the child's own
            # exception name, so the server maps it the way it would have if the
            # caller had run that tool directly. Wrapping it in RunnerError threw
            # the classification away: AMASSS saying "No scan found in ...
            # Supported extensions: .nii.gz, ..." -- a sentence written to be
            # read by whoever sent the request -- reached the client as "Tool
            # execution failed", indistinguishable from a crash. The tool name is
            # prepended because in a chain the caller cannot otherwise tell which
            # step is talking.
            if kind in CALLER_FACING_ERRORS:
                raise type(kind, (Exception,), {})(f"{tool}: {message}")

            raise RunnerError(
                f"Supervised tool '{tool}' failed (exit {completed.returncode}). "
                f"{kind}: {message}" if message else
                f"Supervised tool '{tool}' failed (exit {completed.returncode}). {kind}."
            )
        return self._result(tool, nested_dir)

    def _failure_parts(self, tool: str, nested_dir: str):
        """`(exception class name, message)` the callee recorded, read back."""
        path = os.path.join(nested_dir, RESULT_FILE)
        try:
            with open(path, encoding="utf-8") as handle:
                error = json.load(handle).get("error") or {}
        except (OSError, ValueError):
            return ("Error", f"It wrote no readable result; see its output and {path}.")
        return (error.get("type", "Error"), error.get("message", "").strip())

    def channels(self, wanted: int = 0) -> int:
        """How many of `wanted` items this run may process at once. Never below 1.

        The tool asks; the machine answers. `wanted` is the tool's OWN count,
        in the unit the tool actually loops over -- landmarks of a scan, teeth
        of a mesh, slices of a volume -- taken at run time, which is the one
        moment it is knowable. Omitted or zero means "as many as I can afford".

        The answer is what this run's SHARE can pay for: the room it holds
        (`SADT_CHANNEL_BUDGET`) divided by what one channel of this tool was
        MEASURED to cost. Nothing is over-committed by answering this late,
        because the share was reserved before the process started -- see
        `execution/concurrency.py`'s header. The cost of that bargain, stated
        there and worth repeating: a run that opens one channel still holds its
        share for its whole life.

        **Why this is not just `num_workers`.** It is the same number arriving
        from the other direction, and the direction is the point. The server
        filling in `num_workers` has to work out how many items a request holds
        from the OUTSIDE, before the run -- which for several tools it simply
        cannot: `ALI_IOS` counts teeth out of a mesh's label array, `CLIC`
        counts slices of a volume it has not read, and `AutoCrop3D`,
        `AutoMatrix` and `GreedyReg` take two paired folders where splitting
        either alone re-pairs patients. For those the outside answer was "no
        bound", and an unbounded width is RESERVED all the same. Measured here
        on 2026-09-18, on AMASSS, whose channels are its scans and whose axis
        the server could count: 19.1 GiB per channel against a 93.8 GiB host
        budget, granted 3 channels for a request holding ONE scan -- 57.3 GiB
        reserved, so one run fitted where four had. Six concurrent AMASSS went
        143 s -> 275 s, ten went 158 s -> 425 s, while a single run stayed 79 s
        throughout, which is why nothing caught it: the whole cost is in what a
        run stops OTHER runs from doing. Bounding the width by the real item
        count brought six concurrent back to 153.8 s. This is that bound, asked
        of the side that has it.

        **It does not declare the width.** `progress.set_width(n)` still does,
        and the two are not the same statement: this is a PERMISSION, granted
        before the work, and that is a MEASUREMENT, reported where the width is
        actually in force. A tool has phases of different widths -- AMASSS
        reads its cohort four at a time, runs nnUNet one structure at a time,
        then assembles four at a time, and its peak is in the serial middle --
        and the server divides a run's peak by the NARROWEST width reported.
        Declaring this answer automatically would put a wide width in force
        during a serial phase, which under-reserves the next run: the unsafe
        direction, and the one thing a mechanism must never do behind a tool's
        back. Forgetting to declare, by contrast, reads as one channel and
        over-reserves. So they sit beside each other, they agree by
        construction where a tool declares what it opened, and only the
        declaration is read by the cost table.
        """
        room = _channel_budget()
        if room is None:
            # No server around this run at all -- `scripts/run_tool.py`, a test
            # harness, a tool driven straight from Python. It must not find a
            # width nobody accounted for, so it gets the one width that is
            # always affordable.
            return _record_width(1)
        answer = _channels_affordable(self._tool, room)
        try:
            asked = int(wanted)
        except (TypeError, ValueError):
            asked = 0
        if asked > 0:
            answer = min(answer, asked)
        # Never past what admission actually reserved against. The two agree
        # whenever the cost table has not moved under the run -- the room IS
        # the per-channel cost times that number -- and where it has moved,
        # this is the figure the bytes were taken for and the arithmetic above
        # is the one working from stale prices. Absent on a nested level, which
        # was admitted as part of its parent and has only the room to go on.
        reserved = os.environ.get(CHANNELS_ENV)
        if reserved:
            try:
                answer = min(answer, max(1, int(reserved)))
            except (TypeError, ValueError):
                pass
        answer = _capped_here(answer)
        self.log("{} channel(s) of {} asked for".format(answer, asked or "any"))
        return _record_width(answer)

    def progress(self, fraction: float, message: str) -> None:
        try:
            self.log(f"{float(fraction):.0%} {message}")
        except (TypeError, ValueError):
            self.log(str(message))
        _append_progress(fraction, message, self._depth)

    def log(self, message: str) -> None:
        # Through logging, not print: the runner owns handlers and the server
        # reads stderr. The depth prefix is what makes a nested chain readable.
        logging.getLogger("sadt.supervisor").info("%s%s", "  " * self._depth, message)

    # -- internals ----------------------------------------------------------

    def _interpreter(self, tool: str) -> str:
        """The tool's own python, wherever its folder sits.

        Searched one level deep as well as at the top, because a tool may live
        under a GROUPING folder: `ALI_CBCT` and `ALI_IOS` are two tools inside
        `tools/ALI/`, which holds no tool of its own. The folder name is the
        tool name either way -- only its depth varies -- so this stays a lookup
        rather than a scan of every pyproject, which this file could not read
        anyway (stdlib only, and tomllib does not exist before 3.11).

        Depth is capped at 2 on purpose: deeper would start matching a tool's
        own vendored directories.
        """
        for folder in self._candidate_folders(tool):
            for relative in (os.path.join("bin", "python"), os.path.join("Scripts", "python.exe")):
                candidate = os.path.join(folder, ".venv", relative)
                if os.path.isfile(candidate):
                    return candidate
        raise RunnerError(
            f"Supervised tool '{tool}' has no virtualenv under {self._tools_dir} "
            f"(looked for '{tool}/.venv' and '*/{tool}/.venv'). It is not deployed here."
        )

    def _candidate_folders(self, tool: str):
        """`<root>/<tool>` then `<root>/*/<tool>`, for each root we might be under.

        Two roots, because the CALLER may itself be nested. `_tool_dir()` gives
        this tool's folder and the root was taken as its parent -- right for
        tools/AMASSS, wrong for tools/ALI/ALI_CBCT, where the parent is the
        grouping folder and its siblings are the only tools reachable. That is
        latent today, ALI_CBCT calling nothing, and it points straight at
        AREG_IOSCBCT: a nested tool whose whole job is calling the others.
        """
        seen = set()
        for root in self._roots:
            direct = os.path.join(root, tool)
            if direct not in seen:
                seen.add(direct)
                yield direct
            try:
                groups = sorted(os.listdir(root))
            except OSError:
                continue
            for group in groups:
                if group == tool:
                    continue
                nested = os.path.join(root, group, tool)
                if nested not in seen and os.path.isdir(nested):
                    seen.add(nested)
                    yield nested

    @property
    def _roots(self):
        """The tool's own catalogue, its parent when nested, and every other
        catalogue TOOLS_DIR names.

        The first two are derived from where this tool lives, which is the one
        thing the invocation already fixes. That was enough while there was one
        catalogue. It is not enough with several: a tool in the second whose
        child lives in the first would look for it beside itself, not find it,
        and fail a chain the server's boot check had just declared complete.
        """
        roots = [self._tools_dir]
        parent = os.path.dirname(self._tools_dir)
        if parent and parent != self._tools_dir:
            roots.append(parent)
        for entry in os.environ.get("TOOLS_DIR", "").split(os.pathsep):
            entry = entry.strip()
            if entry:
                roots.append(os.path.abspath(entry))
        seen, ordered = set(), []
        for root in roots:
            if root not in seen:
                seen.add(root)
                ordered.append(root)
        return ordered

    def _result(self, tool: str, nested_dir: str):
        path = os.path.join(nested_dir, RESULT_FILE)
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as exc:
            raise RunnerError(f"Supervised tool '{tool}' wrote no readable result: {exc}")
        # Folded BEFORE the error check, deliberately: a child that ran out of
        # memory is the most useful measurement in the whole chain, and
        # discarding it because the chain then failed is how a budget goes on
        # admitting the thing that broke it.
        _fold_child_measurement(payload)
        if "error" in payload:
            error = payload["error"]
            raise RunnerError(
                f"Supervised tool '{tool}' failed: "
                f"{error.get('type', 'Error')}: {error.get('message', '')}"
            )
        value = payload.get("result")
        if isinstance(value, str):
            return Path(value)
        if isinstance(value, dict):
            return {key: Path(item) for key, item in value.items()}
        return value


def _supervisor_for(run, job: dict):
    """A supervisor for this job, or None when the tool does not ask for one."""
    if not _takes_supervisor(run):
        return None
    tools_dir = os.path.dirname(_tool_dir())
    depth = 0
    try:
        depth = int(os.environ.get(SUPERVISOR_DEPTH_ENV, "0"))
    except ValueError:
        pass
    # The chain the parent handed us, plus ourselves. A root job has an empty
    # environment variable and a chain of just its own name, which is what makes
    # a tool calling ITSELF the first thing refused rather than the fifth.
    inherited = tuple(
        name for name in os.environ.get(SUPERVISOR_CHAIN_ENV, "").split(",") if name
    )
    chain = inherited or (job["tool"],)
    return _Supervisor(
        tools_dir=tools_dir,
        job_dir=job["job_dir"],
        depth=depth,
        chain=chain,
        job_id=job.get("job_id"),
        root=job.get("root"),
        data_dir=job.get("data_dir"),
        tool=job["tool"],
    )


def _load_job(job_path: str) -> dict:
    try:
        with open(job_path, encoding="utf-8") as handle:
            job = json.load(handle)
    except (OSError, ValueError) as exc:
        raise RunnerError(f"Cannot read the job file {job_path}: {exc}")

    for field in ("tool", "job_dir", "params"):
        if field not in job:
            raise RunnerError(f"Job file {job_path} has no '{field}' field.")
    if not isinstance(job["params"], dict):
        raise RunnerError("Job field 'params' must be an object of argument name -> value.")
    if not os.path.isdir(job["job_dir"]):
        raise RunnerError(f"Job directory does not exist: {job['job_dir']}")
    return job


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run one SADT tool job.")
    parser.add_argument("--job", required=True, help="Path to the job.json written by the server")
    arguments = parser.parse_args(argv)

    _configure_logging()
    # Started before the tool is even imported: a heavy import is real resident
    # memory and belongs in the figure, and starting it later would miss a tool
    # whose peak is in its own loading.
    _sampler.start()

    job = None
    try:
        job = _load_job(arguments.job)
        module = _import_tool(job["tool"], os.path.join(_tool_dir(), SRC_DIR_NAME))
        run = _run_function(module, job["tool"])
        params = dict(job["params"])
        # Taken back out before the tool ever sees it: no tool declares this
        # argument. The server publishes it for anything that calls another
        # tool, and collecting what a chain produced is done below -- once,
        # here, rather than written into every orchestrating tool.
        keep_intermediate = _wanted_steps(params.pop(KEEP_INTERMEDIATE_ARGUMENT, None))
        if _takes(run, OUTPUT_DIR_ARGUMENT) and OUTPUT_DIR_ARGUMENT not in params:
            # A supervised call arrives without one: the supervisor drops what
            # the caller passed, for the same reason the server strips it from
            # the published schema. Filled from this job's own directory, which
            # is what dispatch does for a request.
            nested_output = os.path.join(job["job_dir"], OUTPUT_DIR_NAME)
            os.makedirs(nested_output, exist_ok=True)
            params[OUTPUT_DIR_ARGUMENT] = nested_output
        _grant_channels(run, params, job["tool"])
        arguments = _call_arguments(run, params)
        supervisor = _supervisor_for(run, job)
        if supervisor is not None:
            # Injected, never taken from job.json: it is not data, and a client
            # that could name one would be naming a process to start.
            arguments[SUPERVISOR_ARGUMENT] = supervisor
        if _takes_data_root(run) and job.get("data_dir"):
            # A path, not a value: the tool builds its own bundle path under it
            # and nobody outside names one.
            arguments[DATA_ROOT_ARGUMENT] = Path(job["data_dir"])
        result = run(**arguments)
        if keep_intermediate:
            # After run(), never before: a tool removes its own scratch on the
            # way out, and what a chain produced has to survive that.
            collected = _collect_supervised_outputs(
                job["job_dir"], arguments.get(OUTPUT_DIR_ARGUMENT), keep_intermediate
            )
            if collected:
                print(
                    "kept what {} supervised call(s) produced".format(len(collected)),
                    file=sys.stderr,
                )
        _write_result(job["job_dir"], result)
    except RunnerError as exc:
        # Ours, and already precise: the traceback would only point back here.
        print(str(exc), file=sys.stderr)
        if job:
            _write_error(job["job_dir"], exc)
        return 1
    except Exception as exc:
        # The tool's own failure. The whole traceback goes to stderr because
        # the server keeps only its tail, and that tail is all anyone will have
        # to work from; the class name goes in the result file, because it is
        # what decides the status code.
        traceback.print_exc()
        if job:
            _write_error(job["job_dir"], exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
