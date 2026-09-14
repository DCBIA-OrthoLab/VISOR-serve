"""Run a tool in its own interpreter, in its own process.

This is the server half of what runner.py does on the other side:

    <TOOLS_DIR>/<tool>/.venv/bin/python <RUNNER_PATH> --job <job dir>/job.json

Why a subprocess at all, when importing the tool works today: importing it
pins the SERVER's Python to the lowest common denominator of every tool, and
two tools wanting different versions of torch (or of numpy: SurgMovPred asks
for numpy==2.4.0, AREG for numpy<2.0.0) cannot coexist in one interpreter at
all. Holding torch in the API process also keeps a CUDA context alive for the
lifetime of the server, so VRAM is never fully released between jobs, and a
segfault inside a CUDA kernel takes the API down with the job.

**Still synchronous.** The HTTP request blocks for exactly as long as it does
today; `subprocess.run` is called from the same worker thread `tool.invoke`
already ran in, so MAX_CONCURRENT_TOOLS keeps arbitrating how many run at
once. No queue, no polling, no change to the contract with the client.

The job directory is handed out by file_utils, which means the request handler
already removes it once the response has been streamed -- outputs included,
and on the error paths too. Nothing here needs its own cleanup timer.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from typing import Any, Optional

import file_utils
from base import ToolUnavailableError
from config import settings
from registry.deployment import deployment_config
from wire import runs

# The folder holding a tool's hosted weights, under DATA_DIR/<data slug>/. The
# same name data_store.py lists models from; stated here rather than imported so
# this module keeps depending on nothing that touches the filesystem for a
# request.
MODELS_DIRNAME = "models"

logger = logging.getLogger("inference_server")

JOB_FILE = "job.json"
RESULT_FILE = "result.json"
# How long a TERMed process group gets before SIGKILL.
_KILL_GRACE_SECONDS = 10.0

STDOUT_LOG = "stdout.log"
STDERR_LOG = "stderr.log"

# Subdirectories every job gets. `input/` is where the server will stage inputs
# once tools stop being handed paths into the request's own work dir; `output/`
# is what a tool writes into and what survives long enough to be streamed back.
# The job directory's output folder. Named here because main.py builds the
# result archive from it and needs to recognise it: an archive named after this
# folder would be called "output.zip" for every tool.
#
# runner.py repeats the literal rather than importing this. It is stdlib-only by
# contract, executed by each tool's own interpreter, so it cannot import a server
# module. The two must agree; a test pins them together.
JOB_OUTPUT_DIRNAME = "output"

JOB_SUBDIRS = ("input", JOB_OUTPUT_DIRNAME)

# How much of the tool's stderr travels with the failure. Enough for a
# traceback, bounded because a failing tool can print megabytes.
_STDERR_TAIL_BYTES = 8192


# The GPU is one device and the tools no longer arbitrate for it themselves
# (see settings.MAX_CONCURRENT_GPU_JOBS). Created lazily so the setting can be
# monkeypatched in a test before the first run.
_gpu_slots: Optional[threading.BoundedSemaphore] = None
_gpu_lock = threading.Lock()

# The argument a tool declares when it can be told where to run, and the values
# that mean "not on the card".
DEVICE_ARGUMENT = "device"
_CPU_DEVICES = ("cpu", "mps")


def _gpu_semaphore() -> threading.BoundedSemaphore:
    global _gpu_slots
    with _gpu_lock:
        if _gpu_slots is None:
            _gpu_slots = threading.BoundedSemaphore(settings.MAX_CONCURRENT_GPU_JOBS)
        return _gpu_slots


def uses_the_gpu(tool, params: dict) -> bool:
    """Will this run take the card?

    **Assumed yes**, unless the run is explicitly on something else. Only a
    tool that declares `device` can say otherwise, and only by resolving it to
    a CPU value.

    The safe default is the strict one. A tool that quietly imports torch
    without declaring `device` -- and several do; it was `settings.DEVICE`
    inside them until they were packaged -- would otherwise never queue, and
    two of them would meet on the card with nothing between them. The cost of
    being wrong the other way is a CPU-only run waiting for a slot it did not
    need; the cost of being wrong this way is an out-of-memory in the middle of
    somebody's cohort.
    """
    spec = tool.arguments.get(DEVICE_ARGUMENT)
    if spec is None:
        return True
    value = params.get(DEVICE_ARGUMENT)
    if value is None:
        value = spec.default if spec.is_choice else settings.DEVICE
    return not str(value).lower().startswith(_CPU_DEVICES)


class ToolFailure(RuntimeError):
    """The tool itself raised, and said which exception it was.

    There is no shared exception type to catch -- there is no shared package --
    so the runner records the class NAME and main.py maps it: the caller's
    fault (422), this deployment's (503), or opaque (500).
    """

    def __init__(self, error_type: str, message: str):
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type
        self.message = message


class ToolExecutionError(RuntimeError):
    """The tool's process failed: non-zero exit, a timeout, or no usable
    result.json.

    Deliberately NOT one of base.py's typed errors: main.py maps those to 422 /
    501, and a tool that crashed is neither the caller's fault nor something
    this deployment can be reconfigured to fix. It falls through to the generic
    500 handler, which logs the detail server-side and answers the client with
    "Tool execution failed." -- exactly what an in-process crash does today.
    """


class RunCancelled(RuntimeError):
    """The client withdrew this run, through `DELETE /runs/{id}`.

    A class of its own, deliberately NOT a ToolFailure: main.py answers it with
    499, and a client has to be able to tell "I stopped this" from "this broke"
    without reading a message -- one closes the panel quietly, the other opens
    an error dialog. Sharing an exception type with a tool that raised would
    make that distinction a string comparison.
    """


def _raise_if_cancelled(run_id: Optional[str]) -> None:
    """One stat, at each point where the run could still be stopped cheaply.

    The marker is the half of cancellation that covers the window with no
    process in it -- inputs staged, the job file about to be written, the run
    sitting in the GPU queue. Once there IS a process, `DELETE` signals its
    group directly and this only notices afterwards.
    """
    if run_id and runs.is_cancelled(run_id):
        raise RunCancelled("The client cancelled this run.")


@contextlib.contextmanager
def _gpu_slot(run_id: Optional[str]):
    """The card, waited for in a way a cancellation can interrupt.

    Without a run id this is the plain blocking acquire it has always been. With
    one, the wait is broken into RUN_CANCEL_POLL_SECONDS slices so a client that
    gives up on a queued run is not held until the slot it no longer wants frees
    -- which, behind a multi-hour cohort, is the longest wait in the system.
    """
    semaphore = _gpu_semaphore()
    if run_id is None:
        with semaphore:
            yield
        return
    while not semaphore.acquire(timeout=settings.RUN_CANCEL_POLL_SECONDS):
        _raise_if_cancelled(run_id)
    try:
        yield
    finally:
        semaphore.release()


def _registered_folder(tool_name: str):
    """The folder discovery recorded for this tool, or None.

    Imported inside the function: registry imports this module, so a module-level
    import would be a cycle.
    """
    try:
        import registry
    except ImportError:
        return None
    tool = registry.TOOLS.get(tool_name)
    return getattr(tool, "folder", None)


def tool_interpreter(tool_name: str) -> str:
    """Path to the interpreter of this tool's virtualenv.

    Searched one level below each catalogue as well as directly under it,
    because a
    tool may live inside a GROUPING folder: ALI_CBCT and ALI_IOS are two tools
    in tools/ALI/, which is not a tool itself. The folder is still named after
    the tool -- only its depth varies -- so this stays a lookup.

    The registry is asked FIRST, because it already resolved this at startup and
    a name-based search cannot answer for a tool whose `[tool.sadt] name` differs
    from its directory. That key exists to make the API identity a decision
    rather than an accident of directory casing, and it could not deliver while
    the run path re-derived the folder from the name -- one rule in two places,
    which is the shape of nearly every defect found in this repository.

    The search below it stays as a fallback: registry.TOOLS is empty in unit
    tests that exercise dispatch alone, and an in-process tool has no folder.

    The direct path is returned when nothing matches, so the caller's error
    names where it looked first.
    """
    folder = _registered_folder(tool_name)
    if folder:
        registered = os.path.join(folder, ".venv", "bin", "python")
        if os.path.isfile(registered):
            return registered

    # Every catalogue TOOLS_DIR names, in order, and the server's own last.
    roots = settings.tool_roots()
    direct = os.path.join(roots[0], tool_name, ".venv", "bin", "python")
    for root in roots:
        here = os.path.join(root, tool_name, ".venv", "bin", "python")
        if os.path.isfile(here):
            return here
        try:
            groups = sorted(os.listdir(root))
        except OSError:
            continue
        for group in groups:
            if group == tool_name:
                continue
            nested = os.path.join(root, group, tool_name, ".venv", "bin", "python")
            if os.path.isfile(nested):
                return nested
    return direct


def _checked_interpreter(tool_name: str) -> str:
    interpreter = tool_interpreter(tool_name)
    if not os.path.isfile(interpreter):
        # The path names a server-side directory, so it goes to the log and not
        # into the exception: a 501 body travels to the client.
        logger.error(
            "Tool '%s' has no interpreter at %s -- its virtualenv is missing.",
            tool_name,
            interpreter,
        )
        raise ToolUnavailableError(
            f"Tool '{tool_name}' is not installed on this server (no virtualenv). "
            f"See the server logs."
        )
    return interpreter


def _checked_runner() -> str:
    runner = settings.RUNNER_PATH
    if not os.path.isfile(runner):
        logger.error("RUNNER_PATH does not point at a file: %s", runner)
        raise ToolUnavailableError("This server is misconfigured: the tool runner is missing.")
    return runner


def _create_job_dir(job_id: str) -> str:
    """A fresh directory for this job, registered for request-scoped cleanup.

    Named after the job so a directory left behind by a crash can be traced
    back to a log line, and created with exist_ok=False so two jobs can never
    end up sharing one.
    """
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    job_dir = os.path.join(settings.TEMP_DIR, f"job_{job_id}")
    os.makedirs(job_dir)
    for subdir in JOB_SUBDIRS:
        os.makedirs(os.path.join(job_dir, subdir))
    return file_utils.register_scratch_dir(job_dir)


def _server_provided(tool, params: dict, job_dir: str) -> dict:
    """The arguments the SERVER fills in, not the caller.

    `output_dir` is the job's own output/: every packaged tool takes it as a
    required argument and writes only there, and a client has no business
    naming a directory on the server.

    `device` is the deployment's, when the caller did not pick one. It used to
    be `settings.DEVICE` read inside each tool; a tool that no longer reads the
    environment would otherwise always run on its own default (cuda), on a
    server configured for CPU.

    A hosted MODEL argument nobody named gets this tool's models DIRECTORY, and
    the tool picks its own bundle inside it. Which weights an engine needs is
    the engine's own business -- ALI_CBCT is the CBCT engine and can want
    nothing else -- but WHERE they sit is this machine's, and a tool is not
    allowed to know (CONTRIBUTING.md: "run() does not read the environment and
    does not know about /DATA. Path resolution belongs to the server.").
    So the server says where, and the tool says which.

    That works because a bundle is recognisable by its own shape and no other:
    measured against DATA/ALI/models/, which holds both, ALI_CBCT's
    `discover_weights` finds its 119 landmarks under `<landmark>/<scale>/*.pth`
    and ignores the intraoral bundle, while ALI_IOS's finds its six flat
    checkpoints by their network and jaw tokens and ignores 238 CBCT files.
    A tool whose weights are not recognisable there simply reports what it
    always reported for a bundle it cannot read.

    Only ever fills a gap: a supervisor chaining ALI, or an API client that
    names its bundle, has already put it in `params` and is left alone. And
    `model` stays a required argument of `run()`, so the tool is still usable
    with no server at all -- a direct call passes a path, as it always did.
    """
    filled = dict(params)
    if getattr(tool, "wants_output_dir", False):
        output_dir = os.path.join(job_dir, JOB_OUTPUT_DIRNAME)
        os.makedirs(output_dir, exist_ok=True)
        filled["output_dir"] = output_dir
    if DEVICE_ARGUMENT in tool.arguments and DEVICE_ARGUMENT not in filled:
        filled[DEVICE_ARGUMENT] = settings.DEVICE
    for name, spec in tool.arguments.items():
        if getattr(spec, "server_selectable", None) != "model" or name in filled:
            continue
        models = os.path.join(
            settings.DATA_DIR, deployment_config.data_slug(tool.name), MODELS_DIRNAME
        )
        if os.path.isdir(models):
            filled[name] = models
    return filled


def _write_job_file(job_dir: str, job_id: str, tool_name: str, params: dict) -> str:
    """Write job.json.

    Every path in `params` is already resolved by the server (uploads streamed
    to disk, server_selectable names looked up through data_store): the tool
    never sees a reference it would have to resolve for its OWN arguments.

    `data_dir` is the one exception, and it is there for supervised calls only.
    A tool asking the supervisor for a neighbour has to hand it a model bundle,
    and nothing resolves that: `_server_provided` runs here, on the top-level
    request, and a supervised call never passes through it. So the root is
    published and `Supervisor.datapath` hands it on, letting a caller build
    `<root>/<tool>/models` from a name it already holds. That works because a
    DATA folder is named after the tool it belongs to -- harmonised 2026-09-10,
    and the reason it was worth doing.
    """
    job = {
        "job_id": job_id,
        "tool": tool_name,
        "job_dir": job_dir,
        "params": params,
        "data_dir": settings.DATA_DIR,
    }
    job_path = os.path.join(job_dir, JOB_FILE)
    with open(job_path, "w", encoding="utf-8") as handle:
        json.dump(job, handle, default=_jsonable)
    return job_path


def _jsonable(value):
    """Same narrow rule as the runner's: a Path is a path, anything else is a
    schema the wire cannot carry and must fail here rather than halfway."""
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError(
        f"Argument value of type {type(value).__name__} cannot be written to {JOB_FILE}."
    )


def _child_environment(job_id: str, job_dir: str, timeout: Optional[float] = None,
                       progress_file: Optional[str] = None) -> dict:
    """The environment the tool process runs in.

    Inherited rather than rebuilt: tools legitimately need PATH, HOME,
    LD_LIBRARY_PATH, CUDA_VISIBLE_DEVICES and whatever else the deployment
    sets. API_TOKEN is removed -- it is the server's credential, a tool has no
    use for it, and the tool venvs hold third-party code.
    """
    environment = dict(os.environ)
    environment.pop("API_TOKEN", None)
    if timeout:
        # An ABSOLUTE instant on the monotonic clock, not a duration: the clock's
        # origin is per-boot rather than per-process, so every level of a
        # supervised chain reads the same one. A duration would restart at each
        # hop and a five-deep chain would quietly get five times the budget.
        #
        # This is what makes an orchestrating tool's timeout mean "the whole
        # chain": AREG_IOSCBCT computes for a second and spends the rest inside
        # its children, so the budget it is given has to be theirs too.
        environment["SADT_SUPERVISOR_DEADLINE"] = repr(time.monotonic() + timeout)
    if progress_file:
        # An absolute path to this run's events.jsonl, set ONLY when the run has
        # a directory -- a tool that finds the variable can rely on the file
        # existing. It is inherited by every supervised child, which is the
        # whole implementation of chain progress: a nested tool appends to the
        # same file as its parent, one depth deeper.
        environment[runs.PROGRESS_FILE_ENV] = progress_file
    else:
        # A stale value from the server's own environment would send a tool's
        # progress into a file belonging to nothing.
        environment.pop(runs.PROGRESS_FILE_ENV, None)
    environment.update(
        {
            "SADT_API": settings.SADT_API,
            "SADT_JOB_ID": job_id,
            "SADT_JOB_DIR": job_dir,
        }
    )
    return environment


def _stderr_tail(job_dir: str) -> str:
    """The last few KB the tool wrote to stderr, for the exception message.

    May well contain a file path, so it is treated exactly like an in-process
    traceback: logged server-side, never returned to the client (main.py
    answers 500 with a fixed message).
    """
    path = os.path.join(job_dir, STDERR_LOG)
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            if size > _STDERR_TAIL_BYTES:
                handle.seek(size - _STDERR_TAIL_BYTES)
            tail = handle.read()
    except OSError:
        return "(no stderr captured)"
    return tail.decode("utf-8", errors="replace").strip() or "(empty stderr)"


def _execute(command: list, job_dir: str, environment: dict, timeout: Optional[float],
             tool_name: str = "", run_id: Optional[str] = None) -> int:
    """Run the tool process to completion; return its exit code.

    stdout and stderr go to FILES, not to pipes: a tool can print for hours
    (nnUNet does), and a pipe means holding all of it in the server's memory.
    Their contents are never logged either -- shapeaxi prints the patient's own
    file name.

    The process gets its own SESSION (`start_new_session`), and a timeout kills
    the whole PROCESS GROUP rather than the one PID we know about. That
    distinction is the entire point: nnUNet, torch's DataLoader and shapeaxi all
    fork workers, and killing the parent leaves those workers running and
    holding VRAM -- a card that stays full after the job that filled it is gone,
    with nothing left to attribute it to. `killpg` reaches them because the
    session made them one group.
    """
    with open(os.path.join(job_dir, STDOUT_LOG), "wb") as out_stream, open(
        os.path.join(job_dir, STDERR_LOG), "wb"
    ) as error_stream:
        try:
            process = subprocess.Popen(
                command,
                stdout=out_stream,
                stderr=error_stream,
                env=environment,
                # A tool writing a relative path lands in its own job directory
                # rather than in the server's source tree.
                cwd=job_dir,
                # POSIX only, which the deployment image is. On Windows this
                # would need CREATE_NEW_PROCESS_GROUP and a different kill.
                start_new_session=True,
            )
        except OSError as exc:
            raise ToolExecutionError(f"Could not start the tool process: {exc}")

        if run_id is not None:
            # Both of these belong to the same instant, and the order matters:
            # the group id is what a DELETE served by ANOTHER uvicorn worker
            # signals, so it goes down before anything is told the run started.
            runs.set_pgid(run_id, process.pid)
            runs.append(run_id, runs.PHASE_RUNNING)
            # The race the marker check exists for: a cancel written between
            # Popen and set_pgid signalled nothing, because there was nothing
            # recorded to signal yet.
            if runs.is_cancelled(run_id):
                _kill_group(process)
                raise RunCancelled("The client cancelled this run.")

        return _wait(process, timeout, tool_name, run_id)


def _wait(process: subprocess.Popen, timeout: Optional[float], tool_name: str,
          run_id: Optional[str]) -> int:
    """Wait for the tool, in slices short enough to notice a cancellation.

    The timeout semantics are exactly the ones a single `process.wait(timeout)`
    had: the same budget, the same kill, the same message naming both knobs.
    What changed is only that the wait is broken up, so a client that cancels a
    two-hour nnUNet is not waiting for the timeout to be the thing that ends it
    -- and so a `DELETE` served by another worker, which kills the group and
    tells this process nothing, is read as a cancellation rather than as a tool
    that exited with -15.
    """
    deadline = None if not timeout else time.monotonic() + timeout
    poll = settings.RUN_CANCEL_POLL_SECONDS if run_id is not None else None
    while True:
        slice_seconds = poll
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_group(process)
                # Naming both knobs, because which one applied is not visible
                # from the outside and the operator's next move differs: a
                # per-tool entry raises it for this tool alone, the global
                # setting for everything.
                raise ToolExecutionError(
                    f"Tool '{tool_name}' did not finish within its timeout ({timeout}s). "
                    f"Raise it with [tools.{tool_name}] timeout_seconds in deployment.toml, "
                    f"or TOOL_TIMEOUT_SECONDS for every tool."
                )
            slice_seconds = remaining if slice_seconds is None else min(slice_seconds, remaining)
        try:
            exit_code = process.wait(timeout=slice_seconds)
        except subprocess.TimeoutExpired:
            if run_id is not None and runs.is_cancelled(run_id):
                _kill_group(process)
                raise RunCancelled("The client cancelled this run.")
            continue
        # It exited. A cancel that arrived while it ran is what killed it, and
        # the exit code says nothing useful about that.
        _raise_if_cancelled(run_id)
        return exit_code


def _kill_group(process: subprocess.Popen) -> None:
    """SIGTERM the process group, then SIGKILL whatever is left.

    TERM first so a tool with a handler can release the card and flush; KILL
    after a grace period because most will not. `os.killpg` needs the group id,
    which is the child's pid precisely because `start_new_session` made it a
    group leader.

    Every failure here is swallowed deliberately: the process may have exited
    between the timeout and the signal, and a ProcessLookupError then is the
    normal case, not an error to propagate over a timeout that already is one.
    """
    for signal_number, grace in ((signal.SIGTERM, _KILL_GRACE_SECONDS), (signal.SIGKILL, None)):
        try:
            os.killpg(os.getpgid(process.pid), signal_number)
        except (ProcessLookupError, PermissionError, OSError):
            break
        if grace is None:
            break
        try:
            process.wait(timeout=grace)
            break
        except subprocess.TimeoutExpired:
            continue
    try:
        process.wait(timeout=_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        # Unreapable after SIGKILL means the process is stuck in the kernel
        # (uninterruptible I/O). Nothing further is possible from here.
        logger.error("Tool process %s survived SIGKILL; it is now a zombie.", process.pid)


def kill_process_group(pgid: int) -> None:
    """SIGTERM a process group named only by its id. What `DELETE /runs/{id}`
    does with the pgid the run directory recorded.

    Only TERM, and only once, which is the whole difference from `_kill_group`
    above: the endpoint holds no handle on that process -- it may well be in
    another uvicorn worker -- so it can neither wait for it nor reap it, and a
    blocking escalation would hold the 204 for the grace period. Escalation
    belongs to the side that HAS the handle: the worker running `_wait` sees
    the cancel marker on its next poll and applies the full TERM-then-KILL
    discipline. This signal is what makes the stop immediate rather than
    RUN_CANCEL_POLL_SECONDS late.

    The pgid guard is not defensive style. `os.killpg(0, ...)` signals the
    CALLER's group, which is this server: a truncated or absent pgid file must
    never be able to take the API down on a cancel.
    """
    if not isinstance(pgid, int) or pgid <= 1:
        logger.warning("refusing to signal an implausible process group")
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        # It finished between the marker and the signal, which is the normal
        # case for a run cancelled just as it ended.
        pass


def _read_result(job_dir: str, tool_name: str) -> Any:
    """The value run() returned, out of result.json.

    A missing file after a zero exit code is its own failure: the runner writes
    the file last and atomically, so "exited fine but produced nothing" means
    the tool never got as far as returning.
    """
    path = os.path.join(job_dir, RESULT_FILE)
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        raise ToolExecutionError(
            f"Tool '{tool_name}' exited successfully but wrote no {RESULT_FILE}."
        )
    except (OSError, ValueError) as exc:
        raise ToolExecutionError(f"Tool '{tool_name}' wrote an unreadable {RESULT_FILE}: {exc}")

    if not isinstance(payload, dict):
        raise ToolExecutionError(f"Tool '{tool_name}': {RESULT_FILE} must be an object.")

    error = payload.get("error")
    if isinstance(error, dict):
        # The tool raised and named its exception class. Which one it was is
        # the difference between "you sent the wrong thing" and "this broke".
        raise ToolFailure(str(error.get("type", "")), str(error.get("message", "")))

    if "result" not in payload:
        raise ToolExecutionError(
            f"Tool '{tool_name}': {RESULT_FILE} must be an object with a 'result' field."
        )

    # The runner measures this on every run precisely so a VRAM budget can be
    # set later from measurements rather than guesses -- but until now nothing
    # read it back, so every measurement was written into a job directory and
    # deleted with it. One log line is what makes the instrumentation exist.
    # It is a number, not patient data.
    peak = payload.get("peak_vram_bytes")
    if isinstance(peak, int):
        logger.info(
            "tool=%s peak_vram_bytes=%d (%.2f GiB)",
            tool_name, peak, peak / 1024 ** 3,
        )

    return payload["result"]


def dispatch(tool, params: dict, job_id: Optional[str] = None) -> Any:
    """Run `tool` on already-validated `params` in the tool's own interpreter.

    Returns exactly what the tool's run() returned, so callers -- Tool.invoke,
    and through it main.py -- cannot tell which side of the flag they are on.
    """
    interpreter = _checked_interpreter(tool.name)
    runner = _checked_runner()
    job_id = job_id or uuid.uuid4().hex
    # Set by the endpoint and carried into this worker thread by anyio's context
    # copy. None whenever the client sent no X-Run-Id, which is what makes every
    # progress and cancellation call below a no-op for such a run.
    run_id = runs.CURRENT_RUN.get()
    job_dir = _create_job_dir(job_id)

    try:
        # Before a byte of the job is written: a cancel that arrived while the
        # inputs were still being staged costs nothing more than this stat.
        _raise_if_cancelled(run_id)
        params = _server_provided(tool, params, job_dir)
        job_path = _write_job_file(job_dir, job_id, tool.name, params)
        command = [interpreter, runner, "--job", job_path]
        # Per tool, falling back to the global setting. 0 means no limit, which
        # a cohort legitimately needs. Computed BEFORE the environment, which
        # carries it down to every supervised level as a deadline.
        timeout = deployment_config.timeout_seconds(tool.name) or None
        environment = _child_environment(
            job_id, job_dir, timeout, runs.progress_file(run_id)
        )

        if uses_the_gpu(tool, params):
            # Announced before the wait, and only here: a multi-minute queue for
            # the card is today indistinguishable from a tool that is running,
            # which is the single most confusing thing a client can show.
            runs.append(run_id, runs.PHASE_QUEUED_GPU)
            # Held for the whole run, and released by leaving the block on any
            # path. Blocking on purpose: the request already waits for the tool,
            # and a queue is what a single card wants.
            with _gpu_slot(run_id):
                _raise_if_cancelled(run_id)
                exit_code = _execute(command, job_dir, environment, timeout,
                                     tool.name, run_id)
        else:
            exit_code = _execute(command, job_dir, environment, timeout,
                                 tool.name, run_id)

        if exit_code != 0:
            # A tool that recorded WHICH exception it was gets to say so; the
            # tail of stderr is the fallback for one that died without writing
            # anything (a segfault in a CUDA kernel, an OOM kill).
            _read_result(job_dir, tool.name)
            raise ToolExecutionError(
                f"Tool '{tool.name}' exited with code {exit_code}:\n{_stderr_tail(job_dir)}"
            )
        return _read_result(job_dir, tool.name)
    except Exception:
        # No response will ever be streamed from this job, so nothing has to
        # survive: take the directory down now rather than leaving confidential
        # inputs waiting for the request handler's cleanup.
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
