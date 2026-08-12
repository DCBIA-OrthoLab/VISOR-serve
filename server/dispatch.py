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

import json
import logging
import os
import shutil
import subprocess
import uuid
from typing import Any, Optional

import file_utils
from base import ToolUnavailableError
from config import settings

logger = logging.getLogger("inference_server")

JOB_FILE = "job.json"
RESULT_FILE = "result.json"
STDOUT_LOG = "stdout.log"
STDERR_LOG = "stderr.log"

# Subdirectories every job gets. `input/` is where the server will stage inputs
# once tools stop being handed paths into the request's own work dir; `output/`
# is what a tool writes into and what survives long enough to be streamed back.
JOB_SUBDIRS = ("input", "output")

# How much of the tool's stderr travels with the failure. Enough for a
# traceback, bounded because a failing tool can print megabytes.
_STDERR_TAIL_BYTES = 8192


class ToolExecutionError(RuntimeError):
    """The tool's process failed: non-zero exit, a timeout, or no usable
    result.json.

    Deliberately NOT one of base.py's typed errors: main.py maps those to 422 /
    501, and a tool that crashed is neither the caller's fault nor something
    this deployment can be reconfigured to fix. It falls through to the generic
    500 handler, which logs the detail server-side and answers the client with
    "Tool execution failed." -- exactly what an in-process crash does today.
    """


def tool_interpreter(tool_name: str) -> str:
    """Path to the interpreter of this tool's virtualenv."""
    return os.path.join(settings.TOOLS_DIR, tool_name, ".venv", "bin", "python")


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


def _write_job_file(job_dir: str, job_id: str, tool_name: str, params: dict) -> str:
    """Write job.json. Every path in `params` is already resolved by the server
    (uploads streamed to disk, server_selectable names looked up through
    data_store): the tool never sees a reference it would have to resolve, and
    never reads DATA_DIR by itself."""
    job = {
        "job_id": job_id,
        "tool": tool_name,
        "job_dir": job_dir,
        "params": params,
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


def _child_environment(job_id: str, job_dir: str) -> dict:
    """The environment the tool process runs in.

    Inherited rather than rebuilt: tools legitimately need PATH, HOME,
    LD_LIBRARY_PATH, CUDA_VISIBLE_DEVICES and whatever else the deployment
    sets. API_TOKEN is removed -- it is the server's credential, a tool has no
    use for it, and the tool venvs hold third-party code.
    """
    environment = dict(os.environ)
    environment.pop("API_TOKEN", None)
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


def _execute(command: list, job_dir: str, environment: dict) -> int:
    """Run the tool process to completion; return its exit code.

    stdout and stderr go to FILES, not to pipes: a tool can print for hours
    (nnUNet does), and a pipe means holding all of it in the server's memory.
    Their contents are never logged either -- shapeaxi prints the patient's own
    file name.
    """
    timeout = settings.TOOL_TIMEOUT_SECONDS or None
    with open(os.path.join(job_dir, STDOUT_LOG), "wb") as out_stream, open(
        os.path.join(job_dir, STDERR_LOG), "wb"
    ) as error_stream:
        try:
            completed = subprocess.run(
                command,
                stdout=out_stream,
                stderr=error_stream,
                env=environment,
                # A tool writing a relative path lands in its own job directory
                # rather than in the server's source tree.
                cwd=job_dir,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            # subprocess.run has already killed it and reaped it.
            raise ToolExecutionError(
                f"The tool did not finish within TOOL_TIMEOUT_SECONDS ({timeout}s)."
            )
        except OSError as exc:
            raise ToolExecutionError(f"Could not start the tool process: {exc}")
    return completed.returncode


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

    if not isinstance(payload, dict) or "result" not in payload:
        raise ToolExecutionError(
            f"Tool '{tool_name}': {RESULT_FILE} must be an object with a 'result' field."
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
    job_dir = _create_job_dir(job_id)

    try:
        job_path = _write_job_file(job_dir, job_id, tool.name, params)
        exit_code = _execute(
            [interpreter, runner, "--job", job_path], job_dir, _child_environment(job_id, job_dir)
        )
        if exit_code != 0:
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
