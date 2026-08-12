#!/usr/bin/env python3
"""The half of a tool run that happens inside the tool's own interpreter.

    /tools/<name>/.venv/bin/python /opt/sadt/runner.py --job /jobs/<id>/job.json

Read the job file, import the tool, call its run(), write result.json. That is
the whole job.

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

The error channel is stderr plus a non-zero exit code. On failure nothing is
written -- an absent result.json IS the failure signal, so a half-written one
must never be mistaken for a result (hence the atomic replace below).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback

# Where the tool's code lives inside its folder, and the environment variable
# that overrides the folder itself (tests and dev checkouts, where the venv is
# not necessarily next to the sources).
SRC_DIR_NAME = "src"
TOOL_DIR_ENV = "SADT_TOOL_DIR"

RESULT_FILE = "result.json"


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


def _import_tool(tool_name: str, src_dir: str):
    """Import the module defining run(), from the tool's src/ and nowhere else.

    The module is named after the tool -- src/<tool>.py or src/<tool>/ -- which
    is the same "the file is named after the folder" rule the in-process
    registry has always used. `src/tool.py` is accepted as well so a tool whose
    name is awkward as an identifier has a way out.
    """
    if not os.path.isdir(src_dir):
        raise RunnerError(f"Tool '{tool_name}' has no '{SRC_DIR_NAME}/' directory at {src_dir}")

    sys.path.insert(0, src_dir)
    candidates = [tool_name, "tool"] if tool_name != "tool" else ["tool"]
    tried = []
    for candidate in candidates:
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


def _write_result(job_dir: str, result) -> None:
    """Write {"result": ...} atomically.

    Serialized in full BEFORE anything touches the disk, so an unserializable
    result is a clean failure rather than a truncated result.json; and replaced
    into place, so the server never reads one being written.
    """
    payload = json.dumps({"result": result}, default=_jsonable)
    final_path = os.path.join(job_dir, RESULT_FILE)
    temp_path = final_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        handle.write(payload)
    os.replace(temp_path, final_path)


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

    try:
        job = _load_job(arguments.job)
        module = _import_tool(job["tool"], os.path.join(_tool_dir(), SRC_DIR_NAME))
        result = _run_function(module, job["tool"])(**job["params"])
        _write_result(job["job_dir"], result)
    except RunnerError as exc:
        # Ours, and already precise: the traceback would only point back here.
        print(str(exc), file=sys.stderr)
        return 1
    except Exception:
        # The tool's own failure. The whole traceback goes to stderr because
        # the server keeps only its tail, and that tail is all anyone will have
        # to work from.
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
