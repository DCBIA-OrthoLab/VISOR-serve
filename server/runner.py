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
import json
import logging
import os
import sys
import traceback
import typing
from pathlib import Path

# Where the tool's code lives inside its folder, and the environment variable
# that overrides the folder itself (tests and dev checkouts, where the venv is
# not necessarily next to the sources).
SRC_DIR_NAME = "src"
TOOL_DIR_ENV = "SADT_TOOL_DIR"

RESULT_FILE = "result.json"

# The tool's package under src/, when there is exactly one. sadt-tools names it
# `sadt_<tool>`, but the rule is the one its own describe.py uses -- the single
# importable package -- rather than the prefix, so a tool that names its
# package something else still loads.
_FALLBACK_PREFIX = "sadt_"


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

    Same rule as sadt-tools' own describe.py: exactly one directory holding an
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

    Deliberately identical to sadt-tools' testkit driver
    (`testkit/src/sadt_testkit/_driver.py`): every tool's integration tests run
    against that one, so a difference here is a suite that passes while
    production fails.
    """
    if annotation is Path:
        return Path(value)
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

    job = None
    try:
        job = _load_job(arguments.job)
        module = _import_tool(job["tool"], os.path.join(_tool_dir(), SRC_DIR_NAME))
        run = _run_function(module, job["tool"])
        result = run(**_call_arguments(run, job["params"]))
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
