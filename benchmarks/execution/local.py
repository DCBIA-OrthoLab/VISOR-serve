"""The `local` path: a tool in its OWN interpreter, with no HTTP anywhere.

This reproduces, line for line, what `server/execution/dispatch.py` does once
it has validated a request:

    <tools_dir>/<tool>/.venv/bin/python <runner.py> --job <job_dir>/job.json

with job.json holding {job_id, tool, job_dir, params} and the environment
carrying SADT_API / SADT_JOB_ID / SADT_JOB_DIR, API_TOKEN removed. runner.py
derives the tool folder from its own sys.prefix, imports the single package
under src/, calls run(**params), and writes result.json.

Two modes, and the difference between them is a research finding rather than a
convenience -- see NOTES-local-path.md:

  container  the virtualenvs INSIDE the deployment image, reached with
             `docker exec`. Byte-identical to what the server dispatches to,
             which is what makes a local-vs-loopback delta attributable to the
             protocol and nothing else. The default.
  host       the virtualenvs in the sadt-tools working checkout. This models a
             scientist who installed the tools on their own workstation. It is
             a DIFFERENT build of the same lockfile and differs from the image
             by the dev dependency group; the notes quantify it.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from ..settings import LOCAL_MODE_CONTAINER, LocalSpec, ToolSpec

# Copied from server/execution/{dispatch,runner}.py. Repeated rather than
# imported: importing the server package would drag pydantic-settings and a
# configured API_TOKEN into a harness that must import with neither.
# A test pins these against the server's own constants.
JOB_FILE = "job.json"
RESULT_FILE = "result.json"
JOB_OUTPUT_DIRNAME = "output"
JOB_INPUT_DIRNAME = "input"
STDOUT_LOG = "stdout.log"
STDERR_LOG = "stderr.log"

_DOCKER_TIMEOUT_SECONDS = 120.0

# How much of a tool's stderr is kept. See LocalRunner.execute.
_STDERR_TAIL_CHARS = 400_000


class LocalPathError(RuntimeError):
    """The local path could not be taken. Distinct from a tool that ran and
    failed: that one is a ToolRunFailure and is a result, not a setup error."""


class ToolRunFailure(RuntimeError):
    """The tool's own process failed. Carries what runner.py recorded."""

    def __init__(self, message: str, error_type: str = "", stderr_tail: str = ""):
        super().__init__(message)
        self.error_type = error_type
        self.stderr_tail = stderr_tail


@dataclass
class LocalRun:
    job_id: str
    job_dir: str
    output_dir: str
    result: object = None
    peak_vram_bytes: Optional[int] = None
    exit_code: int = 0
    command: list = field(default_factory=list)
    stderr_tail: str = ""


class LocalRunner:
    """Invokes runner.py the way the dispatcher does, in one of the two modes."""

    def __init__(self, spec: LocalSpec) -> None:
        self.spec = spec

    # -- availability -------------------------------------------------

    @property
    def in_container(self) -> bool:
        return self.spec.mode == LOCAL_MODE_CONTAINER

    def unavailable_reason(self) -> Optional[str]:
        """Why this path cannot be taken here, or None when it can.

        Checked before a campaign starts so a reviewer on a laptop is told
        "there is no such container" instead of watching six repetitions fail.
        """
        if self.in_container:
            if shutil.which("docker") is None:
                return "docker is not on PATH, so the container-mode local path cannot be taken"
            probe = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", self.spec.container],
                capture_output=True,
                text=True,
                timeout=_DOCKER_TIMEOUT_SECONDS,
            )
            if probe.returncode != 0:
                return f"container '{self.spec.container}' does not exist"
            if probe.stdout.strip() != "true":
                return f"container '{self.spec.container}' is not running"
            return None
        if not os.path.isfile(self.spec.host_runner):
            return f"no runner.py at {self.spec.host_runner}"
        if not os.path.isdir(self.spec.host_tools_dir):
            return f"no tools directory at {self.spec.host_tools_dir}"
        return None

    def interpreter(self, tool: ToolSpec) -> str:
        if tool.local is None or not tool.local.folder:
            raise LocalPathError(
                f"Tool '{tool.name}' declares no local.folder, so its interpreter is unknown."
            )
        return os.path.join(self.spec.tools_dir, tool.local.folder, ".venv", "bin", "python")

    def source_dir(self, tool: ToolSpec) -> str:
        return os.path.join(self.spec.tools_dir, tool.local.folder, "src")

    # -- the filesystem, on whichever side it is ----------------------

    def _exec(self, argv: list, timeout: Optional[float] = None,
              stdin: Optional[bytes] = None) -> subprocess.CompletedProcess:
        """Run a command on the side the tools live on.

        In container mode `-u <user>` matters: /jobs is owned by the unprivileged
        account the image runs as, and a root-owned job directory left behind by
        a benchmark would break the next real request.
        """
        if self.in_container:
            argv = [
                "docker", "exec", "-i", "-u", self.spec.container_user, self.spec.container
            ] + argv
        return subprocess.run(
            argv, capture_output=True, timeout=timeout, input=stdin
        )

    def _write_remote_file(self, path: str, content: bytes) -> None:
        completed = self._exec(
            ["sh", "-c", f"cat > {shlex.quote(path)}"],
            timeout=_DOCKER_TIMEOUT_SECONDS,
            stdin=content,
        )
        if completed.returncode != 0:
            raise LocalPathError(
                f"Could not write {path}: {completed.stderr.decode('utf-8', 'replace')[:400]}"
            )

    def _read_remote_file(self, path: str) -> Optional[bytes]:
        completed = self._exec(["cat", path], timeout=_DOCKER_TIMEOUT_SECONDS)
        if completed.returncode != 0:
            return None
        return completed.stdout

    def _makedirs(self, path: str) -> None:
        if self.in_container:
            completed = self._exec(["mkdir", "-p", path], timeout=_DOCKER_TIMEOUT_SECONDS)
            if completed.returncode != 0:
                raise LocalPathError(
                    f"Could not create {path}: {completed.stderr.decode('utf-8', 'replace')[:400]}"
                )
        else:
            os.makedirs(path, exist_ok=True)

    def remove_job(self, job_dir: str) -> None:
        """Take the job directory down. Always called, including after a
        failure: 59 GB of headroom does not survive many abandoned CBCT jobs."""
        if self.in_container:
            self._exec(["rm", "-rf", job_dir], timeout=_DOCKER_TIMEOUT_SECONDS)
        else:
            shutil.rmtree(job_dir, ignore_errors=True)

    def fetch_outputs(self, job_dir: str, destination: str) -> str:
        """Copy the job's output/ to a directory on THIS machine.

        Needed by B5, which compares what the local path produced against what
        came back over the API, and cannot compare a path inside a container
        with a file in a download folder.
        """
        os.makedirs(destination, exist_ok=True)
        source = os.path.join(job_dir, JOB_OUTPUT_DIRNAME)
        if self.in_container:
            completed = subprocess.run(
                ["docker", "cp", f"{self.spec.container}:{source}/.", destination],
                capture_output=True,
                text=True,
                timeout=_DOCKER_TIMEOUT_SECONDS * 5,
            )
            if completed.returncode != 0:
                raise LocalPathError(f"docker cp failed: {completed.stderr[:400]}")
        else:
            shutil.copytree(source, destination, dirs_exist_ok=True)
        return destination

    # -- inputs -------------------------------------------------------

    def stage_input(self, host_path: str, job_dir: str, argument: str) -> str:
        """The path the tool will be handed for one input file.

        A file already under the mounted DATA directory is REFERENCED, not
        copied: it is the same bytes the server hands its own tools, and copying
        a 207 MB volume per repetition would both dominate the timing and fill
        the disk. Anything else is staged into the job's input/ -- which is what
        the server does with an upload, so the tool sees the same shape either
        way.
        """
        if not os.path.exists(host_path):
            raise LocalPathError(f"Input file does not exist: {host_path}")
        if not self.in_container:
            return host_path

        data_root = os.path.abspath(self.spec.host_data_dir)
        absolute = os.path.abspath(host_path)
        if absolute == data_root or absolute.startswith(data_root + os.sep):
            relative = os.path.relpath(absolute, data_root)
            return os.path.join(self.spec.container_data_dir, relative)

        name = f"{argument}_{os.path.basename(host_path)}"
        target = os.path.join(job_dir, JOB_INPUT_DIRNAME, name)
        completed = subprocess.run(
            ["docker", "cp", host_path, f"{self.spec.container}:{target}"],
            capture_output=True,
            text=True,
            timeout=_DOCKER_TIMEOUT_SECONDS * 5,
        )
        if completed.returncode != 0:
            raise LocalPathError(f"docker cp of the input failed: {completed.stderr[:400]}")
        return target

    def data_path(self, slug: str, kind: str, name: str) -> str:
        """Where the server-hosted file of that name lives, on the side the
        tool runs on. Mirrors data_store.py's layout: DATA/<slug>/<kind>s/<name>."""
        folder = "models" if kind == "model" else "testfiles"
        return os.path.join(self.spec.data_dir, slug, folder, name)

    # -- the run ------------------------------------------------------

    def prepare_job(self, tool: ToolSpec, params: dict, job_id: Optional[str] = None) -> LocalRun:
        job_id = job_id or uuid.uuid4().hex
        job_dir = os.path.join(self.spec.jobs_dir, f"bench_{job_id}")
        for subdir in (JOB_INPUT_DIRNAME, JOB_OUTPUT_DIRNAME):
            self._makedirs(os.path.join(job_dir, subdir))
        return LocalRun(
            job_id=job_id, job_dir=job_dir,
            output_dir=os.path.join(job_dir, JOB_OUTPUT_DIRNAME),
        )

    def write_job_file(self, run: LocalRun, tool: ToolSpec, params: dict) -> None:
        """The four fields the runner's contract declares, and no others."""
        document = {
            "job_id": run.job_id,
            "tool": tool.name,
            "job_dir": run.job_dir,
            "params": params,
        }
        content = json.dumps(document).encode("utf-8")
        path = os.path.join(run.job_dir, JOB_FILE)
        if self.in_container:
            self._write_remote_file(path, content)
        else:
            _write_local_file(path, content)

    def environment(self, run: LocalRun, timeout: Optional[float]) -> dict:
        """What dispatch.py's `_child_environment` builds, minus the parts that
        only make sense inside the server process."""
        environment = dict(os.environ)
        environment.pop("API_TOKEN", None)
        if timeout:
            environment["SADT_SUPERVISOR_DEADLINE"] = repr(time.monotonic() + timeout)
        environment.update(
            {
                "SADT_API": "http://127.0.0.1:8000",
                "SADT_JOB_ID": run.job_id,
                "SADT_JOB_DIR": run.job_dir,
            }
        )
        return environment

    def execute(self, tool: ToolSpec, run: LocalRun, timeout: Optional[float] = None) -> LocalRun:
        """The exec line itself. Returns with `run.result` filled in, or raises
        ToolRunFailure carrying what the tool said went wrong."""
        interpreter = self.interpreter(tool)
        command = [interpreter, self.spec.runner, "--job", os.path.join(run.job_dir, JOB_FILE)]
        run.command = command

        if self.in_container:
            # `env -u API_TOKEN` rather than a rebuilt environment: the container's
            # own PATH, LD_LIBRARY_PATH and CUDA_VISIBLE_DEVICES are exactly what
            # the server's children inherit, and reconstructing them from the host
            # would be a different environment wearing the same name.
            argv = [
                "docker", "exec", "-i", "-u", self.spec.container_user,
                "-w", run.job_dir,
                "-e", f"SADT_JOB_ID={run.job_id}",
                "-e", f"SADT_JOB_DIR={run.job_dir}",
                "-e", "SADT_API=http://127.0.0.1:8000",
                self.spec.container,
                # `env -u`, not an empty value: dispatch.py POPS API_TOKEN, and
                # a tool that reads os.environ.get("API_TOKEN") must see the
                # same absence here as it does in production.
                "env", "-u", "API_TOKEN",
            ] + command
            completed = subprocess.run(argv, capture_output=True, timeout=timeout)
        else:
            completed = subprocess.run(
                command,
                capture_output=True,
                timeout=timeout,
                cwd=run.job_dir,
                env=self.environment(run, timeout),
                start_new_session=True,
            )

        run.exit_code = completed.returncode
        # Generous, because a SUPERVISED run interleaves its own supervisor lines
        # with every child's logging: 8 kB of an AREG_IOSCBCT chain is the tail of
        # ALI_CBCT's landmark progress and none of the "running 'X'" lines that
        # say which children fired. Callers trim to what they store.
        run.stderr_tail = completed.stderr.decode("utf-8", "replace")[-_STDERR_TAIL_CHARS:]

        payload = self._read_result(run)
        if payload is None:
            raise ToolRunFailure(
                f"Tool '{tool.name}' exited with code {run.exit_code} and wrote no {RESULT_FILE}.",
                stderr_tail=run.stderr_tail,
            )
        error = payload.get("error")
        if isinstance(error, dict):
            raise ToolRunFailure(
                str(error.get("message", "")),
                error_type=str(error.get("type", "")),
                stderr_tail=run.stderr_tail,
            )
        if "result" not in payload:
            raise ToolRunFailure(
                f"Tool '{tool.name}': {RESULT_FILE} has no 'result' field.",
                stderr_tail=run.stderr_tail,
            )
        run.result = payload["result"]
        run.peak_vram_bytes = payload.get("peak_vram_bytes")
        return run

    def _read_result(self, run: LocalRun) -> Optional[dict]:
        path = os.path.join(run.job_dir, RESULT_FILE)
        raw = self._read_remote_file(path) if self.in_container else _read_local_file(path)
        if raw is None:
            return None
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    # -- B3: what the isolation costs ---------------------------------

    def measure_startup(self, tool: ToolSpec, repetitions: int = 3) -> dict:
        """Split "a child ran" into "an interpreter started" and "its stack
        imported", with nothing of the tool's own compute in either.

        Two probes with the SAME interpreter, differing by one import:

            python -c "pass"                          -> interpreter start
            python -c "sys.path.insert(0, src); import <package>"

        The second minus the first is what importing torch/monai/nnU-Net costs.
        Both are medians over `repetitions`, and both run cold-cache-free on
        purpose: the page cache is warm after the first, which is exactly the
        state a server that has already served one request is in.
        """
        interpreter = self.interpreter(tool)
        package = tool.local.package if tool.local else None
        if not package:
            raise LocalPathError(
                f"Tool '{tool.name}' declares no local.package, so its import cost "
                f"cannot be separated from its interpreter start."
            )
        source = self.source_dir(tool)
        bare = _median(self._time_probe(interpreter, "pass", repetitions))
        loaded = _median(
            self._time_probe(
                interpreter,
                f"import sys; sys.path.insert(0, {source!r}); import {package}",
                repetitions,
            )
        )
        return {
            "interpreter_start_seconds": bare,
            "import_stack_seconds": max(0.0, loaded - bare),
            "interpreter_plus_import_seconds": loaded,
            "repetitions": repetitions,
            "package": package,
        }

    def measure_exec_overhead(self, repetitions: int = 5) -> dict:
        """What `docker exec` itself costs, with no tool in it.

        Container mode reaches the interpreter through `docker exec`, and the
        server does not: it forks the child directly. So a container-mode local
        run carries a fixed per-exec cost the server never pays -- around 60-70 ms
        on this machine, which is nothing against a 60 s segmentation and
        everything against a tool that answers in 100 ms.

        Measuring it is the only honest way to report the local arm: the number
        is recorded next to every local run so a reviewer can subtract it, rather
        than being buried in a phase and quietly inflating the "local is slower
        than we said" story. Host mode has no wrapper, so the overhead is
        structurally zero and is reported as such.
        """
        if not self.in_container:
            return {
                "mode": self.spec.mode,
                "median_seconds": 0.0,
                "repetitions": 0,
                "note": "host mode starts the interpreter directly; there is no wrapper",
            }
        durations = []
        for _ in range(repetitions):
            started = time.monotonic()
            completed = subprocess.run(
                ["docker", "exec", "-u", self.spec.container_user, self.spec.container, "true"],
                capture_output=True,
                timeout=_DOCKER_TIMEOUT_SECONDS,
            )
            if completed.returncode != 0:
                return {"mode": self.spec.mode, "median_seconds": None, "repetitions": 0,
                        "note": "the no-op probe failed; the overhead is unknown"}
            durations.append(time.monotonic() - started)
        return {
            "mode": self.spec.mode,
            "median_seconds": _median(durations),
            "min_seconds": min(durations),
            "max_seconds": max(durations),
            "repetitions": repetitions,
            "note": "per `docker exec`; a local run makes several (job setup, then the tool)",
        }

    def _time_probe(self, interpreter: str, code: str, repetitions: int) -> list:
        durations = []
        for _ in range(repetitions):
            argv = [interpreter, "-c", code]
            if self.in_container:
                argv = [
                    "docker", "exec", "-u", self.spec.container_user, self.spec.container
                ] + argv
            started = time.monotonic()
            completed = subprocess.run(argv, capture_output=True, timeout=600)
            elapsed = time.monotonic() - started
            if completed.returncode != 0:
                raise LocalPathError(
                    "Startup probe failed: "
                    + completed.stderr.decode("utf-8", "replace")[-600:]
                )
            durations.append(elapsed)
        return durations


def _write_local_file(path: str, content: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(content)


def _read_local_file(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError:
        return None


def _median(values: list) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0
