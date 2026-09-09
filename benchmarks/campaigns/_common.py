"""What every campaign shares: the context it runs in, and one run of one tool.

A campaign module decides WHICH runs happen and in what order. This module
decides what a run IS, so that a `local` run in B1 and a `local` run in B5 are
the same thing measured the same way -- otherwise the two campaigns could not be
read against each other, which is the entire point of having both.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .. import artifacts
from ..execution import local as local_path
from ..execution import remote as remote_path
from ..provenance import utc_now
from ..recording import STATUS_FAILED, STATUS_OK, PhaseTimer, RunRecord
from ..settings import PATH_LAN, PATH_LOCAL, PATH_LOOPBACK, Config, ToolSpec


@dataclass
class Context:
    """Everything a campaign needs that is not in the config file."""

    config: Config
    token: Optional[str]
    provenance: dict
    scratch_root: str
    keep_artifacts: bool = False
    # Filled by run.py once, so a campaign never builds a second client.
    _local: Optional[local_path.LocalRunner] = None
    _clients: dict = field(default_factory=dict)

    _overhead: Optional[dict] = None

    def local_runner(self) -> local_path.LocalRunner:
        if self._local is None:
            self._local = local_path.LocalRunner(self.config.local)
        return self._local

    def local_overhead(self) -> dict:
        """What the local path's wrapper costs, measured once per invocation.

        Recorded next to every local run so it can be subtracted, rather than
        hidden inside a phase. See LocalRunner.measure_exec_overhead.
        """
        if self._overhead is None:
            try:
                self._overhead = self.local_runner().measure_exec_overhead()
            except Exception as error:  # noqa: BLE001 - reported, never fatal
                self._overhead = {
                    "median_seconds": None,
                    "note": f"{type(error).__name__}: {error}",
                }
        return self._overhead

    def client(self, path: str, parallelism: Optional[int] = None) -> remote_path.RemoteClient:
        """One session per (path, parallelism). Reused across repetitions on
        purpose: a fresh TCP connection per run would put connection setup in
        every measurement and hide it in none."""
        key = (path, parallelism)
        if key not in self._clients:
            if not self.token:
                raise RuntimeError(
                    "No API token; the remote paths cannot be taken. "
                    "Set API_TOKEN or put it in the server repo's .env."
                )
            base_url = self.config.server.base_url
            if path == PATH_LAN:
                if not self.config.server.lan_base_url:
                    raise RuntimeError(
                        "The 'lan' path needs server.lan_base_url in config.yaml -- the "
                        "address of THIS server as the remote client reaches it."
                    )
                base_url = self.config.server.lan_base_url
            self._clients[key] = remote_path.RemoteClient(
                self.config.server,
                self.config.transfer,
                self.token,
                base_url=base_url,
                parallelism=parallelism,
            )
        return self._clients[key]

    def workspace(self, label: str) -> str:
        path = os.path.join(self.scratch_root, f"{label}_{uuid.uuid4().hex[:8]}")
        os.makedirs(path, exist_ok=True)
        return path

    def close(self) -> None:
        for client in self._clients.values():
            client.close()


def path_available(context: Context, path: str) -> Optional[str]:
    """Why `path` cannot be taken here, or None. Checked once per campaign."""
    if path == PATH_LOCAL:
        return context.local_runner().unavailable_reason()
    if path in (PATH_LOOPBACK, PATH_LAN):
        try:
            return context.client(path).unavailable_reason()
        except RuntimeError as error:
            return str(error)
    return f"unknown path {path!r}"


def local_params(context: Context, tool: ToolSpec, job: local_path.LocalRun) -> dict:
    """The `params` the server would have written into job.json.

    Uploads become staged paths, server-hosted names become paths under the
    mounted DATA directory, and `output_dir` is filled in by us because on the
    other path it is filled in by the server. Everything else is the tool's own
    arguments, untouched.
    """
    runner = context.local_runner()
    params = dict(tool.args)
    for argument, path in tool.files.items():
        params[argument] = runner.stage_input(path, job.job_dir, argument)
    for argument, hosted in tool.server_files.items():
        params[argument] = runner.data_path(
            tool.data_slug or tool.name, hosted["kind"], hosted["name"]
        )
    if tool.wants_output_dir:
        params["output_dir"] = job.output_dir
    return params


SUPERVISOR_LOGGER = "sadt.supervisor"


def supervisor_log(stderr: str) -> list:
    """The supervisor's own lines out of a chain's stderr, timestamps kept.

    This is what says WHICH children a supervised run actually started, in what
    order, and when -- read off the run itself rather than off the call graph
    documentation, which is a different claim. Kept as a short list rather than
    the whole stream so it can live in every record.
    """
    return [
        line for line in (stderr or "").splitlines() if f" {SUPERVISOR_LOGGER}: " in line
    ]


def run_local(
    context: Context,
    tool: ToolSpec,
    timer: PhaseTimer,
    collect_to: Optional[str] = None,
    timeout: Optional[float] = None,
    capture_stderr: bool = False,
) -> dict:
    """One `local` run. Returns the campaign-facing `extra` fields.

    `collect_to` is where the artifacts are copied for a later comparison; None
    means the job directory is taken down without reading it, which is what the
    latency campaign wants (copying 200 MB out of a container is not part of
    running a tool).

    `capture_stderr` adds the supervisor's log lines to the record. Off by
    default: it is evidence B3 needs and noise everywhere else.
    """
    runner = context.local_runner()
    job = None
    try:
        with timer.phase("job_setup"):
            job = runner.prepare_job(tool, {})
            params = local_params(context, tool, job)
            runner.write_job_file(job, tool, params)
        with timer.phase("compute"):
            runner.execute(tool, job, timeout=timeout)
        collected = None
        with timer.phase("collect"):
            if collect_to:
                collected = runner.fetch_outputs(job.job_dir, collect_to)
        fields = {
            "job_id": job.job_id,
            "job_dir": job.job_dir,
            "result": job.result,
            "peak_vram_bytes": job.peak_vram_bytes,
            "collected_dir": collected,
            "command": job.command,
            "local_mode": context.config.local.mode,
        }
        if capture_stderr:
            fields["supervisor_log"] = supervisor_log(job.stderr_tail)
            fields["stderr_tail"] = job.stderr_tail[-8000:]
        return fields
    finally:
        if job is not None and not context.keep_artifacts:
            runner.remove_job(job.job_dir)


def run_remote(
    context: Context,
    tool: ToolSpec,
    path: str,
    timer: PhaseTimer,
    parallelism: Optional[int] = None,
    keep_workspace: bool = False,
) -> dict:
    """One `loopback` or `lan` run. Returns the campaign-facing `extra` fields."""
    client = context.client(path, parallelism)
    workspace = context.workspace(f"{path}_{tool.name}")
    try:
        outcome = client.run(tool, workspace, timer)
        return {
            "status_code": outcome.status_code,
            "delivery": outcome.delivery,
            "result": outcome.result,
            "result_path": outcome.result_path,
            "unpacked_dir": outcome.unpacked_dir,
            "bytes_uploaded": outcome.bytes_uploaded,
            "bytes_downloaded": outcome.bytes_downloaded,
            "parts_uploaded": outcome.parts_uploaded,
            "chunked_arguments": outcome.chunked_arguments,
            "inline_arguments": outcome.inline_arguments,
            "parallelism": client.parallelism,
            "base_url": client.base_url,
            "workspace": workspace if keep_workspace else None,
        }
    finally:
        if not keep_workspace and not context.keep_artifacts:
            remote_path.clear_workspace(workspace)


def execute(
    context: Context,
    campaign: str,
    tool: ToolSpec,
    path: str,
    repetition: int,
    warmup: bool = False,
    extra: Optional[dict] = None,
    parallelism: Optional[int] = None,
    collect_to: Optional[str] = None,
    keep_workspace: bool = False,
    timeout: Optional[float] = None,
    capture_stderr: bool = False,
) -> RunRecord:
    """One run, through one path, recorded whether it worked or not.

    Never raises for a tool that failed: a failure is a measurement (of the
    server's error path, and of the fact that this tool does not run here) and
    is written to the raw file with its error type and message. It DOES raise
    for a harness-level problem -- no token, no such path -- because that is not
    a property of the tool and repeating it would only produce more of the same
    record.
    """
    timer = PhaseTimer()
    started_wall = utc_now()
    started = time.monotonic()
    fields = dict(extra or {})
    status = STATUS_OK
    error_type = error_message = None

    try:
        if path == PATH_LOCAL:
            # Set BEFORE the run, so a failed local run still says which mode it
            # was taken in and what the wrapper costs there.
            fields["local_mode"] = context.config.local.mode
            fields["exec_overhead"] = context.local_overhead()
            fields.update(
                run_local(
                    context, tool, timer, collect_to=collect_to, timeout=timeout,
                    capture_stderr=capture_stderr,
                )
            )
        else:
            fields.update(
                run_remote(
                    context, tool, path, timer,
                    parallelism=parallelism, keep_workspace=keep_workspace,
                )
            )
    except (local_path.ToolRunFailure, remote_path.RemoteError, local_path.LocalPathError) as error:
        status = STATUS_FAILED
        error_type = type(error).__name__
        error_message = str(error)[:4000]
        if isinstance(error, local_path.ToolRunFailure):
            fields["tool_error_type"] = error.error_type
            fields["stderr_tail"] = error.stderr_tail[-2000:]
            if capture_stderr:
                fields["supervisor_log"] = supervisor_log(error.stderr_tail)
        if isinstance(error, remote_path.RemoteError):
            fields["status_code"] = error.status_code
    except Exception as error:  # noqa: BLE001 - recorded, never dropped
        status = STATUS_FAILED
        error_type = type(error).__name__
        error_message = str(error)[:4000]

    total = time.monotonic() - started
    return RunRecord(
        campaign=campaign,
        tool=tool.name,
        path=path,
        repetition=repetition,
        started_at=started_wall,
        finished_at=utc_now(),
        total_seconds=total,
        status=status,
        phases=timer.finalize(total),
        warmup=warmup,
        error_type=error_type,
        error_message=error_message,
        extra=fields,
        provenance=context.provenance,
    )


def plan_item(campaign: str, tool: ToolSpec, path: str, runs: int, **extra) -> dict:
    """The shape --dry-run prints and the disk guard sizes.

    A plain dict rather than a dataclass on purpose: a campaign adds its own
    keys (payload label, concurrency, mode) and every one of them has to survive
    into the printed plan without a schema change here.
    """
    item = {
        "campaign": campaign,
        "tool": tool.name,
        "path": path,
        "runs": runs,
        "seconds_each": tool.estimated_seconds,
        "output_mb": tool.estimated_output_mb,
    }
    item.update(extra)
    return item


def estimated_seconds(plan: list) -> float:
    return sum(float(item.get("seconds_each", 0)) * int(item.get("runs", 1)) for item in plan)
