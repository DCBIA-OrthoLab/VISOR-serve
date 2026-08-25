"""B5 -- do the two paths produce the same bytes?

For each tool: one `local` run and one remote run, on the same input with the
same arguments, and then a file-by-file comparison of what each produced.

The comparison is `artifacts.py`, and it is deliberately unforgiving. Where the
bytes differ the record names the FILE, and then says as much as can be said
about how it differs: which JSON keys, which text lines, and -- for an imaging
output -- a numeric distance (max, mean, RMS of the voxelwise difference, how
many voxels moved, and whether the geometry is identical). A difference is never
recorded as "essentially identical". If a header timestamp is the whole
difference, the record says the difference is a header timestamp, which is a
stronger claim than a hash match would have been.

Comparing paths, not directory layouts: the local side's artifacts are the
contents of the job's `output/`, the remote side's are the contents of the result
archive after unpacking. Both are keyed by path RELATIVE to their own root, so a
run whose files sit at the same relative paths compares file to file. A tool
whose archive nests its output one level deeper would show up as "only_left" /
"only_right" for every file, which is itself a finding and not something to
paper over with fuzzy matching.
"""

from __future__ import annotations

import os
from typing import Iterator

from .. import artifacts
from ..provenance import utc_now
from ..recording import STATUS_FAILED, STATUS_OK, PhaseTimer, RunRecord
from ..settings import PATH_LOCAL, PATH_LOOPBACK, Config, ConfigError
from . import _common

NAME = "b5"
DESCRIPTION = "Byte-level parity between the local and the remote path"


def build_plan(config: Config, options: dict) -> list:
    section = config.campaign(NAME)
    remote = str(section.get("remote_path", PATH_LOOPBACK))
    if remote == PATH_LOCAL:
        raise ConfigError(f"campaigns.{NAME}.remote_path cannot be 'local'; parity compares "
                          f"the local path against a remote one.")
    repetitions = int(options.get("reps") or section.get("reps", 1))
    wanted = options.get("tools") or section.get("tools") or []
    if not wanted:
        raise ConfigError(f"campaigns.{NAME}.tools is empty; there is nothing to compare.")

    plan = []
    for name in wanted:
        tool = config.tool(name)
        if not tool.supports_local:
            plan.append(
                _common.plan_item(
                    NAME, tool, PATH_LOCAL, 0, skipped=True,
                    skip_reason=(tool.local.reason if tool.local
                                 else "no local.folder in config.yaml"),
                )
            )
            continue
        plan.append(
            _common.plan_item(
                NAME, tool, f"{PATH_LOCAL}+{remote}", repetitions,
                remote_path=remote,
                # Both sides are kept on disk at once for the comparison.
                output_mb=tool.estimated_output_mb * 2,
                seconds_each=tool.estimated_seconds * 2,
            )
        )
    return plan


def execute(item: dict, context: _common.Context) -> Iterator[RunRecord]:
    if item.get("skipped"):
        return
    tool = context.config.tool(item["tool"])
    remote = item["remote_path"]
    section = context.config.campaign(NAME)
    imaging_interpreter = _imaging_interpreter(context, tool, section)

    control = bool(section.get("local_control"))

    for repetition in range(1, int(item["runs"]) + 1):
        workspace = context.workspace(f"parity_{tool.name}")
        local_dir = os.path.join(workspace, "local")
        try:
            local_record = _common.execute(
                context, NAME, tool, PATH_LOCAL, repetition,
                collect_to=local_dir,
                extra={"side": "local", "pair": repetition},
            )
            yield local_record

            # The control the parity claim needs before it can name a cause.
            # "local and remote differ" says nothing about the PATH unless the
            # tool reproduces itself on one path first: a network that is not
            # deterministic differs from itself, and attributing that to the
            # protocol would be wrong. Off by default because it doubles the
            # local side; turned on for whichever tools actually differ.
            control_record = None
            control_dir = os.path.join(workspace, "local_control")
            if control:
                control_record = _common.execute(
                    context, NAME, tool, PATH_LOCAL, repetition,
                    collect_to=control_dir,
                    extra={"side": "local_control", "pair": repetition},
                )
                yield control_record

            remote_record = _common.execute(
                context, NAME, tool, remote, repetition,
                keep_workspace=True,
                extra={"side": "remote", "pair": repetition},
            )
            yield remote_record

            if control_record is not None:
                yield _control_record(
                    context, tool, repetition, local_record, control_record,
                    local_dir, control_dir, imaging_interpreter,
                )

            yield _comparison_record(
                context, tool, repetition, remote, local_record, remote_record,
                local_dir, imaging_interpreter,
            )
        finally:
            if not context.keep_artifacts:
                from ..execution.remote import clear_workspace

                clear_workspace(workspace)


def _control_record(
    context: _common.Context,
    tool,
    repetition: int,
    first: RunRecord,
    second: RunRecord,
    first_dir: str,
    second_dir: str,
    imaging_interpreter,
) -> RunRecord:
    """The same tool, the same path, twice. The determinism baseline.

    Recorded with `side: "local_control"` and `path: "local+local"` so it can
    never be read as a local-versus-remote row. Nothing is renamed here: both
    sides took the same path and gave their artifacts the same names, so a
    difference is the tool's own run-to-run variation and nothing else.
    """
    timer = PhaseTimer()
    started_wall = utc_now()
    extra = {
        "side": "local_control",
        "pair": repetition,
        "local_mode": context.config.local.mode,
        "first_status": first.status,
        "second_status": second.status,
        "imaging_interpreter": imaging_interpreter,
    }
    status = STATUS_OK
    error_type = error_message = None

    if first.status != STATUS_OK or second.status != STATUS_OK:
        status = STATUS_FAILED
        error_type = "IncompleteControlPair"
        error_message = (
            f"first={first.status} ({first.error_type}); "
            f"second={second.status} ({second.error_type})"
        )
    else:
        try:
            report = artifacts.compare(
                artifacts.snapshot(first_dir), artifacts.snapshot(second_dir)
            )
            for name in report.differing:
                report.details[name] = artifacts.describe_difference(
                    name, first_dir, second_dir, imaging_interpreter
                )
            extra["parity"] = report.as_dict()
            extra["deterministic"] = report.ok
        except Exception as error:  # noqa: BLE001 - recorded, never dropped
            status = STATUS_FAILED
            error_type = type(error).__name__
            error_message = str(error)[:4000]

    total = timer.total
    return RunRecord(
        campaign=NAME,
        tool=tool.name,
        path=f"{PATH_LOCAL}+{PATH_LOCAL}",
        repetition=repetition,
        started_at=started_wall,
        finished_at=utc_now(),
        total_seconds=total,
        status=status,
        phases=timer.finalize(total),
        error_type=error_type,
        error_message=error_message,
        extra=extra,
        provenance=context.provenance,
    )


def _remote_artifact_root(record: RunRecord) -> str:
    """Where the remote side's artifacts ended up.

    The unpacked archive when there was one; otherwise the single downloaded
    file, whose directory is its root. A text tool has neither, and the caller
    records an empty snapshot rather than inventing a path.
    """
    unpacked = record.extra.get("unpacked_dir")
    if unpacked:
        return unpacked
    result_path = record.extra.get("result_path")
    if result_path:
        return result_path
    return ""


def _comparison_record(
    context: _common.Context,
    tool,
    repetition: int,
    remote: str,
    local_record: RunRecord,
    remote_record: RunRecord,
    local_dir: str,
    imaging_interpreter,
) -> RunRecord:
    timer = PhaseTimer()
    started_wall = utc_now()
    status = STATUS_OK
    error_type = error_message = None
    extra = {
        "side": "comparison",
        "pair": repetition,
        "remote_path": remote,
        # Which local mode this pair compared. Without it a summary built from
        # a host run and a container run cannot tell the two arms apart, and
        # those two arms are the whole point of running B5 twice.
        "local_mode": context.config.local.mode,
        "local_status": local_record.status,
        "remote_status": remote_record.status,
        "imaging_interpreter": imaging_interpreter,
    }

    if local_record.status != STATUS_OK or remote_record.status != STATUS_OK:
        # Not a parity result: one side never produced anything. Recorded as a
        # failure so the summary's failure count is right, and naming which side.
        status = STATUS_FAILED
        error_type = "IncompleteParityPair"
        error_message = (
            f"local={local_record.status} ({local_record.error_type}); "
            f"{remote}={remote_record.status} ({remote_record.error_type})"
        )
        return _record(tool, repetition, remote, started_wall, timer, status,
                       error_type, error_message, extra, context)

    try:
        remote_root = _remote_artifact_root(remote_record)
        left = artifacts.snapshot(local_dir)
        right = artifacts.snapshot(remote_root) if remote_root else {}
        report = artifacts.compare(left, right)
        for name in report.differing:
            report.details[name] = artifacts.describe_difference(
                name, local_dir, remote_root, imaging_interpreter
            )
        # The second question. The strict comparison above is by NAME, and a
        # travelling input does not keep its name: the server stages it under
        # the argument it arrived as. So a tool that names its outputs after its
        # input produces differently-SPELLED artifacts remotely, which the
        # strict pass reports as only_left/only_right and must go on reporting.
        # This pairs those leftovers by the server's own staging rule and asks
        # whether the CONTENTS match. Both answers are recorded; neither
        # replaces the other.
        renamed = artifacts.pair_renamed(
            report,
            artifacts.rename_substitutions(
                tool.files, remote_record.extra.get("chunked_arguments")
            ),
            local_dir,
            remote_root,
            imaging_interpreter,
        )
        extra.update(
            {
                "local_root": local_dir,
                "remote_root": remote_root,
                "local_artifacts": left,
                "remote_artifacts": right,
                "parity": dict(report.as_dict(), renamed=renamed),
                "local_result": local_record.extra.get("result"),
                "remote_result": remote_record.extra.get("result"),
            }
        )
        # A difference is a RESULT, not an error: the run succeeded and the
        # answer is "they differ". `parity_ok` is false and the summary reports
        # it; the status stays "ok" so it is not counted as a harness failure.
        extra["parity_ok"] = report.ok
        # Same bytes under whatever name each path gave them.
        extra["content_parity_ok"] = renamed.get("content_ok")
    except Exception as error:  # noqa: BLE001 - recorded, never dropped
        status = STATUS_FAILED
        error_type = type(error).__name__
        error_message = str(error)[:4000]

    return _record(tool, repetition, remote, started_wall, timer, status,
                   error_type, error_message, extra, context)


def _record(tool, repetition, remote, started_wall, timer, status,
            error_type, error_message, extra, context) -> RunRecord:
    total = timer.total
    return RunRecord(
        campaign=NAME,
        tool=tool.name,
        path=f"{PATH_LOCAL}+{remote}",
        repetition=repetition,
        started_at=started_wall,
        finished_at=utc_now(),
        total_seconds=total,
        status=status,
        phases=timer.finalize(total),
        error_type=error_type,
        error_message=error_message,
        extra=extra,
        provenance=context.provenance,
    )


def _imaging_interpreter(context: _common.Context, tool, section: dict):
    """Which interpreter reads the images for the numeric distance.

    Per tool first, then a campaign-wide default, then the tool's OWN
    interpreter -- which is the right answer when the harness runs on the same
    machine as the tools, because it is the stack that wrote the file. In
    container mode there is no host-side interpreter to point at, so the config
    has to name one and the record says so when it does not.
    """
    from ..settings import resolve_path

    per_tool = (section.get("imaging_interpreter_by_tool") or {}).get(tool.name)
    if per_tool:
        return resolve_path(str(per_tool))
    default = section.get("imaging_interpreter")
    if default:
        return resolve_path(str(default))
    runner = context.local_runner()
    if not runner.in_container and tool.supports_local:
        candidate = runner.interpreter(tool)
        if os.path.isfile(candidate):
            return candidate
    return None
