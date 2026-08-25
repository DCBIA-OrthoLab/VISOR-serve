"""B3 -- what the supervisor costs.

`AREG_IOSCBCT` runs in three modes and they differ EXACTLY by how many children
the supervisor starts:

    Registration      the orchestrator alone
    Semi-Automated    plus the children a partly-specified input needs
    Fully-Automated   plus the rest of the chain

So the difference between two modes is the chain cost, measured rather than
modelled. That is the first half of this campaign.

The second half separates, per child, the price of ISOLATION from the price of
the work. Each child is a separate interpreter that imports its own stack before
it computes anything, and a sceptical reviewer will ask how much of the chain is
that. Two probes with the child's own interpreter answer it:

    python -c "pass"                                 -> interpreter start
    python -c "sys.path.insert(0, src); import PKG"   -> plus its whole stack

The second minus the first is the import. Neither touches a model or a file, so
neither contains any compute. The child's actual compute is then its measured
`local` run minus the two.

This produces `startup` records, which have no `path` in the usual sense: they
are marked `path: "local"` because they are executed by the local runner, and
carry `measurement: "startup"` in `extra` so a summary can keep them apart from
whole runs.
"""

from __future__ import annotations

from typing import Iterator

from ..provenance import utc_now
from ..recording import STATUS_FAILED, STATUS_OK, PhaseTimer, RunRecord
from ..settings import PATH_LOCAL, Config, ConfigError
from . import _common

NAME = "b3"
DESCRIPTION = "Supervisor cost: chain modes, and the price of isolation per child"

DEFAULT_REPETITIONS = 3
DEFAULT_STARTUP_REPETITIONS = 5

MODE_ARGUMENT = "automation"


def build_plan(config: Config, options: dict) -> list:
    section = config.campaign(NAME)
    repetitions = int(options.get("reps") or section.get("reps", DEFAULT_REPETITIONS))
    orchestrator = str(section.get("tool", "AREG_IOSCBCT"))
    modes = list(section.get("modes", []))
    if not modes:
        raise ConfigError(f"campaigns.{NAME}.modes is empty; the chain cost is the difference "
                          f"between modes, so at least two are needed.")
    path = str(section.get("path", PATH_LOCAL))
    children = list(section.get("children", []))
    startup_repetitions = int(section.get("startup_reps", DEFAULT_STARTUP_REPETITIONS))

    # `--tools` selects a subset here as it does everywhere else: the startup
    # probes are cheap and the chains are not, so "just the isolation cost, in
    # the other local mode" has to be expressible without re-running three
    # chains to get it.
    wanted = options.get("tools")

    tool = config.tool(orchestrator)
    plan = []
    for mode in (modes if (not wanted or orchestrator in wanted) else []):
        plan.append(
            _common.plan_item(
                NAME, tool, path, repetitions,
                measurement="chain",
                mode=str(mode),
                # A deeper chain is a longer run; the estimate is per mode when
                # the config gives one, so the disk guard and the time estimate
                # are not all quoted at the shallowest mode's cost.
                seconds_each=float(
                    (section.get("seconds_by_mode") or {}).get(mode, tool.estimated_seconds)
                ),
            )
        )
    for child in children:
        if wanted and child not in wanted:
            continue
        child_tool = config.tool(child)
        if not child_tool.supports_local:
            plan.append(
                _common.plan_item(
                    NAME, child_tool, PATH_LOCAL, 0,
                    measurement="startup", skipped=True,
                    skip_reason=(child_tool.local.reason if child_tool.local
                                 else "no local.folder in config.yaml"),
                )
            )
            continue
        plan.append(
            _common.plan_item(
                NAME, child_tool, PATH_LOCAL, 1,
                measurement="startup",
                startup_reps=startup_repetitions,
                # Two short probes per repetition; nothing is written.
                seconds_each=startup_repetitions * 2 * 5.0,
                output_mb=0.0,
            )
        )
    return plan


def execute(item: dict, context: _common.Context) -> Iterator[RunRecord]:
    if item.get("skipped"):
        return
    tool = context.config.tool(item["tool"])
    if item.get("measurement") == "startup":
        yield _startup_record(item, context, tool)
        return

    mode = item["mode"]
    section = context.config.campaign(NAME)
    # The mode is an ARGUMENT of the orchestrator, so it is applied to a copy of
    # the tool spec rather than mutating the shared one -- three plan items
    # otherwise end up all running the last mode.
    variant = _with_mode(tool, mode, section)
    for repetition in range(1, int(item["runs"]) + 1):
        yield _common.execute(
            context,
            NAME,
            variant,
            item["path"],
            repetition,
            warmup=repetition == 1,
            extra={"measurement": "chain", "mode": mode},
            capture_stderr=True,
        )


def _with_mode(tool, mode: str, section: dict = None):
    """The tool as this mode runs it: the mode itself, plus what it changes.

    The modes do not take the same arguments. AREG_IOSCBCT's Registration mode
    registers on landmarks the caller supplies; the automated modes predict
    them, and SKIP the prediction when they are supplied -- so handing all three
    the same inputs would silently collapse a four-child chain to one child and
    report the difference as the chain cost. `campaigns.b3.per_mode` carries
    that difference as data, with `null` meaning "this mode does not take that
    input at all".
    """
    from dataclasses import replace

    from ..settings import resolve_path

    overrides = ((section or {}).get("per_mode") or {}).get(mode) or {}
    args = dict(tool.args, **{MODE_ARGUMENT: mode})
    args.update(overrides.get("args") or {})

    files = dict(tool.files)
    for argument, path in (overrides.get("files") or {}).items():
        if path is None:
            files.pop(argument, None)
        else:
            files[argument] = resolve_path(str(path))

    server_files = dict(tool.server_files)
    for argument, hosted in (overrides.get("server_files") or {}).items():
        if hosted is None:
            server_files.pop(argument, None)
        else:
            server_files[argument] = hosted

    return replace(tool, args=args, files=files, server_files=server_files)


def _startup_record(item: dict, context: _common.Context, tool) -> RunRecord:
    """Interpreter start and stack import for one child, with no compute in it."""
    timer = PhaseTimer()
    started_wall = utc_now()
    runner = context.local_runner()
    status = STATUS_OK
    error_type = error_message = None
    # The mode belongs in the record: the image's virtualenvs are
    # hardlink-deduplicated and the host checkout's are not, and whether that
    # changes first-import time is exactly what two runs of these probes answer.
    extra = {"measurement": "startup", "local_mode": context.config.local.mode}
    try:
        measured = runner.measure_startup(tool, int(item.get("startup_reps",
                                                             DEFAULT_STARTUP_REPETITIONS)))
        extra.update(measured)
        timer.add("interpreter_start", measured["interpreter_start_seconds"])
        timer.add("import_stack", measured["import_stack_seconds"])
    except Exception as error:  # noqa: BLE001 - recorded, never dropped
        status = STATUS_FAILED
        error_type = type(error).__name__
        error_message = str(error)[:4000]

    total = timer.total
    return RunRecord(
        campaign=NAME,
        tool=tool.name,
        path=PATH_LOCAL,
        repetition=1,
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
