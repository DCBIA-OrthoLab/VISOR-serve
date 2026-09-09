"""B1 -- the same tool, the same input, three execution paths.

    local     the tool's own interpreter, no HTTP
    loopback  the API on this machine: protocol and packing, no wire
    lan       the API from another machine: the wire as well

The protocol is fixed and stated here rather than in the paper alone, because a
reviewer re-running this has to get the same shape of number:

- at least six repetitions per (tool, path);
- **repetition 1 is discarded** as warm-up. It pays for the model load and a
  cold page cache, which is a real cost but a different one -- it is the cost of
  the FIRST run, not of a run. It is written to the raw file with
  `warmup: true`, so the discarding is visible and reversible;
- median and full range are reported, not mean and standard deviation. Six
  points from a distribution with a hard floor and a long tail have no
  meaningful standard deviation, and quoting one would imply a symmetry the data
  does not have.
"""

from __future__ import annotations

from typing import Iterator

from ..recording import RunRecord
from ..settings import PATHS, Config, ConfigError
from . import _common

NAME = "b1"
DESCRIPTION = "Latency of one tool through local / loopback / lan"

DEFAULT_REPETITIONS = 6
DEFAULT_WARMUP = 1


def build_plan(config: Config, options: dict) -> list:
    section = config.campaign(NAME)
    repetitions = int(options.get("reps") or section.get("reps", DEFAULT_REPETITIONS))
    warmup = int(section.get("warmup", DEFAULT_WARMUP))
    if repetitions <= warmup:
        raise ConfigError(
            f"campaigns.{NAME}: reps ({repetitions}) must exceed warmup ({warmup}), "
            f"or every run would be discarded."
        )
    paths = list(options.get("paths") or section.get("paths", ["local", "loopback"]))
    for path in paths:
        if path not in PATHS:
            raise ConfigError(f"campaigns.{NAME}.paths: unknown path {path!r}; expected {PATHS}")

    wanted = options.get("tools") or section.get("tools") or []
    if not wanted:
        raise ConfigError(f"campaigns.{NAME}.tools is empty; there is nothing to measure.")

    plan = []
    for name in wanted:
        tool = config.tool(name)
        for path in paths:
            if path == "local" and not tool.supports_local:
                reason = (tool.local.reason if tool.local else "no local.folder in config.yaml")
                plan.append(
                    _common.plan_item(
                        NAME, tool, path, 0, warmup=warmup, skipped=True, skip_reason=reason
                    )
                )
                continue
            plan.append(_common.plan_item(NAME, tool, path, repetitions, warmup=warmup))
    return plan


def execute(item: dict, context: _common.Context) -> Iterator[RunRecord]:
    if item.get("skipped"):
        return
    tool = context.config.tool(item["tool"])
    warmup = int(item.get("warmup", DEFAULT_WARMUP))
    for repetition in range(1, int(item["runs"]) + 1):
        yield _common.execute(
            context,
            NAME,
            tool,
            item["path"],
            repetition,
            warmup=repetition <= warmup,
            extra={"payload_label": tool.payload_label},
        )
