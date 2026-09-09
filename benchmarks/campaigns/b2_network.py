"""B2 -- where the time in a remote call actually goes.

One remote call, split into `pack`, `upload`, `server_exec`, `download`,
`unpack` and `other`, at three payload sizes, with parallel transfer on and off.

Two things about the method are load-bearing and are stated here so a reviewer
does not have to infer them from the code:

**The split is honest about `server_exec`.** Inputs are pushed through the
chunked-upload endpoints FIRST, so the POST /run that follows carries only
references. What that request costs is therefore the server's own execution plus
building the result -- there is no upload hidden inside it. This is also what the
real client does for anything over 16 MB, so it is not a benchmark-only shape.

**Parallel on and off differ by ONE number.** Both arms use the same chunked
upload and the same ranged download; `parallelism` is 4 in one and 1 in the
other. Turning parallelism "off" by falling back to a single whole-file POST
would change the protocol as well, and the difference would no longer be
attributable to concurrency.

The expected story -- transfer dominates, parallelism hides most of it, unpacking
is the residual -- is a HYPOTHESIS. This campaign is what confirms or refutes it.
"""

from __future__ import annotations

from typing import Iterator

from ..recording import RunRecord
from ..settings import PATH_LAN, PATH_LOOPBACK, Config, ConfigError
from . import _common

NAME = "b2"
DESCRIPTION = "Decomposition of a remote call, by payload size and parallelism"

DEFAULT_REPETITIONS = 4


def build_plan(config: Config, options: dict) -> list:
    section = config.campaign(NAME)
    repetitions = int(options.get("reps") or section.get("reps", DEFAULT_REPETITIONS))
    path = str(section.get("path", PATH_LOOPBACK))
    if path not in (PATH_LOOPBACK, PATH_LAN):
        raise ConfigError(
            f"campaigns.{NAME}.path must be 'loopback' or 'lan'; 'local' has no transfer to split."
        )
    parallel_settings = section.get("parallelism", [config.transfer.parallelism, 1])
    if not isinstance(parallel_settings, list) or not parallel_settings:
        raise ConfigError(f"campaigns.{NAME}.parallelism must be a non-empty list of integers")

    wanted = options.get("tools") or section.get("tools") or []
    if not wanted:
        raise ConfigError(f"campaigns.{NAME}.tools is empty; there is nothing to measure.")

    plan = []
    for name in wanted:
        tool = config.tool(name)
        for parallelism in parallel_settings:
            plan.append(
                _common.plan_item(
                    NAME, tool, path, repetitions,
                    parallelism=int(parallelism),
                    payload_label=tool.payload_label or "unlabelled",
                )
            )
    return plan


def execute(item: dict, context: _common.Context) -> Iterator[RunRecord]:
    tool = context.config.tool(item["tool"])
    parallelism = int(item["parallelism"])
    for repetition in range(1, int(item["runs"]) + 1):
        yield _common.execute(
            context,
            NAME,
            tool,
            item["path"],
            repetition,
            # Repetition 1 still pays for the model load on the server side; it
            # is marked and excluded from the summary exactly as in B1.
            warmup=repetition == 1,
            parallelism=parallelism,
            extra={
                "parallelism": parallelism,
                "parallel_transfer": parallelism > 1,
                "payload_label": item.get("payload_label"),
            },
        )
