"""The CLI.

    python -m benchmarks.run --campaign b1 --reps 6 --tools Test_Tool,Example_Tool
    python -m benchmarks.run --campaign b2 --dry-run

`--dry-run` validates the config, builds the whole plan, checks the disk against
its projected output and prints it -- and starts no process, opens no socket and
writes no file. It is the first thing to run against an edited config, and the
first thing a reviewer should run to see what a campaign would do.

Order of operations for a real run, and none of it is optional:

1. the config is parsed and validated;
2. the plan is built (so a typo in a tool name fails now, not in forty minutes);
3. the disk is checked against the plan's projected output and the run is
   REFUSED if it would not fit;
4. the hardware fingerprint is collected once;
5. one raw file is opened, O_EXCL;
6. every plan item runs, and every run -- including every failure -- is written;
7. the scratch directory is cleaned and a summary is regenerated.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

from . import __version__, guards, summarize
from .campaigns import _common, b1_latency, b2_network, b3_supervisor, b4_concurrency, b5_parity
from .provenance import collect
from .recording import Recorder
from .settings import BENCHMARKS_ROOT, ConfigError, load, read_token

CAMPAIGNS = {
    module.NAME: module
    for module in (b1_latency, b2_network, b3_supervisor, b4_concurrency, b5_parity)
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.run",
        description="Run one benchmark campaign against the SADT tool server.",
    )
    parser.add_argument(
        "--campaign", required=True, choices=sorted(CAMPAIGNS),
        help="which campaign to run; see benchmarks/README.md for what each measures",
    )
    parser.add_argument(
        "--reps", type=int, default=None,
        help="repetitions per (tool, path), overriding the config. B1 discards the first.",
    )
    parser.add_argument(
        "--tools", default=None,
        help="comma-separated subset of the campaign's tools",
    )
    parser.add_argument(
        "--paths", default=None,
        help="comma-separated subset of local,loopback,lan (B1 only)",
    )
    parser.add_argument(
        "--config", default=None,
        help="path to config.yaml (default: the shipped one; $BENCHMARKS_CONFIG also works)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate the config, print the plan, and do nothing else",
    )
    parser.add_argument(
        "--keep-artifacts", action="store_true",
        help="do not delete job directories and downloaded results (fills the disk fast)",
    )
    parser.add_argument(
        "--skip-disk-check", action="store_true",
        help="run even when the projected output does not fit. Say why in the notes.",
    )
    parser.add_argument(
        "--no-summary", action="store_true",
        help="do not regenerate the summary afterwards",
    )
    parser.add_argument("--root", default=BENCHMARKS_ROOT,
                        help="directory holding results/ (default: the benchmarks package)")
    return parser


def options_from(arguments) -> dict:
    return {
        "reps": arguments.reps,
        "tools": [name.strip() for name in arguments.tools.split(",")] if arguments.tools else None,
        "paths": [name.strip() for name in arguments.paths.split(",")] if arguments.paths else None,
    }


def print_plan(campaign: str, plan: list, disk, config) -> None:
    print(f"campaign      : {campaign} -- {CAMPAIGNS[campaign].DESCRIPTION}")
    print(f"config        : {config.source_path}")
    print(f"harness       : {__version__}")
    print(f"server        : {config.server.base_url}")
    print(f"local path    : {config.local.mode}"
          + (f" ({config.local.container})" if config.local.mode == "container" else ""))
    print(f"plan items    : {len(plan)}")
    total_runs = sum(int(item.get("runs", 0)) for item in plan)
    print(f"total runs    : {total_runs}")
    print(f"est. duration : {_duration(_common.estimated_seconds(plan))}")
    print(f"{disk.describe()}")
    print(f"disk verdict  : {'OK' if disk.ok else 'REFUSE'}")
    print("")
    header = f"{'#':>3}  {'tool':<18} {'path':<18} {'runs':>5} {'s/run':>7} {'MB/run':>7}  notes"
    print(header)
    print("-" * len(header))
    for number, item in enumerate(plan, start=1):
        notes = []
        for key in ("mode", "parallelism", "concurrency", "payload_label", "measurement",
                    "remote_path", "jobs_per_client"):
            if item.get(key) is not None:
                notes.append(f"{key}={item[key]}")
        if item.get("skipped"):
            notes.insert(0, f"SKIPPED: {item.get('skip_reason', 'no reason given')}")
        print(
            f"{number:>3}  {str(item['tool']):<18} {str(item['path']):<18} "
            f"{int(item.get('runs', 0)):>5} {float(item.get('seconds_each', 0)):>7.1f} "
            f"{float(item.get('output_mb', 0)):>7.1f}  {', '.join(notes)}"
        )


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, remainder = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {remainder:02d}s"
    return f"{remainder}s"


def check_paths(plan: list, context: _common.Context) -> dict:
    """Which execution paths this machine can actually take.

    Probed ONCE, before the first run: a reviewer whose server is down should be
    told that in one line rather than in thirty identical failure records.
    """
    verdicts = {}
    for item in plan:
        for path in str(item["path"]).split("+"):
            if path not in verdicts:
                verdicts[path] = _common.path_available(context, path)
    return verdicts


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    campaign = arguments.campaign
    module = CAMPAIGNS[campaign]

    try:
        config = load(arguments.config)
        plan = module.build_plan(config, options_from(arguments))
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 2

    disk = guards.check_disk(
        config.guards.scratch_dir, plan, config.guards.min_free_gb, config.guards.margin_gb
    )
    print_plan(campaign, plan, disk, config)

    if arguments.dry_run:
        print("\ndry run: nothing was executed, nothing was written.")
        return 0 if disk.ok or arguments.skip_disk_check else 1

    if not disk.ok and not arguments.skip_disk_check:
        print("", file=sys.stderr)
        try:
            guards.enforce(disk)
        except guards.DiskSpaceError as error:
            print(str(error), file=sys.stderr)
        return 1

    os.makedirs(config.guards.scratch_dir, exist_ok=True)
    provenance = collect(config.repos, config.guards.scratch_dir)
    context = _common.Context(
        config=config,
        token=read_token(),
        provenance=provenance,
        scratch_root=config.guards.scratch_dir,
        keep_artifacts=arguments.keep_artifacts,
    )

    verdicts = check_paths(plan, context)
    for path, reason in sorted(verdicts.items()):
        print(f"path {path:<10}: {'available' if reason is None else 'UNAVAILABLE -- ' + reason}")
    runnable = [
        item for item in plan
        if not item.get("skipped")
        and all(verdicts.get(part) is None for part in str(item["path"]).split("+"))
    ]
    if not runnable:
        print("\nNo plan item can run on this machine. Nothing was written.", file=sys.stderr)
        context.close()
        return 1
    if len(runnable) != len([i for i in plan if not i.get("skipped")]):
        print(f"\n{len(plan) - len(runnable)} plan item(s) will not run; see above.")

    started = time.monotonic()
    with Recorder(arguments.root, campaign) as recorder:
        print(f"\nraw file      : {recorder.path}\n")
        try:
            for number, item in enumerate(runnable, start=1):
                label = f"[{number}/{len(runnable)}] {item['tool']} via {item['path']}"
                print(f"{label} ...", flush=True)
                for record in module.execute(item, context):
                    recorder.write(record)
                    marker = "ok " if record.status == "ok" else "FAIL"
                    warm = " (warm-up, discarded)" if record.warmup else ""
                    print(
                        f"    {marker} rep {record.repetition:>2} "
                        f"{record.total_seconds:8.3f}s{warm}"
                        + (f"  {record.error_type}: {(record.error_message or '')[:120]}"
                           if record.status != "ok" else "")
                    )
        except KeyboardInterrupt:
            print("\ninterrupted; every completed run is already in the raw file.",
                  file=sys.stderr)
        finally:
            context.close()

        elapsed = time.monotonic() - started
        print(f"\n{recorder.count} record(s), {recorder.failures} failure(s), "
              f"{_duration(elapsed)} elapsed")
        print(f"raw: {recorder.path}")

    removed = guards.clear_scratch(config.guards.scratch_dir)
    if removed:
        print(f"cleaned {removed} scratch entr{'y' if removed == 1 else 'ies'}")

    if not arguments.no_summary:
        outcome = summarize.summarize(campaign, arguments.root)
        print(f"summary: {outcome['markdown']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
