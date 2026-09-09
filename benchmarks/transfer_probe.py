"""B2's transfer-only probe: what the chunked upload costs, with and without
parallel parts, at each payload size.

**Why this exists next to campaign B2.** B2 measures whole calls, so its
`upload` phase is five kept repetitions sitting beside a 60-second tool run.
That is the right shape for "where does a remote call's time go", and the wrong
shape for "does four-way parallelism buy anything", because the effect being
resolved is a few tenths of a second and five points do not resolve it. This
probe moves the same protocol -- the same `POST /uploads`, the same part `PUT`s,
the same SHA-256 per part, the same gzip rule -- into a loop that runs no tool
at all, so a repetition costs a transfer instead of a segmentation and twenty of
them are affordable.

**It is not a substitute for B2 and does not replace any row in it.** It answers
one question B2 raises and cannot settle on its own. Both numbers are reported.

What it does NOT measure: the download side (a result has to be produced by a
run before it can be fetched, so there is nothing to range-GET without one), and
anything the server does after the last part lands.

    python -m benchmarks.transfer_probe --reps 20
    python -m benchmarks.transfer_probe --tools AMASSS --parallelism 4,1 --reps 30

Raw records go to `results/raw/b2probe-<UTC timestamp>.jsonl`, in the same
append-only shape as every campaign's.
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
import tempfile
import time

from . import provenance
from .execution import remote as remote_path
from .recording import STATUS_FAILED, STATUS_OK, PhaseTimer, Recorder, RunRecord, failed
from .settings import BENCHMARKS_ROOT, ConfigError, load, read_token

NAME = "b2probe"
DEFAULT_TOOLS = ("Crown_Seg", "AMASSS", "AREG_IOSCBCT")


def _packed_inputs(tool, workspace: str) -> list:
    """Every travelling input of `tool`, packed exactly as the client packs it.

    Packing is done ONCE and reused across repetitions: zipping a folder is a
    real cost of a real call and B2 reports it, but it is not transfer, and
    paying it again per repetition would put it inside the number this probe
    exists to resolve.
    """
    packed = []
    for argument, path in tool.files.items():
        packed.append(
            (argument, remote_path._pack_if_folder(path, workspace, tool.name, argument))
        )
    return packed


def _probe_once(client, path: str, discard: list) -> dict:
    """One `POST /uploads` plus every part `PUT`. Returns the fields.

    The `DELETE` that frees the session is deliberately NOT done here: the real
    client never issues one on the happy path -- `POST /run` consumes the
    upload -- so timing one inside this measurement would add a round trip the
    protocol does not have. The id goes on `discard`, and the caller deletes it
    after the clock has stopped.
    """
    session = client.open_upload(path)
    discard.append(session["upload_id"])
    sent = client.upload_parts(session)
    return {
        "bytes_uploaded": sent,
        "parts_uploaded": int(session["part_count"]),
        "chunk_bytes": int(session["chunk_size"]),
        "file_bytes": int(session["size"]),
        "gzipped": client.transfer.gzip_parts and remote_path.worth_compressing(path),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.transfer_probe",
        description="Time the chunked upload alone, at each payload size and parallelism.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", default=BENCHMARKS_ROOT)
    parser.add_argument("--tools", default=",".join(DEFAULT_TOOLS))
    parser.add_argument("--parallelism", default="4,1")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1,
                        help="repetitions marked warmup and excluded from the summary")
    arguments = parser.parse_args(argv)

    try:
        config = load(arguments.config)
    except ConfigError as error:
        print(f"config: {error}", file=sys.stderr)
        return 2

    token = read_token()
    if not token:
        print("No API_TOKEN in the environment or in the repository's .env.", file=sys.stderr)
        return 2

    tools = [name.strip() for name in arguments.tools.split(",") if name.strip()]
    levels = [int(value) for value in arguments.parallelism.split(",") if value.strip()]

    fingerprint = provenance.collect(config.repos, config.guards.scratch_dir)
    workspace = tempfile.mkdtemp(prefix="b2probe-")
    rows = []
    try:
        with Recorder(arguments.root, NAME) as recorder:
            print(f"raw file      : {recorder.path}")
            for name in tools:
                tool = config.tool(name)
                if not tool.files:
                    print(f"{name}: no travelling input, nothing to upload -- skipped")
                    continue
                packed = _packed_inputs(tool, workspace)
                for parallelism in levels:
                    client = remote_path.RemoteClient(
                        config.server, config.transfer, token, parallelism=parallelism
                    )
                    try:
                        seconds = []
                        for repetition in range(1, arguments.reps + 1):
                            timer = PhaseTimer()
                            started = provenance.utc_now()
                            clock = time.monotonic()
                            status, error = STATUS_OK, None
                            per_argument = []
                            opened: list = []
                            extra = {"bytes_uploaded": 0, "parts_uploaded": 0}
                            try:
                                with timer.phase("upload"):
                                    for argument, path in packed:
                                        one = _probe_once(client, path, opened)
                                        one["argument"] = argument
                                        per_argument.append(one)
                                        extra["bytes_uploaded"] += one["bytes_uploaded"]
                                        extra["parts_uploaded"] += one["parts_uploaded"]
                            except Exception as exception:  # noqa: BLE001 - recorded
                                status = STATUS_FAILED
                                error = failed(STATUS_FAILED, exception)
                                error.pop("status", None)
                            total = time.monotonic() - clock
                            # After the clock, so the session teardown is not in
                            # the number. Nothing is left on the server either way.
                            for upload_id in opened:
                                client.discard_upload(upload_id)
                            extra["arguments"] = per_argument
                            warmup = repetition <= arguments.warmup
                            extra.update({
                                "parallelism": parallelism,
                                "parallel_transfer": parallelism > 1,
                                "payload_label": tool.payload_label or "unlabelled",
                                "base_url": client.base_url,
                            })
                            record = RunRecord(
                                campaign=NAME, tool=name, path="loopback",
                                repetition=repetition,
                                started_at=started, finished_at=provenance.utc_now(),
                                total_seconds=total, status=status,
                                phases=timer.finalize(total), warmup=warmup,
                                extra=extra, provenance=fingerprint,
                                **(error or {"error_type": None, "error_message": None}),
                            )
                            recorder.write(record)
                            if status == STATUS_OK and not warmup:
                                seconds.append(timer.phases["upload"])
                            mark = " (warm-up)" if warmup else ""
                            print(f"  {name:<14} p={parallelism} rep {repetition:>2} "
                                  f"{status:<6} {total:8.3f}s{mark}")
                    finally:
                        client.close()
                    if seconds:
                        megabytes = extra["bytes_uploaded"] / 1e6
                        rows.append({
                            "tool": name,
                            "payload_mb": round(megabytes, 1),
                            "parts": extra["parts_uploaded"],
                            "files": len(packed),
                            "parallelism": parallelism,
                            "n": len(seconds),
                            "median_s": round(statistics.median(seconds), 4),
                            "min_s": round(min(seconds), 4),
                            "max_s": round(max(seconds), 4),
                            "median_mb_s": round(megabytes / statistics.median(seconds), 1),
                        })
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    if rows:
        print()
        header = list(rows[0])
        print(" | ".join(f"{key}" for key in header))
        for row in rows:
            print(" | ".join(str(row[key]) for key in header))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
