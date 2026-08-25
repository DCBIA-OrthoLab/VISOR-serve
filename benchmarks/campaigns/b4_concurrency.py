"""B4 -- N simultaneous clients on one server.

N in {1, 2, 4, 8}, each client an independent HTTP session issuing the same call
in a loop for a fixed number of jobs. Reported: throughput (completed jobs per
minute), p50 and p95 latency, and the peak VRAM the CARD held for the duration.

Three method notes a reviewer needs:

**The clients start together.** A `threading.Barrier` releases them at the same
instant, so the "8 concurrent clients" window really has eight in it rather than
a ramp with a tail.

**Latency is per JOB, not per client.** Each completed job is one record, with
its client index in `extra`. p50 and p95 are computed over the jobs by the
summariser, from the raw file, so they can be recomputed a different way without
re-running anything.

**Peak VRAM is device-level and sampled.** `nvidia-smi` on a timer, with the
baseline recorded before the clients start and the sample count kept next to the
peak: a peak between two samples is invisible, and a reviewer is entitled to know
how many samples the number rests on. It is attributed to the CAMPAIGN, not to a
job -- with eight jobs on one card, attributing a device peak to one of them
would be a fiction.

The server's own MAX_CONCURRENT_TOOLS (4) and MAX_CONCURRENT_GPU_JOBS (1) are
what N is being tested against; the numbers this campaign produces are a property
of the deployment's settings as much as of the hardware, so the summary records
them.
"""

from __future__ import annotations

import queue
import threading
from typing import Iterator

from ..gpu import VramSampler
from ..recording import RunRecord
from ..settings import PATH_LAN, PATH_LOOPBACK, Config, ConfigError
from . import _common

NAME = "b4"
DESCRIPTION = "Throughput, p50/p95 latency and peak VRAM at N concurrent clients"

DEFAULT_LEVELS = (1, 2, 4, 8)
DEFAULT_JOBS_PER_CLIENT = 2


def build_plan(config: Config, options: dict) -> list:
    section = config.campaign(NAME)
    levels = [int(value) for value in section.get("concurrency", list(DEFAULT_LEVELS))]
    if not levels or min(levels) < 1:
        raise ConfigError(f"campaigns.{NAME}.concurrency must be positive integers")
    jobs_per_client = int(
        options.get("reps") or section.get("jobs_per_client", DEFAULT_JOBS_PER_CLIENT)
    )
    path = str(section.get("path", PATH_LOOPBACK))
    if path not in (PATH_LOOPBACK, PATH_LAN):
        raise ConfigError(f"campaigns.{NAME}.path must be 'loopback' or 'lan'")
    interval = float(section.get("vram_sample_seconds", 0.5))

    wanted = options.get("tools") or section.get("tools") or []
    if not wanted:
        raise ConfigError(f"campaigns.{NAME}.tools is empty; there is nothing to measure.")

    plan = []
    for name in wanted:
        tool = config.tool(name)
        for level in levels:
            runs = level * jobs_per_client
            plan.append(
                _common.plan_item(
                    NAME, tool, path, runs,
                    concurrency=level,
                    jobs_per_client=jobs_per_client,
                    vram_sample_seconds=interval,
                    # Wall clock, not CPU time: `level` clients share one GPU
                    # slot, so the window is roughly serialised.
                    seconds_each=tool.estimated_seconds,
                )
            )
    return plan


def execute(item: dict, context: _common.Context) -> Iterator[RunRecord]:
    tool = context.config.tool(item["tool"])
    level = int(item["concurrency"])
    per_client = int(item["jobs_per_client"])
    path = item["path"]

    sampler = VramSampler(float(item.get("vram_sample_seconds", 0.5))).start()
    results: queue.Queue = queue.Queue()
    barrier = threading.Barrier(level)

    def client(index: int) -> None:
        # Each client gets its OWN session: sharing one would put every request
        # through a single connection pool and measure the pool, not the server.
        parallelism = context.config.transfer.parallelism
        barrier.wait()
        for job in range(1, per_client + 1):
            record = _common.execute(
                context,
                NAME,
                tool,
                path,
                (index - 1) * per_client + job,
                extra={
                    "concurrency": level,
                    "client_index": index,
                    "job_in_client": job,
                },
                parallelism=parallelism,
            )
            results.put(record)

    threads = [
        threading.Thread(target=client, args=(index,), name=f"b4-client-{index}", daemon=True)
        for index in range(1, level + 1)
    ]
    started = _monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    window = _monotonic() - started
    vram = sampler.stop()

    records = []
    while not results.empty():
        records.append(results.get())

    completed = sum(1 for record in records if record.status == "ok")
    for record in records:
        # Every record of this level carries the window and the device peak, so
        # a raw file is readable without a side table.
        record.extra["campaign_window_seconds"] = window
        record.extra["completed_in_window"] = completed
        record.extra["throughput_jobs_per_minute"] = (
            completed / window * 60.0 if window > 0 else None
        )
        record.extra["vram"] = vram
        yield record


def _monotonic() -> float:
    import time

    return time.monotonic()
