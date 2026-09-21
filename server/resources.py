"""What this machine gives the server, and what one job may take of it.

Detection first, configuration second: a deployment that sets nothing gets a
server sized to the machine it was started on, so `docker compose up` stays one
command with no tuning. Every number can be overridden, and the startup banner
says which ones were.

**Read the cgroup before asking the kernel.** In a container `os.cpu_count()`
and `/proc/meminfo` report the HOST -- measured here, 56 threads and 125 GB
inside a container that could have been limited to 8 and 16 -- so a server
sized on them ignores every limit its operator set. The cascade below reads the
limit first and falls back to the machine only when there is none. That is also
why Kubernetes needs no special case: a pod's `resources.limits` sets the same
cgroup files `docker compose` does, so an operator who configures the container
the way they already know how is configuring the server too.

The card is the one resource with no cgroup. `nvidia-smi` answers instead, and
it is present inside the deployment container even though the image never
installs it -- the NVIDIA container toolkit injects it with the driver. Absent,
the VRAM dimension simply does not exist and nothing here fails.
"""

import logging
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

# cgroup v2 first: it is what a current Docker and every Kubernetes node use.
# v1 is still what an older kernel or a --cgroupns=host setup exposes.
_CGROUP_V2_CPU = "/sys/fs/cgroup/cpu.max"
_CGROUP_V2_MEMORY = "/sys/fs/cgroup/memory.max"
_CGROUP_V1_CPU_QUOTA = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
_CGROUP_V1_CPU_PERIOD = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
_CGROUP_V1_MEMORY = "/sys/fs/cgroup/memory/memory.limit_in_bytes"

_MEMINFO = "/proc/meminfo"

# cgroup v1 spells "no limit" as a number rather than a word, and which number
# depends on the page size. Anything at or above this is not a real limit.
_V1_UNLIMITED = 1 << 62

_NVIDIA_SMI_TIMEOUT_SECONDS = 5.0
_VRAM_QUERY = ["--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"]

_UNITS = {
    "B": 1,
    "K": 1000, "KB": 1000, "KIB": 1024,
    "M": 1000 ** 2, "MB": 1000 ** 2, "MIB": 1024 ** 2,
    "G": 1000 ** 3, "GB": 1000 ** 3, "GIB": 1024 ** 3,
    "T": 1000 ** 4, "TB": 1000 ** 4, "TIB": 1024 ** 4,
}

# Detection is the answer when nothing was configured; these name the other two
# cases so the banner can tell an operator which of their settings took effect.
SOURCE_DETECTED = "detected"
SOURCE_SHARE = "share of detected"
SOURCE_CONFIGURED = "configured"
SOURCE_DERIVED = "derived"
SOURCE_UNAVAILABLE = "unavailable"


def _first_line(path: str) -> Optional[str]:
    """The first line of a file, or None if it cannot be read.

    Every caller is probing for a file that legitimately does not exist on some
    kernels, so a missing path is an answer rather than an error.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.readline().strip()
    except OSError:
        return None


def parse_amount(value, whole: Optional[float]) -> Optional[float]:
    """One setting, written three ways: `"75%"`, `"8"`, or `"16GB"`.

    A percentage needs `whole` to resolve against and returns None without one,
    which is how a VRAM percentage behaves on a machine with no card. An empty
    or unparseable value returns None and means "decide it for me" -- never an
    exception, because a typo in a resource knob must not stop a server that
    can size itself perfectly well.
    """
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if text.endswith("%"):
        if whole is None:
            return None
        try:
            fraction = float(text[:-1]) / 100.0
        except ValueError:
            logger.warning("Ignoring unparseable resource percentage %r.", value)
            return None
        return max(0.0, whole * fraction)
    for suffix in sorted(_UNITS, key=len, reverse=True):
        if text.endswith(suffix) and text[: -len(suffix)].strip():
            try:
                return float(text[: -len(suffix)].strip()) * _UNITS[suffix]
            except ValueError:
                break
    try:
        return float(text)
    except ValueError:
        logger.warning("Ignoring unparseable resource amount %r.", value)
        return None


def detect_cpus() -> float:
    """Cores this process may use, fractional, cgroup limit first.

    The affinity mask sits between the cgroup and `os.cpu_count()` because a
    cpuset (`--cpuset-cpus`, or a Slurm allocation) restricts which cores are
    usable without writing a quota, and it is the only one of the three that
    reflects it.
    """
    line = _first_line(_CGROUP_V2_CPU)
    if line:
        parts = line.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota, period = float(parts[0]), float(parts[1])
                if quota > 0 and period > 0:
                    return quota / period
            except ValueError:
                pass

    quota_line = _first_line(_CGROUP_V1_CPU_QUOTA)
    period_line = _first_line(_CGROUP_V1_CPU_PERIOD)
    if quota_line and period_line:
        try:
            quota, period = float(quota_line), float(period_line)
            if quota > 0 and period > 0:
                return quota / period
        except ValueError:
            pass

    try:
        return float(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return float(os.cpu_count() or 1)


def detect_ram_bytes() -> Optional[int]:
    """Bytes of host memory this process may use, cgroup limit first."""
    line = _first_line(_CGROUP_V2_MEMORY)
    if line and line != "max":
        try:
            return int(line)
        except ValueError:
            pass

    line = _first_line(_CGROUP_V1_MEMORY)
    if line:
        try:
            limit = int(line)
            if 0 < limit < _V1_UNLIMITED:
                return limit
        except ValueError:
            pass

    try:
        with open(_MEMINFO, encoding="utf-8") as handle:
            for entry in handle:
                if entry.startswith("MemTotal:"):
                    return int(entry.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        pass
    return None


def detect_vram_bytes() -> tuple:
    """`(total, free)` bytes on the card with the LEAST free memory, or two Nones.

    The least free rather than the sum: the budget it feeds admits a job that
    has to fit on ONE card, and a machine holding two half-full cards has
    nowhere to put a job the size of their total.

    `free` is reported separately from `total` because the two answer different
    questions and only one of them is stable. Measured on this machine, ~3 GiB
    of a 48 GiB card is already held by something that is not this server, so a
    budget computed from `total` would admit a job the card cannot take.
    """
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None, None
    try:
        completed = subprocess.run(
            [binary] + _VRAM_QUERY,
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    if completed.returncode != 0:
        return None, None

    totals, frees = [], []
    for line in completed.stdout.splitlines():
        parts = [piece.strip() for piece in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            totals.append(int(parts[0]) * 1024 ** 2)
            frees.append(int(parts[1]) * 1024 ** 2)
        except ValueError:
            continue
    if not totals:
        return None, None
    smallest = min(range(len(frees)), key=lambda index: frees[index])
    return totals[smallest], frees[smallest]


@dataclass(frozen=True)
class Allocation:
    """What the server may use, and what it will give one job.

    `*_per_job` is the fairness half and the only one that depends on how many
    people are expected at once: it is what stops one clinician's 500-patient
    cohort from taking the machine. With one expected client there is no cap,
    because dividing a machine by one is what a single workstation wants --
    every run as fast as the hardware allows.
    """

    cpus: float
    ram_bytes: Optional[int]
    vram_bytes: Optional[int]
    cpus_per_job: int
    ram_per_job: Optional[int]
    vram_per_job: Optional[int]
    expected_clients: int
    detected: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)

    @property
    def max_parallel_jobs(self) -> int:
        """How many jobs the CPU budget alone allows, at least one.

        A floor rather than the admission decision: VRAM and RAM narrow it
        further, per job, once the costs are known.
        """
        return max(1, int(self.cpus // max(1, self.cpus_per_job)))


def _budget_for(configured, detected: Optional[float], share: float):
    """One resource's total: what was configured, else a share of what is there."""
    chosen = parse_amount(configured, detected)
    if chosen is not None and chosen > 0:
        return chosen, SOURCE_CONFIGURED
    if detected is None:
        return None, SOURCE_UNAVAILABLE
    return detected * share, SOURCE_SHARE


def _per_job(configured, budget, divisor: int, whole: Optional[float]):
    """One resource's per-job cap: configured, else the budget split `divisor` ways.

    **A cap is a fairness target, not a refusal.** Admission has to let a job
    whose measured cost exceeds its cap run anyway, alone -- measured here,
    AMASSS peaks at 2.19 GiB, and a hundred-way split of this card would hand
    it 0.3 GiB and make it permanently unrunnable. Splitting a machine is how
    you stop one cohort taking it, never how you decide what may run on it.
    """
    chosen = parse_amount(configured, whole)
    if chosen is not None and chosen > 0:
        return chosen, SOURCE_CONFIGURED
    if budget is None:
        return None, SOURCE_UNAVAILABLE
    if divisor <= 1:
        return budget, SOURCE_DERIVED
    return budget / divisor, SOURCE_DERIVED


def resolve(config) -> Allocation:
    """Everything the admission policy needs, decided once at startup.

    Takes the settings object rather than reading the environment, so this
    module stays testable with a plain namespace and the rule that nothing but
    `config.py` reads `os.getenv` keeps holding.
    """
    cpus_found = detect_cpus()
    ram_found = detect_ram_bytes()
    vram_total, vram_free = detect_vram_bytes()
    share = max(0.01, min(1.0, float(getattr(config, "SADT_RESOURCE_SHARE", 0.75))))
    clients = max(1, int(getattr(config, "SADT_EXPECTED_CLIENTS", 1)))

    cpus, cpu_source = _budget_for(getattr(config, "SADT_CPU_BUDGET", ""), cpus_found, share)
    ram, ram_source = _budget_for(getattr(config, "SADT_RAM_BUDGET", ""), ram_found, share)
    # Against what is FREE, not what the card holds: the share exists to leave
    # room, and it cannot leave room in memory another process already took.
    vram, vram_source = _budget_for(getattr(config, "SADT_VRAM_BUDGET", ""), vram_free, share)

    cpus = max(1.0, cpus if cpus else 1.0)
    # Never split further than the machine can actually run in parallel. A
    # budget of 42 cores cannot usefully be cut a hundred ways -- past that the
    # divisor stops describing contention and starts inventing it, and every
    # resource gets shares too small for any real tool.
    divisor = min(clients, max(1, int(cpus)))
    cpu_cap, cpu_cap_source = _per_job(getattr(config, "SADT_CPU_PER_JOB", ""), cpus, divisor, cpus)
    ram_cap, ram_cap_source = _per_job(getattr(config, "SADT_RAM_PER_JOB", ""), ram, divisor, ram)
    vram_cap, vram_cap_source = _per_job(
        getattr(config, "SADT_VRAM_PER_JOB", ""), vram, divisor, vram
    )

    return Allocation(
        cpus=cpus,
        ram_bytes=int(ram) if ram else None,
        vram_bytes=int(vram) if vram else None,
        # At least one whole core: a fractional thread count means nothing to
        # OMP_NUM_THREADS, and a job given zero cores would never run.
        cpus_per_job=max(1, int(cpu_cap or cpus)),
        ram_per_job=int(ram_cap) if ram_cap else None,
        vram_per_job=int(vram_cap) if vram_cap else None,
        expected_clients=clients,
        detected={
            "cpus": cpus_found,
            "ram_bytes": ram_found,
            "vram_total_bytes": vram_total,
            "vram_free_bytes": vram_free,
        },
        sources={
            "cpus": cpu_source,
            "ram_bytes": ram_source,
            "vram_bytes": vram_source,
            "cpus_per_job": cpu_cap_source,
            "ram_per_job": ram_cap_source,
            "vram_per_job": vram_cap_source,
        },
    )


_allocation: Optional[Allocation] = None
_allocation_lock = threading.Lock()


def allocation() -> Allocation:
    """This process's budget, resolved once and then reused.

    Lazy rather than computed at import: detection shells out to `nvidia-smi`,
    and a test importing this module has no reason to pay for that or to depend
    on what the machine running it happens to hold.
    """
    global _allocation
    with _allocation_lock:
        if _allocation is None:
            _allocation = resolve(settings)
        return _allocation


def _gib(value: Optional[float]) -> str:
    return "unavailable" if not value else f"{value / 1024 ** 3:.1f} GiB"


def banner(allocation: Allocation) -> str:
    """What was detected, what was decided, and which half came from a setting.

    Printed at startup because the alternative is an operator who set
    `SADT_CPU_BUDGET` in the wrong place and has no way to find out: the only
    visible symptom of a budget that did not take effect is a server that feels
    slow.
    """
    found = allocation.detected
    lines = [
        "Resource budget",
        "  detected      cpus={:.1f}  ram={}  vram={} free of {}".format(
            found.get("cpus") or 0.0,
            _gib(found.get("ram_bytes")),
            _gib(found.get("vram_free_bytes")),
            _gib(found.get("vram_total_bytes")),
        ),
        "  budget        cpus={:.1f} ({})  ram={} ({})  vram={} ({})".format(
            allocation.cpus,
            allocation.sources.get("cpus"),
            _gib(allocation.ram_bytes),
            allocation.sources.get("ram_bytes"),
            _gib(allocation.vram_bytes),
            allocation.sources.get("vram_bytes"),
        ),
        "  per job       cpus={} ({})  ram={}  vram={}".format(
            allocation.cpus_per_job,
            allocation.sources.get("cpus_per_job"),
            _gib(allocation.ram_per_job),
            _gib(allocation.vram_per_job),
        ),
        "  expecting {} client(s); the cpu budget alone allows {} job(s) at once".format(
            allocation.expected_clients, allocation.max_parallel_jobs
        ),
    ]
    # MAX_CONCURRENT_TOOLS caps the worker threads that may be inside a tool run
    # at once, and it sits ABOVE admission. Left below what the budget allows it
    # becomes the real limit, silently, and every budget past it looks identical
    # -- which is a very expensive afternoon to spend not knowing.
    lines.append(
        "  admission    what each run was measured to need, at the width it was "
        "granted; a tool with no measured cost runs alone"
    )
    threads = int(getattr(settings, "MAX_CONCURRENT_TOOLS", 0) or 0)
    if threads and threads < allocation.max_parallel_jobs:
        lines.append(
            "  NOTE  MAX_CONCURRENT_TOOLS={} is below that, so it is the real "
            "limit, not the budget".format(threads)
        )
    return "\n".join(lines)
