"""What machine produced a number, and from which source.

Every record in results/raw/ carries this, because a runtime figure without the
machine it was measured on is not a measurement. Collected ONCE per invocation
(the fingerprint cannot change under us mid-campaign) and then copied into every
record, so a raw file stays readable on its own with no side table to join.

Nothing here may raise. A field that cannot be collected -- no GPU, no
/proc/cpuinfo, no git -- is recorded as null with a `_notes` entry saying why,
which is the difference between "there was no GPU" and "we forgot to look".
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from typing import Optional

# nvidia-smi is asked for exactly these fields, in this order.
_GPU_QUERY = "name,driver_version,memory.total"
# A probe must never become the slow part of a benchmark.
_PROBE_TIMEOUT_SECONDS = 15.0


def utc_now() -> str:
    """The one timestamp format used everywhere: ISO 8601, UTC, seconds."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _run(command: list, cwd: Optional[str] = None) -> Optional[str]:
    """Stdout of `command`, or None if it could not be run.

    Every failure mode collapses to None on purpose: a missing binary, a
    non-zero exit and a timeout are all "this fact is not available here", and
    the caller records the absence rather than crashing a campaign over it.
    """
    if shutil.which(command[0]) is None and not os.path.isabs(command[0]):
        return None
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def cpu_model() -> Optional[str]:
    """The marketing name of the CPU, from /proc/cpuinfo.

    platform.processor() answers "x86_64" on Linux, which names an
    architecture and not a machine, so it is not used.
    """
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def cpu_counts() -> dict:
    """Physical cores and logical threads, as far as the OS will say."""
    logical = os.cpu_count()
    physical = None
    output = _run(["lscpu"])
    if output:
        sockets = cores_per_socket = None
        for line in output.splitlines():
            if line.startswith("Core(s) per socket:"):
                cores_per_socket = _first_int(line)
            elif line.startswith("Socket(s):"):
                sockets = _first_int(line)
        if sockets and cores_per_socket:
            physical = sockets * cores_per_socket
    return {"physical_cores": physical, "logical_threads": logical}


def _first_int(line: str) -> Optional[int]:
    match = re.search(r"(\d+)", line)
    return int(match.group(1)) if match else None


def ram_bytes() -> Optional[int]:
    """Total RAM, from /proc/meminfo (kB), in bytes."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    kilobytes = _first_int(line)
                    return kilobytes * 1024 if kilobytes else None
    except OSError:
        pass
    return None


def gpus() -> dict:
    """Every CUDA device nvidia-smi reports, plus the driver and runtime CUDA.

    Returns {"devices": [...], "cuda_version": ...}. `devices` is an EMPTY LIST
    on a machine with no card and None when nvidia-smi exists but refused to
    answer -- a distinction a reviewer reading a CPU-only run needs.
    """
    output = _run(["nvidia-smi", f"--query-gpu={_GPU_QUERY}", "--format=csv,noheader"])
    if output is None:
        return {
            "devices": [] if shutil.which("nvidia-smi") is None else None,
            "cuda_version": None,
        }
    devices = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            continue
        devices.append({"name": fields[0], "driver_version": fields[1], "memory_total": fields[2]})
    return {"devices": devices, "cuda_version": _cuda_version()}


def _cuda_version() -> Optional[str]:
    """The CUDA version the DRIVER reports, which is the ceiling a tool's torch
    build has to fit under. Not the version of any toolkit that may be
    installed -- no tool here compiles anything."""
    output = _run(["nvidia-smi"])
    if not output:
        return None
    match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
    return match.group(1) if match else None


def network_interfaces() -> dict:
    """{interface: link speed in Mb/s} for every physical interface.

    Virtual interfaces are excluded by their absence from /sys/class/net/*/device:
    a docker bridge reporting 10000 Mb/s is not a wire and would make the LAN
    numbers look impossible.
    """
    speeds = {}
    root = "/sys/class/net"
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return speeds
    for name in names:
        if not os.path.exists(os.path.join(root, name, "device")):
            continue
        try:
            with open(os.path.join(root, name, "speed"), encoding="utf-8") as handle:
                value = int(handle.read().strip())
        except (OSError, ValueError):
            continue
        # -1 is what the kernel reports for a link that is down.
        speeds[name] = value if value >= 0 else None
    return speeds


def git_sha(repo: str) -> Optional[str]:
    """The commit a repository is on, with `-dirty` when it has changes.

    The suffix matters more than the SHA here: these benchmarks are run against
    working trees, and a number measured on uncommitted code must say so rather
    than claim a commit that does not contain it.
    """
    if not os.path.isdir(repo):
        return None
    sha = _run(["git", "-C", repo, "rev-parse", "HEAD"])
    if sha is None:
        return None
    status = _run(["git", "-C", repo, "status", "--porcelain"])
    return f"{sha}-dirty" if status else sha


def disk_free_bytes(path: str) -> Optional[int]:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


def collect(repos: dict, disk_path: str = ".") -> dict:
    """The whole fingerprint, in one dict, ready to be embedded in a record.

    `repos` is {label: path}; each contributes one git SHA. Anything that could
    not be collected is null and named in `_notes`.
    """
    notes = []
    processor = cpu_model()
    if processor is None:
        notes.append("cpu_model unavailable (no readable /proc/cpuinfo)")
    memory = ram_bytes()
    if memory is None:
        notes.append("ram_bytes unavailable (no readable /proc/meminfo)")
    gpu = gpus()
    if gpu["devices"] is None:
        notes.append("gpu unavailable (nvidia-smi is installed but did not answer)")
    elif not gpu["devices"]:
        notes.append("no CUDA device on this machine")

    revisions = {}
    for label, path in sorted(repos.items()):
        revisions[label] = git_sha(path)
        if revisions[label] is None:
            notes.append(f"git sha unavailable for repo '{label}' at {path}")

    return {
        "hostname": socket.gethostname(),
        "collected_at": utc_now(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "python": platform.python_version(),
        },
        "cpu": dict({"model": processor}, **cpu_counts()),
        "ram_bytes": memory,
        "gpu": gpu,
        "network_mbps": network_interfaces(),
        "disk_free_bytes": disk_free_bytes(disk_path),
        "git": revisions,
        "_notes": notes,
    }
