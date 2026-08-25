"""Refuse to start a campaign that will not fit, and clean up after one that did.

This machine has ~59 GB free and the campaigns write CBCT volumes. A campaign
that fills the disk does not merely fail: it takes the server down with it,
because the job directories the server writes into live on the same filesystem.
So the projected output is computed from the plan BEFORE anything runs, and the
harness refuses rather than starts and hopes.

The projection is deliberately pessimistic -- every run's declared
`estimated_output_mb`, with no credit for the fact that most of it is deleted
between runs. Being wrong the safe way costs a config edit; being wrong the
other way costs the server.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

_GB = 1024 ** 3


class DiskSpaceError(RuntimeError):
    """Not enough room for the projected output. Names all three numbers, so
    the operator can decide between freeing space and lowering the margin."""


@dataclass
class DiskReport:
    path: str
    free_bytes: int
    projected_bytes: int
    margin_bytes: int
    minimum_bytes: int

    @property
    def ok(self) -> bool:
        return (
            self.free_bytes >= self.minimum_bytes
            and self.free_bytes - self.projected_bytes >= self.margin_bytes
        )

    def describe(self) -> str:
        return (
            f"disk {self.path}: free {self.free_bytes / _GB:.1f} GB, "
            f"projected output {self.projected_bytes / _GB:.1f} GB, "
            f"required margin {self.margin_bytes / _GB:.1f} GB, "
            f"absolute floor {self.minimum_bytes / _GB:.1f} GB"
        )


def project_output_bytes(plan: list) -> int:
    """How much the plan will write, from each item's own declaration.

    A plan item says `output_mb` (per run) and `runs`. An item that declares
    neither contributes nothing, which is correct for the text-only tools and
    wrong-but-harmless for a tool whose config forgot to say -- hence the
    validation in settings.py that gives every tool a default.
    """
    total = 0.0
    for item in plan:
        total += float(item.get("output_mb", 0.0)) * int(item.get("runs", 1))
    return int(total * 1024 * 1024)


def check_disk(path: str, plan: list, min_free_gb: float, margin_gb: float) -> DiskReport:
    """The report. Raising is the caller's decision -- see `enforce`."""
    probe = path
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    free = shutil.disk_usage(probe or "/").free
    return DiskReport(
        path=path,
        free_bytes=free,
        projected_bytes=project_output_bytes(plan),
        margin_bytes=int(margin_gb * _GB),
        minimum_bytes=int(min_free_gb * _GB),
    )


def enforce(report: DiskReport) -> None:
    if report.ok:
        return
    raise DiskSpaceError(
        "Refusing to start: " + report.describe() + ". Free space, lower "
        "guards.margin_gb / guards.min_free_gb, or reduce the plan (--tools, --reps)."
    )


def clear_scratch(scratch_dir: str) -> int:
    """Remove every job temp directory this harness made. Returns how many.

    Only ever called on the harness's own scratch root, which is a config key
    and never a path a tool chose: a cleanup routine that walks somewhere else
    is a data-loss incident waiting for its first typo.
    """
    if not os.path.isdir(scratch_dir):
        return 0
    removed = 0
    for name in sorted(os.listdir(scratch_dir)):
        target = os.path.join(scratch_dir, name)
        if os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
            removed += 1
        else:
            try:
                os.remove(target)
                removed += 1
            except OSError:
                pass
    return removed
