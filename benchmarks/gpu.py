"""Peak VRAM, sampled from outside the process that allocates it.

torch's own `max_memory_allocated` is what runner.py records, and it is the
better number when it exists -- but it only counts what THAT process allocated
through torch's caching allocator, and B4 asks what the CARD held with eight
clients on it. That is a device-level question, so it is answered with
nvidia-smi, sampled on a timer for the duration of the campaign.

Sampling is lossy by construction: a peak between two samples is invisible.
The interval is a config knob, the number of samples is recorded next to the
peak, and the summaries print both -- a peak from four samples is a different
claim from a peak from four hundred.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from typing import Optional

_QUERY = ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]
_SAMPLE_TIMEOUT_SECONDS = 5.0


def available() -> bool:
    return shutil.which("nvidia-smi") is not None


def sample() -> Optional[dict]:
    """{gpu index: MiB in use}, or None when nvidia-smi cannot answer."""
    if not available():
        return None
    try:
        completed = subprocess.run(
            _QUERY, capture_output=True, text=True, timeout=_SAMPLE_TIMEOUT_SECONDS
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    used = {}
    for line in completed.stdout.strip().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            used[int(fields[0])] = int(fields[1])
        except ValueError:
            continue
    return used or None


class VramSampler:
    """A background thread holding the highest reading it has seen.

    Started and stopped around a campaign, not around a single run: with eight
    concurrent clients there is no single run to bracket, and attributing a
    device-wide peak to one of them would be a fiction.
    """

    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.peak_mib: dict = {}
        self.samples = 0
        self.baseline_mib: dict = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.unavailable_reason: Optional[str] = None

    def start(self) -> "VramSampler":
        if not available():
            self.unavailable_reason = "nvidia-smi is not on PATH"
            return self
        first = sample()
        if first is None:
            self.unavailable_reason = "nvidia-smi is present but returned nothing usable"
            return self
        self.baseline_mib = dict(first)
        self.peak_mib = dict(first)
        self.samples = 1
        self._thread = threading.Thread(target=self._loop, name="vram-sampler", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            reading = sample()
            if reading is None:
                continue
            self.samples += 1
            for index, mib in reading.items():
                self.peak_mib[index] = max(self.peak_mib.get(index, 0), mib)

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds * 4 + _SAMPLE_TIMEOUT_SECONDS)
        return self.report()

    def report(self) -> dict:
        return {
            "peak_mib": dict(self.peak_mib),
            "baseline_mib": dict(self.baseline_mib),
            "samples": self.samples,
            "interval_seconds": self.interval_seconds,
            "unavailable_reason": self.unavailable_reason,
        }

    def __enter__(self) -> "VramSampler":
        return self.start()

    def __exit__(self, *_exception) -> None:
        self.stop()
