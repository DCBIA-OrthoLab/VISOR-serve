"""One run at a time on the Apple GPU, and nothing new for any other device.

The slot is what a Mac lacks and a CUDA card has: an arbiter. It is tested the
way admission is -- with threads that wait on Events with a timeout, never on a
sleep -- and, as important, the other way round: a run on `cpu` or `cuda` must
not touch it at all.
"""

import os
import threading
import types

import pytest

os.environ.setdefault("API_TOKEN", "test-token")

from config import settings
from execution import dispatch, mps_slot

_TIMEOUT = 5.0


def _tool(default="cuda", declares_device=True):
    spec = types.SimpleNamespace(default=default, is_choice=True)
    return types.SimpleNamespace(
        name="Probe", arguments={"device": spec} if declares_device else {}
    )


# ----------------------------------------------------------------------
# Which runs take the slot
# ----------------------------------------------------------------------

@pytest.mark.parametrize("device,expected", [
    ("mps", True), ("MPS", True), ("mps:0", True),
    ("cpu", False), ("cuda", False), ("cuda:1", False),
])
def test_only_a_device_that_resolves_to_mps_takes_the_slot(device, expected):
    assert mps_slot.runs_on_mps(_tool(), {"device": device}) is expected


def test_the_declared_default_decides_when_the_caller_does_not():
    assert mps_slot.runs_on_mps(_tool(default="mps"), {}) is True
    assert mps_slot.runs_on_mps(_tool(default="cuda"), {}) is False


def test_a_tool_with_no_device_argument_follows_the_deployment(monkeypatch):
    """The strict reading, like uses_the_gpu: a tool that imports torch without
    declaring a device would otherwise meet its neighbour in unified memory."""
    tool = _tool(declares_device=False)
    monkeypatch.setattr(settings, "DEVICE", "mps")
    assert mps_slot.runs_on_mps(tool, {}) is True
    monkeypatch.setattr(settings, "DEVICE", "cuda")
    assert mps_slot.runs_on_mps(tool, {}) is False


def test_the_cuda_and_cpu_paths_do_not_enter_the_slot(monkeypatch):
    """The Linux guarantee: the context manager is a no-op, and the shared slot
    is never even looked at."""
    def _boom():
        raise AssertionError("the slot must not be touched")

    monkeypatch.setattr(mps_slot, "slot", _boom)
    for device in ("cuda", "cpu"):
        with dispatch._mps_exclusive(_tool(), {"device": device}, "run-1"):
            pass


def test_uses_the_gpu_is_unchanged_for_mps():
    """MPS is still `not the card` for the budget: the slot is a separate gate."""
    assert dispatch.uses_the_gpu(_tool(), {"device": "mps"}) is False
    assert dispatch.uses_the_gpu(_tool(), {"device": "cuda"}) is True


# ----------------------------------------------------------------------
# The slot itself
# ----------------------------------------------------------------------

class _Holder:
    def __init__(self, slot, **kwargs):
        self.slot = slot
        self.kwargs = kwargs
        self.admitted = threading.Event()
        self.release = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            with self.slot.hold(**self.kwargs):
                self.admitted.set()
                self.release.wait(_TIMEOUT)
        except mps_slot.Cancelled as exc:
            self.error = exc

    def start(self):
        self.thread.start()
        return self

    def finish(self):
        self.release.set()
        self.thread.join(_TIMEOUT)


def test_a_second_run_waits_until_the_first_lets_go():
    slot = mps_slot.Slot(poll_seconds=0.02)
    first = _Holder(slot).start()
    assert first.admitted.wait(_TIMEOUT)
    second = _Holder(slot).start()
    assert not second.admitted.wait(0.3), "two runs held the Apple GPU at once"
    assert slot.waiting() == 1
    first.finish()
    assert second.admitted.wait(_TIMEOUT)
    second.finish()
    assert not slot.held()


def test_waiting_runs_are_served_in_arrival_order():
    slot = mps_slot.Slot(poll_seconds=0.02)
    order = []
    first = _Holder(slot).start()
    assert first.admitted.wait(_TIMEOUT)

    def worker(label, started):
        started.set()
        with slot.hold():
            order.append(label)

    threads = []
    for label in ("b", "c", "d"):
        started = threading.Event()
        thread = threading.Thread(target=worker, args=(label, started), daemon=True)
        thread.start()
        assert started.wait(_TIMEOUT)
        # Let it reach the queue before the next one is started.
        for _ in range(100):
            if slot.waiting() == len(threads) + 1:
                break
            threading.Event().wait(0.01)
        threads.append(thread)
    first.finish()
    for thread in threads:
        thread.join(_TIMEOUT)
    assert order == ["b", "c", "d"]


def test_a_cancelled_wait_leaves_the_queue_and_frees_the_way():
    slot = mps_slot.Slot(poll_seconds=0.02)
    first = _Holder(slot).start()
    assert first.admitted.wait(_TIMEOUT)
    cancelled = threading.Event()
    second = _Holder(slot, is_cancelled=cancelled.is_set).start()
    for _ in range(100):
        if slot.waiting() == 1:
            break
        threading.Event().wait(0.01)
    cancelled.set()
    second.thread.join(_TIMEOUT)
    assert isinstance(second.error, mps_slot.Cancelled)
    assert slot.waiting() == 0
    first.finish()
    third = _Holder(slot).start()
    assert third.admitted.wait(_TIMEOUT)
    third.finish()


def test_on_wait_is_told_once_and_only_to_a_run_that_queues():
    slot = mps_slot.Slot(poll_seconds=0.02)
    calls = []
    with slot.hold(on_wait=lambda: calls.append(1)):
        pass
    assert calls == []

    first = _Holder(slot).start()
    assert first.admitted.wait(_TIMEOUT)
    second = _Holder(slot, on_wait=lambda: calls.append(1)).start()
    for _ in range(100):
        if calls:
            break
        threading.Event().wait(0.01)
    first.finish()
    assert second.admitted.wait(_TIMEOUT)
    second.finish()
    assert calls == [1]


def test_the_slot_is_released_when_the_run_raises():
    slot = mps_slot.Slot(poll_seconds=0.02)
    with pytest.raises(RuntimeError):
        with slot.hold():
            raise RuntimeError("tool failed")
    assert not slot.held()


def test_dispatch_turns_a_cancelled_wait_into_run_cancelled(monkeypatch):
    slot = mps_slot.Slot(poll_seconds=0.02)
    monkeypatch.setattr(mps_slot, "slot", lambda: slot)
    monkeypatch.setattr(dispatch.runs, "is_cancelled", lambda _run_id: True)
    first = _Holder(slot).start()
    assert first.admitted.wait(_TIMEOUT)
    with pytest.raises(dispatch.RunCancelled):
        with dispatch._mps_exclusive(_tool(), {"device": "mps"}, "run-7"):
            pass
    first.finish()
