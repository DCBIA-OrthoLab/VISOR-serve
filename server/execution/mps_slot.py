"""One run at a time on the Apple GPU.

A Mac has one pool of memory shared by the CPU and the GPU, and MPS has no
per-process arbiter: two runs on `mps` simply allocate from the same pool until
one of them fails, or until the system starts compressing and swapping. The
admission budget cannot stand in for that. It reserves what a tool was MEASURED
to need, a tool nothing has measured runs alone but a measured one is admitted
beside its neighbours, and `dispatch.uses_the_gpu` rightly counts MPS as "not
the card" (there is no card budget on a Mac to charge it to).

So MPS gets the thing a CUDA card has and a Mac lacks: a single slot. A run
whose device resolves to `mps` waits here, in arrival order, BEFORE it queues
for room in the budget; every other run (CPU, CUDA) takes the no-op branch and
never touches this module's state.

Per process, like `admission.Budget` and for the same reason: one API process
owns one machine. `uvicorn --workers 2` would make two slots, exactly as it
makes two budgets.
"""

from __future__ import annotations

import contextlib
import itertools
import threading
from typing import Callable, Optional

from base import DEVICE_ARGUMENT
from config import settings

MPS_PREFIX = "mps"


class Cancelled(RuntimeError):
    """The wait for the slot was abandoned because the client withdrew the run."""


def runs_on_mps(tool, params: dict) -> bool:
    """Does this run's device resolve to the Apple GPU?

    Resolved the way `dispatch.uses_the_gpu` resolves it: the caller's value,
    else the declared default of a choice argument, else `settings.DEVICE`. A
    tool that declares no `device` is judged by `settings.DEVICE` alone, so on a
    server started with DEVICE=mps it is assumed to use the GPU. The strict
    reading again: a tool that imports torch without saying so would otherwise
    meet its neighbour in the unified memory with nothing between them.
    """
    spec = tool.arguments.get(DEVICE_ARGUMENT)
    value = params.get(DEVICE_ARGUMENT)
    if value is None:
        value = spec.default if (spec is not None and spec.is_choice) else settings.DEVICE
    return str(value).lower().startswith(MPS_PREFIX)


class Slot:
    """A first-come, first-served mutex that can be abandoned while waiting."""

    def __init__(self, poll_seconds: Optional[float] = None):
        self._condition = threading.Condition()
        self._tickets = itertools.count()
        self._queue: list = []
        self._held = False
        self._poll_seconds = poll_seconds

    def waiting(self) -> int:
        with self._condition:
            return len(self._queue)

    def held(self) -> bool:
        with self._condition:
            return self._held

    @contextlib.contextmanager
    def hold(self, is_cancelled: Optional[Callable[[], bool]] = None,
             on_wait: Optional[Callable[[], None]] = None):
        """Hold the slot for the block, waiting in arrival order.

        `is_cancelled` is polled while waiting, so a client that gives up on a
        queued run is not kept behind a run that may last an hour. `on_wait` is
        called once, and only if the run actually has to queue.
        """
        poll = self._poll_seconds
        if poll is None:
            poll = float(getattr(settings, "RUN_CANCEL_POLL_SECONDS", 1.0) or 1.0)
        ticket = next(self._tickets)
        announced = False
        with self._condition:
            self._queue.append(ticket)
            while True:
                if is_cancelled is not None and is_cancelled():
                    self._queue.remove(ticket)
                    self._condition.notify_all()
                    raise Cancelled("The client cancelled this run.")
                if self._queue[0] == ticket and not self._held:
                    break
                if not announced and on_wait is not None:
                    announced = True
                    # Released around the callback: it writes an event to disk.
                    self._condition.release()
                    try:
                        on_wait()
                    finally:
                        self._condition.acquire()
                    continue
                self._condition.wait(timeout=poll)
            self._queue.remove(ticket)
            self._held = True
        try:
            yield
        finally:
            with self._condition:
                self._held = False
                self._condition.notify_all()


_slot = Slot()


def slot() -> Slot:
    """The process-wide slot."""
    return _slot
