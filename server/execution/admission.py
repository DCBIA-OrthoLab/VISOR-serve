"""Who may run right now, decided in bytes and cores rather than in jobs.

`MAX_CONCURRENT_GPU_JOBS` counted jobs, and a count cannot describe a card: on
the machine this was written for, ALI_CBCT peaks at 0.25 GiB and AMASSS at
2.19 GiB on a 48 GiB device, so a counter of one left ~95% of it idle while
everything queued. Worse, the counter was already wrong in two directions --
`threading.BoundedSemaphore` is per PROCESS, so `uvicorn --workers 2` silently
made two independent counters of one; and a supervised call never re-enters the
server at all, so a chain put several tools on the card with nothing between
them.

**The seam is the point.** `Budget` is one implementation of `reserve()`, and
the only one that exists today. A deployment that grows past this machine
replaces it -- a filesystem-backed budget shared by several workers, or a
submission to a real scheduler, where this vector becomes `--mem`,
`--cpus-per-task` and `--gres`. Nothing outside this module knows which one it
is talking to, which is what keeps that move an implementation change.

Three rules make it safe to be optimistic:

- **An unmeasured tool reserves everything**, so it runs alone and the server
  is never worse than the counter it replaces. The table fills itself: one run
  of each tool is the whole bootstrap.
- **A demand larger than the budget still runs**, alone, once the machine is
  idle. A cap is a fairness target, not a refusal -- refusing would make a tool
  permanently unrunnable on a machine that can in fact run it.
- **The card's actual free memory is checked too.** History cannot know about
  the 3 GiB another process on this host is already holding.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import threading
from dataclasses import dataclass
from typing import Optional

import resources
from config import settings

logger = logging.getLogger("inference_server")

# What a measured peak is multiplied by before it is believed. VRAM depends on
# the input, so a peak learned on a small scan under-estimates a large one, and
# under-estimating is the direction that ends in an out-of-memory halfway
# through somebody's cohort.
SAFETY_MARGIN = 1.3


def whole_machine(budget) -> "Demand":
    """The demand of a run nothing has measured: everything, and alone."""
    return Demand(cpus=budget.cpus_per_job, ram_bytes=budget.ram_bytes,
                  vram_bytes=budget.vram_bytes, measured=False)


@dataclass(frozen=True)
class Demand:
    """What one job is expected to hold while it runs."""

    cpus: int = 1
    ram_bytes: int = 0
    vram_bytes: int = 0
    # False when nothing has ever measured this tool. Carried rather than
    # inferred from zeros, because "measured, and it uses no VRAM" and "never
    # measured" must not admit the same way.
    measured: bool = True
    # How many threads the run may OPEN, which is NOT how many cores it HOLDS.
    # Zero means "nothing said", and `Budget._cpu_grant` then falls back to the
    # declared per-job share.
    #
    # The two were one field until they were measured apart. `cpus` is what a
    # neighbour cannot have, and measuring it was a real fix: ALI_CBCT at eight
    # channels filled 51.7 of 56 cores while admission believed it had let in a
    # ten-core job. But occupancy is not evidence that the threads were WORTH
    # opening -- a BLAS pool that spin-waits is busy doing nothing, and it is
    # counted here exactly as useful work is (`runner._Sampler._sample_cores`
    # divides ticks by elapsed, and a spinning thread burns ticks). Feeding the
    # one number back into the other closed a loop with a bad fixed point:
    #
    #     Surg_Mov_Pred measured 26.11 cores busy -> granted 34 threads
    #                   -> occupies 26 again     -> granted 34 again
    #
    # and 34 is a third slower than the 4 to 8 threads it actually wants. See
    # `_cpu_grant` for the curve.
    threads: int = 0

    def fits_in(self, budget: "Budget") -> bool:
        return (
            self.cpus <= budget.cpus
            and self.ram_bytes <= budget.ram_bytes
            and self.vram_bytes <= budget.vram_bytes
        )


def demand_for(tool_name: str, allocation, cost, uses_gpu: bool = True,
               channels: int = 1, cpus: Optional[float] = None) -> Demand:
    """What to reserve for this run: its measured cost, or the whole budget.

    The per-job cap is a floor here, not a ceiling. A job is TOLD it may use
    `cpus_per_job` threads, so that is what it holds whatever it measured; and
    a tool measured at less than its cap still reserves the cap's worth of
    nothing else, because the figure that matters for admission is what the
    machine cannot give to somebody else.
    """
    # `cpus` narrows the reservation below the declared share, which is how a
    # run that would otherwise queue is admitted anyway. None means the share.
    held = allocation.cpus_per_job if cpus is None else max(1, int(cpus))
    # What the run may OPEN: the declared share, or less where admission
    # narrowed this candidate to get it in. Taken BEFORE measured occupancy is
    # folded into `held` below, which is the whole point -- see `Demand.threads`.
    openable = held
    whole_budget = Demand(
        cpus=held,
        ram_bytes=allocation.ram_bytes or 0,
        vram_bytes=allocation.vram_bytes or 0,
        measured=False,
        threads=openable,
    )
    if cost is None or (not cost.vram_bytes and not cost.ram_bytes):
        return whole_budget
    # Measured on RAM but never on VRAM, on a run that wants the card: that is
    # ABSENCE, not a measurement of zero, and the two admit in opposite
    # directions. A tool measured at zero VRAM shares the card with anything; a
    # tool whose card readings were all refused as contended knows nothing
    # about the card at all, and reserving zero for it would stack runs onto a
    # device the server believes is empty. It runs alone instead, exactly like
    # a tool nothing has measured -- and it does not stay there, because a run
    # that runs alone is by definition solo and its next reading is trusted.
    if uses_gpu and not cost.vram_known:
        return whole_budget
    # What it costs AT THE CHANNEL COUNT being asked for. Reserving the
    # one-channel figure and then letting the run open eight is how a budget
    # comes to admit eight times what it accounted for -- and the cost table
    # learns that only from the run that already took it.
    cost = cost.at(channels)
    vram = int(cost.vram_bytes * SAFETY_MARGIN) if uses_gpu else 0
    # The cores this WIDTH was measured to keep busy, when anything has
    # measured them. Never below the declared share, because a job is told it
    # may open that many threads whatever it was seen to use.
    #
    # Its absence was a real hole: memory scaled with the channel count and
    # cores did not, so ALI_CBCT at eight channels filled 51.7 of 56 cores
    # while admission believed it had let in a ten-core job. A tool whose
    # channels cost no CPU -- CLIC batches slices inside one forward -- is
    # unaffected, which is the point of measuring rather than assuming.
    if cost.cpu_cores:
        held = max(held, int(cost.cpu_cores * SAFETY_MARGIN + 0.5))
    return Demand(
        cpus=held,
        ram_bytes=int(cost.ram_bytes * SAFETY_MARGIN),
        vram_bytes=vram,
        measured=True,
        threads=openable,
    )


def escalated(demand: Demand, attempt: int, growth: float) -> Demand:
    """The same demand, asking for more room after a memory failure.

    The recorded peak cannot be trusted here, and that is the whole reason this
    exists: a run that died of an out-of-memory recorded what it REACHED, not
    what it needed. The next attempt therefore has to ask for more than history
    knows, and keep asking, until the job is admitted alone.

    Only the memory dimensions grow. Giving a job more cores would not stop it
    running out of memory, and would only make it wait longer for room it has
    no use for.
    """
    if attempt <= 0:
        return demand
    factor = growth ** attempt
    return Demand(
        cpus=demand.cpus,
        ram_bytes=int(demand.ram_bytes * factor),
        vram_bytes=int(demand.vram_bytes * factor),
        measured=demand.measured,
        threads=demand.threads,
    )


def holds_everything(demand: Demand, budget: "Budget") -> bool:
    """Is this demand already the whole machine?

    The terminal condition for retrying. Once a job is admitted alone, asking
    for more buys nothing -- there is nothing left to take -- so a failure at
    that point is the tool genuinely not fitting, and trying again just spends
    the card on a run that cannot succeed.
    """
    if not demand.measured:
        return True
    memory_full = (
        budget.vram_bytes and demand.vram_bytes >= budget.vram_bytes
    ) or (budget.ram_bytes and demand.ram_bytes >= budget.ram_bytes)
    return bool(memory_full)


class Grant:
    """What one admitted run may spend, and whether it was ever ALONE.

    Unpacks as `(cores, channels)`, which is all any caller wanted before
    `solo` existed and what every test still reads.

    **`solo` describes the run's whole life, not the instant it was admitted**,
    which is why it is a mutable attribute on an object the budget keeps a
    handle on rather than a value computed once and returned. A run that
    started alone and was joined a second later shared the card for nearly all
    of its work, and a card-wide measurement taken over that window is
    contaminated whatever the machine looked like at admission.

    It outlives the reservation on purpose: what reads it is `dispatch`, after
    the block has exited, because what a run measured is only readable once its
    process has written result.json.
    """

    __slots__ = ("cores", "channels", "solo")

    def __init__(self, cores: int, channels: int = 1, solo: bool = False):
        self.cores = int(cores)
        self.channels = int(channels)
        # False by default, and that is the safe direction: a grant nobody
        # admitted through `reserve` has no evidence the machine was idle, and
        # absence of evidence must never read as evidence of absence.
        self.solo = bool(solo)

    def __iter__(self):
        return iter((self.cores, self.channels))

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "Grant(cores={}, channels={}, solo={})".format(
            self.cores, self.channels, self.solo)


class Cancelled(RuntimeError):
    """The wait was abandoned because the client withdrew the run."""


class Budget:
    """A resource vector, and a queue of jobs waiting for room in it.

    FIFO by ticket: only the job at the head of the queue may be admitted, so a
    heavy job is never starved by a stream of light ones. The cost is
    head-of-line blocking -- a small job waits behind a big one it would have
    fitted beside -- and that is the deliberate trade. Predictable order is
    worth more here than the last few percent of utilisation, because the thing
    being ordered is somebody's patient cohort.
    """

    def __init__(self, allocation, free_vram=None):
        self.cpus = float(allocation.cpus)
        self.ram_bytes = int(allocation.ram_bytes or 0)
        self.vram_bytes = int(allocation.vram_bytes or 0)
        # The declared per-job share. Not a reservation -- `Demand.cpus` is
        # that -- but the FLOOR of what a job is allowed to spend, so a busy
        # machine grants exactly what it always granted.
        self.cpus_per_job = int(allocation.cpus_per_job or 1)
        self._free_vram = free_vram or (lambda: resources.detect_vram_bytes()[1])
        self._condition = threading.Condition()
        self._tickets = itertools.count()
        self._queue: list = []
        self._held_cpus = 0.0
        self._held_ram = 0
        self._held_vram = 0
        self._running = 0
        # Every grant currently holding room. The budget keeps the handles so
        # it can REVOKE their solitude the moment a second run is admitted:
        # solo is a property of a whole run, and it can only be falsified from
        # here, by the arrival that falsifies it.
        self._live: set = set()
        # How many jobs are running that nothing has measured. They hold the
        # machine ALONE, and it has to be symmetric: an unmeasured job may not
        # join anything, and nothing may join an unmeasured job. Counted rather
        # than relying on it having reserved the whole budget, because the
        # numbers on an unmeasured demand are exactly the ones not to trust.
        self._unmeasured = 0

    # -- introspection, for the banner and the tests -------------------
    @property
    def running(self) -> int:
        with self._condition:
            return self._running

    def snapshot(self) -> dict:
        with self._condition:
            return {
                "running": self._running,
                "waiting": len(self._queue),
                "cpus_held": self._held_cpus,
                "ram_held": self._held_ram,
                "vram_held": self._held_vram,
            }

    # -- the decision --------------------------------------------------
    def _would_fit(self, demand: Demand, last_resort: bool = True,
                   among: int = 1) -> bool:
        """Is there room for this demand right now?

        `last_resort` says whether this is the narrowest way the run could be
        admitted. An idle machine takes anything, including a job bigger than
        the whole budget -- refusing that would make a tool this machine can in
        fact run permanently unrunnable. But that escape belongs to the
        narrowest candidate only: when a WIDER one is being offered alongside,
        the run is runnable either way, and letting an idle machine swallow
        eight channels' worth of a budget it does not have is how the escape
        turns from a safety net into the thing it was protecting against.
        """
        if self._running == 0 and last_resort:
            return True
        # `among` is how many runs the free room is being divided between: this
        # one and everything already queued behind it. At 1 the question is the
        # plain one -- is there room. Above it, the question is whether there is
        # room for this run to take its SHARE, which is what stops the head of
        # a long queue from taking the widest shape that happens to fit and
        # leaving forty runs behind it with nothing.
        share = max(1, int(among))
        if not demand.measured or self._unmeasured:
            # Either side of the pairing is enough: an unmeasured job runs
            # alone, so it neither joins nor is joined.
            return False
        if demand.cpus * share + self._held_cpus > self.cpus:
            return False
        if self.ram_bytes and demand.ram_bytes * share + self._held_ram > self.ram_bytes:
            return False
        if self.vram_bytes and demand.vram_bytes * share + self._held_vram > self.vram_bytes:
            return False
        return self._card_has_room(demand)

    def _widest_that_fits(self, candidates, waiting: int):
        """The shape to admit, given what is running AND what is queued.

        Two passes, and the order is the policy:

        1. **Fair.** Would this shape still fit if every run already waiting
           behind it took the same? A head of queue that grabs the widest shape
           that happens to fit leaves forty runs behind it with nothing, and
           the machine ends up running one wide job where it could have been
           running six narrow ones that all started.
        2. **Greedy**, if nothing passed the first. A run alone, or one whose
           queue is short, takes the widest that fits -- and a run that fits
           nowhere at all still gets the idle machine's escape on the narrowest
           shape, so a long queue can never make a run unrunnable.

        Dividing rather than counting is what keeps this free of a threshold:
        there is no "queue too long" number anywhere, because the queue's
        length IS the divisor.
        """
        for among in (waiting + 1, 1):
            if among < 1:
                continue
            fitted = next(
                ((count, want) for index, (count, want) in enumerate(candidates)
                 if self._would_fit(
                     want,
                     # The idle escape belongs to the narrowest shape, and only
                     # on the pass that is no longer trying to be fair.
                     last_resort=(among == 1 and index == len(candidates) - 1),
                     among=among,
                 )),
                None,
            )
            if fitted is not None:
                return fitted
        return None

    def _card_has_room(self, demand: Demand) -> bool:
        """What the card actually has free, not what history says it should.

        History knows what the tools took. It cannot know about the 3 GiB
        another process on this host is already holding, nor about the run that
        has started allocating since it was admitted.
        """
        if not demand.vram_bytes:
            return True
        try:
            free = self._free_vram()
        except Exception:  # noqa: BLE001 - a probe must never fail an admission
            return True
        if free is None:
            return True
        return demand.vram_bytes <= free

    def _cpu_grant(self, openable: float = 0) -> int:
        """How many cores the job just admitted may actually open threads for.

        The declared per-job share, or less where admission narrowed this run to
        fit it in. Never more -- not for an idle machine, and not for a tool
        measured to occupy more.

        **Two ways of granting more have now been measured, and both lost.**

        It used to EXPAND: a job that arrived alone was told it could use the
        whole CPU budget, on the reasoning that idle cores are free. It is not
        free, because a thread that has nothing to do still synchronises.

        Then the expansion was replaced by the measured figure -- the cores the
        tool was actually SEEN to keep busy -- which reads like the safe version
        of the same idea and is not. Occupancy is what a neighbour cannot have;
        it is not evidence the threads earned their keep. A spinning OpenBLAS
        pool is occupancy. So the tool that occupied the most was granted the
        most, and granting it more made it occupy more: a fixed point, and
        nothing in the loop could discover it was the wrong one. Measured on
        this machine on 2026-09-18, `Surg_Mov_Pred` sat exactly there, at 26.11
        recorded cores and therefore 34 granted threads.

        **The curves, best of three, each tool alone, through its own
        virtualenv, on 28 physical cores / 56 logical** (seconds):

            threads          1     4     8    10    14    28    34    42    56
            Surg_Mov_Pred  4.60  4.45  4.45  4.51  4.75  5.32  5.82  6.21  7.80
            AutoMatrix     9.44  7.83  7.43  7.33  7.23  7.13  7.13  7.03  7.18
            AutoCrop3D     1.02  1.07  1.02  1.02  1.02  1.07  1.07  1.02  1.07
            Crown_Seg     54.53 54.33 55.29 54.30 55.59 54.70   --  54.99 55.03
            GreedyReg        --    --   599    --    --   581    --    --   589

        Three shapes, and only one of them is hurt by a wide grant:

        - `Surg_Mov_Pred` DEGRADES above 8, monotonically. Its work is 112
          sequential unpicklings over small matrices, where a BLAS pool costs
          more in synchronisation than it saves. At the 34 it was being granted
          it ran 1.31x its own best; at 56, 1.75x.
        - `AutoMatrix` SCALES, and is the whole price of this rule: 7.33 s at
          the share of ten against 7.03 s at forty-two, so capping it costs
          **4.1%**. It saturates by fourteen and never degrades.
        - `AutoCrop3D`, `Crown_Seg` and `GreedyReg` are INDIFFERENT -- flat to
          within the repeat noise across a 56x range, because their time is
          process startup, file I/O, the card (Crown_Seg) or a search that does
          not thread (GreedyReg: ten real minutes of compute, and 8 threads and
          56 finish within 3% of each other). A wide grant neither helps nor
          hurts them, and they measure 1.24, 1.31 and single-digit cores, so
          they were never granted more than the share anyway.

        4.1% on the one tool that scales, against 31% on the one that does not,
        is why this is a cap rather than a per-tool search. The server cannot
        find a tool's knee by watching: it observes one thread count per run and
        two runs are two different cohorts, so there is nothing to compare. And
        occupancy -- the one signal it does have -- cannot tell the cases apart,
        `Surg_Mov_Pred` occupying 26 of its 34 while getting slower.

        Unmeasured or unstated, it falls back to the declared per-job share,
        which is what every run got before any of this existed.
        """
        return max(1, int(openable or self.cpus_per_job))

    @contextlib.contextmanager
    def reserve(self, candidates, on_wait=None, is_cancelled=None):
        """Hold room for the duration of the block, taking the widest that fits.

        `candidates` is `[(channels, Demand), ...]`, widest FIRST. They are the
        same run at different degrees of parallelism, and what is reserved is
        what that degree actually costs -- so a run cannot be admitted on a
        one-channel reservation and then open eight.

        Trying them in order, rather than picking a count and then queueing for
        it, is what lets a run narrow itself instead of waiting: eight channels
        on a busy machine becomes two now rather than eight in four minutes.
        The narrowest candidate is always the one-channel demand, so the list
        never runs out and this cannot refuse a run the old code would have
        admitted.

        `on_wait` is called once, only if the job actually has to queue, so a
        client is told it is waiting rather than being told so on every run.
        `is_cancelled` is polled while waiting: a client that gives up on a
        queued run must not be held until the room it no longer wants frees,
        which behind a multi-hour cohort is the longest wait in the system.

        Yields a `Grant` -- what it may spend, how widely, and whether it ever
        had the machine to itself. It unpacks as `(cores, channels)`, so a
        caller with no use for the third thing reads it exactly as before.
        """
        # A bare Demand is one candidate at one channel. Accepted because most
        # callers -- and every test that predates channels -- have exactly one
        # degree of parallelism to offer, and making them wrap it would be
        # ceremony with no reader.
        if isinstance(candidates, Demand):
            candidates = [(1, candidates)]
        candidates = list(candidates) or [(1, whole_machine(self))]
        ticket = next(self._tickets)
        announced = False
        with self._condition:
            self._queue.append(ticket)
            while True:
                if is_cancelled is not None and is_cancelled():
                    self._queue.remove(ticket)
                    self._condition.notify_all()
                    raise Cancelled("The client cancelled this run.")
                channels, demand = candidates[-1]
                if self._queue[0] == ticket:
                    fitted = self._widest_that_fits(candidates, len(self._queue) - 1)
                    if fitted is not None:
                        channels, demand = fitted
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
                self._condition.wait(timeout=settings.RUN_CANCEL_POLL_SECONDS)
            self._queue.remove(ticket)
            # SOLO means nothing else held a reservation at ANY point between
            # this run's admission and its release. A run admitted onto an idle
            # machine starts solo; the moment a second reservation is taken
            # every run in flight -- the newcomer included -- stops being solo,
            # permanently. It is never regained, because the window a card-wide
            # measurement covers is the whole run: a neighbour that came and
            # went still allocated inside it.
            grant = Grant(self._cpu_grant(demand.threads), channels,
                          solo=(self._running == 0))
            if not grant.solo:
                for other in self._live:
                    other.solo = False
            self._live.add(grant)
            self._held_cpus += demand.cpus
            self._held_ram += demand.ram_bytes
            self._held_vram += demand.vram_bytes
            self._running += 1
            self._unmeasured += 0 if demand.measured else 1
        try:
            yield grant
        finally:
            with self._condition:
                # Discarded from the live set, never mutated: the caller reads
                # `solo` AFTER this block, from the run's result file.
                self._live.discard(grant)
                self._held_cpus -= demand.cpus
                self._held_ram -= demand.ram_bytes
                self._held_vram -= demand.vram_bytes
                self._running -= 1
                self._unmeasured -= 0 if demand.measured else 1
                self._condition.notify_all()


_budget: Optional[Budget] = None
_budget_lock = threading.Lock()


def budget() -> Budget:
    """This process's admission policy, built once.

    Lazy for the same reason the allocation is: a test must be able to replace
    the setting before the first run reads it.
    """
    global _budget
    with _budget_lock:
        if _budget is None:
            _budget = Budget(
                resources.allocation()
            )
        return _budget


def reset() -> None:
    """Forget the built budget. For tests, and for nothing else."""
    global _budget
    with _budget_lock:
        _budget = None
