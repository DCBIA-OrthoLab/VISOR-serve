"""Admission in bytes and cores, in place of a count of jobs.

The threads here are the point rather than an inconvenience: what is being
tested is whether two runs may hold the machine at the same time, which cannot
be asserted from one thread. Every wait is on an Event with a timeout, never on
a sleep, so a broken implementation fails fast instead of passing slowly.
"""

import os
import threading

import pytest

os.environ.setdefault("API_TOKEN", "test-token")

import resources
from execution import admission
from execution.costs import Cost

_TIMEOUT = 5.0


def _allocation(cpus=8.0, ram=16 * 1024 ** 3, vram=8 * 1024 ** 3, per_job=2):
    return resources.Allocation(
        cpus=cpus, ram_bytes=ram, vram_bytes=vram,
        cpus_per_job=per_job, ram_per_job=ram // 4, vram_per_job=vram // 4,
        expected_clients=4,
    )


def _budget(free_vram=None, **kwargs):
    return admission.Budget(
        _allocation(**kwargs),
        free_vram=free_vram or (lambda: 1 << 60),
    )


def _small(vram=1 * 1024 ** 3, ram=1 * 1024 ** 3, cpus=2):
    return admission.Demand(cpus=cpus, ram_bytes=ram, vram_bytes=vram, measured=True)


class _Holder:
    """Holds a reservation on a thread until told to let go."""

    def __init__(self, budget, demand, **kwargs):
        self.budget = budget
        self.demand = demand
        self.kwargs = kwargs
        self.admitted = threading.Event()
        self.release = threading.Event()
        self.error = None
        # The grant this run was given, kept after the block so a test can read
        # `solo` the way dispatch does: once the run is over.
        self.grant = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            with self.budget.reserve(self.demand, **self.kwargs) as grant:
                self.grant = grant
                self.admitted.set()
                self.release.wait(_TIMEOUT)
        except BaseException as exc:  # noqa: BLE001 - reported to the test
            self.error = exc
            self.admitted.set()

    def start(self):
        self.thread.start()
        return self

    def wait_admitted(self, timeout=_TIMEOUT):
        return self.admitted.wait(timeout)

    def finish(self):
        self.release.set()
        self.thread.join(_TIMEOUT)


# ----------------------------------------------------------------------
# What fits
# ----------------------------------------------------------------------

def test_an_idle_machine_takes_anything():
    """Including a job bigger than the whole budget. Refusing it would make a
    tool this machine can in fact run permanently unrunnable."""
    budget = _budget(vram=2 * 1024 ** 3)
    huge = admission.Demand(cpus=99, ram_bytes=1 << 50, vram_bytes=1 << 50, measured=True)
    with budget.reserve(huge):
        assert budget.running == 1


def test_two_jobs_that_fit_hold_the_machine_together():
    """The whole point: ALI_CBCT peaks at 0.25 GiB on a 48 GiB card, and a
    counter of one left ~95% of it idle."""
    budget = _budget()
    first = _Holder(budget, _small()).start()
    assert first.wait_admitted()
    second = _Holder(budget, _small()).start()
    assert second.wait_admitted(), "the second job was refused room it had"
    assert budget.running == 2
    first.finish()
    second.finish()


def test_a_job_that_does_not_fit_waits_for_one_that_does_to_leave():
    budget = _budget(vram=4 * 1024 ** 3, cpus=8.0)
    big = _small(vram=3 * 1024 ** 3)
    first = _Holder(budget, big).start()
    assert first.wait_admitted()
    second = _Holder(budget, big).start()
    assert not second.wait_admitted(timeout=1.0), "two 3 GiB jobs fitted in 4 GiB"
    first.finish()
    assert second.wait_admitted(), "the queued job never started"
    second.finish()


@pytest.mark.parametrize("dimension", ["cpus", "ram_bytes", "vram_bytes"])
def test_every_dimension_can_be_the_one_that_binds(dimension):
    """A vector, not a VRAM budget with two decorations."""
    budget = _budget(cpus=4.0, ram=4 * 1024 ** 3, vram=4 * 1024 ** 3)
    fields = {"cpus": 1, "ram_bytes": 1024, "vram_bytes": 1024, "measured": True}
    fields[dimension] = {"cpus": 3, "ram_bytes": 3 * 1024 ** 3, "vram_bytes": 3 * 1024 ** 3}[
        dimension
    ]
    demand = admission.Demand(**fields)
    first = _Holder(budget, demand).start()
    assert first.wait_admitted()
    second = _Holder(budget, demand).start()
    assert not second.wait_admitted(timeout=1.0), f"{dimension} did not bind"
    first.finish()
    second.finish()


# ----------------------------------------------------------------------
# What is not known
# ----------------------------------------------------------------------

def test_an_unmeasured_tool_runs_alone():
    """So an empty cost table behaves exactly like the counter this replaces,
    and the server is never worse than what it grew out of."""
    budget = _budget()
    unknown = admission.Demand(cpus=1, ram_bytes=0, vram_bytes=0, measured=False)
    first = _Holder(budget, unknown).start()
    assert first.wait_admitted()
    second = _Holder(budget, _small(vram=1024, ram=1024, cpus=1)).start()
    assert not second.wait_admitted(timeout=1.0), "something ran beside an unmeasured tool"
    first.finish()
    assert second.wait_admitted()
    second.finish()


def test_nothing_may_join_an_unmeasured_tool_either_way_round():
    budget = _budget()
    tiny = _small(vram=1024, ram=1024, cpus=1)
    first = _Holder(budget, tiny).start()
    assert first.wait_admitted()
    unknown = admission.Demand(cpus=1, ram_bytes=0, vram_bytes=0, measured=False)
    second = _Holder(budget, unknown).start()
    assert not second.wait_admitted(timeout=1.0)
    first.finish()
    assert second.wait_admitted()
    second.finish()


# ----------------------------------------------------------------------
# Order
# ----------------------------------------------------------------------

def test_a_heavy_job_is_not_starved_by_a_stream_of_light_ones():
    """FIFO by ticket: only the head of the queue may be admitted. The cost is
    head-of-line blocking, and it is the deliberate trade -- the thing being
    ordered is somebody's patient cohort."""
    budget = _budget(vram=4 * 1024 ** 3)
    blocker = _Holder(budget, _small(vram=3 * 1024 ** 3)).start()
    assert blocker.wait_admitted()

    heavy = _Holder(budget, _small(vram=3 * 1024 ** 3)).start()
    assert not heavy.wait_admitted(timeout=0.5)
    light = _Holder(budget, _small(vram=1024, ram=1024, cpus=1)).start()
    assert not light.wait_admitted(timeout=0.5), "a light job overtook the queue head"

    blocker.finish()
    assert heavy.wait_admitted(), "the head of the queue never ran"
    heavy.finish()
    assert light.wait_admitted()
    light.finish()


# ----------------------------------------------------------------------
# Giving up
# ----------------------------------------------------------------------

def test_a_cancelled_run_stops_waiting():
    """Behind a multi-hour cohort, the queue is the longest wait in the system.
    A client that gave up must not be held in it."""
    budget = _budget(vram=2 * 1024 ** 3)
    holder = _Holder(budget, _small(vram=2 * 1024 ** 3)).start()
    assert holder.wait_admitted()

    cancelled = threading.Event()
    waiter = _Holder(
        budget, _small(vram=2 * 1024 ** 3), is_cancelled=cancelled.is_set
    ).start()
    assert not waiter.wait_admitted(timeout=0.5)
    cancelled.set()
    waiter.thread.join(_TIMEOUT)
    assert isinstance(waiter.error, admission.Cancelled)
    holder.finish()


def test_a_cancelled_waiter_leaves_the_queue_behind_it_free():
    budget = _budget(vram=2 * 1024 ** 3)
    holder = _Holder(budget, _small(vram=2 * 1024 ** 3)).start()
    assert holder.wait_admitted()
    cancelled = threading.Event()
    doomed = _Holder(budget, _small(vram=2 * 1024 ** 3), is_cancelled=cancelled.is_set).start()
    assert not doomed.wait_admitted(timeout=0.5)
    after = _Holder(budget, _small(vram=1024, ram=1024, cpus=1)).start()
    cancelled.set()
    doomed.thread.join(_TIMEOUT)
    holder.finish()
    assert after.wait_admitted(), "the queue stayed blocked by a run that left it"
    after.finish()


def test_a_failing_run_gives_its_room_back():
    budget = _budget(vram=2 * 1024 ** 3)
    demand = _small(vram=2 * 1024 ** 3)
    with pytest.raises(ValueError):
        with budget.reserve(demand):
            raise ValueError("the tool crashed")
    assert budget.snapshot()["vram_held"] == 0
    assert budget.running == 0


# ----------------------------------------------------------------------
# Telling the client
# ----------------------------------------------------------------------

def test_a_run_admitted_at_once_is_never_announced_as_queueing():
    announcements = []
    budget = _budget()
    with budget.reserve(_small(), on_wait=lambda: announcements.append(1)):
        pass
    assert announcements == []


def test_a_run_that_waits_is_announced_once_and_only_once():
    announcements = []
    budget = _budget(vram=2 * 1024 ** 3)
    holder = _Holder(budget, _small(vram=2 * 1024 ** 3)).start()
    assert holder.wait_admitted()
    waiter = _Holder(
        budget, _small(vram=2 * 1024 ** 3), on_wait=lambda: announcements.append(1)
    ).start()
    assert not waiter.wait_admitted(timeout=1.5)
    holder.finish()
    assert waiter.wait_admitted()
    waiter.finish()
    assert announcements == [1], "a waiting run announced itself on every poll"


# ----------------------------------------------------------------------
# The backstop
# ----------------------------------------------------------------------
#
# There is no job COUNTER any more. `MAX_CONCURRENT_GPU_JOBS` was the last one
# and it is gone: it capped every running job despite its name, so pinning it
# to 1 would have serialised a tabular prediction behind a segmentation --
# exactly what the per-tool semaphores it replaced were removed for. What a
# tool costs is a function of the width it was granted, and the budget is the
# whole policy.

def test_the_card_s_actual_free_memory_is_checked_too():
    """History knows what the tools took. It cannot know about the 3 GiB
    another process on this host is already holding."""
    budget = _budget(free_vram=lambda: 512 * 1024 ** 2)
    first = _Holder(budget, _small(vram=1024, ram=1024, cpus=1)).start()
    assert first.wait_admitted()
    second = _Holder(budget, _small(vram=2 * 1024 ** 3)).start()
    assert not second.wait_admitted(timeout=1.0), "admitted onto a card with no room"
    first.finish()
    second.finish()


def test_a_probe_that_cannot_answer_never_blocks_a_run():
    def _broken():
        raise OSError("nvidia-smi went away")

    budget = _budget(free_vram=_broken)
    first = _Holder(budget, _small()).start()
    assert first.wait_admitted()
    second = _Holder(budget, _small()).start()
    assert second.wait_admitted(), "a failed probe refused a run that fitted"
    first.finish()
    second.finish()


def test_a_cpu_only_run_does_not_consult_the_card():
    def _explode():
        raise AssertionError("the card was probed for a run that never wanted it")

    budget = _budget(free_vram=_explode)
    with budget.reserve(_small(vram=0)):
        pass


# ----------------------------------------------------------------------
# Turning a measurement into a demand
# ----------------------------------------------------------------------

def test_a_measured_tool_reserves_its_peak_plus_a_margin():
    """A peak learned on a small scan under-estimates a large one, and
    under-estimating is the direction that ends in an out-of-memory."""
    demand = admission.demand_for(
        "AMASSS", _allocation(), Cost(vram_bytes=2_349_230_080, ram_bytes=1000, samples=3)
    )
    assert demand.vram_bytes == int(2_349_230_080 * admission.SAFETY_MARGIN)
    assert demand.measured is True


def test_an_unmeasured_tool_reserves_the_whole_budget():
    allocation = _allocation()
    demand = admission.demand_for("Brand_New_Tool", allocation, None)
    assert demand.vram_bytes == allocation.vram_bytes
    assert demand.ram_bytes == allocation.ram_bytes
    assert demand.measured is False


def test_a_tool_measured_at_nothing_at_all_is_treated_as_unmeasured():
    demand = admission.demand_for("Odd", _allocation(), Cost(0, 0, 1))
    assert demand.measured is False


def test_a_run_that_never_wanted_the_card_reserves_no_vram():
    """A tabular prediction must not queue behind a segmentation for memory it
    is not going to touch."""
    demand = admission.demand_for(
        "Surg_Mov_Pred",
        _allocation(),
        Cost(vram_bytes=2_000_000_000, ram_bytes=1000, samples=1),
        uses_gpu=False,
    )
    assert demand.vram_bytes == 0
    assert demand.ram_bytes > 0


def test_a_job_always_holds_the_cores_it_was_told_it_could_use():
    """It is told `cpus_per_job` through OMP_NUM_THREADS, so that is what it
    holds whatever it measured."""
    allocation = _allocation(per_job=6)
    demand = admission.demand_for("AMASSS", allocation, Cost(1, 1, 1))
    assert demand.cpus == 6


# ----------------------------------------------------------------------
# What a run may SPEND is not what it HOLDS
# ----------------------------------------------------------------------
#
# Two ways of granting more than the declared share have been measured, and
# both lost. The grant first EXPANDED -- a job alone on the machine was told it
# could use the whole budget, idle cores being free. Then the expansion was
# replaced by the MEASURED occupancy, which reads like the safe version of the
# same idea and is not: a spinning BLAS pool is occupancy, so the tool that
# occupied most was granted most, and granting it more made it occupy more.
#
# Measured on 2026-09-18, best of three, each tool alone in its own virtualenv
# on 28 physical cores (seconds):
#
#     threads          1     4     8    10    14    28    34    42    56
#     Surg_Mov_Pred  4.60  4.45  4.45  4.51  4.75  5.32  5.82  6.21  7.80
#     AutoMatrix     9.44  7.83  7.43  7.33  7.23  7.13  7.13  7.03  7.18
#
# Surg_Mov_Pred was sitting at 34 granted threads, 1.31x its own best; capping
# AutoMatrix, the one tool of the five that scales, costs 4.1%.

def test_a_run_is_granted_what_it_may_open_not_what_it_holds():
    """The two are different numbers and this is the one that must not follow
    the other: a reservation of 34 cores earned by a spinning pool would
    otherwise hand that pool 34 threads and keep it spinning."""
    budget = _budget(cpus=42.0, per_job=10)
    demand = admission.Demand(cpus=34, ram_bytes=1, vram_bytes=1, threads=10)
    with budget.reserve(demand) as (cores, _channels):
        assert cores == 10, "the occupancy figure became the thread limit again"


def test_a_narrowed_run_holds_its_measured_cores_but_opens_only_its_share():
    """`demand_for` fills both, and they part company exactly here: the cores
    the tool was SEEN to keep busy are reserved, because a neighbour cannot have
    them; the threads it is TOLD to open stay at the share."""
    allocation = _allocation(cpus=42.0, per_job=10)
    demand = admission.demand_for(
        "Surg_Mov_Pred", allocation,
        Cost(vram_bytes=0, ram_bytes=1 << 30, samples=5, cpu_cores=26.11),
        uses_gpu=False,
    )
    assert demand.cpus == 34, "the measured occupancy stopped being reserved"
    assert demand.threads == 10, "the measured occupancy became the thread limit"


def test_narrowing_a_run_to_fit_it_in_narrows_what_it_may_open():
    """The case the coupling broke hardest. Admission narrows a candidate's
    cores to squeeze it onto a busy machine, but the occupancy bump is a
    `max()` -- so a run narrowed to four still reserved 34, and was then TOLD
    it could open 34. It kept the reservation (a neighbour really cannot have
    those cores) and lost the narrowing, which is the only half that was
    supposed to make it a smaller run."""
    allocation = _allocation(cpus=42.0, per_job=10)
    demand = admission.demand_for(
        "Surg_Mov_Pred", allocation,
        Cost(vram_bytes=0, ram_bytes=1 << 30, samples=5, cpu_cores=26.11),
        uses_gpu=False, cpus=4,
    )
    assert demand.threads == 4, "a narrowed run was told it could open the lot"


def test_a_memory_retry_does_not_lose_the_thread_limit():
    """`escalated` rebuilds the Demand, and a field it forgets is a field that
    silently reverts to 'nothing said' on the one path already going wrong."""
    demand = admission.Demand(cpus=34, ram_bytes=1 << 30, vram_bytes=0, threads=10)
    assert admission.escalated(demand, attempt=1, growth=2.0).threads == 10


def test_an_unmeasured_run_falls_back_to_the_declared_share():
    """Which is what every run got before any of this existed."""
    budget = _budget(cpus=42.0, per_job=10)
    assert budget._cpu_grant(0) == 10


def test_a_narrowed_run_is_not_then_told_it_may_open_ten():
    """A run admitted on two cores keeps two. Twenty such runs would otherwise
    be two hundred threads on forty cores."""
    budget = _budget(cpus=42.0, per_job=10)
    assert budget._cpu_grant(2) == 2


# ----------------------------------------------------------------------
# ...and the grant has to reach the process
# ----------------------------------------------------------------------

def test_every_thread_variable_carries_the_grant():
    from execution import dispatch

    limits = dispatch._thread_limits(11)
    assert set(limits) == set(dispatch._THREAD_VARIABLES)
    assert set(limits.values()) == {"11"}


def test_a_caller_outside_the_admission_path_still_gets_a_cap(monkeypatch):
    """Omitting the grant must mean the declared share, never the machine.
    `_child_environment` builds its environment before admission has happened,
    and an uncapped default there is the 224-threads-on-28-cores bug again."""
    from execution import dispatch

    monkeypatch.setattr(resources, "allocation", lambda: _allocation(per_job=3))
    assert set(dispatch._thread_limits().values()) == {"3"}


# ----------------------------------------------------------------------
# Whether a run ever had the machine to itself
# ----------------------------------------------------------------------
#
# What reads this is the cost table. A VRAM figure taken from the CARD is only
# this run's when nothing else was allocating on it, and on this deployment the
# card is the only reading available (the driver answers --query-compute-apps
# with nothing at all). Measured this afternoon: ALI_CBCT alone costs 1.12 GiB
# per channel, flat from one channel to fifteen, while the table -- taught by
# contended runs -- held twice that, and ASO 32.43 GiB per channel.

def test_a_run_admitted_onto_an_idle_machine_is_solo():
    budget = _budget()
    with budget.reserve(_small()) as grant:
        assert grant.solo


def test_a_grant_still_unpacks_as_cores_and_channels():
    """Every caller that has no use for `solo` reads it exactly as before."""
    budget = _budget()
    with budget.reserve(_small()) as (cores, channels):
        assert cores >= 1
        assert channels == 1


def test_solo_is_lost_when_a_second_run_is_admitted_and_never_regained():
    """A card-wide reading covers the run's WHOLE life, so a neighbour that came
    and went is still inside it. Solitude is falsified once and for good."""
    budget = _budget()
    first = _Holder(budget, _small()).start()
    assert first.wait_admitted()
    assert first.grant.solo, "the first run had the machine to itself"

    second = _Holder(budget, _small()).start()
    assert second.wait_admitted(), "the second job was refused room it had"
    assert not first.grant.solo, "a run that was joined is no longer alone"
    assert not second.grant.solo, "the newcomer shares the machine too"

    second.finish()
    assert not first.grant.solo, "solitude came back after the neighbour left"
    first.finish()
    # Read the way dispatch reads it: after the reservation is gone, because
    # what a run measured is only readable once its process has exited.
    assert not first.grant.solo


def test_a_run_that_starts_after_another_has_finished_is_solo_again():
    """It is the RUN that is solo, not the machine: nothing held a reservation
    at any point between this one's admission and its release."""
    budget = _budget()
    first = _Holder(budget, _small()).start()
    assert first.wait_admitted()
    first.finish()
    with budget.reserve(_small()) as grant:
        assert grant.solo


def test_a_tool_whose_vram_was_never_believed_reserves_the_whole_budget():
    """Absence, not a measurement of zero, and they admit in opposite
    directions: a tool measured at zero VRAM shares the card with anything,
    while one whose every card reading was refused knows nothing about it."""
    allocation = _allocation()
    unknown = Cost(vram_bytes=0, ram_bytes=1 << 30, samples=3, vram_known=False)
    demand = admission.demand_for("Card_Shy", allocation, unknown, uses_gpu=True)
    assert demand.measured is False
    assert demand.vram_bytes == allocation.vram_bytes
    assert demand.ram_bytes == allocation.ram_bytes


def test_a_cpu_run_is_not_held_back_by_an_unknown_vram():
    """It is never going to touch the card, so what the card never measured
    says nothing about it."""
    known = Cost(vram_bytes=0, ram_bytes=1 << 30, samples=3, vram_known=False)
    demand = admission.demand_for("Surg_Mov_Pred", _allocation(), known,
                                  uses_gpu=False)
    assert demand.measured is True
    assert demand.vram_bytes == 0
