"""How many channels a tool may open inside one run.

`admission.py` decides how many RUNS share the machine. This decides what one
run may do with the share it was given. The two compose, and composing them
wrongly is the whole risk: eight channels each opening seven OpenMP threads is
fifty-six threads on a budget of seven, and a chain whose every level opens
eight is eight to the power of its depth.

Run with: cd server && ../.venv/bin/python -m pytest tests/test_concurrency.py
"""

import json
import os

os.environ.setdefault("API_TOKEN", "test-token")

import pytest

import resources
from config import settings
from execution import admission, concurrency, dispatch


class _Tool:
    def __init__(self, name="Probe", arguments=("num_workers",)):
        self.name = name
        self.arguments = {argument: object() for argument in arguments}


# ----------------------------------------------------------------------
# A capability nobody declared is not one the server may assume
# ----------------------------------------------------------------------

def test_a_tool_that_declares_no_channel_argument_is_left_alone():
    """Opening several channels inside a loop that shares a temporary file
    corrupts its own outputs, silently. The declaration is the tool saying it
    does not -- and most tools do not declare it."""
    assert concurrency.channel_argument(_Tool(arguments=())) is None
    assert concurrency.ceiling(_Tool(arguments=()), {}) == 1
    assert not concurrency.granted(_Tool(arguments=()), 1).applies


def test_num_workers_is_recognised():
    assert concurrency.channel_argument(_Tool()) == "num_workers"


def test_batch_size_is_recognised():
    tool = _Tool(arguments=("batch_size",))
    assert concurrency.channel_argument(tool) == "batch_size"


def test_the_loop_wins_over_the_batch_when_a_tool_declares_both():
    """`num_workers` means items of the tool's own loop and `batch_size` means
    items in one forward pass. The loop is the outer of the two."""
    tool = _Tool(arguments=("batch_size", "num_workers"))
    assert concurrency.channel_argument(tool) == "num_workers"


# ----------------------------------------------------------------------
# What bounds the width
# ----------------------------------------------------------------------
#
# NOT the cores. Tying the two made the CPU bound a tool whose bottleneck is
# not the CPU: ALI_CBCT measures 28.1s on seven cores and 29.4s on forty-two --
# it does not scale with cores at all -- and the coupling still capped its
# channels at `cpus_per_job / 2`. A channel's cost is memory, which admission
# reserves; what it needs of the CPU is a share of the threads the run already
# holds, which `_thread_limits` divides.

def test_the_ceiling_is_what_the_budget_affords_not_the_core_share(monkeypatch):
    """Not `cpus_per_job / 2`, which is what bounded it before and which made
    the CPU cap a tool whose bottleneck is the card."""
    monkeypatch.setattr(settings, "SADT_MAX_CHANNELS", 0)
    monkeypatch.setattr(concurrency, "_affordable", lambda name, budget=None: 12)
    assert concurrency.ceiling(_Tool(), {}) == 12


def test_a_tool_declaring_nothing_has_a_ceiling_of_one(monkeypatch):
    monkeypatch.setattr(settings, "SADT_MAX_CHANNELS", 8)
    assert concurrency.ceiling(_Tool(arguments=()), {}) == 1


def test_a_caller_who_named_a_number_keeps_it(monkeypatch):
    """An API client measuring a value must get the value it asked for."""
    monkeypatch.setattr(settings, "SADT_MAX_CHANNELS", 8)
    assert concurrency.ceiling(_Tool(), {"num_workers": 3}) == 3


def test_a_caller_cannot_ask_for_more_than_the_server_allows(monkeypatch):
    """`sitk.WriteTransform` was measured printing HDF5's "infinite loop
    closing library" at sixteen threads on this deployment, and aborting the
    process outright at eight in another tool. Sixteen is above anything the
    server would grant and exactly what an unclamped request reaches."""
    monkeypatch.setattr(settings, "SADT_MAX_CHANNELS", 8)
    assert concurrency.ceiling(_Tool(), {"num_workers": 64}) == 8


# ----------------------------------------------------------------------
# There is no per-tool cap
# ----------------------------------------------------------------------
#
# `max_workers` existed to protect against two things and neither survived:
# ALI_CBCT losing landmarks at width 4, fixed at its source by scaling the
# per-agent budget with the width; and a tool that must never overlap, which
# cannot arise because a tool declaring no channel argument is never given one.
#
# What bounds a width is what it COSTS and what was ASKED FOR, both measured.
# A number written in a file could only disagree with them.

def test_a_declared_cap_still_clamps_a_width(monkeypatch):
    """There is no cap by default. A deployment that would rather pin one than
    reason about it still can, and it binds a caller too."""
    monkeypatch.setattr(settings, "SADT_MAX_CHANNELS", 4)
    monkeypatch.setattr(concurrency, "_affordable", lambda name, budget=None: 99)
    assert concurrency.ceiling(_Tool(), {}) == 4
    assert concurrency.ceiling(_Tool(), {"num_workers": 64}) == 4


def test_with_no_cap_the_measured_cost_is_what_bounds_it(monkeypatch):
    """The budget divided by what one channel was measured to need -- the
    honest bound, and a finite one, which the shapes ladder needs."""
    monkeypatch.setattr(settings, "SADT_MAX_CHANNELS", 0)
    monkeypatch.setattr(concurrency, "_affordable", lambda name, budget=None: 12)
    assert concurrency.ceiling(_Tool(), {}) == 12


# ----------------------------------------------------------------------
# The supervisor: a chain hands down BYTES, and must not multiply
# ----------------------------------------------------------------------
#
# What a level inherits used to be `max(1, cpus // channels)` -- its parent's
# CORE grant divided by its parent's CHANNEL grant. Measured on this machine on
# 2026-09-18, that was wrong three ways at once: ASO was granted 28 cores and 5
# channels, opened ONE, and handed its child 28 // 5; on the next run 31 cores
# and 4 channels, so the same ALI_CBCT asking for the same seven landmarks got
# 5 one minute and 7 the next, from a number that was never about it.

def _measured(monkeypatch, tool_name, vram, ram=0, allocation=None):
    """A cost table holding one tool, and a machine budget to weigh it against."""
    from execution.costs import Cost

    monkeypatch.setattr(concurrency.costs, "cost_of",
                        lambda name: Cost(vram_bytes=vram, ram_bytes=ram, samples=3)
                        if name == tool_name else None)
    monkeypatch.setattr(
        concurrency, "machine_budget",
        lambda: allocation if allocation is not None else concurrency.Reservation(
            vram_bytes=32 << 30, ram_bytes=64 << 30),
    )


def test_a_nested_level_is_sized_by_its_own_cost_not_by_cores(monkeypatch):
    """The whole correction. A child asks what IT was measured to need of the
    room it holds -- there is no core count anywhere in the answer."""
    monkeypatch.setattr(settings, "SADT_MAX_CHANNELS", 0)
    _measured(monkeypatch, "Probe", vram=1 << 30)
    inherited = concurrency.Reservation(vram_bytes=8 << 30)
    assert concurrency.ceiling(_Tool(), {}, budget=inherited) == 8


def test_a_child_is_the_same_width_whoever_called_it(monkeypatch):
    """Defect 3: the same ALI_CBCT got 7 standalone, 5 under an ASO holding 28
    cores and 7 under one holding 31. The room it holds is what decides now,
    and two parents holding the same room give the same answer."""
    monkeypatch.setattr(settings, "SADT_MAX_CHANNELS", 0)
    _measured(monkeypatch, "Probe", vram=1 << 30)
    room = concurrency.Reservation(vram_bytes=6 << 30)
    assert concurrency.ceiling(_Tool(), {}, budget=room) == 6
    assert concurrency.ceiling(_Tool(), {}, budget=room) == 6


def test_an_unused_parent_grant_no_longer_shrinks_its_child(monkeypatch):
    """Defect 2. A reservation is a per-channel cost TIMES the channel count,
    so dividing it by that count gives one channel's worth however many were
    granted -- five channels reserve five times the bytes and one of those five
    is what a child spends. ASO opening one of the five costs its child nothing.

    The division is the SUPERVISOR's now (`runner._child_channel_budget`), so
    what is asserted here is the property that makes it safe: the room scales
    with the grant exactly, and per-channel it does not move."""
    _measured(monkeypatch, "ASO", vram=5 << 30, ram=10 << 30)
    wide = concurrency.granted(_Tool(name="ASO"), 5)
    narrow = concurrency.granted(_Tool(name="ASO"), 1)
    assert wide.room.per(wide.channels) == narrow.room.per(narrow.channels)
    assert wide.room.vram_bytes == 25 << 30, "five channels' worth, reserved"
    assert wide.room.per(5).vram_bytes == 5 << 30, "one of them, for a child"


def test_a_chain_cannot_multiply(monkeypatch):
    """A parent on six channels whose child opens eight is forty-eight on a
    machine that admitted one job. Each level spends its own room, and the room
    was divided on the way down -- so what the whole chain holds at any depth
    telescopes back to what the root reserved."""
    monkeypatch.setattr(settings, "SADT_MAX_CHANNELS", 0)
    _measured(monkeypatch, "Probe", vram=1 << 30)

    root = concurrency.granted(_Tool(), 4)
    held = root.room.vram_bytes
    # What the root hands one child: its own room over the channels it opened.
    budget, opened = root.room.per(root.channels), root.channels
    for _ in range(4):
        width = concurrency.ceiling(_Tool(), {}, budget=budget)
        level = concurrency.granted(_Tool(), width)
        assert level.room.vram_bytes <= budget.vram_bytes
        opened *= width
        assert opened * (1 << 30) <= held, "the whole depth fits what the root holds"
        budget = level.room.per(level.channels)


def test_a_chain_cannot_reach_zero(monkeypatch):
    """The floor of one is what makes a deep chain safe rather than broken. A
    budget that has divided its way down to nothing still buys one channel."""
    monkeypatch.setattr(settings, "SADT_MAX_CHANNELS", 8)
    _measured(monkeypatch, "Probe", vram=1 << 30)
    budget = concurrency.Reservation(vram_bytes=7)
    for _ in range(10):
        channels = concurrency.ceiling(_Tool(), {}, budget=budget)
        granted = concurrency.granted(_Tool(), channels)
        assert channels >= 1
        budget = budget.per(channels)
        assert budget.vram_bytes >= 0


def test_what_admission_settled_on_is_what_fills_the_argument(monkeypatch):
    """`granted` cannot revise the number -- the reservation is already held
    against it."""
    _measured(monkeypatch, "Probe", vram=2 << 30)
    settled = concurrency.granted(_Tool(), 3)
    assert settled.argument == "num_workers"
    assert settled.channels == 3
    assert settled.room.vram_bytes == 6 << 30, "the three channels it reserved"
    assert settled.room.per(3).vram_bytes == 2 << 30, "one of them, for a child"


def test_a_tool_declaring_nothing_still_leaves_a_budget_for_its_children(monkeypatch):
    """An orchestrator that parallelises nothing itself -- AREG, ASO -- must
    still hand its children room rather than whatever the container had."""
    _measured(monkeypatch, "Probe", vram=3 << 30)
    assert concurrency.granted(_Tool(arguments=()), 1).room.vram_bytes == 3 << 30


def test_a_tool_with_no_measured_cost_still_reserves_everything(monkeypatch):
    """Nothing has measured it, so it is admitted as if it needed the machine
    -- and what it hands a child is the machine, which is right: it is running
    alone. The rule that makes an empty cost table behave like the job counter
    it replaced has to hold at depth too."""
    monkeypatch.setattr(concurrency.costs, "cost_of", lambda name: None)
    whole = concurrency.Reservation(vram_bytes=32 << 30, ram_bytes=64 << 30)
    monkeypatch.setattr(concurrency, "machine_budget", lambda: whole)
    assert concurrency.reservation("Unseen", 4) == whole
    assert concurrency.granted(_Tool(name="Unseen"), 1).room == whole
    # And it is offered exactly one channel, so the ladder holds no shape that
    # could not fit.
    assert concurrency.ceiling(_Tool(name="Unseen"), {}) == 1
    assert concurrency.ceiling(_Tool(name="Unseen"), {}, budget=whole) == 1


def test_the_room_handed_down_carries_no_safety_margin(monkeypatch):
    """`admission.SAFETY_MARGIN` is applied once, over the whole job, and a
    chain is one job. Carrying a margined figure down would compound it again
    at every hop."""
    _measured(monkeypatch, "Probe", vram=4 << 30)
    assert concurrency.reservation("Probe", 2).vram_bytes == 8 << 30

# ----------------------------------------------------------------------
# Channels and threads multiply, so they have to be divided
# ----------------------------------------------------------------------

def test_threads_are_divided_by_the_channels_granted():
    """Eight channels each opening seven threads is fifty-six on a budget of
    seven. What a tool may open in TOTAL is the core grant."""
    limits = dispatch._thread_limits(cpus=42, channels=6)
    assert set(limits.values()) == {"7"}


def test_one_channel_gets_the_whole_core_grant():
    assert set(dispatch._thread_limits(cpus=42, channels=1).values()) == {"42"}


def test_a_run_never_gets_fewer_than_one_thread():
    assert set(dispatch._thread_limits(cpus=2, channels=8).values()) == {"1"}


def test_itk_is_capped_too():
    """ITK runs its own pool, sized from the machine and NOT from OpenMP's
    variable -- so every SimpleITK resample and compressed write was ignoring
    the cap and taking one thread per logical core. It is the busiest CPU path
    in most of this catalogue."""
    assert "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS" in dispatch._thread_limits(cpus=8)


# ----------------------------------------------------------------------
# What reaches the process
# ----------------------------------------------------------------------

def test_the_child_environment_always_carries_a_budget():
    """Written even for one channel, so a nested call finds room rather than
    inheriting whatever the container was started with."""
    granted = concurrency.Grant("num_workers", 1,
                                concurrency.Reservation(1 << 30, 2 << 30), "whatever")
    environment = concurrency.child_budget({}, granted)
    assert environment[concurrency.BUDGET_ENV] == "1073741824,2147483648"


def test_the_width_is_stated_even_when_it_is_one(monkeypatch):
    """It used to be written only above one, and the absence had a second
    meaning: a tool declaring `num_workers: int = 4` kept its own default on a
    run admitted for ONE channel. Absent now means only "no server decided"."""
    granted = concurrency.Grant("num_workers", 1, concurrency.Reservation(0, 0), "x")
    assert concurrency.child_budget({}, granted)[concurrency.CHANNELS_ENV] == "1"


def test_the_budget_is_never_negative():
    granted = concurrency.Grant("num_workers", 8, concurrency.Reservation(-5, 0), "x")
    assert concurrency.child_budget({}, granted)[concurrency.BUDGET_ENV] == "0,0"


def test_the_existing_environment_survives():
    granted = concurrency.Grant(None, 1, concurrency.Reservation(0, 0), "whatever")
    assert concurrency.child_budget({"PATH": "/usr/bin"}, granted)["PATH"] == "/usr/bin"


def test_a_tool_process_is_told_what_a_nested_level_needs_to_decide(monkeypatch):
    """A nested level works its own width out, and it can import none of this
    package to do it. What it needs -- where the learned costs are, which
    argument each tool's width is counted from, and the deployment's cap --
    goes down in the environment, from the server that knows all three."""
    monkeypatch.setattr(settings, "SADT_MAX_CHANNELS", 6)
    environment = dispatch._child_environment("job-1", "/tmp/job-1")
    assert environment[concurrency.COST_TABLE_ENV].endswith("tool_costs.json")
    assert json.loads(environment[concurrency.WIDTH_AXIS_ENV]) == concurrency.width_axes()
    assert environment[concurrency.MAX_CHANNELS_ENV] == "6"


def test_the_cap_is_sent_rather_than_inherited(monkeypatch):
    """pydantic reads `server/.env` without putting what it finds into
    `os.environ`, so a deployment that capped its channels there would have had
    the cap silently not apply at depth."""
    monkeypatch.delenv(concurrency.MAX_CHANNELS_ENV, raising=False)
    monkeypatch.setattr(settings, "SADT_MAX_CHANNELS", 3)
    assert dispatch._child_environment("j", "/tmp/j")[concurrency.MAX_CHANNELS_ENV] == "3"


# ----------------------------------------------------------------------
# The tool side: what the runner does with it
# ----------------------------------------------------------------------

def _run_declaring(*names):
    def run(**kwargs):
        return None
    import inspect
    parameters = [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=1)
                  for name in names]
    run.__signature__ = inspect.Signature(parameters)
    return run


def test_the_runner_fills_in_what_the_server_granted(monkeypatch):
    from execution import runner

    monkeypatch.setenv(runner.CHANNELS_ENV, "6")
    params = {}
    runner._grant_channels(_run_declaring("num_workers"), params)
    assert params["num_workers"] == 6


def test_the_runner_leaves_a_caller_s_own_number_alone(monkeypatch):
    from execution import runner

    monkeypatch.setenv(runner.CHANNELS_ENV, "6")
    params = {"num_workers": 2}
    runner._grant_channels(_run_declaring("num_workers"), params)
    assert params["num_workers"] == 2


def test_a_tool_outside_a_server_stays_serial(monkeypatch):
    """`scripts/run_tool.py` runs a tool with no server around it. It must not
    find a channel count it cannot account for."""
    from execution import runner

    monkeypatch.delenv(runner.CHANNELS_ENV, raising=False)
    params = {}
    runner._grant_channels(_run_declaring("num_workers"), params)
    assert params == {}


def test_a_tool_that_declares_neither_is_not_given_one(monkeypatch):
    from execution import runner

    monkeypatch.setenv(runner.CHANNELS_ENV, "6")
    params = {}
    runner._grant_channels(_run_declaring("scans", "output_dir"), params)
    assert params == {}


@pytest.mark.parametrize("junk", ["", "lots", "-3", "0"])
def test_an_unusable_grant_leaves_the_tool_serial(monkeypatch, junk):
    from execution import runner

    monkeypatch.setenv(runner.CHANNELS_ENV, junk)
    monkeypatch.delenv(runner.CHANNEL_BUDGET_ENV, raising=False)
    params = {}
    runner._grant_channels(_run_declaring("num_workers"), params)
    assert params.get("num_workers", 1) == 1


# ----------------------------------------------------------------------
# A nested level decides its OWN width, from the room it was handed
# ----------------------------------------------------------------------
#
# A top-level run is told its width, because admission reserved against that
# number and the process may not revise it. A nested run never saw admission,
# so it is told only the ROOM -- and works the width out from what IT was
# measured to cost, which is what makes a tool's width its own property rather
# than its caller's.

def _nested(monkeypatch, tmp_path, costs_table=None, axes=None, budget="0,0",
            cap=None):
    """A runner with no server-granted width, holding `budget` bytes."""
    from execution import runner

    monkeypatch.delenv(runner.CHANNELS_ENV, raising=False)
    monkeypatch.setenv(runner.CHANNEL_BUDGET_ENV, budget)
    if costs_table is None:
        monkeypatch.delenv(runner.COST_TABLE_ENV, raising=False)
    else:
        path = tmp_path / "tool_costs.json"
        path.write_text(json.dumps(costs_table), encoding="utf-8")
        monkeypatch.setenv(runner.COST_TABLE_ENV, str(path))
    if axes is None:
        monkeypatch.delenv(runner.WIDTH_AXIS_ENV, raising=False)
    else:
        monkeypatch.setenv(runner.WIDTH_AXIS_ENV, json.dumps(axes))
    if cap is None:
        monkeypatch.delenv(runner.MAX_CHANNELS_ENV, raising=False)
    else:
        monkeypatch.setenv(runner.MAX_CHANNELS_ENV, str(cap))
    return runner


@pytest.fixture(autouse=True)
def _forget_the_width_between_tests(monkeypatch):
    """`_WIDTH_GRANTED` is a module global because one runner process IS one
    job. A test process is many, so it is put back between them."""
    from execution import runner

    monkeypatch.setattr(runner, "_WIDTH_GRANTED", 1)
    monkeypatch.setattr(runner, "_WIDTH_ASKED", None)


def test_a_nested_level_sizes_itself_from_its_own_cost(monkeypatch, tmp_path):
    """Eight GiB of room and a channel measured at one GiB is eight channels.
    No core count is consulted anywhere on this path -- that is the defect."""
    runner = _nested(monkeypatch, tmp_path,
                     costs_table={"Leaf": {"vram_bytes": 1 << 30, "ram_bytes": 0}},
                     budget="{},0".format(8 << 30))
    params = {}
    runner._grant_channels(_run_declaring("num_workers"), params, "Leaf")
    assert params["num_workers"] == 8


def test_a_nested_level_pays_its_tool_s_fixed_cost_once(monkeypatch, tmp_path):
    """The runner's own half of `Cost.channels_within`, which it cannot import
    -- it runs inside the tool's virtualenv. A leaf that costs 4 GiB for its
    first channel and 1 GiB for each one after fits seven channels in 10 GiB,
    not the two a division by the one-channel figure allowed."""
    runner = _nested(monkeypatch, tmp_path,
                     costs_table={"Leaf": {"vram_bytes": 4 << 30,
                                           "vram_marginal_bytes": 1 << 30,
                                           "widths": 2}},
                     budget="{},0".format(10 << 30))
    params = {}
    runner._grant_channels(_run_declaring("num_workers"), params, "Leaf")
    assert params["num_workers"] == 7


def test_a_nested_level_reads_an_older_table_as_it_always_did(monkeypatch,
                                                              tmp_path):
    """No marginal key is a table written before the intercept existed, and
    the purely proportional reading is exactly the model it was written
    under. Eight GiB of room over a 1 GiB channel is eight, as before."""
    runner = _nested(monkeypatch, tmp_path,
                     costs_table={"Leaf": {"vram_bytes": 1 << 30}},
                     budget="{},0".format(8 << 30))
    params = {}
    runner._grant_channels(_run_declaring("num_workers"), params, "Leaf")
    assert params["num_workers"] == 8


def test_a_nested_level_with_room_for_less_than_one_channel_gets_one(
        monkeypatch, tmp_path):
    """The floor survives the intercept. A tool whose first channel costs more
    than the room it holds still runs -- alone, and at one."""
    runner = _nested(monkeypatch, tmp_path,
                     costs_table={"Leaf": {"vram_bytes": 8 << 30,
                                           "vram_marginal_bytes": 1 << 30,
                                           "widths": 3}},
                     budget="{},0".format(2 << 30))
    assert runner._channels_for("Leaf", {}) == 1


def test_a_nested_level_is_bounded_by_its_own_request(monkeypatch, tmp_path):
    """ALI under ASO can afford more channels than it has landmarks. Giving it
    two workers for one landmark spends a process launch to do one's work."""
    runner = _nested(monkeypatch, tmp_path,
                     costs_table={"ALI_CBCT": {"vram_bytes": 1 << 30}},
                     axes={"ALI_CBCT": "landmarks"},
                     budget="{},0".format(32 << 30))
    params = {"landmarks": {"Ba": True, "S": True, "N": True, "RPo": False}}
    runner._grant_channels(_run_declaring("num_workers"), params, "ALI_CBCT")
    assert params["num_workers"] == 3


def test_a_nested_level_nothing_has_measured_runs_at_one(monkeypatch, tmp_path):
    """The same rule admission applies at the top: an unknown cost is reserved
    as if it were everything, so it gets one channel and spends the lot.

    Read off `_channels_for` rather than off the params, because a one is
    decided and then deliberately not written -- see `_grant_channels`."""
    runner = _nested(monkeypatch, tmp_path, costs_table={},
                     budget="{},0".format(64 << 30))
    assert runner._channels_for("Unseen", {}) == 1
    params = {}
    runner._grant_channels(_run_declaring("num_workers"), params, "Unseen")
    assert params == {}, "a one leaves the tool's own default alone"


def test_a_nested_level_with_no_room_left_still_gets_one(monkeypatch, tmp_path):
    """The floor of one, at depth. A chain that has divided its way down to
    nothing must still run rather than stop."""
    runner = _nested(monkeypatch, tmp_path,
                     costs_table={"Leaf": {"vram_bytes": 1 << 30}}, budget="0,0")
    assert runner._channels_for("Leaf", {}) == 1


def test_a_width_of_one_is_never_written_over_a_tool_s_own_default(monkeypatch,
                                                                   tmp_path):
    """CLIC's `batch_size = 4` is four slices in one forward pass and
    Crown_Seg's `num_workers = 2` is two meshes. Writing a one over either
    would quarter one tool and halve the other on every run the machine
    admitted at a single channel."""
    runner = _nested(monkeypatch, tmp_path,
                     costs_table={"CLIC": {"vram_bytes": 8 << 30}},
                     budget="{},0".format(8 << 30))
    params = {}
    runner._grant_channels(_run_declaring("batch_size"), params, "CLIC")
    assert params == {}


def test_the_server_s_own_width_is_not_revised_by_the_budget(monkeypatch, tmp_path):
    """A top-level run holds a reservation taken against the number admission
    chose. The process may not recompute it, however much room it looks like
    it has."""
    runner = _nested(monkeypatch, tmp_path,
                     costs_table={"Leaf": {"vram_bytes": 1 << 30}},
                     budget="{},0".format(64 << 30))
    monkeypatch.setenv(runner.CHANNELS_ENV, "2")
    params = {}
    runner._grant_channels(_run_declaring("num_workers"), params, "Leaf")
    assert params["num_workers"] == 2


def test_a_deployment_cap_binds_a_nested_level_too(monkeypatch, tmp_path):
    runner = _nested(monkeypatch, tmp_path,
                     costs_table={"Leaf": {"vram_bytes": 1 << 30}},
                     budget="{},0".format(64 << 30), cap=4)
    params = {}
    runner._grant_channels(_run_declaring("num_workers"), params, "Leaf")
    assert params["num_workers"] == 4


@pytest.mark.parametrize("junk", ["", "lots", "1", "x,y"])
def test_an_unreadable_budget_leaves_a_nested_level_serial(monkeypatch, tmp_path, junk):
    """Nothing here may guess. A budget that cannot be read is no budget."""
    runner = _nested(monkeypatch, tmp_path,
                     costs_table={"Leaf": {"vram_bytes": 1 << 30}}, budget=junk)
    params = {}
    runner._grant_channels(_run_declaring("num_workers"), params, "Leaf")
    assert params.get("num_workers", 1) == 1


def test_a_level_hands_its_child_one_of_its_own_channels(monkeypatch, tmp_path):
    """What it holds, divided by the channels it is RUNNING AT -- because each
    of those may be inside a `sup.run` at the same instant."""
    runner = _nested(monkeypatch, tmp_path, budget="{},{}".format(12 << 30, 24 << 30))
    runner._record_width(4)
    assert runner._child_channel_budget() == (3 << 30, 6 << 30)


def test_a_level_that_opens_nothing_hands_its_child_everything(monkeypatch, tmp_path):
    """It is blocked inside `sup.run` while the child works, so there is
    nothing to divide."""
    runner = _nested(monkeypatch, tmp_path, budget="{},0".format(12 << 30))
    assert runner._child_channel_budget() == (12 << 30, 0)


def test_the_room_that_arrives_is_divided_once_and_not_twice(monkeypatch, tmp_path):
    """The defect this replaced, pinned end to end across the two modules.

    `granted()` used to divide the reservation by the grant before writing it
    into the environment, and `_child_channel_budget` divided the result by the
    grant again -- so a tool measured at 3 GiB a channel and admitted at four
    handed its child 0.75 GiB where both sides' comments said 3. Nothing caught
    it because the supervisor tests build the variable by hand."""
    _measured(monkeypatch, "Probe", vram=3 << 30)
    environment = concurrency.child_budget({}, concurrency.granted(_Tool(), 4))

    runner = _nested(monkeypatch, tmp_path,
                     budget=environment[concurrency.BUDGET_ENV])
    runner._record_width(4)
    assert runner._child_channel_budget() == (3 << 30, 0)


def test_a_tool_outside_a_server_hands_its_child_no_budget(monkeypatch, tmp_path):
    from execution import runner

    monkeypatch.delenv(runner.CHANNEL_BUDGET_ENV, raising=False)
    assert runner._child_channel_budget() is None


# ----------------------------------------------------------------------
# What a run costs is a function of how widely it was let spread
# ----------------------------------------------------------------------
#
# This is the whole correction. The width used to be chosen AFTER admission,
# so a tool measured at one channel was admitted on one channel's bytes and
# then opened eight -- and the cost table learned that only from the run that
# had already taken them. Now the width is part of what is reserved.

def test_a_measured_cost_scales_with_the_channels_asked_for():
    from execution.costs import Cost

    one = Cost(vram_bytes=4 << 30, ram_bytes=8 << 30, samples=3, channels=1)
    assert one.at(4).vram_bytes == 16 << 30
    assert one.at(4).ram_bytes == 32 << 30


def test_a_tool_seen_at_ONE_width_is_not_divided_by_it(tmp_path, monkeypatch):
    """A run at four channels says the four TOGETHER cost 8 GiB. It does not
    say whether that is four times two or once eight, and dividing decides it
    the unsafe way: the next run of this tool at one channel would then be
    reserved 2 GiB for what may well need all eight. The whole peak stands
    until a second width separates the intercept from the slope."""
    from execution import costs

    monkeypatch.setattr(settings, "SCHEMA_CACHE_DIR", str(tmp_path))
    costs.record("X", 8 << 30, 8 << 30, channels=4)
    measured = costs.cost_of("X")
    assert measured.vram_bytes == 8 << 30
    assert measured.at(4).vram_bytes == 8 << 30, "the width it was seen at fits"
    assert measured.at(8).vram_bytes == 16 << 30, "beyond it, nothing is known"


def test_a_run_at_one_channel_is_stored_as_it_was():
    from execution.costs import Cost

    measured = Cost(vram_bytes=1234, ram_bytes=5678, samples=1)
    assert measured.at(1) is measured


def test_two_widths_of_the_same_tool_are_not_a_varying_memory(tmp_path, monkeypatch):
    """THE reason to divide at record time. Storing totals made one run at one
    channel and one at four look like a fourfold spread -- and `input_dependent`
    would have reported a tool whose memory depends on the REQUEST, when all
    that differed was how widely the server let it spread."""
    from execution import costs

    monkeypatch.setattr(settings, "SCHEMA_CACHE_DIR", str(tmp_path))
    costs.record("X", 1 << 30, 1 << 30, channels=1)
    costs.record("X", 4 << 30, 4 << 30, channels=4)
    measured = costs.cost_of("X")
    assert measured.vram_spread == 1.0
    assert not measured.input_dependent


def test_a_tool_whose_channels_really_do_differ_is_still_caught(tmp_path, monkeypatch):
    """The signal must survive the fix. Batch_Dental_Seg's bundles measured
    x8.1 apart on this deployment at one channel each, and that is a property
    of the request, not of the width."""
    from execution import costs

    monkeypatch.setattr(settings, "SCHEMA_CACHE_DIR", str(tmp_path))
    costs.record("X", 1 << 30, 1 << 30, channels=1)
    costs.record("X", 8 << 30, 1 << 30, channels=1)
    assert costs.cost_of("X").input_dependent


def test_the_demand_reserves_what_that_width_costs():
    from execution.costs import Cost

    allocation = resources.Allocation(
        cpus=8.0, ram_bytes=16 << 30, vram_bytes=8 << 30,
        cpus_per_job=2, ram_per_job=4 << 30, vram_per_job=2 << 30,
        expected_clients=4,
    )
    cost = Cost(vram_bytes=1 << 30, ram_bytes=1 << 30, samples=3, channels=1)
    one = admission.demand_for("X", allocation, cost, channels=1)
    four = admission.demand_for("X", allocation, cost, channels=4)
    assert four.vram_bytes == one.vram_bytes * 4
    assert four.ram_bytes == one.ram_bytes * 4


def test_a_cpu_tool_still_reserves_no_vram_however_wide():
    """A run that never touches the card does not compete for it, at any
    width -- which is what lets a tabular prediction run beside a
    segmentation."""
    from execution.costs import Cost

    allocation = resources.Allocation(
        cpus=8.0, ram_bytes=16 << 30, vram_bytes=8 << 30,
        cpus_per_job=2, ram_per_job=4 << 30, vram_per_job=2 << 30,
        expected_clients=4,
    )
    cost = Cost(vram_bytes=1 << 30, ram_bytes=1 << 30, samples=3, channels=1)
    assert admission.demand_for("X", allocation, cost,
                                uses_gpu=False, channels=8).vram_bytes == 0


# ----------------------------------------------------------------------
# A busy machine narrows a run instead of making it wait
# ----------------------------------------------------------------------

def _allocation(cpus=8.0, ram=16 * 1024 ** 3, vram=8 * 1024 ** 3, per_job=2):
    return resources.Allocation(
        cpus=cpus, ram_bytes=ram, vram_bytes=vram,
        cpus_per_job=per_job, ram_per_job=ram // 4, vram_per_job=vram // 4,
        expected_clients=4,
    )


def _demand(vram, ram=1 << 20, cpus=1):
    return admission.Demand(cpus=cpus, ram_bytes=ram, vram_bytes=vram, measured=True)


def test_the_widest_candidate_that_fits_is_the_one_taken():
    budget = admission.Budget(_allocation(vram=8 << 30), free_vram=lambda: 1 << 60)
    candidates = [(8, _demand(32 << 30)), (4, _demand(16 << 30)), (1, _demand(4 << 30))]
    with budget.reserve(candidates) as (_cores, channels):
        assert channels == 1, "the two wider ones do not fit this budget"


def test_a_run_that_fits_wide_is_not_narrowed():
    budget = admission.Budget(_allocation(vram=8 << 30), free_vram=lambda: 1 << 60)
    candidates = [(8, _demand(1 << 30)), (1, _demand(1 << 28))]
    with budget.reserve(candidates) as (_cores, channels):
        assert channels == 8


def test_the_narrowest_candidate_is_always_reachable():
    """It is the one-channel demand, so the list cannot run out and this cannot
    refuse a run that one channel would have fitted."""
    budget = admission.Budget(_allocation(vram=1 << 30), free_vram=lambda: 1 << 60)
    with budget.reserve([(8, _demand(64 << 30)), (1, _demand(1 << 20))]) as (_c, channels):
        assert channels == 1


def test_a_bare_demand_is_one_candidate_at_one_channel():
    """Every caller that predates channels keeps working, unchanged."""
    budget = admission.Budget(_allocation(), free_vram=lambda: 1 << 60)
    with budget.reserve(_demand(1 << 20)) as (_cores, channels):
        assert channels == 1


def test_what_was_reserved_is_released(monkeypatch):
    """The narrowed demand, not the widest offered -- releasing the wrong one
    would leak the difference on every run."""
    budget = admission.Budget(_allocation(vram=8 << 30), free_vram=lambda: 1 << 60)
    with budget.reserve([(8, _demand(32 << 30)), (1, _demand(4 << 30))]):
        assert budget.snapshot()["vram_held"] == 4 << 30
    assert budget.snapshot()["vram_held"] == 0


# ----------------------------------------------------------------------
# Every size, not a ladder somebody wrote by hand
# ----------------------------------------------------------------------

def _shapes(tool, params=None, per_job=10, min_cpus=2, monkeypatch=None, affordable=8):
    from execution import dispatch

    monkeypatch.setattr(resources, "allocation", lambda: _allocation(per_job=per_job))
    monkeypatch.setattr(settings, "SADT_MIN_CPUS_PER_JOB", min_cpus)
    # What the budget affords at this tool's measured cost, given rather than
    # computed: these are tests about the ladder's SHAPE.
    monkeypatch.setattr(concurrency, "_affordable",
                        lambda name, budget=None: affordable)
    return dispatch._shapes(tool, params or {})


def test_a_run_that_fits_at_three_is_admitted_at_three(monkeypatch):
    """Halving was arbitrary. A run that fits at three channels should be
    admitted at three, not dropped to two because three was not on the list."""
    monkeypatch.setattr(settings, "SADT_MAX_CHANNELS", 8)
    shapes = _shapes(_Tool(), monkeypatch=monkeypatch)
    assert (3, 10) in shapes
    assert (5, 10) in shapes


def test_every_width_asks_for_the_same_cores(monkeypatch):
    """The cores do NOT follow the channels, and that is the point. Tying them
    made the CPU bound a tool whose bottleneck is not the CPU: ALI_CBCT
    measures 28.1s on seven cores and 29.4s on forty-two, and the coupling
    still capped its channels at `cpus_per_job / 2`."""
    monkeypatch.setattr(settings, "SADT_MAX_CHANNELS", 8)
    shapes = _shapes(_Tool(), per_job=10, monkeypatch=monkeypatch)
    assert {cores for channels, cores in shapes if channels > 1} == {10}


def test_a_gpu_bound_tool_reaches_the_cap_on_a_small_core_share(monkeypatch):
    """Four cores per job used to mean two channels, whatever the card had
    free. The cap is what bounds it now, and memory is what refuses it."""
    monkeypatch.setattr(settings, "SADT_MAX_CHANNELS", 8)
    shapes = _shapes(_Tool(), per_job=4, monkeypatch=monkeypatch)
    assert max(channels for channels, _cores in shapes) == 8


def test_a_tool_with_no_channel_argument_only_narrows_cores(monkeypatch):
    """It is the only axis it has, and it is the axis that narrows safely."""
    shapes = _shapes(_Tool(arguments=()), monkeypatch=monkeypatch)
    assert {channels for channels, _cores in shapes} == {1}


def test_the_narrowest_shape_is_the_floor(monkeypatch):
    """The list must end somewhere reachable, or a busy machine refuses a run
    it could have started."""
    shapes = _shapes(_Tool(), per_job=10, min_cpus=3, monkeypatch=monkeypatch)
    assert shapes[-1] == (1, 3)


def test_a_floor_above_the_share_does_not_invert_the_list(monkeypatch):
    """A deployment can set the floor higher than the per-job share -- a small
    machine, or a large expected client count. It must clamp, not produce an
    empty or backwards list."""
    shapes = _shapes(_Tool(arguments=()), per_job=2, min_cpus=8,
                     monkeypatch=monkeypatch)
    assert shapes == [(1, 2)]


# ----------------------------------------------------------------------
# The budget is built from what is running AND what is queued
# ----------------------------------------------------------------------
#
# Admission used to be greedy: the run at the head took the widest shape that
# happened to fit, and everything behind it waited. On a long queue that is a
# machine running one wide job where six narrow ones could all have started.
# There is no "queue too long" threshold anywhere -- the queue's length IS the
# divisor.

def test_a_run_alone_takes_the_widest_that_fits():
    budget = admission.Budget(_allocation(vram=8 << 30), free_vram=lambda: 1 << 60)
    candidates = [(8, _demand(1 << 30)), (1, _demand(1 << 28))]
    with budget.reserve(candidates) as (_cores, channels):
        assert channels == 8, "nothing is waiting, so nothing is being left room for"


def test_a_head_of_queue_leaves_room_for_what_is_behind_it():
    """Four waiting means the head may take a fifth, not everything that fits."""
    budget = admission.Budget(_allocation(vram=10 << 30), free_vram=lambda: 1 << 60)
    # Occupy the queue behind us without admitting anything: tickets ahead in
    # `_queue` are what `waiting` counts.
    budget._queue.extend(object() for _ in range(4))
    wide, narrow = _demand(8 << 30), _demand(1 << 30)
    assert not budget._would_fit(wide, last_resort=False, among=5)
    assert budget._would_fit(narrow, last_resort=False, among=5)
    fitted = budget._widest_that_fits([(8, wide), (1, narrow)], waiting=4)
    assert fitted[0] == 1, "the head took its share, not the machine"


def test_greedy_is_the_fallback_when_nothing_is_fair():
    """A run whose narrowest shape still does not fit a fifth of the machine
    must not be refused -- it falls back to what actually fits."""
    budget = admission.Budget(_allocation(vram=4 << 30), free_vram=lambda: 1 << 60)
    budget._queue.extend(object() for _ in range(9))
    only = _demand(3 << 30)
    fitted = budget._widest_that_fits([(1, only)], waiting=9)
    assert fitted is not None
    assert fitted[0] == 1


def test_a_long_queue_can_never_make_a_run_unrunnable():
    """The idle machine's escape survives on the narrowest shape, on the greedy
    pass. Without it a queue of forty would refuse a tool this machine can in
    fact run."""
    budget = admission.Budget(_allocation(vram=1 << 30), free_vram=lambda: 1 << 60)
    budget._queue.extend(object() for _ in range(40))
    huge = _demand(64 << 30)
    assert budget._widest_that_fits([(1, huge)], waiting=40) is not None


def test_the_fair_pass_divides_every_axis():
    """Cores and host memory as well as the card: a CPU cohort queues on cores,
    and leaving room only on the card would not leave any."""
    budget = admission.Budget(_allocation(cpus=10.0, vram=1 << 40),
                              free_vram=lambda: 1 << 60)
    budget._queue.extend(object() for _ in range(4))
    assert not budget._would_fit(_demand(0, cpus=4), last_resort=False, among=5)
    assert budget._would_fit(_demand(0, cpus=2), last_resort=False, among=5)


# ----------------------------------------------------------------------
# What a run actually opened, not what it was granted
# ----------------------------------------------------------------------
#
# The server grants N channels; a tool opens `min(N, items)`. Recording the
# grant as the measurement made the table believe a channel was cheaper than
# it is -- observed live on this deployment, AMASSS at 2.41 GiB going in as
# 0.48 GiB per channel because one scan was given five channels.
#
# The fallback is ONE, never the grant: a tool that reported nothing has not
# told us it spread, and over-reserving is slow where under-reserving is an
# out-of-memory in somebody's cohort.

def _progress(tmp_path, monkeypatch, records, depth="0"):
    import json as _json
    from execution import runner

    path = tmp_path / "events.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_json.dumps(record) + "\n")
    monkeypatch.setenv(runner.PROGRESS_FILE_ENV, str(path))
    monkeypatch.setenv(runner.SUPERVISOR_DEPTH_ENV, depth)
    return runner


def test_the_width_a_tool_reported_is_what_counts(tmp_path, monkeypatch):
    runner = _progress(tmp_path, monkeypatch, [
        {"fraction": 0.2, "message": "scan 1 of 3", "depth": 0, "width": 3},
        {"fraction": 0.6, "message": "scan 2 of 3", "depth": 0, "width": 3},
    ])
    assert runner._width_reached() == 3


def test_the_narrowest_phase_is_what_the_peak_is_divided_by(tmp_path, monkeypatch):
    """A peak and a width are not measured at the same instant.

    AMASSS reads its cohort four at a time, runs nnUNet one structure at a
    time, then assembles four at a time -- and its VRAM peak is in the SERIAL
    middle. Taking the widest would divide an inference-sized peak by four and
    teach the table that a channel costs a quarter of what it does: the same
    defect as reading the grant, one step further in.

    The narrowest is safe in the direction that matters -- it over-reserves
    where the widest under-reserves -- and exact for a tool that reports
    `width 1` around its serial phase.
    """
    runner = _progress(tmp_path, monkeypatch, [
        {"depth": 0, "width": 4}, {"depth": 0, "width": 1}, {"depth": 0, "width": 4},
    ])
    assert runner._width_reached() == 1


def test_a_tool_that_is_wide_throughout_keeps_its_width(tmp_path, monkeypatch):
    """The narrowest rule must not punish a tool with no serial phase."""
    runner = _progress(tmp_path, monkeypatch, [
        {"depth": 0, "width": 4}, {"depth": 0, "width": 4},
    ])
    assert runner._width_reached() == 4


def test_a_record_that_said_nothing_is_ignored_not_read_as_one(tmp_path, monkeypatch):
    """A tool writes progress before it knows how wide it will be. Counting
    those as one would pin every tool at one for ever."""
    runner = _progress(tmp_path, monkeypatch, [
        {"fraction": 0.1, "message": "discovering", "depth": 0},
        {"depth": 0, "width": 4},
    ])
    assert runner._width_reached() == 4


def test_a_tool_that_reported_nothing_is_taken_as_one(tmp_path, monkeypatch):
    """NOT as what the server granted. Reading a grant as a measurement is the
    whole defect."""
    runner = _progress(tmp_path, monkeypatch, [
        {"fraction": 0.5, "message": "working", "depth": 0},
    ])
    monkeypatch.setenv(runner.CHANNELS_ENV, "8")
    assert runner._width_reached() == 1


def test_a_nested_call_s_width_belongs_to_the_nested_call(tmp_path, monkeypatch):
    """A chain writes one file. A parent reporting its own width must not
    inherit its child's, nor the other way round."""
    records = [{"depth": 0, "width": 2}, {"depth": 1, "width": 7}]
    assert _progress(tmp_path, monkeypatch, records, depth="0")._width_reached() == 2
    assert _progress(tmp_path, monkeypatch, records, depth="1")._width_reached() == 7


def test_a_half_written_line_costs_that_record_and_nothing_else(tmp_path, monkeypatch):
    """The file is appended to by several processes of a chain while this
    reads it."""
    from execution import runner as _runner

    path = tmp_path / "events.jsonl"
    path.write_text('{"depth": 0, "width": 3}\n{"depth": 0, "wid\n')
    monkeypatch.setenv(_runner.PROGRESS_FILE_ENV, str(path))
    monkeypatch.setenv(_runner.SUPERVISOR_DEPTH_ENV, "0")
    assert _runner._width_reached() == 3


def test_no_progress_file_is_one(monkeypatch):
    """`scripts/run_tool.py` runs a tool with no server around it."""
    from execution import runner

    monkeypatch.delenv(runner.PROGRESS_FILE_ENV, raising=False)
    assert runner._width_reached() == 1


@pytest.mark.parametrize("junk", ["lots", -1, 0, None, [3]])
def test_an_unusable_width_is_ignored(tmp_path, monkeypatch, junk):
    runner = _progress(tmp_path, monkeypatch, [{"depth": 0, "width": junk}])
    assert runner._width_reached() == 1


# ----------------------------------------------------------------------
# A tool that calls another tool is ONE job
# ----------------------------------------------------------------------
#
# A nested call is a subprocess of its parent and never re-enters admission.
# That is not an oversight: if it took a ticket it would queue behind its own
# parent's siblings WHILE the parent holds a slot waiting for it, and six
# chains would deadlock the server. The parent's reservation covers both,
# which is why `runner._measurements` folds `own + worst child`.
#
# What follows from "one job" is that the parent's grant has to cover both, and
# the arithmetic below is currently right for a reason worth writing down.

def test_a_child_inherits_one_channel_s_worth_of_threads():
    """The parent's OMP_NUM_THREADS is ALREADY divided by its channels, so a
    child inheriting the environment gets exactly one channel's share -- and N
    children running from N branches come to the whole grant, not N times it.

    Accidental rather than arranged, and pinned for that reason: a
    `_thread_limits` that set the TOTAL instead would break this silently, and
    the symptom would be forty-two threads per child on a six-way chain.
    """
    from execution import dispatch

    parent = dispatch._thread_limits(cpus=42, channels=6)
    per_channel = int(parent["OMP_NUM_THREADS"])
    assert per_channel == 7
    assert per_channel * 6 == 42, "six children inheriting it come to the grant"


def test_a_parent_that_opens_no_channels_hands_its_child_everything():
    """It is blocked inside `sup.run` while the child works, so there is
    nothing to divide -- and dividing anyway would leave a chain slower than
    the same tool called directly."""
    from execution import dispatch

    assert dispatch._thread_limits(cpus=42, channels=1)["OMP_NUM_THREADS"] == "42"


def test_the_room_that_reaches_the_process_is_the_run_s_own(monkeypatch):
    """A parent on six channels whose child opens eight would be forty-eight on
    a machine that admitted one job, so the room is divided before a child sees
    it -- but ONCE, in the supervisor, and not here. What reaches the process
    is what the run holds: six channels of six GiB is the thirty-six admission
    reserved, and `sup.channels()` needs exactly that to answer a width."""
    _measured(monkeypatch, "Probe", vram=6 << 30)
    granted = concurrency.granted(_Tool(), 6)
    child = concurrency.child_budget({}, granted)
    assert child[concurrency.BUDGET_ENV].split(",")[0] == str(36 << 30)


# ----------------------------------------------------------------------
# Measuring a tool whose work happens in CHILDREN
# ----------------------------------------------------------------------
#
# Both per-process figures are blind to it, and both in the direction that
# ends in an out-of-memory:
#
#   * `torch.cuda.max_memory_reserved()` counts what THIS interpreter
#     allocated. ALI_CBCT searches its landmarks in parallel processes, so the
#     runner allocates nothing on the card and reported no VRAM at all -- not a
#     low figure, an absent one.
#   * `RUSAGE_CHILDREN.ru_maxrss` is the largest single child ever seen, never
#     the sum of several alive at once: four workers holding 1.5 GiB each
#     reported 2.38 GiB against 6.9 GiB resident.

def test_the_sampler_sees_the_whole_process_group():
    """Matched by GROUP rather than by walking parent links: `dispatch` starts
    a tool with `start_new_session=True`, so a grandchild a tool shelled out to
    is in the group whether or not it is still attached to its parent."""
    from execution import runner

    sampler = runner._TreeSampler()
    group = sampler._group()
    assert os.getpid() in group


def test_it_sums_resident_memory_rather_than_taking_a_maximum():
    from execution import runner

    sampler = runner._TreeSampler()
    pids = sampler._group()
    assert len(pids) >= 1
    total = sampler._rss(pids)
    assert total > 0
    biggest = max(sampler._rss({pid}) for pid in pids)
    assert total >= biggest, "a sum, never one process's figure"


def test_a_machine_with_no_nvidia_smi_reports_no_vram(monkeypatch):
    """Instrumentation must never fail a run that otherwise succeeded."""
    from execution import runner

    sampler = runner._TreeSampler()
    monkeypatch.setattr(runner.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no such file")))
    assert sampler._vram({os.getpid()}) == 0


def test_a_sample_that_raises_never_stops_the_thread(monkeypatch):
    from execution import runner

    sampler = runner._TreeSampler()
    monkeypatch.setattr(sampler, "_group",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    sampler.start()
    sampler.stop()  # must not raise


def test_torch_s_own_counter_wins_over_the_card_s_growth(tmp_path, monkeypatch):
    """They are not two views of one number.

    torch's is exact and belongs to this process; the fallback is the card's
    growth, which counts whatever else the host allocated while this ran.
    Taking the larger let that contamination win: Batch_Dental_Seg, which
    really costs 4.77 GiB and whose nnUNet runs in the runner's own
    interpreter, went into the table at 38.40 GiB per channel -- more than the
    whole budget, so nothing could ever have been admitted beside it.
    """
    from execution import runner

    monkeypatch.setattr(runner, "_peak_vram_bytes", lambda: 5 << 30)
    monkeypatch.setattr(runner, "_child_vram_bytes", 0)
    monkeypatch.setattr(runner._sampler, "peak_vram", 38 << 30)
    monkeypatch.setattr(runner._sampler, "peak_rss", 0)
    monkeypatch.setattr(runner._sampler, "stop", lambda: None)
    monkeypatch.setenv(runner.PROGRESS_FILE_ENV, "")
    assert runner._measurements()["peak_vram_bytes"] == 5 << 30


def test_the_fallback_is_used_where_torch_saw_nothing(tmp_path, monkeypatch):
    """Its one case: a tool whose GPU work happens in children, where torch in
    this interpreter has nothing to report -- not a low figure, an absent one.

    Torch must still be LOADED here, which it is: ALI_CBCT builds its
    Environment through monai before any worker starts.
    """
    from execution import runner

    monkeypatch.setattr(runner, "_touched_torch", lambda: True)
    monkeypatch.setattr(runner, "_peak_vram_bytes", lambda: 0)
    monkeypatch.setattr(runner, "_child_vram_bytes", 0)
    monkeypatch.setattr(runner._sampler, "peak_vram", 3 << 30)
    monkeypatch.setattr(runner._sampler, "peak_rss", 0)
    monkeypatch.setattr(runner._sampler, "stop", lambda: None)
    monkeypatch.setenv(runner.PROGRESS_FILE_ENV, "")
    assert runner._measurements()["peak_vram_bytes"] == 3 << 30


# ----------------------------------------------------------------------
# When the card cannot be read per process
# ----------------------------------------------------------------------
#
# This deployment's driver answers `--query-compute-apps` with nothing at all,
# on the host as well as in the container. Without a fallback, a tool whose GPU
# work happens in children reports NO vram -- not a low figure, an absent one --
# and the server stacks runs onto a card it believes is empty.

def test_the_card_s_growth_is_used_when_the_listing_is_empty(monkeypatch):
    from execution import runner

    sampler = runner._TreeSampler()
    monkeypatch.setattr(sampler, "_vram", lambda pids: 0)
    monkeypatch.setattr(sampler, "_rss", lambda pids: 1)
    monkeypatch.setattr(sampler, "_group", lambda: {os.getpid()})
    sampler._card_baseline = 2 << 30
    monkeypatch.setattr(sampler, "_card_used", lambda: 5 << 30)
    sampler.sample()
    assert sampler.peak_vram == 3 << 30, "what the card grew by, not what it holds"


def test_the_per_process_figure_wins_once_it_has_answered(monkeypatch):
    """It is the better measurement: the fallback counts another job's
    allocation as this one's."""
    from execution import runner

    sampler = runner._TreeSampler()
    monkeypatch.setattr(sampler, "_rss", lambda pids: 1)
    monkeypatch.setattr(sampler, "_group", lambda: {os.getpid()})
    monkeypatch.setattr(sampler, "_vram", lambda pids: 1 << 30)
    sampler._card_baseline = 0
    monkeypatch.setattr(sampler, "_card_used", lambda: 40 << 30)
    sampler.sample()
    sampler.sample()
    assert sampler.peak_vram == 1 << 30


def test_another_job_freeing_memory_never_reads_as_costing_nothing(monkeypatch):
    """A negative delta is the one outcome this exists to prevent."""
    from execution import runner

    sampler = runner._TreeSampler()
    sampler._card_baseline = 10 << 30
    monkeypatch.setattr(sampler, "_card_used", lambda: 2 << 30)
    assert sampler._fallback_vram() == 0


def test_no_card_at_all_is_not_an_error(monkeypatch):
    from execution import runner

    sampler = runner._TreeSampler()
    monkeypatch.setattr(sampler, "_card_used", lambda: None)
    assert sampler._fallback_vram() == 0


# ----------------------------------------------------------------------
# Never wider than the request has things to do
# ----------------------------------------------------------------------
#
# A tool cannot usefully open more channels than it has items. Giving ALI two
# workers for one landmark spends a process launch -- ~3 s measured -- to do
# one landmark's work, and then the run is MEASURED at a width it never
# reached, so the cost table learns a per-channel figure that is too low. That
# is the unsafe direction.
#
# Counted, never understood: ticked options, list entries, files in a folder.

def _bounded(monkeypatch, params, declared=None, axis=None, cap=8, affordable=None):
    """`ceiling` with the two things it consults stubbed.

    `affordable` is what the budget could hold at this tool's measured cost --
    normally computed, here given, so a test about COUNTING is not also a test
    about the cost table. Defaults to the cap so the item count is what moves.
    """
    monkeypatch.setattr(settings, "SADT_MAX_CHANNELS", cap)
    monkeypatch.setattr(concurrency, "_affordable",
                        lambda name, budget=None: cap if affordable is None else affordable)
    # `resolved`, not `for_tool`: the axis a width is counted from is a
    # DERIVED field, and reading the raw declaration instead is the defect the
    # section below exists for. A helper stubbing the wrong one of the two
    # would go on passing while nothing counted anything.
    monkeypatch.setattr(
        concurrency.deployment_config, "resolved",
        lambda name: type("D", (), {"width_from": declared,
                                    "batch": {"axis": axis}})(),
    )
    return concurrency.ceiling(_Tool(), params)


def test_a_request_for_one_landmark_gets_one_channel(monkeypatch):
    ticked = {"Ba": True, "S": False, "N": False}
    assert _bounded(monkeypatch, {"landmarks": ticked}, declared="landmarks") == 1


def test_seven_landmarks_and_a_cap_of_eight_gets_seven(monkeypatch):
    ticked = {name: True for name in "abcdefg"}
    ticked["unticked"] = False
    assert _bounded(monkeypatch, {"landmarks": ticked}, declared="landmarks") == 7


def test_the_cap_still_wins_over_a_large_request(monkeypatch):
    ticked = {str(index): True for index in range(119)}
    assert _bounded(monkeypatch, {"landmarks": ticked}, declared="landmarks", cap=8) == 8


def test_the_batch_axis_is_the_default_so_most_tools_declare_nothing(monkeypatch):
    """AMASSS's channels are scans and its axis is already `scans`."""
    assert _bounded(monkeypatch, {"scans": ["a", "b", "c"]}, axis="scans") == 3


def test_a_declared_argument_overrides_the_axis(monkeypatch):
    """ALI_CBCT's axis is `input` -- a folder of scans -- but its channels are
    the landmarks searched within one scan."""
    params = {"input": ["one scan"], "landmarks": {"a": True, "b": True}}
    assert _bounded(monkeypatch, params, declared="landmarks", axis="input") == 2


def test_a_tool_can_say_its_width_cannot_be_counted(monkeypatch):
    """CLIC's channel is a slice of a volume, and how many slices there are is
    not known until the volume has been read -- after admission has decided."""
    params = {"scans": ["a", "b"]}
    assert _bounded(monkeypatch, params, declared=False, axis="scans", cap=4) == 4


def test_a_folder_is_counted_by_its_files(monkeypatch, tmp_path):
    for name in ("a.nii.gz", "b.nii.gz", "c.nii.gz"):
        (tmp_path / name).write_bytes(b"x")
    assert _bounded(monkeypatch, {"scans": str(tmp_path)}, axis="scans") == 3


def test_a_single_file_is_one(monkeypatch, tmp_path):
    scan = tmp_path / "only.nii.gz"
    scan.write_bytes(b"x")
    assert _bounded(monkeypatch, {"scans": str(scan)}, axis="scans") == 1


def test_an_uncountable_value_leaves_the_ceiling_alone(monkeypatch):
    """A hosted name the server never staged, an int, a missing argument: none
    of them is a count, and guessing one would be worse than not bounding."""
    for value in (None, 7, "", "a-name-nobody-staged"):
        assert _bounded(monkeypatch, {"scans": value}, axis="scans", cap=4) == 4


def test_nothing_ticked_does_not_collapse_the_width(monkeypatch):
    """An omitted multichoice arrives as every option present and False, and
    for ALI that means "the regions decide" -- not "no work"."""
    ticked = {"Ba": False, "S": False}
    assert _bounded(monkeypatch, {"landmarks": ticked}, declared="landmarks", cap=4) == 4


# ----------------------------------------------------------------------
# ... and the axis has to REACH the place the width is decided
# ----------------------------------------------------------------------
#
# The tests above stub the lookup, which is right for tests about COUNTING and
# is exactly how the defect below survived being written down: `derive` filled
# `ToolDeployment.batch` with the resolved plan, handed it to the `Tool`, and
# nothing wrote it back into the config `width_axis` reads. So the batch-axis
# half of the rule fired for NO tool -- only ALI_CBCT's explicit `landmarks`
# and CLIC's explicit `false` ever bounded anything -- while every stubbed test
# of it passed.
#
# A width nothing bounds is still reserved: a reservation is
# `per-channel cost x channels`. Measured on this machine, on AMASSS, whose
# channels are its scans:
#
#     AMASSS RAM per channel   19.1 GiB
#     host RAM budget          93.8 GiB
#
#     at 1 channel:  reserves 19.1 GiB  ->  4 runs in parallel
#     at 3 channels: reserves 57.3 GiB  ->  1 run
#
# Granted 3 channels for a request holding ONE scan: six concurrent AMASSS
# 143 s -> 275 s, ten concurrent 158 s -> 425 s, while a single run stayed at
# 79 s -- which is why nothing that timed one run at a time ever saw it.
#
# So these tests stub NOTHING between `conventions.derive` and `width_axis`.

def _served(monkeypatch, make_tool_folder, name, arguments, declared=None):
    """One tool loaded the way the registry loads it, into a config of its own.

    `load_tool` is the real one, so the axis is derived by the real
    conventions and published by the real `record_resolved`. What this
    monkeypatches is only WHICH config the run path reads -- the module global
    belongs to the server's own tools, and a test must not add to it.
    """
    from registry import schema_tool
    from registry.deployment import DeploymentConfig

    config = DeploymentConfig({name: declared} if declared is not None else {})
    schema_tool.load_tool(make_tool_folder(name, arguments=arguments), config)
    monkeypatch.setattr(concurrency, "deployment_config", config)
    return config


_COHORT = {
    "scans": {"type": "path", "required": True},
    "num_workers": {"type": "int", "required": False, "default": 1},
}


def test_the_derived_axis_is_readable_from_where_the_width_is_decided(
        monkeypatch, make_tool_folder):
    """The test that would have caught it, and the only one that could have.

    A tool declaring no `width_from` at all -- which is every tool but two --
    must still have an axis where the width is decided, not only on the `Tool`
    object `derive` happened to hand it to.
    """
    _served(monkeypatch, make_tool_folder, "Cohort_Tool", _COHORT)

    assert concurrency.width_axis("Cohort_Tool") == "scans"


def test_a_request_holding_one_item_is_granted_one_channel(
        monkeypatch, make_tool_folder, tmp_path):
    """AMASSS asking for one scan: one channel, and 19.1 GiB reserved rather
    than 57.3 GiB for two channels it cannot use."""
    _served(monkeypatch, make_tool_folder, "Cohort_Tool", _COHORT)
    monkeypatch.setattr(settings, "SADT_MAX_CHANNELS", 0)
    monkeypatch.setattr(concurrency, "_affordable", lambda name, budget=None: 3)
    one_scan = tmp_path / "patient.nii.gz"
    one_scan.write_bytes(b"x")

    assert concurrency.ceiling(_Tool(name="Cohort_Tool"),
                               {"scans": str(one_scan)}) == 1


def test_a_request_holding_six_items_is_granted_at_most_six(
        monkeypatch, make_tool_folder, tmp_path):
    """And no more. The cohort is what there is to do; affordability is only
    the other bound."""
    _served(monkeypatch, make_tool_folder, "Cohort_Tool", _COHORT)
    monkeypatch.setattr(settings, "SADT_MAX_CHANNELS", 0)
    monkeypatch.setattr(concurrency, "_affordable", lambda name, budget=None: 12)
    cohort = tmp_path / "cohort"
    cohort.mkdir()
    for index in range(6):
        (cohort / f"patient{index}.nii.gz").write_bytes(b"x")

    assert concurrency.ceiling(_Tool(name="Cohort_Tool"),
                               {"scans": str(cohort)}) == 6


def test_width_from_false_still_leaves_a_width_unbounded(
        monkeypatch, make_tool_folder, tmp_path):
    """CLIC's channel is a slice of a volume, and ALI_IOS's is a tooth of a
    mesh: neither is countable from the request, and counting the cohort
    instead would bound a single-item run at one channel. `false` is how a
    deployment says so, and it has to keep meaning it now that silence no
    longer does."""
    from registry.deployment import ToolDeployment

    _served(monkeypatch, make_tool_folder, "Slice_Tool", _COHORT,
            declared=ToolDeployment(width_from=False))
    monkeypatch.setattr(settings, "SADT_MAX_CHANNELS", 0)
    monkeypatch.setattr(concurrency, "_affordable", lambda name, budget=None: 4)
    one_scan = tmp_path / "patient.nii.gz"
    one_scan.write_bytes(b"x")

    assert concurrency.width_axis("Slice_Tool") is None
    assert concurrency.ceiling(_Tool(name="Slice_Tool"),
                               {"scans": str(one_scan)}) == 4


def test_a_declared_width_from_still_wins_over_the_derived_axis(
        monkeypatch, make_tool_folder):
    """ALI_CBCT: its axis is `input`, a folder of scans, but its channels are
    the landmarks searched within one of them. The derived axis must not take
    the declaration's place now that it is finally readable."""
    from registry.deployment import ToolDeployment

    _served(monkeypatch, make_tool_folder, "Landmark_Tool",
            {"input": {"type": "path", "required": True},
             "landmarks": {"type": "list[str]", "required": False},
             "num_workers": {"type": "int", "required": False, "default": 1}},
            declared=ToolDeployment(width_from="landmarks"))
    monkeypatch.setattr(settings, "SADT_MAX_CHANNELS", 0)
    monkeypatch.setattr(concurrency, "_affordable", lambda name, budget=None: 8)

    assert concurrency.width_axis("Landmark_Tool") == "landmarks"
    assert concurrency.ceiling(_Tool(name="Landmark_Tool"),
                               {"input": "a folder of scans",
                                "landmarks": ["Ba", "S", "N"]}) == 3


def test_a_tool_needs_no_entry_in_deployment_toml_to_have_an_axis_at_depth(
        monkeypatch, make_tool_folder, tmp_path):
    """Root and nested must bound a width by the same rule, or a chain's levels
    disagree about the same tool's request -- which is worse than either
    answer. The nested level cannot ask the config at all: it reads the dict
    the server sends down, so that dict has to carry the DERIVED axes too."""
    _served(monkeypatch, make_tool_folder, "Cohort_Tool", _COHORT)
    cohort = tmp_path / "cohort"
    cohort.mkdir()
    for name in ("a.nii.gz", "b.nii.gz"):
        (cohort / name).write_bytes(b"x")
    params = {"scans": str(cohort)}

    axes = concurrency.width_axes()
    assert axes["Cohort_Tool"] == "scans", "not in deployment.toml, and still has an axis"

    runner = _nested(monkeypatch, tmp_path,
                     costs_table={"Cohort_Tool": {"vram_bytes": 1 << 30}},
                     budget="{},0".format(8 << 30),
                     axes=axes)
    granted = dict(params)
    runner._grant_channels(_run_declaring("num_workers"), granted, "Cohort_Tool")

    assert granted["num_workers"] == concurrency.items_in("Cohort_Tool", params) == 2


def test_a_tool_that_never_imported_torch_gets_no_card_figure(monkeypatch):
    """The fallback is the CARD's growth, so it counts whatever else the host
    allocated while this ran. A tabular tool would otherwise pick up a
    neighbour's allocation, and the server would reserve card memory for
    something that never touches the card.

    It also made the supervisor test that pins this FLAKY -- passing on an idle
    machine, failing whenever anything else was running, which is the worst
    shape a test can have.
    """
    from execution import runner

    monkeypatch.setattr(runner, "_touched_torch", lambda: False)
    monkeypatch.setattr(runner, "_peak_vram_bytes", lambda: None)
    monkeypatch.setattr(runner, "_child_vram_bytes", 0)
    monkeypatch.setattr(runner._sampler, "peak_vram", 9 << 30)
    monkeypatch.setattr(runner._sampler, "peak_rss", 1 << 20)
    monkeypatch.setattr(runner._sampler, "stop", lambda: None)
    monkeypatch.setenv(runner.PROGRESS_FILE_ENV, "")
    assert "peak_vram_bytes" not in runner._measurements()


def test_a_tool_whose_gpu_work_is_in_children_still_gets_one(monkeypatch):
    """ALI_CBCT builds its Environment through monai before any worker starts,
    so torch IS loaded here even though nothing was allocated here."""
    from execution import runner

    monkeypatch.setattr(runner, "_touched_torch", lambda: True)
    monkeypatch.setattr(runner, "_peak_vram_bytes", lambda: 0)
    monkeypatch.setattr(runner, "_child_vram_bytes", 0)
    monkeypatch.setattr(runner._sampler, "peak_vram", 2 << 30)
    monkeypatch.setattr(runner._sampler, "peak_rss", 1 << 20)
    monkeypatch.setattr(runner._sampler, "stop", lambda: None)
    monkeypatch.setenv(runner.PROGRESS_FILE_ENV, "")
    assert runner._measurements()["peak_vram_bytes"] == 2 << 30


# ----------------------------------------------------------------------
# What a channel costs in CORES, measured like everything else
# ----------------------------------------------------------------------
#
# Memory scaled with the channel count and cores did not, so ALI_CBCT at eight
# channels filled 51.7 of 56 cores while admission believed it had let in a
# ten-core job. Nothing had ever measured what a channel costs in CPU.

def test_a_measured_core_cost_scales_with_the_width():
    from execution.costs import Cost

    one = Cost(vram_bytes=1 << 30, ram_bytes=1 << 30, samples=3, cpu_cores=8.6)
    assert round(one.at(4).cpu_cores, 1) == 34.4


def test_the_reservation_holds_the_cores_the_width_needs():
    from execution.costs import Cost

    allocation = resources.Allocation(
        cpus=42.0, ram_bytes=93 << 30, vram_bytes=33 << 30,
        cpus_per_job=10, ram_per_job=23 << 30, vram_per_job=8 << 30,
        expected_clients=4,
    )
    cost = Cost(vram_bytes=1 << 28, ram_bytes=1 << 28, samples=5, cpu_cores=8.6)
    wide = admission.demand_for("ALI_CBCT", allocation, cost, channels=4)
    assert wide.cpus > 34, "four channels of a ten-core tool is not a ten-core job"


def test_a_tool_whose_channels_cost_no_cpu_is_unaffected():
    """CLIC batches slices inside one forward pass: more channels is more work
    for the card and none for the host. Measuring rather than assuming is what
    keeps it out of ALI's bill."""
    from execution.costs import Cost

    allocation = resources.Allocation(
        cpus=42.0, ram_bytes=93 << 30, vram_bytes=33 << 30,
        cpus_per_job=10, ram_per_job=23 << 30, vram_per_job=8 << 30,
        expected_clients=4,
    )
    cost = Cost(vram_bytes=1 << 28, ram_bytes=1 << 28, samples=5, cpu_cores=0.0)
    assert admission.demand_for("X", allocation, cost, channels=8).cpus == 10


def test_a_measured_cost_never_falls_below_the_declared_share():
    """A job is told it may open that many threads whatever it was seen to
    use, so that is what it holds."""
    from execution.costs import Cost

    allocation = resources.Allocation(
        cpus=42.0, ram_bytes=93 << 30, vram_bytes=33 << 30,
        cpus_per_job=10, ram_per_job=23 << 30, vram_per_job=8 << 30,
        expected_clients=4,
    )
    cost = Cost(vram_bytes=1 << 28, ram_bytes=1 << 28, samples=5, cpu_cores=0.4)
    assert admission.demand_for("X", allocation, cost, channels=1).cpus == 10


def test_the_sampler_reports_the_cores_a_group_kept_busy():
    """A rate, so it needs two readings: the first only marks the baseline."""
    import time as _time

    from execution import runner

    sampler = runner._TreeSampler()
    sampler.sample()
    assert sampler.peak_cores == 0.0, "one reading is not a rate"
    end = _time.time() + 1.1
    while _time.time() < end:
        pass
    sampler.sample()
    assert 0.5 < sampler.peak_cores < 2.0, sampler.peak_cores


def test_an_interval_too_short_to_divide_is_ignored(monkeypatch):
    """Dividing a tick count by a few milliseconds turns rounding into forty
    cores on a machine that has four.

    The clock is driven rather than raced. Three bare `sample()` calls only
    stay under the threshold while the machine is idle enough to make them --
    so the test failed under exactly the load the guard exists for, and passed
    when nothing was running. Pinning the interval tests the guard instead of
    the machine.
    """
    from execution import runner

    ticking = iter([0.0, 0.01, 0.02, 0.03])
    monkeypatch.setattr(runner, "monotonic", lambda: next(ticking))
    # And the ticks, or the assertion is vacuous: over a few real milliseconds
    # a process burns zero whole ticks, so the figure would be 0.0 whether the
    # guard fired or not. One tick per 10 ms interval is a single core's worth
    # of work reported as 100 -- the rounding this exists to refuse.
    burned = iter([0, 1, 2, 3])
    monkeypatch.setattr(runner._TreeSampler, "_cpu_ticks",
                        lambda self, pids: next(burned))

    sampler = runner._TreeSampler()
    sampler.sample()
    sampler.sample()
    sampler.sample()
    assert sampler.peak_cores == 0.0
