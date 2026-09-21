"""How many channels a tool may open inside one run, and who says so.

`admission.py` decides how many RUNS share the machine. This decides what one
run may do with the share it was given -- whether a tool that can process its
scans in parallel is allowed to, and how widely.

**Why this is not the tool's own decision.** A tool knows it *can* parallelise;
it cannot know whether it is alone on the machine or one of six, and nothing in
its signature could tell it. A default written into the tool is therefore either
too small on an idle server or too large on a busy one, permanently. So the
tool declares the capability -- an argument the server recognises -- and the
server fills in the number, exactly as it already fills in `device` and
`output_dir`.

**And why HOW MANY ITEMS there are is not the server's.** The half of the
question the tool owns is the other one: a width is only ever useful up to the
number of things there are to do, and for several tools that number is not in
the request at all. `ALI_IOS`'s unit is a tooth, read out of a mesh's label
array during the run; `CLIC`'s is a slice of a volume, known once the volume is
read; `AutoCrop3D`, `AutoMatrix` and `GreedyReg` take two required folders that
have to stay paired, so splitting either one alone re-pairs patients and
neither is an axis. Counting from outside answers those with a guess or with
nothing -- `width_from = false`, which means "no bound but affordability", and
an unbounded width is reserved for all the same.

So the two halves are asked of the side that knows:

    width = sup.channels(wanted)

`wanted` is the tool's own count, at run time, in whatever unit the tool
actually loops over. The answer is what the SHARE this run holds can pay for,
floored at one. `runner._Supervisor.channels` is where it is computed, beside
`sup.run` and `sup.progress`, and for the same reason: `runner.py` ships with
the server and is injected by path, so there is one implementation at one
version and nothing is copied into a tool's virtualenv. A tool that calls it
adds no file and no import.

**The share IS the reservation, which is what makes this safe to answer at run
time.** Admission cannot wait for the tool -- it reserves before the process
starts -- so what it reserves is a share of the machine, chosen fair-then-greedy
by `admission._widest_that_fits` exactly as it always has been, and handed down
in bytes (`BUDGET_ENV`). `sup.channels(wanted)` then answers
`min(wanted, share / measured per-channel cost)`, floor one. Nothing can be
over-committed, because every channel the answer permits was paid for before
the run began.

**The cost of that, stated plainly: a run that opens one channel still holds
its share.** A tool granted room for four and finding one item to do keeps the
other three channels' worth reserved for its whole life. That is why
`items_in()` below has NOT gone away with the guessing it looks like: where the
server can count honestly, it still must, because the reservation is taken
before anyone can ask. What `sup.channels()` removes is the case where it
cannot count -- there the width was previously unbounded and is now the tool's
own number. Right-sizing the reservation afterwards, when the tool answers,
needs a live tool-to-admission channel that does not exist here yet.

**Why it is tied to the supervisor.** A nested call is a subprocess of its
parent and never re-enters admission (see `runner.Supervisor`), so nothing
between the two levels divides anything. Left alone, ASO opening six channels
and each calling ALI_CBCT, which opens eight, is forty-eight channels on a
machine that admitted one job. The grant is therefore a BUDGET that is divided
on the way down. A chain three deep cannot multiply, because each level hands
on less than it received and the floor is one.

**What is handed down is BYTES, not a channel count**, and that correction is
recent. The budget used to be the parent's CORE grant divided by its channel
grant -- `max(1, cpus // channels)` -- which was wrong in three ways at once,
all three measured on this machine on 2026-09-18:

    tool=ASO granted 28 cores (declared share 7; the machine was idle)
    tool=ASO granted 5 channel(s) on num_workers      <- five granted
    tool=ASO peak_vram=5.73 GiB ... over 1 channel(s) <- one opened

and on the next run 31 cores and 4 channels, so its child inherited `31 // 4`.

1. **The CPU decided a GPU tool's width.** ALI_CBCT's bottleneck is the card --
   28.1 s on seven cores and 29.4 s on forty-two -- and cores were deliberately
   decoupled from its channel count everywhere else; a chain put them back in
   charge through the back door.
2. **An unused grant penalised the child**: ASO was given five channels, opened
   one, and its child's budget was divided by five anyway. The more the parent
   was given, the less its child could take.
3. **The answer depended on the CALLER.** The same ALI_CBCT asking for the same
   seven landmarks got 7 channels standalone, 5 under an ASO holding 28 cores
   and 7 under an ASO holding 31. A tool had no say in its own width.

So a level hands its children the ROOM it holds, divided by the channels it may
open, and each level then sizes itself exactly as a top-level run does: its own
measured cost against the budget it holds, bounded by the items its own request
carries. Cores leave the equation entirely.

**Why dividing by the GRANT stopped penalising anyone.** The obvious objection
to dividing by what the parent MAY open rather than by what it DID is defect 2
above -- and it does not survive the change of numerator. A reservation is
`per-channel cost x channels`, so dividing it by those same channels gives one
channel's worth whatever the count was: ASO at five channels reserves
5 x 5.73 GiB and hands down 5.73; ASO at one reserves 5.73 and hands down 5.73.
The grant inflates the numerator by exactly what it inflates the denominator
by, and cancels. Dividing by what the parent actually opened is the thing that
cannot be done: it is not knowable before the child starts -- the parent may
open another channel a microsecond later -- and a budget that is only right
when the parent behaves is not a budget. The one case where the cancellation
does not hold is a tool nothing has measured, whose reservation is the whole
machine however many channels it asked for; that tool is granted one channel by
`_affordable`, so the division is by one and nothing is lost.

**The margin is applied once, at the top.** What travels here is the measured
reservation BEFORE `admission.SAFETY_MARGIN`, because admission has already
applied that margin over the whole job -- a chain is one job -- and applying it
again per level would compound it a second time at every hop.

**Three shapes of tool, and only the deployment can tell them apart.** A tool
whose loop is CPU-bound with independent items scales until the cores run out.
A tool already saturating the card gains nothing from a second channel and
loses to contention. A tool holding a model per channel multiplies its memory
by the channel count -- and the cost table only learns that AFTER a run has
taken it, which is one out-of-memory too late. What another channel buys is a
property of the tool -- and it is MEASURED rather than declared, because a
number written in a file is wrong the day a model changes and nobody finds
out, while a measurement corrects itself on the next run.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import resources
from config import settings
from execution import costs
from registry.deployment import deployment_config

logger = logging.getLogger("inference_server")

# The argument names a tool uses to say "I can do several of these at once".
# Both are already in `conventions.TECHNICAL`, so a tool that declares one is
# hidden from a clinician with no further configuration -- which is right: how
# many channels to open is not a clinical decision.
#
# Ordered: a tool declaring both is granted on the first, because `num_workers`
# means items of the tool's own loop while `batch_size` means items in one
# forward pass, and the loop is the outer of the two.
CHANNEL_ARGUMENTS = ("num_workers", "batch_size")

# Deliberately no "cores per channel" constant any more.
#
# Tying the two made the CPU bound a tool whose bottleneck is not the CPU:
# ALI_CBCT measured 28.1s on seven cores and 29.4s on forty-two -- it does not
# scale with cores at all -- and yet the coupling capped its channels at
# `cpus_per_job / 2`. A channel's cost is memory, which is reserved; what it
# needs of the CPU is a share of the threads the run was already granted,
# which `dispatch._thread_limits` divides. Nothing else needed the constant.

# What this run was granted, read by `runner.py` and passed to `run()` when the
# tool declares the argument and the caller named no value.
#
# Carried in the ENVIRONMENT rather than in job.json, and the split is the
# point: job.json is what the CALLER asked for, the environment is what this
# machine granted. They are decided at different moments -- the job file is
# written before admission, and how many channels are free is not knowable
# until after it -- and keeping them apart is what lets a retry re-grant
# without rewriting the caller's request.
#
# Present on a run the SERVER admitted, whatever the number, and ABSENT on a
# nested one -- which is how a nested level knows nobody decided its width and
# that it may decide its own. `runner.Supervisor` removes it deliberately.
CHANNELS_ENV = "SADT_CHANNELS"

# The ROOM this run holds, in bytes, as `"<vram>,<ram>"` -- the reservation
# admission actually took, UNDIVIDED. Read by `sup.channels()`, which answers a
# width out of it, and by `runner.Supervisor`, which divides it by the channels
# this level opened and hands the result to each child. Written even for a
# single channel, so a nested call always finds a number rather than inheriting
# whatever the container was started with.
#
# **It used to carry one channel's worth and the division happened twice.**
# `granted()` divided the reservation by the grant before writing it here, and
# `runner._child_channel_budget` divided the result by the grant again -- so a
# tool measured at 3 GiB a channel and admitted at four handed its child
# 12 / 4 / 4 = 0.75 GiB, a quarter of the one channel's worth the comments on
# both sides said it was handing over. Nothing caught it because
# `test_supervisor.py` builds this variable by hand rather than through
# `child_budget`, and the hand-built value was the intended one. Dividing once,
# where the divisor is known -- in the supervisor, by what `sup.channels()`
# answered -- is what removes the second division and the disagreement with it.
#
# Two integers separated by a comma rather than JSON: `runner.py` parses this
# with nothing but `str.split`, and it runs on every interpreter from 3.9 to
# 3.13 inside a tool's own virtualenv.
BUDGET_ENV = "SADT_CHANNEL_BUDGET"

# Where the learned cost table is, so a nested level can read what IT was
# measured to cost. `runner.py` ships with the server and is injected by path,
# so the two are the same version by construction -- but it runs inside a
# tool's venv and can import none of this package, hence a path and a JSON
# read rather than `costs.cost_of`.
COST_TABLE_ENV = "SADT_COST_TABLE"

# `{tool: argument}` -- which argument a tool's width is counted from, for the
# levels that cannot ask `deployment_config`. Same source as `width_axis`, so
# both sides of a chain bound a width by the same rule.
WIDTH_AXIS_ENV = "SADT_WIDTH_AXIS"

# The deployment's backstop, under the name the setting already has. Sent
# explicitly rather than left to be inherited: pydantic reads `server/.env`
# without putting what it finds into `os.environ`.
MAX_CHANNELS_ENV = "SADT_MAX_CHANNELS"


@dataclass(frozen=True)
class Reservation:
    """Room, in bytes, on the two resources a channel is actually paid for in.

    **Cores are deliberately absent**, and their absence is the whole point of
    the type. A channel costs MEMORY, which is reserved; what it needs of the
    CPU is a share of the threads the run already holds, which
    `dispatch._thread_limits` divides. Handing a core count down a chain is
    what let the CPU decide a GPU tool's width.
    """

    vram_bytes: int = 0
    ram_bytes: int = 0

    def per(self, channels: int) -> "Reservation":
        """This room divided between `channels` -- what ONE of them may spend.

        Floored at one channel, which is what stops a deep chain from dividing
        its way to nothing.
        """
        share = max(1, int(channels))
        return Reservation(max(0, int(self.vram_bytes)) // share,
                           max(0, int(self.ram_bytes)) // share)

    def encode(self) -> str:
        """The wire form for `BUDGET_ENV`. See there for why it is not JSON."""
        return "{},{}".format(max(0, int(self.vram_bytes)),
                              max(0, int(self.ram_bytes)))


@dataclass(frozen=True)
class Grant:
    """What one run may open, and why it came to that."""

    argument: Optional[str]     # the argument to fill in, or None to fill none
    channels: int
    # The room THIS run holds -- the reservation admission took, undivided.
    # Named `room` rather than `budget` because it changed meaning: it used to
    # be one channel's worth, already divided, which is what let the runner
    # divide it a second time. What a nested call inherits is this divided by
    # the channels this level opened, and that division now happens once, in
    # `runner.Supervisor`, where the number it divides by is known.
    room: Reservation
    reason: str

    @property
    def applies(self) -> bool:
        return bool(self.argument) and self.channels > 1


def channel_argument(tool) -> Optional[str]:
    """The argument this tool uses to say how many channels it may open.

    None for a tool that declares neither, which is most of them -- and for
    those this whole module is inert. A capability nobody declared is not a
    capability the server may assume: opening several channels in a tool whose
    loop shares a temporary file corrupts its own outputs, silently.
    """
    arguments = getattr(tool, "arguments", None) or {}
    for name in CHANNEL_ARGUMENTS:
        if name in arguments:
            return name
    return None


def items_in(tool_name: str, params: dict) -> Optional[int]:
    """How many things this request actually holds, or None if unknowable.

    **Kept, deliberately, now that a tool can answer the same question better.**
    `sup.channels()` reads the tool's own count at run time and is right where
    this is a guess -- but it is answered AFTER admission has reserved, and a
    reservation that is too wide is paid for whether or not the tool spreads
    into it (see this module's header, and the AMASSS figures below). So this
    stays as the bound on the LADDER, which is the only place a request's size
    can still narrow what gets reserved. It is removable the day a run can give
    room back mid-flight, and not before.

    A tool can never usefully open more channels than it has items, and giving
    it more spends a process launch -- ~3 s for ALI_CBCT -- to do one item's
    work. It also measures the run at a width it never reached, so the cost
    table learns a per-channel figure that is too low, which is the unsafe
    direction.

    **Counted, never understood.** The server does not know that a landmark is
    a point on a skull or that a scan is a patient: it counts the ticked
    options of a multichoice, the entries of a list, or the files in a folder.
    That is what keeps this general across a catalogue it imports nothing from.

    Which argument to count comes from `width_axis`, which is also what the
    nested levels are handed -- so a width is bounded by the same rule wherever
    it is decided.
    """
    argument = width_axis(tool_name)
    if not argument:
        return None
    return _count(params.get(argument))


def width_axis(tool_name: str) -> Optional[str]:
    """The argument this tool's width is counted from, or None for no counting.

    From `deployment.toml`: `width_from`, or the BATCH AXIS when none is
    declared -- the argument a cohort is already split on, which for most tools
    is the same thing. `width_from = false` says this tool's width cannot be
    counted from the request at all, which is true of CLIC: its channel is a
    slice of a volume, and how many slices there are is not known until the
    volume has been read.

    **The batch-axis half of that rule fired for nobody until 2026-09-18**, and
    what it cost is why it is written down here. `conventions.derive` filled
    `ToolDeployment.batch` with the resolved plan, handed the result to the
    `Tool`, and nothing wrote it back into the config this reads -- so the axis
    was whatever `width_from` named and nothing else, ALI_CBCT's `landmarks`
    and CLIC's `false` across the whole catalogue, and every other tool's width
    was bounded only by what it could afford. A width nothing bounds is
    reserved for all the same: a reservation is `per-channel cost x channels`,
    so the fiction is paid for in memory nobody uses. Measured on this machine,
    on AMASSS, whose channels ARE its scans:

        AMASSS RAM per channel   19.1 GiB
        host RAM budget          93.8 GiB

        at 1 channel:  reserves 19.1 GiB  ->  4 runs in parallel
        at 3 channels: reserves 57.3 GiB  ->  1 run

    It was granted 3 channels for a request holding ONE scan, could use two of
    them for nothing, and tripled its own reservation with the third. Against
    the same arms on an idle machine: six concurrent AMASSS 143 s -> 275 s, ten
    concurrent 158 s -> 425 s. A single run was 79 s either way, which is why
    the coverage pass never saw it -- the whole cost is in what a run stops
    OTHER runs from doing. Repairing the cost table made it worse rather than
    better: truer costs mean more affordable channels, and every extra channel
    was 19 GiB held for work that did not exist.

    `deployment_config.resolved` is the fix, and the seam is deliberate: this
    module reads one config object, exactly as it did, and the registry
    publishes into it what `derive` settled on (see
    `DeploymentConfig.record_resolved`). The alternative -- reaching the
    derived deployment through the `Tool` that carries it -- would put the tool
    registry on the import path of a module admission calls on every run, and
    `width_axes()` below has no tool object to reach through in any case.

    None still means "nothing bounds this but affordability", and that stays
    the right answer for a tool whose channel is not an item of the request:
    CLIC's slices, ALI_IOS's teeth. What changed is that it stopped meaning
    that for everyone else.
    """
    declared = deployment_config.resolved(tool_name)
    argument = getattr(declared, "width_from", None)
    if argument is False:
        return None
    if not argument:
        plan = getattr(declared, "batch", None) or {}
        argument = plan.get("axis")
    return argument or None


def width_axes() -> dict:
    """The axis of every tool that has one, for the levels that cannot ask.

    Sent to a tool process in `WIDTH_AXIS_ENV` and read back by `runner.py` for
    its own name. Built from `width_axis`, so a nested level bounds its width
    by the same rule as a top-level one -- which is the only reason this is a
    dict of every tool rather than one entry: a chain's levels disagreeing
    about the axis would be worse than either answer.

    Over `known_tools`, not `configured_tools`: the whole point of the derived
    axis is that a tool needs no entry in `deployment.toml` to have one, so
    iterating what that file happens to name would rebuild the defect at depth
    the same day it was fixed at the top.
    """
    axes = {}
    for name in getattr(deployment_config, "known_tools", ()) or ():
        axis = width_axis(name)
        if axis:
            axes[name] = axis
    return axes


def _count(value) -> Optional[int]:
    """The items in one argument's value, or None where there is no counting it."""
    if value is None:
        return None
    # A multichoice arrives as the COMPLETE {option: ticked} mapping -- every
    # declared option, so an unticked one is present and False rather than
    # missing. What was asked for is the ticked ones.
    if isinstance(value, dict):
        return sum(1 for ticked in value.values() if ticked) or None
    if isinstance(value, (list, tuple, set)):
        return len(value) or None
    text = str(getattr(value, "path", value) or "")
    if not text:
        return None
    if os.path.isdir(text):
        found = 0
        for _root, _dirs, files in os.walk(text):
            found += len(files)
        return found or None
    if os.path.exists(text):
        return 1
    return None


def ceiling(tool, params: dict, budget: Optional[Reservation] = None) -> int:
    """The MOST channels this run may be offered, before the machine has a say.

    Static on purpose. It is read before admission, to build the degrees of
    parallelism admission will choose between, so it cannot depend on what is
    free -- that is admission's half of the decision, and asking twice would
    make the reservation disagree with the grant.

    A tool declaring no channel argument returns 1, and the whole mechanism is
    inert for it: a capability nobody declared is not one the server may
    assume. Opening several channels inside a loop that shares a temporary file
    corrupts its own outputs, silently, and the declaration is the tool saying
    it does not.

    `budget` is the room this level holds. None means the machine's own, which
    is the top-level case; a nested level passes what its parent handed down.
    **Everything below this line is the same arithmetic either way** -- that is
    the correction: a level sizes itself from its own measured cost against the
    room it holds, never from a number computed out of somebody else's cores.
    """
    argument = channel_argument(tool)
    if argument is None:
        return 1
    name = getattr(tool, "name", "?")
    if params.get(argument) not in (None, ""):
        # A caller may ask for fewer and is left alone; it may not ask for more
        # than the machine's caps, which exist to protect it.
        return _clamped(_as_int(params.get(argument), 1), name)
    # The widest that could POSSIBLY fit, computed from what this tool was
    # measured to cost rather than from a number anyone chose. That is both the
    # honest bound and a finite one: the shapes ladder walks down from here, so
    # an unbounded ceiling would be an unbounded loop.
    #
    # The server's cap is the only thing above it, NOT the per-job share. The
    # share is a declaration made before anyone connected, and pre-limiting the
    # ladder by it meant an idle machine offered no more channels than a full
    # one -- the cores flexed and the channels did not, which is the
    # inconsistency this whole design exists to remove. Offering the cap and
    # letting `_would_fit` refuse what does not fit is the same bargain the
    # cores already get.
    widest = _clamped(_affordable(name, budget), name)
    # And never wider than the request has things to do.
    items = items_in(name, params)
    return max(1, min(widest, items)) if items else widest


def machine_budget() -> Reservation:
    """The whole of what this server may spend -- the top level's budget."""
    allocation = resources.allocation()
    return Reservation(int(allocation.vram_bytes or 0), int(allocation.ram_bytes or 0))


def reservation(tool_name: str, channels: int = 1) -> Reservation:
    """The room this run holds while it runs, BEFORE admission's margin.

    The dual of `_affordable`: that one divides a budget by a cost to get a
    width, this one multiplies a cost by a width to get the budget back. They
    have to agree, or a level would hand down room its parent never held.

    A tool nothing has measured holds EVERYTHING, exactly as
    `admission.demand_for` reserves everything for it -- so a chain whose root
    is unmeasured hands its children the whole machine, which is right: nothing
    else is running.

    Unmargined on purpose. `admission.SAFETY_MARGIN` is applied once, by
    admission, over the whole job -- and a chain is one job -- so carrying a
    margined figure down would compound it again at every hop.
    """
    measured = costs.cost_of(tool_name)
    if measured is None or (not measured.vram_bytes and not measured.ram_bytes):
        return machine_budget()
    scaled = measured.at(max(1, int(channels)))
    return Reservation(int(scaled.vram_bytes), int(scaled.ram_bytes))


def _affordable(tool_name: str, budget: Optional[Reservation] = None) -> int:
    """How many channels of this tool the budget could hold at all.

    `cost.at(n)` is `fixed + marginal x n`, and `Cost.channels_within` inverts
    exactly that -- the room left over once the fixed part is paid, divided by
    what a further channel adds, on whichever of the card and host memory runs
    out first. It used to be a plain division by the one-channel cost, which
    charged every channel for the CUDA context and the resident model again:
    on the measured tools that is three quarters of the figure, so a tool got
    roughly a quarter of the channels its room could actually hold.

    It is what makes "the card decides" literal -- a tool whose channel is
    cheap gets many, one whose channel is dear gets few, and neither number
    was written down by anybody.

    `budget` is the room the level holds, defaulting to the machine's. A nested
    level passes what it inherited, and the arithmetic is otherwise identical:
    that identity is what makes a tool's width its own property rather than its
    caller's.

    A tool nothing has measured gets ONE. It reserves the whole budget and runs
    alone by the rule that governs every unmeasured tool, so a wider ladder for
    it would be a list of shapes that cannot fit.
    """
    measured = costs.cost_of(tool_name)
    if measured is None:
        return 1
    room_for = budget if budget is not None else machine_budget()
    return measured.channels_within(vram_bytes=room_for.vram_bytes,
                                    ram_bytes=room_for.ram_bytes)


def granted(tool, channels: int) -> Grant:
    """What admission settled on, as the thing that fills in the argument.

    Admission already chose `channels` -- it reserved the bytes that count
    costs -- so this only names the argument to put it in, works out what a
    nested call may spend, and says why, for the log. It cannot revise the
    number: the reservation is already held against it.

    **No cores anywhere.** They used to decide what a child inherited, which
    made the CPU the bound on a tool whose bottleneck is the card; see this
    module's header for the measurements that removed them.
    """
    argument = channel_argument(tool)
    channels = max(1, int(channels))
    name = getattr(tool, "name", "?")
    # UNDIVIDED, and that is the change. What travels is the room this run
    # holds; the division into a child's share happens once, in the supervisor,
    # by the width `sup.channels()` answered. Dividing here as well made the
    # reservation and the thing handed down disagree by a factor of the grant
    # (see BUDGET_ENV), and it is also what the tool itself needs: a tool that
    # ASKS for a width is asking how much of its own reservation it may spread
    # over, and one channel's worth answers "one" whatever it was admitted at.
    room = reservation(name, channels)
    if argument is None:
        return Grant(None, 1, room, "declares no channel argument")
    return Grant(argument, channels, room,
                 "admission fitted {}; the run holds {:.2f} GiB of card "
                 "and {:.2f} GiB of host".format(channels,
                                                 room.vram_bytes / (1 << 30),
                                                 room.ram_bytes / (1 << 30)))


def _clamped(channels: int, tool_name: str) -> int:
    """`channels`, held to the server's backstop.

    There is no per-tool cap any more. `max_workers` existed to protect against
    two things and neither survived: ALI_CBCT losing landmarks at width 4,
    which is fixed at its source by scaling the per-agent budget with the
    width, and a tool that must never overlap, which cannot arise because a
    tool that declares no channel argument is never given one.

    What bounds a width now is what it costs and what was asked for -- both
    measured, neither declared -- and a number written in a file could only
    disagree with them.
    """
    declared = int(getattr(settings, "SADT_MAX_CHANNELS", 0) or 0)
    # No cap unless a deployment asks for one: what a width costs and how much
    # there is to do are both measured, and they bind long before a number
    # written here would.
    return max(1, min(int(channels), declared if declared > 0 else _NO_CAP))


# Large enough to be no bound at all, finite so the arithmetic above stays
# integer and a stray value can never make a width negative.
_NO_CAP = 1 << 30


def _as_int(value, fallback: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return fallback


def child_budget(environment: dict, granted: Grant) -> dict:
    """The environment a tool's process gets: its width, and the room it holds.

    Both written even when the grant is one channel, so a process always finds
    numbers rather than inheriting whatever the container was started with.

    **The room is this run's own, not one channel's.** `sup.channels()` divides
    it by what the tool was measured to cost per channel to answer a width, and
    `runner.Supervisor` divides it by that answer to hand a child its share --
    so the division happens once, downstream, where the divisor is known. It
    used to happen here as well, and the two compounded; see `BUDGET_ENV`.

    **`CHANNELS_ENV` is written unconditionally now**, where it used to be set
    only when the grant was wider than one -- and the change is about what its
    ABSENCE means, not about what a one does. A nested level reads "no
    `CHANNELS_ENV`" as "nobody decided my width, so I decide it myself"; a
    root run admitted at one channel must not read it that way, since its
    reservation was taken against that one. Present-and-one says what was
    decided. What the tool process then does with a one is `runner`'s own
    business, and it still writes nothing: see `_grant_channels`.
    """
    return {
        **environment,
        CHANNELS_ENV: str(max(1, int(granted.channels))),
        BUDGET_ENV: granted.room.encode(),
    }


def log_grant(tool_name: str, granted: Grant) -> None:
    """Said only when it does something, so an ordinary run stays quiet."""
    if granted.applies:
        logger.info("tool=%s granted %d channel(s) on %s (%s)",
                    tool_name, granted.channels, granted.argument, granted.reason)
