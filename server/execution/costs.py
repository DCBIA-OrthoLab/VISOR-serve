"""What a tool actually costs, learned from the runs that already happened.

**Nothing is declared here and nothing is measured by hand.** `runner.py` has
written a peak figure on every run since the subprocess path landed, and until
now `dispatch.py` only logged it -- so every measurement was written into a job
directory and deleted with it. This keeps them.

The table is a high-water mark per tool, not an average: admission has to fit
the worst run it has seen, and a mean would admit two jobs on the strength of a
small scan and meet a large one.

**But only over a window.** An all-time maximum never comes back down, so one
unusual scan raises a tool's reservation for every run that follows it, for
ever -- an 8 GiB outlier would pin a tool whose ordinary peak is 0.26 GiB and
collapse its parallelism permanently. The worst of the last COST_WINDOW runs
forgets such an outlier once the tool has behaved normally that many times.

**The window is per RESOURCE, not per run** (`_trim`). RAM and cores come back
from every run; a card reading comes back only from a run that was alone, and
the rest are refused. Counting the refusals against the same window let one
busy arm forget everything the tool knew about the card -- and a tool that
knows nothing about the card reserves the whole budget and runs ALONE. Twenty
concurrent ALI_CBCT runs did exactly that on 2026-09-21: the next arm of twenty
took 549.8 s instead of 140.0 s, one run at a time, on a card that never went
past 3.6 GiB of 33.7.

Forgetting is only safe because failing is now cheap: a run that turns out not
to fit is retried with more room (settings.MEMORY_RETRIES) rather than reported
as an error. Without that retry this window would be a way of causing the very
out-of-memory the table exists to avoid.

It is still optimistic, because VRAM depends on the input and a peak learned on
a small volume under-estimates a big one -- which is why admission applies a
margin on top, checks what the card actually has free, and treats an unknown
tool as costing everything.

A tool with no entry runs alone, so the table bootstraps itself: one run of
each tool fills it, and the benchmark harness writes the same file, so a
calibration campaign is an accelerator rather than a second mechanism.

**What a run costs is `fixed + marginal x width`, not `per channel x width`.**
A tool's first channel pays for the CUDA context and the resident model, and
every channel after it pays only for its own working set -- so the cost of a
run is an affine function of its width with an intercept, and the intercept is
most of it. `_fit` is where that is measured rather than declared, and the two
measured tables that say so are in its docstring.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Optional

from config import settings

logger = logging.getLogger("inference_server")

# What ONE channel of this tool costs, the fixed part included. Its meaning is
# unchanged -- it is what `Cost.at(1)` answers, and what it always answered
# while the model had no intercept.
VRAM_KEY = "vram_bytes"
RAM_KEY = "ram_bytes"
# What each channel AFTER the first adds. Absent from a table written before
# the intercept existed, and read back then as the whole one-channel figure,
# which is exactly the purely-proportional model that table was written under.
VRAM_MARGINAL_KEY = "vram_marginal_bytes"
RAM_MARGINAL_KEY = "ram_marginal_bytes"
# How many distinct widths the window holds. Below two, the split above is a
# fallback rather than a measurement, and `Cost.at` says so by answering
# differently -- see `_predict`.
WIDTHS_KEY = "widths"
SAMPLES_KEY = "samples"
UPDATED_KEY = "updated_at"
# The last N peaks, newest last. What admission reserves is the worst of THESE
# rather than the worst ever seen.
RECENT_KEY = "recent"
# How many channels the most recent run opened, for a human reading the file.
CHANNELS_KEY = "channels"

# Above this ratio between a tool's largest and smallest recorded run -- once
# the width each of them ran at has been accounted for -- its memory is
# reported as depending on the request rather than on the tool. 1.5 rather
# than something tighter because two runs of an identical request already
# differ by a few percent -- allocator behaviour, a different cuDNN algorithm
# chosen for the same shapes -- and calling that input-dependent would make the
# signal fire on everything and mean nothing.
SPREAD_THRESHOLD = 1.5

_lock = threading.Lock()


@dataclass(frozen=True)
class Cost:
    """The worst this tool has needed over the recorded window.

    `channels` is how many of its own items the worst run was processing at
    once, which is what makes the figure comparable: a tool measured at four
    channels and one measured at one are not the same measurement, and until
    the count was recorded beside the bytes they were stored as if they were.

    Two numbers per resource, not one: `vram_bytes` is what ONE channel costs
    and `vram_marginal` is what each channel after it adds. "The worst" is
    therefore a LINE rather than a maximum -- every run in the window carried
    to the width being asked for, worst wins -- and `_fit` is where that line
    comes from.
    """

    vram_bytes: int = 0
    ram_bytes: int = 0
    # What each channel AFTER the first adds, so that one channel costs
    # `vram_bytes` and n channels cost `vram_bytes + vram_marginal * (n - 1)`.
    # None means "nothing has measured a smaller slope than the whole thing",
    # which is filled in below as the one-channel figure -- the purely
    # proportional model, and what every caller constructing a `Cost` by hand
    # means by it.
    vram_marginal: Optional[int] = None
    ram_marginal: Optional[int] = None
    # How many distinct widths the figures were fitted over. One point cannot
    # separate an intercept from a slope, so below two the split above is a
    # fallback and `at()` answers the worse of the two readings that one point
    # admits. See `_predict`.
    widths: int = 1
    samples: int = 0
    channels: int = 1
    # Cores one channel was measured to keep busy. 0 means never measured, and
    # admission then falls back to the declared per-job share -- the behaviour
    # that existed before this was recorded at all.
    cpu_cores: float = 0.0
    # How far the window's runs disagreed: worst / smallest, per resource.
    # 1.0 means every recorded run of this tool cost the same.
    vram_spread: float = 1.0
    ram_spread: float = 1.0
    # Did ANY run in the window come back with a VRAM figure the server was
    # willing to believe? `vram_bytes == 0` has two meanings and they admit in
    # opposite directions: a tabular tool measured at zero costs the card
    # nothing and may share it with anything, while a tool whose only readings
    # were refused is simply UNKNOWN and must run alone until one of them is
    # trusted. The bytes cannot carry that difference; this flag does.
    vram_known: bool = True

    def __post_init__(self) -> None:
        # A `Cost` built by hand -- in a test, or by a caller with one figure
        # and no window behind it -- means the proportional model, which is
        # what this server did everywhere before the intercept was measured.
        if self.vram_marginal is None:
            object.__setattr__(self, "vram_marginal", self.vram_bytes)
        if self.ram_marginal is None:
            object.__setattr__(self, "ram_marginal", self.ram_bytes)

    @property
    def vram_fixed(self) -> int:
        """The part of the card that a second channel does not pay for again."""
        return max(0, int(self.vram_bytes) - int(self.vram_marginal))

    @property
    def ram_fixed(self) -> int:
        return max(0, int(self.ram_bytes) - int(self.ram_marginal))

    @property
    def split_measured(self) -> bool:
        """Was the intercept measured, or is it the fallback one point forces?"""
        return self.widths >= 2

    def at(self, channels: int) -> "Cost":
        """What this tool would cost at `channels`: `fixed + marginal x width`.

        **Not proportional, and that was a real defect rather than a
        conservative approximation.** A tool's first channel pays for the CUDA
        context and the resident model; the ones after it pay for their own
        working set only. Multiplying a one-channel figure by the width
        over-reserves a wide run -- which fails safely -- but DIVIDING a wide
        run's peak by its width to get that figure under-reserves every narrow
        run that follows, which does not. See `_fit` for the two tools this was
        measured on and for what each of those errors cost.

        With fewer than two widths behind it the split cannot have been
        measured, and `_predict` answers the worse of the two readings one
        point admits rather than picking one.

        The returned `Cost` carries a TOTAL in `vram_bytes`/`ram_bytes`, as it
        always has -- `channels` says at what width -- so it is an answer, not
        a model to ask again.
        """
        wanted = max(1, int(channels))
        if wanted == 1:
            # Already the one-channel cost, intercept included: that is what
            # `vram_bytes` means, both here and in the table on disk.
            return self
        return Cost(
            vram_bytes=_predict(self.vram_bytes, self.vram_marginal,
                                self.widths, wanted),
            ram_bytes=_predict(self.ram_bytes, self.ram_marginal,
                               self.widths, wanted),
            vram_marginal=self.vram_marginal,
            ram_marginal=self.ram_marginal,
            widths=self.widths,
            # Cores are deliberately left proportional. What the intercept
            # describes is a resident allocation -- a CUDA context and a set of
            # weights -- and those are bytes, not threads; a channel that is
            # not running occupies no core for the next one to inherit.
            cpu_cores=self.cpu_cores * wanted,
            samples=self.samples,
            channels=wanted,
            vram_spread=self.vram_spread,
            ram_spread=self.ram_spread,
            vram_known=self.vram_known,
        )

    def channels_within(self, vram_bytes: int = 0, ram_bytes: int = 0) -> int:
        """How many channels of this tool that much room could pay for.

        The inverse of `at()`, and it has to be written as its inverse rather
        than as a division: with an intercept, the first channel and the tenth
        do not cost the same, so `room // per channel` answers a question this
        model no longer asks. A tool that is three quarters fixed gets far more
        channels out of the same room than that division ever allowed.

        One is the floor. A run that does not fit at all is still admitted
        alone, by the rule that governs every demand bigger than the budget --
        refusing it here would make a tool the machine can in fact run
        permanently unrunnable.
        """
        widest = []
        for one, marginal, room in (
            (int(self.vram_bytes), int(self.vram_marginal), int(vram_bytes or 0)),
            (int(self.ram_bytes), int(self.ram_marginal), int(ram_bytes or 0)),
        ):
            if not one or not room:
                continue  # nothing measured on this resource, or no room declared
            if room < one:
                return 1  # not even one channel's worth; it runs alone
            if not marginal:
                continue  # measured to cost nothing per channel: no bound here
            if self.split_measured:
                widest.append(1 + (room - one) // marginal)
            else:
                widest.append(room // marginal)
        return max(1, min(widest)) if widest else 1

    @property
    def input_dependent(self) -> bool:
        """Does this tool's memory depend on what was ASKED, not just on the tool?

        Admission reserves ONE figure per tool. That is only sound while the
        figure is a property of the tool. A tool whose peak moves with the
        request -- a bigger volume, more structures, a heavier model bundle --
        is one whose reservation is right for the run that happened to be last
        and wrong for the next one, and the way that goes wrong is an
        out-of-memory in somebody's cohort.

        It is not a defect and nothing here refuses such a tool: the window's
        MAXIMUM is reserved, so the estimate is conservative by construction.
        It is worth SAYING, because it is the difference between a reservation
        that is exact and one that is a bet, and nobody can see that from the
        single number the table prints.

        **The width is not the request, and this must not confuse them.** Once
        the cost has an intercept, two runs of an identical request at
        different widths differ by design -- AMASSS costs 5291 MiB at one
        structure and 7587 at three -- and comparing those two peaks directly
        reported a x1.4 variation nothing in the request explains. The spread
        is therefore measured against the FITTED model rather than against the
        raw peaks: what the width accounts for is not variation, and what is
        left over is.
        """
        return max(self.vram_spread, self.ram_spread) >= SPREAD_THRESHOLD


def table_path() -> str:
    return os.path.join(settings.SCHEMA_CACHE_DIR, "tool_costs.json")


def _load() -> dict:
    try:
        with open(table_path(), encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _store(table: dict) -> None:
    """Replace the table atomically, or give up quietly.

    A cost table that cannot be written costs one thing: every tool looks
    unknown and runs alone, which is exactly today's behaviour. It must never
    cost a run.
    """
    directory = settings.SCHEMA_CACHE_DIR
    try:
        os.makedirs(directory, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(table, stream, indent=2, sort_keys=True)
        os.replace(temporary, table_path())
    except OSError as exc:
        logger.debug("could not write the tool cost table: %s", exc)


def _window(entry: dict) -> list:
    """The recorded window as `[vram, ram, channels]`, whatever shape it was.

    Two older shapes are read rather than discarded, each being the best
    estimate that exists for that tool: a table written before the window held
    one scalar pair, and a table written before channels were recorded holds
    pairs. Both are taken as having been measured at ONE channel, which is what
    they were -- nothing could open more before the argument existed.
    """
    recent = entry.get(RECENT_KEY)
    if isinstance(recent, list) and recent:
        window = []
        for item in recent:
            if not isinstance(item, list) or len(item) < 2:
                continue
            channels = int(item[2]) if len(item) > 2 else 1
            cores = float(item[3]) if len(item) > 3 else 0.0
            # A null VRAM slot stays null. It is the one run whose card reading
            # was thrown away as contended, and reading it back as a zero would
            # turn "we do not know" into "it costs nothing".
            vram = None if item[0] is None else int(item[0])
            window.append([vram, int(item[1]), max(1, channels), cores])
        if window:
            return window
    return [[int(entry.get(VRAM_KEY) or 0), int(entry.get(RAM_KEY) or 0), 1, 0.0]]


def _vram_window(window, keep: int) -> list:
    """The runs this tool's CARD figure may be fitted over.

    Not the same runs as the RAM figure, and that is the point: RAM and cores
    are per-process readings that every run comes back with, while a card
    reading is refused whenever the run shared the machine (see
    `dispatch._keep_measurements`). Fitting both over one window makes the
    refusals forget the measurements.

    Two kinds of entry are dropped here, and both are an absence written as a
    number:

    - **A null**, which is the refusal itself.
    - **A zero, once the tool has been seen to take the card at all.** A GPU
      run whose card reading came back as no growth did not discover that the
      tool stopped using the card; it failed to measure it -- the card grew by
      nothing because a neighbour freed as much as this run allocated, which is
      exactly the condition the null guard exists for and which the clamp in
      `runner._TreeSampler._fallback_vram` turns into a zero. A tool that has
      NEVER read anything but zero keeps every one of them, so a tabular tool
      goes on costing the card nothing and sharing it with anything.

    Measured on this deployment, 2026-09-21: one 20-client arm of ALI_CBCT left
    19 nulls and one zero in its 20-run window, and the zero priced the tool --
    which really costs 0.62 GiB a channel -- at nothing.
    """
    seen = [item for item in window if item[0] is not None]
    if any(item[0] for item in seen):
        seen = [item for item in seen if item[0]]
    return seen[-max(1, int(keep)):]


def _trim(window, keep: int) -> list:
    """The window to store: forgetting by RESOURCE, not by run.

    `keep` runs of history per resource, which is more than `keep` entries
    whenever some of them carry no card reading. The alternative -- one flat
    `window[-keep:]` -- lets a burst of contended runs evict everything the
    tool ever knew about the card, and a tool that knows nothing about the card
    reserves the whole budget and runs ALONE.

    **This is the defect the rule prevents, measured rather than feared.** On
    2026-09-21, twenty concurrent ALI_CBCT runs wrote twenty refusals into a
    twenty-run window. The tool became unmeasured, and the next twenty-client
    arm held 93.8 GiB of host and 33.7 GiB of card for a run costing 0.83 GiB,
    one at a time: **549.8 s against 140.0 s**, mean concurrency 1.0 against
    5.3, on a card that never went past 3.6 GiB of the 33.7 it had. The same
    arm from the same table with its card readings intact is the 140.0 s.

    **It still forgets, in the unit that makes forgetting mean something.** The
    window exists so one unusual scan does not pin a tool's reservation for
    ever, and an outlier is a MEASUREMENT: only another `keep` measurements can
    displace it, and a run that measured nothing is not one of them. Counting
    refusals towards it is what let a busy afternoon erase the table without a
    single new fact being learned.

    Bounded at `2 x keep`, because an entry is kept for at most one reason
    beyond being recent.
    """
    keep = max(1, int(keep))
    kept = set(range(max(0, len(window) - keep), len(window)))
    kept.update(index for index, item in enumerate(window)
                if item[0] is not None)
    # Oldest card readings first, so the cap falls on them rather than on the
    # recent runs every resource is fitted over.
    while len(kept) > 2 * keep:
        kept.discard(min(kept))
    return [window[index] for index in sorted(kept)]


def _predict(one: int, marginal: int, widths: int, channels: int) -> int:
    """What `channels` channels cost, given one channel's cost and the slope.

    Two readings, and which applies is whether a slope was ever measured:

    - **Two widths or more**, so the intercept is real: `fixed + marginal x n`,
      written as `one + marginal x (n - 1)` because `one` is the figure the
      table has always stored and every caller has always read.
    - **One width `w`**, so it is not: the whole peak is admissible as fixed
      AND the whole peak is admissible as `w` channels' worth of marginal, and
      with one point there is nothing to choose between them. The worse of the
      two answers -- flat below `w`, proportional above it. Guessing either one
      alone is how a reservation silently becomes too small, in one direction
      or the other.

    At `w == 1` the two coincide exactly (`one x n`), which is why nothing
    changes for a tool that has only ever run narrow -- the overwhelming
    majority of the table, and every entry written before widths existed.
    """
    wanted = max(1, int(channels))
    if int(widths) >= 2:
        return int(one) + int(marginal) * (wanted - 1)
    return max(int(one), int(marginal) * wanted)


def _fit(points) -> tuple:
    """`(one channel, each channel after it)` from runs at several widths.

    `points` is `[(what the run cost in total, how wide it ran), ...]`.

    **Measured on two tools of opposite shape, and they agree:**

        AMASSS    width 1  5291 MiB   width 2  6371   width 3  7587
                  => ~4.1 GiB fixed + ~1.15 GiB per structure  (~75% fixed)
                                                       (2026-09-18)

        ALI_IOS   width 1  1930 MiB   width 2  2516   width 4  3342
                  width 8  4886
                  => ~1.5 GiB fixed + ~422 MiB per channel     (78% fixed)
                                                       (2026-09-21)

    A 3D nnUNet and a 2D UNet over multi-view rendering, nothing in common,
    both roughly three quarters fixed -- because the fixed part is the CUDA
    context and the resident model, which every GPU tool has.

    **The slope is the secant between the narrowest and the widest width**
    seen, each taken at its worst run. The widest baseline is the one least
    disturbed by the few percent two identical runs differ by, and both tools
    above are slightly CONCAVE -- their marginal falls as they widen (ALI_IOS:
    586, 413, 386 MiB per channel across its three steps) -- so a secant over
    the whole range over-states the marginal at the top, which is the safe
    direction to extrapolate in.

    **The line is then SCALED until it covers every run recorded**, and that
    is what "the worst this tool has needed" becomes once the cost has two
    parameters. It is the same operation the proportional model performed, not
    a new one: that model was the line `n` scaled by `max(peak / width)`, the
    worst ratio any recorded run bore to it. This is the same max over the same
    ratios, against a better line. What it cannot do is sit below a run that
    already happened, at any width in the window, which is the direction the
    per-channel maximum protected and the one that ends in an out-of-memory.

    **Rejected: raising the INTERCEPT until the line covers every run.**
    Algebraically tidier -- the max becomes `marginal x n + max(peak -
    marginal x width)` -- and it fails badly on real data, because it charges
    the whole of a wide run's excess to a fixed cost that no narrow run pays.
    Measured against this deployment's own table, CLIC:

        width  1     4     9    39    45
        MiB  654  2342  5226 21350 22750   => ~0.17 GiB fixed + ~0.55 GiB/ch

    its width-39 run sits 8% above the secant, which is noise in 39 working
    sets rather than 1.7 GiB of extra context -- and the additive clamp reads
    it as exactly that, pricing ONE channel of CLIC at 2266 MiB against the
    654 MiB it was measured at. A 3.5x over-reservation on the commonest shape
    of request, to cover a discrepancy at a width 39 times wider. Scaling
    spreads the same excess over both parameters, so every width pays the 12%
    it actually disagreed by and none pays for another's.

    **Rejected: least squares.** A regression line passes THROUGH the cloud, so
    half the recorded runs sit above it and the reservation for a width would
    be smaller than a run already seen at that very width. With two points it
    is the secant anyway; with more it only adds a way to under-reserve.

    **Rejected: the steepest secant over all pairs of widths.** It dominates
    too, and it extrapolates a tool's noisiest adjacent pair for ever: on
    ALI_IOS it reserves 6032 MiB at eight channels against the 4886 measured,
    which is most of the parallelism this whole change exists to recover.

    **Rejected: a cost per width, with no model at all.** It cannot answer a
    width nobody has run, and that is precisely the question admission asks --
    it builds a ladder of candidate widths up to the cap and prices every rung.
    """
    usable = [(int(total), max(1, int(width))) for total, width in points
              if total is not None]
    if not usable:
        return 0, 0
    widths = sorted({width for _total, width in usable})
    worst = {width: max(total for total, at in usable if at == width)
             for width in widths}
    if len(widths) < 2:
        # One point cannot be split. `_predict` keeps both readings of it, and
        # the per-channel one is rounded up for the same reason as below.
        only = widths[0]
        return worst[only], -(-worst[only] // only)
    low, high = widths[0], widths[-1]
    # Never negative: a wide run that measured LESS than a narrow one says
    # nothing costs extra, not that a channel gives memory back.
    marginal = max(0.0, (worst[high] - worst[low]) / (high - low))
    # And never below zero at one channel either, which only a narrowest width
    # above one could produce.
    one = max(0.0, worst[low] - marginal * (low - 1))
    if not one and not marginal:
        return 0, 0
    # Up to the worst run the window holds, and DOWN to it when the secant
    # already clears everything -- the scale is the tightest that still
    # covers, either way.
    scale = max(total / (one + marginal * (width - 1))
                for total, width in usable
                if one + marginal * (width - 1) > 0)
    # Rounded UP, both of them: truncation could leave the line a byte under
    # the very run it was scaled to cover, and "never below a measured run" is
    # the one property this whole function exists to keep.
    return math.ceil(one * scale), math.ceil(marginal * scale)


def _model_of(window) -> tuple:
    """The fitted figures a window implies, for both the table and the reader.

    One function because `record` writes them into the file and `cost_of`
    reads them out of it, and a table whose flat keys disagreed with what
    admission computes would be a table nobody could debug from.
    """
    keep = max(1, int(settings.COST_WINDOW))
    # Each resource over its own runs. The stored window holds more than `keep`
    # entries precisely so the card is not fitted over the same slice as the
    # host -- see `_trim`.
    recent = window[-keep:]
    vram_one, vram_marginal = _fit([(item[0] * item[2], item[2])
                                    for item in _vram_window(window, keep)])
    ram_one, ram_marginal = _fit([(item[1] * item[2], item[2])
                                  for item in recent])
    # Counted over the recent runs rather than per resource. A window with two
    # widths whose VRAM was only ever readable at one of them then prices VRAM
    # with the affine reading over a fallback slope, which over-reserves
    # slightly -- the safe side, and rare enough not to be worth a second flag.
    widths = len({item[2] for item in recent})
    return vram_one, vram_marginal, ram_one, ram_marginal, widths


def cost_of(tool_name: str) -> Optional[Cost]:
    """What this tool has been measured to need, or None if never seen.

    None is not zero, and the difference is the whole safety of the thing: a
    tool nobody has measured is admitted as if it needed the entire budget.
    """
    entry = _load().get(tool_name)
    if not isinstance(entry, dict):
        return None
    window = _window(entry)
    keep = max(1, int(settings.COST_WINDOW))
    recent = window[-keep:]
    # The runs that came back with a VRAM figure at all. A window of nothing
    # but nulls is a tool whose every reading was refused, which is not the
    # same tool as one measured at zero -- see `Cost.vram_known`.
    vram_seen = _vram_window(window, keep)
    vram_one, vram_marginal, ram_one, ram_marginal, widths = _model_of(window)
    # `channels=1` because `vram_bytes` describes ONE channel, intercept
    # included; `at(k)` carries it to a width. What is reserved covers the
    # heaviest run the window holds, at every width the window holds it at --
    # for the same reason the worst run used to be reserved: admission has to
    # fit the heaviest thing it has met, not the average one.
    return Cost(
        vram_bytes=vram_one,
        vram_marginal=vram_marginal,
        vram_known=bool(vram_seen),
        ram_bytes=ram_one,
        ram_marginal=ram_marginal,
        widths=widths,
        samples=int(entry.get(SAMPLES_KEY) or 0),
        channels=1,
        cpu_cores=max((item[3] for item in recent), default=0.0),
        vram_spread=_spread([(item[0], item[2]) for item in vram_seen],
                            vram_one, vram_marginal, widths),
        ram_spread=_spread([(item[1], item[2]) for item in recent],
                           ram_one, ram_marginal, widths),
    )


def _spread(points, one: int, marginal: int, widths: int) -> float:
    """How far the window's runs disagree once their WIDTH is accounted for.

    Worst over smallest, as it always was, but of each run measured against
    what the model predicts for the width it ran at rather than of the raw
    peaks. With an intercept the raw peaks are not comparable: AMASSS at one
    structure and AMASSS at three differ by 43% with nothing whatever varying
    in the request, and reporting that as "memory varies" would fire the one
    signal that exists for a genuinely input-dependent tool on every tool that
    is simply allowed to spread.

    Runs that measured nothing at all are ignored. A tabular tool records a
    VRAM of 0 on every run; dividing by that would report an infinite spread
    for a tool that never touches the card. Zeroes are absence of a
    measurement here, not a measurement of zero.

    For a window at a single width this is exactly the old computation: the
    prediction is then one number, and dividing every peak by the same number
    leaves their ratio untouched.
    """
    ratios = []
    for total, width in points:
        if not total:
            continue
        predicted = _predict(one, marginal, widths, width)
        if predicted <= 0:
            continue
        ratios.append((total * max(1, int(width))) / predicted)
    if len(ratios) < 2:
        return 1.0
    return max(ratios) / min(ratios)


def record(tool_name: str, vram_bytes: Optional[int], ram_bytes: Optional[int],
           channels: int = 1, cpu_cores: float = 0.0,
           vram_known: bool = True) -> None:
    """Fold one run's peaks into the table, keeping the larger of each.

    `vram_known=False` says this run measured RAM and cores but that its VRAM
    reading was thrown away -- a card-wide figure from a run that shared the
    machine, which is everybody's allocation attributed to each of them. The
    slot is written as null rather than as a zero, because a tool whose only
    readings were refused is UNKNOWN and must go on running alone, while a tool
    measured at zero costs the card nothing and may share it.

    Read-modify-write under a process lock and an atomic replace. Two uvicorn
    workers can still interleave and lose one update between them; that costs a
    high-water mark one run late, never a corrupt file, and the next run of that
    tool restores it. A lock file would buy exactness this does not need.
    """
    if not vram_bytes and not ram_bytes:
        return
    keep = max(1, int(settings.COST_WINDOW))
    with _lock:
        table = _load()
        entry = table.get(tool_name)
        if not isinstance(entry, dict):
            entry = {SAMPLES_KEY: 0}
            window = []
        else:
            window = _window(entry)
        # Stored PER CHANNEL, beside the width it was measured at, which is
        # what makes the two recoverable from each other: `_fit` multiplies
        # them straight back into the run's total, and the intercept can only
        # be separated from the slope by comparing totals at different widths.
        # The division survives because the file is read by people and by
        # older servers, for whom one normalised figure per run is the
        # readable form -- and because a table written before any of this
        # holds exactly that, at a width of one.
        # Rounded UP, so the total `_fit` multiplies back is never a few bytes
        # under the one this run actually reached. A remainder is nothing on a
        # figure in gigabytes, but "the fitted line covers every recorded run"
        # is easier to hold as an invariant than as an approximation.
        opened = max(1, int(channels))
        window.append([None if not vram_known else -(-int(vram_bytes or 0) // opened),
                       -(-int(ram_bytes or 0) // opened),
                       opened,
                       round(float(cpu_cores or 0.0) / opened, 3)])
        window = _trim(window, keep)
        entry[RECENT_KEY] = window
        # Kept beside the window, and the same numbers admission reads through
        # cost_of(): a human opening this file, and `runner.py` inside a tool's
        # own virtualenv, both see the fitted model rather than a list to
        # reduce themselves. The runner reads exactly these keys -- it ships
        # with the server and is injected by path, so the two cannot disagree
        # about what they mean.
        (entry[VRAM_KEY], entry[VRAM_MARGINAL_KEY],
         entry[RAM_KEY], entry[RAM_MARGINAL_KEY],
         entry[WIDTHS_KEY]) = _model_of(window)
        entry[CHANNELS_KEY] = max(1, int(channels))
        entry[SAMPLES_KEY] = int(entry.get(SAMPLES_KEY) or 0) + 1
        entry[UPDATED_KEY] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        table[tool_name] = entry
        _store(table)


def known() -> dict:
    """Every tool with a measurement, for the startup banner and the tests."""
    return {name: cost_of(name) for name in sorted(_load())}


def banner() -> str:
    """What this deployment has learned, said once at startup.

    Two things are worth reading before a shift rather than after an incident:
    which tools have never been measured -- those run ALONE, so they are the
    ones costing the parallelism -- and which tools' memory moves with the
    request, because for those the single reserved figure is the worst seen
    rather than a property of the tool.
    """
    table = {name: cost_of(name) for name in sorted(_load())}
    if not table:
        return ("Learned costs  none yet; every tool runs alone until it has "
                "been measured once")
    varying = [(name, cost) for name, cost in table.items()
               if cost and cost.input_dependent]
    lines = [
        "Learned costs  {} tool(s) measured; heaviest {}".format(
            len(table),
            max(table.items(), key=lambda item: item[1].vram_bytes if item[1] else 0)[0],
        )
    ]
    for name, cost in varying:
        lines.append(
            "  MEMORY VARIES  {} ranged x{:.1f} in VRAM and x{:.1f} in RAM over its "
            "last {} run(s), beyond what their widths explain; it reserves the "
            "worst of them".format(
                name, cost.vram_spread, cost.ram_spread, min(cost.samples,
                                                             int(settings.COST_WINDOW)))
        )
    if not varying:
        lines.append("  every measured tool has cost the same across its recorded "
                     "runs, so each reservation is a property of the tool")
    # Which tools have been seen at more than one width, and are therefore
    # priced with a MEASURED intercept rather than with the whole peak. It is
    # worth a line because it is the difference between a wide run reserving
    # what it costs and a wide run reserving a multiple of it: a tool three
    # quarters fixed fits four times as many channels in the same card.
    for name, cost in sorted(table.items()):
        if cost and cost.split_measured and cost.vram_fixed:
            lines.append(
                "  SPLIT MEASURED  {} costs {:.2f} GiB fixed + {:.2f} GiB per "
                "channel over {} widths".format(
                    name, cost.vram_fixed / 1024 ** 3,
                    cost.vram_marginal / 1024 ** 3, cost.widths)
            )
    return "\n".join(lines)
