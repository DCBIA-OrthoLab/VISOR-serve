"""The cost table: what a tool was measured to need, kept across runs.

Nothing in here declares a number. Every figure comes from a run that already
happened, which is the whole reason a budget can exist without a per-tool table
somebody has to maintain.
"""

import json
import os

import pytest

os.environ.setdefault("API_TOKEN", "test-token")

import config
from execution import costs, runner


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "SCHEMA_CACHE_DIR", str(tmp_path))
    return tmp_path


# ----------------------------------------------------------------------
# Recording and reading back
# ----------------------------------------------------------------------

def test_a_tool_nobody_has_run_has_no_cost():
    """None, not zero, and the difference is the safety of the whole thing: an
    unmeasured tool is admitted as if it needed everything."""
    assert costs.cost_of("AMASSS") is None


def test_one_run_is_enough_to_know_a_tool(cache_dir):
    costs.record("AMASSS", 2_349_230_080, 6 * 1024 ** 3)
    measured = costs.cost_of("AMASSS")
    assert measured.vram_bytes == 2_349_230_080
    assert measured.ram_bytes == 6 * 1024 ** 3
    assert measured.samples == 1
    assert os.path.exists(costs.table_path())


def test_the_table_keeps_the_worst_run_not_the_last():
    """A mean would admit two jobs on the strength of a small scan and then
    meet a large one."""
    costs.record("AMASSS", 2_000_000_000, 4 * 1024 ** 3)
    costs.record("AMASSS", 3_000_000_000, 2 * 1024 ** 3)
    costs.record("AMASSS", 1_000_000_000, 8 * 1024 ** 3)
    measured = costs.cost_of("AMASSS")
    assert measured.vram_bytes == 3_000_000_000
    assert measured.ram_bytes == 8 * 1024 ** 3
    assert measured.samples == 3


def test_tools_do_not_borrow_each_other_s_figures():
    costs.record("AMASSS", 2_349_230_080, 0)
    costs.record("ALI_CBCT", 268_400_640, 0)
    assert costs.cost_of("AMASSS").vram_bytes == 2_349_230_080
    assert costs.cost_of("ALI_CBCT").vram_bytes == 268_400_640


def test_a_run_that_measured_nothing_writes_nothing():
    """A tabular tool never touches the card and never imports torch."""
    costs.record("Surg_Mov_Pred", None, None)
    assert costs.cost_of("Surg_Mov_Pred") is None


def test_a_cpu_only_tool_can_still_record_its_memory():
    costs.record("Surg_Mov_Pred", None, 512 * 1024 ** 2)
    measured = costs.cost_of("Surg_Mov_Pred")
    assert measured.vram_bytes == 0
    assert measured.ram_bytes == 512 * 1024 ** 2


def test_the_table_is_replaced_atomically(cache_dir):
    """Never read half-written: the server reads this on the admission path."""
    costs.record("AMASSS", 1, 1)
    leftovers = [name for name in os.listdir(cache_dir) if name.endswith(".tmp")]
    assert leftovers == []


def test_an_unreadable_table_costs_nothing_but_knowledge(cache_dir):
    """Every tool then looks unknown and runs alone, which is today's behaviour.
    It must never cost a run."""
    with open(costs.table_path(), "w", encoding="utf-8") as handle:
        handle.write("{not json")
    assert costs.cost_of("AMASSS") is None
    costs.record("AMASSS", 5, 5)
    assert costs.cost_of("AMASSS").vram_bytes == 5


def test_a_cache_directory_that_cannot_be_written_is_survivable(monkeypatch):
    monkeypatch.setattr(config.settings, "SCHEMA_CACHE_DIR", "/proc/nonexistent/nope")
    costs.record("AMASSS", 1, 1)  # must not raise
    assert costs.cost_of("AMASSS") is None


def test_the_table_is_readable_by_a_human(cache_dir):
    """It is committed knowledge about a deployment, not an opaque cache."""
    costs.record("AMASSS", 2_349_230_080, 1024)
    loaded = json.loads(open(costs.table_path(), encoding="utf-8").read())
    assert loaded["AMASSS"][costs.VRAM_KEY] == 2_349_230_080
    assert loaded["AMASSS"][costs.UPDATED_KEY].endswith("Z")


def test_known_lists_every_measured_tool():
    costs.record("AMASSS", 2, 2)
    costs.record("ALI_CBCT", 1, 1)
    assert sorted(costs.known()) == ["ALI_CBCT", "AMASSS"]


# ----------------------------------------------------------------------
# What the runner measures
# ----------------------------------------------------------------------

def test_the_runner_measures_its_own_resident_memory():
    """Host RAM was never measured, and it is the resource that binds before
    the card on the machine this was written for."""
    assert runner._peak_rss_bytes() > 0


def test_a_chain_reports_the_worst_of_its_steps(monkeypatch):
    """`sup.run` blocks, so children are strictly sequential and only one is
    ever resident. Summing them would refuse chains that fit perfectly."""
    monkeypatch.setattr(runner, "_child_vram_bytes", 0)
    monkeypatch.setattr(runner, "_peak_vram_bytes", lambda: 0)
    runner._fold_child_measurement({"peak_vram_bytes": 1_000})
    runner._fold_child_measurement({"peak_vram_bytes": 3_000})
    runner._fold_child_measurement({"peak_vram_bytes": 2_000})
    assert runner._measurements()["peak_vram_bytes"] == 3_000


def test_an_orchestrator_adds_its_own_hold_to_its_children(monkeypatch):
    """An upper bound, and exact for the orchestrators here, which import no
    torch at all and hold nothing while a child runs."""
    monkeypatch.setattr(runner, "_child_vram_bytes", 0)
    monkeypatch.setattr(runner, "_peak_vram_bytes", lambda: 500)
    runner._fold_child_measurement({"peak_vram_bytes": 3_000})
    assert runner._measurements()["peak_vram_bytes"] == 3_500


def test_a_child_that_measured_nothing_does_not_disturb_the_fold(monkeypatch):
    monkeypatch.setattr(runner, "_child_vram_bytes", 0)
    monkeypatch.setattr(runner, "_peak_vram_bytes", lambda: 0)
    runner._fold_child_measurement({"result": "ok"})
    runner._fold_child_measurement({"peak_vram_bytes": None})
    runner._fold_child_measurement({"peak_vram_bytes": "lots"})
    assert "peak_vram_bytes" not in runner._measurements()


# ----------------------------------------------------------------------
# What dispatch keeps
# ----------------------------------------------------------------------

def test_dispatch_keeps_what_a_successful_run_measured(tmp_path):
    from execution import dispatch

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "result.json").write_text(json.dumps({
        "result": "ok", "peak_vram_bytes": 2_349_230_080, "peak_rss_bytes": 1_000,
    }))
    assert dispatch._read_result(str(job_dir), "AMASSS") == "ok"
    assert costs.cost_of("AMASSS").vram_bytes == 2_349_230_080


def test_dispatch_keeps_what_a_FAILED_run_measured(tmp_path):
    """The run that died of an out-of-memory is the one worth learning from,
    and it is exactly the one that used to record nothing."""
    from execution import dispatch

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "result.json").write_text(json.dumps({
        "error": {"type": "OutOfMemoryError", "message": "CUDA out of memory"},
        "peak_vram_bytes": 40_000_000_000,
        "peak_rss_bytes": 2_000,
    }))
    with pytest.raises(dispatch.ToolFailure):
        dispatch._read_result(str(job_dir), "Batch_Dental_Seg")
    assert costs.cost_of("Batch_Dental_Seg").vram_bytes == 40_000_000_000


# ----------------------------------------------------------------------
# The window: one outlier must not pin a tool for ever
# ----------------------------------------------------------------------

def test_an_outlier_is_forgotten_once_the_tool_behaves_again(monkeypatch):
    """An all-time maximum never comes back down. One unusual scan would then
    raise this tool's reservation for every run that follows it, and collapse
    its parallelism permanently."""
    monkeypatch.setattr(config.settings, "COST_WINDOW", 5)
    costs.record("ALI_CBCT", 8 << 30, 1 << 30)          # the outlier
    assert costs.cost_of("ALI_CBCT").vram_bytes == 8 << 30
    for _ in range(5):
        costs.record("ALI_CBCT", 268_400_640, 1 << 30)  # back to normal
    assert costs.cost_of("ALI_CBCT").vram_bytes == 268_400_640


def test_the_outlier_still_counts_while_it_is_in_the_window(monkeypatch):
    """It is forgotten only after the tool has behaved normally a windowful of
    times -- not on the very next run."""
    monkeypatch.setattr(config.settings, "COST_WINDOW", 5)
    costs.record("ALI_CBCT", 8 << 30, 0)
    for _ in range(4):
        costs.record("ALI_CBCT", 1 << 20, 0)
    assert costs.cost_of("ALI_CBCT").vram_bytes == 8 << 30


def test_the_window_is_bounded(cache_dir, monkeypatch):
    """A tool run ten thousand times must not grow a ten-thousand-entry file.

    Twice the window, not once: an entry is kept while it is among the recent
    runs OR among the recent card readings, and those are two windows -- see
    `costs._trim`.
    """
    monkeypatch.setattr(config.settings, "COST_WINDOW", 4)
    for index in range(50):
        costs.record("ALI_CBCT", index + 1, index + 1)
    loaded = json.loads(open(costs.table_path(), encoding="utf-8").read())
    assert len(loaded["ALI_CBCT"][costs.RECENT_KEY]) == 8


def test_a_burst_of_contended_runs_does_not_erase_the_card_figure(monkeypatch):
    """The regression this window shape exists for.

    A card reading is only learned from a run that was alone, so a busy arm
    writes nothing but refusals. Counted against one flat window they evict
    every measurement the tool ever made, it becomes unmeasured, and an
    unmeasured tool reserves the whole budget and runs ALONE: twenty concurrent
    ALI_CBCT runs took 549.8 s instead of 140.0 s that way on 2026-09-21.
    """
    monkeypatch.setattr(config.settings, "COST_WINDOW", 5)
    costs.record("ALI_CBCT", 665_845_760, 1 << 30)
    for _ in range(20):
        costs.record("ALI_CBCT", 9 << 30, 1 << 30, vram_known=False)
    learned = costs.cost_of("ALI_CBCT")
    assert learned.vram_known
    assert learned.vram_bytes == 665_845_760


def test_the_card_window_still_forgets_an_outlier(monkeypatch):
    """Keeping refusals out of it must not make it an all-time maximum.

    A windowful of real readings displaces the outlier exactly as before; what
    no longer displaces it is a run that measured nothing.
    """
    monkeypatch.setattr(config.settings, "COST_WINDOW", 5)
    costs.record("ALI_CBCT", 8 << 30, 1 << 30)
    for _ in range(5):
        costs.record("ALI_CBCT", 1 << 20, 1 << 30)
    assert costs.cost_of("ALI_CBCT").vram_bytes == 1 << 20


def test_a_zero_does_not_outrank_a_card_the_tool_was_seen_to_take(monkeypatch):
    """A GPU run whose card reading came back as no growth failed to measure
    it; it did not discover the tool stopped using the card."""
    monkeypatch.setattr(config.settings, "COST_WINDOW", 5)
    costs.record("ALI_CBCT", 665_845_760, 1 << 30)
    for _ in range(5):
        costs.record("ALI_CBCT", 0, 1 << 30)
    assert costs.cost_of("ALI_CBCT").vram_bytes == 665_845_760


def test_a_tool_that_has_only_ever_read_zero_still_costs_the_card_nothing(
        monkeypatch):
    """The other direction, and the one that must not move: a tabular tool
    reads zero on every run and has to go on sharing the card with anything."""
    monkeypatch.setattr(config.settings, "COST_WINDOW", 5)
    for _ in range(5):
        costs.record("Surg_Mov_Pred", 0, 1 << 30)
    learned = costs.cost_of("Surg_Mov_Pred")
    assert learned.vram_known
    assert learned.vram_bytes == 0


def test_a_table_written_before_the_window_keeps_its_figure(cache_dir):
    """The scalar is the best estimate that exists for that tool; it seeds the
    window rather than being thrown away."""
    with open(costs.table_path(), "w", encoding="utf-8") as handle:
        json.dump({"AMASSS": {costs.VRAM_KEY: 2_349_230_080,
                              costs.RAM_KEY: 1000, costs.SAMPLES_KEY: 9}}, handle)
    assert costs.cost_of("AMASSS").vram_bytes == 2_349_230_080


def test_the_scalar_pair_stays_readable_beside_the_window(cache_dir, monkeypatch):
    """A human opening this file sees one pair per tool, not a list to reduce."""
    monkeypatch.setattr(config.settings, "COST_WINDOW", 3)
    costs.record("AMASSS", 5, 50)
    costs.record("AMASSS", 9, 10)
    loaded = json.loads(open(costs.table_path(), encoding="utf-8").read())
    assert loaded["AMASSS"][costs.VRAM_KEY] == 9
    assert loaded["AMASSS"][costs.RAM_KEY] == 50


def test_samples_still_counts_every_run_ever(monkeypatch):
    monkeypatch.setattr(config.settings, "COST_WINDOW", 2)
    for _ in range(7):
        costs.record("AMASSS", 1, 1)
    assert costs.cost_of("AMASSS").samples == 7


# ----------------------------------------------------------------------
# A run costs `fixed + marginal x width`, and the fixed part is most of it
# ----------------------------------------------------------------------
#
# Measured on two tools of opposite shape -- a 3D nnUNet and a 2D UNet over
# multi-view rendering, nothing in common:
#
#     AMASSS    width 1  5291 MiB   width 2  6371   width 3  7587   (2026-09-18)
#     ALI_IOS   width 1  1930 MiB   width 2  2516   width 4  3342
#               width 8  4886                                       (2026-09-21)
#
# Both roughly three quarters fixed, because the fixed part is the CUDA
# context and the resident model, which every GPU tool has. The model used to
# be purely proportional, which forced every tool to choose between two errors
# and only one of them was safe: declare the honest width and teach the table
# a per-channel figure 40% below what one channel really needs, or under-
# declare and hold ~4 GiB per concurrent run that nobody needs.

MiB = 1024 ** 2


def _seen(tool_name, *runs):
    """Record `(width, total MiB)` pairs the way a run of that width would."""
    for width, total in runs:
        costs.record(tool_name, total * MiB, total * MiB, channels=width)
    return costs.cost_of(tool_name)


def test_one_width_is_still_read_exactly_as_it_always_was():
    """One point cannot separate an intercept from a slope, and guessing one
    is how a reservation silently becomes too small. Nothing regresses on a
    tool nobody has run wide -- which is most of the table."""
    measured = _seen("ALI_CBCT", (1, 1000), (1, 1200))
    assert measured.vram_bytes == 1200 * MiB
    assert measured.vram_fixed == 0, "nothing measured a fixed part"
    assert measured.at(4).vram_bytes == 4800 * MiB, "four times one channel"


def test_two_widths_recover_the_intercept_and_the_slope():
    """1 GiB of context plus 256 MiB a channel, put in as the totals a run of
    each width would have reported, and read back as the two numbers."""
    measured = _seen("X", (1, 1024 + 256), (4, 1024 + 4 * 256))
    assert measured.vram_fixed == 1024 * MiB
    assert measured.vram_marginal == 256 * MiB
    assert measured.ram_fixed == 1024 * MiB
    assert measured.at(8).vram_bytes == (1024 + 8 * 256) * MiB


def test_a_tool_with_no_fixed_part_at_all_is_unchanged():
    """A tool that really is proportional measures an intercept of zero, and
    then this whole mechanism is the multiplication it replaced."""
    measured = _seen("X", (1, 500), (4, 2000))
    assert measured.vram_fixed == 0
    assert measured.vram_marginal == 500 * MiB
    assert measured.at(3).vram_bytes == 1500 * MiB


def test_a_narrow_run_after_a_wide_one_is_not_reserved_the_quotient():
    """THE defect, in the direction that ends in an out-of-memory. AMASSS at
    width 2 costs 6371 MiB, and dividing that by 2 reserved 3185 for its next
    run at one structure -- which needs 5291, 40% more. Nothing may be
    reserved less than a run the tool has already been seen to make."""
    measured = _seen("AMASSS", (2, 6371))
    assert measured.at(1).vram_bytes >= 5291 * MiB
    assert measured.at(1).vram_bytes == 6371 * MiB


def test_AMASSS_reproduces_its_measured_split():
    """Three quarters of it is the CUDA context and the resident network; the
    rest is ~1.15 GiB per structure."""
    measured = _seen("AMASSS", (1, 5291), (2, 6371), (3, 7587))
    assert round(measured.vram_fixed / MiB) == 4143
    assert round(measured.vram_marginal / MiB) == 1148
    assert measured.vram_fixed / measured.vram_bytes > 0.75
    for width, total in ((1, 5291), (2, 6371), (3, 7587)):
        assert measured.at(width).vram_bytes >= total * MiB, (
            "a fit may never sit below a run that already happened")


def test_ALI_IOS_reproduces_its_measured_split():
    """1508 MiB fixed + 422 a channel is what the secant through its extremes
    says, and 78% fixed is the shape that matters. Both come out 7% larger
    here, and deliberately: ALI_IOS is slightly CONCAVE -- its marginal falls
    from 586 to 386 MiB a channel as it widens -- so its width-2 run cost 164
    MiB more than that secant, and the line is scaled until it covers every
    run recorded. The 7% is the price of never reserving less than something
    already measured, and it is paid by both parameters rather than dumped on
    the intercept, which no wide run would then be paying for."""
    measured = _seen("ALI_IOS", (1, 1930), (2, 2516), (4, 3342), (8, 4886))
    assert round(measured.vram_marginal / MiB) == 452
    assert round(measured.vram_fixed / MiB) == 1613
    assert 0.75 < measured.vram_fixed / measured.vram_bytes < 0.80
    for width, total in ((1, 1930), (2, 2516), (4, 3342), (8, 4886)):
        assert measured.at(width).vram_bytes >= total * MiB


def test_CLIC_is_not_given_a_fixed_cost_its_narrow_runs_never_paid():
    """This deployment's own table, and the reason the line is SCALED to cover
    its runs rather than shifted up to them. CLIC's width-39 run sits 8% above
    the secant, which is noise across 39 working sets -- but read as an
    intercept it becomes 1.7 GiB of context, and one channel of CLIC would be
    priced at 2266 MiB against the 654 it was measured at. Scaling spreads the
    same 12% over both parameters instead."""
    measured = _seen("CLIC", (1, 654), (4, 2342), (9, 5226), (39, 21350),
                     (45, 22750))
    assert round(measured.vram_fixed / MiB) == 170, "small, as the data says"
    assert measured.at(1).vram_bytes < 800 * MiB
    for width, total in ((1, 654), (9, 5226), (45, 22750)):
        assert measured.at(width).vram_bytes >= total * MiB


def test_a_wide_run_that_measured_less_does_not_give_memory_back():
    """Noise, or a cohort that happened to be smaller. A negative slope would
    make a wide run cheaper than a narrow one, which is not a thing memory
    does; it reads as "nothing costs extra" and the peak stands."""
    measured = _seen("X", (1, 4000), (4, 3800))
    assert measured.vram_marginal == 0
    assert measured.at(8).vram_bytes == 4000 * MiB


def test_the_width_a_room_affords_is_the_inverse_of_what_a_width_costs():
    """`concurrency._affordable` and `runner._channels_affordable` both invert
    this, and a division by the one-channel figure charged every channel for
    the context again: ALI_IOS got 4 channels out of 8 GiB where it fits 14."""
    measured = _seen("ALI_IOS", (1, 1930), (2, 2516), (4, 3342), (8, 4886))
    assert measured.channels_within(vram_bytes=8 * 1024 ** 3) == 14
    assert measured.at(14).vram_bytes <= 8 * 1024 ** 3
    assert measured.at(15).vram_bytes > 8 * 1024 ** 3


def test_a_tool_too_big_for_the_room_still_gets_one_channel():
    """A demand larger than the budget still runs, alone. Refusing here would
    make a tool the machine can in fact run permanently unrunnable."""
    measured = _seen("AMASSS", (1, 5291), (3, 7587))
    assert measured.channels_within(vram_bytes=1024 * MiB) == 1


def test_the_banner_says_which_tools_were_split_rather_than_assumed():
    """The difference between a wide run reserving what it costs and a wide
    run reserving a multiple of it, said once at startup."""
    _seen("AMASSS", (1, 5291), (2, 6371), (3, 7587))
    said = costs.banner()
    assert "SPLIT MEASURED" in said
    assert "AMASSS" in said


def test_a_tool_seen_at_one_width_is_not_announced_as_split():
    _seen("AMASSS", (1, 5291), (1, 6371))
    assert "SPLIT MEASURED" not in costs.banner()


# ----------------------------------------------------------------------
# Does a tool's memory depend on the TOOL, or on the request?
# ----------------------------------------------------------------------
#
# Admission reserves one figure per tool, which is only sound while that figure
# is a property of the tool. Nothing here refuses a tool whose peak moves with
# the request -- the window's maximum is reserved, so the estimate stays
# conservative -- but it has to be SAYABLE, because the difference between a
# reservation that is exact and one that is a bet is invisible in the single
# number the table prints.

def test_a_tool_that_costs_the_same_every_run_is_not_input_dependent():
    for _ in range(5):
        costs.record("ALI_CBCT", 268_400_640, 2 * 1024 ** 3)
    measured = costs.cost_of("ALI_CBCT")
    assert measured.vram_spread == 1.0
    assert not measured.input_dependent


def test_a_tool_that_is_mostly_fixed_cost_is_not_a_tool_that_varies():
    """The width is not the request. AMASSS costs 5291 MiB at one structure
    and 7587 at three with nothing whatever varying in what was asked, and
    comparing those peaks directly reported a x1.4 variation -- and ALI_IOS,
    over four widths, a x3.2 one. Both would have fired the one signal that
    exists for a genuinely input-dependent tool. The spread is measured
    against the fitted model, so what the width explains is not variation."""
    measured = _seen("ALI_IOS", (1, 1930), (2, 2516), (4, 3342), (8, 4886))
    assert measured.vram_spread < 1.1
    assert not measured.input_dependent


def test_a_tool_whose_peak_moves_with_the_request_is_flagged():
    """Batch_Dental_Seg's four bundles label different things and are not the
    same size; AMASSS segments as many structures as it was asked for."""
    costs.record("Batch_Dental_Seg", 1 << 30, 1 << 30)
    costs.record("Batch_Dental_Seg", 5 << 30, 1 << 30)
    measured = costs.cost_of("Batch_Dental_Seg")
    assert measured.vram_spread == 5.0
    assert measured.input_dependent
    assert measured.vram_bytes == 5 << 30, "and it reserves the worst of them"


def test_a_few_percent_between_two_identical_runs_is_not_a_signal():
    """The same request twice already differs -- allocator behaviour, a
    different cuDNN algorithm for the same shapes. A threshold that fired on
    that would fire on everything."""
    costs.record("CLIC", 1_000_000_000, 1 << 30)
    costs.record("CLIC", 1_060_000_000, 1 << 30)
    assert not costs.cost_of("CLIC").input_dependent


def test_a_tool_that_never_touches_the_card_has_no_vram_spread():
    """Zero is the absence of a measurement, not a measurement of zero, and
    dividing by it would report an infinite spread for every tabular tool."""
    costs.record("Surg_Mov_Pred", 0, 1 << 30)
    costs.record("Surg_Mov_Pred", 0, 2 << 30)
    measured = costs.cost_of("Surg_Mov_Pred")
    assert measured.vram_spread == 1.0
    assert measured.ram_spread == 2.0


def test_one_run_cannot_disagree_with_itself():
    costs.record("AMASSS", 1 << 30, 1 << 30)
    assert costs.cost_of("AMASSS").vram_spread == 1.0


def test_the_banner_names_the_tool_whose_memory_moves():
    costs.record("Batch_Dental_Seg", 1 << 30, 1 << 30)
    costs.record("Batch_Dental_Seg", 5 << 30, 1 << 30)
    said = costs.banner()
    assert "MEMORY VARIES" in said
    assert "Batch_Dental_Seg" in said


def test_the_banner_says_so_when_nothing_has_been_measured():
    """Which is the state that makes every tool run alone, so it is the one
    worth reading before wondering why the server feels serial."""
    assert "none yet" in costs.banner()


def test_the_banner_says_so_when_every_tool_is_stable():
    costs.record("ALI_CBCT", 1 << 30, 1 << 30)
    assert "property of the tool" in costs.banner()


# ----------------------------------------------------------------------
# Where a VRAM figure came from, and when it may be believed
# ----------------------------------------------------------------------
#
# Measured this afternoon, ALI_CBCT alone on an idle machine, one landmark per
# channel: 0.62 G at one channel, 7.80 G at seven, 16.74 G at fifteen and
# 16.92 G at 119 landmarks over fifteen -- 1.12 G per channel, flat and linear.
# The table meanwhile held ~2.2 G for it and had held 5.66 G earlier the same
# day; ASO reached 32.43 G per channel, two thirds of the card for ONE channel,
# and Crown_Seg reported a spread of x782. The error is systematic, not noise:
# a card-wide reading taken while six runs share the machine is everybody's
# allocation attributed to each of them, and `record` keeps the WORST of its
# window, so one contended run poisons the figure for twenty afterwards. The
# consequence was measured too: a 119-landmark request opened 15 channels where
# 30 would have fitted.

def _runner_measures(monkeypatch, torch_bytes, card_bytes, touched=True,
                     child=None):
    """One `_measurements()` with the two VRAM sources under control.

    `child` is a supervised call's own result payload, folded in the way the
    supervisor folds it -- which is how a chain comes to report a figure its
    own interpreter never allocated.
    """
    monkeypatch.setattr(runner, "_touched_torch", lambda: touched)
    monkeypatch.setattr(runner, "_peak_vram_bytes", lambda: torch_bytes)
    monkeypatch.setattr(runner, "_child_vram_bytes", 0)
    monkeypatch.setattr(runner, "_child_vram_source", runner.VRAM_FROM_NONE)
    monkeypatch.setattr(runner._sampler, "peak_vram", card_bytes)
    monkeypatch.setattr(runner._sampler, "peak_rss", 1 << 20)
    monkeypatch.setattr(runner._sampler, "stop", lambda: None)
    monkeypatch.setenv(runner.PROGRESS_FILE_ENV, "")
    if child is not None:
        runner._fold_child_measurement(child)
    return runner._measurements()


def test_the_runner_names_torch_when_the_tool_counted_its_own_bytes(monkeypatch):
    measured = _runner_measures(monkeypatch, 5 << 30, 38 << 30)
    assert measured["peak_vram_bytes"] == 5 << 30
    assert measured[runner.VRAM_SOURCE_KEY] == runner.VRAM_FROM_TORCH


def test_the_runner_names_the_card_when_it_had_to_fall_back(monkeypatch):
    """The fallback's one case: a tool whose GPU work happens in children,
    where torch in this interpreter has nothing to report."""
    measured = _runner_measures(monkeypatch, 0, 3 << 30)
    assert measured["peak_vram_bytes"] == 3 << 30
    assert measured[runner.VRAM_SOURCE_KEY] == runner.VRAM_FROM_CARD


def test_a_run_with_no_vram_figure_says_so_rather_than_nothing(monkeypatch):
    measured = _runner_measures(monkeypatch, 0, 9 << 30, touched=False)
    assert "peak_vram_bytes" not in measured
    assert measured[runner.VRAM_SOURCE_KEY] == runner.VRAM_FROM_NONE


def test_a_chain_reports_the_weakest_source_it_was_built_from(monkeypatch):
    """ASO's own interpreter allocates nothing: its whole VRAM figure is ALI's.
    Adding a child's card reading inside this process must not relabel it as
    this process's own allocation -- ASO is the tool that reached 32.43 GiB per
    channel in the table."""
    measured = _runner_measures(
        monkeypatch, 0, 0,
        child={"peak_vram_bytes": 30 << 30, "vram_source": runner.VRAM_FROM_CARD})
    assert measured["peak_vram_bytes"] == 30 << 30
    assert measured[runner.VRAM_SOURCE_KEY] == runner.VRAM_FROM_CARD


def test_a_chain_of_torch_readings_stays_a_torch_reading(monkeypatch):
    measured = _runner_measures(
        monkeypatch, 1 << 30, 0,
        child={"peak_vram_bytes": 3 << 30, "vram_source": runner.VRAM_FROM_TORCH})
    assert measured["peak_vram_bytes"] == 4 << 30
    assert measured[runner.VRAM_SOURCE_KEY] == runner.VRAM_FROM_TORCH


def _measured(source, vram=16 * 1024 ** 3):
    """A result.json body carrying one run's peaks and their provenance."""
    return {"result": "ok", "peak_vram_bytes": vram, "peak_rss_bytes": 2 * 1024 ** 3,
            "peak_cpu_cores": 7.5, "channels": 1, "vram_source": source}


@pytest.mark.parametrize("solo", [True, False])
def test_a_torch_figure_is_learned_whether_the_run_was_alone_or_not(solo):
    """It is what the tool's own allocator reported for its own process.
    Concurrency cannot inflate that, so solitude is irrelevant to it."""
    from execution import dispatch

    dispatch._keep_measurements("Batch_Dental_Seg", _measured("torch"), solo=solo)
    assert costs.cost_of("Batch_Dental_Seg").vram_bytes == 16 * 1024 ** 3


def test_a_card_figure_is_learned_from_a_run_that_had_the_machine_to_itself():
    """On an idle machine the card's growth IS this run's allocation, and on
    this deployment it is the only reading the driver ever gives."""
    from execution import dispatch

    dispatch._keep_measurements("ALI_CBCT", _measured("card"), solo=True)
    measured = costs.cost_of("ALI_CBCT")
    assert measured.vram_bytes == 16 * 1024 ** 3
    assert measured.vram_known is True


def test_a_card_figure_from_a_shared_run_is_not_learned():
    """RAM and cores still are: they are per-PROCESS readings, over this run's
    own process group, so a neighbour cannot appear in them."""
    from execution import dispatch

    dispatch._keep_measurements("ALI_CBCT", _measured("card"), solo=False)
    measured = costs.cost_of("ALI_CBCT")
    assert measured.vram_bytes == 0
    assert measured.vram_known is False, "a refusal is absence, not a zero"
    assert measured.ram_bytes == 2 * 1024 ** 3
    assert measured.cpu_cores == 7.5


def test_a_refused_run_does_not_lower_a_figure_already_learned():
    """The table is a high-water mark, and dropping one run's VRAM must not be
    a way of writing a zero into it."""
    from execution import dispatch

    dispatch._keep_measurements("ALI_CBCT", _measured("card", 1_202_590_842), solo=True)
    dispatch._keep_measurements("ALI_CBCT", _measured("card", 32 * 1024 ** 3), solo=False)
    assert costs.cost_of("ALI_CBCT").vram_bytes == 1_202_590_842


def test_a_solo_run_after_a_refusal_teaches_the_table():
    """Which is why the guard does not deadlock the bootstrap: a tool nothing
    has measured runs ALONE, and a run that runs alone is solo."""
    from execution import dispatch

    dispatch._keep_measurements("ALI_CBCT", _measured("card"), solo=False)
    assert costs.cost_of("ALI_CBCT").vram_known is False
    dispatch._keep_measurements("ALI_CBCT", _measured("card", 1_202_590_842), solo=True)
    assert costs.cost_of("ALI_CBCT").vram_bytes == 1_202_590_842


def test_a_tool_whose_only_reading_was_refused_still_reserves_everything():
    """The whole reason a refusal is written as null rather than as 0: zero is
    "costs the card nothing" and would let the tool share it with anything."""
    import resources
    from execution import admission, dispatch

    dispatch._keep_measurements("ALI_CBCT", _measured("card"), solo=False)
    allocation = resources.Allocation(
        cpus=8.0, ram_bytes=16 * 1024 ** 3, vram_bytes=8 * 1024 ** 3,
        cpus_per_job=2, ram_per_job=4 * 1024 ** 3, vram_per_job=2 * 1024 ** 3,
        expected_clients=4,
    )
    demand = admission.demand_for("ALI_CBCT", allocation,
                                  costs.cost_of("ALI_CBCT"), uses_gpu=True)
    assert demand.measured is False
    assert demand.vram_bytes == allocation.vram_bytes


def test_an_unsourced_figure_is_still_learned():
    """Nothing but a result.json written before `vram_source` existed can lack
    it -- the runner is injected by path and is always this server's own -- and
    refusing those would throw away every figure a benchmark campaign wrote."""
    from execution import dispatch

    payload = _measured("torch")
    payload.pop("vram_source")
    dispatch._keep_measurements("AMASSS", payload, solo=False)
    assert costs.cost_of("AMASSS").vram_bytes == 16 * 1024 ** 3


def test_the_refusal_travels_from_admission_to_the_table(tmp_path):
    """The plumbing, end to end: `_read_result` carries what admission observed
    of the run into what the table is allowed to learn from it."""
    from execution import dispatch

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "result.json").write_text(json.dumps(_measured("card")))
    assert dispatch._read_result(str(job_dir), "ALI_CBCT", solo=False) == "ok"
    assert costs.cost_of("ALI_CBCT").vram_known is False
