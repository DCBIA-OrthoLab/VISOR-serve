"""Stopping a run where a reader asked, and nowhere else.

Two mechanisms, one question. A chain already has a boundary per `sup.run()`;
`sup.declareQualityControl(...)` adds one in the middle of a tool's own work,
where no call boundary exists. Both are published as the same check box, and
both are OFF until somebody ticks them -- a run that was not asked to stop
must not stop, on a server other people are queueing on.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

from starlette.datastructures import UploadFile as StarletteUploadFile

import pytest

from registry import schema_tool
from registry.deployment import DeploymentConfig

from test_supervisor import LEAF, make_tool, run_job, tools_dir  # noqa: F401


def _tool(tmp_path, **schema):
    base = {
        "name": "Orchestrator",
        "arguments": {"scans": {"type": "path"}},
        "returns": "path",
    }
    base.update(schema)
    return schema_tool.SchemaTool(str(tmp_path), base, DeploymentConfig({}).for_tool("X"))


# ----------------------------------------------------------------------
# What the server publishes
# ----------------------------------------------------------------------

def test_a_tool_that_calls_another_is_offered_a_stop_at_that_call(tmp_path):
    tool = _tool(tmp_path, supervisor=True, calls=["Leaf", "Other"])
    spec = tool.arguments["stop_after"]
    assert spec.type == "multichoice"
    assert spec.required is False
    assert spec.choices == {"Leaf": False, "Other": False}, (
        "a run nobody asked to stop must not stop"
    )


def test_a_tool_that_declares_its_own_gets_one_with_no_calls_at_all(tmp_path):
    # AMASSS segmenting five structures calls nobody and still has a moment
    # worth looking at.
    tool = _tool(tmp_path, supervisor=True, quality_controls=["after the crop"])
    assert tool.arguments["stop_after"].choices == {"after the crop": False}


def test_a_tool_with_nowhere_to_stop_is_offered_nothing(tmp_path):
    tool = _tool(tmp_path)
    assert "stop_after" not in tool.arguments, (
        "a check box that can never fire is worse than no check box"
    )


def test_calls_and_declarations_are_offered_together_without_repeating(tmp_path):
    tool = _tool(tmp_path, supervisor=True, calls=["ALI_CBCT"],
                 quality_controls=["landmarks", "ALI_CBCT"])
    assert list(tool.arguments["stop_after"].choices) == ["ALI_CBCT", "landmarks"]


def test_the_two_injected_boxes_read_in_the_order_they_are_decided(tmp_path):
    # Where it stops, what it hands back, where that is written.
    tool = _tool(
        tmp_path, supervisor=True, calls=["Leaf"],
        arguments={"scans": {"type": "path"},
                   "out": {"type": "str", "required": False, "section": "Outputs"}},
    )
    assert list(tool.arguments) == ["scans", "stop_after", "keep_intermediate", "out"]


def test_it_has_a_section_of_its_own(tmp_path):
    tool = _tool(tmp_path, supervisor=True, calls=["Leaf"])
    assert tool.arguments["stop_after"].section == schema_tool.STOP_AFTER_SECTION
    assert tool.arguments["stop_after"].section != \
        tool.arguments["keep_intermediate"].section


def test_the_argument_name_is_the_same_on_both_sides():
    """`runner.py` is run by a TOOL's interpreter and cannot import the
    server, so the two spellings are separate constants and have to agree."""
    from execution import runner
    assert runner.STOP_AFTER_ARGUMENT == schema_tool.STOP_AFTER_ARGUMENT
    assert runner.STOP_PATH_SEPARATOR == schema_tool.STOP_PATH_SEPARATOR


# ----------------------------------------------------------------------
# A stop names a PATH, so a chain can offer what its callees offer
# ----------------------------------------------------------------------

def _catalogue(tmp_path, *schemas) -> dict:
    """A registry of `SchemaTool`s, as `_build_registry` would hold them.

    Built by hand rather than discovered: what is under test is the
    composition over a registry, and a folder on disk per tool would only be
    an elaborate way of writing the same `calls` lists.
    """
    from registry import _publish_transitive_stops

    tools = {schema["name"]: _tool(tmp_path, **schema) for schema in schemas}
    _publish_transitive_stops(tools)
    return tools


def test_a_chain_offers_the_checkpoints_of_the_tools_it_calls(tmp_path):
    """`describe.py` sees one tool, so `AREG` can only ever declare `ASO`.
    Composing `ASO/ALI_CBCT` out of that needs the sibling's schema, which is
    known on the server and nowhere else."""
    tools = _catalogue(
        tmp_path,
        {"name": "ALI_CBCT", "supervisor": True, "quality_controls": ["landmarks"]},
        {"name": "ASO", "supervisor": True, "calls": ["ALI_CBCT"]},
        {"name": "AMASSS"},
        {"name": "AREG", "supervisor": True, "calls": ["ASO", "AMASSS"]},
    )
    assert list(tools["ASO"].arguments["stop_after"].choices) == [
        "ALI_CBCT", "ALI_CBCT/landmarks",
    ]
    assert list(tools["AREG"].arguments["stop_after"].choices) == [
        "ASO", "ASO/ALI_CBCT", "ASO/ALI_CBCT/landmarks", "AMASSS",
    ], "a nested checkpoint is unreachable unless it is published"


def test_a_tool_that_calls_nobody_is_offered_exactly_what_it_declares(tmp_path):
    """The composition must not invent a box for a leaf, nor move the one a
    leaf already had."""
    tools = _catalogue(
        tmp_path,
        {"name": "AMASSS", "supervisor": True, "quality_controls": ["after the crop"]},
        {"name": "Plain"},
    )
    assert list(tools["AMASSS"].arguments["stop_after"].choices) == ["after the crop"]
    assert "stop_after" not in tools["Plain"].arguments


def test_a_nested_checkpoint_is_an_option_a_caller_can_actually_send(tmp_path):
    """The published vocabulary and the wire have to agree: a path that
    `validate()` refuses is a check box that cannot be ticked."""
    tools = _catalogue(
        tmp_path,
        {"name": "ALI_CBCT", "supervisor": True, "quality_controls": ["landmarks"]},
        {"name": "ASO", "supervisor": True, "calls": ["ALI_CBCT"]},
    )
    cleaned = tools["ASO"].validate(
        {"scans": str(tmp_path), "stop_after": "ALI_CBCT/landmarks"})
    assert cleaned["stop_after"].selected == ("ALI_CBCT/landmarks",)


def test_a_facade_publishes_the_nested_checkpoints_of_each_of_its_modes(tmp_path):
    """A facade deep-copies its targets' arguments, so the transitive list has
    to be composed BEFORE it is built -- otherwise the tools get the nested
    checkpoints and the facade over them keeps the stale list."""
    from registry import facade

    tools = _catalogue(
        tmp_path,
        {"name": "ALI_CBCT", "supervisor": True, "quality_controls": ["landmarks"]},
        {"name": "ALI_IOS", "supervisor": True, "quality_controls": ["teeth"]},
        {"name": "ASO_CBCT", "supervisor": True, "calls": ["ALI_CBCT"]},
        {"name": "ASO_IOS", "supervisor": True, "calls": ["ALI_IOS"]},
    )
    composed = facade.compose(
        "ASO", {"CBCT": "ASO_CBCT", "IOS": "ASO_IOS"}, tools)
    spec = composed.arguments["stop_after"]
    assert list(spec.choices) == [
        "ALI_CBCT", "ALI_CBCT/landmarks", "ALI_IOS", "ALI_IOS/teeth",
    ]
    assert spec.options_when == {"mode": {
        "CBCT": ["ALI_CBCT", "ALI_CBCT/landmarks"],
        "IOS": ["ALI_IOS", "ALI_IOS/teeth"],
    }}, "a mode is offered the other engine's checkpoints"


def test_a_cycle_is_walked_once_rather_than_for_ever(tmp_path):
    """The registry refuses a cycle at RUN time, by name, with a message. At
    startup the same cycle would be an infinite walk, and a server that hangs
    on boot says nothing at all."""
    tools = _catalogue(
        tmp_path,
        {"name": "A", "supervisor": True, "calls": ["B"]},
        {"name": "B", "supervisor": True, "calls": ["A"]},
        {"name": "Self", "supervisor": True, "calls": ["Self"]},
    )
    assert list(tools["A"].arguments["stop_after"].choices) == ["B", "B/A"]
    assert list(tools["Self"].arguments["stop_after"].choices) == ["Self"]


# ----------------------------------------------------------------------
# What actually happens to a run
# ----------------------------------------------------------------------

DECLARING = """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "before.txt").write_text("done before the checkpoint")
        sup.declareQualityControl("halfway")
        (output_dir / "after.txt").write_text("done after it")
        return output_dir
"""

CALLER = """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        produced = sup.run("Leaf", scans=scans, tag="called")
        (output_dir / "chained.txt").write_text((Path(produced) / "leaf.txt").read_text())
        return output_dir
"""

# It declares a checkpoint under the very name its CALLER arms, which is the
# trap: a child that inherited its parent's stops would stop before writing.
LEAF_THAT_DECLARES = """
    def run(scans: Path, output_dir: Path, tag: str = "leaf", *, sup=None) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sup.declareQualityControl("Leaf")
        (output_dir / "leaf.txt").write_text(tag + ":" + str(scans))
        return output_dir
"""


def test_a_declared_checkpoint_nobody_armed_costs_the_run_nothing(tools_dir, tmp_path):
    make_tool(tools_dir, "Declaring", DECLARING)
    completed, result = run_job(tools_dir, "Declaring", tmp_path / "job",
                                {"scans": str(tmp_path)})
    assert completed.returncode == 0, completed.stderr
    assert "quality_control" not in result.get("result", {})
    output = tmp_path / "job" / "output"
    assert (output / "after.txt").is_file(), "the run stopped when nobody asked"


def test_arming_it_stops_the_run_and_says_where(tools_dir, tmp_path):
    make_tool(tools_dir, "Declaring", DECLARING)
    completed, result = run_job(
        tools_dir, "Declaring", tmp_path / "job",
        {"scans": str(tmp_path), "stop_after": ["halfway"]},
    )
    # Exit 0: the run did what it was told. A failure would be a 500.
    assert completed.returncode == 0, completed.stderr
    assert result["result"]["quality_control"] is True
    assert result["result"]["stopped_after"] == "halfway"

    output = tmp_path / "job" / "output"
    assert (output / "before.txt").is_file(), "what was produced was thrown away"
    assert not (output / "after.txt").exists(), "it carried on past the checkpoint"


def test_a_name_nobody_declared_stops_nothing(tools_dir, tmp_path):
    make_tool(tools_dir, "Declaring", DECLARING)
    completed, _result = run_job(
        tools_dir, "Declaring", tmp_path / "job",
        {"scans": str(tmp_path), "stop_after": ["elsewhere"]},
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "job" / "output" / "after.txt").is_file()


def test_a_chain_stops_after_the_callee_has_written(tools_dir, tmp_path):
    """The whole reason the boundary is AFTER: stopping before the call would
    leave nothing to look at."""
    make_tool(tools_dir, "Leaf", LEAF)
    make_tool(tools_dir, "Caller", CALLER)
    completed, result = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path), "stop_after": ["Leaf"]},
    )
    assert completed.returncode == 0, completed.stderr
    assert result["result"]["stopped_after"] == "Leaf"

    kept = tmp_path / "job" / "output" / "intermediate"
    produced = sorted(path.name for path in kept.rglob("leaf.txt"))
    assert produced == ["leaf.txt"], "the callee's output was not handed back"
    assert not (tmp_path / "job" / "output" / "chained.txt").exists(), (
        "the caller consumed what it was stopped to let somebody look at"
    )


def test_what_a_chain_produced_comes_back_without_a_second_box_ticked(tools_dir, tmp_path):
    """A reader who asked to stop and look is asking for exactly that."""
    make_tool(tools_dir, "Leaf", LEAF)
    make_tool(tools_dir, "Caller", CALLER)
    _completed, result = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path), "stop_after": ["Leaf"]},
    )
    assert result["result"]["produced"], "nothing was reported as produced"


def test_a_child_does_not_inherit_its_parent_s_stops(tools_dir, tmp_path):
    """A stop is a place in THIS tool's work.

    The leaf declares a checkpoint under the very name the caller armed. If
    the set travelled down, the leaf would stop before writing and the caller
    would read a quality-control record as its callee's result.
    """
    make_tool(tools_dir, "Leaf", LEAF_THAT_DECLARES)
    make_tool(tools_dir, "Caller", CALLER)
    completed, result = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path), "stop_after": ["Leaf"]},
    )
    assert completed.returncode == 0, completed.stderr
    assert result["result"]["stopped_after"] == "Leaf"
    kept = tmp_path / "job" / "output" / "intermediate"
    assert list(kept.rglob("leaf.txt")), "the leaf stopped instead of finishing"


# ----------------------------------------------------------------------
# A stop inside a nested call
# ----------------------------------------------------------------------

# It writes down what it was armed with, which is the only way to see from
# outside that a child got its own subset and not its parent's whole set.
REPORTING = """
    import os

    def run(scans: Path, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "armed.txt").write_text(os.environ.get("SADT_STOP_AFTER", ""))
        return output_dir
"""

# Two callees, so an entry addressed to one must not reach the other.
TWO_CALLS = """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sup.run("Leaf", scans=scans)
        sup.run("Other", scans=scans)
        return output_dir
"""

# The top of a three-level chain. It writes AFTER the call, so a file that
# exists is a run that carried on past a stop it should have unwound.
CALLS_MID = """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        produced = sup.run("Mid", scans=scans)
        (output_dir / "carried_on.txt").write_text(str(produced))
        return output_dir
"""


def _armed(job: Path, *slots) -> str:
    """What the tool in `slots` recorded of its own armed set."""
    where = job
    for slot in slots:
        where = where / "sup" / slot
    return (where / "output" / "armed.txt").read_text()


def test_a_child_is_armed_only_with_what_is_addressed_to_it(tools_dir, tmp_path):
    """`Leaf/inner` is for the Leaf call and `Other/elsewhere` for the other
    one; a bare name is a place in the CALLER's own work and descends to
    neither."""
    make_tool(tools_dir, "Leaf", REPORTING)
    make_tool(tools_dir, "Other", REPORTING)
    make_tool(tools_dir, "Caller", TWO_CALLS)
    job = tmp_path / "job"
    completed, _result = run_job(
        tools_dir, "Caller", job,
        {"scans": str(tmp_path),
         "stop_after": ["Leaf/inner", "Other/elsewhere", "Somewhere"]},
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(_armed(job, "01_Leaf")) == ["inner"]
    assert json.loads(_armed(job, "02_Other")) == ["elsewhere"]


def test_a_child_with_nothing_addressed_to_it_inherits_none_of_its_parent_s(
        tools_dir, tmp_path):
    """The environment a child is handed is a COPY of the parent's, so the
    parent's own armed set is there unless it is taken out. A name meant for
    one callee would otherwise be offered to its sibling."""
    make_tool(tools_dir, "Leaf", REPORTING)
    make_tool(tools_dir, "Other", REPORTING)
    make_tool(tools_dir, "Mid", TWO_CALLS)
    make_tool(tools_dir, "Caller", CALLS_MID)
    job = tmp_path / "job"
    completed, _result = run_job(
        tools_dir, "Caller", job,
        {"scans": str(tmp_path), "stop_after": ["Mid/Leaf/inner"]},
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(_armed(job, "01_Mid", "01_Leaf")) == ["inner"]
    assert _armed(job, "01_Mid", "02_Other") == "", (
        "the sibling was handed what Mid was armed with"
    )


def test_a_grandchild_s_stop_stops_the_whole_chain_and_says_where(
        tools_dir, tmp_path):
    """The delicate one. A stopped child exits 0 with a result.json exactly
    like a finished one, so without recognising it the caller carries on with
    a quality-control record where it expected the path its callee wrote."""
    make_tool(tools_dir, "Leaf", LEAF)
    make_tool(tools_dir, "Mid", CALLER)
    make_tool(tools_dir, "Caller", CALLS_MID)
    job = tmp_path / "job"
    completed, result = run_job(
        tools_dir, "Caller", job,
        {"scans": str(tmp_path), "stop_after": ["Mid/Leaf"]},
    )
    assert completed.returncode == 0, completed.stderr
    assert result["result"]["quality_control"] is True
    assert result["result"]["stopped_after"] == "Mid/Leaf", (
        "the root has to report the path that was armed, or nobody can arm it again"
    )
    assert not (job / "output" / "carried_on.txt").exists(), (
        "the root consumed what the chain was stopped to let somebody look at"
    )
    kept = job / "output" / "intermediate"
    assert list(kept.rglob("leaf.txt")), (
        "what the grandchild produced never reached the reader"
    )


def test_a_bare_name_stops_the_level_that_armed_it_and_no_deeper(
        tools_dir, tmp_path):
    """Unchanged from the day a stop was a bare tool name: `Mid` is the
    boundary after the Mid call, and Mid itself runs to the end."""
    make_tool(tools_dir, "Leaf", LEAF)
    make_tool(tools_dir, "Mid", CALLER)
    make_tool(tools_dir, "Caller", CALLS_MID)
    job = tmp_path / "job"
    completed, result = run_job(
        tools_dir, "Caller", job,
        {"scans": str(tmp_path), "stop_after": ["Mid"]},
    )
    assert completed.returncode == 0, completed.stderr
    assert result["result"]["stopped_after"] == "Mid"
    assert not (job / "output" / "carried_on.txt").exists()
    assert (job / "sup" / "01_Mid" / "output" / "chained.txt").is_file(), (
        "the bare name descended and stopped the callee too"
    )


def test_a_chain_nobody_armed_runs_to_the_end_untouched(tools_dir, tmp_path):
    """The resting state, and the one that matters most: three levels, no
    stop_after at all, nothing armed anywhere and nothing stopped."""
    make_tool(tools_dir, "Leaf", REPORTING)
    make_tool(tools_dir, "Other", REPORTING)
    make_tool(tools_dir, "Mid", TWO_CALLS)
    make_tool(tools_dir, "Caller", CALLS_MID)
    job = tmp_path / "job"
    completed, result = run_job(
        tools_dir, "Caller", job, {"scans": str(tmp_path)})
    assert completed.returncode == 0, completed.stderr
    assert "quality_control" not in result["result"]
    assert (job / "output" / "carried_on.txt").is_file()
    assert _armed(job, "01_Mid", "01_Leaf") == ""
    assert _armed(job, "01_Mid", "02_Other") == ""


def test_a_grandchild_stop_is_picked_up_again_from_the_root(tools_dir, tmp_path):
    """One run id, one resume. The root is the only level the server ever
    paused, so carrying on is the root re-entering -- and the checkpoint it is
    standing on is the only one disarmed."""
    make_tool(tools_dir, "Leaf", LEAF)
    make_tool(tools_dir, "Mid", CALLER)
    make_tool(tools_dir, "Caller", CALLS_MID)
    job = tmp_path / "job"
    params = {"scans": str(tmp_path), "stop_after": ["Mid/Leaf"]}

    _completed, result = run_job(tools_dir, "Caller", job, params)
    assert result["result"]["stopped_after"] == "Mid/Leaf"

    completed, result = _resume(tools_dir, "Caller", job, params)
    assert completed.returncode == 0, completed.stderr
    assert "quality_control" not in result["result"], "it stopped on the same breath"
    assert (job / "output" / "carried_on.txt").is_file(), "the chain did not finish"


# ----------------------------------------------------------------------
# Picking a stopped run up again
# ----------------------------------------------------------------------

COUNTING = """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        marks = Path(scans) / "ran.txt"
        marks.write_text(marks.read_text() + "x" if marks.exists() else "x")
        produced = sup.run("Leaf", scans=scans, tag="called")
        (output_dir / "chained.txt").write_text((Path(produced) / "leaf.txt").read_text())
        return output_dir
"""


def _resume(tools_dir, name, job_dir, params):
    return run_job(tools_dir, name, job_dir, params, env={"SADT_RESUME": "1"})


def test_a_resumed_chain_does_not_run_the_call_it_already_made(tools_dir, tmp_path):
    """The whole mechanism. Nothing preserves a Python stack across a process
    that exited, so the tool re-enters from the top -- and every call it
    already made answers from its slot instead of launching."""
    make_tool(tools_dir, "Leaf", LEAF)
    make_tool(tools_dir, "Caller", COUNTING)
    scans = tmp_path / "in"
    scans.mkdir()
    job = tmp_path / "job"
    params = {"scans": str(scans), "stop_after": ["Leaf"]}

    _completed, result = run_job(tools_dir, "Caller", job, params)
    assert result["result"]["stopped_after"] == "Leaf"
    leaf = job / "sup" / "01_Leaf" / "output" / "leaf.txt"
    assert leaf.is_file(), "the slot was moved away and the memo points at nothing"
    written_once = leaf.stat().st_mtime_ns

    completed, result = _resume(tools_dir, "Caller", job, params)
    assert completed.returncode == 0, completed.stderr
    assert "quality_control" not in result["result"], "it stopped on the same breath"
    assert (job / "output" / "chained.txt").is_file(), "the chain did not finish"
    assert leaf.stat().st_mtime_ns == written_once, "the callee ran a second time"
    # The caller DID re-enter -- that is the design, not an accident.
    assert (scans / "ran.txt").read_text() == "xx"


def test_what_the_reader_corrected_is_what_the_chain_carries_on_with(tools_dir, tmp_path):
    """The files a reader edited came off THEIR disk. A resume that trusted
    the server's copy would carry on with exactly the data they rejected."""
    make_tool(tools_dir, "Leaf", LEAF)
    make_tool(tools_dir, "Caller", COUNTING)
    scans = tmp_path / "in"
    scans.mkdir()
    job = tmp_path / "job"
    params = {"scans": str(scans), "stop_after": ["Leaf"]}

    run_job(tools_dir, "Caller", job, params)

    staged = job / "resume" / "01_Leaf"
    staged.mkdir(parents=True)
    (staged / "leaf.txt").write_text("corrected by a human")

    _completed, _result = _resume(tools_dir, "Caller", job, params)
    assert (job / "output" / "chained.txt").read_text() == "corrected by a human"


def test_a_second_checkpoint_still_stops_the_resumed_run(tools_dir, tmp_path):
    """Only the checkpoint it is standing on is disarmed."""
    make_tool(tools_dir, "Leaf", LEAF)
    make_tool(tools_dir, "Second", DECLARING.replace(
        'sup.declareQualityControl("halfway")',
        'sup.run("Leaf", scans=scans, tag="called")\n'
        '        sup.declareQualityControl("halfway")'))
    job = tmp_path / "job"
    params = {"scans": str(tmp_path), "stop_after": ["Leaf", "halfway"]}

    _completed, result = run_job(tools_dir, "Second", job, params)
    assert result["result"]["stopped_after"] == "Leaf"

    _completed, result = _resume(tools_dir, "Second", job, params)
    assert result["result"]["stopped_after"] == "halfway", (
        "continuing past the first stop skipped the second"
    )


def test_a_chain_that_took_another_path_is_not_handed_a_neighbour_s_answer(
        tools_dir, tmp_path):
    """A correction can change an earlier decision. Once the sequence
    differs, every later slot is suspect: they were numbered by a run that
    went another way."""
    make_tool(tools_dir, "Leaf", LEAF)
    make_tool(tools_dir, "Other", LEAF.replace("leaf.txt", "other.txt"))
    make_tool(tools_dir, "Caller", COUNTING)
    job = tmp_path / "job"
    run_job(tools_dir, "Caller", job, {"scans": str(tmp_path), "stop_after": ["Leaf"]})

    # The same slot, recorded for a different tool.
    memo = job / "sup" / "01_Leaf" / "memo.json"
    memo.write_text(json.dumps({"tool": "Other", "result": "nowhere"}))

    completed, _result = _resume(tools_dir, "Caller", job,
                                 {"scans": str(tmp_path), "stop_after": []})
    assert completed.returncode == 0, completed.stderr
    assert "running the rest for real" in completed.stderr
    assert (job / "output" / "chained.txt").is_file(), "it used the wrong answer"


def test_an_ordinary_run_reads_no_memo_at_all(tools_dir, tmp_path):
    """Without the resume flag a slot's record is inert, so a job directory
    that survived for any other reason cannot short-circuit a fresh run."""
    make_tool(tools_dir, "Leaf", LEAF)
    make_tool(tools_dir, "Caller", COUNTING)
    job = tmp_path / "job"
    params = {"scans": str(tmp_path), "stop_after": ["Leaf"]}
    run_job(tools_dir, "Caller", job, params)
    (job / "sup" / "01_Leaf" / "output" / "leaf.txt").write_text("stale")

    _completed, _result = run_job(tools_dir, "Caller", job,
                                  {"scans": str(tmp_path)})
    assert (job / "output" / "chained.txt").read_text() != "stale"


# ----------------------------------------------------------------------
# Over HTTP
# ----------------------------------------------------------------------

def test_the_two_sides_agree_on_what_a_resume_is_called():
    """Three names cross the process boundary, and `runner.py` is executed by
    a TOOL's interpreter so it cannot import the server."""
    from execution import dispatch, runner
    assert dispatch.RESUME_ENV == runner.RESUME_ENV
    assert dispatch.RESUME_DIRNAME == runner.RESUME_DIRNAME
    assert dispatch.SUP_DIRNAME == runner.SUP_DIRNAME


def test_resuming_a_run_that_never_stopped_is_a_404():
    """A run that finished, failed or expired cannot be carried on, and the
    answer says which of those it is rather than a bare 404."""
    from fastapi.testclient import TestClient
    import main

    # NOT as a context manager: entering one runs the app's lifespan and
    # leaving it runs the SHUTDOWN, which tears down state the rest of the
    # suite is still using. `test_security` builds its client the same way
    # and for the same reason.
    client = TestClient(main.app)
    response = client.post(
        "/runs/deadbeefdeadbeefdeadbeef/resume",
        headers={"Authorization": "Bearer " + main.settings.API_TOKEN},
    )
    assert response.status_code == 404
    assert "not stopped at a checkpoint" in response.json()["detail"]


class _AnyTool:
    """Enough of a tool for the extension check: it declares no argument, so
    the global ALLOWED_EXTENSIONS applies -- which is the fallback a real
    tool takes for a field name that is a SLOT rather than one of its own."""

    name = "Orchestrator"
    arguments: dict = {}


def _ran(tmp_path, *slots, produced=("points.mrk.json",)):
    """A job directory that got through `slots`, each having written
    `produced` -- which is what a correction is checked against."""
    for slot in slots:
        output = tmp_path / "sup" / slot / "output"
        output.mkdir(parents=True)
        for name in produced:
            (output / name).write_text("x")
    return str(tmp_path)


def test_a_correction_named_after_nothing_is_refused(tmp_path):
    """The slot becomes a directory, so it is matched before a path is built
    from it -- the discipline every id that arrives over HTTP follows here."""
    import main
    with pytest.raises(main.HTTPException) as raised:
        main._stage_corrections(_AnyTool(), _ran(tmp_path, "01_ALI_CBCT"),
                                _Form([("../etc", _Upload("x.mrk.json"))]))
    assert raised.value.status_code == 400


def test_a_correction_for_a_step_the_run_never_reached_is_refused(tmp_path):
    """A typo silently accepted is a resume the reader believes carries their
    work and does not."""
    import main
    with pytest.raises(main.HTTPException) as raised:
        main._stage_corrections(_AnyTool(), _ran(tmp_path, "01_ALI_CBCT"),
                                _Form([("02_ASO", _Upload("x.mrk.json"))]))
    assert raised.value.status_code == 400
    assert "01_ALI_CBCT" in raised.value.detail, "it does not say what DID run"


def test_a_correction_that_is_not_what_the_step_produced_is_refused(tmp_path):
    """Compared against that step's OWN output, not against the server's
    input whitelist -- which on this deployment is `.nii` and `.nii.gz`, and
    would have refused the very landmark file the reader was handed."""
    import main
    with pytest.raises(main.HTTPException) as raised:
        main._stage_corrections(_AnyTool(), _ran(tmp_path, "01_ALI_CBCT"),
                                _Form([("01_ALI_CBCT", _Upload("points.exe"))]))
    assert raised.value.status_code == 400
    assert ".mrk.json" in raised.value.detail, "it does not say what WAS produced"


def test_a_zip_is_always_allowed_because_a_folder_cannot_travel_otherwise(tmp_path):
    import main
    import zipfile, io as _io
    packed = _io.BytesIO()
    with zipfile.ZipFile(packed, "w") as archive:
        archive.writestr("points.mrk.json", "corrected")
    packed.seek(0)
    upload = _Upload("folder.zip")
    upload.file = packed
    assert main._stage_corrections(
        _AnyTool(), _ran(tmp_path, "01_ALI_CBCT"),
        _Form([("01_ALI_CBCT", upload)])) == ["01_ALI_CBCT"]
    assert (tmp_path / "resume" / "01_ALI_CBCT" / "points.mrk.json").is_file()


def test_an_empty_correction_is_refused(tmp_path):
    """An empty replacement would put NOTHING where the step's output was."""
    import main
    import zipfile, io as _io
    empty = _io.BytesIO()
    zipfile.ZipFile(empty, "w").close()
    empty.seek(0)
    upload = _Upload("nothing.zip")
    upload.file = empty
    with pytest.raises(main.HTTPException) as raised:
        main._stage_corrections(_AnyTool(), _ran(tmp_path, "01_ALI_CBCT"),
                                _Form([("01_ALI_CBCT", upload)]))
    assert raised.value.status_code == 400
    assert not (tmp_path / "resume" / "01_ALI_CBCT").exists(), "it left the hole"


def test_a_correction_is_staged_where_the_resume_reads_it(tmp_path):
    import main
    staged = main._stage_corrections(
        _AnyTool(), _ran(tmp_path, "01_ALI_CBCT"),
        _Form([("01_ALI_CBCT", _Upload("points.mrk.json"))]))
    assert staged == ["01_ALI_CBCT"]
    landed = tmp_path / "resume" / "01_ALI_CBCT" / "points.mrk.json"
    assert landed.read_bytes() == b"corrected"


def test_a_scalar_field_is_not_mistaken_for_a_correction(tmp_path):
    import main
    assert main._stage_corrections(
        _AnyTool(), _ran(tmp_path, "01_ALI_CBCT"),
        _Form([("note", "looks fine")])) == []


class _Upload(StarletteUploadFile):
    """A real UploadFile: the staging isinstance-checks against that class,
    and a stand-in would make the test pass while the endpoint ignored it."""

    def __init__(self, filename):
        super().__init__(file=io.BytesIO(b"corrected"), filename=filename)


class _Form:
    def __init__(self, items):
        self._items = items

    def multi_items(self):
        return list(self._items)


def test_a_detached_stop_puts_its_result_where_a_watcher_can_reach_it():
    """The `paused` event has to carry the reference, because nothing else
    can.

    A detached run's answer normally rides its TERMINAL event -- the response
    was sent as a 202 minutes earlier. A stopped run writes no terminal
    event, on purpose: one would tell the client to stop watching a run it is
    about to resume. So the reference to what the checkpoint produced lived
    in `_detached_run`'s local `response` and nowhere else, and a watcher sat
    until the stream was reaped. Found by reading the client against it, not
    by any test here.
    """
    import main
    from wire import runs

    appended = []
    original = runs.append
    runs.append = lambda run_id, **kwargs: appended.append(kwargs)
    try:
        main.runs.append("r", phase=runs.PHASE_PAUSED,
                         result=main._collectable(
                             main.JSONResponse({"quality_control": True,
                                                "result_ref": {"result_id": "abc"}})))
    finally:
        runs.append = original

    assert appended and appended[0]["phase"] == runs.PHASE_PAUSED
    assert appended[0]["result"]["result_ref"]["result_id"] == "abc", (
        "a watcher cannot collect what the checkpoint produced"
    )
