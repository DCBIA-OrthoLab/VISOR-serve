"""Keeping what a chain produced, and who decides where a callee writes.

A tool that calls another tool gets a `keep_intermediate` check box WITHOUT
declaring it: `describe.py` already publishes the tools it calls, so the server
adds the argument for anything whose `calls` is non-empty, and `runner.py`
takes it back out before `run()` and does the collecting itself. Adding a new
orchestrating tool therefore costs nothing -- which is the property these tests
exist to keep.

The other half is the contract that makes collection possible at all: where a
supervised tool writes is the SUPERVISOR's, not the caller's, exactly as it is
the server's over HTTP. A caller that pointed a callee at its own scratch
directory had those results deleted with it, before anything could collect
them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from registry import schema_tool
from registry.deployment import DeploymentConfig

from test_supervisor import LEAF, make_tool, run_job, tools_dir  # noqa: F401


# ----------------------------------------------------------------------
# What the server publishes
# ----------------------------------------------------------------------

def _tool(tmp_path, **schema):
    base = {
        "name": "Orchestrator",
        "arguments": {"scans": {"type": "path"}},
        "returns": "path",
    }
    base.update(schema)
    return schema_tool.SchemaTool(str(tmp_path), base, DeploymentConfig({}).for_tool("X"))


def test_a_tool_that_calls_another_gets_the_steps_without_declaring_them(tmp_path):
    """The whole point: `sup.run("Leaf", ...)` is the only thing anyone writes.

    One option per step, named after the tool that ran it -- which is what the
    registry has already checked names something real."""
    tool = _tool(tmp_path, supervisor=True, calls=["Leaf", "Other"])

    spec = tool.arguments["keep_intermediate"]
    assert spec.type == "multichoice"
    assert spec.required is False
    assert spec.choices == {"Leaf": False, "Other": False}, (
        "the ordinary run wants the result, not the workings"
    )
    assert spec.select_all is True


def test_the_steps_keep_the_order_the_chain_declares_them_in(tmp_path):
    """A reader matches an option to a step, so the order has to be the tool's
    own rather than whatever a set iterated in."""
    tool = _tool(tmp_path, supervisor=True, calls=["Zebra", "Alpha", "Medium"])

    assert list(tool.arguments["keep_intermediate"].choices) == [
        "Zebra", "Alpha", "Medium"
    ]


def test_the_box_sits_just_above_the_outputs(tmp_path):
    """Section order IS argument order, so where it lands in the dict is where
    the box lands on the panel. What a run returns is read together: where it
    stops, the steps to keep, then where everything is written."""
    tool = _tool(
        tmp_path,
        supervisor=True,
        calls=["Leaf"],
        arguments={
            "scans": {"type": "path"},
            "output_suffix": {"type": "str", "required": False, "section": "Outputs"},
        },
    )

    assert list(tool.arguments) == [
        "scans", "stop_after", "keep_intermediate", "output_suffix",
    ]


def test_a_tool_naming_no_output_argument_still_gets_it_last(tmp_path):
    """The client adds its own Outputs box after every declared section, so the
    end of the list is still just above it."""
    tool = _tool(tmp_path, supervisor=True, calls=["Leaf"])

    assert list(tool.arguments)[-1] == "keep_intermediate"


def test_a_tool_that_calls_nobody_gets_nothing(tmp_path):
    """CLIC, Crown_Seg and Surg_Mov_Pred all declare `sup` and call no one.

    Keyed on `calls` rather than on `supervisor` for exactly this: a check box
    that can only ever collect nothing is worse than no check box.
    """
    assert "keep_intermediate" not in _tool(tmp_path, supervisor=True).arguments
    assert "keep_intermediate" not in _tool(tmp_path).arguments


def test_the_tool_may_name_the_check_box_for_what_it_produces(tmp_path):
    """"Keep intermediate results" is true of every chain and says nothing.

    Only the tool knows its chain produces landmarks rather than meshes, so a
    layout may name the injected argument even though run() never declares it.
    """
    tool = _tool(
        tmp_path,
        supervisor=True,
        calls=["ALI_CBCT"],
        injected_layout={"keep_intermediate": {
            "label": "Steps to keep",
            "option_help": {"ALI_CBCT": "The landmarks the prediction placed."},
        }},
    )

    spec = tool.arguments["keep_intermediate"]
    assert spec.label == "Steps to keep"
    assert spec.option_help == {"ALI_CBCT": "The landmarks the prediction placed."}


def test_without_a_layout_the_generic_wording_stands(tmp_path):
    spec = _tool(tmp_path, supervisor=True, calls=["Leaf"]).arguments["keep_intermediate"]
    assert spec.label == "Steps to keep"
    assert spec.section == "Intermediate results", (
        "a section of its own: this decides what comes back, which is not a knob "
        "on how the run is computed"
    )


# ----------------------------------------------------------------------
# What the runner does with it
# ----------------------------------------------------------------------

# Passes no `output_dir`: the supervisor supplies it, and the caller learns
# where the callee wrote from the value it returned.
CALLER = """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Call the leaf tool once, and say where its output went.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        produced = sup.run("Leaf", scans=scans, tag="called")
        (output_dir / "where.txt").write_text(str(produced))
        return output_dir
"""

# Names a directory of its own, the way ASO and ALI_IOS used to. The supervisor
# drops it, so this tool STILL finds its callee's results through the return
# value and the collection still works.
BOSSY_CALLER = """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Try to name where the callee writes.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mine = Path(sup.tmp) / "i_said_here"
        produced = sup.run("Leaf", scans=scans, output_dir=mine, tag="called")
        (output_dir / "where.txt").write_text(str(produced))
        return output_dir
"""

TWICE = """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Call the leaf tool twice.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sup.run("Leaf", scans=scans, tag="first")
        sup.run("Leaf", scans=scans, tag="second")
        return output_dir
"""


def _chain(tools_dir, body=CALLER):
    make_tool(tools_dir, "Leaf", LEAF)
    make_tool(tools_dir, "Caller", body)


def test_the_callee_is_given_an_output_directory_it_never_asked_for(tools_dir, tmp_path):
    """A supervised call arrives with no `output_dir`, and the tool declares one
    as a required argument. The callee's own runner fills it from its own job
    directory -- the same thing dispatch does for a request."""
    _chain(tools_dir)
    completed, result = run_job(
        tools_dir, "Caller", tmp_path / "job", {"scans": str(tmp_path)}
    )

    assert completed.returncode == 0, completed.stderr
    written = (tmp_path / "job" / "output" / "where.txt").read_text()
    assert written.endswith(os.path.join("sup", "01_Leaf", "output"))


def test_a_caller_naming_the_directory_is_overruled(tools_dir, tmp_path):
    """Silently, and on purpose: no client can name `output_dir` over HTTP
    either. What a caller passed is dropped, and the callee's results land where
    the supervisor put them, which is the one place collection can find them."""
    _chain(tools_dir, BOSSY_CALLER)
    completed, _ = run_job(
        tools_dir, "Caller", tmp_path / "job", {"scans": str(tmp_path)}
    )

    assert completed.returncode == 0, completed.stderr
    written = (tmp_path / "job" / "output" / "where.txt").read_text()
    assert "i_said_here" not in written
    assert written.endswith(os.path.join("sup", "01_Leaf", "output"))


def test_nothing_is_kept_unless_it_is_asked_for(tools_dir, tmp_path):
    _chain(tools_dir)
    run_job(tools_dir, "Caller", tmp_path / "job", {"scans": str(tmp_path)})

    assert not (tmp_path / "job" / "output" / "intermediate").exists()


def test_what_the_chain_produced_comes_back_when_it_is_asked_for(tools_dir, tmp_path):
    """Under `intermediate/`, one folder per call, named as the supervisor named
    it -- so a reader can match a result to the order the chain ran in."""
    _chain(tools_dir)
    completed, _ = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path), "keep_intermediate": True},
    )

    assert completed.returncode == 0, completed.stderr
    kept = tmp_path / "job" / "output" / "intermediate" / "01_Leaf" / "leaf.txt"
    assert kept.is_file()
    assert kept.read_text().startswith("called:")


def test_the_tool_never_sees_the_argument(tools_dir, tmp_path):
    """No tool declares `keep_intermediate`, so `run(**params)` would raise a
    TypeError if it were passed through. The runner takes it back out."""
    _chain(tools_dir)
    completed, result = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path), "keep_intermediate": True},
    )

    assert completed.returncode == 0, completed.stderr
    assert "keep_intermediate" not in completed.stderr
    assert result.get("result")


def test_every_call_in_the_chain_is_kept_separately(tools_dir, tmp_path):
    _chain(tools_dir, TWICE)
    completed, _ = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path), "keep_intermediate": True},
    )

    assert completed.returncode == 0, completed.stderr
    kept = tmp_path / "job" / "output" / "intermediate"
    assert sorted(p.name for p in kept.iterdir()) == ["01_Leaf", "02_Leaf"]
    assert (kept / "01_Leaf" / "leaf.txt").read_text().startswith("first:")
    assert (kept / "02_Leaf" / "leaf.txt").read_text().startswith("second:")


def test_only_the_steps_that_were_ticked_come_back(tools_dir, tmp_path):
    """The reason it is a multichoice and not a check box: a chain of four
    produces four directories, and a caller checking one prediction does not
    want the other three in the archive."""
    _chain(tools_dir, TWICE)
    completed, _ = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path), "keep_intermediate": {"Leaf": True}},
    )

    assert completed.returncode == 0, completed.stderr
    kept = tmp_path / "job" / "output" / "intermediate"
    # Both calls are to Leaf here, so both match -- what this pins is that a
    # ticked name is matched against the tool, not against the folder's number.
    assert sorted(p.name for p in kept.iterdir()) == ["01_Leaf", "02_Leaf"]


def test_a_step_left_unticked_is_not_collected(tools_dir, tmp_path):
    """An unticked option arrives as False rather than missing -- a multichoice
    sends its complete state -- so this is the shape a real request has."""
    _chain(tools_dir)
    completed, _ = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path), "keep_intermediate": {"Leaf": False}},
    )

    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / "job" / "output" / "intermediate").exists()


def test_a_name_nobody_ran_keeps_nothing_and_does_not_fail(tools_dir, tmp_path):
    """A facade passes the union of its engines' steps down; the engine that
    actually ran made only some of them."""
    _chain(tools_dir)
    completed, _ = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path), "keep_intermediate": {"SomethingElse": True}},
    )

    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / "job" / "output" / "intermediate").exists()


def test_a_tool_that_calls_nothing_is_unaffected(tools_dir, tmp_path):
    """The flag can reach any tool -- a facade may pass it down. One that made
    no supervised call collects nothing and still succeeds."""
    make_tool(tools_dir, "Leaf", LEAF)
    completed, _ = run_job(
        tools_dir, "Leaf", tmp_path / "job",
        {"scans": str(tmp_path), "keep_intermediate": True},
    )

    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / "job" / "output" / "intermediate").exists()
