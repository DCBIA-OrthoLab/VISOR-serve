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
