"""Reading back what a step said it did.

Every path here is tested twice over: what it answers when the report is
there and well-formed, and what it answers when it is not. The second half
matters more. Narrowing a replay is an optimisation -- a report that cannot
be read must cost a slower run, never a lost correction -- so every failure
has to fall towards doing MORE work.
"""

import json
import os

import pytest

from execution import reports


def _report(directory, payload, name="run_report.json"):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return directory


ONE_RUN = {
    "tool": "ALI_CBCT",
    "cases": {
        "Pat_0002": {"input": "Pat_0002.nii.gz",
                     "produced": ["Pat_0002_lm_Pred.mrk.json"]},
        "Pat_0003": {"input": "Pat_0003.nii.gz",
                     "produced": ["Pat_0003_lm_Pred.mrk.json"]},
        "Pat_0004": {"input": "Pat_0004.nii.gz",
                     "produced": ["Pat_0004_lm_Pred.mrk.json"]},
    },
}


# ---------------------------------------------------------------------------
# Finding it
# ---------------------------------------------------------------------------

def test_the_report_is_found_whatever_the_tool_called_it(tmp_path):
    """Four tools write `run_report.json` and the rest name it after
    themselves -- `ASO_report.json`, `CLIC_report.json`. The server serves
    both without a table of tool names."""
    where = _report(str(tmp_path / "aso"), ONE_RUN, name="ASO_report.json")
    assert reports.read(where)["tool"] == "ALI_CBCT"


def test_run_report_wins_when_a_tool_wrote_both(tmp_path):
    directory = str(tmp_path / "both")
    _report(directory, {"tool": "general"}, name="run_report.json")
    _report(directory, {"tool": "specific"}, name="Thing_report.json")
    assert reports.read(directory)["tool"] == "general"


def test_a_nested_report_is_not_the_callers(tmp_path):
    """A report describes the results beside it. One a level down belongs to
    a CALLEE, and the caller's report does not speak for it."""
    directory = str(tmp_path / "step")
    os.makedirs(os.path.join(directory, "intermediate", "01_Leaf"))
    _report(os.path.join(directory, "intermediate", "01_Leaf"), ONE_RUN)
    assert reports.read(directory) == {}


# ---------------------------------------------------------------------------
# Narrowing a replay
# ---------------------------------------------------------------------------

def test_the_inputs_of_the_marked_cases_are_what_a_replay_is_fed(tmp_path):
    report = reports.read(_report(str(tmp_path / "s"), ONE_RUN))
    assert reports.inputs_for(report, {"Pat_0002", "Pat_0004"}) == [
        "Pat_0002.nii.gz", "Pat_0004.nii.gz"]


def test_the_cases_nobody_marked_are_left_out(tmp_path):
    report = reports.read(_report(str(tmp_path / "s"), ONE_RUN))
    assert reports.produced_by(report, {"Pat_0003"}) == [
        "Pat_0003_lm_Pred.mrk.json"]


def test_a_file_is_traced_back_to_its_case(tmp_path):
    """What a resume is narrowed BY: the reader sends back the files they
    corrected, and these are the cases those files are about."""
    report = reports.read(_report(str(tmp_path / "s"), ONE_RUN))
    assert reports.cases_of(report, [
        "/jobs/x/sup/01_ALI_CBCT/output/Pat_0004_lm_Pred.mrk.json",
        "Pat_0002_lm_Pred.mrk.json",
    ]) == ["Pat_0004", "Pat_0002"]


def test_a_file_is_matched_on_its_name_not_its_path(tmp_path):
    """One file reaches this server through a job directory, a staging
    directory and an archive -- three absolute paths, one name."""
    report = reports.read(_report(str(tmp_path / "s"), ONE_RUN))
    assert reports.case_of(report, "/anywhere/at/all/Pat_0002_lm_Pred.mrk.json") \
        == "Pat_0002"


def test_an_input_names_its_case_too(tmp_path):
    """A reader may send back a corrected INPUT, not only an output."""
    report = reports.read(_report(str(tmp_path / "s"), ONE_RUN))
    assert reports.case_of(report, "Pat_0003.nii.gz") == "Pat_0003"


def test_one_file_under_two_cases_is_named_once(tmp_path):
    """A registration writes one file for a PAIR, so a tool may legitimately
    record it under both timepoints."""
    shared = {"cases": {
        "P1_T1": {"input": "a.nii.gz", "produced": ["P1_Reg.nii.gz"]},
        "P1_T2": {"input": "b.nii.gz", "produced": ["P1_Reg.nii.gz"]},
    }}
    report = reports.read(_report(str(tmp_path / "s"), shared))
    assert reports.produced_by(report, {"P1_T1", "P1_T2"}) == ["P1_Reg.nii.gz"]


# ---------------------------------------------------------------------------
# Everything that can go wrong, and the direction it fails in
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload, why", [
    ({}, "a report with nothing in it"),
    ({"cases": []}, "cases as a list, which an older tool wrote"),
    ({"scans": {"P1": {"input": "a", "produced": []}}}, "the older vocabulary"),
    ({"patients": {"P1": {"outputs": []}}}, "the other older vocabulary"),
    ({"cases": {"P1": "not a dict"}}, "an entry that is not an entry"),
])
def test_a_report_that_says_nothing_narrows_nothing(tmp_path, payload, why):
    """Which means the replay is the whole cohort: slower, never wrong.

    Reading `scans` as if it were `cases` would be assuming a shape nobody
    promised -- the two mean the same thing to a human and the server has no
    business guessing that.
    """
    report = reports.read(_report(str(tmp_path / "s"), payload))
    assert reports.cases(report) == {}, why
    assert reports.inputs_for(report, {"P1"}) == []
    assert reports.cases_of(report, ["anything.nii.gz"]) == []


def test_no_report_at_all_is_not_an_error(tmp_path):
    empty = str(tmp_path / "nothing")
    os.makedirs(empty)
    assert reports.read(empty) == {}
    assert reports.find(empty) is None


def test_an_unreadable_report_is_not_an_error(tmp_path):
    directory = str(tmp_path / "broken")
    os.makedirs(directory)
    with open(os.path.join(directory, "run_report.json"), "w") as handle:
        handle.write("{ this is not json")
    assert reports.read(directory) == {}


def test_a_directory_that_is_not_there_is_not_an_error(tmp_path):
    assert reports.read(str(tmp_path / "absent")) == {}
    assert reports.find(str(tmp_path / "absent")) is None


def test_a_report_that_is_a_list_is_not_a_report(tmp_path):
    directory = str(tmp_path / "list")
    os.makedirs(directory)
    with open(os.path.join(directory, "run_report.json"), "w") as handle:
        json.dump(["not", "a", "report"], handle)
    assert reports.read(directory) == {}
