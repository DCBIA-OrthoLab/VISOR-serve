"""Splitting a cohort for a tool whose inputs are PAIRED -- without the server
knowing how anything pairs.

A tool declares, in its schema, which of its arguments hold the same subjects
(`"paired": {"axes": [...]}`), and answers `pairs()` from its own code. The
server's part, tested here:

* publish a plan for it under `paired_batch` -- never under `batch`, which an
  older client would honour by splitting one folder and sending the other whole;
* per MODE for a facade, since the client knows the mode before it splits;
* relay a list of file NAMES to the tool's `pairs()`, in the tool's own
  interpreter, and hand back its answer, refusing anything that is not a plain
  list of relative names.
"""

import json
import os
import sys
import textwrap

import pytest
from fastapi.testclient import TestClient

import main
import registry
from config import settings
from execution import dispatch, runner
from registry import conventions, facade, schema_tool
from registry.deployment import DeploymentConfig, ToolDeployment

client = TestClient(main.app)
AUTH = {"Authorization": f"Bearer {settings.API_TOKEN}"}

TWO = {"t1": {"type": "path", "required": True}, "t2": {"type": "path", "required": True}}
ONE = {"input": {"type": "path", "required": True}}


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def test_a_tool_declaring_paired_inputs_gets_a_paired_plan(make_tool_folder):
    folder = make_tool_folder("paired_tool", arguments=TWO, paired={"axes": ["t1", "t2"]})
    tool = schema_tool.load_tool(folder, DeploymentConfig({}))
    assert tool.batch is None
    assert tool.paired_batch == {"axes": ["t1", "t2"], "max_mb": settings.BATCH_MAX_MB,
                                 "max_files": settings.BATCH_MAX_FILES}


def test_without_the_declaration_a_paired_tool_still_travels_whole(make_tool_folder):
    folder = make_tool_folder("undeclared", arguments=TWO)
    assert schema_tool.load_tool(folder, DeploymentConfig({})).paired_batch is None


@pytest.mark.parametrize("paired", [
    {"axes": ["t1"]},
    {"axes": ["t1", "nope"]},
    {"axes": "t1,t2"},
    "t1,t2",
])
def test_a_declaration_the_arguments_do_not_support_publishes_nothing(paired):
    assert conventions.paired_batch_plan(paired, TWO, ToolDeployment(), (400, 25)) is None


def test_batch_false_turns_a_paired_plan_off_too():
    declared = ToolDeployment(batch_enabled=False)
    assert conventions.paired_batch_plan({"axes": ["t1", "t2"]}, TWO, declared, (400, 25)) is None


def test_a_facade_publishes_a_paired_plan_per_mode(make_tool_folder):
    config = DeploymentConfig({})
    tools = {
        "Paired_A": schema_tool.load_tool(make_tool_folder("Paired_A", arguments=TWO,
                                                           paired={"axes": ["t1", "t2"]}), config),
        "Paired_B": schema_tool.load_tool(make_tool_folder("Paired_B", arguments=TWO,
                                                           paired={"axes": ["t1", "t2"]}), config),
        "Other": schema_tool.load_tool(make_tool_folder("Other", arguments=TWO), config),
    }
    composed = facade.compose("Reg", {"A": "Paired_A", "B": "Paired_B", "C": "Other"}, tools)
    assert composed.batch is None
    assert composed.paired_batch["by"] == facade.MODE_ARGUMENT
    assert set(composed.paired_batch["modes"]) == {"A", "B"}


# ---------------------------------------------------------------------------
# GET /tools and POST /tools/{tool}/pairs
# ---------------------------------------------------------------------------

PAIRS_SOURCE = textwrap.dedent('''
    """A tool that pairs by the part of a name before its first underscore."""

    PAIRED = {"axes": ["t1", "t2"]}


    def run(t1, t2, output_dir):
        return output_dir


    def pairs(t1, t2):
        if "boom.txt" in t1:
            raise ValueError("this cohort cannot be paired: boom")
        key = lambda name: name.split("_")[0]
        both = sorted({key(n) for n in t1} & {key(n) for n in t2})
        return {
            "groups": [{"key": k, "keys": [k], "entries": {
                "t1": [n for n in t1 if key(n) == k], "t2": [n for n in t2 if key(n) == k]}}
                for k in both],
            "shared": {},
            "unpaired": {"t1": sorted({key(n) for n in t1} - set(both))},
        }
''')


@pytest.fixture
def paired_tool(tmp_path, monkeypatch):
    """A packaged tool with a real `pairs()`, run through the real runner with
    this interpreter standing in for its virtualenv."""
    folder = tmp_path / "tools" / "Pairy"
    package = folder / "src" / "sadt_pairy"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(PAIRS_SOURCE)

    class Pairy:
        name = "Pairy"
        paired_batch = {"axes": ["t1", "t2"], "max_mb": 400, "max_files": 25}

    monkeypatch.setitem(registry.TOOLS, "Pairy", Pairy())
    monkeypatch.setattr(dispatch, "tool_interpreter", lambda name: sys.executable)
    monkeypatch.setenv(runner.TOOL_DIR_ENV, str(folder))
    monkeypatch.setattr(settings, "TEMP_DIR", str(tmp_path / "temp"))
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    return folder


def test_the_tools_own_pairs_answers_and_nothing_is_left_behind(paired_tool):
    response = client.post("/tools/Pairy/pairs", headers=AUTH, json={
        "inputs": {"t1": ["A_t1.nii.gz", "B_t1.nii.gz", "C_t1.nii.gz"],
                   "t2": ["A_t2.nii.gz", "B_t2.nii.gz"]}})
    assert response.status_code == 200, response.text
    answer = response.json()
    assert answer["axes"] == ["t1", "t2"]
    assert [g["key"] for g in answer["groups"]] == ["A", "B"]
    assert answer["unpaired"] == {"t1": ["C"]}
    assert os.listdir(settings.TEMP_DIR) == []


def test_the_tools_own_refusal_reaches_the_client(paired_tool):
    response = client.post("/tools/Pairy/pairs", headers=AUTH, json={
        "inputs": {"t1": ["boom.txt"], "t2": ["x"]}})
    assert response.status_code == 422
    assert "cannot be paired" in response.json()["detail"]


@pytest.mark.parametrize("inputs", [
    {"t1": ["a"]},
    {"t1": ["a"], "t2": ["b"], "t3": ["c"]},
    {"t1": "a", "t2": ["b"]},
    {"t1": ["../../etc/passwd"], "t2": ["b"]},
    {"t1": ["/etc/passwd"], "t2": ["b"]},
])
def test_anything_but_lists_of_relative_names_for_exactly_the_axes_is_refused(paired_tool, inputs):
    assert client.post("/tools/Pairy/pairs", headers=AUTH, json={"inputs": inputs}).status_code == 422


def test_a_tool_that_does_not_pair_says_so():
    response = client.post("/tools/Test_Tool/pairs", headers=AUTH, json={"inputs": {}})
    assert response.status_code == 404


def test_it_needs_the_token():
    assert client.post("/tools/Pairy/pairs", json={"inputs": {}}).status_code == 401


def test_the_plan_is_published_under_its_own_key(make_tool_folder, monkeypatch):
    """`paired_batch`, never `batch`: an older client honours `batch` by
    splitting one folder and sending the other whole."""
    folder = make_tool_folder("Paired_Pub", arguments=TWO, paired={"axes": ["t1", "t2"]})
    monkeypatch.setitem(registry.TOOLS, "Paired_Pub", schema_tool.load_tool(folder, DeploymentConfig({})))
    published = {t["name"]: t for t in client.get("/tools").json()}
    assert published["Paired_Pub"]["paired_batch"]["axes"] == ["t1", "t2"]
    assert "batch" not in published["Paired_Pub"]
    assert "paired_batch" not in published["Test_Tool"]
