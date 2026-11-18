"""A battery somebody composed: what it may send, and what it never writes down.

Three things are pinned here, each a way a custom battery could quietly lie:

* the plan is the one asked for -- configurations alternated when run one at a
  time, the counts and widths as typed, and every ceiling holding;
* anything the tool would refuse is refused BEFORE a process starts, naming
  the configuration and the argument, rather than surfacing as a 422 the
  battery reports as the tool failing;
* a bench input leaves the server as a path for the battery and never as a
  name in anything a reader sees: the plan's public half, the summary.
"""

import json
import os

import pytest
from fastapi.testclient import TestClient

import benchmark_client
import benchmark_jobs
import benchmark_presets as presets
import main
from config import settings


def arg(required=False, kind="str", selectable=None, initial=None, choices=None, scope=None):
    return {"required": required, "type": kind, "server_selectable": selectable,
            "initial": initial, "choices": choices, "selectable_scope": scope}


SCHEMAS = {
    "SegTool": {"arguments": {
        "input": arg(required=True, kind="path", selectable="testfile"),
        "model": arg(kind="str", selectable="model"),
        "structures": arg(kind="multichoice", choices={"MAND": True, "MAX": True, "CB": False}),
        "mode": arg(kind="choice", choices={"fast": True, "fine": False}),
        "num_workers": arg(kind="int", initial=1),
        "tile_step_size": arg(kind="float", initial=0.5),
        "smooth": arg(kind="bool", initial=False),
    }},
    "PairTool": {"arguments": {
        "t1": arg(required=True, kind="folder"),
        "label": arg(required=True, kind="str"),
    }},
}
HOSTED = {"SegTool": {"testfiles": ["scan.nii.gz"], "models": ["bundle"]}, "PairTool": {}}
BENCH = {"SegTool": [{"name": "hard_case", "kind": "folder", "size": 10}],
         "PairTool": [{"name": "cohort", "kind": "folder", "size": 10}]}


def plan(**spec):
    spec.setdefault("configs", [{"tool": "SegTool", "params": {"input": "scan.nii.gz"}}])
    return presets.build_custom_plan(spec, SCHEMAS, HOSTED.get, BENCH.get)


def refused(**spec) -> str:
    with pytest.raises(presets.PresetError) as caught:
        plan(**spec)
    return str(caught.value)


# --- the plan is the one asked for ---------------------------------------------

def test_one_at_a_time_alternates_the_configurations():
    built = plan(repeats=3, configs=[
        {"tool": "SegTool", "params": {"input": "scan.nii.gz", "num_workers": 1}},
        {"tool": "SegTool", "params": {"input": "scan.nii.gz", "num_workers": 2}},
    ])
    runs = built["arms"][0]["runs"]
    assert [run["config"] for run in runs] == ["A", "B", "A", "B", "A", "B"]
    assert [run["params"]["num_workers"] for run in runs[:2]] == ["1", "2"]
    assert built["arms"][0]["concurrency"] == 1 and built["total_runs"] == 6


def test_in_parallel_runs_rounds_at_each_width_of_the_ladder():
    built = plan(shape="parallel", repeats=2, ladder=[1, 2, 4], stagger=5)
    assert [(a["concurrency"], len(a["runs"])) for a in built["arms"]] == [(1, 2), (2, 4), (4, 8)]
    assert all(a["stagger"] == 5 for a in built["arms"])


def test_values_travel_as_the_form_fields_run_reads():
    params = plan(configs=[{"tool": "SegTool", "params": {
        "input": "scan.nii.gz", "model": "bundle", "structures": ["MAND", "CB"],
        "mode": "fine", "num_workers": "3", "tile_step_size": 0.7, "smooth": True}}])["arms"][0]["runs"][0]["params"]
    assert params == {"input": "scan.nii.gz", "model": "bundle", "structures": "MAND,CB",
                      "mode": "fine", "num_workers": "3", "tile_step_size": "0.7", "smooth": "true"}


@pytest.mark.parametrize("spec,needle", [
    ({"repeats": 61}, "repeats"),
    ({"shape": "parallel", "concurrency": 17}, "width"),
    ({"shape": "parallel", "repeats": 8, "concurrency": 8}, "64 runs"),
    ({"shape": "parallel", "ladder": [1, 2, 3, 4, 5, 6, 7]}, "ladder"),
    ({"stagger": 601}, "stagger"),
    ({"configs": [{"tool": "SegTool", "params": {"input": "scan.nii.gz"}}] * 5}, "At most 4"),
])
def test_every_ceiling_holds(spec, needle):
    assert needle in refused(**spec)


# --- refused before anything starts, saying what ------------------------------

@pytest.mark.parametrize("config,needle", [
    ({"tool": "Nope"}, "no tool called 'Nope'"),
    ({"tool": "SegTool", "params": {"input": "scan.nii.gz", "wat": 1}}, "has no argument 'wat'"),
    ({"tool": "SegTool", "params": {"input": "other.nii.gz"}}, "is not hosted for 'input'"),
    ({"tool": "SegTool", "params": {"input": "scan.nii.gz", "model": "stolen"}}, "is not hosted for 'model'"),
    ({"tool": "SegTool", "params": {"input": "scan.nii.gz", "mode": "warp"}}, "not one of its options"),
    ({"tool": "SegTool", "params": {"input": "scan.nii.gz", "structures": ["TOOTH"]}}, "not one of its options"),
    ({"tool": "SegTool", "params": {"input": "scan.nii.gz", "num_workers": "two"}}, "whole number"),
    ({"tool": "SegTool", "params": {}}, "'input' is required"),
    ({"tool": "SegTool", "params": {}, "bench": {"input": "absent"}}, "no bench input 'absent'"),
    ({"tool": "SegTool", "params": {"input": "scan.nii.gz"}, "bench": {"mode": "hard_case"}}, "not a file input"),
    ({"tool": "PairTool", "params": {"t1": "/etc", "label": "x"}}, "is a file"),
])
def test_what_the_tool_would_refuse_is_refused_here(config, needle):
    message = refused(configs=[config])
    assert needle in message
    assert message.startswith("A")  # the configuration is named: "A (SegTool): ..."


def test_a_bench_input_satisfies_a_required_file():
    built = plan(configs=[{"tool": "PairTool", "params": {"label": "x"}, "bench": {"t1": "cohort"}}])
    run = built["arms"][0]["runs"][0]
    assert run["bench"] == {"t1": "cohort"} and "t1" not in run["params"]


# --- through the server: names stop at the plan the battery reads ---------------

client = TestClient(main.app)


@pytest.fixture
def admin(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ADMIN_TOKEN", "operator-secret")
    # Under the tool's DATA folder, which is not always its name.
    slug = main.deployment_config.data_slug("Example_Tool")
    case = tmp_path / slug / "bench" / "PATIENT_NAME_SHOULD_NOT_LEAK.csv"
    case.parent.mkdir(parents=True)
    case.write_text("a,b\n1,2\n")
    monkeypatch.setattr(main.data_store, "_root", str(tmp_path))
    started = {}

    def fake_start(plan, summary_dir, base, token, scratch):
        started["plan"] = plan
        return {"preset": plan["preset"], "state": "running"}
    monkeypatch.setattr(benchmark_jobs, "start", fake_start)
    return {"X-Admin-Token": "operator-secret"}, started, str(case)


def test_the_form_lists_bench_inputs_to_the_admin_only(admin):
    headers, _, _ = admin
    assert client.get("/benchmark/form/Example_Tool",
                      headers={"Authorization": f"Bearer {settings.API_TOKEN}"}).status_code == 401
    form = client.get("/benchmark/form/Example_Tool", headers=headers).json()
    assert [entry["name"] for entry in form["bench"]] == ["PATIENT_NAME_SHOULD_NOT_LEAK.csv"]
    assert form["bench_folder"].endswith("/bench")


def test_a_bench_input_reaches_the_battery_as_a_path_and_nowhere_as_a_name(admin):
    headers, started, path = admin
    response = client.post("/benchmark/run", headers=headers, json={"custom": {
        "configs": [{"tool": "Example_Tool",
                     "params": {"label": "x", "threshold": 0.5},
                     "bench": {"input": "PATIENT_NAME_SHOULD_NOT_LEAK.csv"}}]}})
    assert response.status_code == 200, response.text
    battery = started["plan"]
    run = battery["arms"][0]["runs"][0]
    assert run["uploads"] == {"input": os.path.realpath(path)}
    assert "bench" not in run
    # The half that is written into the summary carries the argument, not the case.
    assert "PATIENT_NAME" not in json.dumps(battery["configs"])


def test_a_bench_name_that_climbs_out_is_refused(admin):
    headers, _, _ = admin
    response = client.post("/benchmark/run", headers=headers, json={"custom": {
        "configs": [{"tool": "Example_Tool", "params": {"label": "x", "threshold": 0.5},
                     "bench": {"input": "../../etc/passwd"}}]}})
    assert response.status_code == 422


# --- the battery's side ---------------------------------------------------------

def test_a_bench_folder_is_packed_once_with_its_name_as_the_single_root(tmp_path):
    case = tmp_path / "case_folder"
    (case / "sub").mkdir(parents=True)
    (case / "a.nii.gz").write_bytes(b"x" * 10)
    (case / "sub" / "b.nii.gz").write_bytes(b"y" * 5)
    benchmark_client._packed.clear()
    first = benchmark_client._pack(str(case), str(tmp_path / "scratch"))
    again = benchmark_client._pack(str(case), str(tmp_path / "scratch"))
    assert first is again
    archive, name, shape = first
    assert name == "case_folder.zip" and shape == {"kind": "folder", "files": 2, "bytes": 15}
    import zipfile
    with zipfile.ZipFile(archive) as handle:
        assert sorted(handle.namelist()) == ["case_folder/a.nii.gz", "case_folder/sub/b.nii.gz"]


def test_the_comparison_reads_each_configuration_on_its_own_compute():
    def row(config, seconds, running):
        return {"config": config, "tool": "SegTool", "status": "ok", "seconds": seconds,
                "spans": [{"phase": "queued_gpu", "start": 0, "end": 5},
                          {"phase": "running", "start": 5, "end": 5 + running}]}
    table = benchmark_client._compare([row("A", 20, 10), row("B", 14, 6), row("A", 22, 12), row("B", 13, 6)])
    assert [entry["config"] for entry in table] == ["A", "B"]
    assert table[0]["running"]["mean"] == 11 and table[1]["running"]["mean"] == 6
    assert table[0]["seconds"]["sd"] is not None
    assert benchmark_client._compare([row("A", 1, 1)]) == []
