"""The two execution paths, tested for the parts that need no server.

The most valuable test here is `test_the_job_contract_matches_the_servers`: the
local path reproduces `server/execution/dispatch.py` by hand, and a change on
that side that this side does not follow would make B1's local arm measure
something the server no longer does. Nothing else in the harness would notice.
"""

from __future__ import annotations

import ast
import json
import os
import zipfile

import pytest

from benchmarks.execution import local, remote
from benchmarks.settings import REPO_ROOT, LocalSpec, ToolSpec

DISPATCH_SOURCE = os.path.join(REPO_ROOT, "server", "execution", "dispatch.py")
RUNNER_SOURCE = os.path.join(REPO_ROOT, "server", "execution", "runner.py")


def _module_constants(path: str) -> dict:
    """Top-level `NAME = "literal"` assignments, without importing.

    The server modules cannot be imported here: `config.py` builds a pydantic
    Settings at import time and would demand an API_TOKEN, which is exactly what
    a reviewer inspecting the harness does not have.
    """
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
            found[target.id] = node.value.value
    return found


@pytest.mark.skipif(not os.path.isfile(DISPATCH_SOURCE), reason="server/ is not in this checkout")
def test_the_job_contract_matches_the_servers():
    """The file names the two sides agree on. A rename on the server's side
    that is not followed here would silently make the local arm meaningless."""
    dispatch = _module_constants(DISPATCH_SOURCE)
    assert local.JOB_FILE == dispatch["JOB_FILE"]
    assert local.RESULT_FILE == dispatch["RESULT_FILE"]
    assert local.JOB_OUTPUT_DIRNAME == dispatch["JOB_OUTPUT_DIRNAME"]
    assert local.STDOUT_LOG == dispatch["STDOUT_LOG"]
    assert local.STDERR_LOG == dispatch["STDERR_LOG"]


@pytest.mark.skipif(not os.path.isfile(RUNNER_SOURCE), reason="server/ is not in this checkout")
def test_the_runner_still_takes_the_job_argument():
    """The exec line the local path builds is
    `<python> <runner.py> --job <job.json>`; if the runner's flag changed, the
    local path would fail on every tool at once."""
    with open(RUNNER_SOURCE, encoding="utf-8") as handle:
        body = handle.read()
    assert '"--job"' in body


def _spec(mode="host", **overrides) -> LocalSpec:
    fields = {
        "mode": mode,
        "container": "a-container",
        "container_user": "sadt",
        "container_tools_dir": "/tools",
        "container_runner": "/opt/sadt/server/execution/runner.py",
        "container_jobs_dir": "/jobs",
        "container_data_dir": "/DATA",
        "host_tools_dir": "/checkout/tools",
        "host_runner": "/repo/server/execution/runner.py",
        "host_jobs_dir": "/tmp/jobs",
        "host_data_dir": "/repo/DATA",
    }
    fields.update(overrides)
    return LocalSpec(**fields)


def test_the_interpreter_path_follows_the_nesting():
    """ALI_CBCT lives at tools/ALI/ALI_CBCT, not tools/ALI_CBCT."""
    runner = local.LocalRunner(_spec("container"))
    tool = ToolSpec(name="ALI_CBCT", local=local.__dict__ and None)
    tool.local = type("L", (), {"folder": "ALI/ALI_CBCT", "package": "sadt_ali_cbct",
                                "supported": True, "reason": ""})()
    assert runner.interpreter(tool) == "/tools/ALI/ALI_CBCT/.venv/bin/python"
    assert runner.source_dir(tool) == "/tools/ALI/ALI_CBCT/src"


def test_a_tool_with_no_folder_cannot_be_run_locally():
    runner = local.LocalRunner(_spec())
    tool = ToolSpec(name="Test_Tool")
    with pytest.raises(local.LocalPathError, match="local.folder"):
        runner.interpreter(tool)


def test_the_data_path_mirrors_the_stores_layout():
    runner = local.LocalRunner(_spec("container"))
    assert runner.data_path("CrownSeg", "model", "b.pth") == "/DATA/CrownSeg/models/b.pth"
    assert runner.data_path("AMASSS", "testfile", "s.nii.gz") == "/DATA/AMASSS/testfiles/s.nii.gz"


def test_an_input_already_under_data_is_referenced_not_copied(tmp_path):
    """Copying a 207 MB volume per repetition would dominate the timing it is
    supposed to measure, and fill a disk with 59 GB free."""
    data_root = tmp_path / "DATA"
    scan = data_root / "AMASSS" / "testfiles" / "s.nii.gz"
    scan.parent.mkdir(parents=True)
    scan.write_bytes(b"x")
    runner = local.LocalRunner(_spec("container", host_data_dir=str(data_root)))
    assert runner.stage_input(str(scan), "/jobs/j1", "scans") == "/DATA/AMASSS/testfiles/s.nii.gz"


def test_staging_a_missing_input_is_refused_before_docker_is_touched(tmp_path):
    runner = local.LocalRunner(_spec("container", host_data_dir=str(tmp_path)))
    with pytest.raises(local.LocalPathError, match="does not exist"):
        runner.stage_input(str(tmp_path / "absent.nii.gz"), "/jobs/j1", "scans")


def test_the_host_mode_hands_the_tool_the_path_it_was_given(tmp_path):
    scan = tmp_path / "s.nii.gz"
    scan.write_bytes(b"x")
    runner = local.LocalRunner(_spec("host"))
    assert runner.stage_input(str(scan), str(tmp_path), "scans") == str(scan)


def test_the_environment_drops_the_api_token(monkeypatch):
    """dispatch.py pops it: it is the server's credential and the tool venvs
    hold third-party code."""
    monkeypatch.setenv("API_TOKEN", "secret")
    runner = local.LocalRunner(_spec("host"))
    job = local.LocalRun(job_id="j1", job_dir="/tmp/j1", output_dir="/tmp/j1/output")
    environment = runner.environment(job, timeout=None)
    assert "API_TOKEN" not in environment
    assert environment["SADT_JOB_ID"] == "j1"


def test_a_timeout_becomes_an_absolute_deadline():
    """A duration would restart at every hop of a supervised chain and give a
    five-deep chain five times the budget."""
    runner = local.LocalRunner(_spec("host"))
    job = local.LocalRun(job_id="j1", job_dir="/tmp/j1", output_dir="/tmp/j1/output")
    environment = runner.environment(job, timeout=60.0)
    assert float(environment["SADT_SUPERVISOR_DEADLINE"]) > 0


def test_the_job_file_holds_exactly_the_four_declared_fields(tmp_path):
    runner = local.LocalRunner(_spec("host", host_jobs_dir=str(tmp_path)))
    tool = ToolSpec(name="Widget")
    job = runner.prepare_job(tool, {})
    runner.write_job_file(job, tool, {"alpha": 1, "output_dir": job.output_dir})
    with open(os.path.join(job.job_dir, local.JOB_FILE), encoding="utf-8") as handle:
        document = json.load(handle)
    assert sorted(document) == ["job_dir", "job_id", "params", "tool"]
    assert document["tool"] == "Widget"


# ----------------------------------------------------------------------
# The remote client
# ----------------------------------------------------------------------

def test_arguments_are_stringified_the_way_the_slicer_client_does():
    assert remote._stringify(True) == "true"
    assert remote._stringify(False) == "false"
    assert remote._stringify(1.5) == "1.5"
    assert remote._stringify({"summary": True, "preview": False}) == json.dumps(
        {"summary": True, "preview": False}
    )
    assert remote._stringify([1, 2]) == "[1, 2]"


def test_already_compressed_inputs_are_not_gzipped():
    assert not remote.worth_compressing("scan.nii.gz")
    assert not remote.worth_compressing("bundle.zip")
    assert remote.worth_compressing("mesh.vtk")
    assert remote.worth_compressing("table.csv")


def test_a_folder_input_is_packed_and_a_file_is_not(tmp_path):
    folder = tmp_path / "T1"
    folder.mkdir()
    (folder / "a.vtk").write_bytes(b"mesh")
    (folder / "b.nii.gz").write_bytes(b"\x1f\x8b")
    workspace = tmp_path / "ws"
    workspace.mkdir()

    archive = remote._pack_if_folder(str(folder), str(workspace), "AREG_IOSCBCT", "ios")
    assert archive.endswith("AREG_IOSCBCT_ios.zip")
    with zipfile.ZipFile(archive) as handle:
        assert sorted(handle.namelist()) == ["a.vtk", "b.nii.gz"]
        # An already-compressed member is STORED; deflating it costs CPU for
        # nothing.
        assert handle.getinfo("b.nii.gz").compress_type == zipfile.ZIP_STORED
        assert handle.getinfo("a.vtk").compress_type == zipfile.ZIP_DEFLATED

    plain = tmp_path / "one.vtk"
    plain.write_bytes(b"mesh")
    assert remote._pack_if_folder(str(plain), str(workspace), "T", "a") == str(plain)


def test_the_uploads_field_name_matches_the_servers(tmp_path):
    main_source = os.path.join(REPO_ROOT, "server", "main.py")
    if not os.path.isfile(main_source):
        pytest.skip("server/ is not in this checkout")
    with open(main_source, encoding="utf-8") as handle:
        body = handle.read()
    assert f'_UPLOADS_FIELD = "{remote.UPLOADS_FIELD}"' in body
    assert f'_RESULT_DELIVERY_HEADER = "{remote.RESULT_DELIVERY_HEADER}"' in body


def test_host_mode_reports_a_structurally_zero_wrapper_cost():
    """Host mode starts the interpreter directly, exactly as dispatch.py does,
    so there is no wrapper to subtract and the record must say so."""
    runner = local.LocalRunner(_spec("host"))
    overhead = runner.measure_exec_overhead()
    assert overhead["median_seconds"] == 0.0
    assert overhead["repetitions"] == 0
    assert "no wrapper" in overhead["note"]
