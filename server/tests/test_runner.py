"""runner.py, invoked the way the contract says it is invoked:

    <tool>/.venv/bin/python runner.py --job <job dir>/job.json

Driven here through a plain subprocess rather than through dispatch.py, so
what is being tested is the file the tool venvs execute -- the one half of this
repo that must keep working on every Python from 3.9 to 3.13 and must import
nothing that is not in the standard library.
"""

import json
import os
import subprocess
import sys

import pytest

os.environ.setdefault("API_TOKEN", "test-token")

from config import settings

RUNNER = settings.RUNNER_PATH


def _job(tmp_path, tool, params=None) -> str:
    """A job directory laid out the way dispatch.py lays one out."""
    job_dir = tmp_path / "job"
    (job_dir / "input").mkdir(parents=True)
    (job_dir / "output").mkdir(parents=True)
    job_path = job_dir / "job.json"
    job_path.write_text(
        json.dumps(
            {
                "job_id": "a1b2c3d4",
                "tool": tool,
                "job_dir": str(job_dir),
                "params": params if params is not None else {"a": 1, "b": 2},
            }
        )
    )
    return str(job_path)


def _run(python: str, job_path: str, **environment) -> subprocess.CompletedProcess:
    job_dir = os.path.dirname(job_path)
    child_environment = dict(os.environ)
    child_environment.update(
        {"SADT_API": "http://127.0.0.1:8000", "SADT_JOB_ID": "a1b2c3d4", "SADT_JOB_DIR": job_dir}
    )
    child_environment.update(environment)
    return subprocess.run(
        [python, RUNNER, "--job", job_path],
        capture_output=True,
        text=True,
        cwd=job_dir,
        env=child_environment,
    )


def _result(job_path: str):
    with open(os.path.join(os.path.dirname(job_path), "result.json")) as handle:
        return json.load(handle)


def test_it_writes_the_result_of_run(probe_python, probe_name, tmp_path):
    job_path = _job(tmp_path, probe_name)

    completed = _run(probe_python, job_path)

    assert completed.returncode == 0, completed.stderr
    assert _result(job_path)["result"]["total"] == 3


def test_a_raising_tool_records_which_failure_it_was(probe_python, probe_name, tmp_path):
    """There is no shared exception type to catch, so the class NAME is what
    tells the server whether to answer 422, 503 or 500."""
    job_path = _job(tmp_path, probe_name, params={"a": 1, "b": 2, "fail": True})

    completed = _run(probe_python, job_path)

    assert completed.returncode != 0
    assert "_dispatch_probe was asked to fail" in completed.stderr

    with open(os.path.join(os.path.dirname(job_path), "result.json")) as handle:
        recorded = json.load(handle)
    assert recorded == {
        "error": {"type": "RuntimeError", "message": "_dispatch_probe was asked to fail"}
    }
    # And never a result: the two are mutually exclusive.
    assert "result" not in recorded


def test_an_unknown_tool_names_what_it_tried(probe_python, tmp_path):
    job_path = _job(tmp_path, "not_a_tool")

    completed = _run(probe_python, job_path)

    assert completed.returncode != 0
    assert "not_a_tool" in completed.stderr


def test_a_malformed_job_file_is_refused(probe_python, probe_name, tmp_path):
    job_path = _job(tmp_path, probe_name)
    with open(job_path, "w") as handle:
        json.dump({"tool": probe_name, "job_dir": str(tmp_path / "job")}, handle)

    completed = _run(probe_python, job_path)

    assert completed.returncode != 0
    assert "params" in completed.stderr


def test_the_tool_folder_can_be_overridden(probe_python, probe_tool_dir, probe_name, tmp_path):
    """SADT_TOOL_DIR is what lets the runner be exercised from a checkout where
    the venv is not next to the sources."""
    job_path = _job(tmp_path, probe_name)

    completed = _run(probe_python, job_path, SADT_TOOL_DIR=probe_tool_dir)

    assert completed.returncode == 0, completed.stderr


def test_a_non_virtualenv_interpreter_says_so(probe_name, tmp_path):
    """Run by the wrong python, the runner cannot know which tool it is for.
    It has to say that rather than import something at random."""
    if sys.prefix != sys.base_prefix:
        pytest.skip("This test needs an interpreter that is not itself in a virtualenv")
    job_path = _job(tmp_path, probe_name)

    completed = _run(sys.executable, job_path)

    assert completed.returncode != 0
    assert "not a virtualenv" in completed.stderr


def test_it_imports_nothing_outside_the_standard_library(probe_python, probe_name, tmp_path):
    """The probe's venv has no third-party package at all -- not even pip. That
    the runner works there is the proof that it depends on nothing of the
    server's."""
    job_path = _job(tmp_path, probe_name)

    completed = _run(probe_python, job_path)

    assert completed.returncode == 0, completed.stderr
