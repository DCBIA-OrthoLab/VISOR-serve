"""The runner hands a tool a supervisor, and it calls another tool through it.

Everything here runs the REAL runner as a subprocess, against tools built on
the fly in a temporary TOOLS_DIR -- no fixtures pretending to be venvs, because
the thing under test is precisely that the callee gets its own interpreter.

The tools are one-file packages with no dependencies, so `.venv` is a symlink
farm around `sys.executable`; that is enough for the runner, which only ever
asks for `<tool>/.venv/bin/python`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "execution" / "runner.py"


def make_tool(tools_dir: Path, name: str, body: str) -> Path:
    """A runnable tool: `src/sadt_<name>/__init__.py` plus a venv pointing here."""
    package = tools_dir / name / "src" / f"sadt_{name.lower()}"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from pathlib import Path\n\n" + textwrap.dedent(body), encoding="utf-8"
    )
    binaries = tools_dir / name / ".venv" / "bin"
    binaries.mkdir(parents=True)
    # The runner derives the tool folder from sys.prefix, so the interpreter has
    # to LOOK like it lives in this venv. A symlink does that without building
    # one: sys.prefix follows the link's directory, not its target.
    (binaries / "python").symlink_to(sys.executable)
    (tools_dir / name / ".venv" / "pyvenv.cfg").write_text(
        "home = {}\ninclude-system-site-packages = true\n".format(
            os.path.dirname(sys.executable)
        ),
        encoding="utf-8",
    )
    return tools_dir / name


def run_job(tools_dir: Path, name: str, job_dir: Path, params: dict):
    """Invoke the runner exactly as the server does, and hand back result.json."""
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "output").mkdir(exist_ok=True)
    job_path = job_dir / "job.json"
    job_path.write_text(
        json.dumps(
            {"job_id": "t", "tool": name, "job_dir": str(job_dir), "params": params}
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [str(tools_dir / name / ".venv" / "bin" / "python"), str(RUNNER),
         "--job", str(job_path)],
        capture_output=True, text=True, cwd=str(job_dir),
    )
    result = {}
    if (job_dir / "result.json").is_file():
        result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    return completed, result


LEAF = """
    def run(scans: Path, output_dir: Path, tag: str = "leaf") -> Path:
        \"\"\"Write one file and return where it went.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "leaf.txt").write_text(tag + ":" + str(scans))
        return output_dir
"""

CALLER = """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Call the leaf tool, then write what it produced.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if sup is None:
            raise RuntimeError("no supervisor was injected")
        sup.progress(0.5, "calling Leaf")
        produced = sup.run("Leaf", scans=scans, output_dir=sup.tmp / "leaf", tag="called")
        (output_dir / "chained.txt").write_text((Path(produced) / "leaf.txt").read_text())
        return output_dir
"""


@pytest.fixture
def tools_dir(tmp_path):
    folder = tmp_path / "tools"
    folder.mkdir()
    return folder


def test_a_tool_that_asks_for_no_supervisor_is_given_none(tools_dir, tmp_path):
    make_tool(tools_dir, "Leaf", LEAF)

    completed, result = run_job(
        tools_dir, "Leaf", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(result["result"], "leaf.txt").read_text().startswith("leaf:")


def test_a_tool_declaring_sup_receives_one_and_reaches_the_other_tool(tools_dir, tmp_path):
    """The whole point: `*, sup` in the signature, a real second venv on the
    other end, and no import between the two."""
    make_tool(tools_dir, "Leaf", LEAF)
    make_tool(tools_dir, "Caller", CALLER)

    completed, result = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(result["result"], "chained.txt").read_text() == "called:{}".format(
        tmp_path / "in"
    )


def test_the_nested_run_gets_its_own_job_directory(tools_dir, tmp_path):
    """So a chain can be read afterwards: which tool ran, in what order, and
    what it was asked for are all on disk."""
    make_tool(tools_dir, "Leaf", LEAF)
    make_tool(tools_dir, "Caller", CALLER)

    run_job(tools_dir, "Caller", tmp_path / "job",
            {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")})

    nested = tmp_path / "job" / "sup" / "01_Leaf"
    assert (nested / "job.json").is_file()
    assert json.loads((nested / "job.json").read_text())["tool"] == "Leaf"
    assert (nested / "result.json").is_file()


def test_progress_and_log_reach_stderr(tools_dir, tmp_path):
    """The runner owns logging, so a supervised call has to surface through it
    -- a nested tool's output is the only sign of life during a long run."""
    make_tool(tools_dir, "Leaf", LEAF)
    make_tool(tools_dir, "Caller", CALLER)

    completed, _ = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
    )

    assert "50% calling Leaf" in completed.stderr
    assert "running 'Leaf'" in completed.stderr


def test_a_failing_nested_tool_names_itself(tools_dir, tmp_path):
    """A chain that breaks has to say which link broke."""
    make_tool(tools_dir, "Leaf", """
    def run(scans: Path, output_dir: Path, tag: str = "leaf") -> Path:
        \"\"\"Always fail.\"\"\"
        raise ValueError("the leaf refused")
    """)
    make_tool(tools_dir, "Caller", CALLER)

    job_dir = tmp_path / "job"
    completed, result = run_job(
        tools_dir, "Caller", job_dir,
        {"scans": str(tmp_path / "in"), "output_dir": str(job_dir / "output")},
    )

    assert completed.returncode != 0
    assert "Leaf" in completed.stderr
    # The reason travels in the PARENT's error, not only in the child's output:
    # whatever runs the parent may be capturing and trimming stderr, so "see
    # above" is a promise the supervisor cannot keep. This was found by running
    # a real chain through the server, where the child's traceback vanished.
    assert "ValueError: the leaf refused" in completed.stderr
    assert (job_dir / "result.json").is_file()
    error = json.loads((job_dir / "result.json").read_text())["error"]
    assert "the leaf refused" in error["message"]


def test_an_undeployed_tool_says_it_is_not_installed(tools_dir, tmp_path):
    make_tool(tools_dir, "Caller", CALLER)  # no Leaf at all

    completed, _ = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
    )

    assert completed.returncode != 0
    assert "not deployed here" in completed.stderr


def test_a_tool_calling_itself_is_stopped(tools_dir, tmp_path):
    """A cycle would otherwise fork until the machine gives out."""
    make_tool(tools_dir, "Loop", """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Call itself forever.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sup.run("Loop", scans=scans, output_dir=sup.tmp / "again")
        return output_dir
    """)

    completed, _ = run_job(
        tools_dir, "Loop", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
    )

    assert completed.returncode != 0
    assert "nested more than" in completed.stderr


def test_an_empty_optional_path_stays_empty(tools_dir, tmp_path):
    """`Path("")` is `PosixPath(".")` -- the current directory, and truthy.

    Coercing the "not supplied" default of an optional path therefore hands the
    tool a real directory: ASO read an unset `landmarks=""` as a supplied
    landmark folder and walked its whole checkout.
    """
    make_tool(tools_dir, "Optional", """
    def run(output_dir: Path, extra: Path = "") -> Path:
        \"\"\"Report whether the optional path arrived as absence.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "seen.txt").write_text("absent" if not extra else str(extra))
        return output_dir
    """)

    completed, result = run_job(
        tools_dir, "Optional", tmp_path / "job",
        {"output_dir": str(tmp_path / "job" / "output"), "extra": ""},
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(result["result"], "seen.txt").read_text() == "absent"
