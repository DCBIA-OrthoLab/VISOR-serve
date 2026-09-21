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
import time
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


def run_job(tools_dir: Path, name: str, job_dir: Path, params: dict, env: dict = None):
    """Invoke the runner exactly as the server does, and hand back result.json.

    `env` adds to the inherited environment, which is how the server passes the
    variables a run carries -- SADT_PROGRESS_FILE among them.
    """
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
        env=dict(os.environ, **(env or {})),
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

# It names an `output_dir` for the callee and the supervisor drops it -- where a
# supervised tool writes is not the caller's to choose. Left written that way on
# purpose: it is how ASO and ALI_IOS were spelled, and it has to keep working,
# because the caller learns where its callee wrote from the RETURN value.
# `test_keep_intermediate.py` is where that overruling is asserted directly.
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
    # Refused by NAME on the first call, not by the depth cap on the fifth:
    # four processes and whatever each of them loaded are not spent finding out
    # what the chain already says.
    assert "Supervised call cycle" in completed.stderr
    assert "Loop -> Loop" in completed.stderr


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


# ---------------------------------------------------------------------------
# parent / root, and the chain that refuses a cycle by name
# ---------------------------------------------------------------------------

def test_a_child_job_records_its_parent_and_its_root(tools_dir, tmp_path):
    """Every nested job says which call made it and which request started it.

    Not what makes nesting deadlock-free -- a nested call is a subprocess of its
    parent and never re-enters the server's admission queue, so there is no
    queue it could wait in. This is traceability, and the key a VRAM budget
    would later be applied to.
    """
    make_tool(tools_dir, "Leaf", LEAF)
    make_tool(tools_dir, "Parent", """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Call one child.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sup.run("Leaf", scans=scans, output_dir=sup.tmp / "leaf")
        return output_dir
    """)

    job_dir = tmp_path / "job"
    completed, _ = run_job(
        tools_dir, "Parent", job_dir,
        {"scans": str(tmp_path / "in"), "output_dir": str(job_dir / "output")},
    )
    assert completed.returncode == 0, completed.stderr

    child_jobs = sorted(job_dir.glob("sup/*/job.json"))
    assert len(child_jobs) == 1
    record = json.loads(child_jobs[0].read_text(encoding="utf-8"))
    assert record["parent"] == "t"
    assert record["root"] == "t"
    assert record["tool"] == "Leaf"


def test_a_cycle_two_tools_long_is_named_not_counted(tools_dir, tmp_path):
    """A -> B -> A is refused at the third call, naming all three.

    The depth cap would also stop it, but only after five processes and
    whatever each of them imported, and its message names a number rather than
    the mistake.
    """
    make_tool(tools_dir, "Ping", """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Call Pong.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sup.run("Pong", scans=scans, output_dir=sup.tmp / "pong")
        return output_dir
    """)
    make_tool(tools_dir, "Pong", """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Call Ping back.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sup.run("Ping", scans=scans, output_dir=sup.tmp / "ping")
        return output_dir
    """)

    completed, _ = run_job(
        tools_dir, "Ping", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
    )
    assert completed.returncode != 0
    assert "Supervised call cycle" in completed.stderr
    assert "Ping -> Pong -> Ping" in completed.stderr


def test_the_result_carries_no_vram_figure_when_the_tool_never_touched_torch(tools_dir, tmp_path):
    """`sys.modules.get("torch")` and nothing else: a tabular tool pays nothing.

    The absence IS the assertion. Importing torch to ask would cost seconds and
    a CUDA context on every run of every tool that has no use for either.
    """
    make_tool(tools_dir, "Leaf", LEAF)
    _, result = run_job(
        tools_dir, "Leaf", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
    )
    assert "result" in result
    assert "peak_vram_bytes" not in result


def test_a_supervised_tool_is_found_inside_a_grouping_folder(tools_dir, tmp_path):
    """`tools/ALI/ALI_CBCT` is the tool `ALI_CBCT`, not `ALI/ALI_CBCT`.

    A grouping folder holds several tools and is not one itself, so the name a
    caller uses no longer matches a top-level directory. Without this lookup,
    the split that created ALI_CBCT and ALI_IOS silently broke every
    `sup.run("ALI_CBCT", ...)` -- loudly in fact, RunnerError naming the tool,
    which is the right failure but still a broken chain.
    """
    make_tool(tools_dir, "Leaf", LEAF)
    # Move it under a group, exactly as tools/ALI/ALI_CBCT sits.
    group = tools_dir / "Group"
    group.mkdir()
    (tools_dir / "Leaf").rename(group / "Leaf")

    make_tool(tools_dir, "Caller", """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Call a tool that lives under a grouping folder.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sup.run("Leaf", scans=scans, output_dir=sup.tmp / "leaf")
        return output_dir
    """)

    completed, _ = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
    )
    assert completed.returncode == 0, completed.stderr


def test_a_nested_tool_can_call_a_tool_outside_its_group(tools_dir, tmp_path):
    """The caller may itself live under a grouping folder.

    `_tool_dir()` gives the caller's own folder and the tools root was taken as
    its parent -- correct for tools/AMASSS, wrong for tools/ALI/ALI_CBCT, where
    the parent is the group and its siblings would be the only tools reachable.
    Latent while nested tools call nothing; AREG_IOSCBCT is a nested tool whose
    whole job is calling the others.
    """
    make_tool(tools_dir, "Leaf", LEAF)
    make_tool(tools_dir, "Caller", """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Call a tool that sits OUTSIDE this tool's group.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sup.run("Leaf", scans=scans, output_dir=sup.tmp / "leaf")
        return output_dir
    """)
    # Move only the CALLER under a group; Leaf stays at the top level.
    group = tools_dir / "Group"
    group.mkdir()
    (tools_dir / "Caller").rename(group / "Caller")

    job_dir = tmp_path / "job"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "output").mkdir(exist_ok=True)
    job_path = job_dir / "job.json"
    job_path.write_text(json.dumps({
        "job_id": "t", "tool": "Caller", "job_dir": str(job_dir),
        "params": {"scans": str(tmp_path / "in"),
                   "output_dir": str(job_dir / "output")},
    }), encoding="utf-8")
    completed = subprocess.run(
        [str(group / "Caller" / ".venv" / "bin" / "python"), str(RUNNER),
         "--job", str(job_path)],
        capture_output=True, text=True, cwd=str(job_dir),
    )
    assert completed.returncode == 0, completed.stderr


# ---------------------------------------------------------------------------
# Three levels, and the deadline that crosses them
# ---------------------------------------------------------------------------

def test_a_chain_three_levels_deep_runs_and_carries_its_chain(tools_dir, tmp_path):
    """A -> B -> C, which is the shape AREG_IOSCBCT -> ASO -> ALI_CBCT has.

    Everything proven before this was two levels. Three is the first time a
    supervised child is itself a supervisor: the depth and the chain have to
    survive two hops, and the chain check must NOT fire on a tool that appears
    twice in the tree on different branches -- which is exactly what
    AREG_IOSCBCT does, calling ALI_CBCT directly and again through ASO.
    """
    make_tool(tools_dir, "Leaf", LEAF)
    make_tool(tools_dir, "Middle", """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Call the leaf, one level down.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sup.run("Leaf", scans=scans, output_dir=sup.tmp / "leaf")
        return output_dir
    """)
    make_tool(tools_dir, "Top", """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Call the middle tool AND the leaf directly -- two branches.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sup.run("Leaf", scans=scans, output_dir=sup.tmp / "direct")
        sup.run("Middle", scans=scans, output_dir=sup.tmp / "middle")
        return output_dir
    """)

    job_dir = tmp_path / "job"
    completed, _ = run_job(
        tools_dir, "Top", job_dir,
        {"scans": str(tmp_path / "in"), "output_dir": str(job_dir / "output")},
    )
    assert completed.returncode == 0, completed.stderr

    # The grandchild's job file exists, which is what says two hops happened.
    grandchild = sorted(job_dir.glob("sup/*/sup/*/job.json"))
    assert grandchild, "no third level was reached"
    record = json.loads(grandchild[0].read_text(encoding="utf-8"))
    assert record["tool"] == "Leaf"
    assert record["root"] == "t"


def test_the_deadline_crosses_levels_and_is_not_restarted(tools_dir, tmp_path):
    """A child gets the time its parent has LEFT, not a fresh budget.

    The deadline travels as an absolute instant on the monotonic clock, whose
    origin is per-boot rather than per-process. Passing a duration instead would
    restart the budget at every hop, and a three-deep chain would quietly get
    three times what the operator granted.
    """
    make_tool(tools_dir, "Slow", """
    import time
    def run(scans: Path, output_dir: Path) -> Path:
        \"\"\"Outlive any sensible budget.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        time.sleep(60)
        return output_dir
    """)
    make_tool(tools_dir, "Caller", """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Call something slower than the budget allows.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sup.run("Slow", scans=scans, output_dir=sup.tmp / "slow")
        return output_dir
    """)

    job_dir = tmp_path / "job"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "output").mkdir(exist_ok=True)
    job_path = job_dir / "job.json"
    job_path.write_text(json.dumps({
        "job_id": "t", "tool": "Caller", "job_dir": str(job_dir),
        "params": {"scans": str(tmp_path / "in"),
                   "output_dir": str(job_dir / "output")},
    }), encoding="utf-8")

    environment = dict(os.environ)
    # Three seconds from now, as the server would set it.
    environment["SADT_SUPERVISOR_DEADLINE"] = repr(time.monotonic() + 3.0)

    started = time.monotonic()
    completed = subprocess.run(
        [str(tools_dir / "Caller" / ".venv" / "bin" / "python"), str(RUNNER),
         "--job", str(job_path)],
        capture_output=True, text=True, cwd=str(job_dir), env=environment,
    )
    elapsed = time.monotonic() - started

    assert completed.returncode != 0
    assert "ran out of the job's remaining time" in completed.stderr, completed.stderr
    # Killed on the parent's budget, not given a fresh 60s of its own.
    assert elapsed < 30, f"took {elapsed:.0f}s, so the budget was not inherited"


def test_the_caller_facing_names_match_the_servers_own_table():
    """runner.py is stdlib-only and cannot import main.py, so the two lists are
    written twice. If they drift, a child's 422 silently becomes a 500."""
    import main
    from execution import runner

    assert set(runner.CALLER_FACING_ERRORS) == set(main.TOOL_ERROR_STATUS)


def test_a_tool_can_call_one_in_another_catalogue(tools_dir, tmp_path, monkeypatch):
    """TOOLS_DIR may name several catalogues, and a chain may cross them.

    The runner derives its root from where its own tool lives, which is the one
    thing the invocation fixes. With one catalogue that is the whole answer.
    With several it is not: the caller here sits in the first and its child in
    the second, two directories apart, and looking beside itself finds nothing.
    The server's boot check would have declared the chain complete -- it reads
    every catalogue -- so the failure would land at run time, on a tool the
    deployment was told it had.
    """
    make_tool(tools_dir, "Caller", """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Call a tool served from a different catalogue.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sup.run("Leaf", scans=scans, output_dir=sup.tmp / "leaf")
        return output_dir
    """)
    # Deliberately NOT under tools_dir or its parent: either would be reachable
    # by the caller's own two roots, and the test would pass without the lookup
    # it exists to pin.
    second = tmp_path / "elsewhere" / "catalogue-two"
    second.mkdir(parents=True)
    make_tool(second, "Leaf", LEAF)
    monkeypatch.setenv("TOOLS_DIR", str(second))

    completed, _ = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
    )
    assert completed.returncode == 0, completed.stderr


def test_progress_from_a_chain_lands_in_one_file_with_its_depth(tools_dir, tmp_path):
    """`SADT_PROGRESS_FILE` is inherited, so a child appends to its parent's
    file one level deeper. That is the whole implementation of chain progress:
    a panel shows `AREG -> ASO 30%` without the client knowing what a chain is,
    and no level had to be told about any other.

    The supervisor is the COMPATIBILITY path here, not the recommended one --
    a tool appends to that file itself (see RUN_PROGRESS.md). It is what these
    two fixtures happen to have, and it must keep working.
    """
    make_tool(tools_dir, "Leaf", """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Report from one level down.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sup.progress(0.9, "leaf: scan 9 of 10")
        return output_dir
    """)
    make_tool(tools_dir, "Caller", """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Report, then call the leaf, then report again.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sup.progress(0.1, "caller: starting")
        sup.run("Leaf", scans=scans, output_dir=sup.tmp / "leaf")
        sup.progress(1.0, "caller: done")
        return output_dir
    """)

    events_file = tmp_path / "events.jsonl"
    events_file.write_text("", encoding="utf-8")

    completed, _ = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
        env={"SADT_PROGRESS_FILE": str(events_file)},
    )

    assert completed.returncode == 0, completed.stderr
    records = [json.loads(line) for line in
               events_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Interleaved in the order they happened, and the depth is what tells the
    # levels apart -- the child never learned it was in a chain.
    #
    # The two `(1, None)` records are the supervisor's own markers BRACKETING
    # the nested call, which is what makes the call visible to a reader that
    # sees only events: without them a leaf reporting nothing at all would
    # leave a chain indistinguishable from a parent that simply went quiet.
    assert [(record["depth"], record["fraction"]) for record in records] == [
        (0, 0.1), (1, None), (1, 0.9), (1, None), (0, 1.0)
    ]
    # A marker names the tool; a tool's own line never does.
    assert [record.get("tool") for record in records] == [
        None, "Leaf", None, "Leaf", None
    ]
    # Nothing else: a tool's line says how far along it is, and the server
    # stamps the phase, the state and the sequence when it reads them back.
    assert all(set(record) - {"tool"} == {"at", "fraction", "message", "depth"}
               for record in records)


def test_a_tool_reporting_progress_with_no_file_set_is_a_no_op(tools_dir, tmp_path):
    """The variable is absent for every run whose client sent no id, which is
    every run today. sup.progress() must stay exactly what it was: a log line."""
    make_tool(tools_dir, "Leaf", """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Report into nothing at all.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sup.progress(0.5, "halfway")
        return output_dir
    """)

    completed, result = run_job(
        tools_dir, "Leaf", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
    )

    assert completed.returncode == 0, completed.stderr
    assert result["result"]
    assert "50% halfway" in completed.stderr


def test_a_progress_file_that_does_not_exist_never_reaches_the_tool(tools_dir, tmp_path):
    """Best effort means best effort: the append opens without O_CREAT, so a
    stale variable pointing nowhere costs a no-op rather than a failed cohort."""
    make_tool(tools_dir, "Leaf", """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Report into a file that was never created.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sup.progress(0.5, "halfway")
        return output_dir
    """)

    completed, result = run_job(
        tools_dir, "Leaf", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
        env={"SADT_PROGRESS_FILE": str(tmp_path / "gone" / "events.jsonl")},
    )

    assert completed.returncode == 0, completed.stderr
    assert result["result"]
    assert not (tmp_path / "gone").exists()


# ----------------------------------------------------------------------
# What a chain hands down is ROOM, and the child decides its own width
# ----------------------------------------------------------------------
#
# Everything below runs the real thing: two interpreters, a real cost table on
# disk, and the child reporting the number it was actually called with. The
# unit tests in test_concurrency.py pin the arithmetic; these pin that the
# arithmetic survives a process boundary, which is the only place it matters.

WIDE_LEAF = """
    def run(scans: Path, output_dir: Path, num_workers: int = 1) -> int:
        \"\"\"Report the width the server or its parent settled on.\"\"\"
        return num_workers
"""

WIDE_CALLER = """
    def run(scans: Path, output_dir: Path, num_workers: int = 1, *, sup=None) -> int:
        \"\"\"Call the leaf and hand back the width IT was given.\"\"\"
        return sup.run("Leaf", scans=scans, output_dir=output_dir)
"""


def _costs_file(tmp_path, table):
    path = tmp_path / "tool_costs.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    return str(path)


def _chain_env(tmp_path, table, budget, channels, axes=None):
    """The environment a server builds for a root run, minus the tool paths."""
    environment = {
        "SADT_COST_TABLE": _costs_file(tmp_path, table),
        "SADT_CHANNEL_BUDGET": budget,
        "SADT_CHANNELS": str(channels),
        "SADT_MAX_CHANNELS": "0",
    }
    if axes is not None:
        environment["SADT_WIDTH_AXIS"] = json.dumps(axes)
    return environment


def test_a_child_sizes_itself_from_its_own_cost(tools_dir, tmp_path):
    """The parent holds 12 GiB and may open 4 channels, so one channel -- and
    therefore its child -- may spend 3 GiB. A leaf measured at 1 GiB a channel
    takes three of them, and nothing in that sentence is a core count."""
    make_tool(tools_dir, "Leaf", WIDE_LEAF)
    make_tool(tools_dir, "Caller", WIDE_CALLER)

    completed, result = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
        env=_chain_env(
            tmp_path,
            {"Caller": {"vram_bytes": 3 << 30}, "Leaf": {"vram_bytes": 1 << 30}},
            budget="{},0".format(12 << 30), channels=4,
        ),
    )

    assert completed.returncode == 0, completed.stderr
    assert result["result"] == 3


def test_a_chain_cannot_multiply_across_the_process_boundary(tools_dir, tmp_path):
    """The parent's four channels times the child's three is twelve, and twelve
    channels of 1 GiB is exactly the 12 GiB the root was admitted against. The
    bytes were divided on the way down, so the product cannot exceed them."""
    make_tool(tools_dir, "Leaf", WIDE_LEAF)
    make_tool(tools_dir, "Caller", WIDE_CALLER)

    completed, result = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
        env=_chain_env(
            tmp_path,
            {"Caller": {"vram_bytes": 3 << 30}, "Leaf": {"vram_bytes": 1 << 30}},
            budget="{},0".format(12 << 30), channels=4,
        ),
    )

    assert completed.returncode == 0, completed.stderr
    assert 4 * result["result"] * (1 << 30) <= 12 << 30


def test_an_unused_parent_grant_does_not_shrink_the_child(tools_dir, tmp_path):
    """Measured on 2026-09-18: ASO was granted five channels, opened one, and
    its child's budget was divided by five anyway. A reservation is a
    per-channel cost times the channel count, so the grant now inflates the
    numerator by exactly what it inflates the denominator by -- the child gets
    the same width at every grant its parent could have been given."""
    make_tool(tools_dir, "Leaf", WIDE_LEAF)
    make_tool(tools_dir, "Caller", WIDE_CALLER)

    widths = []
    for granted in (1, 5):
        completed, result = run_job(
            tools_dir, "Caller", tmp_path / "job{}".format(granted),
            {"scans": str(tmp_path / "in"),
             "output_dir": str(tmp_path / "job{}".format(granted) / "output")},
            env=_chain_env(
                tmp_path,
                {"Caller": {"vram_bytes": 4 << 30}, "Leaf": {"vram_bytes": 1 << 30}},
                # What admission would have reserved at that width: the
                # per-channel cost times the channels it granted.
                budget="{},0".format(granted * (4 << 30)), channels=granted,
            ),
        )
        assert completed.returncode == 0, completed.stderr
        widths.append(result["result"])

    assert widths == [4, 4]


def test_a_child_nothing_has_measured_still_reserves_everything(tools_dir, tmp_path):
    """An empty cost table has to behave at depth exactly as it does at the
    top: one channel, the whole budget, and no guessing."""
    make_tool(tools_dir, "Leaf", WIDE_LEAF)
    make_tool(tools_dir, "Caller", WIDE_CALLER)

    completed, result = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
        env=_chain_env(tmp_path, {}, budget="{},0".format(64 << 30), channels=1),
    )

    assert completed.returncode == 0, completed.stderr
    assert result["result"] == 1


def test_a_child_is_bounded_by_the_items_its_own_request_carries(tools_dir, tmp_path):
    """ALI under ASO can afford more channels than it has landmarks. The axis
    to count comes down from the server, so both levels bound a width by the
    same rule."""
    make_tool(tools_dir, "Leaf", """
    def run(scans: Path, output_dir: Path, landmarks: dict = None,
            num_workers: int = 1) -> int:
        \"\"\"Report the width, having been asked for a handful of landmarks.\"\"\"
        return num_workers
    """)
    make_tool(tools_dir, "Caller", """
    def run(scans: Path, output_dir: Path, *, sup=None) -> int:
        \"\"\"Ask the leaf for two of its four landmarks.\"\"\"
        return sup.run("Leaf", scans=scans, output_dir=output_dir,
                       landmarks={"Ba": True, "S": True, "N": False, "RPo": False})
    """)

    completed, result = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
        env=_chain_env(
            tmp_path,
            {"Caller": {"vram_bytes": 16 << 30}, "Leaf": {"vram_bytes": 1 << 30}},
            budget="{},0".format(16 << 30), channels=1,
            axes={"Leaf": "landmarks"},
        ),
    )

    assert completed.returncode == 0, completed.stderr
    assert result["result"] == 2, "sixteen affordable, two asked for"


def test_a_child_never_inherits_its_parent_s_own_width(tools_dir, tmp_path):
    """The supervisor copies this process's environment, so SADT_CHANNELS has
    to be REMOVED rather than left. Inheriting it would give the child its
    parent's width, which is the one number that is certainly not its own."""
    make_tool(tools_dir, "Leaf", WIDE_LEAF)
    make_tool(tools_dir, "Caller", WIDE_CALLER)

    completed, result = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
        env=_chain_env(
            tmp_path,
            {"Caller": {"vram_bytes": 6 << 30}, "Leaf": {"vram_bytes": 2 << 30}},
            budget="{},0".format(6 << 30), channels=3,
        ),
    )

    assert completed.returncode == 0, completed.stderr
    assert result["result"] == 1, "6 GiB over 3 channels is 2 GiB, one leaf's worth"


# ----------------------------------------------------------------------
# The tool asks: `sup.channels(wanted)`
# ----------------------------------------------------------------------
#
# The other half of the same budget, arriving from the other direction. The
# server fills in `num_workers` by counting a request's items from the OUTSIDE,
# before the run -- which for several tools it cannot do at all: ALI_IOS counts
# teeth out of a mesh's label array, CLIC counts slices of a volume it has not
# read, and AutoCrop3D, AutoMatrix and GreedyReg take two paired folders where
# splitting either alone re-pairs patients. For those the outside answer is "no
# bound", and an unbounded width is RESERVED all the same: measured on this
# machine on 2026-09-18, AMASSS at 19.1 GiB a channel against a 93.8 GiB host
# budget was granted 3 channels for a request holding ONE scan, so 57.3 GiB was
# held and one run fitted where four had -- six concurrent AMASSS 143 s ->
# 275 s, ten 158 s -> 425 s, with a single run unchanged at 79 s throughout.
#
# `sup.channels(wanted)` asks the side that knows. What makes it safe to answer
# after the run has started is that the SHARE is the reservation: admission
# reserved room for this run before it began, and the answer is only ever what
# that room can pay for.

ASK_LEAF = """
    def run(scans: Path, output_dir: Path, wanted: int = 0, *, sup=None) -> int:
        \"\"\"Ask for a width and hand back what the machine allowed.\"\"\"
        return sup.channels(wanted)
"""

# It declares `num_workers` as well, which is the migration shape: the argument
# stays, because it is how the tool is driven from a CLI with no server around
# it, and the server goes on filling it in. A tool that ASKS ignores it.
ASK_CALLER = """
    def run(scans: Path, output_dir: Path, mine: int = 0, theirs: int = 0,
            num_workers: int = 1, *, sup=None) -> list:
        \"\"\"Take a width of its own, then let the leaf take one of the rest.\"\"\"
        own = sup.channels(mine)
        return [own, sup.run("Leaf", scans=scans, output_dir=output_dir,
                             wanted=theirs)]
"""


def _ask(tools_dir, tmp_path, name, params, table, budget=None, channels=None,
         cap="0", label="job"):
    """Run one asking tool, with the environment a server would have built."""
    environment = {"SADT_COST_TABLE": _costs_file(tmp_path, table),
                   "SADT_MAX_CHANNELS": cap}
    if budget is not None:
        environment["SADT_CHANNEL_BUDGET"] = budget
    if channels is not None:
        environment["SADT_CHANNELS"] = str(channels)
    job = tmp_path / label
    completed, result = run_job(
        tools_dir, name, job,
        dict(params, scans=str(tmp_path / "in"), output_dir=str(job / "output")),
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    return result["result"]


def test_a_tool_asking_for_more_than_its_share_gets_the_share(tools_dir, tmp_path):
    """Three channels were reserved at 2 GiB each, so three is what there is.

    The answer is the room divided by the MEASURED per-channel cost, which is
    what makes it safe to give after the run has started: every channel it
    permits was paid for before the process existed."""
    make_tool(tools_dir, "Leaf", ASK_LEAF)

    assert _ask(tools_dir, tmp_path, "Leaf", {"wanted": 10},
                {"Leaf": {"vram_bytes": 2 << 30}},
                budget="{},0".format(6 << 30), channels=3) == 3


def test_a_tool_asking_for_less_than_its_share_gets_what_it_asked(tools_dir, tmp_path):
    """The point of asking. Its `wanted` is the real item count, read at run
    time, and a run with two things to do opens two channels however much room
    it was given -- which saves the process launches AND stops the cost table
    learning a per-channel figure divided by a width nothing reached."""
    make_tool(tools_dir, "Leaf", ASK_LEAF)

    assert _ask(tools_dir, tmp_path, "Leaf", {"wanted": 2},
                {"Leaf": {"vram_bytes": 2 << 30}},
                budget="{},0".format(6 << 30), channels=3) == 2


def test_a_tool_asking_for_nothing_in_particular_gets_what_it_can_afford(
        tools_dir, tmp_path):
    """Omitted or zero means "as many as I can afford", for a tool whose loop
    has no count to give -- which is the case `width_from = false` describes."""
    make_tool(tools_dir, "Leaf", ASK_LEAF)

    assert _ask(tools_dir, tmp_path, "Leaf", {},
                {"Leaf": {"vram_bytes": 2 << 30}},
                budget="{},0".format(6 << 30), channels=3) == 3


def test_a_tool_asking_with_no_room_accounted_for_gets_one(tools_dir, tmp_path):
    """No `SADT_CHANNEL_BUDGET` is no server: `scripts/run_tool.py`, a test
    harness, a tool driven straight from Python. It must not find a width
    nobody reserved, so it gets the one width that is always affordable."""
    make_tool(tools_dir, "Leaf", ASK_LEAF)

    assert _ask(tools_dir, tmp_path, "Leaf", {"wanted": 10},
                {"Leaf": {"vram_bytes": 2 << 30}}) == 1


def test_a_tool_nothing_has_measured_is_offered_one_channel(tools_dir, tmp_path):
    """The same rule admission applies at the top: an unknown cost is reserved
    as if it were the whole machine, so there is room for exactly one of it."""
    make_tool(tools_dir, "Leaf", ASK_LEAF)

    assert _ask(tools_dir, tmp_path, "Leaf", {"wanted": 10}, {},
                budget="{},0".format(64 << 30), channels=1) == 1


def test_an_answer_never_exceeds_what_admission_reserved_against(
        tools_dir, tmp_path):
    """The cost table is a window and its maximum can fall when an old sample
    drops out of it. The bytes were taken at the price of the day, so the
    number they were taken for wins over an arithmetic working from newer,
    cheaper prices."""
    make_tool(tools_dir, "Leaf", ASK_LEAF)

    assert _ask(tools_dir, tmp_path, "Leaf", {"wanted": 10},
                {"Leaf": {"vram_bytes": 1 << 30}},
                budget="{},0".format(6 << 30), channels=2) == 2


def test_the_deployment_backstop_still_caps_an_answer(tools_dir, tmp_path):
    """`SADT_MAX_CHANNELS` is the one number a deployment writes down, and it
    only ever narrows. A tool asking is not a way around it."""
    make_tool(tools_dir, "Leaf", ASK_LEAF)

    assert _ask(tools_dir, tmp_path, "Leaf", {"wanted": 10},
                {"Leaf": {"vram_bytes": 1 << 30}},
                budget="{},0".format(16 << 30), channels=16, cap="4") == 4


def test_a_nested_call_sizes_itself_from_its_inherited_share(tools_dir, tmp_path):
    """The child's width is ITS cost against the room it was handed -- never
    its parent's width, and never a number its parent computed.

    The caller holds 12 GiB and opens four channels, so one of those channels
    may spend 3 GiB; a leaf measured at 1 GiB takes three of them. Measured on
    2026-09-18, the defect this replaced: the same ALI_CBCT asking for seven
    landmarks got 7 channels standalone, 5 under an ASO holding 28 cores and 7
    under an ASO holding 31."""
    make_tool(tools_dir, "Leaf", ASK_LEAF)
    make_tool(tools_dir, "Caller", ASK_CALLER)

    assert _ask(tools_dir, tmp_path, "Caller", {"mine": 4, "theirs": 10},
                {"Caller": {"vram_bytes": 3 << 30}, "Leaf": {"vram_bytes": 1 << 30}},
                budget="{},0".format(12 << 30), channels=4) == [4, 3]


def test_a_chain_cannot_multiply_what_the_root_was_admitted_for(
        tools_dir, tmp_path):
    """Four channels of the caller times three of the leaf is twelve, and
    twelve channels of 1 GiB is exactly the 12 GiB the root holds. The bytes
    are divided on the way down, so the product cannot exceed them."""
    make_tool(tools_dir, "Leaf", ASK_LEAF)
    make_tool(tools_dir, "Caller", ASK_CALLER)

    own, theirs = _ask(
        tools_dir, tmp_path, "Caller", {"mine": 4, "theirs": 10},
        {"Caller": {"vram_bytes": 3 << 30}, "Leaf": {"vram_bytes": 1 << 30}},
        budget="{},0".format(12 << 30), channels=4)
    assert own * theirs * (1 << 30) <= 12 << 30


def test_a_parent_that_asks_for_one_hands_its_child_everything(tools_dir, tmp_path):
    """The grant it did not use costs its child nothing, and now exactly
    nothing rather than approximately.

    The caller was admitted at four channels and asks for one, so it is running
    at one: it is blocked inside `sup.run` while the child works and there is
    nothing to divide. Dividing by the four it ignored is the defect measured
    on 2026-09-18 -- ASO granted five channels, opening one, its child's budget
    divided by five anyway -- and what closes it here is that asking IS the
    declaration. A tool that never asks is still divided by what it was
    handed, because it never said otherwise."""
    make_tool(tools_dir, "Leaf", ASK_LEAF)
    make_tool(tools_dir, "Caller", ASK_CALLER)

    assert _ask(tools_dir, tmp_path, "Caller", {"mine": 1, "theirs": 99},
                {"Caller": {"vram_bytes": 3 << 30}, "Leaf": {"vram_bytes": 1 << 30}},
                budget="{},0".format(12 << 30), channels=4) == [1, 12]


def test_a_width_at_depth_never_divides_its_way_below_one(tools_dir, tmp_path):
    """The floor that makes a deep chain safe rather than broken. The caller
    holds seven bytes, opens eight channels, and the leaf still gets a channel
    to run in -- over-committed by arithmetic nobody can avoid, and the
    alternative is a chain that stops."""
    make_tool(tools_dir, "Leaf", ASK_LEAF)
    make_tool(tools_dir, "Caller", ASK_CALLER)

    assert _ask(tools_dir, tmp_path, "Caller", {"mine": 8, "theirs": 8},
                {"Caller": {"vram_bytes": 1}, "Leaf": {"vram_bytes": 1 << 30}},
                budget="7,0", channels=8) == [7, 1]
