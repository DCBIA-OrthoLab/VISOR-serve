"""A run that died for want of memory is started again, with more room.

The alternative is telling a clinician "failed" for what is the server's own
misjudgement: admission believed a peak learned on a smaller scan, admitted one
job too many, and the tool paid for it. Retrying is cheap here in a way it is
not in an in-process design -- the tool is a subprocess, so the failure killed
it and nothing else, and its inputs are still staged on disk.

Every other failure must NOT be retried. A bad argument fails identically
however many times it is tried, and three attempts only spend the card to learn
that.
"""

import os

import pytest

os.environ.setdefault("API_TOKEN", "test-token")

import resources
from config import settings
from execution import admission, dispatch
from execution.costs import Cost
from execution.dispatch import RESULT_FILE, ToolExecutionError, ToolFailure


def _demand(vram=1 << 30, ram=1 << 30, cpus=2, measured=True):
    return admission.Demand(
        cpus=cpus, ram_bytes=ram, vram_bytes=vram, measured=measured
    )


def _budget(vram=8 << 30, ram=16 << 30, cpus=8.0):
    allocation = resources.Allocation(
        cpus=cpus, ram_bytes=ram, vram_bytes=vram,
        cpus_per_job=2, ram_per_job=ram // 4, vram_per_job=vram // 4,
        expected_clients=4,
    )
    return admission.Budget(allocation, free_vram=lambda: 1 << 60)


# ----------------------------------------------------------------------
# Which failures are worth trying again
# ----------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(dispatch._MEMORY_ERRORS))
def test_a_tool_that_named_an_out_of_memory_is_retried(name, tmp_path):
    assert dispatch._out_of_memory(ToolFailure(name, "no room"), 1, str(tmp_path))


@pytest.mark.parametrize("name", ["ValueError", "FileNotFoundError", "KeyError",
                                  "ToolInputError", "ToolUnavailableError"])
def test_every_other_failure_is_final(name, tmp_path):
    """A bad argument fails identically however many times it is tried."""
    assert not dispatch._out_of_memory(ToolFailure(name, "wrong"), 1, str(tmp_path))


def test_a_process_the_kernel_killed_counts_as_out_of_memory(tmp_path):
    """Running out of HOST memory is not an exception. The kernel kills the
    process mid-statement and it writes nothing, so a SIGKILL with no result
    file beside it is the only evidence there is."""
    exc = ToolExecutionError("exited with code -9")
    assert dispatch._out_of_memory(exc, -9, str(tmp_path))
    assert dispatch._out_of_memory(exc, 137, str(tmp_path))


def test_a_kill_that_did_leave_a_result_is_not_an_out_of_memory(tmp_path):
    (tmp_path / RESULT_FILE).write_text("{}")
    exc = ToolExecutionError("exited with code -9")
    assert not dispatch._out_of_memory(exc, -9, str(tmp_path))


def test_a_plain_non_zero_exit_is_not_an_out_of_memory(tmp_path):
    exc = ToolExecutionError("exited with code 1")
    assert not dispatch._out_of_memory(exc, 1, str(tmp_path))


# ----------------------------------------------------------------------
# Asking for more
# ----------------------------------------------------------------------

def test_the_first_attempt_asks_for_exactly_what_was_measured():
    demand = _demand()
    assert admission.escalated(demand, 0, 1.5) == demand


def test_each_retry_asks_for_more_memory():
    """The recorded peak is a LOWER bound after an out-of-memory: the run died
    reaching for more than it ever reported."""
    demand = _demand(vram=2 << 30, ram=4 << 30)
    first = admission.escalated(demand, 1, 1.5)
    second = admission.escalated(demand, 2, 1.5)
    assert first.vram_bytes == int((2 << 30) * 1.5)
    assert second.vram_bytes == int((2 << 30) * 2.25)
    assert second.ram_bytes > first.ram_bytes > demand.ram_bytes


def test_escalating_never_asks_for_more_cores():
    """More cores would not stop a tool running out of memory, and would only
    make it wait longer for room it has no use for."""
    demand = _demand(cpus=4)
    assert admission.escalated(demand, 2, 1.5).cpus == 4


def test_escalating_keeps_the_run_measured():
    assert admission.escalated(_demand(), 2, 1.5).measured is True


# ----------------------------------------------------------------------
# When to stop
# ----------------------------------------------------------------------

def test_a_demand_that_already_holds_the_machine_is_terminal():
    """Nothing is left to take, so another attempt asks for the same thing and
    dies the same way, having spent the card to find out."""
    budget = _budget(vram=8 << 30)
    assert admission.holds_everything(_demand(vram=8 << 30), budget)
    assert admission.holds_everything(_demand(vram=9 << 30), budget)


def test_a_demand_with_room_beside_it_is_not_terminal():
    assert not admission.holds_everything(_demand(vram=1 << 30, ram=1 << 30), _budget())


def test_an_unmeasured_demand_is_terminal():
    """It already runs alone, by the rule that governs unmeasured tools."""
    assert admission.holds_everything(_demand(measured=False), _budget())


def test_ram_alone_is_enough_to_be_terminal():
    """A CPU tool that exhausts host memory has taken the whole machine just as
    surely as one that filled the card."""
    budget = _budget(ram=16 << 30)
    assert admission.holds_everything(_demand(vram=0, ram=16 << 30), budget)


# ----------------------------------------------------------------------
# What the retry must clear first
# ----------------------------------------------------------------------

def test_the_failed_attempt_leaves_nothing_the_retry_would_read(tmp_path):
    """A stale result.json would be read as the retry's own answer, and a
    half-written output/ would be packaged beside what the successful run
    produces."""
    job = tmp_path / "job"
    output = job / "output"
    output.mkdir(parents=True)
    (job / RESULT_FILE).write_text('{"error": {"type": "OutOfMemoryError"}}')
    (output / "half-written.nii.gz").write_bytes(b"\x1f\x8b")

    dispatch._reset_job(str(job))

    assert not (job / RESULT_FILE).exists()
    assert output.is_dir(), "the retry still needs somewhere to write"
    assert list(output.iterdir()) == []


def test_resetting_a_job_with_nothing_in_it_is_not_an_error(tmp_path):
    dispatch._reset_job(str(tmp_path))
    assert (tmp_path / "output").is_dir()


# ----------------------------------------------------------------------
# The loop itself
# ----------------------------------------------------------------------

class _Tool:
    name = "_dispatch_probe"
    arguments: dict = {}
    output_kind = "text"
    wants_output_dir = False


@pytest.fixture
def one_attempt(monkeypatch):
    """Drive dispatch's loop with a scripted sequence of outcomes."""
    monkeypatch.setattr(dispatch, "_checked_interpreter", lambda name: "/usr/bin/python3")
    monkeypatch.setattr(dispatch, "_checked_runner", lambda: "/runner.py")
    monkeypatch.setattr(dispatch, "_server_provided", lambda tool, params, job_dir: params)
    monkeypatch.setattr(dispatch, "_write_job_file",
                        lambda job_dir, *a, **k: os.path.join(job_dir, "job.json"))
    monkeypatch.setattr(dispatch, "_child_environment", lambda *a, **k: {})
    monkeypatch.setattr(dispatch, "_stderr_tail", lambda job_dir: "")
    # A MEASURED cost, and small. An unmeasured tool already runs alone, so it
    # is terminal on its first failure by design -- which is right, and which
    # makes it the wrong tool to test the loop with.
    monkeypatch.setattr(dispatch.costs, "cost_of",
                        lambda name: Cost(vram_bytes=1 << 28, ram_bytes=1 << 28, samples=3))
    admission.reset()
    yield
    admission.reset()


def _script(monkeypatch, outcomes):
    """Each entry is the exception to raise from _read_result, or a value."""
    seen = {"attempts": 0}

    def _execute(*_args, **_kwargs):
        return 0

    def _read_result(job_dir, tool_name, solo=False):
        outcome = outcomes[seen["attempts"]]
        seen["attempts"] += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(dispatch, "_execute", _execute)
    monkeypatch.setattr(dispatch, "_read_result", _read_result)
    return seen


def test_an_out_of_memory_is_tried_again_and_can_succeed(one_attempt, monkeypatch):
    seen = _script(monkeypatch, [ToolFailure("OutOfMemoryError", "no room"), "ok"])
    assert dispatch.dispatch(_Tool(), {}) == "ok"
    assert seen["attempts"] == 2


def test_a_bad_argument_is_not_tried_again(one_attempt, monkeypatch):
    seen = _script(monkeypatch, [ToolFailure("ValueError", "bad landmark"), "ok"])
    with pytest.raises(ToolFailure):
        dispatch.dispatch(_Tool(), {})
    assert seen["attempts"] == 1


def test_the_retries_are_bounded(one_attempt, monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_RETRIES", 2)
    oom = ToolFailure("OutOfMemoryError", "no room")
    seen = _script(monkeypatch, [oom, oom, oom, "ok"])
    with pytest.raises(ToolFailure):
        dispatch.dispatch(_Tool(), {})
    assert seen["attempts"] == 3, "one attempt plus MEMORY_RETRIES"


def test_retrying_can_be_turned_off(one_attempt, monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_RETRIES", 0)
    seen = _script(monkeypatch, [ToolFailure("OutOfMemoryError", "no room"), "ok"])
    with pytest.raises(ToolFailure):
        dispatch.dispatch(_Tool(), {})
    assert seen["attempts"] == 1


def test_a_tool_nobody_has_measured_is_not_retried(one_attempt, monkeypatch):
    """It already runs alone. A failure there is the tool not fitting, not the
    server having misjudged, and a second attempt would ask for the same thing."""
    monkeypatch.setattr(dispatch.costs, "cost_of", lambda name: None)
    seen = _script(monkeypatch, [ToolFailure("OutOfMemoryError", "no room"), "ok"])
    with pytest.raises(ToolFailure):
        dispatch.dispatch(_Tool(), {})
    assert seen["attempts"] == 1
