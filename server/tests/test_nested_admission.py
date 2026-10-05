"""Nested calls admitted like runs of their own.

A tool calling another through its supervisor used to run the callee inside its
own reservation. Now the supervisor asks the run's DESK (`execution/nested.py`)
to admit the callee: the same budget, the same queue -- in a band ahead of runs
not yet started -- and the same price and width a run of that tool arriving over
HTTP would get, held for exactly as long as the callee runs.

The properties pinned here are the ones that make that safe:

* a child is admitted, holds room while it runs, and gives it back when its
  connection closes, however the connection closes;
* a chain cannot deadlock on itself: a child that may only run alone runs
  alone apart from its own parents, and six chains on a budget with room for
  one child at a time all finish;
* what the child is started with is what the server granted, and what it cost
  is recorded under its own name while its parent reports itself alone.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("API_TOKEN", "test-token")

import resources
from execution import admission, costs, nested
from execution import runner as runner_module
from test_supervisor import CALLER, LEAF, make_tool, run_job

GiB = 1024 ** 3
_TIMEOUT = 10.0


def _budget(ram=10 * GiB, vram=8 * GiB, cpus=16.0):
    return admission.Budget(
        resources.Allocation(cpus=cpus, ram_bytes=ram, vram_bytes=vram, cpus_per_job=2,
                             ram_per_job=ram // 4, vram_per_job=vram // 4,
                             expected_clients=4),
        free_vram=lambda: 1 << 60,
    )


def _demand(ram=1 * GiB, vram=0, measured=True):
    return admission.Demand(cpus=1, ram_bytes=ram, vram_bytes=vram, measured=measured)


@pytest.fixture
def budget(monkeypatch):
    built = _budget()
    monkeypatch.setattr(admission, "budget", lambda: built)
    return built


class _Root:
    """A root run holding its reservation and its desk, on its own thread."""

    def __init__(self, budget, job_dir: Path, demand=None, price=None, learn=None):
        job_dir.mkdir(parents=True, exist_ok=True)
        self.job_dir = job_dir
        self.learned = []
        self._price = price or (lambda tool, params: [(1, _demand())])
        self._learn = learn or (lambda tool, payload, solo: self.learned.append((tool, payload, solo)))
        self.ready = threading.Event()
        self.done = threading.Event()
        self.desk = None
        self._thread = threading.Thread(target=self._run, args=(budget, demand or _demand()),
                                        daemon=True)
        self._thread.start()
        assert self.ready.wait(_TIMEOUT), "the root run was never admitted"

    def _run(self, budget, demand):
        with budget.reserve(demand) as grant:
            with nested.Desk("root", str(self.job_dir), grant, self._price,
                             lambda tool, g: {"SADT_CHANNELS": str(g.channels), "GRANTED_TO": tool},
                             self._learn) as desk:
                self.desk = desk
                self.ready.set()
                self.done.wait(_TIMEOUT * 3)

    def finish(self):
        self.done.set()
        self._thread.join(_TIMEOUT)

    def child_job(self, tool="Leaf", under: Path = None, slot="01_Leaf") -> Path:
        directory = (under or self.job_dir) / "sup" / slot
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "job.json"
        path.write_text(json.dumps({"job_id": "root.1", "tool": tool, "params": {}}))
        return path


def _lease(root: _Root, job: Path, caller: Path = None, key=None):
    """What the supervisor does, through the real runner function."""
    environment = {runner_module.ADMISSION_SOCKET_ENV: root.desk.path,
                   runner_module.ADMISSION_KEY_ENV: key or root.desk.key}
    old = {name: os.environ.get(name) for name in environment}
    os.environ.update(environment)
    try:
        return runner_module._admission_lease(str(job), str(caller or root.job_dir), lambda m: None)
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _until(condition, timeout=_TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------------------
# One child
# ---------------------------------------------------------------------------

def test_a_child_is_admitted_holds_room_and_gives_it_back(budget, tmp_path):
    root = _Root(budget, tmp_path / "job")
    lease, environment = _lease(root, root.child_job())

    assert environment == {"SADT_CHANNELS": "1", "GRANTED_TO": "Leaf"}
    assert budget.snapshot()["running"] == 2
    runner_module._close_lease(lease)
    assert _until(lambda: budget.snapshot()["running"] == 1)
    root.finish()


def test_what_the_child_cost_is_learned_under_its_own_name(budget, tmp_path):
    root = _Root(budget, tmp_path / "job")
    job = root.child_job()
    lease, _environment = _lease(root, job)
    (job.parent / "result.json").write_text(json.dumps({"result": 1, "peak_rss_bytes": 123}))
    runner_module._close_lease(lease)

    assert _until(lambda: root.learned)
    tool, payload, _solo = root.learned[0]
    assert tool == "Leaf" and payload["peak_rss_bytes"] == 123
    root.finish()


def test_a_grandchild_is_admitted_under_both_of_its_parents(budget, tmp_path):
    root = _Root(budget, tmp_path / "job")
    child_job = root.child_job("Middle", slot="01_Middle")
    child_lease, _ = _lease(root, child_job)
    grandchild_job = root.child_job("Leaf", under=child_job.parent)

    lease, environment = _lease(root, grandchild_job, caller=child_job.parent)

    assert environment["GRANTED_TO"] == "Leaf"
    assert budget.snapshot()["running"] == 3
    runner_module._close_lease(lease)
    runner_module._close_lease(child_lease)
    assert _until(lambda: budget.snapshot()["running"] == 1)
    root.finish()


def test_a_tool_the_server_cannot_price_runs_inside_its_parent(budget, tmp_path):
    root = _Root(budget, tmp_path / "job", price=lambda tool, params: None)
    assert _lease(root, root.child_job()) == (None, None)
    assert budget.snapshot()["running"] == 1
    root.finish()


@pytest.mark.parametrize("case", ["wrong key", "outside the run", "unknown caller"])
def test_a_request_that_is_not_this_run_s_is_not_admitted(budget, tmp_path, case):
    root = _Root(budget, tmp_path / "job")
    job = root.child_job()
    if case == "wrong key":
        answer = _lease(root, job, key="0" * 32)
    elif case == "outside the run":
        elsewhere = tmp_path / "other" / "sup" / "01_Leaf"
        elsewhere.mkdir(parents=True)
        (elsewhere / "job.json").write_text(json.dumps({"tool": "Leaf"}))
        answer = _lease(root, elsewhere / "job.json")
    else:
        answer = _lease(root, job, caller=tmp_path / "nobody")
    assert answer == (None, None)
    assert budget.snapshot()["running"] == 1
    root.finish()


def test_closing_the_desk_lets_go_of_a_child_still_holding(budget, tmp_path):
    root = _Root(budget, tmp_path / "job")
    lease, _ = _lease(root, root.child_job())
    root.desk.close()
    assert _until(lambda: budget.snapshot()["running"] == 1)
    runner_module._close_lease(lease)
    root.finish()


def test_without_a_server_there_is_nobody_to_ask(monkeypatch, tmp_path):
    monkeypatch.delenv(runner_module.ADMISSION_SOCKET_ENV, raising=False)
    assert runner_module._admission_lease(str(tmp_path / "job.json"), str(tmp_path),
                                          lambda m: None) == (None, None)


# ---------------------------------------------------------------------------
# Waiting, and never deadlocking
# ---------------------------------------------------------------------------

def test_a_child_that_does_not_fit_waits_and_says_so(budget, tmp_path):
    big = lambda tool, params: [(1, _demand(ram=6 * GiB))]
    root = _Root(budget, tmp_path / "job", price=big)
    first, _ = _lease(root, root.child_job(slot="01_Leaf"))
    answered = {}

    def second():
        answered["lease"] = _lease(root, root.child_job(slot="02_Leaf"))

    waiter = threading.Thread(target=second, daemon=True)
    waiter.start()
    assert _until(lambda: budget.snapshot()["waiting"] == 1)
    queued = budget.snapshot()["queue"][0]
    assert queued["nested"] is True and queued["parent"] == "root"

    runner_module._close_lease(first)
    waiter.join(_TIMEOUT)
    lease, environment = answered["lease"]
    assert environment["GRANTED_TO"] == "Leaf"
    runner_module._close_lease(lease)
    root.finish()


def test_a_child_nothing_has_measured_does_not_wait_for_its_own_parent(budget, tmp_path):
    """The whole machine minus its parents: an unmeasured callee may only run
    alone, and alone apart from its own chain is all the chain can give it."""
    unmeasured = lambda tool, params: [(1, admission.whole_machine(budget))]
    root = _Root(budget, tmp_path / "job", price=unmeasured)
    lease, environment = _lease(root, root.child_job())
    assert environment
    held = budget.snapshot()
    # It holds what its parent left, not a second whole machine.
    assert held["ram_held"] == budget.ram_bytes
    runner_module._close_lease(lease)
    root.finish()


def test_a_parent_nothing_has_measured_still_gets_its_child_admitted(budget, tmp_path):
    root = _Root(budget, tmp_path / "job", demand=admission.whole_machine(budget))
    lease, environment = _lease(root, root.child_job())
    assert environment
    runner_module._close_lease(lease)
    root.finish()


def test_six_chains_with_room_for_one_child_at_a_time_all_finish(budget, tmp_path):
    """Six parents hold 6 GiB of 10; each child wants 3 GiB, so one runs at a
    time. Every chain finishes: a parent waiting for its child holds only its
    own room, and the children queue ahead of anything not yet started."""
    price = lambda tool, params: [(1, _demand(ram=3 * GiB))]
    finished = []

    def chain(index):
        root = _Root(budget, tmp_path / f"job{index}", price=price)
        lease, environment = _lease(root, root.child_job())
        assert environment
        time.sleep(0.05)
        runner_module._close_lease(lease)
        root.finish()
        finished.append(index)

    threads = [threading.Thread(target=chain, args=(i,), daemon=True) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(_TIMEOUT * 3)
    assert sorted(finished) == list(range(6))
    assert budget.snapshot()["running"] == 0


def test_a_nested_call_queues_ahead_of_high_and_of_normal():
    """An unrelated run fills what the parent left, so the child has to wait --
    and it waits ahead of a HIGH run: HIGH is the next run to START, and a
    chain already started finishes first, or a HIGH run that does not fit
    freezes every chain behind it."""
    budget = _budget(ram=4 * GiB)
    release = threading.Event()
    holders = []

    def hold(ram):
        admitted = threading.Event()

        def go():
            with budget.reserve(_demand(ram=ram)) as grant:
                holders.append(grant)
                admitted.set()
                release.wait(_TIMEOUT)
        threading.Thread(target=go, daemon=True).start()
        assert admitted.wait(_TIMEOUT)
        return holders[-1]

    parent = hold(1 * GiB)
    hold(3 * GiB)
    budget.set_priority("urgent", admission.PRIORITY_HIGH)

    def queue(run_id, **kwargs):
        def go():
            with budget.reserve(_demand(), run_id=run_id, **kwargs):
                pass
        threading.Thread(target=go, daemon=True).start()

    queue("normal")
    assert _until(lambda: budget.snapshot()["waiting"] == 1)
    queue("child", ancestors=(parent,), parent="root")
    assert _until(lambda: budget.snapshot()["waiting"] == 2)
    queue("urgent")
    assert _until(lambda: budget.snapshot()["waiting"] == 3)

    assert [w["run_id"] for w in budget.snapshot()["queue"]] == ["child", "urgent", "normal"]
    release.set()


def test_a_child_with_only_its_parents_running_is_alone():
    """Solo is about the card's window: a child under a parent that holds
    nothing else is alone, and admitting it ends its parent's solitude."""
    budget = _budget()
    with budget.reserve(_demand()) as parent:
        assert parent.solo
        with budget.reserve(_demand(), ancestors=(parent,)) as child:
            assert child.solo
            assert not parent.solo


# ---------------------------------------------------------------------------
# The cost table
# ---------------------------------------------------------------------------

def test_the_first_own_only_figure_forgets_the_chain_figures_before_it(tmp_path, monkeypatch):
    monkeypatch.setattr(costs, "table_path", lambda: str(tmp_path / "costs.json"))
    costs.record("Orchestrator", None, 14 * GiB)
    costs.record("Orchestrator", None, 15 * GiB)
    costs.record("Orchestrator", None, 2 * GiB, own_only=True)
    assert costs.cost_of("Orchestrator").ram_bytes == 2 * GiB
    costs.record("Orchestrator", None, 3 * GiB, own_only=True)
    assert costs.cost_of("Orchestrator").ram_bytes == 3 * GiB


# ---------------------------------------------------------------------------
# Through the real runner
# ---------------------------------------------------------------------------

LEAF_REPORTING = """
    import os

    def run(scans: Path, output_dir: Path, tag: str = "leaf") -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "leaf.txt").write_text(
            os.environ.get("GRANTED_TO", "-") + "," + os.environ.get("SADT_ADMITTED", "-"))
        return output_dir
"""


@pytest.fixture
def tools_dir(tmp_path):
    folder = tmp_path / "tools"
    folder.mkdir()
    return folder


def test_a_supervised_call_is_started_with_what_the_server_granted(budget, tools_dir, tmp_path):
    make_tool(tools_dir, "Leaf", LEAF_REPORTING)
    make_tool(tools_dir, "Caller", CALLER)
    job_dir = tmp_path / "job"
    root = _Root(budget, job_dir)

    completed, result = run_job(tools_dir, "Caller", job_dir, {"scans": "x"},
                                env=root.desk.environment())

    assert completed.returncode == 0, completed.stderr
    assert (job_dir / "output" / "chained.txt").read_text() == "Leaf,1"
    # The parent reports itself alone, and its child was learned on its own.
    assert result.get(runner_module.NESTED_ADMITTED_KEY) is True
    assert _until(lambda: [tool for tool, _, _ in root.learned] == ["Leaf"])
    assert budget.snapshot()["running"] == 1
    root.finish()


def test_without_a_desk_a_supervised_call_runs_as_it_always_did(tools_dir, tmp_path):
    make_tool(tools_dir, "Leaf", LEAF_REPORTING)
    make_tool(tools_dir, "Caller", CALLER)
    job_dir = tmp_path / "job"

    completed, result = run_job(tools_dir, "Caller", job_dir, {"scans": "x"})

    assert completed.returncode == 0, completed.stderr
    assert (job_dir / "output" / "chained.txt").read_text() == "-,-"
    assert runner_module.NESTED_ADMITTED_KEY not in result


LEAF_HEAVY = """
    import time

    def run(scans: Path, output_dir: Path, tag: str = "leaf") -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        held = bytearray(300 * 1024 * 1024)
        held[::4096] = b"x" * len(held[::4096])
        time.sleep(3)  # longer than the sampler's interval
        (output_dir / "leaf.txt").write_text(str(len(held)))
        return output_dir
"""


def test_an_admitted_child_s_memory_is_its_own_and_not_its_parent_s(budget, tools_dir, tmp_path):
    make_tool(tools_dir, "Leaf", LEAF_HEAVY)
    make_tool(tools_dir, "Caller", CALLER)
    job_dir = tmp_path / "job"
    root = _Root(budget, job_dir)

    completed, result = run_job(tools_dir, "Caller", job_dir, {"scans": "x"},
                                env=root.desk.environment())

    assert completed.returncode == 0, completed.stderr
    assert _until(lambda: root.learned)
    child = root.learned[0][1]
    assert child["peak_rss_bytes"] >= 300 * 1024 * 1024
    assert result["peak_rss_bytes"] < 200 * 1024 * 1024
    root.finish()


def test_without_a_desk_the_parent_still_carries_its_child(tools_dir, tmp_path):
    make_tool(tools_dir, "Leaf", LEAF_HEAVY)
    make_tool(tools_dir, "Caller", CALLER)

    completed, result = run_job(tools_dir, "Caller", tmp_path / "job", {"scans": "x"})

    assert completed.returncode == 0, completed.stderr
    assert result["peak_rss_bytes"] >= 300 * 1024 * 1024


def test_dispatch_admits_a_supervised_call_and_learns_both_tools_apart(
        budget, tools_dir, tmp_path, monkeypatch, tracked_scratch_dirs):
    """End to end through `dispatch`: the root run opens a desk, its supervisor
    asks it for the child, the child is started with the server's grant, and
    the cost table ends up with each tool's own figure."""
    from base import ArgSpec, Tool
    from config import settings
    from execution import dispatch

    class CallerTool(Tool):
        name = "Caller"
        arguments = {"scans": ArgSpec(type=str, description="anything")}

        def run(self, scans):
            raise AssertionError("run() is not called on the subprocess path")

    make_tool(tools_dir, "Leaf", LEAF_REPORTING)
    make_tool(tools_dir, "Caller", CALLER)
    monkeypatch.setattr(settings, "TOOLS_DIR", str(tools_dir))
    monkeypatch.setattr(costs, "table_path", lambda: str(tmp_path / "costs.json"))
    priced = []
    monkeypatch.setattr(dispatch, "_nested_candidates",
                        lambda tool, params: priced.append(tool) or [(1, _demand())])
    monkeypatch.setattr(dispatch, "_nested_environment",
                        lambda tool, grant: {"SADT_CHANNELS": str(grant.channels)})

    output = dispatch.dispatch(CallerTool(), {"scans": "x"})

    assert priced == ["Leaf"]
    assert (Path(output) / "chained.txt").read_text() == "-,1"
    assert costs.cost_of("Leaf") is not None
    table = json.loads((tmp_path / "costs.json").read_text())
    assert table["Caller"][costs.OWN_ONLY_KEY] is True
    assert budget.snapshot()["running"] == 0


# The way a tool's own progress module writes: no depth on the record.
WIDTH_RECORD = """
    import json, os

    def _declare(width):
        path = os.environ.get("SADT_PROGRESS_FILE")
        if path:
            with open(path, "a") as handle:
                handle.write(json.dumps({"fraction": 0.5, "message": "x", "width": width}) + "\\n")
"""

LEAF_FOUR_WIDE = WIDTH_RECORD + """
    def run(scans: Path, output_dir: Path, tag: str = "leaf") -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _declare(4)
        (output_dir / "leaf.txt").write_text("four")
        return output_dir
"""


def test_each_level_is_measured_at_the_width_it_declared_itself(budget, tools_dir, tmp_path):
    """A tool's progress records carry no depth, and the whole chain writes one
    file. Read by depth alone, the root took its callee's width as its own and
    the callee found none; read by where they were written, each level keeps
    its own."""
    make_tool(tools_dir, "Leaf", LEAF_FOUR_WIDE)
    make_tool(tools_dir, "Caller", CALLER)
    job_dir = tmp_path / "job"
    root = _Root(budget, job_dir)
    progress = tmp_path / "progress.jsonl"
    progress.write_text("")

    completed, result = run_job(tools_dir, "Caller", job_dir, {"scans": "x"},
                                env={**root.desk.environment(), "SADT_PROGRESS_FILE": str(progress)})

    assert completed.returncode == 0, completed.stderr
    assert _until(lambda: root.learned)
    assert root.learned[0][1]["channels"] == 4
    assert result["channels"] == 1
    root.finish()



# ---------------------------------------------------------------------------
# Chains that would wait on each other for ever
# ---------------------------------------------------------------------------

class _Chain:
    """A chain of reservations, each level holding while the next one runs,
    straight on the budget: what the desk does, without the sockets."""

    def __init__(self, budget, demands, name):
        self.finished = threading.Event()
        self.error = None
        self._thread = threading.Thread(target=self._run, args=(budget, demands, name), daemon=True)
        self._thread.start()

    def _run(self, budget, demands, name):
        try:
            self._level(budget, list(demands), (), name, 0)
            self.finished.set()
        except Exception as exc:  # noqa: BLE001 - surfaced by the test
            self.error = exc

    def _level(self, budget, demands, ancestors, name, depth):
        if not demands:
            return
        with budget.reserve(demands[0], run_id=f"{name}.{depth}", ancestors=ancestors,
                            parent=name if ancestors else None) as grant:
            time.sleep(0.05)
            self._level(budget, demands[1:], ancestors + (grant,), name, depth + 1)


def _all_finish(chains):
    for chain in chains:
        assert chain.finished.wait(_TIMEOUT), "a chain never finished: {}".format(chain.error)


def test_two_chains_whose_parents_leave_room_for_neither_child_both_finish():
    """4 + 4 GiB of parents on 10 GiB, each calling a 3 GiB child: neither child
    fits while the other parent holds. Each parent waits for its child, so
    nothing outside would ever free room -- and the head child is let in."""
    budget = _budget(ram=10 * GiB)
    _all_finish([_Chain(budget, [_demand(ram=4 * GiB), _demand(ram=3 * GiB)], f"r{i}")
                 for i in range(2)])
    assert budget.snapshot()["running"] == 0


def test_two_chains_calling_tools_nothing_has_measured_both_finish():
    budget = _budget(ram=10 * GiB)
    _all_finish([_Chain(budget, [_demand(), admission.whole_machine(budget)], f"r{i}")
                 for i in range(2)])


def test_a_grandchild_is_not_stuck_behind_another_chain_s_waiting_child():
    """Three deep: chain A's child is running and calls a grandchild while
    chain B's child is queued waiting for A. The grandchild fits, and goes."""
    budget = _budget(ram=10 * GiB)
    _all_finish([_Chain(budget, [_demand(), _demand(ram=5 * GiB), _demand()], f"r{i}")
                 for i in range(2)])


def test_a_high_run_that_does_not_fit_does_not_freeze_a_running_chain():
    budget = _budget(ram=10 * GiB)
    budget.set_priority("hi", admission.PRIORITY_HIGH)
    chain_parent = threading.Event()
    done = threading.Event()

    def high():
        with budget.reserve(_demand(ram=8 * GiB), run_id="hi"):
            pass
        done.set()

    chain = _Chain(budget, [_demand(ram=4 * GiB), _demand(ram=3 * GiB)], "r0")
    threading.Thread(target=high, daemon=True).start()
    _all_finish([chain])
    assert done.wait(_TIMEOUT)


def test_a_desk_that_cannot_be_built_never_fails_the_run(budget, tmp_path):
    """No socket here means nested calls run inside their parent, as before."""
    with budget.reserve(_demand()) as grant:
        desk = nested.open_desk("root", str(tmp_path / "missing"), grant,
                                lambda *a: None, lambda *a: {}, lambda *a: None)
    assert desk is None
