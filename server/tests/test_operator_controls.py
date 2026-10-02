"""The operator's hold on the queue: moving a waiting run, and giving one priority.

Admission is FIFO for every run nobody touches, and these controls are the
only way that order changes. What they must guarantee:

* a move happens within a band -- a normal run never jumps a HIGH one;
* HIGH puts a run ahead of every normal waiter, whether it was marked before
  it queued or while it waited, and admits it on the widest shape that fits;
* a run already admitted is never touched;
* only the admin token can do any of it, and with no admin token configured
  the controls do not exist.
"""

import threading

import anyio
import pytest
from fastapi.testclient import TestClient

import main
import resources
from config import settings
from execution import admission

_TIMEOUT = 5.0
GIB = 1024 ** 3


def _budget(vram=8 * GIB):
    allocation = resources.Allocation(
        cpus=8.0, ram_bytes=16 * GIB, vram_bytes=vram, cpus_per_job=2,
        ram_per_job=4 * GIB, vram_per_job=vram // 4, expected_clients=4,
    )
    return admission.Budget(allocation, free_vram=lambda: 1 << 60)


def _demand(vram=1 * GIB):
    return admission.Demand(cpus=1, ram_bytes=GIB // 4, vram_bytes=vram, measured=True)


class _Run:
    """A run on its own thread: queues, records the order it was admitted in."""

    def __init__(self, budget, run_id, candidates, order):
        self.admitted = threading.Event()
        self.release = threading.Event()
        self.channels = None
        self.thread = threading.Thread(target=self._go, args=(budget, run_id, candidates, order), daemon=True)

    def _go(self, budget, run_id, candidates, order):
        with budget.reserve(candidates, run_id=run_id) as grant:
            self.channels = grant.channels
            order.append(run_id)
            self.admitted.set()
            self.release.wait(_TIMEOUT)

    def start(self):
        self.thread.start()
        return self

    def finish(self):
        self.release.set()
        self.thread.join(_TIMEOUT)


def _queued(budget, count):
    for _ in range(200):
        if budget.snapshot()["waiting"] == count:
            return True
        threading.Event().wait(0.01)
    return False


def _whole(budget):
    return [(1, _demand(vram=budget.vram_bytes))]


def test_a_run_moved_to_the_top_is_admitted_next():
    budget, order = _budget(), []
    holder = _Run(budget, "holder", _whole(budget), order).start()
    assert holder.admitted.wait(_TIMEOUT)
    waiters = []
    for name in ("a", "b", "c"):
        waiters.append(_Run(budget, name, _whole(budget), order).start())
        assert _queued(budget, len(waiters))
    snapshot = budget.move("c", "top")
    assert [entry["run_id"] for entry in snapshot["queue"]] == ["c", "a", "b"]
    holder.finish()
    for run in waiters:
        run.admitted.wait(_TIMEOUT)
        run.finish()
    assert order == ["holder", "c", "a", "b"]


@pytest.mark.parametrize("where, expected", [
    ("up", ["a", "c", "b"]), ("down", ["a", "b", "c"]), ("bottom", ["a", "b", "c"]),
])
def test_moves_shift_one_place_or_to_the_end(where, expected):
    budget, order = _budget(), []
    holder = _Run(budget, "holder", _whole(budget), order).start()
    assert holder.admitted.wait(_TIMEOUT)
    waiters = []
    for name in ("a", "b", "c"):
        waiters.append(_Run(budget, name, _whole(budget), order).start())
        assert _queued(budget, len(waiters))
    snapshot = budget.move("c", where)
    assert [entry["run_id"] for entry in snapshot["queue"]] == expected
    holder.finish()
    for run in waiters:
        run.admitted.wait(_TIMEOUT)
        run.finish()


def test_moving_a_run_that_is_not_waiting_is_refused():
    with pytest.raises(admission.NotQueued):
        _budget().move("nobody", "top")
    with pytest.raises(ValueError):
        _budget().move("nobody", "sideways")


def test_a_run_marked_high_before_it_queues_goes_ahead_of_earlier_ones():
    budget, order = _budget(), []
    holder = _Run(budget, "holder", _whole(budget), order).start()
    assert holder.admitted.wait(_TIMEOUT)
    early = _Run(budget, "early", _whole(budget), order).start()
    assert _queued(budget, 1)
    budget.set_priority("urgent", admission.PRIORITY_HIGH)
    urgent = _Run(budget, "urgent", _whole(budget), order).start()
    assert _queued(budget, 2)
    assert [e["run_id"] for e in budget.snapshot()["queue"]] == ["urgent", "early"]
    holder.finish()
    for run in (urgent, early):
        run.admitted.wait(_TIMEOUT)
        run.finish()
    assert order == ["holder", "urgent", "early"]
    # Forgotten once admitted.
    assert budget.priority_of("urgent") == admission.PRIORITY_NORMAL


def test_marking_a_waiting_run_high_and_back_restores_arrival_order():
    budget, order = _budget(), []
    holder = _Run(budget, "holder", _whole(budget), order).start()
    assert holder.admitted.wait(_TIMEOUT)
    runs = []
    for name in ("a", "b", "c"):
        runs.append(_Run(budget, name, _whole(budget), order).start())
        assert _queued(budget, len(runs))
    queue = budget.set_priority("c", admission.PRIORITY_HIGH)["queue"]
    assert [e["run_id"] for e in queue] == ["c", "a", "b"]
    assert queue[0]["priority"] == admission.PRIORITY_HIGH
    queue = budget.set_priority("c", admission.PRIORITY_NORMAL)["queue"]
    assert [e["run_id"] for e in queue] == ["a", "b", "c"]
    holder.finish()
    for run in runs:
        run.admitted.wait(_TIMEOUT)
        run.finish()


def test_a_normal_run_cannot_be_moved_ahead_of_a_high_one():
    budget, order = _budget(), []
    holder = _Run(budget, "holder", _whole(budget), order).start()
    assert holder.admitted.wait(_TIMEOUT)
    runs = []
    for name in ("a", "b"):
        runs.append(_Run(budget, name, _whole(budget), order).start())
        assert _queued(budget, len(runs))
    budget.set_priority("b", admission.PRIORITY_HIGH)
    queue = budget.move("a", "top")["queue"]
    assert [e["run_id"] for e in queue] == ["b", "a"]
    holder.finish()
    for run in runs:
        run.admitted.wait(_TIMEOUT)
        run.finish()


def _wide_admission(high):
    """Who gets how many channels when a run with others behind it is let in.

    The wide shape (5 GiB of an 8 GiB card) fits on its own but not "as if
    the run behind took the same", which is the fair pass's question.
    """
    budget, order = _budget(vram=8 * GIB), []
    holder = _Run(budget, "holder", _whole(budget), order).start()
    assert holder.admitted.wait(_TIMEOUT)
    if high:
        budget.set_priority("first", admission.PRIORITY_HIGH)
    first = _Run(budget, "first", [(4, _demand(vram=5 * GIB)), (1, _demand(vram=1 * GIB))], order).start()
    assert _queued(budget, 1)
    behind = _Run(budget, "behind", [(1, _demand(vram=1 * GIB))], order).start()
    assert _queued(budget, 2)
    holder.finish()
    assert first.admitted.wait(_TIMEOUT)
    channels = first.channels
    first.finish()
    behind.admitted.wait(_TIMEOUT)
    behind.finish()
    return channels


def test_a_high_run_is_admitted_on_the_widest_shape_that_fits():
    assert _wide_admission(high=False) == 1
    assert _wide_admission(high=True) == 4


# ---------------------------------------------------------------------------
# The tool slot, the first queue a run meets
# ---------------------------------------------------------------------------

def test_a_high_run_goes_past_a_full_tool_slot(monkeypatch):
    monkeypatch.setattr(admission, "_budget", _budget())

    async def scenario():
        monkeypatch.setattr(main, "_tool_limiter", anyio.CapacityLimiter(1))
        monkeypatch.setattr(main, "_PRIORITY_POLL_SECONDS", 0.01)
        occupant = object()
        await main._tool_limiter.acquire_on_behalf_of(occupant)
        entered = []

        async def run():
            async with main._tool_slot("vip"):
                entered.append("vip")

        async with anyio.create_task_group() as group:
            group.start_soon(run)
            await anyio.sleep(0.1)
            assert entered == [], "a normal run got past a full slot"
            admission.budget().set_priority("vip", admission.PRIORITY_HIGH)
            with anyio.fail_after(2):
                while not entered:
                    await anyio.sleep(0.01)
        main._tool_limiter.release_on_behalf_of(occupant)
        assert main._tool_limiter.borrowed_tokens == 0

    anyio.run(scenario)


def test_a_normal_run_takes_and_gives_back_its_slot(monkeypatch):
    monkeypatch.setattr(admission, "_budget", _budget())

    async def scenario():
        monkeypatch.setattr(main, "_tool_limiter", anyio.CapacityLimiter(1))
        async with main._tool_slot("ordinary"):
            assert main._tool_limiter.borrowed_tokens == 1
        assert main._tool_limiter.borrowed_tokens == 0

    anyio.run(scenario)


# ---------------------------------------------------------------------------
# The endpoints, and who may use them
# ---------------------------------------------------------------------------

client = TestClient(main.app)
AUTH = {"Authorization": f"Bearer {settings.API_TOKEN}"}


@pytest.fixture
def admin(monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_TOKEN", "operator-secret")
    monkeypatch.setattr(admission, "_budget", _budget())
    return {**AUTH, "X-Admin-Token": "operator-secret"}


def test_the_controls_do_not_exist_without_an_admin_token(monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_TOKEN", "")
    response = client.post("/admin/runs/x/priority", json={"priority": "high"},
                           headers={**AUTH, "X-Admin-Token": ""})
    assert response.status_code == 403


def test_the_api_token_alone_is_not_enough(admin):
    assert client.get("/admin/check", headers=AUTH).status_code == 401
    wrong = {**AUTH, "X-Admin-Token": "guess"}
    assert client.post("/admin/runs/x/priority", json={"priority": "high"}, headers=wrong).status_code == 401
    # And the admin token alone is not enough either.
    assert client.get("/admin/check", headers={"X-Admin-Token": "operator-secret"}).status_code == 401


def test_the_admin_token_can_mark_a_run_and_status_shows_it(admin):
    assert client.get("/admin/check", headers=admin).json() == {"admin": True}
    response = client.post("/admin/runs/some-run/priority", json={"priority": "high"}, headers=admin)
    assert response.status_code == 200
    assert response.json()["priorities"] == {"some-run": "high"}
    status = client.get("/status", headers=AUTH).json()
    assert status["admission"]["priorities"] == {"some-run": "high"}


def test_bad_requests_are_named(admin):
    assert client.post("/admin/runs/x/priority", json={"priority": "urgent"}, headers=admin).status_code == 422
    assert client.post("/admin/queue/x/move", json={"to": "sideways"}, headers=admin).status_code == 422
    assert client.post("/admin/queue/x/move", json={"to": "top"}, headers=admin).status_code == 409


def test_the_page_is_told_whether_the_controls_exist(admin, monkeypatch):
    assert client.get("/server-debug.json", headers=AUTH).json()["server"]["admin_enabled"] is True
    monkeypatch.setattr(settings, "ADMIN_TOKEN", "")
    assert client.get("/server-debug.json", headers=AUTH).json()["server"]["admin_enabled"] is False
