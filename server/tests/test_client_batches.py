"""A workstation's cohort batches: how they are named, grouped and gated.

A client cutting a cohort into batches tags every batch with one id, its index
and the total. The server keeps that with the run, groups the batches per
workstation on the dashboard, and -- per workstation -- lets them run side by
side or one at a time in index order. What must hold:

* a malformed batch is ignored, never a refused run;
* serial means one batch of a cohort at a time, lowest index first, whatever
  order they arrive in; parallel lets them all through;
* two cohorts, or two workstations, never wait on each other;
* the rule is the operator's: it survives a restart and only the admin token
  changes it.
"""

import json

import anyio
import pytest
from fastapi.testclient import TestClient

import main
import resources
import telemetry
from config import settings
from execution import admission
from wire import clients, runs

client = TestClient(main.app)
AUTH = {"Authorization": f"Bearer {settings.API_TOKEN}"}


def _panel():
    """The admin token, read when a request is made: tests set it per test."""
    return {"X-Admin-Token": settings.ADMIN_TOKEN}


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "TEMP_DIR", str(tmp_path / "temp"))
    clients.reset()
    telemetry.reset()
    yield
    clients.reset()
    telemetry.reset()


def _batch(index, total=3, batch_id="cohort-aaaaaaaa"):
    return {"id": batch_id, "index": index, "total": total}


# ---------------------------------------------------------------------------
# The headers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    (("cohort-aaaaaaaa", "2", "5"), {"id": "cohort-aaaaaaaa", "index": 2, "total": 5}),
    ((None, "1", "2"), None),
    (("short", "1", "2"), None),
    (("cohort/../x-aaaa", "1", "2"), None),
    (("cohort-aaaaaaaa", "0", "2"), None),
    (("cohort-aaaaaaaa", "3", "2"), None),
    (("cohort-aaaaaaaa", "one", "2"), None),
])
def test_a_batch_is_read_only_when_it_is_plausible(raw, expected):
    assert runs.parse_batch(*raw) == expected


def test_a_run_keeps_its_batch_and_the_listing_shows_it():
    runs.register("batch-run-0000000000000001", tool="AMASSS", client="10.0.0.5", batch=_batch(2))
    assert runs.meta("batch-run-0000000000000001")["batch"] == _batch(2)
    listed = [r for r in runs.active() if r["run_id"] == "batch-run-0000000000000001"][0]
    assert listed["batch"] == _batch(2)
    assert telemetry.ledger_record("batch-run-0000000000000001")["batch"] == _batch(2)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def _offer(address, batch, run_id):
    clients.wait(address, batch, run_id)
    return clients.may_start(address, batch, run_id)


def test_a_higher_batch_arriving_first_waits_for_the_lower_ones(monkeypatch):
    """Sent together, batch 4 can reach the server before batch 1."""
    assert not _offer("10.0.0.5", _batch(3), "r3"), "batch 3 started before 1 and 2 arrived"
    assert _offer("10.0.0.5", _batch(1), "r1")


def test_a_lower_batch_that_never_comes_holds_the_rest_only_briefly(monkeypatch):
    monkeypatch.setattr(clients, "ARRIVAL_GRACE_SECONDS", 0.0)
    assert _offer("10.0.0.5", _batch(3), "r3")


def test_a_serial_client_sending_one_at_a_time_never_waits_for_the_grace():
    """Batch 2 arrives after batch 1 has finished, which it remembers."""
    assert _offer("10.0.0.5", _batch(1), "r1")
    clients.leave("10.0.0.5", _batch(1), "r1")
    assert _offer("10.0.0.5", _batch(2), "r2")


def test_serial_lets_one_batch_through_lowest_index_first():
    clients.wait("10.0.0.5", _batch(3), "r3")
    clients.wait("10.0.0.5", _batch(1), "r1")
    clients.wait("10.0.0.5", _batch(2), "r2")
    assert not clients.may_start("10.0.0.5", _batch(3), "r3"), "index 3 jumped index 1"
    assert clients.may_start("10.0.0.5", _batch(1), "r1")
    assert not clients.may_start("10.0.0.5", _batch(2), "r2"), "two batches ran at once"
    clients.leave("10.0.0.5", _batch(1), "r1")
    assert clients.may_start("10.0.0.5", _batch(2), "r2")
    clients.leave("10.0.0.5", _batch(2), "r2")
    assert clients.may_start("10.0.0.5", _batch(3), "r3")


def test_parallel_lets_every_batch_through():
    clients.set_policy("10.0.0.5", clients.PARALLEL)
    assert all(_offer("10.0.0.5", _batch(i), f"r{i}") for i in (1, 2, 3))


def test_two_cohorts_and_two_workstations_never_wait_on_each_other():
    assert _offer("10.0.0.5", _batch(1, batch_id="cohort-aaaaaaaa"), "a1")
    assert _offer("10.0.0.5", _batch(1, batch_id="cohort-bbbbbbbb"), "b1")
    assert _offer("10.0.0.6", _batch(1, batch_id="cohort-aaaaaaaa"), "c1")


def test_a_batch_that_gives_up_frees_its_turn():
    clients.wait("10.0.0.5", _batch(1), "r1")
    clients.wait("10.0.0.5", _batch(2), "r2")
    clients.leave("10.0.0.5", _batch(1), "r1")      # cancelled before its turn
    assert clients.may_start("10.0.0.5", _batch(2), "r2")


def test_the_gate_holds_a_serial_batch_until_its_sibling_ends(monkeypatch):
    allocation = resources.Allocation(cpus=8.0, ram_bytes=16 << 30, vram_bytes=8 << 30, cpus_per_job=2,
                                      ram_per_job=4 << 30, vram_per_job=2 << 30, expected_clients=4)
    monkeypatch.setattr(admission, "_budget", admission.Budget(allocation, free_vram=lambda: 1 << 60))
    for index, run_id in ((1, "slot-batch-0000000000000001"), (2, "slot-batch-0000000000000002")):
        runs.register(run_id, tool="AMASSS", client="10.0.0.5", batch=_batch(index, total=2))

    async def scenario():
        monkeypatch.setattr(main, "_PRIORITY_POLL_SECONDS", 0.01)
        order, release_first = [], anyio.Event()

        async def run(run_id, hold=None):
            async with main._batch_turn(run_id):
                order.append(run_id[-1])
                if hold is not None:
                    await hold.wait()

        async with anyio.create_task_group() as group:
            group.start_soon(run, "slot-batch-0000000000000001", release_first)
            await anyio.sleep(0.05)
            group.start_soon(run, "slot-batch-0000000000000002")
            await anyio.sleep(0.1)
            assert order == ["1"], "the second batch started beside the first"
            release_first.set()
        assert order == ["1", "2"]

    anyio.run(scenario)


# ---------------------------------------------------------------------------
# The rule, and who sets it
# ---------------------------------------------------------------------------

def test_the_rule_survives_a_restart(tmp_path):
    clients.configure(str(tmp_path))
    clients.set_policy("10.0.0.5", clients.PARALLEL)
    clients.reset()
    assert clients.policy_for("10.0.0.5") == clients.SERIAL
    clients.configure(str(tmp_path))
    assert clients.policy_for("10.0.0.5") == clients.PARALLEL
    stored = json.loads((tmp_path / clients.POLICY_FILE).read_text())
    assert set(stored) == {"10.0.0.5"}


def test_setting_the_default_forgets_the_exception():
    clients.set_policy("10.0.0.5", clients.PARALLEL)
    clients.set_policy("10.0.0.5", clients.SERIAL)
    assert clients.policies() == {}


def test_a_workstation_reads_its_own_rule():
    answer = client.get("/clients/me", headers=AUTH).json()
    assert answer["batches"] == clients.SERIAL and answer["max_parallel"] == 1
    clients.set_policy(answer["client"], clients.PARALLEL)
    answer = client.get("/clients/me", headers=AUTH).json()
    assert answer["batches"] == clients.PARALLEL
    budget = admission.budget()
    assert answer["max_parallel"] == max(1, int(budget.cpus // budget.cpus_per_job))


def test_only_the_admin_token_changes_a_rule(monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_TOKEN", "operator-secret")
    path = "/admin/clients/10.0.0.5/policy"
    assert client.post(path, json={"batches": "parallel"}, headers=AUTH).status_code == 401
    admin = {**AUTH, "X-Admin-Token": "operator-secret"}
    assert client.post(path, json={"batches": "sideways"}, headers=admin).status_code == 422
    assert client.post(path, json={"batches": "parallel"}, headers=admin).json() == {
        "client": "10.0.0.5", "batches": "parallel"}


# ---------------------------------------------------------------------------
# What the dashboard draws
# ---------------------------------------------------------------------------

def test_the_dashboard_groups_a_workstations_batches_into_cohorts():
    for index in (1, 2):
        run_id = f"cohort-done-00000000000000{index}"
        runs.register(run_id, tool="AMASSS", client="10.0.0.5", batch=_batch(index, total=4))
        telemetry.record_run_end(run_id, "done", "done")
    runs.register("cohort-live-0000000000000003", tool="AMASSS", client="10.0.0.5", batch=_batch(3, total=4))
    runs.append("cohort-live-0000000000000003", runs.PHASE_RUNNING)
    payload = client.get("/admin-panel.json", headers=_panel()).json()
    row = [c for c in payload["clients"] if c["client"] == "10.0.0.5"][0]
    assert row["batches"] == clients.SERIAL and row["running"] == 1
    cohort = row["cohorts"][0]
    assert (cohort["tool"], cohort["total"], cohort["done"], cohort["running"]) == ("AMASSS", 4, 2, 1)
    assert row["active_cohorts"] == 1
