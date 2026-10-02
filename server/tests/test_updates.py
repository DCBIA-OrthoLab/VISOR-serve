"""The admin panel's half of updating: the operator's door, and the request file.

The pulling itself happens on the host (scripts/update_agent.py, tested in
test_update_agent.py). What the server must get right:

* an operator can stop new runs for hours, and an updater's short lease must
  not cut that short -- only the operator reopens it;
* a request is one file, one at a time, withdrawable;
* the panel is told whether an agent is there at all, so "Update" never looks
  like it worked when nothing is listening.
"""

import json
import os
import time

import pytest
from fastapi.testclient import TestClient

import main
from config import settings
from wire import maintenance, updates

client = TestClient(main.app)


def _admin():
    return {"X-Admin-Token": settings.ADMIN_TOKEN}


@pytest.fixture(autouse=True)
def _open_door():
    maintenance.reopen()
    yield
    maintenance.reopen()


def test_an_operator_closes_the_door_for_hours_and_new_runs_are_refused():
    answer = client.post("/admin/door", json={"accepting": False, "hours": 3}, headers=_admin()).json()
    assert answer["accepting"] is False and answer["by_operator"] is True
    assert answer["closed_for"] > 2.9 * 3600
    refused = client.post("/run/Test_Tool", data={"text_1": "a", "text_2": "b"},
                          headers={"Authorization": f"Bearer {settings.API_TOKEN}"})
    assert refused.status_code == 503
    client.post("/admin/door", json={"accepting": True}, headers=_admin())
    assert maintenance.accepting()


def test_an_updaters_lease_does_not_shorten_an_operators_closure():
    maintenance.close_by_operator(3600, "before the update")
    maintenance.close(60, "updater lease")
    snap = maintenance.snapshot()
    assert snap["by_operator"] is True and snap["closed_for"] > 3000


def test_an_operator_closure_is_bounded():
    granted = maintenance.close_by_operator(10 * 24 * 3600)
    assert granted == maintenance.OPERATOR_MAX_CLOSE_SECONDS


def test_without_an_agent_the_panel_says_so():
    report = client.get("/admin-panel/updates.json", headers=_admin()).json()
    assert report["agent"] == {"seen": False, "alive": False, "heartbeat": None}
    assert report["status"] is None and report["request"] is None


def test_a_live_agents_status_is_passed_through():
    os.makedirs(settings.UPDATE_DIR, exist_ok=True)
    status = {"heartbeat": time.time(), "server": {"behind": 2}, "tools": {"behind": 0}}
    with open(os.path.join(settings.UPDATE_DIR, updates.STATUS_FILE), "w") as handle:
        json.dump(status, handle)
    report = client.get("/admin-panel/updates.json", headers=_admin()).json()
    assert report["agent"]["alive"] is True
    assert report["status"]["server"]["behind"] == 2


def test_a_silent_agent_is_reported_as_not_running():
    os.makedirs(settings.UPDATE_DIR, exist_ok=True)
    with open(os.path.join(settings.UPDATE_DIR, updates.STATUS_FILE), "w") as handle:
        json.dump({"heartbeat": time.time() - updates.AGENT_SILENT_SECONDS - 5}, handle)
    report = client.get("/admin-panel/updates.json", headers=_admin()).json()
    assert report["agent"]["seen"] is True and report["agent"]["alive"] is False


def test_one_request_at_a_time_and_it_can_be_withdrawn():
    first = client.post("/admin/update", json={"target": "tools"}, headers=_admin())
    assert first.status_code == 200 and first.json()["target"] == "tools"
    assert client.post("/admin/update", json={"target": "all"}, headers=_admin()).status_code == 409
    assert client.get("/admin-panel/updates.json", headers=_admin()).json()["request"]["target"] == "tools"
    assert client.delete("/admin/update", headers=_admin()).json() == {"withdrawn": True}
    assert client.delete("/admin/update", headers=_admin()).json() == {"withdrawn": False}


def test_an_unknown_target_is_refused():
    assert client.post("/admin/update", json={"target": "everything"}, headers=_admin()).status_code == 422


def test_only_the_admin_token_can_touch_any_of_it():
    api = {"Authorization": f"Bearer {settings.API_TOKEN}"}
    assert client.post("/admin/door", json={"accepting": False}, headers=api).status_code == 401
    assert client.post("/admin/update", json={"target": "all"}, headers=api).status_code == 401
    assert client.get("/admin-panel/updates.json", headers=api).status_code == 401
