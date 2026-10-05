"""The door an update closes, and the guarantee that it opens again.

Two properties are asserted here and nothing else will do:

* while the door is shut, the endpoints that START work are refused and every
  endpoint a client needs to FINISH work still answers. An update that stranded
  a clinician mid-run would be worse than no update at all;
* the door reopens by itself. The updater runs on another machine's terms --
  it can be killed, it can lose the network -- and a server that stayed shut
  because nobody came back is the failure this design exists to remove.
"""

import time

import pytest
from fastapi.testclient import TestClient

from config import settings
from main import app
from wire import maintenance

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {settings.API_TOKEN}"}
RUN_ARGS = {"text_1": "hello", "text_2": "world"}


@pytest.fixture(autouse=True)
def door_open_again():
    """Every test starts and ends with the door open.

    The state is a module global, so a test that closed it and failed would
    otherwise refuse every run in every test that follows -- and the failure
    would be reported against innocent tests.
    """
    maintenance.reopen()
    yield
    maintenance.reopen()


# ---------------------------------------------------------------------------
# The door itself
# ---------------------------------------------------------------------------

def test_a_server_starts_accepting_work():
    assert maintenance.accepting() is True
    assert maintenance.snapshot() == {"accepting": True, "closed_for": None, "reason": "", "by_operator": False}


def test_the_door_reopens_on_its_own_when_nobody_comes_back():
    """The deadman. This is the property that makes it safe to close the door
    from another machine: the updater may die between closing it and restarting
    the process, and no operator has to notice."""
    maintenance.close(0.3, "an updater that is about to vanish")
    assert maintenance.accepting() is False

    time.sleep(0.4)

    assert maintenance.accepting() is True
    assert maintenance.snapshot()["closed_for"] is None


def test_no_caller_can_shut_the_server_for_longer_than_the_cap():
    """Asked for a day, granted the cap -- and told so, rather than believing
    it has a day. A caller that needs longer asks again, which is also what
    proves it is still alive."""
    granted = maintenance.close(86400, "far too long")
    assert granted == maintenance.MAX_CLOSE_SECONDS
    assert maintenance.snapshot()["closed_for"] <= maintenance.MAX_CLOSE_SECONDS


def test_closing_for_no_time_is_the_same_as_opening():
    maintenance.close(0)
    assert maintenance.accepting() is True


# ---------------------------------------------------------------------------
# What is refused, and what must not be
# ---------------------------------------------------------------------------

def test_a_shut_door_refuses_a_run_with_the_delay_to_wait():
    maintenance.close(60, "update")
    response = client.post("/run/Test_Tool", headers=AUTH, data=RUN_ARGS)

    assert response.status_code == 503
    # Without this a client has nothing to schedule a retry from.
    assert response.headers["Retry-After"] == str(maintenance.RETRY_AFTER_SECONDS)


def test_a_shut_door_refuses_a_new_upload_session():
    maintenance.close(60, "update")
    response = client.post(
        "/uploads", headers=AUTH, json={"filename": "scan.nii.gz", "size": 1024}
    )
    assert response.status_code == 503


def test_a_shut_door_still_lets_a_client_finish_what_it_started():
    """The endpoints a run in flight depends on. Refusing these would strand
    the very people an update is meant to serve."""
    maintenance.close(60, "update")

    assert client.get("/health").status_code == 200
    assert client.get("/status", headers=AUTH).status_code == 200
    assert client.get("/tools").status_code == 200
    # Unknown ids, so what is asserted is that the door let the request THROUGH
    # to the handler -- a 404 or a 410 is the handler answering, a 503 is not.
    assert client.get("/results/does-not-exist", headers=AUTH).status_code != 503
    assert client.get("/runs/does-not-exist", headers=AUTH).status_code != 503
    assert client.delete("/runs/does-not-exist", headers=AUTH).status_code != 503


def test_a_transfer_already_under_way_is_not_cut_off():
    """A PUT of an upload part writes into TEMP_DIR, which survives the reload
    an update triggers. Refusing it would throw away bytes already sent."""
    opened = client.post(
        "/uploads", headers=AUTH, json={"filename": "scan.nii.gz", "size": 8}
    )
    assert opened.status_code == 200
    upload_id = opened.json()["upload_id"]

    maintenance.close(60, "update")
    response = client.put(
        f"/uploads/{upload_id}/parts/0", headers=AUTH, content=b"12345678"
    )
    assert response.status_code != 503


def test_a_shut_door_refuses_every_way_of_starting_a_run():
    """A resume IS a run, and so is a rewind.

    This is the hole the first draft of the gate had. It matched paths by
    prefix, and `"/run/"` does not match `/runs/{id}/resume` -- so a paused
    cohort could be resumed, on the GPU, in the middle of an update. The
    codebase had already made the mirror-image mistake next door:
    `_UNCOUNTED_PATHS`'s `"/runs/"` swallows the same endpoint and stops
    counting it as work. Matching by hand is what both have in common.
    """
    maintenance.close(60, "update")

    assert client.post("/runs/whatever/resume", headers=AUTH, json={}).status_code == 503
    assert client.post("/runs/whatever/rewind", headers=AUTH, json={}).status_code == 503


def test_every_endpoint_that_starts_work_is_gated():
    """The drift guard, and the reason a dependency is safe to forget.

    A new POST route arrives open. That is the right default -- a route that
    should have been gated and was not costs one refused-too-late request
    during an update, while a route wrongly gated refuses a clinician for
    ever. But "open by default" only stays honest if adding one is a DECISION,
    so this test fails until the new route is either gated or named below.
    """
    from main import app
    from wire.maintenance import require_accepting

    # Endpoints that legitimately answer while the door is shut. Each one is
    # something a client needs to FINISH work it already started, or the
    # control that opens the door again.
    # The operator's controls start nothing: they reorder or mark runs, or
    # set how a workstation's batches run, which is exactly what an operator
    # may need to do while an update is draining the server.
    ALLOWED_OPEN = {"/maintenance", "/admin/queue/{run_id}/move", "/admin/runs/{run_id}/priority",
                    "/admin/clients/{address}/policy", "/admin/door", "/admin/update", "/admin/data",
                    "/admin/updates/check",
                    # Answers which file NAMES go together; starts no run.
                    "/tools/{tool_name}/pairs"}
    # This is also what catches a route that does not exist yet. A benchmark
    # battery is a run too, and the day `POST /benchmark/run` lands it arrives
    # here ungated and fails this test until somebody decides.

    open_routes = {
        route.path
        for route in app.routes
        if "POST" in (getattr(route, "methods", None) or set())
        and require_accepting not in [d.call for d in route.dependant.dependencies]
    }
    assert open_routes == ALLOWED_OPEN, (
        f"POST routes neither gated nor declared open: {sorted(open_routes - ALLOWED_OPEN)}"
    )


def test_an_open_door_changes_nothing():
    """The whole mechanism is invisible when no update is being applied."""
    response = client.post("/run/Test_Tool", headers=AUTH, data=RUN_ARGS)
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# The endpoint the updater uses
# ---------------------------------------------------------------------------

def test_shutting_the_door_needs_the_token():
    """It is a control over whether this deployment serves anyone."""
    assert client.post("/maintenance", json={"accepting": False}).status_code == 401


def test_the_updater_closes_and_reopens_over_http():
    closed = client.post(
        "/maintenance", headers=AUTH,
        json={"accepting": False, "seconds": 30, "reason": "applying an update"},
    )
    assert closed.status_code == 200
    assert closed.json()["accepting"] is False
    assert closed.json()["reason"] == "applying an update"
    assert client.post("/run/Test_Tool", headers=AUTH, data=RUN_ARGS).status_code == 503

    reopened = client.post("/maintenance", headers=AUTH, json={"accepting": True})
    assert reopened.status_code == 200
    assert reopened.json()["accepting"] is True
    assert client.post("/run/Test_Tool", headers=AUTH, data=RUN_ARGS).status_code == 200


def test_the_endpoint_answers_with_what_it_granted_not_what_was_asked():
    response = client.post(
        "/maintenance", headers=AUTH, json={"accepting": False, "seconds": 86400}
    )
    assert response.json()["closed_for"] <= maintenance.MAX_CLOSE_SECONDS


def test_status_says_whether_this_server_is_accepting_work():
    """The updater polls this to know its own request landed, and an operator
    reads it to tell an update in progress from a door left shut."""
    assert client.get("/status", headers=AUTH).json()["maintenance"]["accepting"] is True

    maintenance.close(45, "update")
    reported = client.get("/status", headers=AUTH).json()["maintenance"]
    assert reported["accepting"] is False
    assert reported["reason"] == "update"
    assert 0 < reported["closed_for"] <= 45
