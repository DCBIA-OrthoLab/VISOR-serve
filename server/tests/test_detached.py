"""A run that outlives the request that asked for it.

`POST /run` blocked for the whole inference, and that contract did not do what
it looked like it did: a client that disconnected never stopped the run -- no
worker thread is cancelled by anything in Starlette -- it only threw away the
answer. Meanwhile the Slicer client's own read timeout is 600 s against a
server sized for cohorts that take longer.

Detaching keeps every other part: the same staging, the same admission, the
same progress stream. What changes is that the result travels as a reference on
the terminal EVENT instead of in a response nobody is waiting for any more.

Opt-in by header, so a client that says nothing gets exactly what it always
got. Every test here asserts one half of that bargain.
"""

import io
import json
import os
import secrets

os.environ.setdefault("API_TOKEN", "test-token")

import pytest
from fastapi.testclient import TestClient

import main
from main import app
from wire import runs

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
DETACHED = {"X-Run-Delivery": "detached"}


def _new_id() -> str:
    return secrets.token_urlsafe(24)


@pytest.fixture
def detached_client():
    with TestClient(app, raise_server_exceptions=False) as instance:
        yield instance


def _post(client, run_id=None, headers=None, **kwargs):
    sent = dict(AUTH, **DETACHED)
    if run_id:
        sent["X-Run-Id"] = run_id
    sent.update(headers or {})
    return client.post(
        "/run/Test_Tool",
        data={"text_1": "a", "text_2": "b"},
        headers=sent,
        **kwargs,
    )


# ----------------------------------------------------------------------
# The bargain
# ----------------------------------------------------------------------

def test_a_client_that_says_nothing_still_blocks(detached_client):
    """The whole opt-in: byte for byte what it always did."""
    response = detached_client.post(
        "/run/Test_Tool", data={"text_1": "a", "text_2": "b"}, headers=AUTH
    )
    assert response.status_code == 200
    assert response.json() == {"result": "a b"}


def test_a_detached_run_is_accepted_at_once(detached_client):
    run_id = _new_id()
    response = _post(detached_client, run_id)
    assert response.status_code == 202
    assert response.json() == {"run_id": run_id, "status": "accepted"}
    runs.discard(run_id)


def test_a_detached_run_needs_somewhere_to_report(detached_client):
    """The event stream is the only channel it has left, and an id is what
    addresses it."""
    response = _post(detached_client)
    assert response.status_code == 400
    assert "X-Run-Id" in response.json()["detail"]


def test_a_detached_run_refuses_a_file_in_the_body(detached_client):
    """A multipart body is backed by a temporary file the framework closes when
    the response ends -- which here is before the tool has read a byte."""
    run_id = _new_id()
    response = detached_client.post(
        "/run/Test_Tool",
        data={"text_1": "a", "text_2": "b"},
        files={"whatever": ("scan.nii.gz", io.BytesIO(b"\x1f\x8b"), "application/gzip")},
        headers=dict(AUTH, **DETACHED, **{"X-Run-Id": run_id}),
    )
    assert response.status_code == 400
    assert "POST /uploads" in response.json()["detail"]


def test_a_refused_detached_run_leaves_no_directory_behind(detached_client):
    run_id = _new_id()
    detached_client.post(
        "/run/Test_Tool",
        data={"text_1": "a"},
        files={"whatever": ("scan.nii.gz", io.BytesIO(b"\x1f\x8b"), "application/gzip")},
        headers=dict(AUTH, **DETACHED, **{"X-Run-Id": run_id}),
    )
    with pytest.raises(runs.RunError):
        runs.run_directory(run_id)


# ----------------------------------------------------------------------
# How the answer gets back
# ----------------------------------------------------------------------

def _terminal(run_id):
    events = runs.read_events(run_id)
    assert events, "the run wrote no events at all"
    return events[-1]


def test_the_answer_arrives_on_the_terminal_event(detached_client):
    """The response that used to carry it was sent before the tool started."""
    run_id = _new_id()
    assert _post(detached_client, run_id).status_code == 202
    last = _terminal(run_id)
    assert last["state"] == "done"
    assert last["result"] == {"result": "a b"}
    runs.discard(run_id)


def test_the_run_directory_survives_the_response(detached_client):
    """The client has not read the terminal event yet -- writing one and then
    deleting it in the same breath would be pointless."""
    run_id = _new_id()
    _post(detached_client, run_id)
    assert runs.run_directory(run_id)
    runs.discard(run_id)


def test_a_failing_detached_run_says_so_without_naming_a_path(detached_client):
    """Nobody is left to raise to, so the failure has to travel as an event --
    under the same rule a response body follows."""
    run_id = _new_id()
    response = detached_client.post(
        "/run/Test_Tool",
        data={"text_1": "only one"},
        headers=dict(AUTH, **DETACHED, **{"X-Run-Id": run_id}),
    )
    assert response.status_code == 202
    last = _terminal(run_id)
    assert last["state"] == "failed"
    assert last["message"]
    assert "/" not in last["message"], "a failure message named a path"
    runs.discard(run_id)


def test_an_unknown_tool_fails_the_run_rather_than_the_request(detached_client):
    run_id = _new_id()
    response = detached_client.post(
        "/run/No_Such_Tool",
        data={"text_1": "a"},
        headers=dict(AUTH, **DETACHED, **{"X-Run-Id": run_id}),
    )
    assert response.status_code == 202
    assert _terminal(run_id)["state"] == "failed"
    runs.discard(run_id)


# ----------------------------------------------------------------------
# What a tool may not do
# ----------------------------------------------------------------------

def test_a_tool_cannot_write_its_own_result_reference(run_directory_for):
    """A tool appends to the same events.jsonl as the server. One able to write
    a `result` of its own could hand a client a pointer to somebody else's
    bytes."""
    run_id, directory = run_directory_for
    with open(os.path.join(directory, runs.EVENTS_FILE), "a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "at": 1.0, "fraction": 1.0, "message": "done",
            "result": {"result_ref": {"result_id": "stolen"}},
        }) + "\n")
    events = runs.read_events(run_id)
    assert "result" not in events[-1]


def test_the_server_s_own_result_survives_normalisation(run_directory_for):
    run_id, _directory = run_directory_for
    runs.finish(run_id, runs.PHASE_DONE, result={"result_ref": {"result_id": "mine"}})
    assert runs.read_events(run_id)[-1]["result"] == {"result_ref": {"result_id": "mine"}}


@pytest.fixture
def run_directory_for():
    run_id = _new_id()
    runs.register(run_id)
    yield run_id, runs.run_directory(run_id)
    runs.discard(run_id)


# ----------------------------------------------------------------------
# The failure message rule, on its own
# ----------------------------------------------------------------------

def test_a_caller_facing_message_travels_and_an_opaque_one_does_not():
    from execution import dispatch

    caller = dispatch.ToolFailure("ValueError", "landmark 'Zz' is not one of the 119")
    opaque = dispatch.ToolFailure("KeyError", "/opt/sadt/server/secret/path")
    assert main._failure_message(caller) == "landmark 'Zz' is not one of the 119"
    assert main._failure_message(opaque) == "Tool execution failed."
    assert "/opt" not in main._failure_message(opaque)
