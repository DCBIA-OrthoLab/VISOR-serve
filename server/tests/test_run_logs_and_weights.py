"""A tool's log lines, a weighted chain's single bar, and why a run failed.

Three things a run's events carry beyond phases and fractions:

- `sup.log(...)` lines, for the operator or for the requester, which a reader
  gets only when it asks -- a client released before they existed would draw
  one as progress.
- `sup.run(..., _progress=(start, end))`, which folds a callee's own 0..1 into
  the span of its caller's bar, so the root's bar moves by exactly the share
  each level was given.
- The diagnosis of a failed run: which tool in the chain, at which line, doing
  what -- redacted, and kept in the ledger after the run is gone.

No GPU, no tool process: the records are written the way the runner writes
them, which `test_supervisor.py` pins from the other side.
"""

import json
import os
import secrets

os.environ.setdefault("API_TOKEN", "test-token")

import pytest
from fastapi.testclient import TestClient

import registry
import telemetry
from base import Tool
from config import settings
from execution import dispatch
from main import _diagnosis, app
from redact import scrub
from wire import runs

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {settings.API_TOKEN}"}


def _panel():
    return {"X-Admin-Token": settings.ADMIN_TOKEN}


@pytest.fixture
def run_id():
    identifier = runs.register(secrets.token_urlsafe(24), tool="Caller")
    yield identifier
    runs.discard(identifier)


def _write(run_id, *records):
    path = os.path.join(settings.TEMP_DIR, "runs", run_id, runs.EVENTS_FILE)
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _open(call, tool, span=None, depth=None):
    record = {"fraction": None, "message": "", "depth": depth or call.count(".") + 1,
              "tool": tool, "call": call, "edge": "open"}
    if span:
        record["span"] = list(span)
    return record


def _close(call, tool, span=None, depth=None):
    return dict(_open(call, tool, span, depth), edge="close")


def _fractions(run_id):
    return [event["fraction"] for event in runs.read_events(run_id)
            if event.get("phase") == runs.PHASE_RUNNING]


# ---------------------------------------------------------------------------
# One bar for a weighted chain
# ---------------------------------------------------------------------------

def test_a_weighted_call_fills_its_span_of_the_callers_bar(run_id):
    """ASO gives ALI_CBCT 0.2..0.6: ALI at half way is ASO at 0.4, and when
    ALI is done ASO stands at 0.6 until it says otherwise."""
    _write(run_id,
           {"fraction": 0.2, "message": "centred", "depth": 0},
           _open("1", "ALI_CBCT", (0.2, 0.6)),
           {"fraction": 0.5, "message": "scan 2 of 4", "depth": 1, "call": "1"},
           _close("1", "ALI_CBCT", (0.2, 0.6)),
           {"fraction": 0.8, "message": "orienting", "depth": 0})

    assert _fractions(run_id) == [0.2, 0.2, 0.4, 0.6, 0.8]
    child = runs.read_events(run_id)[2]
    # The callee's own figure is still there for a client that wants it.
    assert child["own_fraction"] == 0.5


def test_spans_compose_down_a_chain(run_id):
    """AREG gives ASO the first half; ASO gives ALI its second half. ALI at
    half way is ASO at 0.75, which is AREG at 0.375."""
    _write(run_id,
           _open("1", "ASO", (0.0, 0.5)),
           _open("1.1", "ALI_CBCT", (0.5, 1.0)),
           {"fraction": 0.5, "message": "", "depth": 2, "call": "1.1"})

    assert _fractions(run_id)[-1] == 0.375


def test_a_tools_own_untagged_line_is_attributed_to_the_one_open_call(run_id):
    """A tool's `progress.py` writes neither depth nor call. Its caller is
    blocked in `sup.run` meanwhile, so the line is the callee's."""
    _write(run_id,
           _open("1", "Crown_Seg", (0.5, 1.0)),
           {"fraction": 0.5, "message": "mesh 1 of 2"})

    event = runs.read_events(run_id)[-1]
    assert event["fraction"] == 0.75
    assert event["depth"] == 1


def test_a_chain_nobody_weighted_reads_exactly_as_it_did(run_id):
    """No span anywhere: every record keeps the fraction it was written with,
    the callee's own 0..1 included -- the behaviour every tool has today."""
    _write(run_id,
           {"fraction": 0.1, "message": "", "depth": 0},
           _open("1", "Leaf"),
           {"fraction": 0.9, "message": "", "depth": 1, "call": "1"},
           _close("1", "Leaf"),
           {"fraction": 1.0, "message": "", "depth": 0})

    events = runs.read_events(run_id)
    assert [event["fraction"] for event in events] == [0.1, None, 0.9, None, 1.0]
    assert not any("own_fraction" in event for event in events)


def test_two_calls_open_at_once_make_an_untagged_line_a_guess_that_is_not_made(run_id):
    _write(run_id,
           _open("1", "ASO", (0.0, 0.5)),
           _open("2", "ASO", (0.5, 1.0)),
           {"fraction": 0.3, "message": ""})

    assert runs.read_events(run_id)[-1]["fraction"] == 0.3


def test_the_snapshot_and_the_listing_report_the_runs_bar(run_id):
    _write(run_id,
           _open("1", "ALI_CBCT", (0.2, 0.6)),
           {"fraction": 0.5, "message": "", "depth": 1, "call": "1"})

    assert runs.snapshot(run_id)["fraction"] == 0.4
    listed = [entry for entry in runs.active() if entry["run_id"] == run_id]
    assert listed[0]["fraction"] == 0.4


# ---------------------------------------------------------------------------
# Log lines reach only the reader that asked for them
# ---------------------------------------------------------------------------

def _log(message, level="info", audience="admin", **extra):
    return dict({"kind": "log", "level": level, "audience": audience,
                 "message": message, "depth": 0}, **extra)


def test_a_reader_that_asked_for_nothing_gets_no_log_line(run_id):
    """What the Slicer client released before this needs: a log line drawn as
    progress would blank its bar and show the operator's words."""
    _write(run_id, {"fraction": 0.5, "message": "halfway"},
           _log("an operator's line"), _log("a requester's line", audience="user"))

    events = runs.read_events(run_id)
    assert [event.get("kind") for event in events] == [None]
    assert runs.snapshot(run_id)["message"] == "halfway"


def test_a_client_may_ask_for_its_own_lines_and_never_the_operators(run_id):
    _write(run_id, _log("operator only"), _log("scan 4 skipped", "warning", "user"),
           {"fraction": 0.5, "message": "halfway"})

    for asked in ("user", "user,admin", "admin"):
        events = client.get(f"/runs/{run_id}?logs={asked}", headers=AUTH).json()["events"]
        logs = [event for event in events if event.get("kind") == "log"]
        expected = [] if asked == "admin" else ["scan 4 skipped"]
        assert [event["message"] for event in logs] == expected
    snapshot = client.get(f"/runs/{run_id}?logs=user", headers=AUTH).json()
    # Where the run stands is its progress, never a log line after it.
    assert snapshot["message"] == "halfway" and snapshot["fraction"] == 0.5


def test_the_event_stream_interleaves_requested_lines_in_file_order(run_id):
    _write(run_id, {"fraction": 0.1, "message": "a"},
           _log("visible", "info", "user"), _log("hidden"))
    runs.finish(run_id, runs.PHASE_DONE)

    with client.stream("GET", f"/runs/{run_id}/events?logs=user", headers=AUTH) as response:
        events = [json.loads(line[6:]) for line in response.iter_lines()
                  if line.startswith("data: ")]

    assert [event.get("kind") or event["phase"] for event in events] == ["running", "log", "done"]
    # One sequence for both, so a client deduplicating on `seq` loses nothing.
    assert [event["seq"] for event in events] == [0, 1, 3]


def test_a_log_line_cannot_claim_a_phase_or_end_a_stream(run_id):
    _write(run_id, dict(_log("pretending", audience="user"), phase="done", state="done"))

    event = runs.read_events(run_id, logs=("user",))[-1]
    assert event["state"] == runs.STATE_RUNNING and event["phase"] == runs.PHASE_RUNNING
    assert "fraction" not in event


def test_a_line_with_no_audience_is_the_operators(run_id):
    _write(run_id, {"kind": "log", "level": "error", "message": "unmarked"})

    assert runs.read_events(run_id, logs=("user",)) == []
    assert runs.read_events(run_id, logs=("admin",))[0]["audience"] == "admin"


def test_a_log_line_names_the_tool_whose_call_wrote_it(run_id):
    _write(run_id, _open("1", "ALI_CBCT"), _log("3 of 7 landmarks", "warning", call="1", depth=1))

    line = runs.read_events(run_id, logs=("admin",))[-1]
    assert line["source"] == "ALI_CBCT"


# ---------------------------------------------------------------------------
# The operator's view: redacted, and kept
# ---------------------------------------------------------------------------

def test_the_admin_console_shows_the_tools_lines_redacted(run_id):
    _write(run_id, _open("1", "AMASSS"),
           _log("no mandible in /tmp/in/P05_T1.nii.gz", "warning", call="1", depth=1))

    lines = client.get(f"/admin-panel/runs/{run_id}.json", headers=_panel()).json()["lines"]
    logged = [line for line in lines if line.get("kind") == "log"]
    assert logged[0]["level"] == "warn"
    assert logged[0]["text"] == "[AMASSS] no mandible in <path>"


def test_warnings_and_errors_outlive_the_run_redacted(run_id):
    _write(run_id, _log("chatter"), _log("skipped C_0001", "warning"),
           _log("cannot read /data/x.nrrd", "error"))
    runs.finish(run_id, runs.PHASE_FAILED, failure={
        "tool": "Caller", "chain": ["Caller"], "error_type": "KeyError", "reason": "<id>"})
    runs.discard(run_id)

    record = telemetry.ledger_record(run_id)
    assert [line["message"] for line in record["logs"]] == [
        "chatter", "skipped <id>", "cannot read <path>"]
    assert record["failure"]["error_type"] == "KeyError"
    payload = client.get(f"/admin-panel/runs/{run_id}.json", headers=_panel()).json()
    assert payload["reaped"] is True
    assert payload["lines"][-1]["text"].startswith("failed in Caller: KeyError")


class _Boom(Tool):
    """Fails the way an out-of-process tool does: with the runner's record."""

    name = "Boom_Tool"
    arguments = {}

    def run(self):
        raise dispatch.ToolFailure(
            "RuntimeError", "cannot open /tmp/job/P05_T1_Or.nii.gz",
            origin={"tool": "ALI_CBCT", "chain": ["AREG", "ASO", "ALI_CBCT"],
                    "error_type": "RuntimeError",
                    "message": "cannot open /tmp/job/P05_T1_Or.nii.gz",
                    "where": "sadt_ali_cbct/engine.py:742 in _predict",
                    "stage": "scan 3 of 8 (P05)", "fraction": 0.375})


def test_a_failed_run_says_where_and_why_and_never_whose(monkeypatch):
    monkeypatch.setitem(registry.TOOLS, "Boom_Tool", _Boom())
    identifier = secrets.token_urlsafe(24)

    response = client.post("/run/Boom_Tool", headers=dict(AUTH, **{"X-Run-Id": identifier}))

    assert response.status_code == 500
    assert response.json() == {"detail": "Tool execution failed."}
    failure = telemetry.ledger_record(identifier)["failure"]
    assert failure == {
        "tool": "ALI_CBCT", "chain": ["AREG", "ASO", "ALI_CBCT"],
        "error_type": "RuntimeError", "reason": "cannot open <path>", "status": 500,
        "where": "sadt_ali_cbct/engine.py:742 in _predict",
        "stage": "scan 3 of 8 (<id>)", "fraction": 0.375,
    }


def test_a_diagnosis_without_an_origin_falls_back_on_what_the_server_knows():
    failure = _diagnosis(dispatch.ToolExecutionError(
        "Tool 'AMASSS' exited with code -9:\nTraceback ... /tmp/P05.nii.gz"), "AMASSS")

    assert failure["chain"] == ["AMASSS"]
    assert failure["error_type"] == "ToolExecutionError"
    # The first line only: the stderr tail after it stays in the server's log.
    assert failure["reason"] == "Tool 'AMASSS' exited with code -9:"


def test_a_location_that_is_not_one_is_dropped():
    failure = _diagnosis(dispatch.ToolFailure("KeyError", "x", origin={
        "where": "/home/someone/P05.nii.gz", "chain": ["A", "../etc"]}), "A")

    assert "where" not in failure
    assert failure["chain"] == ["A"]


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("No scan found in /tmp/job/P05_T1_Or.nii.gz.", "No scan found in <path>"),
    ("skipped pairs: C_0001 (no T2), P_002", "skipped pairs: <id> (no T2), <id>"),
    ("read case.nrrd", "read <file>"),
    ("from 10.0.0.12 by a@b.org", "from <address> by <email>"),
    ("scan 14 of 40, mesh 3/40", "scan 14 of 40, mesh 3/40"),
    ("CUDA out of memory. Tried to allocate 2.00 GiB", "CUDA out of memory. Tried to allocate 2.00 GiB"),
    ("two\nlines", "two lines"),
])
def test_redaction_keeps_the_sentence_and_drops_the_names(text, expected):
    assert scrub(text) == expected


def test_a_chatty_tool_cannot_push_its_error_out_of_the_history(run_id):
    _write(run_id, _log("the one error", "error"), *[_log(f"step {i}") for i in range(60)])
    runs.finish(run_id, runs.PHASE_DONE)

    kept = telemetry.ledger_record(run_id)["logs"]
    assert kept[0]["message"] == "the one error"
    assert len(kept) == 21 and kept[-1]["message"] == "step 59"
