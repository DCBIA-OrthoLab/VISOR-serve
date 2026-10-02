"""What the operator page's timelines, history and graphs are built from.

`runs.timeline` turns a run's events into the bars the page draws -- its
phases, the calls it made to other tools, and the chain open right now -- and
the ledger keeps that shape after the run's directory is gone, on disk so it
survives a restart. Like the rest of the page, it is built from phases, times
and tool names only: a message is a tool's free text and can name a patient's
file, so the tests below plant one and check it never comes out.
"""

import json
import os

import pytest
from fastapi.testclient import TestClient

import telemetry
from config import settings
from main import app
from wire import runs

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {settings.API_TOKEN}"}


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "TEMP_DIR", str(tmp_path / "temp"))
    telemetry.reset()
    telemetry.configure_history(None)
    yield
    telemetry.reset()
    telemetry.configure_history(None)


def _event(at, phase="running", depth=0, tool=None, message="", measured=None):
    event = {"at": at, "phase": phase, "depth": depth, "message": message}
    if tool:
        event["tool"] = tool
    if measured:
        event["measured"] = measured
    return event


# ---------------------------------------------------------------------------
# runs.timeline
# ---------------------------------------------------------------------------

def test_spans_follow_the_root_phases_and_stop_at_the_terminal_one():
    shape = runs.timeline([
        _event(10, "received"), _event(11, "staging"), _event(12, "queued_gpu"),
        _event(20, "running"), _event(25, "running"), _event(40, "packaging"),
        _event(42, "done"),
    ])
    assert [(s["phase"], s["start"], s["end"]) for s in shape["spans"]] == [
        ("received", 10, 11), ("staging", 11, 12), ("queued_gpu", 12, 20),
        ("running", 20, 40), ("packaging", 40, 42),
    ]


def test_a_live_run_has_an_open_last_span():
    shape = runs.timeline([_event(1, "staging"), _event(2, "running")])
    assert shape["spans"][-1] == {"phase": "running", "start": 2, "end": None}


def test_nested_calls_pair_by_tool_and_depth_and_the_open_ones_are_the_chain():
    """AREG calls ASO, which calls ALI_CBCT; ASO has returned once already."""
    shape = runs.timeline([
        _event(1, "running"),
        _event(2, depth=1, tool="AMASSS"), _event(5, depth=1, tool="AMASSS"),
        _event(6, depth=1, tool="ASO"),
        _event(7, depth=2, tool="ALI_CBCT"),
        _event(8, depth=2, message="scan_of_jane_doe.nii.gz"),
    ])
    assert [(c["tool"], c["depth"], c["start"], c["end"]) for c in shape["nested"]] == [
        ("AMASSS", 1, 2, 5), ("ASO", 1, 6, None), ("ALI_CBCT", 2, 7, None),
    ]
    assert shape["chain"] == ["ASO", "ALI_CBCT"]
    # Child lines at depth > 0 do not open root spans.
    assert [s["phase"] for s in shape["spans"]] == ["running"]


def test_no_message_reaches_the_timeline():
    shape = runs.timeline([
        _event(1, "running", message="scan_of_jane_doe.nii.gz"),
        _event(2, depth=1, tool="ASO", message="jane_doe"),
        _event(3, "done", message="failed on jane_doe.nii.gz"),
    ])
    assert "jane" not in json.dumps(shape)


def test_measured_is_the_last_one_reported():
    shape = runs.timeline([
        _event(1, "running", measured={"vram_bytes": 1}),
        _event(2, "packaging", measured={"vram_bytes": 2}),
    ])
    assert shape["measured"] == {"vram_bytes": 2}


# ---------------------------------------------------------------------------
# The ledger, across a restart
# ---------------------------------------------------------------------------

def _finish(run_id, tool="AMASSS", timeline=None):
    telemetry.record_run_start(run_id, tool, "152.19.200.7")
    telemetry.record_run_end(run_id, "done", "done", timeline=timeline)


def test_a_finished_run_is_read_back_after_a_restart(tmp_path):
    telemetry.configure_history(str(tmp_path / "history"))
    shape = {"spans": [{"phase": "running", "start": 1, "end": 2}],
             "nested": [{"tool": "ASO", "depth": 1, "start": 1, "end": 2}],
             "measured": {"vram_bytes": 5}}
    _finish("run-one", timeline=shape)

    telemetry.reset()                       # the process ends
    assert telemetry.ledger_record("run-one") is None
    assert telemetry.configure_history(str(tmp_path / "history")) == 1

    record = telemetry.ledger_record("run-one")
    assert record["tool"] == "AMASSS"
    assert record["spans"] == shape["spans"]
    assert record["nested"] == shape["nested"]
    assert record["measured"] == {"vram_bytes": 5}


def test_a_run_still_going_is_not_written(tmp_path):
    telemetry.configure_history(str(tmp_path / "history"))
    telemetry.record_run_start("live", "AMASSS")
    assert not os.path.exists(tmp_path / "history" / telemetry.HISTORY_FILE)


def test_the_history_file_is_compacted_to_the_newest_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, "LEDGER_SIZE", 5)
    telemetry.configure_history(str(tmp_path / "history"))
    for index in range(5 * telemetry._COMPACT_FACTOR + 3):
        _finish(f"run-{index}")
    path = tmp_path / "history" / telemetry.HISTORY_FILE
    lines = path.read_text().splitlines()
    assert len(lines) <= 5 * telemetry._COMPACT_FACTOR
    assert json.loads(lines[-1])["run_id"] == f"run-{5 * telemetry._COMPACT_FACTOR + 2}"


def test_a_torn_line_is_skipped_rather_than_losing_the_history(tmp_path):
    directory = tmp_path / "history"
    directory.mkdir()
    (directory / telemetry.HISTORY_FILE).write_text(
        json.dumps({"run_id": "good", "tool": "ALI"}) + "\n" + '{"run_id": "torn', encoding="utf-8")
    assert telemetry.configure_history(str(directory)) == 1
    assert telemetry.ledger_record("good")["tool"] == "ALI"


def test_an_unusable_history_directory_leaves_the_ledger_in_memory(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")
    assert telemetry.configure_history(str(blocker / "history")) == 0
    _finish("still-recorded")
    assert telemetry.ledger_record("still-recorded")["outcome"] == "done"


# ---------------------------------------------------------------------------
# The resource trace
# ---------------------------------------------------------------------------

def test_the_trace_keeps_what_it_was_given_and_says_nothing_it_was_not():
    point = telemetry.sample_trace({"running": 2, "waiting": 1}, (100, 40))
    assert point["vram"] == 60
    assert point["running"] == 2 and point["waiting"] == 1
    bare = telemetry.sample_trace(None, (None, None))
    assert "vram" not in bare and "running" not in bare


def test_the_trace_is_thinned_without_losing_its_ends_ordering():
    for _ in range(50):
        telemetry.sample_trace()
    thinned = telemetry.trace(max_points=10)
    assert len(thinned) <= 10
    assert [p["at"] for p in thinned] == sorted(p["at"] for p in thinned)


# ---------------------------------------------------------------------------
# The endpoints the page reads
# ---------------------------------------------------------------------------

def test_server_debug_json_carries_the_trace_and_each_runs_chain():
    runs.register("chain-run-0000000000000001", tool="AREG", client="152.19.200.7")
    runs.append("chain-run-0000000000000001", runs.PHASE_RUNNING)
    path = runs.progress_file("chain-run-0000000000000001")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": 1e9, "depth": 1, "tool": "ASO"}) + "\n")
    payload = client.get("/server-debug.json", headers=AUTH).json()
    assert "trace" in payload
    entry = [r for r in payload["runs"] if r["run_id"] == "chain-run-0000000000000001"][0]
    assert entry["chain"] == ["ASO"]
    # /status keeps the listing it always had.
    status = client.get("/status", headers=AUTH).json()
    assert all("chain" not in r for r in status["runs"])


def test_a_reaped_run_is_still_drawn_from_the_ledger():
    _finish("reaped-run-000000000000001", timeline={
        "spans": [{"phase": "running", "start": 1, "end": 9}], "nested": [], "measured": None})
    payload = client.get("/server-debug/runs/reaped-run-000000000000001.json", headers=AUTH).json()
    assert payload["reaped"] is True
    assert payload["timeline"]["spans"] == [{"phase": "running", "start": 1, "end": 9}]
    assert payload["record"]["tool"] == "AMASSS"


def test_an_unknown_run_is_still_a_404():
    response = client.get("/server-debug/runs/never-seen-00000000000001.json", headers=AUTH)
    assert response.status_code == 404


def test_the_tool_view_summarises_that_tools_runs_only():
    _finish("a1", "AMASSS", {"spans": [{"phase": "queued_gpu", "start": 0, "end": 4},
                                      {"phase": "running", "start": 4, "end": 10}],
                            "nested": [], "measured": None})
    _finish("a2", "AMASSS", {"spans": [{"phase": "running", "start": 0, "end": 2}],
                            "nested": [], "measured": None})
    _finish("b1", "ALI")
    payload = client.get("/server-debug/tools/AMASSS.json", headers=AUTH).json()
    assert {r["run_id"] for r in payload["runs"]} == {"a1", "a2"}
    assert payload["summary"]["runs"] == 2 and payload["summary"]["ok"] == 2
    phases = {row["phase"]: row for row in payload["phases"]}
    assert phases["running"]["seconds"] == 8.0 and phases["running"]["mean"] == 4.0
    assert phases["queued_gpu"]["runs"] == 1
    assert "trace" in payload


def test_the_tool_view_needs_the_token():
    assert client.get("/server-debug/tools/AMASSS.json").status_code == 401
