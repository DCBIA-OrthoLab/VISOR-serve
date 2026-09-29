"""What `/server-debug` is drawn from, and the two rules it must not break.

The page itself is a string of HTML and cannot usefully be asserted on. What
CAN be asserted is everything underneath it: the samplers in `telemetry.py`,
the ledger they keep, and the shape of `GET /server-debug.json` that the
diagram reads. So that is what these cover.

Two of them are not shape tests but rules, and they are the reason this file
exists at all:

* **No argument VALUE ever reaches the ledger.** A value is a path and a path
  is a patient's file name. `record_run_inputs` takes names only, and the test
  below passes it a dict to prove that what lands is the keys.
* **An unavailable reading is `None`, never `0`.** The page draws `None`
  hatched and `0` as an empty tank, and "empty" reads as "idle" -- which is a
  claim about a machine nobody measured.
"""

import os

import pytest
from fastapi.testclient import TestClient

import telemetry
from config import settings
from main import app

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {settings.API_TOKEN}"}


@pytest.fixture(autouse=True)
def _clean_telemetry():
    """Each test starts from a server that has measured nothing."""
    telemetry.reset()
    yield
    telemetry.reset()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_the_first_cpu_reading_is_none_because_there_is_no_interval_yet():
    """A percentage is a delta. With one sample there is nothing to subtract.

    Answering 0 would be a lie a reader cannot detect, and answering the
    machine's lifetime average would be a different lie.
    """
    assert telemetry.cpu_percent() is None


def test_the_second_cpu_reading_is_a_percentage():
    telemetry.cpu_percent()
    value = telemetry.cpu_percent()
    # None is still legitimate here: two calls can land inside one jiffy.
    assert value is None or 0.0 <= value <= 100.0


def test_ram_reports_available_rather_than_free():
    """MemAvailable, not MemFree: a server that has just written a 2 GB result
    has almost no free memory and is under no memory pressure at all."""
    reading = telemetry.ram()
    if reading is None:  # no /proc on this platform
        return
    assert reading["total"] > 0
    assert 0 <= reading["available"] <= reading["total"]
    assert reading["used"] == reading["total"] - reading["available"]


def test_a_directory_that_does_not_exist_reports_absent_not_zero(tmp_path):
    missing = str(tmp_path / "nowhere")
    report = telemetry.disk({"gone": missing})
    assert report["gone"]["exists"] is False
    assert report["gone"]["used_here"] is None


def test_disk_counts_what_is_under_the_path(tmp_path):
    (tmp_path / "blob").write_bytes(b"x" * 20000)
    report = telemetry.disk({"here": str(tmp_path)})
    assert report["here"]["exists"] is True
    assert report["here"]["used_here"] >= 20000
    assert report["here"]["total"] > 0


def test_in_flight_requests_count_up_and_back_down():
    telemetry.request_started()
    telemetry.request_started()
    assert telemetry.inflight() == {"now": 2, "peak": 2}
    telemetry.request_finished()
    assert telemetry.inflight() == {"now": 1, "peak": 2}


def test_in_flight_never_goes_negative():
    """A double-decrement would otherwise make the gauge read below empty and
    never recover, since the peak is taken from the same counter."""
    telemetry.request_finished()
    assert telemetry.inflight()["now"] == 0


# ---------------------------------------------------------------------------
# The queue
# ---------------------------------------------------------------------------

def test_the_queue_is_ranked_by_total_wait_not_by_the_worst_one():
    """The decision the page is built on. A tool that queues four seconds a
    hundred times costs this server more than one that waited a minute once."""
    for _ in range(10):
        telemetry.record_admission("CLIC", 4.0, channels=1)
    telemetry.record_admission("AMASSS", 30.0, channels=1)

    ranked = telemetry.queue_history()["ranked"]
    assert [row["tool"] for row in ranked] == ["CLIC", "AMASSS"]
    assert ranked[0]["total_wait"] == 40.0
    # The worst is kept beside it, because it is the one somebody felt.
    assert ranked[1]["worst_wait"] == 30.0


def test_a_run_that_did_not_wait_is_counted_but_not_called_queued():
    telemetry.record_admission("Test_Tool", 0.0, channels=1)
    row = telemetry.queue_history()["ranked"][0]
    assert row["runs"] == 1
    assert row["queued"] == 0


def test_the_queue_log_is_bounded():
    for index in range(telemetry.QUEUE_LOG_SIZE + 50):
        telemetry.record_admission("CLIC", 0.0, channels=1)
    history = telemetry.queue_history()
    assert history["held"] == telemetry.QUEUE_LOG_SIZE


def test_a_bad_admission_record_is_dropped_rather_than_raising():
    """Telemetry must never fail a run. A record it cannot parse is one lost
    row on a page, not an exception on the path of a cohort."""
    telemetry.record_admission("CLIC", "not a number", channels=1)
    assert telemetry.queue_history()["held"] == 0


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

def test_the_ledger_records_what_a_run_held_and_how_it_ended():
    telemetry.record_run_start("run-a", "AMASSS")
    telemetry.record_run_grant("run-a", channels=2, cpus=7,
                               ram_bytes=1024, vram_bytes=2048, waited=4.25)
    telemetry.record_run_inputs("run-a", 40, 999, ["scans", "model"])
    telemetry.record_run_end("run-a", "done", "done")

    record = telemetry.run_ledger()[0]
    assert record["tool"] == "AMASSS"
    assert (record["channels"], record["cpus"]) == (2, 7)
    assert record["vram_bytes"] == 2048
    assert record["files"] == 40
    assert record["outcome"] == "done"
    assert record["seconds"] is not None


def test_the_ledger_keeps_argument_names_and_never_their_values():
    """The rule this whole page is constrained by. A value is a path, and a
    path is the patient's."""
    telemetry.record_run_start("run-b", "ALI_CBCT")
    telemetry.record_run_inputs("run-b", 1, 10, {
        "input": "/jobs/ab12/Smith_John_T1.nii.gz",
        "landmarks": ["Ba", "S"],
    })

    record = telemetry.run_ledger()[0]
    assert record["arguments"] == ["input", "landmarks"]
    serialised = repr(record)
    assert "Smith_John" not in serialised
    assert ".nii.gz" not in serialised


def test_the_ledger_is_bounded_and_keeps_the_newest():
    for index in range(telemetry.LEDGER_SIZE + 20):
        telemetry.record_run_start(f"run-{index}", "CLIC")
    ledger = telemetry.run_ledger(limit=telemetry.LEDGER_SIZE + 50)
    assert len(ledger) == telemetry.LEDGER_SIZE
    assert ledger[0]["run_id"] == f"run-{telemetry.LEDGER_SIZE + 19}"


def test_ending_a_run_nobody_started_is_ignored():
    telemetry.record_run_end("never-seen", "done")
    assert telemetry.run_ledger() == []


# ---------------------------------------------------------------------------
# Per-tool uptime
# ---------------------------------------------------------------------------

def test_uptime_sums_the_time_each_tool_occupied_the_server():
    telemetry.record_run_start("u1", "AMASSS")
    telemetry.record_run_end("u1", "done", "done")
    telemetry.record_run_start("u2", "AMASSS")
    telemetry.record_run_end("u2", "failed", "failed")
    telemetry.record_run_start("u3", "CLIC")

    rows = {row["tool"]: row for row in telemetry.tool_uptime()}
    assert rows["AMASSS"]["runs"] == 2
    assert rows["AMASSS"]["ok"] == 1
    assert rows["AMASSS"]["failed"] == 1
    assert rows["AMASSS"]["running"] == 0
    # Still going: counted as in progress rather than left out until it ends.
    assert rows["CLIC"]["running"] == 1
    assert rows["CLIC"]["ok"] == 0


def test_uptime_is_ordered_busiest_first():
    telemetry.record_run_start("s1", "Quick")
    telemetry.record_run_end("s1", "done")
    telemetry.record_run_start("s2", "Slow")
    ledger = telemetry._ledger  # noqa: SLF001 - reaching in to age a record
    ledger["s2"]["started_at"] -= 600
    telemetry.record_run_end("s2", "done")

    assert [row["tool"] for row in telemetry.tool_uptime()][0] == "Slow"


# ---------------------------------------------------------------------------
# The endpoint the diagram reads
# ---------------------------------------------------------------------------

def test_server_debug_json_needs_the_token():
    assert client.get("/server-debug.json").status_code == 401


def test_server_debug_json_carries_every_block_the_diagram_draws():
    """The page reads these by name. A rename here is a blank panel there."""
    payload = client.get("/server-debug.json", headers=AUTH).json()
    for key in ("budget", "admission", "card", "runs", "costs", "hardware",
                "cpu_percent", "ram", "inflight", "queue", "ledger", "uptime",
                "disk"):
        assert key in payload, key
    assert set(payload["queue"]) >= {"order", "ranked", "capacity", "held"}
    assert set(payload["disk"]) == {"temp", "data"}


def test_server_debug_json_agrees_with_status_on_what_both_report():
    """They are one function plus samples, deliberately, so that two endpoints
    can never disagree about the same number."""
    status = client.get("/status", headers=AUTH).json()
    debug = client.get("/server-debug.json", headers=AUTH).json()
    assert debug["budget"] == status["budget"]
    assert debug["costs"] == status["costs"]


def test_watching_the_server_is_not_counted_as_asking_it_for_work():
    """The observer must not appear in its own observation.

    The dashboard polls every two seconds, so counting its own request made the
    chip read "1 request" on a completely idle machine -- a claim about load
    that came entirely from the thing measuring the load.
    """
    telemetry.reset()
    client.get("/server-debug.json", headers=AUTH)
    client.get("/status", headers=AUTH)
    client.get("/health")
    assert telemetry.inflight() == {"now": 0, "peak": 0}


def test_a_request_that_asks_for_something_is_counted():
    """The other half: excluding the watchers must not exclude everything."""
    telemetry.reset()
    client.get("/tools")
    assert telemetry.inflight()["peak"] == 1


def test_the_debug_page_is_served_without_a_token_and_holds_none():
    """The page is public; the readings are not. A token baked into the HTML
    would hand every reader the key to the whole API."""
    response = client.get("/server-debug")
    assert response.status_code == 200
    assert settings.API_TOKEN not in response.text


def test_the_debug_page_asks_for_nothing_over_the_network():
    """This server runs where there is no internet. A remote stylesheet is a
    broken page, not a degraded one."""
    body = client.get("/server-debug").text
    for absent in ("http://", "https://", "//cdn", "<link"):
        assert absent not in body, absent
