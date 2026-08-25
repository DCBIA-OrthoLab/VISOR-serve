"""Raw data is append-only, and a failure is a record. Both are tested here,
because both are claims the paper makes about its own evidence."""

from __future__ import annotations

import json
import os

import pytest

from benchmarks.recording import (
    PHASES,
    STATUS_FAILED,
    STATUS_OK,
    PhaseTimer,
    Recorder,
    RunRecord,
    load_records,
    read_raw,
)


def _record(**overrides) -> RunRecord:
    fields = {
        "campaign": "b1",
        "tool": "Widget",
        "path": "loopback",
        "repetition": 1,
        "started_at": "2026-08-25T00:00:00+00:00",
        "finished_at": "2026-08-25T00:00:01+00:00",
        "total_seconds": 1.0,
        "status": STATUS_OK,
    }
    fields.update(overrides)
    return RunRecord(**fields)


def test_a_record_round_trips_through_json():
    record = _record(phases={"server_exec": 0.5, "other": 0.5}, extra={"payload": "16MB"})
    restored = json.loads(record.as_json_line())
    assert restored["tool"] == "Widget"
    assert restored["phases"]["server_exec"] == 0.5
    assert restored["extra"]["payload"] == "16MB"
    assert restored["harness_version"]


def test_the_raw_file_is_never_reopened(isolated_root):
    recorder = Recorder(isolated_root, "b1", timestamp="20260825T000000Z")
    recorder.write(_record())
    recorder.close()
    with pytest.raises(FileExistsError):
        Recorder(isolated_root, "b1", timestamp="20260825T000000Z")


def test_records_are_appended_not_overwritten(isolated_root):
    with Recorder(isolated_root, "b1", timestamp="20260825T000001Z") as recorder:
        for repetition in range(1, 4):
            recorder.write(_record(repetition=repetition))
        path = recorder.path
    assert len(read_raw(path)) == 3


def test_a_failure_is_counted_and_kept(isolated_root):
    with Recorder(isolated_root, "b1", timestamp="20260825T000002Z") as recorder:
        recorder.write(_record())
        recorder.write(_record(status=STATUS_FAILED, error_type="RemoteError",
                               error_message="500"))
        assert recorder.count == 2
        assert recorder.failures == 1
        records = read_raw(recorder.path)
    assert [r["status"] for r in records] == [STATUS_OK, STATUS_FAILED]
    assert records[1]["error_message"] == "500"


def test_a_truncated_line_is_reported_not_raised(isolated_root, tmp_path):
    path = os.path.join(isolated_root, "results", "raw", "b1-broken.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_record().as_json_line() + "\n")
        handle.write('{"campaign": "b1", "too')
    records = read_raw(path)
    assert len(records) == 2
    assert records[1]["_unreadable_line"] == 2


def test_load_records_filters_by_campaign(isolated_root):
    with Recorder(isolated_root, "b1", timestamp="20260825T000003Z") as recorder:
        recorder.write(_record())
    with Recorder(isolated_root, "b2", timestamp="20260825T000003Z") as recorder:
        recorder.write(_record(campaign="b2"))
    assert len(load_records(isolated_root, "b1")) == 1
    assert len(load_records(isolated_root)) == 2


def test_phase_timer_derives_other():
    timer = PhaseTimer()
    with timer.phase("server_exec"):
        pass
    phases = timer.finalize(total=10.0)
    assert phases["other"] == pytest.approx(10.0 - phases["server_exec"], abs=1e-6)


def test_phase_timer_records_a_phase_that_raised():
    """A partial decomposition is worth more than none when diagnosing."""
    timer = PhaseTimer()
    with pytest.raises(ValueError):
        with timer.phase("upload"):
            raise ValueError("connection reset")
    assert "upload" in timer.phases


def test_an_unknown_phase_is_refused():
    timer = PhaseTimer()
    with pytest.raises(ValueError, match="Unknown phase"):
        with timer.phase("teleport"):
            pass


def test_every_phase_name_is_declared_once():
    assert len(PHASES) == len(set(PHASES))
    assert "other" in PHASES
