"""Summaries are derived, so they can be tested on synthetic records without any
of the machinery that produces real ones."""

from __future__ import annotations

import csv
import os

from benchmarks import summarize
from benchmarks.recording import STATUS_FAILED, STATUS_OK, Recorder, RunRecord


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
        "provenance": {"hostname": "bench-1", "cpu": {"model": "Xeon"}, "gpu": {"devices": []}},
    }
    fields.update(overrides)
    return RunRecord(**fields)


def test_the_warmup_is_excluded_from_the_statistics_and_counted():
    records = [
        _record(repetition=1, total_seconds=9.0, warmup=True).__dict__,
        _record(repetition=2, total_seconds=1.0).__dict__,
        _record(repetition=3, total_seconds=2.0).__dict__,
        _record(repetition=4, total_seconds=3.0).__dict__,
    ]
    row = summarize.table_b1(records)[0]
    assert row["warmup_discarded"] == 1
    assert row["n"] == 3
    assert row["median_s"] == 2.0
    # The 9 s warm-up must not appear in the range.
    assert row["max_s"] == 3.0


def test_failures_are_counted_in_the_row_they_belong_to():
    records = [
        _record(repetition=1, total_seconds=1.0).__dict__,
        _record(repetition=2, status=STATUS_FAILED, error_type="RemoteError",
                error_message="500").__dict__,
    ]
    row = summarize.table_b1(records)[0]
    assert row["failed"] == 1
    assert row["n"] == 1
    assert "500" in row["first_error"]


def test_a_row_with_no_successful_run_is_still_a_row():
    """A tool that fails everywhere must not vanish from the table."""
    records = [_record(status=STATUS_FAILED, error_type="RemoteError").__dict__]
    rows = summarize.table_b1(records)
    assert len(rows) == 1
    assert rows[0]["n"] == 0
    assert rows[0]["median_s"] is None


def test_b2_reports_one_row_per_parallelism_setting():
    records = []
    for parallelism in (4, 1):
        for repetition in (1, 2):
            records.append(
                _record(
                    campaign="b2",
                    repetition=repetition,
                    total_seconds=float(parallelism),
                    phases={"upload": 1.0, "server_exec": 2.0, "other": 0.5},
                    extra={"parallelism": parallelism, "payload_label": "95MB_CBCT",
                           "bytes_uploaded": 95_000_000, "bytes_downloaded": 1_000},
                ).__dict__
            )
    rows = summarize.table_b2(records)
    assert len(rows) == 2
    assert {row["parallelism"] for row in rows} == {1, 4}
    assert rows[0]["upload_s"] == 1.0
    assert rows[0]["uploaded_mb"] == 95.0


def test_b3_keeps_startup_measurements_apart_from_chain_runs():
    records = [
        _record(campaign="b3", extra={"measurement": "chain", "mode": "Registration"}).__dict__,
        _record(
            campaign="b3", tool="ALI_CBCT",
            extra={"measurement": "startup", "interpreter_start_seconds": 0.05,
                   "import_stack_seconds": 4.2, "package": "sadt_ali_cbct"},
        ).__dict__,
    ]
    rows = summarize.table_b3(records)
    kinds = {row["measurement"] for row in rows}
    assert kinds == {"chain", "startup"}
    startup = next(row for row in rows if row["measurement"] == "startup")
    assert startup["import_stack_s"] == 4.2


def test_b4_reports_p50_p95_and_the_sample_count_behind_the_vram_peak():
    records = []
    for index, seconds in enumerate([1.0, 2.0, 3.0, 10.0], start=1):
        records.append(
            _record(
                campaign="b4", repetition=index, total_seconds=seconds,
                extra={
                    "concurrency": 4,
                    "campaign_window_seconds": 12.0,
                    "throughput_jobs_per_minute": 20.0,
                    "vram": {"peak_mib": {0: 8000}, "baseline_mib": {0: 1000},
                             "samples": 24, "unavailable_reason": None},
                },
            ).__dict__
        )
    row = summarize.table_b4(records)[0]
    assert row["p50_s"] == 2.5
    assert row["p95_s"] == 10.0
    assert row["peak_vram_mib"] == 8000
    assert row["vram_samples"] == 24


def test_b5_reports_the_differing_files_by_name():
    records = [
        _record(
            campaign="b5", tool="AMASSS", path="local+loopback",
            extra={
                "side": "comparison", "pair": 1, "parity_ok": False,
                "parity": {
                    "identical": ["a.nii.gz"], "differing": ["report.txt"],
                    "only_left": [], "only_right": [], "ok": False,
                    "identical_count": 1, "differing_count": 1,
                    "details": {
                        "report.txt": {
                            "bytes": {"left_size": 10, "right_size": 10,
                                      "first_differing_offset": 3,
                                      "differing_bytes": 2, "differing_fraction": 0.2},
                            "text": {"readable": True, "changed_lines": [
                                {"line": 1, "left": "at 10:00", "right": "at 10:05"}]},
                        }
                    },
                },
            },
        ).__dict__
    ]
    row = summarize.table_b5(records)[0]
    assert row["differing"] == 1
    assert "report.txt" in row["differing_files"]
    prose = summarize.parity_details(records)
    assert "report.txt" in prose
    assert "at 10:05" in prose


def test_a_summary_regenerates_from_the_raw_files(isolated_root):
    with Recorder(isolated_root, "b1", timestamp="20260825T010000Z") as recorder:
        recorder.write(_record(repetition=1, warmup=True))
        recorder.write(_record(repetition=2, total_seconds=1.5))
    outcome = summarize.summarize("b1", isolated_root, stamp="20260825T010001Z")
    assert outcome["records"] == 2
    assert os.path.isfile(outcome["markdown"])
    with open(outcome["csv"], encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["tool"] == "Widget"

    body = open(outcome["markdown"], encoding="utf-8").read()
    assert "failed runs" in body
    assert "bench-1" in body, "the summary must name the machine that measured"


def test_two_machines_in_one_campaign_are_reported_as_such():
    def _on(hostname, model):
        return _record(
            provenance={"hostname": hostname, "cpu": {"model": model}, "gpu": {"devices": []}}
        ).__dict__

    records = [_on("a", "X"), _on("b", "Y")]
    assert "several machines" in summarize._hardware_line(records)


def test_p95_is_an_observation_not_an_interpolation():
    """With 8 points, an interpolated p95 is a number no job actually took."""
    stats = summarize._numbers([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert stats["p95"] in (7.0, 8.0)


def test_a_summary_can_be_restricted_to_one_raw_file(isolated_root):
    """A campaign run in several sittings summarises as one; an exploratory run
    must be excludable from a published one."""
    with Recorder(isolated_root, "b1", timestamp="20260825T020000Z") as recorder:
        recorder.write(_record(tool="Widget", total_seconds=1.0))
        first = os.path.basename(recorder.path)
    with Recorder(isolated_root, "b1", timestamp="20260825T020001Z") as recorder:
        recorder.write(_record(tool="Gadget", total_seconds=2.0))

    both = summarize.summarize("b1", isolated_root, stamp="20260825T020002Z")
    assert both["records"] == 2

    one = summarize.summarize("b1", isolated_root, stamp="20260825T020003Z", only=[first])
    assert one["records"] == 1
    assert [row["tool"] for row in one["rows"]] == ["Widget"]
