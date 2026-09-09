"""59 GB of headroom, campaigns that write CBCT volumes, and a server whose own
job directories are on the same filesystem. The guard is not a nicety."""

from __future__ import annotations

import os

import pytest

from benchmarks import guards

_GB = 1024 ** 3


def _plan(*items) -> list:
    return [{"output_mb": mb, "runs": runs} for mb, runs in items]


def test_projection_multiplies_by_repetitions():
    assert guards.project_output_bytes(_plan((100, 6))) == 600 * 1024 * 1024


def test_an_item_with_no_declared_output_contributes_nothing():
    assert guards.project_output_bytes([{"runs": 4}]) == 0


def test_a_plan_that_fits_is_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        guards.shutil, "disk_usage",
        lambda _path: type("Usage", (), {"free": 50 * _GB})(),
    )
    report = guards.check_disk(str(tmp_path), _plan((1000, 4)), min_free_gb=10, margin_gb=8)
    assert report.ok
    guards.enforce(report)


def test_a_plan_that_would_not_leave_the_margin_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        guards.shutil, "disk_usage",
        lambda _path: type("Usage", (), {"free": 20 * _GB})(),
    )
    report = guards.check_disk(str(tmp_path), _plan((5000, 4)), min_free_gb=10, margin_gb=8)
    assert not report.ok
    with pytest.raises(guards.DiskSpaceError) as raised:
        guards.enforce(report)
    # All three numbers have to be in the message: the operator's next move
    # differs depending on which one is binding.
    assert "free" in str(raised.value) and "projected" in str(raised.value)


def test_a_disk_already_below_the_floor_is_refused_whatever_the_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(
        guards.shutil, "disk_usage",
        lambda _path: type("Usage", (), {"free": 3 * _GB})(),
    )
    report = guards.check_disk(str(tmp_path), [], min_free_gb=10, margin_gb=0)
    assert not report.ok


def test_the_scratch_directory_is_emptied(tmp_path):
    scratch = tmp_path / "scratch"
    (scratch / "job_a").mkdir(parents=True)
    (scratch / "job_a" / "big.nii.gz").write_bytes(b"x" * 100)
    (scratch / "loose.zip").write_bytes(b"x")
    assert guards.clear_scratch(str(scratch)) == 2
    assert os.listdir(scratch) == []


def test_clearing_a_directory_that_does_not_exist_is_not_an_error(tmp_path):
    assert guards.clear_scratch(str(tmp_path / "nowhere")) == 0
