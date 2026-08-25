"""B5's claim is parity. This is the machinery that would have to be wrong for a
difference to be missed, so it is tested on differences that are easy to miss."""

from __future__ import annotations

import json
import os

from benchmarks import artifacts


def _write(root, name, content):
    path = os.path.join(root, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(path, mode) as handle:
        handle.write(content)
    return path


def test_snapshot_keys_are_relative(tmp_path):
    root = tmp_path / "output"
    _write(str(root), "seg/mandible.nii.gz", b"abc")
    found = artifacts.snapshot(str(root))
    assert list(found) == [os.path.join("seg", "mandible.nii.gz")]
    assert found[os.path.join("seg", "mandible.nii.gz")]["size"] == 3


def test_two_identical_trees_compare_equal(tmp_path):
    left, right = tmp_path / "l", tmp_path / "r"
    for root in (left, right):
        _write(str(root), "a.txt", "same")
        _write(str(root), "sub/b.bin", b"\x00\x01")
    report = artifacts.compare(artifacts.snapshot(str(left)), artifacts.snapshot(str(right)))
    assert report.ok
    assert len(report.identical) == 2


def test_absolute_paths_are_never_compared(tmp_path):
    """The two sides live in different directories by construction -- a job
    directory in a container and a download folder. Comparing absolute paths
    would report every file as differing."""
    left, right = tmp_path / "jobs" / "abc" / "output", tmp_path / "downloads" / "unpacked"
    _write(str(left), "report.txt", "identical")
    _write(str(right), "report.txt", "identical")
    report = artifacts.compare(artifacts.snapshot(str(left)), artifacts.snapshot(str(right)))
    assert report.ok


def test_a_file_only_one_side_produced_is_named(tmp_path):
    left, right = tmp_path / "l", tmp_path / "r"
    _write(str(left), "extra.txt", "x")
    _write(str(left), "shared.txt", "x")
    _write(str(right), "shared.txt", "x")
    report = artifacts.compare(artifacts.snapshot(str(left)), artifacts.snapshot(str(right)))
    assert report.only_left == ["extra.txt"]
    assert report.only_right == []
    assert not report.ok


def test_byte_difference_finds_the_first_differing_offset(tmp_path):
    left = _write(str(tmp_path), "l.bin", b"aaaaXaaaa")
    right = _write(str(tmp_path), "r.bin", b"aaaaYaaaa")
    detail = artifacts.byte_difference(left, right)
    assert detail["first_differing_offset"] == 4
    assert detail["differing_bytes"] == 1
    assert detail["same_size"] is True


def test_byte_difference_reports_a_size_mismatch(tmp_path):
    left = _write(str(tmp_path), "l.bin", b"abc")
    right = _write(str(tmp_path), "r.bin", b"abcd")
    detail = artifacts.byte_difference(left, right)
    assert detail["same_size"] is False
    assert detail["first_differing_offset"] == 3


def test_json_difference_names_the_key_not_the_byte(tmp_path):
    """A landmark file whose only difference is a timestamp is a different
    finding from one whose coordinates moved."""
    left = _write(str(tmp_path), "l.json",
                  json.dumps({"generated": "A", "points": [{"x": 1.0}]}))
    right = _write(str(tmp_path), "r.json",
                   json.dumps({"generated": "B", "points": [{"x": 1.0}]}))
    detail = artifacts.json_difference(left, right)
    assert detail["readable"]
    assert detail["differing_keys"] == ["generated"]


def test_json_difference_reaches_into_lists(tmp_path):
    left = _write(str(tmp_path), "l.json", json.dumps({"points": [{"x": 1.0}, {"x": 2.0}]}))
    right = _write(str(tmp_path), "r.json", json.dumps({"points": [{"x": 1.0}, {"x": 2.5}]}))
    assert artifacts.json_difference(left, right)["differing_keys"] == ["points[1].x"]


def test_text_difference_reports_the_changed_lines(tmp_path):
    left = _write(str(tmp_path), "l.txt", "alpha\nbeta\ngamma\n")
    right = _write(str(tmp_path), "r.txt", "alpha\nBETA\ngamma\n")
    detail = artifacts.text_difference(left, right)
    assert detail["changed_lines"] == [{"line": 2, "left": "beta", "right": "BETA"}]


def test_numeric_distance_says_why_it_is_unavailable(tmp_path):
    detail = artifacts.numeric_distance("a.nii.gz", "b.nii.gz", None)
    assert detail["available"] is False
    assert "imaging_interpreter" in detail["reason"]

    detail = artifacts.numeric_distance("a.nii.gz", "b.nii.gz", str(tmp_path / "no-python"))
    assert detail["available"] is False
    assert "does not exist" in detail["reason"]


def test_describe_difference_picks_the_right_readers(tmp_path):
    left, right = tmp_path / "l", tmp_path / "r"
    _write(str(left), "a.json", json.dumps({"k": 1}))
    _write(str(right), "a.json", json.dumps({"k": 2}))
    detail = artifacts.describe_difference("a.json", str(left), str(right))
    assert "bytes" in detail and "json" in detail
    assert "numeric" not in detail


def test_describe_difference_asks_for_a_numeric_distance_on_an_image(tmp_path):
    left, right = tmp_path / "l", tmp_path / "r"
    _write(str(left), "seg.nii.gz", b"\x1f\x8b")
    _write(str(right), "seg.nii.gz", b"\x1f\x8c")
    detail = artifacts.describe_difference("seg.nii.gz", str(left), str(right))
    assert "numeric" in detail
    assert detail["numeric"]["available"] is False  # no interpreter given
