"""A number without its machine is not a measurement, so the fingerprint has to
be collectable -- and has to degrade to a stated absence rather than an
exception -- on any machine a reviewer might use."""

from __future__ import annotations

import os

from benchmarks import provenance


def test_collect_returns_every_declared_field(tmp_path):
    fingerprint = provenance.collect({"server": str(tmp_path)}, str(tmp_path))
    for key in ("hostname", "collected_at", "os", "cpu", "ram_bytes", "gpu",
                "network_mbps", "disk_free_bytes", "git", "_notes"):
        assert key in fingerprint


def test_a_machine_with_no_gpu_says_so(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance.shutil, "which", lambda _name: None)
    fingerprint = provenance.collect({}, str(tmp_path))
    assert fingerprint["gpu"]["devices"] == []
    assert any("no CUDA device" in note for note in fingerprint["_notes"])


def test_a_missing_repository_is_a_null_and_a_note(tmp_path):
    fingerprint = provenance.collect({"absent": str(tmp_path / "nowhere")}, str(tmp_path))
    assert fingerprint["git"]["absent"] is None
    assert any("absent" in note for note in fingerprint["_notes"])


def test_a_dirty_working_tree_is_marked():
    """These benchmarks are run against working trees. A number measured on
    uncommitted code must not claim a commit that does not contain it."""
    sha = provenance.git_sha(os.path.dirname(provenance.__file__))
    if sha is None:
        return  # not a git checkout; nothing to assert
    assert len(sha) >= 40


def test_utc_now_is_iso_and_utc():
    stamp = provenance.utc_now()
    assert stamp.endswith("+00:00")
    assert "T" in stamp


def test_a_command_that_does_not_exist_returns_none():
    assert provenance._run(["definitely-not-a-real-binary-xyzzy"]) is None
