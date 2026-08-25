"""The tests that need something running. Every one of them skips with a reason.

A reviewer with no server, no Docker and no GPU should see these as skips
naming what is absent -- never as failures, and never as silent passes.
"""

from __future__ import annotations

import os
import uuid

import pytest

from benchmarks import gpu
from benchmarks.execution import local as local_path


@pytest.mark.needs_server
def test_the_server_answers_health(live_server):
    assert live_server.health()["status"] == "ok"


@pytest.mark.needs_server
def test_a_chunked_upload_round_trips(live_server, tmp_path):
    """POST /uploads plus parallel PUTs plus DELETE, with no tool involved.

    This is the transfer half of B2 exercised on its own, so a failure in the
    upload protocol is found in seconds rather than in the middle of a 200 MB
    campaign.
    """
    payload = tmp_path / "payload.bin"
    payload.write_bytes(os.urandom(20 * 1024 * 1024))
    session = live_server.open_upload(str(payload))
    assert session["part_count"] >= 2, "20 MB must span several 8 MB parts"
    try:
        sent = live_server.upload_parts(session)
        assert sent > 0
        status = live_server.session.get(
            f"{live_server.base_url}/uploads/{session['upload_id']}",
            headers=live_server.headers, timeout=15,
        ).json()
        assert status["missing_parts"] == []
    finally:
        live_server.discard_upload(session["upload_id"])


@pytest.mark.needs_server
def test_an_unauthenticated_call_is_refused(live_server):
    response = live_server.session.post(f"{live_server.base_url}/run/Test_Tool", timeout=15)
    assert response.status_code in (401, 403)


@pytest.mark.needs_docker
def test_the_local_path_can_reach_a_tool_interpreter(live_container, shipped_config):
    """The exec line dispatch.py uses, run by hand, for one configured tool."""
    tool = next(
        (spec for spec in shipped_config.tools.values() if spec.supports_local), None
    )
    if tool is None:
        pytest.skip("no tool in config.yaml declares a local path")
    interpreter = live_container.interpreter(tool)
    completed = live_container._exec([interpreter, "-c", "import sys; print(sys.prefix)"],
                                     timeout=120)
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")[:500]
    assert tool.local.folder.split("/")[-1] in completed.stdout.decode()


@pytest.mark.needs_docker
def test_a_job_directory_can_be_created_and_removed(live_container, shipped_config):
    tool = next(
        (spec for spec in shipped_config.tools.values() if spec.supports_local), None
    )
    if tool is None:
        pytest.skip("no tool in config.yaml declares a local path")
    job = live_container.prepare_job(tool, {})
    try:
        live_container.write_job_file(job, tool, {"output_dir": job.output_dir})
        raw = live_container._read_remote_file(os.path.join(job.job_dir, local_path.JOB_FILE))
        assert raw and b'"tool"' in raw
    finally:
        live_container.remove_job(job.job_dir)


@pytest.mark.needs_gpu
def test_vram_can_be_sampled():
    if not gpu.available():
        pytest.skip("nvidia-smi is not on PATH")
    reading = gpu.sample()
    if reading is None:
        pytest.skip("nvidia-smi is present but returned nothing usable")
    assert all(isinstance(value, int) for value in reading.values())


@pytest.mark.needs_gpu
def test_the_sampler_reports_how_many_samples_a_peak_rests_on():
    if not gpu.available():
        pytest.skip("nvidia-smi is not on PATH")
    sampler = gpu.VramSampler(interval_seconds=0.1).start()
    if sampler.unavailable_reason:
        pytest.skip(sampler.unavailable_reason)
    import time

    time.sleep(0.5)
    report = sampler.stop()
    assert report["samples"] >= 1
    assert report["peak_mib"]


def test_the_sampler_degrades_cleanly_with_no_card(monkeypatch):
    """Runs everywhere, including on a machine with no GPU: the point is that
    the absence is reported rather than raised."""
    monkeypatch.setattr(gpu.shutil, "which", lambda _name: None)
    report = gpu.VramSampler(interval_seconds=0.1).start().stop()
    assert report["peak_mib"] == {}
    assert report["unavailable_reason"] == "nvidia-smi is not on PATH"
