"""The macOS readings, exercised on any machine by faking the platform.

The Linux half of each is as important as the Mac half: a guard that did not
guard would change what a Docker deployment measures, and nothing else in the
suite would notice.
"""

import os
import subprocess
import sys
import types

import pytest

os.environ.setdefault("API_TOKEN", "test-token")

import resources
from config import settings
from execution import dispatch as dispatch_module
from execution import runner
from registry import schema_tool
from registry.deployment import DeploymentConfig


# ----------------------------------------------------------------------
# resources.py
# ----------------------------------------------------------------------

def _fake_sysctl(monkeypatch, stdout="51539607552\n", returncode=0, explode=None):
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if explode:
            raise explode
        return types.SimpleNamespace(returncode=returncode, stdout=stdout)

    monkeypatch.setattr(resources.subprocess, "run", fake_run)
    return calls


def test_a_mac_reads_its_memory_from_sysctl(monkeypatch):
    monkeypatch.setattr(resources.sys, "platform", "darwin")
    calls = _fake_sysctl(monkeypatch)
    assert resources._darwin_ram_bytes() == 51539607552
    assert calls == [["/usr/sbin/sysctl", "-n", "hw.memsize"]]


@pytest.mark.parametrize("failure", [
    {"returncode": 1}, {"stdout": "not a number\n"}, {"explode": OSError("no sysctl")},
    {"explode": subprocess.TimeoutExpired("sysctl", 5)},
])
def test_a_failing_sysctl_is_an_unknown_not_a_crash(monkeypatch, failure):
    monkeypatch.setattr(resources.sys, "platform", "darwin")
    _fake_sysctl(monkeypatch, **failure)
    assert resources._darwin_ram_bytes() is None


def test_linux_never_asks_sysctl(monkeypatch):
    """The whole guarantee: off macOS the new branch is not even entered."""
    monkeypatch.setattr(resources.sys, "platform", "linux")
    calls = _fake_sysctl(monkeypatch)
    assert resources._darwin_ram_bytes() is None
    assert resources._platform_detected() == {}
    assert calls == []


def test_ram_falls_through_to_the_mac_reading_only_when_proc_is_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(resources, "_CGROUP_V2_MEMORY", str(tmp_path / "none"))
    monkeypatch.setattr(resources, "_CGROUP_V1_MEMORY", str(tmp_path / "none"))
    monkeypatch.setattr(resources, "_MEMINFO", str(tmp_path / "no-meminfo"))
    monkeypatch.setattr(resources.sys, "platform", "darwin")
    _fake_sysctl(monkeypatch)
    assert resources.detect_ram_bytes() == 51539607552
    # And on Linux the same absence is still "unknown", exactly as before.
    monkeypatch.setattr(resources.sys, "platform", "linux")
    assert resources.detect_ram_bytes() is None


def test_meminfo_still_wins_where_it_exists(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       1024 kB\n")
    monkeypatch.setattr(resources, "_CGROUP_V2_MEMORY", str(tmp_path / "none"))
    monkeypatch.setattr(resources, "_CGROUP_V1_MEMORY", str(tmp_path / "none"))
    monkeypatch.setattr(resources, "_MEMINFO", str(meminfo))
    monkeypatch.setattr(resources.sys, "platform", "darwin")
    calls = _fake_sysctl(monkeypatch)
    assert resources.detect_ram_bytes() == 1024 * 1024
    assert calls == []


def test_the_banner_mentions_unified_memory_only_on_a_mac(monkeypatch):
    base = dict(cpus=8.0, ram_bytes=16 * 1024 ** 3, vram_bytes=None, cpus_per_job=2,
                ram_per_job=None, vram_per_job=None, expected_clients=1)
    linux = resources.Allocation(**base, detected={"cpus": 8.0, "ram_bytes": 1})
    mac = resources.Allocation(
        **base, detected={"cpus": 8.0, "ram_bytes": 1, "unified_memory_bytes": 48 * 1024 ** 3})
    assert "unified" not in resources.banner(linux)
    assert "unified" in resources.banner(mac)


# ----------------------------------------------------------------------
# runner.py: the process-group sampler and the MPS reading
# ----------------------------------------------------------------------

@pytest.mark.parametrize("text,seconds", [
    ("0:00.05", 0.05), ("1:02.50", 62.5), ("1:02:03", 3723.0),
    ("2-01:00:00", 2 * 86400 + 3600.0), ("garbage", None),
])
def test_ps_cpu_time_is_parsed(text, seconds):
    assert runner._ps_cpu_seconds(text) == seconds


def test_the_group_is_listed_from_ps_when_there_is_no_proc(monkeypatch):
    listing = (
        "  100   100   2048  0:01.50\n"     # the runner, our group
        "  101   100  10240  1:00.00\n"     # a worker in our group
        "  200   200 999999  9:00.00\n"     # someone else's group
        "  oops\n"
    )

    def fake_run(command, **_kwargs):
        assert command[0] == "/bin/ps"
        return types.SimpleNamespace(returncode=0, stdout=listing.encode())

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    sampler = runner._TreeSampler()
    found = sampler._group_from_ps(100)
    assert found == {100, 101}
    assert sampler._rss(found) == (2048 + 10240) * 1024
    ticks = sampler._cpu_ticks(found)
    assert ticks == int((1.5 + 60.0) * sampler._ticks_per_second)


def test_a_failing_ps_is_an_empty_group(monkeypatch):
    def boom(*_a, **_k):
        raise OSError("no ps")

    monkeypatch.setattr(runner.subprocess, "run", boom)
    assert runner._TreeSampler()._group_from_ps(1) == set()


def test_linux_sampling_still_reads_proc(monkeypatch):
    """No snapshot unless macOS asked for one: the /proc readers keep their path."""
    sampler = runner._TreeSampler()
    assert sampler._ps_snapshot is None
    if sys.platform.startswith("linux"):
        assert os.getpid() in sampler._group() or os.getpgrp() != os.getpid()
        assert sampler._ps_snapshot is None


class _FakeTorch:
    def __init__(self, mps_available=True, cuda=False):
        self.cuda = types.SimpleNamespace(is_available=lambda: cuda)
        self.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps_available))
        self.mps = types.SimpleNamespace(
            driver_allocated_memory=lambda: 3_800_000_000,
            current_allocated_memory=lambda: 150_000_000,
            recommended_max_memory=lambda: 40_200_896_512,
        )


def test_mps_memory_is_read_on_a_mac_without_cuda(monkeypatch):
    monkeypatch.setattr(runner.sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    assert runner._peak_vram_bytes() == 3_800_000_000
    assert runner._mps_recommended_max_bytes() == 40_200_896_512


def test_mps_memory_is_never_read_off_a_mac(monkeypatch):
    """A CPU-only Linux deployment reads None, as it always did."""
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    assert runner._peak_vram_bytes() is None
    assert runner._mps_recommended_max_bytes() is None


def test_cuda_wins_over_mps(monkeypatch):
    fake = _FakeTorch(cuda=True)
    fake.cuda.max_memory_reserved = lambda: 7
    fake.cuda.max_memory_allocated = lambda: 5
    monkeypatch.setattr(runner.sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert runner._peak_vram_bytes() == 7


def test_no_mps_backend_is_none(monkeypatch):
    monkeypatch.setattr(runner.sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch(mps_available=False))
    assert runner._peak_vram_bytes() is None
    assert runner._mps_recommended_max_bytes() is None


# ----------------------------------------------------------------------
# registry/schema_tool.py: the server's DEVICE reaches a tool that names none
# ----------------------------------------------------------------------

def _schema_tool_with_device(tmp_path):
    schema = {
        "name": "Dev",
        "arguments": {
            "scans": {"type": "path"},
            "device": {"type": "str", "required": False, "default": "cuda", "choices": ["cuda", "mps", "cpu"]},
        },
        "returns": "path",
    }
    return schema_tool.SchemaTool(str(tmp_path), schema, DeploymentConfig({}).for_tool("X"))


def _capture_dispatch(monkeypatch):
    seen = []
    monkeypatch.setattr(dispatch_module, "dispatch", lambda tool, args: seen.append(args) or "ok")
    return seen


def test_a_mac_gives_a_tool_naming_no_device_the_servers_device(monkeypatch, tmp_path):
    seen = _capture_dispatch(monkeypatch)
    monkeypatch.setattr(schema_tool.sys, "platform", "darwin")
    monkeypatch.setattr(settings, "DEVICE", "mps")
    _schema_tool_with_device(tmp_path).invoke({"scans": "/x"})
    assert seen[0]["device"] == "mps"


def test_a_device_the_caller_named_is_kept_on_a_mac(monkeypatch, tmp_path):
    seen = _capture_dispatch(monkeypatch)
    monkeypatch.setattr(schema_tool.sys, "platform", "darwin")
    monkeypatch.setattr(settings, "DEVICE", "mps")
    _schema_tool_with_device(tmp_path).invoke({"scans": "/x", "device": "cpu"})
    assert seen[0]["device"] == "cpu"


def test_linux_keeps_the_tools_declared_default_device(monkeypatch, tmp_path):
    seen = _capture_dispatch(monkeypatch)
    monkeypatch.setattr(schema_tool.sys, "platform", "linux")
    monkeypatch.setattr(settings, "DEVICE", "mps")
    _schema_tool_with_device(tmp_path).invoke({"scans": "/x"})
    assert seen[0]["device"] == "cuda", "off a Mac the injection must not happen"
