"""Sizing the server to the machine it was started on.

Every test here drives detection through the module's file-path and
subprocess-lookup constants rather than through the real /sys and the real
card, because the whole point of the cascade is what it does on machines this
suite will never run on: a container with a quota, a cgroup v1 kernel, a host
with no GPU at all.
"""

import os
import subprocess
import types

import pytest

os.environ.setdefault("API_TOKEN", "test-token")

import resources


# ----------------------------------------------------------------------
# One setting, three spellings
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "written, whole, expected",
    [
        ("75%", 100.0, 75.0),
        ("50%", 8.0, 4.0),
        ("8", None, 8.0),
        ("16GB", None, 16 * 1000 ** 3),
        ("16GiB", None, 16 * 1024 ** 3),
        ("512MB", None, 512 * 1000 ** 2),
        ("  4  ", None, 4.0),
    ],
)
def test_an_amount_can_be_written_three_ways(written, whole, expected):
    assert resources.parse_amount(written, whole) == pytest.approx(expected)


def test_a_percentage_without_a_whole_is_undecidable():
    """Which is how a VRAM percentage behaves on a machine with no card."""
    assert resources.parse_amount("90%", None) is None


@pytest.mark.parametrize("written", ["", None, "   ", "lots", "%", "GB", "8 cores"])
def test_an_unusable_amount_is_ignored_rather_than_fatal(written):
    """A typo in a resource knob must not stop a server that can size itself."""
    assert resources.parse_amount(written, 100.0) is None


# ----------------------------------------------------------------------
# CPU detection: the cgroup before the kernel
# ----------------------------------------------------------------------

def _files(monkeypatch, mapping):
    """Answer resources._first_line from a dict, and nothing else."""
    monkeypatch.setattr(resources, "_first_line", lambda path: mapping.get(path))


def test_a_cgroup_v2_quota_wins_over_the_host(monkeypatch):
    """The container was limited to 8; the host has 56. The limit is the answer."""
    _files(monkeypatch, {resources._CGROUP_V2_CPU: "800000 100000"})
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(56)))
    assert resources.detect_cpus() == pytest.approx(8.0)


def test_a_fractional_quota_is_kept_fractional(monkeypatch):
    _files(monkeypatch, {resources._CGROUP_V2_CPU: "150000 100000"})
    assert resources.detect_cpus() == pytest.approx(1.5)


def test_an_unlimited_v2_cgroup_falls_through_to_the_affinity_mask(monkeypatch):
    """`max 100000` is what this machine's container reports: no limit set."""
    _files(monkeypatch, {resources._CGROUP_V2_CPU: "max 100000"})
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(12)))
    assert resources.detect_cpus() == pytest.approx(12.0)


def test_cgroup_v1_is_read_when_v2_is_absent(monkeypatch):
    _files(monkeypatch, {
        resources._CGROUP_V1_CPU_QUOTA: "400000",
        resources._CGROUP_V1_CPU_PERIOD: "100000",
    })
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(56)))
    assert resources.detect_cpus() == pytest.approx(4.0)


def test_a_v1_quota_of_minus_one_means_no_limit(monkeypatch):
    _files(monkeypatch, {
        resources._CGROUP_V1_CPU_QUOTA: "-1",
        resources._CGROUP_V1_CPU_PERIOD: "100000",
    })
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(6)))
    assert resources.detect_cpus() == pytest.approx(6.0)


def test_the_affinity_mask_is_preferred_to_cpu_count(monkeypatch):
    """A cpuset restricts which cores are usable without writing any quota, and
    the mask is the only one of the three that reflects it."""
    _files(monkeypatch, {})
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2})
    monkeypatch.setattr(os, "cpu_count", lambda: 56)
    assert resources.detect_cpus() == pytest.approx(3.0)


def test_detection_never_answers_zero(monkeypatch):
    _files(monkeypatch, {})
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert resources.detect_cpus() >= 1.0


# ----------------------------------------------------------------------
# RAM detection
# ----------------------------------------------------------------------

def test_a_cgroup_v2_memory_limit_wins(monkeypatch):
    _files(monkeypatch, {resources._CGROUP_V2_MEMORY: str(64 * 1024 ** 3)})
    assert resources.detect_ram_bytes() == 64 * 1024 ** 3


def test_an_unlimited_v2_memory_cgroup_falls_through(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       131072000 kB\nMemFree: 1 kB\n")
    _files(monkeypatch, {resources._CGROUP_V2_MEMORY: "max"})
    monkeypatch.setattr(resources, "_MEMINFO", str(meminfo))
    assert resources.detect_ram_bytes() == 131072000 * 1024


def test_a_v1_sentinel_is_not_mistaken_for_a_limit(monkeypatch, tmp_path):
    """cgroup v1 spells 'no limit' as a huge number, not as a word."""
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       1024 kB\n")
    _files(monkeypatch, {resources._CGROUP_V1_MEMORY: str(1 << 63)})
    monkeypatch.setattr(resources, "_MEMINFO", str(meminfo))
    assert resources.detect_ram_bytes() == 1024 * 1024


# ----------------------------------------------------------------------
# VRAM detection
# ----------------------------------------------------------------------

def _nvidia_smi(monkeypatch, stdout, returncode=0, present=True):
    monkeypatch.setattr(
        resources.shutil, "which", lambda _name: "/usr/bin/nvidia-smi" if present else None
    )
    monkeypatch.setattr(
        resources.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=returncode, stdout=stdout),
    )


def test_no_card_is_an_answer_not_a_failure(monkeypatch):
    _nvidia_smi(monkeypatch, "", present=False)
    assert resources.detect_vram_bytes() == (None, None)


def test_vram_is_read_in_bytes(monkeypatch):
    _nvidia_smi(monkeypatch, "49140, 45426\n")
    total, free = resources.detect_vram_bytes()
    assert total == 49140 * 1024 ** 2
    assert free == 45426 * 1024 ** 2


def test_the_least_free_card_is_the_one_that_counts(monkeypatch):
    """A job has to fit on ONE card; two half-full cards have room for neither."""
    _nvidia_smi(monkeypatch, "49140, 45426\n49140, 2048\n")
    _total, free = resources.detect_vram_bytes()
    assert free == 2048 * 1024 ** 2


def test_a_failing_nvidia_smi_is_survivable(monkeypatch):
    _nvidia_smi(monkeypatch, "", returncode=9)
    assert resources.detect_vram_bytes() == (None, None)


def test_a_timeout_is_survivable(monkeypatch):
    monkeypatch.setattr(resources.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    def _explode(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 5.0)

    monkeypatch.setattr(resources.subprocess, "run", _explode)
    assert resources.detect_vram_bytes() == (None, None)


# ----------------------------------------------------------------------
# Resolving a budget
# ----------------------------------------------------------------------

def _machine(monkeypatch, cpus=56.0, ram=125 * 1024 ** 3, vram_free=45 * 1024 ** 3):
    monkeypatch.setattr(resources, "detect_cpus", lambda: cpus)
    monkeypatch.setattr(resources, "detect_ram_bytes", lambda: ram)
    monkeypatch.setattr(
        resources, "detect_vram_bytes", lambda: (48 * 1024 ** 3, vram_free)
    )


def _config(**overrides):
    base = {
        "SADT_CPU_BUDGET": "", "SADT_RAM_BUDGET": "", "SADT_VRAM_BUDGET": "",
        "SADT_CPU_PER_JOB": "", "SADT_RAM_PER_JOB": "", "SADT_VRAM_PER_JOB": "",
        "SADT_RESOURCE_SHARE": 0.75, "SADT_EXPECTED_CLIENTS": 1,
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


def test_nothing_configured_takes_a_share_of_the_machine(monkeypatch):
    _machine(monkeypatch)
    allocation = resources.resolve(_config())
    assert allocation.cpus == pytest.approx(42.0)
    assert allocation.sources["cpus"] == resources.SOURCE_SHARE


def test_one_client_gets_the_whole_budget_per_job(monkeypatch):
    """A single workstation wants latency: dividing the machine by one is right."""
    _machine(monkeypatch)
    allocation = resources.resolve(_config())
    assert allocation.cpus_per_job == 42
    assert allocation.max_parallel_jobs == 1


def test_ten_clients_split_the_budget(monkeypatch):
    _machine(monkeypatch)
    allocation = resources.resolve(_config(SADT_EXPECTED_CLIENTS=10))
    assert allocation.cpus_per_job == 4
    assert allocation.max_parallel_jobs == 10


def test_a_configured_budget_beats_detection(monkeypatch):
    _machine(monkeypatch)
    allocation = resources.resolve(_config(SADT_CPU_BUDGET="16"))
    assert allocation.cpus == pytest.approx(16.0)
    assert allocation.sources["cpus"] == resources.SOURCE_CONFIGURED


def test_a_configured_percentage_is_of_what_was_detected(monkeypatch):
    _machine(monkeypatch, cpus=56.0)
    allocation = resources.resolve(_config(SADT_CPU_BUDGET="50%"))
    assert allocation.cpus == pytest.approx(28.0)


def test_a_per_job_cap_beats_the_client_split(monkeypatch):
    _machine(monkeypatch)
    allocation = resources.resolve(
        _config(SADT_EXPECTED_CLIENTS=10, SADT_CPU_PER_JOB="2")
    )
    assert allocation.cpus_per_job == 2
    assert allocation.sources["cpus_per_job"] == resources.SOURCE_CONFIGURED


def test_a_job_always_gets_at_least_one_whole_core(monkeypatch):
    """100 clients on 4 cores would otherwise mean a fraction of a thread, which
    means nothing to OMP_NUM_THREADS and would never run."""
    _machine(monkeypatch, cpus=4.0)
    allocation = resources.resolve(_config(SADT_EXPECTED_CLIENTS=100))
    assert allocation.cpus_per_job == 1


def test_the_split_stops_at_what_the_machine_can_run_in_parallel(monkeypatch):
    """A 42-core budget cannot usefully be cut a hundred ways: past that the
    divisor stops describing contention and starts inventing it."""
    _machine(monkeypatch)
    hundred = resources.resolve(_config(SADT_EXPECTED_CLIENTS=100))
    forty_two = resources.resolve(_config(SADT_EXPECTED_CLIENTS=42))
    assert hundred.vram_per_job == forty_two.vram_per_job
    assert hundred.ram_per_job == forty_two.ram_per_job


def test_a_share_can_still_be_smaller_than_a_real_tool_needs(monkeypatch):
    """Which is the whole reason a cap is a fairness target and not a refusal.

    Split 42 ways this card gives 0.8 GiB, and AMASSS measures 2.19. The
    machine genuinely cannot run 42 of those at once, so the honest answer is
    that admission lets a job over its cap run ALONE -- never that it declines
    to run it. Pinned here so the contract is not quietly broken later by
    turning the cap into a limit.
    """
    _machine(monkeypatch)
    allocation = resources.resolve(_config(SADT_EXPECTED_CLIENTS=42))
    assert allocation.vram_per_job < 2.19 * 1024 ** 3


def test_the_vram_budget_is_taken_from_what_is_free(monkeypatch):
    """~3 GiB of this machine's card is held by something that is not the
    server, and a budget from `total` would admit a job the card cannot take."""
    _machine(monkeypatch, vram_free=45 * 1024 ** 3)
    allocation = resources.resolve(_config(SADT_VRAM_BUDGET="100%"))
    assert allocation.vram_bytes == pytest.approx(45 * 1024 ** 3, rel=1e-6)


def test_a_machine_with_no_card_simply_has_no_vram_dimension(monkeypatch):
    monkeypatch.setattr(resources, "detect_cpus", lambda: 8.0)
    monkeypatch.setattr(resources, "detect_ram_bytes", lambda: 16 * 1024 ** 3)
    monkeypatch.setattr(resources, "detect_vram_bytes", lambda: (None, None))
    allocation = resources.resolve(_config())
    assert allocation.vram_bytes is None
    assert allocation.vram_per_job is None
    assert allocation.sources["vram_bytes"] == resources.SOURCE_UNAVAILABLE
    assert allocation.cpus == pytest.approx(6.0)


def test_a_share_is_clamped_to_something_sane(monkeypatch):
    _machine(monkeypatch, cpus=8.0)
    assert resources.resolve(_config(SADT_RESOURCE_SHARE=5.0)).cpus == pytest.approx(8.0)
    assert resources.resolve(_config(SADT_RESOURCE_SHARE=-1)).cpus >= 1.0


def test_fewer_than_one_client_is_still_one(monkeypatch):
    _machine(monkeypatch)
    assert resources.resolve(_config(SADT_EXPECTED_CLIENTS=0)).expected_clients == 1


# ----------------------------------------------------------------------
# The banner
# ----------------------------------------------------------------------

def test_the_banner_names_what_was_found_and_what_was_decided(monkeypatch):
    _machine(monkeypatch)
    text = resources.banner(resources.resolve(_config(SADT_EXPECTED_CLIENTS=10)))
    assert "detected" in text
    assert "cpus=56.0" in text  # what the machine has
    assert "cpus=42.0" in text  # what the server took
    assert "10 client(s)" in text


def test_the_banner_says_which_numbers_came_from_a_setting(monkeypatch):
    """The only symptom of a budget that did not take effect is a slow server."""
    _machine(monkeypatch)
    text = resources.banner(resources.resolve(_config(SADT_CPU_BUDGET="16")))
    assert resources.SOURCE_CONFIGURED in text


def test_the_banner_says_what_decides_admission(monkeypatch):
    """The first thing to look at when a server feels more serial than it
    should -- and there is only one answer now, no counter above the budget."""
    _machine(monkeypatch)
    said = resources.banner(resources.resolve(_config()))
    assert "measured to need" in said
    assert "runs alone" in said, "the rule that governs an empty cost table"


def test_the_banner_survives_a_machine_with_no_card(monkeypatch):
    monkeypatch.setattr(resources, "detect_cpus", lambda: 4.0)
    monkeypatch.setattr(resources, "detect_ram_bytes", lambda: None)
    monkeypatch.setattr(resources, "detect_vram_bytes", lambda: (None, None))
    assert "unavailable" in resources.banner(resources.resolve(_config()))


# ----------------------------------------------------------------------
# The cap, as the tool process actually receives it
# ----------------------------------------------------------------------

def _budget(monkeypatch, cpus_per_job):
    """Pin the process-wide allocation for one test."""
    from execution import dispatch

    allocation = resources.Allocation(
        cpus=float(cpus_per_job), ram_bytes=None, vram_bytes=None,
        cpus_per_job=cpus_per_job, ram_per_job=None, vram_per_job=None,
        expected_clients=1,
    )
    monkeypatch.setattr(dispatch.resources, "allocation", lambda: allocation)
    return dispatch


def test_a_tool_process_is_told_how_many_threads_it_may_open(monkeypatch):
    """Nothing set these before, so every tool defaulted to one thread per
    logical core and four of them meant 224 threads on 28 physical ones."""
    dispatch = _budget(monkeypatch, 4)
    environment = dispatch._child_environment("job-1", "/tmp/job-1")
    for name in dispatch._THREAD_VARIABLES:
        assert environment[name] == "4", name


def test_the_whole_family_is_set_together(monkeypatch):
    """A tool honouring one variable and not another still oversubscribes."""
    dispatch = _budget(monkeypatch, 8)
    assert "OMP_NUM_THREADS" in dispatch._THREAD_VARIABLES
    assert "MKL_NUM_THREADS" in dispatch._THREAD_VARIABLES
    assert "OPENBLAS_NUM_THREADS" in dispatch._THREAD_VARIABLES


def test_an_inherited_thread_count_does_not_win(monkeypatch):
    """A stray OMP_NUM_THREADS from whatever started the container is exactly
    the accident this exists to stop. SADT_CPU_PER_JOB is the knob."""
    monkeypatch.setenv("OMP_NUM_THREADS", "56")
    dispatch = _budget(monkeypatch, 4)
    assert dispatch._child_environment("job-1", "/tmp/job-1")["OMP_NUM_THREADS"] == "4"


def test_the_cap_never_reaches_zero(monkeypatch):
    dispatch = _budget(monkeypatch, 1)
    assert dispatch._child_environment("job-1", "/tmp/job-1")["OMP_NUM_THREADS"] == "1"


def test_capping_threads_leaves_the_rest_of_the_environment_alone(monkeypatch):
    """The environment is inherited on purpose: tools need PATH, LD_LIBRARY_PATH
    and CUDA_VISIBLE_DEVICES, and API_TOKEN must still be the one thing removed."""
    monkeypatch.setenv("API_TOKEN", "secret")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    dispatch = _budget(monkeypatch, 2)
    environment = dispatch._child_environment("job-1", "/tmp/job-1")
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert "API_TOKEN" not in environment
