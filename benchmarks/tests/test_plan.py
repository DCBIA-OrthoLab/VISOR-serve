"""Every campaign must be able to build its plan from the SHIPPED config, on a
machine with nothing running. That is what `--dry-run` promises a reviewer."""

from __future__ import annotations

import pytest

from benchmarks.campaigns import _common
from benchmarks.run import CAMPAIGNS
from benchmarks.settings import ConfigError


@pytest.mark.parametrize("campaign", sorted(CAMPAIGNS))
def test_every_campaign_builds_a_plan(campaign, shipped_config):
    plan = CAMPAIGNS[campaign].build_plan(shipped_config, {})
    assert plan, f"{campaign} produced an empty plan"
    for item in plan:
        assert item["campaign"] == campaign
        assert item["tool"] in shipped_config.tools
        assert "path" in item and "runs" in item


@pytest.mark.parametrize("campaign", sorted(CAMPAIGNS))
def test_a_plan_can_be_sized(campaign, shipped_config):
    plan = CAMPAIGNS[campaign].build_plan(shipped_config, {})
    from benchmarks.guards import project_output_bytes

    assert project_output_bytes(plan) >= 0
    assert _common.estimated_seconds(plan) >= 0


def test_b1_marks_the_warmup_and_refuses_to_discard_everything(shipped_config):
    plan = CAMPAIGNS["b1"].build_plan(shipped_config, {"reps": 6})
    runs = [item for item in plan if not item.get("skipped")]
    assert runs and all(item["runs"] == 6 for item in runs)
    assert all(item["warmup"] == 1 for item in runs)

    with pytest.raises(ConfigError, match="must exceed warmup"):
        CAMPAIGNS["b1"].build_plan(shipped_config, {"reps": 1})


def test_b1_skips_a_tool_with_no_local_path_and_says_why(shipped_config):
    plan = CAMPAIGNS["b1"].build_plan(shipped_config, {"paths": ["local"]})
    skipped = [item for item in plan if item.get("skipped")]
    assert skipped, "Test_Tool and Example_Tool have no local path; they must be skipped"
    for item in skipped:
        assert item["skip_reason"], f"{item['tool']} skipped without a reason"


def test_b1_refuses_an_unknown_path(shipped_config):
    with pytest.raises(ConfigError, match="unknown path"):
        CAMPAIGNS["b1"].build_plan(shipped_config, {"paths": ["carrier_pigeon"]})


def test_b2_covers_both_parallelism_settings_for_every_payload(shipped_config):
    plan = CAMPAIGNS["b2"].build_plan(shipped_config, {})
    by_tool = {}
    for item in plan:
        by_tool.setdefault(item["tool"], set()).add(item["parallelism"])
    assert by_tool, "b2 produced nothing"
    for tool, settings_seen in by_tool.items():
        assert len(settings_seen) >= 2, f"{tool} has only one parallelism setting"


def test_b2_refuses_the_local_path(shipped_config):
    """There is no transfer to decompose without a wire."""
    document = dict(shipped_config.campaigns)
    document["b2"] = dict(document["b2"], path="local")
    shipped_config.campaigns = document
    with pytest.raises(ConfigError, match="local"):
        CAMPAIGNS["b2"].build_plan(shipped_config, {})


def test_b3_plans_a_chain_row_per_mode_and_a_startup_row_per_child(shipped_config):
    plan = CAMPAIGNS["b3"].build_plan(shipped_config, {})
    chains = [item for item in plan if item.get("measurement") == "chain"]
    startups = [item for item in plan if item.get("measurement") == "startup"]
    assert len(chains) == len(shipped_config.campaigns["b3"]["modes"])
    assert len(startups) == len(shipped_config.campaigns["b3"]["children"])
    assert len({item["mode"] for item in chains}) == len(chains), "two rows share a mode"


def test_b4_scales_the_run_count_with_the_concurrency(shipped_config):
    plan = CAMPAIGNS["b4"].build_plan(shipped_config, {})
    for item in plan:
        assert item["runs"] == item["concurrency"] * item["jobs_per_client"]


def test_b5_pairs_a_local_run_with_a_remote_one(shipped_config):
    plan = CAMPAIGNS["b5"].build_plan(shipped_config, {})
    for item in plan:
        if item.get("skipped"):
            continue
        assert item["path"] == f"local+{item['remote_path']}"


def test_a_campaign_naming_an_undefined_tool_fails_before_anything_runs(shipped_config):
    shipped_config.campaigns["b1"] = dict(shipped_config.campaigns["b1"], tools=["Nonexistent"])
    with pytest.raises(ConfigError, match="Nonexistent"):
        CAMPAIGNS["b1"].build_plan(shipped_config, {})
