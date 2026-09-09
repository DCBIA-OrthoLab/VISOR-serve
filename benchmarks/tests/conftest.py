"""Fixtures for the harness's own suite.

Every test in this directory runs on a machine with no GPU, no Docker daemon and
no server. The ones that genuinely need one of those carry a marker and skip with
a reason naming what is missing -- never a silent pass, and never a failure that
looks like a bug in the harness.
"""

from __future__ import annotations

import os
import shutil

import pytest

from benchmarks import settings
from benchmarks.execution.remote import RemoteClient
from benchmarks.settings import BENCHMARKS_ROOT

# The shipped file by default; $BENCHMARKS_CONFIG lets a reviewer point the
# whole suite at their own deployment's config without editing anything.
SHIPPED_CONFIG = os.environ.get(
    settings.CONFIG_ENVIRONMENT_VARIABLE, os.path.join(BENCHMARKS_ROOT, "config.yaml")
)


@pytest.fixture
def shipped_config() -> settings.Config:
    """The config.yaml this repository ships.

    Loaded rather than synthesised on purpose: it is a published artifact and a
    campaign that cannot build its plan from it is a broken deliverable, not a
    broken test fixture.
    """
    return settings.load(SHIPPED_CONFIG)


@pytest.fixture
def minimal_document() -> dict:
    """The smallest document `parse` accepts, for tests about ONE key."""
    return {
        "server": {"base_url": "http://example.invalid:8000"},
        "local": {"mode": "host"},
        "tools": {
            "Widget": {
                "args": {"alpha": 1},
                "local": {"folder": "Widget", "package": "sadt_widget"},
                "estimated_seconds": 2,
                "estimated_output_mb": 3,
            }
        },
        "campaigns": {
            "b1": {"reps": 6, "warmup": 1, "paths": ["local"], "tools": ["Widget"]},
        },
    }


@pytest.fixture
def isolated_root(tmp_path) -> str:
    """A results/ tree of its own, so no test writes into the published one."""
    root = tmp_path / "harness"
    (root / "results" / "raw").mkdir(parents=True)
    (root / "results" / "summary").mkdir(parents=True)
    return str(root)


@pytest.fixture
def live_server(shipped_config):
    """A client against the configured server, or a skip naming why not."""
    token = settings.read_token()
    if not token:
        pytest.skip("no API_TOKEN in the environment or in the server repo's .env")
    client = RemoteClient(shipped_config.server, shipped_config.transfer, token)
    reason = client.unavailable_reason()
    if reason is not None:
        client.close()
        pytest.skip(reason)
    yield client
    client.close()


@pytest.fixture
def live_container(shipped_config):
    """The deployment container, or a skip naming why not."""
    from benchmarks.execution.local import LocalRunner

    if shutil.which("docker") is None:
        pytest.skip("docker is not on PATH")
    runner = LocalRunner(shipped_config.local)
    reason = runner.unavailable_reason()
    if reason is not None:
        pytest.skip(reason)
    return runner
