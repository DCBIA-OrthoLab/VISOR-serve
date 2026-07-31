"""Integration tests against real server-side data (DATA_DIR).

Unlike the synthetic fixtures in test_main.py and tools/*/test/, these tests
run a tool end-to-end against the real models/test files a maintainer drops
under DATA/<tool_name>/{models,testfiles}/ (see data_store.py). DATA/ is
gitignored -- it holds confidential medical data -- so those files only ever
exist locally, never in the repo.

Because of that, a tool is SKIPPED (not failed) when no matching file is
present under DATA_DIR: a machine without the confidential dataset must
still be able to push. Add a file under DATA/<tool_name>/testfiles/ (and
DATA/<tool_name>/models/ if the tool also takes a model) to turn a skip into
a real run.

**These tests are opt-in**, and that is not a matter of taste. They run real
inference against real weights: ALI with its full bundle present is ~11
minutes on a GPU and hours on a CPU, because the request carries no selection
and the tool therefore predicts all 119 landmarks. The pre-push hook cannot
wait for that, and a hook that takes hours is a hook people disable.

    docker compose run --rm test-gpu     # sets RUN_REAL_DATA_TESTS=1
    RUN_REAL_DATA_TESTS=1 pytest tests/test_data_integration.py

The plain `docker compose run --rm test` (what .githooks/pre-push runs) skips
this module entirely and stays a few seconds long.
"""

import os

os.environ.setdefault("API_TOKEN", "test-token")

import pytest
from fastapi.testclient import TestClient

if not os.environ.get("RUN_REAL_DATA_TESTS"):
    pytest.skip(
        "Real-data integration tests are opt-in: they run real inference against the "
        "bundles under DATA/, which is minutes to hours. Set RUN_REAL_DATA_TESTS=1, or "
        "run `docker compose run --rm test-gpu`.",
        allow_module_level=True,
    )

from data_store import data_store
from main import app
from registry import TOOLS

client = TestClient(app)
TOKEN = os.environ["API_TOKEN"]


def _tools_backed_by_server_data():
    """Tools whose every required argument can be supplied from DATA_DIR.

    A tool with a required argument that is neither server_selectable nor
    something this generic test can fabricate a value for is left out --
    there is no sensible way to guess it from here.
    """
    return [
        tool
        for tool in TOOLS.values()
        if any(spec.server_selectable for spec in tool.arguments.values())
        and all(spec.server_selectable for spec in tool.arguments.values() if spec.required)
    ]


@pytest.mark.parametrize(
    "tool", _tools_backed_by_server_data(), ids=lambda t: t.name
)
def test_tool_runs_against_real_data(tool):
    form_data = {}
    for arg_name, spec in tool.arguments.items():
        if not spec.server_selectable:
            continue
        available = (
            data_store.list_models(tool.name)
            if spec.server_selectable == "model"
            else data_store.list_testfiles(tool.name)
        )
        if not available:
            if spec.required:
                pytest.skip(
                    f"No {spec.server_selectable} file for tool '{tool.name}' -- "
                    f"add one under DATA/{tool.name}/{spec.server_selectable}s/ "
                    f"to run this test."
                )
            continue
        form_data[arg_name] = available[0]

    response = client.post(
        f"/run/{tool.name}",
        headers={"Authorization": f"Bearer {TOKEN}"},
        data=form_data,
    )

    assert response.status_code == 200, response.text
    if tool.output_kind in ("file", "segmentation"):
        assert len(response.content) > 0
