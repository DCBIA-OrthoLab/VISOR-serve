"""`deployment.toml`: what THIS server does with a tool, as opposed to what
the tool is.

The split is the point. A tool's `.schema.json` is generated from its source
and is identical wherever it is installed; which of its arguments can be filled
from this machine's DATA_DIR, and how much this machine accepts as an upload,
are properties of the deployment. Putting either in the schema would make every
installation inherit one server's paths and limits.
"""

import hashlib
import io
import json
import os

import pytest

os.environ.setdefault("API_TOKEN", "test-token")

from fastapi.testclient import TestClient

from registry import deployment
import main
from registry import schema_tool
from config import settings
from registry.deployment import DeploymentConfig, DeploymentConfigError, ToolDeployment

client = TestClient(main.app)
AUTH = {"Authorization": "Bearer test-token"}


def _write(tmp_path, text: str) -> str:
    path = tmp_path / "deployment.toml"
    path.write_text(text)
    return str(path)


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def test_no_file_at_all_is_the_normal_case(tmp_path):
    """A server with no deployment.toml serves every tool with upload-only
    inputs. This must never be an error."""
    config = deployment.load(str(tmp_path / "absent.toml"))

    assert config.configured_tools == ()
    assert config.for_tool("anything") == ToolDeployment()
    assert config.upload_limit_mb("anything") == settings.MAX_UPLOAD_MB


def test_it_reads_what_the_contract_declares(tmp_path):
    path = _write(
        tmp_path,
        """
        [tools.amasss]
        server_selectable = { model = "model", scan = "testfile" }
        max_upload_mb = 500
        """,
    )

    entry = deployment.load(path).for_tool("amasss")

    assert entry.server_selectable == {"model": "model", "scan": "testfile"}
    assert entry.max_upload_mb == 500


def test_a_typo_in_a_key_is_refused(tmp_path):
    """Silent otherwise: `server_selectible` would leave every argument
    upload-only, with no dropdown and nothing saying why."""
    path = _write(tmp_path, '[tools.amasss]\nserver_selectible = { model = "model" }\n')

    with pytest.raises(DeploymentConfigError, match="unknown key"):
        deployment.load(path)


def test_an_unknown_selectable_kind_is_refused(tmp_path):
    path = _write(tmp_path, '[tools.amasss]\nserver_selectable = { model = "weights" }\n')

    with pytest.raises(DeploymentConfigError, match="expected one of"):
        deployment.load(path)


def test_a_nonsense_upload_limit_is_refused(tmp_path):
    path = _write(tmp_path, "[tools.amasss]\nmax_upload_mb = 0\n")

    with pytest.raises(DeploymentConfigError, match="positive integer"):
        deployment.load(path)


def test_a_malformed_file_names_itself(tmp_path):
    path = _write(tmp_path, "[tools.amasss\n")

    with pytest.raises(DeploymentConfigError, match="Cannot read"):
        deployment.load(path)


# ----------------------------------------------------------------------
# Applied to a schema tool
# ----------------------------------------------------------------------

def test_server_selectable_reaches_the_published_schema(make_tool_folder):
    folder = make_tool_folder(
        "selectable", arguments={"scan": {"type": "path", "required": True}}
    )
    config = DeploymentConfig({"selectable": ToolDeployment(server_selectable={"scan": "testfile"})})

    tool = schema_tool.load_tool(folder, config)

    assert tool.arguments["scan"].server_selectable == "testfile"


def test_a_tool_with_no_entry_gets_the_conventions(make_tool_folder):
    """Adding a tool must need no edit to this repository, so an empty
    deployment.toml is the normal case rather than a lapse."""
    folder = make_tool_folder(
        "plain",
        arguments={
            "scan": {"type": "path", "required": True},
            "model": {"type": "path", "required": True},
            "device": {"type": "str", "required": False},
        },
    )

    tool = schema_tool.load_tool(folder, DeploymentConfig({}))

    assert tool.arguments["scan"].server_selectable == "testfile"
    # A name, never an upload: weights do not travel from a clinician's laptop.
    assert tool.arguments["model"].server_selectable == "model"
    assert tool.arguments["model"].type is str
    assert tool.arguments["device"].hidden is True


def test_selecting_an_argument_the_tool_does_not_declare_is_refused(make_tool_folder):
    """A dropdown that never appears, otherwise."""
    folder = make_tool_folder("mismatched")
    config = DeploymentConfig({"mismatched": ToolDeployment(server_selectable={"nope": "model"})})

    with pytest.raises(schema_tool.SchemaError, match="declares no such argument"):
        schema_tool.load_tool(folder, config)


def test_selecting_a_non_path_argument_is_refused(make_tool_folder):
    """The two files disagree about what the argument is; only one of them was
    generated from the source."""
    folder = make_tool_folder("wrongtype")
    config = DeploymentConfig({"wrongtype": ToolDeployment(server_selectable={"a": "model"})})

    with pytest.raises(schema_tool.SchemaError, match="rather than 'path'"):
        schema_tool.load_tool(folder, config)


# ----------------------------------------------------------------------
# max_upload_mb, over real HTTP
# ----------------------------------------------------------------------

@pytest.fixture
def one_megabyte_limit(monkeypatch):
    """A per-tool limit well under the global one, applied to a tool that takes
    a file. It works for an imported tool as much as for a schema one: an
    upload limit is deployment policy and says nothing about the tool."""
    monkeypatch.setattr(
        main,
        "deployment_config",
        DeploymentConfig({"Example_Tool": ToolDeployment(max_upload_mb=1)}),
    )


def test_a_multipart_upload_over_the_tool_limit_is_413(one_megabyte_limit):
    payload = b"col\n" + b"1\n" * (1024 * 1024)

    response = client.post(
        "/run/Example_Tool",
        headers=AUTH,
        data={"label": "x"},
        files={"input": ("big.csv", io.BytesIO(payload), "text/csv")},
    )

    assert response.status_code == 413
    assert "1 MB" in response.json()["detail"]


def test_a_chunked_upload_over_the_tool_limit_is_413_before_it_is_claimed(one_megabyte_limit):
    """The chunked path opens its session without naming a tool, so the size is
    checked at the only point the tool is known: the claim. The blob must not
    be moved into the run's work dir first."""
    payload = b"col\n" + b"1\n" * (1024 * 1024)
    session = client.post(
        "/uploads", headers=AUTH, json={"filename": "big.csv", "size": len(payload)}
    ).json()
    chunk_size = session["chunk_size"]
    for index in range(session["part_count"]):
        part = payload[index * chunk_size : (index + 1) * chunk_size]
        client.put(
            f"/uploads/{session['upload_id']}/parts/{index}",
            headers={**AUTH, "X-Part-SHA256": hashlib.sha256(part).hexdigest()},
            content=part,
        )

    response = client.post(
        "/run/Example_Tool",
        headers=AUTH,
        data={"label": "x", "__uploads__": json.dumps({"input": session["upload_id"]})},
    )

    assert response.status_code == 413
    # And the refused session does not linger on disk holding patient data.
    assert client.get(f"/uploads/{session['upload_id']}", headers=AUTH).status_code == 404


def test_under_the_limit_still_runs(one_megabyte_limit):
    response = client.post(
        "/run/Example_Tool",
        headers=AUTH,
        data={"label": "x", "threshold": "0.5"},
        files={"input": ("small.csv", io.BytesIO(b"col\n1\n2\n"), "text/csv")},
    )

    assert response.status_code == 200, response.text


def test_the_global_limit_applies_when_no_tool_declares_one():
    assert deployment.deployment_config.upload_limit_mb("Example_Tool") == settings.MAX_UPLOAD_MB
