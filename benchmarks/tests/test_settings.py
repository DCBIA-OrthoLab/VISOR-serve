"""The config file is the harness's only input, so its validation is the only
thing standing between a typo and a wasted GPU hour."""

from __future__ import annotations

import os

import pytest

from benchmarks import settings
from benchmarks.settings import ConfigError


def test_shipped_config_loads(shipped_config):
    assert shipped_config.server.base_url.startswith("http")
    assert shipped_config.tools, "the shipped config defines no tools"
    assert set(shipped_config.campaigns) >= {"b1", "b2", "b3", "b4", "b5"}


def test_shipped_config_paths_resolve_against_the_repo(shipped_config):
    """A relative path in config.yaml must not depend on the current directory."""
    tool = shipped_config.tools["Crown_Seg"]
    for path in tool.files.values():
        assert os.path.isabs(path)
        assert path.startswith(settings.REPO_ROOT)


def test_every_tool_declares_a_local_decision(shipped_config):
    """Either a folder and a package, or an explicit refusal with a reason.

    A tool that says nothing would silently drop out of the local arm of B1, and
    the missing row would look like a measurement rather than an omission.
    """
    for name, tool in shipped_config.tools.items():
        assert tool.local is not None, f"{name} says nothing about its local path"
        if tool.supports_local:
            assert tool.local.folder, f"{name} supports local but names no folder"
            assert tool.local.package, f"{name} supports local but names no package"
        else:
            assert tool.local.reason, f"{name} refuses local without saying why"


def test_missing_base_url_is_named(minimal_document):
    del minimal_document["server"]["base_url"]
    with pytest.raises(ConfigError, match="base_url"):
        settings.parse(minimal_document)


def test_unknown_local_mode_is_refused(minimal_document):
    minimal_document["local"]["mode"] = "somewhere_else"
    with pytest.raises(ConfigError, match="local.mode"):
        settings.parse(minimal_document)


def test_container_mode_needs_a_container(minimal_document):
    minimal_document["local"] = {"mode": "container"}
    with pytest.raises(ConfigError, match="names no container"):
        settings.parse(minimal_document)


def test_server_file_needs_a_kind(minimal_document):
    """'model' and 'testfile' live in different folders, and the local path has
    no server to ask which one a name is in."""
    minimal_document["tools"]["Widget"]["server_files"] = {"model": "some_bundle"}
    with pytest.raises(ConfigError, match="kind"):
        settings.parse(minimal_document)


def test_server_file_kind_is_checked(minimal_document):
    minimal_document["tools"]["Widget"]["server_files"] = {
        "model": {"kind": "weights", "name": "b"}
    }
    with pytest.raises(ConfigError, match="must be 'model' or 'testfile'"):
        settings.parse(minimal_document)


def test_unknown_tool_in_a_campaign_names_the_tool(minimal_document):
    configuration = settings.parse(minimal_document)
    with pytest.raises(ConfigError, match="Gadget"):
        configuration.tool("Gadget")


def test_data_slug_defaults_to_the_tool_name(minimal_document):
    configuration = settings.parse(minimal_document)
    assert configuration.tools["Widget"].data_slug == "Widget"


def test_token_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv(settings.TOKEN_ENVIRONMENT_VARIABLE, "from-env")
    assert settings.read_token() == "from-env"


def test_token_comes_from_a_dotenv_when_the_environment_is_empty(monkeypatch, tmp_path):
    monkeypatch.delenv(settings.TOKEN_ENVIRONMENT_VARIABLE, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("# a comment\nDEVICE=cuda\nAPI_TOKEN='from-file'\n")
    assert settings.read_token(str(env_file)) == "from-file"


def test_no_token_is_an_actionable_error(monkeypatch, tmp_path):
    monkeypatch.delenv(settings.TOKEN_ENVIRONMENT_VARIABLE, raising=False)
    with pytest.raises(ConfigError, match="API_TOKEN"):
        settings.require_token(str(tmp_path / "absent.env"))


def test_the_token_is_not_in_the_config_file():
    """The published config must never carry a credential."""
    with open(os.path.join(settings.BENCHMARKS_ROOT, "config.yaml"), encoding="utf-8") as handle:
        body = handle.read()
    assert "API_TOKEN:" not in body
    assert "dev-token" not in body
