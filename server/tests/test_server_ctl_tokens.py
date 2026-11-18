"""The two tokens `server_ctl.py` keeps in .env: the API token every
workstation holds, and the admin panel's. Both are generated once and kept --
regenerating either would lock out whoever already has it."""

import argparse
import importlib.util
import os
import sys

import pytest

_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts",
)


def _load_server_ctl():
    path = os.path.join(_SCRIPTS, "server_ctl.py")
    if not os.path.isfile(path):
        pytest.skip("scripts/server_ctl.py is not mounted here")
    spec = importlib.util.spec_from_file_location("server_ctl_tokens_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ctl = _load_server_ctl()


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    """Point every .env read and write at a file of the test's own."""
    path = str(tmp_path / ".env")
    read, write = ctl.read_env, ctl.write_env
    monkeypatch.setattr(ctl, "ENV_PATH", path)
    monkeypatch.setattr(ctl, "read_env", lambda p=None: read(p or path))
    monkeypatch.setattr(ctl, "write_env", lambda updates, p=None: write(updates, p or path))
    monkeypatch.setattr(ctl, "effective_port", lambda: 8000)
    return path


def test_a_fresh_deployment_gets_both_tokens_and_they_differ(env_file):
    ctl.ensure_env(ctl.GPU_SERVICE)
    env = ctl.read_env(env_file)
    assert env["API_TOKEN"] and env["ADMIN_TOKEN"]
    assert env["API_TOKEN"] != env["ADMIN_TOKEN"], "a workstation's token must not open the admin panel"


def test_existing_tokens_are_kept(env_file):
    with open(env_file, "w") as handle:
        handle.write("API_TOKEN=api\nADMIN_TOKEN=admin\n")
    ctl.ensure_env(ctl.GPU_SERVICE)
    env = ctl.read_env(env_file)
    assert (env["API_TOKEN"], env["ADMIN_TOKEN"]) == ("api", "admin")


def test_an_admin_token_is_added_to_a_deployment_that_had_none(env_file):
    with open(env_file, "w") as handle:
        handle.write("API_TOKEN=api\n")
    ctl.ensure_env(ctl.GPU_SERVICE)
    env = ctl.read_env(env_file)
    assert env["API_TOKEN"] == "api" and env["ADMIN_TOKEN"]


def test_token_prints_the_api_token_and_admin_the_admin_one(env_file):
    with open(env_file, "w") as handle:
        handle.write("API_TOKEN=api\nADMIN_TOKEN=admin\n")
    assert ctl.cmd_token(argparse.Namespace(admin=False)) == {"token": "api"}
    assert ctl.cmd_token(argparse.Namespace(admin=True)) == {"token": "admin"}


def test_a_missing_admin_token_says_how_to_get_one(env_file):
    with open(env_file, "w") as handle:
        handle.write("API_TOKEN=api\n")
    with pytest.raises(ctl.ServerCtlError, match="ADMIN_TOKEN"):
        ctl.cmd_token(argparse.Namespace(admin=True))


def test_the_cli_takes_admin(env_file, capsys):
    with open(env_file, "w") as handle:
        handle.write("API_TOKEN=api\nADMIN_TOKEN=admin\n")
    assert ctl.main(["token", "--admin"]) == 0
    assert capsys.readouterr().out.strip() == "admin"
