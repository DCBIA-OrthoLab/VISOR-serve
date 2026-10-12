"""`server_ctl.py uninstall`, without a docker: what goes, what stays, what refuses.

The command removes a deployment's clone, so the tests here are mostly about
what it must NOT do -- remove a directory that is not a clone, lose the token
every client is configured with, lose DATA/, touch a tools checkout nobody
asked about, or delete anything at all when it cannot keep what it promised.
"""

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
    spec = importlib.util.spec_from_file_location("server_ctl_uninstall_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ctl = _load_server_ctl()


@pytest.fixture(autouse=True)
def no_docker(monkeypatch):
    """No docker, no systemctl: only the filesystem half runs."""
    monkeypatch.setattr(ctl, "compose_command", lambda: None)
    monkeypatch.setattr(ctl, "_which", lambda name: None)


def _write(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _clone(tmp_path, env="API_TOKEN=abc\nADMIN_TOKEN=def\nHOST_PORT=8000\n", data=True):
    root = tmp_path / "VISOR-serve"
    _write(str(root / ".git" / "HEAD"), "ref: refs/heads/deploy\n")
    _write(str(root / "scripts" / "server_ctl.py"))
    _write(str(root / "docker-compose.yml"))
    _write(str(root / ".env"), env)
    _write(str(root / "server" / ".env"), "SADT_CPU_PER_JOB=8\n")
    if data:
        _write(str(root / "DATA" / "AMASSS" / "models" / "weights.pth"), "x" * 10)
    return root


def _args(root, **overrides):
    values = dict(repo_root=str(root), data=False, tools_clone=False, dry_run=False, yes=True)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_a_dry_run_removes_nothing(tmp_path):
    root = _clone(tmp_path)
    result = ctl.cmd_uninstall(_args(root, dry_run=True))
    assert result["dry_run"]
    assert (root / "DATA" / "AMASSS" / "models" / "weights.pth").exists()
    assert not os.path.exists(ctl.kept_dir(str(root)))


def test_the_tokens_and_the_data_are_kept_and_the_clone_goes(tmp_path):
    root = _clone(tmp_path)
    result = ctl.cmd_uninstall(_args(root))
    kept = tmp_path / "VISOR-serve.kept"

    assert result["removed_clone"] and not root.exists()
    assert (kept / "DATA" / "AMASSS" / "models" / "weights.pth").read_text() == "x" * 10
    env = (kept / ".env").read_text()
    assert "API_TOKEN=abc" in env and "ADMIN_TOKEN=def" in env
    assert "HOST_PORT" not in env, "only identity is kept; the next install writes the rest"
    assert (kept / "server.env").read_text() == "SADT_CPU_PER_JOB=8\n"
    assert oct(os.stat(kept / ".env").st_mode)[-3:] == "600"


def test_data_goes_too_when_asked(tmp_path):
    root = _clone(tmp_path)
    ctl.cmd_uninstall(_args(root, data=True))
    assert not root.exists()
    assert not (tmp_path / "VISOR-serve.kept" / "DATA").exists()


@pytest.mark.parametrize("missing", [".git", os.path.join("scripts", "server_ctl.py"), "docker-compose.yml"])
def test_a_directory_that_is_not_a_clone_is_refused(tmp_path, missing):
    root = _clone(tmp_path)
    target = root / missing
    if target.is_dir():
        import shutil
        shutil.rmtree(target)
    else:
        target.unlink()
    with pytest.raises(ctl.ServerCtlError):
        ctl.cmd_uninstall(_args(root))
    assert (root / ".env").exists()


def test_a_home_directory_is_never_a_clone(tmp_path, monkeypatch):
    root = _clone(tmp_path)
    monkeypatch.setenv("HOME", str(root))
    assert not ctl._is_this_clone(str(root))


def test_data_that_cannot_move_aside_stops_everything(tmp_path, monkeypatch):
    """A DATA/ docker created as root cannot be renamed by the user. Stopping
    then, before any removal, leaves a deployment that still works."""
    root = _clone(tmp_path)

    def refuse(*_args):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(ctl.os, "rename", refuse)
    with pytest.raises(ctl.ServerCtlError, match="nothing was removed"):
        ctl.cmd_uninstall(_args(root))
    assert (root / "DATA" / "AMASSS" / "models" / "weights.pth").exists()
    assert (root / ".env").exists()


def test_a_second_uninstall_does_not_overwrite_kept_data(tmp_path):
    root = _clone(tmp_path)
    _write(str(tmp_path / "VISOR-serve.kept" / "DATA" / "old"), "kept before")
    with pytest.raises(ctl.ServerCtlError, match="already holds data"):
        ctl.cmd_uninstall(_args(root))
    assert root.exists()
    assert (tmp_path / "VISOR-serve.kept" / "DATA" / "old").read_text() == "kept before"


def test_a_tools_checkout_is_left_alone_unless_asked(tmp_path):
    tools = tmp_path / "SADT-VISOR"
    _write(str(tools / ".git" / "HEAD"))
    root = _clone(tmp_path, env=f"API_TOKEN=abc\nSADT_TOOLS={tools}\n")
    plan = ctl.uninstall_plan(str(root))
    assert plan["tools_clone"] is None and plan["tools_clone_kept"] == str(tools)

    ctl.cmd_uninstall(_args(root))
    assert tools.exists()


def test_a_tools_checkout_goes_when_asked(tmp_path):
    tools = tmp_path / "SADT-VISOR"
    _write(str(tools / ".git" / "HEAD"))
    root = _clone(tmp_path, env=f"API_TOKEN=abc\nSADT_TOOLS={tools}\n")
    ctl.cmd_uninstall(_args(root, tools_clone=True))
    assert not tools.exists()


def test_only_the_images_this_deployment_names_are_listed(tmp_path):
    root = _clone(tmp_path, env="API_TOKEN=abc\nSADT_TOOLS_IMAGE=ghcr.io/x/tools:tools-1\n")
    assert ctl.uninstall_plan(str(root))["images"] == [ctl.DEFAULT_SERVER_IMAGE, "ghcr.io/x/tools:tools-1"]
    root2 = _clone(tmp_path / "other", env="VISOR_SERVER_IMAGE=local:1\n")
    assert ctl.uninstall_plan(str(root2))["images"] == ["local:1"]


def test_without_a_terminal_it_will_not_proceed_unconfirmed(tmp_path, monkeypatch):
    root = _clone(tmp_path)
    monkeypatch.setattr(ctl.sys.stdin, "isatty", lambda: False)
    with pytest.raises(ctl.ServerCtlError, match="--yes"):
        ctl.cmd_uninstall(_args(root, yes=False))
    assert root.exists()
