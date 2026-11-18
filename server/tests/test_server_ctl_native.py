"""The native backend of server_ctl.py and the Mac install recipe, with no network and no uv run.

What is pinned here: which backend a command picks (Docker stays the default
anywhere but a Mac), what the recipe keeps and drops from a lock, and that
`down` really takes a tool run with it -- the one thing a process, unlike a
container, does not do by itself.
"""

import importlib.util
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time
import types

import pytest

_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts",
)


def _load(name):
    path = os.path.join(_SCRIPTS, name + ".py")
    if not os.path.isfile(path):
        pytest.skip("scripts/%s.py is not mounted here" % name)
    if _SCRIPTS not in sys.path:
        sys.path.insert(0, _SCRIPTS)
    spec = importlib.util.spec_from_file_location(name + "_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ctl = _load("server_ctl")
tools = _load("native_tools")
native = _load("server_ctl_native")


def _config():
    with open(os.path.join(_SCRIPTS, "native_tools.json"), encoding="utf-8") as handle:
        return json.load(handle)


# ----------------------------------------------------------------------
# Which backend
# ----------------------------------------------------------------------

def _args(**kwargs):
    return types.SimpleNamespace(**kwargs)


def test_docker_is_the_default_off_a_mac(monkeypatch):
    monkeypatch.delenv("VISOR_BACKEND", raising=False)
    monkeypatch.setattr(ctl.sys, "platform", "linux")
    assert ctl._wants_native(_args(native=False)) is False
    monkeypatch.setattr(ctl.sys, "platform", "win32")
    assert ctl._wants_native(_args(native=False)) is False


def test_native_is_the_default_on_a_mac_and_docker_can_be_forced(monkeypatch):
    monkeypatch.delenv("VISOR_BACKEND", raising=False)
    monkeypatch.setattr(ctl.sys, "platform", "darwin")
    assert ctl._wants_native(_args(native=False)) is True
    monkeypatch.setenv("VISOR_BACKEND", "docker")
    assert ctl._wants_native(_args(native=False)) is False


def test_the_flag_and_the_variable_ask_for_native_on_linux(monkeypatch):
    monkeypatch.delenv("VISOR_BACKEND", raising=False)
    monkeypatch.setattr(ctl.sys, "platform", "linux")
    assert ctl._wants_native(_args(native=True)) is True
    monkeypatch.setenv("VISOR_BACKEND", "native")
    assert ctl._wants_native(_args(native=False)) is True


def test_a_docker_command_never_imports_the_native_modules(monkeypatch):
    """`main` loads server_ctl_native only on the native branch."""
    monkeypatch.delenv("VISOR_BACKEND", raising=False)
    monkeypatch.setattr(ctl.sys, "platform", "linux")
    for name in ("server_ctl_native", "native_tools"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    seen = []
    monkeypatch.setattr(ctl, "cmd_logs", lambda a: seen.append(a.command) or {"service": "x", "lines": 1})
    parser = ctl.build_parser()
    args = parser.parse_args(["logs"])
    assert args.native is False
    assert ctl._wants_native(args) is False
    assert "server_ctl_native" not in sys.modules


def test_the_parser_still_defaults_every_docker_option():
    args = ctl.build_parser().parse_args(["up"])
    assert args.native is False and args.device == "auto" and args.bind is None


# ----------------------------------------------------------------------
# The recipe: what is kept from a lock
# ----------------------------------------------------------------------

LOCK = [
    "../../naming",
    "numpy==2.3.2",
    "torch==2.8.0+cu128",
    "torchvision==0.23.0+cu128",
    "nvidia-cudnn-cu12==9.10.2.21 ; sys_platform == 'linux'",
    "cuda-toolkit==12.8.1",
    "triton==3.4.0 ; sys_platform == 'linux'",
    "pytorch3d==0.7.9+pt2110cu128",
    "llama-cpp-python==0.3.35",
    "sympy==1.14.0",
    "colorama==0.4.6 ; sys_platform == 'win32'",
    "-e ../../testkit",
]


def test_a_cuda_build_is_dropped_and_everything_else_is_kept(tmp_path):
    kept, dropped = tools.split_requirements(LOCK, _config(), str(tmp_path / "tools" / "T"))
    assert set(dropped) == {"torch", "torchvision", "nvidia-cudnn-cu12", "cuda-toolkit",
                            "triton", "pytorch3d", "llama-cpp-python"}
    names = [line.split("==")[0].split(" ")[0] for line in kept]
    assert "numpy" in names and "sympy" in names and "colorama" in names
    assert not any("cu128" in line or "torch" in line for line in kept)


def test_path_dependencies_become_absolute(tmp_path):
    tool = tmp_path / "tools" / "T"
    kept, _dropped = tools.split_requirements(LOCK, _config(), str(tool))
    assert str(tmp_path / "naming") in kept
    assert "-e " + str(tmp_path / "testkit") in kept


def _lock_for(name, extra=()):
    return ["numpy==2.3.2", "torch==2.8.0+cu128", "torchvision==0.23.0+cu128"] + list(extra)


def _make_tool(tmp_path, name="T"):
    folder = tmp_path / "tools" / name
    folder.mkdir(parents=True)
    (folder / "pyproject.toml").write_text(
        '[project]\nname = "x"\nrequires-python = ">=3.11,<3.12"\n[tool.sadt]\ntool = true\nname = "%s"\n' % name)
    (folder / "uv.lock").write_text("# lock\n")
    return str(folder)


def test_a_torch_tool_gets_the_recent_torch_from_the_config(tmp_path):
    folder = _make_tool(tmp_path, "AMASSS")
    plan = tools.plan_for("AMASSS", folder, _config(), _lock_for("AMASSS"))
    assert plan["torch"]["torch"] == "2.14.1"
    assert plan["pytorch3d"] is None
    assert plan["note"] is None
    assert plan["python"] == ">=3.11,<3.12"


def test_a_tool_importing_pytorch3d_is_held_to_the_torch_its_wheel_was_built_for(tmp_path):
    folder = _make_tool(tmp_path, "ALI_IOS")
    plan = tools.plan_for("ALI_IOS", folder, _config(),
                          _lock_for("ALI_IOS", ["pytorch3d==0.7.9+pt2110cu128"]))
    assert plan["torch"] == {"torch": "2.8.0", "torchvision": "0.23.0"}
    assert plan["pytorch3d"] == "0.7.9+pt280cpu"
    assert plan["extras"] == ["embreex==4.4.0.post1"]
    assert "2.8.0" in plan["note"]


def test_a_tool_without_torch_gets_none(tmp_path):
    folder = _make_tool(tmp_path, "ASO")
    plan = tools.plan_for("ASO", folder, _config(), ["numpy==2.3.2"])
    assert plan["torch"] is None and plan["pytorch3d"] is None and plan["llama_cpp"] is None


def test_cne_gets_the_metal_llama_cpp_wheel(tmp_path):
    folder = _make_tool(tmp_path, "CNE")
    plan = tools.plan_for("CNE", folder, _config(),
                          ["llama-cpp-python==0.3.35", "nvidia-cublas-cu12==12.8.4.1", "pymupdf==1.28.2"])
    assert plan["llama_cpp"] == "llama-cpp-python==0.3.35"
    assert plan["llama_cpp_index"].endswith("/whl/metal")
    assert plan["requirements"] == ["pymupdf==1.28.2"]


def test_the_fingerprint_moves_when_the_recipe_or_the_lock_does(tmp_path):
    folder = _make_tool(tmp_path, "AMASSS")
    config = _config()
    before = tools.fingerprint(tools.plan_for("AMASSS", folder, config, _lock_for("AMASSS")), folder)
    same = tools.fingerprint(tools.plan_for("AMASSS", folder, config, _lock_for("AMASSS")), folder)
    assert before == same
    config["torch_by_tool"]["AMASSS"] = {"torch": "2.15.0"}
    assert tools.fingerprint(tools.plan_for("AMASSS", folder, config, _lock_for("AMASSS")), folder) != before
    with open(os.path.join(folder, "uv.lock"), "a") as handle:
        handle.write("# changed\n")
    assert tools.fingerprint(tools.plan_for("AMASSS", folder, _config(), _lock_for("AMASSS")), folder) != before


def test_discovery_follows_the_dockerfile_rule(tmp_path):
    _make_tool(tmp_path, "AMASSS")
    group = tmp_path / "tools" / "ALI"
    nested = group / "ALI_CBCT"
    nested.mkdir(parents=True)
    (nested / "pyproject.toml").write_text('[project]\nname="a"\n[tool.sadt]\ntool = true\nname = "ALI_CBCT"\n')
    (nested / "uv.lock").write_text("")
    common = group / "common"
    common.mkdir()
    (common / "pyproject.toml").write_text('[project]\nname="c"\n')
    (common / "uv.lock").write_text("")
    assert sorted(tools.discover_tools(str(tmp_path))) == ["ALI_CBCT", "AMASSS"]


def test_the_config_names_only_tools_that_exist_in_the_recipe_shape():
    config = _config()
    assert config["default_torch"]["torch"] == "2.14.1"
    assert config["pytorch3d"]["requires"]["torch"] == "2.8.0"


# ----------------------------------------------------------------------
# The process
# ----------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("ps") is None, reason="needs ps, which a slim image may not have")
def test_stop_takes_a_tool_in_its_own_session_with_it(tmp_path):
    """A tool run is started with start_new_session=True. Docker's `stop` kills
    it with the container; here it has to be found by parentage."""
    child_pidfile = tmp_path / "child.pid"
    server_code = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)\n"
        "open(%r, 'w').write(str(p.pid))\n"
        "time.sleep(60)\n" % str(child_pidfile)
    )
    server = subprocess.Popen([sys.executable, "-c", server_code], start_new_session=True)
    try:
        for _ in range(100):
            if child_pidfile.exists() and child_pidfile.read_text():
                break
            time.sleep(0.05)
        child = int(child_pidfile.read_text())
        assert native.pid_alive(server.pid) and native.pid_alive(child)
        outcome = native.stop_process_tree(server.pid, grace=10)
        server.wait(timeout=10)
        for _ in range(100):
            if not native.pid_alive(child):
                break
            time.sleep(0.05)
        assert not native.pid_alive(child), "the tool run survived `down`"
        assert outcome["killed"] is False
    finally:
        for pid in (server.pid,):
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass


@pytest.mark.skipif(shutil.which("ps") is None, reason="needs ps")
def test_a_dead_pid_is_not_alive_and_a_zombie_is_not_either():
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    assert native.pid_alive(process.pid) is False


def test_the_pidfile_of_something_else_is_not_our_server(tmp_path, monkeypatch):
    monkeypatch.setenv("VISOR_STATE_DIR", str(tmp_path))
    with open(native.pidfile_path(), "w") as handle:
        json.dump({"pid": os.getpid(), "port": 1}, handle)
    assert native.read_pidfile()["pid"] == os.getpid()
    assert native.running_server() is None  # this pytest is not a uvicorn main:app


def test_state_and_logs_live_where_a_mac_user_expects(monkeypatch):
    monkeypatch.delenv("VISOR_STATE_DIR", raising=False)
    monkeypatch.delenv("VISOR_LOG_DIR", raising=False)
    monkeypatch.setattr(native.sys, "platform", "darwin")
    assert native.log_dir() == os.path.expanduser("~/Library/Logs/VISOR")
    assert native.state_dir().startswith(os.path.expanduser("~/Library/Application Support/VISOR"))
    monkeypatch.setattr(native.sys, "platform", "linux")
    assert native.state_dir().startswith(os.path.expanduser("~/.local/state/visor"))


def test_the_default_device_is_the_platforms_gpu(monkeypatch):
    monkeypatch.setattr(native.sys, "platform", "darwin")
    assert native.default_device() == "mps"
    monkeypatch.setattr(native.sys, "platform", "linux")
    monkeypatch.setattr("shutil.which", lambda _n: None)
    assert native.default_device() == "cpu"


def test_run_native_sh_is_valid_bash_and_forces_the_subprocess_dispatch():
    script = os.path.join(_SCRIPTS, "run-native.sh")
    assert subprocess.run(["bash", "-n", script]).returncode == 0
    text = open(script, encoding="utf-8").read()
    assert "SADT_DISPATCH_MODE=subprocess" in text
    assert "BIND_ADDR:-127.0.0.1" in text
    assert 'serve.py --host "$BIND"' in text
    assert "DEVICE=mps" in text


def test_the_native_status_keeps_the_keys_a_docker_client_reads(tmp_path, monkeypatch):
    """The Slicer panel reads `docker`, `compose`, `gpu`, `container` from the
    status; a native one must still answer them, with what is true here."""
    monkeypatch.setenv("VISOR_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("VISOR_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(ctl, "health", lambda *a, **k: False)
    args = _args(url="http://localhost:1", check_remote=False, branch=None, tools_dir=str(tmp_path))
    status = native.cmd_status(ctl, args)
    for key in ("git", "docker", "compose", "gpu", "clone", "container", "server", "env"):
        assert key in status
    assert status["docker"]["daemon"] is False
    assert status["container"]["running"] is False
    assert status["backend"] == "native"
    json.dumps(status)  # --json prints it


# ----------------------------------------------------------------------
# run-native.sh, run for real with a stub in place of the API interpreter
# ----------------------------------------------------------------------

def _fake_api_venv(state):
    """An api-venv whose python records its argv and environment instead of serving."""
    import hashlib

    binary = state / "api-venv" / "bin"
    binary.mkdir(parents=True)
    python = binary / "python"
    python.write_text('#!/bin/sh\nenv > "$STUB_OUT.env"\nprintf "%s\\n" "$@" > "$STUB_OUT.args"\n')
    python.chmod(0o755)
    reqs = os.path.join(_SCRIPTS, "..", "server", "requirements-api.txt")
    with open(reqs, "rb") as handle:
        (state / "api-venv" / ".requirements.sha").write_text(hashlib.sha256(handle.read()).hexdigest())


def _run_script(tmp_path, **extra):
    state = tmp_path / "state"
    tools = tmp_path / "sadt"
    (tools / "tools").mkdir(parents=True, exist_ok=True)
    if not (state / "api-venv").exists():
        _fake_api_venv(state)
    out = tmp_path / "out"
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("VISOR_", "TORCH_HOME", "TEMP_DIR", "BIND_ADDR", "TLS_", "PUBLISH_ADDR", "SADT_TLS_ACTIVE"))}
    env.update({"VISOR_STATE_DIR": str(state), "UV": "/bin/true", "API_TOKEN": "t",
                "SADT_TOOLS": str(tools), "DATA_DIR": str(tmp_path / "DATA"),
                "STUB_OUT": str(out), "HOME": str(tmp_path)})
    env.update(extra)
    done = subprocess.run(["bash", os.path.join(_SCRIPTS, "run-native.sh")], env=env,
                          capture_output=True, text=True, check=False)
    return done, out, state


def _environment_of(out):
    text = pathlib.Path(str(out) + ".env").read_text()
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def test_run_native_sh_gives_this_instance_its_own_temp_dir_and_torch_home(tmp_path):
    done, out, state = _run_script(tmp_path)
    assert done.returncode == 0, done.stderr
    seen = _environment_of(out)
    assert seen["TEMP_DIR"] == str(state / "tmp")
    assert os.path.isdir(state / "tmp")
    assert seen["TORCH_HOME"] == str(state / "torch_home")
    assert seen["SADT_API"].startswith("https://127.0.0.1:")
    # No certificate supplied: serve.py generates its own, into this instance's state folder.
    assert seen["TLS_DIR"] == str(state / "tls")
    assert "TLS_CERT_FILE" not in seen and "TLS_KEY_FILE" not in seen


def test_a_temp_dir_set_by_the_operator_is_kept(tmp_path):
    done, out, _state = _run_script(tmp_path, TEMP_DIR=str(tmp_path / "mine"))
    assert done.returncode == 0, done.stderr
    assert _environment_of(out)["TEMP_DIR"] == str(tmp_path / "mine")


def test_run_native_sh_starts_serve_py_and_never_uvicorn_directly(tmp_path):
    """The trap conftest.py hides: it sets SADT_TLS_ACTIVE=1, so a server started through
    `uvicorn main:app` passes every test while main.py's lifespan refuses it for real."""
    done, out, _state = _run_script(tmp_path)
    assert done.returncode == 0, done.stderr
    args = pathlib.Path(str(out) + ".args").read_text().split("\n")[:-1]
    assert args == ["serve.py", "--host", "127.0.0.1", "--port", "8000"]
    seen = _environment_of(out)
    # serve.py sets this itself once the listener has its certificate; the script must not.
    assert "SADT_TLS_ACTIVE" not in seen
    assert seen["PUBLISH_ADDR"] == "127.0.0.1"
    text = open(os.path.join(_SCRIPTS, "run-native.sh"), encoding="utf-8").read()
    assert "-m uvicorn" not in text


def test_run_native_sh_hands_the_supplied_tls_files_to_serve_py(tmp_path):
    done, out, _state = _run_script(tmp_path, TLS_CERT_FILE=str(tmp_path / "c"), TLS_KEY_FILE=str(tmp_path / "k"))
    assert done.returncode == 1 and "missing" in done.stderr  # files must exist
    (tmp_path / "c").write_text("x")
    (tmp_path / "k").write_text("x")
    done, out, _state = _run_script(tmp_path, TLS_CERT_FILE=str(tmp_path / "c"), TLS_KEY_FILE=str(tmp_path / "k"))
    assert done.returncode == 0, done.stderr
    seen = _environment_of(out)
    assert seen["TLS_CERT_FILE"] == str(tmp_path / "c") and seen["TLS_KEY_FILE"] == str(tmp_path / "k")
    assert seen["SADT_API"].startswith("https://127.0.0.1:")
    assert "https://127.0.0.1" in done.stderr
    assert "--ssl-keyfile" not in pathlib.Path(str(out) + ".args").read_text()


def test_run_native_sh_refuses_a_certificate_off_loopback_and_half_a_pair(tmp_path):
    (tmp_path / "c").write_text("x")
    (tmp_path / "k").write_text("x")
    both = {"TLS_CERT_FILE": str(tmp_path / "c"), "TLS_KEY_FILE": str(tmp_path / "k")}
    done, _out, _state = _run_script(tmp_path, BIND_ADDR="0.0.0.0", **both)
    assert done.returncode == 1 and "127.0.0.1 only" in done.stderr
    done, _out, _state = _run_script(tmp_path, TLS_CERT_FILE=str(tmp_path / "c"))
    assert done.returncode == 1 and "go together" in done.stderr


# ----------------------------------------------------------------------
# Local TLS in server_ctl_native, with real certificates
# ----------------------------------------------------------------------

@pytest.fixture
def envfile(tmp_path, monkeypatch):
    """The .env of the checkout is not touched: read_env/write_env get a path of their own."""
    path = str(tmp_path / "dot-env")
    read, write = ctl.read_env, ctl.write_env
    monkeypatch.setattr(ctl, "read_env", lambda p=path: read(p))
    monkeypatch.setattr(ctl, "write_env", lambda updates, p=path: write(updates, p))
    return path


def _make_cert(directory, name="cert", san="IP:127.0.0.1,IP:::1,DNS:localhost"):
    if not shutil.which("openssl"):
        pytest.skip("openssl is not installed")
    cert, key = directory / (name + ".pem"), directory / (name + "-key.pem")
    done = subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes",
         "-keyout", str(key), "-out", str(cert), "-days", "2", "-subj", "/CN=VISOR test",
         "-addext", "subjectAltName=" + san], capture_output=True, check=False)
    assert done.returncode == 0, done.stderr
    key.chmod(0o600)
    return str(cert), str(key)


def test_a_key_readable_by_others_is_refused_without_a_word_about_its_content(tmp_path):
    cert, key = _make_cert(tmp_path)
    os.chmod(key, 0o644)
    with pytest.raises(native._Error) as raised:
        native.validate_tls(cert, key)
    assert "readable by other accounts" in str(raised.value)
    os.chmod(key, 0o640)
    with pytest.raises(native._Error):
        native.validate_tls(cert, key)
    os.chmod(key, 0o600)
    assert native.validate_tls(cert, key) == (cert, key)


def test_a_key_that_is_not_the_certificates_is_refused_with_a_generic_message(tmp_path):
    cert, _key = _make_cert(tmp_path)
    _other, other_key = _make_cert(tmp_path, "other")
    with pytest.raises(native._Error) as raised:
        native.validate_tls(cert, other_key)
    assert "BEGIN" not in str(raised.value) and other_key not in str(raised.value)


def test_missing_certificate_and_symlinked_or_absent_keys_are_refused(tmp_path):
    cert, key = _make_cert(tmp_path)
    with pytest.raises(native._Error):
        native.validate_tls(str(tmp_path / "nope.pem"), key)
    with pytest.raises(native._Error):
        native.validate_tls(cert, str(tmp_path / "nope-key.pem"))
    with pytest.raises(native._Error):
        native.validate_tls(cert, str(tmp_path))  # a directory


def test_tls_arguments_go_together_and_none_means_the_servers_own_certificate(tmp_path, envfile):
    cert, key = _make_cert(tmp_path)
    with pytest.raises(ctl.ServerCtlError, match="go together"):
        native.tls_settings(ctl, _args(tls_cert=cert, tls_key=None))
    with pytest.raises(ctl.ServerCtlError, match="go together"):
        native.tls_settings(ctl, _args(tls_cert=None, tls_key=key))
    assert native.tls_settings(ctl, _args(tls_cert=cert, tls_key=key)) == (cert, key)
    assert native.tls_settings(ctl, _args()) is None


def test_the_pair_is_remembered_in_env_under_the_upstream_names(tmp_path, envfile):
    cert, key = _make_cert(tmp_path)
    args = _args(tls_cert=cert, tls_key=key, bind=None)
    tls = native.tls_settings(ctl, args)
    native.remember_tls(ctl, args, tls)
    assert ctl.read_env()["TLS_CERT_FILE"] == cert and ctl.read_env()["TLS_KEY_FILE"] == key
    # no arguments: .env supplies the pair (what `update` relies on)
    assert native.tls_settings(ctl, _args(bind=None)) == (cert, key)
    # a remembered key that became readable by others stops the next start
    os.chmod(key, 0o644)
    with pytest.raises(ctl.ServerCtlError, match="readable by other accounts"):
        native.tls_settings(ctl, _args(bind=None))


def test_the_report_names_the_certificate_and_never_the_key(tmp_path):
    cert, key = _make_cert(tmp_path)
    report = native.tls_report((cert, key))
    assert report == {"enabled": True, "cert": cert}
    assert key not in json.dumps(report)
    # No supplied pair: the certificate serve.py generates, whose path is known before it exists.
    assert native.tls_report(None) == {"enabled": True, "cert": native.generated_cert_path()}


def test_status_says_https_with_the_served_certificate_whatever_the_public_url(tmp_path, monkeypatch, envfile):
    cert, key = _make_cert(tmp_path)
    monkeypatch.setenv("VISOR_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("VISOR_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(ctl, "tailnet_url", lambda: "https://somewhere.ts.net")
    seen = []
    monkeypatch.setattr(ctl, "health", lambda url=None, timeout=5.0, cafile=None: seen.append((url, cafile)) or False)
    args = _args(url=None, check_remote=False, branch=None, tools_dir=str(tmp_path))
    generated = native.cmd_status(ctl, args)
    assert generated["tls"] == {"enabled": True, "cert": native.generated_cert_path()}
    assert generated["url"] == "https://127.0.0.1:%d" % ctl.effective_port()
    assert seen[-1] == (generated["url"], native.generated_cert_path())
    ctl.write_env({"HOST_PORT": "8123", "TLS_CERT_FILE": cert, "TLS_KEY_FILE": key})
    secure = native.cmd_status(ctl, args)
    assert secure["url"] == "https://127.0.0.1:8123"
    assert secure["tls"] == {"enabled": True, "cert": cert}
    assert seen[-1] == ("https://127.0.0.1:8123", cert)
    assert key not in json.dumps(secure)


def test_the_pinned_probe_context_is_the_servers_loopback_context(tmp_path):
    import tls as server_tls

    cert, _key = _make_cert(tmp_path)
    mine, theirs = ctl.pinned_context(cert), server_tls.loopback_context(cert)
    assert (mine.check_hostname, mine.verify_mode, mine.verify_flags) == \
           (theirs.check_hostname, theirs.verify_mode, theirs.verify_flags)


def _free_port():
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("supplied", [True, False], ids=["supplied-pair", "generated-certificate"])
def test_a_real_serve_py_answers_https_to_its_own_certificate_only(tmp_path, monkeypatch, envfile, supplied):
    """up --native through run-native.sh, a real serve.py, the real main:app.

    /health with verify=<cert> passes; with another certificate it fails; in
    plain http it fails; and the health check of server_ctl itself goes through
    the same pinned file. The certificate is the supplied one, or the one
    serve.py generated, and the report names the file really served.
    """
    pytest.importorskip("uvicorn")
    requests = pytest.importorskip("requests")
    other, _other_key = _make_cert(tmp_path, "other", san="IP:127.0.0.1")
    cert = key = None
    if supplied:
        cert, key = _make_cert(tmp_path)
    state = tmp_path / "state"
    sadt = tmp_path / "sadt"
    (sadt / "tools").mkdir(parents=True)
    _fake_api_venv(state)
    python = state / "api-venv" / "bin" / "python"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    monkeypatch.setenv("VISOR_STATE_DIR", str(state))
    monkeypatch.setenv("VISOR_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("UV", "/bin/true")
    monkeypatch.setattr(native.native_tools, "find_uv", lambda: "/bin/true")
    monkeypatch.setattr(ctl, "ensure_data_dir", lambda: str(tmp_path / "DATA"))
    monkeypatch.setattr(ctl, "data_dir", lambda: str(tmp_path / "DATA"))
    port = _free_port()
    args = _args(port=port, url=None, token="t" * 40, bind=None, wait=True, timeout=90, device="cpu",
                 tools_dir=str(sadt), tls_cert=cert, tls_key=key)
    result = None
    try:
        result = native.cmd_up(ctl, args)
        assert result["healthy"] is True
        assert result["url"] == f"https://127.0.0.1:{port}"
        served = cert or str(state / "tls" / "cert.pem")
        assert result["tls"] == {"enabled": True, "cert": served}
        assert os.path.isfile(served)
        if key:
            assert key not in json.dumps(result)
        url = result["url"] + "/health"
        assert requests.get(url, verify=served, timeout=10).json() == {"status": "ok"}
        with pytest.raises(requests.exceptions.SSLError):
            requests.get(url, verify=other, timeout=10)
        with pytest.raises(requests.exceptions.RequestException):
            requests.get(f"http://127.0.0.1:{port}/health", timeout=10)
        assert ctl.health(result["url"], cafile=served) is True
        assert ctl.health(result["url"], cafile=other) is False
        assert ctl.health(result["url"]) is False  # the system store does not know it
        if key:
            assert key not in pathlib.Path(native.log_path()).read_text(encoding="utf-8")
    finally:
        if result:
            native.cmd_down(ctl, _args())


def test_a_changed_certificate_makes_update_relaunch(tmp_path, monkeypatch, envfile):
    cert, key = _make_cert(tmp_path)
    new_cert, new_key = _make_cert(tmp_path, "new")
    monkeypatch.setenv("VISOR_STATE_DIR", str(tmp_path / "state"))
    sadt = tmp_path / "sadt"
    (sadt / "tools").mkdir(parents=True)
    os.makedirs(native.state_dir())
    live = {"pid": os.getpid(), "port": 8000, "tls_sha": native.cert_digest((cert, key))}
    monkeypatch.setattr(native, "running_server", lambda: live)
    calls = []
    monkeypatch.setattr(native, "cmd_down", lambda c, a: calls.append("down"))
    monkeypatch.setattr(native, "cmd_up", lambda c, a: calls.append("up") or {"healthy": True})
    monkeypatch.setattr(native.native_tools, "run", lambda *a, **k: {"tools": [], "failed": []})
    monkeypatch.setattr(ctl, "clone_status", lambda **k: {"is_git_repo": False, "checked_remote": False,
                                                         "error": None, "dirty": False, "behind": 0})
    base = {"url": None, "branch": None, "tools_dir": str(sadt), "force": False, "tool": None,
            "bind": None}
    native.cmd_update(ctl, _args(tls_cert=cert, tls_key=key, **base))
    assert calls == []  # same certificate: nothing to relaunch
    result = native.cmd_update(ctl, _args(tls_cert=new_cert, tls_key=new_key, **base))
    assert calls == ["down", "up"] and result["restarted"] is True
    assert result["tls"]["cert"] == new_cert and result["url"].startswith("https://127.0.0.1:")


# ----------------------------------------------------------------------
# The recipe: dependencies the server insists on, and the staged backbone
# ----------------------------------------------------------------------

def test_asking_for_areg_ios_adds_aso_then_ali_cbct():
    config = _config()
    found = {name: "/x/" + name for name in ("AREG_IOS", "ASO", "ALI_CBCT", "ALI_IOS")}
    assert tools.with_requirements(["AREG_IOS"], config, found) == ["AREG_IOS", "ASO", "ALI_CBCT"]
    assert tools.with_requirements(["ALI_IOS"], config, found) == ["ALI_IOS"]
    assert tools.with_requirements(["ASO", "AREG_IOS"], config, found) == ["ASO", "ALI_CBCT", "AREG_IOS"]
    with pytest.raises(tools.RecipeError, match="ALI_CBCT"):
        tools.with_requirements(["ASO"], config, {"ASO": "/x"})


def test_the_backbone_is_downloaded_checked_and_only_then_put_in_place(tmp_path):
    import hashlib

    payload = b"not really a network"
    source = tmp_path / "src.pth"
    source.write_bytes(payload)
    config = {"backbones": {"DOCShapeAXI": {"file": "b.pth", "url": source.as_uri(),
                                            "sha256": hashlib.sha256(payload).hexdigest()}}}
    home = str(tmp_path / "home")
    report = tools.stage_backbones(["DOCShapeAXI", "ALI_IOS"], home, config)
    target = os.path.join(home, "hub", "checkpoints", "b.pth")
    assert [r["action"] for r in report] == ["staged"] and pathlib.Path(target).read_bytes() == payload
    assert tools.stage_backbones(["DOCShapeAXI"], home, config)[0]["action"] == "kept"
    # a corrupted copy is replaced
    pathlib.Path(target).write_bytes(b"junk")
    assert tools.stage_backbones(["DOCShapeAXI"], home, config)[0]["action"] == "staged"
    # a download with the wrong digest never reaches the real name
    config["backbones"]["DOCShapeAXI"]["sha256"] = "0" * 64
    os.unlink(target)
    failed = tools.stage_backbones(["DOCShapeAXI"], home, config)[0]
    assert failed["action"] == "failed" and not os.path.exists(target) and not os.path.exists(target + ".part")
    # a dry run downloads nothing
    assert tools.stage_backbones(["DOCShapeAXI"], str(tmp_path / "dry"), config, dry_run=True)[0]["action"] == "plan"
    assert not os.path.exists(tmp_path / "dry")


def test_the_recipes_backbone_is_the_published_efficientnet_b0():
    spec = _config()["backbones"]["DOCShapeAXI"]
    assert spec["file"] == "efficientnet-b0-355c32eb.pth"
    assert spec["sha256"].startswith("355c32eb"), "torch's file names carry the first 8 digits of the digest"
    assert spec["url"].startswith("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/")


# ----------------------------------------------------------------------
# Audit fixes: loopback only, down kills only its own server
# ----------------------------------------------------------------------

def test_run_native_sh_refuses_any_bind_but_loopback_even_without_a_certificate(tmp_path):
    for bind in ("0.0.0.0", "192.168.1.5", "::"):
        done, out, _state = _run_script(tmp_path, BIND_ADDR=bind)
        assert done.returncode == 1, bind
        assert "127.0.0.1 only" in done.stderr
        assert not pathlib.Path(str(out) + ".args").exists(), "uvicorn was started"
    done, _out, _state = _run_script(tmp_path, BIND_ADDR="127.0.0.1")
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("argv", [["--bind", "0.0.0.0"], ["--bind", "10.0.0.2"], ["--bind", ""]])
def test_up_native_refuses_a_non_loopback_bind_and_starts_nothing(tmp_path, monkeypatch, envfile, argv):
    monkeypatch.setenv("VISOR_STATE_DIR", str(tmp_path / "state"))
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: started.append(a))
    args = _args(bind=argv[1], tls_cert=None, tls_key=None, port=None, token=None)
    with pytest.raises(ctl.ServerCtlError, match="127.0.0.1 only"):
        native.cmd_up(ctl, args)
    assert started == []
    assert not os.path.exists(envfile), ".env was written before the refusal"


def test_up_native_refuses_a_bind_addr_left_in_env_by_a_docker_deployment(tmp_path, monkeypatch, envfile):
    monkeypatch.setenv("VISOR_STATE_DIR", str(tmp_path / "state"))
    pathlib.Path(envfile).write_text("BIND_ADDR=100.64.0.7\n")
    args = _args(bind=None, tls_cert=None, tls_key=None, port=None, token=None)
    with pytest.raises(ctl.ServerCtlError, match="127.0.0.1 only"):
        native.cmd_up(ctl, args)


def _decoy(command_words, tmp_path):
    """A process whose argv[0] imitates a uvicorn command line."""
    return subprocess.Popen(
        ["bash", "-c", "exec -a \"$0\" sleep 90", command_words], start_new_session=True)


def _write_pidfile(pid, **extra):
    info = {"pid": pid, "port": 18765}
    info.update(extra)
    with open(native.pidfile_path(), "w") as handle:
        json.dump(info, handle)


@pytest.mark.skipif(shutil.which("ps") is None, reason="needs ps")
def test_down_never_kills_a_process_that_only_looks_like_a_uvicorn(tmp_path, monkeypatch):
    monkeypatch.setenv("VISOR_STATE_DIR", str(tmp_path / "state"))
    os.makedirs(native.state_dir())
    # Same words as the old check wanted, but another state folder and another port.
    decoy = _decoy("/usr/bin/python3 -m uvicorn main:app --host 127.0.0.1 --port 9999 DECOY", tmp_path)
    try:
        time.sleep(0.3)
        _write_pidfile(decoy.pid)
        assert native.running_server() is None
        outcome = native.cmd_down(ctl, _args())
        assert outcome["was_running"] is False
        assert decoy.poll() is None, "down killed a foreign process"
        assert not os.path.exists(native.pidfile_path()), "the stale pidfile was not cleaned"
    finally:
        decoy.kill()
        decoy.wait()


@pytest.mark.skipif(shutil.which("ps") is None, reason="needs ps")
def test_a_reused_pid_with_the_right_words_but_another_start_time_is_not_ours(tmp_path, monkeypatch):
    monkeypatch.setenv("VISOR_STATE_DIR", str(tmp_path / "state"))
    os.makedirs(native.state_dir())
    words = f"{native.state_dir()}/api-venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 18765"
    ours = _decoy(words, tmp_path)
    try:
        time.sleep(0.3)
        _write_pidfile(ours.pid, proc_start=native.process_start_time(ours.pid))
        assert native.running_server() is not None  # the real thing is recognised
        _write_pidfile(ours.pid, proc_start="Mon Jan  1 00:00:00 2001")
        assert native.running_server() is None
        native.cmd_down(ctl, _args())
        assert ours.poll() is None
        # The right instance but another port: not ours either.
        _write_pidfile(ours.pid, port=1234)
        assert native.running_server() is None
        # And without a recorded start time (an older pidfile) the command line decides.
        _write_pidfile(ours.pid)
        assert native.running_server() is not None
        outcome = native.cmd_down(ctl, _args())
        assert outcome["was_running"] is True
        ours.wait(timeout=10)
    finally:
        try:
            os.killpg(ours.pid, signal.SIGKILL)
        except OSError:
            pass
