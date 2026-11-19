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


def _tool_with_common(tmp_path):
    folder = _make_tool(tmp_path, "AREG_CBCT")
    common = tmp_path / "tools" / "common"
    (common / "sadt_areg_common").mkdir(parents=True)
    (common / "sadt_areg_common" / "__init__.py").write_text("X = 1\n")
    return folder, str(common)


def test_the_fingerprint_moves_when_a_path_package_source_does(tmp_path):
    folder, common = _tool_with_common(tmp_path)
    lock = ["numpy==2.3.2", common]
    plan = lambda: tools.plan_for("AREG_CBCT", folder, _config(), lock)  # noqa: E731
    before = tools.fingerprint(plan(), folder)
    assert tools.fingerprint(plan(), folder) == before
    # Bytecode and caches are not source.
    os.makedirs(os.path.join(common, "sadt_areg_common", "__pycache__"))
    with open(os.path.join(common, "sadt_areg_common", "__pycache__", "a.pyc"), "wb") as handle:
        handle.write(b"x")
    assert tools.fingerprint(plan(), folder) == before
    with open(os.path.join(common, "sadt_areg_common", "__init__.py"), "a") as handle:
        handle.write("def patient_folders(): ...\n")
    changed = tools.fingerprint(plan(), folder)
    assert changed != before
    with open(os.path.join(common, "sadt_areg_common", "new.py"), "w") as handle:
        handle.write("")
    assert tools.fingerprint(plan(), folder) != changed


def test_an_editable_path_package_is_not_hashed(tmp_path):
    folder, common = _tool_with_common(tmp_path)
    plan = tools.plan_for("AREG_CBCT", folder, _config(), ["-e " + common])
    before = tools.fingerprint(plan, folder)
    with open(os.path.join(common, "sadt_areg_common", "__init__.py"), "a") as handle:
        handle.write("Y = 2\n")
    assert tools.fingerprint(plan, folder) == before


def test_a_path_package_is_reinstalled_from_source_never_from_the_uv_cache(tmp_path, monkeypatch):
    folder, common = _tool_with_common(tmp_path)
    plan = tools.plan_for("AREG_CBCT", folder, _config(), ["numpy==2.3.2", common])
    calls = []
    monkeypatch.setattr(tools, "_run", lambda command, cwd=None, env=None, attempts=2: calls.append(list(command)))
    monkeypatch.setattr(tools, "verify_one", lambda uv, plan, config=None: {"pip_check_ok": False})
    tools.install_one("uv", plan, force=True)
    installs = [c for c in calls if c[1:3] == ["pip", "install"]]
    local = [c for c in installs if common in c]
    assert len(local) == 1
    assert "--no-cache" in local[0] and "--reinstall" in local[0] and "--no-deps" in local[0]
    # And the path package is not also sent through the cached requirements file.
    assert all(common not in c for c in installs if c is not local[0])


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


def _update_world(tmp_path, monkeypatch, envfile):
    """A live server, a tools checkout, and recorders for what update does to the process."""
    monkeypatch.setenv("VISOR_STATE_DIR", str(tmp_path / "state"))
    sadt = tmp_path / "sadt"
    (sadt / "tools").mkdir(parents=True)
    os.makedirs(native.state_dir())
    live = {"pid": os.getpid(), "port": 8000, "tls_sha": None}
    calls = []
    monkeypatch.setattr(native, "running_server", lambda: None if "down" in calls else live)
    monkeypatch.setattr(native.native_tools, "find_uv", lambda: "/usr/bin/uv")
    monkeypatch.setattr(native.native_tools, "run", lambda *a, **k: {"tools": [], "failed": []})
    monkeypatch.setattr(ctl, "clone_status", lambda **k: {"is_git_repo": False, "checked_remote": False,
                                                         "error": None, "dirty": False, "behind": 0})
    monkeypatch.setattr(native, "cmd_down", lambda c, a: calls.append("down"))
    # Whatever listens on 8000 of the machine running the tests is not the subject here.
    monkeypatch.setattr(ctl, "port_in_use", lambda port: False)
    # main() imports the backend by name: hand it the module under test.
    monkeypatch.setitem(sys.modules, "server_ctl_native", native)
    # ...and main() reaches its own module through sys.modules[__name__], which another
    # test file loading server_ctl under the same alias may have replaced.
    monkeypatch.setitem(sys.modules, ctl.__name__, ctl)
    base = {"url": None, "branch": None, "tools_dir": str(sadt), "force": True, "tool": None,
            "bind": None, "tls_cert": None, "tls_key": None}
    return calls, base


def _fake_launch(monkeypatch, started):
    """Only the `bash run-native.sh` launch is faked; `ps` and the rest still run."""
    real = subprocess.Popen

    def popen(command, *a, **k):
        if command and command[0] == "bash":
            started.append(command)
            return types.SimpleNamespace(pid=os.getpid())
        return real(command, *a, **k)

    monkeypatch.setattr(subprocess, "Popen", popen)


def test_update_native_through_the_real_parser_relaunches_without_a_token_argument(tmp_path, monkeypatch, envfile):
    """The defect measured on the Mac: AttributeError 'token' after the old server was stopped."""
    calls, _base = _update_world(tmp_path, monkeypatch, envfile)
    started = []
    _fake_launch(monkeypatch, started)
    monkeypatch.setattr(native, "_wait_for_health", lambda *a, **k: True)
    sadt = str(tmp_path / "sadt")
    code = ctl.main(["--json", "update", "--native", "--force", "--tools-dir", sadt])
    assert code == 0 and calls == ["down"] and len(started) == 1


def test_update_takes_a_token_like_up_and_keeps_the_existing_one_without(tmp_path, monkeypatch, envfile):
    parsed = ctl.build_parser().parse_args(["update", "--native", "--token", "abc"])
    assert parsed.token == "abc"  # the parser takes it; native.check_token judges it
    assert ctl.build_parser().parse_args(["update", "--native"]).token is None
    calls, _base = _update_world(tmp_path, monkeypatch, envfile)
    _fake_launch(monkeypatch, [])
    monkeypatch.setattr(native, "_wait_for_health", lambda *a, **k: True)
    sadt = str(tmp_path / "sadt")
    assert ctl.main(["update", "--native", "--force", "--tools-dir", sadt, "--token", "from-update" + "x" * 20]) == 0
    assert ctl.read_env()["API_TOKEN"] == "from-update" + "x" * 20
    assert ctl.main(["update", "--native", "--force", "--tools-dir", sadt]) == 0
    assert ctl.read_env()["API_TOKEN"] == "from-update" + "x" * 20


def test_update_docker_refuses_a_token_it_would_ignore(monkeypatch):
    monkeypatch.setattr(ctl, "cmd_update", lambda a: pytest.fail("ran"))
    assert ctl.main(["update", "--token", "x"]) == 1


@pytest.mark.parametrize("what", ["non-loopback bind", "no uv"])
def test_update_refuses_a_relaunch_that_cannot_happen_before_stopping_the_old_server(
        tmp_path, monkeypatch, envfile, what):
    calls, base = _update_world(tmp_path, monkeypatch, envfile)
    if what == "non-loopback bind":
        base["bind"] = "0.0.0.0"
    elif what == "no uv":
        monkeypatch.setattr(native.native_tools, "find_uv", lambda: None)
    monkeypatch.setattr(native, "cmd_up", lambda c, a: pytest.fail("started"))
    with pytest.raises(ctl.ServerCtlError):
        native.cmd_update(ctl, _args(**base))
    assert calls == [], "the old server was stopped for a start that was refused"


def test_update_says_clearly_when_the_old_server_is_stopped_and_the_new_one_cannot_start(
        tmp_path, monkeypatch, envfile):
    calls, base = _update_world(tmp_path, monkeypatch, envfile)

    def boom(c, a):
        raise OSError("disk full")

    monkeypatch.setattr(native, "cmd_up", boom)
    with pytest.raises(ctl.ServerCtlError, match=r"old native server was stopped.*disk full.*up --native"):
        native.cmd_update(ctl, _args(**base))
    assert calls == ["down"]


# ----------------------------------------------------------------------
# Putting the managed tools clone on the requested branch
# ----------------------------------------------------------------------

def _git(cwd, *argv):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t", GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
    done = subprocess.run(["git", "-C", str(cwd)] + list(argv), capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def _commit(repo, name, text="x"):
    (pathlib.Path(repo) / name).write_text(text)
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", name)


@pytest.fixture
def tools_remote(tmp_path):
    """origin with main and feat/x; a clone of it at tmp/clone (the managed checkout, on main)."""
    if not shutil.which("git"):
        pytest.skip("git is not installed")
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "-q", "-b", "main")
    _commit(seed, "a")
    _git(seed, "checkout", "-q", "-b", "feat/x")
    _commit(seed, "b")
    _git(seed, "checkout", "-q", "main")
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(seed), str(origin)], check=True)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    (clone / "tools").mkdir()
    return types.SimpleNamespace(seed=seed, origin=origin, clone=clone)


def _switch(repo, branch, url=None):
    return native.switch_tools_branch(ctl, str(repo.clone), branch, url)


def _branch(repo):
    return _git(repo.clone, "rev-parse", "--abbrev-ref", "HEAD")


def test_the_tools_clone_is_switched_to_the_requested_branch_and_tracks_it(tools_remote):
    assert _branch(tools_remote) == "main"
    assert _switch(tools_remote, "feat/x") is True
    assert _branch(tools_remote) == "feat/x"
    assert _git(tools_remote.clone, "rev-parse", "--abbrev-ref", "@{u}") == "origin/feat/x"
    assert (tools_remote.clone / "b").exists()
    assert _switch(tools_remote, "feat/x") is False  # already there
    _git(tools_remote.clone, "rev-parse", "--verify", "main")  # the branch left is kept


def test_a_branch_that_moved_upstream_is_fast_forwarded(tools_remote):
    _switch(tools_remote, "feat/x")
    _git(tools_remote.seed, "checkout", "-q", "feat/x")
    _commit(tools_remote.seed, "c")
    _git(tools_remote.seed, "push", "-q", str(tools_remote.origin), "feat/x")
    assert _switch(tools_remote, "feat/x") is True
    assert (tools_remote.clone / "c").exists()


def test_a_dirty_tree_is_never_switched(tools_remote):
    (tools_remote.clone / "a").write_text("edited")
    with pytest.raises(ctl.ServerCtlError, match="uncommitted"):
        _switch(tools_remote, "feat/x")
    assert _branch(tools_remote) == "main" and (tools_remote.clone / "a").read_text() == "edited"


def test_a_local_branch_with_commits_the_remote_lacks_is_not_touched(tools_remote):
    _git(tools_remote.clone, "switch", "-q", "-c", "feat/x", "origin/main")
    _commit(tools_remote.clone, "mine")
    local = _git(tools_remote.clone, "rev-parse", "HEAD")
    _git(tools_remote.clone, "switch", "-q", "main")
    with pytest.raises(ctl.ServerCtlError, match="commits that .* lacks"):
        _switch(tools_remote, "feat/x")
    assert _branch(tools_remote) == "main"
    assert _git(tools_remote.clone, "rev-parse", "feat/x") == local
    # Already on it and diverged: refused too.
    _git(tools_remote.clone, "switch", "-q", "feat/x")
    with pytest.raises(ctl.ServerCtlError, match="diverged"):
        _switch(tools_remote, "feat/x")


def test_a_branch_missing_at_the_remote_is_refused_and_nothing_moves(tools_remote):
    with pytest.raises(ctl.ServerCtlError, match="could not fetch|does not exist"):
        _switch(tools_remote, "feat/nope")
    assert _branch(tools_remote) == "main"


def test_a_detached_head_holding_commits_no_branch_has_is_not_left(tools_remote):
    _git(tools_remote.clone, "switch", "-q", "--detach")
    _commit(tools_remote.clone, "orphan")
    with pytest.raises(ctl.ServerCtlError, match="detached HEAD"):
        _switch(tools_remote, "feat/x")


def test_a_checkout_nested_in_another_repository_is_never_switched(tools_remote, tmp_path):
    nested = tools_remote.clone / "inner"
    nested.mkdir()
    with pytest.raises(ctl.ServerCtlError, match="not the top folder"):
        native.switch_tools_branch(ctl, str(nested), "feat/x")
    assert _branch(tools_remote) == "main"


@pytest.mark.parametrize("branch,url", [("-x", None), ("a..b", None), ("main", "ext::sh -c id"),
                                        ("main", "-oProxyCommand=id"), ("main", "file:///etc"),
                                        ("main", "/some/local/path")])
def test_unusable_branch_names_and_urls_are_refused_before_any_git_call(tools_remote, monkeypatch, branch, url):
    real = ctl._capture

    def only_the_name_check(command, **kwargs):
        if command[3] != "check-ref-format":
            pytest.fail("git ran: %r" % (command,))
        return real(command, **kwargs)

    monkeypatch.setattr(ctl, "_capture", only_the_name_check)
    with pytest.raises(ctl.ServerCtlError):
        _switch(tools_remote, branch, url)


def test_another_repository_is_fetched_into_its_own_ref_and_origin_is_left_alone(tools_remote, tmp_path, monkeypatch):
    other = tmp_path / "other.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(tools_remote.seed), str(other)], check=True)
    _git(tools_remote.seed, "checkout", "-q", "-b", "only-there")
    _commit(tools_remote.seed, "z")
    _git(tools_remote.seed, "push", "-q", str(other), "only-there")
    monkeypatch.setattr(native, "_URL_PREFIXES", ("/",))  # a local path stands in for https in the test
    before = _git(tools_remote.clone, "remote", "get-url", "origin")
    assert _switch(tools_remote, "only-there", str(other)) is True
    assert _branch(tools_remote) == "only-there" and (tools_remote.clone / "z").exists()
    assert _git(tools_remote.clone, "remote", "get-url", "origin") == before
    done = subprocess.run(["git", "-C", str(tools_remote.clone), "config", "--get", "branch.only-there.remote"],
                          capture_output=True, text=True)
    assert done.stdout.strip() == "", "a branch from another URL must not track origin"


def test_update_refuses_a_tools_url_without_a_branch_and_a_switch_before_stopping_the_old_server(
        tmp_path, monkeypatch, envfile, tools_remote):
    calls, base = _update_world(tmp_path, monkeypatch, envfile)
    base["tools_dir"] = str(tools_remote.clone)
    monkeypatch.setattr(native, "cmd_up", lambda c, a: pytest.fail("started"))
    with pytest.raises(ctl.ServerCtlError, match="needs --tools-branch"):
        native.cmd_update(ctl, _args(tools_url="https://example.org/x.git", tools_branch=None, **base))
    (tools_remote.clone / "a").write_text("edited")
    with pytest.raises(ctl.ServerCtlError, match="uncommitted"):
        native.cmd_update(ctl, _args(tools_url=None, tools_branch="feat/x", **base))
    assert calls == []


def test_update_switches_the_tools_clone_then_goes_on(tmp_path, monkeypatch, envfile, tools_remote):
    calls, base = _update_world(tmp_path, monkeypatch, envfile)
    base["tools_dir"] = str(tools_remote.clone)
    monkeypatch.setattr(native, "cmd_up", lambda c, a: calls.append("up") or {"healthy": True})
    result = native.cmd_update(ctl, _args(tools_url=None, tools_branch="feat/x", **base))
    assert result["tools_switched"] is True and _branch(tools_remote) == "feat/x"
    assert calls == ["down", "up"]


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


@pytest.mark.skipif(shutil.which("ps") is None, reason="needs ps")
def test_the_serve_py_command_line_run_native_sh_ends_in_is_ours_and_nothing_wider_is(tmp_path, monkeypatch):
    # The state folder has a space in it on a Mac ("Application Support").
    monkeypatch.setenv("VISOR_STATE_DIR", str(tmp_path / "Application Support" / "VISOR" / "native"))
    os.makedirs(native.state_dir())
    real = f"{native.state_dir()}/api-venv/bin/python serve.py --host 127.0.0.1 --port 18765"
    ours = _decoy(real, tmp_path)
    other_venv = _decoy("/usr/bin/python3 serve.py --host 127.0.0.1 --port 18765", tmp_path)
    other_script = _decoy(f"{native.state_dir()}/api-venv/bin/python other.py --port 18765", tmp_path)
    other_port = _decoy(f"{native.state_dir()}/api-venv/bin/python serve.py --host 127.0.0.1 --port 187650", tmp_path)
    try:
        time.sleep(0.3)
        _write_pidfile(ours.pid, proc_start=native.process_start_time(ours.pid))
        assert native.running_server() is not None
        for decoy in (other_venv, other_script, other_port):
            _write_pidfile(decoy.pid)
            assert native.running_server() is None, "recognised a process that is not this instance's server"
            native.cmd_down(ctl, _args())
            assert decoy.poll() is None, "down killed a foreign process"
        _write_pidfile(ours.pid)
        assert native.cmd_down(ctl, _args())["was_running"] is True
        ours.wait(timeout=10)
    finally:
        for proc in (ours, other_venv, other_script, other_port):
            proc.kill()
            proc.wait()


# ----------------------------------------------------------------------
# The health probe on a TLS stack without VERIFY_X509_PARTIAL_CHAIN (macOS /usr/bin/python3)
# ----------------------------------------------------------------------

def _stack_without_the_flag(monkeypatch):
    """LibreSSL 2.8 (macOS /usr/bin/python3): no flag, and a pinned self-signed leaf anchors no chain.

    OpenSSL 3 would accept that leaf on its own, so the second half is simulated: a
    context built from a cafile on that stack trusts nothing it was handed.
    """
    import ssl as _ssl

    monkeypatch.delattr(_ssl, "VERIFY_X509_PARTIAL_CHAIN", raising=False)
    monkeypatch.setattr(ctl, "pinned_context", lambda cafile: _ssl.create_default_context(cafile=os.devnull))


def _https_health_server(cert, key):
    import http.server
    import ssl as _ssl
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = b'{"status": "ok"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    context = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, "https://127.0.0.1:%d" % server.server_address[1]


@pytest.mark.parametrize("flag", [True, False], ids=["with-partial-chain", "without-partial-chain"])
def test_the_probe_accepts_the_pinned_certificate_and_refuses_any_other(tmp_path, monkeypatch, flag):
    import ssl as _ssl

    if flag and not hasattr(_ssl, "VERIFY_X509_PARTIAL_CHAIN"):
        pytest.skip("this interpreter has no such flag")
    if not flag:
        _stack_without_the_flag(monkeypatch)
    cert, key = _make_cert(tmp_path, "served")
    other, _other_key = _make_cert(tmp_path, "other")
    server, url = _https_health_server(cert, key)
    try:
        assert ctl.health(url, cafile=cert) is True
        assert ctl.health(url, cafile=other) is False, "a certificate that is not the served one was accepted"
    finally:
        server.shutdown()


def test_without_the_flag_a_server_with_another_certificate_is_refused_even_for_the_same_name(tmp_path, monkeypatch):
    _stack_without_the_flag(monkeypatch)
    pinned, _k = _make_cert(tmp_path, "pinned")
    impostor, impostor_key = _make_cert(tmp_path, "impostor")  # same subject and names, other key
    server, url = _https_health_server(impostor, impostor_key)
    try:
        assert ctl.health(url, cafile=pinned) is False
    finally:
        server.shutdown()


# ----------------------------------------------------------------------
# Audit follow-ups: token rotation, ignored files, ps width, token and URL hygiene
# ----------------------------------------------------------------------

def test_update_with_a_new_token_writes_it_and_relaunches_without_force(tmp_path, monkeypatch, envfile):
    calls, base = _update_world(tmp_path, monkeypatch, envfile)
    base["force"] = False
    ctl.write_env({"API_TOKEN": "OLD" * 10})
    monkeypatch.setattr(native, "cmd_up", lambda c, a: calls.append("up") or {"healthy": True})
    new = "N" * 30
    quiet = native.cmd_update(ctl, _args(token=None, **base))
    assert quiet["restarted"] is False and calls == []
    result = native.cmd_update(ctl, _args(token=new, **base))
    assert ctl.read_env()["API_TOKEN"] == new
    assert result["restarted"] is True and calls == ["down", "up"]


def test_update_with_a_new_token_and_no_live_server_still_stores_it(tmp_path, monkeypatch, envfile):
    calls, base = _update_world(tmp_path, monkeypatch, envfile)
    base["force"] = False
    monkeypatch.setattr(native, "running_server", lambda: None)
    native.cmd_update(ctl, _args(token="T" * 30, **base))
    assert ctl.read_env()["API_TOKEN"] == "T" * 30 and calls == []


@pytest.mark.parametrize("token", ["abc", "a" * 23, "a" * 30 + "\nSADT_TOOLS=/evil", "a" * 30 + " b", "a" * 29 + "\x00"])
def test_a_bad_token_is_refused_before_the_old_server_is_stopped_or_env_written(tmp_path, monkeypatch, envfile, token):
    calls, base = _update_world(tmp_path, monkeypatch, envfile)
    with pytest.raises(ctl.ServerCtlError, match="--token must be") as raised:
        native.cmd_update(ctl, _args(token=token, **base))
    assert token not in str(raised.value)
    assert calls == [] and not os.path.exists(envfile)
    with pytest.raises(ctl.ServerCtlError, match="--token must be"):
        native.cmd_up(ctl, _args(token=token, bind=None, tls_cert=None, tls_key=None, port=None))
    assert not os.path.exists(envfile)


def test_a_file_git_ignores_is_never_overwritten_by_a_switch_or_a_pull(tools_remote):
    seed, clone = tools_remote.seed, tools_remote.clone
    _git(clone, "config", "core.excludesFile", os.devnull)
    (clone / ".gitignore").write_text("weights/\n")
    _git(clone, "add", ".gitignore")
    _git(clone, "commit", "-q", "-m", "ignore weights")
    _git(clone, "push", "-q", "origin", "main")
    (clone / "weights").mkdir()
    (clone / "weights" / "model.bin").write_text("MINE")
    # A branch that tracks that very path.
    _git(seed, "pull", "-q", str(tools_remote.origin), "main")
    _git(seed, "checkout", "-q", "-b", "hostile")
    (seed / "weights").mkdir()
    (seed / "weights" / "model.bin").write_text("THEIRS")
    _git(seed, "add", "-f", "weights/model.bin")
    _git(seed, "commit", "-q", "-m", "tracks weights")
    _git(seed, "push", "-q", str(tools_remote.origin), "hostile")
    with pytest.raises(ctl.ServerCtlError, match="overwritten|git switch refused"):
        _switch(tools_remote, "hostile")
    assert (clone / "weights" / "model.bin").read_text() == "MINE" and _branch(tools_remote) == "main"
    # The pull path: main moves to track it too.
    _git(seed, "checkout", "-q", "main")
    (seed / "weights").mkdir(exist_ok=True)
    (seed / "weights" / "model.bin").write_text("THEIRS")
    _git(seed, "add", "-f", "weights/model.bin")
    _git(seed, "commit", "-q", "-m", "main tracks weights")
    _git(seed, "push", "-q", str(tools_remote.origin), "main")
    assert native._pull_tools(ctl, str(clone)) is False
    assert (clone / "weights" / "model.bin").read_text() == "MINE"


def test_a_git_without_the_option_is_refused_with_the_version_it_needs(tools_remote, monkeypatch):
    real = ctl._capture

    def old_git(command, **kwargs):
        if "--no-overwrite-ignore" in command:
            return 129, "", "error: unknown option `no-overwrite-ignore'"
        return real(command, **kwargs)

    monkeypatch.setattr(ctl, "_capture", old_git)
    with pytest.raises(ctl.ServerCtlError, match="git 2.23 or newer"):
        _switch(tools_remote, "feat/x")
    assert _branch(tools_remote) == "main"


@pytest.mark.skipif(shutil.which("ps") is None, reason="needs ps")
def test_the_server_is_recognised_even_when_columns_would_truncate_ps(tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("VISOR_STATE_DIR", str(tmp_path / "Application Support" / "VISOR" / "native"))
    os.makedirs(native.state_dir())
    words = f"{native.state_dir()}/api-venv/bin/python serve.py --host 127.0.0.1 --port 18765"
    assert len(words) > 80
    ours = _decoy(words, tmp_path)
    try:
        time.sleep(0.3)
        _write_pidfile(ours.pid, proc_start=native.process_start_time(ours.pid))
        assert native.running_server() is not None
        assert native.cmd_down(ctl, _args())["was_running"] is True
        ours.wait(timeout=10)
    finally:
        ours.kill()
        ours.wait()


def test_credentials_in_a_tools_url_never_reach_a_message(tools_remote):
    with pytest.raises(ctl.ServerCtlError) as raised:
        _switch(tools_remote, "main", "https://user:SECRET@127.0.0.1:1/x.git")
    assert "SECRET" not in str(raised.value) and "could not fetch" in str(raised.value)
    with pytest.raises(ctl.ServerCtlError) as raised:
        _switch(tools_remote, "main", "ftp://user:SECRET@host/x.git")
    assert "SECRET" not in str(raised.value)


# ----------------------------------------------------------------------
# `up` is idempotent over our own server, as Docker's is
# ----------------------------------------------------------------------

@pytest.fixture
def up_world(tmp_path, monkeypatch, envfile):
    """`up --native` with a launcher that starts a real process wearing run-native.sh's command line."""
    if shutil.which("ps") is None:
        pytest.skip("needs ps")
    monkeypatch.setenv("VISOR_STATE_DIR", str(tmp_path / "Application Support" / "native"))
    os.makedirs(native.state_dir())
    sadt = tmp_path / "sadt"
    (sadt / "tools").mkdir(parents=True)
    spawned = []
    health = {"answers": True}
    real = subprocess.Popen

    def popen(command, *a, **k):
        if command and command[0] == "bash" and len(command) == 2:
            if world.fail_launch:
                raise OSError("no fork")
            port = k["env"]["HOST_PORT"]
            proc = _decoy(f"{native.state_dir()}/api-venv/bin/python serve.py --host 127.0.0.1 --port {port}", tmp_path)
            time.sleep(0.3)
            spawned.append(proc)
            return proc
        return real(command, *a, **k)

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(native.native_tools, "find_uv", lambda: "/usr/bin/uv")
    real_wait = native._wait_for_health
    monkeypatch.setattr(native, "_wait_for_health", lambda *a, **k: True)
    monkeypatch.setattr(native, "_health", lambda ctl_, url, tls, **k: health["answers"]() if callable(health["answers"]) else health["answers"])
    monkeypatch.setattr(native, "port_owner", lambda port: None)
    busy = {"ports": set()}
    monkeypatch.setattr(ctl, "port_in_use", lambda port: port in busy["ports"])
    monkeypatch.setattr(native, "TERM_GRACE_SECONDS", 3)
    world = types.SimpleNamespace(spawned=spawned, health=health, busy=busy, sadt=str(sadt), tmp=tmp_path,
                                  fail_launch=False, real_wait=real_wait, monkeypatch=monkeypatch)

    def args(**extra):
        base = {"bind": None, "tls_cert": None, "tls_key": None, "port": None, "token": None, "wait": True,
                "timeout": 5, "device": None, "tools_dir": str(sadt), "url": None}
        base.update(extra)
        return _args(**base)

    world.args = args
    yield world
    for proc in spawned:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        proc.wait()


TOKEN_A, TOKEN_B = "A" * 30, "B" * 30


def test_up_twice_leaves_the_server_alone_and_says_so(up_world):
    first = native.cmd_up(ctl, up_world.args(token=TOKEN_A))
    again = native.cmd_up(ctl, up_world.args())
    assert len(up_world.spawned) == 1 and again["pid"] == first["pid"]
    assert again["already_running"] is True and again["restarted"] is False
    assert "Stop it first" not in json.dumps(again)
    assert again["token"] == TOKEN_A and again["url"] == first["url"] and again["healthy"] is True
    # The same token and port restated change nothing either.
    third = native.cmd_up(ctl, up_world.args(token=TOKEN_A, port=int(first["url"].rsplit(":", 1)[1])))
    assert third["pid"] == first["pid"] and len(up_world.spawned) == 1


def test_the_idempotent_result_has_the_shape_of_a_fresh_up(up_world):
    first = native.cmd_up(ctl, up_world.args(token=TOKEN_A))
    again = native.cmd_up(ctl, up_world.args())
    assert set(again) - {"already_running", "restarted"} == set(first)
    assert again["tls"] == first["tls"] and again["tls"]["enabled"] is True


def test_up_with_another_token_restarts_and_the_new_token_is_served(up_world):
    first = native.cmd_up(ctl, up_world.args(token=TOKEN_A))
    again = native.cmd_up(ctl, up_world.args(token=TOKEN_B))
    assert again["restarted"] is True and again["pid"] != first["pid"] and again["token"] == TOKEN_B
    assert up_world.spawned[0].poll() is not None, "the old server was not stopped"
    assert ctl.read_env()["API_TOKEN"] == TOKEN_B


def test_up_with_another_certificate_pair_restarts(up_world):
    cert, key = _make_cert(up_world.tmp, "one")
    other, other_key = _make_cert(up_world.tmp, "two")
    first = native.cmd_up(ctl, up_world.args(tls_cert=cert, tls_key=key))
    same = native.cmd_up(ctl, up_world.args(tls_cert=cert, tls_key=key))
    assert same["pid"] == first["pid"] and same["restarted"] is False
    changed = native.cmd_up(ctl, up_world.args(tls_cert=other, tls_key=other_key))
    assert changed["restarted"] is True and changed["pid"] != first["pid"]
    assert changed["tls"]["cert"] == other


def test_a_second_up_waits_for_a_server_still_starting_and_never_restarts_it(up_world):
    first = native.cmd_up(ctl, up_world.args(token=TOKEN_A))
    up_world.monkeypatch.setattr(native, "_wait_for_health", up_world.real_wait)
    answers = iter([False, False, False, True])
    up_world.health["answers"] = lambda: next(answers, True)
    up_world.monkeypatch.setattr(ctl, "HEALTH_POLL_SECONDS", 0.05)
    up_world.monkeypatch.setattr(native.time, "sleep", lambda s_: None)
    again = native.cmd_up(ctl, up_world.args(timeout=30))
    assert again["pid"] == first["pid"] and len(up_world.spawned) == 1
    assert again["already_running"] is True and again["restarted"] is False and again["healthy"] is True
    assert up_world.spawned[0].poll() is None, "the starting server was killed"


def test_a_second_up_with_no_wait_leaves_a_starting_server_alone(up_world):
    first = native.cmd_up(ctl, up_world.args(token=TOKEN_A))
    up_world.health["answers"] = False
    again = native.cmd_up(ctl, up_world.args(wait=False))
    assert again["pid"] == first["pid"] and again["healthy"] is False and again["restarted"] is False
    assert up_world.spawned[0].poll() is None


def test_a_server_that_never_answers_is_relaunched_after_the_full_delay(up_world):
    first = native.cmd_up(ctl, up_world.args(token=TOKEN_A))
    up_world.monkeypatch.setattr(native, "_wait_for_health", up_world.real_wait)
    up_world.health["answers"] = False
    again = native.cmd_up(ctl, up_world.args(timeout=1))
    assert again["restarted"] is True and again["pid"] != first["pid"]
    assert up_world.spawned[0].poll() is not None


def test_a_dead_server_is_simply_started_again(up_world):
    first = native.cmd_up(ctl, up_world.args(token=TOKEN_A))
    os.killpg(first["pid"], signal.SIGKILL)
    up_world.spawned[0].wait()
    again = native.cmd_up(ctl, up_world.args())
    assert again["pid"] != first["pid"] and len(up_world.spawned) == 2


def test_up_never_stops_a_foreign_process_holding_the_port(up_world):
    foreign = _decoy("/usr/bin/python3 -m http.server 8000", up_world.tmp)
    try:
        time.sleep(0.3)
        up_world.busy["ports"].add(8000)
        _write_pidfile(foreign.pid, port=8000)  # a pidfile that points at it does not make it ours
        with pytest.raises(ctl.ServerCtlError, match="Port 8000 is already used"):
            native.cmd_up(ctl, up_world.args(token=TOKEN_A))
        assert foreign.poll() is None and up_world.spawned == []
    finally:
        foreign.kill()
        foreign.wait()


def test_down_then_up_still_starts_a_fresh_server(up_world):
    first = native.cmd_up(ctl, up_world.args(token=TOKEN_A))
    assert native.cmd_down(ctl, _args())["was_running"] is True
    again = native.cmd_up(ctl, up_world.args())
    assert "already_running" not in again and again["pid"] != first["pid"]


def test_a_start_that_fails_after_the_stop_says_so_and_how_to_start_it(up_world):
    native.cmd_up(ctl, up_world.args(token=TOKEN_A))
    up_world.fail_launch = True
    with pytest.raises(ctl.ServerCtlError, match=r"old native server was stopped.*no fork.*up --native"):
        native.cmd_up(ctl, up_world.args(token=TOKEN_B))


def test_the_port_error_names_the_owner_and_carries_stable_fields(up_world, monkeypatch, capsys):
    foreign = _decoy("/usr/bin/python3 -m http.server 8000", up_world.tmp)
    try:
        time.sleep(0.2)
        up_world.busy["ports"].add(8000)
        monkeypatch.setattr(native, "port_owner", lambda port: {"pid": foreign.pid, "command": "Python"})
        with pytest.raises(ctl.ServerCtlError) as raised:
            native.cmd_up(ctl, up_world.args(token=TOKEN_A))
        text = str(raised.value)
        assert "Python" in text and str(foreign.pid) in text and "--port" in text and "Slicer" in text
        assert raised.value.details == {"error_code": "port_in_use", "port": 8000,
                                        "port_owner": {"pid": foreign.pid, "command": "Python"}}
        # and through main(): the JSON carries them next to the text
        monkeypatch.setitem(sys.modules, "server_ctl_native", native)
        monkeypatch.setitem(sys.modules, ctl.__name__, ctl)
        assert ctl.main(["--json", "up", "--native", "--tools-dir", up_world.sadt, "--token", TOKEN_A]) == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["error_code"] == "port_in_use" and payload["port"] == 8000 and "error" in payload
        assert foreign.poll() is None
    finally:
        foreign.kill()
        foreign.wait()


def test_port_owner_gives_an_abbreviated_command_and_a_pid_and_degrades_without_lsof(monkeypatch):
    class Done:
        stdout = "p4242\ncPython-with-a-very-long-command-name-that-is-cut-short\nn*:8000\n"

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/sbin/lsof")
    monkeypatch.setattr(subprocess, "run", lambda command, **k: (
        pytest.fail("arguments requested") if "-Fpc" not in command else Done()))
    owner = native.port_owner(8000)
    assert owner["pid"] == 4242 and len(owner["command"]) <= 32
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert native.port_owner(8000) is None


@pytest.mark.skipif(shutil.which("ps") is None, reason="needs ps")
def test_a_server_of_this_instance_without_a_pidfile_is_adopted_and_nothing_else_is(up_world, monkeypatch):
    ours_line = f"{native.state_dir()}/api-venv/bin/python serve.py --host 127.0.0.1 --port 8000"
    other_state = str(up_world.tmp / "elsewhere")
    decoys = [_decoy("/usr/bin/python3 serve.py --host 127.0.0.1 --port 8000", up_world.tmp),
              _decoy(f"{native.state_dir()}/api-venv/bin/python other.py --port 8000", up_world.tmp),
              _decoy(f"{other_state}/api-venv/bin/python serve.py --host 127.0.0.1 --port 8000", up_world.tmp)]
    try:
        time.sleep(0.3)
        up_world.busy["ports"].add(8000)
        assert not os.path.exists(native.pidfile_path())
        with pytest.raises(ctl.ServerCtlError, match="Port 8000 is already used"):
            native.cmd_up(ctl, up_world.args(token=TOKEN_A))
        assert not os.path.exists(native.pidfile_path()), "a pidfile was written for a process that is not ours"
        assert all(d.poll() is None for d in decoys) and up_world.spawned == []
        real = _decoy(ours_line, up_world.tmp)
        try:
            time.sleep(0.3)
            ctl.write_env({"API_TOKEN": TOKEN_A})
            again = native.cmd_up(ctl, up_world.args())
            assert again["pid"] == real.pid and again["already_running"] is True and again["restarted"] is False
            assert up_world.spawned == [] and real.poll() is None
            info = native.read_pidfile()
            assert info["pid"] == real.pid and info["port"] == 8000 and info["proc_start"]
            assert native.running_server() is not None
            assert all(d.poll() is None for d in decoys)
        finally:
            real.kill()
            real.wait()
    finally:
        for d in decoys:
            d.kill()
            d.wait()
