"""The native backend of server_ctl.py: no Docker, the server is a plain process.

Why it exists: on a Mac, Docker cannot give a container the Apple GPU, so MPS is
only reachable from a process running on macOS itself. This backend starts that
process (`scripts/run-native.sh`, which runs uvicorn), keeps a pidfile, writes a
log under `~/Library/Logs/VISOR/`, and builds the tools' virtualenvs with the
recipe in `native_tools.py`.

It is loaded by `server_ctl.py` only when `--native` is given (or on macOS, where
there is no other way to reach the GPU), and it reuses that module's helpers
instead of repeating them, so the two backends cannot disagree about `.env`, the
port, the token or the health check. Nothing here runs in the Docker path.

The contract server_ctl.py keeps is kept: the result of each command is a dict
that `--json` prints alone on stdout, narration goes to stderr, and only the
standard library is used (this runs inside Slicer's interpreter too).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import ssl
import stat
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import native_tools  # noqa: E402  (after the sys.path line above, which is the point)

SERVICE = "native"
# Seconds a first start may take: the API venv is built on the first run.
STARTUP_TIMEOUT = 300
TERM_GRACE_SECONDS = 20
KILL_GRACE_SECONDS = 5


# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------

def is_darwin() -> bool:
    return sys.platform == "darwin"


def state_dir() -> str:
    """Pidfile, API venv, schema cache. Overridable, which is what the tests use."""
    explicit = os.environ.get("VISOR_STATE_DIR")
    if explicit:
        return explicit
    if is_darwin():
        return os.path.expanduser("~/Library/Application Support/VISOR/native")
    return os.path.expanduser("~/.local/state/visor/native")


def log_dir() -> str:
    explicit = os.environ.get("VISOR_LOG_DIR")
    if explicit:
        return explicit
    if is_darwin():
        return os.path.expanduser("~/Library/Logs/VISOR")
    return os.path.join(state_dir(), "logs")


def torch_home() -> str:
    """Where the tools' torch hub cache lives (run-native.sh exports the same path as TORCH_HOME)."""
    return os.path.join(state_dir(), "torch_home")


def pidfile_path() -> str:
    return os.path.join(state_dir(), "server.pid")


def log_path() -> str:
    return os.path.join(log_dir(), "server.log")


# ---------------------------------------------------------------------------
# The process
# ---------------------------------------------------------------------------

def read_pidfile():
    try:
        with open(pidfile_path(), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("pid"), int) else None


def _ps_rows():
    """`[(pid, ppid, pgid, command)]` for every process. `ps` is on both Linux and macOS."""
    try:
        done = subprocess.run(["ps", "-ww", "-axo", "pid=,ppid=,pgid=,command="], stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in done.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), int(parts[2]), parts[3] if len(parts) > 3 else ""))
        except ValueError:
            continue
    return rows


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie answers kill(0) too, and it is not a server.
    try:
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return True
    return bool(state) and not state.startswith("Z")


def process_start_time(pid: int):
    """`ps -o lstart=` for `pid` (the same text twice for the same process), or None."""
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True, timeout=10, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return " ".join(out.split()) or None


def _is_serve_py_command(command: str) -> bool:
    """The line run-native.sh ends in: `<state>/api-venv/bin/python serve.py --host H --port P`.

    The interpreter must be THIS instance's API venv (a prefix match, since the
    macOS state folder has a space in it), and `serve.py` must be one of the
    words after it. Any other python running any other serve.py is not ours.
    """
    interpreter = os.path.join(state_dir(), "api-venv", "bin", "python")
    if not command.startswith(interpreter):
        return False
    return re.search(r"(^|\s)serve\.py(\s|$)", command[len(interpreter):]) is not None


def _looks_like_our_server(pid: int, info=None) -> bool:
    """Is `pid` the server of THIS instance, and not a process that reused the number?

    The command line must be `serve.py` run by this instance's API venv (or the older
    uvicorn `main:app`, or the `run-native.sh` of this
    checkout, before its `exec`) AND name this instance: its state folder (the
    API venv lives there) and its port. When the pidfile recorded the start time,
    it must match too. A second VISOR server, a dev uvicorn or a shell that merely
    mentions those words is therefore never signalled. No pattern search: only the
    one pid of the pidfile is looked at.
    """
    if info is None:
        info = read_pidfile() or {}
    command = None
    for row_pid, _ppid, _pgid, row_command in _ps_rows():
        if row_pid == pid:
            command = row_command
            break
    if command is None:
        return False
    script = os.path.join(_HERE, "run-native.sh")
    exec_form = ("uvicorn" in command and "main:app" in command) or _is_serve_py_command(command)
    if not exec_form and not (script in command and command.lstrip().startswith(("bash ", "/bin/bash ", "/usr/bin/bash "))):
        return False
    if exec_form:
        if state_dir() not in command:
            return False
        port = info.get("port")
        if isinstance(port, int) and not re.search(rf"--port {port}(\s|$)", command):
            return False
    recorded = info.get("proc_start")
    return not recorded or process_start_time(pid) == recorded


def running_server():
    """The pidfile's contents if it names a live uvicorn of ours, else None."""
    info = read_pidfile()
    if info and pid_alive(info["pid"]) and _looks_like_our_server(info["pid"], info):
        return info
    return None


def _descendants(root_pid: int) -> list:
    """Every process below `root_pid`. Tool runs are in sessions of their own, so a group kill misses them."""
    rows = _ps_rows()
    children = {}
    for pid, ppid, _pgid, _command in rows:
        children.setdefault(ppid, []).append(pid)
    found, stack = [], [root_pid]
    while stack:
        for child in children.get(stack.pop(), []):
            found.append(child)
            stack.append(child)
    return found


def stop_process_tree(pid: int, grace: float = TERM_GRACE_SECONDS) -> dict:
    """SIGTERM the server's group and every tool group under it, then SIGKILL what stayed.

    The tools are started with `start_new_session=True` (so a cancel can kill a
    run and nothing else): they are not in the server's process group, and a
    plain `kill -TERM -<pgid>` would leave a segmentation running on the GPU
    after `down`. Docker's `stop` kills the whole container, so this is the
    native equivalent.
    """
    below = _descendants(pid)
    rows = {row[0]: row for row in _ps_rows()}
    groups = {rows[p][2] for p in below if p in rows}
    try:
        server_group = os.getpgid(pid)
    except OSError:
        server_group = None
    if server_group is not None:
        groups.add(server_group)
    # Never signal our own group: `os.killpg(own)` would stop this very command.
    groups.discard(os.getpgrp())
    for group in sorted(groups):
        try:
            os.killpg(group, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    everyone = [pid] + below
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not any(pid_alive(p) for p in everyone):
            return {"signalled_groups": sorted(groups), "killed": False}
        time.sleep(0.25)
    for group in sorted(groups):
        try:
            os.killpg(group, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    end = time.monotonic() + KILL_GRACE_SECONDS
    while time.monotonic() < end and any(pid_alive(p) for p in everyone):
        time.sleep(0.2)
    return {"signalled_groups": sorted(groups), "killed": True}


# ---------------------------------------------------------------------------
# Configuration: the same .env as the Docker path
# ---------------------------------------------------------------------------

def default_device() -> str:
    if is_darwin():
        return "mps"
    return "cuda" if shutil.which("nvidia-smi") else "cpu"


def resolve_device(ctl, args) -> str:
    """--device gpu|cpu wins, then what .env says, then the platform's default.

    `gpu` means the platform's GPU: mps on a Mac, cuda elsewhere. Storing it in
    .env keeps a later `up` on the same device without asking again.
    """
    if getattr(args, "device", None) == "cpu":
        return "cpu"
    if getattr(args, "device", None) == "gpu":
        return "mps" if is_darwin() else "cuda"
    existing = ctl.read_env().get("DEVICE")
    return existing or default_device()


def sadt_root(ctl, args=None) -> str:
    """The SADT-VISOR checkout: --tools-dir, $SADT_TOOLS, .env, then the sibling folder."""
    explicit = getattr(args, "tools_dir", None) if args is not None else None
    candidate = (explicit or os.environ.get("SADT_TOOLS") or ctl.read_env().get("SADT_TOOLS")
                 or os.path.join(os.path.dirname(ctl.REPO_ROOT), "SADT-VISOR"))
    return os.path.abspath(os.path.expanduser(candidate))


def venv_report(root: str) -> dict:
    found = native_tools.discover_tools(root)
    return {name: os.path.isfile(os.path.join(path, ".venv", "bin", "python")) for name, path in sorted(found.items())}


# ---------------------------------------------------------------------------
# TLS: always, on loopback only. serve.py (the upstream server's one entry point)
# serves https; the certificate is the operator's (TLS_CERT_FILE/TLS_KEY_FILE) or
# the one serve.py generates into TLS_DIR. The client pins whichever is served.
# ---------------------------------------------------------------------------

LOOPBACK = "127.0.0.1"


class _Error(Exception):
    """Turned into the caller's ServerCtlError, which this module cannot import (no cycle)."""


def _check_key_file(key: str) -> None:
    """The key must be a regular file nobody else can read. Its content is never read here."""
    try:
        info = os.stat(key)
    except OSError:
        raise _Error("The TLS key file cannot be read.") from None
    if not stat.S_ISREG(info.st_mode):
        raise _Error("The TLS key is not a regular file.")
    if os.name == "posix":
        if info.st_mode & 0o077:
            raise _Error(f"The TLS key file is readable by other accounts (mode {info.st_mode & 0o777:o}). "
                         "Make it owner-only: chmod 600 on that file.")
        if info.st_uid != os.getuid():
            raise _Error("The TLS key file belongs to another account.")


def validate_tls(cert: str, key: str):
    """Absolute paths of a usable pair, or `_Error`. Nothing about the key is ever reported."""
    cert = os.path.abspath(os.path.expanduser(cert))
    key = os.path.abspath(os.path.expanduser(key))
    if not os.path.isfile(cert):
        raise _Error(f"The TLS certificate {cert} does not exist.")
    _check_key_file(key)
    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
    except (ssl.SSLError, OSError, ValueError):
        # No detail on purpose: a parser's message may quote the file it choked on.
        raise _Error("The TLS certificate and key do not form a usable pair (unreadable, "
                     "not PEM, or the key does not belong to the certificate).") from None
    return cert, key


def generated_cert_path() -> str:
    """Where serve.py puts the certificate it generates (run-native.sh sets TLS_DIR to this folder)."""
    return os.path.join(state_dir(), "tls", "cert.pem")


def tls_settings(ctl, args):
    """`(cert, key)` of the pair the operator supplied, or None for serve.py's own certificate.

    Arguments win, `.env` remembers (TLS_CERT_FILE/TLS_KEY_FILE, the upstream
    server's own names). Validated every time it is used, so a key that became
    world-readable since the last `up` stops the next start instead of being
    served from.
    """
    cert, key = getattr(args, "tls_cert", None), getattr(args, "tls_key", None)
    if bool(cert) != bool(key):
        raise ctl.ServerCtlError("--tls-cert and --tls-key go together.")
    if not cert:
        env = ctl.read_env()
        cert, key = env.get("TLS_CERT_FILE"), env.get("TLS_KEY_FILE")
        if not cert or not key:
            return None
    try:
        return validate_tls(cert, key)
    except _Error as exc:
        raise ctl.ServerCtlError(str(exc))


def served_cert(tls) -> str:
    """The certificate path the server serves: the supplied one, else the generated one."""
    return tls[0] if tls else generated_cert_path()


def refuse_non_loopback(ctl, args) -> str:
    """The native server listens on loopback only, with or without a supplied certificate.

    The certificate the client pins names loopback only, and the upstream
    server will not publish a self-signed one elsewhere. A BIND_ADDR left in
    .env by a Docker deployment counts as much as --bind.
    """
    bind = getattr(args, "bind", None)
    if bind is None:
        bind = ctl.read_env().get("BIND_ADDR") or LOOPBACK
    if bind != LOOPBACK:
        raise ctl.ServerCtlError(
            f"The native server listens on {LOOPBACK} only (asked: {bind!r}), with or without a certificate. "
            "Use the Docker backend to serve another address.")
    return bind


def remember_tls(ctl, args, tls) -> None:
    """Write the pair into .env when the arguments gave one. Paths only."""
    if tls and getattr(args, "tls_cert", None):
        ctl.write_env({"TLS_CERT_FILE": tls[0], "TLS_KEY_FILE": tls[1]})


def tls_report(tls) -> dict:
    """What the JSON says: always enabled, and the certificate served. Never the key, not even its path."""
    return {"enabled": True, "cert": served_cert(tls)}


def cert_digest(tls):
    """sha256 of the SUPPLIED certificate (a public file) the process was started with, or None.

    None for serve.py's generated certificate: it is made at start and kept, so
    there is nothing to compare, and a relaunch would not change it.
    """
    if not tls:
        return None
    with open(tls[0], "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def server_url(ctl, args, port=None) -> str:
    """Always https on loopback, whatever the public URL is: the native server has no other front door."""
    if getattr(args, "url", None):
        return args.url
    return f"https://{LOOPBACK}:{port or ctl.effective_port()}"


def _health(ctl, url, tls, **kwargs) -> bool:
    return ctl.health(url, cafile=served_cert(tls), **kwargs)


# ---------------------------------------------------------------------------
# The commands
# ---------------------------------------------------------------------------

def cmd_status(ctl, args) -> dict:
    env = ctl.read_env()
    # Status reports what is remembered; it does not judge the files (up does).
    remembered = (env.get("TLS_CERT_FILE"), env.get("TLS_KEY_FILE"))
    tls = (os.path.abspath(remembered[0]), remembered[1]) if all(remembered) else None
    url = server_url(ctl, args)
    clone = ctl.clone_status(check_remote=args.check_remote, want_branch=args.branch)
    root = sadt_root(ctl, args)
    info = read_pidfile()
    live = running_server()
    return {
        "backend": SERVICE,
        "repo_root": ctl.REPO_ROOT,
        "git": ctl.git_info(),
        "platform": {"system": sys.platform, "machine": os.uname().machine if hasattr(os, "uname") else None},
        "uv": native_tools.find_uv(),
        "service": SERVICE,
        # The keys the Docker status has and the Slicer panel reads (`docker`,
        # `compose`, `gpu`, `container`), present so that a client written
        # against that contract gets an answer of the same shape. They say what
        # is true here: Docker is not used, the GPU is the platform's own.
        "docker": {"available": False, "version": None, "daemon": False,
                   "error": "not used by the native backend"},
        "compose": {"available": False, "version": None},
        "gpu": {"nvidia_runtime": False, "nvidia_smi": bool(shutil.which("nvidia-smi")),
                "gpu_access": "mps" if is_darwin() else None, "error": None},
        "container": {"service": SERVICE, "exists": bool(info), "running": bool(live),
                      "state": "running" if live else ("exited" if info else None), "error": None},
        "clone": clone,
        "process": {
            "pid": live["pid"] if live else None,
            "running": bool(live),
            "pidfile": pidfile_path(),
            "stale_pidfile": bool(info and not live),
            "log": log_path(),
        },
        "url": url,
        "tls": tls_report(tls),
        "server": {"url": url, "healthy": _health(ctl, url, tls)},
        "tools": {"sadt": root, "present": os.path.isdir(os.path.join(root, "tools")),
                  "venvs": venv_report(root) if os.path.isdir(os.path.join(root, "tools")) else {}},
        "env": {
            "path": ctl.ENV_PATH,
            "exists": os.path.isfile(ctl.ENV_PATH),
            "has_token": bool(env.get("API_TOKEN")),
            "bind_addr": env.get("BIND_ADDR"),
            "device": env.get("DEVICE") or default_device(),
        },
        "data_dir": ctl.data_dir(),
    }


def _tail(path: str, lines: int) -> list:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 256 * 1024))
            data = handle.read().decode("utf-8", "replace")
    except OSError:
        return []
    return data.splitlines()[-lines:]


def _wait_for_health(ctl, url: str, pid: int, timeout: int, tls=None) -> bool:
    deadline = time.monotonic() + timeout
    ctl.log(f"Waiting for {url}/health (up to {timeout // 60} min; the first start builds the API environment)...")
    while time.monotonic() < deadline:
        if _health(ctl, url, tls):
            ctl.log("The server is up.")
            return True
        if not pid_alive(pid):
            ctl.log("The server process exited before it answered. Last log lines:")
            for line in _tail(log_path(), 40):
                ctl.log("  | " + line)
            return False
        time.sleep(ctl.HEALTH_POLL_SECONDS if timeout > 60 else 1)
    ctl.log(f"Still no answer from {url}/health after {timeout}s. Last log lines:")
    for line in _tail(log_path(), 40):
        ctl.log("  | " + line)
    return False


def _launch_prerequisites(ctl, args):
    """`(run-native.sh, tools checkout)` if a start can happen, else ServerCtlError. Changes nothing.

    Shared by `up` and by `update` BEFORE it stops the old server, so a start
    that cannot happen is refused while the old server is still serving.
    """
    script = os.path.join(_HERE, "run-native.sh")
    if not os.path.isfile(script):
        raise ctl.ServerCtlError(f"{script} is missing.")
    if native_tools.find_uv() is None:
        raise ctl.ServerCtlError(
            "uv is not installed. It builds the server's and the tools' Python environments. "
            "Install it from https://docs.astral.sh/uv/ and run this again.")
    root = sadt_root(ctl, args)
    if not os.path.isdir(os.path.join(root, "tools")):
        raise ctl.ServerCtlError(
            f"No SADT-VISOR checkout at {root} (no tools/ folder). Pass --tools-dir PATH; it is "
            f"remembered in .env.")
    return script, root


TOKEN_MIN_LENGTH = 24  # serve.py refuses a shorter API_TOKEN, after the old server would be stopped


def check_token(ctl, token) -> None:
    """A `--token` serve.py will accept and `.env` can hold on one line, or ServerCtlError. Never echoes it."""
    if token is None:
        return
    if len(token) < TOKEN_MIN_LENGTH or not re.fullmatch(r"[A-Za-z0-9_\-.~+/=]+", token):
        raise ctl.ServerCtlError(
            f"--token must be at least {TOKEN_MIN_LENGTH} characters, using letters, digits and _-.~+/= only "
            "(no space, line break or control character).")


def adopt_running_server(ctl):
    """The pidfile entry of a server of THIS instance that runs without a (valid) pidfile, or None.

    Same identity as `_looks_like_our_server`: this instance's API venv
    interpreter running `serve.py`, with a `--port`. Exactly one such process is
    adopted (two is a state nobody should guess at), and nothing else ever is.
    The pidfile is rewritten with what `.env` says it was started with.
    """
    found = []
    for pid, _ppid, _pgid, command in _ps_rows():
        if not _is_serve_py_command(command):
            continue
        match = re.search(r"--port (\d+)(\s|$)", command)
        if match and pid != os.getpid() and pid_alive(pid):
            found.append((pid, int(match.group(1))))
    if len(found) != 1:
        return None
    pid, port = found[0]
    env = ctl.read_env()
    remembered = (env.get("TLS_CERT_FILE"), env.get("TLS_KEY_FILE"))
    try:
        tls = validate_tls(*remembered) if all(remembered) else None
    except _Error:
        return None
    info = {"pid": pid, "port": port, "device": env.get("DEVICE") or default_device(),
            "tools": env.get("SADT_TOOLS"), "started": None, "log": log_path(), "tls": True,
            "tls_sha": cert_digest(tls), "proc_start": process_start_time(pid), "adopted": True}
    os.makedirs(state_dir(), exist_ok=True)
    with open(pidfile_path(), "w", encoding="utf-8") as handle:
        json.dump(info, handle)
    ctl.log(f"Adopted the running native server (pid {pid}, port {port}): its pidfile was missing or stale.")
    return info


def port_owner(port: int):
    """`{"pid", "command"}` of what listens on `port` (abbreviated command, never its arguments), or None."""
    if not shutil.which("lsof"):
        return None
    try:
        out = subprocess.run(["lsof", "-nP", f"-iTCP:{int(port)}", "-sTCP:LISTEN", "-Fpc"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    pid = command = None
    for line in out.splitlines():
        if line.startswith("p") and pid is None:
            pid = int(line[1:]) if line[1:].isdigit() else None
        elif line.startswith("c") and command is None:
            command = line[1:][:32]
    return {"pid": pid, "command": command} if pid and command else None


def _port_in_use_error(ctl, port):
    owner = port_owner(port)
    who = f"{owner['command']} (pid {owner['pid']})" if owner else "another program"
    error = ctl.ServerCtlError(
        f"Port {port} is already used by {who}, and it is not this deployment's native server. "
        f"Quit that program, or serve on another port: pass --port N to `server_ctl.py up --native`, "
        f"or give the SlicerCloud server panel another port, then start again.")
    error.details = {"error_code": "port_in_use", "port": port, "port_owner": owner}
    return error


def _differences(ctl, args, live, tls) -> list:
    """What `up` was asked for that the running server of ours was not started with."""
    found = []
    if args.port and args.port != live.get("port"):
        found.append("another port")
    if live.get("tls_sha") != cert_digest(tls):
        found.append("another certificate")
    wanted = getattr(args, "token", None)
    stored = ctl.read_env().get("API_TOKEN")
    if not stored or (wanted and wanted != stored):
        found.append("another token")
    if getattr(args, "device", None) and resolve_device(ctl, args) != live.get("device"):
        found.append("another device")
    if getattr(args, "tools_dir", None) and live.get("tools") and sadt_root(ctl, args) != live.get("tools"):
        found.append("another tools checkout")
    return found


def _up_report(ctl, args, live, tls, url, healthy: bool) -> dict:
    root = sadt_root(ctl, args)
    missing = sorted(name for name, present in venv_report(root).items() if not present) \
        if os.path.isdir(os.path.join(root, "tools")) else []
    return {
        "service": SERVICE, "url": url, "tls": tls_report(tls), "healthy": healthy,
        "pid": live["pid"], "device": live.get("device"), "token": ctl.read_env().get("API_TOKEN"),
        "log": log_path(), "data_dir": ctl.data_dir(), "tools_without_venv": missing,
        "already_running": True, "restarted": False,
    }


def _up_over_our_server(ctl, args, live, tls) -> dict:
    """`up` while our own server runs: idempotent, as Docker's is.

    Nothing asked differs and /health answers: report the running server, touch
    nothing. A server that does not answer YET is starting (23 tools take up to
    a minute, and the port is bound last): it is waited for, never stopped, up
    to the usual startup delay; only a dead process, or one still mute after
    that delay, is relaunched, as is any difference in what was asked. A process
    that is not ours never gets here (running_server checks its identity).
    """
    port = live.get("port")
    differences = _differences(ctl, args, live, tls)
    url = server_url(ctl, args, port)
    if not differences:
        if _health(ctl, url, tls):
            ctl.log(f"The native server is already running (pid {live['pid']}, port {port}) with this configuration; left alone.")
            return _up_report(ctl, args, live, tls, url, True)
        if not args.wait:
            ctl.log(f"The native server (pid {live['pid']}) is not answering yet; not waiting, not touching it.")
            return _up_report(ctl, args, live, tls, url, False)
        ctl.log(f"The native server (pid {live['pid']}) is still starting; waiting for it instead of restarting it.")
        if _wait_for_health(ctl, url, live["pid"], args.timeout, tls):
            return _up_report(ctl, args, live, tls, url, True)
        ctl.log("It did not come up: relaunching it.")
        differences = ["no answer"]
    else:
        ctl.log("The native server is running but " + ", ".join(differences) + ": relaunching it.")
    up = _relaunch(ctl, args, live)
    up.update({"already_running": True, "restarted": True})
    return up


def _relaunch(ctl, args, live) -> dict:
    """Stop our running server and start it again, in the order that cannot strand the operator.

    Everything a start needs is checked first; if the start still fails once the
    old server is stopped, the error says so and how to start it by hand.
    """
    _launch_prerequisites(ctl, args)
    cmd_down(ctl, args)
    args.port = args.port or live.get("port")
    try:
        return cmd_up(ctl, args)
    except Exception as exc:  # noqa: BLE001 - whatever stopped the start, the operator must be told
        raise ctl.ServerCtlError(
            f"The old native server was stopped, and the new one did not start ({type(exc).__name__}: {exc}). "
            f"Nothing is serving now. Fix the cause, then start it with: server_ctl.py up --native "
            f"(log: {log_path()})") from exc


def cmd_up(ctl, args) -> dict:
    check_token(ctl, getattr(args, "token", None))
    live = running_server()
    # Checked before anything is written: a refused TLS request leaves .env as it was.
    refuse_non_loopback(ctl, args)
    tls = tls_settings(ctl, args)
    live = live or adopt_running_server(ctl)
    port = args.port or (live or {}).get("port") or ctl.effective_port()
    if live:
        return _up_over_our_server(ctl, args, live, tls)
    if ctl.port_in_use(port):
        raise _port_in_use_error(ctl, port)
    script, root = _launch_prerequisites(ctl, args)

    ctl.ensure_data_dir()
    device = resolve_device(ctl, args)
    token = ctl.ensure_env(SERVICE, args.bind, token=getattr(args, "token", None), port=port)
    ctl.write_env({"DEVICE": device, "SADT_TOOLS": root})
    remember_tls(ctl, args, tls)
    env_file = ctl.read_env()
    url = server_url(ctl, args, port)

    venvs = venv_report(root)
    missing = sorted(name for name, present in venvs.items() if not present)
    if missing:
        ctl.log("No virtualenv yet for: " + ", ".join(missing) +
                ". Those tools answer 503 until `server_ctl.py update --native` builds them.")

    os.makedirs(log_dir(), exist_ok=True)
    os.makedirs(state_dir(), exist_ok=True)
    environment = dict(os.environ)
    environment.update({
        "API_TOKEN": token,
        "DEVICE": device,
        "HOST_PORT": str(port),
        "BIND_ADDR": env_file.get("BIND_ADDR", "127.0.0.1"),
        "SADT_TOOLS": root,
        "DATA_DIR": ctl.data_dir(),
        "VISOR_STATE_DIR": state_dir(),
    })
    # Only the paths travel, in the environment of the one process that needs them.
    environment.pop("TLS_CERT_FILE", None)
    environment.pop("TLS_KEY_FILE", None)
    environment["TLS_DIR"] = os.path.dirname(generated_cert_path())
    if tls:
        environment["TLS_CERT_FILE"], environment["TLS_KEY_FILE"] = tls
    log_file = open(log_path(), "ab")
    log_file.write(("\n--- %s starting (device %s) ---\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), device)).encode())
    log_file.flush()
    ctl.log(f"Starting the native server (device {device}, https); log: {log_path()}")
    process = subprocess.Popen(
        ["bash", script], cwd=ctl.REPO_ROOT, env=environment, stdin=subprocess.DEVNULL,
        stdout=log_file, stderr=subprocess.STDOUT,
        # Its own session: it must outlive this command and the Slicer that ran it.
        start_new_session=True,
    )
    log_file.close()
    with open(pidfile_path(), "w", encoding="utf-8") as handle:
        json.dump({"pid": process.pid, "port": port, "device": device, "tools": root,
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "log": log_path(),
                   "tls": True, "tls_sha": cert_digest(tls),
                   "proc_start": process_start_time(process.pid)}, handle)

    healthy = _wait_for_health(ctl, url, process.pid, args.timeout, tls) if args.wait else False
    return {
        "service": SERVICE,
        "url": url,
        "tls": tls_report(tls),
        "healthy": healthy,
        "pid": process.pid,
        "device": device,
        "token": token,
        "log": log_path(),
        "data_dir": ctl.data_dir(),
        "tools_without_venv": missing,
    }


def cmd_down(ctl, args) -> dict:
    info = read_pidfile()
    live = running_server()
    if not live:
        if info:
            os.unlink(pidfile_path())
        ctl.log("The native server is not running.")
        return {"service": SERVICE, "stopped": True, "was_running": False}
    ctl.log(f"Stopping the native server (pid {live['pid']}) and the tool runs under it...")
    outcome = stop_process_tree(live["pid"])
    try:
        os.unlink(pidfile_path())
    except OSError:
        pass
    return {"service": SERVICE, "stopped": not pid_alive(live["pid"]), "was_running": True, **outcome}


def cmd_logs(ctl, args) -> dict:
    for line in _tail(log_path(), args.lines):
        ctl.log("  | " + line)
    return {"service": SERVICE, "lines": args.lines, "log": log_path()}


def cmd_update(ctl, args) -> dict:
    """Pull this clone and the tools' checkout, rebuild the venvs that changed, relaunch if running.

    The code half is the Docker path's rule word for word: fast-forward only,
    never over uncommitted edits. The venv half is `native_tools`: a venv whose
    inputs (pyproject, lock, recipe) did not move is left alone, so an update
    with nothing new costs two git calls.
    """
    check_token(ctl, getattr(args, "token", None))
    tls = tls_settings(ctl, args)
    # What a relaunch needs, asked now: nothing below may stop the old server
    # for a start that was never going to be allowed.
    refuse_non_loopback(ctl, args)
    url = server_url(ctl, args)
    result = {"service": SERVICE, "url": url, "tls": tls_report(tls), "pulled": False, "tools_pulled": False,
              "venvs": [], "restarted": False, "healthy": False, "switched_branch": False}

    if args.branch:
        result["switched_branch"] = ctl.checkout_branch(args.branch)
    clone = ctl.clone_status(check_remote=True, want_branch=args.branch)
    result["clone"] = clone
    if clone["is_git_repo"] and clone["checked_remote"]:
        if clone["dirty"] and clone["behind"]:
            raise ctl.ServerCtlError(
                f"This clone has uncommitted changes and is {clone['behind']} commit(s) behind "
                f"{ctl._upstream()}. Refusing to pull over local edits - commit or discard them first.")
        if clone["behind"]:
            ctl.log(f"{clone['behind']} new commit(s). Fast-forwarding...")
            if ctl._stream(["git", "pull", "--ff-only"], cwd=ctl.REPO_ROOT) != 0:
                raise ctl.ServerCtlError("'git pull --ff-only' failed. The local branch has probably diverged.")
            result["pulled"] = True
        else:
            ctl.log("The clone is already up to date.")
    elif clone["error"]:
        ctl.log(f"Skipping the code update: {clone['error']}")

    root = sadt_root(ctl, args)
    if not os.path.isdir(os.path.join(root, "tools")):
        raise ctl.ServerCtlError(f"No SADT-VISOR checkout at {root}. Pass --tools-dir PATH.")
    is_checkout = os.path.isdir(os.path.join(root, ".git")) or os.path.isfile(os.path.join(root, ".git"))
    wanted_branch, wanted_url = getattr(args, "tools_branch", None), getattr(args, "tools_url", None)
    if wanted_url and not wanted_branch:
        raise ctl.ServerCtlError("--tools-url needs --tools-branch: say which branch of that repository to serve.")
    if wanted_branch and not is_checkout:
        raise ctl.ServerCtlError(f"--tools-branch was given but {root} is not a git checkout: nothing to switch.")
    if is_checkout:
        if wanted_branch:
            result["tools_switched"] = switch_tools_branch(ctl, root, wanted_branch, wanted_url)
        result["tools_pulled"] = _pull_tools(ctl, root)

    ctl.write_env({"SADT_TOOLS": root})
    wanted_token = getattr(args, "token", None)
    token_changed = bool(wanted_token) and wanted_token != ctl.read_env().get("API_TOKEN")
    if token_changed:
        # Written even when nothing else asks for a relaunch: a rotated token that is
        # only kept in memory of this command would leave the old one in force.
        ctl.write_env({"API_TOKEN": wanted_token})
    live = running_server()
    # The running process serves the certificate it started with: a different
    # one (or a renewed file at the same path) needs a relaunch to take effect.
    tls_changed = bool(live) and live.get("tls_sha") != cert_digest(tls)
    remember_tls(ctl, args, tls)
    outcome = native_tools.run(root, tools=getattr(args, "tool", None), force=args.force,
                               torch_home=torch_home())
    result["venvs"] = outcome["tools"]
    result["venvs_failed"] = outcome["failed"]

    live = running_server()
    changed = result["pulled"] or result["tools_pulled"] or any(r.get("action") == "built" for r in outcome["tools"])
    if live and (changed or args.force or tls_changed or token_changed):
        ctl.log("Relaunching the native server so it sees what changed...")
        args.port = live.get("port")
        up = _relaunch(ctl, args, live)
        result["restarted"] = True
        result["healthy"] = up["healthy"]
    elif live:
        result["healthy"] = _health(ctl, url, tls)
    return result


_URL_PREFIXES = ("https://", "ssh://", "git@")
# `git switch` needs git 2.23; `--no-overwrite-ignore` is on `switch` and `merge` from then on.
# An older git is refused rather than run without the protection.
_OLD_GIT = "git 2.23 or newer is required (git switch --no-overwrite-ignore)"


def _OLD_GIT_HINT_IF(err: str) -> str:
    return f" ({_OLD_GIT})" if "overwrite-ignore" in (err or "") and "unknown option" in (err or "") else ""


def _mask(text: str) -> str:
    """Hide the credentials of any URL in `text` (https://user:SECRET@host becomes https://***@host)."""
    return re.sub(r"//[^/@\s]*@", "//***@", text or "")


def switch_tools_branch(ctl, root: str, branch: str, url=None) -> bool:
    """Put the managed tools clone `root` on `branch` of `url` (origin when None). True if HEAD moved.

    Refuses, with nothing changed, rather than lose anything: a dirty tree, a
    detached HEAD whose commits no branch holds, a local branch of that name
    that has commits the remote one lacks (or that has diverged), a branch that
    does not exist at that URL. Only ever runs git inside `root`, and refuses
    when `root` is not the top of its own repository (so a checkout nested in
    another project is never switched). The branch we leave keeps its commits.
    """
    def git(*arguments, timeout=300):
        return ctl._capture(["git", "-C", root] + list(arguments), timeout=timeout)

    def refuse(message):
        raise ctl.ServerCtlError(f"Tools checkout {root}: {_mask(message)}" + _OLD_GIT_HINT_IF(message))

    if not branch or branch.startswith("-") or git("check-ref-format", "--branch", branch)[0] != 0:
        refuse(f"{branch!r} is not a usable branch name.")
    if url is not None and (not url.startswith(_URL_PREFIXES) or "::" in url):
        refuse(f"{_mask(url)!r} is not an https://, ssh:// or git@ repository address.")
    rc, top, _err = git("rev-parse", "--show-toplevel")
    if rc != 0 or os.path.realpath(top) != os.path.realpath(root):
        refuse("it is not the top folder of its own git repository; leaving it alone.")

    rc, origin_url, _err = git("remote", "get-url", "origin")
    from_origin = url is None or (rc == 0 and url.rstrip("/").removesuffix(".git") == origin_url.rstrip("/").removesuffix(".git"))
    if from_origin:
        source, target = "origin", f"refs/remotes/origin/{branch}"
        fetch = ["fetch", "--quiet", "origin", f"+refs/heads/{branch}:{target}"]
    else:
        source, target = _mask(url), f"refs/remotes/visor-requested/{branch}"
        fetch = ["fetch", "--quiet", "--", url, f"+refs/heads/{branch}:{target}"]
    rc, _out, err = git(*fetch)
    if rc != 0:
        refuse(f"could not fetch branch {branch!r} from {source}: {err or 'git refused'}.")
    rc, want, _err = git("rev-parse", "--verify", "--quiet", target + "^{commit}")
    if rc != 0:
        refuse(f"branch {branch!r} does not exist at {source}.")

    def is_ancestor(older, newer):
        return git("merge-base", "--is-ancestor", older, newer)[0] == 0

    _rc, current, _err = git("rev-parse", "--abbrev-ref", "HEAD")
    _rc, head, _err = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain")[1])

    if current == branch:
        if head == want or is_ancestor(want, head):
            ctl.log(f"The tools checkout is on '{branch}' already, at or ahead of {source}.")
            return False
        if not is_ancestor(head, want):
            refuse(f"'{branch}' has diverged from {source}: it has commits the remote branch lacks. "
                   f"Merge or rebase it by hand.")
        if dirty:
            refuse(f"'{branch}' is behind {source} but the tree has uncommitted changes. Commit or discard them.")
        if git("merge", "--no-overwrite-ignore", "--ff-only", want)[0] != 0:
            refuse("the fast-forward failed.")
        ctl.log(f"Fast-forwarded the tools checkout's '{branch}' to {want[:9]}.")
        return True

    if dirty:
        refuse(f"it is on '{current}' and '{branch}' was asked for, but the tree has uncommitted "
               f"changes. Commit or discard them, then update again.")
    if current == "HEAD" and not git("for-each-ref", "--contains", "HEAD", "refs/heads", "refs/remotes", "refs/tags")[1]:
        refuse("it is on a detached HEAD whose commits no branch holds: switching would lose them. "
               "Create a branch there first.")
    ctl.log(f"Switching the tools checkout from '{current}' to '{branch}' ({source})...")
    if git("rev-parse", "--verify", "--quiet", f"refs/heads/{branch}")[0] == 0:
        _rc, local, _err = git("rev-parse", f"refs/heads/{branch}")
        if local != want and not is_ancestor(local, want):
            refuse(f"the local branch '{branch}' has commits that {source} lacks (or has diverged): "
                   f"not touching it. Merge or rebase it by hand.")
        rc, _out, err = git("switch", "--no-overwrite-ignore", branch)
        if rc != 0:
            refuse(f"git switch refused: {err}")
        if local != want and git("merge", "--no-overwrite-ignore", "--ff-only", want)[0] != 0:
            refuse("the fast-forward failed.")
    else:
        command = ["switch", "--no-overwrite-ignore", "-c", branch, "--track", target] if from_origin else \
                  ["switch", "--no-overwrite-ignore", "-c", branch, "--no-track", target]
        rc, _out, err = git(*command)
        if rc != 0:
            refuse(f"git switch refused: {err}")
    return True


def _pull_tools(ctl, root: str) -> bool:
    """Fast-forward the SADT-VISOR checkout, or say why not. Never over edits."""
    def git(*arguments, timeout=300):
        return ctl._capture(["git", "-C", root] + list(arguments), timeout=timeout)

    rc, dirty, _err = git("status", "--porcelain")
    if rc != 0:
        ctl.log(f"{root} is not a usable git checkout; leaving it alone.")
        return False
    rc, _out, err = git("fetch", "--quiet", "origin")
    if rc != 0:
        ctl.log(f"Could not fetch {root}: {_mask(err)}")
        return False
    rc, counts, _err = git("rev-list", "--left-right", "--count", "HEAD...@{u}")
    behind = int(counts.split()[1]) if rc == 0 and len(counts.split()) == 2 else 0
    if not behind:
        ctl.log("The tools checkout is already up to date.")
        return False
    if dirty:
        ctl.log(f"The tools checkout has uncommitted changes and is {behind} commit(s) behind: not pulling over them.")
        return False
    # fetch is done above: merge (not pull, which has no such option) so an ignored local file
    # (downloaded weights) that the new commits track is refused, not overwritten.
    rc, _out, err = git("merge", "--no-overwrite-ignore", "--ff-only", "@{u}")
    if rc != 0:
        ctl.log(f"Not fast-forwarding the tools checkout: {_mask(err) or 'git refused'}" + _OLD_GIT_HINT_IF(err))
    return rc == 0


def cmd_install_tools(ctl, args) -> dict:
    return native_tools.run(sadt_root(ctl, args), tools=getattr(args, "tool", None),
                            dry_run=getattr(args, "dry_run", False), force=args.force,
                            torch_home=torch_home())


def print_status(status: dict) -> None:
    def mark(ok):
        return "ok " if ok else "-- "

    process = status["process"]
    print(f"Repository   {status['repo_root']}")
    print(f"  {mark(status['git']['available'])}git       {status['git']['version'] or 'not installed'}")
    print(f"  {mark(status['uv'])}uv        {status['uv'] or 'not installed'}")
    print(f"  platform   {status['platform']['system']} {status['platform']['machine']}")
    clone = status["clone"]
    if clone["is_git_repo"]:
        print(f"\nClone        {clone['branch']}@{clone['commit']}"
              f"{'  [uncommitted changes]' if clone['dirty'] else ''}")
    print(f"Backend      native, device {status['env']['device']}")
    print(f"Process      {'pid %d' % process['pid'] if process['running'] else 'not running'}"
          f"{'  (stale pidfile)' if process['stale_pidfile'] else ''}   log {process['log']}")
    print(f"Health       {status['server']['url']}  {'answering' if status['server']['healthy'] else 'no answer'}")
    tools = status["tools"]
    built = sorted(name for name, present in tools["venvs"].items() if present)
    print(f"Tools        {tools['sadt']}  {len(built)}/{len(tools['venvs'])} venvs built")
    print(f"Token        {'set in .env' if status['env']['has_token'] else 'not generated yet'}")


HANDLERS = {
    "status": (cmd_status, print_status),
    "up": (cmd_up, None),
    "down": (cmd_down, None),
    "logs": (cmd_logs, None),
    "update": (cmd_update, None),
    "install-tools": (cmd_install_tools, None),
}
