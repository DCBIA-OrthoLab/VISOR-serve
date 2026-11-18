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
        done = subprocess.run(["ps", "-axo", "pid=,ppid=,pgid=,command="], stdout=subprocess.PIPE,
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


def _looks_like_our_server(pid: int, info=None) -> bool:
    """Is `pid` the server of THIS instance, and not a process that reused the number?

    The command line must be uvicorn `main:app` (or the `run-native.sh` of this
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
    exec_form = "uvicorn" in command and "main:app" in command
    if not exec_form and not (script in command and command.lstrip().startswith(("bash ", "/bin/bash ", "/usr/bin/bash "))):
        return False
    if exec_form:
        if state_dir() not in command:
            return False
        port = info.get("port")
        if isinstance(port, int) and f"--port {port}" not in command:
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
# Local TLS: a certificate the client pins, served on loopback only
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


def tls_settings(ctl, args):
    """`(cert, key)` to serve with, or None. Arguments win; `.env` remembers; `--no-tls` forgets.

    Validated every time it is used, so a key that became world-readable since
    the last `up` stops the next start instead of being served from.
    """
    if getattr(args, "no_tls", False):
        if getattr(args, "tls_cert", None) or getattr(args, "tls_key", None):
            raise ctl.ServerCtlError("--no-tls cannot be combined with --tls-cert/--tls-key.")
        return None
    cert, key = getattr(args, "tls_cert", None), getattr(args, "tls_key", None)
    if bool(cert) != bool(key):
        raise ctl.ServerCtlError("--tls-cert and --tls-key go together.")
    if not cert:
        env = ctl.read_env()
        cert, key = env.get("TLS_CERT"), env.get("TLS_KEY")
        if not cert or not key:
            return None
    try:
        pair = validate_tls(cert, key)
    except _Error as exc:
        raise ctl.ServerCtlError(str(exc))
    bind = getattr(args, "bind", None)
    if bind is None:
        bind = ctl.read_env().get("BIND_ADDR", LOOPBACK)
    if bind != LOOPBACK:
        raise ctl.ServerCtlError(
            f"TLS is only served on {LOOPBACK} (the bind address is {bind!r}). The certificate the "
            "client pins names loopback and nothing routable.")
    return pair


def refuse_non_loopback(ctl, args) -> str:
    """The native server listens on loopback only, with or without TLS.

    Plain HTTP on a routable address would hand the token and the scans to the
    LAN, and the certificate the client pins names loopback only. A BIND_ADDR
    left in .env by a Docker deployment counts as much as --bind.
    """
    bind = getattr(args, "bind", None)
    if bind is None:
        bind = ctl.read_env().get("BIND_ADDR") or LOOPBACK
    if bind != LOOPBACK:
        raise ctl.ServerCtlError(
            f"The native server listens on {LOOPBACK} only (asked: {bind!r}), with or without TLS. "
            "Use the Docker backend to serve another address.")
    return bind


def remember_tls(ctl, args, tls) -> None:
    """Write the pair into .env when the arguments gave one or took it away. Paths only."""
    if tls and getattr(args, "tls_cert", None):
        ctl.write_env({"TLS_CERT": tls[0], "TLS_KEY": tls[1]})
    elif getattr(args, "no_tls", False):
        ctl.write_env({"TLS_CERT": "", "TLS_KEY": ""})


def tls_report(tls) -> dict:
    """What the JSON says: enabled and the certificate. Never the key, not even its path."""
    return {"enabled": bool(tls), "cert": tls[0] if tls else None}


def cert_digest(tls):
    """sha256 of the certificate (a public file) the process was started with, or None."""
    if not tls:
        return None
    with open(tls[0], "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def server_url(ctl, args, tls, port=None) -> str:
    if getattr(args, "url", None):
        return args.url
    if tls:
        return f"https://{LOOPBACK}:{port or ctl.effective_port()}"
    return ctl.url_for()


def _health(ctl, url, tls, **kwargs) -> bool:
    return ctl.health(url, cafile=tls[0], **kwargs) if tls else ctl.health(url, **kwargs)


# ---------------------------------------------------------------------------
# The commands
# ---------------------------------------------------------------------------

def cmd_status(ctl, args) -> dict:
    env = ctl.read_env()
    # Status reports what is remembered; it does not judge the files (up does).
    remembered = (env.get("TLS_CERT"), env.get("TLS_KEY"))
    tls = (os.path.abspath(remembered[0]), remembered[1]) if all(remembered) else None
    url = server_url(ctl, args, tls)
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


def cmd_up(ctl, args) -> dict:
    live = running_server()
    port = args.port or (live or {}).get("port") or ctl.effective_port()
    # Checked before anything is written: a refused TLS request leaves .env as it was.
    refuse_non_loopback(ctl, args)
    tls = tls_settings(ctl, args)
    if live:
        raise ctl.ServerCtlError(
            f"The native server is already running (pid {live['pid']}, port {live.get('port')}). "
            f"Stop it first with: server_ctl.py down --native")
    if ctl.port_in_use(port):
        raise ctl.ServerCtlError(
            f"Something is already listening on port {port} of this machine, and it is not this "
            f"deployment's native server. Pick another with --port, or stop whatever holds it.")
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

    ctl.ensure_data_dir()
    device = resolve_device(ctl, args)
    token = ctl.ensure_env(SERVICE, args.bind, token=args.token, port=port)
    ctl.write_env({"DEVICE": device, "SADT_TOOLS": root})
    remember_tls(ctl, args, tls)
    env_file = ctl.read_env()
    url = server_url(ctl, args, tls, port)

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
    environment.pop("VISOR_TLS_CERT", None)
    environment.pop("VISOR_TLS_KEY", None)
    if tls:
        environment["VISOR_TLS_CERT"], environment["VISOR_TLS_KEY"] = tls
    log_file = open(log_path(), "ab")
    log_file.write(("\n--- %s starting (device %s) ---\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), device)).encode())
    log_file.flush()
    ctl.log(f"Starting the native server (device {device}{', https' if tls else ''}); log: {log_path()}")
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
                   "tls": bool(tls), "tls_sha": cert_digest(tls),
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
    tls = tls_settings(ctl, args)
    url = server_url(ctl, args, tls)
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
    if os.path.isdir(os.path.join(root, ".git")) or os.path.isfile(os.path.join(root, ".git")):
        result["tools_pulled"] = _pull_tools(ctl, root)

    ctl.write_env({"SADT_TOOLS": root})
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
    if live and (changed or args.force or tls_changed):
        ctl.log("Relaunching the native server so it sees what changed...")
        cmd_down(ctl, args)
        args.port = live.get("port")
        up = cmd_up(ctl, args)
        result["restarted"] = True
        result["healthy"] = up["healthy"]
    elif live:
        result["healthy"] = _health(ctl, url, tls)
    return result


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
        ctl.log(f"Could not fetch {root}: {err}")
        return False
    rc, counts, _err = git("rev-list", "--left-right", "--count", "HEAD...@{u}")
    behind = int(counts.split()[1]) if rc == 0 and len(counts.split()) == 2 else 0
    if not behind:
        ctl.log("The tools checkout is already up to date.")
        return False
    if dirty:
        ctl.log(f"The tools checkout has uncommitted changes and is {behind} commit(s) behind: not pulling over them.")
        return False
    return ctl._stream(["git", "-C", root, "pull", "--ff-only"]) == 0


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
