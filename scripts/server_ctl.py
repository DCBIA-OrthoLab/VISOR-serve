#!/usr/bin/env python3
"""Bring this server up, keep it current, and choose what lands in DATA/.

This is the engine behind scripts/setup-server.sh, and it is what the Slicer
"Slicer Cloud" module drives: every button in that panel is one subcommand
here. It manages the clone it lives in — `scripts/server_ctl.py` sitting one
directory below the repository root is how it finds everything else.

    python3 scripts/server_ctl.py status --json
    python3 scripts/server_ctl.py up
    python3 scripts/server_ctl.py update
    python3 scripts/server_ctl.py catalog --json
    python3 scripts/server_ctl.py models --tool AMASSS --tool ALI
    python3 scripts/server_ctl.py down

Standard library only, same rule as fetch_data.py and for the same reason:
this runs on a bare host before anything is installed, and inside Slicer's
interpreter, where nothing may be pip-installed on the user's behalf.

Two conventions the GUI depends on:

* **Progress and log lines go to stderr, machine-readable output to stdout.**
  `--json` therefore prints exactly one JSON object on stdout, whatever else
  is being narrated at the same time, and a caller can stream the narration
  into a log pane without having to filter it out of the result.
* **Nothing here ever prints the API token**, except the one subcommand whose
  whole job is to hand it over (`token`). A status dump routinely ends up in a
  log pane, a screenshot, or a bug report.
"""

import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)

DEFAULT_PORT = 8000

# The GPU service and its cardless twin. See docker-compose.yml for why they
# are two services rather than one plus an override file.
GPU_SERVICE = "inference"
CPU_SERVICE = "inference-cpu"
CPU_PROFILE = "cpu"

# A first `up` pulls a multi-GB image and then runs pip inside the container,
# so "not answering yet" is the normal state for a long while.
HEALTH_POLL_SECONDS = 3
DEFAULT_STARTUP_TIMEOUT = 1800


class ServerCtlError(Exception):
    """Something the user can act on: a missing prerequisite, a dirty clone, a
    container that never became healthy."""


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------

def log(message: str) -> None:
    """Narration. Always stderr, so `--json` owns stdout unconditionally."""
    print(message, file=sys.stderr, flush=True)


def _which(name: str):
    return shutil.which(name)


def _capture(cmd, cwd=None, timeout=60):
    """Run `cmd`, returning (returncode, stdout, stderr) and never raising for
    a non-zero exit — callers here decide what a failure means."""
    try:
        completed = subprocess.run(
            cmd, cwd=cwd, timeout=timeout, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", str(exc)
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def _stream(cmd, cwd=None, prefix=""):
    """Run `cmd`, echoing its output to stderr line by line as it arrives.

    Line by line and unbuffered on purpose: `docker compose up` on a fresh host
    spends ten minutes pulling layers, and a caller showing a log pane has
    nothing else to display in the meantime.
    """
    log(f"$ {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
    except OSError as exc:
        raise ServerCtlError(f"Could not run {cmd[0]}: {exc}") from None
    with process:
        for line in process.stdout:
            log(prefix + line.rstrip())
    return process.returncode


# ---------------------------------------------------------------------------
# Host probing
# ---------------------------------------------------------------------------

def git_info() -> dict:
    path = _which("git")
    if not path:
        return {"available": False, "version": None}
    _rc, out, _err = _capture(["git", "--version"])
    return {"available": True, "version": out}


def docker_info() -> dict:
    """Whether docker is installed AND its daemon is reachable by this user.

    The two are worth separating: a fresh `install-docker.sh` leaves the binary
    in place but the user outside the `docker` group, so `docker info` fails
    with a permission error until they log out and back in. Reporting that as
    "docker missing" would send them to reinstall it.
    """
    path = _which("docker")
    if not path:
        return {"available": False, "version": None, "daemon": False, "error": "docker is not in PATH"}
    _rc, version, _err = _capture(["docker", "--version"])
    rc, _out, err = _capture(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=30)
    if rc != 0:
        message = err or "the docker daemon did not answer"
        if "permission denied" in message.lower():
            message = (
                "the docker daemon refused this user. Add yourself to the 'docker' group "
                "(sudo usermod -aG docker $USER) and log out and back in."
            )
        return {"available": True, "version": version, "daemon": False, "error": message}
    return {"available": True, "version": version, "daemon": True, "error": None}


def compose_command():
    """The compose entry point to use: the v2 plugin, else the legacy binary."""
    if _which("docker"):
        rc, _out, _err = _capture(["docker", "compose", "version"])
        if rc == 0:
            return ["docker", "compose"]
    if _which("docker-compose"):
        return ["docker-compose"]
    return None


def compose_info() -> dict:
    command = compose_command()
    if command is None:
        return {"available": False, "version": None, "command": None}
    _rc, out, _err = _capture(command + ["version"])
    return {"available": True, "version": out, "command": command}


def gpu_info() -> dict:
    """Whether docker can actually hand a container an nvidia device.

    `nvidia-smi` on the host is not the answer: the container toolkit is a
    separate install, and without it the GPU service fails to start on a
    machine whose card works perfectly outside docker. What decides is whether
    docker itself knows an "nvidia" runtime.
    """
    runtime = False
    error = None
    if _which("docker"):
        rc, out, err = _capture(["docker", "info", "--format", "{{json .Runtimes}}"], timeout=30)
        if rc == 0:
            try:
                runtime = "nvidia" in json.loads(out or "{}")
            except ValueError:
                runtime = "nvidia" in out
        else:
            error = err or "could not read the docker runtimes"
    return {"nvidia_runtime": runtime, "nvidia_smi": bool(_which("nvidia-smi")), "error": error}


def pick_service(force=None) -> str:
    """Which compose service to drive. `force` is "gpu"/"cpu" from --device."""
    if force == "gpu":
        return GPU_SERVICE
    if force == "cpu":
        return CPU_SERVICE
    return GPU_SERVICE if gpu_info()["nvidia_runtime"] else CPU_SERVICE


def compose_base(service: str):
    """The compose invocation for `service`, profile included when it needs one."""
    command = compose_command()
    if command is None:
        raise ServerCtlError(
            "docker compose was not found. Install Docker Engine with the compose plugin "
            "(scripts/install-docker.sh does it on Linux) and try again."
        )
    if service == CPU_SERVICE:
        return command + ["--profile", CPU_PROFILE]
    return list(command)


# ---------------------------------------------------------------------------
# .env — the only place the deployment's secret lives
# ---------------------------------------------------------------------------

ENV_PATH = os.path.join(REPO_ROOT, ".env")

# Keys docker-compose.yml interpolates. Anything else in the file is left alone.
_ENV_KEYS = ("API_TOKEN", "DEVICE", "BIND_ADDR", "HOST_PORT")
_ENV_BANNER = "# Written by scripts/server_ctl.py. This file is gitignored — keep it that way."


def read_env(path=ENV_PATH) -> dict:
    values = {}
    if not os.path.isfile(path):
        return values
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def write_env(updates: dict, path=ENV_PATH) -> None:
    """Merge `updates` into the .env, preserving every other line verbatim.

    Rewriting the file wholesale would drop whatever the operator added by
    hand — DEVICE overrides, an extra setting read by server/config.py — and
    doing that silently on an "Update" click is exactly the kind of surprise
    this file must not spring.
    """
    lines = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()

    remaining = dict(updates)
    output = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                output.append(f"{key}={remaining.pop(key)}")
                continue
        output.append(line)
    if remaining:
        if output and output[-1].strip():
            output.append("")
        if not any(line.startswith(_ENV_BANNER[:20]) for line in output):
            output.append(_ENV_BANNER)
        for key, value in remaining.items():
            output.append(f"{key}={value}")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(output).rstrip("\n") + "\n")
    # The API token is in here. Owner-only, best effort: a filesystem without
    # POSIX modes (a Windows bind mount) simply ignores it.
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def effective_port() -> int:
    """The host port this deployment publishes on.

    Read from the same `.env` compose interpolates, so `status` and `up` cannot
    disagree about where the server is — the environment wins for a one-off
    override, exactly as it does for compose itself.
    """
    raw = os.environ.get("HOST_PORT") or read_env().get("HOST_PORT")
    try:
        return int(raw) if raw else DEFAULT_PORT
    except ValueError:
        return DEFAULT_PORT


def url_for(port=None) -> str:
    return f"http://localhost:{port or effective_port()}"


def ensure_env(service: str, bind_addr: str, token=None, port=None) -> str:
    """Make sure the deployment has a token and knows where to bind. Returns the token.

    An existing token is kept: regenerating one on every `up` would silently
    lock out every client already configured against this server.
    """
    existing = read_env()
    api_token = token or existing.get("API_TOKEN") or secrets.token_urlsafe(32)
    updates = {"API_TOKEN": api_token, "BIND_ADDR": bind_addr, "HOST_PORT": str(port or effective_port())}
    if service == GPU_SERVICE:
        updates["DEVICE"] = existing.get("DEVICE") or "cuda"
    write_env(updates)
    return api_token


# ---------------------------------------------------------------------------
# The clone
# ---------------------------------------------------------------------------

def _git(args, timeout=120):
    return _capture(["git"] + args, cwd=REPO_ROOT, timeout=timeout)


def _upstream() -> str:
    """The ref this clone tracks: its configured upstream, else origin/<branch>."""
    rc, out, _err = _git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if rc == 0 and out:
        return out
    _rc, branch, _err = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    return f"origin/{branch or 'main'}"


def checkout_branch(branch: str) -> bool:
    """Move the clone onto `branch`, creating a tracking branch if needed.

    Returns whether anything moved. Refuses a dirty tree for the same reason
    `update` refuses to pull over one: a checkout that would discard someone's
    edits is not something a button gets to decide.
    """
    _rc, current, _err = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    if current == branch:
        return False

    _rc, dirty, _err = _git(["status", "--porcelain"])
    if dirty:
        raise ServerCtlError(
            f"The clone is on '{current}' and this deployment asks for '{branch}', but it has "
            f"uncommitted changes. Commit or discard them, then update again."
        )

    log(f"Switching the clone from '{current}' to '{branch}'...")
    _git(["fetch", "--quiet", "origin"], timeout=300)
    rc, _out, _err = _git(["rev-parse", "--verify", "--quiet", branch])
    if rc == 0:
        rc, _out, err = _git(["checkout", branch])
    else:
        rc, _out, err = _git(["checkout", "-b", branch, "--track", f"origin/{branch}"])
    if rc != 0:
        raise ServerCtlError(
            f"Could not switch the clone to '{branch}': {err or 'git refused'}. "
            f"Check that the branch exists on the remote."
        )
    return True


def clone_status(check_remote: bool = False, want_branch=None) -> dict:
    """What this clone is, and whether it has fallen behind its remote.

    `check_remote` is off by default because it needs the network: the status
    the panel refreshes on every visit must not hang for 30s on a machine
    that is offline. The Update button asks for it explicitly.
    """
    info = {
        "path": REPO_ROOT, "is_git_repo": False, "branch": None, "commit": None,
        "remote_url": None, "dirty": False, "ahead": 0, "behind": 0,
        "checked_remote": False, "error": None,
        # What the caller ASKED for, next to what is actually checked out. A
        # clone is only ever created once, so a deployment reconfigured onto
        # another branch afterwards would otherwise keep following the old one
        # in complete silence -- the worst kind of "my change had no effect".
        "configured_branch": want_branch, "branch_mismatch": False,
    }
    if not _which("git"):
        info["error"] = "git is not installed"
        return info
    rc, out, _err = _git(["rev-parse", "--is-inside-work-tree"])
    if rc != 0 or out != "true":
        info["error"] = f"{REPO_ROOT} is not a git clone (it was probably unpacked from an archive)"
        return info

    info["is_git_repo"] = True
    _rc, info["branch"], _err = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    _rc, info["commit"], _err = _git(["rev-parse", "--short", "HEAD"])
    _rc, info["remote_url"], _err = _git(["remote", "get-url", "origin"])
    info["branch_mismatch"] = bool(want_branch) and info["branch"] != want_branch
    _rc, dirty, _err = _git(["status", "--porcelain"])
    info["dirty"] = bool(dirty)

    if check_remote:
        rc, _out, err = _git(["fetch", "--quiet", "origin"], timeout=300)
        info["checked_remote"] = rc == 0
        if rc != 0:
            info["error"] = err or "could not reach the remote"
            return info

    upstream = _upstream()
    rc, out, _err = _git(["rev-list", "--left-right", "--count", f"HEAD...{upstream}"])
    if rc == 0 and out:
        parts = out.split()
        if len(parts) == 2:
            info["ahead"], info["behind"] = int(parts[0]), int(parts[1])
    elif not check_remote:
        # No local copy of the upstream ref yet — a fetch will produce one.
        info["error"] = f"no local record of {upstream}; run 'update' to fetch it"
    return info


# ---------------------------------------------------------------------------
# The container
# ---------------------------------------------------------------------------

def container_status(service: str) -> dict:
    """Whether `service`'s container exists and what state it is in."""
    info = {"service": service, "exists": False, "running": False, "state": None, "error": None}
    try:
        base = compose_base(service)
    except ServerCtlError as exc:
        info["error"] = str(exc)
        return info

    rc, out, err = _capture(base + ["ps", "-a", "-q", service], cwd=REPO_ROOT, timeout=60)
    if rc != 0:
        info["error"] = err or "docker compose ps failed"
        return info
    container = out.splitlines()[0].strip() if out else ""
    if not container:
        return info

    info["exists"] = True
    rc, state, err = _capture(["docker", "inspect", "--format", "{{.State.Status}}", container], timeout=30)
    if rc == 0:
        info["state"] = state
        info["running"] = state == "running"
    else:
        info["error"] = err or "docker inspect failed"
    return info


def port_in_use(port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> bool:
    """Whether anything is already listening where we would publish.

    A plain TCP connect, not a health check: the squatter is usually *another*
    server (a second clone, a hand-started container, someone's dev uvicorn),
    and it does not have to speak our protocol to hold the port.
    """
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def health(url=None, timeout: float = 5.0) -> bool:
    """GET /health. Never raises — an unreachable server just means "not up"."""
    url = url or url_for()
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=timeout) as response:
            if response.status != 200:
                return False
            return json.loads(response.read().decode("utf-8")).get("status") == "ok"
    except Exception:  # noqa: BLE001 - a probe answers False, it never raises
        # Deliberately broad: urlopen raises http.client.HTTPException (not an
        # OSError) when something that is not our server holds the port, and a
        # yes/no probe must never be able to abort the command around it.
        return False


def wait_for_health(url: str, timeout: int, service: str) -> bool:
    """Poll /health until it answers, narrating the wait.

    The narration is the point. A first start pulls a multi-GB image and then
    runs `pip install` inside the container; a caller that prints nothing for
    fifteen minutes is indistinguishable from one that has hung, and gets
    killed just before it would have worked.
    """
    deadline = time.monotonic() + timeout
    log(f"Waiting for {url}/health (up to {timeout // 60} min)...")
    while time.monotonic() < deadline:
        if health(url):
            log("The server is up.")
            return True
        state = container_status(service)
        # "restarting" belongs here with the dead states: `restart:
        # unless-stopped` turns a container that fails at boot into a loop, and
        # a loop never becomes healthy — without this the caller waited out the
        # full 30-minute timeout on a failure visible in three seconds.
        if state["exists"] and state["state"] in ("exited", "dead", "restarting"):
            log(f"The '{service}' container is not running ({state['state']}).")
            _rc, out, _err = _capture(
                compose_base(service) + ["logs", "--tail", "200", service], cwd=REPO_ROOT, timeout=60
            )
            if _DEPS_FATAL_MARKER in out:
                log(
                    "Its dependencies are not installed and pip could not reach its index. "
                    "This container needs network access once; connect and start it again."
                )
            log("Last log lines:")
            cmd_logs_tail(service, 40)
            return False
        time.sleep(HEALTH_POLL_SECONDS)
    log(f"Still no answer from {url}/health after {timeout}s. Last log lines:")
    cmd_logs_tail(service, 40)
    return False


_DEPS_SKIPPED_MARKER = "DEPENDENCY-INSTALL-SKIPPED"
_DEPS_FATAL_MARKER = "DEPENDENCY-INSTALL-FATAL"


def warn_if_deps_skipped(service: str) -> bool:
    """Say so when the container started without re-running its dependency install.

    Harmless in the offline case it exists for — everything was already
    installed — but it is the one state where a change to requirements.txt has
    NOT taken effect while the server looks perfectly healthy. That has to be
    visible rather than buried in `docker compose logs`.
    """
    rc, out, _err = _capture(
        compose_base(service) + ["logs", "--tail", "200", service], cwd=REPO_ROOT, timeout=60
    )
    if rc != 0 or _DEPS_SKIPPED_MARKER not in out:
        return False
    log(
        "NOTE: the dependency install did not run this start (no network?). The server is up "
        "on the packages already in its container. If you just changed requirements.txt, it "
        "has NOT taken effect — re-run 'update' with the network available."
    )
    return True


def cmd_logs_tail(service: str, lines: int) -> None:
    base = compose_base(service)
    _stream(base + ["logs", "--tail", str(lines), service], cwd=REPO_ROOT, prefix="  | ")


# ---------------------------------------------------------------------------
# DATA/ — what the manifest offers against what is on disk
# ---------------------------------------------------------------------------

def _load_fetch_data():
    """Import the download engine that lives next to this file."""
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)
    try:
        import fetch_data  # noqa: PLC0415 - deliberately local, see docstring
    except ImportError as exc:
        raise ServerCtlError(f"scripts/fetch_data.py could not be imported: {exc}") from None
    return fetch_data


def data_dir() -> str:
    return os.path.join(REPO_ROOT, "DATA")


def ensure_data_dir() -> str:
    """Create DATA/ as the invoking user, BEFORE docker can create it as root.

    `./DATA:/data:ro` is a bind mount with `create_host_path: true`, so a
    missing host path is created by the docker DAEMON — owned by root. Every
    later `models` download then dies on "Permission denied" against the very
    directory the server reads, on a brand-new install, for a reason nothing
    on screen explains. Creating it first is the entire fix; the check below is
    for an install where docker already won that race.
    """
    root = data_dir()
    try:
        os.makedirs(root, exist_ok=True)
    except OSError as exc:
        raise ServerCtlError(f"Could not create {root}: {exc}") from None
    if not os.access(root, os.W_OK):
        raise ServerCtlError(
            f"{root} exists but this user cannot write to it — docker created it as root "
            f"before anything else did. Fix it once with:\n\n"
            f"    sudo chown -R $(id -u):$(id -g) {root}"
        )
    return root


def catalog() -> dict:
    """Per tool: what the manifest offers, and how much of it is already here.

    This is what makes a partial install legible. `missing_size` is the honest
    figure — what a download would actually transfer — and it is what the
    client shows, because "AMASSS: 1.5 GB" next to an already-complete AMASSS
    is the number that makes someone skip a tool they could have for free.
    """
    fetch_data = _load_fetch_data()
    try:
        manifest = fetch_data._parse_manifest(fetch_data._DEFAULT_MANIFEST)
    except fetch_data.ManifestError as exc:
        raise ServerCtlError(str(exc)) from None

    root = data_dir()
    tools = []
    for name in sorted(manifest):
        tool = {"name": name, "size": 0, "missing_size": 0, "entries": 0, "present": 0}
        for kind in fetch_data.KINDS:
            entries = manifest[name].get(kind, [])
            counts = {"entries": len(entries), "present": 0, "size": 0, "missing": 0, "missing_size": 0}
            for entry in entries:
                size = entry.get("size") or 0
                counts["size"] += size
                target = fetch_data._target_path(root, {**entry, "tool": name, "kind": kind})
                if os.path.exists(target):
                    counts["present"] += 1
                else:
                    counts["missing"] += 1
                    counts["missing_size"] += size
            tool[kind] = counts
            tool["size"] += counts["size"]
            tool["missing_size"] += counts["missing_size"]
            tool["entries"] += counts["entries"]
            tool["present"] += counts["present"]
        tool["complete"] = tool["entries"] > 0 and tool["present"] == tool["entries"]
        tool["partial"] = 0 < tool["present"] < tool["entries"]
        tools.append(tool)

    try:
        free = shutil.disk_usage(REPO_ROOT).free
    except OSError:
        free = None
    return {
        "data_dir": root,
        "disk_free": free,
        "total_size": sum(t["size"] for t in tools),
        "total_missing_size": sum(t["missing_size"] for t in tools),
        "tools": tools,
    }


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_status(args) -> dict:
    url = args.url or url_for()
    service = pick_service(args.device)
    clone = clone_status(check_remote=args.check_remote, want_branch=args.branch)
    env = read_env()
    return {
        "repo_root": REPO_ROOT,
        "git": git_info(),
        "docker": docker_info(),
        "compose": compose_info(),
        "gpu": gpu_info(),
        "service": service,
        "clone": clone,
        "container": container_status(service),
        "server": {"url": url, "healthy": health(url)},
        # Deliberately no token here: a status dump lands in log panes and bug
        # reports. `server_ctl.py token` is the one way to read it.
        "env": {
            "path": ENV_PATH,
            "exists": os.path.isfile(ENV_PATH),
            "has_token": bool(env.get("API_TOKEN")),
            "bind_addr": env.get("BIND_ADDR"),
            "device": env.get("DEVICE"),
        },
        "data_dir": data_dir(),
    }


def cmd_token(_args) -> dict:
    token = read_env().get("API_TOKEN")
    if not token:
        raise ServerCtlError(
            f"No API_TOKEN in {ENV_PATH}. Run 'server_ctl.py up' — it generates one."
        )
    return {"token": token}


def _preflight(service: str, port=None) -> None:
    docker = docker_info()
    if not docker["available"]:
        raise ServerCtlError(
            "Docker is not installed. On Linux: sudo sh scripts/install-docker.sh\n"
            "On macOS/Windows: install Docker Desktop from https://docs.docker.com/get-docker/"
        )
    if not docker["daemon"]:
        raise ServerCtlError(f"Docker is installed but not usable: {docker['error']}")
    if compose_command() is None:
        raise ServerCtlError(
            "The docker compose plugin is missing. On Linux: sudo sh scripts/install-docker.sh"
        )
    if service == GPU_SERVICE and not gpu_info()["nvidia_runtime"]:
        raise ServerCtlError(
            "The GPU service was requested but docker has no 'nvidia' runtime, so the "
            "container could not start at all. Install the NVIDIA Container Toolkit, or "
            "run with --device cpu."
        )

    # Both checks below are about BINDING, so both are skipped when `port` is
    # None -- which is how `down` asks for the prerequisite checks without
    # being refused permission to stop the very container being complained
    # about.
    #
    # The two services publish the same port, so starting one while the other
    # runs fails on the bind -- and "address already in use" buried in compose
    # output reads as a broken machine rather than as "your server is already
    # running, under its other name".
    other = CPU_SERVICE if service == GPU_SERVICE else GPU_SERVICE
    if port and container_status(other)["running"]:
        raise ServerCtlError(
            f"The '{other}' container is already running and holds port {port}. "
            f"Stop it first (server_ctl.py down --device "
            f"{'gpu' if other == GPU_SERVICE else 'cpu'}), or keep using it."
        )

    # ... and the squatter is just as often something this compose project
    # cannot see at all: a SECOND CLONE of this repository is its own compose
    # project, so `compose ps` here reports nothing while its container holds
    # the port. Checked by connecting, since whatever owns it need not speak
    # our protocol. Skipped when our own container is the one running --
    # compose stops it before starting its replacement.
    if port and not container_status(service)["running"] and port_in_use(port):
        raise ServerCtlError(
            f"Something is already listening on port {port} of this machine, and it is "
            f"not this deployment's container. Two servers cannot publish the same port.\n"
            f"If it is another clone of this repository, stop it from there "
            f"(python3 scripts/server_ctl.py down); otherwise stop whatever holds the port."
        )


def cmd_up(args) -> dict:
    service = pick_service(args.device)
    port = args.port or effective_port()
    _preflight(service, port)
    ensure_data_dir()
    token = ensure_env(service, args.bind, token=args.token, port=port)
    # After ensure_env, so a --port lands in .env before the URL is derived
    # from it -- compose and the health check must not read different ports.
    url = args.url or url_for()

    if service == CPU_SERVICE:
        log("No nvidia runtime in docker: starting the CPU service. Everything works, slowly.")

    command = compose_base(service) + ["up", "-d"]
    if args.force_recreate:
        # Not cosmetic: the container installs requirements.txt as part of its
        # *command*, into a writable layer that survives `restart`. Only a
        # fresh container re-resolves them. See CLAUDE.md, 2026-07-31.
        command.append("--force-recreate")
    command.append(service)
    if _stream(command, cwd=REPO_ROOT) != 0:
        raise ServerCtlError(f"'docker compose up' failed for service '{service}'.")

    healthy = wait_for_health(url, args.timeout, service) if args.wait else False
    deps_skipped = warn_if_deps_skipped(service) if healthy else False
    return {
        "deps_install_skipped": deps_skipped,
        "service": service,
        "url": url,
        "healthy": healthy,
        "token": token,
        "data_dir": data_dir(),
    }


def cmd_update(args) -> dict:
    """Fetch, fast-forward if there is something to fast-forward to, relaunch.

    "Relaunch" is `up -d --force-recreate` rather than `restart` for the reason
    in CLAUDE.md: the container pip-installs requirements.txt in its command,
    into a layer `restart` keeps. A new requirements.txt that is never
    re-resolved is a silent no-op update, which is worse than a failed one.
    """
    service = pick_service(args.device)
    port = effective_port()
    url = args.url or url_for(port)

    result = {"service": service, "url": url, "pulled": False, "recreated": False,
              "healthy": False, "switched_branch": False}

    # --- the code half: pure git ------------------------------------------
    # Deliberately BEFORE the docker preflight. Updating the clone needs
    # neither a working docker nor a free port, and those are exactly the
    # things a user may be updating in order to fix — refusing to fetch new
    # code because port 8000 is busy is the tool getting in its own way.
    if args.branch:
        # Before the drift check, not after: "behind" is meaningless while the
        # clone is still on another branch than the one being deployed.
        result["switched_branch"] = checkout_branch(args.branch)
    clone = clone_status(check_remote=True, want_branch=args.branch)
    result["clone"] = clone

    if clone["is_git_repo"] and clone["checked_remote"]:
        if clone["dirty"] and clone["behind"]:
            raise ServerCtlError(
                f"This clone has uncommitted changes and is {clone['behind']} commit(s) behind "
                f"{_upstream()}. Refusing to pull over local edits — commit or discard them first."
            )
        if clone["behind"]:
            log(f"{clone['behind']} new commit(s) on {_upstream()}. Fast-forwarding...")
            if _stream(["git", "pull", "--ff-only"], cwd=REPO_ROOT) != 0:
                raise ServerCtlError(
                    "'git pull --ff-only' failed. The local branch has probably diverged; "
                    "resolve it by hand in the clone."
                )
            result["pulled"] = True
        else:
            log("The clone is already up to date.")
    elif clone["error"]:
        log(f"Skipping the code update: {clone['error']}")

    # --- the container half -----------------------------------------------
    _preflight(service, port)
    up_to_date = not result["pulled"] and not result["switched_branch"]
    running = container_status(service)["running"]
    if up_to_date and running and health(url) and not args.force:
        log("Nothing to update and the server is answering. Leaving it alone.")
        result["healthy"] = True
        return result

    # The image tag is pinned in docker-compose.yml, so this is a no-op unless
    # the pull above moved it — but it is what makes a tag bump take effect.
    _stream(compose_base(service) + ["pull", service], cwd=REPO_ROOT)
    result["pulled_image"] = True

    ensure_data_dir()
    ensure_env(service, args.bind)
    if _stream(compose_base(service) + ["up", "-d", "--force-recreate", service], cwd=REPO_ROOT) != 0:
        raise ServerCtlError(f"'docker compose up --force-recreate' failed for service '{service}'.")
    result["recreated"] = True
    result["healthy"] = wait_for_health(url, args.timeout, service)
    if result["healthy"]:
        result["deps_install_skipped"] = warn_if_deps_skipped(service)
    return result


def cmd_down(args) -> dict:
    service = pick_service(args.device)
    # No port argument: stopping never binds anything, so the conflict check
    # would refuse to stop the very container it is complaining about.
    _preflight(service, port=None)
    # `stop` rather than `down`: down would also remove the network and, on a
    # future compose file with volumes, invite --volumes. Stopping is what the
    # panel's button means.
    rc = _stream(compose_base(service) + ["stop", service], cwd=REPO_ROOT)
    return {"service": service, "stopped": rc == 0}


def cmd_logs(args) -> dict:
    service = pick_service(args.device)
    cmd_logs_tail(service, args.lines)
    return {"service": service, "lines": args.lines}


def cmd_catalog(_args) -> dict:
    return catalog()


def cmd_models(args) -> dict:
    """Download the selected tools' data into DATA/.

    Delegated to fetch_data.py as a subprocess rather than imported: it prints
    its own progress, and streaming that straight into the caller's log pane is
    the whole point. It also skips whatever is already on disk, which is what
    makes coming back later to add one more tool cost only that tool.
    """
    fetch = os.path.join(_SCRIPT_DIR, "fetch_data.py")
    if not os.path.isfile(fetch):
        raise ServerCtlError(f"scripts/fetch_data.py not found next to {__file__}.")

    known = {tool["name"] for tool in catalog()["tools"]}
    unknown = sorted(set(args.tool or []) - known)
    if unknown:
        raise ServerCtlError(
            f"No such tool in the manifest: {', '.join(unknown)}. Known: {', '.join(sorted(known))}"
        )

    command = [sys.executable, fetch, "--data-dir", ensure_data_dir(), "--progress", "always"]
    for kind in args.kind or ["models", "testfiles"]:
        command += ["--kind", kind]
    for tool in args.tool or []:
        command += ["--tool", tool]
    if args.force:
        command.append("--force")

    rc = _stream(command, cwd=REPO_ROOT)
    after = catalog()
    return {
        "returncode": rc,
        "tools": args.tool or [tool["name"] for tool in after["tools"]],
        "catalog": after,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _human(size) -> str:
    return _load_fetch_data()._human(size)


def _print_status(status: dict) -> None:
    def mark(ok):
        return "ok " if ok else "-- "

    print(f"Repository   {status['repo_root']}")
    print(f"  {mark(status['git']['available'])}git       {status['git']['version'] or 'not installed'}")
    docker = status["docker"]
    print(f"  {mark(docker['daemon'])}docker    {docker['version'] or 'not installed'}"
          f"{'' if docker['daemon'] else '  (' + str(docker['error']) + ')'}")
    compose = status["compose"]
    print(f"  {mark(compose['available'])}compose   {compose['version'] or 'not installed'}")
    gpu = status["gpu"]
    print(f"  {mark(gpu['nvidia_runtime'])}gpu       "
          f"{'nvidia runtime available' if gpu['nvidia_runtime'] else 'no nvidia runtime in docker'}")

    clone = status["clone"]
    if clone["is_git_repo"]:
        drift = "up to date" if not clone["behind"] else f"{clone['behind']} commit(s) behind"
        if not clone["checked_remote"]:
            drift += " (against the last fetch)"
        print(f"\nClone        {clone['branch']}@{clone['commit']}  {drift}"
              f"{'  [uncommitted changes]' if clone['dirty'] else ''}")
        if clone.get("branch_mismatch"):
            print(f"             ! this deployment asks for '{clone['configured_branch']}'. "
                  f"'update' will switch the clone onto it.")
    else:
        print(f"\nClone        {clone['error']}")

    container = status["container"]
    print(f"Service      {status['service']}  "
          f"{container['state'] or 'no container yet'}")
    print(f"Health       {status['server']['url']}  "
          f"{'answering' if status['server']['healthy'] else 'no answer'}")
    print(f"Token        {'set in .env' if status['env']['has_token'] else 'not generated yet'}")


def _print_catalog(data: dict) -> None:
    free = "" if data["disk_free"] is None else f", {_human(data['disk_free'])} free on this disk"
    print(f"{data['data_dir']}")
    print(f"{_human(data['total_size'])} in the manifest, "
          f"{_human(data['total_missing_size'])} still to download{free}")
    print()
    for tool in data["tools"]:
        state = "complete" if tool["complete"] else ("partial" if tool["partial"] else "missing")
        print(f"  {tool['name']:<16} {state:<9} "
              f"{tool['present']}/{tool['entries']} item(s)  "
              f"{_human(tool['size']):>10} total, {_human(tool['missing_size']):>10} to fetch")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start, update and provision this inference server.",
        epilog="Progress goes to stderr; --json prints one object on stdout.",
    )
    parser.add_argument("--json", action="store_true", help="Print the result as JSON on stdout.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub, with_url=True):
        sub.add_argument(
            "--device", choices=("auto", "gpu", "cpu"), default="auto",
            help="Which compose service to drive. Default: gpu when docker has an nvidia runtime.",
        )
        sub.add_argument(
            "--branch",
            help="The branch this deployment should be on. `status` reports a mismatch; "
                 "`update` switches the clone onto it. Default: leave the clone alone.",
        )
        if with_url:
            sub.add_argument("--url", default=None,
                             help="Where to health-check. Default: http://localhost:<HOST_PORT>.")

    status = subparsers.add_parser("status", help="Report every prerequisite, the clone, and the container.")
    add_common(status)
    status.add_argument(
        "--check-remote", action="store_true",
        help="git fetch first, so 'behind' is current. Needs the network.",
    )
    status.set_defaults(func=cmd_status, printer=_print_status)

    token = subparsers.add_parser("token", help="Print this deployment's API token.")
    token.set_defaults(func=cmd_token, printer=lambda result: print(result["token"]))

    up = subparsers.add_parser("up", help="Generate .env if needed and start the server.")
    add_common(up)
    up.add_argument("--token", help="Use this API token instead of generating/keeping one.")
    up.add_argument(
        "--bind", default="127.0.0.1",
        help="Host address the port is published on. Default: 127.0.0.1 — this deployment "
             "speaks plain HTTP, so it stays on loopback. Pass an empty string to publish on "
             "every interface (IPv4 and IPv6), which is only acceptable behind a TLS "
             "terminator; '0.0.0.0' does the same for IPv4 only.",
    )
    up.add_argument(
        "--port", type=int, default=None,
        help="Host port to publish on. Remembered in .env; only needed when something else "
             "already holds the default (8000). The container always serves 8000 internally.",
    )
    up.add_argument("--force-recreate", action="store_true", help="Recreate the container from scratch.")
    up.add_argument("--no-wait", dest="wait", action="store_false", help="Return without waiting for /health.")
    up.add_argument("--timeout", type=int, default=DEFAULT_STARTUP_TIMEOUT, help="Seconds to wait for /health.")
    up.set_defaults(func=cmd_up, printer=None)

    update = subparsers.add_parser("update", help="Pull new commits and relaunch if anything changed.")
    add_common(update)
    update.add_argument("--bind", default="127.0.0.1", help="See 'up --bind'.")
    update.add_argument("--force", action="store_true", help="Recreate even when nothing changed.")
    update.add_argument("--timeout", type=int, default=DEFAULT_STARTUP_TIMEOUT, help="Seconds to wait for /health.")
    update.set_defaults(func=cmd_update, printer=None)

    down = subparsers.add_parser("down", help="Stop the server container.")
    add_common(down, with_url=False)
    down.set_defaults(func=cmd_down, printer=None)

    logs = subparsers.add_parser("logs", help="Show the last lines of the server log.")
    add_common(logs, with_url=False)
    logs.add_argument("-n", "--lines", type=int, default=100)
    logs.set_defaults(func=cmd_logs, printer=None)

    cat = subparsers.add_parser("catalog", help="Per tool: manifest size, and how much is already on disk.")
    cat.set_defaults(func=cmd_catalog, printer=_print_catalog)

    models = subparsers.add_parser("models", help="Download the selected tools' data into DATA/.")
    models.add_argument("--tool", action="append", help="Restrict to this tool (repeatable). Default: all.")
    models.add_argument("--kind", action="append", choices=("models", "testfiles"), help="Default: both.")
    models.add_argument("--force", action="store_true", help="Re-download even what is present.")
    models.set_defaults(func=cmd_models, printer=None)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # Subcommands that take no --url/--device still go through code paths that
    # read them; give them the defaults rather than sprinkling getattr() around.
    for name, default in (("url", None), ("device", "auto"), ("bind", "127.0.0.1"), ("port", None),
                          ("check_remote", False), ("timeout", DEFAULT_STARTUP_TIMEOUT), ("branch", None),
                          ("force", False), ("force_recreate", False), ("wait", True)):
        if not hasattr(args, name):
            setattr(args, name, default)
    if getattr(args, "device", "auto") == "auto":
        args.device = None

    try:
        result = args.func(args)
    except ServerCtlError as exc:
        log(f"error: {exc}")
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        return 1
    except Exception as exc:  # noqa: BLE001 - deliberate, see below
        # An UNEXPECTED failure must still travel. Letting it escape printed a
        # traceback on stderr and exited 1 with an empty stdout, and the GUI
        # -- which reads stdout for the result -- could then say nothing better
        # than "exit code 1". The traceback still goes to stderr for the log;
        # this is what reaches the user's dialog.
        import traceback
        log(traceback.format_exc())
        if args.json:
            print(json.dumps({
                "error": f"{type(exc).__name__}: {exc}",
                "unexpected": True,
            }, indent=2))
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    elif args.printer:
        args.printer(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
