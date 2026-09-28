#!/usr/bin/env python3
"""Bring this server up, keep it current, and choose what lands in DATA/.

This is the engine behind scripts/setup-server.sh, and it is what the Slicer
"Slicer Cloud" module drives: every button in that panel is one subcommand
here. It manages the clone it lives in - `scripts/server_ctl.py` sitting one
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
import getpass
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
    a non-zero exit - callers here decide what a failure means."""
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
        # Reported as a FLAG, not only as prose. This is the one failure here
        # with a fixed, one-command remedy, and a GUI has to be able to tell it
        # apart from "the daemon is down" without matching on English text --
        # the panel answers the two with completely different screens.
        needs_group = "permission denied" in message.lower()
        if needs_group:
            message = (
                f"the docker daemon refused this user. Add yourself to the 'docker' group:\n"
                f"    sudo usermod -aG docker {getpass.getuser()}\n"
                "then log out and back in -- a group only applies to a NEW session."
            )
        return {"available": True, "version": version, "daemon": False,
                "error": message, "needs_group": needs_group}
    return {"available": True, "version": version, "daemon": True,
            "error": None, "needs_group": False}


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


def _has_nvidia_cdi_device(raw: str) -> bool:
    """True when docker has discovered a CDI device for an nvidia GPU.

    The entries look like `{"Source": "cdi", "ID": "nvidia.com/gpu=all"}`. The
    vendor prefix is what identifies them: the same list carries every other
    CDI vendor registered on the host.
    """
    try:
        devices = json.loads(raw or "null")
    except ValueError:
        return False
    if not isinstance(devices, list):
        return False
    return any(
        isinstance(device, dict)
        and device.get("Source") == "cdi"
        and str(device.get("ID", "")).startswith("nvidia.com/gpu")
        for device in devices
    )


def gpu_info() -> dict:
    """Whether docker can actually hand a container an nvidia device.

    `nvidia-smi` on the host is not the answer: the container toolkit is a
    separate install, and without it the GPU service fails to start on a
    machine whose card works perfectly outside docker.

    TWO mechanisms answer "yes", and checking only the first was a bug. The
    legacy one is a runtime named "nvidia" in daemon.json, which is what
    `nvidia-ctk runtime configure` writes. The current one is CDI: docker >= 25
    reads /etc/cdi and /var/run/cdi and resolves `nvidia.com/gpu` devices with
    no runtime registered at all -- and the toolkit has generated those specs
    by itself since 1.17, so a host set up today commonly runs GPU containers
    perfectly while `docker info` lists only runc.

    Measured on exactly such a host (RTX 6000 Ada, driver 580, docker 29,
    toolkit 1.18, no /etc/docker/daemon.json): `docker run --gpus all`, `docker
    run --device nvidia.com/gpu=all` and a compose `driver: nvidia` reservation
    -- the one this repository's `inference` service uses -- all reach the card,
    while the runtime-only check reported "no GPU". `pick_service` therefore
    started `inference-cpu`, and the panel told the user their card was unusable.

    `nvidia_runtime` keeps its name: every caller means "can docker give a
    container the card", which is what it still answers. `gpu_access` names the
    mechanism that replied, so nothing has to print "no nvidia runtime" at
    someone whose GPU works.
    """
    runtime = False
    cdi = False
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
        # A second call, and a failure here is deliberately NOT an error:
        # `.DiscoveredDevices` does not exist before docker 28, where an unknown
        # field makes the whole template fail rather than return nothing. A host
        # too old to have CDI must read as "no CDI", not as "could not check".
        rc, out, _err = _capture(
            ["docker", "info", "--format", "{{json .DiscoveredDevices}}"], timeout=30)
        if rc == 0:
            cdi = _has_nvidia_cdi_device(out)
    return {
        "nvidia_runtime": runtime or cdi,
        "nvidia_smi": bool(_which("nvidia-smi")),
        "gpu_access": "runtime" if runtime else ("cdi" if cdi else None),
        "error": error,
    }


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
# .env - the only place the deployment's secret lives
# ---------------------------------------------------------------------------

ENV_PATH = os.path.join(REPO_ROOT, ".env")

# Keys docker-compose.yml interpolates. Anything else in the file is left alone.
_ENV_KEYS = ("API_TOKEN", "DEVICE", "BIND_ADDR", "HOST_PORT")
_ENV_BANNER = "# Written by scripts/server_ctl.py. This file is gitignored - keep it that way."


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
    hand - DEVICE overrides, an extra setting read by server/config.py - and
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
    disagree about where the server is - the environment wins for a one-off
    override, exactly as it does for compose itself.
    """
    raw = os.environ.get("HOST_PORT") or read_env().get("HOST_PORT")
    try:
        return int(raw) if raw else DEFAULT_PORT
    except ValueError:
        return DEFAULT_PORT


def url_for(port=None, bind_addr=None) -> str:
    """Where this deployment answers, seen from the machine running it.

    `localhost` is right when the port is published on loopback or on EVERY
    interface. It is wrong when it is published on one specific address, which
    is exactly what `--bind <vpn address>` is for: the port is then simply not
    on 127.0.0.1, so a health check against localhost polls an address nothing
    listens on and reports a perfectly healthy server as dead.

    Read from `.env` rather than passed around, so every caller agrees without
    having to thread the value through. `cmd_up` calls `ensure_env` first, so
    what is read here is already the binding the container is about to get.
    """
    if bind_addr is None:
        bind_addr = read_env().get("BIND_ADDR", "127.0.0.1")
    host = "localhost"
    # "" and 0.0.0.0/:: all mean "every interface", and loopback is included in
    # that -- so localhost stays the right thing to talk to.
    if bind_addr and bind_addr not in ("127.0.0.1", "localhost", "0.0.0.0", "::"):
        host = f"[{bind_addr}]" if ":" in bind_addr else bind_addr
    return f"http://{host}:{port or effective_port()}"


def ensure_env(service: str, bind_addr=None, token=None, port=None) -> str:
    """Make sure the deployment has a token and knows where to bind. Returns the token.

    An existing token is kept: regenerating one on every `up` would silently
    lock out every client already configured against this server.

    `bind_addr` is kept the same way, and for the same kind of reason. It used
    to default to "127.0.0.1" at the ARGPARSE level, so `update` -- which never
    asks anyone where to bind -- rewrote BIND_ADDR to localhost on every run.
    A deployment deliberately published on a network address went back to being
    unreachable the first time someone pressed "Update server", with nothing
    said about it. Passing None now means "leave it as it is".

    `is not None` rather than a falsy test, because the EMPTY STRING is a real
    and different value here: it is how the compose file is told to publish on
    every interface (see its `${BIND_ADDR:+...}` comment), so "" and unset must
    not collapse into the same branch.
    """
    existing = read_env()
    api_token = token or existing.get("API_TOKEN") or secrets.token_urlsafe(32)
    if bind_addr is None:
        bind_addr = existing.get("BIND_ADDR", "127.0.0.1")
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
        # No local copy of the upstream ref yet - a fetch will produce one.
        info["error"] = f"no local record of {upstream}; run 'update' to fetch it"
    return info


# ---------------------------------------------------------------------------
# What a remote is offering, and what it would cost to take it
# ---------------------------------------------------------------------------

# A pull delivers SOURCE. It cannot deliver a dependency, because the
# virtualenvs a tool runs in are built into the image and are never touched by
# git -- `docker-compose.dev.yml` mounts `src/` and deliberately leaves the
# `.venv` beside it alone. So a change to any of these paths means the image's
# environments can no longer satisfy the source that would land next to them,
# and pulling it would produce a deployment that is broken in a way no log line
# explains: a tool importing a package that is not installed.
#
# Refusing is the whole point. "Updated, and every run of that tool now fails"
# is worse than "not updated, and here is why".
_NEEDS_IMAGE_SUFFIXES = ("/pyproject.toml", "/uv.lock")
_NEEDS_IMAGE_EXACT = ("pyproject.toml", "uv.lock")
_NEEDS_IMAGE_PREFIXES = ("docker/", "server/requirements")


def remote_head(ref: str, remote: str = "origin", timeout: int = 60):
    """The sha at the tip of `ref` on the remote, fetching nothing.

    `git ls-remote` rather than the GitHub API on purpose. The API allows 60
    unauthenticated requests an hour, which caps polling at once a minute and
    would need a token on every clinic machine; `ls-remote` speaks the git
    protocol, has no such quota, and measured 0.21-0.56 s against the real
    remotes. It also works for any git host, which the API does not.

    Returns None rather than raising: a poll that cannot reach the network is
    "nothing new to say", not an error worth stopping a loop over.
    """
    rc, out, _err = _git(["ls-remote", remote, f"refs/heads/{ref}"], timeout=timeout)
    if rc != 0 or not out:
        return None
    return out.split()[0]


def changed_paths(old: str, new: str) -> list:
    """Every path that differs between two commits, as git reports them.

    Both must be present locally, so this runs after a fetch and before a
    pull -- which is exactly the window in which the decision below has to be
    made.
    """
    rc, out, _err = _git(["diff", "--name-only", f"{old}..{new}"])
    if rc != 0:
        return []
    return [line for line in out.splitlines() if line.strip()]


def needs_image(paths) -> list:
    """The changed paths a pull cannot deliver, or [] when a pull is enough.

    Returned rather than answered yes/no so the refusal can NAME what it is
    refusing over. "An image is needed" sends someone reading a diff; "an image
    is needed because tools/AMASSS/uv.lock changed" sends them to the commit.
    """
    blockers = []
    for entry in paths:
        path = entry.strip()
        if (
            path in _NEEDS_IMAGE_EXACT
            or path.endswith(_NEEDS_IMAGE_SUFFIXES)
            or path.startswith(_NEEDS_IMAGE_PREFIXES)
        ):
            blockers.append(path)
    return blockers


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
    """GET /health. Never raises - an unreachable server just means "not up"."""
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
        # a loop never becomes healthy - without this the caller waited out the
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
            if _CONFIG_AHEAD_MARKER in out:
                # The one boot failure whose obvious repair is the wrong one.
                # deployment.toml is mounted from this checkout, so it can be
                # newer than the server in the image -- and the traceback below
                # reads as "your file is wrong", which invites deleting the line
                # that a newer server needs. Said here in one sentence, above
                # the trace, because that is what gets read.
                log(
                    "Its deployment.toml asks for something this image's server does not know: "
                    "the file is mounted from this checkout and has moved ahead of the image. "
                    "Rebuild the image (`docker compose --profile venvs build inference-venvs`) "
                    "rather than editing the file back."
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
# registry.deployment.CONFIG_AHEAD_MARKER. Written down rather than imported:
# this script runs in Slicer's interpreter too, where the server package is not
# importable. A test pins the two strings to each other.
_CONFIG_AHEAD_MARKER = "DEPLOYMENT-CONFIG-AHEAD-OF-SERVER"


def warn_if_deps_skipped(service: str) -> bool:
    """Say so when the container started without re-running its dependency install.

    Harmless in the offline case it exists for - everything was already
    installed - but it is the one state where a change to requirements.txt has
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
        "has NOT taken effect - re-run 'update' with the network available."
    )
    return True


def cmd_logs_tail(service: str, lines: int) -> None:
    base = compose_base(service)
    _stream(base + ["logs", "--tail", str(lines), service], cwd=REPO_ROOT, prefix="  | ")


# ---------------------------------------------------------------------------
# DATA/ - what the manifest offers against what is on disk
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
    missing host path is created by the docker DAEMON - owned by root. Every
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
            f"{root} exists but this user cannot write to it - docker created it as root "
            f"before anything else did. Fix it once with:\n\n"
            f"    sudo chown -R $(id -u):$(id -g) {root}"
        )
    return root


def catalog() -> dict:
    """Per tool: what the manifest offers, and how much of it is already here.

    This is what makes a partial install legible. `missing_size` is the honest
    figure - what a download would actually transfer - and it is what the
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

# ---------------------------------------------------------------------------
# The update gate
# ---------------------------------------------------------------------------

# Whether this deployment follows its branch at all, and how far it goes.
#
#   off     the default, and deliberately so. A machine holding patient
#           imaging does not start executing code from the internet because
#           somebody installed a timer; turning this on is a decision taken
#           per deployment, by whoever is accountable for that machine.
#   notify  poll and say what would happen. What a site runs while it decides
#           whether to trust the mechanism.
#   apply   drain, pull and restart.
#
# Read from the environment first so a systemd unit or a container can set it
# without editing a file, then from the .env `up` already writes and `update`
# already preserves line by line.
AUTO_UPDATE_OFF = "off"
AUTO_UPDATE_NOTIFY = "notify"
AUTO_UPDATE_APPLY = "apply"
AUTO_UPDATE_MODES = (AUTO_UPDATE_OFF, AUTO_UPDATE_NOTIFY, AUTO_UPDATE_APPLY)


def auto_update_mode() -> str:
    """`off` / `notify` / `apply`, from the environment or the .env."""
    raw = os.environ.get("SADT_AUTO_UPDATE") or read_env().get("SADT_AUTO_UPDATE") or ""
    mode = raw.strip().lower()
    if mode not in AUTO_UPDATE_MODES:
        if mode:
            log(f"SADT_AUTO_UPDATE is '{raw}', which is not one of "
                f"{', '.join(AUTO_UPDATE_MODES)}. Treating it as '{AUTO_UPDATE_OFF}'.")
        return AUTO_UPDATE_OFF
    return mode


# How long the server is asked to hold the door for at a time. It reopens by
# itself after this, so the number is a DEADMAN and not a plan: the updater
# renews it while it works, and an updater that is killed leaves a clinic
# waiting this long rather than until somebody notices.
GATE_LEASE_SECONDS = 60

# How often the drain asks whether the server has gone idle. The Slicer
# dashboard already polls at two seconds, so this adds nothing a running server
# does not field anyway.
DRAIN_POLL_SECONDS = 2

# A run nothing has said anything about for this long is one whose client
# vanished; the server's own reaper owns it and the drain must not sit behind
# it. Mirrors RUN_TTL_SECONDS in server/config.py -- if that moves, this is the
# other half that has to move with it.
GHOST_AFTER_SECONDS = 900

# States in which a run is still going to want the machine. From
# runs._STATE_OF_PHASE: received/staging/queued_gpu are "pending", and
# running/packaging are "running". done, failed, cancelled and paused are not
# here, and paused is the interesting absence -- see `wait_until_idle`.
BUSY_STATES = ("pending", "running")


def _api(url: str, token: str, path: str, payload=None, timeout: float = 10.0):
    """One JSON request to the server, or None if it could not be made.

    None is deliberately not distinguished from an error: every caller below
    treats "I could not ask" and "the answer was no" the same way, which is to
    not proceed with an update. Raising here would only move that decision.
    """
    request = urllib.request.Request(
        f"{url.rstrip('/')}{path}",
        headers={"Authorization": f"Bearer {token}"},
    )
    if payload is not None:
        request.data = json.dumps(payload).encode("utf-8")
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # An older server, with no gate. The updater and the server ship in
            # the same pull, so the FIRST run of a new updater always talks to
            # an old server -- this is the expected case once, not a fault.
            return {"unsupported": True}
        return None
    except Exception:  # noqa: BLE001 - a probe answers None, it never raises
        return None


def set_accepting(url: str, token: str, accepting: bool,
                  seconds: int = GATE_LEASE_SECONDS, reason: str = ""):
    """Open or close the server's door. Returns its answer, or None."""
    return _api(url, token, "/maintenance",
                {"accepting": accepting, "seconds": seconds, "reason": reason})


def server_load(url: str, token: str):
    """What is still holding the machine, or None if the server did not answer.

    Three sources, because no one of them is complete:

    * `admission.running` and `waiting` miss a run that is still STAGING its
      inputs -- the reservation is taken after the upload is written to disk,
      so a cohort being streamed reads as zero;
    * `runs` misses a run sent without an `X-Run-Id`, which creates no run
      directory at all (curl, the test suite and the benchmarks do this);
    * so both are read, and the union is what has to go quiet.
    """
    status = _api(url, token, "/status")
    if not status or status.get("unsupported"):
        return None
    admission = status.get("admission") or {}
    now = time.time()
    busy = []
    for run in status.get("runs") or []:
        if run.get("state") not in BUSY_STATES:
            continue
        updated = run.get("updated_at") or 0
        if updated and now - updated > GHOST_AFTER_SECONDS:
            # The client went away; the server's reaper owns this one.
            continue
        busy.append(run)
    return {
        "running": admission.get("running", 0),
        "waiting": admission.get("waiting", 0),
        "busy": busy,
        "paused": [r for r in (status.get("runs") or []) if r.get("state") == "paused"],
        "accepting": (status.get("maintenance") or {}).get("accepting", True),
    }


def _describe(busy) -> str:
    return ", ".join(
        f"{run.get('tool') or 'a run'} ({run.get('phase') or run.get('state')})"
        for run in busy[:4]
    ) or "an unnamed run"


def wait_until_idle(url: str, token: str, timeout=None) -> bool:
    """Wait, with the door OPEN, until nothing is left running. True if it went.

    The door stays open for this, and that is the sequencing the whole design
    turns on. Closing it first and then waiting would refuse clinicians for the
    length of a cohort; waiting first means the door is shut only for the
    restart, which is seconds.

    **A paused run does not block.** It holds no reservation, no thread and no
    subprocess -- it is a record on disk plus a staged cohort, and a restart
    that keeps TEMP_DIR keeps both. Waiting for one would mean waiting up to
    its four-hour idle timeout for a clinician who may have gone home, and the
    only thing that would end that wait is the reaper deleting the very work
    the wait was protecting.

    `timeout=None` waits for as long as it takes, which is the honest default:
    a cohort legitimately runs for hours, and no update is worth interrupting
    one.
    """
    started = time.monotonic()
    announced = None
    while True:
        load = server_load(url, token)
        if load is None:
            log("The server did not answer; not updating while its state is unknown.")
            return False
        if not load["busy"] and not load["running"] and not load["waiting"]:
            if announced is not None:
                log("The server is idle.")
            return True

        busy = load["busy"] or []
        summary = _describe(busy) if busy else f"{load['running']} run(s) admitted"
        if summary != announced:
            # Only when it CHANGES: a silent wait gets killed just before it
            # would have worked, and a line every two seconds for an hour is
            # the same thing with more noise.
            waiting_for = f", {load['waiting']} queued" if load["waiting"] else ""
            log(f"Waiting for {summary}{waiting_for}.")
            announced = summary

        if timeout is not None and time.monotonic() - started >= timeout:
            log(f"Still busy after {timeout}s. Leaving this server alone; "
                f"the next poll will try again.")
            return False
        time.sleep(DRAIN_POLL_SECONDS)


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
            f"No API_TOKEN in {ENV_PATH}. Run 'server_ctl.py up' - it generates one."
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
            "The GPU service was requested but docker cannot reach a GPU -- it has neither "
            "an 'nvidia' runtime nor a CDI device for one -- so the container could not "
            "start at all. Install the NVIDIA Container Toolkit and register it with\n"
            "    sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker\n"
            "or run with --device cpu."
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
        # "Everything works, slowly" was true until CNE moved to the CUDA build
        # of llama-cpp-python. That build's `libggml-cuda.so.0` names
        # `libcuda.so.1` in DT_NEEDED, and the driver is shipped by no wheel --
        # so on a machine with no driver it is `import llama_cpp` itself that
        # fails, not a GPU call inside it. Every other tool still runs here.
        #
        # Said at startup rather than left in a README: the person reading this
        # line is the one doing the install, and they are the only one who can
        # act on it before a clinician meets a 503.
        log("Docker cannot reach a GPU (no nvidia runtime, no CDI device): starting the CPU "
            "service. Every tool works, slowly -- except CNE, which needs an NVIDIA driver "
            "to load at all and will answer 503. To run it here, point its "
            "[tool.uv.sources] at the whl/cpu index and re-sync; see tools/CNE/README.md.")

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
    # things a user may be updating in order to fix - refusing to fetch new
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
                f"{_upstream()}. Refusing to pull over local edits - commit or discard them first."
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
    # the pull above moved it - but it is what makes a tag bump take effect.
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


def cmd_watch(args) -> dict:
    """Follow a branch, and apply what lands on it without dropping a run.

    The loop, once every `--poll` seconds:

        git ls-remote          ~0.3 s, no quota, nothing fetched
          -> same sha?         sleep
          -> new sha:          fetch, classify the diff, drain, pull, restart

    **It runs on the host, and it has to.** The container is given only
    `server/`; the deployment's `.git` is not in it, so the thing that pulls
    code is necessarily outside the thing that serves it -- which is also what
    lets this survive the restart it causes.

    **It never interrupts a run.** `wait_until_idle` waits with the door OPEN
    for as long as the work takes, and the door is shut only for the restart.
    A run that outlasts `--drain-timeout` does not get killed; this server is
    left alone and the next poll tries again.

    **It refuses what a pull cannot deliver.** A change to a lockfile or to
    `requirements.txt` means the virtualenvs baked into the image can no longer
    satisfy the source a pull would land next to them. That is reported and not
    applied: "not updated, and here is the path that needs an image" beats
    "updated, and every run of that tool now fails".

    And because tier one is exactly the case where no dependency moved,
    `docker compose restart` is the right restart -- it keeps TEMP_DIR, and
    with it every paused run's staged cohort, every parked result a client has
    not collected and every upload in flight. `up -d --force-recreate`, which
    `update` uses, destroys all three.
    """
    mode = args.mode or auto_update_mode()
    if mode == AUTO_UPDATE_OFF:
        # Not an error. A unit installed on every machine and inert until a
        # site turns it on is the shape this is meant to have, so the quiet
        # exit is the normal path, not a failure to report.
        log("SADT_AUTO_UPDATE is 'off'; this deployment does not follow its branch.")
        return {"mode": AUTO_UPDATE_OFF, "applied": [], "refused": []}

    branch = args.branch or _upstream().split("/")[-1]
    url = args.url or url_for()
    # `pick_service` answers `inference` or `inference-cpu`, which is what the
    # rest of this file drives -- and .env.example says plainly that
    # `inference-venvs` "is the real deployment" and that "neither of the two
    # below is managed by server_ctl.py". Restarting the wrong container is a
    # silent no-op: the update lands on disk, the server never reloads, and
    # every report says it worked. So the service is nameable, and the default
    # stays what every other command here already assumes.
    service = args.service or pick_service(args.device)
    token = read_env().get("API_TOKEN")
    if not token:
        raise ServerCtlError(
            "No API_TOKEN in the .env, so this cannot ask the server to stop taking "
            "work before it restarts it. Run 'up' first, or set one."
        )

    log(f"Watching {branch} every {args.poll}s in '{mode}' mode. "
        f"The server is at {url}.")
    applied, refused = [], []
    while True:
        outcome = _watch_once(branch, url, service, token, args, apply=mode == AUTO_UPDATE_APPLY)
        if outcome.get("applied"):
            applied.append(outcome["applied"])
        if outcome.get("needs_image"):
            refused.append(outcome["needs_image"])
        if args.once:
            return {
                "mode": mode, "branch": branch, "applied": applied,
                "refused": refused, "head": outcome.get("head"),
            }
        time.sleep(args.poll)


def _watch_once(branch: str, url: str, service: str, token: str, args,
                apply: bool = True) -> dict:
    """One poll. Separated so `--once` and the loop are the same code path.

    `apply=False` is `notify` mode: everything up to and including the
    classification happens, and nothing is drained, pulled or restarted. That
    is where the reporting is, so a site evaluating the mechanism sees exactly
    what it would have done.
    """
    remote = remote_head(branch)
    if remote is None:
        log(f"Could not reach the remote for {branch}; will try again.")
        return {}

    _rc, local, _err = _git(["rev-parse", "HEAD"])
    if local == remote:
        return {"head": local}

    log(f"{branch} is at {remote[:9]}; this clone is at {(local or '?')[:9]}. Fetching.")
    rc, _out, err = _git(["fetch", "--quiet", "origin", branch], timeout=300)
    if rc != 0:
        log(f"Could not fetch {branch}: {err or 'git refused'}")
        return {}

    blockers = needs_image(changed_paths(local, remote))
    if blockers:
        # Said once per sha, not once per poll: a refusal repeated every 30
        # seconds is a log nobody reads.
        if remote != getattr(args, "_last_refused", None):
            log(f"{remote[:9]} changes {', '.join(blockers[:4])}, which a pull cannot "
                f"deliver -- the virtualenvs live in the image. Not updating. "
                f"Build and publish an image for this one.")
            args._last_refused = remote
        return {"head": local, "needs_image": {"sha": remote, "paths": blockers}}

    if not apply:
        log(f"{remote[:9]} is a source-only change and would be applied now. "
            f"SADT_AUTO_UPDATE is 'notify', so nothing was.")
        return {"head": local, "would_apply": {"sha": remote, "branch": branch}}

    if not wait_until_idle(url, token, args.drain_timeout):
        return {"head": local}

    # Shut the door only now. Everything above happened with it open.
    gate = set_accepting(url, token, False, GATE_LEASE_SECONDS, f"updating to {remote[:9]}")
    if gate is None:
        log("The server would not take the maintenance request; not updating.")
        return {"head": local}
    if gate.get("unsupported"):
        log("This server has no maintenance endpoint (it predates it). Updating "
            "without the gate -- a run started in the next second will be lost.")

    try:
        again = server_load(url, token)
        if again and (again["busy"] or again["running"] or again["waiting"]):
            log("A run arrived while the door was closing; leaving this one for later.")
            return {"head": local}

        rc, _out, err = _git(["pull", "--ff-only", "origin", branch], timeout=300)
        if rc != 0:
            log(f"'git pull --ff-only' failed: {err or 'the branch has diverged'}. "
                f"Resolve it by hand in the clone.")
            return {"head": local}

        log(f"Pulled {remote[:9]}. Restarting {service}.")
        _stream(compose_base(service) + ["restart", service], cwd=REPO_ROOT, prefix="  ")
        wait_for_health(url, args.timeout, service)
        return {"head": remote, "applied": {"sha": remote, "branch": branch}}
    finally:
        # On every path out, including the ones above that return early. The
        # lease is the backstop for this process being killed, not for it
        # taking a branch it forgot to clean up after.
        set_accepting(url, token, True)


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

    # Resolved through fetch_data, which is the one place that knows a manifest
    # key can differ from a served tool name: `Crown_Seg` from GET /tools is
    # `CrownSeg` here, and `ALI_CBCT` is served by the `ALI` bundle. Checking
    # names against the catalog keys alone refused every name a client actually
    # has.
    if args.tool:
        fetch_data = _load_fetch_data()
        try:
            fetch_data.resolve_tools(
                fetch_data._parse_manifest(fetch_data._DEFAULT_MANIFEST),
                args.tool,
            )
        except fetch_data.ManifestError as error:
            raise ServerCtlError(str(error))

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
    access = gpu.get("gpu_access")
    if access == "cdi":
        gpu_text = "available to docker through CDI"
    elif access == "runtime":
        gpu_text = "available to docker through the nvidia runtime"
    elif gpu["nvidia_smi"]:
        gpu_text = ("a card is present, but docker cannot reach it "
                    "(no nvidia runtime and no CDI device)")
    else:
        gpu_text = "no GPU on this host"
    print(f"  {mark(gpu['nvidia_runtime'])}gpu       {gpu_text}")

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
            help="Which compose service to drive. Default: gpu when docker can reach a card, "
                 "through either an nvidia runtime or a CDI device.",
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
        "--bind", default=None,
        help="Host address the port is published on. Omitted, it keeps whatever the "
             "deployment already uses, and 127.0.0.1 on a first install - this deployment "
             "speaks plain HTTP, so it stays on loopback unless someone says otherwise. "
             "Pass an empty string to publish on every interface (IPv4 and IPv6), which is "
             "only acceptable behind a TLS terminator; '0.0.0.0' does the same for IPv4 "
             "only. A single address (say a VPN one) publishes on that interface alone.",
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
    # No default, deliberately: `update` is not where anyone decides where the
    # server listens, so it must carry the existing choice over rather than
    # quietly reimpose loopback. See ensure_env.
    update.add_argument("--bind", default=None, help="See 'up --bind'.")
    update.add_argument("--force", action="store_true", help="Recreate even when nothing changed.")
    update.add_argument("--timeout", type=int, default=DEFAULT_STARTUP_TIMEOUT, help="Seconds to wait for /health.")
    update.set_defaults(func=cmd_update, printer=None)

    watch = subparsers.add_parser(
        "watch",
        help="Follow a branch and apply what lands on it, without dropping a run.",
    )
    add_common(watch)
    watch.add_argument("--poll", type=int, default=30,
                       help="Seconds between remote checks. Each one is a "
                            "'git ls-remote', about 0.3s, and consumes no API quota.")
    watch.add_argument("--drain-timeout", type=int, default=None,
                       help="Give up waiting for a busy server after this many seconds "
                            "and retry at the next poll. The default waits for as long "
                            "as the work takes: a cohort legitimately runs for hours, "
                            "and no update is worth interrupting one.")
    watch.add_argument("--timeout", type=int, default=DEFAULT_STARTUP_TIMEOUT,
                       help="Seconds to wait for /health after a restart.")
    watch.add_argument("--service", default=None,
                       help="Which compose service to restart once a pull lands. "
                            "Default: the same one 'up' and 'update' drive. Name "
                            "'inference-venvs' if that is what this deployment runs.")
    watch.add_argument("--mode", choices=AUTO_UPDATE_MODES, default=None,
                       help="Override SADT_AUTO_UPDATE for this invocation. "
                            "Default: whatever the environment or the .env says, "
                            "and 'off' when neither says anything.")
    watch.add_argument("--once", action="store_true",
                       help="Check once and exit, instead of looping. What a timer "
                            "or a cron entry calls, and what the tests drive.")
    watch.set_defaults(func=cmd_watch, printer=None)

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
