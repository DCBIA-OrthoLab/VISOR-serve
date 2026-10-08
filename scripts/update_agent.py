#!/usr/bin/env python3
"""The host half of updating a deployment from its admin panel.

The server cannot update itself: its container sees neither this checkout's
`.git` nor the tools library's. This runs on the HOST, beside both, and does
two things:

* **It says what is waiting.** For the server repository and the tools library
  (SADT-VISOR), every `--poll` seconds: how many commits the branch each one
  follows is ahead, what they are, and what they touch -- for the server,
  whether a restart is enough or the container must be recreated or an image
  rebuilt; for the tools, WHICH tools, and whether only their code changed or
  their environment has to be rebuilt. It writes that to
  `<UPDATE_DIR>/status.json`, which the admin panel shows.
* **It applies an update when an operator asks.** The panel writes
  `<UPDATE_DIR>/request.json`; this picks it up and:

      close the door to new runs      (POST /maintenance, renewed as it goes)
      wait for the runs in flight     (withdrawing the request here aborts)
      pull                            (fast-forward only)
      rebuild changed environments    (uv sync --frozen --all-extras, per changed tool)
      restart the server              (compose restart, or recreate if needed)
      reopen the door, report

It also downloads a tool's models and test files when an operator asks
(`{"kind": "data", "tool": ...}`), from the tools library's manifest into
`DATA/`, which the container can only read. And each survey reports, per
manifest entry, whether it is on disk -- what the panel's tool view lists.

It acts on its own only when SADT_AUTO_UPDATE is `apply` (in the environment or
the .env, re-read at every survey): a survey that finds the server or the tools
behind their branch then files the same request an operator would, so the door
closes FIRST, the runs in flight finish, and only then is anything pulled.
Closing before waiting is what lets an update happen on a busy server at all;
waiting for an idle moment first never ends while clients keep sending. A
request filed this way can be withdrawn like any other (delete request.json),
and one that fails is not retried until the branch moves again. `notify` logs
what would be applied; `off`, the default, does nothing. A change that needs an
image rebuilt is reported and refused, because a pull cannot deliver it.

Standard library only, like the other host scripts: it runs before anything is
installed and must not need anything that an update could break.

    python3 scripts/update_agent.py                 # the loop, beside the server
    python3 scripts/update_agent.py --once          # one status pass, then exit
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import server_ctl  # noqa: E402 - beside this file, and stdlib only too

REPO_ROOT = server_ctl.REPO_ROOT
STATUS_FILE = "status.json"
REQUEST_FILE = "request.json"

# How a changed tool's environment is rebuilt. `--all-extras`, as the image
# builds them: without it a sync REMOVES what an extra installed. Crown_Seg's
# segmentation engine is one, and a deployment synced without it refused every
# mesh that was not already labelled -- AREG IOS on a raw intraoral scan
# included.
SYNC_COMMAND = ["uv", "sync", "--frozen", "--all-extras", "--quiet"]

# How often the request file is looked for. Cheap -- one stat -- and it is the
# delay between an operator's click and the door closing.
REQUEST_POLL_SECONDS = 2
# The door's lease while an update is in progress, renewed well before it runs
# out. If this process is killed, the server reopens by itself this soon after.
DOOR_LEASE_SECONDS = 90
LOG_LINES = 40

def _load_release_diff():
    """The classification the server's own "Check for updates" uses, loaded by
    path so the panel and this agent can never disagree about a commit."""
    path = os.path.join(REPO_ROOT, "server", "wire", "release_diff.py")
    spec = importlib.util.spec_from_file_location("release_diff_for_agent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release_diff = _load_release_diff()
RESTART, RECREATE, IMAGE = release_diff.RESTART, release_diff.RECREATE, release_diff.IMAGE
classify_server = release_diff.classify_server
classify_tools = release_diff.classify_tools


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


# ---------------------------------------------------------------------------
# git, in a given checkout
# ---------------------------------------------------------------------------

def git(repo: str, args, timeout: int = 120):
    try:
        done = subprocess.run(["git", "-C", repo] + list(args), capture_output=True,
                              text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, "", str(exc)
    return done.returncode, done.stdout.strip(), done.stderr.strip()


def tracking(repo: str):
    """`(remote, branch)` this checkout follows, or None."""
    rc, upstream, _err = git(repo, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if rc != 0 or "/" not in upstream:
        return None
    remote, branch = upstream.split("/", 1)
    return remote, branch


def refresh(repo: str) -> str:
    """Fetch the tracked branch when the remote moved. Returns an error or ""."""
    followed = tracking(repo)
    if followed is None:
        return "this checkout follows no remote branch"
    remote, branch = followed
    rc, out, _err = git(repo, ["ls-remote", remote, f"refs/heads/{branch}"], timeout=60)
    if rc != 0 or not out:
        return f"could not reach {remote}"
    tip = out.split()[0]
    _rc, known, _err = git(repo, ["rev-parse", f"{remote}/{branch}"])
    if known != tip:
        rc, _out, err = git(repo, ["fetch", "--quiet", remote, branch], timeout=300)
        if rc != 0:
            return f"could not fetch {remote}/{branch}: {err or 'git refused'}"
    return ""


# ---------------------------------------------------------------------------
# One checkout
# ---------------------------------------------------------------------------

def inspect(repo: str, kind: str, fetch: bool = True) -> dict:
    """Everything the panel shows about one checkout."""
    info = {"kind": kind, "path": repo, "error": ""}
    if not repo or not os.path.isdir(os.path.join(repo, ".git")) and git(repo, ["rev-parse"])[0] != 0:
        info["error"] = "not a git checkout"
        return info
    if fetch:
        info["error"] = refresh(repo)
    followed = tracking(repo)
    if followed is None:
        info["error"] = info["error"] or "this checkout follows no remote branch"
        return info
    remote, branch = followed
    upstream = f"{remote}/{branch}"
    _rc, local = git(repo, ["rev-parse", "HEAD"])[:2]
    _rc, target = git(repo, ["rev-parse", upstream])[:2]
    _rc, behind = git(repo, ["rev-list", "--count", f"HEAD..{upstream}"])[:2]
    _rc, ahead = git(repo, ["rev-list", "--count", f"{upstream}..HEAD"])[:2]
    _rc, dirty = git(repo, ["status", "--porcelain", "--untracked-files=no"])[:2]
    _rc, log_out = git(repo, ["log", "--format=%H%x1f%s%x1f%an%x1f%ct", "-n", "30", f"HEAD..{upstream}"])[:2]
    _rc, diff = git(repo, ["diff", "--name-only", f"HEAD..{upstream}"])[:2]
    _rc, head_subject = git(repo, ["log", "-1", "--format=%s%x1f%ct", "HEAD"])[:2]
    commits = []
    for line in log_out.splitlines():
        bits = line.split("\x1f")
        if len(bits) == 4:
            commits.append({"sha": bits[0][:9], "subject": bits[1], "author": bits[2], "at": int(bits[3])})
    paths = [p for p in diff.splitlines() if p.strip()]
    subject, _, when = head_subject.partition("\x1f")
    info.update({
        "remote": remote, "branch": branch,
        "local": local[:9], "target": target[:9],
        "current": {"subject": subject, "at": int(when) if when.isdigit() else None},
        "behind": int(behind) if behind.isdigit() else 0,
        "ahead": int(ahead) if ahead.isdigit() else 0,
        "dirty": bool(dirty),
        "commits": commits,
        "changes": release_diff.classify(kind, paths, repo),
    })
    return info


def blockers(info: dict) -> list:
    """Why this checkout cannot be updated from the panel, or []."""
    reasons = []
    if info.get("error"):
        reasons.append(info["error"])
    if info.get("dirty") and info.get("behind"):
        reasons.append("it has uncommitted changes; a pull would refuse")
    if info.get("ahead"):
        reasons.append(f"it has {info['ahead']} local commit(s) the remote does not; a fast-forward is impossible")
    if info.get("kind") == "server" and (info.get("changes") or {}).get("action") == IMAGE:
        reasons.append("it changes the image (" + ", ".join(info["changes"]["heavy"][:3]) +
                       "), which a pull cannot deliver: build and publish an image")
    return reasons


# ---------------------------------------------------------------------------
# Data: what the manifest lists, and what is on disk
# ---------------------------------------------------------------------------

def manifest_paths(tools_repo: str):
    """`(fetch_data.py, data-manifest.yml)` from the tools library when it
    carries them, else this repository's copy -- the same order setup-server
    follows."""
    for root in (tools_repo, REPO_ROOT):
        if not root:
            continue
        engine = os.path.join(root, "scripts", "fetch_data.py")
        manifest = os.path.join(root, "scripts", "data-manifest.yml")
        if os.path.isfile(engine) and os.path.isfile(manifest):
            return engine, manifest
    return None, None


def load_engine(path: str):
    spec = importlib.util.spec_from_file_location("fetch_data_for_agent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def data_summary(tools_repo: str, data_dir: str) -> dict:
    """Per manifest entry: what it is, how big, and whether it is on disk."""
    engine_path, manifest_path = manifest_paths(tools_repo)
    if not engine_path:
        return {"error": "no data manifest found", "tools": {}}
    try:
        engine = load_engine(engine_path)
        manifest = engine._parse_manifest(manifest_path)
    except Exception as exc:  # noqa: BLE001 - reported, never fatal to the agent
        return {"error": f"the manifest could not be read: {exc}", "tools": {}}
    tools = {}
    for key, sections in manifest.items():
        entries = []
        for kind in engine.KINDS:
            for entry in sections.get(kind, []):
                if "name" not in entry:
                    continue
                target = engine._target_path(data_dir, {**entry, "tool": key, "kind": kind})
                entries.append({
                    "kind": kind,
                    "name": entry.get("dest") or entry["name"],
                    "size": entry.get("size"),
                    "present": os.path.exists(target),
                })
        tools[key] = {"provides": sections.get("provides") or [], "entries": entries}
    return {"error": "", "manifest": os.path.relpath(manifest_path, tools_repo or REPO_ROOT),
            "data_dir": data_dir, "tools": tools}


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, args):
        self.args = args
        self.update_dir = args.update_dir
        self.url = args.url
        self.token = args.token
        self.state = {"last": None, "applying": None}
        previous = self._read(STATUS_FILE) or {}
        self.state["last"] = previous.get("last")
        self.server = {}
        self.tools = {}
        self.data = {}
        self.log_lines = []

    # -- files ---------------------------------------------------------
    def _path(self, name):
        return os.path.join(self.update_dir, name)

    def _read(self, name):
        try:
            with open(self._path(name), encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else None
        except (OSError, ValueError):
            return None

    def write_status(self):
        os.makedirs(self.update_dir, exist_ok=True)
        status = {
            "heartbeat": time.time(),
            "poll": self.args.poll,
            "server": self.server,
            "tools": self.tools,
            "data": self.data,
            "applying": self.state["applying"],
            "last": self.state["last"],
        }
        for info in (self.server, self.tools):
            if info:
                info["blockers"] = blockers(info)
        staging = self._path(STATUS_FILE) + ".tmp"
        with open(staging, "w", encoding="utf-8") as handle:
            json.dump(status, handle)
        os.replace(staging, self._path(STATUS_FILE))

    def phase(self, text: str):
        log(text)
        self.log_lines = (self.log_lines + [time.strftime("%H:%M:%S ") + text])[-LOG_LINES:]
        if self.state["applying"] is not None:
            self.state["applying"]["phase"] = text
            self.state["applying"]["log"] = self.log_lines
        self.write_status()

    # -- the server ----------------------------------------------------
    def _api(self, path, payload=None, method=None, timeout=10):
        request = urllib.request.Request(f"{self.url.rstrip('/')}{path}",
                                         headers={"Authorization": f"Bearer {self.token}"},
                                         method=method)
        if payload is not None:
            request.data = json.dumps(payload).encode("utf-8")
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001 - a probe answers None
            return None

    def door(self, accepting: bool, reason: str = "") -> None:
        self._api("/maintenance", {"accepting": accepting, "seconds": DOOR_LEASE_SECONDS, "reason": reason})

    def in_flight(self):
        """Runs still holding or wanting the machine, or None if unreachable."""
        status = self._api("/status")
        if status is None:
            return None
        now = time.time()
        return [run for run in status.get("runs") or []
                if run.get("state") in ("running", "pending")
                and now - (run.get("updated_at") or now) < server_ctl.GHOST_AFTER_SECONDS]

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url.rstrip('/')}/health", timeout=5) as response:
                return response.status == 200
        except Exception:  # noqa: BLE001
            return False

    # -- status --------------------------------------------------------
    def survey(self, fetch: bool = True):
        self.server = inspect(self.args.server_repo, "server", fetch=fetch)
        self.tools = inspect(self.args.tools_repo, "tools", fetch=fetch) if self.args.tools_repo else {
            "kind": "tools", "error": "no tools checkout configured (SADT_TOOLS or --tools-repo)"}
        self.data = data_summary(self.args.tools_repo, self.args.data_dir)
        self.write_status()

    def apply_data(self, request: dict) -> None:
        """Download one tool's data from the manifest. No door, no restart: the
        server reads DATA/ live, and a tool reads its data when it runs."""
        tool = request.get("tool") or ""
        force = bool(request.get("force"))
        self.log_lines = []
        self.state["applying"] = {"id": request.get("id"), "kind": "data", "tool": tool,
                                  "started_at": time.time(), "phase": "", "log": []}
        ok, message = False, ""
        try:
            try:
                os.remove(self._path(REQUEST_FILE))
            except FileNotFoundError:
                pass
            engine, manifest = manifest_paths(self.args.tools_repo)
            if not engine:
                message = "No data manifest found in the tools library."
                return
            command = [sys.executable, engine, "--manifest", manifest, "--data-dir", self.args.data_dir,
                       "--tool", tool, "--progress", "always"] + (["--force"] if force else [])
            self.phase(("Re-downloading" if force else "Downloading what is missing for") + f" {tool}.")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            last_write = 0.0
            for line in process.stdout:
                line = line.rstrip()
                if not line:
                    continue
                self.log_lines = (self.log_lines + [line])[-LOG_LINES:]
                self.state["applying"]["phase"] = line.strip()
                self.state["applying"]["log"] = self.log_lines
                if time.monotonic() - last_write > 1:
                    self.write_status()
                    last_write = time.monotonic()
            process.wait()
            ok = process.returncode == 0
            message = (f"The data of {tool} is in place." if ok
                       else f"Some of {tool}'s data could not be downloaded; see the log.")
        except Exception as exc:  # noqa: BLE001
            message = f"The download stopped on an unexpected error: {exc}"
        finally:
            self.state["last"] = {"id": request.get("id"), "kind": "data", "tool": tool, "ok": ok,
                                  "message": message, "at": time.time(), "log": self.log_lines}
            self.state["applying"] = None
            log(message)
            self.survey(fetch=False)

    # -- applying ------------------------------------------------------
    def apply(self, request: dict) -> None:
        target = request.get("target") or "all"
        self.log_lines = []
        self.state["applying"] = {"id": request.get("id"), "target": target,
                                  "started_at": time.time(), "phase": "", "log": []}
        ok, message = False, ""
        try:
            self.survey(fetch=True)
            parts = [info for name, info in (("server", self.server), ("tools", self.tools))
                     if target in ("all", name) and info.get("behind")]
            if not parts:
                ok, message = True, "Nothing to update: already at the latest version."
                return
            refused = [f"{info['kind']}: {reason}" for info in parts for reason in blockers(info)]
            if refused:
                message = "Not updated. " + "; ".join(refused)
                return

            self.phase("Closing the server to new runs.")
            self.door(False, "updating")
            last_lease = time.monotonic()
            while True:
                if not os.path.exists(self._path(REQUEST_FILE)):
                    message = "Withdrawn before anything was changed."
                    return
                busy = self.in_flight()
                if busy == []:
                    break
                if busy is None:
                    self.phase("The server is not answering; waiting for it.")
                else:
                    self.phase(f"Waiting for {len(busy)} run(s) to finish: " +
                               ", ".join(sorted({r.get('tool') or '?' for r in busy})))
                if time.monotonic() - last_lease > DOOR_LEASE_SECONDS / 3:
                    self.door(False, "updating")
                    last_lease = time.monotonic()
                time.sleep(REQUEST_POLL_SECONDS * 2)

            # The point of no return: the request is consumed, the pull begins.
            try:
                os.remove(self._path(REQUEST_FILE))
            except FileNotFoundError:
                pass
            restart = RESTART
            applied = {}
            for info in parts:
                self.phase(f"Pulling the {info['kind']} ({info['behind']} commit(s)).")
                self.door(False, "updating")
                rc, _out, err = git(info["path"], ["pull", "--ff-only"], timeout=600)
                if rc != 0:
                    message = f"The {info['kind']} pull failed: {err or 'git refused'}. Nothing was restarted."
                    return
                applied[info["kind"]] = info["target"]
                if info["kind"] == "server" and info["changes"]["action"] == RECREATE:
                    restart = RECREATE
                if info["kind"] == "tools":
                    for folder in info["changes"]["environments"]:
                        if self.args.no_sync:
                            self.phase(f"Skipping the environment rebuild of {folder} (--no-sync).")
                            continue
                        self.phase(f"Rebuilding the environment of {folder}.")
                        self.door(False, "updating")
                        # A local package installed by copy is reinstalled, or
                        # the environment keeps the version it was built with.
                        reinstall = []
                        for package in info["changes"].get("reinstall", {}).get(folder, []):
                            reinstall += ["--reinstall-package", package]
                        done = subprocess.run(SYNC_COMMAND + reinstall,
                                              cwd=os.path.join(info["path"], folder),
                                              capture_output=True, text=True, timeout=3600)
                        if done.returncode != 0:
                            message = (f"Rebuilding {folder} failed: {(done.stderr or done.stdout)[-400:]}. "
                                       f"The code is pulled; the server was not restarted.")
                            return

            self.phase("Restarting the server." if restart == RESTART else "Recreating the server's container.")
            if self.args.restart_cmd:
                done = subprocess.run(self.args.restart_cmd, shell=True, capture_output=True, text=True, timeout=600)
            else:
                service = self.args.service or server_ctl.pick_service()
                verb = ["restart", service] if restart == RESTART else ["up", "-d", "--force-recreate", service]
                done = subprocess.run(server_ctl.compose_base(service) + verb, cwd=REPO_ROOT,
                                      capture_output=True, text=True, timeout=900)
            if done.returncode != 0:
                message = f"The restart failed: {(done.stderr or done.stdout)[-400:]}"
                return
            deadline = time.monotonic() + self.args.health_timeout
            self.phase("Waiting for the server to answer again.")
            while time.monotonic() < deadline and not self.healthy():
                time.sleep(2)
            if not self.healthy():
                message = "The server did not come back after the restart; look at its logs."
                return
            ok = True
            message = "Updated " + ", ".join(f"{kind} to {sha}" for kind, sha in applied.items()) + "."
        except Exception as exc:  # noqa: BLE001 - an update must always end with the door open
            message = f"The update stopped on an unexpected error: {exc}"
        finally:
            self.door(True)
            try:
                if not ok and os.path.exists(self._path(REQUEST_FILE)):
                    os.remove(self._path(REQUEST_FILE))
            except OSError:
                pass
            self.state["last"] = {"id": request.get("id"), "target": target, "ok": ok,
                                  "message": message, "at": time.time(), "log": self.log_lines}
            self.state["applying"] = None
            log(message)
            self.survey(fetch=False)

    def claim(self) -> bool:
        """One agent per deployment: refuse to start beside a live one."""
        os.makedirs(self.update_dir, exist_ok=True)
        pid_path = self._path("agent.pid")
        try:
            with open(pid_path, encoding="utf-8") as handle:
                other = int(handle.read().strip())
            if other != os.getpid():
                with open(f"/proc/{other}/cmdline", "rb") as handle:
                    if b"update_agent.py" in handle.read():
                        log(f"Another update agent is already running (pid {other}).")
                        return False
        except (OSError, ValueError):
            pass
        with open(pid_path, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        return True

    def auto_request(self):
        """The request SADT_AUTO_UPDATE=apply files for what the survey found
        waiting, or None. See the module docstring for the order it keeps."""
        waiting = [info for info in (self.server, self.tools) if info and info.get("behind")]
        if not waiting:
            self._auto_failed = None
            return None
        targets = sorted(f"{info['kind']}:{info.get('target') or '?'}" for info in waiting)
        mode = server_ctl.auto_update_mode()
        if mode == server_ctl.AUTO_UPDATE_NOTIFY:
            if targets != getattr(self, "_auto_noted", None):
                log("SADT_AUTO_UPDATE is 'notify': would update " + ", ".join(targets) + ".")
                self._auto_noted = targets
            return None
        if mode != server_ctl.AUTO_UPDATE_APPLY:
            return None
        refused = [f"{info['kind']}: {reason}" for info in waiting for reason in blockers(info)]
        if refused:
            if targets != getattr(self, "_auto_noted", None):
                log("Not updating automatically: " + "; ".join(refused))
                self._auto_noted = targets
            return None
        if targets == getattr(self, "_auto_failed", None):
            return None  # failed once on these commits; wait for the branch to move
        return {"id": "auto-" + "-".join(t.split(":", 1)[1][:9] for t in targets),
                "target": "all", "auto": True, "targets": targets}

    def run(self):
        if not self.args.once and not self.claim():
            return
        log(f"SADT_AUTO_UPDATE is '{server_ctl.auto_update_mode()}'.")
        self.survey()
        if self.args.once:
            return
        next_survey = time.monotonic() + self.args.poll
        while True:
            request = self._read(REQUEST_FILE)
            if request:
                if request.get("kind") == "data":
                    self.apply_data(request)
                else:
                    self.apply(request)
                    last = self.state["last"] or {}
                    if request.get("auto") and not last.get("ok"):
                        self._auto_failed = request.get("targets")
                    if last.get("ok") and last.get("target") in ("all", "server") \
                            and "server" in (last.get("message") or ""):
                        # This file may have been updated with the rest: run the
                        # new one rather than go on with the code it replaced.
                        log("Restarting the agent on the updated code.")
                        os.execv(sys.executable, [sys.executable] + sys.argv)
                next_survey = time.monotonic() + self.args.poll
            elif time.monotonic() >= next_survey:
                self.survey()
                automatic = self.auto_request()
                if automatic:
                    log("SADT_AUTO_UPDATE is 'apply': updating " + ", ".join(automatic["targets"]) + ".")
                    staging = self._path(REQUEST_FILE) + ".tmp"
                    with open(staging, "w", encoding="utf-8") as handle:
                        json.dump(automatic, handle)
                    os.replace(staging, self._path(REQUEST_FILE))
                    continue  # picked up at once, as an operator's would be
                next_survey = time.monotonic() + self.args.poll
            else:
                # The heartbeat, so the panel can tell a quiet agent from a
                # dead one between surveys.
                if int(time.monotonic()) % 30 < REQUEST_POLL_SECONDS:
                    self.write_status()
            time.sleep(REQUEST_POLL_SECONDS)


def _default_tools_repo() -> str:
    """The tools checkout this deployment serves, from SADT_TOOLS in the .env."""
    tools = os.environ.get("SADT_TOOLS") or server_ctl.read_env().get("SADT_TOOLS") or ""
    if not tools:
        return ""
    rc, top, _err = git(tools, ["rev-parse", "--show-toplevel"])
    return top if rc == 0 else ""


def main(argv=None) -> int:
    env = server_ctl.read_env()
    parser = argparse.ArgumentParser(description="Report and apply updates for the admin panel.")
    parser.add_argument("--server-repo", default=REPO_ROOT)
    parser.add_argument("--tools-repo", default=None, help="Default: the checkout holding SADT_TOOLS.")
    parser.add_argument("--update-dir", default=os.path.join(REPO_ROOT, "server", ".update"))
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "DATA"),
                        help="Where models and test files are downloaded (the server's DATA/).")
    parser.add_argument("--url", default=None, help="Default: this deployment's own address.")
    parser.add_argument("--token", default=None, help="Default: API_TOKEN from the .env.")
    parser.add_argument("--service", default=None, help="The compose service to restart.")
    parser.add_argument("--restart-cmd", default=None,
                        help="A shell command that restarts the server, instead of docker compose.")
    parser.add_argument("--poll", type=int, default=60, help="Seconds between surveys of the remotes.")
    parser.add_argument("--health-timeout", type=int, default=600)
    parser.add_argument("--no-sync", action="store_true", help="Never run uv sync (for testing).")
    parser.add_argument("--once", action="store_true", help="One survey, then exit.")
    args = parser.parse_args(argv)
    if args.tools_repo is None:
        args.tools_repo = _default_tools_repo()
    args.url = args.url or server_ctl.url_for()
    args.token = args.token or os.environ.get("API_TOKEN") or env.get("API_TOKEN") or ""
    Agent(args).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
