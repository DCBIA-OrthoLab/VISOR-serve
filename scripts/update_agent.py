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

A deployment that runs a tool library from its PREBUILT image
(SADT_TOOLS_IMAGE, docker/compose.tools-image.yml) has no tools checkout. Its
tools are then followed by TAG: they are behind when the registry serves a
different digest under that tag than the image on this machine, and what
changed comes from the image's labels and the GitHub repository they name.
Updating them is pulling the image and recreating the container, inside the
same door and the same wait. The server image (VISOR_SERVER_IMAGE) is followed
the same way, so a server commit that needs a new image is applied once that
image is published, instead of being refused.

Standard library only, like the other host scripts: it runs before anything is
installed and must not need anything that an update could break.

    python3 scripts/update_agent.py                 # the loop, beside the server
    python3 scripts/update_agent.py --once          # one status pass, then exit
"""

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
import types
import uuid
import urllib.error
import urllib.request

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import image_release  # noqa: E402 - beside this file, stdlib only
import server_ctl  # noqa: E402 - beside this file, and stdlib only too

REPO_ROOT = server_ctl.REPO_ROOT
STATUS_FILE = "status.json"
REQUEST_FILE = "request.json"
# The tool a data request names to bring every installed tool up to date.
ALL_DATA = "all"
# The runs that prove an update left the tools working, beside status.json
# unless --smoke-file names another. Deployment-specific on purpose: which
# tools and which hosted test files are worth a check is this deployment's
# knowledge, and the server holds none about any tool.
SMOKE_FILE = "smoke.json"
SMOKE_TIMEOUT_SECONDS = 1800

# How a changed tool's environment is rebuilt. `--all-extras`, as the image
# builds them: without it a sync REMOVES what an extra installed. Crown_Seg's
# segmentation engine is one, and a deployment synced without it refused every
# mesh that was not already labelled -- AREG IOS on a raw intraoral scan
# included.
SYNC_COMMAND = ["uv", "sync", "--frozen", "--all-extras", "--quiet"]

# How often the request file is looked for. Cheap -- one stat -- and it is the
# delay between an operator's click and the door closing.
REQUEST_POLL_SECONDS = 2
# How long the last wait before a restart tolerates a server that does not
# answer. An update that changes the server's own code can take the running one
# down before the container is recreated -- which is the restart that would
# bring it back -- so waiting for an answer then waits for ever. A server that
# is down has no run in flight to protect.
UNREACHABLE_GRACE_SECONDS = 120
# The door's lease while an update is in progress, renewed well before it runs
# out. If this process is killed, the server reopens by itself this soon after.
DOOR_LEASE_SECONDS = 90
LOG_LINES = 40
# The header POST /maintenance is authenticated by: the operator's admin token,
# never the API token every workstation holds (server/wire/security.py).
ADMIN_HEADER = "X-Admin-Token"
# door()'s answer when the server could not be reached at all, as opposed to
# one that answered and refused.
DOOR_UNREACHABLE = "the server did not answer"

def _release_diff_source() -> str:
    """The source of server/wire/release_diff.py as COMMITTED at HEAD.

    Never the working tree: the server container bind-mounts server/ read-write,
    so a file there is whatever the container last wrote, and executing it here
    would run container-controlled code on the host. HEAD is what the operator
    fetched, and the container sees .git read-only. A tree with no .git at all
    (a test image mounting scripts/ and server/ alone) has no HEAD to read and
    no agent deployed in it, and falls back to the file.
    """
    relative = "server/wire/release_diff.py"
    if not os.path.exists(os.path.join(REPO_ROOT, ".git")):
        with open(os.path.join(REPO_ROOT, *relative.split("/")), encoding="utf-8") as handle:
            return handle.read()
    done = subprocess.run(["git", "-C", REPO_ROOT, "show", f"HEAD:{relative}"],
                          capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        raise RuntimeError(f"cannot read {relative} at HEAD: {done.stderr.strip() or 'git refused'}")
    return done.stdout


def _load_release_diff():
    """The classification the server's own "Check for updates" uses, so the
    panel and this agent can never disagree about a commit."""
    module = types.ModuleType("release_diff_for_agent")
    module.__file__ = "<HEAD:server/wire/release_diff.py>"
    exec(compile(_release_diff_source(), module.__file__, "exec"), module.__dict__)
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


# A pulled image's release series, from its version label: the tools releases
# are `tools-YYYY.MM.DD-N` and share their repository with `data-` releases.
_RELEASE_SERIES = re.compile(r"^([a-z]+-)\d")


def inspect_image(image: str, kind: str, fetch: bool = True, cache: dict = None) -> dict:
    """What the panel shows about a component the deployment runs as an image.

    Shaped like `inspect`, so the panel renders a library from an image as it
    renders one from a checkout: `branch` is the tag followed, `local` the
    release this machine runs, `behind` the commits the published release adds.
    The registry is asked at every survey (one request); GitHub only when that
    digest moves, through `cache`, which keeps an agent polling every minute
    far under the API's anonymous limit.
    """
    cache = cache if cache is not None else {}
    _registry, repository, tag = image_release.split_reference(image)
    info = {"kind": kind, "path": image, "image": image, "error": "", "remote": repository,
            "branch": tag, "behind": 0, "ahead": 0, "dirty": False, "commits": [],
            "changes": {"action": RECREATE, "tools": [], "environments": [], "heavy": [image]}}
    local = image_release.local_image(image)
    if not local["present"]:
        info["error"] = f"{image} is not on this machine"
        return info
    labels = local["labels"]
    release = labels.get("org.opencontainers.image.version") or "unlabelled"
    revision = labels.get("org.opencontainers.image.revision") or ""
    source = labels.get("org.opencontainers.image.source") or ""
    info.update({"local": release, "target": release, "revision": revision[:9],
                 "current": {"subject": f"image {release}" + (f", commit {revision[:9]}" if revision else ""),
                             "at": None}})
    if not local["digest"]:
        info["error"] = f"{image} was built on this machine, not pulled: there is no published version to follow"
        return info
    remote = image_release.remote_digest(image) if fetch else cache.get(("digest", image), "")
    cache[("digest", image)] = remote
    if not remote:
        info["warning"] = "the registry could not be asked; shown as of the last pull"
        return info
    if remote == local["digest"]:
        return info
    known = cache.get(("release", remote))
    if known is None:
        series = _RELEASE_SERIES.match(release)
        latest = image_release.latest_release(source, series.group(1)) if series else {}
        difference = image_release.compare(source, revision, latest.get("revision", ""))
        known = {"latest": latest, "difference": difference}
        cache[("release", remote)] = known
    latest, difference = known["latest"], known["difference"]
    info.update({
        "target": latest.get("tag") or remote.split(":", 1)[-1][:12],
        "behind": max(1, difference.get("behind") or 0),
        "commits": difference.get("commits") or [],
        "pending_digest": remote,
    })
    if kind == "tools":
        classified = release_diff.classify_tools(difference.get("paths") or [])
        info["changes"] = {**classified, "action": RECREATE, "heavy": [image]}
    return info


def _pinned(image: str, digest: str) -> str:
    """`image` with its tag replaced by `digest` (`name@sha256:...`), or
    `image` itself when no digest is known."""
    if not digest or not digest.startswith("sha256:"):
        return image
    name = image.split("@", 1)[0]
    if ":" in name.rsplit("/", 1)[-1]:
        name = name.rsplit(":", 1)[0]
    return f"{name}@{digest}"


def blockers(info: dict) -> list:
    """Why this checkout cannot be updated from the panel, or []."""
    reasons = []
    if info.get("error"):
        reasons.append(info["error"])
    if info.get("dirty") and info.get("behind"):
        reasons.append("it has uncommitted changes; a pull would refuse")
    if info.get("ahead"):
        reasons.append(f"it has {info['ahead']} local commit(s) the remote does not; a fast-forward is impossible")
    if info.get("kind") == "server" and (info.get("changes") or {}).get("action") == IMAGE \
            and not info.get("image_update"):
        reasons.append("it changes the image (" + ", ".join(info["changes"]["heavy"][:3]) +
                       "), which a pull cannot deliver: " + (
                           "waiting for the server image to be published" if info.get("server_image")
                           else "build and publish an image"))
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
                located = {**entry, "tool": key, "kind": kind}
                if hasattr(engine, "entry_state"):
                    state = engine.entry_state(located, data_dir)
                else:  # a tools library carrying an older engine: presence only
                    state = "current" if os.path.exists(engine._target_path(data_dir, located)) else "missing"
                entries.append({
                    "kind": kind,
                    "name": entry.get("dest") or entry["name"],
                    "size": entry.get("size"),
                    "present": state != "missing",
                    "state": state,
                })
        tools[key] = {"provides": sections.get("provides") or [], "entries": entries}
    return {"error": "", "manifest": os.path.relpath(manifest_path, tools_repo or REPO_ROOT),
            "data_dir": data_dir, "tools": tools, "refresh": refresh_targets(tools)}


def refresh_targets(tools: dict) -> list:
    """The tools whose data an update brings up to date: every tool with
    something on disk and something not current -- replaced in the manifest,
    or added to it. A tool with nothing on disk was never installed here, and
    is left for an operator to download on purpose."""
    return sorted(key for key, info in tools.items()
                  if any(e["present"] for e in info["entries"])
                  and any(e["state"] != "current" for e in info["entries"]))


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

def _code_stamp() -> int:
    """When this file last changed, or 0 when that cannot be read."""
    try:
        return os.stat(os.path.abspath(__file__)).st_mtime_ns
    except OSError:
        return 0


class Agent:
    def __init__(self, args):
        # Images are followed only when named: arguments built by an older
        # caller -- or a test -- carry neither, and mean checkouts only.
        for name in ("server_image", "tools_image"):
            if not hasattr(args, name):
                setattr(args, name, "")
        self.args = args
        self.update_dir = args.update_dir
        self.url = args.url
        self.token = args.token
        self.admin_token = getattr(args, "admin_token", "") or ""
        self.state = {"last": None, "applying": None}
        previous = self._read(STATUS_FILE) or {}
        self.state["last"] = previous.get("last")
        self.server = {}
        self.tools = {}
        self.data = {}
        self.server_image = {}
        self.image_cache = {}
        self.log_lines = []
        self._code_stamp = _code_stamp()

    def _code_changed(self) -> bool:
        """Whether this file is not the one the running agent was started from."""
        return _code_stamp() != self._code_stamp

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
            with server_ctl.server_urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001 - a probe answers None
            return None

    def door(self, accepting: bool, reason: str = ""):
        """Open or close the server to new runs. None once the server took it,
        else why it did not -- DOOR_UNREACHABLE when nothing answered.

        Sent with the ADMIN token: closing a clinic's server is an operator's
        act, and the API token every workstation holds cannot do it.
        """
        if not self.admin_token:
            return "no ADMIN_TOKEN in the deployment's .env, so the server cannot be closed to new runs"
        request = urllib.request.Request(
            f"{self.url.rstrip('/')}/maintenance", method="POST",
            data=json.dumps({"accepting": accepting, "seconds": DOOR_LEASE_SECONDS,
                             "reason": reason}).encode("utf-8"),
            headers={ADMIN_HEADER: self.admin_token, "Content-Type": "application/json"})
        try:
            with server_ctl.server_urlopen(request, timeout=10) as response:
                response.read()
            return None
        except urllib.error.HTTPError as exc:
            if exc.code == 403:
                return "the server has no ADMIN_TOKEN set, so it cannot be closed to new runs"
            if exc.code == 401:
                return "the server refused this deployment's ADMIN_TOKEN"
            return f"the server answered {exc.code} to closing it"
        except Exception:  # noqa: BLE001 - reported, the caller decides
            return DOOR_UNREACHABLE

    def in_flight(self):
        """Runs still holding or wanting the machine, or None if unreachable."""
        status = self._api("/status")
        if status is None:
            return None
        now = time.time()
        tracked = [run for run in status.get("runs") or []
                   if run.get("state") in ("running", "pending")
                   and now - (run.get("updated_at") or now) < server_ctl.GHOST_AFTER_SECONDS]
        if tracked:
            return tracked
        # A client that sends no X-Run-Id is not in `runs` at all, yet its run
        # holds the machine like any other -- and a restart would kill it in
        # the middle. What admission still holds or queues is the ground truth.
        admission = status.get("admission") or {}
        held = int(admission.get("running") or 0) + int(admission.get("waiting") or 0)
        return [{"tool": "a run without a run id", "state": "running"}] * held

    def healthy(self) -> bool:
        try:
            with server_ctl.server_urlopen(f"{self.url.rstrip('/')}/health", timeout=5) as response:
                return response.status == 200
        except Exception:  # noqa: BLE001
            return False

    # -- status --------------------------------------------------------
    def survey(self, fetch: bool = True):
        self.server = inspect(self.args.server_repo, "server", fetch=fetch)
        if self.args.server_image:
            # Not a panel entry of its own: it decides whether a server commit
            # that needs a new image can be applied (the published one has it).
            self.server_image = inspect_image(self.args.server_image, "server-image", fetch, self.image_cache)
            self.server["server_image"] = self.args.server_image
            self.server["image_update"] = bool(self.server_image.get("behind"))
        if self.args.tools_image:
            self.tools = inspect_image(self.args.tools_image, "tools", fetch, self.image_cache)
        elif self.args.tools_repo:
            self.tools = inspect(self.args.tools_repo, "tools", fetch=fetch)
        else:
            self.tools = {"kind": "tools",
                          "error": "no tools configured (SADT_TOOLS_IMAGE, SADT_TOOLS or --tools-repo)"}
        self.data = data_summary(self.args.tools_repo, self.args.data_dir)
        self.write_status()

    def apply_data(self, request: dict) -> None:
        """Download one tool's data from the manifest -- or, for the tool
        `all`, bring every installed tool's data up to date (refresh_targets).
        No door, no restart: the server reads DATA/ live, and a tool reads its
        data when it runs."""
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
            if tool == ALL_DATA:
                tools = data_summary(self.args.tools_repo, self.args.data_dir).get("refresh") or []
                if not tools:
                    ok, message = True, "Every installed tool's data is already up to date."
                    return
            else:
                tools = [tool]
            ok, message = self.fetch_data(tools, force)
        except Exception as exc:  # noqa: BLE001
            message = f"The download stopped on an unexpected error: {exc}"
        finally:
            self.state["last"] = {"id": request.get("id"), "kind": "data", "tool": tool, "ok": ok,
                                  "message": message, "at": time.time(), "log": self.log_lines}
            self.state["applying"] = None
            log(message)
            self.survey(fetch=False)

    def fetch_data(self, tools: list, force: bool = False):
        """Run the manifest's engine for `tools`, streaming its output into the
        status. Returns `(ok, message)`; never raises for a failed download."""
        engine, manifest = manifest_paths(self.args.tools_repo)
        if not engine:
            return False, "No data manifest found in the tools library."
        named = ", ".join(tools)
        command = [sys.executable, engine, "--manifest", manifest, "--data-dir", self.args.data_dir,
                   "--progress", "always"] + [arg for t in tools for arg in ("--tool", t)]
        command += ["--force"] if force else []
        self.phase(("Re-downloading" if force else "Downloading what is missing or changed for") + f" {named}.")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        last_write = 0.0
        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue
            self.log_lines = (self.log_lines + [line])[-LOG_LINES:]
            if self.state.get("applying") is not None:
                self.state["applying"]["phase"] = line.strip()
                self.state["applying"]["log"] = self.log_lines
            if time.monotonic() - last_write > 1:
                self.write_status()
                last_write = time.monotonic()
        process.wait()
        if process.returncode == 0:
            return True, f"The data of {named} is in place."
        return False, f"Some of the data of {named} could not be downloaded; see the log."

    # -- applying ------------------------------------------------------
    def apply(self, request: dict) -> None:
        target = request.get("target") or "all"
        keeper = None
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
            # A server that answers and refuses will refuse every renewal too:
            # updating anyway would restart it under runs admitted meanwhile.
            # One that does not answer is waited for below, and must take the
            # door before anything changes.
            refusal = self.door(False, "updating")
            if refusal and refusal != DOOR_UNREACHABLE:
                message = f"Not updated: {refusal}."
                return
            keeper = self.hold_door()
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
                time.sleep(REQUEST_POLL_SECONDS * 2)

            refusal = self.door(False, "updating")
            if refusal:
                message = f"Not updated: {refusal}. Nothing was changed."
                return
            # The point of no return: the request is consumed, the pull begins.
            try:
                os.remove(self._path(REQUEST_FILE))
            except FileNotFoundError:
                pass
            restart = RESTART
            applied = {}
            # What each image is NOW, before the pull moves its tag: once the
            # new one is running, this is the copy to drop.
            replaced = {info["image"]: self.image_id(info["image"])
                        for info in parts if info.get("image")}
            for info in parts:
                if info.get("image"):
                    self.phase(f"Pulling {info['image']} ({info['target']}).")
                    if not self.pull_image(info["image"], info.get("pending_digest", "")):
                        message = f"Pulling {info['image']} failed; see the log. Nothing was restarted."
                        return
                    applied[info["kind"]] = info["target"]
                    restart = RECREATE
                    continue
                self.phase(f"Pulling the {info['kind']} ({info['behind']} commit(s)).")
                self.door(False, "updating")
                rc, _out, err = git(info["path"], ["pull", "--ff-only"], timeout=600)
                if rc != 0:
                    message = f"The {info['kind']} pull failed: {err or 'git refused'}. Nothing was restarted."
                    return
                applied[info["kind"]] = info["target"]
                if info["kind"] == "server" and info["changes"]["action"] == RECREATE:
                    restart = RECREATE
                if info["kind"] == "server" and info["changes"]["action"] == IMAGE:
                    self.phase(f"Pulling the published server image {self.args.server_image}.")
                    if not self.pull_image(self.args.server_image,
                                           self.server_image.get("pending_digest", "")):
                        message = ("The server code is pulled but its image could not be; "
                                   "the server was not restarted.")
                        return
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

            self.wait_idle("Before restarting", unreachable_grace=UNREACHABLE_GRACE_SECONDS)
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
            self.drop_replaced(replaced)
            message = "Updated " + ", ".join(f"{kind} to {sha}" for kind, sha in applied.items()) + "."
        except Exception as exc:  # noqa: BLE001 - an update must always end with the door open
            message = f"The update stopped on an unexpected error: {exc}"
        finally:
            if keeper is not None:
                keeper.set()
            self.door(True)
            try:
                if not ok and os.path.exists(self._path(REQUEST_FILE)):
                    os.remove(self._path(REQUEST_FILE))
            except OSError:
                pass
            smoke = None
            if ok:
                message += self.refresh_data()
                # After the door reopens: /run refuses everyone while it is
                # shut, this check included, and clients need not wait for it.
                smoke = self.run_smoke()
                if smoke:
                    failed = [entry for entry in smoke if not entry["ok"]]
                    message += (f" Smoke test: {len(smoke)} of {len(smoke)} passed." if not failed else
                                f" Smoke test FAILED for {len(failed)} of {len(smoke)}: " +
                                "; ".join(f"{entry['label']} ({entry['detail']})" for entry in failed))
            self.state["last"] = {"id": request.get("id"), "target": target, "ok": ok,
                                  "message": message, "at": time.time(), "log": self.log_lines,
                                  "smoke": smoke}
            self.state["applying"] = None
            log(message)
            self.survey(fetch=False)

    def refresh_data(self) -> str:
        """After an update: download what the manifest now pins and DATA/ does
        not hold yet, for every installed tool. Run with the door open: a file
        is downloaded and checked beside its target and only then put in place
        (the old one removed, the new one renamed), so a run in flight could
        find it absent for an instant at most. Before the
        smoke test, which may need it. A failure is reported, never undoes the update. Returns the
        sentence to add to the update's message."""
        try:
            tools = data_summary(self.args.tools_repo, self.args.data_dir).get("refresh") or []
            if not tools:
                return ""
            done, message = self.fetch_data(tools)
        except Exception as exc:  # noqa: BLE001 - the update itself is applied
            done, message = False, f"The data could not be refreshed: {exc}"
        return " Models and test files: " + (f"brought up to date for {', '.join(tools)}." if done else message)

    # -- the smoke test ------------------------------------------------
    def smoke_plan(self) -> list:
        """`[{"tool", "form", "label"?, "timeout"?}]` from the smoke file, or []
        when there is none. A malformed file is reported, never fatal."""
        path = getattr(self.args, "smoke_file", "") or self._path(SMOKE_FILE)
        try:
            with open(path, encoding="utf-8") as handle:
                plan = json.load(handle)
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as exc:
            log(f"The smoke file {path} could not be read ({exc}); no smoke test.")
            return []
        if not isinstance(plan, list) or not all(isinstance(item, dict) and item.get("tool")
                                                 for item in plan):
            log(f"The smoke file {path} must be a list of {{\"tool\", \"form\"}}; no smoke test.")
            return []
        return plan

    def run_smoke(self) -> list:
        """Run every smoke check, one at a time, and say how each went."""
        results = []
        for item in self.smoke_plan():
            label = item.get("label") or item["tool"]
            self.phase(f"Smoke test: {label}.")
            started = time.monotonic()
            code, detail = self._post_run(item["tool"], item.get("form") or {},
                                          float(item.get("timeout") or SMOKE_TIMEOUT_SECONDS))
            results.append({"label": label, "tool": item["tool"], "ok": code == 200, "http": code,
                            "seconds": round(time.monotonic() - started, 1),
                            "detail": "ok" if code == 200 else detail})
        return results

    def _post_run(self, tool: str, form: dict, timeout: float):
        """POST /run/<tool> with `form` as multipart fields; (status, detail).

        Sent with a run id, so the run is listed in /status like any client's
        and the next update waits for it. The result is read and discarded:
        what is checked is that the tool ran to the end.
        """
        boundary = uuid.uuid4().hex
        body = b"".join(
            (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n").encode("utf-8")
            for key, value in form.items()) + f"--{boundary}--\r\n".encode("utf-8")
        request = urllib.request.Request(
            f"{self.url.rstrip('/')}/run/{tool}", data=body, method="POST",
            headers={"Authorization": f"Bearer {self.token}", "X-Run-Id": str(uuid.uuid4()),
                     "Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            with server_ctl.server_urlopen(request, timeout=timeout) as response:
                while response.read(1 << 20):
                    pass
                return response.status, ""
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8") or "{}").get("detail")
            except (OSError, ValueError):
                detail = None
            return exc.code, str(detail or exc.reason)[:160]
        except Exception as exc:  # noqa: BLE001 - a smoke check reports, it never raises
            return 0, f"{type(exc).__name__}: {exc}"[:160]

    def hold_door(self) -> threading.Event:
        """Keep the door shut until the returned event is set.

        Renewed from a thread of its own, every third of the lease, whatever
        the update is doing. It used to be renewed between steps and on a
        pull's output lines -- and `docker pull` of a tools image prints
        nothing for minutes while a layer downloads, as `uv sync` does while it
        builds. The lease lapsed, the server reopened, and a run admitted then
        would have been killed by the recreate that followed.
        """
        released = threading.Event()

        def renew():
            while not released.wait(DOOR_LEASE_SECONDS / 3):
                self.door(False, "updating")

        threading.Thread(target=renew, name="door-keeper", daemon=True).start()
        return released

    def pull_image(self, image: str, digest: str = "") -> bool:
        """`docker pull`, its notable lines kept for the panel's log.

        With `digest` -- the one the survey compared -- the exact manifest is
        pulled and then tagged as `image`, so what is installed is what was
        judged, not whatever the tag points at by the time the pull starts.
        """
        reference = _pinned(image, digest)
        process = subprocess.Popen(["docker", "pull", reference], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            line = line.rstrip()
            if line and not line.endswith(("Waiting", "Pulling fs layer", "Download complete")):
                self.log_lines = (self.log_lines + [line])[-LOG_LINES:]
        if process.wait() != 0:
            return False
        if reference == image:
            return True
        done = subprocess.run(["docker", "tag", reference, image], capture_output=True, text=True)
        if done.returncode != 0:
            self.log_lines = (self.log_lines + [(done.stderr or "docker tag failed").strip()])[-LOG_LINES:]
        return done.returncode == 0

    def image_id(self, image: str) -> str:
        """The local ID of `image` (`sha256:...`), or "" when it is not here."""
        try:
            done = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"],
                                  capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return ""
        return done.stdout.strip() if done.returncode == 0 else ""

    def drop_replaced(self, replaced: dict) -> None:
        """Remove the images an update has just replaced, once the new ones answer.

        A pull leaves the previous image on the disk without a name, and a
        tools image is 25 GB: two updates fill a disk the size of a system
        partition, and the third dies on `No space left on device` with the
        server down. Only here, after the health check passed: until then the
        old image is the way back.

        By ID, never a prune -- the machine may hold images that are not ours --
        and without `--force`, so Docker itself refuses one a container still
        uses. A refusal is logged and is not a failed update.
        """
        for image, before in replaced.items():
            if not before or before == self.image_id(image):
                continue
            try:
                done = subprocess.run(["docker", "image", "rm", before], capture_output=True,
                                      text=True, timeout=120)
                line = (f"Removed the previous {image} ({before[7:19]})." if done.returncode == 0 else
                        f"Could not remove the previous {image}: {(done.stderr or 'docker refused').strip()}")
            except (OSError, subprocess.SubprocessError) as exc:
                line = f"Could not remove the previous {image}: {exc}"
            self.log_lines = (self.log_lines + [line])[-LOG_LINES:]

    def wait_idle(self, why: str, unreachable_grace: float = None) -> None:
        """Block until nothing is in flight. The last check before a restart:
        whatever slipped in while the door was open must finish first.

        With `unreachable_grace`, a server that has not answered for that many
        seconds stops being waited for: it is down, so nothing runs on it, and
        the restart that follows is what brings it back."""
        silent_since = None
        while True:
            busy = self.in_flight()
            if busy == []:
                return
            if busy is not None:
                silent_since = None
            elif unreachable_grace is not None:
                silent_since = silent_since if silent_since is not None else time.monotonic()
                if time.monotonic() - silent_since >= unreachable_grace:
                    self.phase(f"{why}: the server has not answered for {int(unreachable_grace)} s; "
                               "nothing can be running, going on.")
                    return
            self.phase(f"{why}: waiting for " + (
                "the server to answer" if busy is None else
                f"{len(busy)} run(s): " + ", ".join(sorted({r.get('tool') or '?' for r in busy}))))
            time.sleep(REQUEST_POLL_SECONDS * 2)

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
            if not request and self._code_changed():
                # Pulled by an update that never finished, or by hand: the agent
                # in memory would go on talking to the server the way it did
                # when it started, which may no longer be the way it answers.
                log("The agent's own code changed on disk; restarting on it.")
                os.execv(sys.executable, [sys.executable] + sys.argv)
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
    parser.add_argument("--tools-image", default=None,
                        help="Follow the tools as this image instead of a checkout. Default: SADT_TOOLS_IMAGE.")
    parser.add_argument("--server-image", default=None,
                        help="The server image to follow. Default: VISOR_SERVER_IMAGE, else the compose "
                             "file's default.")
    parser.add_argument("--update-dir", default=os.path.join(REPO_ROOT, "server", ".update"))
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "DATA"),
                        help="Where models and test files are downloaded (the server's DATA/).")
    parser.add_argument("--url", default=None, help="Default: this deployment's own address.")
    parser.add_argument("--token", default=None, help="Default: API_TOKEN from the .env.")
    parser.add_argument("--admin-token", default=None,
                        help="What closes the server during an update. Default: ADMIN_TOKEN from the "
                             "environment, else the .env. Never printed.")
    parser.add_argument("--service", default=None, help="The compose service to restart.")
    parser.add_argument("--restart-cmd", default=None,
                        help="A shell command that restarts the server, instead of docker compose.")
    parser.add_argument("--poll", type=int, default=60, help="Seconds between surveys of the remotes.")
    parser.add_argument("--health-timeout", type=int, default=600)
    parser.add_argument("--no-sync", action="store_true", help="Never run uv sync (for testing).")
    parser.add_argument("--once", action="store_true", help="One survey, then exit.")
    parser.add_argument("--smoke-file", default=None,
                        help="Runs to check after every successful update, as a JSON list of "
                             '{"tool", "form", "label"?, "timeout"?}. Default: SADT_SMOKE_FILE, '
                             "else smoke.json in the update dir; none, no check.")
    args = parser.parse_args(argv)
    if args.tools_image is None:
        args.tools_image = os.environ.get("SADT_TOOLS_IMAGE") or env.get("SADT_TOOLS_IMAGE") or ""
    if args.tools_repo is None:
        args.tools_repo = "" if args.tools_image else _default_tools_repo()
    if args.smoke_file is None:
        args.smoke_file = os.environ.get("SADT_SMOKE_FILE") or env.get("SADT_SMOKE_FILE") or ""
    if args.server_image is None:
        args.server_image = (os.environ.get("VISOR_SERVER_IMAGE") or env.get("VISOR_SERVER_IMAGE")
                             or server_ctl.DEFAULT_SERVER_IMAGE)
    args.url = args.url or server_ctl.url_for()
    args.token = args.token or os.environ.get("API_TOKEN") or env.get("API_TOKEN") or ""
    args.admin_token = args.admin_token or os.environ.get("ADMIN_TOKEN") or env.get("ADMIN_TOKEN") or ""
    Agent(args).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
