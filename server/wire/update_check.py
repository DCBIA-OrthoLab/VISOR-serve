"""The server's own "Check for updates": what is waiting, with no agent needed.

Read-only from end to end. For the server repository and the tools library:

* the commit deployed here, read from the checkout's `.git` -- mounted
  read-only into the container for the server, already present for the tools;
* the tip of the branch it follows, from `git ls-remote`, which writes nothing;
* the commits in between and the files they touch, from GitHub's compare API,
  because a read-only `.git` cannot fetch them. Unauthenticated, so bounded to
  60 calls an hour: a check runs when an operator asks, or at most every
  `AUTO_CHECK_SECONDS` while the panel is open.

What it cannot do is apply anything; that needs the host (the update agent, or
`server_ctl.py update`). It says so rather than offering a button that would do
nothing.
"""

from __future__ import annotations

import calendar
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from typing import Optional

from config import settings
from wire import release_diff

AUTO_CHECK_SECONDS = 30 * 60
_GITHUB = re.compile(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?/?$")

_lock = threading.Lock()
_last: Optional[dict] = None
_running = False


def _git(repo: str, args, timeout: int = 30):
    try:
        done = subprocess.run(["git", "-c", f"safe.directory={repo}", "-C", repo] + list(args),
                              capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, "", str(exc)
    return done.returncode, done.stdout.strip(), done.stderr.strip()


def _compare(url: str, base: str, head: str) -> dict:
    """GitHub's `compare/base...head`, or raises."""
    match = _GITHUB.search(url)
    if not match:
        raise ValueError("not a GitHub remote, so the commits in between cannot be listed")
    owner, name = match.group(1), match.group(2)
    request = urllib.request.Request(
        f"https://api.github.com/repos/{owner}/{name}/compare/{base}...{head}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "visor-serve-update-check"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def check_repo(repo: str, kind: str) -> dict:
    """Everything the panel shows about one checkout, read-only."""
    info = {"kind": kind, "path": repo, "error": "", "source": "server", "checked_at": time.time()}
    if not repo or _git(repo, ["rev-parse", "--git-dir"])[0] != 0:
        info["error"] = "the checkout's history is not visible to the server"
        return info
    rc, upstream, _err = _git(repo, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if rc != 0 or "/" not in upstream:
        info["error"] = "this checkout follows no remote branch"
        return info
    remote, branch = upstream.split("/", 1)
    _rc, local, _err = _git(repo, ["rev-parse", "HEAD"])
    _rc, url, _err = _git(repo, ["remote", "get-url", remote])
    _rc, head_line, _err = _git(repo, ["log", "-1", "--format=%s%x1f%ct", "HEAD"])
    subject, _, when = head_line.partition("\x1f")
    info.update({"remote": remote, "branch": branch, "local": local[:9],
                 "current": {"subject": subject, "at": int(when) if when.isdigit() else None},
                 "behind": 0, "ahead": 0, "commits": [], "changes": release_diff.classify(kind, [])})
    rc, out, err = _git(repo, ["ls-remote", url, f"refs/heads/{branch}"], timeout=30)
    if rc != 0:
        info["error"] = f"could not reach {remote}: {err or 'no answer'}"
        return info
    if not out:
        info["error"] = f"the branch it follows, {branch}, no longer exists on {remote}"
        return info
    tip = out.split()[0]
    info["target"] = tip[:9]
    if tip == local:
        return info
    try:
        compared = _compare(url, local, tip)
    except Exception as exc:  # noqa: BLE001 - reported on the panel, never raised
        info["behind"] = None
        info["error"] = f"an update is waiting, but its details could not be read ({exc})"
        return info
    info["behind"] = compared.get("ahead_by", 0)
    info["ahead"] = compared.get("behind_by", 0)
    info["commits"] = [{
        "sha": c.get("sha", "")[:9],
        "subject": ((c.get("commit") or {}).get("message") or "").splitlines()[0] if c.get("commit") else "",
        "author": (((c.get("commit") or {}).get("author")) or {}).get("name", ""),
        "at": _epoch(((c.get("commit") or {}).get("author") or {}).get("date")),
    } for c in reversed(compared.get("commits") or [])][:30]
    info["changes"] = release_diff.classify(kind, [f.get("filename", "") for f in compared.get("files") or []],
                                            repo)
    return info


def _epoch(stamp) -> Optional[int]:
    try:
        return calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))
    except (TypeError, ValueError):
        return None


def blockers(info: dict) -> list:
    reasons = []
    if info.get("ahead"):
        reasons.append(f"the deployed commit is not on the branch it follows ({info['ahead']} commit(s) apart)")
    if info.get("kind") == "server" and (info.get("changes") or {}).get("action") == release_diff.IMAGE:
        reasons.append("it changes the image (" + ", ".join(info["changes"]["heavy"][:3]) +
                       "), which a pull cannot deliver: build and publish an image")
    return reasons


def tools_repo() -> str:
    tools = str(settings.TOOLS_DIR).split(os.pathsep)[0]
    rc, top, _err = _git(tools, ["rev-parse", "--show-toplevel"])
    return top if rc == 0 else ""


def run_check() -> dict:
    """Check both checkouts now. Blocking: seconds, mostly the network."""
    global _last, _running
    result = {"at": time.time()}
    for kind, repo in (("server", settings.SERVER_REPO), ("tools", tools_repo())):
        info = check_repo(repo, kind)
        info["blockers"] = blockers(info)
        result[kind] = info
    with _lock:
        _last = result
        _running = False
    return result


def last(auto: bool = True) -> Optional[dict]:
    """The latest check, starting a new one in the background when the last is
    older than `AUTO_CHECK_SECONDS` -- so an open panel stays current without a
    poll ever waiting on the network."""
    global _running
    with _lock:
        stale = _last is None or time.time() - _last["at"] > AUTO_CHECK_SECONDS
        start = auto and stale and not _running
        if start:
            _running = True
        current = dict(_last) if _last else None
    if start:
        _background()
    if current is not None:
        current["running"] = _running
    return current


def _background() -> None:
    """Run a check off the request path. Replaced in tests: a panel read must
    not reach the network there."""
    threading.Thread(target=run_check, name="update-check", daemon=True).start()


def reset() -> None:
    """Forget the last check. For tests."""
    global _last, _running
    with _lock:
        _last, _running = None, False
