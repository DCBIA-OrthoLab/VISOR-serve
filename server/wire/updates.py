"""What can be updated on this deployment, and the operator's request to do it.

The server cannot update itself: the container sees neither the deployment's
`.git` nor the tools checkout's, so the thing that pulls code runs on the HOST
(`scripts/update_agent.py`). The two meet in one folder both can write,
`UPDATE_DIR` -- `server/.update` in the bind-mounted checkout:

* `status.json` is written by the agent: for the server repository and the
  tools library, the commits waiting and what they touch (which tools, whether
  an environment has to be rebuilt), plus what the agent is doing right now and
  how its last update ended. Its `heartbeat` says the agent is alive.
* `request.json` is written here, when an operator clicks "Update" on the
  admin panel, and removed by the agent once it has acted on it. Removing it
  before the agent has started is how a request is withdrawn.

Nothing in this module runs git or touches a process. It reads and writes two
small files, atomically, and that is the whole of the server's part.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from typing import Optional

from config import settings

STATUS_FILE = "status.json"
REQUEST_FILE = "request.json"
TARGETS = ("all", "server", "tools")

# An agent that has not written for this long is reported as not running. Its
# own poll is a minute by default; three missed polls is not a slow network.
AGENT_SILENT_SECONDS = 300


class UpdateError(Exception):
    def __init__(self, message: str, status_code: int = 409):
        super().__init__(message)
        self.status_code = status_code


def _path(name: str) -> str:
    return os.path.join(settings.UPDATE_DIR, name)


def _read(name: str) -> Optional[dict]:
    try:
        with open(_path(name), encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _write(name: str, value: dict) -> None:
    os.makedirs(settings.UPDATE_DIR, exist_ok=True)
    staging = _path(name) + ".tmp"
    with open(staging, "w", encoding="utf-8") as handle:
        json.dump(value, handle)
    os.replace(staging, _path(name))


def overview() -> dict:
    """Everything the panel shows about updates, in one object."""
    status = _read(STATUS_FILE)
    request = _read(REQUEST_FILE)
    heartbeat = (status or {}).get("heartbeat") or 0
    return {
        "agent": {
            "seen": bool(status),
            "alive": bool(status) and time.time() - heartbeat < AGENT_SILENT_SECONDS,
            "heartbeat": heartbeat or None,
        },
        "status": status,
        "request": request,
    }


def request_update(target: str, by: Optional[str] = None) -> dict:
    """Ask the agent to apply what is waiting. One request at a time."""
    if target not in TARGETS:
        raise UpdateError(f"Unknown target {target!r}. Expected one of: {', '.join(TARGETS)}", 422)
    if _read(REQUEST_FILE):
        raise UpdateError("An update has already been requested; withdraw it or wait for it to finish.")
    request = {"id": secrets.token_hex(6), "target": target, "at": time.time(), "by": by}
    try:
        _write(REQUEST_FILE, request)
    except OSError as exc:
        raise UpdateError(f"Could not record the request in {settings.UPDATE_DIR}: {exc}", 500)
    return request


def withdraw() -> bool:
    """Remove a pending request. True if there was one.

    The agent checks for the file while it waits for runs to finish, so a
    request withdrawn during that wait stops the update and reopens the door.
    Once it is pulling, the file is already gone and this answers False.
    """
    try:
        os.remove(_path(REQUEST_FILE))
        return True
    except FileNotFoundError:
        return False
