"""Whether this server is accepting new work, and the guarantee that it reopens.

An update has to stop new runs arriving, let the ones in flight finish, and
only then have the process replaced. Nothing here decides WHEN that happens:
the updater runs on the host, outside the container, because the container
cannot see the deployment's `.git`. This module is only the door, plus the one
property that makes a door safe to close remotely -- that it cannot be left
shut.

**Closed is a DEADLINE, not a boolean, and that is the whole design.** A
boolean needs someone to come back and clear it; a deadline clears itself. The
failure this removes is the updater dying between closing the door and
restarting the process, which would otherwise leave a clinic with a server that
answers `/health` and refuses every run until a human notices. There are two
independent ways back:

* the deadline passes, and `accepting()` answers True again;
* the state lives in memory, so any restart -- including the `--reload` the
  update itself triggers -- starts with the door open.

The window is meant to be short. The updater waits for the server to go idle
with the door OPEN, and only closes it once there is nothing left to wait for,
so what is held is the restart, not the drain. `MAX_CLOSE_SECONDS` is a
backstop against a caller that asks for more, not the expected duration.
"""

import threading
import time
from typing import Optional

from fastapi import HTTPException, status

# The longest any single request may shut the door for. Generous against a
# restart measured in seconds, and short enough that the worst case of a
# updater that vanished is a clinic waiting minutes rather than until someone
# looks. A caller that needs longer asks again, which is also what proves it is
# still alive.
MAX_CLOSE_SECONDS = 300.0

# What a refused caller is told to wait. Deliberately not the full remaining
# deadline: the common case is a restart, after which the door is open, and a
# client that backed off for five minutes would be idle long after the server
# came back.
RETRY_AFTER_SECONDS = 10

# How long an OPERATOR may close the door for from the admin panel. Longer
# than an updater's lease because the decision is a person's, made on purpose
# -- "stop taking runs, I am about to update" -- and the drain it waits for can
# be a multi-hour cohort. Still bounded: an operator who forgets is a clinic
# refused for at most this long, and any restart opens it.
OPERATOR_MAX_CLOSE_SECONDS = 12 * 3600.0

_lock = threading.Lock()
_closed_until: Optional[float] = None
_reason: str = ""
_by_operator: bool = False


def close(seconds: float, reason: str = "") -> float:
    """Stop accepting new work for at most `seconds`. Returns what was granted.

    Clamped at both ends: a non-positive value reopens rather than closing for
    no time, and anything past `MAX_CLOSE_SECONDS` is cut to it. The caller is
    told what it actually got rather than what it asked for, so an updater that
    wanted an hour learns it has five minutes and refreshes.
    """
    global _closed_until, _reason, _by_operator

    with _lock:
        if _by_operator and _closed_until is not None and time.monotonic() < _closed_until:
            # An operator closed it for longer: an updater's lease must not
            # shorten that, only an operator reopening it may.
            return round(_closed_until - time.monotonic(), 1)
    granted = min(float(seconds), MAX_CLOSE_SECONDS)
    if granted <= 0:
        reopen()
        return 0.0
    with _lock:
        _closed_until = time.monotonic() + granted
        _reason = reason
        _by_operator = False
    return granted


def close_by_operator(seconds: float, reason: str = "") -> float:
    """Stop accepting new work for at most `OPERATOR_MAX_CLOSE_SECONDS`.

    What the admin panel's "stop accepting new runs" does: the runs already in
    are untouched and finish, new ones are refused with a 503 saying why.
    """
    global _closed_until, _reason, _by_operator

    granted = min(float(seconds), OPERATOR_MAX_CLOSE_SECONDS)
    if granted <= 0:
        reopen()
        return 0.0
    with _lock:
        _closed_until = time.monotonic() + granted
        _reason = reason or "closed by an operator"
        _by_operator = True
    return granted


def reopen() -> None:
    """Accept work again, now."""
    global _closed_until, _reason, _by_operator

    with _lock:
        _closed_until = None
        _reason = ""
        _by_operator = False


def accepting() -> bool:
    """Whether this server will take new work.

    Reading it is what expires it: no timer, no task, nothing to schedule and
    nothing that can fail to run. The deadman is the comparison itself.
    """
    with _lock:
        if _closed_until is None:
            return True
        return time.monotonic() >= _closed_until


def snapshot() -> dict:
    """`{accepting, closed_for, reason}` for `/status` and for the updater.

    `closed_for` is the seconds remaining, rounded, or None when open -- what
    an operator needs to tell "an update is in progress" from "something shut
    this and went away", the second being visible only as a number that keeps
    counting down without the server ever restarting.
    """
    with _lock:
        if _closed_until is None:
            return {"accepting": True, "closed_for": None, "reason": "", "by_operator": False}
        remaining = _closed_until - time.monotonic()
        if remaining <= 0:
            return {"accepting": True, "closed_for": None, "reason": "", "by_operator": False}
        return {
            "accepting": False,
            "closed_for": round(remaining, 1),
            "reason": _reason,
            "by_operator": _by_operator,
        }


def require_accepting() -> None:
    """FastAPI dependency: refuse this request while the door is shut.

    A dependency and not middleware, and the reason is a bug this codebase has
    already made once. Middleware has to recognise a route by its PATH, which
    means rewriting the router's matching by hand -- and `_UNCOUNTED_PATHS` in
    `main.py` shows what that costs: its `"/runs/"` prefix quietly swallows
    `POST /runs/{id}/resume`, a real run of real GPU, and stops counting it.
    The first draft of this gate made the mirror-image mistake: a `"/run/"`
    prefix that does NOT match `/runs/{id}/resume`, so a paused cohort could be
    resumed in the middle of an update.

    A dependency is matched by the router itself, so there is nothing to keep
    in sync. It also costs nothing on the read paths -- `GET /runs/{id}/events`
    never evaluates it -- and it is visible at the route it guards.

    What it must NOT be attached to is as deliberate as what it guards: a
    client has to be able to watch a run it already started, collect its
    result, resume an upload it is halfway through, and cancel. Refusing those
    would strand exactly the people an update is meant to serve.
    """
    if accepting():
        return
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=(
            "This server is being updated and is not accepting new work. "
            "Nothing already running has been interrupted, and anything you "
            "have already uploaded is still here. Try again shortly."
        ),
        # Without this a client has nothing to schedule a retry from -- and it
        # is also what tells this 503 apart from ToolUnavailableError's, which
        # means the opposite thing: waiting will not help.
        headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
    )
