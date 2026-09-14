"""Ephemeral, id-addressed directories under TEMP_DIR, and how they expire.

`transfer.py` and `runs.py` both hold state that outlives the request that made
it -- an upload still being sent, a result waiting to be fetched, a run still
reporting progress -- and both hold it in a directory named by a client-minted
id. That shape brings two rules with it, and they are the same rules both
times, so they are written once here:

* **an id is matched against a pattern BEFORE any path is built from it.**
  "../../etc" must never be looked up, not even to be reported as missing.
* **expiry is IDLE, not age.** Every write and every read stamps the directory,
  so work still in flight is never reaped under itself however long it takes,
  while a directory whose client vanished goes minutes later. That is what lets
  the TTL be minutes rather than the hours an age limit would need.

What stays with each caller is what genuinely differs: which roots it sweeps,
which TTL setting bounds it, and what it calls the things it removed.
"""

import logging
import os
import re
import shutil
import time
from typing import Iterable, Optional

logger = logging.getLogger("inference_server.scratch")

# Ids are minted with `secrets.token_urlsafe(24)`, whose alphabet is exactly
# [A-Za-z0-9_-], so a legitimate id always passes. This enforces the SHAPE and
# nothing else: it cannot enforce entropy, and the id plus the bearer token is
# what authorises reaching the directory behind it.
ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


def is_valid_id(value: str) -> bool:
    """Whether `value` may be turned into a path at all."""
    return bool(value) and ID_RE.match(value) is not None


def touch(directory: str) -> None:
    """Mark a directory as still in use, right now.

    Best effort: a failed `utime` costs an early reap, never a failed request.
    """
    try:
        os.utime(directory)
    except OSError as exc:
        logger.debug("could not touch %s: %s", directory, exc)


def reap(roots: Iterable[str], ttl: float, label: str,
         now: Optional[float] = None) -> int:
    """Delete every directory under `roots` untouched for `ttl` seconds.

    Returns how many were removed. A missing root is not an error -- nothing
    has been written there yet.
    """
    deadline = (time.time() if now is None else now) - ttl
    removed = 0
    for root in roots:
        try:
            entries = os.listdir(root)
        except FileNotFoundError:
            continue
        for entry in entries:
            directory = os.path.join(root, entry)
            try:
                if os.path.getmtime(directory) > deadline:
                    continue
            except OSError:
                # Gone between listdir and getmtime: another worker won.
                continue
            # ignore_errors: every uvicorn worker runs its own reaper, so
            # losing the race to delete the same directory is expected.
            shutil.rmtree(directory, ignore_errors=True)
            removed += 1
    if removed:
        logger.info("%s reaper removed %d expired directory(ies)", label, removed)
    return removed
