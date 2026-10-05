"""Per-workstation rules for cohorts, and the gate that enforces them.

A clinician who processes a cohort sends it as several runs -- the client cuts
the folder into batches by the limits `GET /tools` publishes -- and each batch
carries the same batch id, its index and the total (`X-Batch-*`). What this
module decides is whether one workstation's batches may run side by side:

* **serial** (the default): one batch of a cohort at a time, in index order.
  That is how every client behaved before, and it leaves the rest of the
  machine to the other workstations.
* **parallel**: as many at once as the tool slots and the admission budget
  allow. An operator turns this on for a cohort that genuinely matters, and
  that workstation then takes a large share of the server.

Keyed by the address the server sees (`_client_address` in main.py), which is
the workstation once the proxy's `X-Forwarded-For` is trusted. There is no
other identity: one API token is shared by every workstation.

The rules are kept beside the run history so an operator's decision survives
an update; the gate itself is in memory, like admission, and covers this
process.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Optional

from config import settings

logger = logging.getLogger("inference_server")

SERIAL = "serial"
PARALLEL = "parallel"
POLICIES = (SERIAL, PARALLEL)
POLICY_FILE = "client_policies.json"

_lock = threading.Lock()
_path: Optional[str] = None
_policies: dict = {}           # address -> {"batches": ..., "updated_at": ...}


def configure(directory: Optional[str]) -> None:
    """Keep the rules under `directory`, and read back what is already there."""
    global _path, _policies
    with _lock:
        _path, _policies = None, {}
        if not directory:
            return
        path = os.path.join(directory, POLICY_FILE)
        try:
            os.makedirs(directory, exist_ok=True)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as handle:
                    loaded = json.load(handle) or {}
                _policies = {address: rule for address, rule in loaded.items()
                             if isinstance(rule, dict) and rule.get("batches") in POLICIES}
            _path = path
        except (OSError, ValueError) as exc:
            logger.warning("Client rules kept in memory only, %s is not usable: %s", directory, exc)


def default_policy() -> str:
    return settings.DEFAULT_BATCH_POLICY if settings.DEFAULT_BATCH_POLICY in POLICIES else SERIAL


def policy_for(address: Optional[str]) -> str:
    with _lock:
        rule = _policies.get(address or "")
    return rule["batches"] if rule else default_policy()


def set_policy(address: str, batches: str) -> dict:
    if batches not in POLICIES:
        raise ValueError(f"Unknown batch rule {batches!r}. Expected one of: {', '.join(POLICIES)}")
    if not address or len(address) > 64:
        raise ValueError("A client is named by its address.")
    with _lock:
        if batches == default_policy():
            _policies.pop(address, None)
        else:
            _policies[address] = {"batches": batches, "updated_at": time.time()}
        snapshot = dict(_policies)
        path = _path
    if path:
        try:
            staging = path + ".tmp"
            with open(staging, "w", encoding="utf-8") as handle:
                json.dump(snapshot, handle)
            os.replace(staging, path)
        except OSError as exc:
            logger.warning("Could not save the client rules to %s: %s", path, exc)
    return {"client": address, "batches": policy_for(address)}


def policies() -> dict:
    """Every address with a rule that is not the default."""
    with _lock:
        return {address: dict(rule) for address, rule in _policies.items()}


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
#
# A serial cohort lets exactly one of its batches past at a time, the lowest
# index waiting. The client used to guarantee that by sending one at a time;
# the server guarantees it now, so a client may send its batches together and
# the operator's rule -- not the client's config -- decides how they run.

_gate_lock = threading.Lock()
_active: dict = {}             # (address, batch id) -> set of run ids
_waiting: dict = {}            # (address, batch id) -> {run id: index}
# Every index of a cohort that has reached the gate, and when the cohort was
# last seen, so a batch can tell "batch 1 has not arrived yet" from "batch 1
# already ran". Forgotten after a while of nothing.
_seen: dict = {}               # (address, batch id) -> {"indices": set, "at": t}
_arrived: dict = {}            # run id -> when it reached the gate

# How long a serial batch waits for a LOWER index that has not arrived yet.
# A client sending its batches together sends them within a second or two; a
# lower batch that never comes (cancelled before it was sent) must not hold
# the rest of the cohort for ever.
ARRIVAL_GRACE_SECONDS = 10.0
_SEEN_TTL_SECONDS = 6 * 3600


def _key(address, batch) -> tuple:
    return (address or "", batch["id"])


def wait(address: Optional[str], batch: dict, run_id: str) -> None:
    """Declare that this batch wants to start. Pair with `may_start`/`leave`."""
    now = time.time()
    key = _key(address, batch)
    with _gate_lock:
        for stale in [k for k, v in _seen.items() if now - v["at"] > _SEEN_TTL_SECONDS]:
            _seen.pop(stale, None)
        _waiting.setdefault(key, {})[run_id] = batch["index"]
        seen = _seen.setdefault(key, {"indices": set(), "at": now})
        seen["indices"].add(batch["index"])
        seen["at"] = now
        _arrived.setdefault(run_id, now)


def may_start(address: Optional[str], batch: dict, run_id: str) -> bool:
    """True, and the batch is marked running, when it may go now.

    Parallel: always. Serial: when no other batch of this cohort is running,
    this one has the lowest index of those waiting, and every lower index has
    already arrived -- or has been given `ARRIVAL_GRACE_SECONDS` to.
    """
    key = _key(address, batch)
    with _gate_lock:
        waiting = _waiting.get(key, {})
        if policy_for(address) == SERIAL:
            if _active.get(key):
                return False
            if waiting and min(waiting.values()) < waiting.get(run_id, batch["index"]):
                return False
            seen = _seen.get(key, {"indices": set()})["indices"]
            missing = any(index not in seen for index in range(1, batch["index"]))
            if missing and time.time() - _arrived.get(run_id, 0) < ARRIVAL_GRACE_SECONDS:
                return False
        waiting.pop(run_id, None)
        _arrived.pop(run_id, None)
        if not waiting:
            _waiting.pop(key, None)
        _active.setdefault(key, set()).add(run_id)
        return True


def leave(address: Optional[str], batch: dict, run_id: str) -> None:
    """The batch ended, or gave up waiting."""
    key = _key(address, batch)
    with _gate_lock:
        _arrived.pop(run_id, None)
        waiting = _waiting.get(key)
        if waiting is not None:
            waiting.pop(run_id, None)
            if not waiting:
                _waiting.pop(key, None)
        active = _active.get(key)
        if active is not None:
            active.discard(run_id)
            if not active:
                _active.pop(key, None)


def reset() -> None:
    """Forget every rule and every gate. For tests."""
    global _path, _policies
    with _lock:
        _path, _policies = None, {}
    with _gate_lock:
        _active.clear()
        _waiting.clear()
        _seen.clear()
        _arrived.clear()
