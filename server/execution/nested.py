"""Admission for nested calls: a tool another tool called through its supervisor
queues, and is sized, exactly like a run that arrived over HTTP.

A nested call used to be a subprocess living inside its parent's reservation.
That had two costs. The parent reserved its whole chain's worst case for the
whole run -- an orchestrator spending most of its life on a CPU registration
held the card its segmentation step needed an hour earlier -- and a child could
never be wider than the share it inherited, however empty the machine was.

Now each run gets a DESK: a Unix socket in its own job directory. When the
supervisor is about to start a child it connects, names the child's job file,
and waits. The desk prices the child from the child's own measured cost, queues
it in the same `admission.Budget` every run goes through -- in the nested band,
ahead of runs not yet started -- and answers with what was granted: the width,
the threads and the room, as the environment the child is started with.

**The connection IS the reservation.** The supervisor keeps it open while the
child runs and closes it when the child returns. A process that dies closes its
sockets, so a crashed level releases what it held without anybody noticing it
crashed; nothing here can leak a reservation past the process that held it.

What is NOT here: starting or killing anything. The supervisor starts the child,
in the root's process group as it always did, so a timeout or a cancel that
signals the root's group still stops every level at once.

Standard library only on the other end: the runner speaks this with `socket`
and `json`, from inside a tool's own virtualenv.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import secrets
import select
import socket
import tempfile
import threading
from typing import Callable, Optional

from execution import admission

logger = logging.getLogger(__name__)

# What the root run's environment carries down every level. The key is checked
# on every request: the socket is the server's, and only the processes of this
# run were handed the key.
SOCKET_ENV = "SADT_ADMISSION_SOCKET"
KEY_ENV = "SADT_ADMISSION_KEY"
SOCKET_NAME = "admission.sock"
JOB_FILE = "job.json"
RESULT_FILE = "result.json"

# A Unix socket path is limited to ~108 bytes. A job directory is normally far
# below that; a deployment with a long TEMP_DIR gets a short private one instead.
_MAX_SOCKET_PATH = 100
_MAX_REQUEST_BYTES = 64 * 1024
_POLL_SECONDS = 1.0


def open_desk(*args, **kwargs) -> Optional["Desk"]:
    """A desk, or None when one cannot be built here -- in which case the run's
    nested calls run inside their parent's room, as they did before desks
    existed. Never a reason to fail a run."""
    try:
        return Desk(*args, **kwargs)
    except OSError as exc:
        logger.warning("No nested admission for this run (%s); its nested calls "
                       "run inside their parent's reservation.", exc)
        return None


class Desk:
    """The admission desk of one root run. Built by `dispatch` once the run is
    admitted, closed when its process exits.

    `price(tool_name, params)` returns the candidates to queue on, widest first,
    or None for a tool this server cannot price -- which the supervisor answers
    by running the child the way it always did, inside its parent's room.
    `environment(tool_name, grant)` turns a grant into the variables the child
    is started with. `learn(tool_name, payload, solo)` keeps what the child was
    measured to cost, under its own name.
    """

    def __init__(self, run_id: Optional[str], job_dir: str, grant: admission.Grant,
                 price: Callable, environment: Callable, learn: Callable):
        self.run_id = run_id
        self.root = os.path.realpath(job_dir)
        self.key = secrets.token_hex(16)
        self._price = price
        self._environment = environment
        self._learn = learn
        self._closed = threading.Event()
        self._lock = threading.Lock()
        # job directory -> the grants of the chain down to and including it.
        # A request names its caller's directory, which is how a grandchild
        # finds the two grants above it without anybody passing them along.
        self._chains = {self.root: (grant,)}
        self._private_dir = None
        self.path = self._socket_path()
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(self.path)
        os.chmod(self.path, 0o600)
        self._listener.listen(16)
        self._listener.settimeout(_POLL_SECONDS)
        self._thread = threading.Thread(target=self._accept, name="nested-desk", daemon=True)
        self._thread.start()

    def _socket_path(self) -> str:
        path = os.path.join(self.root, SOCKET_NAME)
        # Stale from a previous attempt of this run (a memory retry reuses the
        # directory); bind refuses an existing path.
        try:
            os.unlink(path)
        except OSError:
            pass
        if len(os.fsencode(path)) <= _MAX_SOCKET_PATH:
            return path
        self._private_dir = tempfile.mkdtemp(prefix="sadt-desk-")
        os.chmod(self._private_dir, 0o700)
        return os.path.join(self._private_dir, SOCKET_NAME)

    def environment(self) -> dict:
        """What the root run's process is started with, and passes down."""
        return {SOCKET_ENV: self.path, KEY_ENV: self.key}

    def close(self) -> None:
        """Stop admitting. Every held or waiting child lets go within a poll."""
        self._closed.set()
        try:
            self._listener.close()
        except OSError:
            pass
        for leftover in (self.path, self._private_dir):
            if not leftover:
                continue
            try:
                if os.path.isdir(leftover):
                    os.rmdir(leftover)
                else:
                    os.unlink(leftover)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    # -- the listener ----------------------------------------------------
    def _accept(self) -> None:
        while not self._closed.is_set():
            try:
                connection, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return  # closed under us
            threading.Thread(target=self._serve, args=(connection,),
                             name="nested-call", daemon=True).start()

    def _serve(self, connection: socket.socket) -> None:
        with connection:
            try:
                self._handle(connection)
            except Exception:  # noqa: BLE001 - one bad request must not take the desk down
                logger.exception("Nested admission request failed; the child runs in its parent's room.")
                _send(connection, {"inherit": "the server could not admit this call"})

    def _handle(self, connection: socket.socket) -> None:
        request = _read_line(connection)
        if not isinstance(request, dict) or not hmac.compare_digest(
                str(request.get("key") or ""), self.key):
            return
        job_path = os.path.realpath(str(request.get("job") or ""))
        child_dir = os.path.dirname(job_path)
        caller_dir = os.path.realpath(str(request.get("caller") or ""))
        # Only a job file inside THIS run's tree, named as the supervisor names
        # it. The path is read, so it must not be able to point anywhere else.
        if os.path.basename(job_path) != JOB_FILE or \
                not child_dir.startswith(self.root + os.sep):
            return
        with self._lock:
            chain = self._chains.get(caller_dir)
        if chain is None:
            _send(connection, {"inherit": "the caller is not a level of this run"})
            return
        with open(job_path, encoding="utf-8") as handle:
            job = json.load(handle)
        tool_name = str(job.get("tool") or "")
        candidates = self._price(tool_name, job.get("params") or {})
        if not candidates:
            _send(connection, {"inherit": f"'{tool_name}' cannot be priced here"})
            return

        def announce():
            _send(connection, {"queued": True})

        def given_up():
            return self._closed.is_set() or _hung_up(connection)

        try:
            with admission.budget().reserve(
                    candidates, on_wait=announce, is_cancelled=given_up,
                    run_id=job.get("job_id"), ancestors=chain, parent=self.run_id,
                    tool=tool_name) as grant:
                with self._lock:
                    self._chains[child_dir] = chain + (grant,)
                try:
                    _send(connection, {"granted": {
                        "channels": grant.channels, "cores": grant.cores,
                        "environment": self._environment(tool_name, grant)}})
                    self._hold(connection)
                finally:
                    with self._lock:
                        self._chains.pop(child_dir, None)
        except admission.Cancelled:
            return
        self._learn_from(tool_name, child_dir, grant.solo)

    def _hold(self, connection: socket.socket) -> None:
        """Until the supervisor hangs up -- the child returned, or died."""
        while not self._closed.is_set():
            readable, _, _ = select.select([connection], [], [], _POLL_SECONDS)
            if readable:
                try:
                    if not connection.recv(4096):
                        return
                except OSError:
                    return

    def _learn_from(self, tool_name: str, child_dir: str, solo: bool) -> None:
        path = os.path.join(child_dir, RESULT_FILE)
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return  # killed before it wrote anything: nothing to learn
        if isinstance(payload, dict):
            self._learn(tool_name, payload, solo)


def _send(connection: socket.socket, message: dict) -> None:
    try:
        connection.sendall((json.dumps(message) + "\n").encode("utf-8"))
    except OSError:
        pass


def _read_line(connection: socket.socket):
    connection.settimeout(10)
    data = b""
    while b"\n" not in data and len(data) < _MAX_REQUEST_BYTES:
        chunk = connection.recv(4096)
        if not chunk:
            break
        data += chunk
    connection.settimeout(None)
    try:
        return json.loads(data.split(b"\n", 1)[0].decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def _hung_up(connection: socket.socket) -> bool:
    """Has the other end closed? Without consuming anything it sent."""
    try:
        readable, _, _ = select.select([connection], [], [], 0)
        if not readable:
            return False
        return connection.recv(1, socket.MSG_PEEK) == b""
    except OSError:
        return True
