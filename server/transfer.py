"""Chunked resumable uploads, and results served in byte ranges.

Why this exists next to the plain multipart path in main.py: one HTTP request
moves one file over one TCP connection, and on a long-haul link a single
connection is bound by the congestion window long before it is bound by
bandwidth. A 100 MB CBCT to a remote GPU server is minutes that way, and a
connection dropped at 95% starts again from zero. Splitting the file into
parts the client sends over SEVERAL connections at once removes both problems:
the transfer scales with the number of streams instead of with one window, and
a part that fails is a part that is retried, not a file that is restarted.

The endpoints in main.py are thin wrappers; everything that touches the
filesystem, validates an id, or decides what a session is allowed to do lives
here so it can be unit-tested without an HTTP client.

Two invariants make the concurrency safe without a single lock:

- **Parts never overlap.** Part `n` occupies `[n * chunk_size, (n+1) *
  chunk_size)` and is written with `os.pwrite` at that offset, so concurrent
  parts write disjoint ranges of one preallocated file. There is no
  reassembly pass afterwards -- the blob IS the file as soon as the last part
  lands, so each uploaded byte is written to disk exactly once.
- **State lives on disk, not in this process.** A received part is recorded by
  creating a zero-byte marker file, which is atomic. Nothing here is a module
  global, so the server keeps working under `uvicorn --workers N` (parts of
  one upload may legitimately be served by different workers) and a session
  survives the `--reload` that a code edit triggers mid-transfer.

Integrity is per part, not per file: the client sends each part's SHA-256 and
the server refuses to write a part whose bytes do not match. Since the parts
tile the file exactly and every one of them is checked, the assembled blob is
verified in full without either side ever making a second pass over it. This
matters more here than the speed does -- a truncated or corrupted CBCT that
reaches a tool is a wrong result, not an error.
"""

import errno
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import time
from dataclasses import dataclass
from typing import Optional

from config import settings

logger = logging.getLogger("inference_server.transfer")

# Ids go straight into a filesystem path, so they are matched against this
# BEFORE any path is built from them -- "../../etc" must never be looked up,
# not even to be reported as missing. token_urlsafe's alphabet is exactly
# [A-Za-z0-9_-], so a legitimate id always passes.
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")

_META_NAME = "meta.json"
_BLOB_NAME = "blob"
_PARTS_DIR = "parts"


class TransferError(Exception):
    """Anything the CLIENT got wrong: an unknown id, a part out of range, a
    checksum that does not match. main.py maps this to a 4xx -- it is never a
    server bug, so it must reach the caller with its message intact."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _upload_root() -> str:
    return os.path.join(settings.TEMP_DIR, "uploads")


def _result_root() -> str:
    return os.path.join(settings.TEMP_DIR, "results")


def _validated_dir(root: str, identifier: str, kind: str) -> str:
    if not _ID_RE.match(identifier or ""):
        raise TransferError(f"Malformed {kind} id.", status_code=404)
    path = os.path.join(root, identifier)
    if not os.path.isdir(path):
        raise TransferError(
            f"Unknown or expired {kind} '{identifier}'. Start it again.", status_code=404
        )
    return path


def _read_meta(directory: str) -> dict:
    with open(os.path.join(directory, _META_NAME), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_meta(directory: str, meta: dict) -> None:
    """Written once, at creation, and never mutated. That is what lets every
    later request read it without a lock: there is no version to race on."""
    with open(os.path.join(directory, _META_NAME), "w", encoding="utf-8") as handle:
        json.dump(meta, handle)


# ----------------------------------------------------------------------
# Expiry
# ----------------------------------------------------------------------

def touch(directory: str) -> None:
    """Mark a session or result as still in use, right now.

    Called on every part written and every range read, which is what makes
    TRANSFER_TTL_SECONDS an IDLE timeout rather than an age limit: a transfer
    that is still moving can take as long as it needs, and one whose client
    vanished expires quickly. Without it the two cases are indistinguishable
    and the TTL has to be sized for the slowest imaginable transfer, which is
    the opposite of what a confidentiality bound wants.

    Best effort: a failed `utime` costs a directory an early reap, never a
    failed request, and the reap only happens if nothing touches it again for
    the whole TTL.
    """
    try:
        os.utime(directory)
    except OSError as exc:
        logger.debug("could not touch %s: %s", directory, exc)


def reap_expired(now: Optional[float] = None) -> int:
    """Delete upload sessions and results untouched for TRANSFER_TTL_SECONDS.

    Runs both on a timer (see main.py's lifespan) and opportunistically when a
    session or result is created. The timer is the one that matters: the case
    where an abandoned transfer sits longest is precisely the case where no new
    request arrives to trigger the opportunistic sweep, so creation-time
    reaping alone would let an idle server hold patient data indefinitely.

    An abandoned transfer is the only way these directories leak, since every
    normal path deletes its own. But "abandoned" is the common case for the
    very situation chunked transfer exists to survive: a client that lost its
    connection, or a Slicer that was closed mid-download, and never came back.
    """
    now = time.time() if now is None else now
    deadline = now - settings.TRANSFER_TTL_SECONDS
    removed = 0
    for root in (_upload_root(), _result_root()):
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
                continue
            # ignore_errors: several uvicorn workers run their own reaper, so
            # losing the race to delete the same directory is expected.
            shutil.rmtree(directory, ignore_errors=True)
            removed += 1
    if removed:
        logger.info("transfer reaper removed %d expired directory(ies)", removed)
    return removed


# ----------------------------------------------------------------------
# Uploads
# ----------------------------------------------------------------------

@dataclass
class UploadSession:
    upload_id: str
    directory: str
    filename: str
    size: int
    chunk_size: int

    @property
    def blob_path(self) -> str:
        return os.path.join(self.directory, _BLOB_NAME)

    @property
    def part_count(self) -> int:
        # A zero-byte file is one (empty) part rather than none, so "every part
        # received" stays the same check for it as for everything else.
        if self.size == 0:
            return 1
        return (self.size + self.chunk_size - 1) // self.chunk_size

    def received_parts(self) -> set:
        try:
            return {int(name) for name in os.listdir(os.path.join(self.directory, _PARTS_DIR))}
        except FileNotFoundError:
            return set()

    def missing_parts(self) -> list:
        received = self.received_parts()
        return [index for index in range(self.part_count) if index not in received]


def create_upload(filename: str, size: int, chunk_size: Optional[int] = None) -> UploadSession:
    """Reserve space for `size` bytes and hand back the session to fill it.

    The blob is created at its final length up front (`os.truncate`) so every
    part can be written at its own offset from the first request onwards. On
    every filesystem the server realistically runs on this is a sparse file:
    it costs no blocks until the parts actually arrive, so a client that
    abandons a 2 GB upload after one part leaves one part's worth of disk
    behind, not 2 GB.
    """
    if size < 0:
        raise TransferError("Declared size cannot be negative.")
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if size > max_bytes:
        # The whole point of declaring the size first: a file over the limit is
        # refused before a single byte of it travels, which the multipart path
        # cannot do (it only finds out while receiving).
        raise TransferError(
            f"File exceeds the {settings.MAX_UPLOAD_MB} MB limit.", status_code=413
        )

    chunk_size = _clamped_chunk_size(chunk_size)
    reap_expired()

    root = _upload_root()
    os.makedirs(root, exist_ok=True)
    upload_id = secrets.token_urlsafe(24)
    directory = os.path.join(root, upload_id)
    os.makedirs(os.path.join(directory, _PARTS_DIR))

    session = UploadSession(
        upload_id=upload_id,
        directory=directory,
        # Stored for the extension check main.py runs on it, never used to
        # build a path: the blob is always called "blob", so a client-supplied
        # name cannot decide where anything is written.
        filename=os.path.basename(filename or "upload"),
        size=size,
        chunk_size=chunk_size,
    )
    with open(session.blob_path, "wb"):
        pass
    os.truncate(session.blob_path, size)
    _write_meta(
        directory,
        {"filename": session.filename, "size": size, "chunk_size": chunk_size},
    )
    logger.info(
        "upload session created: %d byte(s) in %d part(s) of %d",
        size, session.part_count, chunk_size,
    )
    return session


def _clamped_chunk_size(requested: Optional[int]) -> int:
    """The client may ask for a part size; the server decides.

    Left to the client, a too-small value turns a 2 GB upload into thousands of
    requests (each with its own headers, auth check and fsync) and a too-large
    one gives the parallelism nothing to work with -- one 2 GB part is exactly
    the single-stream transfer this module exists to replace.
    """
    default = settings.UPLOAD_CHUNK_MB * 1024 * 1024
    if not requested:
        return default
    return max(1024 * 1024, min(int(requested), 64 * 1024 * 1024))


def get_upload(upload_id: str) -> UploadSession:
    directory = _validated_dir(_upload_root(), upload_id, "upload")
    meta = _read_meta(directory)
    return UploadSession(
        upload_id=upload_id,
        directory=directory,
        filename=meta["filename"],
        size=meta["size"],
        chunk_size=meta["chunk_size"],
    )


def write_part(session: UploadSession, index: int, data: bytes, sha256: Optional[str] = None) -> int:
    """Write one part at its offset. Idempotent: re-sending a part it already
    holds is how a client resumes, so it must not be an error.

    Returns how many parts the session still expects, so the caller can answer
    without a second listdir.
    """
    if index < 0 or index >= session.part_count:
        raise TransferError(
            f"Part {index} is outside this upload's {session.part_count} part(s)."
        )

    offset = index * session.chunk_size
    expected = min(session.chunk_size, session.size - offset)
    if len(data) != expected:
        # Only the LAST part is short, and only by exactly this much. A
        # mismatch means the two sides disagree about the layout, and writing
        # anyway would silently produce a file with a hole or an overlap.
        raise TransferError(
            f"Part {index} is {len(data)} bytes, expected {expected}."
        )

    if sha256:
        actual = hashlib.sha256(data).hexdigest()
        if not secrets.compare_digest(actual, sha256.lower()):
            # Never written to disk: the client retries this part alone, which
            # is the whole reason integrity is checked per part rather than
            # once over the assembled file.
            raise TransferError(f"Part {index} failed its checksum; send it again.")

    file_descriptor = os.open(session.blob_path, os.O_WRONLY)
    try:
        written = 0
        while written < len(data):
            written += os.pwrite(file_descriptor, data[written:], offset + written)
    finally:
        os.close(file_descriptor)

    marker = os.path.join(session.directory, _PARTS_DIR, str(index))
    try:
        # O_EXCL rather than a plain create: two workers racing on the same
        # part is legitimate (a client retrying a part whose response it never
        # saw), and this makes the loser a no-op instead of a second write.
        os.close(os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        pass

    return session.part_count - len(session.received_parts())


def claim_upload(upload_id: str, destination: str) -> str:
    """Hand the finished blob over to the request that is going to use it, and
    close the session.

    `os.rename` rather than a copy: the session and the request work dir are
    both under TEMP_DIR, so this is a directory entry change -- a 2 GB upload
    becomes a tool's input in microseconds instead of a second full read and
    write. The fallback covers the one case where that is not true (TEMP_DIR
    spanning two mounts), which no supported deployment has but which would
    otherwise fail the request outright.
    """
    session = get_upload(upload_id)
    missing = session.missing_parts()
    if missing:
        shown = ", ".join(str(index) for index in missing[:5])
        suffix = ", ..." if len(missing) > 5 else ""
        raise TransferError(
            f"Upload '{upload_id}' is incomplete: {len(missing)} part(s) missing ({shown}{suffix})."
        )

    actual_size = os.path.getsize(session.blob_path)
    if actual_size != session.size:
        raise TransferError(
            f"Upload '{upload_id}' is {actual_size} bytes, {session.size} were declared."
        )

    try:
        os.rename(session.blob_path, destination)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copyfile(session.blob_path, destination)
    shutil.rmtree(session.directory, ignore_errors=True)
    return destination


def discard_upload(upload_id: str) -> None:
    """Drop a session the client no longer wants. Best effort on purpose: a
    client cancelling a run should never be handed an error for it, and the
    reaper collects whatever this misses."""
    try:
        directory = _validated_dir(_upload_root(), upload_id, "upload")
    except TransferError:
        return
    shutil.rmtree(directory, ignore_errors=True)


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------

@dataclass
class StoredResult:
    result_id: str
    directory: str
    filename: str
    media_type: str
    size: int

    @property
    def blob_path(self) -> str:
        return os.path.join(self.directory, _BLOB_NAME)

    def as_reference(self) -> dict:
        """What /run answers with instead of the bytes, so the client can come
        back for them over as many connections as it likes."""
        return {
            "result_id": self.result_id,
            "filename": self.filename,
            "media_type": self.media_type,
            "size": self.size,
        }


def store_result(path: str, media_type: str) -> StoredResult:
    """Move a tool's output somewhere it outlives the request that made it.

    Needed because a range-served download is N separate requests: the work dir
    the tool wrote into is deleted the moment /run answers, so the file has to
    leave it first. Same rename-not-copy reasoning as claim_upload.
    """
    reap_expired()
    root = _result_root()
    os.makedirs(root, exist_ok=True)
    result_id = secrets.token_urlsafe(24)
    directory = os.path.join(root, result_id)
    os.makedirs(directory)

    blob_path = os.path.join(directory, _BLOB_NAME)
    try:
        os.rename(path, blob_path)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copyfile(path, blob_path)

    stored = StoredResult(
        result_id=result_id,
        directory=directory,
        # The name the client should save it under -- the blob itself is always
        # called "blob", so nothing derived from a tool's output name can
        # decide where anything is written.
        filename=os.path.basename(path),
        media_type=media_type,
        size=os.path.getsize(blob_path),
    )
    _write_meta(
        directory,
        {"filename": stored.filename, "media_type": stored.media_type, "size": stored.size},
    )
    return stored


def get_result(result_id: str) -> StoredResult:
    directory = _validated_dir(_result_root(), result_id, "result")
    meta = _read_meta(directory)
    return StoredResult(
        result_id=result_id,
        directory=directory,
        filename=meta["filename"],
        media_type=meta["media_type"],
        size=os.path.getsize(os.path.join(directory, _BLOB_NAME)),
    )


def discard_result(result_id: str) -> None:
    """Called by the client once it has the whole file. The reaper is the
    safety net for the client that never gets that far."""
    try:
        directory = _validated_dir(_result_root(), result_id, "result")
    except TransferError:
        return
    shutil.rmtree(directory, ignore_errors=True)


def parse_range(header: Optional[str], size: int) -> Optional[tuple]:
    """`Range: bytes=start-end` -> inclusive (start, end), or None for "send it all".

    Only the single-range form is honoured. A multipart/byteranges response is
    a different content type with its own framing, and nothing needs it: a
    client wanting several ranges at once gets more out of asking for them on
    several connections, which is the entire point here.

    Raises TransferError(416) for a range that cannot be satisfied, which is
    what tells a client its idea of the file's length is stale.
    """
    if not header:
        return None
    match = re.fullmatch(r"\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*", header)
    if not match:
        return None
    raw_start, raw_end = match.group(1), match.group(2)
    if not raw_start and not raw_end:
        return None

    if not raw_start:
        # "bytes=-500": the LAST 500 bytes, not the first.
        length = int(raw_end)
        if length == 0:
            raise TransferError("Unsatisfiable range.", status_code=416)
        start = max(0, size - length)
        end = size - 1
    else:
        start = int(raw_start)
        end = int(raw_end) if raw_end else size - 1
        end = min(end, size - 1)

    if start >= size or start > end:
        raise TransferError("Unsatisfiable range.", status_code=416)
    return start, end


def read_range(path: str, start: int, end: int, chunk_size: int = 1024 * 1024):
    """Yield `[start, end]` of `path`. A generator, so the response streams
    straight off the disk and a range the size of the whole file costs the same
    memory as a small one."""
    remaining = end - start + 1
    with open(path, "rb") as handle:
        handle.seek(start)
        while remaining > 0:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                return
            remaining -= len(chunk)
            yield chunk
