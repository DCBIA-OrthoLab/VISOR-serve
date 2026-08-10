# This server processes confidential medical imaging data.
# It must be deployed in an appropriate jurisdiction (EU / a certified health
# host, depending on context) and only ever reached over TLS (see README.md).
# De-identification of patient data happens on the client side before upload;
# this server never logs file contents, argument values, or patient metadata.

import contextlib
import functools
import gzip
import json
import logging
import mimetypes
import os
import shutil
import tempfile
import time
import zlib
from typing import Optional

import anyio.to_thread
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.datastructures import UploadFile as StarletteUploadFile

import file_utils
import transfer
from base import (
    FILE_TYPES,
    FOLDER_TYPE,
    ResolvedPath,
    ToolArgumentError,
    ToolUnavailableError,
)
from config import settings
from data_store import DataNotFoundError, data_store
from registry import TOOLS, get_tool
from security import verify_token

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("inference_server")

os.makedirs(settings.TEMP_DIR, exist_ok=True)


async def _reaper_loop() -> None:
    """Sweep expired transfer directories for as long as the server runs.

    A timer, not only the opportunistic sweep transfer.py already does when a
    session is created: the case where an abandoned upload or an undownloaded
    result sits longest is exactly the case where no new request comes in to
    trigger that sweep. On a server holding confidential imaging, "cleaned up
    the next time somebody happens to use it" is not a bound.
    """
    while True:
        await anyio.sleep(settings.TRANSFER_SWEEP_SECONDS)
        try:
            await anyio.to_thread.run_sync(transfer.reap_expired)
        except Exception:  # noqa: BLE001 - one bad sweep must not end the loop
            logger.exception("transfer reaper sweep failed")


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(_reaper_loop)
        try:
            yield
        finally:
            # The loop never returns on its own; cancelling the scope is how it
            # ends, and without this the task group waits for it forever on
            # shutdown.
            task_group.cancel_scope.cancel()


app = FastAPI(lifespan=_lifespan)

_CHUNK_SIZE_BYTES = 1024 * 1024  # read/write in 1 MB chunks, never load the full file into RAM
_MAX_UPLOAD_BYTES = settings.MAX_UPLOAD_MB * 1024 * 1024
_MAX_EXTRACTED_BYTES = settings.MAX_EXTRACTED_MB * 1024 * 1024
_RESULT_REFERENCE_MIN_BYTES = settings.RESULT_REFERENCE_MIN_MB * 1024 * 1024


class _UploadTooLargeError(Exception):
    pass


_ACCEPT_ALL_EXTENSIONS = "*"

# Form field carrying {argument name: upload id} for inputs that travelled
# through the chunked-upload endpoints instead of this request's body (see
# transfer.py). Double-underscored so it can never collide with a tool's own
# argument name, and popped before anything looks at `args`.
_UPLOADS_FIELD = "__uploads__"

# Sent by a client that would rather be handed a reference to the result and
# fetch the bytes itself, over as many parallel range requests as it wants,
# than have them streamed down the same connection that carried the run. A
# client that does not send it gets exactly the response it always got.
_RESULT_DELIVERY_HEADER = "X-Result-Delivery"
_DELIVER_BY_REFERENCE = "reference"

# Caps how many tool executions run at once (see settings.MAX_CONCURRENT_TOOLS).
# Dedicated to tool runs only, so waiting inference jobs can never starve the
# threadpool used for everything else (sync endpoints, background cleanup).
# Created lazily on first use: anyio requires a running event loop to
# instantiate a CapacityLimiter, so it can't be built at import time.
_tool_limiter: Optional[anyio.CapacityLimiter] = None


def _get_tool_limiter() -> anyio.CapacityLimiter:
    global _tool_limiter
    if _tool_limiter is None:
        _tool_limiter = anyio.CapacityLimiter(settings.MAX_CONCURRENT_TOOLS)
    return _tool_limiter


def _extract_extension(filename: str) -> str:
    """Return the file's extension, preserving compound extensions like .nii.gz."""
    lower = filename.lower()
    parts = lower.split(".")
    if len(parts) >= 3 and parts[-1] in ("gz", "bz2", "xz"):
        return "." + ".".join(parts[-2:])
    if len(parts) >= 2:
        return "." + parts[-1]
    return ""


def _expected_extensions(tool, field_name: str) -> Optional[tuple]:
    """Return the specific extensions expected for this argument, across every
    file type it declares (see base.FILE_TYPES). None means "no specific type
    declared" -- fall back to settings.ALLOWED_EXTENSIONS.
    """
    spec = tool.arguments.get(field_name)
    if spec is None or not spec.is_file:
        return None
    return spec.extensions


def _matched_extension(filename: str, expected: Optional[tuple]) -> Optional[str]:
    """Return the extension to use for the saved file, or None to reject it.

    `expected` is the specific extension tuple for this argument (from the
    tool's own schema) if any; otherwise settings.ALLOWED_EXTENSIONS is used.
    "*" in either list accepts every extension, preserved as-is.
    """
    candidates = expected if expected is not None else settings.ALLOWED_EXTENSIONS

    if _ACCEPT_ALL_EXTENSIONS in candidates:
        return _extract_extension(filename)

    lower = filename.lower()
    for extension in candidates:
        if lower.endswith(extension):
            return extension
    return None


async def _stream_to_disk(upload: UploadFile, destination: str) -> int:
    """Write the upload to disk in chunks, never buffering the whole file in RAM."""
    size = 0
    with open(destination, "wb") as out_file:
        while chunk := await upload.read(_CHUNK_SIZE_BYTES):
            size += len(chunk)
            if size > _MAX_UPLOAD_BYTES:
                raise _UploadTooLargeError()
            out_file.write(chunk)
    return size


def _type_name(arg_type) -> str:
    return arg_type if isinstance(arg_type, str) else arg_type.__name__


def _resolved_kind(spec, path: str) -> str:
    """Which declared type a path on disk corresponds to, for ResolvedPath.kind."""
    if os.path.isdir(path):
        return FOLDER_TYPE
    if not spec.is_file:
        return "file"
    return spec.match_type(_extract_extension(os.path.basename(path)))


def _extract_folder_argument(spec, archive_path: str, work_dir: str, field_name: str) -> str:
    """Extract an archive sent for a "folder"-typed argument, so run() gets a
    directory. HTTP has no notion of a folder: the client zips it, the server
    unpacks it here, and the tool never sees the archive step at all.
    """
    try:
        return file_utils.extract_zip(
            archive_path,
            os.path.join(work_dir, f"{field_name}_folder"),
            strip_single_root=True,
            max_total_bytes=_MAX_EXTRACTED_BYTES,
        )
    except file_utils.BadArchiveError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Argument '{field_name}': {exc}",
        )


def _temp_root_of(path: str) -> Optional[str]:
    """Top-level folder under settings.TEMP_DIR containing `path`, or None if
    `path` lives outside TEMP_DIR. Used to clean up a tool's own scratch dir
    (see file_utils.make_scratch_dir) once its output has been streamed."""
    temp_dir = os.path.realpath(settings.TEMP_DIR)
    resolved = os.path.realpath(path)
    if os.path.commonpath([temp_dir, resolved]) != temp_dir or resolved == temp_dir:
        return None
    relative = os.path.relpath(resolved, temp_dir)
    return os.path.join(temp_dir, relative.split(os.sep)[0])


def _output_paths(result) -> list:
    """Normalize what run() returned into a list of output paths. Empty when
    the returned value isn't path-shaped at all."""
    if isinstance(result, (str, os.PathLike)):
        return [str(result)]
    if isinstance(result, (list, tuple)):
        return [str(path) for path in result]
    return []


def _discard(work_dir: Optional[str], scratch_dirs: list) -> None:
    """Remove everything this request created, right now. Used on the error
    paths, where no response will ever stream and background tasks won't run."""
    for directory in ([work_dir] if work_dir else []) + list(scratch_dirs):
        shutil.rmtree(directory, ignore_errors=True)


def _media_type_of(path: str) -> str:
    """Content-Type for a file about to be streamed. Derived from the real
    extension so an .xlsx is never mislabeled as a generic zip (see the
    2026-07-24 changelog entry); the gzip/octet-stream fallback covers bare
    .gz files (e.g. .nii.gz), which mimetypes cannot name."""
    media_type, _ = mimetypes.guess_type(str(path))
    if media_type is None:
        media_type = "application/gzip" if str(path).endswith(".gz") else "application/octet-stream"
    return media_type


def _human_bytes(size: int) -> str:
    """Byte count in the largest unit that keeps it readable.

    Logged alongside the exact figure, never instead of it: the round number
    is what a human scans for, the exact one is what stays greppable and
    comparable across requests.
    """
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024


def _log_served(tool_name: str, start_time: float, received: int, sent: Optional[int]) -> None:
    """One line per successfully served request.

    Called at each return point rather than before them, so `duration` covers
    packing the response too -- zipping a multi-GB segmentation is not free,
    and a duration that stopped at the end of run() understated the request by
    however long that took.

    `sent` is None for a "text" tool: its result travels as JSON and no output
    file exists to measure. Nothing here may name a file, an argument value,
    or any patient metadata (see the note at the top of this module).
    """
    duration = time.monotonic() - start_time
    sent_field = (
        "" if sent is None else f" sent={sent}B ({_human_bytes(sent)})"
    )
    logger.info(
        "endpoint=/run/%s status=200 duration=%.2fs received=%dB (%s)%s",
        tool_name,
        duration,
        received,
        _human_bytes(received),
        sent_field,
    )


def _output_roots(outputs: list, work_dir: str) -> set:
    """TEMP_DIR folders holding the tool's outputs, excluding the request's own
    work dir (already scheduled for cleanup by the caller)."""
    work_dir_real = os.path.realpath(work_dir)
    roots = set()
    for path in outputs:
        root = _temp_root_of(path if os.path.isdir(path) else os.path.dirname(path))
        if root is not None and root != work_dir_real:
            roots.add(root)
    return roots


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _extensions_of(spec) -> Optional[dict]:
    """{type name: [extension, ...]} for every file type an argument accepts.

    Published so a client never has to mirror FILE_TYPES: a type name does not
    reliably spell out its extensions ("nifti_file" is .nii/.nii.gz,
    "volume_or_zip_file" is seven of them), so a client guessing from the name
    alone gets them wrong the moment a type is added here.

    Keyed by type rather than flattened because the caller needs the split:
    the extensions of "folder" are what a zipped folder may be uploaded as,
    not what its file picker should offer.
    """
    per_type = {}
    for declared in spec.types:
        name = _type_name(declared)
        if name in FILE_TYPES:
            extensions = FILE_TYPES[name]
            per_type[name] = list(extensions) if extensions else None
    return per_type or None


@app.get("/tools")
def list_tools() -> list:
    """Let clients discover every registered tool and its expected arguments."""
    return [
        {
            "name": tool.name,
            "arguments": {
                arg_name: {
                    # "type" stays a single string for clients that predate
                    # multi-type arguments; "types" is the full list and is
                    # what a client should read to build its file picker.
                    "type": _type_name(spec.types[0]),
                    "types": [_type_name(declared) for declared in spec.types],
                    "required": spec.required,
                    "description": spec.description,
                    "server_selectable": spec.server_selectable,
                    # For "choice"/"multichoice": the options to render, each
                    # with its initial state. None for every other type.
                    "choices": spec.choices,
                    # For a SCALAR argument: the value a client should pre-fill
                    # its widget with, so a spin box does not start at Qt's 0
                    # while the tool's own default reads 5. null when the tool
                    # declares none, and for choice types (whose initial state
                    # is in "choices" above).
                    "initial": spec.initial,
                    # {type name: accepted extensions} for the file types
                    # above, so a client can build a file dialog's filters
                    # without a copy of FILE_TYPES hardcoded on its side --
                    # and without drifting when this table changes. null for
                    # a type the server does not restrict (the generic
                    # "file", which falls back to ALLOWED_EXTENSIONS), and
                    # None for an argument that takes no file at all.
                    "extensions": _extensions_of(spec),
                    # Presentation hints (see ArgSpec). Purely about how a
                    # client should lay this argument out and when to show it;
                    # every one of them is null on a tool that declares none,
                    # which is what keeps a client's default rendering the
                    # rendering every existing tool already gets.
                    # The field label. null means "prettify the argument name",
                    # which is what every client already did on its own -- so a
                    # tool declaring none renders exactly as before.
                    "label": spec.label,
                    "section": spec.section,
                    "visible_when": spec.visible_when,
                    "ui": spec.ui,
                    # Tuples would serialize as JSON arrays anyway; listed
                    # explicitly so the wire shape does not depend on how a
                    # tool happened to spell its catalog.
                    "groups": (
                        {name: list(options) for name, options in spec.groups.items()}
                        if spec.groups
                        else None
                    ),
                }
                for arg_name, spec in tool.arguments.items()
            },
            "output_kind": tool.output_kind,
        }
        for tool in TOOLS.values()
    ]


@app.get("/tools/{tool_name}/data", dependencies=[Depends(verify_token)])
def list_tool_data(tool_name: str) -> dict:
    """List models and test files available on the server for this tool, so
    a client can pick one instead of uploading its own (see ArgSpec.server_selectable).
    """
    try:
        tool = get_tool(tool_name)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    return {
        "models": data_store.list_models(tool.name),
        "testfiles": data_store.list_testfiles(tool.name),
    }


def _remove_path(path: str) -> None:
    """Remove a file or a whole directory tree — for backend-materialized temp
    copies (ResolvedFile.is_temporary), which can be either."""
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        os.remove(path)


@app.get("/tools/{tool_name}/testfiles/{filename}", dependencies=[Depends(verify_token)])
async def download_testfile(tool_name: str, filename: str, background_tasks: BackgroundTasks):
    """Stream one of the tool's hosted test files to the client, so a user can
    fill an input with reference data instead of hunting for a scan of their
    own. The valid names are what GET /tools/{tool_name}/data lists.

    Only test files are downloadable. Models are deliberately NOT: they are
    selected by name and used in place (see ArgSpec.server_selectable), and
    nothing a client does should ever pull one off the server.

    A test entry that is a FOLDER is zipped on the fly — one response carries
    one blob — and the client unpacks it back into a directory on its side.
    """
    start_time = time.monotonic()
    try:
        tool = get_tool(tool_name)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    try:
        resolved = data_store.resolve_testfile(tool.name, filename)
    except DataNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    if resolved.is_temporary:
        background_tasks.add_task(_remove_path, resolved.path)

    path = resolved.path
    if os.path.isdir(path):
        # DATA_DIR is read-only: the archive is built in its own staging dir
        # under TEMP_DIR, which must outlive the response stream — hence the
        # background task, and the inline cleanup on the one path where no
        # response (and so no background task) will ever run.
        staging_dir = tempfile.mkdtemp(dir=settings.TEMP_DIR)
        archive_name = f"{os.path.basename(filename)}.zip"
        try:
            path = await anyio.to_thread.run_sync(
                file_utils.make_zip, path, os.path.join(staging_dir, archive_name)
            )
        except Exception:
            logger.exception("endpoint=/tools/%s/testfiles status=500 (packing)", tool_name)
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail="Could not package the test folder.")
        background_tasks.add_task(shutil.rmtree, staging_dir, ignore_errors=True)

    # Same log shape as /run: tool, status, duration, size — never the file
    # name (see the confidentiality note at the top of this module).
    size = os.path.getsize(path)
    logger.info(
        "endpoint=/tools/%s/testfiles status=200 duration=%.2fs sent=%dB (%s)",
        tool_name,
        time.monotonic() - start_time,
        size,
        _human_bytes(size),
    )
    return FileResponse(
        path,
        media_type=_media_type_of(path),
        filename=os.path.basename(path),
        background=background_tasks,
    )


# ----------------------------------------------------------------------
# Chunked upload / range-served results (see transfer.py for the why)
# ----------------------------------------------------------------------

class _NewUpload(BaseModel):
    filename: str
    size: int
    chunk_size: Optional[int] = None


def _transfer_error(exc: transfer.TransferError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


@app.post("/uploads", dependencies=[Depends(verify_token)])
async def create_upload(spec: _NewUpload) -> dict:
    """Open a session the client then fills with parallel PUTs.

    Answering with `chunk_size` rather than accepting the client's is what
    keeps the layout single-sourced: part n is always
    `[n * chunk_size, (n+1) * chunk_size)`, and both sides compute it from the
    one number returned here.
    """
    try:
        session = await anyio.to_thread.run_sync(
            functools.partial(
                transfer.create_upload, spec.filename, spec.size, spec.chunk_size
            )
        )
    except transfer.TransferError as exc:
        raise _transfer_error(exc)
    return {
        "upload_id": session.upload_id,
        "chunk_size": session.chunk_size,
        "part_count": session.part_count,
    }


@app.get("/uploads/{upload_id}", dependencies=[Depends(verify_token)])
async def upload_status(upload_id: str) -> dict:
    """What is still missing, this is what makes a transfer resumable: a
    client coming back after a dropped connection sends only these parts."""
    try:
        session = await anyio.to_thread.run_sync(transfer.get_upload, upload_id)
        missing = await anyio.to_thread.run_sync(session.missing_parts)
    except transfer.TransferError as exc:
        raise _transfer_error(exc)
    return {
        "upload_id": session.upload_id,
        "size": session.size,
        "chunk_size": session.chunk_size,
        "part_count": session.part_count,
        "missing_parts": missing,
    }


@app.put("/uploads/{upload_id}/parts/{index}", dependencies=[Depends(verify_token)])
async def upload_part(upload_id: str, index: int, request: Request) -> dict:
    """Receive one part, verify it, write it at its offset.

    The body is the raw bytes, no multipart framing, because there is exactly
    one thing in it and parsing a boundary out of a 8 MB body buys nothing.
    `Content-Encoding: gzip` is honoured for inputs that are not already
    compressed (an uncompressed .nii or a .vtk mesh is 3-4x smaller deflated,
    which on a remote link is 3-4x less time), and `X-Part-SHA256` is checked
    against what lands on disk either way.
    """
    body = await request.body()
    if request.headers.get("Content-Encoding", "").lower() == "gzip":
        try:
            body = await anyio.to_thread.run_sync(gzip.decompress, body)
        except (OSError, EOFError, zlib.error) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Part {index} is not readable gzip: {exc}",
            )
    try:
        session = await anyio.to_thread.run_sync(transfer.get_upload, upload_id)
        remaining = await anyio.to_thread.run_sync(
            functools.partial(
                transfer.write_part,
                session,
                index,
                body,
                request.headers.get("X-Part-SHA256"),
            )
        )
    except transfer.TransferError as exc:
        raise _transfer_error(exc)
    # An upload that is still moving must never be reaped, however long it
    # takes. This is what makes TRANSFER_TTL_SECONDS an idle timeout.
    await anyio.to_thread.run_sync(transfer.touch, session.directory)
    return {"received": index, "missing_count": remaining}


@app.delete("/uploads/{upload_id}", dependencies=[Depends(verify_token)])
async def delete_upload(upload_id: str) -> dict:
    await anyio.to_thread.run_sync(transfer.discard_upload, upload_id)
    return {"status": "ok"}


@app.get("/results/{result_id}", dependencies=[Depends(verify_token)])
async def download_result(result_id: str, request: Request):
    """Serve a stored result, honouring `Range`.

    That header is the whole point: it lets the client pull one file down over
    several connections at once, which on a long-haul link is the difference
    between one congestion window's worth of throughput and several. A client
    that sends no Range still gets the entire file in one response.
    """
    try:
        stored = await anyio.to_thread.run_sync(transfer.get_result, result_id)
    except transfer.TransferError as exc:
        raise _transfer_error(exc)

    try:
        span = transfer.parse_range(request.headers.get("Range"), stored.size)
    except transfer.TransferError as exc:
        # The size is what the client got wrong, so the real one has to travel
        # with the refusal, otherwise it can only guess again.
        return JSONResponse(
            {"detail": str(exc)},
            status_code=exc.status_code,
            headers={"Content-Range": f"bytes */{stored.size}"},
        )

    # Stamped before the body streams, not after: a download in progress is a
    # download that must survive the reaper, and the next range may be minutes
    # away on a slow link.
    await anyio.to_thread.run_sync(transfer.touch, stored.directory)

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f'attachment; filename="{stored.filename}"',
    }
    if span is None:
        start, end, code = 0, stored.size - 1, status.HTTP_200_OK
    else:
        start, end = span
        code = status.HTTP_206_PARTIAL_CONTENT
        headers["Content-Range"] = f"bytes {start}-{end}/{stored.size}"
    headers["Content-Length"] = str(max(0, end - start + 1))

    return StreamingResponse(
        transfer.read_range(stored.blob_path, start, end),
        status_code=code,
        media_type=stored.media_type,
        headers=headers,
    )


@app.delete("/results/{result_id}", dependencies=[Depends(verify_token)])
async def delete_result(result_id: str) -> dict:
    """Sent by a client that has the whole file. Not required for correctness
   , the reaper collects what is never claimed, but it is what keeps TEMP_DIR
    flat under load instead of holding every result for the full TTL."""
    await anyio.to_thread.run_sync(transfer.discard_result, result_id)
    return {"status": "ok"}


def _upload_references(raw) -> dict:
    """{argument name: upload id} from the request's `__uploads__` field."""
    if not raw:
        return {}
    try:
        references = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed '{_UPLOADS_FIELD}' field: {exc}",
        )
    if not isinstance(references, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in references.items()
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{_UPLOADS_FIELD}' must be an object of argument name -> upload id.",
        )
    return references


def _checked_extension(tool, field_name: str, filename: str) -> str:
    """The extension an input will be saved under, or a 400 naming what was
    allowed. Shared by the multipart path and the chunked one so an upload is
    validated identically however its bytes arrived."""
    expected = _expected_extensions(tool, field_name)
    extension = _matched_extension(filename or "", expected)
    if extension is None:
        allowed = expected if expected is not None else settings.ALLOWED_EXTENSIONS
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension for '{field_name}'. Allowed: {allowed}",
        )
    return extension


def _reject_upload_for_scalar(spec, field_name: str) -> None:
    """A scalar-typed argument must never arrive as a file -- e.g. a
    server-side-only model (ArgSpec(type=str, server_selectable="model")) is
    selected by name, never sent by the client. Without this check the
    uploaded file's temp path would be silently passed through as the
    argument's string value."""
    if spec is not None and not spec.is_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Argument '{field_name}' expects a plain value, not an uploaded file.",
        )


async def _as_resolved_path(spec, input_path: str, extension: str, work_dir: str, field_name: str):
    """Tag an input with the declared type it actually is, unpacking a
    "folder" argument's archive so run() only ever sees a directory."""
    kind = spec.match_type(extension) if spec is not None and spec.is_file else "file"
    if kind != FOLDER_TYPE:
        return ResolvedPath(input_path, kind)
    extracted = await anyio.to_thread.run_sync(
        functools.partial(_extract_folder_argument, spec, input_path, work_dir, field_name)
    )
    return ResolvedPath(extracted, FOLDER_TYPE)


@app.post("/run/{tool_name}", dependencies=[Depends(verify_token)])
async def run_tool(tool_name: str, request: Request, background_tasks: BackgroundTasks):
    start_time = time.monotonic()

    try:
        tool = get_tool(tool_name)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    # Generic argument collection: whatever scalar fields and/or files the
    # caller sends, regardless of which tool it targets. Each uploaded file
    # is matched to the tool's argument of the same name (a tool can declare
    # several "file"-typed arguments, e.g. "fixed_image" + "moving_image").
    # The tool's own schema (validated in tool.invoke) decides what is
    # actually accepted.
    form = await request.form()
    args: dict = {}
    uploaded_files: dict = {}
    for key, value in form.multi_items():
        if isinstance(value, StarletteUploadFile):
            uploaded_files[key] = value
        else:
            args[key] = value

    # Inputs that came up through the chunked-upload endpoints reference their
    # session here instead of carrying their bytes in this request. Popped
    # before `args` is looked at, so it can never reach a tool as an argument.
    upload_references = _upload_references(args.pop(_UPLOADS_FIELD, None))

    # An argument declared with ArgSpec(server_selectable=...) can be sent as
    # a plain form value (the file name) instead of an upload -- resolved
    # below into a path already present on the server (see data_store.py).
    # Pulled out of `args` before the upload loop below so a genuinely
    # uploaded file for the same field name is never mistaken for one.
    server_file_args: dict = {}
    for field_name in list(args):
        spec = tool.arguments.get(field_name)
        if spec is not None and spec.server_selectable:
            server_file_args[field_name] = args.pop(field_name)

    work_dir = None
    input_paths = []
    resolved_files = []
    size = 0

    if uploaded_files or upload_references:
        work_dir = tempfile.mkdtemp(dir=settings.TEMP_DIR)

    try:
        for field_name, upload in uploaded_files.items():
            spec = tool.arguments.get(field_name)
            _reject_upload_for_scalar(spec, field_name)
            extension = _checked_extension(tool, field_name, upload.filename or "")
            input_path = os.path.join(work_dir, f"{field_name}{extension}")
            try:
                size += await _stream_to_disk(upload, input_path)
            except _UploadTooLargeError:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds the {settings.MAX_UPLOAD_MB} MB limit.",
                )
            input_paths.append(input_path)
            # An argument can accept several types (e.g. ("csv_file", "folder")):
            # decide here which one this upload actually is, and hand run() a
            # path tagged with it. A "folder" arrives zipped and is unpacked
            # now, so the tool only ever sees a directory.
            args[field_name] = await _as_resolved_path(
                spec, input_path, extension, work_dir, field_name
            )

        # Same treatment, for the inputs whose bytes are already on disk: the
        # session's blob is RENAMED into the work dir rather than copied, so a
        # chunked upload costs no extra pass over the file at all.
        for field_name, upload_id in upload_references.items():
            spec = tool.arguments.get(field_name)
            _reject_upload_for_scalar(spec, field_name)
            try:
                session = await anyio.to_thread.run_sync(transfer.get_upload, upload_id)
                extension = _checked_extension(tool, field_name, session.filename)
                input_path = os.path.join(work_dir, f"{field_name}{extension}")
                await anyio.to_thread.run_sync(transfer.claim_upload, upload_id, input_path)
            except transfer.TransferError as exc:
                raise _transfer_error(exc)
            size += session.size
            input_paths.append(input_path)
            args[field_name] = await _as_resolved_path(
                spec, input_path, extension, work_dir, field_name
            )
    except HTTPException:
        # Nothing has been queued for cleanup yet, and no response will stream,
        # so the work dir has to go now. The sessions that were never claimed
        # go too: the reaper would get them eventually, but "eventually" is a
        # TTL's worth of confidential imaging sitting on disk.
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
        for upload_id in upload_references.values():
            await anyio.to_thread.run_sync(transfer.discard_upload, upload_id)
        raise

    for field_name, filename in server_file_args.items():
        spec = tool.arguments[field_name]
        resolver = data_store.resolve_model if spec.server_selectable == "model" else data_store.resolve_testfile
        try:
            resolved = resolver(tool.name, filename)
        except DataNotFoundError as exc:
            if work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
        resolved_files.append(resolved)

        # Server-side data can be a real folder or a single file; tag it the
        # same way as an upload so run() branches on .kind either way. An
        # archive standing in for a "folder" argument is unpacked here too, so
        # the two routes stay indistinguishable from the tool's point of view.
        kind = _resolved_kind(spec, resolved.path)
        path = resolved.path
        if kind == FOLDER_TYPE and not os.path.isdir(path):
            # DATA_DIR is read-only: extract into the request's own work dir.
            if work_dir is None:
                work_dir = tempfile.mkdtemp(dir=settings.TEMP_DIR)
            try:
                path = await anyio.to_thread.run_sync(
                    functools.partial(_extract_folder_argument, spec, path, work_dir, field_name)
                )
            except HTTPException:
                shutil.rmtree(work_dir, ignore_errors=True)
                raise
        args[field_name] = ResolvedPath(path, kind)

    # Anything the tool creates through file_utils.make_scratch_dir() lands
    # here, so it can be removed even if run() raises before returning a path.
    scratch_dirs = file_utils.track_scratch_dirs()

    try:
        # Run the tool in a worker thread, NOT on the event loop: tool.invoke
        # is synchronous CPU-bound work (model loading, inference) and would
        # otherwise freeze the whole server -- even /health -- for its entire
        # duration. Offloaded like this, requests are served fully in
        # parallel, bounded by MAX_CONCURRENT_TOOLS. Tools are safe to run
        # concurrently: they are stateless (everything arrives via args),
        # each request gets its own work_dir, and DATA_DIR is read-only.
        result = await anyio.to_thread.run_sync(
            tool.invoke, args, limiter=_get_tool_limiter()
        )
    except ToolArgumentError as exc:
        _discard(work_dir, scratch_dirs)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))
    except ToolUnavailableError as exc:
        # The request is fine; this deployment cannot serve it (a dependency
        # the image does not carry). 501, not 500: nothing the caller changes
        # will help, and unlike a crash the reason is exactly what they need to
        # read -- it names a missing package, never a path or patient data.
        logger.warning("endpoint=/run/%s status=501", tool_name)
        _discard(work_dir, scratch_dirs)
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc))
    except Exception:
        logger.exception("endpoint=/run/%s status=500", tool_name)
        _discard(work_dir, scratch_dirs)
        raise HTTPException(status_code=500, detail="Tool execution failed.")
    finally:
        # Uploaded inputs are never needed again past this point.
        for input_path in input_paths:
            if os.path.exists(input_path):
                os.remove(input_path)
        # Server-side data (DATA_DIR) is persistent and must never be
        # deleted; only backend-materialized temp copies are (see
        # ResolvedFile.is_temporary in data_store.py).
        for resolved in resolved_files:
            if resolved.is_temporary and os.path.exists(resolved.path):
                os.remove(resolved.path)

    if tool.output_kind in ("file", "segmentation", "files"):
        # `result` is a path to the output file the tool wrote -- or, for
        # "files", a list of paths / a single directory to bundle into a zip.
        if work_dir is None:
            work_dir = tempfile.mkdtemp(dir=settings.TEMP_DIR)

        # A tool whose inputs all came from the read-only data store writes its
        # output in its own scratch dir under TEMP_DIR instead of the upload
        # work dir: those folders have to be cleaned up as well, whether the
        # response goes out or the packing below fails. `scratch_dirs` covers
        # everything requested through file_utils.make_scratch_dir();
        # _output_roots also catches a tool that wrote under TEMP_DIR by hand.
        try:
            outputs = _output_paths(result)
            if not outputs:
                raise ValueError(
                    f"Tool '{tool.name}' declares output_kind={tool.output_kind!r} but "
                    f"run() returned {type(result).__name__}, not a path (or a list of paths)."
                )
            output_roots = _output_roots(outputs, work_dir) | set(scratch_dirs)
            if tool.output_kind == "files":
                # Built inside work_dir, never inside the tool's own scratch
                # dir: the archive has to outlive the files it was made from,
                # right up until the response has finished streaming.
                archive_name = (
                    f"{os.path.basename(outputs[0].rstrip(os.sep))}.zip"
                    if len(outputs) == 1 and os.path.isdir(outputs[0])
                    else f"{tool.name}_output.zip"
                )
                result = await anyio.to_thread.run_sync(
                    file_utils.make_zip, outputs, os.path.join(work_dir, archive_name)
                )
        except Exception:
            logger.exception("endpoint=/run/%s status=500 (packing output)", tool_name)
            _discard(work_dir, _output_roots(_output_paths(result), work_dir) | set(scratch_dirs))
            raise HTTPException(status_code=500, detail="Tool execution failed.")

        media_type = _media_type_of(str(result))

        # A client asking for reference delivery gets the result MOVED out of
        # the work dir (a rename, not a copy) and a JSON pointer to it, so it
        # can pull the bytes down over several range requests at once instead
        # of through this one connection. Done before the cleanup tasks are
        # queued, since those are what would otherwise take the file with them.
        #
        # Only above RESULT_REFERENCE_MIN_MB, and the reason is cleanup rather
        # than speed. A streamed response deletes its file server-side the
        # moment the response ends, with no dependency at all on the client
        # coming back for it; a reference depends on a DELETE or on the reaper.
        # Parallel ranges buy nothing on a small result, so there is no reason
        # to trade the stronger guarantee away for one.
        deliver_by_reference = (
            request.headers.get(_RESULT_DELIVERY_HEADER, "").lower() == _DELIVER_BY_REFERENCE
            and os.path.getsize(result) >= _RESULT_REFERENCE_MIN_BYTES
        )
        stored = None
        if deliver_by_reference:
            try:
                stored = await anyio.to_thread.run_sync(
                    transfer.store_result, str(result), media_type
                )
            except OSError:
                # Reference delivery is an optimisation; failing it must not
                # fail a run that has already done the expensive part. Falls
                # through to streaming the file the way it always did.
                logger.exception("endpoint=/run/%s (storing result by reference)", tool_name)

        background_tasks.add_task(shutil.rmtree, work_dir, ignore_errors=True)
        for output_root in output_roots:
            background_tasks.add_task(shutil.rmtree, output_root, ignore_errors=True)

        if stored is not None:
            _log_served(tool_name, start_time, size, stored.size)
            return JSONResponse({"result_ref": stored.as_reference()}, background=background_tasks)

        # The size of the file about to be streamed. Measured rather than
        # accumulated: for output_kind="files" what goes out is the archive
        # built just above, not the sum of what run() produced.
        _log_served(tool_name, start_time, size, os.path.getsize(result))
        return FileResponse(
            result,
            media_type=media_type,
            filename=os.path.basename(result),
            background=background_tasks,
        )

    # A "text" tool can still have written scratch files along the way.
    for directory in ([work_dir] if work_dir else []) + list(scratch_dirs):
        background_tasks.add_task(shutil.rmtree, directory, ignore_errors=True)
    _log_served(tool_name, start_time, size, None)
    return {"result": result}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
