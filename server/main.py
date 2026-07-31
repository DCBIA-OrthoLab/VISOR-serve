# This server processes confidential medical imaging data.
# It must be deployed in an appropriate jurisdiction (EU / a certified health
# host, depending on context) and only ever reached over TLS (see README.md).
# De-identification of patient data happens on the client side before upload;
# this server never logs file contents, argument values, or patient metadata.

import functools
import logging
import mimetypes
import os
import shutil
import tempfile
import time
from typing import Optional

import anyio.to_thread
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

import file_utils
from base import FILE_TYPES, FOLDER_TYPE, ResolvedPath, ToolArgumentError
from config import settings
from data_store import DataNotFoundError, data_store
from registry import TOOLS, get_tool
from security import verify_token

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("inference_server")

os.makedirs(settings.TEMP_DIR, exist_ok=True)

app = FastAPI()

_CHUNK_SIZE_BYTES = 1024 * 1024  # read/write in 1 MB chunks, never load the full file into RAM
_MAX_UPLOAD_BYTES = settings.MAX_UPLOAD_MB * 1024 * 1024
_MAX_EXTRACTED_BYTES = settings.MAX_EXTRACTED_MB * 1024 * 1024


class _UploadTooLargeError(Exception):
    pass


_ACCEPT_ALL_EXTENSIONS = "*"

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

    if uploaded_files:
        work_dir = tempfile.mkdtemp(dir=settings.TEMP_DIR)
        for field_name, upload in uploaded_files.items():
            # A scalar-typed argument must never arrive as an upload -- e.g. a
            # server-side-only model (ArgSpec(type=str, server_selectable="model"))
            # is selected by name, never sent by the client. Without this check
            # the uploaded file's temp path would be silently passed through as
            # the argument's string value.
            spec = tool.arguments.get(field_name)
            if spec is not None and not spec.is_file:
                shutil.rmtree(work_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Argument '{field_name}' expects a plain value, not an uploaded file.",
                )
            expected = _expected_extensions(tool, field_name)
            extension = _matched_extension(upload.filename or "", expected)
            if extension is None:
                shutil.rmtree(work_dir, ignore_errors=True)
                allowed = expected if expected is not None else settings.ALLOWED_EXTENSIONS
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file extension for '{field_name}'. Allowed: {allowed}",
                )
            input_path = os.path.join(work_dir, f"{field_name}{extension}")
            try:
                size += await _stream_to_disk(upload, input_path)
            except _UploadTooLargeError:
                shutil.rmtree(work_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds the {settings.MAX_UPLOAD_MB} MB limit.",
                )
            input_paths.append(input_path)

            # An argument can accept several types (e.g. ("csv_file", "folder")):
            # decide here which one this upload actually is, and hand run() a
            # path tagged with it. A "folder" arrives zipped and is unpacked
            # now, so the tool only ever sees a directory.
            kind = spec.match_type(extension) if spec is not None and spec.is_file else "file"
            if kind == FOLDER_TYPE:
                try:
                    extracted = await anyio.to_thread.run_sync(
                        functools.partial(
                            _extract_folder_argument, spec, input_path, work_dir, field_name
                        )
                    )
                except HTTPException:
                    shutil.rmtree(work_dir, ignore_errors=True)
                    raise
                args[field_name] = ResolvedPath(extracted, FOLDER_TYPE)
            else:
                args[field_name] = ResolvedPath(input_path, kind)

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

        background_tasks.add_task(shutil.rmtree, work_dir, ignore_errors=True)
        for output_root in output_roots:
            background_tasks.add_task(shutil.rmtree, output_root, ignore_errors=True)

        media_type, _ = mimetypes.guess_type(str(result))
        if media_type is None:
            media_type = "application/gzip" if str(result).endswith(".gz") else "application/octet-stream"
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
