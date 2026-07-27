# This server processes confidential medical imaging data.
# It must be deployed in an appropriate jurisdiction (EU / a certified health
# host, depending on context) and only ever reached over TLS (see README.md).
# De-identification of patient data happens on the client side before upload;
# this server never logs file contents, argument values, or patient metadata.

import logging
import mimetypes
import os
import shutil
import tempfile
import time
from typing import Optional

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from base import FILE_TYPES, ToolArgumentError
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


class _UploadTooLargeError(Exception):
    pass


_ACCEPT_ALL_EXTENSIONS = "*"


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
    """Return the specific extensions expected for this argument, if the tool
    declares a specific file type for it (see base.FILE_TYPES). None means
    "no specific type declared" -- fall back to settings.ALLOWED_EXTENSIONS.
    """
    spec = tool.arguments.get(field_name)
    if spec is None:
        return None
    return FILE_TYPES.get(spec.type)


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


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/tools")
def list_tools() -> list:
    """Let clients discover every registered tool and its expected arguments."""
    return [
        {
            "name": tool.name,
            "arguments": {
                arg_name: {
                    "type": _type_name(spec.type),
                    "required": spec.required,
                    "description": spec.description,
                    "server_selectable": spec.server_selectable,
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
            args[field_name] = input_path
            input_paths.append(input_path)

    for field_name, filename in server_file_args.items():
        spec = tool.arguments[field_name]
        resolver = data_store.resolve_model if spec.server_selectable == "model" else data_store.resolve_testfile
        try:
            resolved = resolver(tool.name, filename)
        except DataNotFoundError as exc:
            if work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
        args[field_name] = resolved.path
        resolved_files.append(resolved)

    try:
        result = tool.invoke(args)
    except ToolArgumentError as exc:
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))
    except Exception:
        logger.exception("endpoint=/run/%s status=500", tool_name)
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
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

    duration = time.monotonic() - start_time
    logger.info(
        "endpoint=/run/%s status=200 duration=%.2fs size=%dB", tool_name, duration, size
    )

    if tool.output_kind in ("file", "segmentation"):
        # `result` is expected to be a path to the output file written by the tool.
        if work_dir is None:
            work_dir = tempfile.mkdtemp(dir=settings.TEMP_DIR)
        background_tasks.add_task(shutil.rmtree, work_dir, ignore_errors=True)
        media_type, _ = mimetypes.guess_type(str(result))
        if media_type is None:
            media_type = "application/gzip" if str(result).endswith(".gz") else "application/octet-stream"
        return FileResponse(
            result,
            media_type=media_type,
            filename=os.path.basename(result),
            background=background_tasks,
        )

    if work_dir:
        background_tasks.add_task(shutil.rmtree, work_dir, ignore_errors=True)
    return {"result": result}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
