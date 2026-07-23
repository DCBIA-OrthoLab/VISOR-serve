# This server processes confidential medical imaging data.
# It must be deployed in an appropriate jurisdiction (EU / a certified health
# host, depending on context) and only ever reached over TLS (see README.md).
# De-identification of patient data happens on the client side before upload;
# this server never logs file contents, argument values, or patient metadata.

import logging
import os
import shutil
import tempfile
import time
from typing import Optional

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from base import ToolArgumentError
from config import settings
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


def _matched_extension(filename: str) -> Optional[str]:
    lower = filename.lower()
    for extension in settings.ALLOWED_EXTENSIONS:
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
                }
                for arg_name, spec in tool.arguments.items()
            },
            "output_kind": tool.output_kind,
        }
        for tool in TOOLS.values()
    ]


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

    work_dir = None
    input_paths = []
    size = 0

    if uploaded_files:
        work_dir = tempfile.mkdtemp(dir=settings.TEMP_DIR)
        for field_name, upload in uploaded_files.items():
            extension = _matched_extension(upload.filename or "")
            if extension is None:
                shutil.rmtree(work_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file extension for '{field_name}'. Allowed: {settings.ALLOWED_EXTENSIONS}",
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

    duration = time.monotonic() - start_time
    logger.info(
        "endpoint=/run/%s status=200 duration=%.2fs size=%dB", tool_name, duration, size
    )

    if tool.output_kind in ("file", "segmentation"):
        # `result` is expected to be a path to the output file written by the tool.
        if work_dir is None:
            work_dir = tempfile.mkdtemp(dir=settings.TEMP_DIR)
        background_tasks.add_task(shutil.rmtree, work_dir, ignore_errors=True)
        return FileResponse(
            result,
            media_type="application/gzip" if str(result).endswith(".gz") else "application/octet-stream",
            filename=os.path.basename(result),
            background=background_tasks,
        )

    if work_dir:
        background_tasks.add_task(shutil.rmtree, work_dir, ignore_errors=True)
    return {"result": result}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
