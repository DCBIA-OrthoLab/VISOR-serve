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
import re
import shutil
import subprocess
import tempfile
import time
import zlib
from typing import Optional

import anyio.to_thread
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, UploadFile, status
from fastapi.responses import (
    RedirectResponse,
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel
from starlette.datastructures import UploadFile as StarletteUploadFile

# `runner` is executed BY a tool's interpreter and must never import this
# server -- but the other direction is safe and is what lets the rewind
# edit the same record the runner wrote. It is standard library only.
from execution import admission, costs, dispatch, reports, runner
from registry import facade
from registry.facade import FacadeTool
import file_utils
import redact
import resources
import benchmark_jobs
import benchmark_presets
import telemetry
from wire import (benchmark_launch_page, benchmark_page, debug_page, doc_page,
                  maintenance, runs, status_page, transfer)
from base import (
    FILE_TYPES,
    FOLDER_TYPE,
    PATH_TYPE,
    ResolvedPath,
    ToolArgumentError,
    ToolUnavailableError,
)
from config import settings
from data_store import DataNotFoundError, data_store
from registry.deployment import deployment_config
from registry import TOOLS, get_tool
from wire import clients, update_check, updates
from wire.security import verify_admin, verify_token

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("inference_server")

os.makedirs(settings.TEMP_DIR, exist_ok=True)


async def _reaper_loop() -> None:
    """Sweep expired transfer and run directories for as long as the server runs.

    A timer, not only the opportunistic sweep transfer.py does when a session
    is created: an abandoned upload sits longest exactly when no new request
    comes in to trigger that sweep.

    Runs ride the same loop rather than a second one. Their normal cleanup is
    the request that owns them, so this is the safety net for a client that
    vanished mid-POST -- and it is needed because a progress message is written
    by a tool and can name a file.
    """
    while True:
        await anyio.sleep(settings.TRANSFER_SWEEP_SECONDS)
        for sweep in (transfer.reap_expired, runs.reap_expired):
            try:
                await anyio.to_thread.run_sync(sweep)
            except Exception:  # noqa: BLE001 - one bad sweep must not end the loop
                logger.exception("reaper sweep failed")


async def _trace_loop() -> None:
    """Sample the machine for the operator page's graphs, for as long as the
    server runs. One point every `telemetry.TRACE_SECONDS`, off the event loop
    because reading the card is a subprocess."""
    def take() -> None:
        try:
            card = resources.detect_vram_bytes()
        except Exception:  # noqa: BLE001 - no card is a missing line, not an error
            card = (None, None)
        telemetry.sample_trace(admission.budget().snapshot(), card)

    while True:
        try:
            await anyio.to_thread.run_sync(take)
        except Exception:  # noqa: BLE001 - one bad sample must not end the loop
            logger.exception("trace sample failed")
        await anyio.sleep(telemetry.TRACE_SECONDS)


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Said out loud, once, before anything runs. A budget that did not take
    # effect -- a variable set in the wrong file, a container limit nobody
    # applied -- has no symptom other than a server that feels slow, so the
    # numbers it decided on have to be readable in the log beside the ones it
    # found. Resolved here rather than at import so the detection it does is
    # part of starting the server, not of importing it.
    for line in resources.banner(resources.allocation()).splitlines():
        logger.info("%s", line)
    # Said beside the budget, because the budget is only as good as these: a
    # tool nobody has measured reserves everything, and a tool whose memory
    # moves with the request is reserving the worst run anyone has seen rather
    # than what this one will cost.
    for line in costs.banner().splitlines():
        logger.info("%s", line)
    clients.configure(settings.HISTORY_DIR)
    restored = telemetry.configure_history(settings.HISTORY_DIR)
    if restored:
        logger.info("Run history: %d finished run(s) read back from %s",
                    restored, settings.HISTORY_DIR)
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(_reaper_loop)
        task_group.start_soon(_trace_loop)
        try:
            yield
        finally:
            # The loop never returns on its own: without this cancel the task
            # group would wait for it forever on shutdown.
            task_group.cancel_scope.cancel()


app = FastAPI(lifespan=_lifespan)


# Paths that WATCH this server rather than ask it for work. They are not
# counted as in-flight requests: the dashboard polls every two seconds, so
# counting its own poll made the chip read "1 request" on a completely idle
# machine -- the observer appearing in its own observation, and a claim the
# page could not support. Health checks are excluded for the same reason.
_UNCOUNTED_PATHS = ("/admin-panel", "/server-debug", "/status", "/health", "/runs/")


def _is_observer(path: str) -> bool:
    return any(path == prefix.rstrip("/") or path.startswith(prefix)
               for prefix in _UNCOUNTED_PATHS)


@app.middleware("http")
async def _count_inflight(request: Request, call_next):
    """How many requests are being served at this instant, for `/admin-panel`.

    A counter and not a log: what the page needs is the CONCURRENT figure, and
    that is knowable only from inside the request's own lifetime. The
    `try/finally` is what makes it safe -- a handler that raises, a client that
    disconnects mid-body, and a cancelled `POST /run` all have to decrement, or
    the number only ever climbs.

    What it counts is work ASKED OF this server: a run, an upload, a result
    being fetched. A page watching the server is not that, and `_is_observer`
    is what keeps the two apart.

    It is deliberately the first middleware and does nothing else: every
    request in this process passes through it, so anything expensive here is
    expensive everywhere.
    """
    if _is_observer(request.url.path):
        return await call_next(request)
    telemetry.request_started()
    try:
        return await call_next(request)
    finally:
        telemetry.request_finished()


_CHUNK_SIZE_BYTES = 1024 * 1024  # read/write in 1 MB chunks, never load the full file into RAM
_MAX_EXTRACTED_BYTES = settings.MAX_EXTRACTED_MB * 1024 * 1024
_RESULT_REFERENCE_MIN_BYTES = settings.RESULT_REFERENCE_MIN_MB * 1024 * 1024


class _UploadTooLargeError(Exception):
    pass


_ACCEPT_ALL_EXTENSIONS = "*"

# A packaged tool's exception class NAME -> the status it means. The tools do
# not share a base class, so this maps by name, which is the convention
# SADT-VISOR documents:
#
#   the caller's fault, and every message is written to be read by whoever
#   sent it -- a bad structure code, a table with no patient column;
#   ToolUnavailableError -- the tool is installed, its engine is not (crownseg
#   without its `segmentation` extra), which no request can fix;
#   anything else is opaque and answers 500 with a fixed message, because a
#   crash inside a tool can name server-side paths.
def _tool_error_status(kinds) -> int:
    """The status for a tool's exception, from its class name or, failing
    that, the nearest base class this table names -- a tool's own subclass of
    `ToolInputError` is the caller's fault the way its base is. 500 otherwise."""
    for kind in kinds or ():
        if kind in TOOL_ERROR_STATUS:
            return TOOL_ERROR_STATUS[kind]
    return status.HTTP_500_INTERNAL_SERVER_ERROR


TOOL_ERROR_STATUS = {
    "ToolInputError": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "ValueError": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "FileNotFoundError": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "ToolUnavailableError": status.HTTP_503_SERVICE_UNAVAILABLE,
}

# Form field carrying {argument name: upload id} for inputs that travelled
# through the chunked-upload endpoints instead of this request's body (see
# transfer.py). Double-underscored so it can never collide with a tool's own
# argument name, and popped before anything looks at `args`.
_UPLOADS_FIELD = "__uploads__"

# Sent by a client that would rather be handed a reference to the result and
# fetch the bytes over parallel range requests. A client that does not send it
# gets exactly the response it always got.
_RESULT_DELIVERY_HEADER = "X-Result-Delivery"
_DELIVER_BY_REFERENCE = "reference"

# Opts one run out of the blocking contract: the POST answers 202 as soon as the
# inputs are staged, and the run reports through the event stream it already
# has. Modelled on the header above, and opt-in for the same reason -- a client
# that does not send it reaches byte-for-byte the behaviour it always had.
#
# What this fixes is not hypothetical. The Slicer client's POST read timeout is
# 600 s and its ceiling is an hour, while the server is sized for cohorts that
# legitimately take longer; and a disconnect never stopped a run, it only threw
# away the answer, because nothing in Starlette cancels a worker thread.
_RUN_DELIVERY_HEADER = "X-Run-Delivery"
_RUN_DETACHED = "detached"

# The run id, minted by the client with secrets.token_urlsafe(24) and sent on
# the run it identifies. Optional in both directions: a client that sends none
# gets exactly the behaviour it always got, and one that sends it to an older
# server simply finds no /runs endpoints.
_RUN_ID_HEADER = "X-Run-Id"
# A run that is one batch of a divided cohort says which, so the server can
# group them and decide whether they run side by side (wire/clients.py).
_BATCH_ID_HEADER = "X-Batch-Id"
_BATCH_INDEX_HEADER = "X-Batch-Index"
_BATCH_TOTAL_HEADER = "X-Batch-Total"

# nginx's, and non-standard on purpose: no standard code means "the caller
# withdrew this". The client has to tell a cancellation from a failure without
# reading a message -- one closes the panel quietly, the other opens an error
# dialog.
CLIENT_CLOSED_REQUEST = 499

# Caps how many tool executions run at once (settings.MAX_CONCURRENT_TOOLS).
# Dedicated to tool runs, so waiting inference jobs never starve the threadpool
# used for everything else. Created lazily: anyio needs a running event loop to
# instantiate a CapacityLimiter.
_tool_limiter: Optional[anyio.CapacityLimiter] = None


def _get_tool_limiter() -> anyio.CapacityLimiter:
    global _tool_limiter
    if _tool_limiter is None:
        _tool_limiter = anyio.CapacityLimiter(settings.MAX_CONCURRENT_TOOLS)
    return _tool_limiter




async def _run_in_slot(call):
    """`call` in a worker thread, inside a tool slot (see `_tool_slot`)."""
    async with _tool_slot(runs.CURRENT_RUN.get()):
        return await anyio.to_thread.run_sync(call)


# How often a run waiting for a slot checks whether an operator has given it
# priority. Half a second is invisible next to a wait for a slot, which is a
# wait for a whole other run to finish.
_PRIORITY_POLL_SECONDS = 0.5


@contextlib.asynccontextmanager
async def _tool_slot(run_id: Optional[str]):
    """One of the MAX_CONCURRENT_TOOLS slots, or none for a run given priority.

    The slot is the FIRST queue a run meets, before admission's, and it is
    anyio's own FIFO: a run an operator marks HIGH while it waits here would
    otherwise sit behind every run that arrived first, which is the opposite
    of what the mark means. So the wait is raced against the mark, and a
    marked run goes through without a slot. Admission still decides what it
    may hold -- the slot bounds worker threads, the budget bounds the machine.
    """
    limiter = _get_tool_limiter()
    borrower = object()
    acquired = False
    budget = admission.budget()
    # A batch of a cohort first waits its turn among its siblings, when its
    # workstation's rule is serial (wire/clients.py). Before the slot, so a
    # batch waiting on a sibling holds no slot another workstation could use.
    # Priority goes past this gate too.
    info = runs.meta(run_id) if run_id else {}
    batch, address = info.get("batch"), info.get("client")
    if batch:
        clients.wait(address, batch, run_id)
        while not clients.may_start(address, batch, run_id):
            if budget.priority_of(run_id) == admission.PRIORITY_HIGH:
                clients.leave(address, batch, run_id)
                break
            if runs.is_cancelled(run_id):
                clients.leave(address, batch, run_id)
                raise HTTPException(status_code=CLIENT_CLOSED_REQUEST, detail="The client cancelled this run.")
            await anyio.sleep(_PRIORITY_POLL_SECONDS)
    if run_id is None or budget.priority_of(run_id) != admission.PRIORITY_HIGH:
        async with anyio.create_task_group() as group:
            async def take() -> None:
                nonlocal acquired
                await limiter.acquire_on_behalf_of(borrower)
                acquired = True
                group.cancel_scope.cancel()

            async def watch() -> None:
                while budget.priority_of(run_id) != admission.PRIORITY_HIGH:
                    await anyio.sleep(_PRIORITY_POLL_SECONDS)
                group.cancel_scope.cancel()

            group.start_soon(take)
            if run_id is not None:
                group.start_soon(watch)
    try:
        yield
    finally:
        if acquired:
            limiter.release_on_behalf_of(borrower)
        if batch:
            clients.leave(address, batch, run_id)


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
    # A packaged tool's "path" takes whatever the tool reads -- a .vtk mesh, a
    # .csv of measurements, a .zip of a whole cohort, which the server unpacks.
    # Its schema cannot say more than "a path", and falling back to
    # ALLOWED_EXTENSIONS here would leave every packaged tool accepting .nii
    # only: Surg_Mov_Pred could not be sent its own .csv.
    if spec.accepts is None and PATH_TYPE in spec.types:
        return (_ACCEPT_ALL_EXTENSIONS,)
    declared = spec.extensions
    if declared and PATH_TYPE in spec.types and ".zip" not in declared:
        # A path argument may be given a FOLDER, and a folder reaches the
        # server as a .zip it unpacks -- `_is_folder_upload` a few lines down
        # says exactly that. What a tool declares is what it READS, not how a
        # directory travels: a cohort of `.vtk` sent as one archive was refused
        # for not being a `.vtk`, which is the transport answering for the
        # content. Adding ".zip" to every tool's declaration would be restating
        # the same transport in each of them.
        return tuple(declared) + (".zip",)
    return declared


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


async def _stream_to_disk(upload: UploadFile, destination: str, max_bytes: int) -> int:
    """Write the upload to disk in chunks, never buffering the whole file in RAM."""
    size = 0
    with open(destination, "wb") as out_file:
        while chunk := await upload.read(_CHUNK_SIZE_BYTES):
            size += len(chunk)
            if size > max_bytes:
                raise _UploadTooLargeError()
            out_file.write(chunk)
    return size


def _upload_limit_mb(tool) -> int:
    """The upload limit for this tool: deployment.toml's `max_upload_mb` when
    it declares one, MAX_UPLOAD_MB otherwise.

    Applied here rather than in POST /uploads because that endpoint opens a
    session for a file, not for a tool, and does not know which tool the bytes
    are for. The chunked path is therefore bounded by the global limit while
    the transfer runs, and by the tool's own the moment it is claimed below.
    """
    return deployment_config.upload_limit_mb(tool.name)


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


def _discard(work_dir: Optional[str], scratch_dirs: list) -> None:
    """Remove everything this request created, right now. Used on the error
    paths, where no response will ever stream and background tasks won't run."""
    for directory in ([work_dir] if work_dir else []) + list(scratch_dirs):
        shutil.rmtree(directory, ignore_errors=True)


def _media_type_of(path: str) -> str:
    """Content-Type for a file about to be streamed. Derived from the real
    extension so an .xlsx is never mislabeled as a generic zip; the
    gzip/octet-stream fallback covers bare .gz files (e.g. .nii.gz), which
    mimetypes cannot name."""
    media_type, _ = mimetypes.guess_type(str(path))
    if media_type is None:
        media_type = "application/gzip" if str(path).endswith(".gz") else "application/octet-stream"
    return media_type


def _gib(count) -> str:
    """A byte count in GiB, or "unknown" when there is none.

    `Allocation.ram_bytes` and `.vram_bytes` are Optional and vram IS None on a
    machine with no card -- CI, a CPU deployment. Dividing it took three tests
    of the debug page down at once. Same lesson as `_human_bytes` below, which
    records the first time a None reached a formatter.
    """
    return "unknown" if count is None else f"{count / 1073741824:.0f} GiB"


def _human_bytes(size) -> str:
    """Byte count in the largest unit that keeps it readable. Logged alongside
    the exact figure, never instead of it.

    An absent count reads as absent rather than raising: this is a log line,
    and a log line must never be what takes a finished run down. It already
    did once -- a resume carries no input bytes and passed None through.
    """
    if size is None:
        return "-"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024


def _log_served(tool_name: str, start_time: float, received: int, sent: Optional[int]) -> None:
    """One line per successfully served request.

    Called at each return point rather than before them, so `duration` covers
    packing the response too -- zipping a multi-GB segmentation is not free.
    `sent` is None for a "text" tool, whose result travels as JSON. Nothing
    here may name a file, an argument value, or patient metadata (see the note
    at the top of this module).
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





def _writable(directory: str) -> bool:
    """Whether this process could write a file in `directory`, creating it if need be."""
    if os.path.isdir(directory):
        return os.access(directory, os.W_OK | os.X_OK)
    parent = os.path.dirname(os.path.abspath(directory))
    return os.path.isdir(parent) and os.access(parent, os.W_OK | os.X_OK)


def _battery_dir() -> str:
    """Where a battery launched from the page writes its summary (see config)."""
    if settings.SADT_BATTERY_DIR:
        return settings.SADT_BATTERY_DIR
    if _writable(settings.SADT_BENCHMARK_DIR):
        return settings.SADT_BENCHMARK_DIR
    return os.path.join(settings.SCHEMA_CACHE_DIR, "batteries")


def _campaign_files() -> dict:
    """{summary name: path}, over the campaigns and the batteries alike.

    Two directories, because a deployment may write the one and only read the
    other; the first to hold a name keeps it.
    """
    files = {}
    seen = set()
    for directory in (settings.SADT_BENCHMARK_DIR, _battery_dir()):
        real = os.path.realpath(directory)
        if real in seen:
            continue
        seen.add(real)
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            continue
        for name in names:
            if _CAMPAIGN_NAME.fullmatch(name) and name not in files:
                files[name] = os.path.join(directory, name)
    return files


def _campaign_index() -> list:
    """One entry per campaign or battery summary, newest first.

    A missing directory is not an error and neither is an empty one: a
    production deployment runs no campaign, and "nothing measured here" is the
    honest answer rather than a 500. An unreadable summary costs that summary
    alone -- a campaign still being written must not take the endpoint down for
    the ones already finished.
    """
    index = []
    for name, path in _campaign_files().items():
        try:
            size = os.path.getsize(path)
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(report, dict):
            continue
        index.append({
            "source": name,
            "generated_at": report.get("generated_at"),
            "arms": len(report.get("arms") or ()),
            "bytes": size,
        })
    index.sort(key=lambda entry: entry["generated_at"] or 0, reverse=True)
    return index



# A campaign is addressed BY NAME from a URL. One path segment, starting with
# an alphanumeric and ending in the suffix the report writer uses: no
# separator and no leading dot, which makes a traversal unrepresentable rather
# than merely detected. `wire/transfer.py`'s ID_RE is the local precedent.
#
# Matched with `fullmatch`, not `match`: Python's `$` also matches before a
# trailing newline, and a file name on Linux may contain one -- so an anchored
# `match` would accept `b6-....json\n` as if it were the name beside it.
_CAMPAIGN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\.json")









@app.get("/benchmarks", dependencies=[Depends(verify_admin)])
def benchmark_report(campaign: Optional[str] = None) -> dict:
    """The campaign a reader asked for, and the list of the others.

    Behind the ADMIN token, like every benchmark route: a campaign is the
    operator's and the developer's reading of how far this machine can be
    pushed, and a workstation's API token has no business with it.

    Named `benchmark_report`, not `benchmarks`: see the note below on what a
    handler shadowing a module-level name costs.
    """
    index = _campaign_index()
    if campaign is None:
        chosen = index[0]["source"] if index else None
    else:
        # NOT a fallback to the newest. A reader who asked for yesterday and
        # silently got today would compare two campaigns believing they were
        # one; the page is written to expect this 404 and to say so once.
        chosen = None
        if _CAMPAIGN_NAME.fullmatch(campaign):
            chosen = next(
                (entry["source"] for entry in index if entry["source"] == campaign),
                None,
            )
        if chosen is None:
            raise HTTPException(status_code=404, detail="No such campaign.")
    if chosen is None:
        return {"campaign": None, "campaigns": []}
    path = _campaign_files().get(chosen)
    try:
        with open(path or "", encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, ValueError):
        # Gone or half-written between the listing and the read.
        raise HTTPException(status_code=404, detail="No such campaign.")
    # The file does not carry its own name; the picker keys every option on it.
    report["source"] = chosen
    return {"campaign": report, "campaigns": index}



# NOT named `status`, `benchmarks` or anything else already bound at module
# level: a handler shadowing an imported name breaks it for every line below
# it -- a function called `status` here once took out every `status.HTTP_*` in
# this module at import time.

@app.get("/benchmarks/view", include_in_schema=False)
def benchmark_view() -> HTMLResponse:
    """The campaign, drawn. Unauthenticated: the page holds no measurement, it
    asks for one with the admin token the reader types into it."""
    return HTMLResponse(benchmark_page.BENCHMARK_PAGE)

# Per-tool documentation. Unauthenticated like the other two pages: it
# describes what a tool DOES and what it costs on this deployment, which is
# what `GET /tools` already publishes in machine-readable form. A tool nobody
# has written up answers 404 rather than an empty page -- a blank document
# reads as "there is nothing to say" when the truth is "nobody wrote it".




@app.get("/doc/{tool}", include_in_schema=False)
def tool_document(tool: str) -> HTMLResponse:
    body = doc_page.page_for(tool)
    if not body:
        raise HTTPException(status_code=404, detail="No document for that tool.")
    return HTMLResponse(body)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _extensions_of(spec) -> Optional[dict]:
    """{type name: [extension, ...]} for every file type an argument accepts.

    Published so a client never has to mirror FILE_TYPES. Keyed by type rather
    than flattened because the caller needs the split: the extensions of
    "folder" are what a zipped folder may be uploaded as, not what its file
    picker should offer.
    """
    per_type = {}
    for declared in spec.types:
        name = _type_name(declared)
        if name in FILE_TYPES:
            # ArgSpec.accepts overrides the declared type's own list: it is how
            # an argument declared as a generic file (a .schema.json can only
            # say "path") still tells the client what it reads.
            extensions = FILE_TYPES[name] if spec.accepts is None else spec.accepts
            per_type[name] = list(extensions) if extensions else None
    return per_type or None


@app.get("/status", dependencies=[Depends(verify_token)])
def server_status() -> dict:
    # NOT named `status`: `fastapi.status` is imported in this module and a
    # function of that name shadows it, so every `status.HTTP_*` below becomes
    # an AttributeError at import time.
    """What this server is doing, right now.

    The budget said at startup what the machine has; this says what is being
    spent of it. Without it a slow server is indistinguishable from a busy one,
    and admission -- the whole point of which is to make a run WAIT -- is
    invisible: a client sees `queued_gpu` and cannot tell whether it is behind
    one job or twelve.

    Bearer-protected, and deliberately narrower than `GET /runs/{id}`. That
    endpoint answers to whoever holds a run's id, which is the client that
    started it; this one lists every run to anyone holding the shared API
    token, which on this deployment is every workstation. So no progress
    MESSAGE appears here -- a message is free text written by a tool and can
    name the file it is working on, which is a patient's. Phases and counts say
    what the server is doing without saying whose data it is doing it to.
    """
    allocation = resources.allocation()
    budget = admission.budget()
    free_vram, total_vram = None, None
    try:
        total_vram, free_vram = resources.detect_vram_bytes()
    except Exception:  # noqa: BLE001 - a status page must not fail on a probe
        pass
    learned = costs.known()
    return {
        "budget": {
            "cpus": allocation.cpus,
            "ram_bytes": allocation.ram_bytes,
            "vram_bytes": allocation.vram_bytes,
            "cpus_per_job": allocation.cpus_per_job,
            "ram_per_job": allocation.ram_per_job,
            "vram_per_job": allocation.vram_per_job,
            "expected_clients": allocation.expected_clients,
            "max_parallel_jobs": allocation.max_parallel_jobs,
        },
        "admission": budget.snapshot(),
        # Whether this server is taking new work, and for how long it is not.
        # The updater polls this to know its own request landed; an operator
        # reads it to tell "an update is in progress" from "something shut this
        # and went away" -- the second being a countdown that keeps running
        # without the server ever restarting.
        "maintenance": maintenance.snapshot(),
        "card": {"free_bytes": free_vram, "total_bytes": total_vram},
        "runs": runs.active(),
        "costs": {
            name: {
                "vram_bytes": cost.vram_bytes,
                "ram_bytes": cost.ram_bytes,
                "samples": cost.samples,
                "vram_spread": round(cost.vram_spread, 3),
                "ram_spread": round(cost.ram_spread, 3),
                "input_dependent": cost.input_dependent,
            }
            for name, cost in learned.items() if cost
        },
    }


class _Maintenance(BaseModel):
    accepting: bool
    # Ignored when reopening. Clamped by `maintenance.MAX_CLOSE_SECONDS`, and
    # the response says what was actually granted rather than what was asked.
    seconds: Optional[float] = None
    reason: str = ""


@app.post("/maintenance", dependencies=[Depends(verify_token)])
def set_maintenance(wanted: _Maintenance) -> dict:
    """Close or reopen this server to new work.

    Called by the updater, which runs on the HOST and not in this container --
    the container has only `server/` bind-mounted and cannot see the
    deployment's `.git`, so the thing that pulls code is necessarily outside,
    and this is how it reaches in.

    **It closes for a bounded time and cannot shut this server for good.** An
    updater that dies between closing the door and restarting the process would
    otherwise leave a clinic with a server that answers `/health` and refuses
    every run until a human notices. See `wire/maintenance.py` for the two
    independent ways back.

    The expected sequence holds the door for well under a second: the updater
    waits for `admission.running == 0` with the door OPEN, closes it, re-reads
    `/status` to confirm nothing slipped in, pulls, and lets the restart clear
    the flag. What is held is the restart, not the drain.
    """
    if not wanted.accepting:
        granted = maintenance.close(
            wanted.seconds if wanted.seconds is not None else 60.0, wanted.reason
        )
        logger.info(
            "Not accepting new work for %.0fs (%s)", granted, wanted.reason or "no reason given"
        )
    else:
        maintenance.reopen()
        logger.info("Accepting new work again")
    return maintenance.snapshot()


class _Move(BaseModel):
    to: str


class _Priority(BaseModel):
    priority: str


class _ClientRule(BaseModel):
    batches: str


@app.get("/clients/me", dependencies=[Depends(verify_token)])
def client_me(request: Request) -> dict:
    """How this workstation's cohort batches will be run, so the client can
    send them accordingly: one at a time when serial, together when parallel
    -- the server enforcing it either way."""
    address = _client_address(request)
    batches = clients.policy_for(address)
    return {
        "client": address,
        "batches": batches,
        "max_parallel": settings.MAX_CONCURRENT_TOOLS if batches == clients.PARALLEL else 1,
    }


@app.post("/admin/clients/{address}/policy", dependencies=[Depends(verify_admin)])
def admin_client_policy(address: str, wanted: _ClientRule) -> dict:
    """Let one workstation's batches run side by side, or one at a time."""
    try:
        answer = clients.set_policy(address, wanted.batches)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))
    logger.info("Operator set the batch rule of %s to %s", address, wanted.batches)
    return answer


class _Door(BaseModel):
    accepting: bool
    # How long to stay closed, when closing. Bounded by the maintenance module.
    hours: float = 2.0
    reason: str = ""


class _UpdateRequest(BaseModel):
    target: str = "all"


@app.post("/admin/door", dependencies=[Depends(verify_admin)])
def admin_door(wanted: _Door) -> dict:
    """Stop accepting new runs, or start again -- the operator's own switch.

    Closing refuses new runs with a 503 that says why; every run already
    received finishes as it would have. What an operator does before an update
    or a restart they are about to make by hand.
    """
    if wanted.accepting:
        maintenance.reopen()
        logger.info("Operator reopened the server to new work")
    else:
        granted = maintenance.close_by_operator(
            wanted.hours * 3600, wanted.reason or "closed by an operator before maintenance")
        logger.info("Operator closed the server to new work for %.0fs", granted)
    return maintenance.snapshot()


@app.get("/admin-panel/updates.json", dependencies=[Depends(verify_admin)])
def panel_updates() -> dict:
    """What the host's update agent says can be updated, and what it is doing."""
    report = updates.overview()
    # The server's own read-only check, which needs no agent; refreshed in the
    # background when stale, so this poll never waits on the network.
    report["check"] = update_check.last()
    report["maintenance"] = maintenance.snapshot()
    report["load"] = {
        "running": admission.budget().snapshot()["running"],
        "in_flight": sum(1 for r in runs.active() if r.get("state") in ("running", "pending")),
    }
    return report


@app.post("/admin/updates/check", dependencies=[Depends(verify_admin)])
def admin_updates_check() -> dict:
    """Check now, from the server itself, what is waiting for the server and
    the tools library. Read-only; applying still needs the host."""
    return update_check.run_check()


@app.post("/admin/update", dependencies=[Depends(verify_admin)])
def admin_update(wanted: _UpdateRequest) -> dict:
    """Ask the host's update agent to apply what is waiting.

    It stops new runs, waits for those in flight to finish, pulls, rebuilds
    what has to be rebuilt and restarts the server. The request is a file the
    agent picks up (wire/updates.py); without an agent running, nothing happens
    and the panel says so.
    """
    try:
        request = updates.request_update(wanted.target, by="admin panel")
    except updates.UpdateError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    logger.info("Operator requested an update (%s)", wanted.target)
    return request


class _DataRequest(BaseModel):
    tool: str
    force: bool = False


@app.post("/admin/data", dependencies=[Depends(verify_admin)])
def admin_data(wanted: _DataRequest) -> dict:
    """Ask the host's agent to download a tool's models and test files.

    `DATA/` is read-only inside the container, so the download happens on the
    host, from the tools library's manifest -- what is missing, or everything
    again with `force`. Nothing is stopped: a tool reads its data when it runs.
    """
    try:
        request = updates.request_data(wanted.tool, wanted.force, by="admin panel")
    except updates.UpdateError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    logger.info("Operator requested the data of %s (force=%s)", wanted.tool, wanted.force)
    return request


@app.delete("/admin/update", dependencies=[Depends(verify_admin)])
def admin_update_withdraw() -> dict:
    """Withdraw a pending update, while the agent is still waiting on runs."""
    return {"withdrawn": updates.withdraw()}


@app.get("/admin/check", dependencies=[Depends(verify_admin)])
def admin_check() -> dict:
    """Whether the admin token the dashboard holds is the right one."""
    return {"admin": True}


@app.post("/admin/queue/{run_id}/move", dependencies=[Depends(verify_admin)])
def admin_move(run_id: str, wanted: _Move) -> dict:
    """Move a run waiting for room: `top`, `up`, `down` or `bottom`."""
    try:
        snapshot = admission.budget().move(run_id, wanted.to)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))
    except admission.NotQueued as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    logger.info("Operator moved run %s %s", run_id, wanted.to)
    return snapshot


@app.post("/admin/runs/{run_id}/priority", dependencies=[Depends(verify_admin)])
def admin_priority(run_id: str, wanted: _Priority) -> dict:
    """Mark a run `high` or back to `normal`, whether it is queued yet or not.

    HIGH puts it ahead of every normal run waiting for room, lets it past the
    wait for a tool slot, and admits it on the widest shape that fits. A run
    already admitted keeps what it was given.
    """
    try:
        snapshot = admission.budget().set_priority(run_id, wanted.priority)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))
    logger.info("Operator set run %s to %s priority", run_id, wanted.priority)
    return snapshot


@app.get("/admin-panel.json", dependencies=[Depends(verify_admin)])
def server_debug_data() -> dict:
    """Everything `/status` reports, plus what the machine says about itself.

    A superset rather than a second opinion: the budget, the admission holds
    and the learned costs are read through `server_status` so the two endpoints
    cannot drift into disagreeing about the same number. What is added here is
    the half that `/status` deliberately does not carry -- live CPU, live host
    memory, disk, in-flight requests, and the queue log -- because it is
    sampled per call rather than being state the server keeps.

    Like `/status`, it carries no progress MESSAGE: a message is free text
    written by a tool and can name the file it is working on, which is a
    patient's. Phases and counts say what the machine is doing without saying
    whose data it is doing it to.
    """
    report = server_status()
    allocation = resources.allocation()
    try:
        node = os.uname().nodename
    except (AttributeError, OSError):
        node = "this server"
    report["hardware"] = (
        f"{node} · {allocation.cpus:.0f} cpus budgeted · "
        f"{_gib(allocation.ram_bytes)} ram · "
        f"{_gib(allocation.vram_bytes)} vram"
    )
    # Which server this is, for a page that will be open beside two others.
    # The origin is the browser's own and is not sent from here; what only this
    # side knows is which machine answered, which process, which card it was
    # told to use, and which tool tree it is serving -- the last being the one
    # that actually distinguishes a checkout from the deployed image. The
    # BASENAME only: a full path is deployment layout, and this page is read by
    # anyone holding the shared token.
    report["server"] = {
        "node": node,
        "pid": os.getpid(),
        "device": settings.DEVICE,
        "tools": os.path.basename(str(settings.TOOLS_DIR).rstrip("/").split(os.pathsep)[0]),
        # So the page can say when a paused run will be let go rather than
        # leaving a reader to wonder whether it is stuck forever.
        "run_ttl_seconds": settings.RUN_TTL_SECONDS,
        "paused_ttl_seconds": settings.PAUSED_RUN_TTL_SECONDS,
        # Whether the operator controls exist here at all, so the page can
        # offer them or say why it does not. Never the token itself.
        "admin_enabled": bool(settings.ADMIN_TOKEN),
    }
    # The live listing again, with each run's open nested calls: `/status`
    # keeps the narrower one it always had, and only this page needs to say
    # that an AREG is, right now, inside ASO inside ALI_CBCT.
    report["runs"] = runs.active(with_chain=True)
    report["trace"] = telemetry.trace(since=time.time() - TRACE_WINDOW_SECONDS)
    report["cpu_percent"] = telemetry.cpu_percent()
    report["ram"] = telemetry.ram()
    report["inflight"] = telemetry.inflight()
    report["queue"] = telemetry.queue_history()
    # The ledger and the live listing are sent side by side rather than merged:
    # `runs` is read from disk and is the truth about what is happening NOW
    # across every worker, while `ledger` is this process's memory of what it
    # admitted and what that cost. The page joins them on `run_id`, and a run
    # present in one and not the other is a fact worth being able to see.
    report["ledger"] = telemetry.run_ledger()
    report["uptime"] = telemetry.tool_uptime()
    report["clients"] = _client_activity(report["runs"], telemetry.run_ledger(limit=telemetry.LEDGER_SIZE))
    # TEMP_DIR is where uploads, results and run directories live -- the space a
    # download actually costs this machine. DATA_DIR is mounted read-only and
    # cannot grow, but it is the other half of "what is this disk holding".
    report["disk"] = telemetry.disk({
        "temp": settings.TEMP_DIR,
        "data": settings.DATA_DIR,
    })
    return report


_ACTIVITY_LEVEL = {
    runs.PHASE_DONE: "ok",
    runs.PHASE_FAILED: "error",
    runs.PHASE_CANCELLED: "warn",
    runs.PHASE_QUEUED_GPU: "warn",
    runs.PHASE_PAUSED: "warn",
}

# What the server says a phase MEANS. The console shows these instead of the
# tool's own words, and that substitution is the whole point of the endpoint
# below: a phase is this server's vocabulary and carries nothing but itself.
_ACTIVITY_TEXT = {
    runs.PHASE_RECEIVED: "request received",
    runs.PHASE_STAGING: "staging inputs",
    runs.PHASE_QUEUED_GPU: "waiting for room on this machine",
    runs.PHASE_RUNNING: "running",
    runs.PHASE_PACKAGING: "packaging the result",
    runs.PHASE_PAUSED: "paused, holding its work",
    runs.PHASE_DONE: "finished",
    runs.PHASE_FAILED: "failed",
    runs.PHASE_CANCELLED: "cancelled by the client",
}


# How long a workstation stays on the dashboard after its last run, and how
# long a finished cohort stays listed under it.
_CLIENT_SEEN_SECONDS = 24 * 3600
_BATCH_SHOWN_SECONDS = 15 * 60


def _client_activity(live: list, ledger: list) -> list:
    """Per workstation: its batch rule, what it has in flight, and its cohorts.

    A cohort is every run sharing a batch id from one address. Its `done` and
    `failed` come from the ledger, its `running` and `waiting` from the live
    listing, and `total` is what the client said it would send -- so a serial
    cohort shows "3 of 12" before batches 4 to 12 have even been sent.
    """
    now = time.time()
    rules = clients.policies()
    rows: dict = {}

    def row(address):
        return rows.setdefault(address, {
            "client": address, "batches": clients.policy_for(address),
            "custom": address in rules, "last_seen": 0.0,
            "running": 0, "waiting": 0, "runs_today": 0, "cohorts": {},
        })

    def cohort(entry, batch, tool):
        return entry["cohorts"].setdefault(batch["id"], {
            "id": batch["id"], "tool": tool, "total": batch.get("total"),
            "done": 0, "failed": 0, "running": 0, "waiting": 0,
            "started_at": None, "last_at": 0.0,
        })

    for record in ledger:
        address = record.get("client")
        seen = record.get("ended_at") or record.get("started_at") or 0
        if not address or now - seen > _CLIENT_SEEN_SECONDS:
            continue
        entry = row(address)
        entry["last_seen"] = max(entry["last_seen"], seen)
        if record.get("started_at") and now - record["started_at"] < 24 * 3600:
            entry["runs_today"] += 1
        batch = record.get("batch")
        if batch and record.get("ended_at"):
            group = cohort(entry, batch, record.get("tool"))
            group["done" if record.get("outcome") == "done" else "failed"] += 1
            group["last_at"] = max(group["last_at"], record["ended_at"])
            start = record.get("started_at")
            if start and (group["started_at"] is None or start < group["started_at"]):
                group["started_at"] = start
    for run in live:
        address = run.get("client")
        if not address or run.get("state") not in ("running", "pending"):
            continue
        entry = row(address)
        entry["last_seen"] = max(entry["last_seen"], run.get("updated_at") or now)
        waiting = run.get("phase") in (runs.PHASE_QUEUED_GPU, runs.PHASE_RECEIVED)
        entry["waiting" if waiting else "running"] += 1
        batch = run.get("batch")
        if batch:
            group = cohort(entry, batch, run.get("tool"))
            group["waiting" if waiting else "running"] += 1
            group["last_at"] = now
            start = run.get("started_at")
            if start and (group["started_at"] is None or start < group["started_at"]):
                group["started_at"] = start
    for address in rules:
        row(address)
    result = []
    for entry in rows.values():
        groups = [g for g in entry["cohorts"].values()
                  if g["running"] or g["waiting"] or now - g["last_at"] < _BATCH_SHOWN_SECONDS]
        groups.sort(key=lambda g: g["started_at"] or 0, reverse=True)
        entry["cohorts"] = groups
        entry["active_cohorts"] = sum(1 for g in groups if g["running"] or g["waiting"])
        result.append(entry)
    result.sort(key=lambda e: (-(e["running"] + e["waiting"]), -e["last_seen"]))
    return result


# How much of the resource trace rides every poll of the operator page. The
# per-tool view asks for its own, wider window.
TRACE_WINDOW_SECONDS = 30 * 60


@app.get("/admin-panel/history.json", dependencies=[Depends(verify_admin)])
def panel_history(limit: int = 500) -> dict:
    """Every finished run the history holds, newest first, for the panel's full
    history window. The same records the dashboard's strip shows, more of them:
    timings, shapes, the tools each run called -- never a value or a file name."""
    limit = max(1, min(int(limit), telemetry.LEDGER_SIZE))
    records = [r for r in telemetry.run_ledger(limit=telemetry.LEDGER_SIZE) if r.get("ended_at")]
    return {"runs": records[:limit], "held": len(records), "capacity": telemetry.LEDGER_SIZE}


@app.get("/admin-panel/tools/{tool_name}.json", dependencies=[Depends(verify_admin)])
def server_debug_tool(tool_name: str, limit: int = 120) -> dict:
    """One tool over its recent runs: what the operator page draws when a tool
    is clicked.

    Built from the ledger, which keeps finished runs across restarts, so the
    graphs have a past: each run's phases and nested calls for the Gantt, the
    mean time spent in each phase, and the machine's trace over the window
    those runs cover. The same rule as the rest of the page: timings, shapes
    and tool names, never an argument value or a file name.
    """
    limit = max(1, min(int(limit), telemetry.LEDGER_SIZE))
    records = telemetry.run_ledger(limit=limit, tool=tool_name)
    phases = {}
    for record in records:
        for span in record.get("spans") or []:
            if span.get("end") is None or span.get("start") is None:
                continue
            row = phases.setdefault(span["phase"], {"phase": span["phase"], "seconds": 0.0, "runs": 0})
            row["seconds"] += max(0.0, span["end"] - span["start"])
            row["runs"] += 1
    for row in phases.values():
        row["mean"] = round(row["seconds"] / row["runs"], 2) if row["runs"] else 0.0
        row["seconds"] = round(row["seconds"], 1)
    finished = [r for r in records if r.get("seconds") is not None]
    since = min((r["started_at"] for r in records), default=time.time()) if records else None
    learned = costs.known().get(tool_name)
    # What this deployment holds for the tool, as a workstation would see it.
    # A tool in the history that is no longer served has none.
    try:
        hosted = list_tool_data(tool_name)
        hosted["folder"] = deployment_config.data_slug(get_tool(tool_name).name)
    except HTTPException:
        hosted = None
    return {
        "tool": tool_name,
        "runs": records,
        "phases": sorted(phases.values(), key=lambda row: row["seconds"], reverse=True),
        "summary": {
            "runs": len(records),
            "ok": sum(1 for r in records if r.get("outcome") == "done"),
            "failed": sum(1 for r in records if r.get("outcome") == "failed"),
            "running": sum(1 for r in records if r.get("ended_at") is None),
            "mean_seconds": round(sum(r["seconds"] for r in finished) / len(finished), 1)
            if finished else None,
            "mean_wait": round(sum(r.get("waited") or 0 for r in finished) / len(finished), 2)
            if finished else None,
        },
        "cost": {
            "vram_bytes": learned.vram_bytes,
            "ram_bytes": learned.ram_bytes,
            "samples": learned.samples,
        } if learned else None,
        "trace": telemetry.trace(since=since, max_points=900),
        "data": hosted,
    }


@app.get("/admin-panel/runs/{run_id}.json", dependencies=[Depends(verify_admin)])
def server_debug_run(run_id: str) -> dict:
    """One run's activity, COMPOSED by this server rather than quoted from the
    tool.

    It reads like a console and is deliberately not one. A tool's real stdout
    names the file it is working on -- nnUNet prints every case, and shapeaxi's
    output is swallowed elsewhere in this codebase for exactly that reason --
    and this page is read by anyone holding the shared API token, which on this
    deployment is every workstation. So the tool's own words never travel:
    every line here is built from the phase, the fraction and the depth, which
    are this server's vocabulary and say what is happening without saying whose
    data it is happening to.

    A tool CAN still make itself heard, and usefully: a progress record it
    wrote turns into "42% · scan 14 of 40"-shaped text built from the fraction
    alone, so a chatty tool produces a busier console than a silent one without
    a character of its text being republished.

    **The one exception is a tool's LOG**, `sup.log(...)`: lines written to be
    read by whoever runs the server, shown here at their level -- and passed
    through `redact.scrub` first, so the sentence reaches the operator and any
    path, file name or identifier in it does not. A failed run ends with its
    diagnosis: which tool in the chain broke, at which line, and why.
    """
    record = telemetry.ledger_record(run_id)
    try:
        events = runs.read_events(run_id, logs=runs.LOG_AUDIENCES)
    except runs.RunError as exc:
        # Reaped, which is the normal state of a run that finished more than a
        # few minutes ago. Its timeline survives in the ledger, so the page can
        # still draw it; only the console is gone.
        if record is None:
            raise _run_error(exc)
        return {
            "run_id": run_id, "lines": _kept_lines(record), "reaped": True, "record": record,
            "timeline": {"spans": record.get("spans") or [],
                         "nested": record.get("nested") or [],
                         "chain": [], "measured": record.get("measured")},
        }
    lines = []
    for event in events:
        if event.get("kind") == runs.LOG_KIND:
            lines.append(_log_line(event))
            continue
        phase = event.get("phase") or runs.PHASE_RUNNING
        fraction = event.get("fraction")
        text = _ACTIVITY_TEXT.get(phase, phase)
        if fraction is not None:
            text = f"{text} · {fraction * 100:.0f}%"
        lines.append({
            "seq": event.get("seq"),
            "at": event.get("at"),
            "depth": event.get("depth", 0),
            "phase": phase,
            "state": event.get("state"),
            "fraction": fraction,
            "level": _ACTIVITY_LEVEL.get(phase, "info"),
            "text": text,
        })
    if record and record.get("failure"):
        lines.append(_failure_line(record["failure"], record.get("ended_at")))
    return {"run_id": run_id, "lines": lines, "reaped": False, "record": record,
            "timeline": runs.timeline(events)}


_LOG_CONSOLE_LEVEL = {"debug": "info", "info": "info", "warning": "warn", "error": "error"}


def _log_line(event: dict) -> dict:
    """A tool's log line as a console line: its level, its source, redacted."""
    source = event.get("source")
    text = redact.scrub(event.get("message"))
    if event.get("audience") == runs.LOG_AUDIENCE_USER:
        text = "(shown to the user) " + text
    return {
        "seq": event.get("seq"), "at": event.get("at"),
        "depth": event.get("depth", 0), "phase": runs.PHASE_RUNNING,
        "state": runs.STATE_RUNNING, "fraction": None, "kind": runs.LOG_KIND,
        "level": _LOG_CONSOLE_LEVEL.get(event.get("level"), "info"),
        "text": f"[{source}] {text}" if source else text,
    }


def _failure_line(failure: dict, at) -> dict:
    return {"seq": None, "at": at, "depth": 0, "phase": runs.PHASE_FAILED,
            "state": runs.STATE_FAILED, "fraction": None, "kind": "failure",
            "level": "error", "text": "failed in " + _described(failure)}


def _kept_lines(record: Optional[dict]) -> list:
    """What a reaped run's console can still show: the warnings and errors its
    ledger kept, already redacted, and how it failed."""
    if not record:
        return []
    lines = [_log_line(dict(line, kind=runs.LOG_KIND, message=line.get("message")))
             for line in record.get("logs") or ()]
    if record.get("failure"):
        lines.append(_failure_line(record["failure"], record.get("ended_at")))
    return lines


def _benchmark_schemas() -> dict:
    """Every served tool's arguments, projected to what a battery plan reads.

    Read live rather than cached: a bundle staged into `DATA/` since startup
    should make its tool runnable without a restart, and the whole point of
    resolving from what is hosted is that it tracks the deployment.
    """
    # Only what the resolver reads, projected straight off the ArgSpec rather
    # than through `/tools`'s full publication: duplicating forty lines of
    # presentation here would be a second place for the two to drift.
    schemas = {
        name: {"arguments": {
            arg: {"required": spec.required,
                  "server_selectable": spec.server_selectable,
                  "initial": spec.initial,
                  "choices": spec.choices,
                  # Carried because the resolver needs it, and for one reason:
                  # a facade publishes every input as optional -- `t1` belongs
                  # to AREG's CBCT modes, `ios` to its CBCT-to-IOS one -- and
                  # `visible_when` is the only thing that says which. Without
                  # it here a battery sends the mode alone and collects a 422
                  # from the tool the facade dispatched to.
                  "visible_when": spec.visible_when,
                  # Carried for the same reason `visible_when` is: a scoped
                  # argument draws from ONE subfolder, and a resolver reading
                  # the whole catalogue picks a name that argument cannot
                  # resolve. `AREG_CBCT.t1` is scoped to `T1`; unscoped, the
                  # first hosted name is `IOSCBCT`, a sibling, and every AREG
                  # arm answered 404 before a process started.
                  "selectable_scope": spec.selectable_scope,
                  "type": _type_name(spec.types[0])}
            for arg, spec in tool.arguments.items()
        }}
        for name, tool in TOOLS.items()
    }
    return schemas


def _benchmark_hosted(name: str) -> dict:
    """What this server hosts for one tool, per argument scope."""
    slug = deployment_config.data_slug(name)
    # One list per scope the tool's arguments actually name, beside the
    # unscoped catalogue an unscoped argument still draws from.
    scopes = {
        spec.selectable_scope
        for spec in TOOLS[name].arguments.values()
        if getattr(spec, "selectable_scope", None)
    }
    return {"models": data_store.list_models(slug),
            "testfiles": data_store.list_testfiles(slug),
            "testfiles_by_scope": {
                scope: data_store.list_testfiles(slug, scope)
                for scope in sorted(scopes)
            },
            "models_by_scope": {
                scope: data_store.list_models(slug, scope)
                for scope in sorted(scopes)
            }}


def _benchmark_bench(name: str) -> list:
    """`DATA/<tool>/bench/`, described: what a custom battery may take as input."""
    try:
        return data_store.describe_bench(deployment_config.data_slug(name))
    except OSError:
        return []


def _benchmark_resolution() -> dict:
    """Which tools this deployment could actually run a preset with."""
    return benchmark_presets.runnable_tools(_benchmark_schemas(), _benchmark_hosted)


@app.get("/benchmark", include_in_schema=False)
def benchmark_home() -> HTMLResponse:
    """Both halves in one place: what was measured, and what to measure next.

    Unauthenticated like the other pages -- it holds no reading and starts
    nothing by itself; every call it makes carries the admin token the reader
    typed, the one the admin panel keeps.
    """
    return HTMLResponse(benchmark_launch_page.LAUNCH_PAGE)


@app.get("/benchmark/presets", dependencies=[Depends(verify_admin)])
def benchmark_catalogue() -> dict:
    """Every preset, and what it would do on THIS deployment."""
    resolved = _benchmark_resolution()
    return {
        "presets": benchmark_presets.catalogue(resolved),
        "tools": {name: {"ready": not entry["missing"],
                         "missing": entry["missing"],
                         "params": sorted(entry["params"])}
                  for name, entry in sorted(resolved.items())},
        "limits": {"max_runs": benchmark_presets.MAX_RUNS,
                   "max_concurrency": benchmark_presets.MAX_CONCURRENCY},
        "custom_limits": {"max_runs": benchmark_presets.CUSTOM_MAX_RUNS,
                          "max_concurrency": benchmark_presets.CUSTOM_MAX_CONCURRENCY,
                          "max_configs": benchmark_presets.CUSTOM_MAX_CONFIGS,
                          "max_ladder": benchmark_presets.CUSTOM_MAX_LADDER,
                          "max_stagger": benchmark_presets.CUSTOM_MAX_STAGGER},
        "running": benchmark_jobs.current(),
    }


@app.get("/benchmark/form/{tool_name}", dependencies=[Depends(verify_admin)])
def benchmark_form(tool_name: str) -> dict:
    """What the custom launcher needs to fill one tool's form, beside `/tools`.

    `/tools` already publishes the schema the form is drawn from; this adds
    what only an operator may see: the names this deployment hosts for each
    argument, the bench inputs staged for the tool, and what a preset would
    have sent, so the form opens on a run that works and only what matters has
    to be changed.
    """
    if tool_name not in TOOLS:
        raise HTTPException(status_code=404, detail=f"No tool called '{tool_name}'.")
    schemas = _benchmark_schemas()
    hosted = _benchmark_hosted(tool_name)
    resolved = benchmark_presets.resolve_arguments(schemas[tool_name], hosted, tool_name)
    return {"tool": tool_name, "defaults": resolved["params"],
            "missing": resolved["missing"], "hosted": hosted,
            "bench": _benchmark_bench(tool_name),
            "bench_folder": f"DATA/{deployment_config.data_slug(tool_name)}/bench"}


class BatteryRequest(BaseModel):
    preset: Optional[str] = None
    tools: Optional[list] = None
    concurrency: Optional[int] = None
    # A battery composed in the launcher instead of a preset; see
    # `benchmark_presets.build_custom_plan` for its shape.
    custom: Optional[dict] = None


def _custom_plan(spec: dict) -> dict:
    """A custom battery's plan, with every bench name turned into a path.

    The names stop here. The plan the battery process reads carries the path
    it uploads from, and the summary it writes carries the input's SHAPE --
    files, bytes -- never its name: a bench case is a clinical one, and its
    name can be a patient's.
    """
    plan = benchmark_presets.build_custom_plan(
        spec, _benchmark_schemas(), _benchmark_hosted, _benchmark_bench)
    for arm in plan["arms"]:
        for run in arm["runs"]:
            slug = deployment_config.data_slug(run["tool"])
            uploads = {}
            for argument, entry in run.pop("bench").items():
                try:
                    uploads[argument] = data_store.resolve_bench(slug, entry).path
                except DataNotFoundError as exc:
                    raise benchmark_presets.PresetError(str(exc))
            run["uploads"] = uploads
    return plan


def _local_base(request: Request) -> str:
    """`http://127.0.0.1:<port>`: this server, on its own loopback.

    The battery runs in the same container as uvicorn, so it never needs to
    leave it: loopback traffic does not reach the Docker network, the host or
    anything beyond, which is the property that makes plain HTTP acceptable
    here -- the same plain HTTP the reverse proxy already speaks to uvicorn
    after terminating TLS, only shorter. The port is the one uvicorn accepted
    this request on.
    """
    server = request.scope.get("server") or (None, 8000)
    port = server[1] or 8000
    return f"http://127.0.0.1:{port}"


@app.post("/benchmark/run", dependencies=[Depends(verify_admin), Depends(maintenance.require_accepting)])
def benchmark_start(request: Request, body: BatteryRequest) -> dict:
    """Start a battery. One at a time, and never on the event loop.

    The plan is built HERE so a bad preset is a 422 before anything is spawned,
    and the runs themselves happen in another process -- a battery executed
    inside this one would hold the very slots it is measuring.
    """
    try:
        if body.custom is not None:
            plan = _custom_plan(body.custom)
        elif body.preset:
            plan = benchmark_presets.build_plan(
                body.preset, _benchmark_resolution(),
                tools=body.tools, concurrency=body.concurrency)
        else:
            raise benchmark_presets.PresetError("Name a preset, or compose a custom battery.")
    except benchmark_presets.PresetError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    allocation = resources.allocation()
    plan["hardware"] = (
        f"{allocation.cpus:.0f} cpus · "
        f"{_gib(allocation.ram_bytes)} ram · "
        f"{_gib(allocation.vram_bytes)} vram"
    )
    # The battery talks to this server over HTTP like any other client, so it
    # needs an address to reach it at -- and it runs HERE, beside uvicorn, so
    # that is this server's own loopback, never the address the operator
    # typed. Behind the reverse proxy those differ: the request arrives as
    # https://<public address>, which the container cannot reach (its connect
    # hung in SYN_SENT and the battery sent nothing). Loopback never leaves the
    # container, so nothing a battery sends crosses any network.
    base = _local_base(request)
    try:
        return benchmark_jobs.start(
            plan, _battery_dir(), base, settings.API_TOKEN,
            os.path.join(settings.TEMP_DIR, "benchmarks"))
    except benchmark_jobs.BatteryError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not start it: {exc}")


@app.get("/benchmark/status", dependencies=[Depends(verify_admin)])
def benchmark_status() -> dict:
    return {"running": benchmark_jobs.current()}


@app.delete("/benchmark/run", dependencies=[Depends(verify_admin)])
def benchmark_stop() -> dict:
    """Stop the battery, and everything it started."""
    return {"stopped": benchmark_jobs.stop()}


@app.get("/admin-panel", include_in_schema=False)
def admin_panel() -> HTMLResponse:
    """The operator's panel. The page itself holds no reading and is served to
    anyone; everything it shows comes from `/admin-panel*.json`, which answers
    only to the ADMIN token. A clinician's workstation holds the API token,
    which opens nothing here."""
    return HTMLResponse(debug_page.DEBUG_PAGE)


@app.get("/server-debug", include_in_schema=False)
@app.get("/panel-admin", include_in_schema=False)
def server_debug_moved() -> RedirectResponse:
    """The panel's earlier addresses: both land on it rather than on a 404."""
    return RedirectResponse(url="admin-panel", status_code=status.HTTP_308_PERMANENT_REDIRECT)


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
                    # Which subfolder of the tool's hosted files this argument
                    # draws from, when a deployment scoped it. The client picks
                    # its list out of `scoped` with this; without it published,
                    # the scoped lists are sent and nobody can tell which is
                    # whose.
                    "selectable_scope": spec.selectable_scope,
                    # For "choice"/"multichoice": the options to render, each
                    # with its initial state. null for every other type.
                    "choices": spec.choices,
                    # For a SCALAR argument: the value a client should pre-fill
                    # its widget with, so a spin box does not start at Qt's 0
                    # while the tool's own default reads 5.
                    "initial": spec.initial,
                    # {type name: accepted extensions}, so a client builds its
                    # file dialog filters without a copy of FILE_TYPES on its
                    # side. null for the generic "file" type (which falls back
                    # to ALLOWED_EXTENSIONS) and for a non-file argument.
                    "extensions": _extensions_of(spec),
                    # Presentation hints (see ArgSpec): how a client lays this
                    # argument out and when to show it. All null on a tool that
                    # declares none, so its panel renders exactly as before.
                    "label": spec.label,
                    "section": spec.section,
                    "visible_when": spec.visible_when,
                    "options_when": spec.options_when,
                    # True: do not render this at all. Still published, because
                    # a client that hides it must still know it exists rather
                    # than treat it as an argument the server invented.
                    "hidden": spec.hidden,
                    "ui": spec.ui,
                    # A vec2's two axes. The ranges are not presentation -- the
                    # server validates against them -- but the client needs them
                    # to build the pad, and the end labels to say what each end
                    # means: "0.8" carries no meaning in a mouth, "mid"/"out"
                    # does. Lists rather than tuples, so the wire shape does not
                    # depend on how the schema spelled them.
                    "x_range": list(spec.x_range) if spec.x_range else None,
                    "y_range": list(spec.y_range) if spec.y_range else None,
                    "x_labels": list(spec.x_labels) if spec.x_labels else None,
                    # How many columns this argument's section is laid out in.
                    "section_columns": spec.section_columns,
                    # Arguments naming one cell are drawn together in it.
                    "cell": spec.cell,
                    # What each of the two numbers is, written beside its box.
                    "x_label": spec.x_label,
                    "y_label": spec.y_label,
                    "y_labels": list(spec.y_labels) if spec.y_labels else None,
                    # Listed explicitly so the wire shape does not depend on
                    # whether a tool spelled its catalog as a tuple or a list.
                    "groups": (
                        {name: list(options) for name, options in spec.groups.items()}
                        if spec.groups
                        else None
                    ),
                    # Only when a facade actually has one, unlike every other
                    # hint above. Emitting `"groups_when": null` on every
                    # argument of every tool would change the published shape of
                    # tools that have nothing to do with facades, and that shape
                    # is pinned byte for byte by tests/golden/tools_response.json
                    # -- a fixture whose whole point is that the Slicer client
                    # builds its panel from it, so it is not what gets updated.
                    **({"groups_when": spec.groups_when} if spec.groups_when else {}),
                    # Same reasoning, same shape: omitted rather than null, so a
                    # tool that names none of its options publishes exactly what
                    # it published before this field existed.
                    **({"option_help": spec.option_help} if spec.option_help else {}),
                    **({"option_kind": spec.option_kind} if spec.option_kind else {}),
                    # Same shape again: omitted rather than null, because an
                    # empty multichoice is a meaningful answer everywhere it is
                    # not declared, and saying so on every argument of every
                    # tool would change a shape pinned byte for byte.
                    **({"min_selected": spec.min_selected} if spec.min_selected else {}),
                    **({"select_all": True} if spec.select_all else {}),
                }
                for arg_name, spec in tool.arguments.items()
            },
            "output_kind": tool.output_kind,
            # How to split a folder of inputs into several runs:
            # `{"axis": the argument to split, "max_mb": ..., "max_files": ...}`,
            # whichever cap binds first. Advisory -- a client that ignores it
            # sends the cohort whole, exactly as every client did before this
            # existed, and every request is still a request.
            #
            # Omitted rather than null, like the argument-level hints above and
            # for the same reason: tests/golden/tools_response.json pins the
            # published shape byte for byte, and a tool that cannot be split
            # must publish what it published before the field existed.
            **({"batch": tool.batch} if tool.batch else {}),
            # Under its own key: a client that knows only `batch` must not
            # split one of a pair of folders and send the other whole.
            **({"paired_batch": tool.paired_batch} if getattr(tool, "paired_batch", None) else {}),
        }
        for tool in TOOLS.values()
    ]


class _PairsRequest(BaseModel):
    # {argument: [relative file path, ...]} -- names only, as a client lists a
    # folder it is about to split.
    inputs: dict
    # What else the client has chosen; a facade needs its mode to know which
    # engine's pairing to ask.
    arguments: dict = {}


# Bounds on a pairing request: it carries names, not files, and a cohort of a
# few thousand scans is a few hundred kilobytes of them.
_PAIRS_MAX_NAMES = 50000
_PAIRS_MAX_NAME = 1024
_PAIRS_TIMEOUT_SECONDS = 120


def _pairs_target(tool_name: str, arguments: dict):
    """The tool whose `pairs()` answers, and its plan; a facade resolves to the
    engine of the chosen mode."""
    try:
        tool = get_tool(tool_name)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(tool, FacadeTool):
        mode = arguments.get(facade.MODE_ARGUMENT)
        if not mode:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                                detail=f"{tool.name} needs its '{facade.MODE_ARGUMENT}' to pair inputs.")
        try:
            tool = get_tool(tool.target_for(str(mode)))
        except (ToolArgumentError, KeyError) as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))
    plan = getattr(tool, "paired_batch", None)
    if not plan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"{tool.name} does not pair its inputs.")
    return tool, plan


@app.post("/tools/{tool_name}/pairs", dependencies=[Depends(verify_token)])
def tool_pairs(tool_name: str, wanted: _PairsRequest) -> dict:
    """Which of these file names go together, answered by the tool itself.

    For a client splitting a cohort for a tool whose inputs are paired: every
    batch has to carry the same subjects on every axis, and only the tool knows
    how it pairs them. This runs the tool's own `pairs()` in its own
    interpreter, on NAMES only -- no file is uploaded, read or kept -- and
    hands back its groups. The server knows nothing of the rule.

    The names are a patient's file names, so they are never logged.
    """
    tool, plan = _pairs_target(tool_name, wanted.arguments or {})
    axes = plan["axes"]
    if sorted(wanted.inputs) != sorted(axes):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                            detail=f"{tool.name} pairs {', '.join(axes)}; send exactly those.")
    inputs = {}
    for axis in axes:
        names = wanted.inputs[axis]
        if not isinstance(names, list) or len(names) > _PAIRS_MAX_NAMES or not all(
                isinstance(n, str) and 0 < len(n) <= _PAIRS_MAX_NAME and not os.path.isabs(n)
                and ".." not in n.replace("\\", "/").split("/") for n in names):
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                                detail=f"'{axis}' must be a list of relative file names.")
        inputs[axis] = names
    interpreter = dispatch.tool_interpreter(tool.name)
    if not os.path.isfile(interpreter):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"{tool.name} is not installed on this server.")
    job_dir = tempfile.mkdtemp(prefix="pairs_", dir=settings.TEMP_DIR)
    try:
        with open(os.path.join(job_dir, "job.json"), "w", encoding="utf-8") as handle:
            json.dump({"tool": tool.name, "job_dir": job_dir, "params": inputs,
                       "entry": runner.PAIRS_ENTRY}, handle)
        try:
            completed = subprocess.run(
                [interpreter, settings.RUNNER_PATH, "--job", os.path.join(job_dir, "job.json")],
                capture_output=True, text=True, timeout=_PAIRS_TIMEOUT_SECONDS, cwd=job_dir,
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                                detail=f"{tool.name} took too long to pair these names.")
        try:
            with open(os.path.join(job_dir, "result.json"), encoding="utf-8") as handle:
                body = json.load(handle)
        except (OSError, ValueError):
            logger.error("pairs for %s failed without a result: %s", tool.name, completed.stderr[-2000:])
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"{tool.name} could not pair these inputs.")
        if "error" in body:
            error = body["error"] or {}
            bases = error.get("bases") if isinstance(error.get("bases"), list) else []
            code = _tool_error_status([error.get("type")] + bases)
            detail = error.get("message") if code < 500 else f"{tool.name} could not pair these inputs."
            raise HTTPException(status_code=code, detail=detail)
        answer = body.get("result") or {}
        return {"tool": tool.name, "axes": axes, **answer}
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


@app.get("/tools/{tool_name}/data", dependencies=[Depends(verify_token)])
def list_tool_data(tool_name: str) -> dict:
    """List models and test files available on the server for this tool, so
    a client can pick one instead of uploading its own (see ArgSpec.server_selectable).
    """
    try:
        tool = get_tool(tool_name)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    slug = deployment_config.data_slug(tool.name)
    # `models` and `testfiles` stay lists of NAMES, exactly as they were: an
    # older client reads them unchanged. `entries` is additive, and carries the
    # two things a name cannot say -- whether an entry is one file or a whole
    # folder, and how many bytes picking it costs, now that the client
    # downloads what a user picks rather than naming it to the server.
    # One list per SCOPE, beside the tool's own. A tool serving several
    # modalities -- AREG registers CBCT volumes, intraoral surfaces, and one
    # onto the other -- has one folder of test data per modality, and an
    # argument that draws from one of them must not be offered the others: the
    # CBCT baseline picker was listing intraoral meshes. Additive, so a client
    # that reads only the two flat lists behaves exactly as before.
    scopes = sorted({
        spec.selectable_scope for spec in tool.arguments.values()
        if getattr(spec, "selectable_scope", None)
    })
    scoped = {
        scope: {
            "models": data_store.list_models(slug, scope),
            "testfiles": data_store.list_testfiles(slug, scope),
            "entries": {
                "models": data_store.describe(slug, "models", scope),
                "testfiles": data_store.describe(slug, "testfiles", scope),
            },
        }
        for scope in scopes
    }
    return {
        "models": data_store.list_models(slug),
        "testfiles": data_store.list_testfiles(slug),
        "entries": {
            "models": data_store.describe(slug, "models"),
            "testfiles": data_store.describe(slug, "testfiles"),
        },
        **({"scoped": scoped} if scoped else {}),
    }


def _remove_path(path: str) -> None:
    """Remove a file or a whole directory tree -- for backend-materialized temp
    copies (ResolvedFile.is_temporary), which can be either."""
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        os.remove(path)


# HEAD as well as GET, and it is not decoration. The client probes with a HEAD
# to learn the size and whether ranges are served, and only then splits the
# transfer across parallel connections. Starlette does not add HEAD to a GET
# route, so the probe was answered `405 Method Not Allowed` -- every probe
# failed, and every test file came down one connection at a time. Invisible on
# a loopback at 378 MB/s; the whole point of the parallel path on the link a
# clinician actually has.
@app.api_route("/tools/{tool_name}/testfiles/{filename}", methods=["GET", "HEAD"],
               dependencies=[Depends(verify_token)])
async def download_testfile(tool_name: str, filename: str,
                            background_tasks: BackgroundTasks, scope: str = ""):
    """Stream one of the tool's hosted test files, so a user can fill an input
    with reference data. The valid names are what GET /tools/{name}/data lists.

    `scope` names the subfolder the name was listed under, for a tool whose
    deployment scopes an argument's hosted files. It has to be said, not
    guessed: a name is bare and two scopes may legitimately hold the same one.
    Omitted, the tool's own folder is read, which is every unscoped
    deployment. A run resolves the same name through the ARGUMENT it was sent
    for, and this route had to learn the same trick -- without it the picker
    listed a file it could not then download.

    Only test files are downloadable. Models are deliberately NOT: they are
    selected by name and used in place (see ArgSpec.server_selectable).

    A test entry that is a FOLDER is zipped on the fly and the client unpacks
    it back into a directory on its side.
    """
    start_time = time.monotonic()
    try:
        tool = get_tool(tool_name)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    try:
        resolved = data_store.resolve_testfile(
            deployment_config.data_slug(tool.name), filename,
            *((scope,) if scope else ()),
        )
    except DataNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    if resolved.is_temporary:
        background_tasks.add_task(_remove_path, resolved.path)

    path = resolved.path
    # Built for THIS request, rather than read off the disk. What that costs is
    # decided a dozen lines below, where the response says whether it may be
    # ranged.
    generated = os.path.isdir(path)
    if generated:
        # DATA_DIR is read-only: the archive is built in its own staging dir
        # under TEMP_DIR, which must outlive the response stream -- hence the
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

    # Same log shape as /run: tool, status, duration, size -- never the file
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
        # An archive built for one request has no byte range worth offering, and
        # offering one is far worse than useless: the client probes, sees
        # ranges, and splits a 339 MB cohort across 43 parallel parts -- so the
        # server builds the same 339 MB archive 43 times to deliver it once.
        # Measured against AREG's CBCT_FullyAuto: one 8 MB range costs 40% of
        # the entire download, and the transfer took 39.1s ranged against 1.5s
        # in a single stream. `Accept-Ranges: none` is exactly what the client
        # probes for, so it falls back to one connection, and one build.
        #
        # Only the ADVERTISEMENT is withdrawn. Starlette still answers a Range
        # a client sends anyway, and that stays correct because two builds of
        # one folder are byte-identical (pinned by a test) -- it is merely slow,
        # which is the right way round for a client that ignores the header.
        #
        # A real file is untouched: it IS on disk, its ranges are free, and
        # parallel connections are the whole point of the probe.
        headers={"Accept-Ranges": "none"} if generated else None,
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


@app.post("/uploads", dependencies=[Depends(verify_token), Depends(maintenance.require_accepting)])
async def create_upload(spec: _NewUpload) -> dict:
    """Open a session the client then fills with parallel PUTs.

    Answering with `chunk_size` rather than accepting the client's keeps the
    layout single-sourced: part n is always
    `[n * chunk_size, (n+1) * chunk_size)`, computed by both sides from the one
    number returned here.
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

    The body is the raw bytes, no multipart framing: there is exactly one thing
    in it. `Content-Encoding: gzip` is honoured for inputs not already
    compressed (an uncompressed .nii or .vtk is 3-4x smaller deflated), and
    `X-Part-SHA256` is checked against what lands on disk either way.
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
    several connections at once. A client that sends no Range still gets the
    entire file in one response.
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
    -- the reaper collects what is never claimed -- but it keeps TEMP_DIR flat
    under load instead of holding every result for the full TTL."""
    await anyio.to_thread.run_sync(transfer.discard_result, result_id)
    return {"status": "ok"}


# ----------------------------------------------------------------------
# Run progress and cancellation
# ----------------------------------------------------------------------
#
# All three are optional in both directions, exactly as the chunked-transfer
# endpoints are: a client that never sends X-Run-Id never reaches them, and one
# that calls them against an older server gets a 404 it is told to read as "no
# progress here" rather than as an error. See RUN_PROGRESS.md.


def _run_error(exc: runs.RunError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


def _client_logs(logs: str) -> tuple:
    """The log audiences a CLIENT may ask for: `user`, or none.

    `admin` is not among them whatever is asked: an operator's line is read on
    the operator page, redacted, and the API token is held by every
    workstation. Absent is none, which is what keeps a client released before
    log lines existed from rendering one as progress.
    """
    wanted = {part.strip() for part in (logs or "").split(",")}
    return (runs.LOG_AUDIENCE_USER,) if runs.LOG_AUDIENCE_USER in wanted else ()


@app.get("/runs/{run_id}", dependencies=[Depends(verify_token)])
async def run_snapshot(run_id: str, logs: str = "") -> dict:
    """Where a run stands, and every event it has written.

    For tests, for debugging, and for a client that cannot hold a streaming
    connection open. The Slicer client watches the event stream instead, so
    nothing here is on any hot path. `?logs=user` adds the log lines a tool
    wrote for the person who started the run.
    """
    try:
        return await anyio.to_thread.run_sync(
            functools.partial(runs.snapshot, run_id, logs=_client_logs(logs)))
    except runs.RunError as exc:
        raise _run_error(exc)


@app.get("/runs/{run_id}/events", dependencies=[Depends(verify_token)])
async def run_events(run_id: str, logs: str = "") -> StreamingResponse:
    """Server-Sent Events, oldest first, INCLUDING what was written before this
    watcher connected.

    That last part is the whole design: the client opens this from a second
    thread the moment it has minted the id, while the first thread is still
    blocked inside the POST, and the two cannot be ordered. A watcher that
    attaches late must never be behind, so the stream starts from the beginning
    of the file rather than from the moment of connection.

    The file is read in a worker thread, never on the event loop: this handler
    lives for the whole run -- hours, for a cohort -- and a blocking read here
    would stall every other request for as long as it took.

    `?logs=user` interleaves the log lines a tool wrote for the requester, as
    events with `"kind": "log"`. Without it there are none, which is what a
    client released before they existed needs: it would draw one as progress.
    """
    try:
        directory = await anyio.to_thread.run_sync(runs.run_directory, run_id)
    except runs.RunError as exc:
        raise _run_error(exc)
    audiences = _client_logs(logs)

    async def frames():
        reader = runs.EventReader(directory, logs=audiences)
        while True:
            for event in await anyio.to_thread.run_sync(reader.read):
                yield f"data: {json.dumps(event)}\n\n".encode("utf-8")
            if reader.finished:
                # A terminal event was delivered, or the directory went away
                # with the request that owned it. Either way the run is over
                # and holding the connection open would only look like one
                # still going.
                return
            await anyio.sleep(settings.RUN_EVENT_POLL_SECONDS)

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            # nginx buffers a proxied response by default, which for a stream
            # means the client sees nothing until the run ends -- the exact
            # failure this endpoint exists to prevent.
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/runs/{run_id}", dependencies=[Depends(verify_token)],
            status_code=status.HTTP_204_NO_CONTENT)
async def cancel_run(run_id: str) -> Response:
    """Stop a run. Idempotent; `404` for an id this server never had.

    Two things happen, because neither covers the whole window. The marker is
    written first and is what a run with no process yet -- staging its inputs,
    or queued for the card -- notices on its next poll. Then, if a process
    group has been recorded, it is signalled: that is what actually stops a
    two-hour nnUNet, and it works from a uvicorn worker that holds no handle on
    that process because the group id travelled through the run directory
    rather than through this process's memory.
    """
    try:
        pgid = await anyio.to_thread.run_sync(runs.request_cancel, run_id)
    except runs.RunError as exc:
        raise _run_error(exc)
    if pgid is not None:
        await anyio.to_thread.run_sync(dispatch.kill_process_group, pgid)
    logger.info("endpoint=/runs status=204 action=cancel signalled=%s", pgid is not None)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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



# Characters kept from a client-supplied filename. Everything else is dropped
# rather than escaped: this string becomes a path component on the server's
# disk, and a whitelist is the only form of that decision which cannot be
# reasoned around.
_SAFE_STEM = re.compile(r"[^A-Za-z0-9_.-]+")
# How much of the name survives. Long enough for a real patient identifier,
# short enough that it cannot push the path past a filesystem limit.
_MAX_STEM = 64


def _safe_stem(filename: str, extension: str) -> str:
    """The patient-identifying part of an upload's name, made safe to write.

    Sanitized, NOT discarded, and the distinction is clinical. Naming the temp
    file after the form field alone -- which is what this replaces -- meant every
    scan in a batch arrived as `scans.nii.gz`, so every tool that names its
    outputs after its input handed back `scans_Pred_MAND.nii.gz` for every
    patient. Identity survived only in the order the requests were made. Measured
    on three unrelated tools (AMASSS, Batch_Dental_Seg, Crown_Seg) before being
    fixed here, once, where the name is chosen.

    The real risk was never the name's presence, it is writing an unsanitized
    client string to disk. So:

    - the declared extension is removed first, not `Path.stem`, which only strips
      the last suffix and would leave `.nii` on a `.nii.gz`;
    - everything outside `[A-Za-z0-9_.-]` is dropped, which removes separators
      and control characters outright;
    - leading dots go, so nothing becomes a hidden file or a relative path;
    - a result that is empty, or that is all dots (`.`, `..`), returns "" and the
      caller falls back to the field name alone. Traversal cannot survive a
      whitelist that excludes `/`, but `..` is refused explicitly because it is
      the one leftover that is still a meaningful path;
    - a run of dots collapses to one, so `..` cannot appear anywhere in the
      result. `..` between two underscores cannot traverse -- the whitelist
      already removed every separator -- but a dot run is never part of a real
      patient name, and leaving it forces anyone auditing this to reason about
      adjacency instead of reading one flat rule. Found by the URL-encoded
      payload `..%2f..%2fpasswd`, whose percent signs became underscores and
      left the dots behind.
    """
    name = os.path.basename(filename or "")
    if extension and name.lower().endswith(extension.lower()):
        name = name[: -len(extension)]
    cleaned = _SAFE_STEM.sub("_", name)
    cleaned = re.sub(r"\.{2,}", ".", cleaned).strip("._")
    if not cleaned or set(cleaned) <= {"."}:
        return ""
    return cleaned[:_MAX_STEM]


def _staged_input_path(work_dir: str, field_name: str, filename: str, extension: str) -> str:
    """Where an uploaded file lands: `<work dir>/<argument>/<the file's name>`.

    The argument gets a DIRECTORY, not a filename prefix. The prefix said the
    same thing -- which argument a file belongs to -- but it said it inside the
    NAME, and a tool that pairs two inputs by patient reads that name:
    `A1_T1.nii.gz` sent as `files` and `A1_T1_transform.mat` sent as
    `transforms` became patients `files_A1` and `transforms_A1`, and AutoMatrix
    answered "1 file(s) had no transform, 1 transform(s) had no file" to a
    request that was completely correct. GreedyReg's t1/t2 and AutoCrop3D's
    scans/roi break the same way. Zipping each argument hid it, because the
    prefix then landed on the archive rather than on the files inside.

    A directory keeps the argument readable, keeps the patient's own name
    intact, and cannot collide between two arguments. `_safe_stem` still
    sanitises the name; an unusable one falls back to the argument's own.
    """
    stem = _safe_stem(filename or "", extension)
    directory = os.path.join(work_dir, field_name)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"{stem or field_name}{extension}")


_GZIP_PROBE_CHUNK = 1 << 20


def _reject_a_truncated_gzip(path: str, field_name: str, filename: str) -> None:
    """Refuse an empty upload, and a `.gz` that never reaches its end marker.

    ITK's NIfTI reader does not report a truncated gzip. It reads the header,
    believes the dimensions it declares, and ZERO-FILLS whatever the stream
    could not supply -- measured on a scan cut to 4 kB: `sitk.ReadImage`
    returned size (512, 512, 365), 95.7M voxels, of which **10 394 were
    non-zero**, and raised nothing.

    So a transfer that dropped halfway is not an error anywhere. It is a
    successful run on a volume that is 99.99% empty: AMASSS segmented one in
    28.6 s, Batch_Dental_Seg in 8.9 s, AutoCrop3D in 0.3 s, ASO in 0.7 s, and
    ALI spent **21 minutes** of GPU on it. Every one answered 200.

    Checked here rather than in each tool because it is the same check for all
    of them, it costs no dependency (`gzip` is standard library, and the API
    venv deliberately has nothing heavier), and it belongs where the bytes
    arrive. It reads the file once; on a 94 MB scan that is a fraction of what
    receiving it cost.
    """
    # An empty upload first, and for every extension: zero bytes is not a
    # volume, a mesh or a table, and `gzip.open` reads an empty file without
    # complaining -- which is how a zero-byte `.nii.gz` reached ASO and came
    # back as a successful orientation report.
    try:
        if os.path.getsize(path) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"'{filename or field_name}' is empty. An upload of zero "
                    f"bytes is refused here rather than read as an empty "
                    f"volume. Send it again."
                ),
            )
    except OSError:
        pass

    if not path.lower().endswith(".gz"):
        return
    try:
        with gzip.open(path, "rb") as handle:
            while handle.read(_GZIP_PROBE_CHUNK):
                pass
    except (EOFError, OSError) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"'{filename or field_name}' is not a complete gzip file "
                f"({type(error).__name__}). A transfer that stopped early is "
                f"read by the imaging libraries as a volume of zeros, so it is "
                f"refused here rather than segmented. Send it again."
            ),
        )


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


def _reject_upload_for_unknown_argument(tool, field_name: str) -> None:
    """An upload naming an argument the tool does not declare is a 422 saying
    exactly that.

    It has to be checked BEFORE the extension is, or the caller is told
    something untrue: `_expected_extensions` falls back to the global
    ALLOWED_EXTENSIONS for an argument it cannot find, so a client sending
    `scan=@x.nii.gz` to a tool whose argument is `scans` was answered
    "Unsupported file extension for 'scan'. Allowed: ('.nii', '.nii.gz')" --
    about a file whose extension is exactly right. Found by running
    Example_Tool through the API with the wrong field name.

    The message matches `Tool.validate`'s, which is what an unknown SCALAR
    argument already gets, so a typo answers the same way whether or not the
    value happened to arrive as a file.
    """
    if field_name not in tool.arguments:
        known = ", ".join(sorted(tool.arguments))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unexpected argument(s) for tool '{tool.name}': {field_name}. "
                f"This tool takes: {known}."
            ),
        )


def _reject_upload_for_scalar(spec, field_name: str) -> None:
    """A scalar-typed argument must never arrive as a file: a server-side-only
    model (ArgSpec(type=str, server_selectable="model")) is selected by name.
    Without this check the uploaded file's temp path would silently become the
    argument's string value."""
    if spec is not None and not spec.is_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Argument '{field_name}' expects a plain value, not an uploaded file.",
        )


def _unpacks_to_a_folder(kind: str, extension: str) -> bool:
    """Does this input arrive as an archive that must be extracted first?

    Always for a "folder" argument, and for a "path" argument that received a
    .zip: no packaged tool unpacks archives any more -- each used to carry its
    own extraction, zip-bomb cap and scratch directory, which is exactly the
    duplication moving them out removed.
    """
    return kind == FOLDER_TYPE or (kind == PATH_TYPE and extension.lower() == ".zip")


async def _as_resolved_path(spec, input_path: str, extension: str, work_dir: str, field_name: str):
    """Tag an input with the declared type it actually is, unpacking an archive
    so run() only ever sees a real file or directory."""
    kind = spec.match_type(extension) if spec is not None and spec.is_file else "file"
    if not _unpacks_to_a_folder(kind, extension):
        return ResolvedPath(input_path, kind)
    extracted = await anyio.to_thread.run_sync(
        functools.partial(_extract_folder_argument, spec, input_path, work_dir, field_name)
    )
    return ResolvedPath(extracted, FOLDER_TYPE)


def _registered_run(request: Request, tool_name: str) -> Optional[str]:
    """Claim the run id the client sent, or None when it sent none.

    Called as the FIRST thing the handler does, before `await request.form()`,
    and the ordering is the point. Parsing the form IS the multipart upload --
    minutes of it for a CBCT, which is precisely the stretch a client today can
    say nothing about -- and the client opens its event stream from a second
    thread the instant it has minted the id. Registering after the form was
    parsed would answer that watcher a 404, which the client is told to read as
    "an older server, stop watching": the feature would silently do nothing on
    exactly the runs it exists for. Registering first leaves a window of
    microseconds, and the client tolerates a brief 404 to close it.
    """
    raw = request.headers.get(_RUN_ID_HEADER)
    if not raw:
        return None
    # Which batch of a divided cohort this is, when the client says so. Read
    # here, with the id, so the dashboard can group the runs from the moment
    # they exist and the gate in `_tool_slot` can order them.
    batch = runs.parse_batch(request.headers.get(_BATCH_ID_HEADER),
                             request.headers.get(_BATCH_INDEX_HEADER),
                             request.headers.get(_BATCH_TOTAL_HEADER))
    try:
        return runs.register(raw, tool=tool_name, client=_client_address(request), batch=batch)
    except runs.RunError as exc:
        raise _run_error(exc)


def _client_address(request: Request) -> Optional[str]:
    """Which workstation asked for this run.

    **The peer this process actually sees, never a header read here.**
    `X-Forwarded-For` is written by the client and is trivially forged, so
    taking it would turn an attribution into a suggestion -- and attribution is
    the entire reason this field exists. Behind the TLS terminator the README
    documents, the peer is the proxy; uvicorn replaces it with the address the
    proxy reports only when that proxy is in `FORWARDED_ALLOW_IPS`, which
    docker-compose.yml sets to Docker's bridge gateways. Which proxy to believe
    is a deployment decision, so it lives there and not in this function.

    An address is not patient data, but it does identify a person's machine, so
    it travels no further than `/status` and `/admin-panel` already do: behind
    the shared token, for an operator asking "who is hammering this server".
    """
    client = request.client
    return client.host if client else None


def _failure_message(exc: BaseException) -> str:
    """What a detached run may say about its own failure.

    The same rule the response body follows: a message the tool wrote to be
    read by whoever sent the request travels, anything else is opaque. A
    traceback can name a server-side path, and this one is going into a file a
    client reads.
    """
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    if isinstance(exc, dispatch.ToolFailure):
        if _tool_error_status(exc.kinds) != 500:
            return exc.message
        return "Tool execution failed."
    if isinstance(exc, (ToolArgumentError, ToolUnavailableError)):
        return str(exc)
    return "Tool execution failed."


# What a diagnosis may carry, field by field. A location in a tool's source is
# code, not data, so it travels as written -- but only in this shape.
_ERROR_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,63}")
_SOURCE_LOCATION = re.compile(r"[A-Za-z0-9_./-]{1,200}:[0-9]{1,6} in [A-Za-z0-9_<>]{1,80}")


def _diagnosis(exc: BaseException, tool_name: Optional[str],
               run_id: Optional[str] = None) -> dict:
    """Why a run failed and where, for the OPERATOR: `{tool, chain, error_type,
    reason, where, stage, fraction, status}`, every free-text field redacted.

    The runner records where a tool raised (`runner._origin_of`) and every
    level above relays it, so a failure three calls down names the leaf, its
    line, and the last thing it said it was doing. When nothing below said
    anything -- a crash, a timeout, a 422 from validation -- what this server
    knows stands in: the tool asked for, the calls still open, the exception.
    """
    found, status_code = None, None
    seen, current = 0, exc
    while current is not None and seen < 6:
        if isinstance(current, HTTPException) and status_code is None:
            status_code = current.status_code
        if isinstance(current, dispatch.ToolFailure):
            found = current
            break
        if found is None and not isinstance(current, HTTPException):
            found = current
        current = current.__cause__ or current.__context__
        seen += 1
    found = found or exc

    origin = getattr(found, "origin", None) or {}
    chain = [name for name in (runs._clean_tool_name(entry)
                               for entry in origin.get("chain") or ()) if name]
    if not chain and run_id:
        try:
            chain = [tool_name] + runs.timeline(runs.read_events(run_id))["chain"]
        except runs.RunError:
            chain = []
    chain = [name for name in chain if name] or ([tool_name] if tool_name else [])
    error_type = origin.get("error_type") or (
        found.error_type if isinstance(found, dispatch.ToolFailure) else type(found).__name__)
    if isinstance(found, dispatch.ToolFailure):
        reason = origin.get("message") or found.message
    elif isinstance(found, HTTPException):
        reason = found.detail
    else:
        # The first line only: a ToolExecutionError carries the stderr tail
        # after it, and that belongs in the server's log, not on a page.
        reason = str(found).split("\n", 1)[0]
    diagnosis = {
        "tool": runs._clean_tool_name(origin.get("tool")) or (chain[-1] if chain else None),
        "chain": chain,
        "error_type": error_type if _ERROR_TYPE.fullmatch(str(error_type)) else "Error",
        "reason": redact.scrub(reason),
    }
    if status_code is not None:
        diagnosis["status"] = status_code
    where = origin.get("where")
    if isinstance(where, str) and _SOURCE_LOCATION.fullmatch(where):
        diagnosis["where"] = where
    if origin.get("stage"):
        diagnosis["stage"] = redact.scrub(origin["stage"])
    fraction = origin.get("fraction")
    if isinstance(fraction, (int, float)) and 0.0 <= fraction <= 1.0:
        diagnosis["fraction"] = fraction
    return diagnosis


def _described(diagnosis: dict) -> str:
    """A diagnosis as one log line: `AREG > ASO > ALI_CBCT: KeyError at ... -- why`."""
    text = "{}: {}".format(" > ".join(diagnosis.get("chain") or ["?"]),
                           diagnosis.get("error_type"))
    if diagnosis.get("where"):
        text += " at " + diagnosis["where"]
    if diagnosis.get("stage"):
        text += " during '{}'".format(diagnosis["stage"])
    if diagnosis.get("reason"):
        text += " -- " + diagnosis["reason"]
    return text


def _run_diagnosis(exc: BaseException, run_id: str, tool_name: Optional[str] = None):
    """`_diagnosis` for a registered run, or None for one that was cancelled.
    Never raises: it runs on a failure path, where a second error would hide
    the first."""
    if isinstance(exc, dispatch.RunCancelled):
        return None
    try:
        return _diagnosis(exc, tool_name or runs.meta(run_id).get("tool"), run_id)
    except Exception:  # noqa: BLE001
        logger.exception("could not diagnose a failed run")
        return None


def _collectable(response) -> dict:
    """The terminal event's payload: how to collect what the run produced."""
    if isinstance(response, JSONResponse):
        return json.loads(bytes(response.body).decode("utf-8"))
    if isinstance(response, dict):
        return dict(response)
    return {}


async def _detached_run(tool_name: str, request: Request, run_id: str) -> None:
    """The whole run, after the 202 has already gone out.

    The terminal event carries the `result_ref`, because the response that used
    to carry it was sent minutes ago. Nothing else about the run changes: the
    same staging, the same admission, the same progress events on the same
    stream the client is already watching.

    The run directory is deliberately NOT discarded here. The client has not
    read the terminal event yet -- that is the whole point of writing one -- so
    it expires the way an abandoned one does, on the idle TTL that every read
    pushes back.
    """
    cleanup = BackgroundTasks()
    token = runs.CURRENT_RUN.set(run_id)
    try:
        response = await _run_tool(tool_name, request, cleanup, detached=True)
        if runs.paused_at(run_id) is not None:
            # A stopped run has not finished, so no terminal event: one would
            # tell the client to stop watching a run it is about to resume.
            #
            # But the `paused` event `_finish_stopped_run` wrote carries no
            # result, and the response that does was built two lines ago and
            # is about to be dropped -- nothing answers a detached run here,
            # so the reference to what the checkpoint produced existed in
            # this function and nowhere else. A watcher would have sat until
            # the stream was reaped. Appended again, WITH the payload, since
            # the stream is append-only and the last event is what a reader
            # takes as the state.
            runs.append(run_id, phase=runs.PHASE_PAUSED,
                        result=_collectable(response))
            logger.info("endpoint=/run/%s paused (detached)", tool_name)
            return
        runs.finish(run_id, runs.PHASE_DONE, result=_collectable(response))
    except dispatch.RunCancelled:
        runs.finish(run_id, runs.PHASE_CANCELLED)
    except BaseException as exc:  # noqa: BLE001 - nobody is left to raise to
        diagnosis = _run_diagnosis(exc, run_id, tool_name)
        # Redacted, like every line about a failure: the exception's own text
        # is a tool's words and may name the file it died on.
        logger.warning("endpoint=/run/%s detached failure: %s", tool_name,
                       _described(diagnosis))
        runs.finish(run_id, runs.PHASE_FAILED, message=_failure_message(exc),
                    failure=diagnosis)
    finally:
        runs.CURRENT_RUN.reset(token)
        try:
            await cleanup()
        except Exception:  # noqa: BLE001 - cleanup must not outlive its own failure
            logger.exception("endpoint=/run/%s (detached cleanup)", tool_name)


async def _detach(tool_name: str, request: Request, run_id, background_tasks):
    """Accept the run, answer at once, and finish it after the response.

    Two things are refused rather than half-supported. A detached run needs a
    run id, because the event stream is the only channel it has left. And it
    needs its file inputs to have arrived through `POST /uploads`, because a
    multipart body is backed by a temporary file the framework closes when the
    response ends -- which here is before the tool has read a byte of it. The
    client already sends anything worth detaching that way.
    """
    if run_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"A detached run needs an {_RUN_ID_HEADER} header to report through.",
        )
    form = await request.form()
    if any(isinstance(value, StarletteUploadFile) for _, value in form.multi_items()):
        runs.discard(run_id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "A detached run cannot take a file in the request body. Send it "
                "through POST /uploads first and name it in __uploads__."
            ),
        )
    runs.append(run_id, runs.PHASE_RECEIVED)
    background_tasks.add_task(_detached_run, tool_name, request, run_id)
    logger.info("endpoint=/run/%s status=202 detached", tool_name)
    return JSONResponse(
        {"run_id": run_id, "status": "accepted"},
        status_code=status.HTTP_202_ACCEPTED,
        background=background_tasks,
    )


# One directory per slot, named as the slot is (`01_ALI_CBCT`). The runner
# moves each over that slot's output before answering from its memo.
_RESUME_STEP = r"[0-9]{2}_[A-Za-z0-9_-]{1,64}"
# One step, or a path through nested ones: `01_ALI_CBCT`, or `01_ASO/01_ALI_CBCT`
# for a step a CALLEE made. Bounded at four levels because the name comes over
# HTTP and becomes a directory, and a chain that deep does not exist.
_RESUME_SLOT = re.compile(r"^%s(?:/%s){0,3}$" % (_RESUME_STEP, _RESUME_STEP))


def _suffixes_of(name: str) -> set:
    """`{".json", ".mrk.json"}` for `points.mrk.json`.

    Both, because this ecosystem's extensions are compound half the time --
    `.nii.gz`, `.nrrd.gz`, `.mrk.json` -- and the single form alone would
    tell a reader to send a `.json` when what came out was a `.mrk.json`.
    Derived from the real file rather than from a table, so a tool that
    starts writing something else needs no edit here.
    """
    parts = name.lower().split(".")
    found = set()
    if len(parts) >= 2:
        found.add("." + parts[-1])
    if len(parts) >= 3:
        found.add("." + ".".join(parts[-2:]))
    return found


def _checked_correction(step_dir: str, slot: str, filename: str) -> None:
    """Refuse a correction that is not the kind of thing that went out.

    NOT `settings.ALLOWED_EXTENSIONS`: that is the whitelist for a tool's
    INPUTS, and a correction replaces a step's OUTPUT -- landmarks, a
    labelled mesh, a transform. On this deployment the input list is
    `('.nii', '.nii.gz')`, so reusing it refused the very `.mrk.json` the
    reader had just been handed.

    What it is compared against instead is that step's own output, which is
    on disk a few directories away: a reader sends back what they were
    given. A `.zip` is always allowed -- a folder has no other way to
    travel.

    `step_dir` is the step's own directory, resolved by the caller: a step
    of a CALLEE is not under `<job>/sup` but under its own caller's, and
    building the path here would only work for the first level.
    """
    produced = os.path.join(step_dir, dispatch.JOB_OUTPUT_DIRNAME)
    allowed = {".zip"}
    for _root, _directories, names in os.walk(produced):
        for name in names:
            allowed.update(_suffixes_of(name))
    if _matched_extension(filename, tuple(sorted(allowed))) is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(f"'{filename}' is not what step '{slot}' produced. Send one "
                    f"of {tuple(sorted(allowed))}, or a .zip of the folder."),
        )


def _located_step(job_dir: str, slot: str):
    """Where a step's own output is, and where its correction is staged.

    A chain nests: the root supervisor works in `<job>`, and the one inside
    `01_Mid` works in `<job>/sup/01_Mid`. Each stages what came back for its
    OWN calls in `<its job dir>/resume/<step>`, so a correction for
    `01_Mid/01_Leaf` belongs in `<job>/sup/01_Mid/resume/01_Leaf` -- which is
    exactly where the supervisor that re-runs Mid will look for it.

    At module scope rather than inside the staging, because the rewind reads
    the same two places to work out which cases came back.

    Returns `(step directory, staging directory)`, or None for a path no step
    of this run answers to.
    """
    here = job_dir
    steps = slot.split("/")
    for step in steps[:-1]:
        here = os.path.join(here, dispatch.SUP_DIRNAME, step)
        if not os.path.isdir(here):
            return None
    produced = os.path.join(here, dispatch.SUP_DIRNAME, steps[-1])
    if not os.path.isdir(produced):
        return None
    return produced, os.path.join(here, dispatch.RESUME_DIRNAME, steps[-1])


def _staged_files(job_dir: str, slot: str) -> list:
    """Every file a reader sent back for `slot`, at any depth."""
    located = _located_step(job_dir, slot)
    if located is None:
        return []
    _produced, staged = located
    found = []
    for root, _directories, names in os.walk(staged):
        found.extend(os.path.join(root, name) for name in sorted(names))
    return found


def _stage_corrections(tool, job_dir: str, form) -> list:
    """Put what a reader sends back where the resumed run will read it.

    The files came off THEIR disk: a resume that trusted the server's copy
    would carry on with exactly the data they stopped to reject. A `.zip` is
    unpacked -- a folder has no other way to travel -- and anything else is
    written as the single file it is.

    The slot name is matched against a pattern BEFORE any path is built from
    it, the same discipline every id in `wire/` follows: it arrives over HTTP
    and it becomes a directory.
    """
    def _ran() -> list:
        """Every step of this run, nested ones written as a path."""
        found = []

        def walk(directory: str, prefix: str) -> None:
            root = os.path.join(directory, dispatch.SUP_DIRNAME)
            if not os.path.isdir(root):
                return
            for name in sorted(os.listdir(root)):
                path = prefix + name
                found.append(path)
                walk(os.path.join(root, name), path + "/")

        walk(job_dir, "")
        return found

    staged = []
    for slot, value in form.multi_items():
        if not isinstance(value, StarletteUploadFile):
            continue
        if not _RESUME_SLOT.match(slot):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(f"'{slot}' is not a step of that run. Name a correction "
                        "after the folder it came back in, such as 01_ALI_CBCT."),
            )
        located = _located_step(job_dir, slot)
        if located is None:
            # A correction for a step that never ran is a typo, and a typo
            # silently accepted is a resume the reader believes carries their
            # work and does not.
            ran = _ran()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(f"That run has no step '{slot}'. It ran: "
                        f"{', '.join(ran) or 'nothing'}."),
            )
        produced, destination = located
        _checked_correction(produced, slot, value.filename or "")

        shutil.rmtree(destination, ignore_errors=True)
        os.makedirs(destination, exist_ok=True)
        name = os.path.basename(value.filename or slot)
        landed = os.path.join(destination, name)
        with open(landed, "wb") as handle:
            shutil.copyfileobj(value.file, handle)
        if landed.lower().endswith(".zip"):
            # Untrusted: `extract_zip` refuses zip slip, symlink members and
            # anything over MAX_EXTRACTED_MB before a byte is written.
            file_utils.extract_zip(landed, destination)
            os.remove(landed)
        if not any(os.scandir(destination)):
            # An empty replacement would put NOTHING where the step's output
            # was, and the chain would carry on with an empty folder rather
            # than with the reader's correction. Refused: the reader meant to
            # send something.
            shutil.rmtree(destination, ignore_errors=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"The correction sent for '{slot}' is empty.",
            )
        staged.append(slot)
    return staged


async def _tracked_run(run_id: str, label: str, background_tasks: BackgroundTasks,
                       call):
    """Run `call()` as `run_id`, writing the terminal event whichever way it ends.

    Shared by `POST /run/{tool}` and `POST /runs/{id}/resume`, because a
    resume IS a run: it queues for the machine, it can be cancelled, it can
    fail, and it can stop again at a second checkpoint. Resume reached this
    file without it and wrote no terminal event at all, so every resumed run
    stayed `running / packaging` in the registry for ever -- visible on the
    status page as work that finished hours ago and is somehow still going,
    and never delivering a terminal event to a watcher.

    A run that is PAUSED when `call` returns is deliberately left alone:
    it has not finished, and its directory is what a resume looks the work
    up through.
    """
    token = runs.CURRENT_RUN.set(run_id)
    try:
        response = await call()
    except dispatch.RunCancelled:
        runs.finish(run_id, runs.PHASE_CANCELLED)
        runs.discard(run_id)
        logger.info("endpoint=%s status=%d", label, CLIENT_CLOSED_REQUEST)
        raise HTTPException(
            status_code=CLIENT_CLOSED_REQUEST, detail="Run cancelled by the client."
        )
    except BaseException as exc:
        # Every failure path, the 404 for an unknown tool included -- the tool
        # is resolved after the run is registered, so that one now has a
        # directory to clean up like any other.
        runs.finish(run_id, runs.PHASE_FAILED,
                    failure=_run_diagnosis(exc, run_id))
        runs.discard(run_id)
        raise
    finally:
        runs.CURRENT_RUN.reset(token)

    if runs.paused_at(run_id) is not None:
        # The run STOPPED where it was asked to. Not finished, so no terminal
        # event; and above all not discarded -- the run directory is what
        # `POST /runs/{id}/resume` looks the work up through, and taking it
        # down here is what made the whole feature unreachable the first time
        # it was tried end to end. It lives on the idle TTL, like an
        # abandoned transfer, and every read pushes that back.
        logger.info("endpoint=%s paused", label)
        return response

    runs.finish(run_id, runs.PHASE_DONE)
    # Queued rather than done now, so the directory survives until the response
    # has finished streaming -- which gives a watcher the whole download to
    # collect the terminal event. Waiting out the TTL instead is not an option:
    # a progress message is written by a tool and can name a file.
    background_tasks.add_task(runs.discard, run_id)
    return response


def _tool_of(job_dir: str):
    """The tool a stopped run was started for, from the job file it was given.

    Read back rather than passed in: the request is on disk and is the one
    thing about a paused run that cannot have drifted.
    """
    try:
        with open(os.path.join(job_dir, dispatch.JOB_FILE), encoding="utf-8") as handle:
            name = json.load(handle)["tool"]
    except (OSError, ValueError, KeyError):
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="The work this run stopped in is no longer on the server.",
        )
    try:
        return get_tool(name)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


# The form fields a client names its marked cases in. One per case rather than
# one joined string: a patient identifier is whatever a clinic calls its
# folders, so any separator picked here is one that will turn up inside a name
# and split it in half.
_CASE_FIELD_PREFIX = "case_"


def _declared_cases(form) -> list:
    """The cases a reader asked to have done again, as the client named them.

    Order is the form's, duplicates dropped. Nothing is validated here: a name
    that is not a case of this run reaches `dispatch.narrow_to_cases`, which
    refuses to narrow at all rather than narrowing on half an agreement -- and
    the replay is then the whole cohort, which is the direction everything on
    this path fails in.
    """
    named = []
    for field, value in form.multi_items():
        if not str(field).startswith(_CASE_FIELD_PREFIX):
            continue
        case = (value or "").strip() if isinstance(value, str) else ""
        if case and case not in named:
            named.append(case)
    return named


@app.post("/runs/{run_id}/rewind", dependencies=[Depends(verify_token), Depends(maintenance.require_accepting)])
async def rewind_run(run_id: str, request: Request,
                     background_tasks: BackgroundTasks):
    """Send a stopped run BACK to a checkpoint it already went past.

    A reader looking at a bad orientation cannot fix it where they are: the
    orientation was computed from landmarks decided two steps back. So they
    ask to return to the last stop where something can actually be changed,
    and this arms that checkpoint again.

    Nothing is re-run to get there and no memo is dropped. The step's result
    is on disk, which is precisely what the reader wants to look at; the run
    re-enters its tool, every call answers from what it recorded, and the
    checkpoint fires again the moment that step is reached. A GPU pass to
    reproduce a file that is already there would be paid for nothing.

    `to` is a step the run was stopped at, named as it was published --
    `ALI_CBCT`, or `ASO/ALI_CBCT` for one inside a callee.
    """
    paused = await anyio.to_thread.run_sync(runs.paused_at, run_id)
    if paused is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(f"Run '{run_id}' is not stopped at a checkpoint. A run that "
                    "finished, failed or expired cannot be sent back."),
        )
    form = await request.form()
    target = (form.get("to") or "").strip()
    if not target:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Name the step to go back to, as 'to'.",
        )
    job_dir = paused["job_dir"]
    staged = _stage_corrections(_tool_of(job_dir), job_dir, form)
    # What the reader SAID, before anything is deduced. A bad registration is
    # marked and not corrected -- the landmarks that caused it are two steps
    # back, so there is no file to send from where the reader is standing --
    # and without this the only evidence was the corrections, so a reader who
    # marked three of forty and edited nothing replayed all forty.
    marked = _declared_cases(form)
    # Then which cases the reader sent back, read out of the step's own report
    # -- the tool stated which of its files belong to which case, so nothing is
    # deduced from a file name here. A step that reported nothing narrows
    # nothing, and the replay is the whole cohort: slower, never wrong.
    for slot in staged:
        located = _located_step(job_dir, slot)
        if located is None:
            continue
        step_dir, _staging = located
        report = reports.read(os.path.join(step_dir, dispatch.JOB_OUTPUT_DIRNAME))
        marked.extend(name for name in reports.cases_of(
            report, _staged_files(job_dir, slot)) if name not in marked)
    # The union, deliberately: a reader who corrected one patient's landmarks
    # AND marked another wants both done again. Taking only one source would
    # silently drop half of what they said.
    if marked:
        await anyio.to_thread.run_sync(
            dispatch.narrow_to_cases, job_dir, _tool_of(job_dir), marked)

    if not await anyio.to_thread.run_sync(runner.rewind_to, job_dir, target):
        # Asking to return somewhere the run has not been. Refused rather
        # than ignored: a silent no-op leaves a reader waiting at a
        # checkpoint that will never come.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(f"This run never stopped at '{target}', so there is "
                    "nothing to go back to."),
        )
    return await resume_run(run_id, request, background_tasks,
                            already_staged=True)


@app.post("/runs/{run_id}/resume", dependencies=[Depends(verify_token), Depends(maintenance.require_accepting)])
async def resume_run(run_id: str, request: Request,
                     background_tasks: BackgroundTasks,
                     already_staged: bool = False):
    """Tell a run that stopped at a checkpoint to carry on.

    The request may carry corrections: one file field per step, named after
    the folder that step came back in. They are staged and the run re-enters
    its tool from the top -- nothing preserves a Python stack across a
    process that exited -- with every call it already made answering from
    what it recorded instead of running again.

    It queues for the machine like any other run, because it IS one: the work
    left to do is real work, and a resumed cohort must not jump a clinician
    who has been waiting.
    """
    paused = await anyio.to_thread.run_sync(runs.paused_at, run_id)
    if paused is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(f"Run '{run_id}' is not stopped at a checkpoint. A run that "
                    "finished, failed or expired cannot be carried on."),
        )
    job_dir = paused["job_dir"]
    try:
        with open(os.path.join(job_dir, dispatch.JOB_FILE), encoding="utf-8") as handle:
            tool_name = json.load(handle)["tool"]
    except (OSError, ValueError, KeyError):
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="The work this run stopped in is no longer on the server.",
        )

    try:
        tool = get_tool(tool_name)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if not already_staged:
        # A rewind stages before it narrows, and the upload's handles are
        # read once: staging the same form twice would find every file empty
        # and refuse the correction the reader just made.
        _stage_corrections(tool, job_dir, await request.form())
    # Cleared BEFORE the run, not after: it is standing on this checkpoint
    # right up until it moves, and a second resume arriving while the first
    # is running must not be offered the same directory.
    await anyio.to_thread.run_sync(runs.clear_pause, run_id)

    async def carry_on():
        logger.info("endpoint=/runs/resume tool=%s after=%s",
                    tool_name, paused.get("stopped_after"))
        return await _run_tool(tool_name, request, background_tasks,
                               resume_from=job_dir)

    return await _tracked_run(run_id, "/runs/resume", background_tasks, carry_on)


@app.post("/run/{tool_name}", dependencies=[Depends(verify_token), Depends(maintenance.require_accepting)])
async def run_tool(tool_name: str, request: Request, background_tasks: BackgroundTasks):
    """The run, with its progress recorded when the client asked for it.

    Everything the run actually does is in `_run_tool`; this is only the shell
    that owns the run directory -- registering it before anything is read,
    writing the terminal event whichever way the run ends, and taking the
    directory down afterwards. A client that sends no `X-Run-Id` takes the
    first branch and reaches byte-for-byte the behaviour it always had.
    """
    run_id = _registered_run(request, tool_name)
    if request.headers.get(_RUN_DELIVERY_HEADER, "").lower() == _RUN_DETACHED:
        return await _detach(tool_name, request, run_id, background_tasks)
    if run_id is None:
        return await _run_tool(tool_name, request, background_tasks)

    # The run id is read by dispatch in the worker thread anyio copies this
    # context into. It is request scope, not tool input, so it travels the way
    # file_utils tracks scratch dirs rather than through Tool.invoke's
    # signature -- which every tool and both dispatch paths agree on.
    async def start():
        runs.emit(runs.PHASE_RECEIVED)
        return await _run_tool(tool_name, request, background_tasks)

    return await _tracked_run(run_id, f"/run/{tool_name}", background_tasks, start)


async def _run_tool(tool_name: str, request: Request, background_tasks: BackgroundTasks,
                    detached: bool = False, resume_from: Optional[str] = None):
    """`resume_from` is the job directory of a run that STOPPED at a
    checkpoint. Everything after the tool has run is identical -- the same
    packing, the same delivery, the same cleanup -- which is the whole reason
    a resume comes through here rather than through a second endpoint that
    would have to learn all of it again. What it skips is the front half:
    there is no form to read and no argument to validate, because a resume is
    the same request carrying on and its inputs are already staged.
    """
    start_time = time.monotonic()

    try:
        tool = get_tool(tool_name)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    if resume_from:
        return await _finish_run(
            tool, tool_name, start_time, background_tasks, detached,
            # Zero, not None: a resume carries no input bytes -- its inputs
            # were staged by the run it is picking up -- and the log line
            # wants a number.
            size=0,
            work_dir=None, scratch_dirs=file_utils.track_scratch_dirs(),
            result=await _run_in_slot(
                functools.partial(dispatch.dispatch, tool, {}, resume_from=resume_from)),
        )

    # Generic argument collection: whatever scalar fields and/or files the
    # caller sends, whichever tool it targets. Each uploaded file is matched to
    # the tool's argument of the same name; the tool's own schema (validated in
    # tool.invoke) decides what is actually accepted.
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

    # A facade is a name standing for a choice between tools. Resolved HERE,
    # before anything else looks at `tool`: from this line on the request is an
    # ordinary run of an ordinary tool, with its own validation, timeout, GPU
    # slot and job directory. Nothing downstream knows a facade was involved,
    # which is what keeps this from being a second execution path.
    if isinstance(tool, FacadeTool):
        chosen = args.pop(facade.MODE_ARGUMENT, None)
        if not chosen:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Missing required argument '{}' for '{}'. Choose one of: {}.".format(
                    facade.MODE_ARGUMENT, tool.name, ", ".join(sorted(tool.targets))),
            )
        try:
            target_name = tool.target_for(str(chosen))
        except ToolArgumentError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
        logger.info("facade=%s mode=%s -> tool=%s", tool.name, chosen, target_name)
        # The name follows the tool: every later lookup -- this deployment's
        # server_selectable entries, its upload limit, the log line -- is about
        # what actually runs.
        tool_name = target_name
        tool = get_tool(target_name)

        # Arguments the chosen mode does not declare. The facade publishes the
        # union of its targets', so a panel that was switched from one mode to
        # another legitimately still holds the other's fields; forwarding them
        # would be an "unexpected argument" 422 for something the caller never
        # typed. Dropped rather than refused, and only ever the ones the FACADE
        # published: anything else is still an unexpected argument.
        for extra in [name for name in args if name not in tool.arguments]:
            args.pop(extra)
        for extra in [name for name in uploaded_files if name not in tool.arguments]:
            uploaded_files.pop(extra)

    # An argument declared with ArgSpec(server_selectable=...) can be sent as a
    # plain form value (the file name) instead of an upload, resolved below
    # into a path already on the server (see data_store.py). Pulled out of
    # `args` before the upload loop so a genuine upload for the same field name
    # is never mistaken for one.
    server_file_args: dict = {}
    for field_name in list(args):
        spec = tool.arguments.get(field_name)
        if spec is not None and spec.server_selectable:
            server_file_args[field_name] = args.pop(field_name)


    work_dir = None
    input_paths = []
    resolved_files = []
    size = 0
    upload_limit_mb = _upload_limit_mb(tool)
    upload_limit_bytes = upload_limit_mb * 1024 * 1024

    if uploaded_files or upload_references:
        work_dir = tempfile.mkdtemp(dir=settings.TEMP_DIR)

    # For the staging events below. A position in the batch, never a file name:
    # the message travels to a panel and is written to disk, and an input's name
    # is the patient's.
    staged_total = len(uploaded_files) + len(upload_references)
    staged = 0

    try:
        for field_name, upload in uploaded_files.items():
            staged += 1
            runs.emit(runs.PHASE_STAGING, message=f"input {staged} of {staged_total}")
            _reject_upload_for_unknown_argument(tool, field_name)
            spec = tool.arguments.get(field_name)
            _reject_upload_for_scalar(spec, field_name)
            extension = _checked_extension(tool, field_name, upload.filename or "")
            input_path = _staged_input_path(work_dir, field_name,
                                            upload.filename or "", extension)
            try:
                size += await _stream_to_disk(upload, input_path, upload_limit_bytes)
            except _UploadTooLargeError:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds the {upload_limit_mb} MB limit.",
                )
            _reject_a_truncated_gzip(input_path, field_name, upload.filename or "")
            input_paths.append(input_path)
            # An argument can accept several types (e.g. ("csv_file",
            # "folder")): decide here which one this upload is and tag the path
            # with it. A "folder" arrives zipped and is unpacked now.
            args[field_name] = await _as_resolved_path(
                spec, input_path, extension, work_dir, field_name
            )

        # Same treatment, for the inputs whose bytes are already on disk: the
        # session's blob is RENAMED into the work dir rather than copied, so a
        # chunked upload costs no extra pass over the file at all.
        for field_name, upload_id in upload_references.items():
            staged += 1
            runs.emit(runs.PHASE_STAGING, message=f"input {staged} of {staged_total}")
            _reject_upload_for_unknown_argument(tool, field_name)
            spec = tool.arguments.get(field_name)
            _reject_upload_for_scalar(spec, field_name)
            try:
                session = await anyio.to_thread.run_sync(transfer.get_upload, upload_id)
                extension = _checked_extension(tool, field_name, session.filename)
                # The tool is only known here, so this is where a per-tool
                # limit lower than the global one is enforced -- before the
                # blob is claimed, never after.
                if session.size > upload_limit_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds the {upload_limit_mb} MB limit.",
                    )
                # Staged exactly as the multipart branch stages it, and for
                # the same clinical reason -- see `_staged_input_path`. The
                # chunked route is the one a client takes for any file large
                # enough to be worth splitting, which is every CBCT, so in
                # production it is the route that matters.
                input_path = _staged_input_path(work_dir, field_name,
                                                session.filename, extension)
                await anyio.to_thread.run_sync(transfer.claim_upload, upload_id, input_path)
            except transfer.TransferError as exc:
                raise _transfer_error(exc)
            # The route every CBCT takes, so the route where a dropped transfer
            # actually happens. Each PART is checksummed, but the parts only
            # tile what the client SENT -- a client that stopped early sends a
            # complete set of parts for an incomplete file.
            _reject_a_truncated_gzip(input_path, field_name, session.filename)
            size += session.size
            input_paths.append(input_path)
            args[field_name] = await _as_resolved_path(
                spec, input_path, extension, work_dir, field_name
            )
    except HTTPException:
        # Nothing is queued for cleanup yet and no response will stream, so the
        # work dir goes now. So do the unclaimed sessions: the reaper would get
        # them eventually, but that is a TTL's worth of confidential imaging
        # sitting on disk.
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
        for upload_id in upload_references.values():
            await anyio.to_thread.run_sync(transfer.discard_upload, upload_id)
        raise

    for field_name, filename in server_file_args.items():
        spec = tool.arguments[field_name]
        resolver = data_store.resolve_model if spec.server_selectable == "model" else data_store.resolve_testfile
        try:
            # Not tool.name: the packaged tools are lowercase while the data
            # staged under DATA/ is not, so deployment.toml maps the two.
            # The scope travels with the ARGUMENT, never with the name the
            # client sent: the name stays bare, so the traversal defence in
            # data_store._resolve is untouched.
            scope = getattr(spec, "selectable_scope", None) or ""
            # Passed only when there IS one: a DataStore is an extension point
            # (see the abstract base), and a backend written before scopes
            # existed must keep serving a deployment that uses none. One that
            # is handed a scope it cannot honour fails loudly here rather than
            # quietly serving the wrong folder.
            resolved = resolver(
                deployment_config.data_slug(tool.name), filename,
                *( (scope,) if scope else () ),
            )
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
        if not os.path.isdir(path) and _unpacks_to_a_folder(
            kind, _extract_extension(os.path.basename(path))
        ):
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

    # What this request asked for, in the only terms that may be shown to
    # whoever holds the shared token: how many inputs and how big, and WHICH
    # arguments were named -- never what they were set to. An argument's value
    # is a path, and a path is a patient's file name.
    # Through the ContextVar for the same reason `runs.emit` is: a request that
    # sent no `X-Run-Id` has no run to record against, and reading it here keeps
    # the two paths from forking.
    telemetry.record_run_inputs(
        runs.CURRENT_RUN.get(), staged_total, size, args.keys(),
        detail=telemetry.describe_inputs(args, {
            name: {"server_selectable": spec.server_selectable}
            for name, spec in tool.arguments.items()
        }))

    # Anything the tool creates through file_utils.make_scratch_dir() lands
    # here, so it can be removed even if run() raises before returning a path.
    scratch_dirs = file_utils.track_scratch_dirs()

    try:
        # Run the tool in a worker thread, NOT on the event loop: tool.invoke
        # is synchronous CPU-bound work and would otherwise freeze the whole
        # server -- even /health -- for its entire duration. Concurrency is
        # bounded by MAX_CONCURRENT_TOOLS and safe: tools are stateless
        # (everything arrives via args), each request gets its own work_dir,
        # and DATA_DIR is read-only.
        result = await _run_in_slot(functools.partial(tool.invoke, args))
    except ToolArgumentError as exc:
        _discard(work_dir, scratch_dirs)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))
    except dispatch.ToolFailure as exc:
        # The tool raised and named its exception class. There is no shared
        # exception type to isinstance-check -- there is no shared package --
        # so the NAME decides, and only the names that mean "the caller can fix
        # this" let their message through.
        code = _tool_error_status(exc.kinds)
        logger.warning("endpoint=/run/%s status=%d error=%s", tool_name, code, exc.error_type)
        if code == 500:
            # The ONLY place this exists. A 4xx carries its message to the
            # caller, but a 500 answers "Tool execution failed." and the job
            # directory holding stderr.log is discarded on the next line, so
            # without this the tool's traceback is gone -- which is exactly
            # what makes a failing tool undiagnosable from the outside.
            # Server-side only, and redacted: the message is the tool's own
            # words and may name the file it died on (see redact.py). Where it
            # broke and why survive that; the name does not.
            logger.error("endpoint=/run/%s failure: %s", tool_name,
                         _described(_diagnosis(exc, tool_name)))
        _discard(work_dir, scratch_dirs)
        raise HTTPException(
            status_code=code,
            detail=exc.message if code != 500 else "Tool execution failed.",
        )
    except ToolUnavailableError as exc:
        # The request is fine; this deployment cannot serve it (a dependency the
        # image does not carry). 501 rather than 500: nothing the caller changes
        # will help, and the reason names a missing package, never a path.
        logger.warning("endpoint=/run/%s status=501", tool_name)
        _discard(work_dir, scratch_dirs)
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc))
    except dispatch.RunCancelled:
        # The client withdrew this run. Cleaned up exactly like a failure --
        # nothing will be streamed and the inputs are confidential -- but
        # re-raised unchanged, because only the caller of this function knows
        # the run id and so only it can record how the run ended.
        _discard(work_dir, scratch_dirs)
        raise
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

    return await _finish_run(
        tool, tool_name, start_time, background_tasks, detached, work_dir,
        scratch_dirs, result, size,
        wants_reference=(
            request.headers.get(_RESULT_DELIVERY_HEADER, "").lower()
            == _DELIVER_BY_REFERENCE
        ),
    )


async def _finish_stopped_run(tool, tool_name: str, start_time: float,
                              background_tasks: BackgroundTasks, work_dir,
                              scratch_dirs, record: dict, size):
    """Deliver a run that stopped at a checkpoint.

    What goes back is the record -- which checkpoint, and what was produced
    -- with the files themselves parked as a result reference. The job
    directory stays where it is: it is what a resume reads, and packing is a
    copy out of it rather than a move.
    """
    outputs = file_utils.output_paths(record)
    reference = None
    # The archive is built in a directory of its OWN, never in the request's
    # work dir. That work dir is where an uploaded folder was unpacked, and
    # deleting it left the resume with no inputs: the tool ran again, was
    # handed the same paths, and answered "Path not found". It never showed
    # in testing because a hosted test file lives in `DATA/` and belongs to
    # no request -- so every upload, which is every clinical use, was broken
    # and every test passed.
    if outputs:
        packing = tempfile.mkdtemp(dir=settings.TEMP_DIR)
        archive = await anyio.to_thread.run_sync(
            file_utils.make_zip, outputs,
            os.path.join(packing, f"{tool_name}_stopped.zip"),
        )
        stored = await anyio.to_thread.run_sync(
            transfer.store_result, str(archive), "application/zip"
        )
        reference = stored.as_reference()
        background_tasks.add_task(shutil.rmtree, packing, ignore_errors=True)
    # What the run still needs is handed to the pause record instead of being
    # deleted here; `runs.discard` takes it when the run finally ends, and the
    # reaper takes it from a reader who never came back.
    runs.keep_while_paused(
        runs.CURRENT_RUN.get(None),
        [directory for directory in ([work_dir] if work_dir else [])
         + list(scratch_dirs)])

    # Last, so the phase a watcher reads is where the run actually is: the
    # zip is built and the reference is parked, and from here it waits.
    runs.emit(runs.PHASE_PAUSED)
    _log_served(tool_name, start_time, size, None)
    return JSONResponse(
        {
            "quality_control": True,
            "stopped_after": record.get("stopped_after"),
            "produced": record.get("produced") or [],
            "result_ref": reference,
        },
        background=background_tasks,
    )


async def _finish_run(tool, tool_name: str, start_time: float,
                      background_tasks: BackgroundTasks, detached: bool,
                      work_dir, scratch_dirs, result, size=None,
                      wants_reference: bool = False):
    """Turn what a tool returned into what the caller receives.

    Split out of `_run_tool` so a RESUME reaches it too. Everything here is
    about the answer and nothing about the request, which is exactly the half
    the two paths share: the same packing, the same reference delivery, the
    same background cleanup. A second endpoint would have had to learn all of
    it again, and would have drifted.

    `size` is how many bytes the request carried IN, logged beside what goes
    out. A resume carried none: its inputs were staged by the run it is
    picking up.
    """
    # The tool has returned; what is left is building the response. For a cohort
    # that is a multi-GB archive and minutes of it, so it is a phase of its own
    # rather than a gap between the last progress message and the download.
    stopped = isinstance(result, dict) and result.get("quality_control")
    if not stopped:
        # Not for a stopped run: `packaging` would be appended AFTER the
        # `paused` event dispatch already wrote, and the latest phase is what
        # a watcher reads as the state. The run would report itself as
        # running, packaging, for as long as it sat there waiting to be
        # picked up.
        runs.emit(runs.PHASE_PACKAGING)

    if stopped:
        # A run that STOPPED has an answer of a different shape: not the
        # tool's declared output, which it never got to produce, but
        # everything the chain got through plus the name of the checkpoint it
        # is standing on. Delivered as JSON with a reference rather than as
        # the archive alone, because the client needs both -- the files to
        # look at, and the fact that this run can be told to carry on.
        return await _finish_stopped_run(tool, tool_name, start_time,
                                         background_tasks, work_dir,
                                         scratch_dirs, result, size)

    if tool.output_kind in ("file", "segmentation", "files"):
        # `result` is a path to the output file the tool wrote -- or, for
        # "files", a list of paths / a single directory to bundle into a zip.
        if work_dir is None:
            work_dir = tempfile.mkdtemp(dir=settings.TEMP_DIR)

        # A tool whose inputs all came from the read-only data store writes its
        # output in its own scratch dir under TEMP_DIR, which must be cleaned up
        # too -- whether the response goes out or the packing below fails.
        # `scratch_dirs` covers file_utils.make_scratch_dir(); _output_roots
        # also catches a tool that wrote under TEMP_DIR by hand.
        try:
            outputs = file_utils.output_paths(result)
            if not outputs:
                raise ValueError(
                    f"Tool '{tool.name}' declares output_kind={tool.output_kind!r} but "
                    f"run() returned {type(result).__name__}, not a path (or a list of paths)."
                )
            if tool.output_kind != "files":
                # One file goes back, so there has to be exactly one. Taking
                # the first of several silently would return part of a result
                # and call it a success.
                if len(outputs) > 1:
                    raise ValueError(
                        f"Tool '{tool.name}' declares output_kind={tool.output_kind!r} but "
                        f"returned {len(outputs)} paths. Declare 'files' to return several."
                    )
                # Normalized rather than reused as-is: `result` may have been a
                # mapping of named outputs, and what is streamed is a path.
                result = outputs[0]
            output_roots = _output_roots(outputs, work_dir) | set(scratch_dirs)
            if tool.output_kind == "files":
                # Built inside work_dir, never inside the tool's own scratch
                # dir: the archive has to outlive the files it was made from,
                # right up until the response has finished streaming.
                # Always carries the tool's name. Naming the archive after the
                # single directory a tool produced dates from the in-process
                # era, when that directory had a name the tool chose; a packaged
                # tool writes into the job's own `output/`, so the rule was
                # handing every one of them the same `output.zip`. Nine tools
                # run from the client landed in the download folder under one
                # name, each overwriting the last, and the name leaked a
                # server-side directory rather than saying what produced it.
                stem = ""
                if len(outputs) == 1 and os.path.isdir(outputs[0]):
                    stem = os.path.basename(outputs[0].rstrip(os.sep))
                if not stem or stem == dispatch.JOB_OUTPUT_DIRNAME:
                    stem = "output"
                archive_name = f"{tool.name}_{stem}.zip"
                result = await anyio.to_thread.run_sync(
                    file_utils.make_zip, outputs, os.path.join(work_dir, archive_name)
                )
        except Exception:
            logger.exception("endpoint=/run/%s status=500 (packing output)", tool_name)
            _discard(
                work_dir,
                _output_roots(file_utils.output_paths(result), work_dir) | set(scratch_dirs),
            )
            raise HTTPException(status_code=500, detail="Tool execution failed.")

        media_type = _media_type_of(str(result))

        # A client asking for reference delivery gets the result MOVED out of
        # the work dir (a rename, not a copy) and a JSON pointer to it, so it
        # can pull the bytes over several range requests. Done before the
        # cleanup tasks are queued, which would otherwise take the file away.
        #
        # Only above RESULT_REFERENCE_MIN_MB, for cleanup rather than speed: a
        # streamed response deletes its file the moment the response ends, with
        # no dependency on the client, while a reference waits for a DELETE or
        # for the reaper. Parallel ranges buy nothing on a small result.
        # A detached run has no response left to stream into, so it always
        # takes a reference -- the size floor below is about cleanup, and a
        # detached run's cleanup is the reaper either way.
        deliver_by_reference = detached or (
            wants_reference
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
        if detached and stored is None:
            # Nothing to fall through to: the response went out as a 202 long
            # ago, so a result that cannot be parked is a result nobody can
            # ever collect. Better a failed run than a silent one.
            raise ToolExecutionError(
                f"Tool '{tool_name}' produced a result that could not be stored "
                "for collection."
            )

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




@app.get("/", include_in_schema=False)
def dashboard() -> HTMLResponse:
    """What this server is doing, in a browser. `GET /status` is the same thing
    as JSON, and the page fetches it with the reader's own token."""
    return HTMLResponse(status_page.STATUS_PAGE)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
