"""The `loopback` and `lan` paths: the HTTP API, spoken the way the real client
speaks it.

There is only one remote client in this harness and both paths use it; `lan`
differs from `loopback` by the base URL and by nothing else. That is the point
of the comparison -- any difference between the two arms is the wire.

The protocol is not invented here. It is what
`SlicerAutomatedDentalToolsCloud/ServerToolsCore/ServerToolsCoreLib/` does, down
to the constants: 8 MB parts, four concurrent transfers, `X-Part-SHA256` over
the plaintext part, `Content-Encoding: gzip` on anything not already compressed,
`X-Result-Delivery: reference`, ranged GETs, and a DELETE when the bytes are in.
Measuring a client nobody runs would produce numbers nobody can use.

Phase boundaries, which is what B2 is actually about:

  pack         zipping a folder input, before a byte moves
  upload       POST /uploads plus every PUT of a part
  server_exec  the POST /run round trip. With the inputs already uploaded this
               is the server's own execution plus building the result -- there
               is nothing else in it, which is exactly why the chunked path is
               used even when a file would fit in one request.
  download     the ranged GETs of the result (or the streamed body, for a small
               one)
  unpack       extracting the result archive
  other        total minus the above; see recording.PhaseTimer
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import requests
from requests.adapters import HTTPAdapter

from ..settings import ServerSpec, ToolSpec, TransferSpec

# The client's own table: extensions whose bytes are already compressed, so
# gzipping a part of them costs CPU and saves nothing.
PRECOMPRESSED_EXTENSIONS = (
    ".gz", ".bz2", ".xz", ".zip", ".7z",
    ".xlsx", ".ods", ".docx", ".pptx",
    ".png", ".jpg", ".jpeg",
)
GZIP_LEVEL = 1
READ_BUFFER_BYTES = 1024 * 1024
UPLOADS_FIELD = "__uploads__"
RESULT_DELIVERY_HEADER = "X-Result-Delivery"
RESULT_DELIVERY_REFERENCE = "reference"
PART_TIMEOUT = (15, 180)


class RemoteError(RuntimeError):
    """The server refused, or the transfer could not be completed.

    Carries the status when there is one: a 422 is a config problem, a 500 is
    the tool, and a benchmark record that says only "failed" makes the two
    indistinguishable at analysis time.
    """

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class RemoteRun:
    tool: str
    status_code: int = 0
    delivery: str = ""  # "json" | "streamed" | "reference"
    result: object = None
    result_path: Optional[str] = None
    unpacked_dir: Optional[str] = None
    bytes_uploaded: int = 0
    bytes_downloaded: int = 0
    parts_uploaded: int = 0
    chunked_arguments: list = field(default_factory=list)
    inline_arguments: list = field(default_factory=list)


def worth_compressing(path: str) -> bool:
    return not path.lower().endswith(PRECOMPRESSED_EXTENSIONS)


class RemoteClient:
    """One HTTP session, one server, one set of transfer settings.

    `parallelism` overrides the configured value, which is how B2 runs the same
    payload with parallel transfer on and off: the PROTOCOL stays identical and
    only the number of workers changes, so the difference is attributable to
    parallelism and not to having switched to a different upload path.
    """

    def __init__(
        self,
        server: ServerSpec,
        transfer: TransferSpec,
        token: str,
        base_url: Optional[str] = None,
        parallelism: Optional[int] = None,
    ) -> None:
        self.server = server
        self.transfer = transfer
        self.base_url = (base_url or server.base_url).rstrip("/")
        self.parallelism = parallelism or transfer.parallelism
        self.headers = {"Authorization": f"Bearer {token}"}
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=transfer.connection_pool,
            pool_maxsize=transfer.connection_pool,
            max_retries=0,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "RemoteClient":
        return self

    def __exit__(self, *_exception) -> None:
        self.close()

    # -- probes -------------------------------------------------------

    def health(self) -> dict:
        response = self.session.get(
            f"{self.base_url}/health", timeout=10, verify=self.server.verify_tls
        )
        response.raise_for_status()
        return response.json()

    def unavailable_reason(self) -> Optional[str]:
        try:
            self.health()
        except Exception as error:  # noqa: BLE001 - reported, never swallowed
            return f"{self.base_url} did not answer /health: {type(error).__name__}: {error}"
        return None

    # -- upload -------------------------------------------------------

    def open_upload(self, path: str) -> dict:
        size = os.path.getsize(path)
        response = self.session.post(
            f"{self.base_url}/uploads",
            headers=self.headers,
            json={
                "filename": os.path.basename(path),
                "size": size,
                "chunk_size": self.transfer.chunk_bytes,
            },
            timeout=PART_TIMEOUT,
            verify=self.server.verify_tls,
        )
        if response.status_code >= 400:
            raise RemoteError(
                f"POST /uploads -> {response.status_code}: {response.text[:400]}",
                response.status_code,
            )
        session = response.json()
        session["size"] = size
        session["path"] = path
        return session

    def upload_parts(self, session: dict) -> int:
        """Every part, over `self.parallelism` connections. Returns bytes sent.

        Each worker opens its own handle and seeks; nothing is shared but the
        counter. The server recomputes each part's offset from the chunk size IT
        returned, so the client never invents a layout.
        """
        path = session["path"]
        size = session["size"]
        chunk = int(session["chunk_size"])
        count = int(session["part_count"])
        gzip_parts = self.transfer.gzip_parts and worth_compressing(path)
        sent = [0]
        lock = threading.Lock()

        def send(index: int) -> None:
            offset = index * chunk
            length = min(chunk, size - offset)
            with open(path, "rb") as handle:
                handle.seek(offset)
                data = handle.read(length)
            headers = dict(self.headers)
            headers["X-Part-SHA256"] = hashlib.sha256(data).hexdigest()
            headers["Content-Type"] = "application/octet-stream"
            if gzip_parts:
                data = gzip.compress(data, GZIP_LEVEL)
                headers["Content-Encoding"] = "gzip"
            response = self.session.put(
                f"{self.base_url}/uploads/{session['upload_id']}/parts/{index}",
                headers=headers,
                data=data,
                timeout=PART_TIMEOUT,
                verify=self.server.verify_tls,
            )
            if response.status_code >= 400:
                raise RemoteError(
                    f"PUT part {index} -> {response.status_code}: {response.text[:300]}",
                    response.status_code,
                )
            with lock:
                sent[0] += len(data)

        workers = max(1, min(self.parallelism, count))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(send, range(count)))
        return sent[0]

    def discard_upload(self, upload_id: str) -> None:
        try:
            self.session.delete(
                f"{self.base_url}/uploads/{upload_id}",
                headers=self.headers,
                timeout=15,
                verify=self.server.verify_tls,
            )
        except requests.RequestException:
            pass

    # -- the run ------------------------------------------------------

    def invoke(self, tool_name: str, data: dict, files: dict) -> requests.Response:
        headers = dict(self.headers)
        if self.transfer.result_delivery_reference:
            headers[RESULT_DELIVERY_HEADER] = RESULT_DELIVERY_REFERENCE
        handles = []
        try:
            payload = {}
            for argument, path in files.items():
                handle = open(path, "rb")
                handles.append(handle)
                payload[argument] = (os.path.basename(path), handle)
            response = self.session.post(
                f"{self.base_url}/run/{tool_name}",
                headers=headers,
                data=data,
                files=payload or None,
                timeout=self.server.request_timeout_seconds,
                verify=self.server.verify_tls,
                stream=True,
            )
        finally:
            for handle in handles:
                handle.close()
        return response

    # -- download -----------------------------------------------------

    def download_ranged(self, result_id: str, size: int, destination: str) -> int:
        """Pull one result over `self.parallelism` connections.

        The file is preallocated and every worker `pwrite`s at its own absolute
        offset, so the spans need no ordering and no buffer between them. Spans
        are one chunk each, NOT one per worker: a four-way split of a 500 MB
        result would leave three connections idle while the slowest finished.
        """
        chunk = self.transfer.chunk_bytes
        spans = [(start, min(start + chunk, size) - 1) for start in range(0, size, chunk)]
        descriptor = os.open(destination, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        received = [0]
        lock = threading.Lock()
        try:
            os.ftruncate(descriptor, size)

            def fetch(span) -> None:
                start, end = span
                headers = dict(self.headers)
                headers["Range"] = f"bytes={start}-{end}"
                # identity, deliberately: a transfer-compressed body would not
                # line up with the offsets these writes assume.
                headers["Accept-Encoding"] = "identity"
                response = self.session.get(
                    f"{self.base_url}/results/{result_id}",
                    headers=headers,
                    timeout=PART_TIMEOUT,
                    verify=self.server.verify_tls,
                    stream=True,
                )
                if response.status_code >= 400:
                    raise RemoteError(
                        f"GET range {start}-{end} -> {response.status_code}",
                        response.status_code,
                    )
                offset = start
                for block in response.iter_content(READ_BUFFER_BYTES):
                    if not block:
                        continue
                    os.pwrite(descriptor, block, offset)
                    offset += len(block)
                    with lock:
                        received[0] += len(block)

            workers = max(1, min(self.parallelism, len(spans)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(fetch, spans))
        finally:
            os.close(descriptor)
        return received[0]

    def release_result(self, result_id: str) -> None:
        try:
            self.session.delete(
                f"{self.base_url}/results/{result_id}",
                headers=self.headers,
                timeout=15,
                verify=self.server.verify_tls,
            )
        except requests.RequestException:
            pass

    # -- one whole call, decomposed -----------------------------------

    def run(self, tool: ToolSpec, workspace: str, timer, unpack: bool = True) -> RemoteRun:
        """One complete remote call, with every phase timed into `timer`.

        `workspace` is a directory this call owns: packed inputs, the downloaded
        result and its extraction all land there and the caller removes it.
        """
        os.makedirs(workspace, exist_ok=True)
        outcome = RemoteRun(tool=tool.name)

        data = {name: _stringify(value) for name, value in tool.args.items()}
        for argument, hosted in tool.server_files.items():
            # A server-hosted model or test file is NAMED, not uploaded -- the
            # bytes never leave the server, which is exactly what makes the
            # "94 MB payload" claims about the payload and not about the model.
            data[argument] = hosted["name"] if isinstance(hosted, dict) else str(hosted)

        packed: dict = {}
        with timer.phase("pack"):
            for argument, path in tool.files.items():
                packed[argument] = _pack_if_folder(path, workspace, tool.name, argument)

        references: dict = {}
        inline: dict = {}
        with timer.phase("upload"):
            for argument, path in packed.items():
                if os.path.getsize(path) >= self.transfer.min_chunked_bytes:
                    session = self.open_upload(path)
                    outcome.bytes_uploaded += self.upload_parts(session)
                    outcome.parts_uploaded += int(session["part_count"])
                    references[argument] = session["upload_id"]
                    outcome.chunked_arguments.append(argument)
                else:
                    inline[argument] = path
                    outcome.inline_arguments.append(argument)
            if references:
                data[UPLOADS_FIELD] = json.dumps(references)

        try:
            with timer.phase("server_exec"):
                response = self.invoke(tool.name, data, inline)
                # For an inline (small) input the bytes travel INSIDE this
                # request, so they are counted here and the upload phase is
                # honestly zero for it -- the alternative would be inventing a
                # split the protocol does not have.
                for path in inline.values():
                    outcome.bytes_uploaded += os.path.getsize(path)
                outcome.status_code = response.status_code
                if response.status_code >= 400:
                    raise RemoteError(
                        f"POST /run/{tool.name} -> {response.status_code}: "
                        f"{response.text[:600]}",
                        response.status_code,
                    )
                content_type = response.headers.get("Content-Type", "")
                body = response.json() if content_type.startswith("application/json") else None
        except Exception:
            for upload_id in references.values():
                self.discard_upload(upload_id)
            raise

        if body is not None and "result_ref" in body:
            reference = body["result_ref"]
            outcome.delivery = "reference"
            filename = os.path.basename(str(reference.get("filename") or "")) or "result.bin"
            destination = os.path.join(workspace, filename)
            with timer.phase("download"):
                try:
                    outcome.bytes_downloaded = self.download_ranged(
                        str(reference["result_id"]), int(reference["size"]), destination
                    )
                finally:
                    self.release_result(str(reference["result_id"]))
            outcome.result_path = destination
        elif body is not None:
            # A "text" tool. Nothing to download and nothing to unpack, and the
            # phases say so with zeros rather than being absent.
            outcome.delivery = "json"
            outcome.result = body.get("result")
            timer.add("download", 0.0)
        else:
            outcome.delivery = "streamed"
            filename = _filename_from(response) or f"{tool.name}_result.bin"
            destination = os.path.join(workspace, os.path.basename(filename))
            with timer.phase("download"):
                total = 0
                with open(destination, "wb") as handle:
                    for block in response.iter_content(READ_BUFFER_BYTES):
                        if block:
                            handle.write(block)
                            total += len(block)
                outcome.bytes_downloaded = total
            outcome.result_path = destination

        with timer.phase("unpack"):
            if unpack and outcome.result_path and outcome.result_path.lower().endswith(".zip"):
                extracted = os.path.join(workspace, "unpacked")
                os.makedirs(extracted, exist_ok=True)
                with zipfile.ZipFile(outcome.result_path) as archive:
                    archive.extractall(extracted)
                outcome.unpacked_dir = extracted

        return outcome


def _stringify(value) -> str:
    """Exactly what the Slicer client sends: booleans lowercased, containers as
    JSON, everything else str()."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    return str(value)


def _pack_if_folder(path: str, workspace: str, tool_name: str, argument: str) -> str:
    """A directory input becomes a .zip, because HTTP has no notion of one.

    Members already compressed are STORED rather than deflated -- the client's
    own rule, and the reason a folder of .nii.gz packs at disk speed.
    """
    if not os.path.isdir(path):
        return path
    archive_path = os.path.join(workspace, f"{tool_name}_{argument}.zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for root, _directories, names in os.walk(path):
            for name in sorted(names):
                absolute = os.path.join(root, name)
                relative = os.path.relpath(absolute, path)
                compression = (
                    zipfile.ZIP_STORED if not worth_compressing(name) else zipfile.ZIP_DEFLATED
                )
                archive.write(absolute, relative, compress_type=compression)
    return archive_path


def _filename_from(response) -> Optional[str]:
    disposition = response.headers.get("Content-Disposition", "")
    for part in disposition.split(";"):
        part = part.strip()
        if part.lower().startswith("filename="):
            return part.split("=", 1)[1].strip().strip('"')
    return None


def clear_workspace(workspace: str) -> None:
    shutil.rmtree(workspace, ignore_errors=True)
