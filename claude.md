# CLAUDE.md — GPU Inference Server (tool-registry architecture) + 3D Slicer client

## Project context

We are offloading heavy computation from a 3D Slicer extension to a remote GPU
server. The Slicer module is a **thin client**: it sends inputs to the server, the
server runs the selected tool, and returns the result.

The extension will expose **at least 15 tools**, and this number will grow. The
whole design must therefore be **scalable and low-friction to extend**: adding a new
tool must require writing one self-contained class and nothing else — no edits to
the server core, no new route, no manual registration list to keep in sync.

**The data is confidential medical imaging.** Confidentiality and transport security
(TLS, auth, temp-file cleanup) are first-class requirements.

The HTTP contract is **blocking request/response** (the client gets its result in
the same response), but the server itself executes tools **in parallel**: each
`tool.invoke` runs in a worker thread, capped by `MAX_CONCURRENT_TOOLS` (see
`config.py`), so a long inference never freezes the event loop or other requests.
Do **not** add Celery/Redis/async job queues yet.

## Core design: a `Tool` base class

Every tool is a **class deriving from a common `Tool` base class**. A tool declares:

1. **A unique name** (used to select it over the wire, e.g. `"test_tool"`).
2. **The arguments it expects**, as a typed schema (each argument has a name, a type,
   and whether it is required). This schema is what lets the server **validate**
   incoming arguments *before* running the tool.
3. **A `run(...)` method** that performs the actual work and returns the result.

The base class provides the shared machinery: argument validation against the
declared schema, a uniform way to be invoked, and metadata for discovery. Subclasses
only declare their schema and implement `run`.

### Validation contract (important)
- When a tool is invoked with a set of arguments, the base class **validates those
  arguments against the tool's declared schema** first:
  - every **required** argument is present,
  - no **unknown** arguments are passed,
  - each argument **matches its declared type** (coerce where sensible, e.g. form
    strings → int/float/bool; otherwise reject).
- If validation fails, **raise a clear, specific error** (a dedicated exception type,
  e.g. `ToolArgumentError`) naming what is wrong (missing arg, unexpected arg, wrong
  type). The server maps this to an HTTP `422` with the message.
- Only if validation passes is `run(...)` called. `run` can then trust its inputs.

### Suggested base-class shape (guidance, adapt as needed)
```python
@dataclass
class ArgSpec:
    type: type          # str, int, float, bool, "file", ...
    required: bool = True
    description: str = ""

class Tool(ABC):
    name: str                      # unique tool id, set on each subclass
    arguments: dict[str, ArgSpec]  # declared schema, set on each subclass
    output_kind: str = "text"      # "text" | "file" | "segmentation" | ...

    def validate(self, args: dict) -> dict:
        """Check args against self.arguments; return cleaned/coerced args or
        raise ToolArgumentError."""
        ...

    def invoke(self, args: dict):
        cleaned = self.validate(args)
        return self.run(**cleaned)

    @abstractmethod
    def run(self, **kwargs):
        """Do the actual work. Trusts that args are already validated."""
        ...
```
Use whatever concrete mechanism is cleanest (dataclasses, pydantic models, or a
simple dict of `ArgSpec`). The **hard requirement** is: declared typed schema +
validation-before-run + clear error on mismatch.

## Scalable tool discovery (the registry)

Tools live in a **`tools/` package**, one file per tool (or grouped logically). The
server builds its registry by **auto-discovering every `Tool` subclass** in that
package at startup — e.g. import all modules in `tools/`, collect every subclass of
`Tool`, and index them by their `name`. Adding a tool = dropping a new file in
`tools/` that defines a `Tool` subclass. **No central list to edit.**

- Detect and reject **duplicate tool names** at startup with a clear error.
- Expose the registry as `name -> Tool instance` (or class).

## The one tool that exists now

Only a single **test tool** exists for now:
- name: `"test_tool"`
- arguments: `text_1` (str, required), `text_2` (str, required)
- `run(text_1, text_2) -> str`: returns some simple combination (e.g. concatenation
  or an echo), enough to prove the round trip. Output kind: `"text"`.

All other ~14+ tools are future work; the architecture must accommodate them without
change to the core. Include `# TODO` guidance showing how to add another tool.

## Target architecture

```
┌──────────────────────┐        HTTPS / TLS              ┌──────────────────────────┐
│  3D Slicer module    │  ── POST /run/{tool_name} ────► │  FastAPI server          │
│  (client, requests)  │      (args [+ file], token)     │  (Uvicorn)               │
│                      │                                 │  - verify token          │
│                      │                                 │  - registry.get(name)    │
│                      │  ◄── 200 + result ──────────────│  - tool.validate(args)   │
│                      │                                 │  - tool.run(**args)      │
│                      │                                 │  - return result         │
└──────────────────────┘                                 │  - cleanup temp files    │
                                                          └──────────────────────────┘
```

## Expected repo structure

```
.
├── CLAUDE.md
├── server/
│   ├── main.py              # FastAPI app: generic /run/{tool_name} endpoint
│   ├── base.py              # Tool base class, ArgSpec, ToolArgumentError
│   ├── registry.py          # auto-discovery of Tool subclasses in tools/
│   ├── data_store.py        # DataStore interface + LocalDataStore (server-side models/testfiles)
│   ├── security.py          # Bearer token verification
│   ├── config.py            # config from environment variables
│   ├── tools/
│   │   ├── __init__.py
│   │   └── test_tool.py     # the only tool for now
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md
├── DATA/                    # DATA_DIR mount, read-only: <tool_name>/{models,testfiles}/
└── slicer_client/
    └── inference_client.py  # generic client used by every Slicer module
```

---

## PART 1 — Server (`server/` directory)

### Required stack
- **FastAPI** + **Uvicorn**, `python-multipart`.
- **No** database, queue, Celery, or Redis in this version.
- Python 3.10+.

### `base.py`
- `ArgSpec`, `Tool` (abstract base with `name`, `arguments`, `output_kind`,
  `validate`, `invoke`, abstract `run`), and `ToolArgumentError`.
- `validate` enforces: required present, no unknowns, type match/coercion; raises
  `ToolArgumentError` with a precise message otherwise.

### `registry.py`
- Auto-discovers every `Tool` subclass under the `tools/` package at import time.
- Builds `TOOLS: dict[str, Tool]` keyed by `name`.
- Raises on duplicate names.
- `get_tool(name) -> Tool` raises a not-found error (→ `404`) for unknown names.

### `data_store.py` — server-side models and test files
Lets a tool argument be satisfied by a file already present on the server (an AI
model, a reference test dataset) instead of the client uploading it every call.
- `ArgSpec.server_selectable: Optional[str]` — `"model"` or `"testfile"` on a
  file-typed argument opts it into this; `None` (default) means upload-only.
- Layout on disk: `DATA_DIR/<tool_name>/models/` and
  `DATA_DIR/<tool_name>/testfiles/` (`DATA_DIR` from config, mounted **read-only**).
- `GET /tools/{tool_name}/data` (Bearer-protected) lists the available file names
  in both folders, so a client can present them (e.g. a dropdown) instead of a file
  picker.
- In `POST /run/{tool_name}`, a `server_selectable` argument sent as a plain form
  value (the file name) instead of an upload is resolved against `data_store`
  rather than streamed to a temp dir.
- **Backend abstraction (`DataStore`):** `main.py` and every `Tool` only ever call
  `data_store.list_models/list_testfiles/resolve_model/resolve_testfile` — never
  the filesystem directly. `LocalDataStore` is the only implementation today. To
  swap to an external database or object store later: add a new `DataStore`
  subclass in `data_store.py` returning a `ResolvedFile(path, is_temporary)` from
  each `resolve_*` (`is_temporary=True` if the backend had to materialize a temp
  copy, so `main.py`'s cleanup deletes it — persistent local paths must never be
  deleted), then select it in `build_data_store()` via `settings.DATA_BACKEND`.
  No change needed anywhere else.

### `tools/test_tool.py`
- Defines `TestTool(Tool)` with `name = "test_tool"`, the two required string args,
  and a `run` returning a str. Serves as the copy-paste template for future tools.

### Endpoints (`main.py`)
1. `GET /health` → `{"status": "ok"}`, no auth.
2. `GET /tools` → list of `{name, arguments, output_kind}` from the registry, no
   auth. Lets clients discover tools and their expected arguments.
3. `POST /run/{tool_name}` — generic, Bearer-token protected:
   - `404` if `tool_name` not in registry.
   - Collect arguments from the request (form fields for scalars; an optional
     uploaded `file` streamed to a temp dir in chunks — never fully into RAM).
   - Call `tool.invoke(args)` → this runs `validate` then `run`.
   - On `ToolArgumentError` → `422` with the message.
   - Return the result: JSON for text/scalar outputs, `FileResponse` for file
     outputs (correct media type).
   - **Guaranteed cleanup** of any temp files (input and output), including on error
     (`BackgroundTask` for the streamed output, `try/finally` for the input).

### `security.py`, `config.py`
- Bearer token from env (`API_TOKEN`), constant-time compare, `401` on failure.
- Config from env: `API_TOKEN`, `DEVICE`, `MAX_UPLOAD_MB`, `MAX_EXTRACTED_MB`,
  `TEMP_DIR`, `ALLOWED_EXTENSIONS`, `DATA_DIR`, `DATA_BACKEND`. Sensible dev
  defaults.

### Security / confidentiality — hard requirements
- **TLS mandatory**; README documents HTTPS + self-signed cert for dev, real cert
  for prod, never plain HTTP.
- Upload size limit (`MAX_UPLOAD_MB` → `413`).
- Client archives extracted for `"folder"` arguments are untrusted: zip slip,
  symlink members, and zip bombs (`MAX_EXTRACTED_MB`) all rejected with `400`.
- Delete all temp files after processing.
- Never log file contents, arguments values, or patient metadata. Logs limited to
  timestamp, endpoint, tool name, status, duration, size.
- Comment in `main.py`: deploy in the appropriate jurisdiction; de-identification
  happens client-side.

### `.env.example`, server `README.md`
- All env vars with dummy values.
- Install, self-signed cert generation, HTTPS run command, `curl` examples hitting
  `/tools` and `/run/test_tool` (with token). A short "how to add a tool" section:
  create a file in `tools/`, subclass `Tool`, done.

---

## PART 2 — Slicer client (`slicer_client/inference_client.py`)

Runs inside Slicer's Python interpreter. Available: `requests`, `slicer`, `qt`,
`vtk`, `os`, `tempfile`. No other external deps.

Provide a small, generic client mirroring the server:
- `check_server(server_url, verify_tls) -> bool` (`GET /health`).
- `list_tools(server_url, verify_tls) -> list` (`GET /tools`), so the client can see
  each tool's expected arguments.
- `run_tool(server_url, token, tool_name, args: dict, file_path: str = None,
  verify_tls=True) -> response`: POST to `/run/{tool_name}` with Bearer token,
  sending `args` as form fields and optionally streaming `file_path`. Blocking, with
  a generous configurable timeout. Returns the parsed result (JSON text or a written
  output file path depending on content type).
- Clean error handling mapping `401` / `404` / `422` / `413` / timeout / network
  errors to clear messages (via `slicer.util.errorDisplay` when wired into UI).

### Client-side security
- `verify_tls` defaults to **True**; False only for local dev, documented as such.
- Token not hardcoded: read from Slicer settings / env, passed as a parameter.
- Never log token or argument contents.

### UI integration (guidance only)
- Show how a module's `onApplyButton` calls `run_tool` with its own `tool_name` and
  an `args` dict, and how the same client serves all tools.
- Use `showStatusMessage` / wait cursor during the blocking call.

---

## Code style and conventions
- Clear code; comments at security, cleanup, and `# TODO` extension points.
- Explicit error handling, no silent `except: pass`.
- Type hints throughout the server.
- No hardcoded secrets.
- **All code, identifiers, comments, docstrings, and log messages in English.**

## Definition of done
- Server runs over HTTPS; `/health`, `/tools`, `/run/{tool_name}` all work.
- `test_tool` round-trips: send `text_1` + `text_2`, get a string back.
- Passing wrong/missing/extra args to a tool yields a `422` with a clear message
  (validation happens in the `Tool` base class before `run`).
- Unknown tool → `404`; no token → `401`; oversized file → `413`.
- Adding a new tool requires only a new file in `tools/` subclassing `Tool` — no
  core changes, no manual registration.
- No temp files left behind.

## Out of scope for this iteration (do not implement)
- Job queue / Celery / Redis / async polling.
- Real GPU inference (the test tool is trivial).
- Model/GPU memory management (note it for later, don't build it).
- Scaling (multi-process / multi-machine) / database. (In-process parallelism IS
  implemented: tool runs execute concurrently in worker threads, see changelog.)

## Changelog

### 2026-07-27 — Parallel request handling (threadpool execution of tools)
**Motivation:** `run_tool` called `tool.invoke(args)` synchronously inside an
`async def` endpoint, i.e. directly on the uvicorn event loop. Any inference in
progress froze the entire server — a second `/run`, or even `/health`, could not
be answered until it finished. The server was effectively serial.

**Design (`server/main.py`, `server/config.py`):** `tool.invoke` now runs via
`anyio.to_thread.run_sync(...)` in a worker thread, with a dedicated
`anyio.CapacityLimiter` capping simultaneous tool executions at
`MAX_CONCURRENT_TOOLS` (new setting, default 4) — excess requests wait for a
slot instead of piling unbounded work onto RAM/CPU/GPU. The limiter is created
lazily (anyio needs a running event loop) and is dedicated to tool runs so
queued inference can't starve the default threadpool used by sync endpoints.
This is safe because tools are stateless (everything arrives via `args`), each
request has its own `work_dir`, and `DATA_DIR` is read-only. The HTTP contract
is unchanged: still blocking request/response, no job queue.

**Test (`server/tests/test_main.py::test_concurrent_requests_run_in_parallel`):**
registers a probe tool whose `run()` blocks on a 2-party `threading.Barrier`,
then fires two requests through ONE shared event loop (`TestClient` as context
manager — two bare `client.post` calls from separate threads would each get
their own loop and pass even against a serial server). The barrier only opens
if both requests are inside `run()` at the same time; serial execution times it
out and fails the test.

### 2026-07-27 — surg_mov_pred: the model is server-side only, selected by name
**Motivation:** the client still had to provide the model as a zip upload (or
optionally pick a server-side one). The model should live exclusively in the
server's data store: the client asks for the list of available models
(`GET /tools/surg_mov_pred/data`, already existing) and sends only the *name*
of the chosen one — no model package ever travels from the client.

**Design (`server/tools/surg_mov_pred/surg_mov_pred.py`):** the `model`
argument changed from `ArgSpec(type="zip_file", server_selectable="model")` to
`ArgSpec(type=str, server_selectable="model")`. The resolution path is
unchanged — `main.py` already resolves any `server_selectable` argument sent
as a plain form value through `data_store`, so `run()` still receives a local
path to the model zip. What changed is the contract: a scalar (non-file) type
means "name only". To enforce it, `main.py` now rejects with a 400 any file
*upload* targeting a non-file-typed argument (previously the uploaded temp
path would have been silently passed through as the argument's string value).
`base.py`'s `server_selectable` comment documents the two flavors: on a
file-typed argument the client may still upload its own file (e.g.
`surg_mov_pred`'s `input`); on a scalar argument the server-side file is the
only option. `GET /tools` needs no change — it already exposes `type` and
`server_selectable`, which is all a client needs to render a dropdown of
server-side model names instead of a file picker (the Slicer client's
`ServerToolsCoreLib` does exactly that; see `SlicerAutomatedDentalToolsCloud`'s
ARCHITECTURE.md).

**Tests (`server/tests/test_main.py`):** an upload for `surg_mov_pred`'s
`model` is a 400; an unknown model name is a 404; a synthetic tool with a
str-typed `server_selectable="model"` argument resolves the name through
`data_store` and `run()` gets the file's path. The real-data integration test
(`test_data_integration.py`) already sent names as form values and covers the
new schema unchanged.

### 2026-07-27 — Pre-push test gate + real-data integration tests
**Motivation:** the test suite (`server/tests/`, `server/tools/*/test/`) only ran
on synthetic fixtures and only when someone remembered to invoke it manually.
Nothing stopped a regression from being pushed, and there was no way to
exercise a tool against a real testfile without writing one-off scripts.

**Design:** a new `docker-compose.yml` service, `test`, runs the exact same
image as `inference` (so no local Python environment to install/maintain)
without its GPU `deploy` reservation, installs `requirements.txt` +
`requirements-dev.txt`, and runs `python -m pytest`
(`docker compose run --rm test`). A git hook, `.githooks/pre-push`, runs that
service before every push and blocks it on any test failure; it is opt-in per
clone via `git config core.hooksPath .githooks` (git hooks aren't versioned
by default), and can be bypassed for a single push with git's built-in
`git push --no-verify`.

`server/tests/test_data_integration.py` complements the synthetic tests: for
every registered tool whose required arguments are all `server_selectable`
(see the 2026-07-24 entry below), it looks up real files via `data_store`
under `DATA/<tool_name>/{models,testfiles}/` and runs the tool end-to-end
against them. `DATA/` is gitignored (confidential medical data), so a tool
with no matching file present is **skipped**, never failed — a clone without
the dataset can still push. A maintainer turns a skip into a real run by
dropping a file under the relevant `DATA/<tool_name>/...` folder locally.

### 2026-07-24 — Server-side data store: models and test files without re-upload
**Motivation:** tools like `surg_mov_pred` required the client to re-upload the
same model package on every single call, and there was no way for a client to
say "run this against the server's reference test data" instead of streaming
its own file. Confidential-data constraints rule out a generic upload cache, so
this needed to be explicit, per-tool, read-only server-side storage instead.

**Design:** `server/data_store.py` introduces a `DataStore` interface
(`list_models`, `list_testfiles`, `resolve_model`, `resolve_testfile`) with a
`LocalDataStore` implementation reading `DATA_DIR/<tool_name>/{models,testfiles}/`
(new `DATA_DIR`/`DATA_BACKEND` settings in `config.py`). `ArgSpec` gained
`server_selectable: Optional[str]` (`"model"` | `"testfile"`); a new
`GET /tools/{tool_name}/data` endpoint lists what's available so a client can
offer a selection instead of a file picker. In `POST /run/{tool_name}`, a
`server_selectable` argument sent as a plain form value (a file name) rather
than an upload is resolved through `data_store` and excluded from the temp-file
cleanup that applies to genuine uploads (`server/main.py`). `surg_mov_pred`'s
`model` and `input` arguments now both opt in.

**Deliberately abstracted for a future external database/object store:** neither
`main.py` nor any `Tool` touches the filesystem directly — only `data_store`.
Each `resolve_*` returns a `ResolvedFile(path, is_temporary)`; `is_temporary`
lets a future backend that must materialize a remote blob to a local temp copy
(e.g. downloaded from a DB or S3) mark it for cleanup, while `LocalDataStore`'s
persistent paths are never deleted. Swapping backends later is a change
contained entirely to `data_store.py` (a new `DataStore` subclass, wired up in
`build_data_store()` via `settings.DATA_BACKEND`) — see the `# TODO` there.

**Also:** `docker-compose.yml` now mounts a single `./DATA:/data:ro` (previously
two inconsistent mounts, `./models:/models` and `./data:/data`, the latter not
matching the `DATA/` folder actually used on disk); `.gitignore` excludes `DATA/`
so model weights and test datasets are never committed.

### 2026-07-24 — Correct `Content-Type` for file-kind tool outputs
**Problem:** `POST /run/{tool_name}` responses with `output_kind in ("file",
"segmentation")` always sent `media_type="application/octet-stream"` (or
`application/gzip` for `.gz` files), regardless of the output's real format. The
new `surg_mov_pred` tool returns a `.xlsx` file, and since an `.xlsx` is
internally a zip container (`PK\x03\x04` signature), a client that decides
whether to unzip a downloaded "file" result by sniffing magic bytes instead of
trusting `Content-Type` could not tell it apart from an actual zip archive —
it silently extracted the Excel file's internal XML parts instead of saving
the `.xlsx` itself.

**Fix (`server/main.py`):** the `FileResponse` for file-kind outputs now
derives `media_type` from the output file's extension via
`mimetypes.guess_type()`, falling back to the previous
`application/gzip`/`application/octet-stream` logic only when the type can't
be guessed (still the case for bare `.gz` files, e.g. `.nii.gz` segmentation
outputs — unchanged behavior). This makes `.xlsx` responses carry the correct
`application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`
`Content-Type`, and also fixes `.zip` (`application/zip`), `.csv`
(`text/csv`), `.ods` (`application/vnd.oasis.opendocument.spreadsheet`), etc.
`Content-Disposition`'s `filename` was already correct (`os.path.basename(result)`,
carrying the real extension) and is unchanged.

**Client-side follow-up (not part of this repo):** if the Slicer client (see
`SlicerAutomatedDentalTools`, e.g. the `SurgMovPred` module) decides whether to
unzip a downloaded tool result by sniffing the response body's magic bytes
rather than reading `Content-Type` / `Content-Disposition`, it must be updated
to trust those headers instead — magic-byte sniffing can never distinguish a
real `.xlsx`/`.docx`/`.pptx`/`.ods` file from an actual zip archive, since
those formats are zip containers by design.