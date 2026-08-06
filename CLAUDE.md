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

## The registered tools

(Originally only `test_tool` existed; see the changelog for how the rest arrived.)

- `test_tool` — two required strings in, their concatenation out. Proves the
  round trip and serves as the minimal copy-paste template.
- `example_tool` — the feature showcase: multi-type input (`csv_file` or
  `folder`), `choice`/`multichoice` arguments, `output_kind = "files"`.
- `SurgMovPred` — surgical movement prediction from tabular measurements
  (stacking models, server-side model bundles).
- `AMASSS` — CBCT skull structure segmentation (nnUNet v2, GPU).
- `ALI` — automatic landmark identification, on CBCT volumes (deep-RL agents)
  or intraoral surface scans (multi-view rendering + 2D UNet). The engine is
  chosen from the data, not from an argument.
- `CrownSeg` — per-tooth labelling of intraoral scans (shapeaxi). Its own tool
  rather than a helper inside ALI, because ASO, AREG and FlexReg need it too.

The extension will eventually expose ~15+ tools; the architecture must
accommodate them without change to the core. See `ADDING_A_TOOL.md`.

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

## Repo structure

```
.
├── CLAUDE.md
├── ADDING_A_TOOL.md         # the full contract for writing a tool
├── docker-compose.yml       # inference service + test service (profile "test")
├── .githooks/pre-push       # runs `docker compose run --rm test` before a push
├── scripts/                 # populate DATA/ from the public GitHub releases
│   ├── setup-models.sh      #   curl-pipeable entry points
│   ├── setup-testfiles.sh
│   ├── fetch_data.py        #   the engine (stdlib only)
│   └── data-manifest.yml    #   what to download, and where it goes
├── server/
│   ├── main.py              # FastAPI app: generic /run/{tool_name} endpoint
│   ├── base.py              # Tool base class, ArgSpec, ToolArgumentError
│   ├── registry.py          # auto-discovery of Tool subclasses in tools/
│   ├── data_store.py        # DataStore interface + LocalDataStore (server-side models/testfiles)
│   ├── file_utils.py        # shared helpers: zip extraction/creation, tabular loading
│   ├── security.py          # Bearer token verification
│   ├── config.py            # config from environment variables
│   ├── tools/               # one folder per tool: tools/<name>/<name>.py (+ src/, test/)
│   │   ├── test_tool/
│   │   ├── example_tool/
│   │   ├── SurgMovPred/
│   │   ├── AMASSS/
│   │   ├── CrownSeg/
│   │   └── ALI/
│   │       └── src/         # ALILogic.py dispatches; one folder per engine
│   │           ├── ALI_CBCT/
│   │           ├── ALI_IOS/
│   │           └── markups/ # shared by both engines, no other tool needs it yet
│   ├── tests/               # HTTP-layer + integration tests
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── .env.example
│   └── README.md
└── DATA/                    # DATA_DIR mount, read-only, gitignored: <tool_name>/{models,testfiles}/
```

The Slicer client (thin modules + the generic inference client) lives in its
own repo, `SlicerAutomatedDentalToolsCloud` — not here.

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
- `GET /tools/{tool_name}/testfiles/{filename}` (Bearer-protected) streams one of
  the listed **test files** to the client (a folder entry arrives zipped) — backs
  the Slicer client's "Test file" button. Models are deliberately not
  downloadable: selected by name, used in place.
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
  `TEMP_DIR`, `MAX_CONCURRENT_TOOLS`, `AMASSS_MAX_GPU_JOBS`, `ALLOWED_EXTENSIONS`,
  `DATA_DIR`, `DATA_BACKEND`. Sensible dev defaults.
- **Every setting goes through `config.Settings`** — no tool reads `os.getenv`
  directly, even for a knob only it uses, so the whole configuration stays
  discoverable in one file and documented in `.env.example`.
- A setting a tool reads must be reachable: an argument default like
  `device: str = "cpu"` in a function signature short-circuits
  `device or settings.DEVICE` and makes the environment variable dead. Default
  such parameters to `None` and let the setting be the single source of truth.

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

### 2026-08-06 — Presentation hints: the schema says how to lay a panel out

**Ported here from the `aso` branch, where it was written for ASO's four-mode
panel; this branch needs it for a different reason.** ALI's `landmarks`
argument (see the entry below) publishes 118 options, and the generic client
renders a `multichoice` as one check box per line — a column of 118 on top of
the two selections ALI already always shows. The alternative was a hand-written
Qt panel, which puts the anatomy back in the client and makes "add a landmark
server-side" a client release again.

**Five optional `ArgSpec` fields, published by `GET /tools`, ignored by
`validate()` and `run()`:** `label` (the field label), `section` (which
collapsible box), `visible_when` (`{other_arg: value}`, show only while every
entry matches), `ui` (`"tabs"`/`"grid"`/`"inline"` — how a multichoice's boxes
are laid out) and `groups` (`{group name: (option, ...)}` for the two grouped
layouts). Every one is `null` on every existing tool, so every existing panel
renders unchanged.

**`label` closes the last thing the client was still inventing.** Field labels
were built client-side from the argument name, by two different rules in the
same panel: `formgen.build` used the raw schema name while the file-input rows
prettified it, so a panel showed "Input" directly above "cbct_regions". No
naming rule can do better than "Cbct regions" for an acronym. Every
user-visible word describing a tool is now the tool's: label, section title,
tab names, option names, tooltip. The client keeps only its own chrome (Apply,
Cancel, "Output folder", All / None / Default).

**None of the layout fields names an anatomical concept**, which is the whole
point: `groups` says what to group, `ui` how to lay it out, `visible_when` when
it applies. ALI's tabs are `ALI_CBCT/landmarks.GROUP_LABELS` — the same table
the engine groups its output files by, published rather than restated.

**`check_schema` rejects them at startup, and that matters more here than for
a real type.** A wrong `visible_when` hides a field for good, and a client
cannot tell that from a field the tool never declared — the failure is silent
everywhere else, so it has to be loud at boot: unknown layout, `groups` without
a layout that uses them, a group naming an option that does not exist, a
`visible_when` on an undeclared argument, on a non-`choice` argument, or
expecting a value outside its `choices`. An option no group mentions is *not*
an error: the client renders the leftovers rather than dropping a selection the
tool genuinely offers.

**`visible_when` is presentation, not validation.** A hidden argument is not
sent, so its declared default applies, and a tool's own cross-argument checks
still run for a direct API call that sends whatever it likes. What it does fix
is a real wire problem: a multichoice is read back as the *complete*
`{option: checked}` dict and the server reads what it receives **as** the
selection, so a panel was sending the inert mode's selection too — a choice the
user was never shown, frozen at whatever the invisible widget was built with
even after the default changed here.

### 2026-07-31 — ALI's model bundle is matched to the detected mode; a wrong bundle is a 422

**Found by running ALI IOS from Slicer**, on the first request after the
torch-2.8/pytorch3d image rebuild. The client's model dropdown was left on
`ALI_CBCT_Models` for an intraoral scan: the IOS engine looked for
jaw/network-token checkpoints in the CBCT folder, listed all 119 CBCT files
as unrecognized -- and Slicer showed `500 -- The tool failed on the server`,
with the one message actually written for the user buried in the server log.

**`model` is now optional, and the mode picks it (`ALILogic.select_bundle`).**
The tool's own rule -- the mode is detected, not declared -- applied one
argument further: the server already knows which engine will run, and each
engine recognises its own bundle layout through its `discover_weights`
(`<landmark>/<scale>/*.pth` folders vs flat jaw/network-token checkpoints;
mutually exclusive by construction, and file-name parsing only, so probing a
bundle costs a directory walk, never a model load). Among the hosted bundles,
the one the chosen engine recognises runs. No match is a 422 naming
`setup-models.sh`; several matches are a 422 naming the candidates rather
than a silent pick -- which model vintage ran must never be a surprise. For
the same reason the report gains `model_bundle`, so "which weights placed
these landmarks?" has an answer even when nobody named them. A backend temp
copy materialized for a probe (`ResolvedFile.is_temporary`) is deleted
whether or not it was picked.

**A named-but-wrong bundle answers 422, not 500.** The five
`FileNotFoundError`s the two engines raised on a mismatched or empty bundle
are `ToolArgumentError` now -- same category as an empty
`cbct_regions`/`ios_networks` selection, and the message (which names the
bundle, the selection and the expected checkpoint naming) reaches Slicer
verbatim instead of dying in the log. The two "not a directory" messages
carried the full server path; a 422 body travels, so they name the basename
only.

**Client (`SlicerAutomatedDentalToolsCloud`):** the dropdown of an OPTIONAL
scalar `server_selectable` argument now leads with an "(automatic)" entry
whose item data is `""` -- `collectArgs` already drops empty optionals, so
the default selection sends no `model` at all and the server decides.
Generic in `base_widget`/`formgen`, so any future tool with an optional
hosted-file argument gets the entry for free. `test_data_integration.py`
needs no change: it still sends every `server_selectable` argument
explicitly, names included.

**Tests:** 7 new server-side (auto-pick per mode, no hosted bundle,
no matching bundle, ambiguity, temp-probe cleanup, engine-level 422, and an
end-to-end CBCT run with no `model` whose report names the bundle), plus the
client's schema mirror, its combo stub gaining item data, and its "input
alone is a complete request" contract tests. Verified live end to end: an
IOS request with no `model` returned 200 in 17s with
`model_bundle: ALI_IOS_Models` in the report.

### 2026-07-31 — 501 for "this server cannot do that", instead of a blank 500

**Found by running ALI in IOS mode.** The preflight added earlier did its job:
it raised immediately, before any mesh was read, with a message naming
pytorch3d and adding that the CBCT engine is unaffected. That message went to
the server log — and the Slicer user got `500 — The tool failed on the server.`

500 hides its detail deliberately, and rightly: a crash inside a tool can name
server-side paths and internals. A missing optional dependency is the opposite
kind of event. The request was valid, nothing the caller changes will help,
and the reason names a package, nothing sensitive. Swallowing it sends the
user hunting through their own arguments for a problem that is entirely ours.

`base.ToolUnavailableError` plus a `501 Not Implemented` mapping in `main.py`.
Every dependency-import failure across `ALI`, `CrownSeg` and `AMASSS` raises it
now — twelve sites — because the same condition answering 500 in one tool and
501 in another is worse than either. **No client release needed:**
`error_for_status` shows the server's `detail` verbatim for any status it does
not map specially, so 501 already reads correctly in Slicer. Verified end to
end: an IOS request returns 501 with the pytorch3d message in the body.

### 2026-07-31 — Test files are downloadable: `GET /tools/{tool}/testfiles/{filename}`

**Motivation:** the Slicer client grows a per-input "Test file" button that
fills a file input with reference data, so a user can try a tool without
hunting for a scan of their own. The hosted-name route
(`server_selectable="testfile"`, sent as a form value) runs a tool on a test
file without it ever travelling — but it cannot put the file *in the user's
hands*, and the button's whole point is a local copy the user can look at and
re-run at will.

**Design (`server/main.py`):** one new Bearer-protected endpoint streams a
test file by name, resolved through `data_store.resolve_testfile` (so the
backend abstraction and its traversal checks apply unchanged; unknown name →
404). A folder entry is zipped on the fly into a staging dir under `TEMP_DIR`
(`file_utils.make_zip`, run off the event loop) and removed by background task
once the response has streamed; a backend temp copy
(`ResolvedFile.is_temporary`) is likewise removed after streaming. **Test
files only** — models are selected by name and used in place; nothing a client
does should ever pull one off the server. The log line carries tool, status,
duration and size, never the file name, matching `/run`. The media-type guess
is factored into `_media_type_of()`, shared with `/run`.

Also: `AMASSS`'s `input` is now `server_selectable="testfile"` like ALI's and
SurgMovPred's, so the reference scan under `DATA/AMASSS/testfiles/` feeds its
"Test file" button. (The Slicer GUI dropped its hosted-name dropdown for file
arguments in the same change — the input field always holds a local file now;
the name route stays in the API and in `test_data_integration.py`.) The
client grays its button off this server's actual `GET /tools/{tool}/data`
listing, so an empty `testfiles/` folder is a grayed button explaining itself
in its tooltip, not a 404.

**Tests (`tests/test_main.py`):** 401 without a token, 404 for unknown
tool/file, a plain file streamed with the right `Content-Disposition`/type and
the store left untouched, a folder zipped with `TEMP_DIR` clean afterward, and
an `is_temporary` copy removed after streaming.

### 2026-07-31 — `monai` pinned: an unpinned entry was replacing the image's torch

**Caught by reading a `pip install` log, not by a failing test.** Adding
`monai` (unpinned) to `requirements.txt` for ALI made every container start
resolve `monai 1.6.0`, which requires `torch>=2.8.0` — so pip downloaded a
`torch 2.13.0` wheel plus the whole CUDA 13 stack **over the image's
`2.5.1+cu124`**, on every start. Three consequences, none of which fail a
test: ~3 GB downloaded per container start (`--no-cache-dir`), `torchaudio
2.5.1+cu124` left unsatisfiable, and the image's purpose-built CUDA torch
shadowed by a wheel that happened to still find the card here — on a host
whose driver predates CUDA 13 it would not have.

This is exactly the failure `requirements.txt`'s own comment warns about
("never pin or force-reinstall torch"), reached from the other direction: not
by listing torch, but by listing something that outranks it.

`monai==1.5.1` is the fix — it asks for `torch>=2.4.1`, which the image
already satisfies, so pip leaves torch alone. Every transform and network ALI
uses (`EnsureChannelFirst`, `BorderPad`, `ScaleIntensity`, `SpatialCrop`,
`DenseNet`, `UNet`) exists there, confirmed by re-running the full real-data
suite on GPU. Move to 1.6 only together with the image rebuild to torch >= 2.8.

**The general rule this earns:** an unpinned dependency can upgrade torch
transitively. When adding one to `requirements.txt`, check its torch
requirement against the image before trusting that "pip sees torch satisfied
and skips it".

**And an operational one, learned the hard way when it bit a live container.**
The `inference` service installs requirements as part of its *command*, so a
container that has been up for days is running whatever `requirements.txt` said
when it last started — uvicorn's `--reload` picks up new Python code but never
re-runs pip. Worse, `pip --user` writes into `/home/lab/.local`, i.e. the
container's writable layer, so `docker compose restart` keeps a bad resolution
(the shadowing torch 2.13 survived it). After changing `requirements.txt`:

    docker compose up -d --force-recreate inference

`--force-recreate`, not `restart`: only a fresh layer discards what the
previous resolution installed.

**A dependency failure is a run-level failure** (`check_dependencies()` in both
ALI engines). The missing `itk` above surfaced through the per-scan
`try/except`, so it was reported as if one patient's data were at fault: every
scan failed identically, each only *after* a complete histogram correction, and
the run ended on "ALI produced no landmarks for any scan" — which buries the
one line naming what to install. Both engines now import their whole lazy stack
once, before the loop, so the install message is what the caller sees and a
200-scan cohort does not discover it 200 times.

### 2026-07-31 — Real-data tests are opt-in; `test-gpu` service; ALI's GPU cap off 1

**Where the device is actually decided.** `inference` already runs on the GPU
(`DEVICE=${DEVICE:-cuda}` plus the nvidia reservation), and every tool reads
`settings.DEVICE`, falling back to CPU only when `torch.cuda.is_available()`
says so. Nothing hardcodes a device. The one service deliberately on CPU was
`test`.

**`test-gpu` (`docker-compose.yml`).** The unit tests stub every model and
gain nothing from a card, but `tests/test_data_integration.py` runs each tool
end to end against the real bundles under `DATA/` — minutes on a GPU, hours on
a CPU. A compose device reservation is all-or-nothing, so putting it on `test`
would make the pre-push hook fail outright on any clone without an nvidia
card. A second service instead, sharing everything else through a YAML anchor
so the two suites cannot drift. The hook keeps pointing at `test`.

**The real-data suite is now opt-in (`RUN_REAL_DATA_TESTS`), and had to
become so.** It was written to skip when `DATA/` is empty, which made it free
— but the moment a full ALI bundle lands, "skip" turns into eleven minutes of
GPU inference, or hours on the CPU the hook uses, because the request carries
no selection and ALI therefore predicts all 119 landmarks. `docker compose run
--rm test` was left hanging for hours. It now skips the module at collection
and stays ~10s; `test-gpu` sets the variable and runs the real thing. A
pre-push hook that takes hours is a hook people disable, and coverage nobody
runs is not coverage.

**`ALI_MAX_GPU_JOBS` 1 → 4.** Measured on the real bundle (RTX 6000 Ada): an
ALI CBCT run peaks at **256 MiB** of VRAM — one small DenseNet over a 64³ crop
— on a 48 GB card. At a limit of 1, two concurrent requests fully serialized,
the second waiting ~6.5 s per landmark for the first, for a resource neither
was close to exhausting. The default now matches `MAX_CONCURRENT_TOOLS`, which
makes it effectively "no extra limit" while still bounding things if that is
ever raised a lot. The figure is a property of the models, not of the card, so
4 × 256 MiB = 1 GB is safe on any GPU able to run this at all.
`AMASSS_MAX_GPU_JOBS` stays at 1: nothing here measured it, and a 3d_fullres
nnUNet is a different order of magnitude.

### 2026-07-31 — ALI (both engines) + CrownSeg

**Motivation:** port `ALI` — automatic landmark identification — from a pair
of Slicer CLI modules. It is the first tool with *two* engines that share
nothing but their output format, and the first whose IOS half depends on a
library the deployment image does not yet carry.

**One tool, two engine folders (`tools/ALI/src/{ALI_CBCT,ALI_IOS}/`).** One
entry in `GET /tools`, one `DATA/ALI/`, one Slicer module — mirroring the
original. `ALILogic.py` owns everything before inference (unpacking, DICOM
conversion, mode detection, the run report) so each engine only has to know
how to place landmarks. `src/markups/` holds the Slicer `.mrk.json` writer
both engines use; it sits beside them rather than in a tool of its own
because no *other* tool needs it yet.

**The mode is detected, not declared.** There is deliberately no `mode`
argument: a `.zip` can hold either kind of data and a DICOM series has no
extension at all, so nothing in the request distinguishes them — only the
data does. `ALILogic.detect` walks the input, and an archive holding both
kinds is a 422 rather than a guess. (This supersedes ALI_PORT_CONTEXT.md
§3.1, which proposed an explicit `mode` choice; the Slicer client was already
written against the detected-mode contract, and its 24 schema tests pass
against what shipped.)

The accepted cost is that the schema cannot say "this argument only applies
in mode X": `cbct_regions` and `ios_networks` are both optional, both always
rendered by the client, and one is inert on any run. Emptying the selection
for the mode that *actually ran* raises `ToolArgumentError` → 422 naming the
argument to fill in. That 422 is how a mode mismatch explains itself.

**`CrownSeg` is a tool, not a helper.** ALI's IOS engine needs a mesh
carrying per-tooth labels. The Slicer module got them by running the
`dentalmodelseg` executable out of Slicer's own bin — which is only the
console-script entry point of the `shapeaxi` PyPI package
(`dentalmodelseg = shapeaxi.dental_model_seg:cml`), so nothing needed porting:
`tools/CrownSeg/src/` calls `shapeaxi.dental_model_seg.main()` directly with
the namespace its `cml()` would have built. It lives in its own tool because
ASO, AREG and FlexReg call it too, and because ALI's IOS half needs pytorch3d
— inside ALI, one absent dependency would take four tools out of the registry
instead of one. ALI asks for segmentation without naming CrownSeg's data:
`model` is optional there and falls back to `settings.CROWNSEG_MODEL`. (The
library's own fallback downloads the checkpoint from GitHub mid-request; a
server holding patient data does not make outbound calls, so a missing file
is an error naming `setup-models.sh`.) shapeaxi's stdout is swallowed — it
prints "Saving results to <path>" per mesh, and that path is the patient's
own file name.

**Defects fixed by construction**, all of which lost results silently:

- **One unknown landmark cost the whole patient.** `LABEL_GROUPS[landmark]`
  was indexed with no guard inside the save loop, and its `KeyError` was
  caught far above — so nothing at all was written for that scan, including
  every landmark correctly found. The two spellings that triggered it (the UI
  said `UR3OI…`, the CLI `UR3OIP…`) are now aliases of one vocabulary, and
  `group_of()` cannot raise.
- **Homonyms overwrote each other in batch.** The patient key was
  `file.name`, so two `scan.nii.gz` in different subfolders collided twice
  over — in the working dict and in the flat output folder. Scans are keyed by
  path relative to the input root, and the output mirrors the input tree.
- **A missing mandibular IOS model was a silently-caught `KeyError`**, so the
  jaw vanished from the results. Reported now. The jaw must also be named
  explicitly in the checkpoint's name: "not Lower ⇒ Upper" meant a bundle
  missing its mandibular model quietly predicted the lower arch with the
  maxillary one. One naming rule (an `O`/`C` token plus an `Upper`/`Lower`
  one) replaces the two the UI and the CLI disagreed on — verified against
  the published archive, whose files are `Upper_O_model.pth` &co.
- **DICOM conversion wrote into the user's own folder** (`<input>/NIFTI/`),
  which the next run then re-ingested as input scans. Everything goes to the
  request scratch dir. Likewise the segmentation CSV, which the module wrote
  into the extension's own source tree.
- **`.stl` was accepted then ignored**: the UI counted them, the CLI globbed
  for `.vtk` only. Read for real now, and `surface_or_zip_file` (new
  `FILE_TYPES` entry — the only core edit) advertises exactly what discovery
  walks.
- **`R`, `RIP`, `OIP`** were selectable and predicted by nothing. Not offered.
- **`SaveId` was read by nothing**; `prediction_ID` is a real argument.
- **Output extensions disagreed** (`.mrk.json` vs `.json` for identical
  content, only the first of which Slicer recognises). Uniform, and **one
  file per scan** instead of one per anatomical region — the split forced
  every downstream tool to recombine them by hand.
- **`display.visibility: false`**, in both CLIs, ported faithfully at first and
  then corrected. It switches the markups *display* node off: Slicer loads the
  file, builds the node, lists it in the Markups module, and draws nothing —
  independently of each control point's own `visibility`, which was already
  true. Invisible inside the old module (which loaded the nodes itself and had
  a panel to toggle them), fatal the moment anyone opens a result file, which
  is exactly what a returned archive is for. Two tests pin it, because a valid
  file that renders nothing fails no other check.
- Two latent bugs in the search: `new_pos.all() > 0` reduced the array to one
  boolean *before* comparing, so it tested "no component is zero" and let
  negative coordinates through; and `Focus`'s convergence loop had no bound at
  all, which in a worker thread is a request that never returns. Also, the IOS
  masks were argmax'd over logits cast to `int16`, turning near ties into real
  ones that resolved toward the background channel.

**Sequencing (`ALI_PORT_CONTEXT.md` §8):** the CBCT engine runs today on
`monai` + `itk`, both added to `requirements.txt`. The IOS engine and
CrownSeg are written and tested but **cannot execute until the base image is
rebuilt on torch ≥ 2.8 with pytorch3d compiled in** — pytorch3d has no PyPI
distribution at all, and every shapeaxi release wants a newer torch than the
image's 2.5.1. Both are imported lazily, so ALI loads, publishes its schema,
and fails only an IOS *run*, with a message naming what is missing. Neither
is in `requirements.txt`: adding them would install nothing useful and could
shadow the CUDA torch build.

**Tests:** 37 for ALI, 20 for CrownSeg, all without GPU, weights, or network.
The agent's search and shapeaxi's segmentation are stubbed; everything around
them runs for real, including a full CBCT run through real preprocessing,
monai transforms and markups writing. Each test that pins a fixed defect says
which one in its docstring.

### 2026-07-30 — Dead-code and duplication cleanup

- `base.py` had an entire block (`FOLDER_TYPE`, `SCALAR_TYPES`,
  `CHOICE_TYPES`, `Selection`, `ResolvedPath`) declared TWICE, plus the
  remnants of the retired `SELECTION_TYPE` API (the constant, a second
  shadowing `choices` field, `choice_groups`, `multiple`,
  `_TRUE_TOKENS`/`_FALSE_TOKENS`). All removed. **The current API is
  `"choice"`/`"multichoice"`** (one `choices` dict of option → default);
  the 2026-07-28 entry below describes the interim design it replaced.
- `main.py._describe_argument` (unused, referenced the removed fields) and
  `file_utils.zip_directory` (unused duplicate of `make_zip`) removed.
- `requirements-amasss.txt` removed: torch/nnunetv2/vtk had been added to
  `requirements.txt` ("FIX : AMASSS functional"), leaving the file a pure
  duplicate. The heavy stack stays lazily imported, and torch stays unpinned
  so the image's CUDA build is never shadowed.
- The `test` service in `docker-compose.yml` (commented out earlier) is back
  under a `profiles: ["test"]` guard: `docker compose up` ignores it, the
  pre-push hook's `docker compose run --rm test` works again.
- Docs realigned with the code: `claude.md` → `CLAUDE.md`, `surgMovPred` →
  `SurgMovPred` (post-rename casing), README's selection-argument section
  rewritten for `choice`/`multichoice`, `ADDING_A_TOOL.md` §7 now describes
  the real requirements layout.

### 2026-07-28 — AMASSS tool + grouped selection arguments

**Motivation:** port `AMASSS_CLI.py` (CBCT skull structure segmentation,
nnUNet v2) from the Slicer extension to this server. AMASSS is the first
tool that is genuinely an *API*: AREG already calls it programmatically
today, and more modules will. It is also the first to need an argument the
schema could not express — "pick several structures from a list" — and the
first with a GPU deep-learning stack.

**Core additions (`base.py`, `main.py`):**

- Choice arguments: `ArgSpec.choices`, with the `"choice"` (exactly one) and
  `"multichoice"` (any number) types. One `{option name: on by default}` dict
  declares the options **and** their initial state, so a client renders the
  widget straight from `GET /tools` with no structure list of its own, and the
  defaults are written down once. Accepted on the wire as `"MAND,MAX"` or
  `{"MAND": true}`; an invalid option is a 422 naming what is allowed.
  This is a change to the *type system*, made once, not a per-tool change —
  the same category as adding a `FILE_TYPES` entry. Adding a structure the
  day its model ships is a one-line server edit with no client release.

  > **Superseded — see the 2026-07-30 entry.** This shipped as a
  > `SELECTION_TYPE` type with `choice_groups` / `multiple` / `default` fields
  > and a `Tool._coerce_selection` coercer accepting a JSON-list form. None of
  > that survived: the grouped-presentation metadata was never rendered, the
  > coercer was never written, the JSON-list form 422s, and the `default`
  > **field** silently shadowed the `default` **property** that `validate()`
  > relies on. The text above describes what the server actually does today.
- `FILE_TYPES["volume_or_zip_file"]`: one argument accepting either a single
  volume or a zip of a folder of them, since the schema cannot express
  "exactly one of these two arguments". Existing `is_file_type()` on the
  client (`endswith("_file")`) picks it up with no change.
- `file_utils.zip_directory`, the counterpart of `extract_zip`.

**The tool (`tools/AMASSS/`):** `AMASSS.py` declares the schema;
`src/AMASSSLogic.py` holds the pipeline; `src/nnunet_runner.py` isolates
inference; `src/vtk_export.py` handles surfaces. `segment()` is the reusable
API (returns a `SegmentationRun` with the produced files and a report);
`main()` is the thin HTTP adapter that zips it, and carries a note pointing
at itself as the only thing to change once multi-file responses land.

Three defects mattered specifically *because* this is a shared server, and
are fixed by construction rather than patched:

- the CLI set `os.environ['nnUNet_results']` before shelling out to
  `nnUNetv2_predict`. Tools run concurrently in worker threads and
  `os.environ` is process-global, so two overlapping AMASSS requests would
  have silently swapped model paths. `initialize_from_trained_model_folder`
  takes an explicit path — the race cannot occur.
- the CLI polled the output file's size and killed the predictor once it
  stopped growing for 3s, which could interrupt nnUNet mid-postprocessing.
  The Python API just returns.
- GPU work is serialized by AMASSS's own semaphore (`AMASSS_MAX_GPU_JOBS`,
  default 1), independently of `MAX_CONCURRENT_TOOLS` — four concurrent
  3d_fullres inferences do not fit on one card. (Read straight from
  `os.getenv` here; moved into `config.Settings` on 2026-07-30.)

Also corrected in the port: NRRD/GIPL inputs are now really converted via
SimpleITK instead of being renamed to `.nii.gz`; folder scanning is recursive
and excludes AMASSS's own previous outputs (the CLI re-ingested them on a
second run); structures with no shipped model (RC/TEETH/MCAN) are no longer
offered at all; a missing or failed structure is reported in
`AMASSS_report.json` instead of vanishing into a log line; `sys.exit()` is
gone. Inference now loads each checkpoint once per structure instead of once
per (scan x structure).

**Dependencies:** `numpy` + `SimpleITK` go in `requirements.txt` (imported at
module load). `torch`/`nnunetv2`/`vtk` go in a separate
`requirements-amasss.txt` and are imported **lazily**: `registry.py` imports
every tool at startup, so a missing heavy stack must not stop the server from
booting. It doesn't — only AMASSS fails, with an actionable message. The
deployment image already ships torch built against its CUDA version, which is
the other reason not to let a compose-time `pip install` shadow it.

**Tests (`tools/AMASSS/test/test_AMASSS.py`):** 35 tests with
`nnunet_runner.predict_folder` stubbed, so no GPU and no real models are
needed; everything around inference (discovery, output filtering, model
resolution, conversion, label merging, naming, report, and every accepted
selection wire format) runs for real.

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

### 2026-07-27 — surgMovPred: the model is server-side only, selected by name
**Motivation:** the client still had to provide the model as a zip upload (or
optionally pick a server-side one). The model should live exclusively in the
server's data store: the client asks for the list of available models
(`GET /tools/surgMovPred/data`, already existing) and sends only the *name*
of the chosen one — no model package ever travels from the client.

**Design (`server/tools/surgMovPred/surgMovPred.py`):** the `model`
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
`surgMovPred`'s `input`); on a scalar argument the server-side file is the
only option. `GET /tools` needs no change — it already exposes `type` and
`server_selectable`, which is all a client needs to render a dropdown of
server-side model names instead of a file picker (the Slicer client's
`ServerToolsCoreLib` does exactly that; see `SlicerAutomatedDentalToolsCloud`'s
ARCHITECTURE.md).

**Tests (`server/tests/test_main.py`):** an upload for `surgMovPred`'s
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
**Motivation:** tools like `surgMovPred` required the client to re-upload the
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
cleanup that applies to genuine uploads (`server/main.py`). `surgMovPred`'s
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
new `surgMovPred` tool returns a `.xlsx` file, and since an `.xlsx` is
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