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
- `ASO` — automated standardized orientation, CBCT and intra-oral scans.

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
│   │   └── ASO/
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

### 2026-07-31 — ASO ported: four modes, one tool, and the defects it inherited

**Motivation:** port ASO (Automated Standardized Orientation) from a 2587-line
Slicer widget plus four CLI modules onto this server. ASO is the step every
longitudinal study runs before anything else — put every scan in the same
coordinate frame — and AREG needs it programmatically.

**One tool, `server/tools/ASO/`, two engines under `src/`, four modes:**

|          | Semi-Automated | Fully-Automated |
|---|---|---|
| **CBCT** | your landmarks, ICP onto a gold set | landmarks predicted first, then the same ICP |
| **IOS**  | your landmarks, ICP per jaw | tooth centroids of an already segmented mesh |

`modality` and `automation` are explicit `choice` arguments, never inferred:
a `.zip` can hold either kind of data, and guessing wrong orients a patient
against the wrong reference and calls it a success. Every mode-specific
argument is `required=False`, with the cross-argument rules raised as
`ToolArgumentError` **before** any file is read (`ALI_PORT_CONTEXT.md` §3.1).

**Fully-Automated CBCT is wired but inert.** It needs landmark prediction,
which is ALI's job, and ALI is not deployed here yet — so that one mode answers
422 with a message naming the missing tool and pointing at Semi-Automated. The
seam is `src/ali_client.py`; the day `tools/ALI/` is registered, ASO needs no
change. **The call is in-process, not an HTTP request to our own `/run/ALI`,
and that is load-bearing**: a tool run holds one of `MAX_CONCURRENT_TOOLS`
slots for its whole duration, so four concurrent ASO runs each waiting on a
fifth slot would deadlock the server — `/health` included — until they timed
out. `Tool.invoke` is the same entry point `main.py` uses, validation included.

**Fully-Automated IOS takes already-segmented meshes only.** Crown segmentation
is `shapeaxi`/pytorch3d, absent from the image, and belongs to a future
`tools/CrownSeg` that ALI, AREG and FlexReg want too. `segment_unlabelled()` is
where it plugs in. **No** labelled mesh in the batch is a 422 (wrong mode);
*some* unlabelled is a per-patient report entry and the rest of the batch is
processed — "one of your forty meshes was bad" is not a reason to return
nothing.

**DICOM input is supported and writes nowhere near the caller's data.**
`convertdicom2nifti` wrote `<input>/NIFTI/`, which a later run then
re-discovered as input scans; its fallback path renamed an arbitrary earlier
output onto the current patient; and it only looked one directory down.

**The defects that cost data, all fixed by construction and each with a named
test:**

- **`SEMI_ASO_CBCT` could not work at all.** It read `data["tfm"]`
  unconditionally, but only the fully-automated chain ever produced one, so
  every patient of a semi-automated run died on a `KeyError` caught 90 lines
  above. Recentring now always runs, and the landmarks are moved into the
  centred space with it (`center_landmarks`), which is the state the ICP was
  always written to expect.
- **One landmark could lose a patient.** `GetDistDifference` indexed the
  reference's pairwise table with the *input's* keys, so a landmark present in
  one and absent from the other was a `KeyError`. The two sides are intersected
  first and what was dropped is reported, per landmark, with the reason.
- **Patient keys collided.** `GetPatients` keyed on the base name, so two
  `scan.nii.gz` in different subfolders became one patient — in the working
  dict and again in the flat output folder. Keys are paths relative to the
  input root and the output mirrors the input tree. It also stripped
  `_T1`/`_T2`, collapsing two timepoints of one subject into a single patient.
- **`MergeJson` merged a patient's landmark files by writing into the caller's
  input folder and DELETING the sources.** The merge is in memory.
- **A second run re-ingested the first.** `patient1_Or.nii.gz` sorts before
  `patient1_scan.nii.gz`. Previous outputs are set aside and used only when a
  patient has nothing else.
- **`UpperOrLower` defaulted to Lower**, so a maxillary mesh named
  `patient1.vtk` was registered against the mandibular reference and returned
  as a success. A file whose name does not say its jaw is now refused.
- **`Files_vtk_json.organise` paired with `vtk_name in json_name`**, so patient
  `1` matched patient `10` — and padded its list with a literal
  `"Upper_nioegfjhdfjkdffdhjmndfhnmdfhj"` sentinel. Exact stem, per directory.
- **Both jaws wrote the same `.tfm`.** `<patient>_SegOr.tfm` for Upper and
  Lower; the second silently overwrote the first. Named per jaw.

**Concurrency, which only matters because this is a shared server:**

- `InitIcp` wrote `source.npy`/`target.npy` into **its own installed package
  directory** (`ASO_IOS_utils/cache/`) and re-`np.load`ed one of them on every
  iteration of a 2500-iteration search. That is a write into the install tree,
  thousands of pointless round trips per patient, and — the path being fixed —
  two concurrent requests overwriting each other's landmarks. The search is
  pure and in memory (`src/geometry.py`, shared by both engines, which had
  carried two drifted copies of it).
- The triplet search drew from the **global** `numpy` generator, so the same
  request gave a different orientation every time and concurrent requests
  consumed each other's state. Every ordered triplet is now enumerated when
  there are at most `ASO_ICP_MAX_TRIPLETS` of them (7 landmarks is 210), which
  is deterministic, faster *and* better than sampling; above that a local
  generator seeded with `ASO_ICP_SEED` is used.

**Latent bugs found while reading, each now a test:** `np.arccos` of a dot
product rounding past 1.0 gave NaN that propagated silently through the
rotation matrix (`pre_icp.py` clamped one end only); `RotationMatrix` divided
by a zero-length axis for two already-parallel vectors; `ASO_IOS_utils/utils.py`
defined `PatientNumber` twice, the second shadowing the first; `WriteSurf` used
`vtkPolyDataWriter`'s ASCII default (bigger, ~130x slower to parse, and *less*
accurate — see the 2026-07-30 entry); the `.off` reader referenced an undefined
`line` on its single-vertex branch; and `SEMI_ASO_IOS` wrote every transform as
`matrix_file_0.npy`, `matrix_file_1.npy` because an `isinstance(file, dict)`
guard was false on the path that mattered. **The IOS matrix composition order
was also backwards** (`M_init @ M_icp`, where the ICP runs on points the
initialisation has already moved); the CBCT engine always had it right, which
is what makes it a transcription slip rather than a choice.

**Also removed rather than ported:** `PRE_ASO_CBCT`'s `model_folder`, `SmallFOV`
and `temp_folder` arguments (read, never used, since the learned orientation
step was removed upstream), the `<filter-progress>` prints, `sys.exit()`,
`tqdm`, the `time.sleep(0.2)` progress theatre (0.6s per patient), the
`*Error.txt` files written into the output folder, the `if not os.path.exists`
skip-if-exists guards, and the reference *scan* the CBCT ICP read and never
used — which it nonetheless required, so a reference bundle holding only the
landmarks it registers against died on an `IndexError`.

**`scripts/data-manifest.yml`:** the ASO block dropped 600 MB nothing reads —
`PreASOModels` (the removed deep-learning step), `identification_landmark_ios_model`
(fully-automated IOS runs no landmark network) and `segmentation_model` (the
blocked CrownSeg path) — and gained the seven ALI landmark bundles the default
reference planes are built on, curated to 758 MB rather than all 112 at ~12 GB.

**Core changes, both sanctioned:** one `FILE_TYPES` entry (`surface_file`), and
three `config.Settings` fields (`ASO_ICP_MAX_TRIPLETS`, `ASO_ICP_SEED`,
`ASO_LANDMARK_TOOL`). No GPU: everything ASO computes is SimpleITK/VTK/numpy,
so it needs no semaphore of its own. `main.py` and `registry.py` untouched.

**Tests:** 195 server tests, 65 new — 64 in `tools/ASO/test/test_ASO.py` (the
landmark seam stubbed, everything else real, against synthetic volumes, DICOM
series and meshes) and one in `tests/test_main.py` asserting every extension
`surface_file` advertises is accepted on upload and a `.txt` is a 400.

Two of those cover the outputs a clinician actually relies on, and both hold to
the float: **the written landmarks land exactly where the resampling put the
voxels** (volume and markups are moved by two different code paths — if they
disagree the markups file opens floating beside its scan, and nothing in the
report would say so), and **the `.tfm` maps ORIENTED -> ORIGINAL**, recentring
included. That direction is asserted rather than assumed because getting it
backwards is silent: the file still loads, and still transforms.

### 2026-07-30 — AMASSS surfaces: binary, and decimated by default

**Motivation:** a five-structure run with surfaces returned a 41.9MB archive
that Slicer could not open — the client froze on the main thread and the user
read it as "the server never sent the .vtk". It had: `curl` against the same
endpoint pulled all 41,889,544 bytes with a correct `Content-Length` and every
mesh re-read cleanly. The transfer was never the problem. The **geometry** was.

**Marching cubes runs on the original scan grid**, so a 0.33mm CBCT produces a
triangle per voxel face: 1.6M for a cranial base, 3.5M across five structures,
11.8M for a merged nine-structure volume. That is not detail — the mask
underneath is only accurate to about half a voxel — it is just resolution
nobody asked for, and it is what made the results unusable to ship and to open.

**`vtkPolyDataWriter` was writing ASCII** (its default), which is where the
bulk went: 848.5MB for the merged surface alone against 6.4MB for every
segmentation in the same run. Binary is the same geometry — and the *more*
accurate of the two, which is the opposite of the reflex: it round-trips the
float32 vertices exactly, while ASCII prints ~6 significant digits and moved
points by up to 5e-05mm on read-back. It is also 133x faster to parse (a 1.6M
triangle cranial base: 2.67s ASCII, 0.02s binary).

**`surface_decimation` (new schema argument, default 90).** `vtkDecimatePro`
with `PreserveTopologyOn`, applied after smoothing and *before* the per-cell
colour array is built — the array is sized to the mesh that is actually
written. Measured on the cranial base, against a 0.33mm voxel:

| reduction | triangles | mean dev | p95 | max |
|---|---|---|---|---|
| 50% | 811,222 | 0.0034mm | 0.004mm | 0.277mm |
| 80% | 324,488 | 0.0338mm | 0.125mm | 0.493mm |
| **90%** | **162,244** | **0.0590mm** | **0.171mm** | 0.692mm |
| 95% | 81,122 | 0.0951mm | 0.264mm | 1.223mm |

90 costs a fifth of a voxel on average and buys a factor of ten. 0 keeps the
raw mesh. The value is recorded in `AMASSS_report.json` — these surfaces are
lossy by default now, so a run has to say by how much.

**End to end, the same five-structure request, real HTTP:** archive
41,889,544 -> **5,417,443 bytes** (7.7x), triangles 3,519,420 -> **351,938**
(10x), total client-side mesh parsing 2.7s+ -> **0.01s**. Decimation adds ~12s
of server time for five structures.

**A caveat measured and worth keeping:** binary alone did NOT shrink the
download. DEFLATE was already squeezing ASCII at 6.2:1 and binary only
compresses 2.7:1, so the archive went 223.4MB -> 227.8MB on a nine-structure
run. Binary pays off in disk, RAM, write time, zip time and parse time — not
on the wire. Only removing geometry moved the download, which is what
decimation does.

**Still on the table, in the client repo:** `AMASSS.py`'s
`MAX_RESULTS_TO_LOAD = 12` caps by *file count* while the cost is in triangles,
so a 10-file run loaded 3.5M triangles on the UI thread while a 20-file run of
small masks was correctly skipped. Decimation makes this survivable; it does
not make the cap correct.

**Tests:** 119 server tests (4 new: binary header + exact round trip, temp file
removal, decimation reduces triangles with 0 disabling it and the colour array
matching the final cell count, and the report field).

### 2026-07-30 — AMASSS: the GPU was idle seven eighths of the run

**Motivation:** AMASSS looked GPU-bound and was not. Profiling one structure on
a 512x512x365 CBCT at 0.33mm: **14.6s** resampling the input to the model's
0.4mm grid, **4.5s** of actual inference, **6.9s** resampling the logits back.
Both resamplings are scipy splines pinned to a single core. The card was doing
an eighth of the work and holding 2.7GB of 48GB.

**The tempting fix was the wrong one, and was measured before being discarded.**
At a 128^3 patch the network already saturates the SMs at batch 1: throughput
is flat at ~43 patches/s from batch 1 through batch 12. Cutting the GPU's work
by 5x (`tile_step_size` 0.5 -> 1.0) moved the total from 37.3s to 34.5s and
dropped utilisation from 36% to 11% — direct evidence the GPU was never the
constraint. **Free VRAM is not convertible into speed here; idle time is.**

**GPU resampling (`nnunet_runner._enable_gpu_resampling`, new setting
`AMASSS_GPU_RESAMPLING`, default on).** nnUNet already ships torch versions of
both resamplers, so nothing is reimplemented — only selected. It is selected by
NAME: nnUNet resolves `resampling_fn_data` / `resampling_fn_probabilities` out
of the configuration dict via `recursive_find_resampling_fn_by_name`, so
rewriting those two strings redirects both ends with no monkeypatching. Two
things make that mutation safe, and both would have been silent bugs:
`PlansManager` hands out a `deepcopy` of the configuration, so it touches
neither the shared plans nor a concurrent request — and consequently the
`torch.device` placed in the kwargs never reaches the `plans.json` nnUNet writes
beside its output, which `json.dump` could not serialize. The two properties are
`@property @lru_cache`, so the swap clears them.

The GPU path uses `predict_from_files_sequential`: `predict_from_files` fans
preprocessing and export out to *spawned* processes, each of which would need
its own CUDA context to run a GPU resampler. That trades away the CPU/GPU
overlap on multi-scan batches — recovering it with a reader thread feeding the
GPU is the obvious next step, and the reason `num_processes_*` were left at 2.

**Measured end to end, real models, default five structures, one scan:
195.9s -> 77.0s (2.5x).** Per structure 34.2s -> 13.2s.

**It is not numerically free, and the defaults say so.** torch has no 3D cubic
interpolation, so the input resampling drops from spline order 3 to order 1.
Dice against the scipy pipeline: MAND 0.998, UAW 0.997, MAX 0.995, CB 0.991,
**CV 0.978**. The cervical vertebra is consistently the outlier — thinnest
structure, closest to the edge of the field of view. `AMASSS_GPU_RESAMPLING=false`
restores bit-identical nnUNet output, and a bundle whose plans pin a non-default
resampler opts itself out automatically. `AMASSS_report.json` now records
`gpu_resampling` and `tile_step_size`, because a mask is only reproducible
next to the values that produced it.

**`AMASSS_TILE_STEP_SIZE` (default 0.5, unchanged).** The one knob here that
moves the segmentation for a *pure* speed gain, so it is exposed rather than
tuned: 0.7 measured Dice 0.995 against 0.5 and saves ~2.5s of GPU per structure.

**`_convert_to_nifti` stopped casting to float32.** The cast was never what made
the conversion real — the read and write are — and it doubled the bytes gzipped
per scan and gunzipped again by nnUNet, costing 2.4s + 0.4s for nothing:
nnUNet's reader casts to float32 itself, and int16 CBCT values are exact there.

**`vtk_export` cleanups.** Cell colours are built in numpy instead of a
`SetTuple` per cell (identical bytes, 32x, but only ~80ms on a mandible — the
honest win is small). The marching-cubes temp file had a *fixed* name, so every
surface in a run wrote over the same path; it is now unique per call and removed
after use rather than held until request cleanup.

**Tests:** 115 server tests (5 new, covering the cpu skip, the non-default-plans
skip, the swap itself including cache invalidation, the report fields, and the
preserved voxel type).

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