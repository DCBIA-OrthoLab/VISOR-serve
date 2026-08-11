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
- `ASO` — automated standardized orientation, CBCT and intra-oral scans. Its
  fully-automated CBCT mode calls `ALI` in-process for the landmarks.
- `ALI` — automatic landmark identification, on CBCT volumes (deep-RL agents)
  or intraoral surface scans (multi-view rendering + 2D UNet). The engine is
  chosen from the data, not from an argument.
- `CrownSeg` — per-tooth labelling of intraoral scans (shapeaxi). Its own tool
  rather than a helper inside ALI, because ASO, AREG and FlexReg need it too.
- `AREG` — registering two timepoints of the same patient onto each other.
  CBCT (masked elastix) and IOS (palatal-patch ICP); its automated modes drive
  AMASSS, ASO and CrownSeg in-process.

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
│   ├── main.py              # FastAPI app: /run/{tool_name}, /uploads, /results
│   ├── base.py              # Tool base class, ArgSpec, ToolArgumentError
│   ├── registry.py          # auto-discovery of Tool subclasses in tools/
│   ├── data_store.py        # DataStore interface + LocalDataStore (server-side models/testfiles)
│   ├── file_utils.py        # shared helpers: zip extraction/creation, tabular loading
│   ├── transfer.py          # chunked resumable uploads, range-served results
│   ├── security.py          # Bearer token verification
│   ├── config.py            # config from environment variables
│   ├── tools/               # one folder per tool: tools/<name>/<name>.py (+ src/, test/)
│   │   ├── test_tool/
│   │   ├── example_tool/
│   │   ├── SurgMovPred/
│   │   ├── AMASSS/
│   │   ├── ASO/
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

### Chunked transfer (`transfer.py`)

A file arriving in one request rides one TCP connection, and one connection to a
remote client is bound by its congestion window long before it is bound by
bandwidth, which is why a 100 MB CBCT took minutes and a connection dropped at
95% started again from zero. These endpoints let a client use several at once.
All Bearer-protected, all optional: a client that ignores them still works, and
a client that uses them against an older server falls back on the `404`.

4. `POST /uploads` `{filename, size, chunk_size?}` → `{upload_id, chunk_size,
   part_count}`. The server clamps `chunk_size` to [1, 64] MB and **answers with
   what it used**: part `n` is always `[n * chunk_size, (n+1) * chunk_size)`, and
   both sides compute offsets from that one number. A file over `MAX_UPLOAD_MB`
   is refused here, before a byte of it travels, which the multipart path
   cannot do.
5. `PUT /uploads/{id}/parts/{n}`, raw body, no multipart framing. `os.pwrite`
   at the part's offset into a pre-`truncate`d (sparse) blob, so concurrent
   parts write disjoint ranges of one file and there is **no reassembly pass**:
   each uploaded byte is written to disk exactly once. Idempotent, re-sending a
   part is how a client resumes. `X-Part-SHA256` is verified before anything is
   written, so a bad part is one retried part; since the parts tile the file,
   that verifies the whole upload without either side making a second pass.
   `Content-Encoding: gzip` is honoured (worth ~3x on an uncompressed `.nii` or
   a `.vtk`), and the checksum covers the *decompressed* bytes, what lands on
   disk, not what travelled.
6. `GET /uploads/{id}` → `missing_parts`. What makes a transfer resumable.
7. `DELETE /uploads/{id}`.
8. `GET /results/{id}`, honouring `Range` → `206`. `POST /run` hands back a
   *reference* instead of the bytes when the client sends
   `X-Result-Delivery: reference` **and the result is at least
   `RESULT_REFERENCE_MIN_MB`**; `DELETE /results/{id}` releases it.

   That threshold is about cleanup, not speed. A streamed `FileResponse`
   deletes its file through a `BackgroundTask` when the response ends, and that
   fires even when the client disconnects mid-body (measured), so it depends on
   nothing the client does. A reference does depend on the client: the file
   waits for a `DELETE`, or for the reaper. Parallel ranges buy nothing under
   16 MB, so there is no reason to trade the stronger guarantee away for one,
   and the overwhelming majority of runs keep exactly the cleanup behaviour they
   have always had.
9. In `POST /run/{tool_name}`, an input that came up this way is named in the
   reserved `__uploads__` form field (`{argument name: upload id}`) instead of
   being sent as bytes. Its blob is **renamed** into the request's work dir, not
   copied, same filesystem, so a 2 GB upload becomes a tool's input in
   microseconds. Extensions are validated identically on both routes.

State lives on disk (a `meta.json` written once and never mutated, plus a
zero-byte marker file per received part), not in a module global: part `n` and
part `n+1` of one upload may legitimately be served by different `uvicorn
--workers`, and a session has to survive the `--reload` a code edit triggers
mid-transfer. It also means no lock anywhere, parts never overlap, and a marker
is created with `O_EXCL`. Ids are `secrets.token_urlsafe(24)`, matched against
`[A-Za-z0-9_-]{16,64}` *before* any path is built from them.

**Cleanup, and why it is a timer.** A reference is the one thing here that
survives its request, so it needs a bound that does not depend on the client
behaving. Three layers, in the order they normally fire:

1. The client `DELETE`s the result as soon as it has it, from a `finally`, so a
   download that failed halfway or an archive that failed its integrity check
   releases it too. Retried once.
2. `transfer.reap_expired` runs **on a timer** (`_reaper_loop`, every
   `TRANSFER_SWEEP_SECONDS`), not only opportunistically when a session is
   created. That opportunistic sweep alone was a hole: the case where an
   abandoned transfer sits longest is exactly the case where no new request
   arrives to trigger one, so an idle server would have held it indefinitely.
3. `TRANSFER_TTL_SECONDS` is an **idle** timeout, not an age limit. Every part
   written and every range read stamps its directory (`transfer.touch`), so a
   transfer still in flight is never at risk however long it takes, while one
   whose client vanished expires 15 minutes later. That is what lets the number
   be minutes instead of the hours an age limit would need in order to survive
   the slowest imaginable transfer.

Worst case, therefore, for patient data left on disk by a client that died
mid-download: `TRANSFER_TTL_SECONDS + TRANSFER_SWEEP_SECONDS`, about 16
minutes, with no upper bound depending on when the next request happens to
arrive.

### `security.py`, `config.py`
- Bearer token from env (`API_TOKEN`), constant-time compare, `401` on failure.
- Config from env: `API_TOKEN`, `DEVICE`, `MAX_UPLOAD_MB`, `MAX_EXTRACTED_MB`,
  `TEMP_DIR`, `MAX_CONCURRENT_TOOLS`, `AMASSS_MAX_GPU_JOBS`, `ALLOWED_EXTENSIONS`,
  `DATA_DIR`, `DATA_BACKEND`, `UPLOAD_CHUNK_MB`, `TRANSFER_TTL_SECONDS`,
  `TRANSFER_SWEEP_SECONDS`, `RESULT_REFERENCE_MIN_MB`. Sensible dev defaults.
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

### 2026-08-10 — ALI gains the mucogingival network, so AREG stops asking for landmarks

The entry below shipped AREG's lower-arch registration with the 13 mucogingival
landmarks as a *required upload*, because this server's ALI had the Occlusal and
Cervical networks only — the wrong shape for a tool whose point is that the
computation happens here. Ported from upstream's `ALI_IOS_utils/{model,agent}.py`.

- **MG is a different network, not a third set of channels.** Occlusal and
  Cervical take one rendered image per camera as a batch (4 channels in, 4
  classes out). MG stacks its **three** buccal views into ONE input of 12
  channels and predicts 3 classes: it has a single landmark to find and needs
  the three views together. `_build_network` declares both shapes rather than
  inferring one, every argument there being part of the checkpoint's shape.
- **Its cameras are aimed per tooth, and that is why it works at all.** The
  crown networks orbit a tooth on a fixed sphere of directions, but the
  mucogingival point is buccal to the crown and "buccal" is a different
  direction for every tooth: measured against the true normal, the sphere
  scheme is off by 2-5° on the incisors but **35° on teeth 19/30 and 53° on
  tooth 31**, so on the molars the cameras looked along the arch and the
  landmark was not in the image. The arch tangent comes from the neighbouring
  teeth's centroids, the buccal normal is its horizontal perpendicular, and the
  cameras aim at `MG_AIM_OFFSET`, a prior measured on upstream's 155 scans.
- **Three things are deliberately not the crown path**, each of which is why a
  naive version returns nothing: the predicted faces are NOT filtered to the
  tooth (`faces_on_tooth` keeps only faces carrying that tooth's label, and the
  landmark is on gingiva — it would drop every one); the argmax is on the raw
  logits, never on an int16 cast; and a tooth that wins no pixel falls back to
  its 50 most likely ones rather than leaving a hole in a line AREG then fits a
  spline through.
- **A landmark that had to be helped says so, in the file.** `markups.write`
  takes a `descriptions` mapping, so a point placed from an arch fit ("tooth
  not segmented") or forced ("confidence 0.31") carries the reason in its own
  control point, and the report lists them under `landmarks_degraded`. A
  degraded point is indistinguishable from a good one otherwise.
- **Mucogingival is OFF by default in `ios_networks`**, unlike the other two:
  it is one point per lower tooth wanted by AREG and by nobody asking for crown
  landmarks, and on by default it would add a third pass over every mesh of
  every existing request. It is also declared lower-jaw only (`NETWORK_JAWS`),
  so a maxilla is skipped rather than reported as a missing checkpoint.
- **AREG's side is one seam.** `mgl_landmarks` became optional: absent, AREG
  calls ALI in-process for both timepoints, merges the two into one folder
  (they cannot collide, being keyed on a scan key carrying the timepoint) and
  paints from that. A deployment without ALI answers 422 saying to send them.
- The model bundle moved with it: `data-manifest.yml`'s ALI IOS entry points at
  the 2026-08-05 release (Occlusal, Cervical, Mucogingival, 236 MB). The MG
  file is `Lower_MG_v6.pth`, which `discover_weights`' single naming rule reads
  with no special case.

**Verified live over HTTP**, on the reference mandible published with ASO's IOS
gold files (101,463 points, PredictedID labels 18-31):

* `POST /run/ALI` with Mucogingival alone → **all 13 landmarks in 9 s**, none
  forced, none estimated, tracing a symmetric arch;
* `POST /run/AREG` on a T1/T2 pair displaced by a known 3.5° + 2.7 mm, **sending
  no landmarks at all** → 200 in 5.5 s, report saying
  `"mgl_landmarks": "predicted by 'ALI'"`. The patch covers 21% of the mesh with
  **0 vertices on a crown**, and a 3.077 mm mean displacement comes back with a
  residual of **0.004 mm mean, 0.008 mm max**. That also settles the open
  question from the entry below: the ~1.1 mm residual measured there was the
  synthetic swept ridge having no relief for the ICP to lock onto, not the port.

**Debugged against the caller's own data afterwards, and it found two things.**

**A landmark file could not be named the way people name them.** The matching
rule was a BLACKLIST of words to strip from a file name (`mg`, `pred`, `lm`), so
`H10_T1_L_MG_edited.mrk.json` — a hand-corrected file sitting right beside
`H10_T1_L.vtk` — reduced to `H10_T1_edited`, matched nothing, and was reported
missing. A blacklist can never cover `_edited`, `_corrected`, `_v2` or someone's
initials. The rule is now "the scan's tokens are a PREFIX of the landmark file's
tokens", which is what people already do when naming these, with an ambiguity
refused rather than guessed. It stays safe because the comparison is on whole
tokens: `('p1',)` is not a prefix of `('p10', ...)`.

**And the port reproduces the Slicer module exactly.** On the caller's real H10
mandible pair with their own edited landmarks, the registered T2 came out
**0.53 mm** from their Slicer output. The whole difference was the band height:
their run used 4 mm, the default here is 5. At 4 mm the patch is identical point
for point (Jaccard 1.000, 15,330 of 15,330 vertices) and the registered mesh
lands **0.0000 mm mean, 0.0001 mm max** from theirs. Also measured on that pair:
registering on ALI's predicted landmarks instead of the hand-corrected ones
moves the result by **1.08 mm** on average.

**Not ported:** upstream's `--force_topk` / `--no-force_landmarks` switches. The
forcing is always on here (`MG_FORCE_TOPK = 50`, upstream's default), the only
caller being AREG's spline, which wants as many of the 13 points as it can get
and now has the per-point caveat to judge them by.

**Tests:** 409 server tests (+6 in ALI's suite), plus AREG's two seam tests.

**Also fixed, and it was the test suite's fault rather than the server's.**
`file_utils.make_scratch_dir` registers what it hands out with the REQUEST being
served, and `main.py` is what deletes it — so a test calling `AREGLogic.main()`
directly has no request and nothing cleans up. The suite had been leaving one
scratch directory per such call in the server's real `TEMP_DIR`. An autouse
fixture points `TEMP_DIR` at the test's own `tmp_path`; verified from both ends.

### 2026-08-10 — AREG registers the LOWER arch too, on the mucogingival line

The rest of AREG was ported from `SlicerAutomatedDentalToolsCloud`, which
predates upstream's `AREG_IOS` MGL mode. Ported now from
`AREG_IOS/AREG_IOS_utils/mgl_patch.py` and the `RunMGL`/`SortLower` half of its
driver.

**The upper arch is registered on the palate; the mandible has no palate.** It
has the mucogingival line, where attached gingiva meets alveolar mucosa: ALI's
MG model places 13 landmarks along it, they are joined into a spline, every
sample is snapped onto the mesh, and a band grows around the curve. That band
plays the same role as the palatal patch and is written as a 0/1 point array of
the same shape — under its own name, `Bottom_MGL`, so a mandible is never
labelled after the palate.

- **Picking the patch picks the arch, so it is one `choice`, not a toggle.**
  `ios_patch` selects Palate (upper) or Mucogingival line (lower); with the
  palate the mandible is carried along by the maxilla's transform, and with the
  mucogingival line the maxillae are left untouched, as upstream leaves them.
- **It needs no network at all**, which is the operational point: the palatal
  patch needs pytorch3d, and this is a spline, a shortest-path walk and a label
  lookup. `net.check_dependencies()` is not called on this path, so a
  deployment without pytorch3d answers 501 for the upper arch and registers
  lower arches at full speed.
- **Two properties of the band are load-bearing, both upstream's:** every spline
  sample is snapped onto the mesh, because a curve interpolated between
  landmarks floats off the surface in the concavities between teeth; and the
  band grows **geodesically**, never through space, so a buccal patch cannot
  leak onto the lingual side wherever the ridge is thinner than the radius.
  `_adjacency` now reads the numpy adjacency `postprocess.Adjacency` already
  builds instead of walking every cell through a `vtkIdList`.

**The defect an end-to-end run caught, in this port's own code.** The MG
landmarks are **per scan** and were keyed **per patient**: both timepoints' files
routinely sit in one folder, so `P1_T1_..._MG_Pred.json` and
`P1_T2_..._MG_Pred.json` collapsed to one ambiguous entry and every MGL run
failed with "2 landmark files match 'P1'". A patient has one key and two scans;
`landmarks.scan_key` keeps the timepoint token that `pairing.patient_stem`
strips. Nothing in the unit tests would have found it — one had encoded the
wrong behaviour as if it were correct — and the first real request did.

**Upstream's own matching is not ported.** `FindLandmarkFile` fell back to any
json whose name merely CONTAINS the scan's stem, taking `sorted(...)[0]` with a
warning — so with `P1`'s file missing, `P1` takes `P10`'s and registers a
mandible against another patient's mucogingival line while reporting success.
Files are matched through the shared key rule, and an ambiguity is an error
naming the candidates.

**`SortLower` is why the pairing gained a `registered_jaw`.** `Sort` only kept a
lower pair when the matching UPPER pair existed, so a study that only scanned
mandibles paired to nothing at all.

**Observed, not changed:** `LOWER_TOOTH_LABELS = range(18, 32)` leaves the two
lower third molars (17 and 32) in the band if it ever reaches them, which reads
like an off-by-one on both ends of "the lower teeth" — and the synthetic
mandible used to verify this port has 2 vertices of a tooth-32 crown inside its
patch for that reason. It is also precisely the span ALI's MG model is trained
on, so it may be deliberate. Kept as-is: widening it changes which vertices
drive a clinical registration, and that is the upstream author's call.

**Verified live over HTTP:** a lower-arch pair displaced by a known 4° + 2.6 mm
rigid transform, 200 with the registered mandibles, the transform and the
report; the maxillae untouched; the patch confined to -4.5..+4.8 mm around a
line asked for a 5 mm half-height. The residual on that phantom is ~1.1 mm,
which is ICP on a mathematically swept ridge with almost no relief — the same
code registers a flat labelled patch to 2e-5 mm. No real IOS pair with MG
landmarks was available, and no clinical accuracy is claimed.

**Tests:** 403 server tests (+22).

### 2026-08-10 — AREG ported: five modes, and an 8 mm transform nobody could see

Port AREG (Automated REGistration) from the Slicer extension's `AREG/` module
and its `AREG_CBCT` / `AREG_IOS` CLIs. AREG is the step every longitudinal study
runs *after* ASO, and the first tool here that is mostly **other tools**: its
automated modes drive AMASSS, ASO and CrownSeg.

|          | Semi-Automated                  | Fully-Automated               | Oriented + Fully-Automated |
|----------|---------------------------------|-------------------------------|----------------------------|
| **CBCT** | your T1 masks, masked elastix   | AMASSS segments the T1 masks  | ASO orients the T1 first   |
| **IOS**  | your segmented, oriented meshes | CrownSeg labels + ASO orients | —                          |

`modality` and `automation` are explicit `choice` arguments, never inferred. The
pair is validated (`IOS` has no oriented mode) and answered with a 422 naming
what that modality offers, because the schema can hide an *argument* per mode
but not one *option* of a choice.

The calls into AMASSS/ASO/CrownSeg go through `registry.TOOLS[...].invoke`, not
HTTP and not an import (`src/tools_client.py`). HTTP is a deadlock and bites
harder here than for ASO, an AREG run chaining up to three tool slots. The
registry rather than a direct import because AREG needs the *availability*
answer as much as the result: a deployment may legitimately not carry AMASSS,
and AREG must then answer "use Semi-Automated and send your own masks".

**The defect that mattered most is one line, and it is worth 8 mm.** elastix
reports a rigid result as three Euler angles, a translation, **and a centre of
rotation** — the transform is `y = R(x - c) + c + t`. `MatrixRetrieval` dropped
`c`, building a SimpleITK `Euler3DTransform` rotating about the physical origin:
a different transform by exactly `(I - R)c`. Measured against a known
ground-truth transform on a phantom whose origin is at (-140, -90, 60) mm:

| | max error vs. ground truth |
|---|---|
| centre dropped (what shipped) | **8.371 mm** |
| centre honoured (here) | **0.025 mm** |

It is invisible on data that happens to be centred — exactly what the oriented
mode produces, since ASO recentres — and silently wrong on data that is not.
When `c` is the origin the two agree to the float, so the fix cannot change a
case that was already right. The same defect made
`AREG_IOS_utils.transformation.read_matrix` build its 4x4 from
`GetTranslation()`; SimpleITK has no accessor returning the composed offset, so
`t + c - Rc` is computed rather than read, and a test pins it.

**Two lines removed rather than "fixed", both after measuring.**
`ImagePyramidSchedule = 8,8, 4,4, 2,2` is six values for a *three*-dimensional
image over *three* resolutions, where elastix wants nine — and elastix does not
error, it discards a mismatched schedule and uses its default. Verified three
ways: the original six-value schedule and no schedule at all give bit-identical
results (0.025 mm), a corrected nine-value 8/4/2 schedule gives a different one
(0.113 mm). Deleting the dead line keeps the behaviour every published result
was produced with. `ErodeMask=true` is likewise inert (it applies to an elastix
mask, and none is ever set) and is kept with a comment saying so.

**The T2 pre-centring is gone, and with it the reason the `.tfm` was unusable.**
The original resampled every T2 onto a recentred grid before registering, then
wrote the transform between the T1 and *that* — while the recentred copy lived
in a `<t2_folder>_Center` directory next to the caller's own data and was never
returned. `AutomaticTransformInitialization` aligns the two volumes' centres by
itself, so the pass bought nothing and cost every T2 an extra interpolation. The
transform now maps the T1 frame to the T2 frame the caller sent, and a test
asserts it by resampling the *original* T2 with the returned `.tfm`.

**The defects that cost data, all fixed by construction and each with a named
test:**

- **`"cb" in basename.lower()` makes every file whose name contains CBCT a
  cranial-base mask.** So does `"max"` for a patient called MAX_01, and `"md"`
  for almost anything. Masks are matched on whole tokens of the stem, and must
  say **both** that they are a segmentation and which structure they cover.
- **Patient keys collided.** `GetPatients` keyed on the base name, so
  `scan.nii.gz` under two subject folders became one patient. Keys are paths
  relative to the input root and the output mirrors the input tree.
  `.split(".")[0]` also truncated a name at its first dot.
- **A second run re-ingested the first.** `P1_CB_Reg.nii.gz` sorts before
  `P1_scan.nii.gz`. Previous outputs are set aside and used only when a patient
  has nothing else — and the suffix is matched as a trailing *token*, so a
  patient called Regina is not a previous run of suffix "Reg".
- **The masked fixed image was written to `<temp>/fixed_image_masked.nii.gz`** —
  one fixed name shared by every patient of a run and every concurrent request,
  so two overlapping runs registered against each other's anatomy. The
  conversion is in memory; a test asserts `TEMP_DIR` is untouched.
- **A mask of a different geometry was forced into agreement**
  (`fixed_seg.SetOrigin(...)`, unconditionally), and a `SegmentationLabel` the
  mask did not hold fell through to using the *whole* mask. Both are per-patient
  failures naming the problem now.
- **IOS: a mesh whose name did not say its jaw was treated as a lower arch.**
  `Sort` split on "is it Upper" and defaulted everything else to Lower. The
  Upper vocabulary also matched the bare `_U`/`U_` as substrings, so a patient
  identifier like `P_U12` was an upper arch whatever the file held.
- **`vtkICP.__call__` returned its source unmoved** — it built a
  `vtkTransformPolyDataFilter`, ran it and threw the output away — so every
  method after the first in `ICP`'s list ran on unaligned points while its
  matrix was composed as if it had not. One method was ever configured.
- **`RemoveIslands(surf, labels, 33, 500)`** is the first of four
  post-processing steps and has never run: the label array is binary. Not
  ported.
- **Background pixels voted for the last face.** pytorch3d's pixel-to-face map
  is **-1** where a pixel hit nothing, and -1 indexes the last element; those
  pixels were zeroed before a softmax that turned the zeros back into an even
  0.5/0.5 vote.
- Two latent numerical bugs in the canonical orientation: `np.arccos` clamped at
  +1 only, so a dot product rounding past -1 gave NaN that propagated into every
  vertex; and `RotationMatrix` normalised an axis it never checked, so two
  parallel vectors divided by zero. Also `reshape(-1, 4)` on a non-triangle
  mesh, which does not fail — it reads the wrong point indices.

**Also removed rather than ported:** the `ApproxReg` argument (passed to
`VoxelBasedRegistration` and never referenced inside it), the
`<filter-progress>` prints and their `time.sleep(0.2)` (0.6 s per patient), the
log file the widget polled, `sys.exit()`, and the four-level nested `try/except`
wrappers whose only effect was to log and re-raise. DICOM conversion no longer
writes `<input>/NIFTI/` into the caller's own folder.

**Core changes, both sanctioned:** one `config.Settings` field
(`AREG_MAX_GPU_JOBS`) and one `requirements.txt` entry (`itk-elastix==0.25.4`).
The pin was checked against the rule the `monai` entry earns: it depends on
`itk-core`/`itk-filtering` and nothing else — no torch, so it cannot shadow the
image's CUDA build — but it *does* move the image's own `itk` 5.4.6 → 5.4.7 in
the pip `--user` layer, a patch bump of the series ALI's CBCT engine runs on.
`main.py`, `registry.py` and `base.py` untouched.

**Not ported: the IOSCBCT mode** (registering an intra-oral scan onto a CBCT of
the same patient). Genuinely a different problem — cross-modality,
landmark-driven, a four-folder input contract — and the least settled of the
three CLIs: its jaw detection is `re.search(r'[_]?[uU][_]?', filename)`, which
matches any `u` anywhere in a name; its `run_icp_point_to_plane` computes the
surface normals and never uses them (it is a point-to-*point* Kabsch solve); and
its patient ids are normalised so that `P001` and `P1` become one subject. Its
test files stay in `data-manifest.yml` and it is next in line.

**Tests:** 381 server tests (+53), no GPU and no weights: the patch network is
stubbed, everything else runs for real, including a full CBCT registration
through elastix on a 48³ phantom.

**Verified live end to end over HTTP**, both CBCT modes:

* Semi-Automated, two synthetic patients: 200 in 11 s, with the registered
  volumes, the transforms and the report, and `TEMP_DIR` clean afterwards.
* **Fully-Automated on the real 512×512×365 CBCT at 0.33 mm** hosted as AMASSS's
  test file, displaced by a known 2.9° + 3.2 mm rigid transform: **200 in 66 s**
  — AMASSS segmenting the cranial-base mask on the GPU, elastix, and a 100 MB
  upload plus a 106 MB download. The known displacement came back to **0.048 mm,
  a seventh of a voxel**.

That last run also illustrates why the centre-of-rotation defect survived: this
scan's origin is (-84.5, -84.5, -60.1) mm for a 169×169×120 mm field of view, so
its geometric centre is within a millimetre of the physical origin — `(I - R)c`
is 0.04 mm on it. The reference scan everyone tests with is a centred one.

### 2026-08-10 — Transfers use several connections instead of one

**"A 100 MB file takes an eternity to upload".** The entry below measured the
HTTP stack as innocent *over loopback* and pointed at "the wire or the client's
path to it". That was right, and this is that part: a remote deployment is the
target, and one HTTP request rides one TCP connection, which is bound by its
congestion window long before it is bound by bandwidth. No amount of tuning a
single stream fixes that, only more streams do.

**`transfer.py`: uploads arrive as independent parts, results leave as byte
ranges.** `POST /uploads` opens a session and answers with the part size it
chose; `PUT /uploads/{id}/parts/{n}` `os.pwrite`s each part at its offset into a
pre-`truncate`d sparse blob. Parts never overlap, so there is no lock anywhere
and **no reassembly pass** — the blob *is* the file once the last part lands,
and `claim_upload` then *renames* it into the run's work dir. That also removes
the double disk write the old path had (Starlette spools the multipart body to a
temp file, `_stream_to_disk` copied it out again). On the way back,
`X-Result-Delivery: reference` makes `/run` return a pointer, and
`GET /results/{id}` honours `Range`.

Measured client-to-server for 100 MB, through a relay capping each connection at
12 MB/s:

| | upload | download |
|---|---|---|
| one request, one connection | 9.1 s | 8.7 s |
| 4 connections (the client default) | 2.5 s, **3.7x** | 2.5 s, **3.6x** |
| 8 connections | 1.5 s, **6.2x** | 1.4 s, **6.2x** |

Over loopback, where there is no window to be limited by, the upload is still
1.5x faster from losing the double write, and the client's peak RSS for a 100 MB
upload drops from 200 MB to 32 MB (`requests` reads a `files=` argument entirely
into memory, then builds the whole encoded body next to it).

**Integrity is per part, not per file.** Each `PUT` carries `X-Part-SHA256` and
is refused before anything is written, so corruption costs one retried part. The
parts tile the file exactly, so the assembled blob is verified in full without
either side making a second pass — which matters more than the speed: a silently
truncated CBCT reaching a tool is a wrong result, not an error.
`Content-Encoding: gzip` is honoured (worth ~3x on an uncompressed `.nii` or a
`.vtk`), and the checksum covers the decompressed bytes.

**Resumability falls out of the same design.** `GET /uploads/{id}` reports
`missing_parts`, so a client coming back from a dropped connection sends only
the gap. State is on disk (immutable `meta.json` + one `O_EXCL` marker per
part), so it survives `--reload` and works under `uvicorn --workers N`.

**Nothing is mandatory.** A client that ignores all of this gets byte-identical
behaviour to before; a client that uses it against an older server sees a `404`
on `POST /uploads` and falls back. Abandoned sessions and unclaimed results are
reaped after `TRANSFER_TTL_SECONDS` of **inactivity** (15 min) by a sweep on a
timer, so an idle server cleans up too. This is a confidentiality bound, not a
disk-space one.

**And the cleanup guarantee is deliberately not weakened for the common case.**
A result under `RESULT_REFERENCE_MIN_MB` (16 MB) is streamed in the /run
response and deleted server-side when that response ends, with no dependency on
the client at all. Only results big enough to genuinely need several connections
take the reference route. Worst case for a client that dies mid-download: about
16 minutes.

**The client-side note from the entry below is now closed too:**
`slicer_io.zip_folder` deflated already-compressed uploads at level 6, on the
user's own machine, on the main thread. 105 MB of gzipped CBCT took 2.3 s to
pack into an archive of exactly the same size; it now stores those members and
takes 0.16 s.

**Tests:** `server/tests/test_transfer.py` (+31), including the cleanup
guarantees specifically: a transfer still making progress is never reaped, a
result being range-read is never reaped, an abandoned one does expire, the timed
sweep fires with no request at all, and a small result is streamed leaving
nothing in `results/`. Plus parts in any order, resume from a reported gap, a
refused checksum, a wrong-length part, gzip round-trip, an over-size file
refused before transfer, traversal ids, the reaper, `/run` through a session,
and range / suffix-range / 416 on results.

### 2026-08-07 — Result archives stop re-deflating already-compressed members

**"The zip and the unzip are super long, and a 90 MB transfer is slow on a 1 Gb
link".** Measured before touching anything, on the live container: the HTTP
stack is innocent — the 94 MB testfile downloads at ~600 MB/s and uploads at
~450 MB/s over loopback. A 1 Gb wire moves 90 MB in ~1 s. So everything slower
than that was CPU, and it was the zip.

`zipfile` DEFLATE level 6 runs at 30–46 MB/s on one core, and most of what this
server ships is already compressed. The 94 MB input everyone tests with is a
`.nii.gz`; deflating it again cost 3.3 s to shrink the archive by 0%. The client
then pays the same tax twice more on arrival: its `_verify_download` CRCs every
member (a full decompression) and its extraction inflates them again — both ~5x
faster on a STORED member.

`make_zip` now picks the compression per member (`_STORED_EXTENSIONS`):
`.gz`/`.zip`/OOXML/image members are stored as-is, everything else deflates at
the new `settings.ZIP_COMPRESSLEVEL`, default 1 — measured twice as fast as
level 6 for ~3% of size on the one member class still worth compressing (binary
`.vtk`, ~2.7:1 at either level). Storing *everything* would have traded real
wire bytes for nothing. A mixed archive is an ordinary zip, so the Slicer client
needs no change and still benefits on its CRC and extract passes. Measured on
the real testfiles (94 MB `.nii.gz` + 16 MB `.vtk`): **3.32 s → 0.31 s** for a
105 MB archive where level 6 produced 104 MB.

**Not touched, deliberately:** the client's `slicer_io.zip_folder` deflates
uploads at level 6 with the same waste — the biggest *perceived* win, since it
runs on the user's own machine, but it lives in the client repo. The extraction
side was measured fast enough to leave alone (~1 s per 200 MB).

**Tests:** 297 server tests (+3): the `.nii.gz` and `.XLSX` members stored
whatever their case, the `.vtk` still genuinely deflated, and a mixed archive
round-tripping byte-identically through `extract_zip`.
### 2026-08-06 — ALI can be asked for named landmarks, which is what ASO needs

ASO's fully-automated CBCT mode calls ALI in-process and checks that its schema
exposes `("input", "model", "landmarks")`. ALI declared `cbct_regions` and
`ios_networks` and no `landmarks`, so the call failed on the contract check.

The name was the smaller half. ASO registers on seven points (Ba, S, N, RPo,
LPo, ROr, LOr) straddling the Cranial base and Upper regions, so asking by
region runs **58 agents to use 7** — and one agent is a full two-scale walk of
the volume. The engine always worked at landmark granularity internally; only
the schema was coarser.

- `landmarks` is a multichoice over all 119 catalog labels, **every option off
  by default** — unlike `cbct_regions`, whose options are all on. "All off" is
  what an omitted multichoice arrives as, so the default state means "nothing
  said here, the regions decide", which is what every earlier request keeps
  meaning.
- Naming a landmark **REPLACES** the region selection rather than narrowing it.
  Narrowing would agree for ASO only because it leaves the regions all on, and
  would silently drop landmarks for a caller that set both. The run report says
  which drove the run: `regions` is empty when `landmarks_selected` is not.
- The 119 options are readable because the schema says how to group them:
  `ui="tabs"` with `groups=LANDMARK_GROUPS`, which is `GROUP_LABELS` — the same
  table the engine names its output files by, published rather than restated,
  so a landmark added to it gets its tab with no client release. ALI also
  gained sections and a `label` on every argument.

**Tests:** 197 server tests (+3), including ASO's exact argument dict surviving
`tool.validate` with `input` as a resolved directory. Client-side, 34 ALI tests.

### 2026-08-06 — Presentation hints: the schema says how to lay a panel out

ASO's panel was unusable: four modes share one schema, so a generic client
rendered 130 CBCT landmarks, 32 teeth, 8 landmark types and 2 jaws as a single
column of ~180 check boxes with CBCT and IOS options interleaved, while any run
uses one half or the other. ALI has the same shape for a different reason: 119
landmark options and no `mode` field to hide the inert selection behind. The
old Slicer modules solved this with hand-written QStackedWidgets and ~700 lines
of checkbox plumbing, with the anatomy written inside the widget — exactly what
the ports removed.

**Five optional `ArgSpec` fields, published by `GET /tools`, ignored by
`validate()` and `run()`:** `label`, `section`, `visible_when`
(`{other_arg: value}`), `ui` (`"tabs"`/`"grid"`/`"inline"`) and `groups`. Every
one is `null` on a tool that declares none, so existing panels render unchanged.

- `label` closes the last thing the client was inventing. Labels were built
  client-side by two different rules in the same panel, so ASO showed
  "Reference" above "cbct_landmarks". No naming rule can produce "Scan /
  Landmark Folder" from `input`. Every user-visible word describing a tool is
  now the tool's; the client keeps only its own chrome.
- **None of the layout fields names an anatomical concept**: `groups` says what
  to group, `ui` how to lay it out, `visible_when` when it applies. ASO's
  `TOOTH_GROUPS` is derived from `TOOTH_IDS`, and ALI's tabs are
  `GROUP_LABELS`.
- The two tools use different amounts of it. ASO has a `modality` choice, so
  `visible_when` makes its two selections mutually exclusive; ALI has no such
  field on purpose, so it gets `section` only.
- `check_schema` rejects them at startup, and that matters more here than for a
  real type: a wrong `visible_when` hides a field for good, and a client cannot
  tell that from a field the tool never declared. An option no group mentions
  is *not* an error — the client renders the leftovers.
- `visible_when` is presentation, not validation: a hidden argument is not
  sent, so its declared default applies, and cross-argument checks still run
  for a direct API call. What it fixes is a real wire problem — a multichoice
  is read back as the complete `{option: checked}` dict, so a panel was sending
  the inert mode's selection, frozen at whatever the invisible widget held.

### 2026-07-31 — ALI's model bundle is matched to the detected mode; a wrong bundle is a 422

Found by running ALI IOS from Slicer with the dropdown left on
`ALI_CBCT_Models`: the IOS engine listed all 119 CBCT files as unrecognized and
Slicer showed `500 — The tool failed on the server`, the one message written
for the user buried in the log.

- **`model` is now optional and the mode picks it** (`ALILogic.select_bundle`).
  Each engine recognises its own bundle layout through `discover_weights`
  (`<landmark>/<scale>/*.pth` folders vs flat jaw/network-token checkpoints;
  mutually exclusive, file-name parsing only, so probing costs a directory walk
  and never a model load). No match is a 422 naming `setup-models.sh`; several
  matches is a 422 naming the candidates rather than a silent pick — which
  model vintage ran must never be a surprise. The report gains `model_bundle`.
  A temp copy materialized for a probe (`ResolvedFile.is_temporary`) is deleted
  whether or not it was picked.
- **A named-but-wrong bundle answers 422, not 500.** The five
  `FileNotFoundError`s the engines raised are `ToolArgumentError` now, so the
  message reaches Slicer verbatim. The two "not a directory" messages named the
  full server path; a 422 body travels, so they name the basename only.

**Client:** the dropdown of an optional scalar `server_selectable` argument
leads with an "(automatic)" entry whose item data is `""`, so the default
selection sends no `model` at all. Generic in `base_widget`/`formgen`.

**Tests:** 7 new server-side, plus an end-to-end CBCT run with no `model`.
Verified live: an IOS request with no `model` returned 200 in 17s with
`model_bundle: ALI_IOS_Models`.

### 2026-07-31 — 501 for "this server cannot do that", instead of a blank 500

Found by running ALI in IOS mode: the preflight raised immediately with a
message naming pytorch3d, and that message went to the server log while the
Slicer user got `500 — The tool failed on the server.`

500 hides its detail rightly: a crash inside a tool can name server-side paths.
A missing optional dependency is the opposite — the request was valid, nothing
the caller changes will help, and the reason names a package.

`base.ToolUnavailableError` plus a `501 Not Implemented` mapping in `main.py`.
Every dependency-import failure across `ALI`, `CrownSeg` and `AMASSS` raises it
(twelve sites), because the same condition answering 500 in one tool and 501 in
another is worse than either. **No client release needed:** `error_for_status`
shows the server's `detail` verbatim for any unmapped status.

### 2026-07-31 — Test files are downloadable: `GET /tools/{tool}/testfiles/{filename}`

The Slicer client grows a per-input "Test file" button filling a file input
with reference data. The hosted-name route runs a tool on a test file without
it travelling, but cannot put the file in the user's hands.

One Bearer-protected endpoint streams a test file by name, resolved through
`data_store.resolve_testfile` (so the backend abstraction and its traversal
checks apply; unknown name → 404). A folder entry is zipped on the fly into a
staging dir under `TEMP_DIR` and removed by background task once streamed; a
backend temp copy is likewise removed. **Test files only** — models are
selected by name and used in place. The log line carries tool, status, duration
and size, never the file name.

Also: `AMASSS`'s `input` is now `server_selectable="testfile"`. The client grays
its button off the actual `GET /tools/{tool}/data` listing, so an empty
`testfiles/` folder is a grayed button explaining itself, not a 404.

**Tests:** 401 without a token, 404 for unknown tool/file, a plain file streamed
with the right headers, a folder zipped with `TEMP_DIR` clean afterward, and an
`is_temporary` copy removed after streaming.

### 2026-07-31 — `monai` pinned: an unpinned entry was replacing the image's torch

Caught by reading a `pip install` log, not by a failing test. Adding `monai`
unpinned made every container start resolve `monai 1.6.0`, which requires
`torch>=2.8.0` — so pip downloaded `torch 2.13.0` plus the whole CUDA 13 stack
**over the image's `2.5.1+cu124`**, on every start. Three consequences, none of
which fail a test: ~3 GB per container start, `torchaudio 2.5.1+cu124` left
unsatisfiable, and the image's purpose-built CUDA torch shadowed by a wheel
that happened to still find the card here.

`monai==1.5.1` asks for `torch>=2.4.1`, which the image satisfies, so pip
leaves torch alone. Every transform and network ALI uses exists there. Move to
1.6 only together with an image rebuild to torch >= 2.8.

**The general rule:** an unpinned dependency can upgrade torch transitively.
When adding one, check its torch requirement against the image.

**And an operational one.** The `inference` service installs requirements as
part of its *command*, so a container up for days runs whatever
`requirements.txt` said when it last started — uvicorn's `--reload` picks up
new Python code but never re-runs pip. Worse, `pip --user` writes into the
container's writable layer, which `docker compose restart` keeps. After
changing `requirements.txt`:

    docker compose up -d --force-recreate inference

**A dependency failure is a run-level failure** (`check_dependencies()` in both
ALI engines). The missing `itk` surfaced through the per-scan `try/except`, so
it was reported as if one patient's data were at fault: every scan failed
identically, each only after a complete histogram correction, and the run ended
on "ALI produced no landmarks for any scan". Both engines now import their
whole lazy stack once, before the loop.

### 2026-07-31 — Real-data tests are opt-in; `test-gpu` service; ALI's GPU cap off 1

`inference` already runs on the GPU and every tool reads `settings.DEVICE`.
Nothing hardcodes a device; the one service deliberately on CPU was `test`.

- **`test-gpu`.** The unit tests stub every model and gain nothing from a card,
  but `tests/test_data_integration.py` runs each tool end to end against the
  real bundles — minutes on a GPU, hours on a CPU. A compose device reservation
  is all-or-nothing, so putting it on `test` would make the pre-push hook fail
  on any clone without a card. A second service instead, sharing everything
  through a YAML anchor. The hook keeps pointing at `test`.
- **The real-data suite is opt-in (`RUN_REAL_DATA_TESTS`).** It was written to
  skip when `DATA/` is empty, but the moment a full ALI bundle lands "skip"
  turns into eleven minutes of GPU inference, or hours on the CPU the hook
  uses. It now skips at collection and stays ~10s. A pre-push hook that takes
  hours is a hook people disable.
- **`ALI_MAX_GPU_JOBS` 1 → 4.** Measured on the real bundle (RTX 6000 Ada): an
  ALI CBCT run peaks at **256 MiB** of VRAM on a 48 GB card. At a limit of 1,
  two concurrent requests fully serialized for a resource neither was close to
  exhausting. The figure is a property of the models, not the card.
  `AMASSS_MAX_GPU_JOBS` stays at 1: a 3d_fullres nnUNet is a different order of
  magnitude and nothing here measured it.

### 2026-07-31 — ALI (both engines) + CrownSeg

Port `ALI` — automatic landmark identification — from a pair of Slicer CLI
modules. The first tool with *two* engines sharing nothing but their output
format, and the first whose IOS half depends on a library the image lacks.

**One tool, two engine folders (`tools/ALI/src/{ALI_CBCT,ALI_IOS}/`).** One
entry in `GET /tools`, one `DATA/ALI/`, one Slicer module. `ALILogic.py` owns
everything before inference (unpacking, DICOM conversion, mode detection, the
run report) so each engine only has to place landmarks. `src/markups/` holds
the Slicer `.mrk.json` writer both engines use.

**The mode is detected, not declared.** There is deliberately no `mode`
argument: a `.zip` can hold either kind of data and a DICOM series has no
extension, so only the data distinguishes them. An archive holding both kinds
is a 422 rather than a guess. The accepted cost is that the schema cannot say
"this argument only applies in mode X": `cbct_regions` and `ios_networks` are
both optional and one is inert on any run. Emptying the selection for the mode
that actually ran is a 422 naming the argument to fill in.

**`CrownSeg` is a tool, not a helper.** ALI's IOS engine needs a mesh carrying
per-tooth labels. The Slicer module got them by running the `dentalmodelseg`
executable out of Slicer's bin — which is only the console-script entry point
of the `shapeaxi` PyPI package, so nothing needed porting: `tools/CrownSeg/src/`
calls `shapeaxi.dental_model_seg.main()` directly. It lives in its own tool
because ASO, AREG and FlexReg call it too, and because ALI's IOS half needs
pytorch3d — inside ALI, one absent dependency would take four tools out of the
registry instead of one. `model` is optional there and falls back to
`settings.CROWNSEG_MODEL`; the library's own fallback downloads the checkpoint
from GitHub mid-request, and a server holding patient data does not make
outbound calls. shapeaxi's stdout is swallowed — it prints the patient's own
file name.

**Defects fixed by construction**, all of which lost results silently:

- **One unknown landmark cost the whole patient.** `LABEL_GROUPS[landmark]` was
  indexed with no guard inside the save loop, and its `KeyError` was caught far
  above — so nothing at all was written for that scan. The two spellings that
  triggered it (`UR3OI…` in the UI, `UR3OIP…` in the CLI) are aliases of one
  vocabulary now, and `group_of()` cannot raise.
- **Homonyms overwrote each other in batch.** The patient key was `file.name`,
  so two `scan.nii.gz` in different subfolders collided twice over. Scans are
  keyed by path relative to the input root, and the output mirrors the tree.
- **A missing mandibular IOS model was a silently-caught `KeyError`.** Reported
  now. The jaw must be named explicitly in the checkpoint's name: "not Lower ⇒
  Upper" meant a bundle missing its mandibular model quietly predicted the
  lower arch with the maxillary one. One naming rule replaces the two the UI
  and CLI disagreed on, verified against the published archive.
- **DICOM conversion wrote into the user's own folder** (`<input>/NIFTI/`),
  which the next run re-ingested as input scans. Everything goes to the request
  scratch dir, as does the segmentation CSV the module wrote into the
  extension's own source tree.
- **`.stl` was accepted then ignored**: the UI counted them, the CLI globbed
  for `.vtk` only. `surface_or_zip_file` (new `FILE_TYPES` entry — the only
  core edit) advertises exactly what discovery walks.
- **`R`, `RIP`, `OIP`** were selectable and predicted by nothing. Not offered.
- **`SaveId` was read by nothing**; `prediction_ID` is a real argument.
- **Output extensions disagreed** (`.mrk.json` vs `.json` for identical
  content, only the first of which Slicer recognises). Uniform, and one file
  per scan instead of one per region — the split forced every downstream tool
  to recombine them by hand.
- **`display.visibility: false`**, in both CLIs. It switches the markups
  *display* node off, so Slicer loads the file, builds the node and draws
  nothing. Invisible inside the old module, fatal the moment anyone opens a
  result file — which is what a returned archive is for. Two tests pin it.
- Two latent search bugs: `new_pos.all() > 0` reduced the array to one boolean
  *before* comparing, letting negative coordinates through; and `Focus`'s
  convergence loop had no bound, which in a worker thread is a request that
  never returns. The IOS masks were also argmax'd over logits cast to `int16`,
  turning near ties into real ones resolving toward the background channel.

**Sequencing:** the CBCT engine runs today on `monai` + `itk`. The IOS engine
and CrownSeg are written and tested but cannot execute until the base image is
rebuilt on torch ≥ 2.8 with pytorch3d compiled in — pytorch3d has no PyPI
distribution at all. Both are imported lazily, so ALI loads, publishes its
schema, and fails only an IOS *run*.

**Tests:** 37 for ALI, 20 for CrownSeg, no GPU, weights or network.

### 2026-07-31 — ASO ported: four modes, one tool, and the defects it inherited

Port ASO (Automated Standardized Orientation) from a 2587-line Slicer widget
plus four CLI modules. ASO is the step every longitudinal study runs before
anything else, and AREG needs it programmatically.

|          | Semi-Automated | Fully-Automated |
|---|---|---|
| **CBCT** | your landmarks, ICP onto a gold set | landmarks predicted first, then the same ICP |
| **IOS**  | your landmarks, ICP per jaw | tooth centroids of an already segmented mesh |

`modality` and `automation` are explicit `choice` arguments, never inferred: a
`.zip` can hold either kind of data, and guessing wrong orients a patient
against the wrong reference and calls it a success. Every mode-specific
argument is `required=False`, with cross-argument rules raised as
`ToolArgumentError` **before** any file is read.

The call into the landmark tool is **in-process, not HTTP to our own /run/ALI**:
a tool run holds one of `MAX_CONCURRENT_TOOLS` slots for its whole duration, so
four concurrent ASO runs each waiting on a fifth slot would deadlock the
server, `/health` included. `Tool.invoke` is the same entry point `main.py`
uses, validation included.

**Fully-Automated IOS takes already-segmented meshes only** (crown segmentation
is CrownSeg's job; `segment_unlabelled()` is where it plugs in). **No** labelled
mesh in the batch is a 422 (wrong mode); *some* unlabelled is a per-patient
report entry and the rest of the batch is processed.

**The defects that cost data**, each with a named test:

- **`SEMI_ASO_CBCT` could not work at all.** It read `data["tfm"]`
  unconditionally, but only the fully-automated chain produced one, so every
  semi-automated patient died on a `KeyError` caught 90 lines above.
  Recentring always runs now, and the landmarks are moved with it.
- **One landmark could lose a patient.** `GetDistDifference` indexed the
  reference's pairwise table with the *input's* keys. The two sides are
  intersected first and what was dropped is reported.
- **Patient keys collided.** `GetPatients` keyed on the base name, and stripped
  `_T1`/`_T2`, collapsing two timepoints into one patient.
- **`MergeJson` merged a patient's landmark files by writing into the caller's
  input folder and DELETING the sources.** The merge is in memory.
- **A second run re-ingested the first.** `patient1_Or.nii.gz` sorts before
  `patient1_scan.nii.gz`. Previous outputs are set aside and used only when a
  patient has nothing else.
- **`UpperOrLower` defaulted to Lower**, so a maxillary mesh named
  `patient1.vtk` was registered against the mandibular reference and returned
  as a success. A file whose name does not say its jaw is refused.
- **`Files_vtk_json.organise` paired with `vtk_name in json_name`**, so patient
  `1` matched patient `10` — and padded its list with a literal
  `"Upper_nioegfjhdfjkdffdhjmndfhnmdfhj"` sentinel. Exact stem, per directory.
- **Both jaws wrote the same `.tfm`.** Named per jaw now.
- **The published IOS reference was rejected outright.** Refusing a mesh whose
  name does not say its jaw is right, but the first version also required an
  identifier *before* the jaw token — and the published `Gold_file.zip` is
  `Upper_gold.vtk` / `Lower_gold.vtk`, jaw first. Found by reading the real
  archive rather than assuming its shape.

**Concurrency, which only matters because this is a shared server:**

- `InitIcp` wrote `source.npy`/`target.npy` into **its own installed package
  directory** and re-`np.load`ed one on every iteration of a 2500-iteration
  search — a write into the install tree, thousands of round trips per patient,
  and two concurrent requests overwriting each other's landmarks. The search is
  pure and in memory (`src/geometry.py`, shared by both engines, which had
  carried two drifted copies).
- The triplet search drew from the **global** numpy generator, so the same
  request gave a different orientation every time. Every ordered triplet is now
  enumerated when there are at most `ASO_ICP_MAX_TRIPLETS` of them (7 landmarks
  is 210) — deterministic, faster *and* better than sampling; above that a
  local generator seeded with `ASO_ICP_SEED` is used.

**Latent bugs found while reading, each now a test:** `np.arccos` of a dot
product rounding past 1.0 gave NaN propagating through the rotation matrix;
`RotationMatrix` divided by a zero-length axis for two parallel vectors;
`ASO_IOS_utils/utils.py` defined `PatientNumber` twice; `WriteSurf` used
`vtkPolyDataWriter`'s ASCII default; the `.off` reader referenced an undefined
`line`; and `SEMI_ASO_IOS` wrote every transform as `matrix_file_0.npy`.
**The IOS matrix composition order was also backwards** (`M_init @ M_icp`,
where the ICP runs on points the initialisation already moved); the CBCT engine
always had it right, which makes it a transcription slip.

**Also removed rather than ported:** `PRE_ASO_CBCT`'s `model_folder`, `SmallFOV`
and `temp_folder` arguments (read, never used), the `<filter-progress>` prints,
`sys.exit()`, `tqdm`, the `time.sleep(0.2)` progress theatre, the `*Error.txt`
files written into the output folder, the skip-if-exists guards, and the
reference *scan* the CBCT ICP read and never used — which it nonetheless
required, so a reference bundle holding only landmarks died on an `IndexError`.

**Core changes, both sanctioned:** one `FILE_TYPES` entry (`surface_file`) and
three `config.Settings` fields. No GPU: everything ASO computes is
SimpleITK/VTK/numpy. `main.py` and `registry.py` untouched.

**The two published references carry DISJOINT landmark sets**, and the schema's
defaults only match one: Frankfurt Horizontal + Midsagittal has
`Ba, S, N, RPo, LPo, ROr, LOr`; Occlusal + Midsagittal has
`ANS, IF, PNS, UL6O, UR1O, UR6O`. Picking the second and leaving the defaults
would drop every landmark as "not in the reference" and fail all forty patients
separately. `_check_selection_against_reference` turns that into a single 422
naming what the reference offers, raised after discovery but before a scan is
read.

**Tests:** 201 server tests, 71 new. Two of them cover the outputs a clinician
relies on, and both hold to the float: the written landmarks land exactly where
the resampling put the voxels (volume and markups move by two different code
paths — if they disagree the markups file opens floating beside its scan), and
the `.tfm` maps ORIENTED → ORIGINAL, recentring included. That direction is
asserted rather than assumed because getting it backwards is silent.

### 2026-07-30 — AMASSS surfaces: binary, and decimated by default

A five-structure run with surfaces returned a 41.9 MB archive Slicer could not
open — the client froze on the main thread and the user read it as "the server
never sent the .vtk". It had: `curl` pulled all 41,889,544 bytes and every mesh
re-read cleanly. The geometry was the problem.

- **Marching cubes runs on the original scan grid**, so a 0.33 mm CBCT produces
  a triangle per voxel face: 1.6M for a cranial base, 3.5M across five
  structures, 11.8M for a merged nine-structure volume. The mask underneath is
  only accurate to about half a voxel, so that is resolution nobody asked for.
- **`vtkPolyDataWriter` was writing ASCII** (its default): 848.5 MB for the
  merged surface against 6.4 MB for every segmentation in the same run. Binary
  is the same geometry and the *more* accurate of the two — it round-trips the
  float32 vertices exactly, while ASCII prints ~6 significant digits and moved
  points by up to 5e-05 mm on read-back. It is also 133x faster to parse (a
  1.6M triangle cranial base: 2.67s ASCII, 0.02s binary).
- **`surface_decimation` (new argument, default 90).** `vtkDecimatePro` with
  `PreserveTopologyOn`, applied after smoothing and *before* the per-cell colour
  array is built. Measured on the cranial base against a 0.33 mm voxel:

  | reduction | triangles | mean dev | p95 | max |
  |---|---|---|---|---|
  | 50% | 811,222 | 0.0034mm | 0.004mm | 0.277mm |
  | 80% | 324,488 | 0.0338mm | 0.125mm | 0.493mm |
  | **90%** | **162,244** | **0.0590mm** | **0.171mm** | 0.692mm |
  | 95% | 81,122 | 0.0951mm | 0.264mm | 1.223mm |

  90 costs a fifth of a voxel on average and buys a factor of ten. 0 keeps the
  raw mesh. The value is recorded in `AMASSS_report.json`, these surfaces being
  lossy by default now.

**End to end, real HTTP:** archive 41,889,544 → **5,417,443 bytes** (7.7x),
triangles 3,519,420 → **351,938** (10x), client-side mesh parsing 2.7s+ →
**0.01s**. Decimation adds ~12s of server time for five structures.

**A caveat worth keeping:** binary alone did NOT shrink the download. DEFLATE
was already squeezing ASCII at 6.2:1 and binary only compresses 2.7:1, so the
archive went 223.4 MB → 227.8 MB on a nine-structure run. Binary pays off in
disk, RAM, write time, zip time and parse time — not on the wire. Only removing
geometry moved the download.

**Still on the table, in the client repo:** `AMASSS.py`'s
`MAX_RESULTS_TO_LOAD = 12` caps by *file count* while the cost is in triangles.

**Tests:** 119 server tests (+4).

### 2026-07-30 — AMASSS: the GPU was idle seven eighths of the run

Profiling one structure on a 512x512x365 CBCT at 0.33 mm: **14.6s** resampling
the input to the model's 0.4 mm grid, **4.5s** of inference, **6.9s** resampling
the logits back. Both resamplings are scipy splines pinned to a single core.

**The tempting fix was measured and discarded.** At a 128³ patch the network
already saturates the SMs at batch 1: throughput is flat from batch 1 through
12. Cutting the GPU's work 5x (`tile_step_size` 0.5 → 1.0) moved the total from
37.3s to 34.5s and dropped utilisation from 36% to 11% — direct evidence the
GPU was never the constraint. **Free VRAM is not convertible into speed here;
idle time is.**

- **GPU resampling** (`nnunet_runner._enable_gpu_resampling`, new setting
  `AMASSS_GPU_RESAMPLING`, default on). nnUNet already ships torch versions of
  both resamplers, so nothing is reimplemented — only selected, and selected by
  NAME: nnUNet resolves them out of the configuration dict via
  `recursive_find_resampling_fn_by_name`. Two things make that mutation safe:
  `PlansManager` hands out a `deepcopy`, so it touches neither the shared plans
  nor a concurrent request — and consequently the `torch.device` never reaches
  the `plans.json` nnUNet writes, which `json.dump` could not serialize. Both
  properties are `@property @lru_cache`, so the swap clears them.
- The GPU path uses `predict_from_files_sequential`: `predict_from_files` fans
  preprocessing and export out to *spawned* processes, each of which would need
  its own CUDA context. That trades away the CPU/GPU overlap on multi-scan
  batches; recovering it with a reader thread is the obvious next step.

**Measured end to end, real models, five structures, one scan: 195.9s → 77.0s
(2.5x).** Per structure 34.2s → 13.2s.

**Not numerically free.** torch has no 3D cubic interpolation, so the input
resampling drops from spline order 3 to order 1. Dice against the scipy
pipeline: MAND 0.998, UAW 0.997, MAX 0.995, CB 0.991, **CV 0.978**. The
cervical vertebra is consistently the outlier — thinnest structure, closest to
the edge of the field of view. `AMASSS_GPU_RESAMPLING=false` restores
bit-identical output, and a bundle whose plans pin a non-default resampler opts
itself out. `AMASSS_report.json` records `gpu_resampling` and `tile_step_size`.

**`AMASSS_TILE_STEP_SIZE` (default 0.5, unchanged).** The one knob here that
moves the segmentation for a *pure* speed gain, so it is exposed rather than
tuned: 0.7 measured Dice 0.995 against 0.5 and saves ~2.5s of GPU per structure.

**`_convert_to_nifti` stopped casting to float32.** The cast was never what made
the conversion real, and it doubled the bytes gzipped per scan and gunzipped
again by nnUNet, costing 2.4s + 0.4s for nothing.

**`vtk_export` cleanups.** Cell colours are built in numpy instead of a
`SetTuple` per cell. The marching-cubes temp file had a *fixed* name, so every
surface in a run wrote over the same path; it is unique per call now and
removed after use.

**Tests:** 115 server tests (+5).

### 2026-07-30 — Dead-code and duplication cleanup

- `base.py` had an entire block (`FOLDER_TYPE`, `SCALAR_TYPES`, `CHOICE_TYPES`,
  `Selection`, `ResolvedPath`) declared TWICE, plus the remnants of the retired
  `SELECTION_TYPE` API. All removed. **The current API is
  `"choice"`/`"multichoice"`** (one `choices` dict of option → default).
- `main.py._describe_argument` (unused, referenced the removed fields) and
  `file_utils.zip_directory` (unused duplicate of `make_zip`) removed.
- `requirements-amasss.txt` removed: torch/nnunetv2/vtk had been added to
  `requirements.txt`, leaving the file a pure duplicate. The heavy stack stays
  lazily imported, and torch stays unpinned so the image's CUDA build is never
  shadowed.
- The `test` service in `docker-compose.yml` is back under a
  `profiles: ["test"]` guard.
- Docs realigned with the code: `claude.md` → `CLAUDE.md`, `surgMovPred` →
  `SurgMovPred`, README's selection-argument section rewritten for
  `choice`/`multichoice`, `ADDING_A_TOOL.md` §7 now describes the real
  requirements layout.

### 2026-07-28 — AMASSS tool + grouped selection arguments

Port `AMASSS_CLI.py` (CBCT skull structure segmentation, nnUNet v2). The first
tool that is genuinely an *API* — AREG already calls it programmatically — the
first to need an argument the schema could not express, and the first with a
GPU deep-learning stack.

**Core additions (`base.py`, `main.py`):**

- Choice arguments: `ArgSpec.choices`, with the `"choice"` (exactly one) and
  `"multichoice"` (any number) types. One `{option: on by default}` dict
  declares the options **and** their initial state, so a client renders the
  widget straight from `GET /tools` and the defaults are written down once.
  Accepted on the wire as `"MAND,MAX"` or `{"MAND": true}`; an invalid option
  is a 422 naming what is allowed. This is a change to the *type system*, made
  once — the same category as adding a `FILE_TYPES` entry.
- `FILE_TYPES["volume_or_zip_file"]`: one argument accepting either a single
  volume or a zip of a folder of them, since the schema cannot express "exactly
  one of these two arguments".
- `file_utils.zip_directory`, the counterpart of `extract_zip`.

**The tool (`tools/AMASSS/`):** `AMASSS.py` declares the schema;
`src/AMASSSLogic.py` holds the pipeline; `src/nnunet_runner.py` isolates
inference; `src/vtk_export.py` handles surfaces. `segment()` is the reusable
API; `main()` is the thin HTTP adapter.

Three defects mattered specifically *because* this is a shared server:

- the CLI set `os.environ['nnUNet_results']` before shelling out. Tools run
  concurrently in worker threads and `os.environ` is process-global, so two
  overlapping requests would have silently swapped model paths.
  `initialize_from_trained_model_folder` takes an explicit path.
- the CLI polled the output file's size and killed the predictor once it
  stopped growing for 3s, which could interrupt nnUNet mid-postprocessing.
- GPU work is serialized by AMASSS's own semaphore (`AMASSS_MAX_GPU_JOBS`,
  default 1), independently of `MAX_CONCURRENT_TOOLS`.

Also corrected: NRRD/GIPL inputs are really converted via SimpleITK instead of
being renamed; folder scanning is recursive and excludes previous outputs;
structures with no shipped model are no longer offered; a missing or failed
structure is reported in `AMASSS_report.json`; `sys.exit()` is gone. Inference
loads each checkpoint once per structure instead of once per (scan × structure).

**Dependencies:** `numpy` + `SimpleITK` in `requirements.txt`;
`torch`/`nnunetv2`/`vtk` imported **lazily**, since `registry.py` imports every
tool at startup and a missing heavy stack must not stop the server booting.

**Tests:** 35 with `nnunet_runner.predict_folder` stubbed, so no GPU and no
real models are needed.

### 2026-07-27 — Parallel request handling (threadpool execution of tools)

`run_tool` called `tool.invoke(args)` synchronously inside an `async def`
endpoint, i.e. directly on the uvicorn event loop. Any inference in progress
froze the entire server — a second `/run`, or even `/health`, could not be
answered until it finished.

`tool.invoke` now runs via `anyio.to_thread.run_sync(...)` in a worker thread,
with a dedicated `anyio.CapacityLimiter` capping simultaneous executions at
`MAX_CONCURRENT_TOOLS` (new setting, default 4). The limiter is created lazily
(anyio needs a running event loop) and is dedicated to tool runs so queued
inference cannot starve the default threadpool. Safe because tools are
stateless, each request has its own `work_dir`, and `DATA_DIR` is read-only.
The HTTP contract is unchanged.

**Test:** a probe tool whose `run()` blocks on a 2-party `threading.Barrier`,
fired from two requests through ONE shared event loop (`TestClient` as a
context manager — two bare `client.post` calls from separate threads would each
get their own loop and pass even against a serial server).

### 2026-07-27 — SurgMovPred: the model is server-side only, selected by name

The model should live exclusively in the server's data store: the client asks
for the list (`GET /tools/SurgMovPred/data`) and sends only the *name*.

The `model` argument changed from `ArgSpec(type="zip_file",
server_selectable="model")` to `ArgSpec(type=str, server_selectable="model")`.
The resolution path is unchanged — `main.py` already resolves any
`server_selectable` argument sent as a form value — but the contract is: a
scalar type means "name only". To enforce it, `main.py` rejects with a 400 any
file *upload* targeting a non-file-typed argument, which previously would have
passed the temp path through as the argument's string value.

**Tests:** an upload for `model` is a 400; an unknown model name is a 404; a
synthetic str-typed `server_selectable` argument resolves through `data_store`.

### 2026-07-27 — Pre-push test gate + real-data integration tests

The suite only ran on synthetic fixtures and only when someone remembered to
invoke it. A new `docker-compose.yml` service, `test`, runs the same image as
`inference` without its GPU reservation (`docker compose run --rm test`). A git
hook, `.githooks/pre-push`, runs it before every push; it is opt-in per clone
via `git config core.hooksPath .githooks` and bypassable with `--no-verify`.

`server/tests/test_data_integration.py` complements the synthetic tests: for
every tool whose required arguments are all `server_selectable`, it looks up
real files via `data_store` and runs the tool end to end. `DATA/` is gitignored,
so a tool with no matching file is **skipped**, never failed.

### 2026-07-24 — Server-side data store: models and test files without re-upload

Tools like `SurgMovPred` required the client to re-upload the same model on
every call, and there was no way to say "run this against the server's
reference data". Confidential-data constraints rule out a generic upload cache,
so this is explicit, per-tool, read-only server-side storage.

`server/data_store.py` introduces a `DataStore` interface with a
`LocalDataStore` reading `DATA_DIR/<tool_name>/{models,testfiles}/`. `ArgSpec`
gained `server_selectable` (`"model"` | `"testfile"`); `GET
/tools/{tool_name}/data` lists what is available. In `POST /run/{tool_name}`, a
`server_selectable` argument sent as a plain form value is resolved through
`data_store` and excluded from the temp-file cleanup that applies to uploads.

**Deliberately abstracted for a future external database/object store:**
neither `main.py` nor any `Tool` touches the filesystem directly. Each
`resolve_*` returns a `ResolvedFile(path, is_temporary)`; `is_temporary` lets a
future backend mark a materialized temp copy for cleanup, while
`LocalDataStore`'s persistent paths are never deleted. Swapping backends is
contained entirely to `data_store.py`.

Also: `docker-compose.yml` now mounts a single `./DATA:/data:ro` (previously two
inconsistent mounts), and `.gitignore` excludes `DATA/`.

### 2026-07-24 — Correct `Content-Type` for file-kind tool outputs

`POST /run/{tool_name}` responses with `output_kind in ("file", "segmentation")`
always sent `application/octet-stream` (or `application/gzip`), regardless of
the real format. An `.xlsx` is internally a zip container, so a client deciding
whether to unzip by sniffing magic bytes could not tell it from a real archive
— it silently extracted the Excel file's internal XML parts.

The `FileResponse` now derives `media_type` from the extension via
`mimetypes.guess_type()`, falling back to the previous logic only when the type
cannot be guessed (still the case for bare `.gz` files, e.g. `.nii.gz`). This
also fixes `.zip`, `.csv` and `.ods`.

**Client-side follow-up (not in this repo):** a client deciding whether to unzip
by sniffing magic bytes must trust `Content-Type`/`Content-Disposition`
instead — sniffing can never distinguish a real `.xlsx`/`.docx`/`.pptx` from an
actual zip archive, those formats being zip containers by design.
