# Inference server (tool-registry architecture)

FastAPI server exposing a generic `/run/{tool_name}` endpoint.
Tools are self-contained classes auto-discovered from `tools/` at startup —
adding a tool never requires touching the server core.

Requests are served **in parallel**: each tool execution runs in a worker
thread (never on the event loop), so a long inference never blocks other
requests — `/health`, `/tools` and other `/run` calls all stay responsive
while a tool is working. The number of tool executions allowed to run
simultaneously is capped by `MAX_CONCURRENT_TOOLS` (env var, default 4);
requests beyond the cap wait for a free slot. The HTTP call itself remains
blocking request/response: the client sends a request and gets the result in
the same response (no job queue / polling).

## Installation

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env
# edit .env and set a real API_TOKEN
```

## Server-side data (`DATA/`)

Models and reference test files are not in the repository. Fetch them from
the repo root with the scripts in [`scripts/`](../scripts/README.md), which
write the exact layout `data_store.py` reads:

```bash
./scripts/setup-models.sh --tool AMASSS      # or omit --tool for everything (~29 GB)
./scripts/setup-testfiles.sh
python3 scripts/fetch_data.py --list         # what's available, and how big
```

## Generating a self-signed certificate (development only)

```bash
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout key.pem -out cert.pem -days 365 \
  -subj "/CN=localhost"
```

This certificate is **for local development only** and will trigger a trust
warning. **Production must use a real certificate** (Let's Encrypt or an
institutional CA) — never run this server over plain HTTP outside a fully
isolated dev environment.

## Running over HTTPS

```bash
./venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 \
  --ssl-keyfile key.pem --ssl-certfile cert.pem
```

## Testing with curl

```bash
# Health check, no auth
curl -k https://localhost:8000/health

# Discover registered tools and their expected arguments
curl -k https://localhost:8000/tools

# Run the test tool
curl -k -X POST https://localhost:8000/run/test_tool \
  -H "Authorization: Bearer change-me-to-a-long-random-secret" \
  -F "text_1=hello" \
  -F "text_2=world"
# -> {"result": "hello world"}
```

`-k` disables certificate verification and is only acceptable against the
self-signed dev certificate above — never use it against a production server.

A tool argument can also expect a file: add `-F "input=@/path/to/data.zip"`
(field name = argument name) to the same call; the server streams it to a
temp dir and passes its path to the tool under that argument name. A tool can
declare more than one file-typed argument, each uploaded as its own multipart
field in the same request.

An argument declared with `ArgSpec(server_selectable=...)` can instead be
satisfied by a file already hosted on the server (under
`DATA_DIR/<tool>/{models,testfiles}/`): send the file's *name* as a plain form
value. On a scalar (non-file-typed) argument that is the only option — e.g.
`SurgMovPred`'s `model` (`ArgSpec(type=str, server_selectable="model")`) is
always picked by name, never uploaded (an upload for it is rejected with a 400):

```bash
# List the models/testfiles hosted server-side for a tool (Bearer-protected)
curl -k https://localhost:8000/tools/SurgMovPred/data \
  -H "Authorization: Bearer change-me-to-a-long-random-secret"
# -> {"models": ["stacking_v2.zip"], "testfiles": ["demo_measurements.zip"]}

# Download one of the listed test files (a folder entry arrives zipped).
# Test files only — models are selected by name and never leave the server.
curl -k -O https://localhost:8000/tools/SurgMovPred/testfiles/demo_measurements.zip \
  -H "Authorization: Bearer change-me-to-a-long-random-secret"

# Run it: the model is a name, the input is a genuine upload
curl -k -X POST https://localhost:8000/run/SurgMovPred \
  -H "Authorization: Bearer change-me-to-a-long-random-secret" \
  -F "model=stacking_v2.zip" \
  -F "input=@/path/to/measurements.zip"
```

## File-typed arguments

Instead of a single server-wide extension whitelist, each file-typed argument
declares its own specific **file type** in `ArgSpec`, from the registry in
`base.py`:

```python
FILE_TYPES = {
    "file": None,              # generic -- falls back to config.ALLOWED_EXTENSIONS
    "zip_file": (".zip",),
    "csv_file": (".csv",),
    "xlsx_file": (".xlsx",),
    "ods_file": (".ods",),
    "nifti_file": (".nii", ".nii.gz"),
    "volume_or_zip_file": (".nii", ".nii.gz", ".nrrd", ".nrrd.gz",
                           ".gipl", ".gipl.gz", ".zip"),
}
```

`volume_or_zip_file` is the pattern for a tool that works on one scan *or* a
batch: the schema cannot express "exactly one of these two arguments", so a
single argument accepts both and the tool dispatches on what it received
(file / zip / folder). `AMASSS` uses it.

```python
"input": ArgSpec(type="zip_file", required=True, description="..."),
```

This is what both the server (extension check on upload) and the client
(`GET /tools`, which reports `"type": "zip_file"` per argument) use to know
exactly what's expected — no shared global list to keep in sync as tools with
different file needs are added. To accept a new kind of file, add an entry to
`FILE_TYPES` in `base.py`. Only arguments left as generic `"file"` fall back
to `config.ALLOWED_EXTENSIONS`.

## Choice arguments (pick one or several from a server-defined list)

An argument typed `"choice"` (combo box, exactly one option) or
`"multichoice"` (check boxes, any number of options) declares its options —
and their defaults — in one `choices` dict, so a client renders the widget
entirely from `GET /tools` with nothing hardcoded:

```python
"structures": ArgSpec(
    type="multichoice", required=False,
    choices={"Mandible": True, "Maxilla": True, "Skull": False},
    description="Anatomical structures to segment",
),
```

`GET /tools` reports `choices` for every argument. On the wire, `"choice"` is
just the option name; `"multichoice"` accepts a JSON object or a
comma-separated shorthand of the enabled options:

```bash
-F 'structures={"Mandible":true,"Maxilla":true,"Skull":false}'   # JSON object
-F 'structures=Mandible,Maxilla'                                  # equivalent
```

Either form is the complete selection; omitting the argument entirely falls
back to the declared defaults. An invalid option is a `422` naming what is
allowed. Adding an option is a one-line server change with no client release.
See [`ADDING_A_TOOL.md`](../ADDING_A_TOOL.md) for the full contract,
including what `run()` receives (`base.Selection`).

## How to add a new tool

1. Create a folder `tools/<your_tool>/` with an `__init__.py` (can be empty)
   and a `tools/<your_tool>/<your_tool>.py` file — **the file name must match
   the folder name**, that's the one file the registry imports. Any other
   file in the folder (helpers, data, ...) is ignored by discovery, though
   your main file is free to import from them.
2. Subclass `Tool` (from `base.py`), set a unique `name`, declare `arguments`
   as a dict of `ArgSpec` (type, required, description), implement `run(**kwargs)`.
   For a file argument, pick a type from `base.FILE_TYPES` (or add a new one).
3. That's it — `registry.py` auto-discovers it at startup, `/tools` lists it,
   and `/run/<your_tool>` becomes available immediately. No route to add, no
   registration list to update. See `tools/test_tool/` for a minimal example,
   or `tools/example_tool/` for one with a file argument.

If your tool needs to unzip an upload, zip its results, and/or load
CSV/XLSX/ODS files, reuse the shared helpers in `file_utils.py`
(`extract_zip`, `make_zip`, `make_scratch_dir`, `load_tabular_file`,
`load_tabular_directory`) instead of reimplementing them.

### After changing `requirements.txt`, recreate the container

The `inference` service installs its requirements as part of its **command**,
so a container that has been up for days is running whatever the file said
when it last *started*. Uvicorn's `--reload` picks up new Python code but never
re-runs pip — which shows up as a tool failing on a `ModuleNotFoundError` for
something you can plainly see in `requirements.txt`.

```bash
docker compose up -d --force-recreate inference
```

`--force-recreate`, not `restart`: `pip --user` writes into the container's
writable layer, so a plain restart keeps whatever the previous resolution
installed — including a torch it should never have replaced. Only a fresh
layer discards it.

### Heavy or optional dependencies

`registry.py` imports **every** tool at startup, so a tool whose module-level
imports fail is skipped (see `ADDING_A_TOOL.md`). A tool needing a heavy or
optional stack must therefore import it **lazily, inside the functions that
use it**. `AMASSS` does this for torch/nnunetv2/vtk: those packages are listed
in `requirements.txt` (the deployment image already ships torch built against
its CUDA version, so pip skips it there), but on a machine where they are
missing the server still starts normally, every other tool works, and only
AMASSS fails — with a message saying what to install.

## How a tool run is executed (`SADT_DISPATCH_MODE`)

Two paths, one contract. `Tool.invoke` validates the arguments and then either
calls `run()` or hands the job to another process; the HTTP request blocks for
the run either way, and the client cannot tell which happened.

| `SADT_DISPATCH_MODE` | what happens |
|---|---|
| `inprocess` (default) | `registry.py` imported the tool at startup; `run()` is called in a worker thread |
| `subprocess` | the server writes a `job.json` and runs the tool in **its own virtualenv** through `runner.py` |

```
<TOOLS_DIR>/<tool>/.venv/bin/python <RUNNER_PATH> --job <job dir>/job.json
```

with `SADT_API`, `SADT_JOB_ID` and `SADT_JOB_DIR` in the environment (and
`API_TOKEN` deliberately removed from it). The runner imports the tool from
`<TOOLS_DIR>/<tool>/src/`, calls `run(**params)`, and writes
`{"result": ...}` to `result.json`. On failure it writes **nothing** and exits
non-zero, with stderr as the error channel — so a missing `result.json` is
itself the failure signal.

Why: importing every tool into the server pins the server's Python to the
lowest common denominator across all of them, makes two incompatible pins
(`numpy==2.4.0` against `numpy<2.0.0`) unresolvable, keeps a CUDA context alive
for the life of the process, and lets a segfault in a CUDA kernel take the API
down with the job.

`inprocess` is still the default and nothing has moved over yet; the flag is
temporary and goes away once every tool has. `tools/_dispatch_probe/` is the
fixture the path is tested with — a tool that needs no dependency at all, so a
failure there is a failure of the dispatch machinery and of nothing else.

## AMASSS: CBCT skull structure segmentation

```bash
# What model bundles / test scans are hosted server-side
curl -k https://localhost:8000/tools/AMASSS/data \
  -H "Authorization: Bearer $API_TOKEN"

# One scan, mandible + maxilla, merged output
curl -k -X POST https://localhost:8000/run/AMASSS \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "model=AMASSS_Models" \
  -F "input=@/path/to/scan.nii.gz" \
  -F 'structures={"Mandible":true,"Maxilla":true}' \
  -F "merge=One merged segmentation file" \
  --output result.zip

# Both output forms at once: merge is a multichoice, so it takes the
# comma-separated shorthand (or the full JSON object) like structures does.
  -F "merge=One merged segmentation file,Separated segmentation files"
```

`structures` and `merge` are both **`multichoice`** arguments, so they speak the
*declared option names* — the same labels the Slicer panel shows — not the
internal codes (`MAND`, `MERGED`). Sending a code is a **422** naming the
options it expected. `merge` was briefly a `"choice"`, which silently produced
runs with no segmentation files in them at all; see the 2026-07-31 changelog
entry in `CLAUDE.md`.

Models live under `DATA/AMASSS/models/<bundle>/<CODE>/**/*__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth`
(one subfolder per structure code) and are selected **by name** — they never
travel from the client. `input` takes either one scan or a zip of a folder of
scans for a batch.

The response is a zip: one `<scan>_<ID>_SegOut/` folder per scan plus an
`AMASSS_report.json` listing what was predicted, what was skipped for lack of
a model, and what failed — a structure without a model used to disappear with
nothing but a log line, which is invisible in a 200-scan batch.

Segmentations are **always written compressed**, whatever the input was: a scan
sent as a plain `.nii` comes back as `.nii.gz` masks. These are label volumes,
so the difference is not marginal — 191 MB down to 0.4 MB per structure on a
0.33mm CBCT, measured. The input's format is preserved (`.nrrd` stays `.nrrd`,
compressed internally — ITK has no `.nrrd.gz` writer); only the compression is
imposed.

**Calling AMASSS from another server-side tool**: import
`tools.AMASSS.src.AMASSSLogic.segment()` directly. It returns a
`SegmentationRun` exposing the produced files (`segmentation_files`,
`surface_files`) and the report, with no zip round trip. The zip only exists
because one HTTP response carries one blob.

Two knobs, both environment variables: `AMASSS_MAX_GPU_JOBS` (default 1) caps
how many AMASSS inferences touch the GPU at once, independently of
`MAX_CONCURRENT_TOOLS`; `DEVICE` selects cuda/cpu.

## ALI: automatic landmark identification

```bash
# One CBCT scan, cranial base only. No `model`: the server picks the hosted
# bundle whose layout matches the detected mode.
curl -k -X POST https://localhost:8000/run/ALI \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "input=@/path/to/scan.nii.gz" \
  -F 'cbct_regions={"Cranial base":true,"Upper":false,"Lower":false,"Impacted canine":false}' \
  --output result.zip

# A whole cohort (CBCT, IOS or DICOM series) as one archive. `model` names a
# bundle explicitly — only needed when several bundles of the same kind are
# hosted (a 422 lists them if the pick is ambiguous).
zip -r cohort.zip cohort/
curl -k -X POST https://localhost:8000/run/ALI \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "model=ALI_IOS_Models" -F "input=@cohort.zip" --output result.zip
```

**There is no `mode` argument.** The server inspects the input and picks the
CBCT or the IOS engine itself: a `.zip` can hold either kind, and a DICOM
series has no extension at all, so nothing in the request distinguishes them —
only the data does. An input holding both kinds is a 422 rather than a guess.
The optional `model` argument follows the same rule: left out, the hosted
bundle whose content the chosen engine recognises is the one that runs, and
the run report's `model_bundle` field says which it was.

The consequence a client has to live with is that the schema cannot say "this
argument only applies in mode X". `cbct_regions` and `ios_networks` are both
optional and both always rendered; one is inert on any given run. Emptying the
selection for the mode that actually ran is a 422 naming the argument to fill
in — that 422 is how a mode mismatch explains itself.

The response is a zip mirroring the input's folder tree, with **one
`<scan>_lm_<ID>.mrk.json` per scan** holding every landmark found (the two
original CLIs wrote one file per anatomical region, and disagreed on the
extension), plus a `run_report.json`. The report separates *"the bundle has no
weights for this landmark"* (`landmarks_without_model` → use another bundle)
from *"the agent did not converge on this scan"* (`landmarks_failed` → this
scan is hard). Both look identical in the Slicer scene and need opposite
fixes.

Model bundles are selected by name and never travel. CBCT bundles hold
`<landmark>/<scale>/*.pth` with scale folders named `1` and `0-3`; IOS bundles
hold checkpoints named with an `O`/`C` token and an `Upper`/`Lower` one, e.g.
`Upper_O_model.pth`.

`ALI_MAX_GPU_JOBS` caps its GPU use; `ALI_SEARCH_MAX_SECONDS` bounds one CBCT
landmark's search (unset = 15s on GPU, 60s on CPU).

**The IOS half needs pytorch3d**, which has no PyPI distribution and must be
compiled into the deployment image; the current one predates that. ALI still
loads and still publishes its schema — only an IOS run fails, naming what is
missing. The CBCT engine works today.

**Calling ALI from another server-side tool**: import
`tools.ALI.src.ALILogic.identify()`, which returns the run report with the
produced files named in it.

## CrownSeg: per-tooth labelling of intraoral scans

```bash
curl -k -X POST https://localhost:8000/run/CrownSeg \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "input=@/path/to/arch.vtk" --output result.zip
```

Adds a point-data array naming the tooth each vertex belongs to — the
precondition for ALI's IOS engine, and for the IOS modes of ASO, AREG and
FlexReg. It is a tool of its own for exactly that reason: burying it inside
ALI would make those three depend on ALI, whose IOS half needs pytorch3d, so
one absent dependency would take four tools out of the registry instead of
one.

`model` is **optional**: a caller that names nothing gets the server's
configured checkpoint (`CROWNSEG_MODEL`), which is what lets ALI ask for crown
segmentation without knowing where CrownSeg keeps its data. Handing the
underlying library no model at all would make it download one from GitHub
mid-request, which a server holding confidential data does not do.

A mesh already carrying tooth labels is passed through untouched unless
`skip_segmented=false` — re-running the network on it costs minutes and
changes nothing.

**From another server-side tool**: `tools.CrownSeg.src.CrownSegLogic.segment_crowns()`.
Needs `shapeaxi` + pytorch3d, same image caveat as ALI's IOS engine.

A tool folder missing its `<name>.py` file, duplicate tool names, or a tool
missing its `name`, all fail loudly at server
startup rather than silently overwriting each other.

## Testing

```bash
./venv/bin/pip install -r requirements-dev.txt   # pytest, httpx
./venv/bin/pytest                                 # everything
./venv/bin/pytest tests/                          # HTTP layer only (main.py, routing, auth)
./venv/bin/pytest tools/SurgMovPred/test/        # one tool's own logic, in isolation
```

Or, without installing anything locally, run the exact same suite in Docker
(see "Pre-push tests" below for why this is the recommended way):

```bash
docker compose run --rm test
```

For a tool with non-trivial internal logic (its own `src/` folder), add a
sibling `test/` folder next to it (`tools/<name>/test/test_<name>_logic.py`)
that imports directly from `tools.<name>.src.<name>_logic` and exercises its
functions without going through HTTP at all. `registry.py` only scans the
immediate children of `tools/` for a `<name>/<name>.py` file, so a nested
`test/` folder is invisible to tool discovery — it's picked up by pytest only.
See `tools/SurgMovPred/test/` for an example covering column-name cleaning,
patient-ID detection, zip extraction, prediction, and the full pipeline
end-to-end with synthetic data.

### Testing against real data (`tests/test_data_integration.py`)

The tests above only ever use synthetic/fabricated data. `tests/test_data_integration.py`
additionally runs each tool that supports server-side data (`ArgSpec(server_selectable=...)`,
see `data_store.py`) against whatever **real** file a maintainer has placed
under `../DATA/<tool_name>/{models,testfiles}/` at the repo root.

`DATA/` is gitignored — it holds confidential medical data and must never be
committed — so those files only ever exist locally. Accordingly, a tool with
no matching file under `DATA/` is **skipped**, not failed: a machine without
the confidential dataset can still run the suite and push. To turn a skip
into a real run, drop a file in the relevant folder, e.g.
`DATA/SurgMovPred/testfiles/my_real_input.zip` (and
`DATA/SurgMovPred/models/my_real_model.zip` if you want the real model
exercised too, instead of leaving that argument unfulfilled and the test skipped).

**This module is opt-in**, and it is the only one that wants a GPU —
everything else stubs its models. It runs when `RUN_REAL_DATA_TESTS` is set,
which is what `test-gpu` is for:

```bash
docker compose run --rm test-gpu               # DEVICE=cuda, GPU reserved, real data
RUN_REAL_DATA_TESTS=1 ./venv/bin/pytest tests/test_data_integration.py
```

Plain `docker compose run --rm test` — what the pre-push hook runs — skips it
at collection and stays around ten seconds.

Two reasons it is not simply always-on. It sends only the arguments it can
fill from `DATA/`, so each tool runs with its **schema defaults**: for ALI
that is all four regions, 119 landmarks at ~6.5 s each, about **11 minutes**
for one scan on a GPU and hours on a CPU. And a hook that takes hours is a
hook people disable. `test-gpu` is a separate service rather than a flag on
`test` because a compose device reservation is all-or-nothing — it cannot
start without an nvidia card, which is why the hook keeps pointing at `test`.

### Pre-push tests

A git hook runs the suite above (via `docker compose run --rm test`, so
nothing needs installing locally) before every `git push`, and blocks the push
if any test fails. It runs `test`, never `test-gpu` — the real-data module is
opt-in for the reasons just given, so the hook stays about ten seconds
whatever is sitting in `DATA/`. Run `test-gpu` by hand when you touch a tool's
inference path.

The hook is **not active by default** — enable it once per clone with:

```bash
git config core.hooksPath .githooks
```

To push without running the suite (e.g. Docker unavailable, or a
documentation-only change), use git's built-in bypass for a single push:

```bash
git push --no-verify
```
