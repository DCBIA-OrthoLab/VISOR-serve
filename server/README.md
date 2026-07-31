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

### Heavy or optional dependencies

`registry.py` imports **every** tool at startup, so a tool whose module-level
imports fail is skipped (see `ADDING_A_TOOL.md`). A tool needing a heavy or
optional stack must therefore import it **lazily, inside the functions that
use it**. `AMASSS` does this for torch/nnunetv2/vtk: those packages are listed
in `requirements.txt` (the deployment image already ships torch built against
its CUDA version, so pip skips it there), but on a machine where they are
missing the server still starts normally, every other tool works, and only
AMASSS fails — with a message saying what to install.

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

### Pre-push tests

A git hook runs the full suite above (via `docker compose run --rm test`,
so nothing needs installing locally) before every `git push`, and blocks the
push if any test fails. It is **not active by default** — enable it once per
clone with:

```bash
git config core.hooksPath .githooks
```

To push without running the suite (e.g. Docker unavailable, or a
documentation-only change), use git's built-in bypass for a single push:

```bash
git push --no-verify
```
