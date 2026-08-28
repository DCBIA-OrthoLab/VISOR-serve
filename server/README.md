# Inference server (tool-registry architecture)

FastAPI server exposing a generic `/run/{tool_name}` endpoint. **It knows no
dental tool.** The tools live in [`sadt-tools`](https://github.com/DCBIA-OrthoLab/SADT-VISOR),
one isolated project each - its own interpreter, its own lockfile, its own
torch - and this server discovers them from `TOOLS_DIR`, publishes them, and
runs them **without importing a line of them**:

```
<TOOLS_DIR>/<tool>/.venv/bin/python <RUNNER_PATH> --job <job dir>/job.json
```

That is the whole architecture. Importing a tool would pin this server's Python
to the lowest common denominator across all of them, make two incompatible pins
(`numpy==2.4.0` against `numpy<2.0.0`) unresolvable, keep a CUDA context alive
for the life of the process, and let a segfault in a CUDA kernel take the API
down with the job.

Two in-process tools remain in `tools/` - `Test_Tool` and `Example_Tool` - as
the demonstration that the `Tool`/`ArgSpec` path still works. Nothing new is
written that way; see [`ADDING_A_TOOL.md`](../ADDING_A_TOOL.md).

Requests are served **in parallel**: each tool execution runs in a worker
thread (never on the event loop), so a long inference never blocks other
requests - `/health`, `/tools` and other `/run` calls all stay responsive
while a tool is working. `MAX_CONCURRENT_TOOLS` (default 4) caps how many run
at once, and `MAX_CONCURRENT_GPU_JOBS` (default 1) caps how many of those may
touch the card - **one counter across all tools**, since an `AMASSS` run and a
`Crown_Seg` run want the same device. The HTTP call itself remains blocking
request/response: the client sends a request and gets the result in the same
response (no job queue / polling).

## Where things live

| | |
|---|---|
| `main.py` | the routes, the upload/response handling, the error mapping |
| `base.py` | `Tool`, `ArgSpec`, `Selection`, `ToolArgumentError` - the object every tool becomes |
| `registry/` | discovery (`__init__`), `.schema.json` → `Tool` (`schema_tool`), `source_hash` (`schema_hash`), zero-config defaults (`conventions`), the exceptions (`deployment`) |
| `execution/` | `dispatch` (server side of a run), `runner` (tool side, stdlib only, injects the supervisor), `parity` (run one tool both ways and diff what a caller receives) |
| `wire/` | `transfer` (chunked uploads, ranged results), `security` (Bearer token) |
| `data_store.py` | `DATA_DIR/<tool>/{models,testfiles}/`, behind a swappable `DataStore` |
| `config.py` | every setting, and nothing reads `os.getenv` outside it |

## The endpoints

| | | auth |
|---|---|---|
| `GET /health` | `{"status": "ok"}` | - |
| `GET /tools` | every tool, its arguments and its layout hints. A client builds its whole UI from this. | - |
| `POST /run/{tool}` | the run. Form fields for scalars, multipart for uploads. | Bearer |
| `GET /tools/{tool}/data` | what is hosted for it: `{models, testfiles}` | Bearer |
| `GET /tools/{tool}/testfiles/{name}` | stream one test file (a folder arrives zipped). **Test files only** - a model is selected by name and used in place. | Bearer |
| `POST /uploads` | open a chunked transfer; answers with the `chunk_size` it will use | Bearer |
| `PUT /uploads/{id}/parts/{n}` | one part, raw body, `X-Part-SHA256` verified before anything is written | Bearer |
| `GET /uploads/{id}` | `missing_parts` - what makes a transfer resumable | Bearer |
| `DELETE /uploads/{id}` | | Bearer |
| `GET /results/{id}` | honours `Range` → `206` | Bearer |
| `DELETE /results/{id}` | release a result held by reference | Bearer |

The chunked endpoints are **optional**: a client that ignores them still works,
and one that uses them against an older server falls back on the `404`. They
exist because a file in one request rides one TCP connection, bound by its
congestion window long before it is bound by bandwidth - and because a
connection dropped at 95% otherwise starts again from zero. An input that
arrived that way is named in the reserved `__uploads__` form field of the run,
and its blob is **renamed** into the job's work directory rather than copied,
so a 2 GB upload becomes a tool's input in microseconds.

A result over `RESULT_REFERENCE_MIN_MB` can be handed back as a reference
(`X-Result-Delivery: reference`) and fetched over parallel ranges. Below that
it is streamed inline, deliberately: a streamed response deletes its file when
the response ends, which fires even when the client disconnects mid-body, while
a reference waits for a `DELETE` or for the reaper. `TRANSFER_TTL_SECONDS` is
an **idle** timeout - every part written and every range read stamps its
directory - so a transfer still in flight is never at risk however long it
takes, and one whose client vanished expires 15 minutes later.

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
institutional CA) - never run this server over plain HTTP outside a fully
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
curl -k -X POST https://localhost:8000/run/Test_Tool \
  -H "Authorization: Bearer change-me-to-a-long-random-secret" \
  -F "text_1=hello" \
  -F "text_2=world"
# -> {"result": "hello world"}
```

`-k` disables certificate verification and is only acceptable against the
self-signed dev certificate above - never use it against a production server.

A tool argument can also expect a file: add `-F "input=@/path/to/data.zip"`
(field name = argument name) to the same call; the server streams it to a
temp dir and passes its path to the tool under that argument name. A tool can
declare more than one file-typed argument, each uploaded as its own multipart
field in the same request.

An argument the server hosts can instead be satisfied by a file already on the
server (under `DATA_DIR/<tool>/{models,testfiles}/`): send the file's *name* as
a plain form value. **Which arguments those are is derived from their names**
(`registry/conventions.py`): `model`, `*_model` and `*_reference` are picked
from `models/` and can never be uploaded - that is a safety property, a
clinician must not be able to send weights from a laptop - while any other
`path` may be uploaded *or* filled from `testfiles/`. An upload aimed at a
model argument is a **400**:

```bash
# List the models/testfiles hosted server-side for a tool (Bearer-protected)
curl -k https://localhost:8000/tools/Surg_Mov_Pred/data \
  -H "Authorization: Bearer change-me-to-a-long-random-secret"
# -> {"models": ["all_models"], "testfiles": ["TestFiles"]}

# Download one of the listed test files (a folder entry arrives zipped).
# Test files only - models are selected by name and never leave the server.
curl -k -O https://localhost:8000/tools/Surg_Mov_Pred/testfiles/TestFiles \
  -H "Authorization: Bearer change-me-to-a-long-random-secret"

# Run it: the model is a name, the input is a genuine upload
curl -k -X POST https://localhost:8000/run/Surg_Mov_Pred \
  -H "Authorization: Bearer change-me-to-a-long-random-secret" \
  -F "model=all_models" \
  -F "measurements=@/path/to/measurements.xlsx"
```

## File-typed arguments

**A packaged tool declares one file type: `path`.** The narrow vocabulary below
belongs to `ArgSpec` and to the two in-process demos. A `path` argument
publishes the extensions its tool named (`extensions` in the schema, from the
tool's `layout.py`); with none, it falls back to `ALLOWED_EXTENSIONS`, which
accepts a `.nii.gz` either way - it just does not offer it in the dialog. A
`.zip` sent for a `path` argument is unpacked by the server, zip-bomb cap and
single-root strip included, because no packaged tool unpacks archives itself.

The rest of this section is the in-process vocabulary. Each file-typed
`ArgSpec` declares its own specific **file type**, from the registry in
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
exactly what's expected - no shared global list to keep in sync as tools with
different file needs are added. To accept a new kind of file, add an entry to
`FILE_TYPES` in `base.py`. Only arguments left as generic `"file"` fall back
to `config.ALLOWED_EXTENSIONS`.

## Choice arguments (pick one or several from a server-defined list)

An argument typed `"choice"` (combo box, exactly one option) or
`"multichoice"` (check boxes, any number of options) declares its options - 
and their defaults - in one `choices` dict, so a client renders the widget
entirely from `GET /tools` with nothing hardcoded:

```python
"structures": ArgSpec(
    type="multichoice", required=False,
    choices={"Mandible": True, "Maxilla": True, "Skull": False},
    description="Anatomical structures to segment",
),
```

A packaged tool declares the same thing in its **signature** - 
`Literal["MERGED", "SEPARATE"]` for exactly-one, `list[Literal[...]]` for
several-of - and `describe.py` publishes the options as `choices`. That is the
point of the move: an `ArgSpec.choices` table was a second declaration and it
drifted, while a `Literal` and the value the tool switches on are the same
list.

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

**You do not add it here.** A tool is a folder in `sadt-tools`; this server
discovers it, publishes it and runs it with no change to this repository - not
a route, not a registration list, not a line of `deployment.toml` in the normal
case. [`ADDING_A_TOOL.md`](../ADDING_A_TOOL.md) is the whole contract; the
short version is one `run()` with annotated parameters, and the schema is
generated from that signature by `describe.py` running in **the tool's own
interpreter**, so it cannot drift from the code.

To see a checkout served without building an image:

```bash
SADT_TOOLS=~/code/sadt-tools ./run-local.sh      # port 8001, schemas generated on start
```

Adding an in-process tool is still possible and is what `Test_Tool` and
`Example_Tool` are: a folder `tools/<name>/` with `tools/<name>/<name>.py` - 
**the file name must match the folder name**, that is the one file discovery
imports - defining a `Tool` subclass with a unique `name`, an `arguments` dict
of `ArgSpec`, and a `run(**kwargs)`. It is documented because the path still
exists, not because anything new should use it.

### After changing `requirements.txt`, recreate the container

The `inference` service installs its requirements as part of its **command**,
so a container that has been up for days is running whatever the file said
when it last *started*. Uvicorn's `--reload` picks up new Python code but never
re-runs pip - which shows up as a `ModuleNotFoundError` for something you can
plainly see in `requirements.txt`.

```bash
docker compose up -d --force-recreate inference
```

`--force-recreate`, not `restart`: `pip --user` writes into the container's
writable layer, so a plain restart keeps whatever the previous resolution
installed - including a torch it should never have replaced. Only a fresh
layer discards it.

This applies to the **dev** image. It does not apply to a packaged tool: its
dependencies are in its own `uv.lock`, resolved into its own `.venv` when the
deployment image is built, and nothing the API installs can reach them.

### `requirements.txt` and `requirements-api.txt`

`requirements-api.txt` is what the API itself needs - fastapi, uvicorn,
python-multipart, pydantic-settings - and a test asserts it stays that way. An
API that quietly regrows numpy is pinned to what the tools can agree on all
over again, which is the problem the split exists to remove.

`requirements.txt` is heavier and is what a dev checkout and the current
`inference` service install. It is legacy weight: the tools that needed torch,
nnUNet, SimpleITK and VTK are gone, and it shrinks when the last in-process
path does.

## How a tool is declared

Two kinds, discovered side by side. `GET /tools` publishes them identically - 
a client cannot tell which is which.

**Declared by a `.schema.json`** (every clinical tool). A folder under
`TOOLS_DIR` holding the schema, the tool's own `.venv/`, and its `src/` - none
of which the server imports:

```
<TOOLS_DIR>/AMASSS/
├── .schema.json     what run() takes, and the hash of the src/ it was read from
├── .venv/           the tool's own interpreter and dependencies
└── src/sadt_amasss/ the tool's code
```

```json
{
  "name": "AMASSS",
  "description": "Segment craniofacial structures on a CBCT scan.",
  "arguments": {
    "scans":      {"type": "path", "required": true,
                   "description": "A CBCT scan, or a folder of them",
                   "extensions": [".nii", ".nii.gz", ".nrrd"]},
    "model":      {"type": "path", "required": true},
    "structures": {"type": "list[str]", "required": false,
                   "default": ["MAND", "MAX", "CB", "CV", "UAW"],
                   "choices": ["MAND", "MAX", "CB", "CV", "UAW", "SKIN",
                               "CBMASK", "MANDMASK", "MAXMASK"],
                   "section": "Structures", "ui": "inline"},
    "device":     {"type": "str", "required": false, "default": "cuda",
                   "choices": ["cuda", "cpu"], "hidden": true}
  },
  "returns": "path",
  "source_hash": "966bf6d9…"
}
```

- **Types** are `path`, `str`, `int`, `float`, `bool` and `list[...]`. A
  `list[str]`/`str` carrying `choices` is a multichoice/choice; without
  `choices` it is free text.
- **`required` comes only from the absence of a default** in `run()`'s
  signature. There is no second declaration that can disagree with it.
- **`returns`** is `path` (one output, or a directory), `dict[str, path]`
  (several named ones) or `text` (any JSON value). A directory or several files
  come back as one archive.
- **`supervisor: true`** means the tool calls another tool and must be handed
  one; `ASO` and `AREG` declare it.
- **Presentation keys** - `label`, `section`, `ui`, `groups`, `visible_when`,
  `options_when`, `hidden` - travel from the tool's `layout.py` to the client
  untouched. `validate()` and `run()` ignore every one of them.
- **An unknown key is a WARNING, never a refusal.** This is the seam between
  two repositories: a field one side adds must not stop the other from
  starting. It must not vanish in silence either, which is why
  `registry/schema_tool.py` names each key it reads rather than passing the
  object through wholesale - the five presentation keys were once dropped here
  and the effect was invisible from both ends.
- **`output_dir` is never published.** Every tool takes it as a required
  argument and writes only there; the server fills it in with the job's own
  `output/`. A client has no business picking a directory on the server, and a
  file picker for one is the fastest way to make every run a 422.

**Dropping a `.schema.json` into a folder is what moves a tool off the imported
path** - a folder that has one is never imported. The folder must be named
after the tool: its interpreter is looked up by tool name.

**The schema is a cache, not a source.** `describe.py` generates it from
`run()`'s signature with the tool's own interpreter - the deployment image at
build time, a dev server at startup, into `SCHEMA_CACHE_DIR` (the tool folders
are read-only to the process serving them). `source_hash` is what says the
cache is behind, and a stale one **regenerates** rather than refusing to start.

**Imported** (`Test_Tool`, `Example_Tool`). `tools/<name>/<name>.py` subclasses
`Tool` and the server imports it at startup. A packaged tool of the same name
supersedes it silently rather than colliding with it - comparison ignores case
and separators, so `Batch_Dental_Seg` and `BatchDentalSeg` are one tool.

### What a discovery failure costs

**One tool, never the server.** A missing dependency, a syntax error, a schema
that cannot be generated: the tool is skipped, reported in a full-width banner
at startup *and* again at the end of it, kept in `FAILED_TOOLS`, and named by
`get_tool()` - so a request for it answers "failed to load at startup", not
"unknown tool", which reads like a typo. With 15+ tools, one unavailable model
must not block all the others.

A folder whose name starts with `_` is not a tool at all and is never
discovered (`_dispatch_probe`, `_AREG`). A folder holding no Python and no
schema is treated as a leftover checkout rather than a broken tool, because git
cannot delete a directory that still holds a `__pycache__/`.

### What `run()` returns

A **`Path`** - normally `output_dir` itself - or a **`dict[str, Path]`** when
there are several outputs worth naming. `describe.py` publishes which, as
`returns`.

```python
return output_dir                              # AMASSS, ALI, ASO, AREG, …
return {"excel": xlsx, "csv": csv}             # Surg_Mov_Pred
```

The names do not reach the client: an HTTP response is one file or one archive,
so a tool whose outputs need identifying writes a report beside them
(`AMASSS_report.json`, `run_report.json`). What the names ARE for is the next
**tool**: a supervisor gets the `dict[str, Path]` back verbatim, which is how
`AREG` feeds one specific mask of an `AMASSS` run into the next step instead of
guessing by extension or by the order paths came back in.

The older `{"outputs": {name: path}}` wrapper and a bare list of paths are both
still accepted by `file_utils.output_paths`; neither is the form to write.

### `source_hash`, and what a mismatch actually does

The schema declares the signature requests are validated against; the hash says
which source tree that signature was read from. If the two drift, the server
validates against a function that no longer exists - an argument the tool has
since renamed passes validation and fails inside the tool, and one it has since
added can never be sent.

**A mismatch REGENERATES the schema**, because the schema is a cache and this
field is what says the cache is behind (`registry/schema_tool.resolve_schema`
runs `describe.py` with the tool's interpreter). What is fatal is a mismatch
that cannot be resolved - no `describe.py`, or a generator that fails - and a
schema carrying **no** hash at all, which is unverifiable and is skipped: it
must not serve, but it endangers only itself.

The hash is sha256 over `<relative posix path>\0<sha256 of contents>\n` per
file, sorted, `__pycache__` and `*.pyc` excluded - every clause of which exists
to make it reproducible on another machine. The generator lives in the other
repository and the two must agree byte for byte, so this ships an executable
reference implementation rather than a description to reimplement:

```bash
python server/registry/schema_hash.py <TOOLS_DIR>/AMASSS/src
```

> Getting this wrong is not theoretical. The first version hashed each file's
> digest and sorted strings while the generator hashed raw bytes and sorted
> `Path` objects - and `a/b.py` sorts before `a.py` one way and after it the
> other, so every tool would have looked stale.

### Conventions, and `deployment.toml` for the exceptions

A schema is generated from the tool's source and is the same wherever that tool
is installed. Which of its arguments may be filled from *this* server's
`DATA_DIR`, how much *this* server accepts as an upload, and how long it lets a
run take are properties of the deployment, not of the tool.

**None of it normally needs writing down.** `registry/conventions.py` derives it
from the argument names:

| named | published as |
|---|---|
| `model`, `*_model`, `*_reference` | a name picked from `DATA/<tool>/models/`, **never an upload** |
| any other `path` | uploadable, and also fillable from `DATA/<tool>/testfiles/` |
| `device`, `gpu_resampling`, `tile_step_size`, `num_workers`, `seed`, … (`TECHNICAL`) | not rendered to a clinician |

The first row is a safety property rather than a convenience: a model is
published as a name so a clinician cannot upload weights from their laptop. The
third is about who owns a knob - those arguments change the result, they are
recorded in the run report, and they are set by whoever deploys the server.

`DATA/` is found by the tool's name with underscores stripped, so
`Batch_Dental_Seg` reads `DATA/BatchDentalSeg/`; a literal match wins wherever
it exists. `server/deployment.toml` is therefore an empty file of comments
today, and that is the measure of whether the conventions are right.

Write a section only for a genuine exception. Anything stated wins **per
argument** over the convention:

```toml
[tools.AMASSS]
server_selectable = { atlas = "model" }   # hosted, but not named *_model
hidden = ["iterations"]                   # technical, not in TECHNICAL
data_dir = "AMASSS_v2"                    # DATA/ folder named differently
max_upload_mb = 2000                      # this tool alone takes more
timeout_seconds = 7200                    # killed after this; 0 means no limit
```

An entry naming an argument the tool does not declare, or a non-`path` one, is
a startup **error** - the failure is otherwise a dropdown that silently never
appears. One naming a tool this server does not serve is a **warning**: a
deployment file outliving one tool must not stop the others.

`max_upload_mb` is enforced where the tool is known: on the multipart body, and
when a chunked upload is claimed by a run. `POST /uploads` opens a session for a
file rather than for a tool, so the global limit still bounds the transfer
itself.

`timeout_seconds` is per tool because the right number differs by two orders of
magnitude - a `Surg_Mov_Pred` prediction is seconds, an `AMASSS` cohort is
hours - and one global value has to be set for the slowest tool, which means
every fast tool that hangs holds a slot until then. The timeout kills the whole
**process group**: nnUNet, torch's DataLoader and shapeaxi all fork workers, and
killing the parent alone leaves those running and holding VRAM, with nothing
left to attribute it to. SIGTERM first, SIGKILL after a grace period.

## How a tool run is executed

`Tool.invoke` validates the arguments and then either calls `run()` or hands
the job to another process. The HTTP request blocks for the run either way, and
the client cannot tell which happened.

| | what happens |
|---|---|
| a folder with a `.schema.json` | **always** dispatched to its own virtualenv, whatever `SADT_DISPATCH_MODE` says |
| an imported tool, `SADT_DISPATCH_MODE=inprocess` (default) | `run()` is called in a worker thread |
| an imported tool, `SADT_DISPATCH_MODE=subprocess` | dispatched the same way, if a venv exists for it |

The flag is temporary and only concerns the two demo tools; it goes away with
the in-process path.

### The job

```
<TOOLS_DIR>/<tool>/.venv/bin/python <RUNNER_PATH> --job <job dir>/job.json
```

with `SADT_API`, `SADT_JOB_ID` and `SADT_JOB_DIR` in the environment, and
**`API_TOKEN` deliberately removed from it** - the server's bearer token has no
business in a venv full of third-party code. `cwd` is the job directory, so a
tool writing a relative path lands there rather than in the server's source
tree.

The runner imports the tool from `<TOOLS_DIR>/<tool>/src/` (finding "the single
package under `src/`", the same rule `describe.py` uses), coerces each value by
its annotation, calls `run(**params)` and writes `result.json`. It is
**standard library only** and runs on Python 3.9 through 3.13, because each
tool pins its own interpreter - and it ships with the SERVER, injected by
absolute path, never installed into a tool venv, so runner and server are
always the same version.

The job directory is a tracked scratch directory, so the request handler
removes it - outputs included - once the response has streamed, and on every
error path. A failed run is deleted immediately: its inputs are confidential
and its outputs are worthless.

### An empty string is absence, and must stay a string

JSON has no path type, so paths arrive as strings and are coerced to `Path` by
annotation. `Path("")` is `PosixPath(".")` - the current directory, and truthy -
so coercing the "not supplied" default of an optional path hands the tool a
real directory. That is not hypothetical: it made `ASO` read an unset
`landmarks=""` as a supplied landmark folder and walk an entire checkout,
`.venv` included.

### How a failure travels back

The runner writes `{"error": {"type": ..., "message": ...}}` and exits non-zero.
There is no shared exception type to catch - there is no shared package - so
`main.py` maps the class **NAME**:

| the tool raised | the caller gets |
|---|---|
| `ToolInputError`, `ValueError`, `FileNotFoundError` | **422**, message passed through verbatim |
| `ToolUnavailableError` | **503** - installed, but its engine is not |
| anything else | **500** with a fixed message |

stdout and stderr go to **files**, not pipes: a tool can print for hours
(nnUNet does) and a pipe means holding all of it in the server's memory. Only
the last 8 kB of stderr travels with a failure, into the server log. Nothing a
tool prints is ever parsed, and nothing it prints is logged - shapeaxi prints
the patient's own file name.

A missing virtualenv is the opposite case and answers **501** through
`ToolUnavailableError`: the request was valid and no request can fix it.

### The supervisor: a tool calling another tool

A tool whose `run()` declares `*, sup` - keyword-only and **unannotated** - is
handed a supervisor and can call a sibling. That call re-enters `runner.py`
with the sibling's interpreter, so chaining and nesting are one recursion
rather than a feature, and the callee gets its own venv and its own dependency
set. `ASO` needs landmarks **mid-run**, after it has recentred its scans;
`AREG` drives four tools.

Five members, duck-typed, nothing shared: `sup.run(tool, **params)` (blocking,
returns what that tool's `run()` returned), `sup.out`, `sup.tmp`,
`sup.progress(fraction, message)`, `sup.log(message)`. A tool never imports the
class - it cannot, its venv holds none of the server - and the same shape is
produced by `sadt-tools`' `scripts/run_tool.py` and faked in its tests.

- **A cycle is refused by name.** The chain travels in the environment, so a
  tool asking for one already running above it fails immediately, naming the
  chain, rather than after starting four processes. Five deep is the backstop.
- **It does not go back through the server.** A nested call is a subprocess of
  its parent, so it never queues for a slot the parent is already holding - 
  which is exactly the deadlock the in-process version had, where four
  concurrent `ASO` runs each waited on a fifth slot. The cost: nested work is
  invisible to `MAX_CONCURRENT_GPU_JOBS`, so a deployment running several
  supervised jobs at once has to size for more than one tool on the card.
- Nested jobs live under `<job>/sup/NN_<tool>/` and are removed with the job.

### Who gets the card

`MAX_CONCURRENT_GPU_JOBS` is **one counter across all tools** - the per-tool
semaphores went with the tools that held them, and would cap nothing now that
each tool is its own process anyway.

**A run is assumed to want the GPU** unless it declares `device` and resolves
it to a CPU value. The safe default is the strict one: a tool that quietly
imports torch without declaring `device` would otherwise never queue at all,
and two of them would meet on the card with nothing between them. Being wrong
the other way costs a CPU-only run a slot it did not need; being wrong this way
is an out-of-memory in the middle of somebody's cohort.

`device` itself is injected from `settings.DEVICE` when the caller picks none,
because a tool that no longer reads the environment would otherwise run on its
own default (`cuda`) on a CPU server.

### Proving two paths agree

`execution/parity.py` runs one tool in both forms on the same arguments and
compares **what a caller receives**: every file produced, by name and by hash,
and the returned value with paths resolved to the artifacts they name. Absolute
paths are never compared - one run wrote into a job directory, the other into a
scratch directory, and neither name means anything to a client.

```bash
cd server && python -m execution.parity --imported AMASSS --args case.json
```

`tools/_dispatch_probe/` is the fixture the dispatch path is tested with - a
tool that needs no dependency at all, so a failure there is a failure of the
machinery and of nothing else.

## The tools this server serves

Every clinical tool lives in `sadt-tools`, and **its own `README.md` there is
authoritative** for what it computes, what it was validated against and which
pins produced that result. The sections below are the server-facing view:
what to send, what comes back, and which bundle has to be staged under `DATA/`.
When one disagrees with `GET /tools`, `GET /tools` is right - it is generated
from the code.

### AMASSS - CBCT skull structure segmentation

```bash
# What model bundles / test scans are hosted server-side
curl -k https://localhost:8000/tools/AMASSS/data \
  -H "Authorization: Bearer $API_TOKEN"

# One scan, mandible + maxilla, merged output
curl -k -X POST https://localhost:8000/run/AMASSS \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "model=AMASSS_Models" \
  -F "scans=@/path/to/scan.nii.gz" \
  -F "structures=MAND,MAX" \
  -F "merge=MERGED" \
  --output result.zip

# Both output forms at once: merge is a multichoice, so it takes the
# comma-separated shorthand (or the full JSON object) like structures does.
  -F "merge=MERGED,SEPARATE"
```

`structures` and `merge` are both **`list[Literal[...]]`** in the tool's
signature, so they speak the codes the network emits - `MAND`, `MAX`, `CB`,
`CV`, `UAW`, `SKIN`, `CBMASK`, `MANDMASK`, `MAXMASK`, and `MERGED`/`SEPARATE`.
An option outside that list is a **422** naming what was expected. `merge` was
briefly exactly-one, which silently produced runs with no segmentation files in
them at all.

Models live under
`DATA/AMASSS/models/<bundle>/<CODE>/**/*__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth`
(one subfolder per structure code) and are selected **by name** - they never
travel from the client. `scans` takes one scan or a folder of them (send a
`.zip`; the server unpacks it).

The response is a zip: one `<scan>_<ID>_SegOut/` folder per scan plus an
`AMASSS_report.json` listing what was predicted, what was skipped for lack of a
model, and what failed - a structure without a model used to disappear with
nothing but a log line, which is invisible in a 200-scan batch.

Segmentations are **always written compressed**, whatever the input was: a scan
sent as a plain `.nii` comes back as `.nii.gz` masks. These are label volumes,
so the difference is not marginal - 191 MB down to 0.4 MB per structure on a
0.33 mm CBCT, measured. The input's format is preserved (`.nrrd` stays `.nrrd`,
compressed internally - ITK has no `.nrrd.gz` writer); only the compression is
imposed.

Surfaces (`generate_surface`) are binary `.vtk`, decimated by
`surface_decimation` (default 90%, which costs a fifth of a voxel on average
and buys a factor of ten in triangles). `tile_step_size` and `gpu_resampling`
trade accuracy for speed and are `hidden` - a clinician is not asked - but both
are recorded in the run report, because they change the mask.

**Calling AMASSS from another tool**: `sup.run("AMASSS", scans=…, model=…,
output_dir=…)`. `AREG`'s fully-automated CBCT mode does exactly that.

### ALI - automatic landmark identification

Places anatomical landmarks and writes Slicer markups (`.mrk.json`). Two
engines that share nothing but their output format, and - since the split that
gave each its own pins - **two tools**:

- **`ALI_CBCT`** - one deep-RL agent per landmark walks the volume at 1 mm and
  then at 0.3 mm until it converges. 119 landmarks across four regions
  (`regions`: Cranial base, Upper, Lower, Impacted canine), or named
  individually through `landmarks`.
- **`ALI_IOS`** - per tooth, the mesh is rendered from a dozen viewpoints and a
  2D UNet predicts masks projected back onto the surface. `networks`:
  **Occlusal**, **Cervical**, **Mucogingival**.

```bash
# One CBCT scan, cranial base only
curl -k -X POST https://localhost:8000/run/ALI_CBCT \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "model=ALI_CBCT_Models" \
  -F "input=@/path/to/scan.nii.gz" \
  -F "regions=Cranial base" \
  --output result.zip

# A whole cohort as one archive, asking for seven landmarks by name
zip -r cohort.zip cohort/
curl -k -X POST https://localhost:8000/run/ALI_CBCT \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "model=ALI_CBCT_Models" -F "input=@cohort.zip" \
  -F "landmarks=Ba,S,N,RPo,LPo,ROr,LOr" --output result.zip
```

**Naming landmarks REPLACES the region selection** rather than narrowing it.
`ASO` registers on seven points straddling two regions, so asking by region
would run 58 agents to use 7 - and one agent is a full two-scale walk of the
volume. Narrowing would agree for ASO only because it leaves the regions all
on, and would silently drop landmarks for a caller that set both. The run
report says which drove the run.

The 119 options are readable because the schema says how to group them: the
tool's `layout.py` publishes `ui: "tabs"` with `groups` **computed from the
same catalog the engine names its output files by** - so a landmark added to
the catalog gets its tab with no client release, and cannot be published
without one.

The response is a zip mirroring the input's folder tree, with **one
`<scan>_lm_<ID>.mrk.json` per scan** holding every landmark found (the two
original CLIs wrote one file per anatomical region, and disagreed on the
extension), plus a `run_report.json`. The report separates *"the bundle has no
weights for this landmark"* (`landmarks_without_model` → use another bundle)
from *"the agent did not converge on this scan"* (`landmarks_failed` → this
scan is hard). Both look identical in the Slicer scene and need opposite fixes.

Model bundles are selected by name and never travel. CBCT bundles hold
`<landmark>/<scale>/*.pth` with scale folders named `1` and `0-3`; IOS bundles
hold checkpoints named with an `O`/`C`/`MG` token and an `Upper`/`Lower` one,
e.g. `Upper_O_model.pth`.

`ALI_IOS`'s **Mucogingival** network places one point per lower tooth on the
gingival margin. It is lower-jaw only (that is what it was trained on, so a
maxilla is skipped rather than reported as a missing model) and it is what
`AREG`'s lower-arch registration asks for. A point placed from a fit of the
arch - or forced out of the most likely pixels when the class won none - 
carries the reason in its own `description` field and in the report's
`landmarks_degraded`, because it looks exactly like a good one otherwise.

**`ALI_IOS` needs pytorch3d**, which publishes no usable wheel and is compiled
from source against the pinned torch. It is an optional extra of that tool, so
the tool loads and publishes its schema either way and only a *run* answers
**503**, naming the command to run. `ALI_CBCT` needs neither pytorch3d nor the
torch version it forces, which is precisely why the two were split.

`search_seconds` bounds one CBCT landmark's search and is `hidden`; `device`
selects cuda/cpu and is likewise the deployment's business.

### Crown_Seg - per-tooth labelling of intraoral scans

```bash
curl -k -X POST https://localhost:8000/run/Crown_Seg \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "model=07-21-22_val-loss0.169.pth" \
  -F "meshes=@/path/to/arch.vtk" --output result.zip
```

Adds a point-data array naming the tooth each vertex belongs to - the
precondition for `ALI_IOS`, and for the IOS modes of `ASO`, `AREG` and FlexReg.
It is a tool of its own for exactly that reason: burying it inside ALI would
make those three depend on a tool whose IOS half needs pytorch3d.

`model` is a `.pth` file, not a folder, and is picked by name from
`DATA/Crown_Seg/models/`. It is **required**: the library's own fallback
downloads a checkpoint from GitHub mid-request, and a server holding
confidential data does not make outbound calls.

`numbering` picks `Universal` or `FDI`. A mesh already carrying tooth labels is
passed through untouched unless `skip_segmented=false` - re-running the network
on it costs minutes and changes nothing. `run_report.json` names every mesh
that now carries labels, whether this run produced them or found them, which is
what the next tool in a chain reads.

> **A guard that was never reached.** `Crown_Seg` imports its segmentation
> engine inside the branch that segments, so a batch of already-labelled meshes
> returned a clean report on a deployment where the engine could not run at
> all. A tool's availability must not depend on its input data - one that can
> serve some batches and not others is not available, it has only not been
> asked the right question yet.

**From another tool**: `sup.run("Crown_Seg", meshes=…, model=…, output_dir=…)`.

### Batch_Dental_Seg - teeth and jaw structures on dental CT/CBCT

```bash
curl -k -X POST https://localhost:8000/run/Batch_Dental_Seg \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "model=PediatricDentalSeg" \
  -F "scans=@/path/to/cohort.zip" \
  -F "separate_segments=true" --output result.zip
```

**The hosted bundle name IS the model.** Four trained networks label different
things - DentalSegmentator and PediatricDentalSeg put the maxilla inside Upper
Skull, NasoMaxillaDentSeg separates it (which shifts every later value), and
UniversalLab labels all 32 permanent teeth, 20 deciduous ones and 3 structures -
so the folder's basename selects the weights **and** their label table
together. An earlier version had a second `dental_model` choice beside it,
which meant a caller could pair bundle X with the table of Y and get a
plausible volume with every structure named wrong. The label table used is
published with every run, for the same reason.

### ASO - automated standardized orientation

```bash
# Fully-automated CBCT: the landmarks are predicted mid-run, by ALI
curl -k -X POST https://localhost:8000/run/ASO \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "modality=CBCT" -F "automation=Fully-Automated" \
  -F "landmark_model=CBCT_landmark_models" \
  -F "reference=CBCT_Gold_Occlusal_Midsagittal_Plane" \
  -F "input=@cohort.zip" --output result.zip
```

Four modes across two modalities and two automation levels. `modality` and
`automation` are explicit choices, never inferred: a `.zip` can hold either
kind of data, and guessing wrong orients a patient against the wrong reference
and calls it a success. `visible_when` is what keeps a CBCT user from being
shown 32 teeth and 8 landmark types.

**This is the tool the supervisor exists for.** Fully-automated CBCT recentres
each scan, predicts landmarks **on the centred volumes**, then registers - the
order the original Slicer chain used. Running the landmark tool first and
handing ASO the markups reorders those two steps, and that reordering is not
provably exact. So `ASO` declares `*, sup` and calls `ALI_CBCT` from the middle
of its own run. Passing `landmarks` (a folder of `.mrk.json`) skips the call
entirely, which is also what makes ASO usable with no supervisor at all.

**The two published references carry disjoint landmark sets** - Frankfurt
Horizontal + Midsagittal wants `Ba, S, N, RPo, LPo, ROr, LOr`, Occlusal +
Midsagittal wants `ANS, IF, PNS, UL6O, UR1O, UR6O`. Picking one and leaving the
other's defaults would drop every landmark as "not in the reference" and fail
all forty patients separately; it is one **422** naming what the reference
offers, raised after discovery but before a scan is read.

The `.tfm` a run returns maps **oriented → original**, recentring included.
That direction is asserted by a test rather than assumed, because getting it
backwards is silent.

### Surg_Mov_Pred - surgical movement prediction

```bash
curl -k -X POST https://localhost:8000/run/Surg_Mov_Pred \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "model=all_models" \
  -F "measurements=@/path/to/patients.xlsx" --output result.zip
```

Tabular in, tabular out: it is the one tool here that never touches an image,
returns `dict[str, Path]` (`excel` and `csv`) rather than a directory, and
declares no `device` - so it never queues for the card.

### AREG - registering two timepoints onto each other

```bash
# Semi-Automated CBCT: you send the T1 masks the registration is confined to
curl -k -X POST https://localhost:8000/run/AREG \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "modality=CBCT" -F "automation=Semi-Automated" \
  -F "cbct_regions=Cranial base,Mandible" \
  -F "t1=@/path/to/T1.zip" -F "t2=@/path/to/T2.zip" \
  -F "t1_masks=@/path/to/masks.zip" --output result.zip

# Fully-Automated CBCT: AMASSS produces those masks server-side
curl -k -X POST https://localhost:8000/run/AREG \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "modality=CBCT" -F "automation=Fully-Automated" \
  -F "cbct_regions=Cranial base" \
  -F "segmentation_model=AMASSS_Models" \
  -F "t1=@/path/to/T1.zip" -F "t2=@/path/to/T2.zip" --output result.zip
```

Five modes, two engines:

|          | Semi-Automated                  | Fully-Automated                 | Oriented + Fully-Automated |
|----------|---------------------------------|---------------------------------|----------------------------|
| **CBCT** | your T1 masks, masked elastix   | AMASSS segments the T1 masks    | ASO orients the T1 first   |
| **IOS**  | your segmented, oriented meshes | `Crown_Seg` labels + `ASO` orients | -                       |

`modality` and `automation` are explicit arguments, never inferred: a `.zip`
can hold either kind of data, and guessing wrong registers a patient's
follow-up against the wrong anatomy and calls it a success. A mode a modality
does not have (`IOS` + `Oriented`) is a **422** naming what that modality
offers.

**T1 and T2 are paired by name**, up to the timepoint token and a trailing
`_scan`/`_Or`/`_Seg`…, so `P1_T1_scan.nii.gz` in one folder pairs with
`P1_T2.nii.gz` in the other. Subjects present at only one timepoint are listed
in `AREG_report.json` rather than dropped in silence.

**CBCT: each region is a separate registration.** `cbct_regions` runs one
masked registration per selected region into its own output folder
(`CB/`, `MAND/`, `MAX/`) - registering on the cranial base and on the mandible
answer two different clinical questions. In Semi-Automated mode a mask has to
say **both** that it is a segmentation (`mask`/`seg`/`pred`) and which
structure it covers (`cb`/`mand`/`max`), matched as whole tokens of the name:
`P1_T1_CB_seg.nii.gz`. `segmentation_label` picks one label out of a
multi-label mask; a label the mask does not hold is refused rather than
silently falling back to the whole mask.

**The `.tfm` a run returns maps the T1 frame to the T2 frame you sent**, which
is what `sitk.ResampleImageFilter` consumes - so resampling your own T2 with it
reproduces the registered volume in the archive. (In the automated IOS modes,
where the meshes are oriented before registration, the orientation is composed
in, so the transform still refers to the mesh you sent.)

**IOS registers on a region that does not move**, and which region that is
depends on the arch - `ios_patch` picks one, and with it the arch:

- **Palate (upper arch)** - a network predicts the patch on each maxilla and the
  two are aligned by ICP; the mandible is carried by the maxilla's transform, so
  the occlusion the pair was captured in survives. Needs pytorch3d (same image
  caveat as ALI's IOS engine) and the `registration_model` checkpoint.
- **Mucogingival line (lower arch)** - the mandible has no palate, but it has the
  mucogingival line. The 13 MG landmarks are joined into a spline, every sample
  is snapped onto the mesh, and a band grows **along the surface** (geodesic, so
  a buccal patch cannot leak to the lingual side where the ridge is thin);
  vertices that land on a crown are dropped, because the crowns are what moves.
  `mgl_patch_height` is the half-height in mm, and **0** registers on the
  landmarks alone - the control case for measuring what the surface around the
  line adds.

**The mucogingival band itself needs no model and no GPU** - a spline, a
shortest-path walk and a label lookup - so a server without pytorch3d registers
lower arches at full speed while answering **503** for the palate. The landmarks
it is built from come from `ALI_IOS`'s Mucogingival network, **through the
supervisor**: send nothing and they are produced for both timepoints.
`mgl_landmarks` is there only to reuse landmarks you already have, which also
skips paying for the prediction twice - and it is optional, so an empty file
row must not block Apply on the client either.

```bash
# The landmarks are predicted server-side; nothing to supply but the scans
curl -k -X POST https://localhost:8000/run/AREG \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "modality=IOS" -F "automation=Semi-Automated" \
  -F "ios_patch=Mucogingival line (lower arch)" \
  -F "mgl_patch_height=5.0" \
  -F "t1=@T1.zip" -F "t2=@T2.zip" \
  --output result.zip
```

**AREG drives four tools**, all through the supervisor: `AMASSS` for the T1
masks its fully-automated CBCT mode registers on, `ASO` for the orientation the
third mode adds, `Crown_Seg` for tooth labels, and `ALI_IOS` for the
mucogingival landmarks. It is the deepest chain here - `AREG → ASO → ALI_CBCT`
is three levels - and the reason the depth cap is five rather than four.

**Not ported: the IOSCBCT mode** (registering an intra-oral scan onto a CBCT of
the same patient). It is a genuinely different problem - cross-modality,
landmark-driven, its own four-folder input contract - and 829 lines upstream
that this port does not have, with no entry in its catalogs and no schema field
to say so. It will need the supervisor too: it orchestrates both chains.

The CBCT engine is elastix on the CPU; the IOS patch prediction is the GPU half
and queues on `MAX_CONCURRENT_GPU_JOBS` like everything else. Both dependencies
are in that tool's own lockfile, not in `requirements.txt`.

## Testing

```bash
./venv/bin/pip install -r requirements-dev.txt   # pytest, httpx
./venv/bin/pytest                                 # the server's own suite
```

Or, without installing anything locally, run the exact same suite in Docker
(see "Pre-push tests" below for why this is the recommended way):

```bash
docker compose run --rm test
```

**220 tests, about six seconds**, and no GPU, no model weights and no network:
every inference is stubbed and everything around it runs for real.

`pytest.ini` collects **only `tests/`**, deliberately. A packaged tool's own
tests belong to that tool and run in ITS interpreter against ITS pinned
dependencies - collecting them with the API's interpreter would import a tool
the API cannot import. They are not skipped here, they are somebody else's job:
`cd <TOOLS_DIR>/<tool> && uv run pytest`.

What the server's suite covers is the seam rather than the science: the
discovery of both kinds of tool, the schema vocabulary, `source_hash` against
the real generator (`test_tool_contract.py` runs `sadt-tools`' `describe.py`
with the real tool interpreters, and skips where that checkout is absent), the
dispatch loop through `_dispatch_probe`, the supervisor, the transfer
endpoints, and `tests/golden/tools_response.json` - `GET /tools` captured per
tool, per argument, in order. The Slicer client builds its entire UI from that
response, so **if the golden test fails, the client breaks; the fixture is not
what gets updated.**

### Testing against real data (`tests/test_data_integration.py`)

The tests above only ever use synthetic/fabricated data.
`tests/test_data_integration.py` additionally runs each tool whose required
arguments can all be filled from the server's own data store (see
`data_store.py` and `registry/conventions.py`) against whatever **real** file a
maintainer has placed under `../DATA/<tool_name>/{models,testfiles}/` at the
repo root.

`DATA/` is gitignored - it holds confidential medical data and must never be
committed - so those files only ever exist locally. Accordingly, a tool with
no matching file under `DATA/` is **skipped**, not failed: a machine without
the confidential dataset can still run the suite and push. To turn a skip
into a real run, drop a file in the relevant folder, e.g.
`DATA/SurgMovPred/testfiles/TestFiles/` (and `DATA/SurgMovPred/models/all_models/`
if you want the real model exercised too, instead of leaving that argument
unfulfilled and the test skipped).

**This module is opt-in**, and it is the only one that wants a GPU - 
everything else stubs its models. It runs when `RUN_REAL_DATA_TESTS` is set,
which is what `test-gpu` is for:

```bash
docker compose run --rm test-gpu               # DEVICE=cuda, GPU reserved, real data
RUN_REAL_DATA_TESTS=1 ./venv/bin/pytest tests/test_data_integration.py
```

Plain `docker compose run --rm test` - what the pre-push hook runs - skips it
at collection and stays around ten seconds.

Two reasons it is not simply always-on. It sends only the arguments it can
fill from `DATA/`, so each tool runs with its **schema defaults**: for
`ALI_CBCT` that is all four regions, 119 landmarks at ~6.5 s each, about
**11 minutes** for one scan on a GPU and hours on a CPU. And a hook that takes hours is a
hook people disable. `test-gpu` is a separate service rather than a flag on
`test` because a compose device reservation is all-or-nothing - it cannot
start without an nvidia card, which is why the hook keeps pointing at `test`.

### Pre-push tests

A git hook runs the suite above (via `docker compose run --rm test`, so
nothing needs installing locally) before every `git push`, and blocks the push
if any test fails. It runs `test`, never `test-gpu` - the real-data module is
opt-in for the reasons just given, so the hook stays about ten seconds
whatever is sitting in `DATA/`. Run `test-gpu` by hand when you touch a tool's
inference path.

The hook is **not active by default** - enable it once per clone with:

```bash
git config core.hooksPath .githooks
```

To push without running the suite (e.g. Docker unavailable, or a
documentation-only change), use git's built-in bypass for a single push:

```bash
git push --no-verify
```
