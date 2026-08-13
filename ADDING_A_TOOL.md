# Adding a tool to the inference server

This guide is the long-form companion to the short "How to add a new tool"
section in [`server/README.md`](README.md). It covers the whole contract: the
folder layout, the argument schema, server-side data, outputs, dependencies,
and tests.

**The one thing to remember:** a tool is a self-contained folder. You never
edit `main.py`, `registry.py`, or `base.py` to add one. No route, no
registration list.

> **Which kind of tool is this?** Everything below describes a tool the server
> **imports** — a `Tool` subclass under `server/tools/`. There is now a second
> kind, declared by a `.schema.json` and run in its own virtualenv, which the
> server never imports; it is packaged in the `sadt-tools` repository and
> described in [`server/README.md`](server/README.md#how-a-tool-is-declared).
> The two are discovered side by side and are indistinguishable over the wire.
> Follow this guide for a tool that lives in this repository.

---

## 1. The 30-second version

```bash
mkdir -p server/tools/my_tool
touch server/tools/my_tool/__init__.py
```

```python
# server/tools/my_tool/my_tool.py
from base import ArgSpec, Tool


class MyTool(Tool):
    name = "my_tool"
    arguments = {
        "scan": ArgSpec(type="nifti_file", required=True, description="CBCT scan to process"),
        "threshold": ArgSpec(type=float, required=False, description="Binarization threshold"),
    }
    output_kind = "text"

    def run(self, scan: str, threshold: float = 0.5) -> str:
        return f"processed {scan} at {threshold}"
```

Restart the server. `GET /tools` now lists `my_tool`, and
`POST /run/my_tool` works. That's the whole thing.

---

## 2. Folder layout — the rules the registry enforces

`registry.py` scans the **immediate children** of `tools/` at startup. For a
folder named `X`, it imports exactly one file: `tools/X/X.py`.

```
server/tools/my_tool/
├── __init__.py            # required, may be empty
├── my_tool.py             # REQUIRED, name must match the folder
├── src/                   # optional: your actual logic
│   ├── __init__.py
│   └── my_tool_logic.py
└── test/                  # optional: your unit tests (invisible to discovery)
    └── test_my_tool.py
```

Three rules, all checked at startup:

| Rule | What happens if you break it |
| --- | --- |
| `tools/X/` must contain `X.py` | `Tool folder 'tools/X/' is missing its 'tools/X/X.py' file.` |
| Every `Tool` subclass must set a non-empty `name` | `Tool class 'MyTool' has no 'name' set.` |
| Tool `name`s must be unique across the whole registry | `Duplicate tool name detected: 'my_tool'` |

### When a tool fails to load

Breaking one of those rules — or a missing dependency, a syntax error, an
invalid `ArgSpec` — does **not** take the server down. With 15+ tools, one
unavailable model must not block all the others. The tool is skipped and the
failure is printed as a banner with its full traceback, then summarised again
at the very end of startup so it survives the wall of uvicorn output:

```
==============================================================================
  TOOL FAILED TO LOAD:  AMASSS
  ModuleNotFoundError: No module named 'SimpleITK'

  This tool is NOT registered. Every other tool still works.
------------------------------------------------------------------------------
Traceback (most recent call last):
  ...
==============================================================================
```

The tool is then absent from `GET /tools`, and `POST /run/<name>` answers
**404** with `failed to load at server startup` rather than `Unknown tool` —
so a client developer isn't sent hunting for a typo.

`registry.FAILED_TOOLS` maps each failed folder to its reason. Because a
skipped tool is otherwise silent, `tests/test_registry.py` asserts that map is
empty: that test is what turns a silently missing tool back into a red build.
If it fails with a `ModuleNotFoundError`, the environment is usually just
stale — `pip install -r requirements.txt`.

Everything else in the folder (`src/`, `test/`, data, helpers) is invisible to
discovery — but `my_tool.py` is free to import from it. Folders starting with
`_` or `.` are skipped entirely.

### Imports inside a tool

The server runs with `server/` as its working directory, so the core modules
are imported **absolutely**, and your own submodules **relatively**:

```python
from base import ArgSpec, Tool                      # core: absolute
from config import settings                         # core: absolute
from file_utils import extract_zip                  # core: absolute
from .src.my_tool_logic import MyToolLogic          # your own code: relative
```

Getting this backwards is the single most common startup failure.

---

## 3. Declaring arguments (`ArgSpec`)

```python
ArgSpec(
    type=...,                  # str | int | float | bool, a file-type key,
                               # "choice" / "multichoice", or a TUPLE of
                               # file-type keys — see below
    required=True,             # default True
    description="",            # shown to clients via GET /tools — write a real one
    server_selectable=None,    # "model" | "testfile" | None — see §5
    choices=None,              # {name: on_by_default} — "choice"/"multichoice" only
)
```

`registry.py` calls `Tool.check_schema()` at startup, so an invalid declaration
(unknown type, a scalar mixed with a file type) raises `ToolSchemaError` on
boot rather than on the first request that happens to reach that tool.

`Tool.validate()` runs **before** `run()` on every request. It rejects missing
required arguments, unknown arguments, and un-coercible types with a
`ToolArgumentError`, which `main.py` maps to **HTTP 422** with the message.
By the time `run()` is called, its inputs are guaranteed valid — no defensive
checks needed inside it.

Coercion is deliberately narrow, because everything arrives as multipart form
strings: `int`/`float` go through the constructor; `bool` accepts
`true/false/1/0` (case-insensitive); `str` must already be a string.

### Scalar arguments

```python
"label":      ArgSpec(type=str,   required=True,  description="Free-text run label"),
"threshold":  ArgSpec(type=float, required=True,  description="Numeric threshold"),
"iterations": ArgSpec(type=int,   required=False, description="Optional iteration count"),
"smooth":     ArgSpec(type=bool,  required=False, description="Apply Laplacian smoothing"),
```

For an optional argument, give the Python parameter a default in `run()` —
`validate()` simply omits absent optional arguments from the call. (Choice
arguments are the one exception: see below.)

### Choice arguments — combo box and check boxes

When an argument picks from a fixed set of options, declare them instead of
taking a free-form `str`. You get the right widget on the client, and an
out-of-range value is rejected by `validate()` with a **422** instead of
reaching `run()`.

Both types share one declaration shape: **a dict of option name → is it on by
default**. They differ only in how many options can be on at once.

```python
# Combo box — exactly one option. The True one is the default.
"preview_format": ArgSpec(
    type="choice", required=False,
    choices={"csv": True, "json": False},
    description="Format of the preview file",
),

# Check boxes — any number of options.
"structures": ArgSpec(
    type="multichoice", required=False,
    choices={"mandible": True, "maxilla": True, "skull": False},
    description="Anatomical structures to segment",
),
```

What `run()` receives:

| Type | Value | Use it as |
| --- | --- | --- |
| `"choice"` | the chosen name, a `str` | `if preview_format == "json":` |
| `"multichoice"` | a `base.Selection` — a `dict` of **every** declared option → bool | `if structures["skull"]:` or `for s in structures.selected:` |

`Selection` always holds every option the tool declared, in declaration order,
so `structures["skull"]` can't raise `KeyError` and `.get(name, False)` is
never needed. `.selected` is the tuple of enabled names.

> **Pick the right one of the two, and check what `run()` actually receives.**
> The types differ in the *shape* of that value, not just in how many options
> may be on — `"choice"` gives a bare `str`, `"multichoice"` a `Selection`. Code
> written for one and fed the other does not raise: a `str` is iterable, so
> `tuple(value)` yields its **characters**, every membership test fails, and the
> tool takes the "nothing was selected" path while reporting success. AMASSS
> shipped exactly that — its `merge` argument was declared `"choice"` while its
> description said "Both may be selected" — and a nine-structure run spent 154s
> on the GPU to return a report saying it had produced no files. If an argument
> can ever have two options on at once, it is `"multichoice"`; and validate the
> translated codes before doing minutes of work on them, so a mismatch fails
> loudly at the front instead of quietly at the end.

**The declared booleans are also the defaults.** If the caller omits an
optional choice argument, `validate()` hands `run()` exactly what `choices`
declared — so don't repeat those defaults in `run()`'s signature, where the two
copies would drift apart.

On the wire, `"choice"` is just the option name. `"multichoice"` accepts a JSON
object, or a comma-separated shorthand of the enabled ones:

```bash
-F 'structures={"mandible": true, "maxilla": false, "skull": true}'
-F 'structures=mandible,skull'          # equivalent
```

Either form is the **complete** selection: an option you don't mention is off,
regardless of its declared default. Omitting the argument entirely is what
falls back to the defaults.

Rules `check_schema()` enforces at startup: `choices` is required on these two
types and forbidden on every other one, names are non-empty strings mapping to
real booleans, and a `"choice"` declares exactly one `True`.

### File arguments

A file argument declares **its own** accepted extensions via a key of
`base.FILE_TYPES`, rather than relying on a server-wide whitelist:

```python
FILE_TYPES = {
    "file":       None,                 # generic → falls back to config.ALLOWED_EXTENSIONS
    "folder":     (".zip",),            # sent as a zip, reaches run() as a DIRECTORY
    "zip_file":   (".zip",),
    "csv_file":   (".csv",),
    "xlsx_file":  (".xlsx",),
    "ods_file":   (".ods",),
    "nifti_file": (".nii", ".nii.gz"),
    "volume_or_zip_file": (...),        # one medical volume OR a zip of a folder of them
    "surface_file": (".vtk", ".vtp", ".stl", ".obj", ".off"),   # a 3D mesh
}
```

Every extension a type lists is one the tool declaring it must be able to
**read**: ALI's IOS mode advertised `.stl` and then discovered only `.vtk`, so
a caller's meshes were accepted and silently never processed.

```python
"scan": ArgSpec(type="nifti_file", required=True, description="Oriented CBCT (.nii/.nii.gz)"),
```

What this buys you:

- `main.py` validates the uploaded file's extension against **that argument's**
  list (compound extensions like `.nii.gz` are handled correctly) and returns
  **400** otherwise.
- `GET /tools` reports `"type": "nifti_file"`, so the Slicer client can put the
  right filter on its file picker without any tool-specific client code.
- No global list to keep in sync as tools with unrelated file needs pile up.

**`run()` receives a local path**, not bytes or a file object. `main.py` has
already streamed the upload to disk in 1 MB chunks (never loading it into RAM)
at `<work_dir>/<argument_name><ext>`.

A tool may declare **several** file arguments (`fixed_image` + `moving_image`);
each is uploaded as its own multipart field in the same request.

To accept a file kind that doesn't exist yet, add one entry to `FILE_TYPES` in
`base.py` — that is the only core edit a new tool can legitimately require.

### Folder arguments

HTTP has no notion of a directory, so `"folder"` is the one type where the
server does work on your behalf: the client uploads a `.zip`, `main.py`
extracts it, and **`run()` receives the path of a real directory**. If the
archive holds a single root folder (what zipping `patients/` produces on every
OS), that wrapper is stripped, so your tool sees the files directly.

Extraction of client archives is hardened in `file_utils.extract_zip()`: members
escaping the extraction directory (zip slip) and symlink members are refused,
and the uncompressed total is capped by `settings.MAX_EXTRACTED_MB` — all three
return a **400**. Don't reimplement this; call the helper.

### Arguments accepting several types

Pass a **tuple** when one argument legitimately takes more than one shape —
almost always "one file, or a whole folder of them":

```python
"input": ArgSpec(
    type=("csv_file", "folder"),
    required=True,
    description="A single .csv, or a folder of .csv/.xlsx/.ods files sent as a .zip",
),
```

`run()` then receives a `base.ResolvedPath`: a plain path **string** that also
carries `.kind`, the declared type it was actually resolved as.

```python
def run(self, input: str) -> list:
    if input.kind == "folder":
        table = file_utils.load_tabular_directory(input)
    else:
        table = file_utils.load_tabular_file(input)
```

- `.kind` is always one of the types **you** declared, so branching on it is
  exhaustive — no `os.path.isdir()` probing, no extension re-parsing.
- It subclasses `str`, so `open()`, `os.path.*` and `pandas.read_csv()` all keep
  working, and a single-type argument can ignore `.kind` entirely.
- **Declaration order breaks ties** when two types share an extension:
  `("zip_file", "folder")` hands you the archive, `("folder", "zip_file")`
  extracts it first.
- Only file types can be combined — a scalar type must stand alone.

`tools/Example_Tool/` is a working example of this.

### Presentation hints — laying the client's panel out

Four optional `ArgSpec` fields say *how* an argument should be presented. The
server never reads them: `validate()` and `run()` ignore every one, they only
travel through `GET /tools`. A tool declaring none renders exactly as it always
did, which is the right default for almost every tool.

They exist because past a certain size, "what is this argument" stops producing
a usable panel. `ASO` declares 130 CBCT landmarks, 32 teeth, 8 landmark types
and 2 jaws in one schema, and a generic client renders that as one column of
~180 check boxes with the CBCT and IOS options interleaved — while any given
run uses one half or the other. **Reach for these when your tool has that
shape** (several modes sharing a schema, or a catalog too long to scroll
through), not by default.

```python
"cbct_landmarks": ArgSpec(
    label="Landmarks",                            # what the user reads
    type="multichoice",
    required=False,
    choices=catalogs.CBCT_LANDMARK_CHOICES,      # 130 options
    description="CBCT only: the landmarks to register on...",
    section="Landmark Reference",                 # its own collapsible box
    visible_when={"modality": "CBCT"},            # hidden for an IOS run
    ui="tabs",                                    # one tab per group
    groups=catalogs.CBCT_LANDMARK_GROUPS,         # {"Cranial base": (...), ...}
),
```

| Field | Applies to | Meaning |
|---|---|---|
| `label` | any argument | The field label. Absent → the client prettifies the argument *name* (`output_suffix` → "Output suffix"), which is a guess: it renders an acronym as "Cbct landmarks" and cannot produce "Scan / Landmark Folder" from `input` |
| `section` | any argument | The collapsible box it belongs in. Absent → the client's default box ("Inputs"). Boxes appear in the order your schema first mentions them, so declaration order is the reading order |
| `visible_when` | any argument | `{other_argument: value}` — show only while every entry matches. A list/tuple of values means "any of these". The named arguments must be `"choice"` arguments of the same tool |
| `ui` | `"multichoice"` only | `"tabs"` (one scrollable tab per group), `"grid"` (one row per group, options as columns — for options whose position carries meaning, like a dental chart), `"inline"` (one horizontal row) |
| `groups` | `ui="tabs"`/`"grid"` | `{group name: (option, ...)}`. Every option must exist in `choices`; options no group mentions are rendered after the groups, not dropped |

`check_schema()` rejects all of these at **startup** — an unknown layout,
`groups` without a layout that uses them, a group naming an option that does
not exist, a `visible_when` on an argument you do not declare or expecting a
value outside its `choices`. That is deliberate for something purely cosmetic:
a wrong `visible_when` hides a field for good, and a client cannot tell that
from a field the tool never declared, so the failure is silent everywhere else.

Two things to keep in mind:

- **`visible_when` is presentation, not validation.** A hidden argument is
  simply not sent, so its declared default applies. Your tool's own
  cross-argument checks still have to hold — a direct API call sends whatever
  it likes.
- **Derive `groups` from the same table the logic uses**, never write it out a
  second time. `catalogs.TOOTH_GROUPS` is built from `TOOTH_IDS`, so a tooth
  added to the label table appears in its own arch on screen.
- **`label` is worth declaring even on a tool that needs none of the rest.**
  Every user-visible word describing your tool — the field label, the section
  title, the group names, the option names — belongs here, next to
  `description` and `choices`. A client that has to invent one can only guess
  from an identifier you chose for Python, not for a reader.

---

## 4. Writing `run()`

```python
def run(self, scan: str, threshold: float, iterations: int = 10) -> str:
    ...
```

- Signature parameters must match the argument names exactly (they're passed as
  `**kwargs`).
- Trust your inputs — validation already happened.
- **Never `except: pass`.** Let exceptions propagate: `main.py` catches them,
  logs the traceback server-side, and returns a generic **500** to the client.
  Swallowing an error silently returns a bogus 200 instead.
- Read `settings.DEVICE` (`"cuda"` / `"cpu"`) if your tool does GPU work,
  rather than hardcoding a device.
- **Never log file contents, argument values, or patient metadata.** This
  server processes confidential medical data; logs are limited to timestamp,
  endpoint, tool name, status, duration, and size.

### Keeping `run()` thin

If the tool has real logic, put it in `src/` and keep `my_tool.py` as a
declaration-plus-delegation shim. This is what `SurgMovPred` does, and it's
what makes the logic unit-testable without HTTP:

```python
from base import ArgSpec, Tool
from .src.my_tool_logic import MyToolLogic


class MyTool(Tool):
    name = "my_tool"
    arguments = {...}
    output_kind = "file"

    def run(self, model: str, input: str) -> str:
        return MyToolLogic.main(model, input)
```

### Shared helpers

Before writing your own zip/tabular handling, check `file_utils.py`:

| Helper | Does |
| --- | --- |
| `make_scratch_dir(prefix)` | Fresh writable directory under `TEMP_DIR` for this request; cleaned up for you |
| `extract_zip(zip_path, extract_dir=None, strip_single_root=False, max_total_bytes=None)` | Safely extracts an archive (zip slip / symlink / size guards), returns the directory |
| `make_zip(sources, destination)` | Bundles a list of paths, or one directory, into an archive — what backs `output_kind="files"` |
| `split_scan_extension(name)` | `'scan.nii.gz'` → `('scan', '.nii.gz')`, compound extensions preserved |
| `compressed_extension(ext)` | The compressed spelling ITK can write for a scan extension |
| `load_tabular_file(path)` | Loads one `.csv` / `.xlsx` / `.ods` into a DataFrame |
| `load_tabular_directory(path)` | Loads and concatenates every tabular file directly in a directory |

A helper more than one tool needs belongs here, not copied into each of them —
tools never import each other.

---

## 5. Server-side data: models and reference files

Some inputs shouldn't travel over the wire on every call — an AI model weighing
hundreds of MB, or a reference test dataset. `ArgSpec.server_selectable` lets
the client send just a **name** instead:

```
DATA_DIR/<tool_name>/models/       →  server_selectable="model"
DATA_DIR/<tool_name>/testfiles/    →  server_selectable="testfile"
```

`GET /tools/<tool_name>/data` (Bearer-protected) lists what's available, so the
client can render a dropdown instead of a file picker. `main.py` resolves the
name through `data_store` and `run()` still receives a **local path** — your
code doesn't change depending on which route the file took.

There are two flavours, and the difference is the *type*:

```python
# Scalar type → name-only. An upload for this argument is rejected with a 400.
"model": ArgSpec(type=str, required=True, server_selectable="model",
                 description="Name of a model hosted on the server (see GET /tools/<tool>/data)"),

# File type → the client may EITHER pick a server-side file by name OR upload its own.
"input": ArgSpec(type="zip_file", required=True, server_selectable="testfile",
                 description="Zip archive of CSV/XLSX/ODS files"),
```

Use `type=str` when the artefact must never leave the server (model weights);
use a file type when the client legitimately has its own data to send.

Server-side entries can be a single file **or** a whole folder, and either way
`run()` gets a path tagged with `.kind` — `"folder"` for a directory, the
matching file type otherwise. A `"folder"`-typed argument backed by a zipped
entry in the data store is extracted for you, so the tool sees a directory
whichever route the data took.

Test files (and only test files) are also **downloadable**:
`GET /tools/<tool_name>/testfiles/<name>` (Bearer-protected) streams the entry
to the client — a folder entry arrives zipped. This backs the Slicer client's
"Test file" button, which pulls a hosted test file down and fills the input
field with the local copy; the button grays itself when this endpoint would
404, i.e. when `DATA/<tool_name>/testfiles/` has nothing to offer. Models are
deliberately not downloadable: they are selected by name and used in place.
Nothing to declare per tool — drop a file under `testfiles/` and both the
dropdown and the button light up.

Two constraints worth internalizing:

- `DATA_DIR` is mounted **read-only** (`./DATA:/data:ro`). Never write there.
- `DATA/` is gitignored — it holds confidential data and must never be
  committed. A clone without it still runs, with the relevant tests skipped.

Public model bundles and reference test files are fetched by the scripts in
[`scripts/`](scripts/README.md), driven by `scripts/data-manifest.yml`. If your
tool has published weights, add an entry there rather than documenting a manual
download — `./scripts/setup-models.sh --tool <your_tool>` then populates
`DATA/<your_tool>/models/` for everyone.

Path traversal is already handled by `data_store.py` (bare-name check plus a
`realpath` containment check against symlinks); you don't need to re-validate.

---

## 6. Outputs (`output_kind`)

| `output_kind` | `run()` returns | Client receives |
| --- | --- | --- |
| `"text"` (default) | any JSON-serializable value | `{"result": <value>}` |
| `"file"` / `"segmentation"` | a **path** to the output file | the file itself, as a `FileResponse` |
| `"files"` | a **list of paths**, or one **directory** path | a single zip of them all |

For file outputs, `main.py` derives `Content-Type` from the extension via
`mimetypes.guess_type()`, and `Content-Disposition`'s filename from
`os.path.basename(result)` — so give your output a real, meaningful name and
the correct extension. (This matters more than it looks: `.xlsx`, `.docx`, and
`.ods` are zip containers, so a client sniffing magic bytes can't tell them
apart from an actual `.zip`. Correct headers are what makes that unambiguous.)

### Returning several files

Declare `output_kind = "files"` and return the paths you produced. `main.py`
zips them and streams the archive — **no zip code in your tool**:

```python
output_kind = "files"

def run(self, scan: str) -> list:
    output_dir = file_utils.make_scratch_dir("my_tool_")
    ...
    return [predictions_csv, report_pdf, overlay_nii]   # → my_tool_output.zip
```

Returning the directory itself works too, and is the better choice when the
file count isn't known up front — its *contents* land at the archive root, and
the archive is named after the folder:

```python
    return output_dir        # results/ → results.zip, holding a.csv, nested/b.csv, …
```

Two rules:

- **Base names must be unique.** `a/result.csv` and `b/result.csv` would collide
  inside the archive and silently hand the client fewer files than you produced,
  so it's rejected with a 500. Return the parent directory instead, which keeps
  the tree.
- If you already have a finished archive to send back, use `output_kind="file"`
  and return its path — `"files"` would wrap it in a second zip.

### Where to write the output

Cleanup rules, so nothing leaks on a server handling medical data:

- **Uploaded inputs** are deleted right after `run()` returns, in a `finally`
  block — including when `run()` raises.
- **The per-request work dir** (created when there is at least one upload) is
  removed by a background task *after* the response has finished streaming.
- **Every `file_utils.make_scratch_dir()` folder** is registered per request and
  removed the same way — including when `run()` raises halfway through, which
  is exactly when half-written patient data would otherwise be left behind.
- **`DATA_DIR` files are never deleted** — only temp copies materialized by a
  future non-local `DataStore` backend are.

So: **write your output inside the request's own work dir**, which is the
directory containing an uploaded input:

```python
output_dir = os.path.join(os.path.dirname(uploaded_input_path), "output")
os.makedirs(output_dir, exist_ok=True)
```

The whole tree is cleaned up for you once the response is sent.

⚠️ **When there is no upload at all** — every file argument satisfied
server-side — no work dir exists and `os.path.dirname(...)` would point into
read-only `DATA_DIR`. Use `file_utils.make_scratch_dir()` instead:

```python
output_dir = file_utils.make_scratch_dir("my_tool_")
```

`main.py` registers every folder handed out by `make_scratch_dir()` for the
request being served and removes them all once it is over — on success after
the response has streamed, on failure immediately. Calling
`tempfile.mkdtemp(dir=settings.TEMP_DIR)` by hand bypasses that registry, and
is what leaks when a tool crashes.

---

## 7. Dependencies

There is no per-tool requirements mechanism — two files, whatever the number
of tools:

| File | Holds | Installed |
| --- | --- | --- |
| `requirements.txt` | everything the server and its tools import — `fastapi`, `pandas`, `numpy`, `SimpleITK`, and the heavy DL stack (`vtk`, `torch`, `nnunetv2`) | every container start, by the `inference` service's command |
| `requirements-dev.txt` | `pytest`, `httpx` | only where the test suite runs |

**Never pin or force-reinstall `torch`.** The base image
(`ghcr.io/jules-gp/lab-ai:2026.07`) ships `2.5.1+cu124`; pip sees the
unpinned `torch` entry satisfied and leaves it alone, whereas a pinned or
`--upgrade` install would shadow it with a CPU-only wheel and every GPU tool
would silently drop to CPU. Pin versions for anything where a silent upgrade
could change numerical results.

(A dependency that cannot install from PyPI at all — e.g. `pytorch3d`, which
needs a source build against the image's torch + CUDA — must be baked into
the image, not added to `requirements.txt`.)

### Import heavy dependencies lazily

A heavy or optional stack must be imported **inside the function that uses
it**, not at module level:

```python
def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ToolUnavailableError(f"This tool needs torch: {_INSTALL_HINT}") from exc
    return torch
```

This is not a style preference. A module-level `import torch` in a tool whose
dependency is missing raises at **discovery** time; `registry.py` then skips
the whole tool (§2). Deferring the import means the tool still loads, still
publishes its schema through `GET /tools`, and only an actual run fails — with
a message naming what to install. `tools/AMASSS/src/nnunet_runner.py` is the
reference implementation.

`ToolUnavailableError` (not a bare `RuntimeError`) is what carries that message
out to the caller as a **501** instead of a blank 500 — see §9.

**Call every one of these importers once, up front**, before the work starts:

```python
def check_dependencies() -> None:
    _import_torch()
    _import_vtk()
```

A missing dependency belongs to the server, not to one input. Discovering it
inside a per-item loop means a 200-scan batch fails 200 times identically —
each time only *after* that scan's preprocessing — and the run then ends on a
summary that buries the one line saying what to install. `tools/ALI/`'s two
engines do this; it is the difference between an actionable failure in a second
and an opaque one in an hour.

The same applies to a `src/` module other tools import: keep the heavy imports
inside functions so importing the module never drags in the stack.

---

## 8. Tests

Two layers, matching the two layers of the tool.

### Tool logic, in isolation

Add `tools/<name>/test/test_<name>.py` importing directly from your `src/`
modules (`tools.<name>.src...`). No HTTP, no server, synthetic data only —
this is where you cover the actual algorithm. `registry.py` only looks for
`<name>/<name>.py`, so a nested `test/` folder is invisible to discovery and
picked up by pytest alone. See `tools/SurgMovPred/test/` for a worked
example.

### HTTP layer

`tests/test_main.py` uses FastAPI's `TestClient` against the real app. Add a
case here if your tool exercises something new at the boundary — an unusual
argument shape, a new file type, a rejection path.

### Real data

`tests/test_data_integration.py` runs every tool whose required arguments are
all `server_selectable` against whatever real files a maintainer has placed
under `DATA/<tool_name>/{models,testfiles}/`. Missing files mean **skip**, not
fail, so a machine without the confidential dataset can still run the suite.
Making your tool's required arguments `server_selectable` is what opts it into
this coverage for free.

### Running

```bash
docker compose run --rm test          # recommended: same image as production
# or locally
./venv/bin/pytest                     # everything
./venv/bin/pytest tools/my_tool/test/ # just your tool
```

A pre-push hook runs the full suite and blocks the push on failure. Enable it
once per clone:

```bash
git config core.hooksPath .githooks
git push --no-verify                  # bypass for a single push
```

---

## 9. Verifying by hand

```bash
# Is it registered, with the schema you expect?
curl -k https://localhost:8000/tools

# Server-side models/testfiles for it
curl -k https://localhost:8000/tools/my_tool/data \
  -H "Authorization: Bearer $API_TOKEN"

# Run it
curl -k -X POST https://localhost:8000/run/my_tool \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "threshold=0.5" \
  -F "scan=@/path/to/scan.nii.gz" \
  -o result.out
```

`-k` skips certificate verification and is only acceptable against the
self-signed dev certificate — never against production.

A `"folder"` argument is sent as a zip, exactly like any other upload — the
server unpacks it:

```bash
zip -r cohort.zip cohort/
curl -k -X POST https://localhost:8000/run/my_tool \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "input=@cohort.zip" \
  -o output.zip
```

Expected failure modes, all of which should already work without any code of
yours: no token → **401**, unknown tool → **404**, missing/unknown/mistyped
argument or an option outside `choices` → **422**, wrong file extension, an
upload on a scalar argument, or an unsafe/oversized archive → **400**, file over
`MAX_UPLOAD_MB` → **413**.

A rule spanning *several* arguments can't be expressed in the schema. Raise
`ToolArgumentError` from `run()` and it gets the same **422** treatment as a
schema violation, with your message.

When it is the *server* that can't serve a perfectly valid request — almost
always a dependency the deployment image doesn't carry — raise
`ToolUnavailableError` instead, and the client gets a **501** with your
message. That is what the lazy importers of §7 do. Use it only for
"this deployment cannot do this": nothing the caller changes may fix it, and
the message must name a package, never a path or anything about the data.

**Any other exception becomes a blank 500**, on purpose — a crash can name
server-side internals, so its detail is never shown to the client.

---

## 10. What the client gets for free

`GET /tools` returns, per argument, `type` / `types` / `required` /
`description` / `server_selectable` / `choices` / `initial` / `extensions`,
plus the four presentation hints (`section` / `visible_when` / `ui` /
`groups`). The Slicer client builds its panel from that: a file picker filtered
on the declared extensions, a dropdown of server-side names for
`server_selectable` arguments, a numeric field for `int`/`float`, a combo box
or a group of check boxes pre-ticked from `choices`, your `description` as the
label or tooltip — and, where you asked for them, collapsible sections, tabs, a
chart grid, and fields that appear only in the modes they apply to.

`types` is the full list of accepted types; `type` repeats the first one so
clients written before multi-type arguments keep working. An argument whose
`types` contain `"folder"` is the client's cue to also offer a *directory*
picker and zip the selection before uploading.

Which means: **the quality of your `arguments` dict is the quality of the
generated UI.** Write real descriptions.

---

## Checklist

- [ ] `tools/<name>/__init__.py` exists
- [ ] `tools/<name>/<name>.py` — file name matches folder name
- [ ] Class subclasses `Tool`, `name` set and unique
- [ ] Core imports absolute (`from base import ...`), own imports relative (`from .src...`)
- [ ] Every argument has a real `description`
- [ ] File arguments use a `FILE_TYPES` key (new type added to `base.py` if needed)
- [ ] Multi-type arguments branch on `.kind`, not on `os.path.isdir()` or the extension
- [ ] A fixed set of options is a `"choice"`/`"multichoice"`, not a free-form `str`
- [ ] Choice defaults declared once, in `choices` — not repeated in `run()`'s signature
- [ ] `output_kind` correct; file outputs return a path with a meaningful name and extension
- [ ] Outputs written inside the request work dir, or in a `make_scratch_dir()`
- [ ] No secret, no patient data, no argument value in any log
- [ ] New dependencies added to `requirements.txt`
- [ ] Unit tests under `tools/<name>/test/`
- [ ] `docker compose run --rm test` passes
- [ ] Nothing edited in `main.py` / `registry.py` (a `FILE_TYPES` entry in `base.py` is the only allowed core change)

---

## Full example

A hypothetical registration tool: two uploaded volumes, a server-side model, a
numeric parameter, a file output.

```python
# server/tools/areg_cbct/areg_cbct.py
"""CBCT-to-CBCT automated registration."""

import os

from base import ArgSpec, Tool

from .src.areg_cbct_logic import register


class AregCbctTool(Tool):
    name = "areg_cbct"
    arguments = {
        "fixed_image": ArgSpec(
            type="nifti_file",
            required=True,
            description="Reference CBCT the moving image is registered onto (.nii/.nii.gz)",
        ),
        "moving_image": ArgSpec(
            type="nifti_file",
            required=True,
            description="CBCT to register (.nii/.nii.gz)",
        ),
        "model": ArgSpec(
            type=str,
            required=True,
            server_selectable="model",
            description="Name of a registration model hosted on the server (see GET /tools/areg_cbct/data)",
        ),
        "iterations": ArgSpec(
            type=int,
            required=False,
            description="Optimizer iterations (default 200)",
        ),
    }
    output_kind = "file"

    def run(self, fixed_image: str, moving_image: str, model: str, iterations: int = 200) -> str:
        # Uploaded inputs share the request's work dir, which main.py cleans up
        # after the response is streamed — write the output inside it.
        output_dir = os.path.join(os.path.dirname(fixed_image), "output")
        os.makedirs(output_dir, exist_ok=True)
        return register(
            fixed_image=fixed_image,
            moving_image=moving_image,
            model_path=model,
            iterations=iterations,
            output_path=os.path.join(output_dir, "registered.nii.gz"),
        )
```

Called as:

```bash
curl -k -X POST https://localhost:8000/run/areg_cbct \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "fixed_image=@/data/patient_T1.nii.gz" \
  -F "moving_image=@/data/patient_T2.nii.gz" \
  -F "model=areg_cbct_v3.pth" \
  -F "iterations=300" \
  -o registered.nii.gz
```

No route was added. No list was updated. That is the point.
