# Adding a tool to the inference server

This guide is the long-form companion to the short "How to add a new tool"
section in [`server/README.md`](README.md). It covers the whole contract: the
folder layout, the argument schema, server-side data, outputs, dependencies,
and tests.

**The one thing to remember:** a tool is a self-contained folder. You never
edit `main.py`, `registry.py`, or `base.py` to add one. No route, no
registration list.

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
    └── test_my_tool_logic.py
```

Three rules, all enforced loudly at startup (the server refuses to boot rather
than silently misbehaving):

| Rule | What happens if you break it |
| --- | --- |
| `tools/X/` must contain `X.py` | `RuntimeError: Tool folder 'tools/X/' is missing its 'tools/X/X.py' file.` |
| Every `Tool` subclass must set a non-empty `name` | `RuntimeError: Tool class 'MyTool' has no 'name' set.` |
| Tool `name`s must be unique across the whole registry | `RuntimeError: Duplicate tool name detected: 'my_tool'` |

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
    type=...,                  # str | int | float | bool, or a file-type key
    required=True,             # default True
    description="",            # shown to clients via GET /tools — write a real one
    server_selectable=None,    # "model" | "testfile" | None — see §5
)
```

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
`validate()` simply omits absent optional arguments from the call.

### File arguments

A file argument declares **its own** accepted extensions via a key of
`base.FILE_TYPES`, rather than relying on a server-wide whitelist:

```python
FILE_TYPES = {
    "file":       None,                 # generic → falls back to config.ALLOWED_EXTENSIONS
    "zip_file":   (".zip",),
    "csv_file":   (".csv",),
    "xlsx_file":  (".xlsx",),
    "ods_file":   (".ods",),
    "nifti_file": (".nii", ".nii.gz"),
}
```

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
declaration-plus-delegation shim. This is what `surg_mov_pred` does, and it's
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
| `extract_zip(zip_path, extract_dir=None)` | Extracts an archive (default: an `extracted/` folder next to it), returns the directory |
| `load_tabular_file(path)` | Loads one `.csv` / `.xlsx` / `.ods` into a DataFrame |
| `load_tabular_directory(path)` | Loads and concatenates every tabular file directly in a directory |

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

Two constraints worth internalizing:

- `DATA_DIR` is mounted **read-only** (`./DATA:/data:ro`). Never write there.
- `DATA/` is gitignored — it holds confidential data and must never be
  committed. A clone without it still runs, with the relevant tests skipped.

Path traversal is already handled by `data_store.py` (bare-name check plus a
`realpath` containment check against symlinks); you don't need to re-validate.

---

## 6. Outputs (`output_kind`)

| `output_kind` | `run()` returns | Client receives |
| --- | --- | --- |
| `"text"` (default) | any JSON-serializable value | `{"result": <value>}` |
| `"file"` / `"segmentation"` | a **path** to the output file | the file itself, as a `FileResponse` |

For file outputs, `main.py` derives `Content-Type` from the extension via
`mimetypes.guess_type()`, and `Content-Disposition`'s filename from
`os.path.basename(result)` — so give your output a real, meaningful name and
the correct extension. (This matters more than it looks: `.xlsx`, `.docx`, and
`.ods` are zip containers, so a client sniffing magic bytes can't tell them
apart from an actual `.zip`. Correct headers are what makes that unambiguous.)

### Where to write the output

Cleanup rules, so nothing leaks on a server handling medical data:

- **Uploaded inputs** are deleted right after `run()` returns, in a `finally`
  block — including when `run()` raises.
- **The per-request work dir** (created when there is at least one upload) is
  removed by a background task *after* the response has finished streaming.
- **`DATA_DIR` files are never deleted** — only temp copies materialized by a
  future non-local `DataStore` backend are.

So: **write your output inside the request's own work dir**, which is the
directory containing an uploaded input:

```python
output_dir = os.path.join(os.path.dirname(uploaded_input_path), "output")
os.makedirs(output_dir, exist_ok=True)
```

The whole tree is cleaned up for you once the response is sent.

⚠️ **Caveat when there is no upload at all.** If every file argument was
satisfied server-side, no work dir exists and `os.path.dirname(...)` points
into read-only `DATA_DIR`. In that case create your own directory under
`settings.TEMP_DIR` (`tempfile.mkdtemp(dir=settings.TEMP_DIR)`) — and note that
the automatic background cleanup won't cover it, so a tool in that situation
should manage its own retention.

---

## 7. Dependencies

Extra Python packages go in `server/requirements.txt`. Both the `inference`
and `test` services install it at container start, so a rebuild isn't needed —
but pin versions for anything where a silent upgrade could change numerical
results.

The base image (`ghcr.io/jules-gp/lab-ai:2026.07`) already carries the heavy
CUDA/ML stack; only add what's genuinely missing.

---

## 8. Tests

Two layers, matching the two layers of the tool.

### Tool logic, in isolation

Add `tools/<name>/test/test_<name>_logic.py` importing directly from
`tools.<name>.src.<name>_logic`. No HTTP, no server, synthetic data only —
this is where you cover the actual algorithm. `registry.py` only looks for
`<name>/<name>.py`, so a nested `test/` folder is invisible to discovery and
picked up by pytest alone. See `tools/surg_mov_pred/test/` for a worked
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

Expected failure modes, all of which should already work without any code of
yours: no token → **401**, unknown tool → **404**, missing/unknown/mistyped
argument → **422**, wrong file extension or an upload on a scalar argument →
**400**, file over `MAX_UPLOAD_MB` → **413**.

---

## 10. What the client gets for free

`GET /tools` returns, per argument, `type` / `required` / `description` /
`server_selectable`. The Slicer client builds its panel from that: a file
picker filtered on the declared extensions, a dropdown of server-side names for
`server_selectable` arguments, a numeric field for `int`/`float`, and your
`description` as the label or tooltip.

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
- [ ] `output_kind` correct; file outputs return a path with a meaningful name and extension
- [ ] Outputs written inside the request work dir
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
