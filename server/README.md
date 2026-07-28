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
`surgMovPred`'s `model` (`ArgSpec(type=str, server_selectable="model")`) is
always picked by name, never uploaded (an upload for it is rejected with a 400):

```bash
# List the models/testfiles hosted server-side for a tool (Bearer-protected)
curl -k https://localhost:8000/tools/surgMovPred/data \
  -H "Authorization: Bearer change-me-to-a-long-random-secret"
# -> {"models": ["stacking_v2.zip"], "testfiles": ["demo_measurements.zip"]}

# Run it: the model is a name, the input is a genuine upload
curl -k -X POST https://localhost:8000/run/surgMovPred \
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
}
```

```python
"input": ArgSpec(type="zip_file", required=True, description="..."),
```

This is what both the server (extension check on upload) and the client
(`GET /tools`, which reports `"type": "zip_file"` per argument) use to know
exactly what's expected — no shared global list to keep in sync as tools with
different file needs are added. To accept a new kind of file, add an entry to
`FILE_TYPES` in `base.py`. Only arguments left as generic `"file"` fall back
to `config.ALLOWED_EXTENSIONS`.

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

If your tool needs to unzip an upload and/or load CSV/XLSX/ODS files, reuse
the shared helpers in `file_utils.py` (`extract_zip`, `load_tabular_file`,
`load_tabular_directory`) instead of reimplementing them.

A tool folder missing its `<name>.py` file, duplicate tool names, or a tool
missing its `name`, all fail loudly at server
startup rather than silently overwriting each other.

## Testing

```bash
./venv/bin/pip install -r requirements-dev.txt   # pytest, httpx
./venv/bin/pytest                                 # everything
./venv/bin/pytest tests/                          # HTTP layer only (main.py, routing, auth)
./venv/bin/pytest tools/surgMovPred/test/        # one tool's own logic, in isolation
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
See `tools/surgMovPred/test/` for an example covering column-name cleaning,
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
`DATA/surgMovPred/testfiles/my_real_input.zip` (and
`DATA/surgMovPred/models/my_real_model.zip` if you want the real model
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
