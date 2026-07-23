# Inference server (tool-registry architecture)

Synchronous FastAPI server exposing a generic `/run/{tool_name}` endpoint.
Tools are self-contained classes auto-discovered from `tools/` at startup —
adding a tool never requires touching the server core.

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

A tool that also expects a file just adds `-F "file=@/path/to/volume.nii.gz"`
to the same call; the server streams it to a temp dir and passes its path to
the tool as the `file` argument.

## How to add a new tool

1. Create a folder `tools/<your_tool>/` with an `__init__.py` (can be empty)
   and a `tools/<your_tool>/<your_tool>.py` file — **the file name must match
   the folder name**, that's the one file the registry imports. Any other
   file in the folder (helpers, data, ...) is ignored by discovery, though
   your main file is free to import from them.
2. Subclass `Tool` (from `base.py`), set a unique `name`, declare `arguments`
   as a dict of `ArgSpec` (type, required, description), implement `run(**kwargs)`.
3. That's it — `registry.py` auto-discovers it at startup, `/tools` lists it,
   and `/run/<your_tool>` becomes available immediately. No route to add, no
   registration list to update. See `tools/test_tool/` for a minimal example.

A tool folder missing its `<name>.py` file, duplicate tool names, or a tool
missing its `name`, all fail loudly at server
startup rather than silently overwriting each other.
