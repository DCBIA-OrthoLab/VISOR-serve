# Architecture

A 3D Slicer extension offloads heavy computation to this server. The client
sends inputs, the server runs the selected tool, the result comes back in the
same response.

## The shape

```
Slicer client  --HTTPS-->  FastAPI server  --subprocess-->  tool virtualenv
                                |                                |
                           registry, dispatch              its own pinned deps
                           transfer, security              its own Python
```

## Three rules that explain most decisions

**A tool is a folder with its own virtualenv.** The server never imports one.
It reads the tool's `.schema.json`, validates the request against it, and runs
`<tool>/.venv/bin/python runner.py --job job.json` as a subprocess. Two tools
whose dependencies conflict is therefore a non-problem: ALI_CBCT runs torch
2.8 while ALI_IOS runs 2.11, in the same deployment.

**Discovery is declared, not inferred.** A directory is a tool when its
`pyproject.toml` carries `[tool.sadt] tool = true`, scanned two levels deep so a
grouping folder can hold several. The API name comes from `[tool.sadt] name`.
No central list to keep in sync.

**A tool calls another through the supervisor.** `sup.run("ASO", ...)` starts a
sibling in its own virtualenv, as a subprocess of the caller, so a chain never
queues for a slot its parent holds. The names are published in the schema's
`calls`, and the server refuses to start when one of them is not served.

## Endpoints

| | |
|---|---|
| `GET /health` | liveness, no auth |
| `GET /tools` | every tool and its arguments, no auth |
| `POST /run/{tool}` | run it, Bearer token, blocking |
| `POST /uploads` + `PUT .../parts/{n}` | chunked resumable upload |
| `GET /results/{id}` | range-served result, for large outputs |

## What the server guarantees

Arguments are validated before a tool starts. Every temp file is deleted,
including on error. Uploaded archives are checked for zip slip, symlinks and
expansion. A run is capped by a per-tool timeout that kills the whole process
group, so a killed job leaves nothing holding the GPU. Logs carry timestamp,
endpoint, tool, status, duration and size, never file contents or arguments.

## Deployment

One container, N virtualenvs, hardlinked so shared dependencies are stored once.
`DATA/` is mounted read-only and holds each tool's models and test files.
TLS is mandatory in front; the container speaks plain HTTP on localhost.
