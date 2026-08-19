# Internals

How the server is built, and why each mechanism is the shape it is. Read
`ARCHITECTURE.md` first for the one-page version.

## Module map

```
server/
  main.py            HTTP layer: routes, uploads, results, cleanup
  base.py            Tool, ArgSpec, the typed argument schema and its errors
  config.py          every setting, read from the environment, nowhere else
  file_utils.py      scratch dirs, archives, output discovery
  registry/
    __init__.py      discovery, the startup checks, TOOLS
    schema_tool.py   a Tool built from .schema.json, never imported
    schema_hash.py   source_hash, the reference implementation
    deployment.py    deployment.toml: per-tool server-side config
    conventions.py   what an argument's name implies
  execution/
    dispatch.py      run a tool in its own venv, as a subprocess
    runner.py        the other half, executed BY the tool's venv, stdlib only
    parity.py        run a tool both ways and compare what a caller receives
  wire/
    transfer.py      chunked uploads, range-served results
    security.py      bearer token, constant-time compare
```

## Discovery

`_tool_folders` walks `TOOLS_DIR` to depth 2 and returns `(api name, path)` for
every directory whose `pyproject.toml` declares `[tool.sadt] tool = true`. It
does not descend into a folder that is already a tool, and a leading underscore
or dot excludes one.

Depth 2 exists so a grouping folder can hold several tools: `tools/ALI/` is not
a tool, `tools/ALI/ALI_CBCT/` is. A shared package like `tools/ALI/common/` has
a `pyproject.toml` but no `[tool.sadt]`, so it is importable and never served.

The name comes from `[tool.sadt] name`, falling back to the directory. The
directory must still be NAMED after the tool, because the interpreter is looked
up by name; what declaring it buys is that the tool's DEPTH may change without
the name a client sends moving with it.

### Failures that are fatal, and why

Most discovery failures cost one tool: it is skipped, recorded in
`FAILED_TOOLS`, and the rest are served. Three are fatal instead.

A **stale `source_hash`** means the cached schema no longer describes the source
beside it, so the server would validate requests against a signature that has
changed under it. A **`sup.run()` naming an unserved tool** means a chain that
starts, accepts a request, runs for an hour and fails on a name that was already
wrong at deploy time. **Every packaged tool failing** is not a tool fault at all
but a deployment one, and starting with only the in-process fixtures reads as a
small deployment rather than a broken one.

## Running a tool

`dispatch.dispatch` writes `job.json`, picks the tool's interpreter, and runs
`runner.py` inside it. `runner.py` ships with the SERVER and is injected by
path, never installed into a tool venv, so the two are always the same version.
It is standard library only and must run on Python 3.9 through 3.13, because
each tool pins its own.

The child gets its own session (`start_new_session`), so a timeout can send
SIGTERM to the whole process GROUP. That distinction is the point: nnUNet,
torch's DataLoader and shapeaxi all fork workers, and killing the one PID leaves
them holding a CUDA context.

`API_TOKEN` is stripped from the child environment. The rest is inherited.

## The supervisor

A tool whose `run()` declares a keyword-only, unannotated `sup` receives one.
Unannotated is the marker rather than an accident: every other parameter must be
annotated, so nothing else has that shape, and `describe.py` uses the same rule
to keep it out of the published schema.

Five members, frozen: `run(tool, **params)`, `out`, `tmp`, `progress(frac, msg)`,
`log(msg)`.

`sup.run()` re-enters `runner.py` with the sibling's interpreter, so nesting is
one recursion rather than a feature. It does **not** go back through the server:
a nested call is a subprocess of its parent and never re-enters admission, so it
cannot wait for a slot the parent already holds.

### Nesting

A nested call carries three things in its environment: the depth, the chain of
tools already running above it, and the job's deadline.

The chain is what refuses a cycle, by NAME and at the first call, listing the
chain in the error. The depth cap (10) is only a backstop for a chain that grows
without repeating. A tool appearing twice on different BRANCHES is not a cycle:
AREG_IOSCBCT calls ALI_CBCT directly and again through ASO.

A nested call deliberately does **not** get its own session. Without one it
inherits its parent's process group, so the SIGTERM the server sends to the root
reaches every level. Giving each level its own session would detach the
grandchildren from the group the server kills.

### The deadline

The server turns a tool's `timeout_seconds` into an absolute instant on the
monotonic clock and passes it down; every level gives its child only what is
left. A duration would restart at each hop and a deep chain would silently get a
multiple of the budget granted.

For an orchestrating tool the timeout is not about its own compute:
AREG_IOSCBCT computes for under a second and spends the rest inside four
children. Rule of thumb, `sum of the timeouts of every tool in calls * 1.2`.

## Transfer

A file in one request rides one TCP connection and is bound by its congestion
window long before its bandwidth. `POST /uploads` opens a session, parts are
`PUT` at computed offsets with `os.pwrite` into a pre-truncated sparse blob, so
concurrent parts write disjoint ranges and there is no reassembly pass. A part
carries `X-Part-SHA256`, verified before anything is written; since the parts
tile the file, that verifies the whole upload without a second pass.

State lives on disk, not in a module global: two parts of one upload may be
served by different workers, and a session must survive a reload. Ids are
`secrets.token_urlsafe(24)`, matched against `[A-Za-z0-9_-]{16,64}` before any
path is built from them.

Cleanup is a timer, not an opportunistic sweep: an abandoned transfer sits
longest exactly when no new request arrives. `TRANSFER_TTL_SECONDS` is an IDLE
timeout, so a transfer still in flight is never at risk however long it takes.

## Security

There is no database, no ORM and no query anywhere, so SQL injection has no
surface. What exists is a file-handling API, and the guards match:

- the bearer token is compared in constant time, and never echoed;
- ids are pattern-matched before a path is built from them;
- an uploaded filename is sanitized to `[A-Za-z0-9_.-]` and capped, keeping the
  patient's name (which a batch needs to stay distinguishable) without writing a
  client string to disk unchecked;
- archives are refused on zip slip, absolute members, symlink members, and
  expansion past `MAX_EXTRACTED_MB`;
- every temp file is removed on every path, including error and client
  disconnect.

`tests/test_security.py` asserts each of these as a refusal rather than as an
absence of crash.

## Testing

`server/pytest.ini` restricts collection to `server/tests`. A packaged tool's
tests belong to that tool and must run in ITS interpreter against its own pins;
collecting them here would import tools the API deliberately cannot import.

`tests/golden/tools_response.json` is `GET /tools` captured before the migration
began and asserted per tool, per argument, in order. The Slicer client builds
its entire UI from that response. If that test fails, the client breaks, and the
fixture is not what gets updated.
