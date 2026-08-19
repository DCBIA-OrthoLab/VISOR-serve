# `_dispatch_probe` — the dispatch path, without a tool

A test fixture, not a tool. It exists so the whole subprocess loop — job
directory, `job.json`, another interpreter, `runner.py`, `result.json`,
cleanup — can be proven end to end without a model, a GPU, or a single heavy
dependency in the way.

The leading underscore is load-bearing: `registry/` skips folders starting
with `_`, so this never appears in `GET /tools` and never reaches a client.
The tests drive it directly through `dispatch()` and through a `Tool` subclass
they define themselves.

## Its virtualenv

There isn't one here. `tests/conftest.py` copies this folder into a temporary
directory, builds the venv *there*, and points `TOOLS_DIR` at it — which is
the same knob the deployment image uses to put the tools at `/tools/<name>/`,
so the layout under test is the real one.

Not built in place, for two reasons with teeth: the docker `test` service
mounts `server/` from the host, so a venv created inside the container would
leave root-owned files on the host pointing at an interpreter that does not
exist there; and a venv is a build artifact that a clone on another machine
should not inherit.

It is created with `uv venv` when uv is installed and with the standard
library's `venv` (no pip) otherwise — the probe imports nothing, so there is
nothing to install and no network to reach.

To build one by hand, the way the deployment image will build every tool's:

    cd server/tools/_dispatch_probe && uv venv && uv sync

## What it proves

`run()` returns the interpreter that executed it, so a test can assert it was
**not** the server's; the job id, job dir and `SADT_API` it was given; whether
the server's `API_TOKEN` leaked into its environment (it must not); and the
file it wrote into `output/`. `fail=True` makes it raise, which is how the
failure half of the contract is tested: a non-zero exit, a `result.json`
carrying `{"error": {"type", "message"}}` — the class NAME is what `main.py`
maps to a status — and the stderr tail attached to the error.

It also has a packaged twin, which is what `execution/parity.py` is proven
against: agreement is reported as agreement, a twin returning a different total
is caught, and a twin writing a *different file with the same answer* is caught
too — that last one is the failure a smoke test misses.

The **supervisor** is tested separately (`tests/test_supervisor.py`), against
one-file tools built on the fly into a temporary `TOOLS_DIR`, because what is
under test there is precisely that the callee gets its own interpreter.
