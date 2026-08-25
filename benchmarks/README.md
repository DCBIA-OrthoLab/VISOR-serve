# Benchmark harness

This directory holds everything needed to reproduce the runtime, network,
supervisor, concurrency and parity numbers reported in the accompanying paper.
It is written to be re-run by someone who did not write it.

Nothing outside this directory is modified by anything in it.

---

## 1. What is measured

| Campaign | Question | Output |
|---|---|---|
| **B1** | How long does one tool take through each execution path? | `results/summary/b1-*.md` |
| **B2** | Where does the time in a remote call actually go? | `results/summary/b2-*.md` |
| **B3** | What does the supervisor cost, and what does isolation cost? | `results/summary/b3-*.md` |
| **B4** | What happens with N clients at once? | `results/summary/b4-*.md` |
| **B5** | Do the two paths produce the same bytes? | `results/summary/b5-*.md` |

Three execution paths appear throughout:

- **`local`** -- the tool in its own interpreter, with no HTTP anywhere. This is
  the server's own dispatch mechanism, invoked by hand.
  **Read [`NOTES-local-path.md`](NOTES-local-path.md)**: it documents exactly
  what is reproduced, and the four places where the reproduction is not exact.
- **`loopback`** -- through the HTTP API on the same machine. Protocol and
  packing cost, no wire.
- **`lan`** -- through the same API from a different machine. Adds the wire.
  Requires `server.lan_base_url` in `config.yaml`.

The HTTP client is not a benchmark-only client. It speaks the protocol the real
Slicer client speaks -- 8 MB parts, four concurrent transfers, `X-Part-SHA256`
over each plaintext part, `Content-Encoding: gzip` on inputs that are not
already compressed, `X-Result-Delivery: reference`, ranged GETs, and a `DELETE`
when the bytes are in. Those constants are in `config.yaml` under `transfer:`
and should not be changed to make a number look better.

## 2. Prerequisites

| Need | For | Absent? |
|---|---|---|
| Python 3.11+ with `requests` and `PyYAML` | everything | nothing runs |
| A running server (`GET /health` -> `{"status":"ok"}`) | `loopback`, `lan`, B2, B4, B5 | those campaigns refuse to start, with a reason |
| `API_TOKEN` in the environment or in the repository's `.env` | any authenticated call | refused with a message naming both places |
| The deployment container running, or a `sadt-tools` checkout with built virtualenvs | `local`, B3, B5 | refused with a reason |
| An NVIDIA card and `nvidia-smi` | B4's VRAM column, and every GPU tool | the column reads "unavailable" with the reason; GPU tools fail and are recorded as failures |
| Docker | `local` in `container` mode only | switch `local.mode` to `host` |

**Inspecting the harness needs none of these.** The package imports and the unit
tests pass on a machine with no GPU, no server and no Docker daemon; the tests
that need one of those skip with a reason naming what is missing. See section 7.

### Set it up

```bash
cd <server repo>/benchmarks
python -m venv .venv           # or: uv venv .venv
.venv/bin/pip install -r requirements.txt
```

### The token

Never in this repository. The harness reads `$API_TOKEN` first, then
`API_TOKEN=` in the server repository's `.env` -- which `.gitignore` excludes,
so it cannot be committed. There is no default and no fallback:

```bash
export API_TOKEN=...            # or put it in ../.env
```

### Starting the server

From the repository root (the deployment image, one container, N virtualenvs):

```bash
docker compose --profile venvs up -d inference-venvs
curl -s http://localhost:8000/health          # {"status":"ok"}
curl -s http://localhost:8000/tools | head    # no auth needed for either
```

`config.yaml`'s `local.container` must name that container
(`docker ps --format '{{.Names}}'`).

## 3. Running a campaign

Always start here:

```bash
cd <server repo>
benchmarks/.venv/bin/python -m benchmarks.run --campaign b1 --dry-run
```

`--dry-run` parses and validates the config, builds the whole plan, checks the
free disk against the plan's projected output, and prints all of it. It starts
no process, opens no socket and writes no file. Run it after every config edit.

Then, for real:

```bash
python -m benchmarks.run --campaign b1 --reps 6
python -m benchmarks.run --campaign b2
python -m benchmarks.run --campaign b3
python -m benchmarks.run --campaign b4
python -m benchmarks.run --campaign b5
```

Useful flags:

| Flag | Effect |
|---|---|
| `--tools A,B` | run only these tools from the campaign's list |
| `--paths local,loopback` | B1 only: run only these execution paths |
| `--reps N` | override the repetition count |
| `--config PATH` (or `$BENCHMARKS_CONFIG`) | point at your own config file |
| `--keep-artifacts` | keep job directories and downloads (fills a disk fast) |
| `--skip-disk-check` | run anyway when the projection does not fit. Say why in your notes. |
| `--no-summary` | do not regenerate the summary afterwards |

### How long, and how much disk

From `--dry-run` against the shipped config, on the hardware in section 6:

| Campaign | Runs | Estimated wall clock | Projected output |
|---|---|---|---|
| B1 | 84 | ~2 h 20 m | ~7 GB |
| B2 | 24 | ~1 h | ~4 GB |
| B3 | 13 | ~30 m | ~2 GB |
| B4 | 30 | ~30 m | ~1 GB |
| B5 | 6 pairs | ~25 m | ~1 GB |

These are the config's own `estimated_seconds` / `estimated_output_mb`, which
exist to size the plan and guard the disk. They are not measurements. The real
numbers are whatever the campaign produces.

**The disk guard is not optional.** This machine has ~59 GB free and the
campaigns write CBCT volumes; a campaign that fills the disk takes the server
down with it, because the server's own job directories are on the same
filesystem. `guards.min_free_gb` is an absolute floor and `guards.margin_gb` is
what must remain after the projected output. The run is refused, not warned
about.

## 4. Adding a tool

Edit `config.yaml`. Nothing else.

```yaml
tools:
  My_Tool:
    args: {device: cuda}                    # scalars, sent as form values
    files: {scans: DATA/MyTool/testfiles/case.nii.gz}   # inputs that TRAVEL
    server_files:                           # inputs the server already holds
      model: {kind: model, name: MyTool_Models}
    data_slug: MyTool                       # the DATA/ folder it reads
    local: {folder: My_Tool, package: sadt_mytool}
    estimated_seconds: 90
    estimated_output_mb: 60
```

then add `My_Tool` to whichever campaign's `tools:` list, and `--dry-run`.

`server_files` needs an explicit `kind` (`model` or `testfile`) because the
local path has no server to ask which folder a name lives in. A tool with no
separate interpreter says so, with the reason, so the missing row cannot be
mistaken for a measurement:

```yaml
    local: false
    local_reason: in-process tool, no per-tool virtualenv
```

If you find yourself editing Python to run a tool, that is a defect in the
harness -- please report it.

## 5. What the outputs mean

```
results/
  raw/<campaign>-<UTC timestamp>.jsonl     append-only, never regenerated
  summary/<campaign>-<UTC timestamp>.{csv,md}   derived, delete and rebuild freely
```

### `results/raw/` is the evidence

One JSON object per line, **one line per individual run**, created with `O_EXCL`
so a file is never reopened and never overwritten. Fields:

| Field | Meaning |
|---|---|
| `campaign`, `tool`, `path`, `repetition` | which run this is |
| `started_at`, `finished_at`, `total_seconds` | when, and how long (UTC, ISO 8601) |
| `status` | `ok` or `failed` |
| `error_type`, `error_message` | what went wrong, for a failed run |
| `phases` | wall-clock seconds per phase; see below |
| `warmup` | true for a run the protocol discards |
| `extra` | campaign-specific: payload size, concurrency, VRAM, the parity report |
| `provenance` | CPU, RAM, GPU, driver, CUDA, NIC speed, hostname, git SHA of every repo |
| `harness_version` | bumped whenever a phase boundary moves |

**A failed run is a record, never a deletion.** Every summary reports the
failure count, and a table row computed from four successes out of six says so.

**Phases**, with `other` always derived as `total - sum(named phases)`:

| Phase | Boundary |
|---|---|
| `pack` | zipping a folder input, before a byte moves |
| `upload` | `POST /uploads` plus every part `PUT` |
| `server_exec` | the `POST /run` round trip. With inputs already uploaded, this is server-side execution plus building the result -- nothing else is in it |
| `download` | the ranged `GET`s of the result (or the streamed body) |
| `unpack` | extracting the result archive |
| `job_setup` | (local) creating the job directory and writing `job.json` |
| `compute` | (local) the tool's own process |
| `collect` | (local) copying artifacts out, for B5 |
| `interpreter_start`, `import_stack` | (B3) the two isolation probes |
| `other` | whatever the named phases did not account for |

A negative `other` would mean the phases overlap. It is a bug report about the
instrumentation, not a plausible number.

### `results/summary/` is derived

Rebuild at any time:

```bash
python -m benchmarks.summarize --campaign b1
python -m benchmarks.summarize --campaign b1 --raw b1-20260827T090000Z.jsonl
```

Every raw file of the campaign is used by default, which is right when a
campaign was run in several sittings. `--raw` restricts the summary to named
files, which is right when an exploratory run should not be mixed with a
published one. The summary always lists the files it read.

**Statistics are median with the full range, never mean and standard
deviation.** Six repetitions of a process with a hard floor and an open tail are
not normally distributed; a mean is pulled by the tail and a standard deviation
implies a symmetry the data does not have. B4's p95 is by nearest rank rather
than interpolation, so the reported tail is a job that really took that long.

**Repetition 1 is discarded** in B1 and B2 as warm-up -- it pays for the model
load and a cold page cache, which is the cost of the *first* run, not of a run.
It stays in the raw file marked `warmup: true`, so the discarding is visible and
reversible.

### B5 in particular

The comparison names the file. Where the bytes differ it also says which JSON
keys, which text lines, and -- for an imaging output -- a numeric distance: max,
mean and RMS of the voxelwise difference, how many voxels moved, and whether the
geometry (size, spacing, origin, direction) is identical. Nothing is softened
into "essentially identical".

The numeric distance is computed by a **tool's own interpreter**
(`campaigns.b5.imaging_interpreter` in `config.yaml`), because the tools carry
SimpleITK and this harness deliberately does not. Where no such interpreter is
configured, the record says the distance is unavailable and why; it is never
silently skipped.

## 6. The hardware the published numbers came from

Every record carries its own fingerprint, collected automatically at run start.
The numbers in the paper were measured on:

| Component | Value |
|---|---|
| CPU | Intel Xeon w7-3555, 28 cores / 56 threads |
| RAM | 125 GiB |
| GPU | NVIDIA RTX 6000 Ada Generation, 48 GB (49140 MiB) |
| Driver / CUDA | 595.84 / 13.2 |
| Storage | NVMe, 456 GB, 59 GB free at the time of measurement |
| Network | 1 Gb/s Ethernet (`enp0s31f6`) |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0 |
| Deployment image | Debian 13 (trixie), Python 3.13 API venv, per-tool venvs |
| Server settings | `MAX_CONCURRENT_TOOLS=4`, `MAX_CONCURRENT_GPU_JOBS=1`, `DEVICE=cuda` |

**1 Gb/s, not 10.** A 233 MB input cannot cross that wire faster than about
1.9 s at line rate. Read every transfer number against that ceiling.

**One GPU.** No multi-GPU scaling claim is available from this data.

## 7. The tests

```bash
cd <server repo>/benchmarks
.venv/bin/python -m pytest -q
```

The suite runs on a machine with no GPU, no Docker and no server. Tests needing
one of those skip with a reason naming what is missing:

```
SKIPPED [1] tests/test_live.py:53: docker is not on PATH
SKIPPED [1] tests/test_live.py:87: nvidia-smi is not on PATH
SKIPPED [1] tests/test_live.py:18: http://127.0.0.1:9 did not answer /health: ...
```

Two of them are worth knowing about:

- `test_the_job_contract_matches_the_servers` reads `server/execution/dispatch.py`
  as text and pins the file names the local path reproduces. A rename on the
  server's side that this side does not follow would otherwise make B1's local
  arm measure something the server no longer does, and nothing else would notice.
- `test_a_chunked_upload_round_trips` exercises `POST /uploads` plus the parallel
  part `PUT`s plus `DELETE` with no tool involved, so a break in the transfer
  protocol is found in seconds rather than in the middle of a 233 MB campaign.

## 8. Layout

```
benchmarks/
  README.md               this file
  NOTES-local-path.md     how the no-HTTP path reproduces the server's dispatch
  config.yaml             the ONLY input; adding a tool is an edit here
  requirements.txt
  run.py                  the CLI
  settings.py             config parsing and validation; the token
  provenance.py           hardware / git / host fingerprint
  recording.py            the append-only JSONL writer, RunRecord, PhaseTimer
  guards.py               the disk guard and scratch cleanup
  gpu.py                  nvidia-smi sampling
  artifacts.py            B5's file-by-file comparison
  _imaging_diff.py        run BY A TOOL's interpreter, for the numeric distance
  summarize.py            raw -> CSV + markdown
  execution/
    local.py              the no-HTTP path (container or host mode)
    remote.py             the HTTP client, as the real Slicer client speaks it
  campaigns/
    _common.py            what one run IS, shared by all five
    b1_latency.py  b2_network.py  b3_supervisor.py
    b4_concurrency.py  b5_parity.py
  tests/                  pytest; runs with no GPU, no server, no Docker
  fixtures/               small inputs committed for the trivial tools
  results/
    raw/                  append-only evidence
    summary/              derived, regenerable
```

## 9. Known limitations

- **Do not run two campaigns at once.** The local path takes no GPU semaphore
  (the server holds one; there is no server process in that arm), so a local run
  overlapping any other GPU work produces two wrong numbers. Plan items run
  sequentially within a campaign; keeping campaigns apart is the operator's job.
- **`local` in container mode pays a `docker exec` tax** of about 69 ms per exec
  on this machine. It is measured once per invocation and written into every
  local record as `extra.exec_overhead` so it can be subtracted. Report it
  explicitly for any tool faster than about 5 s.
- **Every local number is a warm-cache number.** Repetition 1 is discarded, and
  the page cache is not dropped between repetitions.
- **`extra.peak_vram_bytes`** (from `runner.py`, per process, via torch's caching
  allocator) and **B4's `peak_vram_mib`** (device-level, from `nvidia-smi`) are
  different measurements and must not be quoted in the same column.
- **`Test_Tool` and `Example_Tool` have no `local` arm.** They are in-process
  tools with no virtualenv of their own. `--dry-run` prints this as a skip with
  the reason.
- **`Example_Tool` fails on the venvs deployment image**: it calls
  `file_utils.load_tabular_file`, which imports pandas, and the API virtualenv
  deliberately carries none. It is kept in the config because a deterministic
  failure is what demonstrates that failures are recorded rather than dropped.
