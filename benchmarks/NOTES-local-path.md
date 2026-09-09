# The `local` execution path: what it reproduces, and exactly how

**Question.** B1 compares a tool run through the HTTP API against the same tool
run with no HTTP at all. For that comparison to mean anything, the no-HTTP arm
has to be *the same execution* the server's dispatcher performs, minus the
protocol -- not a different way of running the tool that happens to produce the
same file.

**Answer, in one line.** It is reproducible exactly, in two places, and the
harness does it in both. The per-tool virtualenvs exist **on the host** (in the
`sadt-tools` working checkout) **and inside the deployment image**, and the exec
line is the same on either side.

Everything below is what that claim rests on, including the four places where
the reproduction is *not* exact and what the harness does about each.

---

## 1. What the server actually does

`server/execution/dispatch.py`, function `dispatch()`, after `Tool.invoke` has
validated the arguments:

```
command = [interpreter, runner, "--job", job_path]
```

with

| Piece | Value | Where it comes from |
|---|---|---|
| `interpreter` | `<TOOLS_DIR>/<tool>/.venv/bin/python` | `dispatch.tool_interpreter()` |
| `runner` | `settings.RUNNER_PATH` | `server/execution/runner.py` |
| `job_path` | `<TEMP_DIR>/job_<uuid>/job.json` | `dispatch._create_job_dir()` |

`tool_interpreter()` searches **two levels**, because a tool may live inside a
grouping folder: `ALI_CBCT` is at `tools/ALI/ALI_CBCT/`, not `tools/ALI_CBCT/`.
The registry is asked first, and a name-based search under `TOOLS_DIR` is the
fallback.

The job directory is created with `input/` and `output/` subdirectories, and
`job.json` holds exactly four fields:

```json
{"job_id": "...", "tool": "ASO", "job_dir": "/jobs/job_<uuid>", "params": {...}}
```

`params` are the caller's validated arguments plus two the **server** fills in
(`dispatch._server_provided`):

- `output_dir` -> `<job_dir>/output`, for every tool declaring it;
- `device` -> `settings.DEVICE`, when the tool declares `device` and the caller
  did not choose one.

The child process gets (`dispatch._child_environment`):

- the server's own environment, inherited (PATH, LD_LIBRARY_PATH,
  CUDA_VISIBLE_DEVICES, ...);
- **`API_TOKEN` removed** -- it is the server's credential and the tool venvs
  hold third-party code;
- `SADT_API`, `SADT_JOB_ID`, `SADT_JOB_DIR`;
- `SADT_SUPERVISOR_DEADLINE`, an **absolute** `time.monotonic()` instant, when
  the tool has a timeout. Absolute rather than a duration so a supervised chain
  does not get the full budget again at every hop.

and is started with `cwd=job_dir` and `start_new_session=True` (so a timeout can
`killpg` the whole group -- nnU-Net, torch's DataLoader and shapeaxi all fork).
stdout and stderr go to `stdout.log` / `stderr.log` **files**, never pipes.

On the other side, `runner.py`:

1. derives the tool folder from **its own `sys.prefix`** (`<tool>/.venv` ->
   `<tool>/`), which is why no extra argument is needed -- the choice of
   interpreter *is* the choice of tool. `SADT_TOOL_DIR` overrides it;
2. puts `<tool>/src` first on `sys.path` and imports the single package under it
   (the same rule `describe.py` uses), refusing a module imported from anywhere
   else;
3. coerces JSON strings to `Path` where the annotation asks for one;
4. injects a supervisor as `sup` when the signature declares one;
5. calls `run(**params)` and writes `result.json` atomically, plus
   `peak_vram_bytes` when torch was in `sys.modules`.

## 2. Where the virtualenvs live -- the question that had to be settled

**Both.** They are two independent builds of the same lockfiles.

**In the image** (`docker/Dockerfile`), built with

```
uv sync --frozen --no-install-project --no-dev --all-extras
```

in a single layer (so overlayfs cannot break the hardlinks that deduplicate
eleven torch stacks from ~63 GB down to ~26 GB), at `/tools/<name>/.venv`.
Verified present, 2026-08-25, in `slicer-remote-tool-server-inference-venvs-1`:

```
/tools/AMASSS  /tools/ASO  /tools/Batch_Dental_Seg  /tools/Crown_Seg
/tools/Surg_Mov_Pred  /tools/ALI/ALI_CBCT  /tools/ALI/ALI_IOS
/tools/AREG/AREG_CBCT  /tools/AREG/AREG_IOS  /tools/AREG/AREG_IOSCBCT
```

**On the host**, in the `sadt-tools` working checkout at
`/home/luciacev/code/sadt-tools/tools/<name>/.venv`. The Dockerfile excludes
these from the build context (`--exclude=**/.venv`) precisely *because* they are
there. Verified working, with CUDA:

```
Crown_Seg      Python 3.11.15   torch 2.11.0+cu128   cuda_available True
AMASSS         Python 3.11.15   torch 2.8.0+cu128    cuda_available True
ASO            Python 3.11.15   no torch
Surg_Mov_Pred  Python 3.12.13   no torch
```

### How much the two builds differ

Measured by listing installed distributions in each and diffing, 2026-08-25.
Every runtime dependency is identical -- same torch, same numpy, same monai,
same SimpleITK, same versions. The difference is **exactly the dev group plus
the project itself**:

| Tool | host distributions | image distributions | lines differing |
|---|---|---|---|
| ASO | 25 | 20 | 5 |
| Crown_Seg | 169 | 164 | 5 |
| AMASSS | 103 | 98 | 5 |
| Surg_Mov_Pred | 22 | 16 | 6 |
| AREG_IOSCBCT | 38 | 33 | 5 |
| ALI_CBCT | 42 | 36 | 6 |
| ALI_IOS | 53 | 48 | 5 |
| Batch_Dental_Seg | 102 | 97 | 5 |
| AREG_CBCT | 48 | 43 | 5 |
| AREG_IOS | 54 | 49 | 5 |

The extra host entries are, in every case, a subset of:
`pytest`, `pluggy`, `iniconfig`, `packaging`, `sadt-testkit`, and the tool's own
package (`sadt-aso`, `sadt-crownseg`, ...) which the image deliberately does not
install (`--no-install-project`; `runner.py` puts `src/` on `sys.path` instead,
and refuses a module imported from anywhere but `src/`, so the installed copy
cannot win even where it exists).

Python patch versions match exactly (3.11.15 / 3.12.13 on both sides).

The **operating systems do not match**: the host is Ubuntu 22.04.5 with glibc
2.35, the image is Debian 13 (trixie) with glibc 2.41. Every tool's Python
dependencies are pinned; the C libraries under them are not.

## 3. What the harness does

`benchmarks/execution/local.py` builds exactly the command above and writes
exactly that `job.json`. Two modes, selected by `local.mode` in `config.yaml`:

### `mode: container` (the default)

```
docker exec -i -u sadt -w <job_dir> \
  -e SADT_JOB_ID=... -e SADT_JOB_DIR=... -e SADT_API=http://127.0.0.1:8000 \
  slicer-remote-tool-server-inference-venvs-1 \
  env -u API_TOKEN \
  /tools/<name>/.venv/bin/python /opt/sadt/server/execution/runner.py --job <job_dir>/job.json
```

Same interpreter, same runner file, same `DATA` mount, same job layout as the
server's own children. `-u sadt` matters: `/jobs` belongs to the unprivileged
account the image runs as, and a root-owned directory left behind by a benchmark
would break the next real request. `env -u API_TOKEN` reproduces
`dispatch._child_environment`'s `pop`, not an empty value.

**Verified end to end**, 2026-08-25: a job with deliberately empty `params`
reached `run()` and came back as
`TypeError: run() missing 2 required positional arguments: 'input' and 'reference'`
with `result.json` written and the job directory removed -- i.e. the interpreter
started, `runner.py` imported the tool from `src/`, and the contract held.

### `mode: host`

```
/home/luciacev/code/sadt-tools/tools/<name>/.venv/bin/python \
  /home/luciacev/code/slicer-remote-tool-server/server/execution/runner.py \
  --job <job_dir>/job.json
```

with `cwd=job_dir`, `start_new_session=True`, and the environment
`dispatch._child_environment` builds. This models a scientist who installed the
tools on their own workstation, which is the deployment the paper contrasts
against remote execution.

**Verified end to end**, 2026-08-25, on `tools/_template` (numpy only): six runs
through the harness, median 0.117 s, `result.json` read back and the returned
path collected.

## 4. Where the reproduction is NOT exact

Four differences. None is hidden; each is recorded or reported.

### 4.1 No GPU semaphore

`dispatch.dispatch()` takes a `MAX_CONCURRENT_GPU_JOBS` semaphore (default 1)
around the run. The local path does not, because there is no server process to
hold one in. **Consequence:** a `local` run started while the server is executing
a GPU job will contend for the card, and both numbers will be wrong.
**Mitigation:** B1's local arm must not be run concurrently with any other
campaign, and the harness runs plan items sequentially. This is a scheduling
constraint on the operator, not something the harness can enforce.

### 4.2 No argument validation

`Tool.invoke` validates arguments against the published schema *before*
dispatching. The local path writes `params` straight from `config.yaml`.
**Consequence:** the local arm excludes validation cost -- which is
microseconds, and is server-side work rather than tool work, so excluding it is
arguably the right boundary for a "the tool itself" measurement. **It does mean a
mis-typed argument fails differently on the two arms**: the API answers 422, the
local path reaches `run()` and raises `TypeError`. Both are recorded as failed
runs with their own error, so the difference is visible in the raw file.

### 4.3 Container mode pays for `docker exec`

The server forks its child directly; container mode reaches it through
`docker exec`. Measured on this machine: **69 ms median per exec**
(min 67, max 70, 5 repetitions). A local run makes several -- job directory
creation, writing `job.json`, then the tool itself.

The harness measures this once per invocation and writes it into **every** local
record as `extra.exec_overhead`, so it can be subtracted rather than argued
about. It also keeps `job_setup` and `compute` as separate phases: `job_setup`
carries the setup execs, `compute` carries one. Against a 60 s segmentation it is
0.1%; against a tool answering in 100 ms it is most of the measurement. **Report
it explicitly for any tool faster than about 5 s.** `mode: host` has no wrapper
and records the overhead as structurally zero.

### 4.4 The two in-process tools have no local path at all

`Test_Tool` and `Example_Tool` are `Tool` subclasses that the API **imports**
(`server/tools/<Name>/<Name>.py`). They have no virtualenv, so `base.Tool.invoke`
falls through to `self.run(**cleaned)` in the API process even with
`SADT_DISPATCH_MODE=subprocess`. There is no separate interpreter to invoke, so
there is no `local` arm to measure.

This is a property of those two tools, not a gap in the harness. `config.yaml`
states it (`local: false` with a `local_reason`) and `--dry-run` prints it as a
skip with the reason attached, so the missing row cannot be mistaken for a
measurement.

## 5. Ambiguities that remain

- **Which mode should the published numbers use?** Settled 2026-08-25, and the
  two campaigns take different answers because they ask different questions.
  **B1 takes `container`**: it is byte-for-byte the execution the server
  dispatches, so the local-to-loopback delta is attributable to the protocol and
  nothing else. **B5 takes `host` as its primary and `container` as its
  control**: the host checkout is Ubuntu 22.04 / glibc 2.35 and the image is
  Debian 13 / glibc 2.41, so byte-identical artifacts *across that gap* is a
  reproducibility result, where byte-identical inside one image is nearly
  tautological; running `container` as well is what separates an OS difference
  from a protocol one. Neither is a default any more: `--local-mode` names the
  mode on the command line and every record carries `extra.local_mode`.

- **Cold versus warm start is not separated inside the local path.** Repetition 1
  is discarded as warm-up, which is the protocol, but the harness does not drop
  the page cache between repetitions. Every reported local number is therefore a
  warm-cache number, and should be described that way.

- **The image's virtualenvs are hardlink-deduplicated and the host's are not.**
  Whether that changes first-import time measurably has not been tested. If B3's
  per-child import cost differs between the two modes by more than the exec
  overhead, this is the first thing to look at.

- **`peak_vram_bytes` comes from `runner.py`, which reads
  `torch.cuda.max_memory_allocated()`** -- per process, through torch's caching
  allocator. It is not comparable with the device-level `nvidia-smi` figure B4
  reports, and the two must not be quoted in the same column.
