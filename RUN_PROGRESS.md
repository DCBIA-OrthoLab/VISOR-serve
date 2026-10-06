# Run progress, and cancelling a run

`POST /run/{tool}` blocks until the tool is finished, which for a cohort is
hours. The only thing a client could show for that was an elapsed timer, so a
five-minute wait for the GPU queue, a two-hour segmentation and a tool stuck in
a loop all looked identical - and there was no way at all to call one off
except closing Slicer.

This is the contract that fixes both. Three repositories implement it and each
is written against this file rather than against the other two:

| | does |
|---|---|
| `VISOR-serve` | the run registry, the three endpoints, the phases, the kill |
| `AutomatedDentalToolsRemote` | mints the id, watches the stream, draws the bar, offers Cancel |
| `SADT-VISOR` | a tool appends its own messages, if it has anything to say |

**Everything here is optional, in both directions, and that is a hard
requirement rather than a courtesy.** A client that sends no run id gets the
behaviour it has today, byte for byte - no run directory is created and nothing
is written. A client that calls these endpoints against an older server gets a
`404` it must read as "this server does not do progress", stop watching, and
say nothing. The extension ships on its own schedule and the server does not
wait for it.

---

## 1. The run id

The **client** mints it, with a CSPRNG, and sends it on the run:

```python
run_id = secrets.token_urlsafe(24)      # matches ^[A-Za-z0-9_-]{16,64}$
```

```
POST /run/AMASSS
X-Run-Id: 3PqVh0v-Rn2mYcT8xK1LZs9dJf4wQeUa
```

**Why the client and not the server.** A server-assigned id would have to
travel in the response, and the response is the last thing that happens - the
entire point is to say something while the request is still in flight. A
`POST /runs` handshake before every run would buy a server-generated id at the
price of a round trip paid before every two-second run, and a second kind of
orphaned state to reap. Neither is worth it.

**The id is a capability.** Knowing it, plus the bearer token, is what
authorises reading a run's progress and cancelling it - the same model
`transfer.py` already applies to upload and result ids. So a client MUST mint it
with `secrets.token_urlsafe`: never a counter, never a timestamp, and never
anything derived from a patient. The server enforces the shape and nothing more,
because it cannot enforce entropy.

| the server sees | it answers |
|---|---|
| no header | the run proceeds exactly as today; no run directory exists |
| an id failing the regex | `400` |
| an id already registered, in flight or finished and not yet reaped | `409` |

**The run is registered before the form is read.** That ordering is not an
implementation detail: `await request.form()` *is* the multipart upload, minutes
of it for a CBCT, and it is precisely the stretch a client can say nothing about
today. The client also opens its event stream from a second thread the instant
it has minted the id, while the first thread is still inside the POST, and the
two cannot be ordered. Registering after the form was parsed would answer that
watcher a `404` - which the client is told to read as "an older server" - so the
feature would silently do nothing on exactly the runs it exists for. Registering
first leaves a window of microseconds, and the client tolerating a brief `404`
closes it.

One consequence, worth stating because it looks odd: the tool is resolved
*after* the run is registered, so a `404` for an unknown tool now has a run
directory of its own to take down.

---

## 2. The endpoints

All three Bearer-protected, all three `404` for an id this server never had or
has already reaped.

### `GET /runs/{run_id}` - one snapshot

```json
{"run_id": "...", "state": "running", "phase": "running",
 "fraction": 0.35, "message": "scan 14 of 40",
 "depth": 0, "started_at": 1757400000.0, "updated_at": 1757400123.0,
 "events": [ ...every event so far, oldest first... ]}
```

For tests, for debugging, and for a client that cannot hold a streaming
connection. The Slicer client does not use it. `?logs=user` adds the tool's
log lines for the requester (§4e).

### `GET /runs/{run_id}/events` - Server-Sent Events

`Content-Type: text/event-stream`, `Cache-Control: no-store`,
`X-Accel-Buffering: no`. One `data:` frame per event, oldest first,
**including the events written before this watcher connected** - a watcher that
attaches late is never behind. The stream ends when a terminal event has been
delivered, or when the run directory disappears.

`?logs=user` interleaves the tool's log lines written for the requester, as
events with `"kind": "log"` (§4e). **Without it there are none**: a client
released before log lines existed would draw one as progress.

The last header is not decoration: nginx buffers a proxied response by default,
which for a stream means the client sees nothing until the run ends - the exact
failure the endpoint exists to prevent. `curl` needs `-N` for the same reason.

### `DELETE /runs/{run_id}` - cancel

`204`, idempotent. Writes the cancel marker and, if a process group has been
recorded, signals it. See §6.

### And the POST's own answer, when it was cancelled

```
499  {"detail": "Run cancelled by the client."}
```

`499` is nginx's and non-standard, and that is deliberate: no standard code
means "the caller withdrew this", and the client MUST be able to tell a
cancellation from a failure without inspecting a message. One closes the panel
quietly; the other opens an error dialog.

---

## 3. The event

One JSON object per SSE frame, and one per line of the run's `events.jsonl`:

```json
{"seq": 12, "at": 1757400123.4, "state": "running", "phase": "running",
 "fraction": 0.35, "message": "scan 14 of 40", "depth": 0}
```

- `seq` - monotonic from 0, per run. The client dedupes and orders on it.
- `at` - `time.time()`, for elapsed display only.
- `state` - `pending | running | done | failed | cancelled`. The last three are
  terminal.
- `phase` - a small CLOSED vocabulary, so the client shows the clinician's word
  rather than the server's: `received`, `staging`, `queued_gpu`, `running`,
  `packaging`, `done`, `failed`, `cancelled`.
- `fraction` - `0.0..1.0`, or `null` when unknown. **Never fabricated**: a
  progress bar believes what it is given.
- `message` - free text, truncated by the server to 200 characters.
- `depth` - supervised nesting depth, `0` being the tool the client asked for.
  It is what lets a panel show `AREG -> ASO  30%` without the client knowing
  what a chain is.

Optional, and ignored by a client that does not know them:

- `call` - which supervised call wrote the record (`"1"`, `"1.2"`), absent for
  the root's own records.
- `tool`, `edge`, `span` - on the supervisor's markers bracketing a nested
  call: the callee's name, `open` or `close`, and the span of the caller's bar
  it was given, if any.
- `own_fraction` - when a weighted chain rescaled `fraction` (§4d), what the
  record itself said, on its own tool's 0..1.

`seq` is **the line number, assigned when the record is read**, and nothing
writes it. It could not be written: a value monotonic per run would have to be
counted first, and between the count and the write a supervised child in another
process could append its own. The file's order is the sequence, every reader
derives the same numbering from it, and the client gets exactly the monotonic
key it was promised.

---

## 4. Where events come from

### 4a. The server, for every tool, with no tool cooperation at all

This is what makes the feature useful on day one rather than after fifteen tools
have been edited:

| phase | when |
|---|---|
| `received` | the run id was registered, before a byte of the body is read |
| `staging` | each input written to the work dir, as `input 3 of 8` |
| `queued_gpu` | before the GPU slot is acquired, and only when the run wants the card |
| `running` | immediately after the tool's process starts |
| `packaging` | the tool returned; the response or the archive is being built |
| `done` / `failed` / `cancelled` | terminal, one of them always |

`queued_gpu` is the one worth pointing at. `MAX_CONCURRENT_GPU_JOBS` is one
counter across every tool, so behind a cohort that wait is measured in hours -
and today it is indistinguishable from a tool that is running.

### 4b. The tool, through a file it appends to

The server sets **`SADT_PROGRESS_FILE`** in the tool's environment: an absolute
path, set only when the run has a directory. A tool reports progress by
appending one JSON object per line to it:

```json
{"fraction": 0.35, "message": "scan 14 of 40"}
```

`fraction` may be `null`. The server stamps `seq`, `state` and `phase` itself
and ignores anything a tool sends for them - a tool's line is a running tool,
whatever it claims, and only the server declares a run done, failed or
cancelled. `at` and `depth` are facts only the writer knows, so they are read
when present and plausible: `at` falls back to the moment the line was first
seen, which the poll keeps within a quarter second of the truth, and an
implausible `depth` is read as the root's.

Rules for the tool side, all four load-bearing:

- **Standard library only, and best effort.** Wrap the whole thing so a failure
  to write can never propagate into the tool. A tool that cannot report its
  progress is still a tool; turning a failed `write()` into a failed cohort
  would be an absurd trade.
- **One `write()` of under 4096 bytes to an `O_APPEND` handle.** POSIX
  guarantees that is atomic below `PIPE_BUF`, which is the entire locking
  strategy: the server process, the tool's process and every supervised level
  below it append to one file with nothing between them. Truncate the message to
  200 characters and it cannot come close to the bound.
- **Open without `O_CREAT`.** The server creates the file when it registers the
  run. A stale variable inherited from somewhere else must cost a no-op, not a
  file littered wherever it pointed.
- **Never a patient identifier.** Position in the batch, never the file name.
  `ALI_CBCT`'s existing log line states this rule and obeys it; it is the model.
  A progress message is written to disk and travels to a panel, so it is held to
  the same standard as a log line.

Roughly fifteen lines, and each tool carries its own copy, the way AREG's three
engines each carry their own `tools.py`. A shared package here is exactly what
the split between these repositories removed.

**Why not `sup.progress()`.** It is the obvious place and it is the wrong one.
Only 4 of the 15 tools take a supervisor, and `SADT-VISOR`'s own CONTRIBUTING
tells authors in as many words to reach for one *only* where the ordering
genuinely forbids plain chaining - so making the supervisor the progress channel
would mean growing one on nine tools purely as a conduit, against that
repository's own rule. A second injected keyword-only parameter is worse:
`describe.py`'s unannotated-keyword-only marker is a single slot, one string and
one boolean, and threading another through would touch ten code sites and some
forty call sites. An environment variable costs none of that and reaches all
fifteen tools rather than the four that happen to have a supervisor already.

`sup.progress()` nevertheless keeps working and writes to the **same file**, so
the four tools that already call it need no change whatever. It is the
compatibility path, not the recommended one.

### 4c. Nesting

A supervised child re-enters `runner.py` with its own environment.
`SADT_PROGRESS_FILE` is inherited, so a child appends to the SAME file as its
parent with `depth` one higher. That is the whole implementation of chain
progress: no plumbing, and no level had to be told about any other.

---

### 4d. Weighting a nested call: one bar for the whole chain

Without it, each level's records carry that level's own 0..1, so a chain's bar
runs ASO's 0..0.2, then ALI's 0..1, then ASO's 0.6..1. A caller fixes that by
saying how much of ITS bar a call fills:

```python
sup.run("ALI_CBCT", ..., _progress=(0.2, 0.6))
```

`_progress` is the supervisor's, taken out before the callee sees it, and it
rides the call's two markers as `span`. The server folds every record into a
tree of calls as it reads the file: a level's position is the furthest of what
it said itself and `start + (end - start) * position` of each weighted call,
a finished call counting as its whole span. ALI at half way is ASO at 0.4, and
under an AREG that gave ASO 0..0.5, AREG at 0.2.

`fraction` on the wire is then the position of the OUTERMOST call the weighting
reaches -- the root, when every level was weighted -- and `own_fraction` keeps
what the record said. A client that knows nothing of this draws a bar that only
moves forward.

- **Nothing weighted, nothing changed.** A record with no weighted call above or
  below it keeps the fraction it was written with, so a chain nobody weighted
  reads exactly as it did.
- **One span per call.** Two calls in a loop sharing one span read as the
  furthest of the two; give each its own slice.
- **Whose record it is** comes from its `call`. A record with none -- a tool's
  own `progress.py` written before it learned to stamp one -- is attributed to
  the one call open when it was read, since the caller is blocked in
  `sup.run`; with two calls open at once that is a guess, and no guess is made:
  the record keeps its own fraction.

### 4e. Log lines

```python
sup.log("3 of 7 landmarks found; orienting on those", level="warning")
sup.log("Scan 4 has no mandible and was skipped", level="warning", user=True)
```

```json
{"kind": "log", "at": 1757400123.4, "level": "warning", "audience": "user",
 "message": "Scan 4 has no mandible and was skipped", "depth": 1, "call": "1"}
```

Same file, same append rules as a progress record (§4b). `level` is `debug`,
`info`, `warning` or `error`; `audience` is `user` or `admin`, and a line with
none is the operator's.

| audience | reaches | as |
|---|---|---|
| `admin` (default) | the operator page's run console; warnings and errors are kept in the run's history | **redacted** (`server/redact.py`) |
| `user` | the client's stream and snapshot, when it asked with `?logs=user`; the operator console too | as written for the requester |

**Never a file name or a patient's.** A `user` line is held to the progress
message's rule (§4b) because it is read in the same panel. An `admin` line is
redacted anyway -- any path, file name with a data suffix, address, or token
mixing letters with two or more digits is replaced by what it was -- but the
redaction is a net, not a licence.

A log line is never progress: it carries no fraction, is counted apart from
`MAX_RUN_EVENTS` (500 per reader), and is read as a running tool's whatever
phase or state it claims.

`sup.log(message)`, one argument, is what the method always accepted and stays
an `info` line for the operator. The supervisor's own narration (`running
'ALI_CBCT'`, a channel grant) goes to stderr only and never into the file.

### 4f. Why a run failed

The runner records, beside the exception's class name, where it came from:

```json
{"error": {"type": "KeyError", "message": "...",
           "origin": {"tool": "ALI_CBCT", "chain": ["AREG", "ASO", "ALI_CBCT"],
                      "error_type": "KeyError", "message": "...",
                      "where": "sadt_ali_cbct/engine.py:742 in _predict",
                      "stage": "scan 3 of 8", "fraction": 0.375}}}
```

Written by the level that raised and relayed unchanged by every level above,
so the root's record names the tool that actually broke. `where` is the
innermost frame in the tool's own `src/`, a place in code. `stage` is the last
progress message that level wrote.

The server turns it into a diagnosis -- redacted, every field checked against
the shape it may have -- and keeps it in the run's ledger record, shown at the
top of the operator page's run dialog and as the console's last line. When
nothing below recorded an origin (a crash, a timeout, a 422 from validation),
what the server knows stands in: the tool asked for, the calls still open, and
the first line of the exception. The failure's server log line is the same
diagnosis, redacted the same way.

None of it is on the client's stream: the requester gets the status code and
the message it always got.

## 5. Storage, and why it is not the job directory

`TEMP_DIR/runs/<run_id>/`, modelled on `wire/transfer.py` in every respect:

| | |
|---|---|
| `events.jsonl` | append-only, one record per line, created empty at registration |
| `pgid` | the tool's process group, written once, immediately after its process starts |
| `cancel` | a marker file, created by `DELETE` |

Not the job directory, for three reasons with teeth: progress starts before the
job directory exists (during staging), the job directory is deleted on every
error path, and a run may never reach dispatch at all.

Every append and every read stamps the directory, so `RUN_TTL_SECONDS` is an
**idle** timeout exactly as `TRANSFER_TTL_SECONDS` is - a cohort reporting
progress for six hours is never reaped under itself. The same `_reaper_loop`
that sweeps transfers sweeps these.

The reaper is a safety net rather than the normal path. **A run is removed with
the request that owned it**, on success (as a background task, so the directory
survives until the response has finished streaming and a watcher has the whole
download to collect the terminal event) and immediately on every failure. What
the reaper bounds is the run whose client vanished mid-POST, and it needs
bounding because a progress message is written by a tool and can name a file.

### Confidentiality

A progress message is stored under `TEMP_DIR`, deleted with the run, and
**never logged** beyond its phase name. Nothing in the server's log lines
carries a message, a file name or an argument value - timestamp, endpoint, tool,
status, duration, size, and nothing else.

`MAX_RUN_EVENTS` caps how many events a run reports, so a tool printing per
slice in a 500-patient cohort cannot fill the disk; past the cap one "not
reported" event is delivered and the rest are dropped. A terminal event is never
dropped - a stream that could not end would leave every watcher hanging on a run
that finished half an hour ago.

---

## 6. Cancellation, in detail

Two mechanisms, because neither covers the whole window.

1. **`DELETE` signals the process group**, when a `pgid` has been recorded.
   Immediate, and what actually stops a two-hour nnUNet. It is a `SIGTERM` to
   the whole group, not to the one pid: nnUNet, torch's DataLoader and shapeaxi
   all fork workers, and killing only the parent leaves those holding VRAM on a
   card nothing can be attributed to any more.
2. **The run polls the cancel marker**, which covers the window where there is
   no process yet - inputs still staging, or the run sitting in the GPU queue.
   Checked before the job file is written, before and after the GPU slot is
   acquired, immediately after the process starts (the race between the marker
   being written and the pgid being recorded), and every
   `RUN_CANCEL_POLL_SECONDS` while the tool runs.

**The endpoint holds no handle on the process, and does not need one.** The
`DELETE` may be served by a different `uvicorn --workers` process than the one
blocked in the POST; that is why the group id goes on disk rather than in a
module global. `os.killpg` reaches across processes of the same user, and
`start_new_session` made the child a group leader. Escalation from `TERM` to
`KILL` belongs to the side that *does* hold the handle - it can wait for the
process and reap it, where the endpoint could only block its own `204` for the
grace period.

A cancelled run raises a class of its own, never the one a failing tool raises,
so the two can never be confused; the job directory is discarded as on any other
error path; and the POST answers `499`.

While the tool runs, the wait is broken into `RUN_CANCEL_POLL_SECONDS` slices
rather than one long one. **The timeout semantics are unchanged** - the same
budget, the same process-group kill, the same message naming both
`[tools.X] timeout_seconds` and `TOOL_TIMEOUT_SECONDS`.

---

## 7. Configuration

| setting | default | what it is |
|---|---|---|
| `RUN_TTL_SECONDS` | `900` | idle TTL of a run directory |
| `RUN_EVENT_POLL_SECONDS` | `0.25` | how often the SSE endpoint tails the file |
| `RUN_CANCEL_POLL_SECONDS` | `1.0` | how often a running tool checks the cancel marker |
| `MAX_RUN_EVENTS` | `2000` | events reported per run |

All four go through `config.Settings` and are documented in
`server/.env.example`, like every other setting. None of them appears in the
root `.env.example`, which holds only what docker compose interpolates before a
container exists.

---

## 8. What the client does

- Mints a run id per run and sends it.
- Runs the POST in the thread it already uses, and opens a **second** thread for
  the event stream - the run thread is blocked inside the POST and cannot poll.
- Delivers events to the Qt main thread by whatever mechanism the existing
  `progress_cb` already uses. No new cross-thread mechanism.
- Renders `phase` translated into clinician language, plus `message`, plus a
  determinate bar when `fraction` is not null.
- Offers a Cancel button per in-flight run. Cancelling a run still queued
  client-side removes it with no HTTP call at all.
- Treats `499` as "Cancelled" and closes the run quietly. It MUST NOT raise an
  error dialog.
- Treats a `404` from the events endpoint as an older server: stop watching, say
  nothing, keep the elapsed timer.
- May ask for `?logs=user` and show those lines in a log pane, never in place of
  the progress message. A server that predates log lines ignores the parameter.

---

## 9. Where it lives, on this side

| | |
|---|---|
| `server/wire/runs.py` | the registry: register, append, read, cancel, reap |
| `server/main.py` | the header, the phases, the three endpoints, `499` |
| `server/execution/dispatch.py` | the cancel checks, the `pgid`, `queued_gpu`, the kill |
| `server/execution/runner.py` | `sup.progress()` and `sup.log()` appending to the file, call ids, spans, a failure's origin |
| `server/redact.py` | what a tool's words lose before an operator reads them |
| `server/tests/test_runs.py` | all of it, with no GPU, no weights and no network |

`runs.py` is `transfer.py`'s discipline applied to a second kind of ephemeral
state, deliberately and line for line: the id matched against a regex before any
path is built from it, state on disk rather than in a module global, `touch()`
making the TTL an idle timeout, and the reaper on a timer. Read one and you have
read the other.
