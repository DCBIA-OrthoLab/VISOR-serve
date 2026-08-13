# Moving one tool out of the server

This is the companion to [`ADDING_A_TOOL.md`](ADDING_A_TOOL.md), which
describes a tool the server **imports**. This one describes moving such a tool
to the other side: a folder the server never imports, running in its own
virtualenv.

**One tool at a time, and reversible.** Nothing here is a big-bang cutover: the
registry serves both kinds side by side, so a tool can move while the other
fourteen carry on exactly as before. Deleting `server/tools/` happens once —
after the last tool has made this trip.

---

## What actually moves

| | before | after |
|---|---|---|
| the code | `server/tools/AMASSS/` | `<TOOLS_DIR>/amasss/src/` (packaged in `sadt-tools`) |
| the argument schema | `ArgSpec` in Python | `.schema.json`, generated from the source |
| the dependencies | `server/requirements.txt`, shared with everything | `pyproject.toml` + `uv.lock`, the tool's own |
| `server_selectable` | in the tool's `ArgSpec` | `deployment.toml`, **server-side** |
| how it runs | imported, `run()` called in-process | `runner.py`, in the tool's interpreter |

The last row is the point: after the move, this tool's numpy is nobody else's
business.

---

## The loop, per tool

### 1. Package it (in `sadt-tools`)

Produce a folder named **exactly after the tool**:

```
amasss/
├── pyproject.toml     its own dependencies, its own requires-python
├── uv.lock            committed, so the image build is `uv sync --frozen`
├── .schema.json       generated from src/, including source_hash
└── src/
    └── amasss.py      defines run(**params)
```

The folder name is not cosmetic: `dispatch.py` looks the interpreter up at
`<TOOLS_DIR>/<tool name>/.venv/bin/python`, so a folder named anything else
registers a tool that cannot be run. The server refuses it at startup.

`run()` returns named outputs — `{"outputs": {"mandible": "/jobs/…/x.nii.gz"}}`
— or a path, or a list of paths. All three work; the first is the one to write
(see `server/README.md`).

### 2. Translate the schema, and know what it costs

The schema's types are `path`, `str`, `int`, `float`, `bool`, `list[str]`.
That is narrower than what `ArgSpec` can express, so some of what the Slicer
panel shows today has to be declared differently or is lost:

| `ArgSpec` | `.schema.json` | what happens |
|---|---|---|
| `str` / `int` / `float` / `bool` | same | unchanged |
| `"csv_file"`, `"volume_or_zip_file"`, … | `"path"` + `"extensions": [...]` | declare `extensions` or the file dialog offers everything |
| `"folder"` | `"path"` | the client still zips it, the server still unpacks it |
| `"multichoice"` | `list[Literal[...]]` | check boxes, kept: the generator publishes the options as `choices` |
| `"choice"` | `Literal[...]` | a combo box, kept, same way |
| `label`, `section`, `ui`, `groups`, `visible_when` | — | not expressible: the panel falls back to one flat column |
| `server_selectable` | `deployment.toml` | see step 3 |
| `initial` | `"default"` | unchanged |

What that costs, tool by tool, measured on the current registry:

| tool | args | what it loses without more work |
|---|---|---|
| `test_tool` | 2 | nothing |
| `SurgMovPred` | 2 | nothing beyond `extensions` |
| `example_tool` | 6 | 1 choice, 1 multichoice |
| `CrownSeg` | 6 | 1 choice |
| `BatchDentalSeg` | 4 | 4 labels, 4 sections |
| `AMASSS` | 8 | 2 multichoices (structures, merge) |
| `ALI` | 6 | 3 multichoices, 3 tabbed layouts, 6 labels, 6 sections |
| `ASO` | 12 | 3 choices, 4 multichoices, **7 `visible_when`**, 4 layouts, 12 labels |

Read that column as an order of migration, not as a blocker: `test_tool`,
`SurgMovPred` and `BatchDentalSeg` are nearly free, and `ASO` — whose four
modes only make sense with `visible_when` hiding the inert half — should be
last, and probably wants the schema to grow before it goes.

> **The panel is the deliverable, not the schema.** A tool whose 119 landmarks
> arrive as one flat column of check boxes is a regression a clinician sees,
> even though every test passes. If a tool needs `groups`/`ui`/`visible_when`,
> say so before porting it rather than after.

### 2b. What the server fills in, so the tool must not

Three things a packaged tool declares but a caller never sends:

- **`output_dir`** — every tool takes it as a required `Path` and writes only
  there. It is removed from the published schema and filled in with the job's
  own `output/`.
- **`device`** — injected from `settings.DEVICE` when the caller picks none,
  because a tool that no longer reads the environment would otherwise run on
  its own default (`cuda`) on a CPU server.
- **the GPU slot** — the per-tool semaphores are gone with the tools that held
  them, so the server serialises card work through `MAX_CONCURRENT_GPU_JOBS`,
  one counter across all tools. **Every run is assumed to want the card**; the
  only way out is a `device` argument resolving to a CPU value. A tool that
  imports torch without declaring `device` would otherwise never queue at all,
  and two of them would meet on the same device.

### 3. Move the server-side bits to `deployment.toml`

Which arguments may be filled from *this* server's `DATA_DIR` is not a property
of the tool, so it does not travel with it:

```toml
[tools.amasss]
server_selectable = { model = "model", scans = "testfile" }
max_upload_mb = 2000
# The packaged tools are lowercase; the bundles staged under DATA/ are not.
data_dir = "AMASSS"
```

Checked at startup: an argument the tool does not declare, or one that is not
a `path`, is an error rather than a dropdown that never appears.

### 4. Build the image with it

```bash
TOOLS_CONTEXT=../sadt-tools/dist docker compose --profile venvs build inference-venvs
docker compose --profile venvs up -d inference-venvs        # port 8001
docker run --rm <image> /opt/sadt/.venv/bin/python /opt/sadt/verify_dedup.py
```

See [`docker/README.md`](docker/README.md).

### 5. Prove it — this is the step that is not optional

```bash
cd server
python parity.py --imported AMASSS --args case.json
```

It runs both forms on the same arguments and compares **what a caller
receives**: every file produced, by name and by hash, and the returned value
with paths resolved to the artifacts they name. Absolute paths are never
compared — one run wrote into a job directory, the other into a scratch
directory, and neither name means anything to a client.

Where the two schemas differ (a `multichoice` became a `list[str]`), pass the
second argument set explicitly:

```bash
python parity.py --imported AMASSS --args imported.json --packaged-args packaged.json
```

**A difference is not automatically a defect.** The packaged tool runs against
its own pinned dependencies: a different numpy moves voxels, a newer SimpleITK
writes a different header. What is not acceptable is a difference nobody
looked at. Run it on a real case, on real data, and read every line it prints.

### 6. Flip it

Drop the packaged folder into `TOOLS_DIR`. A folder with a `.schema.json` is
**never imported**, so the in-process copy stops being used the moment the
packaged one is present — under the same name, so no client changes.

**Rolling back is deleting the `.schema.json`.** The folder is imported again
on the next start. Which is why the in-process copy stays in `server/tools/`
until every tool has moved: it is the rollback.

### 7. Only then, phase 4

Once **every** tool has been through steps 1–6 and is running packaged in
production:

- delete `server/tools/`;
- delete the `SADT_DISPATCH_MODE` flag and the in-process branch of
  `Tool.invoke`;
- delete the legacy-import half of `registry.py` (it already tolerates the
  `tools/` package being absent);
- retire the heavy `server/requirements.txt`, leaving `requirements-api.txt`.

Not before. Each of those is one line of deletion and no way back.

---

## Checklist

- [ ] folder name == `.schema.json` `name` == what the client sends to `/run/<name>`
- [ ] `source_hash` regenerated — `python server/schema_hash.py <tool>/src`
- [ ] `uv.lock` committed, and `requires-python` matches what the pins actually
      support (numpy 1.26 has no wheel for 3.13 — an old pin drags an old
      interpreter behind it)
- [ ] `extensions` declared on every `path` argument that is not "any file"
- [ ] `description` on every argument (the client shows it under the field)
- [ ] `server_selectable` / `max_upload_mb` moved to `deployment.toml`
- [ ] the image builds and `verify_dedup.py` exits 0
- [ ] `parity.py` run on a real case, output read line by line
- [ ] the Slicer panel opened once, by eye, before and after
