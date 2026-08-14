# Adding a tool

**You never edit this repository.** A tool is a folder in `sadt-tools`; the
server discovers it, publishes it, and runs it with no change here.

## The whole thing

```
tools/MyTool/
├── pyproject.toml      its dependencies, its Python
├── uv.lock
└── src/sadt_mytool/
    └── __init__.py     defines run()
```

```python
from pathlib import Path

def run(scans: Path, model: Path, output_dir: Path, device: str = "cuda") -> dict:
    ...
    return {"outputs": {"segmentation": output_dir / "seg.nii.gz"}}
```

Build the image, and `GET /tools` lists `MyTool`. That is all.

## What the signature buys you

The schema is generated from `run()` by `describe.py`, run with **your**
interpreter, so it cannot drift from the code.

| in `run()` | on the wire | in the client |
|---|---|---|
| `scans: Path` | `path` | file picker |
| `threshold: float = 0.5` | `float` | spin box at 0.5 |
| `merge: bool = False` | `bool` | check box |
| `mode: Literal["A", "B"]` | `choice` | combo box |
| `parts: list[Literal["a", "b"]]` | `multichoice` | check boxes |
| `output_dir: Path` | — | filled in by the server |
| `*, sup` | — | the supervisor, never published |

## Calling another tool

Declare `*, sup` — keyword-only and **unannotated** — and the runner hands you a
supervisor. Being unannotated is the marker: every other parameter must be
annotated, so nothing else has that shape, and a tool cannot grow one by
forgetting a type.

```python
def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
    landmarks = sup.run("ALI", input=scans, model=bundle, output_dir=sup.tmp / "ali")
```

Five members, nothing more: `sup.run(tool, **params)` (blocking, returns what
that tool's `run()` returned), `sup.out`, `sup.tmp`, `sup.progress(fraction,
message)`, `sup.log(message)`.

- **Never import a supervisor type.** It is duck-typed on purpose: a tool
  importing one would need a package shared with this repository, which is what
  the split removes. The same shape is produced by `sadt-tools`'
  `scripts/run_tool.py` and faked in its tests.
- `sup.run("ALI", ...)`, never `sup.ALI(...)`. A typo in a string is greppable;
  a typo in an attribute is an `AttributeError` an hour into a job.
- **Give the caller a way in.** Accept the dependency's output as an ordinary
  argument too (`landmarks: Path = ""`) and skip the call when it is supplied.
  That is what keeps the tool usable with no supervisor at all — `uv run`, a
  notebook, a deployment that has not installed the sibling.
- Default it to `None`. A tool that cannot run without one should say so itself,
  with a message naming the way forward.

Reach for it only when the ordering forbids plain chaining. Where one tool's
output is simply another's input, the caller runs both and passes a folder;
`ASO` needs `ALI` **mid-run**, after it has recentred the scans, which is why it
takes a supervisor and `ALI` does not.

Nested calls each get their own job directory under `<job>/sup/NN_<tool>/`, and
are capped at four deep. A nested run is a subprocess of its parent, so it never
queues behind the slot its parent already holds — but it is invisible to
`MAX_GPU_JOBS`, so a deployment running several supervised jobs at once has to
size for more than one tool on the card.

Types are limited to `path`, `str`, `int`, `float`, `bool`, `list[str]`.

## The conventions, so nothing needs configuring

`server/conventions.py` derives from your argument **names**:

| named | becomes |
|---|---|
| `model`, `*_model`, `*_reference` | a dropdown of `DATA/<tool>/models/`, **never an upload** |
| any other `Path` | may be filled from `DATA/<tool>/testfiles/`, or uploaded |
| `device`, `tile_step_size`, `num_workers`, `seed`, … (see `TECHNICAL`) | not rendered to a clinician |

So name your model argument `model` and your device argument `device`, and the
panel is right with nothing written down.

`DATA/` is found by name, underscores stripped: `Batch_Dental_Seg` reads
`DATA/BatchDentalSeg/`.

## Returning results

Return `{"outputs": {name: path}}` and write only into `output_dir`. The server
zips what you produced and streams it back. The names are what tool-to-tool
wiring will use, so give them meaning.

## Failing

Raise `ValueError` or `FileNotFoundError` for something the caller can fix —
the message reaches them as a `422`. Anything else is a `500` with a fixed
message, and the traceback stays in the server log.

## The exceptions

Only if a convention is wrong for you, add a section to
`server/deployment.toml` — the one file here a tool may ever need:

```toml
[tools.MyTool]
server_selectable = { atlas = "model" }   # hosted, but not named *_model
hidden = ["iterations"]                   # technical, not in TECHNICAL
data_dir = "MyToolData"                   # DATA/ folder named differently
max_upload_mb = 500
```

---

The tools still living in `server/tools/` are the old, in-process kind: a
`Tool` subclass the server imports. They are being repackaged, and nothing new
should be written that way — see `MIGRATING_A_TOOL.md`.
