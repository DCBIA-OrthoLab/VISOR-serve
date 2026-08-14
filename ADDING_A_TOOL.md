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
