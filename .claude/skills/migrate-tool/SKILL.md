---
name: migrate-tool
description: Migrate a Slicer module into a packaged SADT tool, or split an existing tool whose two engines have incompatible dependencies. Use when porting a tool from SlicerAutomatedDentalTools, when a tool needs its own pinned torch, or when adding a tool to sadt-tools.
---

# Migrating a tool

A tool is a folder with its own virtualenv. The server never imports it: it
reads the tool's schema, validates the request, and runs the tool's own Python
as a subprocess. Everything below follows from that.

## 1. Read upstream before writing anything

Open the upstream module and **enumerate** what it declares. Do not filter with
a regex you wrote from memory: parameter names mix case (`DCMInput`,
`lm_type`), and a pattern like `[a-z_]+` silently drops half of them.

```bash
# method classes, when there are any
grep -h "^class " <upstream>/<Tool>_Method/*.py

# CLI parameters, when it is a CLI module
grep -oE "<name>[^<]+</name>" <upstream>/<Tool>_CLI/<Tool>_CLI.xml
```

Set that list against what the port exposes and sort every difference into three
piles: ported, deliberately dropped with the reason written down, and missing by
oversight. Only the third is work. The first two exist to isolate it.

## 2. One tool, or several?

Split when the engines have **incompatible dependencies**, not when the code
looks separable. The test is concrete:

```bash
grep -rhoE "^\s*(import|from) (torch|pytorch3d|monai|itk|vtk)[a-z0-9_.]*" <engine>/*.py \
  | awk '{print $2}' | cut -d. -f1 | sort -u
```

If one engine needs pytorch3d (and therefore an exact torch) and the other does
not, a shared virtualenv forces one pin onto the other. That has been measured
to move results: bumping ALI_CBCT to its sibling's torch shifted a landmark by
4 voxels and dropped two others entirely.

Split into `tools/<Family>/<Family>_<ENGINE>/`, with `tools/<Family>/` holding
no `pyproject.toml` of its own.

## 3. What goes in a shared `common/`

**Duplicate the implementation, share the formats.** Two copies of an algorithm
cost a divergence; a coupling costs an entire class of failure, and the copy
usually wins. The exception is anything defining the shape of bytes leaving the
repository, or a convention both engines must agree on:

- output file formats (a Slicer `.mrk.json` writer);
- input vocabularies (which extensions are volumes, which are surfaces);
- identity derivation (how a patient key is read from a filename).

A shared package declares **`dependencies = []`** and must keep doing so: it
installs into environments whose pins are deliberately incompatible.

Never share anything containing `sup.run()`. `describe.py` derives the schema's
`calls` field by reading each tool's own `src/`, and the server refuses to start
on a call naming an unserved tool. Shared orchestration is invisible to both.

## 4. Write `run()`

One public callable. Annotations are stdlib only: `Path`, `str`, `int`, `float`,
`bool`, `Literal[...]`, `list[...]` of those. No `Optional`, no unions. No
default means required.

Heavy imports go **inside** `run()` or the engine, never at module level: CI
imports the package on every PR to publish the schema, and that must not cost a
CUDA stack.

If the tool needs another tool, declare `*, sup=None` -- keyword-only and
unannotated, which is what keeps it out of the published schema.

## 5. Declare it

```toml
[tool.sadt]
tool = true
name = "Crown_Seg"
```

The section is what makes the directory a tool. The name is the API identity:
what a client sends, what `deployment.toml` is keyed by, what `sup.run()` names.
The directory must still be named after the tool, because the interpreter is
looked up by name.

Pin sources explicitly, never as extra index URLs:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
```

**A `[tool.uv.sources]` entry only applies to a DECLARED dependency.** Left
transitive it is ignored, and the package comes from PyPI built against a
different torch: it imports fine and dies on the first CUDA kernel.

## 6. Verify, in this order

Each step catches what the previous one cannot.

```bash
# 1. it resolves
cd tools/<Tool> && uv lock

# 2. it installs and imports
uv sync && .venv/bin/python -c "import sadt_<tool>"

# 3. no name is undefined on a path only a run reaches
uv run --no-project --with pyflakes -- python -m pyflakes src

# 4. the schema generates, with the tool's own interpreter
.venv/bin/python ../../scripts/describe.py .

# 5. it runs, on real data, and the result is compared
```

Step 3 is not optional. An import proves the module loads and says nothing about
a name inside a branch only a real run reaches -- four such defects survived
import AND schema generation in one split.

Step 5 is what "done" means. Run it directly in its venv, then through
`POST /run/{tool}`, and diff the two. State what matched exactly and what varied
within a measured tolerance. If the tool is non-deterministic, measure its
spread by running it twice on the same input: the reference is that spread, not
zero.

## 7. After a split, check the callers

A rename breaks every `sup.run()` naming the old tool, and the failure is at run
time.

```bash
grep -rn "sup.run(" tools/*/src tools/*/*/src
```

Also check `deployment.toml`: a tool's data folder is looked up by name, so a
split tool no longer matches `DATA/<old name>/` and needs `data_dir`.

## Commits

One sentence saying what the commit does, prefixed `ADD:`, `FIX:`, `DEL:`,
`UPDATE:` or `CLEAN:`. Several commits rather than one large one. No AI
attribution trailers.
