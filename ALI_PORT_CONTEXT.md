# ALI port — working context

> **Status, 2026-07-31: the port is done.** `tools/ALI/` and `tools/CrownSeg/`
> exist, are registered, and are tested. Read this file as the *record of why*
> — the analysis and the decisions — not as a plan. Where it and the code
> disagree, the code is right; the two places that happened are flagged inline
> (§3.1 on the `mode` argument, §6 on the open questions). The changelog entry
> in `CLAUDE.md` for 2026-07-31 is the summary of what actually shipped.

Handoff note for the port of **ALI** (Automatic Landmark Identification) from a
pair of Slicer CLI modules to a tool on this inference server. Written to be
read cold, by a session that has none of the preceding conversation.

Read alongside, in this order:

1. [`ADDING_A_TOOL.md`](ADDING_A_TOOL.md) — the contract every tool obeys. Non-negotiable.
2. `server/tools/AMASSS/` — the closest precedent. Same shape of problem (a
   Slicer CBCT CLI turned into a tool), same idioms. Copy its structure, not
   its content.
3. `SADT_tools_analysis/ALI.md` in the **client** repo (see below) — a
   line-referenced audit of the original ALI. It is in French; its numbered
   "Incohérences et pièges" section is the defect list this port must fix.

---

## 1. The two repositories

| Repo | Path | Role |
| --- | --- | --- |
| Server | `slicer-remote-tool-server/` | This one. FastAPI + tool registry. The port happens here. |
| Client | `SlicerAutomatedDentalToolsCloud/` | The Slicer extension. Holds the ORIGINAL ALI sources being ported, and the thin client modules. |

Original sources to port, all in the client repo:

```
ALI/ALI.py                          # Slicer widget: UI, landmark/teeth catalogs, mode switch
ALI/ALI_Method/{Method,CBCT,IOS}.py # per-mode orchestration, still Slicer-bound
ALI_CBCT/ALI_CBCT.py                # CBCT CLI entry point
ALI_CBCT/ALI_CBCT_utils/            # constants, io, preprocess, environment, agent, brain
ALI_IOS/ALI_IOS.py                  # IOS CLI entry point
ALI_IOS/ALI_IOS_utils/              # model, io, surface, agent, render, mask_renderer
```

---

## 2. What ALI does

One Slicer module, **two engines that share nothing** but the output format
(Slicer markups JSON):

- **CBCT** — one deep-RL agent per landmark navigates the volume at two
  spacings (1 mm then 0.3 mm) until it converges on the point. One `.pth`
  weight set per (landmark, scale). Dependencies: torch, monai, itk, SimpleITK.
- **IOS** — per tooth, multi-view offscreen rendering of the mesh + a 2D UNet
  predicting RGB masks that are reprojected onto the mesh faces. Requires the
  input mesh to already carry per-point tooth labels. Dependencies: torch,
  monai, **pytorch3d**, vtk.

---

## 3. Decisions already taken

### 3.1 One tool named `ALI`, both engines inside `src/`

```
server/tools/ALI/
├── __init__.py
├── ALI.py                  # schema only + delegation, no logic
├── src/
│   ├── __init__.py
│   ├── ALILogic.py         # mode dispatch, input discovery, run report
│   ├── cbct/               # ported from ALI_CBCT_utils/
│   └── ios/                # ported from ALI_IOS_utils/
└── test/
    └── test_ALI_logic.py
```

Rationale: keeps a single module in the Slicer UI, a single `DATA/ALI/` data
root, and mirrors the original module. Rejected alternatives: two tools
(`ALI_CBCT` + `ALI_IOS`) — cleaner schemas but splits one UI entry into two;
three tools — a dispatcher adding a name to `GET /tools` for zero logic.

**Accepted cost, do not try to work around it in the schema.** The schema
cannot express "this argument only exists in mode X". Consequences, all
deliberate:

- every mode-specific argument is `required=False`, otherwise the inactive
  mode would block the request;
- `run()` raises `ToolArgumentError` (→ 422) when the *active* mode's own
  selection is empty. This is the cross-argument rule §9 of the guide covers;
- ~~`mode` is an explicit `"choice"`, not inferred from the input extension: a
  `.zip` can hold either kind of data.~~

> **Superseded — there is no `mode` argument.** The observation is right and
> the conclusion inverted: because a `.zip` says nothing, and a DICOM series
> has no extension at all, *nothing in the request* can distinguish the two —
> only the data can. `ALILogic.detect` walks the input and picks the engine;
> an input holding both kinds is a 422 rather than a guess. Asking the caller
> would move the error from the door to several minutes into a GPU run. The
> Slicer client was already written against this contract (its `ALI_SCHEMA`
> fixture has no `mode`), and its 24 schema tests pass against what shipped.
>
> The teeth selection is also gone: the networks predict every tooth they are
> pointed at, the original's checkbox grid only filtered the *output*
> afterwards, and every tooth present in the mesh is what a landmarking run
> wants. Two selection groups ship, not three: `cbct_regions` and
> `ios_networks`.

The Slicer panel therefore always shows both checkbox groups, one of which is
inert at any time. Hiding one needs a dependency key in the schema plus
`formgen.py` support on the client — out of scope here, do not fake it with
prefixed option labels.

### 3.2 Crown segmentation runs server-side too, but from its own tool

ALI's IOS engine needs a mesh carrying a per-point tooth-label array
(`PredictedID`, `UniversalID`, or `Universal_ID`). The original Slicer module
produced it by calling the `dentalmodelseg` executable from Slicer's own
Python bin.

**`dentalmodelseg` is not something to port.** It is the console-script entry
point of the `shapeaxi` PyPI package:

```
[console_scripts]
dentalmodelseg = shapeaxi.dental_model_seg:cml
```

So server-side crown segmentation is `pip install shapeaxi` plus a call into
`shapeaxi.dental_model_seg` — a Python API, no subprocess, no Slicer binary.
The detour through Slicer's bin directory only existed because the caller was
inside Slicer.

**Decision: the goal is a single button with everything computed in the
cloud** — ALI, given a raw mesh, segments it itself. The one amendment to
"put the segmentation logic in `ALI/src/ios/`" is *where the file lives*:

- crown segmentation goes in **`tools/CrownSeg/src/`**, exposing something
  like `segment_crowns(...)`;
- **`ALI/src/ios/` imports it lazily** when input discovery finds an
  unsegmented mesh.

Rationale, in order of weight:

1. `dentalmodelseg` is called by **ALI, ASO, AREG and FlexReg**. If it lives
   inside ALI, those three must `from tools.ALI.src.ios... import`, i.e.
   depend on ALI. And ALI's IOS half needs pytorch3d — when that is missing,
   ALI fails to load and `registry.py` **skips** it, so the other three would
   import a module belonging to a tool the registry has just rejected. One
   absent dependency would take down four tools instead of one.
2. Crown segmentation alone is a legitimate user-facing operation
   (DOCShapeAXI and BATCHDENTALSEG do exactly that). Buried inside ALI it
   cannot be invoked on its own.
3. It matches the reuse pattern `ADDING_A_TOOL.md` documents for
   `AMASSSLogic.segment()`: the tool is the HTTP adapter, the `src/` function
   is the API other tools call.

This costs the one-button experience nothing.

Until `CrownSeg` exists, ALI's IOS mode refuses an unsegmented mesh with a
`ToolArgumentError` naming the expected arrays. Design the seam now: input
discovery returns "segmented" and "unsegmented" lists; the unsegmented branch
raises today and calls `CrownSeg` tomorrow. Nothing else changes.

### 3.3 One requirements file plus lazy imports — no per-tool files

There is **no per-tool requirements mechanism**: `requirements.txt` is the
single dependency list, installed by the `inference` service command in
`docker-compose.yml` at every container start. It includes the heavy DL stack
(`vtk`, `torch`, `nnunetv2`), unpinned on purpose: the image already ships
torch built against its CUDA version, so pip sees the entry satisfied and
leaves it untouched. (`requirements-amasss.txt` duplicated exactly those
three lines and has been removed.) `requirements-dev.txt` adds pytest/httpx
for the test suite. `ADDING_A_TOOL.md` §7 documents this layout.

ALI introduces no new file: monai/itk go into `requirements.txt`. pytorch3d
has no PyPI distribution at all and must be baked into the image (see §4) —
never added to `requirements.txt`.

The rule that makes this work, and that ALI must follow: **a heavy or optional
dependency is imported lazily, inside the function that uses it**, raising an
actionable message when absent. See `tools/AMASSS/src/nnunet_runner.py`
(`_import_torch`, `_import_predictor`, `_INSTALL_HINT`) for the exact pattern.
Consequence for ALI: CBCT must work on a server with no pytorch3d.

---

## 4. Measured facts about the deployment image

Probed directly (`docker run --rm --entrypoint python ghcr.io/jules-gp/lab-ai:2026.07`),
2026-07-28. Do not re-derive from memory, re-probe if in doubt.

- Python **3.11.10**, nvcc **CUDA 12.4** present.
- **torch 2.5.1+cu124** and **scipy 1.17.1** already installed. Never `pip
  install torch` — it shadows the CUDA build with a CPU-only wheel.
- Installable from PyPI: monai **1.6.0**, itk **5.4.6**, vtk **9.6.2**,
  nnunetv2 **2.8.1**, SimpleITK **2.5.5**, dicom2nifti **2.6.2**,
  shapeaxi **2.0.2**, ocnn **2.3.2**.
- **pytorch3d has no distribution on PyPI at all.** Source build only
  (feasible — nvcc is present — but a one-off image build of roughly half an
  hour). shapeaxi's own install notes confirm it: `pip install
  "git+https://github.com/facebookresearch/pytorch3d.git"`.
- **Every shapeaxi release wants a torch this image does not have**:
  1.0.10 (the version ALI's conda env pins) requires `torch==2.0.1` exactly,
  1.1.0 requires `>=2.7.0`, 2.0.2 requires `>=2.8,<2.13`. The image is on
  2.5.1. Downgrading torch is not an option — it would break AMASSS.

**Everything IOS-related therefore hangs on one ops task**: rebuild the base
image on **torch >= 2.8 with pytorch3d compiled in**. That single change
unlocks ALI's IOS mode, `CrownSeg`, and later the IOS modes of AREG, ASO and
FlexReg. Before bumping, confirm nnunetv2 2.8.1 (AMASSS) still works on
torch 2.8.

The user has explicitly authorised replacing outdated pinned libraries with
current ones. The original code pins monai 0.7.0 / 1.3.2 and pytorch3d 0.6.2;
target monai 1.6 and drop the version branches (see §5).

---

## 5. Defects the port must fix

Numbers refer to the "Incohérences et pièges" list in
`SADT_tools_analysis/ALI.md`. Items that are purely client-side UI (1, 5, 11,
12, 15) disappear by construction — the server validates before `run()` and
receives paths, not MRML nodes.

**Data-loss bugs — the important ones:**

- **(4) A single unknown landmark loses every output for that patient.**
  `environment.py:123` does `LABEL_GROUPS[landmark]` with no guard, inside a
  loop whose `KeyError` is caught far above at `ALI_CBCT.py:237`. One landmark
  absent from `LABELS` → no file written at all, including for the landmarks
  that were correctly predicted. Use `.get(landmark, "Other")`.
- **Silent collision in batch mode.** The patient key is `file.name`
  (`ALI_CBCT.py:95`), so two identically-named scans in different subfolders
  overwrite each other, in the working dict *and* in the flat output folder.
  Key by path relative to the input root, and preserve the input tree in the
  output. AMASSS solved the same class of problem — see its `discover_scans`.
- **(13) A missing mandibular IOS model raises a `KeyError` that is caught
  silently** (`ALI_IOS.py:335-337`); the jaw simply vanishes from the results.
  Report it in the run report, like AMASSS's `structures_without_model`.

**Vocabulary and schema:**

- **(3) `R`, `RIP`, `OIP`** appear in `SURFACE_LANDMARKS` but no model predicts
  them and `TYPE_LM` does not contain them. Do not publish them in `choices`.
- **(4, naming) Impacted-canine landmarks disagree**: the UI says
  `UR3OI/UL3OI/UR3RI/UL3RI`, the CLI expects `UR3OIP/UL3OIP/UR3RIP/UL3RIP`.
  One vocabulary, defined once server-side, published through `choices`.
- **(14) Network detection disagrees**: the UI requires `_O_`/`_C_` in the
  `.pth` name, the CLI takes `basename.split("_")[1]`. Pick one rule and
  document it in the `model` argument's description.
- **(2) `.stl` is accepted then silently ignored** by the IOS CLI, which only
  discovers `.vtk`. Either read it (VTK can) or reject it in the schema. Not
  both.
- **(6) `SaveId` and `GroupInFolderCheckBox` are read by nothing**; the `Pred`
  suffix is hardcoded in both CLIs. Make `prediction_ID` a real argument, as
  AMASSS did.
- **(16) Output extensions differ between modes**: CBCT writes `.mrk.json`,
  IOS writes `.json` for the same markups content. Uniform `.mrk.json`.

**Filesystem discipline (the server forbids all of these anyway):**

- **(8)** DICOM conversion writes `<input>/NIFTI/` — into the user's own data,
  which a later run then re-discovers as input scans.
- **(9)** `create_csv` writes into the extension's own source folder.
- **(7)** DICOM mode is half-implemented (`NumberScanDCM` and friends return
  `None`, which then divides by `None` at the end of the run). Decide
  explicitly: implement it server-side from a zip of DICOM folders, or leave
  it out of the schema. Do not port it half-way.

**Latent code bugs spotted while reading (not in the analysis doc):**

- `ALI_CBCT_utils/agent.py:193` — `if new_pos.all() > 0 and ...`. `.all()`
  returns a single boolean, so this tests "no component is zero", not "all
  components are positive". Negative coordinates pass the bounds check.
- `environment.py:59-65` and `ALI_CBCT.py` branch on `sys.version_info >=
  (3,10)` to choose `EnsureChannelFirst` vs the removed `AddChannel`. On monai
  1.6 only `EnsureChannelFirst` exists — delete the branch.
- `brain.py:177` calls `torch.load(...)` without `weights_only`. These are
  plain state dicts; pass `weights_only=True` explicitly (torch ≥ 2.6 changes
  the default, and the files come from a data store).
- Verify the import path `monai.networks.nets.densenet.DenseNet` still
  resolves on monai 1.6 before assuming it does.
- `ALI_SEARCH_MAX_TIME` (agent search budget, `agent.py:266-267`) is already a
  local patch in the client repo — keep the idea, but drive it from the tool
  schema or `config.py` rather than an env var read deep in a loop.

---

## 6. Open questions

1. **When is the base image rebuilt on torch >= 2.8 + pytorch3d?** Nothing
   IOS-related can run before that. Check nnunetv2/AMASSS against torch 2.8 in
   the same pass. **Still open, and still the critical path** — it is now the
   *only* thing between the shipped code and a working IOS mode: `tools/ALI/`'s
   IOS engine and `tools/CrownSeg/` are written, registered and tested, and
   fail an actual run with a message naming what is missing. No code change is
   needed when the image lands.
2. ~~**DICOM input**: in or out of the schema.~~ **In.** A DICOM series is
   recognised inside the archive by asking GDCM (slices have no extension, so
   nothing else can), converted into the request's scratch directory, and
   processed like any other CBCT scan. Only directories holding no volume or
   surface file of their own are probed, so a cohort of NIfTI costs nothing.
3. ~~**Where do model bundles live and how are they named?**~~ **Answered, and
   the manifest matches the code.** `DATA/ALI/models/ALI_CBCT_Models/` holds
   the eight region archives unpacked side by side —
   `discover_weights` walks recursively for `<landmark>/<scale>/*.pth`, so
   they form one bundle and one dropdown entry.
   `DATA/ALI/models/ALI_IOS_Models/` holds `Upper_O_model.pth`,
   `Lower_O_model.pth`, `Upper_C_model.pth`, `Lower_C_model.pth` — verified
   against the published archive, not guessed. The two bundle names are what
   tells them apart in the shared dropdown. CrownSeg's checkpoint is its own
   tool's data (`DATA/CrownSeg/models/`, named by `settings.CROWNSEG_MODEL`),
   so ALI never reaches into another tool's folder.

---

## 7. Hard constraints, restated

From `ADDING_A_TOOL.md` and `CLAUDE.md`, the ones easiest to violate here:

- **Nothing outside `tools/ALI/` may be edited**, except a new `FILE_TYPES`
  entry in `base.py` if one is genuinely needed. No route, no registry entry.
- Core imports absolute (`from base import ...`), own imports relative
  (`from .src...`).
- `run()` writes only inside the request work dir or a
  `file_utils.make_scratch_dir()` folder. `DATA_DIR` is read-only.
- **Never log file contents, argument values, or patient metadata.** This
  server processes confidential medical data.
- Never `except: pass`. Let exceptions propagate; `main.py` logs the traceback
  and returns a generic 500.
- Read `settings.DEVICE` rather than deciding on `torch.cuda.is_available()`
  deep in the code — the original does the latter in five different modules.
- Unit tests under `tools/ALI/test/`, synthetic data, no HTTP.

## 8. Status and sequencing

1. **ALI, CBCT engine** — **done and runnable.** `monai` and `itk` are in
   `requirements.txt`.
2. **Base image rebuild** — torch >= 2.8, pytorch3d compiled in. **Still
   open**, ops task, and now the only blocker for 3 and 4 below. Re-check
   AMASSS/nnunetv2 against torch 2.8 in the same pass.
3. **`tools/CrownSeg/`** — **written and tested**, executes once the image
   lands.
4. **ALI, IOS engine** — **written and tested**, same. It segments an
   unlabelled mesh through `CrownSegLogic.segment_crowns()` and goes straight
   on to landmark identification: one button, all computation server-side, no
   second request.

Neither 3 nor 4 needs a code change when the image is rebuilt — both import
their missing stack lazily, so the day it is present they simply start
working. `pip install -r requirements.txt` will not bring it: `pytorch3d` has
no PyPI distribution and `shapeaxi` requires a torch the image does not ship,
which is why neither is listed there.
