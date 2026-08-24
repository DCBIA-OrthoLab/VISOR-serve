# NOTES-AMIA — working file, delete after submission

Consolidation before the AMIA submission of 2026-09-03. One section per
chantier. Everything here is a fact obtained by running something; the paper is
updated from this file, not from memory.

Frozen commits the paper cites, all three verified reachable on 2026-08-24:

| Repository | Commit | Where it is |
|---|---|---|
| `sadt-tools` | `483429fa` | tip of `main` |
| `slicer-remote-tool-server` | `7858d147` | on `origin/main`, 3 commits back from its tip |
| `SlicerAutomatedDentalToolsCloud` | `cfe60d68` | reachable |

No history was rewritten. Nothing was tagged.

---

## Chantier 1 — ASO: the numerical diff, three of its four modes

### Result

**ASO's parity cell moves from "runs end to end, not diffed numerically" to
"bit-identical against the pre-port implementation in three of its four
modes".** The fourth (fully-automated CBCT) is untested and needs the landmark
tool built; see *What is still open*.

| Mode | Artifacts identical | Differing | Verdict |
|---|---|---|---|
| CBCT Semi-Automated | 3 | 1 (`ASO_report.json`) | clinical output **bit-identical** |
| IOS Semi-Automated | 4 | 0 | **bit-identical**, report included |
| IOS Fully-Automated | 3 | 0 | **bit-identical**, report included |
| CBCT Fully-Automated | — | — | not run |

The one difference is not numerical. `ASO_report.json` gained a single key
between the two implementations:

```
> "landmark_source": "alongside the scans"
```

717 bytes before, 761 after. It records where the landmarks came from; the
packaged tool says it, the pre-port one did not. No other byte of any output
differs.

### The exact figures for CBCT Semi-Automated

Real patient CBCT, real ALI-predicted landmarks, published gold reference.
sha256 of what each side wrote, `imported` being the pre-port implementation
and `packaged` the tool as it ships:

| Artifact | Bytes | sha256 (both sides) |
|---|---|---|
| `P_0001_T2_Or.nii.gz` (the oriented scan) | 209,222,332 | `684dddbd00e2b91a095b31b0b08e6d12e92f0d41301dab4780e7c1be60d782ab` |
| `P_0001_T2_Or_transform.tfm` | 1,020 | `8a90194f058aa0e179a9e5864a77ddb9c7eac12a134387286268f861496d9b30` |
| `P_0001_T2_lm_Or.mrk.json` | 6,895 | `d841d8c5fcf6ee9d7c3be12f7269b5e3237182fa2488664a4ef2f4eaa02a9a37` |

**The two runs did not share a dependency stack**, which is what makes the
figure worth quoting:

| | Python | numpy | SimpleITK | vtk |
|---|---|---|---|---|
| pre-port (server venv) | 3.10.12 | 2.2.6 | 2.5.5 | 9.6.2 |
| packaged (`tools/ASO/.venv`) | 3.11.15 | 2.3.2 | 2.5.6 | 9.6.2 |

Different interpreter, different numpy minor, different SimpleITK patch — and a
209 MB resampled volume identical to the byte. That is the isolation claim
holding on real data rather than being asserted.

### Data used

Nothing was downloaded and nothing was generated; all of it was already staged
under `DATA/`.

| Mode | Input | Reference |
|---|---|---|
| CBCT Semi-Auto | `DATA/AREG/testfiles/IOSCBCT_RegTestFiles/T2/P_0001_T2.nii.gz` + `CBCT Landmarks/P_0001_T2_lm_Pred_U.mrk.json` | `DATA/ASO/models/CBCT_Gold_Occlusal_Midsagittal_Plane` |
| IOS Semi-Auto | `.../IOSCBCT_RegTestFiles/T1/P001_T2_U.vtk` + `IOS Landmarks/P001_T2_U_Seg_Upper_O_Pred.json` | `DATA/ASO/models/IOS_Gold_file` |
| IOS Fully-Auto | `DATA/ALI/testfiles/T1_01_U_segmented.vtk` | `DATA/ASO/models/IOS_Gold_file` |

`DATA/ASO/` holds models only — no `testfiles/` — which is why the inputs come
from other tools' bundles. The CBCT case registers on `UL6O, UR1O, UR6O`, the
three landmarks the patient file and the occlusal gold have in common.

### Commits

| Repo | SHA | What |
|---|---|---|
| `slicer-remote-tool-server` | `4f5a134` | `FIX: resolve the imported tool's folder from server/, not from parity.py's own directory` |

Merged into `main` (fast-forward, not pushed). No commit in `sadt-tools`: the
tool itself needed no change — this chantier measured it, it did not alter it.

### Two defects found in the parity harness itself

Both would have stopped anyone re-running the comparison, and neither was
covered by a test.

1. **`parity.py` could not find the side it compares against.** It resolved
   `server/tools/<name>` against its own directory, which stopped being
   `server/` the day the file moved into `execution/`: every lookup landed in
   `server/execution/tools/` and found nothing. Fixed in `4f5a134`, with two
   tests (`test_a_bare_name_resolves_under_server_tools`,
   `test_a_path_is_taken_as_given`).
2. **Its usage line is stale.** The docstring still says `python parity.py`,
   which now fails on `ModuleNotFoundError: file_utils`. It has to be run as a
   module from `server/`: `venv/bin/python -m execution.parity`. Left alone
   rather than edited, since the recipe below supersedes it.

### How to reproduce

The in-process ASO was deleted in `5029781` once its package replaced it, so
the side being compared against no longer exists in the tree. Restore it, run,
remove it — never commit it, or the registry will serve two tools of one name:

```bash
cd slicer-remote-tool-server
git archive '5029781^' server/tools/ASO | tar -x -C .

cd server
PYTHONPATH=$PWD API_TOKEN=parity-token \
TOOLS_DIR=~/code/sadt-tools/tools \
DESCRIBE_PATH=~/code/sadt-tools/scripts/describe.py \
DATA_DIR=../DATA SCHEMA_CACHE_DIR=../.schema-cache DEVICE=cpu \
venv/bin/python <wrapper>.py --imported ASO --packaged ASO \
    --args imported.json --packaged-args packaged.json

cd .. && rm -rf server/tools/ASO
```

`<wrapper>.py` is three lines around `execution.parity.main`: the pre-port ASO
reads `settings.ASO_ICP_MAX_TRIPLETS` and `settings.ASO_ICP_SEED`, which went
with the tool in `5029781`, so they are put back on the settings object at the
values `config.py` carried at `5029781^` — 2500 and 0, the same numbers the
packaged tool now takes as arguments. Restoring the tool means restoring the
environment it ran in; nothing in the algorithm is touched. The scripts and the
argument files are in this session's scratch directory, not in the repository.

The two sides need different argument files: `cbct_landmarks` is a multichoice
on one side (`"UL6O,UR1O,UR6O"`) and a `list[str]` on the other, which is what
`--packaged-args` exists for.

---

## Dependency pins

**None changed.** No lockfile, no `pyproject.toml`, no version was touched.

---

## What is still open

| Cell | State | What it would take |
|---|---|---|
| `ASO` CBCT Fully-Automated | not diffed | `cd tools/ALI && uv sync` (the bundle is on disk, 14 GB) plus a card. This mode calls the landmark tool through the supervisor and then runs the same ICP the semi-automated mode does, which is now bit-identical — so it is the supervised call, not the geometry, that remains unproven. |
| `ALI_IOS` | blocked | unchanged: pytorch3d needs a CUDA toolkit. Provisioning, not code. |
| `AREG_IOS` | not validated | unchanged, and the heaviest of the three. |

`tools/ALI` and `tools/AREG` have **no `.venv` on this machine**; the other six
tools do. Anything needing them starts with a `uv sync`.

---

## Things noticed that the paper should not trip over

- **No closed cell was found to be open.** The six the paper declares closed
  were not re-run — the point of this chantier was the open one — but nothing
  encountered contradicts them.
- **`main` cannot currently serve `FlexReg`.** Its schema declares a `vec2`
  argument type the registry on `main` does not know, so it is skipped at
  startup with `1 TOOL(S) FAILED TO LOAD` and every other tool loads. This is
  the designed behaviour for a tool that will not load, not a regression.
  `FlexReg` is **not one of the nine** the paper counts: it is absent from
  `sadt-tools` `main` (`483429fa` ships seven tool directories plus the
  template) and lives on `feat/flexreg`, with the server-side `vec2` support on
  `feat/vec2`. Neither is merged.
- Consequently `server/tests/test_tool_contract.py::test_a_really_packaged_tool_loads[FlexReg]`
  fails on this machine. It is parameterised over whatever `TOOLS_DIR` holds,
  and this checkout has `sadt-tools` on `feat/flexreg`. Against `main` the
  parameter does not exist. **316 passed, 1 failed, 2 skipped**, and the failure
  reproduces with every change of this chantier stashed.
- **`sadt-tools` has uncommitted work in the tree** (`tools/FlexReg/`, on
  `feat/flexreg`) from an earlier session. Unrelated to AMIA, but it is sitting
  there.

---

## Chantier 2 — AREG_IOSCBCT: the cell does not close, and the reason is not the environment

### Result

**`AREG_IOSCBCT` Registration mode does not reproduce the pre-port
implementation, and the gap is a missing pipeline step rather than version
drift.** The packaged tool stops after the landmark alignment; the pre-port one
refines that alignment with an ICP against a surface it extracts from the CBCT.
The registered upper arch differs by up to **2.68 mm**, the lower by **2.20 mm**.

This is a defect found, not a cell closed. It is the opposite of the ASO
result and should be read that way.

| Mode | Verdict |
|---|---|
| Registration | **differs**, 2.68 mm upper / 2.20 mm lower, cause attributed below |
| Semi-Automated | not run: needs Crown_Seg and ALI_IOS bundles |
| Fully-Automated | not run: additionally needs ASO and a reference |

### The cause, and the proof that it is the cause

`pipeline.register_one` runs the ICP only when it is handed `cbct_points`:

```python
if cbct_points is not None and len(cbct_points) >= 3:
```

It has exactly one caller, `dispatch.py:153`, and that caller never passes
them. **The ICP is therefore unreachable in all three modes**, not only in
Registration. The CBCT is opened for pairing and never sampled. Upstream, by
contrast, derives the surface from the volume itself
(`vol.contour(isosurfaces=[400])` in `load_data`) and calls the ICP with it for
both arches.

The attribution was tested rather than assumed. Taking the packaged output,
rebuilding upstream's CBCT surface exactly as `load_data` does, and replaying
the packaged tool's **own** ICP on it:

| Arch | Packaged vs pre-port | After replaying the missing ICP |
|---|---|---|
| Upper | 2.6838 mm | **0.0008 mm** |
| Lower | 2.1957 mm | **0.0006 mm** |

The lower arch replay returns `rmse 0.3930244, fitness 0.8891596`. The pre-port
run logged `Fitness: 0.8892, Inlier RMSE: 0.3931`. Same estimator, same
convergence, to four figures.

Two things follow. The port's geometry is numerically equivalent to upstream's,
including the ICP it never calls, and the residual 0.0008 mm is the float32
quantisation of upstream's mesh writer, not disagreement. **What is missing is
the call, not the arithmetic.**

### A second, smaller divergence

`geometry.DEFAULT_MAX_DIST = 1.5`, documented in `run()` as "0 uses 1.5, which
is upstream's". 1.5 is the default of upstream's *signature*; the value
upstream actually passes at both call sites (`AREG_IOSCBCT.py:420,425`) is
**1.0**. Nothing depends on this today, since the ICP does not run, but it
would the moment it did.

### Artifacts

Neither side writes the same set, so no sha256 pair is comparable directly.
The meshes are the common output and were compared numerically.

| Side | Artifact | Bytes | sha256 |
|---|---|---|---|
| pre-port | `P1_T2_Reg_U.vtk` | 10,589,056 | `f4f272529166802af98108db920cb8b39e9d07f368e4139610b18409c80248d1` |
| pre-port | `P1_T2_Reg_L.vtk` | 6,570,416 | `7941aeec7860a49833f89f94b96806dd36f2e7a7eed016de8418fb53b9d051a8` |
| pre-port | `P1_T2_lm_Reg_U.mrk.json` | 6,904 | `fdf16d84108f1dad523869e0cb6d2ec242f37bbf374bbd688eebd9db75841707` |
| pre-port | `P1_T2_lm_Reg_L.mrk.json` | 6,909 | `b84cceb993ce8b9bcfdb242c504bf653b9a9a5ea2fc67a888734ebd709405efd` |
| packaged | `1/P001_T2_U_Reg.vtk` | 11,981,285 | `50146d31b00eb7f0196fa1e8de0197ed13417a2c123deb1dae7d887e7a96ef6a` |
| packaged | `1/P001_T2_L_Reg.vtk` | 7,525,737 | `c4cf72bf482004dba51cd51a1f58059ffeb1cf462fbcdd564e5ca41ea0850182` |
| packaged | `1/P001_T2_U_Reg_matrix.npy` | 256 | `8af804874106ebc5cbc165b17743dafa357dabad50ba67200a2cf94d028bceb6` |
| packaged | `1/P001_T2_L_Reg_matrix.npy` | 256 | `2aef2e2ad09acbb77db1ed461e94a51c432aa2a8d4612e366fc8a6768f24269d` |
| packaged | `AREG_report.json` | 902 | `b49099b08f26bba340dfdc0713cf8605eba26d4a6e57dee5e8c616677ebf92a2` |

Three differences of kind, none of them numerical: the packaged tool writes the
4x4 matrix and a report and no registered landmarks; the pre-port one writes
registered landmarks and neither matrix nor report; and the two name their
outputs differently (`P1_T2_Reg_U` against `1/P001_T2_U_Reg`). Point counts and
face arrays are identical on both arches (116,019 and 79,610 points), so the
meshes are the same surface under different transforms. The pre-port writer
emits points as **float32**, the packaged one as **float64**, which is most of
the size difference.

### The two dependency stacks

The pre-port CLI does not exist as a server `Tool`, so there was no server venv
carrying its dependencies. It was run in a throwaway interpreter built for this
comparison, deliberately older than the packaged one on every axis:

| | Python | numpy | SimpleITK | vtk | pyvista | scipy |
|---|---|---|---|---|---|---|
| pre-port (throwaway venv) | 3.10.12 | 2.2.6 | 2.5.5 | 9.4.2 | 0.45.3 | 1.15.3 |
| packaged (`AREG_IOSCBCT/.venv`) | 3.11.15 | 2.3.2 | 2.5.6 | 9.6.2 | 0.48.4 | 1.16.2 |

Six axes apart, and **the environment explains none of the difference**: the
replay above reproduces the pre-port result to 0.0008 mm across that same gap.
The vtk split is not arbitrary. pyvista 0.45 pins `vtk<9.5`, which is why the
older stack cannot carry vtk 9.6.2, and is the same constraint the tool's
`pyproject.toml` documents for 0.44.

### Data used

Nothing downloaded, nothing generated; one patient, already staged.

| Role | Path (under `DATA/AREG/testfiles/IOSCBCT_RegTestFiles/`) |
|---|---|
| IOS meshes | `T1/P001_T2_U.vtk`, `T1/P001_T2_L.vtk` |
| CBCT | `T2/P_0001_T2.nii.gz` |
| IOS landmarks | `IOS Landmarks/P001_T2_{U,L}_Seg_{Upper,Lower}_O_Pred.json` |
| CBCT landmarks | `CBCT Landmarks/P_0001_T2_lm_Pred_{U,L}.mrk.json` |

Six landmarks per arch were shared and used, none dropped (`UL1O UL3O UL6O UR1O
UR3O UR6O`, and the L equivalents).

### Why the parity harness was not used

`execution/parity.py` compares a `Tool` this server imports against the
packaged folder. **AREG_IOSCBCT never existed as a server tool.** The in-process
`_AREG` parked at `37f009c` and deleted at `ab24e34` registers two timepoints of
one modality onto each other, CBCT or IOS; it has no IOS-to-CBCT mode, and no
file in the server's history mentions one. The pre-port implementation is the
upstream Slicer CLI, `SlicerAutomatedDentalTools/AREG_IOSCBCT/AREG_IOSCBCT.py`,
which the harness has no way to invoke. Its comparison primitives are the right
ones and were reused in spirit: artifacts hashed by name, absolute paths never
compared. Nothing was written to replace it.

### Reproducing

```bash
# pre-port side
uv venv --python 3.10 preport-venv
VIRTUAL_ENV=$PWD/preport-venv uv pip install \
    numpy==2.2.6 SimpleITK==2.5.5 pyvista==0.45.3 scipy==1.15.3
D=DATA/AREG/testfiles/IOSCBCT_RegTestFiles
preport-venv/bin/python \
    ~/code/SlicerAutomatedDentalTools/AREG_IOSCBCT/AREG_IOSCBCT.py \
    "$D/T1" "$D/T2" "$D/IOS Landmarks" "$D/CBCT Landmarks" preport_out

# packaged side
cd ~/code/sadt-tools/tools/AREG/AREG_IOSCBCT
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); \
from sadt_areg_ioscbct import run; \
run(ios='$D/T1', cbct='$D/T2', output_dir='packaged_out', \
    automation='Registration', ios_landmarks='$D/IOS Landmarks', \
    cbct_landmarks='$D/CBCT Landmarks')"
```

`scipy` is imported inside `run_icp_point_to_plane`, so the pre-port side fails
late and per patient without it, logging `No module named 'scipy'` and
continuing. Everything else it needs is imported at module scope.

### What is still open

| Cell | State | What it would take |
|---|---|---|
| `AREG_IOSCBCT` Registration | **differs, cause known** | a decision on the missing ICP call, then a re-run. This is a code question, not a measurement one. |
| `AREG_IOSCBCT` Semi-Automated | not run | `Crown_Seg` and `ALI_IOS` bundles; `ALI_IOS` is the cell blocked on a CUDA toolkit for pytorch3d |
| `AREG_IOSCBCT` Fully-Automated | not run | the above plus `ASO` and an orientation reference |

### Dependency pins

**None changed.** No lockfile, no `pyproject.toml`, no pin, in either
repository. The pre-port interpreter was built outside both trees and is not
committed.

### Things the paper should not trip over

- **No closed cell was disturbed.** Nothing in either repository was modified;
  `git status` in both is exactly what it was before this chantier, carrying
  only the FlexReg work and the skill edit that were already there.
- `AREG_IOSCBCT` ships **one** test, `test_every_tool_is_named_by_string`.
  Nothing exercises the registration pipeline, and no test mentions the ICP,
  which is how a step could become unreachable without anything going red.
- The tool is correct in what it does do. The landmark alignment matches
  upstream's exactly, and the ICP code it carries matches upstream's estimator
  to four figures. **It is a wiring gap, not a numerical one**, which is the
  distinction the paper draws between an attributed difference and drift.
