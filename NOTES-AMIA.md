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
