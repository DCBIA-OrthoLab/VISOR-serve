# AREG, parked

This is the AREG module as it stood on the `AREG` branch, merged here so the
work lives in `main` rather than in a branch on one laptop. **It does not run,
and it is not meant to.**

The leading underscore keeps it out of the registry — `server/registry/`
skips folders starting with `_`, the same way it skips `_dispatch_probe`. Without
it the registry would try to import AREG on every start, fail on `SimpleITK`, and
fail a test on every run.

## Why it does not run

Two independent reasons:

- **Its dependencies are gone.** `requirements.txt` shed torch, monai, pytorch3d,
  SimpleITK and VTK when the tools were packaged. Nothing in the server's venv
  can import this any more.
- **It is written against an architecture that no longer exists.** `Tool` /
  `ArgSpec` schemas, `registry.TOOLS` for calling other tools, and
  `config.settings` for its knobs. The tools that survive are packages with a
  single `run()`, described by `scripts/describe.py`, and a tool that needs
  another one receives a supervisor.

## What to do with it

Port it into [sadt-tools](https://github.com/Jules-GP/sadt-tools), the way the
other six were. Two things make it the hardest of the seven:

- **It drives four tools** — `AMASSS`, `ASO`, `CrownSeg`, and `ALI` by way of
  ASO. `src/tools_client.py` is the seam to replace with `sup.run(...)`. The
  supervisor handles nesting (`AREG → ASO → ALI`) and is capped at four deep.
- **pytorch3d**, so the same source build `Crown_Seg` and `ALI` need, and the
  same reason its IOS half cannot be validated without a CUDA toolkit.

`src/cbct/elastix.py` is worth reading first — it is the part with no equivalent
anywhere else in the family.

## One more thing on this branch

The `AREG` branch also carried **+512 lines of ALI**, adding a third landmark
network: **Mucogingival**, one point per lower tooth, trained on the mandible
only. Those changes were NOT kept here — `server/tools/ALI/` is deleted and its
package in sadt-tools is where the network belongs. The weights are already
staged: `DATA/ALI/models/ALI_IOS_Models/Lower_MG_v6.pth`, which the packaged ALI
currently reports as unrecognised. Recover the diff from the archive tag:

    git diff archive/AREG~2 archive/AREG -- server/tools/ALI
