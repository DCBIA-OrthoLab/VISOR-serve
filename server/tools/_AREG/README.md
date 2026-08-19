# AREG, parked

> **Superseded.** `AREG` is packaged in
> [sadt-tools](https://github.com/Jules-GP/sadt-tools) and served like every
> other tool; it drives `AMASSS`, `ASO`, `Crown_Seg` and `ALI_IOS` through the
> supervisor, which is what this copy was waiting for. Nothing here is on any
> critical path any more. It is kept only so `git log --follow` still reaches
> the work, and **deleting it is a decision, not a risk** — see
> `MIGRATING_A_TOOL.md`, step 7.

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

## What happened to it

It was ported into [sadt-tools](https://github.com/Jules-GP/sadt-tools), the way
the other six were, and the two things that made it the hardest of the seven
are both resolved:

- **It drives four tools** — `AMASSS`, `ASO`, `Crown_Seg`, and `ALI` by way of
  ASO. `src/tools_client.py` was the seam, and it is `sup.run(...)` now. The
  supervisor handles nesting (`AREG → ASO → ALI_CBCT`), refuses a cycle by
  name, and is capped at five deep.
- **pytorch3d**, the same source build `Crown_Seg` and `ALI_IOS` need, and the
  same reason its IOS half cannot be validated without a CUDA toolkit. It is an
  optional extra of the packaged tool, so the tool loads and only an IOS *run*
  answers 503.

What is genuinely **not** ported is the `IOSCBCT` mode — 829 lines upstream,
with no package, no catalog entry and no schema field on our side, and nothing
anywhere saying so. It needs the supervisor too: it orchestrates both chains.

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
