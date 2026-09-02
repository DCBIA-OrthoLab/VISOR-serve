# Packaging ALI and ASO

Written for whoever works on
[SADT-VISOR](https://github.com/DCBIA-OrthoLab/SADT-VISOR). It is the mirror of your
`What the server has to do`: that one told this side what stopped happening in
the tools, this one tells your side what the server will and will not do for
the last two.

They are the last two, and that matters: `server/tools/` cannot be deleted
until they leave it, and with it goes the in-process path, the
`SADT_DISPATCH_MODE` flag, and every tool dependency in
`server/requirements.txt`. The server then imports nothing heavier than
fastapi, which is the whole point of the split.

Read the two **decisions** first. They are not work, and getting them wrong is
not visible in any test - it is visible in a clinician's panel.

## Where things stand (2026-08-13)

Four tools are packaged and were run end to end this morning through the
packaged path, on real data, on an RTX 6000 Ada:

| tool | input | time | result |
|---|---|---|---|
| `AMASSS` | hosted scan + `AMASSS_Models`, `structures=MAND` | 21.8 s | mandible segmented, `device: cuda` |
| `Batch_Dental_Seg` | **86 MB uploaded** CBCT + `PediatricDentalSeg` | 71 s | 5 segments, label table in the report |
| `Surg_Mov_Pred` | `TestFiles` + `all_models` | 8.6 s | predictions .xlsx + .csv |
| `Crown_Seg` | hosted mesh, already labelled | 0.95 s | passed through untouched |

One caveat on the last row, worth knowing because it is the live state rather
than a defect: that venv was built with a plain `uv sync`, so the
`segmentation` extra is absent and **the engine was never touched**. An
unlabelled mesh answers 503 today, exactly as your own document says it
should.

`tools/ALI/` and `tools/ASO/` hold raw copies of the in-process tools - 
`ALI.py`, `src/ALI_CBCT/`, `src/ALILogic.py` - with no `pyproject.toml`, no
`uv.lock` and no venv. Nothing there can be synced, described or run.

## The blocker both of them share: they call another tool

Two chains, and both are in code today:

```
Crown_Seg → ALI    ALILogic.ensure_segmented()
                   → from tools.CrownSeg.src import CrownSegLogic

ALI       → ASO    ali_client.predict_landmarks()
                   → from registry import TOOLS; TOOLS["ALI"].invoke(...)
```

They are in-process calls rather than HTTP for a reason worth keeping in mind:
a tool run holds one of `MAX_CONCURRENT_TOOLS` for its whole duration, so an
ASO run calling the server's own `/run/ALI` would wait for a slot the outer
run is holding. Four concurrent ASO requests would deadlock the server,
`/health` included.

A packaged tool can do neither. Your document already states the target - the
server chains them, the handoff is files - and that is the right answer. **The
sequencing does not exist in the server yet**, so the work splits:

**Your half - and it unblocks packaging immediately.** Package both taking the
upstream artifact as an ordinary `path` argument, and delete the call:

- **ALI**: `meshes: Path`, documented as *already carrying tooth labels*. Drop
  `ensure_segmented()`. Handed an unlabelled mesh, raise `ToolInputError`
  naming Crown_Seg - the server turns that into a 422 with your message.
- **ASO**: `landmarks: Path`, a directory of `.mrk.json`. Drop
  `ali_client.py` and the `ASO_LANDMARK_TOOL` setting with it.

Both are then self-contained, packageable today, and testable with your own
testkit exactly as `CONTRIBUTING.md` describes - `run_tool("Crown_Seg", …)`
then `run(…)`.

**My half.** `depends_on` sequencing in the server, so a client still sends
one request and the server runs the dependency first and feeds its outputs in.
Until it lands, the chained modes are reachable by a caller willing to make
two calls - which the Slicer client can do today - so this is not as urgent as
it looks.

**What I need from you to build it:** the schema has to *name* the outputs.
`"returns": "dict[str, path]"` says there are several; it does not say what
they are called. A wiring that reads "feed ASO's `landmarks` from ALI's
`markups`" can only be checked at startup, like every other schema mistake, if
those names are declared. An `outputs` block in `describe.py` - from the
docstring's `Returns:` section, or from an annotation, whichever is cheaper - 
is the one addition I would ask for.

## Decision 1: ASO has four modes and one schema

`modality` (CBCT | IOS) × `automation` (Semi | Fully) = four modes. Twelve
arguments, of which **seven apply to some modes and not others**. The server
carries `visible_when` today, so the panel shows only what applies.

The schema has no equivalent, so a packaged ASO publishes all twelve at once:
roughly **180 check boxes** - 130 CBCT landmarks, 32 teeth, 8 landmark types,
2 jaws - CBCT and IOS interleaved, in one column. Any given run uses one half
or the other. That is exactly the panel the port removed, coming back.

Three ways out. I would take the third:

1. **Accept it.** Cheapest, and a clinician sees it every day.
2. **Grow the contract** with presentation fields (`visible_when`, `ui`,
   `groups`). Real work on both sides, and it puts UI vocabulary into a schema
   generated from a signature - which is what deriving `choices` from
   `Literal[...]` deliberately avoided.
3. **Split ASO into four tools**, one `run()` per mode. Each takes only its own
   arguments, so there is nothing conditional left to express: four honest
   panels, four short signatures, and the contract needs nothing new. The cost
   is four entries in `GET /tools` where there was one, and a naming
   convention - `ASO_CBCT_Semi`, `ASO_CBCT_Auto`, `ASO_IOS_Semi`,
   `ASO_IOS_Auto` or similar.

Option 3 also dissolves half of the chaining problem: only `ASO_CBCT_Auto`
needs landmarks from ALI, and the other three modes stop carrying an argument
they never use.

**The same question for ALI, with much smaller stakes.** Its 119 landmarks are
one multichoice, laid out today as tabs by anatomical group (`ui="tabs"` +
`groups`, both derived from the same table the engine names its output files
by). `choices` keeps every option; the tabs become a long scroll. Splitting
does not help - it is one tool with one long list. Live with the scroll, or
the contract grows. It is a worse panel, not a wrong one.

## Decision 2: nothing - this part is known work

**ALI's IOS engine needs pytorch3d**, which publishes no usable wheel: the
newest on PyPI is 0.7.4 with cp38–cp310 wheels built against a much older
torch. `Crown_Seg/pyproject.toml` has already solved exactly this, and it is
copyable line for line:

- the git tag under `[tool.uv.sources]`;
- `[tool.uv.extra-build-dependencies] pytorch3d = ["torch"]`, because its
  `setup.py` imports torch at build time without declaring it, and under uv's
  build isolation that is a `ModuleNotFoundError`;
- the whole thing behind an optional extra, so a plain `uv sync` stays fast and
  CI can still import the package to describe it.

Consequence to accept, and it is the same one Crown_Seg lives with: with the
extra unbuilt, ALI's **IOS half answers 503** and the CBCT half is unaffected - 
that one needs only torch, monai and itk.

**Keep ALI as one tool with two engines.** The mode is detected from the data,
not declared, and deliberately so: a `.zip` can hold either kind and a DICOM
series has no extension to dispatch on. An archive holding both kinds is a
422 rather than a guess. It means one package, one `run()`, and a dependency
set heavier than either half needs alone.

## Smaller things, found by running the packaged tools

- **Reports carry server-side absolute paths.** `AMASSS_report.json` and
  `run_report.json` list `/tmp/inference_server/job_<id>/output/…`, and those
  reports travel to the client inside the result archive. Not patient data,
  but server internals - and a path relative to `output_dir` would be more
  useful to whoever opens the archive.
- **ALI's IOS bundle** reports `models_unrecognized: ["Lower_MG_v6.pth"]`.
  Worth a look while the tool is open.
- `DATA/BatchDentalSeg/` has a model but no `testfiles/`, so the client's
  "Test file" button has nothing to offer. Data staging, not yours.

## What the server does for you, so you do not have to

Restating the parts that bear on these two specifically:

- **Archives are unpacked before `run()`**, with the zip-bomb cap and the
  single-root strip. ALI's discovery walked `.zip` files itself; it must stop.
- **`output_dir` is filled in** with the job's own `output/` and never
  published to a client.
- **`device` is filled in** from the deployment when the caller picks none.
- **The GPU is serialised across tools** - every run is assumed to want the
  card unless its `device` resolves to a CPU value. Neither tool should hold a
  semaphore of its own.
- **Errors map by class name**: `ToolInputError` / `ValueError` /
  `FileNotFoundError` answer 422 with your message passed through verbatim, so
  write those messages for whoever sent the request;
  `ToolUnavailableError` answers 503; anything else answers 500 with a fixed
  message and only the traceback in the server log.
