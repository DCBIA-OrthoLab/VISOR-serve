# Quality-control stops: the map

A run can be stopped so a clinician looks at what it has produced, corrects it,
and lets it carry on. This document exists because that turns one request into
a conversation, and a conversation is easy to get lost in: three repositories,
two directions of travel, and a job directory whose shape is the whole
mechanism. Read it when you cannot remember who is holding what.

## The words

| word | what it means |
|---|---|
| **stop** | a boundary a run may be asked to halt at. Named after the tool whose call it follows (`ALI_CBCT`), or after what a tool declared mid-run |
| **step** / **slot** | one supervised call, as it exists on disk: `01_ALI_CBCT`. The number is the call's position, which is what keeps two calls to one tool apart |
| **memo** | what a completed call returned, recorded so a resumed run answers from it instead of running again |
| **correction** | files a reader sends back for one step |
| **flag** | a patient a reader wants re-done from an earlier stop |
| **replay** | running part of a chain again, narrowed to the flagged patients |

## Who decides what

Three repositories, and the seam between them is the point.

- **`SADT-VISOR`** — a tool declares. It calls `sup.run("ALI_CBCT")`, or
  `sup.declareQualityControl("landmarks")`. It knows nothing about stops,
  pauses or HTTP.
- **`VISOR-serve`** — the server publishes and arms. It reads the tool's
  source by AST, injects a `stop_after` multichoice into the published schema,
  carries the armed stops down the chain, keeps the job directory when a run
  halts, and resumes it.
- **`AutomatedDentalToolsRemote`** — the client reviews. It unpacks what came
  back, opens VISU, and sends back only what the reader changed.

A tool never learns that it was paused. A client never learns how memoisation
works. That is deliberate and worth defending.

## What a job looks like on disk

ASO, fully-automated CBCT, stopped after `ALI_CBCT`:

```
<job>/
  input/                                what the client uploaded
  job.json                              the REQUEST, written once, never rewritten
  stopped.json                          which stop was reached
  sup/
    01_ALI_CBCT/
      job.json                          what the nested call was asked
      memo.json                         what it returned   <- absent if it stopped
      output/                           <- what the resume reads, and where a
        Pat_0002_lm_Pred.mrk.json          correction is laid over
        run_report.json
  resume/
    01_ALI_CBCT/                        what the reader sent back, waiting
  output/
    intermediate/01_ALI_CBCT/           the COPY handed to the reader
```

Two things about that layout carry the whole design:

- **`sup/<step>/output` is the original, `output/intermediate/<step>` is the
  copy.** The reader downloads the copy; the resume reads the original. A
  correction is laid over the original, file by file.
- **A chain nests.** The supervisor inside `01_Mid` works in
  `<job>/sup/01_Mid`, so its own steps are `<job>/sup/01_Mid/sup/01_Leaf` and
  its corrections are staged in `<job>/sup/01_Mid/resume/01_Leaf`. A
  correction for a step a callee made is therefore named by its path:
  `01_Mid/01_Leaf`.

## Round trip 1: stop, correct, carry on

This exists and is proven end to end.

```
client                          server                          tool
  |  POST /run/ASO stop_after=ALI_CBCT
  |------------------------------>|
  |                               |  runs ASO, which calls ALI_CBCT
  |                               |<-------------------------------|
  |                               |  QualityControlStop after the call
  |  200 {quality_control, stopped_after, produced, result_ref}
  |<------------------------------|  state: paused / paused
  |                                  the job directory is KEPT
  |
  |  GET /results/<ref>            the reader's copy
  |<------------------------------|
  |  ... VISU. digest at unpack, digest at Continue ...
  |
  |  POST /runs/<id>/resume   01_ALI_CBCT=<zip of only what changed>
  |------------------------------>|
  |                               |  stage the correction, lay it over
  |                               |  re-enter ASO from the top; the ALI call
  |                               |  answers from its memo, corrected
  |  200 <the finished archive>
  |<------------------------------|  state: done / done
```

**Why the tool re-enters from the top.** Nothing preserves a Python stack
across a process that exited, and asking thirteen orchestrators to become
re-enterable by hand is thirteen chances to get it subtly wrong. So the tool
runs again and the CALLS are what is remembered.

## Round trip 2: go back for some patients

**This is the part being built.** A reader looking at a bad orientation cannot
fix it there: the orientation was computed from landmarks decided two steps
back. So they mark the patients that are wrong and ask to return to the last
stop where something can actually be changed.

Decided:

- **The same run.** No second run id, no merge step. The server drops the memo
  of the target step and of every step after it, and runs the chain again with
  only the flagged patients as input. The rest of the cohort keeps the output
  it already has, and the flagged patients' new results are laid over it.
- **A stop declares whether it is a return point.** The local module already
  makes this distinction and it is the right one: `view` (look only),
  `landmarks` (drag the points), `registration` (drag the scan). Only the last
  two are somewhere to go back TO. A stop that is only ever looked at is
  skipped over, so the button lands where something can be done.
- **The flag IS the return.** One button, worded as the local module words it:
  `Go back and edit this patient` / `Cancel - this patient is fine`, with
  `Go back to <step> for <n> patient(s)` beside it. There is no separate
  "note" any more: a marked patient is a patient that will be re-run.

```
client                          server
  |  the reader flags 3 patients of 40 at the ASO stop
  |
  |  POST /runs/<id>/rewind   to=01_ALI_CBCT  patients=[...]
  |------------------------------>|  drop the memo of 01_ALI_CBCT and after
  |                               |  narrow the input to the 3
  |  200 <that step's files, for those 3 only>
  |<------------------------------|  state: paused, at the EARLIER stop
  |  ... the reader edits the landmarks of those 3 ...
  |  POST /runs/<id>/resume
  |------------------------------>|  replay for the 3; the other 37 untouched
```

### How the patients are narrowed

**No temporary folder, and no manifest.** The client sends back the corrected
files for the flagged patients and nothing else; the server reads WHICH
patients from the files it was given, and replays for those. There is nothing
to keep in step between the two sides, because there is only one list and it
is the one that travelled.

What that leaves to settle, and it is the part that can go wrong in silence:

- **How a patient is identified from a file name.** The local module derives an
  id and normalises it, because four naming conventions coexist
  (`Pat_0002_lm_Pred.mrk.json` is patient `Pat_0002`). VISU's index already
  does this on the client. If the two sides disagree, the wrong patient is
  replayed and nothing says so. They have to share one rule, and it has to be
  tested against all four conventions.
- **What "the steps after it" means across a chain.** Dropping the memo of
  `01_ALI_CBCT` is clear. Dropping everything after it, when "after" spans two
  levels of nesting, is not yet written down.

## State

| | |
|---|---|
| declare a stop (nested call, or `declareQualityControl`) | done |
| publish `stop_after`, no tool edit | done |
| stops inside a callee, qualified paths (`ASO/ALI_CBCT`) | done |
| stop, keep the job, answer before finishing | done |
| memoised resume | done |
| correction laid over a step, file by file | done |
| only the changed files travel back | done |
| corrections for a step a callee made | done |
| VISU: review, flag, lock, save, continue | done |
| a stop declaring what may be edited there (`view`/`landmarks`/`registration`) | declared; not yet read by the server |
| **rewind to an earlier stop for flagged patients** | to build |
| no served tool declares `declareQualityControl` yet | open |
| a reader who repoints the folder sends everything | open |
