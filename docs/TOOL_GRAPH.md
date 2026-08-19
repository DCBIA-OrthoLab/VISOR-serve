# Which tool calls which

Generated from each tool's `.schema.json` `calls` field, which `describe.py`
derives by reading the tool's own source. The server refuses to start when one
of these names is not served, so this graph is checked at every boot rather than
maintained by hand.

## The graph

```
AREG_IOSCBCT ──┬──> Crown_Seg
               ├──> ALI_IOS
               ├──> ALI_CBCT
               └──> ASO ──> ALI_CBCT

AREG_IOS ──────┬──> Crown_Seg
               ├──> ALI_IOS
               └──> ASO ──> ALI_CBCT

AREG_CBCT ─────┬──> AMASSS
               └──> ASO ──> ALI_CBCT

ASO ───────────────> ALI_CBCT

AMASSS              leaf
ALI_CBCT            leaf
ALI_IOS             leaf
Crown_Seg           leaf
Batch_Dental_Seg    leaf
Surg_Mov_Pred       leaf
```

## Read as a table

| tool | calls | depth below it |
|---|---|---|
| `AREG_IOSCBCT` | Crown_Seg, ALI_IOS, ALI_CBCT, ASO | 2 |
| `AREG_IOS` | Crown_Seg, ALI_IOS, ASO | 2 |
| `AREG_CBCT` | AMASSS, ASO | 2 |
| `ASO` | ALI_CBCT | 1 |
| `AMASSS` | | 0 |
| `ALI_CBCT` | | 0 |
| `ALI_IOS` | | 0 |
| `Crown_Seg` | | 0 |
| `Batch_Dental_Seg` | | 0 |
| `Surg_Mov_Pred` | | 0 |

## What the shape tells you

**ASO is the only callee that is itself a caller.** Every other dependency is a
leaf that does its work and returns. ASO, in its fully-automated CBCT mode,
runs ALI_CBCT to predict the landmarks it orients on. That is what makes the
deepest real chain three levels: `AREG_IOSCBCT -> ASO -> ALI_CBCT`.

**ALI_CBCT is reached two ways from AREG_IOSCBCT**, directly and through ASO.
That is not a cycle: the check refuses a tool already running ABOVE the call,
not one appearing twice on different branches.

**A call only fires in the automated modes.** In `Semi-Automated`, and in
AREG_IOSCBCT's `Registration`, the caller takes what it needs as arguments and
the graph above collapses to nothing. The modes are where the cost is:
Registration runs in 1.9s, Semi-Automated in 87s, Fully-Automated in 149s, and
the difference is entirely the children.

**Only orchestrators are light.** AREG_IOSCBCT holds ~12 MB because it imports
no torch at all; a leaf holds ~500 MB. A chain is sequential, so one heavy
process lives at a time whatever the depth. What multiplies memory is
`MAX_CONCURRENT_TOOLS`, not nesting.

## Consequences for configuration

An orchestrating tool's `timeout_seconds` has to cover its whole chain, not its
own compute. `sum of the timeouts of every tool in calls * 1.2` is the rule of
thumb, and a chain is sequential so the terms add rather than max.

A tool listed here must be deployed for its caller to be servable at all. That
is checked at startup, not at request time.
