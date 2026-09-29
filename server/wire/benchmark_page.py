"""The page `GET /benchmarks/view` serves: how this server coped under load.

`GET /benchmarks` answers the campaign as JSON, and the status page shows the
headline of it in one table. That table says an arm took 95 seconds; it cannot
say WHERE the seconds went -- and the whole reason a campaign is run is that
`queued_gpu` and `running` cost the same wall clock while meaning opposite
things. One is a machine computing, the other is admission making a client
wait. This page draws them apart.

**The width goes to the timeline, not to the prose.** A 4 s `queued_gpu` inside
a 400 s run is a sliver at 1080 px and legible at 2400. So the page fills the
screen it is given, while every line of text stays inside a comfortable measure
and the arms are compared against each other in a multi-column strip rather
than scrolled between. A layout that is merely stretched reads worse than a
capped one.

**Concurrency is said in words, not left to be inferred.** An arm with six
clients has six run rows, and the overlap between their `running` bars is the
entire question: did the server run six CLIC at once, or queue five behind one?
The arm header says which -- "6 CLIC at once, 5.8 on average" -- and a
concurrency trace on the same x-scale as the Gantt shows how that number moved
through the arm.

**A run row answers what that ONE run was asked to do.** The arm's setup line
describes the arm, which is not the same thing: B4 runs four different tools
across six clients, so no single sentence above the rows can describe any of
them. And the size of the question moves the runtime by an order of magnitude
-- ALI_CBCT is 26.5 s for one landmark and 476.1 s for all four regions, with
identical memory throughout -- so two rows reading `ALI_CBCT 476 s` and
`ALI_CBCT 35 s` look like a server problem and are a question-size difference.
"3 of 119 landmarks", drawn as a filled proportion, is what resolves it, and it
is on the row itself rather than only inside the panel it opens.

**Self-contained on purpose.** No CDN, no font, no framework, no charting
library: a server holding confidential imaging is deployed on networks that do
not reach the internet, and a page whose stylesheet 404s is worse than no page.
The Gantt is DOM elements positioned in percent and every trace is inline SVG,
so they share one axis exactly and neither needs a script it cannot fetch.

**It holds no token.** The shell says nothing; the numbers live behind
`GET /benchmarks`, which is Bearer-protected. The token is read from (and
written to) `localStorage` under the SAME key the status page uses, so someone
who authenticated on one page is never asked twice. It is sent as a header and
never put in the URL, where it would land in proxy logs and browser history.

**Every URL it fetches is relative** (`../benchmarks`, resolved against
`/benchmarks/view`), so the page works unchanged behind a reverse proxy serving
this server on a sub-path, and cannot be pointed at another host. The campaign
picker writes its choice into the query string with `replaceState`, so a link
to one campaign is shareable and still resolves against whatever host served
the page.

**A limit test is read for whether anything broke, so that is what it says
first.** An arm carrying a `stress` block was not run to find out how fast the
server is: it pushed until admission, the card or the host budget was the
binding constraint, and the only question is whether the machine degraded or
broke. So the verdict sits ABOVE the arm's own numbers -- failures,
out-of-memories and unanswered health checks in one cluster, and `refused` and
`retried` deliberately in another, because a refusal is the budget working and
a retry is the out-of-memory path recovering, and both read as damage in a
table that prints them beside a failure count. Each is said in the page's own
words, since nothing about the word "retried" tells a reader it means a run
that hit an out-of-memory and came back.

**The card is drawn against its budget, not printed beside it.** An arm that
peaked at 13% of the ceiling it names did not test that ceiling, however green
its zeroes are, and a proud "nothing broke" over it is the one wrong answer
this page could give. The queue it sampled is a stacked trace on the Gantt's
own x-scale, held over running: a rising amber band above a flat green one is
admission holding the line, both rising is the machine absorbing it, and the
green band collapsing mid-arm is something dying -- each of which is only
legible against the runs that caused it, which is why it reuses the grid the
Gantt and the VRAM trace already share.

And it shows no progress MESSAGE and no id for any run, for the reason
`GET /status` carries neither: a message is free text a tool wrote and can name
a patient's file, and an id is the capability to read a run. The parameters a
run row opens are the ones the server already reduced to basenames before
publishing them; this page renders what it was given and never a key it was
not.
"""

from __future__ import annotations

BENCHMARK_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VISOR benchmarks</title>
<style>
  /* Three theme states, and each one is a complete palette.

     A token whose sole definition lived inside a media query would be an
     undefined colour for half the readers, so the full light palette is
     defined on bare :root. The dark palette is then declared TWICE on
     purpose: once for the reader whose system says dark and who has not
     chosen otherwise, and once for the reader who picked dark here. Without
     the second block the toggle can only ever darken a light system; without
     the `:not([data-theme="light"])` guard it can never lighten a dark one. */
  :root {
    color-scheme: light;
    --bg: #fbfbfa; --panel: #ffffff; --line: #e4e2dd; --ink: #1c1b19;
    --soft: #6f6b63; --faint: #908b82;
    --track: #efedea; --grid: #e6e4df; --fill: #3d6b8e;
    --warn: #9c5b2a; --ok: #4a7c59; --chip: #f1efeb; --sunk: #f7f6f4;
    --p-transfer: #4a7fa8;
    --p-received: #97a2aa;
    --p-staging: #6d7b86;
    --p-queued: #c9821f;
    --p-queued-alt: #eec489;
    --p-running: #2f6b55;
    --p-packaging: #7d5f9e;
    --p-fetch: #9cc5e2;
    --nest: #e3d5ae; --nest-edge: rgba(0, 0, 0, .42);
    --vram-fill: #d3dde8; --vram-line: #3d6b8e;
    --conc-fill: #d7e5dd; --conc-line: #2f6b55;
    --font-sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
                 "Helvetica Neue", sans-serif;
    --font-mono: ui-monospace, SFMono-Regular, Menlo, Consolas,
                 "Liberation Mono", monospace;
    /* The Gantt's fixed columns. Named here so the run rows, the two traces
       and the axis cannot drift apart, and so a wide screen spends its pixels
       on the track rather than on the label beside it. */
    --lab: clamp(160px, 11vw, 250px);
    --val: 76px;
    --sel: 132px;
    --cost: 96px;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --bg: #17181a; --panel: #1e2023; --line: #2e3135; --ink: #e8e6e3;
      --soft: #93908a; --faint: #7d7a74;
      --track: #26292c; --grid: #2b2e32; --fill: #6ea8d4;
      --warn: #d89a63; --ok: #7fb08c; --chip: #26292c; --sunk: #202326;
      --p-transfer: #4f93c4;
      --p-received: #7d878f;
      --p-staging: #a2aeb6;
      --p-queued: #d89a3f;
      --p-queued-alt: #7a531a;
      --p-running: #4e9c7c;
      --p-packaging: #a382c9;
      --p-fetch: #a9d0ea;
      --nest: #e3d5ae; --nest-edge: rgba(0, 0, 0, .55);
      --vram-fill: #2b3a4a; --vram-line: #6ea8d4;
      --conc-fill: #22362d; --conc-line: #4e9c7c;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --bg: #17181a; --panel: #1e2023; --line: #2e3135; --ink: #e8e6e3;
    --soft: #93908a; --faint: #7d7a74;
    --track: #26292c; --grid: #2b2e32; --fill: #6ea8d4;
    --warn: #d89a63; --ok: #7fb08c; --chip: #26292c; --sunk: #202326;
    --p-transfer: #4f93c4;
    --p-received: #7d878f;
    --p-staging: #a2aeb6;
    --p-queued: #d89a3f;
    --p-queued-alt: #7a531a;
    --p-running: #4e9c7c;
    --p-packaging: #a382c9;
    --p-fetch: #a9d0ea;
    --nest: #e3d5ae; --nest-edge: rgba(0, 0, 0, .55);
    --vram-fill: #2b3a4a; --vram-line: #6ea8d4;
    --conc-fill: #22362d; --conc-line: #4e9c7c;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 13.5px/1.55 var(--font-sans);
    -webkit-font-smoothing: antialiased;
  }
  /* Wide, but not unbounded: past ~2400px a timeline stops gaining resolution
     a reader can use and the page starts needing a head turn. */
  main { width: 100%; max-width: 2400px; margin: 0 auto;
         padding: 20px clamp(14px, 1.6vw, 30px) 80px; }

  /* Typography roles. Five, and every element on the page takes one of them:
     page title, section label, claim, figure, micro-label. */
  h1 { font-size: 15px; font-weight: 600; letter-spacing: .07em;
       text-transform: uppercase; margin: 0 0 3px; }
  h2 { font-size: 11px; font-weight: 600; letter-spacing: .09em;
       text-transform: uppercase; color: var(--soft); margin: 0 0 12px; }
  .sub { color: var(--soft); font-size: 12.5px;
         font-variant-numeric: tabular-nums; }
  /* Prose is capped at a readable measure even on a 2400px screen. The width
     buys timeline resolution; a 300-character line buys nothing. */
  .empty, .note { color: var(--soft); font-size: 12.5px; max-width: 74ch; }
  .empty { padding: 6px 0; }
  .err { color: var(--warn); font-size: 12.5px; }
  .bad { color: var(--warn); }
  .gain { color: var(--ok); font-weight: 600; }
  .scroll { overflow-x: auto; }

  /* Header: identity on the left, the two pieces of chrome on the right. The
     picker labels the page, it is not the subject of it. */
  .top { display: flex; flex-wrap: wrap; align-items: flex-end;
         justify-content: space-between; gap: 12px 20px; margin-bottom: 16px; }
  .chrome { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .pick { display: inline-flex; align-items: center; gap: 7px;
          font-size: 11px; letter-spacing: .08em; text-transform: uppercase;
          color: var(--soft); }
  select { font: inherit; font-size: 12.5px; text-transform: none;
           letter-spacing: 0; color: var(--ink); background: var(--panel);
           border: 1px solid var(--line); border-radius: 6px; padding: 5px 8px;
           max-width: min(46vw, 380px); }
  .stamp { font-size: 12px; color: var(--faint);
           font-variant-numeric: tabular-nums; }
  .ghost { background: transparent; color: var(--soft); border: 1px solid
           var(--line); border-radius: 6px; padding: 5px 10px; font: inherit;
           font-size: 12px; cursor: pointer; min-width: 62px; }
  .ghost:hover { color: var(--ink); }

  section { background: var(--panel); border: 1px solid var(--line);
            border-radius: 9px; padding: 15px clamp(14px, 1.2vw, 20px);
            margin-bottom: 14px; }

  /* Summary strip */
  .strip { display: grid; gap: 11px;
           grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }
  .tile { border: 1px solid var(--line); border-radius: 7px; padding: 9px 12px;
          background: var(--sunk); }
  .tile .k { font-size: 10.5px; letter-spacing: .09em; text-transform: uppercase;
             color: var(--soft); }
  .tile .v { font-size: 21px; font-weight: 600; font-variant-numeric: tabular-nums;
             line-height: 1.25; margin-top: 3px; letter-spacing: -.01em; }
  .tile .v.small { font-size: 12.5px; font-weight: 500; line-height: 1.35;
                   word-break: break-word; letter-spacing: 0; }
  .tile .n { font-size: 11px; color: var(--faint); margin-top: 2px;
             font-variant-numeric: tabular-nums; }

  /* The legend: every phase named, so colour is never the only carrier. */
  .legend { position: sticky; top: 0; z-index: 5; background: var(--bg);
            border-bottom: 1px solid var(--line); padding: 8px 0 9px;
            margin-bottom: 14px; display: flex; flex-wrap: wrap; gap: 6px 14px;
            align-items: center; }
  .lg { display: inline-flex; align-items: center; gap: 6px; font-size: 11.5px;
        color: var(--soft); }
  .sw { width: 22px; height: 10px; border-radius: 2px; flex: none;
        border: 1px solid var(--nest-edge); }

  /* Phase colours. queued_gpu is hatched as well as coloured: it is the one
     phase that must never be mistaken for work, and a hatch survives both a
     greyscale print and a reader who cannot separate amber from green. */
  .ph-transfer  { background: var(--p-transfer); }
  .ph-received  { background: var(--p-received); }
  .ph-staging   { background: var(--p-staging); }
  .ph-queued    { background: repeating-linear-gradient(45deg,
                    var(--p-queued) 0 4px, var(--p-queued-alt) 4px 8px); }
  .ph-running   { background: var(--p-running); }
  .ph-packaging { background: var(--p-packaging); }
  .ph-fetch     { background: var(--p-fetch); }
  .ph-nest      { background: var(--nest); }
  .ph-conc      { background: var(--conc-fill); }
  .ph-wait      { background: var(--p-queued-alt); }

  /* Arms at a glance. The one place the page goes multi-column: a family of
     arms is a handful of short rows, so four of them fit side by side on a
     wide screen and A1/A2/A3 can be read against each other instead of
     scrolled between. */
  .fams { display: grid; gap: 12px;
          grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); }
  .fam { border: 1px solid var(--line); border-radius: 8px; padding: 11px 13px;
         background: var(--sunk); }
  .fhead { display: flex; align-items: baseline; gap: 9px; margin-bottom: 9px; }
  .fkey { font-family: var(--font-mono); font-size: 12px; font-weight: 600;
          letter-spacing: .04em; color: var(--ink); }
  .fwhat { font-size: 12px; color: var(--soft); overflow: hidden;
           text-overflow: ellipsis; white-space: nowrap; }
  .frow { display: grid; align-items: center; gap: 4px 10px; margin-top: 5px;
          grid-template-columns: 30px minmax(84px, auto) 1fr 54px 62px 46px; }
  .flab { font-size: 11.5px; color: var(--soft); white-space: nowrap;
          overflow: hidden; text-overflow: ellipsis; }
  .ftrack { position: relative; height: 15px; background: var(--track);
            border-radius: 3px; overflow: hidden; }
  /* Outer band: the wall clock. Inner bar: the median run inside it. The
     ratio between them IS the concurrency -- six runs in the time of one and
     a half is the arm working; six in the time of six is a queue. */
  .fwall { position: absolute; left: 0; top: 0; height: 100%;
           background: var(--vram-fill); }
  .fmed { position: absolute; left: 0; top: 3px; height: 9px;
          border-radius: 2px; background: var(--p-running); }
  .fn, .fc, .fs { font-size: 11.5px; text-align: right;
                  font-variant-numeric: tabular-nums; }
  .fn { color: var(--ink); }
  .fc { color: var(--soft); }
  .fs { color: var(--faint); }
  .fs.gain { color: var(--ok); }
  .fs.bad { color: var(--warn); font-weight: 600; }
  .fcap { font-size: 11px; color: var(--faint); margin-top: 9px; }

  /* Arm card */
  .fdiv { display: flex; align-items: baseline; gap: 9px; margin: 22px 0 2px;
          padding-bottom: 6px; border-bottom: 1px solid var(--line); }
  .fdiv:first-child { margin-top: 0; }
  .arm { padding-top: 15px; margin-top: 4px; }
  .armhead { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px; }
  .code { font-family: var(--font-mono); font-size: 12px; font-weight: 600;
          letter-spacing: .04em; background: var(--chip);
          border: 1px solid var(--line); border-radius: 5px; padding: 1px 6px; }
  .name { font-size: 13.5px; font-weight: 600; }
  .at { margin-left: auto; font-size: 11.5px; color: var(--faint);
        font-variant-numeric: tabular-nums; }
  /* The sentence the whole arm exists to answer, in the reader's own words. */
  .claim { display: flex; align-items: baseline; gap: 8px; margin: 9px 0 2px;
           font-size: 14.5px; font-variant-numeric: tabular-nums; }
  .claim b { font-weight: 600; }
  .dot { width: 9px; height: 9px; border-radius: 2px; flex: none;
         background: var(--p-running); transform: translateY(-1px); }
  .claimx { color: var(--soft); font-size: 12.5px; }
  .metrics { display: flex; flex-wrap: wrap; gap: 4px 18px; margin: 7px 0 4px;
             font-size: 12px; color: var(--soft);
             font-variant-numeric: tabular-nums; }
  .metrics b { color: var(--ink); font-weight: 600; }
  .setup { font-size: 12px; color: var(--soft); margin: 4px 0 12px;
           font-variant-numeric: tabular-nums; }
  /* Its own line, always: the capture date beside it is pushed to the right
     edge, and a short setup line sharing that line would read as a caption
     for the date rather than for the arm. */
  .armhead .setup { flex: 1 0 100%; margin: 2px 0 0; }

  /* Gantt. One grid, reused by every run row, by both traces and by the axis
     -- which is what makes a peak land above the run that caused it without a
     pixel of arithmetic. The two optional columns appear only for an arm
     whose records carry them; on an older campaign the grid is the three it
     always was. */
  .gantt { min-width: 640px; }
  .grow { display: grid; grid-template-columns: var(--lab) 1fr var(--val);
          gap: 10px; align-items: center; margin-bottom: 3px; }
  .gantt.sel .grow { grid-template-columns:
          var(--lab) 1fr var(--sel) var(--val); }
  .gantt.cost .grow { grid-template-columns:
          var(--lab) 1fr var(--val) var(--cost); }
  .gantt.sel.cost .grow { grid-template-columns:
          var(--lab) 1fr var(--sel) var(--val) var(--cost); }
  .ghead { margin-bottom: 6px; }
  .ghead > div { font-size: 10px; letter-spacing: .09em; text-transform: uppercase;
                 color: var(--faint); }
  .ghead .rv, .ghead .rc { text-align: right; }
  .rl { font-size: 11.5px; color: var(--soft); white-space: nowrap;
        overflow: hidden; text-overflow: ellipsis;
        font-variant-numeric: tabular-nums; }
  .rl .who { font-family: var(--font-mono); font-size: 11px; }
  .rl b { color: var(--ink); font-weight: 500; }
  /* A row that has something to open IS a control: a real button, reachable
     by keyboard, carrying its own aria-expanded. A row whose record has no
     invocation stays a plain label -- an affordance that does nothing is
     worse than one that was never offered. */
  button.rl { background: none; border: 0; padding: 0; margin: 0; font: inherit;
              font-size: 11.5px; color: var(--soft); text-align: left;
              cursor: pointer; display: block; width: 100%; }
  button.rl:hover b, button.rl:focus-visible b { color: var(--fill); }
  .caret { display: inline-block; width: 9px; color: var(--faint); }
  .rt { position: relative; height: 16px; background: var(--track);
        border-radius: 3px; overflow: hidden;
        background-image: linear-gradient(to right, var(--grid) 0,
                          var(--grid) 1px, transparent 1px);
        background-repeat: repeat-x; }
  .rv { text-align: right; font-size: 11.5px; color: var(--soft);
        font-variant-numeric: tabular-nums; }
  .rc { text-align: right; font-size: 11.5px; color: var(--faint);
        font-variant-numeric: tabular-nums; white-space: nowrap; }
  .ph { position: absolute; top: 0; height: 100%; min-width: 2px;
        border-radius: 2px; }
  /* A nested call is drawn INSIDE the parent's running span, because that is
     literally what the parent was doing: waiting inside sup.run(). A thin
     strip along the bottom keeps the row from turning into noise. */
  .nb { position: absolute; bottom: 0; height: 5px; min-width: 2px;
        border-radius: 2px; background: var(--nest);
        box-shadow: 0 0 0 1px var(--nest-edge); }
  .nb.d2 { bottom: 5px; }
  .rerr { grid-column: 2 / -1; font-size: 11px; color: var(--warn);
          white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
          margin: -1px 0 4px; }
  .axis { position: relative; height: 17px; background-repeat: repeat-x;
          background-image: linear-gradient(to right, var(--grid) 0,
                            var(--grid) 1px, transparent 1px); }
  .tick { position: absolute; top: 6px; font-size: 10.5px; color: var(--soft);
          transform: translateX(-50%); font-variant-numeric: tabular-nums; }
  .tick.first { transform: none; }
  .tick.last { transform: translateX(-100%); }
  .trace { height: 44px; }
  .trace.short { height: 34px; }
  .trace svg { display: block; width: 100%; height: 100%; }

  /* How much of what the tool offers this run actually asked for. On the row,
     not only inside the panel: it is the difference between a 26 s run and a
     476 s one, and a page readable at rest has to carry it. */
  .sc { display: flex; align-items: center; gap: 7px; font-size: 11px;
        color: var(--soft); font-variant-numeric: tabular-nums; }
  .sbar { position: relative; flex: 1; min-width: 28px; height: 7px;
          border-radius: 4px; background: var(--track); overflow: hidden; }
  .sfill { position: absolute; left: 0; top: 0; height: 100%;
           background: var(--p-transfer); border-radius: 4px; }
  .sn { white-space: nowrap; }
  .sn b { color: var(--ink); font-weight: 600; }

  /* A limit test. This kind of arm is not read for speed: it is read for
     whether anything broke, and that answer comes before the arm's own
     numbers rather than after them. The three ways it can have broken are
     counted in one cluster, the two things that are NOT breakage in another,
     and the card is drawn against the budget the arm was built to reach --
     because an arm that never got near its budget did not test what it
     claimed to, and a green zero there is a false pass. */
  .stress { border: 1px solid var(--line); border-radius: 8px;
            background: var(--sunk); padding: 11px 13px; margin: 10px 0 2px; }
  .verdict { display: flex; align-items: baseline; gap: 8px;
             font-size: 14.5px; font-variant-numeric: tabular-nums; }
  .verdict b { font-weight: 600; }
  .dot.ok { background: var(--ok); }
  .dot.warn { background: var(--warn); }
  .sgrid { display: grid; gap: 13px 20px; margin-top: 11px;
           grid-template-columns: repeat(auto-fit, minmax(245px, 1fr)); }
  .sfig { display: flex; align-items: baseline; gap: 8px; margin-top: 4px;
          font-size: 12px; color: var(--soft);
          font-variant-numeric: tabular-nums; }
  .sfig .num { font-size: 15px; font-weight: 600; color: var(--ink);
               min-width: 1.7em; text-align: right; flex: none; }
  .sfig .num.bad { color: var(--warn); }
  .sfig .num.zero { color: var(--faint); font-weight: 500; }
  /* Said in the page's own words: a reader meeting "retried" for the first
     time has no way to know it means an out-of-memory that recovered. */
  .sgloss { font-size: 11.5px; color: var(--faint); margin-top: 7px;
            max-width: 48ch; }
  /* Budget is the track, the peak is the fill. Amber when the fill is short:
     the arm did not reach the limit it names. */
  .meter { position: relative; height: 11px; border-radius: 4px;
           background: var(--track); overflow: hidden; margin: 8px 0 2px; }
  .mfill { position: absolute; left: 0; top: 0; height: 100%; border-radius: 4px;
           background: var(--vram-line); }
  .mfill.shy { background: var(--p-queued); }

  /* What one run was asked to do, opened in place. Not a modal: the reader is
     comparing runs, and a dialog over the Gantt defeats the comparison. */
  .det { grid-column: 2 / -1; margin: 2px 0 9px; padding: 11px 13px;
         border: 1px solid var(--line); border-left: 3px solid var(--fill);
         border-radius: 7px; background: var(--sunk); display: grid; gap: 14px;
         grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }
  .det[hidden] { display: none; }
  .dk { font-size: 10px; letter-spacing: .09em; text-transform: uppercase;
        color: var(--faint); margin-bottom: 6px; }
  .sline { display: grid; grid-template-columns: 1fr; gap: 3px;
           margin-bottom: 9px; }
  .sline .sk { font-size: 11.5px; color: var(--soft); }
  .sline .sbig { display: flex; align-items: center; gap: 9px; }
  .sline .sbar { height: 11px; }
  .sline .sn { font-size: 13px; }
  .dio { font-size: 12px; color: var(--soft); line-height: 1.6;
         font-variant-numeric: tabular-nums; }
  .dio b { color: var(--ink); font-weight: 500; }
  .dpar { display: grid; grid-template-columns: auto 1fr; gap: 2px 10px;
          font-size: 11.5px; align-items: baseline; }
  .dpar .pk { color: var(--faint); white-space: nowrap; }
  .dpar .pv { color: var(--soft); font-family: var(--font-mono);
              font-size: 11px; word-break: break-word; }

  /* Per-phase means */
  .means { margin-top: 13px; display: grid;
           grid-template-columns: 100px minmax(120px, 1fr) 58px; gap: 4px 10px;
           align-items: center; max-width: 520px; }
  .mk { font-size: 10.5px; letter-spacing: .07em; text-transform: uppercase;
        color: var(--soft); }
  .mt { height: 7px; background: var(--track); border-radius: 4px;
        overflow: hidden; }
  .mf { height: 100%; border-radius: 4px; }
  .mv { text-align: right; font-size: 11.5px; color: var(--soft);
        font-variant-numeric: tabular-nums; }
  .inside { font-size: 12px; color: var(--soft); margin-top: 11px;
            font-variant-numeric: tabular-nums; max-width: 74ch; }
  .inside b { color: var(--ink); font-weight: 600; }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; font-weight: 500; color: var(--soft); font-size: 10.5px;
       letter-spacing: .08em; text-transform: uppercase; padding: 0 10px 8px 0;
       white-space: nowrap; }
  td { padding: 5px 10px 5px 0; border-top: 1px solid var(--line);
       font-variant-numeric: tabular-nums; vertical-align: middle; }
  td.n { text-align: right; white-space: nowrap; }
  td.tbar { width: 34%; min-width: 120px; }
  .tag { font-size: 11px; padding: 1px 7px; border-radius: 10px;
         border: 1px solid var(--line); color: var(--soft); white-space: nowrap; }
  .tag.run { color: var(--ok); border-color: var(--ok); }
  .tag.wait { color: var(--warn); border-color: var(--warn); }

  .gate { display: flex; gap: 8px; margin-top: 10px; max-width: 520px; }
  input { flex: 1; padding: 8px 10px; border: 1px solid var(--line);
          border-radius: 6px; background: var(--bg); color: var(--ink);
          font: inherit; }
  button.primary { padding: 8px 14px; border: 1px solid var(--line);
           border-radius: 6px; background: var(--fill); color: #fff;
           font: inherit; cursor: pointer; }
</style>
</head>
<body>
<main>
  <header class="top">
    <div>
      <h1>VISOR benchmarks</h1>
      <div class="sub" id="when">loading&hellip;</div>
    </div>
    <div class="chrome">
      <label class="pick" id="pickwrap" hidden>campaign
        <select id="pick"></select>
      </label>
      <div class="stamp" id="stamp"></div>
      <button class="ghost" id="theme" type="button"
              title="Follow the system, or force light or dark">auto</button>
    </div>
  </header>

  <section id="gate" hidden>
    <h2>API token</h2>
    <div class="empty">This campaign is Bearer-protected: it names every tool
      this deployment serves and how hard it can be pushed. The token is kept
      in this browser only and sent as a header, never in the URL.</div>
    <div class="gate">
      <input id="token" type="password" placeholder="API token" autocomplete="off">
      <button class="primary" id="save">Connect</button>
    </div>
    <div class="err" id="gateerr"></div>
  </section>

  <section id="none" hidden>
    <h2>No campaign</h2>
    <div class="empty">Nothing has been measured on this deployment. That is
      the normal case: a campaign is run deliberately, and this page has
      something to draw only once one has written its summary where the server
      reads it.</div>
  </section>

  <section id="summary" hidden>
    <h2>The campaign</h2>
    <div class="strip" id="strip"></div>
  </section>

  <section id="glance" hidden>
    <h2>Arms at a glance</h2>
    <div class="fams" id="glancebody"></div>
  </section>

  <div class="legend" id="legend" hidden></div>

  <section id="arms" hidden>
    <h2>Every arm, run by run</h2>
    <div id="armbody"></div>
  </section>

  <section id="cover" hidden>
    <h2>Every tool, run once</h2>
    <div id="coverbody"></div>
  </section>
</main>
<script>
(function () {
  "use strict";
  // The SAME key the status page writes: a reader who authenticated there is
  // not asked again here.
  var KEY = "visor.token";
  var THEME_KEY = "visor.theme";
  var token = "";
  try { token = window.localStorage.getItem(KEY) || ""; } catch (e) { token = ""; }

  // In the order they occur, which is the order the legend and the means list
  // are read in.
  var PHASES = ["transfer", "received", "staging", "queued_gpu", "running",
                "packaging", "fetch"];
  var WHAT = {
    transfer: "bytes going up",
    received: "the server has them",
    staging: "unpacked, put in place",
    queued_gpu: "waiting for admission",
    running: "the tool computing",
    packaging: "the result zipped",
    fetch: "bytes coming down"
  };
  var DOT = String.fromCharCode(183);
  var TIMES = String.fromCharCode(215);
  var DASH = String.fromCharCode(8212);
  var CROSS = String.fromCharCode(10007);
  var SHUT = String.fromCharCode(9656);
  var OPEN = String.fromCharCode(9662);

  function el(id) { return document.getElementById(id); }
  function show(id, on) { el(id).hidden = !on; }

  function esc(v) {
    return String(v == null ? "" : v)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function cls(phase) {
    return "ph-" + (phase === "queued_gpu" ? "queued" : phase);
  }

  // A campaign file is JSON on disk that a script writes and a person can
  // edit. Nothing here may assume a field that should hold a number does: a
  // string reaching `.toFixed` throws and takes the whole page with it, and
  // arithmetic on one puts "NaN failure(s)" in front of a reader. One gate,
  // and every formatter below is total.
  function fig(v) {
    return (typeof v === "number" && isFinite(v)) ? v : null;
  }
  function count(v) { return fig(v) == null ? 0 : v; }

  // Same gate, for the two shapes a hand-edited campaign file can also bend.
  // A string where a list belongs reaches `.forEach` and throws, which in a
  // promise chain is a blank page rather than a wrong number.
  function list(v) {
    return Object.prototype.toString.call(v) === "[object Array]" ? v : [];
  }
  function obj(v) {
    return (v && typeof v === "object" &&
      Object.prototype.toString.call(v) !== "[object Array]") ? v : {};
  }

  // `stress` is present only on an arm that is a limit test, which is a
  // handful of them at most: every other arm must render exactly as it did
  // before this block existed, so every reader of it goes through here.
  function stressOf(a) {
    var s = a.stress;
    return (s && typeof s === "object" &&
      Object.prototype.toString.call(s) !== "[object Array]") ? s : null;
  }

  // Past this share of its budget, the arm met the limit it names. Below it
  // the arm ran comfortably inside a ceiling it never touched -- which is a
  // measurement of nothing, however clean its failure counts are.
  var REACHED = 0.75;

  // The card's peak over the budget it was given, or null when either side of
  // that ratio was not recorded. A zero peak is treated as "not sampled", not
  // as a card that was never touched: the cost table already learned that a
  // zero is the absence of a measurement.
  function reach(s) {
    var want = fig(obj(s.budget).vram_bytes), peak = fig(s.peak_card_bytes);
    if (want == null || peak == null || !(want > 0) || !(peak > 0)) {
      return null;
    }
    return peak / want;
  }

  // "held", "short" or "broke", for the one narrow cell the glance strip has
  // spare. A stress arm has no speedup to print there and printing a dash
  // would waste the only column that can carry the verdict.
  function glanceVerdict(a) {
    var s = stressOf(a);
    if (!s) { return null; }
    if (count(s.failed) > 0 || count(s.oom) > 0 ||
        count(obj(s.health).failures) > 0) {
      return { word: "broke", cls: " bad",
               title: "this limit test broke -- see the arm below" };
    }
    var r = reach(s);
    if (r != null && r < REACHED) {
      return { word: "short", cls: " bad",
               title: "nothing broke, but the card peaked at " +
                 Math.round(r * 100) + "% of this arm's budget: it did not " +
                 "reach the limit it names" };
    }
    return { word: "held", cls: " gain",
             title: "a limit test in which nothing broke" };
  }

  function bytes(n) {
    n = fig(n);
    if (n == null) { return ""; }
    if (n >= 1073741824) { return (n / 1073741824).toFixed(1) + " GB"; }
    if (n >= 1048576) { return (n / 1048576).toFixed(1) + " MB"; }
    if (n >= 1024) { return (n / 1024).toFixed(0) + " kB"; }
    return n + " B";
  }
  function gib(n) {
    return fig(n) == null ? "" : (n / 1073741824).toFixed(1) + "G";
  }
  function gibLong(n) {
    return fig(n) == null ? DASH : (n / 1073741824).toFixed(1) + " GiB";
  }
  // "4.4x", or nothing at all. Never a bare truthy value sent to `.toFixed`.
  function mult(v) {
    return fig(v) == null ? DASH : v.toFixed(1) + TIMES;
  }

  function secs(v) {
    v = fig(v);
    if (v == null) { return ""; }
    if (v >= 100) { return v.toFixed(0) + "s"; }
    if (v >= 10) { return v.toFixed(1) + "s"; }
    return v.toFixed(2) + "s";
  }

  function clock(v) {
    if (fig(v) == null) { return DASH; }
    if (v < 60) { return Math.round(v) + "s"; }
    var m = Math.floor(v / 60), r = Math.round(v % 60);
    return r ? m + "m" + ("0" + r).slice(-2) : m + "m";
  }

  // How dense the axis may be, which is a function of how wide the screen is:
  // the whole reason this page fills the viewport is that a 4 s wait inside a
  // 400 s run needs pixels to exist at all, and ticks it can be read against.
  function step(duration) {
    var want = Math.max(6, Math.min(22,
      Math.round((window.innerWidth || 1200) / 155)));
    var choices = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
    for (var i = 0; i < choices.length; i++) {
      if (duration / choices[i] <= want) { return choices[i]; }
    }
    return 7200;
  }

  function pct(v) {
    // Every proportion on this page goes through here, so a zero denominator
    // is caught in exactly one place: a selection over an empty catalogue
    // would otherwise put "NaN%" or "Infinity%" into a style attribute.
    if (v == null || isNaN(v) || !isFinite(v)) { return "0%"; }
    return (Math.max(0, Math.min(1, v)) * 100).toFixed(3) + "%";
  }

  function captured(epochSeconds) {
    var when = new Date(epochSeconds * 1000);
    if (isNaN(when.getTime())) { return ""; }
    var pad = function (n) { return (n < 10 ? "0" : "") + n; };
    return when.getFullYear() + "-" + pad(when.getMonth() + 1) + "-" +
      pad(when.getDate()) + " " + pad(when.getHours()) + ":" +
      pad(when.getMinutes());
  }

  // ------------------------------------------------------------------
  // Theme: the third state is "whatever the system says", and it is the
  // default. The two forced states exist because a clinician reading this
  // beside a scan viewer has already decided which they want.
  // ------------------------------------------------------------------

  var theme = "auto";
  try { theme = window.localStorage.getItem(THEME_KEY) || "auto"; } catch (e) { }
  if (theme !== "light" && theme !== "dark") { theme = "auto"; }

  function applyTheme() {
    var root = document.documentElement;
    if (theme === "auto") { root.removeAttribute("data-theme"); }
    else { root.setAttribute("data-theme", theme); }
    el("theme").textContent = theme;
  }

  // ------------------------------------------------------------------
  // The summary strip
  // ------------------------------------------------------------------

  function tile(k, v, note, small) {
    return '<div class="tile"><div class="k">' + esc(k) + '</div><div class="v' +
      (small ? " small" : "") + '">' + esc(v) + "</div>" +
      (note ? '<div class="n">' + esc(note) + "</div>" : "") + "</div>";
  }

  function limitsNote(broke, short) {
    var said = [];
    if (broke.length) { said.push(broke.join(", ") + " broke"); }
    if (short.length) {
      said.push(short.join(", ") + " never reached the budget");
    }
    return said.length ? said.join("; ") : "nothing broke";
  }

  function drawSummary(c) {
    var arms = c.arms || [], total = 0, failed = 0, wall = 0;
    var widest = 0, widestArm = "", vram = 0, vramArm = "", broke = [];
    var limits = 0, limitsBroke = [], limitsShort = [];
    arms.forEach(function (a) {
      if (stressOf(a)) {
        limits += 1;
        var verdict = glanceVerdict(a);
        if (verdict && verdict.word === "broke") {
          limitsBroke.push(a.arm || "?");
        } else if (verdict && verdict.word === "short") {
          limitsShort.push(a.arm || "?");
        }
      }
      total += count(a.total);
      wall += count(a.wall_seconds);
      var lost = count(a.total) - count(a.ok);
      failed += lost;
      if (lost > 0) { broke.push(a.arm || "?"); }
      if (count(a.peak_running) > widest) {
        widest = count(a.peak_running);
        widestArm = (a.arm || "?") + " " + DOT + " " + ((a.tools || [])[0] || "");
      }
      if (count(a.peak_vram_bytes) > vram) {
        vram = count(a.peak_vram_bytes);
        vramArm = a.arm || "?";
      }
    });
    el("strip").innerHTML =
      tile("runs", total, arms.length + " arm(s)") +
      tile("failures", failed, broke.length ? broke.join(", ") : "none") +
      tile("most at once", widest || DASH, widestArm) +
      tile("peak vram", vram ? gibLong(vram) : DASH, vramArm) +
      tile("on the clock", clock(wall), "summed over the arms") +
      // Only when a campaign holds one. A campaign of ordinary arms keeps the
      // six tiles it always had. An arm that never reached its budget is
      // named here too: "nothing broke" over an arm that pushed nothing is
      // the one wrong answer this tile could give.
      (limits ? tile("limit tests", limits, limitsNote(limitsBroke,
        limitsShort)) : "") +
      tile("hardware", c.hardware || "unstated", "", true);
    el("when").textContent = arms.length + " arm(s) " + DOT + " " + total +
      " run(s) " + DOT + " " + failed + " failure(s) " + DOT + " " +
      clock(wall) + " measured";
    show("summary", true);

    el("legend").innerHTML = PHASES.map(function (p) {
      return '<span class="lg" title="' + esc(WHAT[p]) + '"><span class="sw ' +
        cls(p) + '"></span>' + esc(p) + "</span>";
    }).join("") +
      '<span class="lg" title="a supervised tool calling another, drawn inside ' +
      'its parent&#39;s running span"><span class="sw ph-nest"></span>' +
      "nested call</span>" +
      '<span class="lg" title="how many runs were inside running at that ' +
      'instant"><span class="sw ph-conc"></span>at once</span>' +
      (limits ? '<span class="lg" title="runs admission was holding rather ' +
        'than refusing, sampled through a limit test"><span class="sw ' +
        'ph-wait"></span>held</span>' : "");
    show("legend", true);
  }

  // ------------------------------------------------------------------
  // Arms at a glance: one block per family, several abreast on a wide screen
  // ------------------------------------------------------------------

  function familyKey(a) {
    var found = /^[A-Za-z]+/.exec(String(a.arm == null ? "?" : a.arm));
    return found ? found[0].toUpperCase() : "?";
  }

  function families(arms) {
    var order = [], seen = {};
    arms.forEach(function (a) {
      var key = familyKey(a);
      if (!seen[key]) { seen[key] = { key: key, arms: [] }; order.push(seen[key]); }
      seen[key].arms.push(a);
    });
    return order;
  }

  function familyGloss(list) {
    var tools = [], clients = [];
    list.forEach(function (a) {
      (a.tools || []).forEach(function (t) {
        if (tools.indexOf(t) < 0) { tools.push(t); }
      });
      if (a.clients != null && clients.indexOf(a.clients) < 0) {
        clients.push(a.clients);
      }
    });
    clients.sort(function (p, q) { return p - q; });
    var what = !tools.length ? ""
      : tools.length <= 3 ? tools.join(", ") : tools.length + " tools";
    if (!clients.length) { return what; }
    var at = clients.join(", ") + " client" +
      (clients.length > 1 || clients[0] > 1 ? "s" : "");
    return what ? what + " at " + at : "at " + at;
  }

  function familyCard(f) {
    // One scale across the family, which is the entire point: F1 and F4 are
    // the same load with the width forced to 1, and they only say so when
    // their bars are measured against the same ruler.
    var top = 0;
    f.arms.forEach(function (a) {
      if (count(a.wall_seconds) > top) { top = count(a.wall_seconds); }
    });
    if (!(top > 0)) { top = 1; }
    var rows = f.arms.map(function (a) {
      var wall = count(a.wall_seconds), med = count(a.median_seconds);
      var peak = a.peak_running;
      var verdict = glanceVerdict(a);
      return '<div class="frow"><span class="code">' + esc(a.arm || "?") +
        '</span><span class="flab" title="' + esc(a.label || "") + '">' +
        esc(a.label || "") + "</span>" +
        '<div class="ftrack" title="' + esc("wall " + secs(wall) +
          ", median run " + secs(med)) + '">' +
        '<div class="fwall" style="width:' + pct(wall / top) +
        '"></div><div class="fmed" style="width:' + pct(med / top) +
        '"></div></div>' +
        '<div class="fn">' + esc(secs(wall)) + "</div>" +
        '<div class="fc">' + (peak == null ? DASH : esc(peak) + " at once") +
        "</div>" +
        '<div class="fs' + (verdict ? verdict.cls :
          (a.speedup > 1 ? " gain" : "")) + '" title="' +
        esc(verdict ? verdict.title : (a.speedup_basis || "")) + '">' +
        (verdict ? esc(verdict.word) : mult(a.speedup)) + "</div></div>";
    }).join("");
    return '<div class="fam"><div class="fhead"><span class="fkey">' +
      esc(f.key) + '</span><span class="fwhat">' + esc(familyGloss(f.arms)) +
      "</span></div>" + rows +
      '<div class="fcap">band: wall clock  ' + DOT +
      "  bar: median run  " + DOT + "  same scale across the family</div></div>";
  }

  // ------------------------------------------------------------------
  // One arm
  // ------------------------------------------------------------------

  // The sentence the reader asked for, in the words they asked for it in:
  // "five CLIC at once". Every other number on the card qualifies this one.
  function claim(a) {
    var peak = a.peak_running, mean = a.mean_running, tools = a.tools || [];
    if (peak == null) { return ""; }
    var what = tools.length === 1 ? tools[0]
      : tools.length ? "runs across " + tools.length + " tools" : "runs";
    var head = peak <= 1
      ? "one " + (tools.length === 1 ? tools[0] : "run") + " at a time"
      : peak + " " + what + " at once";   // escaped once, where it is placed
    var tail = [];
    if (fig(mean) != null && peak > 1) {
      tail.push(mean.toFixed(1) + " on average");
    }
    if (a.clients != null && a.clients > (peak || 0)) {
      tail.push(a.clients + " clients asked");
    }
    return '<div class="claim"><span class="dot"></span><span><b>' +
      esc(head) + "</b>" + (tail.length ? '<span class="claimx">  ' + DOT +
        "  " + esc(tail.join("  " + DOT + "  ")) + "</span>" : "") +
      "</span></div>";
  }

  function inputLine(i) {
    var bits = [];
    if (i.kind) { bits.push(i.kind); }
    if (i.files) { bits.push(i.files + " file(s)"); }
    if (i.bytes != null) { bits.push(bytes(i.bytes)); }
    return esc(i.argument) + " = " + esc(i.name) +
      (bits.length ? " (" + esc(bits.join(", ")) + ")" : "");
  }

  function setupLine(a) {
    var s = a.setup || {};
    var clients = s.clients == null ? a.clients : s.clients;
    var each = s.runs_per_client;
    if (each == null && clients) { each = Math.round((a.total || 0) / clients); }
    var tools = s.tools || a.tools || [];
    var transfer = s.transfer == null ? a.transfer : s.transfer;
    var detached = s.detached == null ? a.detached : s.detached;

    var parts = [];
    if (clients != null) {
      parts.push(clients + " client(s)" +
        (each ? " " + TIMES + " " + each + " run(s) each" : ""));
    }
    if (tools.length) { parts.push(tools.join(", ")); }
    if (transfer) { parts.push(transfer + " input(s)"); }
    parts.push(detached === false ? "blocking" : "detached");
    // `--arg batch_size=1` is the whole difference between a cohort arm and
    // its serial control. Without it printed, those two arms render as the
    // same experiment run twice.
    Object.keys(s.override || {}).forEach(function (name) {
      parts.push(name + "=" + s.override[name]);
    });
    // WHEN this arm was captured, and it earns its place on every row.
    // `campaign_report.load` keeps the newest record per arm id, so an arm
    // that failed in the latest campaign silently keeps the one before it --
    // measured on other code, against another cost table. Seventeen arms then
    // read as one campaign when one of them is from another day. Printing the
    // date per arm makes that visible without the page having to guess which
    // arms belong together.
    var stamp = a.origin ? captured(a.origin) : "";

    var inputs = (s.inputs || []).map(inputLine);

    return (stamp ? '<span class="at">captured ' + esc(stamp) + "</span>" : "") +
      '<div class="setup">' + esc(parts.join("  " + DOT + "  ")) +
      (inputs.length ? "<br>" + inputs.join("  " + DOT + "  ") : "") + "</div>";
  }

  function metrics(a) {
    var bad = count(a.ok) < count(a.total);
    return '<div class="metrics">' +
      "<span>ok <b" + (bad ? ' class="bad"' : "") + ">" + count(a.ok) +
        "/" + count(a.total) + "</b></span>" +
      "<span>wall <b>" + esc(secs(a.wall_seconds)) + "</b></span>" +
      "<span>median <b>" + esc(secs(a.median_seconds)) + "</b></span>" +
      "<span>fastest <b>" + esc(secs(a.fastest_seconds)) + "</b></span>" +
      "<span>slowest <b>" + esc(secs(a.slowest_seconds)) + "</b></span>" +
      // Green only ABOVE one. A 0.9x painted as a gain says the arm went
      // faster under load, and it went slower.
      '<span title="' + esc(a.speedup_basis || "") + '">speedup <b' +
        (a.speedup > 1 ? ' class="gain"' : "") + ">" + mult(a.speedup) +
        "</b></span>" +
      "<span>peak vram <b>" + esc(count(a.peak_vram_bytes)
        ? gib(a.peak_vram_bytes) : DASH) + "</b></span>" +
      "</div>";
  }

  // ------------------------------------------------------------------
  // A limit test: did anything break, and did it reach the limit it names?
  // ------------------------------------------------------------------

  function figLine(n, text, breakage) {
    return '<div class="sfig"><span class="num' +
      (n > 0 ? (breakage ? " bad" : "") : " zero") + '">' + esc(n) +
      "</span><span>" + esc(text) + "</span></div>";
  }

  // The three ways this arm can have broken, and nothing else in this box.
  function brokeBlock(s) {
    var health = obj(s.health);
    var checks = count(health.checks), missed = count(health.failures);
    var worst = fig(health.worst_ms);
    return '<div><div class="dk">what broke</div>' +
      figLine(count(s.failed), "runs failed", true) +
      figLine(count(s.oom), "of those ran the machine out of memory", true) +
      (checks > 0
        ? figLine(missed, "of " + checks + " health checks went unanswered" +
            (worst == null ? "" :
              " (slowest answer " + Math.round(worst) + " ms)"), true)
        : '<div class="sfig"><span class="num zero">' + DASH +
          "</span><span>health was not polled while this arm ran</span></div>") +
      '<div class="sgloss">A failure is a run a client asked for and did not ' +
      "get back. The server still answering while the arm pushed it is what " +
      "says the rest of the machine stayed usable.</div></div>";
  }

  // The two counts that look like failures in a table and are the opposite.
  function heldBlock(s) {
    return '<div><div class="dk">admission doing its job</div>' +
      figLine(count(s.refused), "runs turned away before they started", false) +
      figLine(count(s.retried), "runs started again after an out-of-memory",
        false) +
      '<div class="sgloss">Neither of these is a failure. A refusal is the ' +
      "budget saying no to a run it could not fit, which is what keeps the " +
      "machine from being oversubscribed; a retry is a run that hit an " +
      "out-of-memory and was started again rather than lost.</div></div>";
  }

  // Whether the arm reached the ceiling it was built to reach. A clean arm
  // that never came near its budget tested nothing, and this is the only
  // place on the page that can say so.
  function cardBlock(s) {
    var budget = obj(s.budget);
    var want = fig(budget.vram_bytes), peak = fig(s.peak_card_bytes);
    var ram = fig(budget.ram_bytes), cpus = fig(budget.cpus);
    var share = reach(s), meter = "", said;
    if (share == null) {
      said = (peak != null && peak > 0)
        ? "The card peaked at " + gibLong(peak) + ", with no budget recorded " +
          "to measure that against."
        : "The card was not sampled while this arm ran, so nothing here says " +
          "how close it came to its limit.";
    } else {
      meter = '<div class="meter"><div class="mfill' +
        (share < REACHED ? " shy" : "") + '" style="width:' + pct(share) +
        '"></div></div>';
      said = share < REACHED
        ? "The card peaked at " + Math.round(share * 100) + "% of that " +
          "budget. An arm that never reaches its limit has not tested it."
        : "The card peaked at " + Math.round(share * 100) + "% of that " +
          "budget, so the limit this arm names is the one it actually met.";
    }
    var given = [];
    if (want != null && want > 0) { given.push(gibLong(want) + " of card"); }
    if (ram != null && ram > 0) { given.push(gibLong(ram) + " of host memory"); }
    if (cpus != null && cpus > 0) { given.push(cpus + " core(s)"); }
    return '<div><div class="dk">the card against its budget</div>' + meter +
      '<div class="sfig"><span class="num">' +
      esc(peak != null && peak > 0 ? gibLong(peak) : DASH) +
      "</span><span>held at once, of " +
      esc(want != null && want > 0 ? gibLong(want) : "no stated budget") +
      "</span></div>" +
      '<div class="sgloss">' + esc(said) + "</div>" +
      (given.length ? '<div class="sgloss">This arm ran against ' +
        esc(given.join(", ")) + ".</div>" : "") + "</div>";
  }

  // The verdict, above the arm's own numbers. A stress arm is read for this
  // sentence and for nothing else until it has been read.
  function stressPanel(a) {
    var s = stressOf(a);
    if (!s) { return ""; }
    var failed = count(s.failed), oom = count(s.oom);
    var missed = count(obj(s.health).failures);
    var wrong = [];
    if (failed > 0) {
      wrong.push(failed + (failed === 1 ? " run failed" : " runs failed") +
        (oom > 0 ? " (" + oom + " out of memory)" : ""));
    } else if (oom > 0) {
      wrong.push(oom + " out of memory");
    }
    if (missed > 0) {
      wrong.push(missed + (missed === 1 ? " health check" : " health checks") +
        " went unanswered");
    }
    var share = reach(s), head, tone, tail = [];
    if (wrong.length) {
      tone = "warn";
      head = CROSS + " " + wrong.join(", ");
      if (share != null) {
        tail.push("at " + Math.round(share * 100) + "% of the card budget");
      }
    } else if (share == null) {
      tone = "ok";
      head = "nothing broke";
      tail.push("the card was not sampled, so how hard this arm pushed is " +
        "not recorded");
    } else if (share < REACHED) {
      tone = "warn";
      head = "nothing broke, and nothing was pushed";
      tail.push("the card peaked at " + Math.round(share * 100) +
        "% of the budget this arm names");
    } else {
      tone = "ok";
      head = "nothing broke";
      tail.push("at " + Math.round(share * 100) + "% of the card budget");
    }
    if (count(s.refused) > 0) { tail.push(count(s.refused) + " turned away"); }
    if (count(s.retried) > 0) { tail.push(count(s.retried) + " retried"); }
    return '<div class="stress"><div class="verdict"><span class="dot ' +
      tone + '"></span><span><b>' + esc(head) + "</b>" +
      (tail.length ? '<span class="claimx">  ' + DOT + "  " +
        esc(tail.join("  " + DOT + "  ")) + "</span>" : "") +
      '</span></div><div class="sgrid">' + brokeBlock(s) + heldBlock(s) +
      cardBlock(s) + "</div></div>";
  }

  function duration(a) {
    var d = 0;
    (a.runs || []).forEach(function (r) {
      (r.spans || []).forEach(function (s) {
        if (count(s.end) > d) { d = count(s.end); }
      });
      (r.nested || []).forEach(function (n) {
        if (count(n.end) > d) { d = count(n.end); }
      });
      if (count(r.started) + count(r.seconds) > d) {
        d = count(r.started) + count(r.seconds);
      }
    });
    (a.vram || []).forEach(function (p) {
      if (count(p.at) > d) { d = count(p.at); }
    });
    // A limit test can be sampled past its last run -- the queue is what it
    // was run to show, so the axis has to cover it or the trace is clipped.
    var s = stressOf(a);
    if (s) {
      list(s.queue).forEach(function (p) {
        if (count(obj(p).at) > d) { d = count(obj(p).at); }
      });
    }
    if (!(d > 0)) { d = count(a.wall_seconds) || 1; }
    return d;
  }

  function gridStyle(tick, d, height) {
    // The same vertical rule on every track, in percent, so the Gantt and the
    // traces below it are ruled identically without either knowing a pixel.
    return "background-size:" + (tick / d * 100).toFixed(4) + "% " +
      (height || "100%") + ";";
  }

  // ------------------------------------------------------------------
  // What ONE run was asked to do
  // ------------------------------------------------------------------

  function selections(r) {
    var inv = r.invocation;
    return (inv && inv.selection) || [];
  }

  // "3 of 119" -- or "all 119" when nothing was left out, which is the
  // "tous les models" end of the same question. An `available` of zero is a
  // catalogue the record could not size: the count still means something, the
  // proportion does not, so the bar is simply not drawn.
  function selectionText(s) {
    var asked = s.asked, avail = s.available;
    if (asked == null) { return DASH; }
    // Escaped even though these are counts: this page renders whatever a
    // campaign file holds, and a hand-edited one holds whatever it holds.
    if (!(avail > 0)) { return "<b>" + esc(asked) + "</b> selected"; }
    if (asked >= avail) { return "<b>all " + esc(avail) + "</b>"; }
    return "<b>" + esc(asked) + "</b> of " + esc(avail);
  }

  function selectionShare(s) {
    return (s.available > 0 && s.asked != null) ? s.asked / s.available : 0;
  }

  function selCell(r) {
    var list = selections(r);
    // Left, where the bar would start -- a dash flush right would read as a
    // missing count rather than as a tool with nothing to choose from.
    if (!list.length) { return '<div class="sc">' + DASH + "</div>"; }
    var first = list[0];
    var extra = list.length > 1 ? " +" + (list.length - 1) : "";
    var title = list.map(function (s) {
      return s.argument + ": " + (s.asked == null ? "?" : s.asked) +
        (s.available > 0 ? " of " + s.available : " selected");
    }).join("  " + DOT + "  ");
    return '<div class="sc" title="' + esc(title) + '">' +
      '<div class="sbar"><div class="sfill" style="width:' +
      pct(selectionShare(first)) + '"></div></div>' +
      '<div class="sn">' + selectionText(first) + esc(extra) + "</div></div>";
  }

  function paramValue(v) {
    if (v == null) { return null; }
    if (Object.prototype.toString.call(v) === "[object Array]") {
      return v.length ? v.join(", ") : null;
    }
    if (typeof v === "object") { return null; }
    return String(v);
  }

  function detailPanel(r, open) {
    var inv = r.invocation;
    if (!inv) { return ""; }
    var blocks = [];

    var picked = selections(r);
    if (picked.length) {
      blocks.push('<div><div class="dk">selection</div>' +
        picked.map(function (s) {
          return '<div class="sline"><span class="sk">' + esc(s.argument) +
            '</span><span class="sbig"><span class="sbar"><span class="sfill" ' +
            'style="width:' + pct(selectionShare(s)) + '"></span></span>' +
            '<span class="sn">' + selectionText(s) + "</span></span></div>";
        }).join("") + "</div>");
    }

    var io = [];
    (inv.inputs || []).forEach(function (i) { io.push(inputLine(i)); });
    if (inv.produced) {
      var made = [];
      if (inv.produced.files != null) {
        made.push(inv.produced.files + " file(s)");
      }
      if (inv.produced.bytes != null) { made.push(bytes(inv.produced.bytes)); }
      if (made.length) { io.push("produced <b>" + esc(made.join(", ")) + "</b>"); }
    }
    if (io.length) {
      blocks.push('<div><div class="dk">input and output</div>' +
        '<div class="dio">' + io.join("<br>") + "</div></div>");
    }

    var pairs = [];
    Object.keys(inv.params || {}).forEach(function (name) {
      var value = paramValue(inv.params[name]);
      if (value == null) { return; }
      pairs.push('<span class="pk">' + esc(name) + '</span><span class="pv" ' +
        'title="' + esc(value) + '">' + esc(value.slice(0, 200)) + "</span>");
    });
    if (pairs.length) {
      blocks.push('<div><div class="dk">parameters</div>' +
        '<div class="dpar">' + pairs.join("") + "</div></div>");
    }

    if (!blocks.length) {
      blocks.push('<div class="dio">This run recorded no configuration.</div>');
    }
    return '<div class="det"' + (open ? "" : " hidden") + ">" +
      blocks.join("") + "</div>";
  }

  function costCell(r) {
    // What this ONE run was measured to cost. Null on every campaign captured
    // before the runner started reporting it, which is the common case -- so
    // the column is absent rather than full of dashes, and this cell only
    // exists for an arm that has at least one measurement.
    var m = r.measured;
    if (!m) { return '<div class="rc">' + DASH + "</div>"; }
    var bits = [];
    if (m.vram_bytes != null) { bits.push("vram " + gibLong(m.vram_bytes)); }
    if (m.ram_bytes != null) { bits.push("ram " + gibLong(m.ram_bytes)); }
    if (m.cpu_cores != null) { bits.push(m.cpu_cores + " core(s)"); }
    if (m.channels != null) { bits.push(m.channels + " channel(s)"); }
    var head = m.vram_bytes != null ? gib(m.vram_bytes)
      : m.ram_bytes != null ? gib(m.ram_bytes) + " ram" : DASH;
    return '<div class="rc" title="' + esc(bits.join("  " + DOT + "  ")) + '">' +
      esc(head) + "</div>";
  }

  // Which optional columns this arm's records can fill. Decided per arm, not
  // per run, so every row of one Gantt keeps the same column edges.
  function columns(a) {
    var cols = { sel: false, cost: false };
    (a.runs || []).forEach(function (r) {
      if (r.measured) { cols.cost = true; }
      if (selections(r).length) { cols.sel = true; }
    });
    return cols;
  }

  // Which panels the reader has opened, kept across a redraw -- the page
  // re-renders on a resize and on a campaign change, and losing every open
  // panel to a window drag would make comparing two runs impossible. Keyed by
  // arm, client and index: the three things that identify a row, none of
  // which is a capability to read anything.
  var opened = {};
  function rowKey(a, r) {
    return (a.arm || "?") + "/" + (r.client == null ? "?" : r.client) + "/" +
      (r.index == null ? "?" : r.index);
  }

  function runRow(a, r, d, tick, cols) {
    // Client and index identify the row. No id for the run appears anywhere on
    // this page: it is the capability to read that run's result.
    var label = "c" + (r.client == null ? "?" : r.client) + DOT +
      (r.index == null ? "?" : r.index);
    var failed = r.status && r.status !== "ok";
    var bars = (r.spans || []).map(function (s) {
      var w = Math.max(0, (s.end - s.start));
      return '<div class="ph ' + cls(s.phase) + '" style="left:' +
        pct(s.start / d) + ";width:" + pct(w / d) + '" title="' +
        esc(s.phase + "  " + secs(w)) + '"></div>';
    }).join("");
    var nested = (r.nested || []).map(function (n) {
      var w = Math.max(0, (n.end - n.start));
      return '<div class="nb' + ((n.depth || 1) > 1 ? " d2" : "") +
        '" style="left:' + pct(n.start / d) + ";width:" + pct(w / d) +
        // The name is the point of the bar: "ASO spent 71% of itself inside
        // ALI_CBCT" is the sentence. It is null on a campaign captured before
        // the supervisor emitted its markers, and then the depth is all there
        // is to say.
        '" title="' + esc((n.tool ? "calls " + n.tool : "nested call") +
        ", depth " + (n.depth || 1) + "  " + secs(w)) + '"></div>';
    }).join("");

    var key = rowKey(a, r);
    var canOpen = !!r.invocation;
    var isOpen = canOpen && !!opened[key];
    var inner = (failed ? CROSS + " " : "") +
      (canOpen ? '<span class="caret">' + (isOpen ? OPEN : SHUT) + "</span>" : "") +
      '<span class="who">' + esc(label) + "</span> <b>" +
      esc(r.tool || "?") + "</b>";
    var head = canOpen
      ? '<button type="button" class="rl" data-open="' + esc(key) +
        '" aria-expanded="' + (isOpen ? "true" : "false") + '" title="' +
        esc(label + " " + (r.tool || "") + " -- what this run was asked to do") +
        '">' + inner + "</button>"
      : '<div class="rl" title="' + esc(label + " " + (r.tool || "")) + '">' +
        inner + "</div>";

    return '<div class="grow">' + head +
      '<div class="rt" style="' + gridStyle(tick, d) + '">' + bars + nested +
      "</div>" +
      (cols.sel ? selCell(r) : "") +
      '<div class="rv' + (failed ? " bad" : "") + '">' +
      esc(secs(r.seconds)) + "</div>" +
      (cols.cost ? costCell(r) : "") +
      (failed && r.error
        ? '<div class="rerr" title="' + esc(r.error) + '">' +
          esc(String(r.error).slice(0, 160)) + "</div>"
        : "") +
      detailPanel(r, isOpen) +
      "</div>";
  }

  // A step path over [{at, v}], as an area and as a line. Discrete
  // observations joined with a slope would draw a ramp nobody measured.
  function stepPaths(points, d, top) {
    var W = 1000, H = 100, PAD = 3;
    function y(v) { return (H - PAD) - (v / top) * (H - 2 * PAD); }
    function x(t) { return Math.max(0, Math.min(W, (t / d) * W)); }
    var area = "M " + x(points[0].at).toFixed(2) + " " + H.toFixed(2);
    var line = "M " + x(points[0].at).toFixed(2) + " " +
      y(points[0].v).toFixed(2);
    var last = points[0];
    points.forEach(function (p, i) {
      var px = x(p.at).toFixed(2);
      if (i > 0) {
        line += " L " + px + " " + y(last.v).toFixed(2);
        area += " L " + px + " " + y(last.v).toFixed(2);
      }
      line += " L " + px + " " + y(p.v).toFixed(2);
      area += " L " + px + " " + y(p.v).toFixed(2);
      last = p;
    });
    area += " L " + x(last.at).toFixed(2) + " " + H.toFixed(2) + " Z";
    return { line: line, area: area };
  }

  // `layers` are painted in the order given, so a stacked trace is drawn
  // total first and the lower band over it: what stays visible is exactly the
  // difference between the two, without either path having to know the other.
  function traceRow(label, value, layers, tick, d, cols, aria, short) {
    return '<div class="grow"><div class="rl"><b>' + esc(label) + "</b></div>" +
      '<div class="rt trace' + (short ? " short" : "") + '" style="' +
      gridStyle(tick, d) + '">' +
      '<svg viewBox="0 0 1000 100" preserveAspectRatio="none" role="img" ' +
      'aria-label="' + esc(aria) + '">' +
      layers.map(function (layer) {
        return '<path d="' + layer.paths.area + '" style="fill:var(' +
          layer.fill + ');stroke:none"></path>' +
          '<path d="' + layer.paths.line + '" style="fill:none;stroke:var(' +
          layer.line + ');stroke-width:1.5" vector-effect="non-scaling-stroke">' +
          "</path>";
      }).join("") +
      "</svg></div>" +
      (cols.sel ? "<div></div>" : "") +
      '<div class="rv">' + esc(value) + "</div>" +
      (cols.cost ? '<div class="rc"></div>' : "") + "</div>";
  }

  // How many runs were inside `running` at each instant. This is the overlap
  // the stacked rows show, counted -- and it is what separates "six ran at
  // once" from "one ran while five waited", which cost identical wall clock.
  function concurrency(runs) {
    var events = [];
    runs.forEach(function (r) {
      (r.spans || []).forEach(function (s) {
        if (s.phase !== "running") { return; }
        events.push({ t: s.start, k: 1 });
        events.push({ t: s.end, k: -1 });
      });
    });
    if (!events.length) { return []; }
    // Ends before starts at the same instant: a run handing over to the next
    // is one at a time, not two.
    events.sort(function (p, q) { return p.t === q.t ? p.k - q.k : p.t - q.t; });
    var open = 0, points = [{ at: 0, v: 0 }];
    events.forEach(function (e) {
      open += e.k;
      points.push({ at: e.t, v: open });
    });
    points.push({ at: events[events.length - 1].t, v: 0 });
    return points;
  }

  function concRow(a, d, tick, cols) {
    var points = concurrency(a.runs || []);
    if (points.length < 3) { return ""; }
    var top = 0;
    points.forEach(function (p) { if (p.v > top) { top = p.v; } });
    // A flat line at one says nothing a single run row does not already say.
    if (top < 2) { return ""; }
    return traceRow("at once", top + " peak",
      [{ paths: stepPaths(points, d, top), fill: "--conc-fill",
         line: "--conc-line" }], tick, d, cols,
      "how many runs were computing at the same time", true);
  }

  // The queue a limit test was run to produce: what the server was RUNNING
  // and what admission was HOLDING, sampled through the arm and drawn on the
  // Gantt's own x-scale -- because the point is to read the depth against the
  // runs that caused it. A rising held band over a flat running one is
  // admission holding the line; both rising is the machine absorbing it; the
  // running band falling to nothing mid-arm is something dying.
  function queueRow(a, d, tick, cols) {
    var s = stressOf(a);
    if (!s) { return ""; }
    var points = list(s.queue).map(function (p) {
      p = obj(p);
      return { at: count(p.at), run: Math.max(0, count(p.running)),
               wait: Math.max(0, count(p.waiting)) };
    });
    if (points.length < 2) { return ""; }
    points.sort(function (p, q) { return p.at - q.at; });
    var top = 0, heldPeak = 0, runPeak = 0;
    points.forEach(function (p) {
      if (p.run + p.wait > top) { top = p.run + p.wait; }
      if (p.wait > heldPeak) { heldPeak = p.wait; }
      if (p.run > runPeak) { runPeak = p.run; }
    });
    if (!(top > 0)) { top = 1; }
    var total = stepPaths(points.map(function (p) {
      return { at: p.at, v: p.run + p.wait };
    }), d, top);
    var running = stepPaths(points.map(function (p) {
      return { at: p.at, v: p.run };
    }), d, top);
    return traceRow("queue", heldPeak + " held",
      [{ paths: total, fill: "--p-queued-alt", line: "--p-queued" },
       { paths: running, fill: "--conc-fill", line: "--conc-line" }],
      tick, d, cols,
      "runs in the server over time: up to " + runPeak + " computing and up " +
      "to " + heldPeak + " held by admission", false);
  }

  function vramRow(a, d, tick, cols) {
    var points = (a.vram || []).map(function (p) {
      return { at: p.at, v: p.bytes };
    });
    if (points.length < 2) { return ""; }
    var top = count(a.peak_vram_bytes);
    points.forEach(function (p) { if (p.v > top) { top = p.v; } });
    if (!(top > 0)) { top = 1; }
    return traceRow("VRAM", gib(top),
      [{ paths: stepPaths(points, d, top), fill: "--vram-fill",
         line: "--vram-line" }], tick, d, cols,
      "VRAM held over the arm", false);
  }

  function headRow(cols) {
    return '<div class="grow ghead"><div class="rl">run</div><div></div>' +
      (cols.sel ? '<div title="how much of what the tool offers this run ' +
        'asked for">asked</div>' : "") +
      '<div class="rv">seconds</div>' +
      (cols.cost ? '<div class="rc" title="what this run was measured to ' +
        'cost">measured</div>' : "") + "</div>";
  }

  function axisRow(d, tick, cols) {
    var marks = "";
    for (var t = 0; t <= d + 0.001; t += tick) {
      var at = t / d;
      marks += '<span class="tick' + (at < 0.02 ? " first" :
        (at > 0.98 ? " last" : "")) + '" style="left:' + pct(at) + '">' +
        esc(clock(t)) + "</span>";
    }
    return '<div class="grow"><div class="rl"></div><div class="axis" style="' +
      gridStyle(tick, d, "6px") + '">' + marks + "</div>" +
      (cols.sel ? "<div></div>" : "") +
      '<div class="rv"></div>' +
      (cols.cost ? '<div class="rc"></div>' : "") + "</div>";
  }

  function meansRows(a) {
    var phases = a.phases || {};
    var top = 0;
    PHASES.forEach(function (p) {
      if (phases[p] != null && phases[p] > top) { top = phases[p]; }
    });
    if (!(top > 0)) { return ""; }
    return '<div class="means">' + PHASES.filter(function (p) {
      return phases[p] != null;
    }).map(function (p) {
      return '<div class="mk">' + esc(p) + '</div><div class="mt">' +
        '<div class="mf ' + cls(p) + '" style="width:' +
        pct(phases[p] / top) + '"></div></div><div class="mv">' +
        esc(secs(phases[p])) + "</div>";
    }).join("") + "</div>";
  }

  // "ASO spent 71% of itself inside ALI_CBCT" -- the one sentence the nested
  // strip along the bottom of a row is drawn to support. Depth 1 only: a
  // deeper call is already inside the one above it, and adding the two would
  // count the same seconds twice.
  function insideLine(a) {
    var runs = a.runs || [], total = 0, inside = {}, order = [];
    runs.forEach(function (r) {
      total += count(r.seconds);
      (r.nested || []).forEach(function (n) {
        if ((n.depth || 1) !== 1) { return; }
        var name = n.tool || "a nested call";
        if (inside[name] == null) { inside[name] = 0; order.push(name); }
        inside[name] += Math.max(0, count(n.end) - count(n.start));
      });
    });
    if (!order.length || !(total > 0)) { return ""; }
    var who = [];
    runs.forEach(function (r) {
      if (r.tool && who.indexOf(r.tool) < 0) { who.push(r.tool); }
    });
    var subject = who.length === 1 ? who[0] : "these runs";
    order.sort(function (p, q) { return inside[q] - inside[p]; });
    return '<div class="inside">' + esc(subject) + " spent " +
      order.map(function (name) {
        return "<b>" + Math.round(inside[name] / total * 100) + "%</b> of " +
          "itself inside " + esc(name);
      }).join(", and ") + ".</div>";
  }

  function armCard(a) {
    var d = duration(a), tick = step(d), cols = columns(a);
    var runs = (a.runs || []).slice().sort(function (p, q) {
      if ((p.started || 0) !== (q.started || 0)) {
        return (p.started || 0) - (q.started || 0);
      }
      if ((p.client || 0) !== (q.client || 0)) {
        return (p.client || 0) - (q.client || 0);
      }
      return (p.index || 0) - (q.index || 0);
    });
    var shape = (cols.sel ? " sel" : "") + (cols.cost ? " cost" : "");
    // A limit test can carry a queue and no per-run detail at all, and that
    // queue is the whole of what it was run to show.
    var queue = queueRow(a, d, tick, cols);
    var gantt = (runs.length || queue)
      ? '<div class="scroll"><div class="gantt' + shape + '">' +
        headRow(cols) + runs.map(function (r) {
          return runRow(a, r, d, tick, cols);
        }).join("") + concRow(a, d, tick, cols) + queue +
        vramRow(a, d, tick, cols) + axisRow(d, tick, cols) + "</div></div>"
      : '<div class="empty">This arm recorded no per-run detail. Its summary ' +
        "numbers above are all there is.</div>";
    return '<div class="arm"><div class="armhead"><span class="code">' +
      esc(a.arm || "?") + '</span><span class="name">' + esc(a.label || "") +
      "</span>" + setupLine(a) + "</div>" + stressPanel(a) + claim(a) +
      metrics(a) + gantt + insideLine(a) + meansRows(a) + "</div>";
  }

  // ------------------------------------------------------------------
  // Coverage
  // ------------------------------------------------------------------

  function drawCoverage(c) {
    var cov = c.coverage || [];
    if (!cov.length) { show("cover", false); return; }
    var top = 0;
    cov.forEach(function (t) {
      if (count(t.seconds) > top) { top = count(t.seconds); }
    });
    if (!(top > 0)) { top = 1; }
    el("coverbody").innerHTML =
      '<div class="scroll"><table><thead><tr><th>tool</th><th>result</th>' +
      "<th>seconds</th><th></th><th>vram</th><th>ram</th><th>out</th>" +
      "</tr></thead><tbody>" +
      cov.map(function (t) {
        var ok = t.status === "ok";
        return "<tr><td>" + esc(t.tool) + '</td><td><span class="tag ' +
          (ok ? "run" : "wait") + '">' + esc(t.status) + "</span>" +
          (t.error ? ' <span class="bad">' + esc(String(t.error).slice(0, 160)) +
            "</span>" : "") +
          '</td><td class="n">' + esc(secs(t.seconds)) +
          '</td><td class="tbar"><div class="mt"><div class="mf ph-running" ' +
          'style="width:' + pct(count(t.seconds) / top) + '"></div></div>' +
          '</td><td class="n">' + esc(gib(t.vram_bytes)) +
          '</td><td class="n">' + esc(gib(t.ram_bytes)) +
          '</td><td class="n">' + esc(bytes(t.bytes)) + "</td></tr>";
      }).join("") + "</tbody></table></div>" +
      ((c.out_of_scope || []).length
        ? '<div class="empty">Not run, deliberately: ' +
          esc(c.out_of_scope.join(", ")) + ".</div>"
        : "");
    show("cover", true);
  }

  // ------------------------------------------------------------------
  // The campaign picker: chrome, not the subject
  // ------------------------------------------------------------------

  var wanted = "";
  try {
    var asked = /[?&]campaign=([^&]*)/.exec(window.location.search || "");
    if (asked && asked[1]) { wanted = decodeURIComponent(asked[1]); }
  } catch (e) { wanted = ""; }

  function remember(name) {
    // replaceState, not pushState: choosing a campaign is not a navigation the
    // back button should have to undo. The URL is relative, so a shared link
    // resolves against whatever host and prefix served this page.
    try {
      window.history.replaceState(null, "",
        name ? "?campaign=" + encodeURIComponent(name) :
          window.location.pathname);
    } catch (e) { /* a browser with history disabled still renders */ }
  }

  function drawPicker(list, current) {
    var options = list || [];
    var stamp = "";
    if (current && current.generated_at) {
      stamp = "captured " + captured(current.generated_at);
    }
    if (current && current.source) {
      stamp = (stamp ? stamp + "  " + DOT + "  " : "") + current.source;
    }
    el("stamp").textContent = stamp;
    // One campaign is not a choice, and a select holding a single option reads
    // as a control that does nothing. An older server sends no list at all,
    // and the stamp above already says which campaign this is.
    if (options.length < 2) { show("pickwrap", false); return; }
    el("pick").innerHTML = options.map(function (o) {
      var bits = [];
      var key = /^[A-Za-z0-9]+/.exec(String(o.source || ""));
      if (key) { bits.push(key[0]); }
      if (o.generated_at) { bits.push(captured(o.generated_at)); }
      bits.push((o.arms || 0) + " arm(s)");
      if (o.bytes) { bits.push(bytes(o.bytes)); }
      var chosen = current && o.source === current.source;
      return '<option value="' + esc(o.source) + '"' +
        (chosen ? " selected" : "") + ">" + esc(bits.join("  " + DOT + "  ")) +
        "</option>";
    }).join("");
    el("pick").value = (current && current.source) || options[0].source;
    show("pickwrap", true);
  }

  // ------------------------------------------------------------------
  // Loading
  // ------------------------------------------------------------------

  var current = null;

  function draw(c) {
    current = c;
    if (!c) {
      ["summary", "glance", "legend", "arms", "cover"].forEach(function (id) {
        show(id, false);
      });
      show("none", true);
      el("when").textContent = "nothing measured here";
      return;
    }
    show("none", false);
    drawSummary(c);
    var arms = c.arms || [];
    var groups = families(arms);
    el("glancebody").innerHTML = groups.map(familyCard).join("");
    show("glance", arms.length > 0);
    el("armbody").innerHTML = arms.length
      ? groups.map(function (f) {
          return '<div class="fdiv"><span class="fkey">' + esc(f.key) +
            '</span><span class="fwhat">' + esc(familyGloss(f.arms)) +
            "</span></div>" + f.arms.map(armCard).join("");
        }).join("")
      : '<div class="empty">The campaign recorded no arm.</div>';
    show("arms", true);
    drawCoverage(c);
  }

  var retried = false;

  function load() {
    if (!token) {
      show("gate", true);
      el("when").textContent = "waiting for a token";
      return;
    }
    // Relative on purpose: resolved against /benchmarks/view this is
    // /benchmarks, and it stays correct behind a proxy on a sub-path.
    var where = "../benchmarks" +
      (wanted ? "?campaign=" + encodeURIComponent(wanted) : "");
    fetch(where, { headers: { Authorization: "Bearer " + token } })
      .then(function (r) {
        if (r.status === 401) { throw new Error("That token was refused."); }
        if (r.status === 404) {
          // The server deliberately does not fall back to the newest: a reader
          // who asked for yesterday and silently got today would compare two
          // campaigns believing they were one. So the page says so, once, and
          // then asks for the newest itself.
          var gone = wanted;
          wanted = "";
          remember("");
          if (!retried) {
            retried = true;
            load();
            throw new Error("");
          }
          throw new Error("No campaign named " + gone + " on this server.");
        }
        if (!r.ok) { throw new Error("The server answered " + r.status + "."); }
        return r.json();
      })
      .then(function (d) {
        if (!d) { return; }
        show("gate", false);
        drawPicker(d.campaigns, d.campaign);
        draw(d.campaign);
      })
      .catch(function (e) {
        if (!e.message) { return; }
        show("gate", true);
        el("gateerr").textContent = e.message;
        el("when").textContent = "not connected";
      });
  }

  el("save").addEventListener("click", function () {
    token = el("token").value.trim();
    try { window.localStorage.setItem(KEY, token); } catch (e) { /* private window */ }
    el("gateerr").textContent = "";
    load();
  });
  el("token").addEventListener("keydown", function (e) {
    if (e.key === "Enter") { el("save").click(); }
  });
  el("pick").addEventListener("change", function () {
    wanted = el("pick").value;
    remember(wanted);
    retried = true;
    load();
  });
  el("theme").addEventListener("click", function () {
    theme = theme === "auto" ? "light" : theme === "light" ? "dark" : "auto";
    try { window.localStorage.setItem(THEME_KEY, theme); } catch (e) { }
    applyTheme();
  });
  // One listener for every run row, bound once to a container the redraw
  // fills rather than replaces -- so a campaign change does not leave a
  // hundred dead handlers behind.
  el("armbody").addEventListener("click", function (e) {
    var button = e.target && e.target.closest
      ? e.target.closest("[data-open]") : null;
    if (!button) { return; }
    var row = button.parentNode;
    var panel = row && row.querySelector ? row.querySelector(".det") : null;
    if (!panel) { return; }
    var nowOpen = panel.hidden;
    panel.hidden = !nowOpen;
    button.setAttribute("aria-expanded", nowOpen ? "true" : "false");
    var caret = button.querySelector ? button.querySelector(".caret") : null;
    if (caret) { caret.textContent = nowOpen ? OPEN : SHUT; }
    var key = button.getAttribute("data-open");
    if (nowOpen) { opened[key] = true; } else { delete opened[key]; }
  });
  // How many ticks an axis can carry depends on how wide the window is, so a
  // resize is a redraw. Debounced: a drag fires this continuously.
  var pending = 0;
  window.addEventListener("resize", function () {
    if (!current) { return; }
    clearTimeout(pending);
    pending = setTimeout(function () { draw(current); }, 180);
  });

  applyTheme();
  load();
}());
</script>
</body>
</html>
"""
