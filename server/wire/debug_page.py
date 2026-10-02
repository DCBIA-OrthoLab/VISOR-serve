"""The page `GET /server-debug` serves: the operator's view of this server.

What it answers, top to bottom:

    is the machine busy   ->  what is waiting  ->  what is running, and inside
    which tool  ->  how each tool behaves over time  ->  what already finished

**Laid out along the path a run takes.** The vitals strip says how loaded the
machine is; the queue, the running runs and the machine sit side by side
because "a run is slow" has four answers and they live at four points of that
path -- nobody has asked yet, it is waiting at the gate, it is computing, or it
is writing. The tools strip and the history are below, because they answer the
slower question of how this server behaves over a day.

**A run that calls another tool says so.** AREG drives AMASSS, ASO and, through
ASO, ALI; the supervisor writes a marker around every such call, and the page
draws the chain open right now (`AREG > ASO > ALI_CBCT`) on the run's card and
every call as a bar under the run's own timeline.

**Detail opens in front of the page.** A run or a tool opens a dialog over the
dashboard rather than a drawer beside it: the timeline of a run and the graphs
of a tool need the width, and the dashboard behind keeps polling.

Rules carried over unchanged, each a thing the page deliberately does NOT do:

* **No argument value and no file name, ever.** A value is a path and a path is
  a patient's file name. A run's detail names its arguments, the SHAPE of what
  was uploaded (files, bytes, extensions) and the hosted bundles this
  deployment staged itself. Console lines are composed by the server from the
  phase, never quoted from what a tool printed.
* **A tank shows held AND used.** Admission knows what it handed out, the
  machine knows what is being consumed; the gap between the two is the
  reading.
* **It says "requests", not "clients".** HTTP has no session.
* **An unavailable reading is hatched, never drawn as zero.** Empty reads as
  idle, and idle is a claim.
* **It admits what it cannot see.** The trace is in memory and resets with the
  server; the history survives restarts but is bounded. Both say so.

Self-contained, like the status and benchmark pages, and for the same reason:
this server runs on networks with no internet, so a stylesheet or a font from a
CDN is a broken page rather than a degraded one. No library, no build step, and
no token in the served HTML -- the reader types one and the browser keeps it.
"""

from __future__ import annotations

DEBUG_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Server dashboard</title>
<style>
  :root {
    --bg: #f3f4f6; --panel: #ffffff; --sunk: #f6f7f9; --line: #e3e6ea;
    --ink: #14171c; --soft: #5f6773; --ghost: #9aa1ab;
    --bar: #e8ebef; --accent: #2f6fdb; --accent-soft: #e6eefc;
    --ok: #1f9d6b; --warn: #d4860f; --hot: #d14343; --violet: #7a5bd6;
    --staging: #8a96a8;
    --shadow: 0 1px 2px rgba(16,24,40,.05), 0 4px 16px rgba(16,24,40,.06);
    --shadow-lg: 0 24px 64px rgba(16,24,40,.22);
    --radius: 12px;
  }
  :root[data-theme="dark"] {
    --bg: #0d1014; --panel: #151a20; --sunk: #11151a; --line: #262d36;
    --ink: #e8ebef; --soft: #9aa3ae; --ghost: #5f6873;
    --bar: #222932; --accent: #6ea2ff; --accent-soft: #1b2638;
    --ok: #3fc48d; --warn: #f0a63a; --hot: #f07070; --violet: #a68cf5;
    --staging: #7d8898;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 4px 16px rgba(0,0,0,.3);
    --shadow-lg: 0 24px 64px rgba(0,0,0,.6);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #0d1014; --panel: #151a20; --sunk: #11151a; --line: #262d36;
      --ink: #e8ebef; --soft: #9aa3ae; --ghost: #5f6873;
      --bar: #222932; --accent: #6ea2ff; --accent-soft: #1b2638;
      --ok: #3fc48d; --warn: #f0a63a; --hot: #f07070; --violet: #a68cf5;
      --staging: #7d8898;
      --shadow: 0 1px 2px rgba(0,0,0,.4), 0 4px 16px rgba(0,0,0,.3);
      --shadow-lg: 0 24px 64px rgba(0,0,0,.6);
    }
  }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  html, body { min-height: 100%; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  body.locked { overflow: hidden; }
  .mono { font-variant-numeric: tabular-nums; }
  button, input, select { font: inherit; color: inherit; }
  button { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
           padding: 6px 12px; cursor: pointer; transition: border-color .15s, background .15s; }
  button:hover { border-color: var(--soft); }
  button.ghost { border-color: transparent; background: transparent; }
  button.ghost:hover { background: var(--sunk); border-color: var(--line); }
  input { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
          padding: 7px 11px; min-width: 220px; outline: none; }
  input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }

  /* ---- shell ---------------------------------------------------------- */
  #shell { max-width: 1680px; margin: 0 auto; padding: 16px 20px 28px;
           display: flex; flex-direction: column; gap: 14px; }
  header { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  #flag { width: 10px; height: 38px; border-radius: 5px; background: var(--ghost); flex: none; }
  h1 { font-size: 20px; font-weight: 700; letter-spacing: -.015em; margin: 0; line-height: 1.15; }
  #host { color: var(--soft); font-size: 12px; }
  .grow { flex: 1; }
  #live { display: inline-flex; align-items: center; gap: 7px; font-size: 12px; color: var(--soft); }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ok);
         box-shadow: 0 0 0 3px color-mix(in srgb, var(--ok) 22%, transparent); }
  .dot.off { background: var(--ghost); box-shadow: none; }
  #filters { display: flex; gap: 6px; flex-wrap: wrap; }
  .fchip { display: inline-flex; align-items: center; gap: 6px; font-size: 12px;
           padding: 4px 9px 4px 11px; border-radius: 999px; background: var(--accent-soft);
           color: var(--accent); cursor: pointer; }
  .fchip i { font-style: normal; opacity: .7; }

  .card { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius);
          box-shadow: var(--shadow); display: flex; flex-direction: column; min-width: 0; }
  .card > h2 { margin: 0; padding: 13px 16px 11px; font-size: 13px; font-weight: 650;
               display: flex; align-items: center; gap: 8px; border-bottom: 1px solid var(--line); }
  .card > h2 .count { font-size: 11.5px; font-weight: 600; color: var(--soft);
                      background: var(--sunk); border: 1px solid var(--line);
                      border-radius: 999px; padding: 0 8px; }
  .card > h2 .note { margin-left: auto; font-weight: 400; color: var(--ghost); font-size: 11.5px; }
  .body { padding: 12px 14px; min-height: 0; }
  .scroll { overflow: auto; }
  .empty { color: var(--ghost); font-size: 12.5px; padding: 14px 4px; text-align: center; }
  .label { font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: var(--ghost);
           font-weight: 600; }

  /* ---- vitals --------------------------------------------------------- */
  #vitals { display: grid; gap: 12px; grid-template-columns: repeat(6, minmax(0, 1fr)); }
  @media (max-width: 1200px) { #vitals { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
  @media (max-width: 640px) { #vitals { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  .vital { padding: 12px 14px 8px; position: relative; overflow: hidden; }
  .vital .v { font-size: 26px; font-weight: 700; letter-spacing: -.02em; line-height: 1.1; margin-top: 4px; }
  .vital .v small { font-size: 13px; font-weight: 500; color: var(--soft); margin-left: 3px; }
  .vital .f { font-size: 11.5px; color: var(--soft); margin-top: 1px; white-space: nowrap;
              overflow: hidden; text-overflow: ellipsis; }
  .vital svg { display: block; width: 100%; height: 30px; margin-top: 6px; }
  .vital.busy .v { color: var(--accent); } .vital.wait .v { color: var(--warn); }
  .vital.hot .v { color: var(--hot); }

  /* ---- main grid ------------------------------------------------------- */
  #main { display: grid; gap: 14px; grid-template-columns: minmax(250px, 300px) minmax(0, 1fr) minmax(320px, 400px); }
  @media (max-width: 1200px) { #main { grid-template-columns: 1fr; } }
  #left { display: flex; flex-direction: column; gap: 14px; min-width: 0; }

  /* the queue */
  .qitem { display: flex; gap: 10px; align-items: center; padding: 8px 10px;
           border: 1px solid var(--line); border-radius: 10px; margin-bottom: 7px; cursor: pointer;
           background: color-mix(in srgb, var(--warn) 6%, var(--panel)); }
  .qitem:hover { border-color: var(--warn); }
  .qitem .pos { width: 24px; height: 24px; border-radius: 50%; flex: none; display: grid;
                place-items: center; font-size: 12px; font-weight: 700; color: #fff; background: var(--warn); }
  .qitem .n { font-weight: 650; font-size: 13.5px; }
  .qitem .m { font-size: 11.5px; color: var(--soft); }
  .qitem.held { background: var(--sunk); border-style: dashed; }
  .qitem.slot { background: var(--sunk); }
  .qitem.slot .pos { background: var(--staging); }
  .qitem.held .pos { background: var(--ghost); }
  .rank { display: flex; align-items: center; gap: 8px; margin-top: 6px; font-size: 12px; }
  .rank .nm { width: 92px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .track { flex: 1; height: 6px; background: var(--bar); border-radius: 3px; overflow: hidden; }
  .track i { display: block; height: 100%; border-radius: 3px; background: var(--warn); }
  .rank .vv { color: var(--soft); min-width: 46px; text-align: right; }

  /* running */
  .runs { display: grid; gap: 10px; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); }
  .run { border: 1px solid var(--line); border-radius: 11px; padding: 11px 13px 12px;
         background: var(--panel); cursor: pointer; position: relative; overflow: hidden;
         transition: border-color .15s, transform .15s, box-shadow .15s; }
  .run:hover { border-color: var(--accent); box-shadow: var(--shadow); transform: translateY(-1px); }
  .run::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--ok); }
  .run.q::before { background: var(--warn); } .run.s::before { background: var(--staging); }
  .run .hd { display: flex; align-items: center; gap: 8px; }
  .run .nm { font-weight: 700; font-size: 15px; }
  .run .pc { margin-left: auto; font-size: 15px; font-weight: 700; color: var(--ok); }
  .run.q .pc { color: var(--warn); }
  .run .sub { font-size: 11.5px; color: var(--soft); margin-top: 3px; display: flex; gap: 6px;
              align-items: center; flex-wrap: wrap; }
  .progress { height: 6px; background: var(--bar); border-radius: 3px; overflow: hidden; margin-top: 10px; }
  .progress i { display: block; height: 100%; background: var(--ok); border-radius: 3px; transition: width .5s ease; }
  .progress.indet i { width: 30% !important; animation: slide 1.4s ease-in-out infinite; }
  .run.q .progress i { background: var(--warn); }
  @keyframes slide { 0% { transform: translateX(-100%); } 100% { transform: translateX(340%); } }
  .chain { display: flex; align-items: center; gap: 4px; flex-wrap: wrap; margin-top: 8px; font-size: 12px; }
  .chain .link { padding: 1px 8px; border-radius: 999px; background: var(--sunk); border: 1px solid var(--line); }
  .chain .link.root { font-weight: 650; }
  .chain .link.now { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); font-weight: 650; }
  .chain .sep { color: var(--ghost); }
  .cells { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; margin-top: 10px; }
  .cell { background: var(--sunk); border-radius: 7px; padding: 4px 7px; }
  .cell u { display: block; font-size: 9.5px; letter-spacing: .06em; color: var(--ghost);
            text-transform: uppercase; text-decoration: none; }
  .cell b { font-size: 12.5px; font-weight: 600; }

  /* machine */
  .tank { margin-bottom: 12px; }
  .tank .lab { display: flex; align-items: baseline; gap: 6px; font-size: 12px; color: var(--soft); }
  .tank .lab b { margin-left: auto; font-size: 15px; font-weight: 700; color: var(--ink); }
  .tank .tr { position: relative; height: 10px; background: var(--bar); border-radius: 5px; overflow: hidden; margin-top: 4px; }
  .tank .tr i { position: absolute; inset: 0 auto 0 0; background: var(--accent); border-radius: 5px;
                opacity: .35; transition: width .4s ease; }
  .tank .tr s { position: absolute; inset: 0 auto 0 0; background: var(--accent); border-radius: 5px;
                transition: width .4s ease; text-decoration: none; }
  .tank .tr.hot s { background: var(--hot); }
  .tank .tr.na { background: repeating-linear-gradient(45deg, var(--bar) 0 5px, var(--panel) 5px 10px); }
  .tank .ft { font-size: 11px; color: var(--ghost); margin-top: 3px; }
  .legend { display: flex; gap: 12px; flex-wrap: wrap; font-size: 11.5px; color: var(--soft); margin: 2px 0 6px; }
  .legend i { display: inline-block; width: 10px; height: 3px; border-radius: 2px; margin-right: 5px; vertical-align: middle; }
  svg.chart { display: block; width: 100%; }
  svg.chart text { font: 10px ui-sans-serif, system-ui, sans-serif; fill: var(--ghost); }
  .disk { display: flex; align-items: center; gap: 8px; font-size: 12px; margin-top: 6px; }
  .disk .nm { width: 44px; color: var(--soft); }
  .disk .vv { color: var(--soft); min-width: 120px; text-align: right; }
  .disk .track i { background: var(--accent); }

  /* tools */
  #tools .grid { display: grid; gap: 10px; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); }
  .tool { border: 1px solid var(--line); border-radius: 11px; padding: 11px 13px; cursor: pointer;
          background: var(--panel); transition: border-color .15s, box-shadow .15s, transform .15s; }
  .tool:hover { border-color: var(--accent); box-shadow: var(--shadow); transform: translateY(-1px); }
  .tool .hd { display: flex; align-items: center; gap: 7px; }
  .tool .nm { font-weight: 700; font-size: 14px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .tool .badge { margin-left: auto; }
  .tool .stats { display: flex; gap: 12px; margin-top: 7px; font-size: 12px; color: var(--soft); }
  .tool .stats b { color: var(--ink); font-weight: 650; }
  .okbar { display: flex; height: 4px; border-radius: 2px; overflow: hidden; margin-top: 9px; background: var(--bar); }
  .okbar i { display: block; height: 100%; }

  /* history */
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  thead th { position: sticky; top: 0; background: var(--panel); z-index: 1; text-align: left;
             font-weight: 600; color: var(--ghost); font-size: 11px; letter-spacing: .05em;
             text-transform: uppercase; padding: 8px 10px; border-bottom: 1px solid var(--line); }
  tbody td { padding: 7px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
  tbody tr { cursor: pointer; }
  tbody tr:hover td { background: var(--sunk); }
  td.r, th.r { text-align: right; }
  #history .scroll { max-height: 420px; }
  .minibar { display: inline-flex; width: 90px; height: 6px; border-radius: 3px; overflow: hidden;
             background: var(--bar); vertical-align: middle; }
  .minibar i { display: block; height: 100%; }

  /* pills */
  .pill { display: inline-block; font-size: 11px; font-weight: 600; padding: 1px 8px; border-radius: 999px;
          background: var(--sunk); color: var(--soft); white-space: nowrap; }
  .pill.running, .pill.done, .pill.packaging { background: color-mix(in srgb, var(--ok) 14%, transparent); color: var(--ok); }
  .pill.queued_gpu, .pill.paused { background: color-mix(in srgb, var(--warn) 16%, transparent); color: var(--warn); }
  .pill.failed { background: color-mix(in srgb, var(--hot) 14%, transparent); color: var(--hot); }
  .pill.staging, .pill.received { background: color-mix(in srgb, var(--staging) 16%, transparent); color: var(--staging); }

  /* ---- the dialog ------------------------------------------------------- */
  #overlay { position: fixed; inset: 0; z-index: 20; display: flex; align-items: flex-start;
             justify-content: center; padding: 5vh 20px; overflow: auto;
             background: color-mix(in srgb, #0b0e12 46%, transparent); backdrop-filter: blur(3px);
             opacity: 0; transition: opacity .16s ease; }
  #overlay.open { opacity: 1; }
  #dialog { width: min(1180px, 100%); background: var(--panel); border: 1px solid var(--line);
            border-radius: 16px; box-shadow: var(--shadow-lg); transform: translateY(10px) scale(.99);
            transition: transform .18s ease; }
  #overlay.open #dialog { transform: none; }
  .dhd { display: flex; align-items: flex-start; gap: 12px; padding: 18px 22px 14px;
         border-bottom: 1px solid var(--line); }
  .dhd h3 { margin: 0; font-size: 21px; font-weight: 750; letter-spacing: -.015em; }
  .dhd .rid { font-size: 11.5px; color: var(--ghost); word-break: break-all; margin-top: 2px; }
  .dhd .acts { margin-left: auto; display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end; }
  .dbody { padding: 18px 22px 22px; display: flex; flex-direction: column; gap: 18px; }
  .section h4 { margin: 0 0 8px; font-size: 12px; letter-spacing: .06em; text-transform: uppercase;
                color: var(--ghost); font-weight: 650; }
  .kpis { display: grid; gap: 10px; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); }
  .kpi { background: var(--sunk); border-radius: 10px; padding: 9px 12px; }
  .kpi u { display: block; font-size: 10.5px; letter-spacing: .05em; color: var(--ghost);
           text-transform: uppercase; text-decoration: none; font-weight: 600; }
  .kpi b { font-size: 17px; font-weight: 700; }
  .two { display: grid; gap: 18px; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
  @media (max-width: 900px) { .two { grid-template-columns: 1fr; } }
  table.io td { padding: 5px 10px 5px 0; white-space: normal; }
  table.io tr { cursor: default; }
  table.io tr:hover td { background: none; }
  table.io td:first-child { color: var(--soft); white-space: nowrap; }
  .con { background: var(--sunk); border: 1px solid var(--line); border-radius: 10px; padding: 9px 11px;
         max-height: 260px; overflow: auto; font: 12px/1.6 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .con .ln { display: flex; gap: 10px; }
  .con .ts { color: var(--ghost); flex: none; }
  .con .ok .tx { color: var(--ok); font-weight: 600; } .con .warn .tx { color: var(--warn); }
  .con .error .tx { color: var(--hot); font-weight: 600; }
  .caveat { font-size: 11.5px; color: var(--ghost); border-top: 1px solid var(--line); padding-top: 12px; }

  /* gantt */
  .gantt { position: relative; }
  .grow-row { display: grid; grid-template-columns: 150px minmax(0, 1fr) 64px; gap: 10px;
              align-items: center; padding: 3px 0; font-size: 12px; }
  .grow-row .lb { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--soft); }
  .grow-row .lb b { color: var(--ink); font-weight: 600; }
  .grow-row .vv { text-align: right; color: var(--soft); }
  .grow-row.click { cursor: pointer; border-radius: 6px; }
  .grow-row.click:hover { background: var(--sunk); }
  .lane { position: relative; height: 18px; background: var(--sunk); border-radius: 5px; overflow: hidden; }
  .lane.thin { height: 12px; background: transparent; }
  .seg { position: absolute; top: 0; bottom: 0; border-radius: 3px; min-width: 2px; }
  .seg.received, .seg.staging { background: var(--staging); opacity: .7; }
  .seg.queued_gpu { background: var(--warn); }
  .seg.running { background: var(--ok); }
  .seg.packaging { background: var(--violet); }
  .seg.paused { background: repeating-linear-gradient(45deg, var(--warn) 0 4px, transparent 4px 8px); }
  .seg.nest { background: var(--accent); opacity: .85; top: 2px; bottom: 2px; }
  .seg.nest.d2 { opacity: .55; }
  .seg.open { box-shadow: inset -3px 0 0 color-mix(in srgb, #fff 55%, transparent); }
  .axis { display: grid; grid-template-columns: 150px minmax(0, 1fr) 64px; gap: 10px;
          font-size: 10.5px; color: var(--ghost); margin-top: 2px; }
  .axis .ticks { position: relative; height: 14px; }
  .axis .ticks span { position: absolute; transform: translateX(-50%); white-space: nowrap; }
  .keys { display: flex; gap: 12px; flex-wrap: wrap; font-size: 11.5px; color: var(--soft); margin-bottom: 8px; }
  .keys i { display: inline-block; width: 12px; height: 8px; border-radius: 2px; margin-right: 5px; vertical-align: middle; }
  .stack { display: flex; height: 22px; border-radius: 7px; overflow: hidden; background: var(--bar); }
  .stack i { display: block; height: 100%; }
  .stack-keys { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: var(--soft); margin-top: 7px; }
  .stack-keys b { color: var(--ink); }

  #gate { max-width: 520px; margin: 14vh auto; }
  .err { color: var(--hot); font-size: 12.5px; margin-top: 8px; }
</style>
</head>
<body>

<div id="gate" class="card" hidden>
  <h2>Server dashboard</h2>
  <div class="body">
    <p style="margin:0 0 12px;font-size:13px;color:var(--soft)">
      The readings come from <code>/server-debug.json</code>, which is
      Bearer-protected like every endpoint that says anything about a run.
      The token stays in this browser and is never written into this page.</p>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <input id="token" type="password" placeholder="API token" autocomplete="off">
      <button id="save" type="button">Connect</button>
    </div>
    <div class="err" id="gateerr"></div>
  </div>
</div>

<div id="shell" hidden>
  <header>
    <span id="flag" title="derived from this page's own origin"></span>
    <div>
      <h1 id="origin">&nbsp;</h1>
      <div id="host" class="mono">connecting&hellip;</div>
    </div>
    <span class="grow"></span>
    <span id="filters"></span>
    <input id="find" style="min-width:290px" type="search" placeholder="Filter: run id, tool or address" autocomplete="off" spellcheck="false">
    <span id="live"><span class="dot" id="dot"></span><span id="livetext">live</span></span>
    <button id="toggle" type="button" class="ghost">Pause</button>
    <button id="theme" type="button" class="ghost" title="theme">Theme</button>
  </header>

  <section id="vitals"></section>

  <div id="main">
    <div id="left">
      <section class="card" id="queue">
        <h2>Queue <span class="count" id="q-count">0</span><span class="note">waiting for the machine</span></h2>
        <div class="body">
          <div id="q-list"></div>
          <div style="margin-top:12px"><div class="label">Total time spent waiting</div><div id="q-rank"></div></div>
        </div>
      </section>
      <section class="card" id="paused">
        <h2>Paused <span class="count" id="p-count">0</span><span class="note">waiting for a person</span></h2>
        <div class="body"><div id="p-list"></div></div>
      </section>
    </div>

    <section class="card" id="running">
      <h2>Running <span class="count" id="r-count">0</span><span class="note">click a run for its timeline</span></h2>
      <div class="body"><div class="runs" id="r-list"></div></div>
    </section>

    <section class="card" id="machine">
      <h2>Machine<span class="note" id="m-note"></span></h2>
      <div class="body">
        <div id="m-tanks"></div>
        <div class="label" style="margin-top:4px">Last 30 minutes</div>
        <div class="legend"><span><i style="background:var(--accent)"></i>cpu</span>
          <span><i style="background:var(--violet)"></i>ram</span>
          <span><i style="background:var(--ok)"></i>vram</span></div>
        <div id="m-chart"></div>
        <div class="legend" style="margin-top:8px"><span><i style="background:var(--ok)"></i>running</span>
          <span><i style="background:var(--warn)"></i>waiting</span></div>
        <div id="m-load"></div>
        <div id="m-disk" style="margin-top:10px"></div>
      </div>
    </section>
  </div>

  <section class="card" id="tools">
    <h2>Tools<span class="note">click a tool for its runs and graphs</span></h2>
    <div class="body"><div class="grid" id="t-list"></div></div>
  </section>

  <section class="card" id="history">
    <h2>History <span class="count" id="h-count">0</span><span class="note" id="h-note"></span></h2>
    <div class="scroll" id="h-body"></div>
  </section>
</div>

<div id="overlay" hidden><div id="dialog" role="dialog" aria-modal="true"></div></div>

<script>
(function () {
  "use strict";
  var KEY = "visor.token", THEME_KEY = "visor.theme";
  var EVERY = 2000;
  var token = "";
  try { token = window.localStorage.getItem(KEY) || ""; } catch (e) { token = ""; }

  var live = true, latest = null;
  // What the dialog shows: a run or a tool, and what was last fetched for it.
  var dlg = { kind: null, id: null, data: null, error: "", fetchedAt: 0 };
  var filters = { tool: "", client: "", outcome: "", text: "" };

  function el(id) { return document.getElementById(id); }
  function esc(text) {
    var box = document.createElement("div");
    box.textContent = String(text == null ? "" : text);
    return box.innerHTML;
  }
  function gib(n) { return (n == null) ? "—" : (n / 1073741824).toFixed(1) + " G"; }
  function bytes(n) {
    if (n == null) { return "—"; }
    if (n >= 1073741824) { return (n / 1073741824).toFixed(1) + " GiB"; }
    if (n >= 1048576) { return (n / 1048576).toFixed(1) + " MiB"; }
    if (n >= 1024) { return (n / 1024).toFixed(0) + " KiB"; }
    return n + " B";
  }
  function pct(a, b) { return b ? Math.max(0, Math.min(100, (a / b) * 100)) : 0; }
  function dur(s) {
    if (s == null) { return "—"; }
    s = Math.max(0, Math.round(s));
    if (s < 60) { return s + "s"; }
    if (s < 3600) { return Math.floor(s / 60) + "m" + ("0" + (s % 60)).slice(-2); }
    return Math.floor(s / 3600) + "h" + ("0" + Math.floor((s % 3600) / 60)).slice(-2);
  }
  function now() { return Date.now() / 1000; }
  function ago(at) { return at ? dur(now() - at) : "—"; }
  function clock(at) {
    var d = new Date(at * 1000);
    return ("0" + d.getHours()).slice(-2) + ":" + ("0" + d.getMinutes()).slice(-2) +
      ":" + ("0" + d.getSeconds()).slice(-2);
  }
  function inFlight(r) { return r.state === "running" || r.state === "pending"; }
  var PHASE_LABEL = { received: "arriving", staging: "staging", queued_gpu: "waiting",
    running: "running", packaging: "packaging", paused: "paused", done: "done",
    failed: "failed", cancelled: "cancelled" };
  var PHASE_COLOR = { received: "var(--staging)", staging: "var(--staging)",
    queued_gpu: "var(--warn)", running: "var(--ok)", packaging: "var(--violet)",
    paused: "var(--warn)" };
  function pill(phase) {
    return '<span class="pill ' + esc(phase || "") + '">' + esc(PHASE_LABEL[phase] || phase || "in flight") + "</span>";
  }

  // ---- theme ----------------------------------------------------------
  function applyTheme(theme) {
    if (theme === "light" || theme === "dark") {
      document.documentElement.setAttribute("data-theme", theme);
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
    el("theme").textContent = theme === "dark" ? "Dark" : theme === "light" ? "Light" : "Auto";
  }
  var theme = "auto";
  try { theme = window.localStorage.getItem(THEME_KEY) || "auto"; } catch (e) { theme = "auto"; }
  applyTheme(theme);

  // ---- filters --------------------------------------------------------
  function readUrl() {
    var raw = (window.location.hash || "").replace(/^#/, "");
    raw.split("&").forEach(function (pair) {
      var bits = pair.split("=");
      var key = decodeURIComponent(bits[0] || "");
      if (key in filters) { filters[key] = decodeURIComponent(bits[1] || ""); }
    });
  }
  function writeUrl() {
    var parts = Object.keys(filters).filter(function (k) { return filters[k]; })
      .map(function (k) { return encodeURIComponent(k) + "=" + encodeURIComponent(filters[k]); });
    var next = parts.length ? "#" + parts.join("&") : "#";
    if (window.location.hash !== next) { window.history.replaceState(null, "", next); }
  }
  function setFilter(key, value) {
    filters[key] = (filters[key] === value) ? "" : value;
    writeUrl();
    if (latest) { draw(latest); }
  }
  function matches(run, led) {
    led = led || {};
    var tool = run.tool || led.tool || "", client = run.client || led.client || "";
    if (filters.tool && tool !== filters.tool) { return false; }
    if (filters.client && client !== filters.client) { return false; }
    if (filters.outcome && (run.phase || led.outcome || "") !== filters.outcome) { return false; }
    if (filters.text) {
      var hay = [run.run_id || led.run_id, tool, client].join(" ").toLowerCase();
      if (hay.indexOf(filters.text.toLowerCase()) === -1) { return false; }
    }
    return true;
  }
  function anyFilter() { return Object.keys(filters).some(function (k) { return filters[k]; }); }
  function drawFilters() {
    var labels = { tool: "tool", client: "from", outcome: "state", text: "text" };
    el("filters").innerHTML = Object.keys(filters).filter(function (k) { return filters[k]; })
      .map(function (k) {
        return '<span class="fchip" data-clear="' + k + '"><i>' + labels[k] + "</i><b>" +
          esc(filters[k]) + "</b>&times;</span>";
      }).join("");
    if (document.activeElement !== el("find") && el("find").value !== filters.text) {
      el("find").value = filters.text;
    }
  }

  // ---- charts (inline SVG, no library) ---------------------------------
  function spark(values, color, max) {
    var w = 200, h = 30, pts = [], i, n = values.length;
    if (n < 2) { return '<svg viewBox="0 0 200 30"></svg>'; }
    var top = max || Math.max.apply(null, values.filter(function (v) { return v != null; }).concat([1]));
    var path = "", started = false;
    for (i = 0; i < n; i++) {
      if (values[i] == null) { started = false; continue; }
      var x = (i / (n - 1)) * w, y = h - 2 - (Math.min(values[i], top) / top) * (h - 4);
      path += (started ? "L" : "M") + x.toFixed(1) + "," + y.toFixed(1);
      started = true;
      pts.push([x, y]);
    }
    var area = pts.length > 1 ? "M" + pts[0][0].toFixed(1) + "," + h + " L" +
      pts.map(function (p) { return p[0].toFixed(1) + "," + p[1].toFixed(1); }).join(" L") +
      " L" + pts[pts.length - 1][0].toFixed(1) + "," + h + " Z" : "";
    return '<svg viewBox="0 0 ' + w + " " + h + '" preserveAspectRatio="none">' +
      '<path d="' + area + '" fill="' + color + '" opacity=".12"/>' +
      '<path d="' + path + '" fill="none" stroke="' + color + '" stroke-width="1.6" ' +
      'vector-effect="non-scaling-stroke" stroke-linejoin="round"/></svg>';
  }

  // A time chart: `series` is [{values: [[t, v], ...], color, step}], on one
  // x axis from t0 to t1, y from 0 to `max`. `marks` are vertical spans
  // drawn behind the lines ({start, end, color}).
  function timeChart(series, t0, t1, opts) {
    opts = opts || {};
    var w = 600, h = opts.height || 110, pad = 18, top = 6, max = opts.max || 100;
    if (!(t1 > t0)) { t1 = t0 + 1; }
    function X(t) { return pad + ((t - t0) / (t1 - t0)) * (w - pad - 4); }
    function Y(v) { return top + (h - top - 16) * (1 - Math.min(v, max) / max); }
    var out = '<svg class="chart" viewBox="0 0 ' + w + " " + h + '" preserveAspectRatio="none" style="height:' + h + 'px">';
    [0, 0.5, 1].forEach(function (f) {
      var y = Y(max * f);
      out += '<line x1="' + pad + '" x2="' + (w - 4) + '" y1="' + y.toFixed(1) + '" y2="' + y.toFixed(1) +
        '" stroke="var(--line)" stroke-dasharray="' + (f ? "3 4" : "0") + '" vector-effect="non-scaling-stroke"/>';
    });
    (opts.marks || []).forEach(function (m) {
      var a = X(Math.max(m.start, t0)), b = X(Math.min(m.end == null ? t1 : m.end, t1));
      if (b > a) {
        out += '<rect x="' + a.toFixed(1) + '" y="' + top + '" width="' + Math.max(1, b - a).toFixed(1) +
          '" height="' + (h - top - 16) + '" fill="' + (m.color || "var(--accent)") + '" opacity=".08"/>';
      }
    });
    series.forEach(function (s) {
      var d = "", started = false;
      s.values.forEach(function (p, i) {
        if (p[1] == null) { started = false; return; }
        var x = X(p[0]), y = Y(p[1]);
        if (s.step && started) {
          d += "H" + x.toFixed(1) + "V" + y.toFixed(1);
        } else {
          d += (started ? "L" : "M") + x.toFixed(1) + "," + y.toFixed(1);
        }
        started = true;
      });
      out += '<path d="' + d + '" fill="none" stroke="' + s.color + '" stroke-width="' + (s.width || 1.7) +
        '" vector-effect="non-scaling-stroke" stroke-linejoin="round" stroke-linecap="round"/>';
    });
    var ticks = opts.ticks || 4, i;
    for (i = 0; i <= ticks; i++) {
      var t = t0 + (t1 - t0) * (i / ticks);
      out += '<text x="' + X(t).toFixed(1) + '" y="' + (h - 3) + '" text-anchor="' +
        (i === 0 ? "start" : i === ticks ? "end" : "middle") + '">' + clock(t).slice(0, 5) + "</text>";
    }
    out += '<text x="2" y="' + (top + 8) + '">' + esc(opts.maxLabel || (max + "%")) + "</text>";
    return out + "</svg>";
  }

  // Gantt rows on one time axis [t0, t1]. A row is {label, sub, value,
  // spans: [{phase, start, end}], nested: [{tool, depth, start, end}], run}.
  function gantt(rows, t0, t1, relative) {
    if (!(t1 > t0)) { t1 = t0 + 1; }
    function L(t) { return pct(t - t0, t1 - t0); }
    var html = '<div class="gantt">';
    rows.forEach(function (row) {
      var lane = "";
      (row.spans || []).forEach(function (s) {
        var a = L(s.start), b = L(s.end == null ? Math.min(now(), t1) : s.end);
        lane += '<span class="seg ' + esc(s.phase) + (s.end == null ? " open" : "") +
          '" style="left:' + a.toFixed(2) + "%;width:" + Math.max(0.3, b - a).toFixed(2) +
          '%" title="' + esc((PHASE_LABEL[s.phase] || s.phase) + " · " +
          dur((s.end == null ? now() : s.end) - s.start)) + '"></span>';
      });
      html += '<div class="grow-row' + (row.run ? ' click" data-run="' + esc(row.run) : "") + '">' +
        '<span class="lb">' + row.label + "</span>" + '<div class="lane">' + lane + "</div>" +
        '<span class="vv mono">' + (row.value || "") + "</span></div>";
      var depthRows = {};
      (row.nested || []).forEach(function (c) {
        (depthRows[c.depth] = depthRows[c.depth] || []).push(c);
      });
      Object.keys(depthRows).sort().forEach(function (depth) {
        var segs = depthRows[depth].map(function (c) {
          var a = L(c.start), b = L(c.end == null ? Math.min(now(), t1) : c.end);
          return '<span class="seg nest' + (depth >= 2 ? " d2" : "") + (c.end == null ? " open" : "") +
            '" style="left:' + a.toFixed(2) + "%;width:" + Math.max(0.3, b - a).toFixed(2) +
            '%" title="' + esc(c.tool + " · " + dur((c.end == null ? now() : c.end) - c.start)) + '"></span>';
        }).join("");
        var tools = [];
        depthRows[depth].forEach(function (c) { if (tools.indexOf(c.tool) < 0) { tools.push(c.tool); } });
        html += '<div class="grow-row"><span class="lb" style="padding-left:' + (depth * 10) + 'px">↳ ' +
          esc(tools.join(", ")) + '</span><div class="lane thin">' + segs + '</div><span class="vv"></span></div>';
      });
    });
    var ticks = "", i;
    for (i = 0; i <= 4; i++) {
      var t = t0 + (t1 - t0) * (i / 4);
      ticks += '<span style="left:' + (i * 25) + '%">' + (relative ? dur(t - t0) : clock(t)) + "</span>";
    }
    return html + '<div class="axis"><span></span><div class="ticks">' + ticks + "</div><span></span></div></div>";
  }

  function phaseKeys() {
    return '<div class="keys">' + ["received", "staging", "queued_gpu", "running", "packaging"].map(function (p) {
      return '<span><i style="background:' + PHASE_COLOR[p] + '"></i>' + PHASE_LABEL[p] + "</span>";
    }).join("") + '<span><i style="background:var(--accent)"></i>nested tool</span></div>';
  }

  // ---- vitals ---------------------------------------------------------
  function traceOf(d, key, total) {
    return (d.trace || []).map(function (p) {
      if (p[key] == null) { return null; }
      return total ? pct(p[key], total) : p[key];
    });
  }
  function vital(label, value, unit, foot, cls, sparkline) {
    return '<div class="card vital ' + (cls || "") + '"><div class="label">' + esc(label) + "</div>" +
      '<div class="v mono">' + value + (unit ? "<small>" + unit + "</small>" : "") + "</div>" +
      '<div class="f">' + foot + "</div>" + (sparkline || "") + "</div>";
  }
  function drawVitals(d) {
    var a = d.admission || {}, b = d.budget || {}, card = d.card || {}, ram = d.ram, f = d.inflight || {};
    var paused = (d.runs || []).filter(function (r) { return r.phase === "paused"; }).length;
    var vramUsed = card.total_bytes == null ? null : card.total_bytes - card.free_bytes;
    var cpu = d.cpu_percent;
    var today = (d.ledger || []).filter(function (r) {
      return r.ended_at && new Date(r.ended_at * 1000).toDateString() === new Date().toDateString();
    });
    el("vitals").innerHTML =
      vital("Running", a.running == null ? "?" : a.running, "", f.now + " request" + (f.now === 1 ? "" : "s") +
        " in flight", a.running ? "busy" : "", spark(traceOf(d, "running"), "var(--ok)")) +
      vital("Waiting", a.waiting == null ? "?" : a.waiting, "", paused + " paused", a.waiting ? "wait" : "",
        spark(traceOf(d, "waiting"), "var(--warn)")) +
      vital("CPU", cpu == null ? "—" : cpu.toFixed(0), "%",
        (a.cpus_held == null ? "?" : a.cpus_held.toFixed(1)) + " of " + (b.cpus == null ? "?" : b.cpus.toFixed(0)) +
        " cores held", cpu > 90 ? "hot" : "", spark(traceOf(d, "cpu"), "var(--accent)", 100)) +
      vital("RAM", ram ? (ram.used / 1073741824).toFixed(0) : "—", ram ? "/ " + (ram.total / 1073741824).toFixed(0) + " G" : "",
        gib(a.ram_held) + " held by runs", ram && pct(ram.used, ram.total) > 90 ? "hot" : "",
        spark(traceOf(d, "ram", ram && ram.total), "var(--violet)", 100)) +
      vital("VRAM", vramUsed == null ? "—" : (vramUsed / 1073741824).toFixed(1),
        card.total_bytes ? "/ " + (card.total_bytes / 1073741824).toFixed(0) + " G" : "",
        card.total_bytes ? gib(a.vram_held) + " held by runs" : "no card", "",
        spark(traceOf(d, "vram", card.total_bytes), "var(--ok)", 100)) +
      vital("Finished today", today.length, "",
        today.filter(function (r) { return r.outcome === "done"; }).length + " succeeded \u00b7 " +
        today.filter(function (r) { return r.outcome === "failed"; }).length + " failed", "", "");
  }

  // ---- queue ------------------------------------------------------------
  function ledgerById(d) {
    var byId = {};
    (d.ledger || []).forEach(function (r) { byId[r.run_id] = r; });
    return byId;
  }
  function drawQueue(d) {
    var byId = ledgerById(d), q = d.queue || {};
    // Two gates, in the order a run meets them: a slot among the
    // MAX_CONCURRENT_TOOLS the server serves at once (the run is "received"
    // until it gets one -- or still uploading, which this page cannot tell
    // apart), then room on the machine ("queued_gpu").
    var waiting = (d.runs || []).filter(function (r) {
      return inFlight(r) && (r.phase === "queued_gpu" || r.phase === "received") && matches(r, byId[r.run_id]);
    }).sort(function (x, y) {
      var gx = x.phase === "queued_gpu" ? 0 : 1, gy = y.phase === "queued_gpu" ? 0 : 1;
      return gx - gy || x.started_at - y.started_at;
    });
    el("q-count").textContent = waiting.length;
    el("q-list").innerHTML = waiting.length ? waiting.map(function (r, i) {
      var slot = r.phase === "received";
      return '<div class="qitem' + (slot ? " slot" : "") + '" data-run="' + esc(r.run_id) + '"><span class="pos">' + (i + 1) + "</span>" +
        '<div style="min-width:0"><div class="n">' + esc(r.tool || "?") + "</div>" +
        '<div class="m mono">' + (slot ? "waiting for a slot " : "waiting for room ") + ago(r.started_at) +
        (r.client ? " · " + esc(r.client) : "") + "</div></div></div>";
    }).join("") : '<div class="empty">' + (anyFilter() ? "Nothing waiting matches this filter." : "Nothing waiting.") + "</div>";

    var held = (d.runs || []).filter(function (r) { return r.phase === "paused" && matches(r, byId[r.run_id]); });
    var ttl = ((d.server || {}).paused_ttl_seconds) || 0;
    el("p-count").textContent = held.length;
    el("p-list").innerHTML = held.length ? held.map(function (r) {
      var left = ttl ? Math.max(0, ttl - (now() - r.started_at)) : null;
      return '<div class="qitem held" data-run="' + esc(r.run_id) + '"><span class="pos">⏸</span>' +
        '<div><div class="n">' + esc(r.tool || "?") + "</div>" +
        '<div class="m mono">held ' + ago(r.started_at) + (left === null ? "" : " · dropped in " + dur(left)) +
        "</div></div></div>";
    }).join("") : '<div class="empty">Nothing paused.</div>';

    var ranked = (q.ranked || []).filter(function (r) { return r.total_wait > 0; }).slice(0, 6);
    var worst = ranked.length ? (ranked[0].total_wait || 1) : 1;
    el("q-rank").innerHTML = ranked.length ? ranked.map(function (r) {
      return '<div class="rank"><span class="nm">' + esc(r.tool) + '</span><span class="track"><i style="width:' +
        pct(r.total_wait, worst).toFixed(1) + '%"></i></span><span class="vv mono">' + dur(r.total_wait) + "</span></div>";
    }).join("") : '<div class="empty" style="padding:6px 0;text-align:left">No run has had to wait yet.</div>';
  }

  // ---- running ------------------------------------------------------------
  function chainHtml(root, chain) {
    var links = ['<span class="link root">' + esc(root || "?") + "</span>"];
    (chain || []).forEach(function (tool, i) {
      links.push('<span class="sep">›</span><span class="link' + (i === chain.length - 1 ? " now" : "") +
        '">' + esc(tool) + "</span>");
    });
    return '<div class="chain">' + links.join("") + "</div>";
  }
  function drawRunning(d) {
    var byId = ledgerById(d);
    var rows = (d.runs || []).filter(function (r) {
      return inFlight(r) && r.phase !== "paused" && r.phase !== "queued_gpu" && r.phase !== "received" &&
        matches(r, byId[r.run_id]);
    });
    el("r-count").textContent = rows.length;
    if (!rows.length) {
      el("r-list").innerHTML = '<div class="empty">' + (anyFilter() ? "Nothing running matches this filter." : "Nothing running.") + "</div>";
      return;
    }
    el("r-list").innerHTML = rows.map(function (r) {
      var led = byId[r.run_id] || {};
      var staging = r.phase === "staging";
      var frac = r.fraction == null ? null : Math.max(0, Math.min(1, r.fraction));
      var chain = r.chain || [];
      return '<div class="run' + (staging ? " s" : "") + '" data-run="' + esc(r.run_id) + '">' +
        '<div class="hd"><span class="nm">' + esc(r.tool || "?") + "</span>" + pill(r.phase) +
        '<span class="pc mono">' + (frac == null ? "" : (frac * 100).toFixed(0) + "%") + "</span></div>" +
        '<div class="sub mono">' + ago(r.started_at) + " in" +
        (led.waited ? " · waited " + dur(led.waited) : "") +
        ((r.client || led.client) ? " · " + esc(r.client || led.client) : "") + "</div>" +
        (chain.length ? chainHtml(r.tool, chain) : "") +
        '<div class="progress' + (frac == null ? " indet" : "") + '"><i style="width:' +
        (frac == null ? 0 : frac * 100).toFixed(1) + '%"></i></div>' +
        '<div class="cells">' +
        '<div class="cell"><u>chan</u><b>' + (led.channels == null ? "—" : led.channels) + "</b></div>" +
        '<div class="cell"><u>cpu</u><b>' + (led.cpus == null ? "—" : led.cpus) + "</b></div>" +
        '<div class="cell"><u>ram</u><b>' + gib(led.ram_bytes) + "</b></div>" +
        '<div class="cell"><u>vram</u><b>' + gib(led.vram_bytes) + "</b></div>" +
        '<div class="cell"><u>files</u><b>' + (led.files == null ? "—" : led.files) + "</b></div>" +
        "</div></div>";
    }).join("");
  }

  // ---- machine ------------------------------------------------------------
  function tank(label, heldPct, usedPct, big, foot) {
    if (heldPct == null && usedPct == null) {
      return '<div class="tank"><div class="lab">' + esc(label) + "<b>—</b></div>" +
        '<div class="tr na"></div><div class="ft">unavailable</div></div>';
    }
    var used = usedPct == null ? heldPct : usedPct;
    return '<div class="tank"><div class="lab">' + esc(label) + "<b>" + esc(big) + "</b></div>" +
      '<div class="tr' + (used > 90 ? " hot" : "") + '">' +
      (heldPct == null ? "" : '<i style="width:' + heldPct.toFixed(1) + '%"></i>') +
      '<s style="width:' + used.toFixed(1) + '%"></s></div><div class="ft">' + foot + "</div></div>";
  }
  function drawMachine(d) {
    var a = d.admission || {}, b = d.budget || {}, card = d.card || {}, ram = d.ram, cpu = d.cpu_percent;
    var vramUsed = card.total_bytes == null ? null : card.total_bytes - card.free_bytes;
    el("m-tanks").innerHTML =
      tank("CPU", b.cpus ? pct(a.cpus_held, b.cpus) : null, cpu, cpu == null ? "—" : cpu.toFixed(0) + "%",
        "pale: held by runs · solid: in use") +
      tank("RAM", b.ram_bytes ? pct(a.ram_held, b.ram_bytes) : null, ram ? pct(ram.used, ram.total) : null,
        ram ? gib(ram.used) : "—", gib(a.ram_held) + " held · " + (ram ? gib(ram.available) + " available" : "no /proc/meminfo")) +
      tank("VRAM", b.vram_bytes ? pct(a.vram_held, b.vram_bytes) : null,
        card.total_bytes ? pct(vramUsed, card.total_bytes) : null, card.total_bytes ? gib(vramUsed) : "—",
        gib(a.vram_held) + " held · " + (card.total_bytes ? gib(card.free_bytes) + " free" : "no card"));

    // The window is the last 30 minutes, or since the trace began if that is
    // shorter: a server up for five minutes should fill the chart, not draw
    // a sliver at its right edge.
    var tr = d.trace || [], t1 = now(), t0 = Math.max(t1 - 1800, tr.length ? tr[0].at : t1 - 1800);
    var ramTotal = ram && ram.total, vramTotal = card.total_bytes;
    el("m-chart").innerHTML = tr.length > 1 ? timeChart([
      { values: tr.map(function (p) { return [p.at, p.cpu]; }), color: "var(--accent)" },
      { values: tr.map(function (p) { return [p.at, p.ram == null || !ramTotal ? null : pct(p.ram, ramTotal)]; }), color: "var(--violet)" },
      { values: tr.map(function (p) { return [p.at, p.vram == null || !vramTotal ? null : pct(p.vram, vramTotal)]; }), color: "var(--ok)" },
    ], t0, t1, { height: 120 }) : '<div class="empty">The trace starts with the server; a point every 5 s.</div>';
    var maxLoad = Math.max.apply(null, tr.map(function (p) { return Math.max(p.running || 0, p.waiting || 0); }).concat([2]));
    el("m-load").innerHTML = tr.length > 1 ? timeChart([
      { values: tr.map(function (p) { return [p.at, p.running]; }), color: "var(--ok)", step: true },
      { values: tr.map(function (p) { return [p.at, p.waiting]; }), color: "var(--warn)", step: true },
    ], t0, t1, { height: 74, max: maxLoad, maxLabel: maxLoad + " runs" }) : "";
    el("m-note").textContent = "trace kept in memory, resets with the server";
    el("m-disk").innerHTML = Object.keys(d.disk || {}).map(function (name) {
      var e = d.disk[name];
      return '<div class="disk"><span class="nm">' + esc(name) + '</span><span class="track"><i style="width:' +
        ((e.total && e.free != null) ? pct(e.total - e.free, e.total) : 0).toFixed(1) + '%"></i></span>' +
        '<span class="vv mono">' + (e.used_here == null ? "—" : gib(e.used_here)) + " here · " +
        gib(e.free) + " free</span></div>";
    }).join("");
  }

  // ---- tools ------------------------------------------------------------
  function drawTools(d) {
    var up = d.uptime || [], seen = {};
    var rows = up.map(function (r) { seen[r.tool] = true; return r; });
    Object.keys(d.costs || {}).forEach(function (name) {
      if (!seen[name]) { rows.push({ tool: name, runs: 0, running: 0, ok: 0, failed: 0, seconds: 0, mean: 0 }); }
    });
    rows = rows.filter(function (r) { return !filters.tool || r.tool === filters.tool; });
    var waits = {};
    ((d.queue || {}).ranked || []).forEach(function (r) { waits[r.tool] = r; });
    el("t-list").innerHTML = rows.length ? rows.map(function (r) {
      var total = (r.ok || 0) + (r.failed || 0);
      var w = waits[r.tool];
      return '<div class="tool" data-tool="' + esc(r.tool) + '"><div class="hd"><span class="nm">' + esc(r.tool) + "</span>" +
        (r.running ? '<span class="pill running badge">' + r.running + " running</span>" : "") + "</div>" +
        '<div class="stats mono"><span><b>' + r.runs + "</b> runs</span><span>mean <b>" + dur(r.mean) +
        "</b></span>" + (w && w.total_wait ? "<span>wait <b>" + dur(w.mean_wait) + "</b></span>" : "") + "</div>" +
        '<div class="okbar">' + (total ? '<i style="width:' + pct(r.ok, total).toFixed(1) + '%;background:var(--ok)"></i>' +
          '<i style="width:' + pct(r.failed, total).toFixed(1) + '%;background:var(--hot)"></i>' : "") + "</div></div>";
    }).join("") : '<div class="empty">No tool has run on this server yet.</div>';
  }

  // ---- history ----------------------------------------------------------
  function phaseBar(spans, total) {
    if (!spans || !spans.length || !total) { return ""; }
    var sums = {};
    spans.forEach(function (s) { if (s.end != null) { sums[s.phase] = (sums[s.phase] || 0) + (s.end - s.start); } });
    return '<span class="minibar">' + ["received", "staging", "queued_gpu", "running", "packaging"].map(function (p) {
      return sums[p] ? '<i style="width:' + pct(sums[p], total).toFixed(1) + "%;background:" + PHASE_COLOR[p] + '"></i>' : "";
    }).join("") + "</span>";
  }
  function drawHistory(d) {
    var liveIds = {};
    (d.runs || []).forEach(function (r) { if (inFlight(r)) { liveIds[r.run_id] = true; } });
    var rows = (d.ledger || []).filter(function (r) { return !liveIds[r.run_id] && r.ended_at && matches(r, r); });
    el("h-count").textContent = rows.length;
    el("h-note").textContent = "kept across restarts, newest " + (d.ledger || []).length + " shown";
    if (!rows.length) {
      el("h-body").innerHTML = '<div class="empty">' + (anyFilter() ? "Nothing finished matches this filter." : "Nothing has finished yet.") + "</div>";
      return;
    }
    el("h-body").innerHTML = "<table><thead><tr><th>tool</th><th>calls</th><th>from</th><th>outcome</th>" +
      "<th>phases</th><th class='r'>took</th><th class='r'>waited</th><th class='r'>chan</th>" +
      "<th class='r'>vram</th><th class='r'>files</th><th class='r'>input</th><th class='r'>ended</th></tr></thead><tbody>" +
      rows.map(function (r) {
        var called = [];
        (r.nested || []).forEach(function (c) { if (called.indexOf(c.tool) < 0) { called.push(c.tool); } });
        return '<tr data-run="' + esc(r.run_id) + '"><td><b>' + esc(r.tool || "?") + "</b></td>" +
          '<td style="color:var(--soft)">' + esc(called.join(", ") || "—") + "</td>" +
          '<td class="mono">' + esc(r.client || "—") + "</td><td>" + pill(r.outcome) + "</td>" +
          "<td>" + phaseBar(r.spans, r.seconds) + "</td>" +
          "<td class='r mono'>" + dur(r.seconds) + "</td>" +
          "<td class='r mono'>" + (r.waited == null ? "—" : dur(r.waited)) + "</td>" +
          "<td class='r mono'>" + (r.channels == null ? "—" : r.channels) + "</td>" +
          "<td class='r mono'>" + gib((r.measured || {}).vram_bytes != null ? r.measured.vram_bytes : r.vram_bytes) + "</td>" +
          "<td class='r mono'>" + (r.files == null ? "—" : r.files) + "</td>" +
          "<td class='r mono'>" + bytes(r.input_bytes) + "</td>" +
          "<td class='r mono'>" + ago(r.ended_at) + " ago</td></tr>";
      }).join("") + "</tbody></table>";
  }

  // ---- the dialog ------------------------------------------------------
  function openDialog(kind, id) {
    dlg = { kind: kind, id: id, data: null, error: "", fetchedAt: 0 };
    el("overlay").hidden = false;
    document.body.classList.add("locked");
    window.requestAnimationFrame(function () { el("overlay").classList.add("open"); });
    drawDialog();
    fetchDialog();
  }
  function closeDialog() {
    dlg = { kind: null, id: null, data: null, error: "", fetchedAt: 0 };
    el("overlay").classList.remove("open");
    document.body.classList.remove("locked");
    window.setTimeout(function () { if (!dlg.kind) { el("overlay").hidden = true; } }, 170);
  }
  function fetchDialog() {
    if (!dlg.kind || !token) { return; }
    var kind = dlg.kind, id = dlg.id;
    var url = kind === "run" ? "server-debug/runs/" + encodeURIComponent(id) + ".json"
                             : "server-debug/tools/" + encodeURIComponent(id) + ".json";
    dlg.fetchedAt = Date.now();
    fetch(url, { headers: { Authorization: "Bearer " + token } })
      .then(function (r) {
        if (r.status === 404) { throw new Error("This run is no longer known to the server."); }
        if (!r.ok) { throw new Error("The server answered " + r.status + "."); }
        return r.json();
      })
      .then(function (data) {
        if (dlg.kind !== kind || dlg.id !== id) { return; }
        dlg.data = data; dlg.error = "";
        drawDialog();
      })
      .catch(function (e) {
        if (dlg.kind !== kind || dlg.id !== id) { return; }
        dlg.error = e.message;
        drawDialog();
      });
  }
  function kpi(label, value) { return '<div class="kpi"><u>' + esc(label) + "</u><b class='mono'>" + value + "</b></div>"; }
  function section(title, body) { return '<div class="section"><h4>' + esc(title) + "</h4>" + body + "</div>"; }

  function drawDialog() {
    if (!dlg.kind) { return; }
    el("dialog").innerHTML = dlg.kind === "run" ? runDialog() : toolDialog();
  }

  function runDialog() {
    var id = dlg.id, data = dlg.data || {};
    var liveRun = ((latest || {}).runs || []).filter(function (r) { return r.run_id === id; })[0] || {};
    var led = data.record || ((latest || {}).ledger || []).filter(function (r) { return r.run_id === id; })[0] || {};
    var tl = data.timeline || { spans: led.spans || [], nested: led.nested || [], chain: [], measured: led.measured };
    var tool = led.tool || liveRun.tool || "?", client = liveRun.client || led.client || "";
    var state = liveRun.phase || led.outcome || "unknown";
    var started = led.started_at || liveRun.started_at;
    var ended = led.ended_at || null;
    var took = led.seconds != null ? led.seconds : (started ? now() - started : null);
    var chain = liveRun.chain || tl.chain || [];
    var measured = tl.measured || led.measured || {};

    var head = '<div class="dhd"><div style="min-width:0"><div style="display:flex;gap:10px;align-items:center">' +
      "<h3>" + esc(tool) + "</h3>" + pill(state) + "</div>" +
      '<div class="rid mono">' + esc(id) + "</div>" + (chain.length ? chainHtml(tool, chain) : "") + "</div>" +
      '<div class="acts">' +
      '<button class="ghost" data-tool="' + esc(tool) + '">Tool view</button>' +
      '<button class="ghost" data-filter="tool" data-value="' + esc(tool) + '">Filter tool</button>' +
      (client ? '<button class="ghost" data-filter="client" data-value="' + esc(client) + '">Filter address</button>' : "") +
      '<button class="ghost" data-copy="' + esc(id) + '">Copy id</button>' +
      '<button data-close="1">Close ✕</button></div></div>';

    var spans = tl.spans || [];
    var t0 = spans.length ? spans[0].start : started, t1 = ended || now();
    (tl.nested || []).forEach(function (c) { if (c.start < t0) { t0 = c.start; } });
    var timeline = spans.length || (tl.nested || []).length
      ? phaseKeys() + gantt([{ label: "<b>" + esc(tool) + "</b>", spans: spans, nested: tl.nested, value: dur(t1 - t0) }], t0, t1, true)
      : '<div class="empty">' + (dlg.data || dlg.error ? "No phase recorded for this run." : "Loading…") + "</div>";

    var kpis = '<div class="kpis">' +
      kpi("took", dur(took)) + kpi("waited", led.waited == null ? "—" : dur(led.waited)) +
      kpi("progress", liveRun.fraction == null ? "—" : (liveRun.fraction * 100).toFixed(0) + "%") +
      kpi("from", client ? esc(client) : "—") +
      kpi("channels", led.channels == null ? "—" : led.channels) +
      kpi("cpus granted", led.cpus == null ? "—" : led.cpus) +
      kpi("ram held", gib(led.ram_bytes)) + kpi("vram held", gib(led.vram_bytes)) +
      kpi("vram peak", gib(measured.vram_bytes)) + kpi("ram peak", gib(measured.ram_bytes)) +
      kpi("input", (led.files == null ? "—" : led.files + " files · ") + bytes(led.input_bytes)) +
      kpi("started", started ? clock(started) : "—") + "</div>";

    var lines = data.lines || [];
    var con = dlg.error ? '<div class="con"><div class="ln error"><span class="tx">' + esc(dlg.error) + "</span></div></div>"
      : data.reaped ? '<div class="empty" style="text-align:left">The console is kept only while a run is on the server; its timeline above is from the history.</div>'
      : lines.length ? '<div class="con">' + lines.map(function (line) {
          var padding = new Array(Math.min(line.depth || 0, 4) + 1).join("  ");
          return '<div class="ln ' + esc(line.level) + '"><span class="ts">' + clock(line.at || 0) +
            '</span><span class="tx">' + esc(padding + line.text) + "</span></div>";
        }).join("") + "</div>"
      : '<div class="empty" style="text-align:left">Nothing reported yet.</div>';

    return head + '<div class="dbody">' + section("Timeline", timeline) + kpis +
      '<div class="two">' + section("Inputs", inputsHtml(led)) + section("Settings", settingsHtml(led)) + "</div>" +
      section("Console", con) +
      '<div class="caveat">Argument values are deliberately not held anywhere on this page: an uploaded input is ' +
      "never named, only its shape (files, bytes, extensions). A hosted bundle is named, because this deployment " +
      "staged it. Console lines are composed by the server from the phase the run reported, never quoted from " +
      "what the tool printed.</div></div>";
  }

  function inputsHtml(led) {
    var rows = led.inputs || [];
    if (!rows.length) { return '<div class="empty" style="text-align:left">not recorded</div>'; }
    return '<table class="io"><tbody>' + rows.map(function (r) {
      var what = r.hosted ? "<b>" + esc(r.hosted) + "</b>" : (r.files === 1 ? "1 file" : r.files + " files") + (r.partial ? "+" : "");
      return "<tr><td>" + esc(r.argument) + "</td><td>" + what + '</td><td class="mono">' + bytes(r.bytes) +
        '</td><td class="mono" style="color:var(--ghost)">' + esc((r.extensions || []).join(" ")) + "</td></tr>";
    }).join("") + "</tbody></table>";
  }
  function settingsHtml(led) {
    var set = led.settings || {}, names = Object.keys(set);
    if (!names.length) { return '<div class="empty" style="text-align:left">none recorded</div>'; }
    return '<table class="io"><tbody>' + names.sort().map(function (name) {
      var v = set[name], shown = Array.isArray(v) ? (v.length ? v.join(", ") : "—") : String(v);
      return "<tr><td>" + esc(name) + '</td><td class="mono">' + esc(shown) + "</td></tr>";
    }).join("") + "</tbody></table>";
  }

  function toolDialog() {
    var name = dlg.id, data = dlg.data;
    var head = '<div class="dhd"><div><h3>' + esc(name) + '</h3><div class="rid">every run of this tool the history holds</div></div>' +
      '<div class="acts"><button class="ghost" data-filter="tool" data-value="' + esc(name) + '">Filter dashboard</button>' +
      '<button data-close="1">Close ✕</button></div></div>';
    if (!data) {
      return head + '<div class="dbody"><div class="empty">' + (dlg.error ? esc(dlg.error) : "Loading…") + "</div></div>";
    }
    var s = data.summary || {}, runsOf = data.runs || [];
    var kpis = '<div class="kpis">' + kpi("runs", s.runs) + kpi("succeeded", s.ok) + kpi("failed", s.failed) +
      kpi("running", s.running) + kpi("mean time", dur(s.mean_seconds)) + kpi("mean wait", dur(s.mean_wait)) +
      kpi("learned vram", data.cost ? gib(data.cost.vram_bytes) : "—") +
      kpi("learned ram", data.cost ? gib(data.cost.ram_bytes) : "—") + "</div>";

    // Where the time goes, on average: one stacked bar of the phase means.
    var phases = data.phases || [], meanTotal = 0;
    phases.forEach(function (p) { meanTotal += p.mean; });
    var order = ["received", "staging", "queued_gpu", "running", "packaging"];
    var byPhase = {};
    phases.forEach(function (p) { byPhase[p.phase] = p; });
    var stack = meanTotal ? '<div class="stack">' + order.map(function (p) {
        return byPhase[p] ? '<i style="width:' + pct(byPhase[p].mean, meanTotal).toFixed(2) + "%;background:" + PHASE_COLOR[p] +
          '" title="' + esc(PHASE_LABEL[p] + " " + dur(byPhase[p].mean)) + '"></i>' : "";
      }).join("") + '</div><div class="stack-keys">' + order.filter(function (p) { return byPhase[p]; }).map(function (p) {
        return '<span><i style="display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;background:' +
          PHASE_COLOR[p] + '"></i>' + PHASE_LABEL[p] + " <b>" + dur(byPhase[p].mean) + "</b></span>";
      }).join("") + "</div>"
      : '<div class="empty" style="text-align:left">No finished run with recorded phases yet.</div>';

    // The newest runs, each on its own clock so their lengths compare.
    var recent = runsOf.filter(function (r) { return (r.spans || []).length; }).slice(0, 25);
    var longest = 1;
    recent.forEach(function (r) {
      var start = r.spans[0].start, end = r.ended_at || now();
      longest = Math.max(longest, end - start);
    });
    var rel = recent.map(function (r) {
      var start = r.spans[0].start;
      function shift(x) {
        return { phase: x.phase, tool: x.tool, depth: x.depth, start: x.start - start, end: x.end == null ? null : x.end - start };
      }
      return {
        label: '<b>' + clock(start).slice(0, 5) + "</b> " + esc(r.client || ""),
        spans: (r.spans || []).map(shift), nested: (r.nested || []).map(shift),
        value: dur((r.ended_at || now()) - start), run: r.run_id,
      };
    });
    var runsGantt = rel.length ? phaseKeys() + gantt(rel, 0, longest, true)
      : '<div class="empty" style="text-align:left">No run with a recorded timeline yet. Runs finished before this version have none.</div>';

    // Duration over time, and the machine while these runs happened.
    var done = runsOf.filter(function (r) { return r.ended_at && r.seconds != null; }).slice().reverse();
    var t0 = done.length ? done[0].started_at : now() - 3600, t1 = now();
    var maxDur = Math.max.apply(null, done.map(function (r) { return r.seconds; }).concat([1]));
    var durations = done.length > 1 ? timeChart([
      { values: done.map(function (r) { return [r.ended_at, r.seconds]; }), color: "var(--accent)" },
      { values: done.map(function (r) { return [r.ended_at, r.waited || 0]; }), color: "var(--warn)" },
    ], t0, t1, { height: 120, max: maxDur, maxLabel: dur(maxDur) }) : '<div class="empty" style="text-align:left">Needs two finished runs.</div>';
    var tr = data.trace || [], marks = runsOf.map(function (r) { return { start: r.started_at, end: r.ended_at }; });
    var ramTotal = latest && latest.ram ? latest.ram.total : null;
    var vramTotal = latest && latest.card ? latest.card.total_bytes : null;
    var machine = tr.length > 1 ? timeChart([
      { values: tr.map(function (p) { return [p.at, p.cpu]; }), color: "var(--accent)" },
      { values: tr.map(function (p) { return [p.at, p.ram == null || !ramTotal ? null : pct(p.ram, ramTotal)]; }), color: "var(--violet)" },
      { values: tr.map(function (p) { return [p.at, p.vram == null || !vramTotal ? null : pct(p.vram, vramTotal)]; }), color: "var(--ok)" },
    ], tr[0].at, Math.max(tr[tr.length - 1].at, tr[0].at + 1), { height: 130, marks: marks })
      : '<div class="empty" style="text-align:left">The machine trace covers the last six hours of this process only.</div>';

    return head + '<div class="dbody">' + kpis + section("Where the time goes, on average", stack) +
      section("Recent runs, each from its own start", runsGantt) +
      '<div class="two">' +
      section("Duration of each run", '<div class="legend"><span><i style="background:var(--accent)"></i>took</span>' +
        '<span><i style="background:var(--warn)"></i>waited</span></div>' + durations) +
      section("The machine while it ran", '<div class="legend"><span><i style="background:var(--accent)"></i>cpu</span>' +
        '<span><i style="background:var(--violet)"></i>ram</span><span><i style="background:var(--ok)"></i>vram</span>' +
        '<span>shaded: this tool running</span></div>' + machine) +
      "</div></div>";
  }

  // ---- assembly -----------------------------------------------------------
  function draw(d) {
    latest = d;
    var srv = d.server || {};
    el("origin").textContent = window.location.host || "this server";
    el("host").textContent = [srv.node, srv.tools ? "tools: " + srv.tools : null, srv.device,
      (d.hardware || "").split(" · ").slice(1).join(" · ") || null,
      "updated " + new Date().toLocaleTimeString()].filter(Boolean).join("  ·  ");
    drawVitals(d);
    drawQueue(d);
    drawRunning(d);
    drawMachine(d);
    drawTools(d);
    drawHistory(d);
    drawFilters();
    if (dlg.kind) {
      // A live run's detail follows it; a tool view is heavier and refreshes slower.
      var every = dlg.kind === "run" ? EVERY : 10000;
      if (Date.now() - dlg.fetchedAt >= every - 200 && !(dlg.data && dlg.data.reaped)) { fetchDialog(); }
      else { drawDialog(); }
    }
  }

  function load() {
    if (!token) { el("gate").hidden = false; el("shell").hidden = true; return; }
    fetch("server-debug.json", { headers: { Authorization: "Bearer " + token } })
      .then(function (r) {
        if (r.status === 401) { throw new Error("That token was refused."); }
        if (!r.ok) { throw new Error("The server answered " + r.status + "."); }
        return r.json();
      })
      .then(function (d) { el("gate").hidden = true; el("shell").hidden = false; draw(d); })
      .catch(function (e) { el("gate").hidden = false; el("shell").hidden = true; el("gateerr").textContent = e.message; });
  }
  function tick() { if (live) { load(); } window.setTimeout(tick, EVERY); }

  el("save").addEventListener("click", function () {
    token = el("token").value.trim();
    try { window.localStorage.setItem(KEY, token); } catch (e) { /* private mode */ }
    el("token").value = "";
    el("gateerr").textContent = "";
    load();
  });
  el("token").addEventListener("keydown", function (event) { if (event.key === "Enter") { el("save").click(); } });
  el("toggle").addEventListener("click", function () {
    live = !live;
    el("toggle").textContent = live ? "Pause" : "Resume";
    el("dot").className = live ? "dot" : "dot off";
    el("livetext").textContent = live ? "live" : "paused";
    if (live) { load(); }
  });
  el("theme").addEventListener("click", function () {
    theme = theme === "auto" ? "dark" : theme === "dark" ? "light" : "auto";
    try { window.localStorage.setItem(THEME_KEY, theme); } catch (e) { /* private mode */ }
    applyTheme(theme);
  });
  var findTimer = null;
  el("find").addEventListener("input", function () {
    window.clearTimeout(findTimer);
    findTimer = window.setTimeout(function () {
      filters.text = el("find").value.trim();
      writeUrl();
      if (latest) { draw(latest); }
    }, 180);
  });

  document.addEventListener("click", function (event) {
    var t = event.target;
    if (t.closest("[data-close]") || t === el("overlay")) { closeDialog(); return; }
    var chip = t.closest("[data-clear]");
    if (chip) { setFilter(chip.getAttribute("data-clear"), filters[chip.getAttribute("data-clear")]); return; }
    var fbtn = t.closest("[data-filter]");
    if (fbtn) { closeDialog(); setFilter(fbtn.getAttribute("data-filter"), fbtn.getAttribute("data-value")); return; }
    var copy = t.closest("[data-copy]");
    if (copy) {
      if (navigator.clipboard) { navigator.clipboard.writeText(copy.getAttribute("data-copy")); }
      copy.textContent = "Copied";
      window.setTimeout(function () { copy.textContent = "Copy id"; }, 1200);
      return;
    }
    var toolNode = t.closest("[data-tool]");
    if (toolNode) { openDialog("tool", toolNode.getAttribute("data-tool")); return; }
    var runNode = t.closest("[data-run]");
    if (runNode) { openDialog("run", runNode.getAttribute("data-run")); }
  });
  document.addEventListener("keydown", function (event) { if (event.key === "Escape" && dlg.kind) { closeDialog(); } });
  window.addEventListener("hashchange", function () { readUrl(); if (latest) { draw(latest); } });

  // A stable hue per origin, so two deployments open side by side never look alike.
  (function () {
    var key = window.location.host || "local", hash = 0, i;
    for (i = 0; i < key.length; i++) { hash = (hash * 31 + key.charCodeAt(i)) % 360; }
    el("flag").style.background = "hsl(" + hash + " 60% 52%)";
  })();

  readUrl();
  drawFilters();
  load();
  window.setTimeout(tick, EVERY);
})();
</script>
</body>
</html>
"""
