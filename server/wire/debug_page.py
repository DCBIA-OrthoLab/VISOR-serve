"""The page `GET /admin-panel` serves: the operator's view of this server.

Opened with the ADMIN token only. The API token every workstation holds opens
nothing here: reading the queue and acting on it are the operator's.

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
<title>VISOR admin panel</title>
<style>
  /* Glass over light: frosted panels on a softly lit ground, after macOS.
     The ground carries a few wide colour glows so the blur behind each panel
     has something to pick up; the panels themselves stay neutral. */
  :root {
    --bg: #eef1f6;
    --glow-1: rgba(120,170,255,.45); --glow-2: rgba(190,150,255,.35); --glow-3: rgba(110,220,200,.30);
    --panel: rgba(255,255,255,.58); --panel-strong: rgba(255,255,255,.78);
    --panel-edge: rgba(255,255,255,.75); --sunk: rgba(255,255,255,.55);
    --line: rgba(60,60,67,.14); --ink: #1d1d1f; --soft: #5f6368; --ghost: #8e8e93;
    --bar: rgba(120,120,128,.16); --accent: #007aff; --accent-soft: rgba(0,122,255,.12);
    --ok: #28a745; --warn: #e8890c; --hot: #ff3b30; --violet: #8e5bd8; --staging: #8e8e93;
    --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
    --shadow: 0 1px 1px rgba(0,0,0,.03), 0 8px 28px rgba(30,40,60,.08);
    --shadow-lg: 0 30px 80px rgba(20,30,50,.28);
    --blur: blur(28px) saturate(180%);
    --radius: 16px;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #0b0d12;
      --glow-1: rgba(40,90,200,.38); --glow-2: rgba(110,60,190,.30); --glow-3: rgba(20,140,120,.24);
      --panel: rgba(30,32,38,.55); --panel-strong: rgba(36,38,44,.82);
      --panel-edge: rgba(255,255,255,.09); --sunk: rgba(255,255,255,.05);
      --line: rgba(235,235,245,.12); --ink: #f5f5f7; --soft: #a1a1a6; --ghost: #6e6e73;
      --bar: rgba(120,120,128,.28); --accent: #0a84ff; --accent-soft: rgba(10,132,255,.18);
      --ok: #30d158; --warn: #ff9f0a; --hot: #ff453a; --violet: #bf5af2; --staging: #8e8e93;
      --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
      --shadow: 0 1px 1px rgba(0,0,0,.2), 0 8px 28px rgba(0,0,0,.35);
      --shadow-lg: 0 30px 80px rgba(0,0,0,.7);
    }
  }
  :root[data-theme="dark"] {
    --bg: #0b0d12;
    --glow-1: rgba(40,90,200,.38); --glow-2: rgba(110,60,190,.30); --glow-3: rgba(20,140,120,.24);
    --panel: rgba(30,32,38,.55); --panel-strong: rgba(36,38,44,.82);
    --panel-edge: rgba(255,255,255,.09); --sunk: rgba(255,255,255,.05);
    --line: rgba(235,235,245,.12); --ink: #f5f5f7; --soft: #a1a1a6; --ghost: #6e6e73;
    --bar: rgba(120,120,128,.28); --accent: #0a84ff; --accent-soft: rgba(10,132,255,.18);
    --ok: #30d158; --warn: #ff9f0a; --hot: #ff453a; --violet: #bf5af2; --staging: #8e8e93;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
    --shadow: 0 1px 1px rgba(0,0,0,.2), 0 8px 28px rgba(0,0,0,.35);
    --shadow-lg: 0 30px 80px rgba(0,0,0,.7);
  }
  * { box-sizing: border-box; }
  /* Scrollbars in the page's own colours: the browser's default is a light
     gutter that sits on a dark theme like a stripe of paint. `color-scheme`
     also gives form controls and the default bars the right palette. */
  :root { color-scheme: light; --thumb: #c5cad2; --thumb-hover: #9aa1ab; }
  :root[data-theme="dark"] { color-scheme: dark; --thumb: #343c47; --thumb-hover: #4a5462; }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) { color-scheme: dark; --thumb: #343c47; --thumb-hover: #4a5462; }
  }
  * { scrollbar-width: thin; scrollbar-color: var(--thumb) transparent; }
  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--thumb); border-radius: 999px;
                              border: 2px solid transparent; background-clip: padding-box; }
  ::-webkit-scrollbar-thumb:hover { background: var(--thumb-hover); background-clip: padding-box; }
  ::-webkit-scrollbar-corner { background: transparent; }
  [hidden] { display: none !important; }
  @media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition: none !important; animation: none !important; } }
  html, body { min-height: 100%; }
  body {
    margin: 0; color: var(--ink); background-color: var(--bg);
    background-image:
      radial-gradient(60vw 50vh at 8% -5%, var(--glow-1), transparent 70%),
      radial-gradient(55vw 55vh at 100% 15%, var(--glow-2), transparent 70%),
      radial-gradient(60vw 50vh at 45% 110%, var(--glow-3), transparent 70%);
    background-attachment: fixed;
    font: 14px/1.45 -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", system-ui, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased; letter-spacing: -.003em;
  }
  body.locked { overflow: hidden; }
  .mono { font-variant-numeric: tabular-nums; }
  button, input, select { font: inherit; color: inherit; }
  button { background: var(--sunk); border: 1px solid var(--line); border-radius: 8px;
           padding: 6px 12px; cursor: pointer; transition: border-color .15s, background .15s; }
  button:hover { border-color: var(--soft); }
  button.ghost { border-color: transparent; background: transparent; }
  button.ghost:hover { background: var(--sunk); border-color: var(--line); }
  input { background: var(--sunk); border: 1px solid var(--line); border-radius: 9px;
          padding: 7px 11px; min-width: 220px; outline: none; }
  input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }

  /* ---- shell ---------------------------------------------------------- */
  #shell { max-width: 1680px; margin: 0 auto; padding: 10px 20px 28px;
           display: flex; flex-direction: column; gap: 14px; }
  header#bar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; position: sticky;
               top: 10px; z-index: 10; padding: 10px 14px; border-radius: var(--radius); }
  header button.ghost { padding: 6px 9px; }
  #flag { width: 10px; height: 38px; border-radius: 5px; background: var(--ghost); flex: none; }
  h1 { font-size: 20px; font-weight: 700; letter-spacing: -.015em; margin: 0; line-height: 1.15; }
  #host { color: var(--soft); font-size: 12px; }
  .grow { flex: 1; }
  #live { display: inline-flex; align-items: center; gap: 7px; font-size: 12px; color: var(--soft); }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ok);
         box-shadow: 0 0 0 3px color-mix(in srgb, var(--ok) 22%, transparent); }
  .dot.off { background: var(--ghost); box-shadow: none; }
  #filters { display: flex; gap: 6px; flex-wrap: wrap; }
  .navlink { color: var(--accent); text-decoration: none; font-size: 13px; font-weight: 600;
             padding: 6px 10px; border-radius: 8px; white-space: nowrap; }
  .navlink:hover { background: var(--accent-soft); }
  .fchip { display: inline-flex; align-items: center; gap: 6px; font-size: 12px;
           padding: 4px 9px 4px 11px; border-radius: 999px; background: var(--accent-soft);
           color: var(--accent); cursor: pointer; }
  .fchip i { font-style: normal; opacity: .7; }

  .card, #vitals, header#bar { background: var(--panel); -webkit-backdrop-filter: var(--blur); backdrop-filter: var(--blur);
          border: 1px solid var(--panel-edge); box-shadow: var(--shadow); }
  .card { border-radius: var(--radius); display: flex; flex-direction: column; min-width: 0; }
  .card > h2 { margin: 0; padding: 13px 16px 10px; font-size: 15px; font-weight: 600; letter-spacing: -.01em;
               display: flex; align-items: center; gap: 8px; border-bottom: 1px solid var(--line); }
  .card > h2 .count { font-size: 12px; font-weight: 600; color: var(--soft); }
  .card > h2 .note { margin-left: auto; font-weight: 400; color: var(--ghost); font-size: 11.5px; }
  .body { padding: 12px 14px; min-height: 0; }
  .scroll { overflow: auto; }
  .empty { color: var(--ghost); font-size: 12.5px; padding: 14px 4px; text-align: center; }
  .label { font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: var(--ghost);
           font-weight: 600; }

  /* ---- vitals --------------------------------------------------------- */
  /* One strip, divided, rather than six tiles: these are readings of one
     machine, and boxing each one made them look like six separate things. */
  #vitals { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); border-radius: var(--radius); }
  @media (max-width: 1200px) { #vitals { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
  @media (max-width: 640px) { #vitals { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  .vital { padding: 10px 14px 6px; position: relative; overflow: hidden; border: none;
           border-radius: 0; background: transparent; border-left: 1px solid var(--line); }
  .vital:first-child { border-left: none; }
  .vital .v { font-size: 26px; font-weight: 600; letter-spacing: -.02em; line-height: 1.15; margin-top: 2px; }
  .vital .v small { font-size: 13px; font-weight: 500; color: var(--soft); margin-left: 3px; }
  .vital .f { font-size: 11.5px; color: var(--soft); margin-top: 1px; white-space: nowrap;
              overflow: hidden; text-overflow: ellipsis; }
  .vital svg { display: block; width: 100%; height: 30px; margin-top: 6px; }
  .vital.busy .v { color: var(--accent); } .vital.wait .v { color: var(--warn); }
  .vital.hot .v { color: var(--hot); }

  /* ---- main grid ------------------------------------------------------- */
  #main { display: grid; gap: 14px; grid-template-columns: minmax(300px, 360px) minmax(0, 1fr) minmax(320px, 400px); }
  /* The queue is what an operator acts on, so it takes the column; paused
     runs wait on a person and keep a short, scrolling strip below it. */
  #queue { flex: 1 1 auto; }
  #queue .body { min-height: 420px; }
  #q-list { max-height: 640px; overflow: auto; }
  #paused { flex: 0 0 auto; }
  #p-list { max-height: 150px; overflow: auto; }
  #paused .qitem { padding: 5px 9px; margin-bottom: 5px; }
  #paused .qitem .pos { width: 20px; height: 20px; font-size: 10px; }
  @media (max-width: 1200px) { #main { grid-template-columns: 1fr; } }
  #left { display: flex; flex-direction: column; gap: 14px; min-width: 0; }

  /* the queue */
  .qitem { display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; padding: 8px 10px;
           border: 1px solid var(--line); border-radius: 12px; margin-bottom: 7px; cursor: pointer;
           background: color-mix(in srgb, var(--warn) 7%, transparent); }
  .qitem:hover { border-color: var(--warn); }
  .qitem .pos { width: 24px; height: 24px; border-radius: 50%; flex: none; display: grid;
                place-items: center; font-size: 12px; font-weight: 700; color: #fff; background: var(--warn); }
  .qitem .n { font-weight: 650; font-size: 13.5px; }
  .qitem .m { font-size: 11.5px; color: var(--soft); }
  .qitem.held { background: transparent; border-style: dashed; }
  .qitem.slot { background: transparent; }
  .qitem.slot .pos { background: var(--staging); }
  .qitem.held .pos { background: var(--ghost); }
  .rank { display: flex; align-items: center; gap: 8px; margin-top: 6px; font-size: 12px; }
  .rank .nm { width: 92px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .track { flex: 1; height: 6px; background: var(--bar); border-radius: 3px; overflow: hidden; }
  .track i { display: block; height: 100%; border-radius: 3px; background: var(--warn); }
  .rank .vv { color: var(--soft); min-width: 46px; text-align: right; }

  /* running */
  .runs { display: grid; gap: 10px; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); }
  .run { border: 1px solid var(--panel-edge); border-radius: 12px; padding: 11px 13px 12px;
         background: var(--sunk); cursor: pointer; position: relative; overflow: hidden;
         transition: border-color .15s; }
  .run:hover { border-color: var(--accent); }
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
  .cell { border-left: 1px solid var(--line); padding: 1px 8px; }
  .cell:first-child { border-left: none; padding-left: 0; }
  .cell u { display: block; font-size: 9.5px; letter-spacing: .06em; color: var(--ghost);
            text-transform: uppercase; text-decoration: none; }
  .cell b { font-size: 12.5px; font-weight: 600; white-space: nowrap; }

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
  .tool { border: 1px solid var(--panel-edge); border-radius: 12px; padding: 11px 13px; cursor: pointer;
          background: var(--sunk); transition: border-color .15s; }
  .tool:hover { border-color: var(--accent); }
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
  .pill { display: inline-block; font-size: 11px; font-weight: 600; padding: 1px 7px; border-radius: 4px;
          background: var(--sunk); color: var(--soft); white-space: nowrap; }
  .pill.running, .pill.done, .pill.packaging { background: color-mix(in srgb, var(--ok) 14%, transparent); color: var(--ok); }
  .pill.queued_gpu, .pill.paused { background: color-mix(in srgb, var(--warn) 16%, transparent); color: var(--warn); }
  .pill.failed { background: color-mix(in srgb, var(--hot) 14%, transparent); color: var(--hot); }
  .pill.staging, .pill.received { background: color-mix(in srgb, var(--staging) 16%, transparent); color: var(--staging); }

  /* ---- the dialog ------------------------------------------------------- */
  #overlay { position: fixed; inset: 0; z-index: 20; display: flex; align-items: flex-start;
             justify-content: center; padding: 5vh 20px; overflow: auto;
             background: rgba(0,0,0,.18); -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px);
             opacity: 0; transition: opacity .16s ease; }
  #overlay.open { opacity: 1; }
  #dialog { width: min(1180px, 100%); background: var(--panel-strong); border: 1px solid var(--panel-edge);
            -webkit-backdrop-filter: var(--blur); backdrop-filter: var(--blur);
            border-radius: 20px; box-shadow: var(--shadow-lg); transform: translateY(10px) scale(.99);
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
  .kpi { border-left: 2px solid var(--line); padding: 2px 12px; }
  .kpi u { display: block; font-size: 10.5px; letter-spacing: .05em; color: var(--ghost);
           text-transform: uppercase; text-decoration: none; font-weight: 600; }
  .kpi b { font-size: 17px; font-weight: 700; }
  .two { display: grid; gap: 18px; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
  @media (max-width: 900px) { .two { grid-template-columns: 1fr; } }
  table.io td { padding: 5px 10px 5px 0; white-space: normal; }
  table.io tr { cursor: default; }
  table.io tr:hover td { background: none; }
  table.io td:first-child { color: var(--soft); white-space: nowrap; }
  .con { background: var(--sunk); border: 1px solid var(--line); border-radius: 12px; padding: 9px 11px;
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

  /* maintenance & updates */
  .mrow { display: grid; grid-template-columns: 1.2fr 1fr 1.3fr auto; }
  @media (max-width: 1000px) { .mrow { grid-template-columns: 1fr; } }
  .mcell { padding: 12px 16px; border-left: 1px solid var(--line); min-width: 0; }
  .mcell:first-child { border-left: none; }
  .mcell .t { font-weight: 600; font-size: 14px; display: flex; align-items: center; gap: 8px; }
  .mcell .s { font-size: 12.5px; color: var(--soft); margin-top: 2px; }
  .mact { display: flex; flex-direction: column; gap: 6px; justify-content: center; }
  .sdot { width: 9px; height: 9px; border-radius: 50%; flex: none; background: var(--ok); }
  .sdot.closed { background: var(--warn); } .sdot.off { background: var(--ghost); } .sdot.up { background: var(--accent); }
  .chips { display: flex; gap: 5px; flex-wrap: wrap; margin-top: 6px; }
  .chip2 { font-size: 11.5px; padding: 1px 8px; border-radius: 6px; background: var(--sunk); border: 1px solid var(--line); }
  .chip2.env { border-color: var(--warn); color: var(--warn); }
  .doorctl { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; align-items: center; }
  .doorctl select { font-size: 12.5px; padding: 4px 8px; border-radius: 8px; border: 1px solid var(--line);
                    background: var(--sunk); color: var(--ink); }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
  button.primary:disabled { opacity: .45; cursor: default; }
  #m-progress:not(:empty) { border-top: 1px solid var(--line); padding: 10px 16px; font-size: 13px; }
  .commit { display: grid; grid-template-columns: 80px minmax(0, 1fr) 120px; gap: 10px; font-size: 12.5px;
            padding: 4px 0; border-bottom: 1px solid var(--line); }
  .commit .sha { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--ghost); }
  .warnbox { border: 1px solid color-mix(in srgb, var(--warn) 50%, transparent); border-radius: 10px;
             padding: 8px 12px; font-size: 12.5px; background: color-mix(in srgb, var(--warn) 8%, transparent); }
  .okbox { border: 1px solid color-mix(in srgb, var(--ok) 45%, transparent); border-radius: 10px;
           padding: 8px 12px; font-size: 12.5px; background: color-mix(in srgb, var(--ok) 8%, transparent); }
  /* full history */
  .hfilters { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .hfilters select, .hfilters input { font-size: 13px; padding: 6px 9px; border-radius: 9px;
      border: 1px solid var(--line); background: var(--sunk); color: var(--ink); min-width: 0; }
  .hfilters input { width: 220px; }
  .htable { max-height: 58vh; overflow: auto; border: 1px solid var(--line); border-radius: 12px; }
  .htable td.calls { white-space: normal; color: var(--soft); max-width: 260px; }
  .htable td small { color: var(--ghost); }
  /* clients */
  .client { display: grid; grid-template-columns: 290px minmax(0, 1fr); gap: 16px; padding: 12px 4px;
            border-bottom: 1px solid var(--line); }
  .client:last-child { border-bottom: none; }
  @media (max-width: 900px) { .client { grid-template-columns: 1fr; } }
  .client .addr { font-weight: 700; font-size: 14px; }
  .client .meta { font-size: 12px; color: var(--soft); margin-top: 2px; }
  .seg2 { display: inline-flex; gap: 2px; padding: 2px; border-radius: 9px; background: var(--bar); margin-top: 8px; }
  .seg2 button { border: none; border-radius: 7px; padding: 4px 11px; font-size: 12px; background: transparent; white-space: nowrap; }
  .seg2 button.on { background: var(--panel-strong); color: var(--ink); font-weight: 600;
                    box-shadow: 0 1px 3px rgba(0,0,0,.12), 0 0 0 .5px rgba(0,0,0,.04); }
  .seg2 button:disabled { cursor: default; opacity: 1; }
  .seg2 button:disabled:not(.on) { color: var(--ghost); }
  .cohort { display: grid; grid-template-columns: 130px minmax(0, 1fr) 150px; gap: 12px; align-items: center;
            font-size: 12.5px; padding: 4px 0; }
  .cohort .tl b { font-weight: 650; }
  .cohort .tl small { display: block; color: var(--ghost); font-size: 11px; }
  .cbar { display: flex; height: 10px; border-radius: 5px; overflow: hidden; background: var(--bar); }
  .cbar i { display: block; height: 100%; }
  .cohort .vv { text-align: right; color: var(--soft); }
  .tag { display: inline-block; font-size: 10.5px; font-weight: 650; padding: 0 6px; border-radius: 5px;
         background: var(--accent-soft); color: var(--accent); }
  /* operator controls */
  .ctl { display: flex; gap: 4px; flex: none; width: 100%; justify-content: flex-end;
         padding-left: 34px; margin-top: -2px; }
  .ctl button { padding: 2px 7px; font-size: 12px; border-radius: 6px; line-height: 1.3; }
  .ctl button.star.on { background: var(--hot); border-color: var(--hot); color: #fff; }
  .qitem.high { border-color: color-mix(in srgb, var(--hot) 55%, transparent); background: color-mix(in srgb, var(--hot) 8%, transparent); }
  .qitem.high .pos { background: var(--hot); }
  .prio { font-size: 10.5px; font-weight: 700; color: var(--hot); letter-spacing: .04em; }
  .adminbox { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .toast { position: fixed; bottom: 18px; left: 50%; transform: translateX(-50%); z-index: 40;
           background: var(--ink); color: var(--panel); padding: 8px 14px; border-radius: 9px;
           font-size: 13px; box-shadow: var(--shadow-lg); }
  #gate { max-width: 520px; margin: 14vh auto; }
  .err { color: var(--hot); font-size: 12.5px; margin-top: 8px; }
</style>
</head>
<body>

<div id="gate" class="card" hidden>
  <h2>Admin panel</h2>
  <div class="body">
    <p style="margin:0 0 12px;font-size:13px;color:var(--soft)">
      This panel opens with the server's admin token (<code>ADMIN_TOKEN</code>).
      The API token a workstation uses for its runs does not open it.
      The token stays in this browser and is never written into this page.</p>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <input id="token" type="password" placeholder="Admin token" autocomplete="off">
      <button id="save" type="button">Connect</button>
    </div>
    <div class="err" id="gateerr"></div>
  </div>
</div>

<div id="shell" hidden>
  <header id="bar">
    <span id="flag" title="derived from this page's own origin"></span>
    <div>
      <h1 id="origin">&nbsp;</h1>
      <div id="host" class="mono">connecting&hellip;</div>
    </div>
    <span class="grow"></span>
    <span id="filters"></span>
    <input id="find" style="min-width:200px;width:220px" type="search" placeholder="Filter: run id, tool or address" autocomplete="off" spellcheck="false">
    <span id="live"><span class="dot" id="dot"></span><span id="livetext">live</span></span>
    <button id="toggle" type="button" class="ghost">Pause</button>
    <a class="navlink" href="benchmark" title="launch a benchmark preset on this server">Run a benchmark</a>
    <a class="navlink" href="benchmarks/view" title="the campaigns already measured, drawn on one time axis">Benchmark results</a>
    <button id="adminbtn" type="button" class="ghost" title="forget the admin token in this browser">Lock</button>
    <button id="theme" type="button" class="ghost" title="theme">Theme</button>
  </header>

  <section id="vitals"></section>

  <section class="card" id="maint">
    <div class="mrow">
      <div class="mcell" id="m-door"></div>
      <div class="mcell" id="m-server"></div>
      <div class="mcell" id="m-tools"></div>
      <div class="mcell mact" id="m-act"></div>
    </div>
    <div id="m-progress"></div>
  </section>

  <div id="main">
    <div id="left">
      <section class="card" id="queue">
        <h2>Queue <span class="count" id="q-count">0</span><span class="note">waiting for the machine</span></h2>
        <div class="body">
          <div id="q-list"></div>
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
        <div class="legend" id="m-legend"></div>
        <div id="m-chart"></div>
        <div class="legend" style="margin-top:8px"><span><i style="background:var(--ok)"></i>running</span>
          <span><i style="background:var(--warn)"></i>waiting</span></div>
        <div id="m-load"></div>
        <div id="m-disk" style="margin-top:10px"></div>
        <div style="margin-top:14px"><div class="label">Time spent waiting, by tool</div><div id="q-rank"></div></div>
      </div>
    </section>
  </div>

  <section class="card" id="clientscard">
    <h2>Clients <span class="count" id="c-count">0</span><span class="note" id="c-note">workstations seen in the last 24 h, and their cohorts</span></h2>
    <div class="body" id="c-list"></div>
  </section>

  <section class="card" id="tools">
    <h2>Tools<span class="note">click a tool for its runs and graphs</span></h2>
    <div class="body"><div class="grid" id="t-list"></div></div>
  </section>

  <section class="card" id="history">
    <h2>History <span class="count" id="h-count">0</span><span class="note" id="h-note"></span>
      <button type="button" class="ghost" data-history="1" style="margin-left:6px">Full history</button></h2>
    <div class="scroll" id="h-body"></div>
  </section>
</div>

<div id="overlay" hidden><div id="dialog" role="dialog" aria-modal="true"></div></div>
<div id="toast" class="toast" hidden></div>

<script>
(function () {
  "use strict";
  var KEY = "visor.admin", THEME_KEY = "visor.theme";
  var EVERY = 2000;
  var token = "";
  try { token = window.localStorage.getItem(KEY) || ""; } catch (e) { token = ""; }

  var live = true, latest = null;
  // Whether the controls are live: the panel is opened with the admin token,
  // so once it reads anything at all, every action is allowed.
  var admin = { ok: false };
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
    return '<div class="vital ' + (cls || "") + '"><div class="label">' + esc(label) + "</div>" +
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
        " cores held", cpu > 90 ? "hot" : "", spark(traceOf(d, "cpu"), "var(--series-1)", 100)) +
      vital("RAM", ram ? (ram.used / 1073741824).toFixed(0) : "—", ram ? "/ " + (ram.total / 1073741824).toFixed(0) + " G" : "",
        gib(a.ram_held) + " held by runs", ram && pct(ram.used, ram.total) > 90 ? "hot" : "",
        spark(traceOf(d, "ram", ram && ram.total), "var(--series-2)", 100)) +
      vital("VRAM", vramUsed == null ? "—" : (vramUsed / 1073741824).toFixed(1),
        card.total_bytes ? "/ " + (card.total_bytes / 1073741824).toFixed(0) + " G" : "",
        card.total_bytes ? gib(a.vram_held) + " held by runs" : "no card", "",
        spark(traceOf(d, "vram", card.total_bytes), "var(--series-3)", 100)) +
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
    // The order admission will actually admit in, which an operator may have
    // changed; runs still waiting for a slot follow, oldest first.
    var adm = d.admission || {}, prios = adm.priorities || {}, position = {}, serialClients = {};
    (d.clients || []).forEach(function (c) { if (c.batches === "serial") { serialClients[c.client] = true; } });
    (adm.queue || []).forEach(function (entry) { if (entry.run_id) { position[entry.run_id] = entry.position; } });
    var runOf = {};
    (d.runs || []).forEach(function (r) { runOf[r.run_id] = r; });
    // Nested calls waiting for room: a tool another tool called, queued like a
    // run of its own. Shown under the run that called it.
    var nestedWaiting = (adm.queue || []).filter(function (e) {
      var parent = runOf[e.parent];
      return e.nested && (!parent || matches(parent, byId[parent.run_id]));
    }).map(function (e) {
      var parent = runOf[e.parent] || {};
      return { run_id: e.run_id, tool: e.tool, phase: "queued_gpu", nested: true, started_at: e.since,
               client: parent.client, calledBy: parent.tool };
    });
    var waiting = (d.runs || []).filter(function (r) {
      return inFlight(r) && (r.phase === "queued_gpu" || r.phase === "received") && matches(r, byId[r.run_id]);
    }).concat(nestedWaiting).sort(function (x, y) {
      var px = position[x.run_id] || 1e6, py = position[y.run_id] || 1e6;
      return px - py || x.started_at - y.started_at;
    });
    el("q-count").textContent = waiting.length;
    el("q-list").innerHTML = waiting.length ? waiting.map(function (r, i) {
      var slot = r.phase === "received", high = prios[r.run_id] === "high";
      // A serial cohort's later batches wait at the batch gate, before any
      // slot: say which batch they are waiting for rather than "a slot".
      var sibling = null;
      if (slot && r.batch && serialClients[r.client]) {
        (d.runs || []).forEach(function (o) {
          if (o.run_id !== r.run_id && o.batch && o.batch.id === r.batch.id && o.client === r.client &&
              inFlight(o) && o.batch.index < r.batch.index && (!sibling || o.batch.index < sibling)) {
            sibling = o.batch.index;
          }
        });
      }
      var ctl = r.nested ? "" : admin.ok ? '<span class="ctl">' +
        (slot ? "" : '<button data-move="top" data-id="' + esc(r.run_id) + '" title="to the top">\u2912</button>' +
          '<button data-move="up" data-id="' + esc(r.run_id) + '" title="up one">\u2191</button>' +
          '<button data-move="down" data-id="' + esc(r.run_id) + '" title="down one">\u2193</button>') +
        '<button class="star' + (high ? " on" : "") + '" data-prio="' + (high ? "normal" : "high") + '" data-id="' +
        esc(r.run_id) + '" title="' + (high ? "back to normal" : "give priority") + '">\u2605</button></span>' : "";
      return '<div class="qitem' + (slot ? " slot" : "") + (high ? " high" : "") + '" data-run="' + esc(r.run_id) + '"><span class="pos">' + (i + 1) + "</span>" +
        '<div style="min-width:0;flex:1"><div class="n">' + esc(r.tool || "?") + (high ? ' <span class="prio">PRIORITY</span>' : "") +
          (r.batch ? ' <span class="tag">batch ' + r.batch.index + "/" + r.batch.total + "</span>" : "") + "</div>" +
        '<div class="m mono">' + (r.nested ? "called by " + esc(r.calledBy || "?") + " \u00b7 " : "") +
        (sibling ? "waiting for batch " + sibling + " " : slot ? "waiting for a slot " : "waiting for room ") + ago(r.started_at) +
        (r.client ? " \u00b7 " + esc(r.client) : "") + "</div></div>" + ctl + "</div>";
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
        (r.batch ? '<span class="tag">batch ' + r.batch.index + "/" + r.batch.total + "</span>" : "") +
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
      { values: tr.map(function (p) { return [p.at, p.cpu]; }), color: "var(--series-1)" },
      { values: tr.map(function (p) { return [p.at, p.ram == null || !ramTotal ? null : pct(p.ram, ramTotal)]; }), color: "var(--series-2)" },
      { values: tr.map(function (p) { return [p.at, p.vram == null || !vramTotal ? null : pct(p.vram, vramTotal)]; }), color: "var(--series-3)" },
    ], t0, t1, { height: 120 }) : '<div class="empty">The trace starts with the server; a point every 5 s.</div>';
    var maxLoad = Math.max.apply(null, tr.map(function (p) { return Math.max(p.running || 0, p.waiting || 0); }).concat([2]));
    el("m-load").innerHTML = tr.length > 1 ? timeChart([
      { values: tr.map(function (p) { return [p.at, p.running]; }), color: "var(--ok)", step: true },
      { values: tr.map(function (p) { return [p.at, p.waiting]; }), color: "var(--warn)", step: true },
    ], t0, t1, { height: 74, max: maxLoad, maxLabel: maxLoad + " runs" }) : "";
    // The current value beside each name: the legend is how a series is
    // identified, and the light aqua is too pale to be read from its line alone.
    var lastP = tr.length ? tr[tr.length - 1] : {};
    function now_(v) { return v == null ? "\u2014" : v.toFixed(0) + "%"; }
    el("m-legend").innerHTML =
      '<span><i style="background:var(--series-1)"></i>cpu <b class="mono">' + now_(lastP.cpu) + "</b></span>" +
      '<span><i style="background:var(--series-2)"></i>ram <b class="mono">' + now_(lastP.ram == null || !ramTotal ? null : pct(lastP.ram, ramTotal)) + "</b></span>" +
      '<span><i style="background:var(--series-3)"></i>vram <b class="mono">' + now_(lastP.vram == null || !vramTotal ? null : pct(lastP.vram, vramTotal)) + "</b></span>";
    el("m-note").textContent = "trace kept in memory, resets with the server";
    el("m-disk").innerHTML = Object.keys(d.disk || {}).map(function (name) {
      var e = d.disk[name];
      return '<div class="disk"><span class="nm">' + esc(name) + '</span><span class="track"><i style="width:' +
        ((e.total && e.free != null) ? pct(e.total - e.free, e.total) : 0).toFixed(1) + '%"></i></span>' +
        '<span class="vv mono">' + (e.used_here == null ? "—" : gib(e.used_here)) + " here · " +
        gib(e.free) + " free</span></div>";
    }).join("");
  }

  // ---- clients ----------------------------------------------------------
  function drawClients(d) {
    var rows = (d.clients || []).filter(function (c) { return !filters.client || c.client === filters.client; });
    el("c-count").textContent = rows.length;
    var enabled = d.server && d.server.admin_enabled;
    el("c-list").innerHTML = rows.length ? rows.map(function (c) {
      var parallel = c.batches === "parallel";
      var rule = '<div class="seg2" title="how this workstation\'s cohort batches run">' +
        ["serial", "parallel"].map(function (level) {
          return '<button ' + (admin.ok ? 'data-rule="' + level + '" data-addr="' + esc(c.client) + '"' : "disabled") +
            ' class="' + (c.batches === level ? "on" : "") + '">' + (level === "serial" ? "One batch at a time" : "Batches in parallel") + "</button>";
        }).join("") + "</div>";
      var cohorts = (c.cohorts || []).map(function (g) {
        var total = g.total || (g.done + g.failed + g.running + g.waiting) || 1;
        var sent = g.done + g.failed + g.running + g.waiting;
        var bar = '<div class="cbar">' +
          '<i style="width:' + pct(g.done, total).toFixed(1) + '%;background:var(--ok)"></i>' +
          '<i style="width:' + pct(g.failed, total).toFixed(1) + '%;background:var(--hot)"></i>' +
          '<i style="width:' + pct(g.running, total).toFixed(1) + '%;background:var(--accent)"></i>' +
          '<i style="width:' + pct(g.waiting, total).toFixed(1) + '%;background:var(--warn)"></i></div>';
        var live = g.running || g.waiting;
        return '<div class="cohort"><span class="tl"><b>' + esc(g.tool || "?") + "</b><small>started " + ago(g.started_at) +
          " ago</small></span>" + bar + '<span class="vv mono">' + g.done + " / " + total + " done" +
          (g.running ? " \u00b7 " + g.running + " running" : "") + (g.waiting ? " \u00b7 " + g.waiting + " waiting" : "") +
          (!live && sent < total ? " \u00b7 stopped" : "") + "</span></div>";
      }).join("");
      return '<div class="client"><div><div class="addr mono">' + esc(c.client) + "</div>" +
        '<div class="meta">' + (c.running ? c.running + " running \u00b7 " : "") + (c.waiting ? c.waiting + " waiting \u00b7 " : "") +
        c.runs_today + " runs today \u00b7 seen " + ago(c.last_seen) + " ago</div>" + rule + "</div>" +
        "<div>" + (cohorts || '<div class="empty" style="text-align:left;padding:6px 0">No cohort in the last 15 minutes.</div>') + "</div></div>";
    }).join("") : '<div class="empty">No workstation has sent a run in the last 24 hours.</div>';
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
        return '<tr data-run="' + esc(r.run_id) + '"><td><b>' + esc(r.tool || "?") + "</b>" +
          (r.batch ? ' <span class="tag">batch ' + r.batch.index + "/" + r.batch.total + "</span>" : "") + "</td>" +
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
  function openDialog(kind, id, back) {
    dlg = { kind: kind, id: id, data: null, error: "", fetchedAt: 0, back: back || null };
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
    if (dlg.kind === "updates") { loadUpdates(true); drawDialog(); return; }
    if (dlg.kind === "tool") { loadUpdates(true); }
    var kind = dlg.kind, id = dlg.id;
    var url = kind === "run" ? "admin-panel/runs/" + encodeURIComponent(id) + ".json"
            : kind === "history" ? "admin-panel/history.json?limit=1000"
            : "admin-panel/tools/" + encodeURIComponent(id) + ".json";
    dlg.fetchedAt = Date.now();
    fetch(url, { headers: { "X-Admin-Token": token } })
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
    el("dialog").innerHTML = dlg.kind === "run" ? runDialog() : dlg.kind === "history" ? historyDialog()
      : dlg.kind === "updates" ? updatesDialog() : toolDialog();
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
      (admin.ok && (state === "queued_gpu" || state === "received" || state === "staging")
        ? '<button class="ghost" data-prio="' + (((latest.admission || {}).priorities || {})[id] === "high" ? "normal" : "high") +
          '" data-id="' + esc(id) + '">' + (((latest.admission || {}).priorities || {})[id] === "high" ? "Remove priority" : "\u2605 Give priority") + "</button>"
        : "") +
      '<button class="ghost" data-copy="' + esc(id) + '">Copy id</button>' +
      (dlg.back ? '<button class="ghost" data-back="1">\u2190 History</button>' : "") +
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
      kpi("batch", (liveRun.batch || led.batch) ? (liveRun.batch || led.batch).index + " of " + (liveRun.batch || led.batch).total : "\u2014") +
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

  // ---- maintenance & updates ------------------------------------------
  var upd = null, updFetchedAt = 0, doorHours = "2", checking = false;
  function loadUpdates(force) {
    if (!token || (!force && Date.now() - updFetchedAt < 4000)) { return; }
    updFetchedAt = Date.now();
    fetch("admin-panel/updates.json", { headers: { "X-Admin-Token": token } })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d) { return; }
        upd = d;
        drawMaint();
        if (dlg.kind === "updates" || dlg.kind === "tool") { drawDialog(); }
      })
      .catch(function () { /* the next poll tries again */ });
  }
  function repoSummary(info, label) {
    if (!info || info.error) {
      return '<div class="t"><span class="sdot off"></span>' + label + '</div><div class="s">' +
        esc(info && info.error ? info.error : "not reported") + "</div>";
    }
    var behind = info.behind || 0;
    var head = '<div class="t"><span class="sdot ' + (behind ? "up" : "") + '"></span>' + label + " \u00b7 " +
      (behind ? behind + " update" + (behind > 1 ? "s" : "") + " waiting" : "up to date") + "</div>" +
      '<div class="s mono">' + esc(info.branch) + " @ " + esc(info.local) +
      (info.current && info.current.subject ? " \u00b7 " + esc(info.current.subject.slice(0, 60)) : "") + "</div>";
    var extra = "";
    if (info.kind === "tools" && behind) {
      extra = '<div class="chips">' + ((info.changes || {}).tools || []).map(function (t) {
        return '<span class="chip2' + (t.environment ? " env" : "") + '" title="' +
          (t.environment ? "dependencies changed: its environment is rebuilt" : "code only") + '">' +
          esc(t.tool) + (t.environment ? " \u00b7 env" : "") + "</span>";
      }).join("") + "</div>";
    }
    if (info.kind === "server" && behind && info.changes) {
      var what = { restart: "a restart applies it", recreate: "the container is recreated",
                   image: "needs a new image: cannot be applied from here" }[info.changes.action] || "";
      extra = '<div class="s">' + esc(what) + "</div>";
    }
    return head + extra;
  }
  function drawMaint() {
    var m = (upd && upd.maintenance) || (latest && latest.maintenance) || { accepting: true };
    var door = m.accepting
      ? '<div class="t"><span class="sdot"></span>Accepting new runs</div>' +
        '<div class="s">New runs are admitted as usual.</div>' +
        '<div class="doorctl"><select id="doorhours">' + ["1", "2", "4", "12"].map(function (h) {
          return '<option value="' + h + '"' + (doorHours === h ? " selected" : "") + ">for " + h + " h</option>";
        }).join("") + "</select>" +
        '<button data-door="close">Stop accepting new runs</button></div>'
      : '<div class="t"><span class="sdot closed"></span>Closed to new runs</div>' +
        '<div class="s">' + esc(m.reason || "") + " \u00b7 reopens by itself in " + dur(m.closed_for) +
        ". Runs already in finish normally.</div>" +
        '<div class="doorctl"><button class="primary" data-door="open">Reopen now</button></div>';
    el("m-door").innerHTML = door;
    if (!upd) {
      el("m-server").innerHTML = '<div class="s">Loading\u2026</div>';
      el("m-tools").innerHTML = ""; el("m-act").innerHTML = ""; el("m-progress").innerHTML = "";
      return;
    }
    // With an agent on the host, its report (and its one-click apply). Without
    // one, the server's own read-only check: what is waiting, and how to apply.
    var alive = upd.agent.alive, st = upd.status || {};
    var view = alive ? st : (upd.check || {});
    if (!alive && !upd.check) {
      el("m-server").innerHTML = '<div class="t"><span class="sdot off"></span>Checking for updates\u2026</div>';
      el("m-tools").innerHTML = "";
    } else {
      el("m-server").innerHTML = repoSummary(view.server, "Server");
      el("m-tools").innerHTML = repoSummary(view.tools, "Tools library");
    }
    var waiting = ((view.server || {}).behind || 0) + ((view.tools || {}).behind || 0);
    var checkedAt = alive ? upd.agent.heartbeat : (upd.check || {}).at;
    el("m-act").innerHTML = '<button class="' + (waiting ? "primary" : "") + '" data-updates="1">' +
      (waiting ? "Review" + (alive ? " &amp; update" : "") : "Details") + "</button>" +
      '<button class="ghost" data-check="1">' + ((upd.check || {}).running || checking ? "Checking\u2026" : "Check now") + "</button>" +
      '<span class="s" style="font-size:11.5px;color:var(--ghost)">' + (checkedAt ? "checked " + ago(checkedAt) + " ago" : "") +
      (alive ? "" : " \u00b7 no update agent") + "</span>";
    var progress = "";
    if (st.applying) {
      progress = '<b>Updating</b> \u00b7 ' + esc(st.applying.phase || "starting") +
        (upd.request ? ' <button class="ghost" data-withdraw="1" style="margin-left:8px">Cancel</button>' : "");
    } else if (upd.request) {
      progress = "<b>Update requested</b> \u00b7 waiting for the agent to pick it up " +
        '<button class="ghost" data-withdraw="1" style="margin-left:8px">Cancel</button>';
    } else if (st.last && now() - st.last.at < 3600) {
      progress = '<span style="color:' + (st.last.ok ? "var(--ok)" : "var(--hot)") + '">' +
        (st.last.ok ? "\u2713 " : "\u2715 ") + esc(st.last.message) + "</span> \u00b7 " + ago(st.last.at) + " ago";
    }
    el("m-progress").innerHTML = progress;
  }
  function commitsHtml(info) {
    if (!info || !(info.commits || []).length) { return '<div class="empty" style="text-align:left">Nothing waiting.</div>'; }
    return info.commits.map(function (c) {
      return '<div class="commit"><span class="sha">' + esc(c.sha) + "</span><span>" + esc(c.subject) +
        '</span><span style="color:var(--ghost)">' + esc(c.author) + " \u00b7 " + ago(c.at) + "</span></div>";
    }).join("");
  }
  function updatesDialog() {
    var alive = upd && upd.agent.alive;
    var st = (upd && upd.status) || {};
    var view = alive ? st : ((upd && upd.check) || {});
    var srv = view.server || {}, tl = view.tools || {};
    function block(info, label, target) {
      var reasons = info.blockers || [];
      var detail = "";
      if (info.kind === "tools" && info.behind) {
        detail = '<table class="io"><tbody>' + ((info.changes || {}).tools || []).map(function (t) {
          return "<tr><td><b>" + esc(t.tool) + "</b></td><td>" + t.files + " file" + (t.files > 1 ? "s" : "") +
            "</td><td>" + (t.environment ? '<span class="chip2 env">environment rebuilt (uv sync)</span>' : "code only, restart") + "</td></tr>";
        }).join("") + "</tbody></table>";
      }
      if (info.kind === "server" && info.behind && info.changes) {
        detail = '<div class="s">' + ({ restart: "Source only: the server is restarted.",
          recreate: "Dependencies or compose changed: the container is recreated (" + esc((info.changes.heavy || []).join(", ")) + ").",
          image: "The image changes: " + esc((info.changes.heavy || []).join(", ")) }[info.changes.action] || "") + "</div>";
      }
      return section(label + (info.branch ? " \u00b7 " + info.branch + " @ " + (info.local || "") : ""),
        (reasons.length ? '<div class="warnbox">Cannot be updated from here: ' + esc(reasons.join("; ")) + "</div>" : "") +
        detail + commitsHtml(info) +
        (info.behind && !reasons.length && alive ? '<div style="margin-top:8px"><button class="primary" data-update="' + target +
          '">Update ' + label.toLowerCase() + "</button></div>" : ""));
    }
    var busy = st.applying || (upd && upd.request);
    var both = alive && (srv.behind && tl.behind && !(srv.blockers || []).length && !(tl.blockers || []).length);
    var howto = alive ? "" : '<div class="warnbox">Applying an update needs the host, where the code lives. Either run ' +
      "<code>python3 scripts/server_ctl.py update</code> there, or start the update agent " +
      "(<code>python3 scripts/server_ctl.py agent</code>) to update from this panel in one click.</div>";
    var last = st.last ? (st.last.ok ? '<div class="okbox">' : '<div class="warnbox">') + esc(st.last.message) +
      " \u00b7 " + ago(st.last.at) + " ago" + ((st.last.log || []).length
        ? '<div class="con" style="margin-top:8px;max-height:180px">' + st.last.log.map(function (l) {
            return '<div class="ln"><span class="tx">' + esc(l) + "</span></div>"; }).join("") + "</div>" : "") + "</div>" : "";
    var head = '<div class="dhd"><div><h3>Updates</h3><div class="rid">Applying one stops new runs, waits for the ones in flight, ' +
      "pulls, rebuilds what changed and restarts the server. Runs already in are never interrupted.</div></div>" +
      '<div class="acts">' + (both && !busy ? '<button class="primary" data-update="all">Update both</button>' : "") +
      '<button data-close="1">Close \u2715</button></div></div>';
    var progress = busy ? '<div class="warnbox"><b>In progress:</b> ' + esc((st.applying || {}).phase || "requested, waiting for the agent") +
      ((st.applying || {}).log ? '<div class="con" style="margin-top:8px;max-height:160px">' + st.applying.log.map(function (l) {
        return '<div class="ln"><span class="tx">' + esc(l) + "</span></div>"; }).join("") + "</div>" : "") + "</div>" : "";
    return head + '<div class="dbody">' + howto + progress + (busy ? "" : '<div class="two">' + block(srv, "Server", "server") +
      block(tl, "Tools library", "tools") + "</div>") + (last ? section("Last update", last) : "") + "</div>";
  }

  // ---- the full history ------------------------------------------------
  function ranSeconds(r) {
    var total = 0;
    (r.spans || []).forEach(function (sp) { if (sp.phase === "running" && sp.end != null) { total += sp.end - sp.start; } });
    return total;
  }
  function callsOf(r) {
    var byTool = {};
    (r.nested || []).forEach(function (c) {
      if (c.end == null) { return; }
      byTool[c.tool] = (byTool[c.tool] || 0) + (c.end - c.start);
    });
    return Object.keys(byTool).map(function (t) { return esc(t) + " <small>" + dur(byTool[t]) + "</small>"; }).join(", ");
  }
  function dateTime(at) {
    if (!at) { return "\u2014"; }
    var d = new Date(at * 1000), today = new Date().toDateString() === d.toDateString();
    return (today ? "" : d.toLocaleDateString() + " ") + clock(at);
  }
  function historyDialog() {
    var data = dlg.data, f = dlg.hf || (dlg.hf = { tool: "", client: "", outcome: "", text: "" });
    var head = '<div class="dhd"><div><h3>History</h3><div class="rid">' +
      (data ? data.held + " finished runs kept on this server (up to " + data.capacity + "), across restarts" : "Loading\u2026") +
      '</div></div><div class="acts"><button class="ghost" data-hrefresh="1">Refresh</button>' +
      '<button data-close="1">Close \u2715</button></div></div>';
    if (!data) { return head + '<div class="dbody"><div class="empty">' + (dlg.error ? esc(dlg.error) : "Loading\u2026") + "</div></div>"; }
    var all = data.runs || [];
    function uniq(key) {
      var seen = {};
      all.forEach(function (r) { if (r[key]) { seen[r[key]] = true; } });
      return Object.keys(seen).sort();
    }
    function select(key, label, values) {
      return '<select data-hf="' + key + '"><option value="">' + label + "</option>" + values.map(function (v) {
        return '<option value="' + esc(v) + '"' + (f[key] === v ? " selected" : "") + ">" + esc(v) + "</option>";
      }).join("") + "</select>";
    }
    var rows = all.filter(function (r) {
      if (f.tool && r.tool !== f.tool) { return false; }
      if (f.client && r.client !== f.client) { return false; }
      if (f.outcome && r.outcome !== f.outcome) { return false; }
      if (f.text) {
        var hay = [r.run_id, r.tool, r.client, (r.batch || {}).id].join(" ").toLowerCase();
        if (hay.indexOf(f.text.toLowerCase()) < 0) { return false; }
      }
      return true;
    });
    var ok = rows.filter(function (r) { return r.outcome === "done"; }).length;
    var ran = 0, waited = 0, took = 0;
    var perTool = {};
    rows.forEach(function (r) {
      var rs = ranSeconds(r);
      ran += rs; waited += r.waited || 0; took += r.seconds || 0;
      var t = perTool[r.tool || "?"] = perTool[r.tool || "?"] || { runs: 0, ran: 0, took: 0, longest: 0, failed: 0 };
      t.runs += 1; t.ran += rs; t.took += r.seconds || 0; t.longest = Math.max(t.longest, r.seconds || 0);
      if (r.outcome !== "done") { t.failed += 1; }
    });
    var filters = '<div class="hfilters">' + select("tool", "All tools", uniq("tool")) +
      select("client", "All workstations", uniq("client")) + select("outcome", "Any outcome", uniq("outcome")) +
      '<input data-hf="text" type="search" placeholder="Run id, batch id\u2026" value="' + esc(f.text) + '">' +
      '<span style="color:var(--soft);font-size:12.5px">' + rows.length + " of " + all.length + " shown</span></div>";
    var kpis = '<div class="kpis">' + kpi("runs", rows.length) +
      kpi("succeeded", rows.length ? Math.round(100 * ok / rows.length) + "%" : "\u2014") +
      kpi("computing", dur(ran)) + kpi("waiting", dur(waited)) + kpi("wall clock", dur(took)) + "</div>";
    var tools = Object.keys(perTool).sort(function (a, b) { return perTool[b].ran - perTool[a].ran; });
    var maxRan = tools.length ? Math.max(1, perTool[tools[0]].ran) : 1;
    var byTool = '<table><thead><tr><th>tool</th><th class="r">runs</th><th class="r">failed</th><th>time computing</th>' +
      '<th class="r">computing</th><th class="r">mean took</th><th class="r">longest</th></tr></thead><tbody>' +
      tools.map(function (name) {
        var t = perTool[name];
        return '<tr data-tool="' + esc(name) + '"><td><b>' + esc(name) + '</b></td><td class="r mono">' + t.runs +
          '</td><td class="r mono">' + (t.failed || "\u2014") + '</td><td><span class="minibar" style="width:160px"><i style="width:' +
          pct(t.ran, maxRan).toFixed(1) + '%;background:var(--ok)"></i></span></td><td class="r mono">' + dur(t.ran) +
          '</td><td class="r mono">' + dur(t.took / t.runs) + '</td><td class="r mono">' + dur(t.longest) + "</td></tr>";
      }).join("") + "</tbody></table>";
    var table = '<div class="htable"><table><thead><tr><th>started</th><th>tool</th><th>calls</th><th>from</th>' +
      "<th>outcome</th><th>phases</th><th class='r'>computing</th><th class='r'>waited</th><th class='r'>took</th>" +
      "<th class='r'>chan</th><th class='r'>cpus</th><th class='r'>vram peak</th><th class='r'>ram peak</th>" +
      "<th class='r'>files</th><th class='r'>input</th></tr></thead><tbody>" +
      rows.map(function (r) {
        var m = r.measured || {};
        return '<tr data-run="' + esc(r.run_id) + '" data-from="history"><td class="mono">' + dateTime(r.started_at) + "</td>" +
          "<td><b>" + esc(r.tool || "?") + "</b>" + (r.batch ? ' <span class="tag">batch ' + r.batch.index + "/" + r.batch.total + "</span>" : "") + "</td>" +
          '<td class="calls">' + (callsOf(r) || "\u2014") + "</td>" +
          '<td class="mono">' + esc(r.client || "\u2014") + "</td><td>" + pill(r.outcome) + "</td>" +
          "<td>" + phaseBar(r.spans, r.seconds) + "</td>" +
          "<td class='r mono'>" + dur(ranSeconds(r)) + "</td>" +
          "<td class='r mono'>" + (r.waited == null ? "\u2014" : dur(r.waited)) + "</td>" +
          "<td class='r mono'>" + dur(r.seconds) + "</td>" +
          "<td class='r mono'>" + (r.channels == null ? "\u2014" : r.channels) + "</td>" +
          "<td class='r mono'>" + (r.cpus == null ? "\u2014" : r.cpus) + "</td>" +
          "<td class='r mono'>" + gib(m.vram_bytes) + "</td><td class='r mono'>" + gib(m.ram_bytes) + "</td>" +
          "<td class='r mono'>" + (r.files == null ? "\u2014" : r.files) + "</td>" +
          "<td class='r mono'>" + bytes(r.input_bytes) + "</td></tr>";
      }).join("") + "</tbody></table></div>";
    return head + '<div class="dbody">' + filters + kpis + section("By tool", byTool) +
      section("Runs \u00b7 click one for its timeline", rows.length ? table
        : '<div class="empty">No finished run matches these filters.</div>') + "</div>";
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

  // ---- a tool's data ---------------------------------------------------
  function norm(name) { return String(name || "").toLowerCase().replace(/[^a-z0-9]/g, ""); }
  function manifestFor(tool, folder) {
    var tools = (((upd || {}).status || {}).data || {}).tools || {};
    var keys = Object.keys(tools), i;
    for (i = 0; i < keys.length; i++) {
      var k = keys[i], provides = (tools[k].provides || []).map(norm);
      if (norm(k) === norm(folder) || norm(k) === norm(tool) || provides.indexOf(norm(tool)) >= 0) {
        return { key: k, info: tools[k] };
      }
    }
    return null;
  }
  function hostedList(entries) {
    if (!entries || !entries.length) { return '<div class="empty" style="text-align:left;padding:4px 0">Nothing.</div>'; }
    return '<table class="io"><tbody>' + entries.map(function (e) {
      return "<tr><td>" + (e.kind === "folder" ? "\ud83d\udcc1 " : "") + esc(e.name) + '</td><td class="mono" style="text-align:right">' +
        bytes(e.size) + "</td></tr>";
    }).join("") + "</tbody></table>";
  }
  function dataSection(name, data) {
    if (!data) { return section("Data", '<div class="empty" style="text-align:left">This tool is not served here.</div>'); }
    var hosted = '<div class="two"><div><div class="label">Models</div>' + hostedList((data.entries || {}).models) +
      '</div><div><div class="label">Test files</div>' + hostedList((data.entries || {}).testfiles) + "</div></div>";
    var scoped = data.scoped ? Object.keys(data.scoped).map(function (scope) {
      var sc = data.scoped[scope];
      return '<div style="margin-top:8px"><span class="label">testfiles/' + esc(scope) + "</span> " +
        '<span style="color:var(--soft);font-size:12.5px">' + esc((sc.testfiles || []).join(", ") || "empty") + "</span></div>";
    }).join("") : "";
    var m = manifestFor(name, data.folder), st = (upd || {}).status || {};
    var manifest = "";
    if (!upd || !upd.agent.alive) {
      manifest = '<div class="warnbox">The update agent is not running on the host, so the manifest cannot be compared ' +
        "and nothing can be downloaded from here. Start it on the host with <code>python3 scripts/server_ctl.py agent</code>.</div>";
    } else if (!m) {
      manifest = '<div class="empty" style="text-align:left">The tools library\'s manifest lists nothing for this tool.</div>';
    } else {
      var entries = m.info.entries || [], missing = entries.filter(function (e) { return !e.present; });
      var missingBytes = missing.reduce(function (a, e) { return a + (e.size || 0); }, 0);
      var running = st.applying && st.applying.kind === "data" && st.applying.tool === m.key;
      var busy = st.applying || upd.request;
      manifest = '<div style="font-size:12.5px;color:var(--soft);margin-bottom:6px">' + (entries.length - missing.length) + " of " +
        entries.length + " entries on disk" + (missing.length ? " \u00b7 " + bytes(missingBytes) + " missing" : "") +
        " \u00b7 manifest key <b>" + esc(m.key) + "</b></div>" +
        '<table class="io"><tbody>' + entries.map(function (e) {
          return "<tr><td>" + esc(e.kind) + "</td><td>" + esc(e.name) + '</td><td class="mono" style="text-align:right">' +
            bytes(e.size) + "</td><td>" + (e.present ? '<span class="pill done">on disk</span>' : '<span class="pill failed">missing</span>') +
            "</td></tr>";
        }).join("") + "</tbody></table>" +
        (running ? '<div class="warnbox" style="margin-top:8px"><b>Downloading</b> \u00b7 ' + esc(st.applying.phase || "") + "</div>"
          : '<div class="doorctl" style="margin-top:10px">' +
            '<button class="primary" data-fetch="' + esc(m.key) + '"' + (missing.length && !busy ? "" : " disabled") + ">Download missing</button>" +
            '<button data-fetch="' + esc(m.key) + '" data-force="1"' + (busy ? " disabled" : "") + ">Re-download everything</button>" +
            (busy && !running ? '<span class="s">the agent is busy with another request</span>' : "") + "</div>");
      if (st.last && st.last.kind === "data" && st.last.tool === m.key && now() - st.last.at < 3600) {
        manifest += '<div class="' + (st.last.ok ? "okbox" : "warnbox") + '" style="margin-top:8px">' + esc(st.last.message) +
          " \u00b7 " + ago(st.last.at) + " ago</div>";
      }
    }
    return section("Data on this server \u00b7 DATA/" + esc(data.folder || name), hosted + scoped) +
      section("From the tools library's manifest", manifest);
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
      { values: tr.map(function (p) { return [p.at, p.cpu]; }), color: "var(--series-1)" },
      { values: tr.map(function (p) { return [p.at, p.ram == null || !ramTotal ? null : pct(p.ram, ramTotal)]; }), color: "var(--series-2)" },
      { values: tr.map(function (p) { return [p.at, p.vram == null || !vramTotal ? null : pct(p.vram, vramTotal)]; }), color: "var(--series-3)" },
    ], tr[0].at, Math.max(tr[tr.length - 1].at, tr[0].at + 1), { height: 130, marks: marks })
      : '<div class="empty" style="text-align:left">The machine trace covers the last six hours of this process only.</div>';

    return head + '<div class="dbody">' + kpis + dataSection(name, data.data) + section("Where the time goes, on average", stack) +
      section("Recent runs, each from its own start", runsGantt) +
      '<div class="two">' +
      section("Duration of each run", '<div class="legend"><span><i style="background:var(--accent)"></i>took</span>' +
        '<span><i style="background:var(--warn)"></i>waited</span></div>' + durations) +
      section("The machine while it ran", '<div class="legend"><span><i style="background:var(--series-1)"></i>cpu</span>' +
        '<span><i style="background:var(--series-2)"></i>ram</span><span><i style="background:var(--series-3)"></i>vram</span>' +
        '<span>shaded: this tool running</span></div>' + machine) +
      "</div></div>";
  }

  // ---- operator controls ----------------------------------------------
  function toast(text) {
    var node = el("toast");
    node.textContent = text;
    node.hidden = false;
    window.clearTimeout(toast.timer);
    toast.timer = window.setTimeout(function () { node.hidden = true; }, 2600);
  }



  function adminAction(path, body, done) {
    fetch(path, { method: "POST", headers: { "X-Admin-Token": token,
                                             "Content-Type": "application/json" }, body: JSON.stringify(body) })
      .then(function (r) {
        return r.json().catch(function () { return {}; }).then(function (payload) {
          if (r.status === 401) { admin.ok = false; }
          if (!r.ok) { throw new Error(payload.detail || ("The server answered " + r.status + ".")); }
          return payload;
        });
      })
      .then(function () { toast(done); load(); })
      .catch(function (e) { toast(e.message); });
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
    drawMaint();
    loadUpdates(false);
    drawClients(d);
    drawTools(d);
    drawHistory(d);
    drawFilters();
    if (dlg.kind === "updates") {
      // Redrawn when the updates report arrives (loadUpdates).
    } else if (dlg.kind === "history") {
      // Fetched when opened and on Refresh only: a poll redrawing it would
      // wipe a filter being typed.
    } else if (dlg.kind) {
      // A live run's detail follows it; a tool view is heavier and refreshes slower.
      var every = dlg.kind === "run" ? EVERY : 10000;
      if (Date.now() - dlg.fetchedAt >= every - 200 && !(dlg.data && dlg.data.reaped)) { fetchDialog(); }
      else { drawDialog(); }
    }
  }

  function load() {
    if (!token) { el("gate").hidden = false; el("shell").hidden = true; return; }
    fetch("admin-panel.json", { headers: { "X-Admin-Token": token } })
      .then(function (r) {
        if (r.status === 401) { throw new Error("That is not this server's admin token."); }
        if (r.status === 403) { throw new Error("This server has no admin token: set ADMIN_TOKEN in its .env and restart it."); }
        if (!r.ok) { throw new Error("The server answered " + r.status + "."); }
        return r.json();
      })
      .then(function (d) {
        el("gate").hidden = true; el("shell").hidden = false;
        admin.ok = true;
        draw(d);
      })
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
  // The history window's own filters, applied in the browser to what it holds.
  document.addEventListener("change", function (event) {
    if (event.target.id === "doorhours") { doorHours = event.target.value; return; }
    var key = event.target.getAttribute && event.target.getAttribute("data-hf");
    if (key && dlg.kind === "history" && event.target.tagName === "SELECT") {
      dlg.hf[key] = event.target.value;
      drawDialog();
    }
  });
  var hfTimer = null;
  document.addEventListener("input", function (event) {
    if (event.target.getAttribute && event.target.getAttribute("data-hf") === "text" && dlg.kind === "history") {
      window.clearTimeout(hfTimer);
      var value = event.target.value;
      hfTimer = window.setTimeout(function () {
        dlg.hf.text = value;
        drawDialog();
        var box = document.querySelector('[data-hf="text"]');
        if (box) { box.focus(); box.setSelectionRange(value.length, value.length); }
      }, 250);
    }
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
    if (t.closest("[data-history]")) { openDialog("history", null); return; }
    if (t.closest("[data-updates]")) { openDialog("updates", null); return; }
    if (t.closest("[data-check]")) {
      if (checking) { return; }
      checking = true; drawMaint();
      fetch("admin/updates/check", { method: "POST", headers: { "X-Admin-Token": token } })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function () { checking = false; updFetchedAt = 0; loadUpdates(true); })
        .catch(function () { checking = false; drawMaint(); });
      return;
    }
    var dr = t.closest("[data-door]");
    if (dr) {
      var opening = dr.getAttribute("data-door") === "open";
      var hours = opening ? 0 : parseFloat((el("doorhours") || {}).value || "2");
      adminAction("admin/door", { accepting: opening, hours: hours },
        opening ? "Accepting new runs again." : "No new runs will be accepted; the ones already in will finish.");
      updFetchedAt = 0;
      return;
    }
    var up = t.closest("[data-update]");
    if (up) {
      adminAction("admin/update", { target: up.getAttribute("data-update") },
        "Update requested. New runs are stopped once the agent picks it up.");
      updFetchedAt = 0;
      window.setTimeout(function () { loadUpdates(true); }, 600);
      return;
    }
    var fe = t.closest("[data-fetch]");
    if (fe && !fe.disabled) {
      var force = fe.getAttribute("data-force") === "1";
      adminAction("admin/data", { tool: fe.getAttribute("data-fetch"), force: force },
        force ? "Re-download requested." : "Download requested.");
      updFetchedAt = 0;
      window.setTimeout(function () { loadUpdates(true); }, 800);
      return;
    }
    if (t.closest("[data-withdraw]")) {
      fetch("admin/update", { method: "DELETE", headers: { "X-Admin-Token": token } })
        .then(function () { toast("Update cancelled."); loadUpdates(true); });
      return;
    }
    if (t.closest("[data-hrefresh]")) { fetchDialog(); return; }
    if (t.closest("[data-back]") && dlg.back) {
      var back = dlg.back;
      openDialog("history", null);
      dlg.data = back.data; dlg.hf = back.hf; drawDialog();
      return;
    }
    var mv = t.closest("[data-move]");
    if (mv) {
      adminAction("admin/queue/" + encodeURIComponent(mv.getAttribute("data-id")) + "/move",
        { to: mv.getAttribute("data-move") }, "Moved " + mv.getAttribute("data-move") + ".");
      return;
    }
    var rl = t.closest("[data-rule]");
    if (rl) {
      var level2 = rl.getAttribute("data-rule");
      adminAction("admin/clients/" + encodeURIComponent(rl.getAttribute("data-addr")) + "/policy",
        { batches: level2 }, level2 === "parallel" ? "Batches of this workstation now run in parallel." : "Batches of this workstation now run one at a time.");
      return;
    }
    var pr = t.closest("[data-prio]");
    if (pr) {
      var level = pr.getAttribute("data-prio");
      adminAction("admin/runs/" + encodeURIComponent(pr.getAttribute("data-id")) + "/priority",
        { priority: level }, level === "high" ? "Priority given." : "Back to normal priority.");
      return;
    }
    if (t.closest("#adminbtn")) {
      token = ""; admin.ok = false;
      try { window.localStorage.removeItem(KEY); } catch (e) { /* private mode */ }
      closeDialog();
      el("gate").hidden = false; el("shell").hidden = true;
      return;
    }
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
    if (runNode) {
      var from = runNode.getAttribute("data-from") === "history" && dlg.kind === "history"
        ? { data: dlg.data, hf: dlg.hf } : null;
      openDialog("run", runNode.getAttribute("data-run"), from);
    }
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && dlg.kind) { closeDialog(); }
  });
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
