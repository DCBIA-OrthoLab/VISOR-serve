"""The page `GET /server-debug` serves: this machine drawn as what it is.

The status page is a set of readings. This is a **schematic** -- the same
numbers, but placed where the thing they describe actually sits on the path a
run takes through this server:

    waiting  ->  running  ->  the machine it is spending  ->  what it left

Drawn rather than tabulated because the question this page exists for is
positional. "A run is slow" has four possible answers and they live at four
points of that line: nobody has asked yet, it is waiting at the gate, it is
computing, or it is writing. A table of twenty numbers makes a reader work out
which; a layout puts the congestion where the eye already is.

**The page IS the diagram now.** It used to be one SVG in the middle of a
column, which meant the flow was drawn at the size of a figure while the screen
around it was empty -- and it could not hold a scrollable history or anything
you could click. So the flow became the geometry of the page itself: the queue
is the left rail, what is running is the centre, the machine it is spending is
the right rail, and what it left is the strip along the bottom. Left to right
is still the path; SVG is kept for the things that are genuinely drawings --
the tanks, the graphs, the progress of a run -- and dropped for the things that
are genuinely lists.

Five decisions, each of them a thing the page deliberately does NOT do:

**A tank shows held AND used, never one of them.** `admission` knows what it
handed out; `/proc` and the card know what is being consumed. They routinely
disagree -- a tool that reserved 8 GiB of VRAM and is still loading its weights
holds all of it and uses none -- so the reservation is the filled body of the
tank and the real consumption is a line drawn across it. The gap between the
two IS the reading.

**It says "requests", not "clients".** HTTP has no session. A client blocked
inside a four-hour POST is one in-flight request; a client whose laptop closed
without a FIN is none at all. The page counts what it can see and names it what
it is.

**The queue is ranked by TOTAL wait, not the worst one.** A tool that queues
four seconds a hundred times costs this deployment more than one that waited a
minute once, and sorting by the worst says the opposite. The worst is shown
beside it, because it is the one a clinician felt.

**A run's detail names its arguments and never their values.** Which knobs a
caller set is an operational fact; what they were set to is a path, and a path
is a patient's file name. The same rule that keeps progress messages off this
page keeps parameters off it -- so the detail carries argument NAMES, a file
COUNT and a byte total, and nothing that could identify whose data it was.

**It admits what it cannot see.** The queue log, the ledger and the graphs are
all bounded and in memory: the first two reset when the server restarts and two
uvicorn workers keep two of them, and the graphs live only as long as the tab,
because keeping a server-side history for a picture nobody may be watching is a
cost paid on every run for a page opened once a week. All three say so on
themselves. A reading that is unavailable -- no `/proc`, no card, a missing
directory -- is hatched and labelled, never drawn as an empty tank, because
empty reads as idle and idle is a claim.

Self-contained, like the status and benchmark pages, and for the same reason:
this server runs on networks with no internet, so a stylesheet or a font from a
CDN is a broken page rather than a degraded one. No library, no build step, and
no token in the served HTML -- the reader types one and the browser keeps it.
"""

from __future__ import annotations

DEBUG_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Server debug</title>
<style>
  :root {
    --bg: #f4f4f2; --panel: #ffffff; --sunk: #edece8; --line: #dedcd5;
    --ink: #17181a; --soft: #6f6b63; --ghost: #a9a59c;
    --bar: #e2dfd8; --fill: #3d6b8e; --warn: #b5722f; --ok: #4a7c59;
    --hot: #a4433a; --flow: #b9c8d4;
    --shadow: 0 1px 2px rgba(0,0,0,.05), 0 8px 24px rgba(0,0,0,.05);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #101113; --panel: #1a1c1f; --sunk: #141619; --line: #2b2e33;
      --ink: #eceae6; --soft: #918d86; --ghost: #5d6167;
      --bar: #24272b; --fill: #74aede; --warn: #dfa163; --ok: #7fb08c;
      --hot: #e08478; --flow: #33424f;
      --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.35);
    }
  }
  * { box-sizing: border-box; }
  /* The shell is a grid and the gate is a flex card, and an author `display`
     beats the user agent's `[hidden] { display: none }` -- so without this both
     are on screen at once and the token box sits across the dashboard. */
  [hidden] { display: none !important; }
  html, body { height: 100%; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .mono { font-variant-numeric: tabular-nums; }

  /* ---- shell: full bleed, three rails and a strip ------------------ */
  #shell { display: grid; height: 100vh; gap: 10px; padding: 10px;
           grid-template-columns: minmax(210px, 240px) minmax(0, 1fr) minmax(300px, 25vw);
           grid-template-rows: auto minmax(0, 1fr) minmax(150px, 28vh);
           grid-template-areas: "top top top" "queue runs machine" "past past past"; }
  @media (max-width: 1100px) {
    #shell { height: auto; grid-template-columns: 1fr;
             grid-template-rows: auto;
             grid-template-areas: "top" "machine" "queue" "runs" "past"; }
  }
  #top { grid-area: top; } #left { grid-area: queue; }
  #runs { grid-area: runs; } #machine { grid-area: machine; }
  #past { grid-area: past; }
  /* Two boxes, not two blocks in one: the queue is waiting for the MACHINE and
     a paused run is waiting for a PERSON. They look alike -- both stopped,
     both orange -- which is exactly why they must not share a frame, or a
     reader counts eight waiting when four of them are waiting on themselves.
     Paused sizes to its content and yields the rest to the queue. */
  #left { display: flex; flex-direction: column; gap: 10px; min-height: 0; }
  #queue { flex: 1 1 auto; min-height: 0; }
  #paused { flex: 0 1 auto; min-height: 92px; max-height: 46%; }

  .card { background: var(--panel); border: 1px solid var(--line);
          border-radius: 10px; display: flex; flex-direction: column;
          min-height: 0; overflow: hidden; }
  .card > h2 { margin: 0; padding: 10px 14px 8px; font-size: 10.5px;
               font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
               color: var(--soft); display: flex; align-items: baseline; gap: 8px;
               border-bottom: 1px solid var(--line); }
  .card > h2 .note { margin-left: auto; text-transform: none; letter-spacing: 0;
                     font-weight: 400; color: var(--ghost); font-size: 10.5px; }
  .body { padding: 10px 12px; overflow: auto; min-height: 0; flex: 1; }
  .body.tight { padding: 8px; }

  /* ---- top bar ----------------------------------------------------- */
  #top { display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
         padding: 2px 4px; }
  /* The title answers "which of the three servers am I looking at" before
     anything else on the page does: two of them are open in other tabs, they
     look identical, and reading the wrong one is the expensive mistake here. */
  #ident { min-width: 0; }
  h1 { font-size: 19px; font-weight: 700; letter-spacing: -.01em; margin: 0;
       line-height: 1.1; }
  #host { color: var(--soft); font-size: 11.5px; }
  /* Colour keyed on the origin, so the three deployments are told apart from
     across the room without a word being legible. */
  #flag { width: 7px; align-self: stretch; min-height: 34px; border-radius: 4px;
          background: var(--ghost); flex: none; }
  .grow { flex: 1; }
  .chip { display: flex; align-items: baseline; gap: 6px; padding: 4px 11px;
          background: var(--panel); border: 1px solid var(--line);
          border-radius: 999px; }
  .chip b { font-size: 17px; font-weight: 650; }
  .chip span { font-size: 10px; letter-spacing: .08em; text-transform: uppercase;
               color: var(--soft); }
  .chip.busy b { color: var(--fill); } .chip.wait b { color: var(--warn); }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--ok);
         box-shadow: 0 0 0 3px color-mix(in srgb, var(--ok) 22%, transparent); }
  .dot.off { background: var(--ghost); box-shadow: none; }
  button, input { font: inherit; color: inherit; }
  button { background: var(--panel); border: 1px solid var(--line);
           border-radius: 7px; padding: 5px 11px; cursor: pointer; }
  button:hover { border-color: var(--soft); }
  input { background: var(--panel); border: 1px solid var(--line);
          border-radius: 7px; padding: 7px 10px; min-width: 220px; }
  #find { min-width: 230px; padding: 5px 10px; font-size: 12.5px; }
  /* An active filter is a removable chip rather than a state you have to
     remember: a dashboard left on a wall filtered to one tool, with nothing
     saying so, is a dashboard that lies by omission. */
  #filters { display: flex; gap: 5px; flex-wrap: wrap; }
  .fchip { display: inline-flex; align-items: center; gap: 5px; font-size: 11px;
           padding: 3px 7px 3px 9px; border-radius: 999px;
           border: 1px solid var(--fill); color: var(--fill); cursor: pointer; }
  .fchip:hover { background: color-mix(in srgb, var(--fill) 12%, transparent); }
  .fchip b { font-weight: 600; }
  .fchip i { font-style: normal; opacity: .65; }
  .fbtn { font-size: 11px; padding: 3px 9px; border-radius: 999px; }

  /* ---- the queue, stacked ------------------------------------------ */
  .stack { display: flex; flex-direction: column-reverse; gap: 5px; }
  .slab { border: 1px solid var(--warn); border-left-width: 3px;
          border-radius: 6px; padding: 6px 9px; cursor: pointer;
          background: color-mix(in srgb, var(--warn) 9%, var(--panel)); }
  .slab:hover { border-color: var(--ink); }
  .slab .n { font-size: 12.5px; font-weight: 600; }
  .slab .m { font-size: 10.5px; color: var(--soft); display: flex; gap: 6px; }
  .next { font-size: 9.5px; letter-spacing: .09em; text-transform: uppercase;
          color: var(--warn); margin-bottom: 4px; }
  /* Paused shares the orange -- both are "stopped, waiting" -- and is told
     apart by the dashed edge and by sitting in its own band. */
  .slab.held { border-style: dashed; border-left-style: solid; }
  .foot { font-size: 10px; color: var(--ghost); margin-top: 5px; line-height: 1.35; }
  .rank { display: flex; align-items: center; gap: 7px; margin-top: 5px;
          font-size: 11px; }
  .rank .nm { width: 78px; overflow: hidden; text-overflow: ellipsis;
              white-space: nowrap; }
  .rank .tr { flex: 1; height: 6px; background: var(--bar); border-radius: 3px; }
  .rank .tr i { display: block; height: 100%; border-radius: 3px;
                background: var(--warn); }
  .rank .tr i.top { background: var(--hot); }
  .rank .vv { color: var(--soft); min-width: 46px; text-align: right; }

  /* ---- running ------------------------------------------------------ */
  .runs { display: grid; gap: 8px;
          grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); }
  .run { position: relative; border: 1px solid var(--line); border-radius: 8px;
         padding: 9px 11px 10px; background: var(--sunk); cursor: pointer;
         overflow: hidden; }
  .run:hover { border-color: var(--soft); }
  .run .prog { position: absolute; inset: 0 auto 0 0; background: var(--fill);
               opacity: .13; transition: width .4s ease; }
  .run > * { position: relative; }
  .run .hd { display: flex; align-items: baseline; gap: 7px; }
  .run .nm { font-weight: 650; font-size: 14px; }
  .run .pc { margin-left: auto; font-size: 15px; font-weight: 650; }
  .run .sub { font-size: 10.5px; color: var(--soft); margin-top: 1px; }
  /* A 4px spine down the left of every card: the state is readable from the
     far side of the room without any text being legible at all. */
  .run { border-left: 4px solid var(--ok); }
  .run .prog { background: var(--ok); }
  .run .pc { color: var(--ok); }
  .run.q { background: color-mix(in srgb, var(--warn) 8%, var(--sunk));
           border-color: var(--warn); border-left-color: var(--warn); }
  .run.q .prog, .run.p .prog { background: var(--warn); }
  .run.q .pc, .run.p .pc { color: var(--warn); }
  .run.p { border-color: var(--warn); border-left-color: var(--warn);
           border-style: dashed; border-left-style: solid;
           background: color-mix(in srgb, var(--warn) 5%, var(--sunk)); }
  .cells { display: flex; gap: 5px; margin-top: 8px; flex-wrap: wrap; }
  .cell { flex: 1 1 52px; background: var(--panel); border: 1px solid var(--line);
          border-radius: 5px; padding: 3px 6px; }
  .cell u { display: block; font-size: 9px; letter-spacing: .07em; color: var(--ghost);
            text-transform: uppercase; text-decoration: none; }
  .cell b { font-size: 12.5px; font-weight: 600; }
  .depth { display: inline-block; font-size: 9.5px; color: var(--soft);
           border: 1px solid var(--line); border-radius: 4px; padding: 0 4px; }

  /* ---- machine ------------------------------------------------------ */
  .tank { margin-bottom: 11px; }
  .tank .lab { display: flex; align-items: baseline; gap: 6px; font-size: 10.5px;
               letter-spacing: .08em; text-transform: uppercase; color: var(--soft); }
  .tank .lab b { margin-left: auto; font-size: 15px; font-weight: 650;
                 letter-spacing: 0; text-transform: none; color: var(--ink); }
  .tank .tr { position: relative; height: 13px; background: var(--bar);
              border-radius: 4px; overflow: hidden; margin-top: 3px; }
  .tank .tr i { position: absolute; inset: 0 auto 0 0; background: var(--fill);
                border-radius: 4px; transition: width .4s ease; }
  .tank .tr i.hot { background: var(--hot); }
  .tank .tr u { position: absolute; top: -2px; bottom: -2px; width: 2px;
                background: var(--ink); transition: left .4s ease; }
  .tank .tr.na { background: repeating-linear-gradient(45deg,
                 var(--bar) 0 5px, var(--panel) 5px 10px); }
  .tank .ft { font-size: 10.5px; color: var(--soft); margin-top: 2px; }
  .graphs { margin-top: 4px; }
  .gr { margin-bottom: 7px; }
  .gr .hd { display: flex; font-size: 10px; letter-spacing: .07em; color: var(--soft);
            text-transform: uppercase; }
  .gr .hd b { margin-left: auto; letter-spacing: 0; text-transform: none;
              color: var(--ink); font-size: 11.5px; }
  svg.plot { display: block; width: 100%; height: 40px; }

  /* ---- uptime -------------------------------------------------------- */
  .up { display: flex; align-items: center; gap: 7px; font-size: 11.5px;
        padding: 2px 0; }
  .up .nm { width: 96px; overflow: hidden; text-overflow: ellipsis;
            white-space: nowrap; }
  .up .tr { flex: 1; height: 7px; background: var(--bar); border-radius: 4px; }
  .up .tr i { display: block; height: 100%; border-radius: 4px; background: var(--fill); }
  .up .tr i.live { background: var(--ok); }
  .up .vv { color: var(--soft); min-width: 74px; text-align: right; }

  /* ---- history strip -------------------------------------------------- */
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  thead th { position: sticky; top: 0; background: var(--panel); z-index: 1;
             text-align: left; font-weight: 500; color: var(--soft);
             font-size: 10px; letter-spacing: .07em; text-transform: uppercase;
             padding: 4px 9px 5px; border-bottom: 1px solid var(--line); }
  tbody td { padding: 4px 9px; border-bottom: 1px solid var(--line); }
  tbody tr { cursor: pointer; }
  tbody tr:hover td { background: var(--sunk); }
  td.r, th.r { text-align: right; }
  .pill { font-size: 10px; padding: 1px 7px; border-radius: 999px;
          border: 1px solid var(--line); color: var(--soft); white-space: nowrap; }
  /* One traffic light for the whole page: green is computing or finished
     well, orange is stopped and waiting on something, red is broken, grey is
     nobody's fault. Queued and paused share the orange deliberately -- they
     are both "stopped, waiting" -- and are told apart by where they sit and by
     the dashed border, which is the same positional reasoning the layout is. */
  .pill.running { color: var(--ok); border-color: var(--ok); }
  .pill.done { color: var(--ok); border-color: var(--ok); }
  .pill.packaging { color: var(--ok); border-color: var(--ok); }
  .pill.queued_gpu, .pill.wait { color: var(--warn); border-color: var(--warn); }
  .pill.paused { color: var(--warn); border-color: var(--warn); border-style: dashed; }
  .pill.failed { color: var(--hot); border-color: var(--hot); }
  .pill.cancelled { color: var(--soft); border-style: dashed; }

  /* ---- detail drawer --------------------------------------------------- */
  #drawer { position: fixed; top: 0; right: 0; bottom: 0; width: min(420px, 94vw);
            background: var(--panel); border-left: 1px solid var(--line);
            box-shadow: var(--shadow); padding: 16px 18px; overflow: auto;
            transform: translateX(101%); transition: transform .22s ease; z-index: 9; }
  #drawer.open { transform: none; }
  #drawer h3 { margin: 0 0 2px; font-size: 17px; }
  #drawer .rid { font-size: 10.5px; color: var(--ghost); word-break: break-all; }
  #drawer dl { display: grid; grid-template-columns: 116px 1fr; gap: 3px 10px;
               margin: 14px 0 0; font-size: 12.5px; }
  #drawer dt { color: var(--soft); }
  #drawer dd { margin: 0; }
  #drawer .args { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
  #drawer .caveat { margin-top: 16px; font-size: 11px; color: var(--ghost);
                    border-top: 1px solid var(--line); padding-top: 10px; }
  .dhead { font-size: 10px; letter-spacing: .09em; text-transform: uppercase;
           color: var(--ghost); margin-bottom: 3px; }
  table.io { width: 100%; border-collapse: collapse; font-size: 12px; }
  table.io td { padding: 3px 8px 3px 0; border-top: 1px solid var(--line);
                vertical-align: top; }
  table.io td:first-child { color: var(--soft); white-space: nowrap; }
  table.io td.r { text-align: right; white-space: nowrap; }
  table.io td.na { color: var(--ghost); }
  #close { position: absolute; top: 12px; right: 14px; }

  /* ---- the run console ---------------------------------------------- */
  .con { margin-top: 8px; background: var(--sunk); border: 1px solid var(--line);
         border-radius: 7px; padding: 7px 9px; max-height: 32vh; overflow: auto;
         font: 11.5px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .con .ln { display: flex; gap: 8px; }
  .con .ts { color: var(--ghost); flex: none; }
  .con .tx { white-space: pre-wrap; word-break: break-word; }
  .con .info .tx { color: var(--ink); }
  .con .ok .tx { color: var(--ok); font-weight: 600; }
  .con .warn .tx { color: var(--warn); }
  .con .error .tx { color: var(--hot); font-weight: 600; }

  .empty { color: var(--ghost); font-size: 12px; padding: 10px 2px; }
  #gate { max-width: 560px; margin: 12vh auto; }
  .err { color: var(--hot); font-size: 12.5px; margin-top: 8px; }
</style>
</head>
<body>

<div id="gate" class="card" hidden>
  <h2>Server debug</h2>
  <div class="body">
    <p style="margin:0 0 10px;font-size:12.5px;color:var(--soft)">
      The readings come from <code>/server-debug.json</code>, which is
      Bearer-protected like every endpoint that says anything about a run.
      The token stays in this browser and is never written into this page.</p>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <input id="token" type="password" placeholder="API token" autocomplete="off">
      <button id="save" type="button">connect</button>
    </div>
    <div class="err" id="gateerr"></div>
  </div>
</div>

<div id="shell" hidden>
  <div id="top">
    <span id="flag" title="derived from this page's own origin"></span>
    <div id="ident">
      <h1 id="origin">&nbsp;</h1>
      <div id="host" class="mono">connecting&hellip;</div>
    </div>
    <span class="grow"></span>
    <input id="find" type="search" placeholder="run id, tool or address"
           autocomplete="off" spellcheck="false">
    <span id="filters"></span>
    <span class="chip" id="c-req" title="work asked of this server; this page's own polling is not counted"><b>0</b><span>requests</span></span>
    <span class="chip" id="c-run"><b>0</b><span>running</span></span>
    <span class="chip" id="c-wait"><b>0</b><span>waiting</span></span>
    <span class="dot" id="dot"></span>
    <button id="toggle" type="button">pause</button>
  </div>

  <div id="left">
    <section class="card" id="queue">
      <h2>Queue<span class="note" id="q-note">waiting for the machine</span></h2>
      <div class="body tight">
        <div id="q-stack"></div>
        <div style="margin-top:14px">
          <div class="note" style="font-size:10px;letter-spacing:.09em;
               text-transform:uppercase;color:var(--ghost)">Time spent waiting</div>
          <div id="q-rank"></div>
        </div>
      </div>
    </section>

    <section class="card" id="paused">
      <h2>Paused<span class="note">waiting for a person</span></h2>
      <div class="body tight"><div id="q-paused"></div></div>
    </section>
  </div>

  <section class="card" id="runs">
    <h2>Running<span class="note">click a run for its detail</span></h2>
    <div class="body"><div class="runs" id="r-list"></div></div>
  </section>

  <section class="card" id="machine">
    <h2>Machine<span class="note" id="m-note"></span></h2>
    <div class="body">
      <div id="m-tanks"></div>
      <div class="graphs" id="m-graphs"></div>
      <div id="m-disk" style="margin-top:6px"></div>
      <div style="margin-top:12px">
        <div class="note" style="font-size:10px;letter-spacing:.09em;
             text-transform:uppercase;color:var(--ghost)">Uptime per tool</div>
        <div id="m-up"></div>
      </div>
    </div>
  </section>

  <section class="card" id="past">
    <h2>History<span class="note" id="h-note"></span></h2>
    <div class="body tight" id="h-body"></div>
  </section>
</div>

<aside id="drawer" aria-live="polite">
  <button id="close" type="button">close</button>
  <div id="d-body"></div>
</aside>

<script>
(function () {
  "use strict";
  var KEY = "visor.token";
  var EVERY = 2000;
  var SPAN = 180;          // samples kept per graph: six minutes at 2s
  var token = "";
  try { token = window.localStorage.getItem(KEY) || ""; } catch (e) { token = ""; }

  var live = true;
  var selected = null;
  var latest = null;
  var console_ = { run: null, lines: [], error: "" };
  // Every list on the page reads these, so a filter is one concept rather than
  // four. Mirrored into the URL so a filtered view can be bookmarked, sent to
  // somebody, or left on a wall and restored after a reload.
  var filters = { tool: "", client: "", outcome: "", text: "" };
  var series = { cpu: [], ram: [], vram: [], disk: [] };

  function el(id) { return document.getElementById(id); }

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
      .map(function (k) {
        return encodeURIComponent(k) + "=" + encodeURIComponent(filters[k]);
      });
    var next = parts.length ? "#" + parts.join("&") : "#";
    if (window.location.hash !== next) {
      // replaceState, not assignment: a filter is a view, not a navigation, and
      // twenty of them should not be twenty presses of the back button.
      window.history.replaceState(null, "", next);
    }
  }

  function setFilter(key, value) {
    filters[key] = (filters[key] === value) ? "" : value;
    writeUrl();
    if (latest) { draw(latest); }
  }

  function matches(run, led) {
    led = led || {};
    var tool = run.tool || led.tool || "";
    var client = run.client || led.client || "";
    if (filters.tool && tool !== filters.tool) { return false; }
    if (filters.client && client !== filters.client) { return false; }
    if (filters.outcome) {
      var state = run.phase || led.outcome || "";
      if (state !== filters.outcome) { return false; }
    }
    if (filters.text) {
      var hay = [run.run_id || led.run_id, tool, client].join(" ").toLowerCase();
      if (hay.indexOf(filters.text.toLowerCase()) === -1) { return false; }
    }
    return true;
  }

  function anyFilter() {
    return Object.keys(filters).some(function (k) { return filters[k]; });
  }

  function drawFilters() {
    var labels = { tool: "tool", client: "from", outcome: "state", text: "text" };
    el("filters").innerHTML = Object.keys(filters)
      .filter(function (k) { return filters[k]; })
      .map(function (k) {
        return '<span class="fchip" data-clear="' + k + '"><i>' + labels[k] +
          "</i><b>" + esc(filters[k]) + "</b>&times;</span>";
      }).join("");
    if (el("find").value !== filters.text) { el("find").value = filters.text; }
  }
  function show(node, on) { node.hidden = !on; }
  function esc(text) {
    var box = document.createElement("div");
    box.textContent = String(text == null ? "" : text);
    return box.innerHTML;
  }
  function gib(n) { return (n == null) ? "—" : (n / 1073741824).toFixed(1) + "G"; }
  function mib(n) { return (n == null) ? "—" : (n / 1048576).toFixed(0) + "M"; }
  function pct(a, b) { return b ? Math.max(0, Math.min(100, (a / b) * 100)) : 0; }
  function dur(s) {
    if (s == null) { return "—"; }
    s = Math.max(0, Math.round(s));
    if (s < 60) { return s + "s"; }
    if (s < 3600) { return Math.floor(s / 60) + "m" + ("0" + (s % 60)).slice(-2); }
    return Math.floor(s / 3600) + "h" + ("0" + Math.floor((s % 3600) / 60)).slice(-2);
  }
  function ago(at) { return at ? dur(Date.now() / 1000 - at) : "—"; }

  // ---- graphs ---------------------------------------------------------
  function push(key, value) {
    var buf = series[key];
    // A missing reading is kept as a gap, not dropped: the line breaks where
    // the machine stopped answering, which is itself the reading.
    buf.push(value == null ? null : value);
    while (buf.length > SPAN) { buf.shift(); }
  }

  function plot(key, label, current, unit) {
    var buf = series[key], w = 300, h = 40, i;
    var step = w / (SPAN - 1), runs = [], cur = [];
    for (i = 0; i < buf.length; i++) {
      if (buf[i] == null) { if (cur.length) { runs.push(cur); cur = []; } continue; }
      var x = (i + (SPAN - buf.length)) * step;
      var y = h - 1.5 - (Math.max(0, Math.min(100, buf[i])) / 100) * (h - 3);
      cur.push(x.toFixed(1) + "," + y.toFixed(1));
    }
    if (cur.length) { runs.push(cur); }
    var paint = runs.filter(function (r) { return r.length > 1; }).map(function (r) {
      var area = "M" + r[0].split(",")[0] + "," + h + " L" + r.join(" L") +
        " L" + r[r.length - 1].split(",")[0] + "," + h + " Z";
      return '<path d="' + area + '" fill="var(--fill)" opacity=".13"/>' +
        '<polyline points="' + r.join(" ") + '" fill="none" stroke="var(--fill)" ' +
        'stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>';
    }).join("");
    return '<div class="gr"><div class="hd">' + esc(label) +
      "<b>" + (current == null ? "—" : esc(current) + (unit || "")) + "</b></div>" +
      '<svg class="plot" viewBox="0 0 ' + w + " " + h +
      '" preserveAspectRatio="none" aria-hidden="true">' + paint + "</svg></div>";
  }

  // ---- machine --------------------------------------------------------
  function tank(label, heldPct, usedPct, big, foot) {
    if (heldPct == null && usedPct == null) {
      return '<div class="tank"><div class="lab">' + esc(label) +
        "<b>—</b></div>" + '<div class="tr na"></div>' +
        '<div class="ft">unavailable</div></div>';
    }
    var body = (heldPct == null ? usedPct : heldPct);
    return '<div class="tank"><div class="lab">' + esc(label) + "<b>" + esc(big) +
      "</b></div>" +
      '<div class="tr"><i class="' + (body > 90 ? "hot" : "") + '" style="width:' +
      body.toFixed(1) + '%"></i>' +
      (usedPct != null && heldPct != null
        ? '<u style="left:' + usedPct.toFixed(1) + '%"></u>' : "") +
      '</div><div class="ft">' + foot + "</div></div>";
  }

  function drawMachine(d) {
    var a = d.admission || {}, b = d.budget || {}, card = d.card || {};
    var ram = d.ram, cpu = d.cpu_percent;
    var store = (d.disk || {}).temp || {};
    var vramUsed = card.total_bytes == null ? null : card.total_bytes - card.free_bytes;
    var diskPct = store.total ? pct(store.total - store.free, store.total) : null;

    push("cpu", cpu);
    push("ram", ram ? pct(ram.used, ram.total) : null);
    push("vram", card.total_bytes ? pct(vramUsed, card.total_bytes) : null);
    push("disk", diskPct);

    el("m-tanks").innerHTML =
      tank("cpu", b.cpus ? pct(a.cpus_held, b.cpus) : null, cpu,
           (cpu == null ? "\u2014" : cpu.toFixed(0) + "%"),
           (a.cpus_held == null ? "?" : a.cpus_held.toFixed(1)) + " held of " +
           (b.cpus == null ? "?" : b.cpus.toFixed(0))) +
      tank("ram", b.ram_bytes ? pct(a.ram_held, b.ram_bytes) : null,
           ram ? pct(ram.used, ram.total) : null,
           ram ? gib(ram.used) : "—",
           gib(a.ram_held) + " held · " +
           (ram ? gib(ram.available) + " available of " + gib(ram.total)
                : "no /proc/meminfo")) +
      tank("vram", b.vram_bytes ? pct(a.vram_held, b.vram_bytes) : null,
           card.total_bytes ? pct(vramUsed, card.total_bytes) : null,
           card.total_bytes ? gib(vramUsed) : "—",
           gib(a.vram_held) + " held · " +
           (card.total_bytes ? gib(card.free_bytes) + " free of " + gib(card.total_bytes)
                             : "no card"));

    el("m-graphs").innerHTML =
      plot("cpu", "cpu", cpu == null ? null : cpu.toFixed(0), "%") +
      plot("ram", "ram", ram ? pct(ram.used, ram.total).toFixed(0) : null, "%") +
      plot("vram", "vram", card.total_bytes
        ? pct(vramUsed, card.total_bytes).toFixed(0) : null, "%") +
      plot("disk", "scratch disk", diskPct == null ? null : diskPct.toFixed(0), "%");

    el("m-note").textContent = "graphs live in this tab only";

    var names = Object.keys(d.disk || {});
    el("m-disk").innerHTML = names.map(function (name) {
      var e = d.disk[name];
      return '<div class="up"><span class="nm">' + esc(name) + "</span>" +
        '<span class="tr"><i style="width:' +
        ((e.total && e.free != null) ? pct(e.total - e.free, e.total) : 0).toFixed(1) +
        '%"></i></span><span class="vv mono">' +
        (e.used_here == null ? "—" : gib(e.used_here)) + " / " + gib(e.free) +
        " free</span></div>";
    }).join("");

    var up = d.uptime || [];
    var busiest = up.length ? (up[0].seconds || 1) : 1;
    el("m-up").innerHTML = up.length ? up.slice(0, 12).map(function (r) {
      return '<div class="up"><span class="nm">' + esc(r.tool) + "</span>" +
        '<span class="tr"><i class="' + (r.running ? "live" : "") + '" style="width:' +
        pct(r.seconds, busiest).toFixed(1) + '%"></i></span>' +
        '<span class="vv mono">' + dur(r.seconds) + " · " + r.runs + "r</span></div>";
    }).join("") : '<div class="empty">No run seen yet.</div>';
  }

  // ---- the queue, stacked --------------------------------------------
  function drawQueue(d) {
    var q = d.queue || {};
    var ledByRun = {};
    (d.ledger || []).forEach(function (r) { ledByRun[r.run_id] = r; });
    var waiting = (d.runs || []).filter(function (r) {
      return r.state === "running" && r.phase === "queued_gpu" &&
        matches(r, ledByRun[r.run_id]);
    });
    el("q-note").textContent = "of " + (q.capacity || 0) + " kept, " +
      (q.held || 0) + " held";

    // Column-reverse in CSS: the oldest waiter sits at the BOTTOM, nearest the
    // gate it is about to go through, and a new arrival lands on top of the
    // pile. A stack that grew downwards would put the next run to be let in
    // furthest from where the eye reads the queue draining.
    el("q-stack").innerHTML = waiting.length
      ? '<div class="next">next out &darr;</div>' + waiting.map(function (r) {
          return '<div class="slab" data-run="' + esc(r.run_id) + '">' +
            '<div class="n">' + esc(r.tool || "?") + "</div>" +
            '<div class="m mono"><span>waiting ' + ago(r.started_at) + "</span></div></div>";
        }).join("")
      : (anyFilter() ? '<div class="empty">Nothing waiting matches this filter.</div>' : '<div class="empty">Nothing waiting.</div>');

    var held = (d.runs || []).filter(function (r) {
      return r.phase === "paused" && matches(r, ledByRun[r.run_id]);
    });
    var ttl = ((d.server || {}).paused_ttl_seconds) || 0;
    el("q-paused").innerHTML = held.length ? held.map(function (r) {
      var age = Date.now() / 1000 - r.started_at;
      var left = ttl ? Math.max(0, ttl - age) : null;
      return '<div class="slab held" data-run="' + esc(r.run_id) + '">' +
        '<div class="n">' + esc(r.tool || "?") + "</div>" +
        '<div class="m mono"><span>held ' + ago(r.started_at) + "</span>" +
        (left === null ? "" : "<span>· dropped in " + dur(left) + "</span>") +
        "</div></div>";
    }).join("") + '<div class="foot">Each is holding its own work so it can be ' +
      "resumed. That is disk, until somebody comes back or the idle timeout " +
      "lets it go.</div>"
      : '<div class="empty">Nothing paused.</div>';

    var ranked = (q.ranked || []).slice(0, 8);
    var worst = ranked.length ? (ranked[0].total_wait || 1) : 1;
    el("q-rank").innerHTML = ranked.length ? ranked.map(function (r, i) {
      return '<div class="rank"><span class="nm">' + esc(r.tool) + "</span>" +
        '<span class="tr"><i class="' + (i === 0 ? "top" : "") + '" style="width:' +
        pct(r.total_wait, worst).toFixed(1) + '%"></i></span>' +
        '<span class="vv mono">' + r.total_wait.toFixed(1) + "s</span></div>";
    }).join("") : '<div class="empty">Nothing has queued.</div>';
  }

  // ---- what is running -------------------------------------------------
  function drawRuns(d) {
    var byId = {};
    (d.ledger || []).forEach(function (r) { byId[r.run_id] = r; });
    // Paused is NOT running. A paused run is stopped and waiting for a
    // PERSON, which is off the path this page is laid out along -- it has its
    // own band in the left rail, beside the queue that is waiting for the
    // machine. Showing it here made three ASO look like work in progress.
    var rows = (d.runs || []).filter(function (r) {
      return r.state === "running" && r.phase !== "paused" &&
        matches(r, byId[r.run_id]);
    });
    if (!rows.length) {
      el("r-list").innerHTML = (anyFilter() ? '<div class="empty">Nothing running matches this filter.</div>' : '<div class="empty">Nothing running.</div>');
      return;
    }
    el("r-list").innerHTML = rows.map(function (r) {
      var led = byId[r.run_id] || {};
      var queued = r.phase === "queued_gpu", paused = r.phase === "paused";
      var frac = r.fraction == null ? null : Math.max(0, Math.min(1, r.fraction));
      return '<div class="run' + (queued ? " q" : "") + (paused ? " p" : "") +
        '" data-run="' + esc(r.run_id) + '">' +
        (frac == null ? "" : '<div class="prog" style="width:' +
          (frac * 100).toFixed(1) + '%"></div>') +
        '<div class="hd"><span class="nm">' + esc(r.tool || "?") + "</span>" +
        (r.depth ? '<span class="depth">depth ' + esc(r.depth) + "</span>" : "") +
        '<span class="pc mono">' + (frac == null ? "" : (frac * 100).toFixed(0) + "%") +
        "</span></div>" +
        '<div class="sub mono"><span class="pill ' + esc(r.phase) + '">' +
        esc(r.phase) + "</span> · " + ago(r.started_at) + " in" +
        (led.waited ? " · waited " + led.waited.toFixed(1) + "s" : "") +
        ((r.client || led.client) ? " · " + esc(r.client || led.client) : "") + "</div>" +
        '<div class="cells">' +
        '<div class="cell"><u>chan</u><b>' + (led.channels == null ? "—" : led.channels) + "</b></div>" +
        '<div class="cell"><u>cpu</u><b>' + (led.cpus == null ? "—" : led.cpus) + "</b></div>" +
        '<div class="cell"><u>ram</u><b>' + gib(led.ram_bytes) + "</b></div>" +
        '<div class="cell"><u>vram</u><b>' + gib(led.vram_bytes) + "</b></div>" +
        '<div class="cell"><u>files</u><b>' + (led.files == null ? "—" : led.files) + "</b></div>" +
        "</div></div>";
    }).join("");
  }

  // ---- history ----------------------------------------------------------
  function drawHistory(d) {
    var inFlight = {};
    (d.runs || []).forEach(function (r) {
      if (r.state === "running") { inFlight[r.run_id] = true; }
    });
    var rows = (d.ledger || []).filter(function (r) {
      return !inFlight[r.run_id] && matches(r, r);
    });
    el("h-note").textContent = rows.length + " kept in this worker's memory, " +
      "cleared on restart";
    if (!rows.length) {
      el("h-body").innerHTML = (anyFilter() ? '<div class="empty">Nothing finished matches this filter.</div>' : '<div class="empty">Nothing has finished yet.</div>');
      return;
    }
    el("h-body").innerHTML = "<table><thead><tr>" +
      "<th>tool</th><th>from</th><th>outcome</th><th class='r'>took</th><th class='r'>waited</th>" +
      "<th class='r'>chan</th><th class='r'>cpu</th><th class='r'>ram</th>" +
      "<th class='r'>vram</th><th class='r'>files</th><th class='r'>in</th>" +
      "<th class='r'>ended</th></tr></thead><tbody>" +
      rows.map(function (r) {
        return '<tr data-run="' + esc(r.run_id) + '"><td>' + esc(r.tool || "?") +
          '</td><td class="mono">' + esc(r.client || "—") +
          '</td><td><span class="pill ' + esc(r.outcome || "") + '">' +
          esc(r.outcome || "in flight") + "</span></td>" +
          "<td class='r mono'>" + dur(r.seconds) + "</td>" +
          "<td class='r mono'>" + (r.waited == null ? "—" : r.waited.toFixed(1) + "s") + "</td>" +
          "<td class='r mono'>" + (r.channels == null ? "—" : r.channels) + "</td>" +
          "<td class='r mono'>" + (r.cpus == null ? "—" : r.cpus) + "</td>" +
          "<td class='r mono'>" + gib(r.ram_bytes) + "</td>" +
          "<td class='r mono'>" + gib(r.vram_bytes) + "</td>" +
          "<td class='r mono'>" + (r.files == null ? "—" : r.files) + "</td>" +
          "<td class='r mono'>" + mib(r.input_bytes) + "</td>" +
          "<td class='r mono'>" + ago(r.ended_at) + "</td></tr>";
      }).join("") + "</tbody></table>";
  }

  // ---- the detail drawer -------------------------------------------------
  function drawDrawer() {
    if (!selected || !latest) { return; }
    var led = (latest.ledger || []).filter(function (r) {
      return r.run_id === selected; })[0] || {};
    var live = (latest.runs || []).filter(function (r) {
      return r.run_id === selected; })[0] || {};
    var tool = led.tool || live.tool || "?";
    var client = live.client || led.client || "";
    var state = live.phase || led.outcome || "unknown";

    function row(k, v) { return "<dt>" + esc(k) + "</dt><dd class='mono'>" + v + "</dd>"; }
    el("d-body").innerHTML =
      "<h3>" + esc(tool) + "</h3>" +
      '<div class="rid">' + esc(selected) + "</div>" +
      '<div style="margin-top:9px;display:flex;gap:6px;flex-wrap:wrap;align-items:center">' +
      '<span class="pill ' + esc(state) + '">' + esc(state) + "</span>" +
      '<button class="fbtn" data-filter="tool" data-value="' + esc(tool) +
      '">filter this tool</button>' +
      (client ? '<button class="fbtn" data-filter="client" data-value="' + esc(client) +
        '">filter this address</button>' : "") +
      '<button class="fbtn" data-copy="' + esc(selected) + '">copy id</button>' +
      "</div>" +
      "<dl>" +
      row("progress", live.fraction == null ? "—" : (live.fraction * 100).toFixed(0) + "%") +
      row("depth", live.depth == null ? (led.run_id ? "0" : "—") : live.depth) +
      row("started", ago(led.started_at || live.started_at) + " ago") +
      row("took", dur(led.seconds != null ? led.seconds
            : (live.started_at ? Date.now() / 1000 - live.started_at : null))) +
      row("asked from", client ? esc(client) : "—") +
      row("waited at gate", led.waited == null ? "—" : led.waited.toFixed(2) + "s") +
      row("channels", led.channels == null ? "—" : led.channels) +
      row("cpus granted", led.cpus == null ? "—" : led.cpus) +
      row("ram reserved", gib(led.ram_bytes)) +
      row("vram reserved", gib(led.vram_bytes)) +
      row("input files", led.files == null ? "—" : led.files) +
      row("input bytes", led.input_bytes == null ? "—" : mib(led.input_bytes)) +
      "</dl>" +
      inputsHtml(led) + settingsHtml(led) +
      '<div style="margin-top:14px"><div style="font-size:10px;letter-spacing:.09em;' +
      'text-transform:uppercase;color:var(--ghost)">Console</div>' +
      consoleHtml() + "</div>" +
      '<div class="caveat">Argument VALUES are deliberately not held anywhere on ' +
      "an uploaded input is never NAMED here \u2014 its name is the patient's. " +
      "What is shown is its shape: how many files, how many bytes, which " +
      "extensions, which is what makes two runs comparable without identifying " +
      "either. A hosted bundle IS named, because this deployment staged it and " +
      "already lists it. Console lines are composed by the server from the phase " +
      "the run reported, never quoted from what the tool printed.</div>";
    el("drawer").classList.add("open");
  }

  function inputsHtml(led) {
    var rows = led.inputs || [];
    if (!rows.length) {
      return '<div style="margin-top:14px"><div class="dhead">Inputs</div>' +
        '<div class="empty">not recorded</div></div>';
    }
    return '<div style="margin-top:14px"><div class="dhead">Inputs</div>' +
      '<table class="io"><tbody>' + rows.map(function (r) {
        var what = r.hosted
          ? '<b>' + esc(r.hosted) + "</b>"
          : (r.files === 1 ? "1 file" : r.files + " files") +
            (r.partial ? "+" : "");
        return "<tr><td>" + esc(r.argument) + "</td><td>" + what + "</td>" +
          '<td class="r mono">' +
          (r.files != null && r.hosted ? r.files + "f " : "") +
          (r.bytes == null ? "\u2014" : bytes(r.bytes)) + "</td>" +
          '<td class="mono na">' + esc((r.extensions || []).join(" ")) +
          "</td></tr>";
      }).join("") + "</tbody></table></div>";
  }

  function settingsHtml(led) {
    var set = led.settings || {};
    var names = Object.keys(set);
    if (!names.length) {
      // Named but not valued, for a run recorded before this existed.
      var older = led.arguments || [];
      if (!older.length) { return ""; }
      return '<div style="margin-top:14px"><div class="dhead">Arguments named</div>' +
        '<div class="args">' + older.map(function (n) {
          return '<span class="pill">' + esc(n) + "</span>"; }).join("") + "</div></div>";
    }
    return '<div style="margin-top:14px"><div class="dhead">Settings</div>' +
      '<table class="io"><tbody>' + names.sort().map(function (name) {
        var value = set[name];
        var shown = Array.isArray(value)
          ? (value.length ? value.join(", ") : "\u2014")
          : String(value);
        return "<tr><td>" + esc(name) + '</td><td colspan="3" class="mono">' +
          esc(shown) + "</td></tr>";
      }).join("") + "</tbody></table></div>";
  }

  function bytes(n) {
    if (n == null) { return "\u2014"; }
    if (n >= 1073741824) { return (n / 1073741824).toFixed(1) + " GiB"; }
    if (n >= 1048576) { return (n / 1048576).toFixed(1) + " MiB"; }
    if (n >= 1024) { return (n / 1024).toFixed(0) + " KiB"; }
    return n + " B";
  }

  function consoleHtml() {
    if (console_.error) {
      return '<div class="con"><div class="ln error"><span class="tx">' +
        esc(console_.error) + "</span></div></div>";
    }
    if (!console_.lines.length) {
      return '<div class="empty">Nothing reported yet.</div>';
    }
    return '<div class="con">' + console_.lines.map(function (line) {
      var when = new Date((line.at || 0) * 1000);
      var stamp = ("0" + when.getHours()).slice(-2) + ":" +
        ("0" + when.getMinutes()).slice(-2) + ":" + ("0" + when.getSeconds()).slice(-2);
      // Depth is drawn as indentation rather than said: a supervised chain is
      // the one thing here whose SHAPE is the information.
      var pad = new Array(Math.min(line.depth || 0, 4) + 1).join("  ");
      return '<div class="ln ' + esc(line.level) + '"><span class="ts">' + stamp +
        '</span><span class="tx">' + esc(pad + line.text) + "</span></div>";
    }).join("") + "</div>";
  }

  function loadConsole(runId) {
    if (!runId || !token) { return; }
    fetch("server-debug/runs/" + encodeURIComponent(runId) + ".json",
          { headers: { Authorization: "Bearer " + token } })
      .then(function (r) {
        if (r.status === 404) { throw new Error("This run has been reaped."); }
        if (!r.ok) { throw new Error("The server answered " + r.status + "."); }
        return r.json();
      })
      .then(function (d) {
        if (selected !== runId) { return; }   // the reader moved on mid-flight
        console_ = { run: runId, lines: d.lines || [], error: "" };
        drawDrawer();
      })
      .catch(function (e) {
        if (selected !== runId) { return; }
        console_ = { run: runId, lines: [], error: e.message };
        drawDrawer();
      });
  }

  function pick(node) {
    var host = node.closest("[data-run]");
    if (!host) { return; }
    selected = host.getAttribute("data-run");
    // Cleared rather than kept: showing the previous run's console under the
    // new run's name for one poll is the worst thing this panel could do.
    console_ = { run: selected, lines: [], error: "" };
    drawDrawer();
    loadConsole(selected);
  }

  // ---- assembly -----------------------------------------------------------
  function draw(d) {
    latest = d;
    var a = d.admission || {}, f = d.inflight || {};
    var srv = d.server || {};
    el("origin").textContent = window.location.host || "this server";
    el("host").textContent = [
      srv.node, srv.tools ? "tools: " + srv.tools : null,
      srv.device, srv.pid ? "pid " + srv.pid : null,
      (d.hardware || "").split(" · ").slice(1).join(" · ") || null,
      new Date().toLocaleTimeString(),
    ].filter(Boolean).join("  ·  ");
    el("c-req").firstElementChild.textContent = f.now == null ? "?" : f.now;
    el("c-req").className = "chip" + (f.now ? " busy" : "");
    el("c-run").firstElementChild.textContent = a.running == null ? "?" : a.running;
    el("c-run").className = "chip" + (a.running ? " busy" : "");
    el("c-wait").firstElementChild.textContent = a.waiting == null ? "?" : a.waiting;
    el("c-wait").className = "chip" + (a.waiting ? " wait" : "");
    drawMachine(d);
    drawQueue(d);
    drawRuns(d);
    drawHistory(d);
    drawFilters();
    if (selected) { drawDrawer(); loadConsole(selected); }
  }

  function load() {
    if (!token) { show(el("gate"), true); show(el("shell"), false); return; }
    fetch("server-debug.json", { headers: { Authorization: "Bearer " + token } })
      .then(function (r) {
        if (r.status === 401) { throw new Error("That token was refused."); }
        if (!r.ok) { throw new Error("The server answered " + r.status + "."); }
        return r.json();
      })
      .then(function (d) {
        show(el("gate"), false);
        show(el("shell"), true);
        draw(d);
      })
      .catch(function (e) {
        show(el("gate"), true);
        show(el("shell"), false);
        el("gateerr").textContent = e.message;
      });
  }

  function tick() { if (live) { load(); } window.setTimeout(tick, EVERY); }

  el("save").addEventListener("click", function () {
    token = el("token").value.trim();
    try { window.localStorage.setItem(KEY, token); } catch (e) { /* private mode */ }
    // Cleared from the field as soon as it has been taken: the browser keeps it,
    // this page does not need to go on showing it, and a dashboard left open on
    // a wall should not have a credential sitting in an input.
    el("token").value = "";
    el("gateerr").textContent = "";
    load();
  });
  el("token").addEventListener("keydown", function (event) {
    if (event.key === "Enter") { el("save").click(); }
  });
  el("toggle").addEventListener("click", function () {
    live = !live;
    el("toggle").textContent = live ? "pause" : "resume";
    el("dot").className = live ? "dot" : "dot off";
    if (live) { load(); }
  });
  el("close").addEventListener("click", function () {
    selected = null;
    el("drawer").classList.remove("open");
  });
  var findTimer = null;
  el("find").addEventListener("input", function () {
    // Debounced: the poll redraws every two seconds anyway, and refiltering on
    // every keystroke of a 32-character run id is work nobody sees.
    window.clearTimeout(findTimer);
    findTimer = window.setTimeout(function () {
      filters.text = el("find").value.trim();
      writeUrl();
      if (latest) { draw(latest); }
    }, 180);
  });

  document.addEventListener("click", function (event) {
    var chip = event.target.closest("[data-clear]");
    if (chip) { setFilter(chip.getAttribute("data-clear"), filters[chip.getAttribute("data-clear")]); return; }
    var fbtn = event.target.closest("[data-filter]");
    if (fbtn) {
      setFilter(fbtn.getAttribute("data-filter"), fbtn.getAttribute("data-value"));
      return;
    }
    var copy = event.target.closest("[data-copy]");
    if (copy) {
      var id = copy.getAttribute("data-copy");
      if (navigator.clipboard) { navigator.clipboard.writeText(id); }
      copy.textContent = "copied";
      window.setTimeout(function () { copy.textContent = "copy id"; }, 1200);
      return;
    }
    if (event.target.closest("#drawer")) { return; }
    pick(event.target);
  });

  window.addEventListener("hashchange", function () {
    readUrl();
    if (latest) { draw(latest); }
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") { el("close").click(); }
  });

  // A stable hue per origin: the same server is always the same colour, and a
  // different port is always a different one. Hashed rather than configured so
  // a new deployment needs no entry anywhere.
  (function () {
    var key = window.location.host || "local", hash = 0, i;
    for (i = 0; i < key.length; i++) { hash = (hash * 31 + key.charCodeAt(i)) % 360; }
    el("flag").style.background = "hsl(" + hash + " 55% 52%)";
  })();

  readUrl();
  drawFilters();
  load();
  tick();
})();
</script>
</body>
</html>
"""
