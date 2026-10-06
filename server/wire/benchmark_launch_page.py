"""The page `GET /benchmark` serves: what was measured, and what to measure next.

Two tabs, and the split is the point. **Results** is the campaign viewer
unchanged -- the Gantt, the VRAM trace, the nested calls drawn inside their
parent's span -- carried here in an iframe rather than reimplemented, because
it already answers its question well and a second copy would be a second thing
to keep right. **Launch** is the new half: press a preset and watch it run.

Why an iframe and not a merge: that page is a self-contained document with its
own head, its own 2400px layout and its own token gate. Inlining it would mean
either flattening it into this one -- losing the width it needs -- or running
two stylesheets against each other. Same origin, no network, nothing external:
the iframe costs a tag and keeps the page that works exactly as it is.

**A preset names tools and a shape, never arguments.** Everything a battery
sends is resolved from what this deployment hosts, so the launcher's job is to
show what WOULD run before anything does: which tools resolve, how many runs,
and which of them cannot be satisfied here. A reader should never press a
button whose cost they could not have predicted.

**One battery at a time, and the page says which.** Two would measure each
other. The launcher disables itself while one runs and shows what it is, so a
second reader on another screen sees why their button is off.

**It looks like the admin panel**, and borrows its stylesheet to: the colour
tokens, the lit ground and the controls come from `glass_style`, so a reader
moving between the operator pages never crosses into another product. The
framed viewer draws no ground of its own, and both documents follow one
theme key.

Self-contained, like every page in this directory: this server runs where there
is no internet, so a stylesheet from a CDN is a broken page rather than a
degraded one. No token is written into the served HTML -- the reader types one
and the browser keeps it, under the same key the admin panel uses.

**It answers to the ADMIN token, not the API one.** Benchmarks are the
developer's and the operator's instrument, part of the admin panel in all but
URL; a workstation holds the API token for its runs, and that opens nothing
here. The battery itself still talks to this server with the API token, as any
client would: that is the thing it measures.
"""

from __future__ import annotations

from wire.glass_style import GLASS_BASE, GLASS_SELECT_JS

LAUNCH_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmarks</title>
<style>
""" + GLASS_BASE + """
  /* The admin panel's shell: a frosted bar over the lit ground, and cards
     beneath it. The Results tab is a frame holding the campaign viewer, which
     draws no ground of its own when framed, so the glows run through both. */
  html, body { height: 100%; }
  body { display: flex; flex-direction: column; }
  header#bar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
               margin: 10px 20px 0; padding: 10px 14px; border-radius: var(--radius); flex: none; }
  #flag { width: 10px; height: 38px; border-radius: 5px; background: var(--ghost); flex: none; }
  h1 { font-size: 20px; font-weight: 700; letter-spacing: -.015em; margin: 0; line-height: 1.15; }
  .sub { color: var(--soft); font-size: 12px; }
  .grow { flex: 1; }
  /* The two tabs as one segmented control, the one the clients card uses. */
  .seg { display: inline-flex; gap: 2px; padding: 2px; border-radius: 9px; background: var(--bar); margin-left: 8px; }
  .seg button { border: none; border-radius: 7px; padding: 5px 14px; font-size: 13px; background: transparent; }
  .seg button.on { background: var(--panel-strong); color: var(--ink); font-weight: 600;
                   box-shadow: 0 1px 3px rgba(0,0,0,.12), 0 0 0 .5px rgba(0,0,0,.04); }

  .card, header#bar { background: var(--panel); -webkit-backdrop-filter: var(--blur); backdrop-filter: var(--blur);
          border: 1px solid var(--panel-edge); box-shadow: var(--shadow); }
  .card { border-radius: var(--radius); margin-bottom: 14px; min-width: 0; }
  .card > h2 { margin: 0; padding: 13px 16px 10px; font-size: 15px; font-weight: 600; letter-spacing: -.01em;
               display: flex; align-items: center; gap: 8px; border-bottom: 1px solid var(--line); }
  .card > h2 .note { margin-left: auto; font-weight: 400; color: var(--ghost); font-size: 11.5px; }
  .body { padding: 12px 16px 14px; }

  main { flex: 1; min-height: 0; display: flex; }
  iframe { flex: 1; width: 100%; border: 0; background: transparent; color-scheme: normal; }
  #launch { flex: 1; overflow: auto; padding: 14px 20px 28px; }
  .wrap { max-width: 1180px; margin: 0 auto; }

  .note { font-size: 12.5px; color: var(--soft); line-height: 1.5; margin: 0; }
  .grid { display: grid; gap: 10px;
          grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); }
  .preset { border: 1px solid var(--panel-edge); border-radius: 12px; padding: 12px 14px;
            background: var(--sunk); display: flex; flex-direction: column; gap: 8px;
            transition: border-color .15s; }
  .preset:hover { border-color: var(--accent); }
  .preset h3 { margin: 0; font-size: 14.5px; font-weight: 700; letter-spacing: -.01em; }
  .preset p { margin: 0; font-size: 12px; color: var(--soft); line-height: 1.45; }
  .facts { display: flex; gap: 5px; flex-wrap: wrap; margin-top: auto; }
  .tag { display: inline-block; font-size: 11px; font-weight: 600; padding: 1px 7px; border-radius: 4px;
         background: var(--bar); color: var(--soft); white-space: nowrap; }
  .tag.n { background: var(--accent-soft); color: var(--accent); }
  .tag.stop { background: color-mix(in srgb, var(--hot) 14%, transparent); color: var(--hot); }
  .preset .go { align-self: flex-start; }
  /* Two controls, because the two questions are different: "which tool" wants
     a list you read top to bottom, "which of these" wants to be clickable one
     at a time without a dropdown closing over the answer. */
  .pick { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  .pick select { font-size: 12.5px; padding-top: 4px; padding-bottom: 4px; max-width: 100%; }
  .chips { display: flex; gap: 5px; flex-wrap: wrap; }
  .chip { font-size: 11.5px; padding: 2px 10px; border-radius: 999px; cursor: pointer;
          border: 1px solid var(--line); color: var(--ghost); user-select: none;
          transition: background .15s, border-color .15s; }
  .chip.on { border-color: transparent; color: var(--accent); background: var(--accent-soft); font-weight: 600; }
  .chip:hover { border-color: var(--soft); }
  .mini { font-size: 11.5px; padding: 2px 9px; border-radius: 999px; }
  .count { font-size: 11.5px; color: var(--soft); }

  .progress { height: 6px; background: var(--bar); border-radius: 3px; overflow: hidden; margin: 10px 0 6px; }
  .progress i { display: block; height: 100%; background: var(--ok); border-radius: 3px; transition: width .5s ease; }
  .live { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .live b { font-size: 17px; font-weight: 700; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ok);
         box-shadow: 0 0 0 3px color-mix(in srgb, var(--ok) 22%, transparent); }
  .dot.off { background: var(--ghost); box-shadow: none; }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th { text-align: left; font-weight: 600; color: var(--ghost); font-size: 11px; letter-spacing: .05em;
       text-transform: uppercase; padding: 8px 10px; border-bottom: 1px solid var(--line); }
  td { padding: 7px 10px; border-bottom: 1px solid var(--line); }
  tbody tr:hover td { background: var(--sunk); }
  .pill { display: inline-block; font-size: 11px; font-weight: 600; padding: 1px 7px; border-radius: 4px; white-space: nowrap; }
  .pill.ready { background: color-mix(in srgb, var(--ok) 14%, transparent); color: var(--ok); }
  .pill.notready { background: color-mix(in srgb, var(--warn) 16%, transparent); color: var(--warn); }
  .err { color: var(--hot); font-size: 12.5px; margin-top: 8px; }
  code { font-size: 12px; background: var(--bar); padding: 0 5px; border-radius: 4px; }
  /* The custom battery: the shape on one line, then one card per
     configuration, side by side so A and B read against each other. */
  .cshape { display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: flex-end; }
  .cf { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: var(--soft); min-width: 0; }
  .cf > span { font-size: 11px; letter-spacing: .05em; text-transform: uppercase; color: var(--ghost); font-weight: 600; }
  .cf > span i { text-transform: none; letter-spacing: 0; font-style: normal; font-weight: 400; }
  .cf input { min-width: 0; width: 120px; padding: 5px 9px; }
  .cf.wide input { width: 260px; }
  .seg.small { margin-left: 0; }
  .seg.small button { padding: 4px 12px; font-size: 12.5px; }
  .configs { display: grid; gap: 12px; margin-top: 14px;
             grid-template-columns: repeat(auto-fit, minmax(min(380px, 100%), 1fr)); }
  .cfg { border: 1px solid var(--panel-edge); border-radius: 14px; background: var(--sunk);
         padding: 10px 12px 12px; min-width: 0; }
  .cfg.add { display: grid; place-items: center; border-style: dashed; border-color: var(--line);
             background: transparent; color: var(--soft); cursor: pointer; min-height: 120px; align-self: start; }
  .cfg.add:hover { border-color: var(--accent); color: var(--accent); }
  .chd { display: flex; align-items: center; gap: 8px; padding-bottom: 8px; margin-bottom: 8px;
         border-bottom: 1px solid var(--line); }
  .letter { width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center; flex: none;
            background: var(--accent); color: #fff; font-weight: 700; font-size: 12.5px; }
  .chd select { font-size: 13.5px; font-weight: 650; padding-top: 4px; padding-bottom: 4px; }
  .chd .grow { flex: 1; }
  .chd button { padding: 3px 9px; font-size: 12px; }
  .fsec { font-size: 10.5px; letter-spacing: .06em; text-transform: uppercase; color: var(--ghost);
          font-weight: 650; margin: 10px 0 4px; }
  .fld { display: grid; grid-template-columns: minmax(110px, 38%) minmax(0, 1fr); gap: 8px;
         align-items: center; padding: 3px 0; font-size: 12.5px; }
  .fld > label { color: var(--soft); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .fld > label b { color: var(--hot); font-weight: 700; margin-left: 3px; }
  .fld.changed > label { color: var(--accent); font-weight: 600; }
  .fld select { width: 100%; min-width: 0; font-size: 12.5px; padding-top: 4px; padding-bottom: 4px; }
  .fld input[type=text], .fld input[type=number] {
    width: 100%; min-width: 0; font-size: 12.5px; padding: 4px 8px; border-radius: 8px;
    border: 1px solid var(--line); background: var(--panel-strong); color: var(--ink); }
  .fld input[type=checkbox] { width: 16px; height: 16px; accent-color: var(--accent); min-width: 0; }
  .fld.wide { grid-template-columns: 1fr; gap: 4px; }
  .mc { display: flex; flex-direction: column; gap: 5px; min-width: 0; }
  .mc .chips { max-height: 132px; overflow: auto; padding: 2px; }
  .mc .chip { font-size: 11px; padding: 1px 8px; }
  .mc .mcbar { display: flex; gap: 6px; align-items: center; font-size: 11.5px; color: var(--soft); }
  .mc .mcbar button { padding: 1px 8px; font-size: 11px; border-radius: 999px; }
  details.tech { margin-top: 8px; border-top: 1px dashed var(--line); padding-top: 6px; }
  details.tech summary { cursor: pointer; font-size: 12px; color: var(--soft); }
  .cfoot { display: flex; align-items: center; gap: 10px; margin-top: 14px; flex-wrap: wrap; }
  .csum { font-size: 13px; color: var(--soft); }
  .csum b { color: var(--ink); }
  @media (max-width: 640px) { header#bar { margin: 8px 10px 0; } #launch { padding: 12px 10px 24px; } }
</style>
</head>
<body>

<header id="bar">
  <span id="flag" title="derived from this page's own origin"></span>
  <div>
    <h1>Benchmarks</h1>
    <div class="sub mono" id="hint">what was measured, and what to measure next</div>
  </div>
  <div class="seg" role="tablist">
    <button id="t-launch" type="button" role="tab">Launch</button>
    <button id="t-results" type="button" role="tab">Results</button>
  </div>
  <span class="grow"></span>
  <a class="navlink" href="admin-panel">Admin panel</a>
  <button id="theme" type="button" class="ghost" title="theme">Auto</button>
</header>

<main>
  <iframe id="results" title="Campaign results" allowtransparency="true"></iframe>
  <div id="launch" hidden>
    <div class="wrap">

      <section class="card" id="gatebox" hidden>
        <h2>Admin token</h2>
        <div class="body">
          <p class="note">Benchmarks open with the server's admin token
            (<code>ADMIN_TOKEN</code>), the one the admin panel asks for. The
            API token a workstation uses for its runs does not open them. The
            token stays in this browser.</p>
          <div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap">
            <input id="token" type="password" placeholder="Admin token" autocomplete="off">
            <button id="save" type="button" class="primary">Connect</button>
          </div>
          <div class="err" id="gateerr"></div>
        </div>
      </section>

      <div id="body" hidden>
        <section class="card" id="runningbox" hidden>
          <h2><span class="dot" id="r-dot"></span>Running now</h2>
          <div class="body">
            <div class="live">
              <b id="r-label"></b>
              <span class="note mono" id="r-detail"></span>
              <span class="grow"></span>
              <button id="stop" type="button">Stop</button>
            </div>
            <div class="progress"><i id="r-bar" style="width:0%"></i></div>
            <div class="note" id="r-foot"></div>
          </div>
        </section>

        <section class="card" id="custombox">
          <h2>Custom battery<span class="note">your tool, your arguments, your case</span></h2>
          <div class="body">
            <div class="cshape">
              <label class="cf wide"><span>Label</span>
                <input data-cs="label" type="text" placeholder="what this battery is asking" maxlength="80"></label>
              <div class="cf"><span>Shape</span>
                <div class="seg small" id="c-shape">
                  <button type="button" data-shape="sequential">One at a time</button>
                  <button type="button" data-shape="parallel">In parallel</button>
                </div></div>
              <label class="cf"><span id="c-repeats-label">Runs per configuration</span>
                <input data-cs="repeats" type="number" min="1" step="1"></label>
              <label class="cf par"><span>At once</span>
                <input data-cs="concurrency" type="number" min="1" step="1"></label>
              <label class="cf par"><span>Ladder <i>optional, e.g. 1,2,4</i></span>
                <input data-cs="ladder" type="text" placeholder="widths in turn"></label>
              <label class="cf par"><span>Stagger <i>seconds between starts</i></span>
                <input data-cs="stagger" type="number" min="0" step="1"></label>
            </div>
            <div class="configs" id="c-configs"></div>
            <div class="cfoot">
              <span class="csum" id="c-sum"></span>
              <span class="grow"></span>
              <button type="button" class="primary" id="c-run">Run this battery</button>
            </div>
            <div class="err" id="c-err"></div>
            <p class="note" style="margin-top:8px">A bench input is a case staged on the server under
              <code>DATA/&lt;tool&gt;/bench/</code>. It is never reachable by name: the battery reads it
              on this machine and uploads it like a workstation would, drawn as its own
              <i>transfer</i> phase, and the results record its size, never its name.</p>
          </div>
        </section>

        <section class="card">
          <h2>Presets<span class="note" id="limits"></span></h2>
          <div class="body">
            <div class="grid" id="presets"></div>
            <div class="err" id="runerr"></div>
          </div>
        </section>

        <section class="card">
          <h2>What this deployment can run</h2>
          <div class="body">
            <p class="note">A preset names tools and a shape, never arguments:
              every value is resolved from what this server hosts, the way a
              clinician's panel fills its form. A tool with nothing staged for a
              required input is listed here rather than failing halfway through a
              battery.</p>
            <div style="margin-top:10px;overflow:auto"><table id="tools"></table></div>
          </div>
        </section>

        <section class="card">
          <h2>What these numbers are, and are not</h2>
          <div class="body">
            <p class="note">A battery sends hosted file NAMES, so nothing is
              uploaded: it measures dispatch, admission and the tools, and says
              nothing about the transfer path. The campaigns under
              <code>benchmarks/</code> remain the instrument for the wire, and a
              battery's summary is named <code>preset-*</code> so the two are
              never read as each other.</p>
          </div>
        </section>
      </div>

    </div>
  </div>
</main>

<script>
(function () {
  "use strict";
  // The admin panel's key: unlocking either page unlocks both.
  var KEY = "visor.admin";
  var token = "";
  try { token = window.localStorage.getItem(KEY) || ""; } catch (e) { token = ""; }
  var timer = null;

  function el(id) { return document.getElementById(id); }
  function show(node, on) { node.hidden = !on; }
  function esc(text) {
    var box = document.createElement("div");
    box.textContent = String(text == null ? "" : text);
    return box.innerHTML;
  }
  function dur(s) {
    s = Math.max(0, Math.round(s || 0));
    if (s < 60) { return s + "s"; }
    return Math.floor(s / 60) + "m" + ("0" + (s % 60)).slice(-2);
  }

  function ask(path, options) {
    options = options || {};
    options.headers = { "X-Admin-Token": token,
                        "Content-Type": "application/json" };
    return fetch(path, options).then(function (r) {
      if (r.status === 401) { throw new Error("That is not this server's admin token."); }
      if (r.status === 403) { throw new Error("This server has no admin token: set ADMIN_TOKEN in its .env and recreate it."); }
      return r.json().then(function (body) {
        if (!r.ok) { throw new Error(body.detail || ("HTTP " + r.status)); }
        return body;
      });
    });
  }

  // ---- tabs ------------------------------------------------------------
  function tab(which) {
    var results = which === "results";
    el("t-results").className = results ? "on" : "";
    el("t-launch").className = results ? "" : "on";
    el("t-results").setAttribute("aria-selected", results ? "true" : "false");
    el("t-launch").setAttribute("aria-selected", results ? "false" : "true");
    // The tab is the fragment, so "Run a benchmark" can link straight to it.
    try { window.history.replaceState(null, "", results ? "#results" : "#launch"); } catch (e) { /* still renders */ }
    show(el("results"), results);
    show(el("launch"), !results);
    if (results && !el("results").getAttribute("src")) {
      // Loaded lazily: the campaign viewer fetches a whole summary on open,
      // and a reader who came here to launch something should not pay for it.
      // The query carries the campaign a shared link asked for.
      el("results").setAttribute("src", "benchmarks/view" + window.location.search);
    }
    if (!results) { refresh(); }
  }
  el("t-results").addEventListener("click", function () { tab("results"); });
  el("t-launch").addEventListener("click", function () { tab("launch"); });

  // ---- the catalogue ---------------------------------------------------
  // Kept across redraws: the page refreshes every three seconds and a reader
  // mid-way through choosing four tools must not have the list reset under them.
  var chosen = {};
  var lastCatalogue = null;

  function presetById(id) {
    var found = ((lastCatalogue || {}).presets || []).filter(function (p) {
      return p.id === id;
    });
    return found[0];
  }

  function selectionFor(p) {
    if (!chosen[p.id]) {
      chosen[p.id] = p.default_tools && p.default_tools.length
        ? p.default_tools.slice()
        : (p.candidates || []).slice();
    }
    // A candidate that stopped resolving since the last poll drops out rather
    // than being sent to a plan that would refuse it.
    return chosen[p.id].filter(function (t) {
      return (p.candidates || []).indexOf(t) !== -1;
    });
  }

  function control(p) {
    if (!p.choose || p.blocked) { return ""; }
    var picked = selectionFor(p);
    if (p.choose === "one") {
      var one = picked[0] || (p.candidates || [])[0];
      return '<div class="pick"><span class="count">tool</span><select data-one="' +
        esc(p.id) + '">' + (p.candidates || []).map(function (t) {
          return '<option value="' + esc(t) + '"' +
            (t === one ? " selected" : "") + ">" + esc(t) + "</option>";
        }).join("") + "</select></div>";
    }
    return '<div class="pick"><button class="mini" data-all="' + esc(p.id) +
      '">All</button><button class="mini" data-none="' + esc(p.id) +
      '">None</button><span class="count">' + picked.length + " of " +
      (p.candidates || []).length + "</span></div>" +
      '<div class="chips">' + (p.candidates || []).map(function (t) {
        return '<span class="chip' + (picked.indexOf(t) !== -1 ? " on" : "") +
          '" data-toggle="' + esc(p.id) + '" data-tool="' + esc(t) + '">' +
          esc(t) + "</span>";
      }).join("") + "</div>";
  }

  function drawPresets(d) {
    lastCatalogue = d;
    var busy = d.running && d.running.state === "running";
    el("limits").textContent =
      "At most " + d.limits.max_runs + " runs and " +
      d.limits.max_concurrency + " at once per battery. One battery at a time: " +
      "two would measure each other.";

    el("presets").innerHTML = (d.presets || []).map(function (p) {
      var facts = [];
      if (p.blocked) {
        facts.push('<span class="tag stop">cannot run here</span>');
      } else {
        facts.push('<span class="tag n">' + p.runs + " run" +
                   (p.runs === 1 ? "" : "s") + "</span>");
        facts.push('<span class="tag">' + esc(p.shape) + "</span>");
        if (p.ladder) {
          facts.push('<span class="tag">' + p.ladder.join(" \\u2192 ") + "</span>");
        }
        (p.tools || []).slice(0, 4).forEach(function (t) {
          facts.push('<span class="tag">' + esc(t) + "</span>");
        });
        if ((p.tools || []).length > 4) {
          facts.push('<span class="tag">+' + (p.tools.length - 4) + "</span>");
        }
      }
      return '<div class="preset"><h3>' + esc(p.label) + "</h3>" +
        "<p>" + esc(p.about) + "</p>" +
        (p.blocked ? '<p class="note" style="color:var(--warn)">' +
          esc(p.blocked) + "</p>" : "") +
        '<div class="facts">' + facts.join("") + "</div>" +
        control(p) +
        '<button class="go primary" data-preset="' + esc(p.id) + '"' +
        (p.blocked || busy ? " disabled" : "") + ">" +
        (busy ? "A battery is running" : "Run this") + "</button></div>";
    }).join("");

    var names = Object.keys(d.tools || {});
    el("tools").innerHTML = "<thead><tr><th>tool</th><th>state</th>" +
      "<th>arguments a battery would send</th></tr></thead><tbody>" +
      names.map(function (name) {
        var t = d.tools[name];
        return "<tr><td><b>" + esc(name) + '</b></td><td><span class="pill ' +
          (t.ready ? "ready" : "notready") + '">' +
          (t.ready ? "ready" : "needs " + esc(t.missing.join(", "))) + "</span>" +
          '</td><td class="mono note">' +
          (t.params.length ? esc(t.params.join(", ")) : "\\u2014") +
          "</td></tr>";
      }).join("") + "</tbody>";
  }

  function drawRunning(job) {
    var busy = job && job.state === "running";
    show(el("runningbox"), !!job);
    if (!job) { return; }
    el("r-label").textContent = job.label || job.preset;
    el("r-detail").textContent = job.total_runs + " run(s) \\u00b7 " +
      dur(job.elapsed) + " elapsed";
    el("stop").disabled = !busy;
    el("stop").textContent = busy ? "Stop" : "Finished";
    el("r-dot").className = busy ? "dot" : "dot off";
    // Elapsed against nothing: a battery's duration is not knowable up front
    // (that is what it is measuring), so the bar paces rather than predicts.
    el("r-bar").style.width = busy ? Math.min(96, (job.elapsed / 6)) + "%" : "100%";
    el("r-foot").textContent = busy
      ? "Its summary lands in the Results tab as each arm finishes."
      : "Finished as " + esc(job.summary) + " \\u2014 open the Results tab.";
  }

  function refresh() {
    if (!token) { show(el("gatebox"), true); show(el("body"), false); return; }
    ask("benchmark/presets").then(function (d) {
      show(el("gatebox"), false);
      show(el("body"), true);
      drawPresets(d);
      drawRunning(d.running);
      if (d.custom_limits) { climits = d.custom_limits; }
      var busy = !!(d.running && d.running.state === "running");
      if (!custom) {
        batteryBusy = busy;
        ensureCustom(Object.keys(d.tools || {}).filter(function (n) { return d.tools[n].ready; }).sort());
      } else if (busy !== batteryBusy) {
        batteryBusy = busy;
        drawCustom();
      }
    }).catch(function (e) {
      show(el("gatebox"), true);
      show(el("body"), false);
      el("gateerr").textContent = e.message;
    });
  }

  function repaint() {
    if (lastCatalogue) { drawPresets(lastCatalogue); }
  }

  document.addEventListener("click", function (event) {
    var chip = event.target.closest("[data-toggle]");
    if (chip) {
      var id = chip.getAttribute("data-toggle"), tool = chip.getAttribute("data-tool");
      var list = chosen[id] || [];
      var at = list.indexOf(tool);
      if (at === -1) { list.push(tool); } else { list.splice(at, 1); }
      chosen[id] = list;
      repaint();
      return;
    }
    var all = event.target.closest("[data-all]");
    if (all) {
      var pa = presetById(all.getAttribute("data-all"));
      chosen[pa.id] = (pa.candidates || []).slice();
      repaint();
      return;
    }
    var none = event.target.closest("[data-none]");
    if (none) {
      chosen[none.getAttribute("data-none")] = [];
      repaint();
      return;
    }

    var go = event.target.closest("[data-preset]");
    if (!go || go.disabled) { return; }
    var preset = presetById(go.getAttribute("data-preset"));
    var tools = preset && preset.choose ? selectionFor(preset) : null;
    if (preset && preset.choose === "one") {
      var box = document.querySelector('[data-one="' + preset.id + '"]');
      tools = box ? [box.value] : tools;
    }
    if (preset && preset.choose && (!tools || !tools.length)) {
      el("runerr").textContent = "Pick at least one tool for " + preset.label + ".";
      return;
    }
    el("runerr").textContent = "";
    go.disabled = true;
    ask("benchmark/run", { method: "POST",
                           body: JSON.stringify({
                             preset: go.getAttribute("data-preset"),
                             tools: tools }) })
      .then(function () { refresh(); })
      .catch(function (e) { el("runerr").textContent = e.message; refresh(); });
  });

  // A dropdown is the one control whose value lives in the DOM rather than in
  // `chosen`, so it is written back on change or a redraw would lose it.
  document.addEventListener("change", function (event) {
    var box = event.target.closest("[data-one]");
    if (box) { chosen[box.getAttribute("data-one")] = [box.value]; }
  });

  el("stop").addEventListener("click", function () {
    ask("benchmark/run", { method: "DELETE" }).then(refresh).catch(function (e) {
      el("runerr").textContent = e.message;
    });
  });

  el("save").addEventListener("click", function () {
    token = el("token").value.trim();
    try { window.localStorage.setItem(KEY, token); } catch (e) { /* private */ }
    el("token").value = "";
    el("gateerr").textContent = "";
    refresh();
  });
  el("token").addEventListener("keydown", function (event) {
    if (event.key === "Enter") { el("save").click(); }
  });

  timer = window.setInterval(function () {
    if (!el("launch").hidden) { refresh(); }
  }, 3000);

  // ---- the custom battery ----------------------------------------------
  // Drawn only when what it shows changes, never by the poll: the catalogue
  // above repaints every three seconds, and a form being typed into must not.
  var VALUE_TYPES = ["str", "int", "float", "bool", "choice", "multichoice", "list[str]"];
  var LETTERS = "ABCD";
  var schemas = null, forms = {}, climits = { max_runs: 60, max_concurrency: 16, max_configs: 4 };
  var batteryBusy = false;
  var CUSTOM_KEY = "visor.bench.custom";
  var custom = null;

  function blank() {
    return { label: "", shape: "sequential", repeats: 3, concurrency: 2, ladder: "", stagger: 0, configs: [] };
  }
  // The last composition, kept in this browser: a battery is usually re-run
  // after a change, and retyping it is where a comparison quietly drifts.
  function loadCustom() {
    try {
      var saved = JSON.parse(window.localStorage.getItem(CUSTOM_KEY) || "null");
      if (saved && Array.isArray(saved.configs)) { return saved; }
    } catch (e) { /* private window, or something unreadable */ }
    return blank();
  }
  function saveCustom() {
    try { window.localStorage.setItem(CUSTOM_KEY, JSON.stringify(custom)); } catch (e) { /* fine */ }
  }

  function loadSchemas() {
    if (schemas) { return Promise.resolve(schemas); }
    return fetch("tools").then(function (r) { return r.json(); }).then(function (list) {
      schemas = {};
      list.forEach(function (t) { schemas[t.name] = t; });
      return schemas;
    });
  }
  function loadForm(tool) {
    if (forms[tool]) { return Promise.resolve(forms[tool]); }
    return ask("benchmark/form/" + encodeURIComponent(tool)).then(function (f) { forms[tool] = f; return f; });
  }

  function argsOf(tool) { return ((schemas || {})[tool] || {}).arguments || {}; }
  function isFile(d) { return VALUE_TYPES.indexOf(d.type) < 0 && d.server_selectable !== "model"; }
  function poolOf(d, hosted) {
    var kind = d.server_selectable, scope = d.selectable_scope;
    if (kind === "model") { return scope ? (hosted.models_by_scope || {})[scope] || [] : hosted.models || []; }
    if (kind === "testfile") { return scope ? (hosted.testfiles_by_scope || {})[scope] || [] : hosted.testfiles || []; }
    return [];
  }

  // What the form opens on: what a preset would have sent, then each
  // argument's own default. Only the first set and what is touched is sent.
  function seed(c) {
    var f = forms[c.tool], args = argsOf(c.tool);
    c.values = {}; c.sent = {};
    Object.keys(args).forEach(function (name) {
      var d = args[name], given = (f.defaults || {})[name];
      if (given !== undefined) { c.sent[name] = true; }
      if (d.server_selectable === "model") { c.values[name] = given === undefined ? "" : String(given); return; }
      if (isFile(d)) { c.values[name] = given === undefined ? "" : "t:" + given; return; }
      if (d.type === "multichoice") {
        var on = given !== undefined ? String(given).split(",") :
          Object.keys(d.choices || {}).filter(function (k) { return d.choices[k]; });
        c.values[name] = on; return;
      }
      if (d.type === "choice") {
        var pick = given !== undefined ? given : d.initial != null ? d.initial :
          Object.keys(d.choices || {}).filter(function (k) { return d.choices[k]; })[0];
        c.values[name] = pick == null ? "" : String(pick); return;
      }
      if (d.type === "bool") { c.values[name] = given !== undefined ? String(given) === "true" : !!d.initial; return; }
      c.values[name] = given !== undefined ? String(given) : d.initial == null ? "" : String(d.initial);
    });
  }

  // The panel's own rule for which fields are in force (formgen.is_visible).
  function applies(d, values) {
    var when = d.visible_when;
    if (!when) { return true; }
    return Object.keys(when).every(function (other) {
      if (!(other in values)) { return false; }
      var allowed = Array.isArray(when[other]) ? when[other] : [when[other]];
      var have = Array.isArray(values[other]) ? values[other] : [values[other]];
      return have.some(function (v) { return allowed.indexOf(v) >= 0; });
    });
  }

  function fieldHtml(i, c, name, d, f) {
    var v = c.values[name], key = ' data-ci="' + i + '" data-arg="' + esc(name) + '"', input;
    if (d.server_selectable === "model") {
      var models = poolOf(d, f.hosted);
      input = "<select" + key + ">" + (d.required ? "" : '<option value="">(automatic)</option>') +
        models.map(function (m) { return '<option value="' + esc(m) + '"' + (v === m ? " selected" : "") + ">" + esc(m) + "</option>"; }).join("") +
        "</select>";
    } else if (isFile(d)) {
      var tests = poolOf(d, f.hosted), bench = f.bench || [];
      function opt(value, text) { return '<option value="' + esc(value) + '"' + (v === value ? " selected" : "") + ">" + esc(text) + "</option>"; }
      input = "<select" + key + ">" + (d.required ? "" : opt("", "(not sent)")) +
        (tests.length ? '<optgroup label="Test files">' + tests.map(function (t) { return opt("t:" + t, t); }).join("") + "</optgroup>" : "") +
        (bench.length ? '<optgroup label="Bench · ' + esc(f.bench_folder) + '">' + bench.map(function (b) {
          return opt("b:" + b.name, b.name + "  ·  " + (b.kind === "folder" ? "folder, " : "") + bytesOf(b.size));
        }).join("") + "</optgroup>" : "") +
        (!tests.length && !bench.length ? opt("", "nothing hosted — stage one in " + f.bench_folder) : "") + "</select>";
    } else if (d.type === "choice") {
      input = "<select" + key + ">" + Object.keys(d.choices || {}).map(function (k) {
        return '<option value="' + esc(k) + '"' + (v === k ? " selected" : "") + ">" + esc(k) + "</option>";
      }).join("") + "</select>";
    } else if (d.type === "multichoice") {
      var all = Object.keys(d.choices || {});
      input = '<div class="mc"><div class="mcbar"><span>' + v.length + " of " + all.length + "</span>" +
        '<button type="button" data-mc="all"' + key + ">All</button>" +
        '<button type="button" data-mc="none"' + key + ">None</button></div>" +
        '<div class="chips">' + all.map(function (k) {
          return '<span class="chip' + (v.indexOf(k) >= 0 ? " on" : "") + '" data-opt="' + esc(k) + '"' + key + ">" + esc(k) + "</span>";
        }).join("") + "</div></div>";
    } else if (d.type === "bool") {
      input = '<input type="checkbox"' + key + (v ? " checked" : "") + ">";
    } else {
      input = '<input type="' + (d.type === "int" || d.type === "float" ? "number" : "text") + '"' +
        (d.type === "float" ? ' step="any"' : "") + key + ' value="' + esc(v) + '">';
    }
    return '<div class="fld' + (d.type === "multichoice" && Object.keys(d.choices || {}).length > 8 ? " wide" : "") + (c.sent[name] && !(forms[c.tool].defaults || {}).hasOwnProperty(name) ? " changed" : "") +
      '"><label title="' + esc(d.description || name) + '">' + esc(d.label || name) + (d.required ? "<b>*</b>" : "") +
      "</label>" + input + "</div>";
  }

  function bytesOf(n) {
    if (n == null) { return "?"; }
    if (n >= 1073741824) { return (n / 1073741824).toFixed(1) + " GB"; }
    if (n >= 1048576) { return (n / 1048576).toFixed(1) + " MB"; }
    return Math.max(1, Math.round(n / 1024)) + " kB";
  }

  function configCard(c, i) {
    var names = Object.keys(schemas || {}).sort();
    var head = '<div class="chd"><span class="letter">' + LETTERS[i] + "</span>" +
      '<select data-ctool="' + i + '">' + names.map(function (n) {
        return '<option value="' + esc(n) + '"' + (n === c.tool ? " selected" : "") + ">" + esc(n) + "</option>";
      }).join("") + '</select><span class="grow"></span>' +
      (custom.configs.length < climits.max_configs ? '<button type="button" data-cdup="' + i + '" title="a copy of this configuration, to change one thing and compare">Compare</button>' : "") +
      (custom.configs.length > 1 ? '<button type="button" class="ghost" data-cdel="' + i + '" title="remove">✕</button>' : "") + "</div>";
    var f = forms[c.tool];
    if (!f || !c.values) { return '<div class="cfg">' + head + '<div class="note">Loading…</div></div>'; }
    var args = argsOf(c.tool), shown = {}, tech = [];
    Object.keys(args).forEach(function (name) {
      var d = args[name];
      if (!applies(d, c.values)) { return; }
      if (d.hidden) { tech.push(name); return; }
      var section = d.section || "Arguments";
      (shown[section] = shown[section] || []).push(name);
    });
    var body = Object.keys(shown).map(function (section) {
      return '<div class="fsec">' + esc(section) + "</div>" +
        shown[section].map(function (name) { return fieldHtml(i, c, name, args[name], f); }).join("");
    }).join("");
    if (tech.length) {
      body += '<details class="tech"' + (c.techOpen ? " open" : "") + ' data-ctech="' + i + '"><summary>Technical · ' + tech.length +
        " argument" + (tech.length > 1 ? "s" : "") + " a clinician never sees</summary>" +
        tech.map(function (name) { return fieldHtml(i, c, name, args[name], f); }).join("") + "</details>";
    }
    if ((f.missing || []).length && !(f.bench || []).length) {
      body = '<div class="note" style="color:var(--warn)">Nothing hosted for ' + esc(f.missing.join(", ")) +
        ": stage a case in " + esc(f.bench_folder) + ".</div>" + body;
    }
    return '<div class="cfg">' + head + body + "</div>";
  }

  function widths() {
    if (custom.shape !== "parallel") { return [1]; }
    var ladder = String(custom.ladder || "").split(",").map(function (x) { return parseInt(x, 10); })
      .filter(function (x) { return x > 0; });
    return ladder.length ? ladder : [Math.max(1, parseInt(custom.concurrency, 10) || 1)];
  }
  function totalRuns() {
    var reps = Math.max(1, parseInt(custom.repeats, 10) || 1);
    if (custom.shape !== "parallel") { return reps * custom.configs.length; }
    return widths().reduce(function (a, w) { return a + w * reps; }, 0);
  }

  function drawCustom() {
    if (!custom || !schemas) { return; }
    el("c-configs").innerHTML = custom.configs.map(configCard).join("") +
      (custom.configs.length < climits.max_configs
        ? '<div class="cfg add" data-cadd="1">+ add a configuration</div>' : "");
    document.querySelectorAll("[data-shape]").forEach(function (b) {
      b.className = b.getAttribute("data-shape") === custom.shape ? "on" : "";
    });
    document.querySelectorAll("#custombox .par").forEach(function (n) { n.hidden = custom.shape !== "parallel"; });
    el("c-repeats-label").textContent = custom.shape === "parallel" ? "Rounds" : "Runs per configuration";
    ["label", "repeats", "concurrency", "ladder", "stagger"].forEach(function (k) {
      var box = document.querySelector('[data-cs="' + k + '"]');
      if (box && document.activeElement !== box) { box.value = custom[k] == null ? "" : custom[k]; }
    });
    var n = totalRuns(), k = custom.configs.length;
    var how = custom.shape === "parallel"
      ? widths().join(" → ") + " at once" + (custom.stagger > 0 ? ", " + custom.stagger + " s apart" : "")
      : "one at a time" + (k > 1 ? ", alternated " + LETTERS.slice(0, k).split("").join(", ") + "…" : "");
    el("c-sum").innerHTML = "<b>" + n + " run" + (n === 1 ? "" : "s") + "</b> · " + esc(how) +
      (n > climits.max_runs ? ' · <span style="color:var(--hot)">over the ' + climits.max_runs + " a battery may start</span>" : "");
    el("c-run").disabled = batteryBusy || n > climits.max_runs || !k;
    el("c-run").textContent = batteryBusy ? "A battery is running" : "Run this battery";
    saveCustom();
  }

  function useTool(i, tool) {
    custom.configs[i] = { tool: tool, values: null, sent: {} };
    drawCustom();
    loadForm(tool).then(function () {
      if (custom.configs[i] && custom.configs[i].tool === tool && !custom.configs[i].values) {
        seed(custom.configs[i]);
      }
      drawCustom();
    }).catch(function (e) { el("c-err").textContent = e.message; });
  }

  function ensureCustom(ready) {
    if (custom) { return; }
    custom = loadCustom();
    loadSchemas().then(function () {
      custom.configs = custom.configs.filter(function (c) { return schemas[c.tool]; });
      if (!custom.configs.length) {
        var first = ready.filter(function (t) { return t !== "Test_Tool" && t !== "Example_Tool"; })[0] || ready[0];
        if (!first) { drawCustom(); return; }
        custom.configs = [{ tool: first, values: null, sent: {} }];
      }
      return Promise.all(custom.configs.map(function (c) { return loadForm(c.tool); })).then(function () {
        custom.configs.forEach(function (c) { if (!c.values) { seed(c); } });
        drawCustom();
      });
    }).catch(function (e) { el("c-err").textContent = e.message; });
  }

  function payload() {
    return {
      label: custom.label, shape: custom.shape,
      repeats: parseInt(custom.repeats, 10) || 1,
      concurrency: parseInt(custom.concurrency, 10) || 1,
      ladder: custom.shape === "parallel" && String(custom.ladder || "").trim() ? widths() : [],
      stagger: custom.shape === "parallel" ? parseFloat(custom.stagger) || 0 : 0,
      configs: custom.configs.map(function (c) {
        var args = argsOf(c.tool), params = {}, bench = {};
        Object.keys(args).forEach(function (name) {
          var d = args[name], v = c.values[name];
          if (!c.sent[name] || !applies(d, c.values)) { return; }
          if (isFile(d) && typeof v === "string") {
            if (v.indexOf("b:") === 0) { bench[name] = v.slice(2); }
            else if (v.indexOf("t:") === 0) { params[name] = v.slice(2); }
            return;
          }
          params[name] = v;
        });
        return { tool: c.tool, params: params, bench: bench };
      }),
    };
  }

  function setValue(node, value) {
    var i = +node.getAttribute("data-ci"), name = node.getAttribute("data-arg"), c = custom.configs[i];
    c.values[name] = value;
    c.sent[name] = true;
    drawCustom();
  }

  el("custombox").addEventListener("change", function (event) {
    var t = event.target;
    if (t.hasAttribute("data-ctool")) { useTool(+t.getAttribute("data-ctool"), t.value); return; }
    if (t.hasAttribute("data-cs")) {
      custom[t.getAttribute("data-cs")] = t.type === "number" ? (t.value === "" ? "" : +t.value) : t.value;
      drawCustom(); return;
    }
    if (t.hasAttribute("data-arg") && !t.hasAttribute("data-opt") && !t.hasAttribute("data-mc")) {
      setValue(t, t.type === "checkbox" ? t.checked : t.value);
    }
  });
  el("custombox").addEventListener("toggle", function (event) {
    var t = event.target;
    if (t.hasAttribute && t.hasAttribute("data-ctech")) { custom.configs[+t.getAttribute("data-ctech")].techOpen = t.open; }
  }, true);
  el("custombox").addEventListener("click", function (event) {
    var t = event.target, n;
    if ((n = t.closest("[data-shape]"))) { custom.shape = n.getAttribute("data-shape"); drawCustom(); return; }
    if ((n = t.closest("[data-opt]"))) {
      var c = custom.configs[+n.getAttribute("data-ci")], name = n.getAttribute("data-arg"), opt = n.getAttribute("data-opt");
      var list = (c.values[name] || []).slice(), at = list.indexOf(opt);
      if (at < 0) { list.push(opt); } else { list.splice(at, 1); }
      setValue(n, list); return;
    }
    if ((n = t.closest("[data-mc]"))) {
      var d = argsOf(custom.configs[+n.getAttribute("data-ci")].tool)[n.getAttribute("data-arg")];
      setValue(n, n.getAttribute("data-mc") === "all" ? Object.keys(d.choices || {}) : []); return;
    }
    if ((n = t.closest("[data-cdup]"))) {
      var from = custom.configs[+n.getAttribute("data-cdup")];
      custom.configs.push(JSON.parse(JSON.stringify(from)));
      drawCustom(); return;
    }
    if ((n = t.closest("[data-cdel]"))) { custom.configs.splice(+n.getAttribute("data-cdel"), 1); drawCustom(); return; }
    if ((n = t.closest("[data-cadd]"))) {
      var last = custom.configs[custom.configs.length - 1];
      if (last) { custom.configs.push(JSON.parse(JSON.stringify(last))); drawCustom(); }
      return;
    }
    if (t.closest("#c-run") && !el("c-run").disabled) {
      el("c-err").textContent = "";
      el("c-run").disabled = true;
      ask("benchmark/run", { method: "POST", body: JSON.stringify({ custom: payload() }) })
        .then(function () { refresh(); })
        .catch(function (e) { el("c-err").textContent = e.message; drawCustom(); });
    }
  });

  // ---- theme ------------------------------------------------------------
  // The key every operator page writes. The frame below listens for it too, so
  // one toggle repaints both documents.
  var THEME_KEY = "visor.theme";
  var theme = "auto";
  try { theme = window.localStorage.getItem(THEME_KEY) || "auto"; } catch (e) { theme = "auto"; }
  function applyTheme() {
    if (theme === "light" || theme === "dark") {
      document.documentElement.setAttribute("data-theme", theme);
    } else {
      theme = "auto";
      document.documentElement.removeAttribute("data-theme");
    }
    el("theme").textContent = theme === "dark" ? "Dark" : theme === "light" ? "Light" : "Auto";
  }
  el("theme").addEventListener("click", function () {
    theme = theme === "auto" ? "dark" : theme === "dark" ? "light" : "auto";
    try { window.localStorage.setItem(THEME_KEY, theme); } catch (e) { /* private */ }
    applyTheme();
  });
  window.addEventListener("storage", function (event) {
    if (event.key === THEME_KEY) { theme = event.newValue || "auto"; applyTheme(); }
  });
  applyTheme();

  // A stable hue per origin, the admin panel's, so two deployments open side
  // by side never look alike.
  (function () {
    var key = window.location.host || "local", hash = 0, i;
    for (i = 0; i < key.length; i++) { hash = (hash * 31 + key.charCodeAt(i)) % 360; }
    el("flag").style.background = "hsl(" + hash + " 60% 52%)";
  })();

  // Launch first: the Benchmarks button is pressed to run something. Results
  // opens when the address asks for it -- its own tab, or a campaign named in
  // a shared link or carried over from the bare /benchmarks/view.
  tab(window.location.hash === "#results" || /[?&]campaign=/.test(window.location.search)
    ? "results" : "launch");
})();
</script>
""" + GLASS_SELECT_JS + """
</body>
</html>
"""
