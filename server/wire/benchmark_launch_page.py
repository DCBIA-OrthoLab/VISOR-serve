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

Self-contained, like every page in this directory: this server runs where there
is no internet, so a stylesheet from a CDN is a broken page rather than a
degraded one. No token is written into the served HTML -- the reader types one
and the browser keeps it, under the same key the other pages use.
"""

from __future__ import annotations

LAUNCH_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmarks</title>
<style>
  :root {
    --bg: #f4f4f2; --panel: #ffffff; --sunk: #edece8; --line: #dedcd5;
    --ink: #17181a; --soft: #6f6b63; --ghost: #a9a59c;
    --bar: #e2dfd8; --fill: #3d6b8e; --warn: #b5722f; --ok: #4a7c59;
    --hot: #a4433a;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #101113; --panel: #1a1c1f; --sunk: #141619; --line: #2b2e33;
      --ink: #eceae6; --soft: #918d86; --ghost: #5d6167;
      --bar: #24272b; --fill: #74aede; --warn: #dfa163; --ok: #7fb08c;
      --hot: #e08478;
    }
  }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  html, body { height: 100%; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
         display: flex; flex-direction: column; }
  .mono { font-variant-numeric: tabular-nums; }

  header { display: flex; align-items: center; gap: 14px; padding: 9px 14px;
           border-bottom: 1px solid var(--line); background: var(--panel);
           flex-wrap: wrap; }
  h1 { font-size: 13px; font-weight: 700; letter-spacing: .12em;
       text-transform: uppercase; margin: 0; }
  .grow { flex: 1; }
  .tabs { display: flex; gap: 4px; }
  .tab { font: inherit; color: var(--soft); background: none; cursor: pointer;
         border: 1px solid transparent; border-radius: 7px; padding: 5px 13px; }
  .tab.on { color: var(--ink); background: var(--sunk); border-color: var(--line); }
  button { font: inherit; color: inherit; background: var(--panel);
           border: 1px solid var(--line); border-radius: 7px; padding: 5px 12px;
           cursor: pointer; }
  button:hover:not(:disabled) { border-color: var(--soft); }
  button:disabled { opacity: .45; cursor: not-allowed; }
  input { font: inherit; color: inherit; background: var(--panel);
          border: 1px solid var(--line); border-radius: 7px; padding: 7px 10px;
          min-width: 220px; }

  main { flex: 1; min-height: 0; display: flex; }
  iframe { flex: 1; width: 100%; border: 0; background: var(--bg); }
  #launch { flex: 1; overflow: auto; padding: 16px; }
  .wrap { max-width: 1100px; margin: 0 auto; }

  .grid { display: grid; gap: 12px;
          grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); }
  .preset { background: var(--panel); border: 1px solid var(--line);
            border-radius: 10px; padding: 13px 15px; display: flex;
            flex-direction: column; gap: 7px; }
  .preset h3 { margin: 0; font-size: 14.5px; }
  .preset p { margin: 0; font-size: 12px; color: var(--soft); line-height: 1.45; }
  .facts { display: flex; gap: 6px; flex-wrap: wrap; margin-top: auto; }
  .tag { font-size: 10.5px; padding: 2px 8px; border-radius: 999px;
         border: 1px solid var(--line); color: var(--soft); }
  .tag.n { color: var(--fill); border-color: var(--fill); font-weight: 600; }
  .tag.stop { color: var(--hot); border-color: var(--hot); }
  .preset .go { align-self: flex-start; }
  /* Two controls, because the two questions are different: "which tool" wants
     a list you read top to bottom, "which of these" wants to be clickable one
     at a time without a dropdown closing over the answer. */
  .pick { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  .pick select { font: inherit; color: inherit; background: var(--panel);
                 border: 1px solid var(--line); border-radius: 7px;
                 padding: 5px 8px; max-width: 100%; }
  .chips { display: flex; gap: 4px; flex-wrap: wrap; }
  .chip { font-size: 11px; padding: 3px 9px; border-radius: 999px; cursor: pointer;
          border: 1px solid var(--line); color: var(--ghost); user-select: none; }
  .chip.on { border-color: var(--fill); color: var(--fill);
             background: color-mix(in srgb, var(--fill) 10%, transparent); }
  .chip:hover { border-color: var(--soft); }
  .mini { font-size: 10.5px; padding: 2px 8px; border-radius: 999px; }
  .count { font-size: 11px; color: var(--soft); }

  .panel { background: var(--panel); border: 1px solid var(--line);
           border-radius: 10px; padding: 14px 16px; margin-bottom: 14px; }
  h2 { font-size: 10.5px; font-weight: 600; letter-spacing: .1em;
       text-transform: uppercase; color: var(--soft); margin: 0 0 10px; }
  .note { font-size: 12px; color: var(--soft); line-height: 1.5; }
  .bar { height: 8px; background: var(--bar); border-radius: 4px; overflow: hidden;
         margin: 9px 0 6px; }
  .bar i { display: block; height: 100%; background: var(--fill);
           transition: width .4s ease; }
  .live { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
  .live b { font-size: 17px; }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th { text-align: left; font-weight: 500; color: var(--soft); font-size: 10px;
       letter-spacing: .07em; text-transform: uppercase; padding: 0 8px 6px 0; }
  td { padding: 4px 8px 4px 0; border-top: 1px solid var(--line); }
  td.r, th.r { text-align: right; }
  .ready { color: var(--ok); } .notready { color: var(--warn); }
  .err { color: var(--hot); font-size: 12.5px; margin-top: 8px; }
  #gate { max-width: 560px; margin: 12vh auto; }
</style>
</head>
<body>

<header>
  <h1>Benchmarks</h1>
  <div class="tabs">
    <button class="tab on" id="t-results" type="button">Results</button>
    <button class="tab" id="t-launch" type="button">Launch</button>
  </div>
  <span class="grow"></span>
  <span class="note mono" id="hint"></span>
</header>

<main>
  <iframe id="results" title="Campaign results"></iframe>
  <div id="launch" hidden>
    <div class="wrap">

      <div class="panel" id="gatebox" hidden>
        <h2>Token</h2>
        <p class="note">Starting a battery is a write, so it is Bearer-protected
          like every other endpoint that makes this server do work. The token
          stays in this browser.</p>
        <div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap">
          <input id="token" type="password" placeholder="API token" autocomplete="off">
          <button id="save" type="button">connect</button>
        </div>
        <div class="err" id="gateerr"></div>
      </div>

      <div id="body" hidden>
        <div class="panel" id="runningbox" hidden>
          <h2>Running now</h2>
          <div class="live">
            <b id="r-label"></b>
            <span class="note mono" id="r-detail"></span>
            <span class="grow"></span>
            <button id="stop" type="button">stop</button>
          </div>
          <div class="bar"><i id="r-bar" style="width:0%"></i></div>
          <div class="note" id="r-foot"></div>
        </div>

        <div class="panel">
          <h2>Presets</h2>
          <p class="note" id="limits"></p>
          <div class="grid" id="presets" style="margin-top:12px"></div>
          <div class="err" id="runerr"></div>
        </div>

        <div class="panel">
          <h2>What this deployment can run</h2>
          <p class="note">A preset names tools and a shape, never arguments:
            every value is resolved from what this server hosts, the way a
            clinician's panel fills its form. A tool with nothing staged for a
            required input is listed here rather than failing halfway through a
            battery.</p>
          <div style="margin-top:10px;overflow:auto"><table id="tools"></table></div>
        </div>

        <div class="panel">
          <h2>What these numbers are, and are not</h2>
          <p class="note">A battery sends hosted file NAMES, so nothing is
            uploaded: it measures dispatch, admission and the tools, and says
            nothing about the transfer path. The campaigns under
            <code>benchmarks/</code> remain the instrument for the wire, and a
            battery's summary is named <code>preset-*</code> so the two are
            never read as each other.</p>
        </div>
      </div>

    </div>
  </div>
</main>

<script>
(function () {
  "use strict";
  var KEY = "visor.token";
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
    options.headers = { Authorization: "Bearer " + token,
                        "Content-Type": "application/json" };
    return fetch(path, options).then(function (r) {
      if (r.status === 401) { throw new Error("That token was refused."); }
      return r.json().then(function (body) {
        if (!r.ok) { throw new Error(body.detail || ("HTTP " + r.status)); }
        return body;
      });
    });
  }

  // ---- tabs ------------------------------------------------------------
  function tab(which) {
    var results = which === "results";
    el("t-results").className = "tab" + (results ? " on" : "");
    el("t-launch").className = "tab" + (results ? "" : " on");
    show(el("results"), results);
    show(el("launch"), !results);
    if (results && !el("results").getAttribute("src")) {
      // Loaded lazily: the campaign viewer fetches a whole summary on open,
      // and a reader who came here to launch something should not pay for it.
      el("results").setAttribute("src", "benchmarks/view");
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
      '">all</button><button class="mini" data-none="' + esc(p.id) +
      '">none</button><span class="count">' + picked.length + " of " +
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
        '<button class="go" data-preset="' + esc(p.id) + '"' +
        (p.blocked || busy ? " disabled" : "") + ">" +
        (busy ? "a battery is running" : "run this") + "</button></div>";
    }).join("");

    var names = Object.keys(d.tools || {});
    el("tools").innerHTML = "<thead><tr><th>tool</th><th>state</th>" +
      "<th>arguments a battery would send</th></tr></thead><tbody>" +
      names.map(function (name) {
        var t = d.tools[name];
        return "<tr><td>" + esc(name) + '</td><td class="' +
          (t.ready ? "ready" : "notready") + '">' +
          (t.ready ? "ready" : "needs " + esc(t.missing.join(", "))) +
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
    el("stop").textContent = busy ? "stop" : "finished";
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
      el("hint").textContent = "";
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

  tab("results");
})();
</script>
</body>
</html>
"""
