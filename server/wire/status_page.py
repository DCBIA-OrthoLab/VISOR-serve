"""The page `GET /` serves: what this server is doing, in a browser.

`GET /status` answers JSON, and `benchmarks/live.py` renders it in a terminal.
Neither helps someone who opens the server's address in a browser -- which is
the first thing anybody does -- and who until now got a bare 404 from a server
that was running perfectly.

**Self-contained on purpose.** No CDN, no font, no framework: a server holding
confidential imaging is deployed on networks that do not reach the internet,
and a page whose stylesheet 404s is worse than no page. Everything below is one
file with no request of its own except to this server.

**It holds no token.** The shell is served to anyone who asks, because it says
nothing -- the numbers live behind `GET /status`, which is Bearer-protected.
The page asks for the token, keeps it in `localStorage` so a reload does not
ask again, and sends it as a header. It is never put in the URL, where it would
land in proxy logs and browser history.

And it shows no progress MESSAGE, for the same reason `GET /status` carries
none: a message is free text a tool wrote and can name a patient's file.
"""

from __future__ import annotations

STATUS_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VISOR</title>
<style>
  :root {
    --bg: #fbfbfa; --panel: #ffffff; --line: #e4e2dd; --ink: #1c1b19;
    --soft: #6f6b63; --bar: #d9d6cf; --fill: #3d6b8e; --warn: #9c5b2a;
    --ok: #4a7c59;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #17181a; --panel: #1e2023; --line: #2e3135; --ink: #e8e6e3;
      --soft: #93908a; --bar: #2e3135; --fill: #6ea8d4; --warn: #d89a63;
      --ok: #7fb08c;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  main { max-width: 900px; margin: 0 auto; padding: 28px 20px 60px; }
  h1 { font-size: 15px; font-weight: 600; letter-spacing: .06em;
       text-transform: uppercase; margin: 0 0 2px; }
  .sub { color: var(--soft); font-size: 12.5px; margin-bottom: 22px; }
  section { background: var(--panel); border: 1px solid var(--line);
            border-radius: 8px; padding: 16px 18px; margin-bottom: 14px; }
  h2 { font-size: 11px; font-weight: 600; letter-spacing: .09em;
       text-transform: uppercase; color: var(--soft); margin: 0 0 12px; }
  .row { display: grid; grid-template-columns: 74px 1fr 128px;
         gap: 10px; align-items: center; margin-bottom: 7px; }
  .k { color: var(--soft); font-size: 12px; }
  .track { height: 7px; background: var(--bar); border-radius: 4px; overflow: hidden; }
  .fill { height: 100%; background: var(--fill); border-radius: 4px;
          transition: width .4s ease; }
  .v { text-align: right; font-variant-numeric: tabular-nums; font-size: 12.5px;
       color: var(--soft); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; font-weight: 500; color: var(--soft); font-size: 11px;
       letter-spacing: .06em; text-transform: uppercase;
       padding: 0 8px 8px 0; }
  td { padding: 5px 8px 5px 0; border-top: 1px solid var(--line);
       font-variant-numeric: tabular-nums; }
  td.n { text-align: right; }
  .tag { font-size: 11px; padding: 1px 7px; border-radius: 10px;
         border: 1px solid var(--line); color: var(--soft); }
  .tag.run { color: var(--ok); border-color: var(--ok); }
  .tag.wait { color: var(--warn); border-color: var(--warn); }
  .varies { color: var(--warn); font-size: 11.5px; }
  .empty { color: var(--soft); font-size: 12.5px; padding: 6px 0; }
  a { color: var(--fill); }
  .gain { color: var(--ok); font-weight: 600; }
  .bad { color: var(--warn); }
  .scroll { overflow-x: auto; }
  .arm { color: var(--soft); font-size: 11.5px; }
  .gate { display: flex; gap: 8px; margin-top: 10px; }
  input { flex: 1; padding: 8px 10px; border: 1px solid var(--line);
          border-radius: 6px; background: var(--bg); color: var(--ink);
          font: inherit; }
  button { padding: 8px 14px; border: 1px solid var(--line); border-radius: 6px;
           background: var(--fill); color: #fff; font: inherit; cursor: pointer; }
  .err { color: var(--warn); font-size: 12.5px; }
  .nest { color: var(--soft); }
</style>
</head>
<body>
<main>
  <h1>VISOR</h1>
  <div class="sub" id="when">connecting&hellip;</div>

  <section id="gate" hidden>
    <h2>API token</h2>
    <div class="empty">This server's numbers are Bearer-protected. The token is
      kept in this browser only and sent as a header, never in the URL.</div>
    <div class="gate">
      <input id="token" type="password" placeholder="API token" autocomplete="off">
      <button id="save">Connect</button>
    </div>
    <div class="err" id="gateerr"></div>
  </section>

  <section id="budget" hidden>
    <h2>What is being spent</h2>
    <div id="bars"></div>
    <div class="empty" id="seats"></div>
  </section>

  <section id="runs" hidden>
    <h2>Runs</h2>
    <div id="runbody"></div>
  </section>

  <section id="bench" hidden>
    <h2>What this deployment was measured to do</h2>
    <div class="empty" id="benchwhen"></div>
    <div id="benchbody"></div>
  </section>

  <section id="cover" hidden>
    <h2>Every tool, run once</h2>
    <div id="coverbody"></div>
  </section>

  <section id="costs" hidden>
    <h2>What each tool was measured to need</h2>
    <div id="costbody"></div>
  </section>
</main>
<script>
(function () {
  "use strict";
  var KEY = "visor.token";
  var token = "";
  try { token = window.localStorage.getItem(KEY) || ""; } catch (e) { token = ""; }

  function el(id) { return document.getElementById(id); }
  function show(id, on) { el(id).hidden = !on; }
  function gib(n) { return (n == null) ? "?" : (n / 1073741824).toFixed(1) + "G"; }

  function escaped(text) {
    // The only place this page builds markup from data. Both values come from
    // the server's own campaign file rather than from a client, but building
    // HTML from anything unescaped is a habit worth not having.
    var box = document.createElement("div");
    box.textContent = String(text == null ? "" : text);
    return box.innerHTML;
  }

  function ago(seconds) {
    var s = Math.max(0, Math.round(Date.now() / 1000 - seconds));
    if (s < 60) { return s + "s"; }
    if (s < 3600) { return Math.floor(s / 60) + "m" + ("0" + (s % 60)).slice(-2); }
    return Math.floor(s / 3600) + "h" + ("0" + Math.floor((s % 3600) / 60)).slice(-2);
  }

  function bar(label, used, total, note) {
    var pct = total ? Math.min(100, (used / total) * 100) : 0;
    return '<div class="row"><div class="k">' + label + '</div>' +
      '<div class="track"><div class="fill" style="width:' + pct.toFixed(1) + '%"></div></div>' +
      '<div class="v">' + note + "</div></div>";
  }

  function draw(d) {
    var b = d.budget, a = d.admission, card = d.card || {};
    el("bars").innerHTML =
      bar("cpus", a.cpus_held, b.cpus, a.cpus_held.toFixed(1) + " / " + b.cpus.toFixed(1)) +
      bar("ram", a.ram_held, b.ram_bytes, gib(a.ram_held) + " / " + gib(b.ram_bytes)) +
      bar("vram", a.vram_held, b.vram_bytes, gib(a.vram_held) + " / " + gib(b.vram_bytes)) +
      (card.total_bytes
        ? bar("card", card.total_bytes - (card.free_bytes || 0), card.total_bytes,
              gib(card.total_bytes - (card.free_bytes || 0)) + " / " + gib(card.total_bytes))
        : "");
    el("seats").textContent =
      a.running + " running, " + a.waiting + " waiting \\u2014 the budget allows " +
      b.max_parallel_jobs + " at once for " + b.expected_clients + " declared client(s)";

    var runs = d.runs || [];
    if (!runs.length) {
      el("runbody").innerHTML = '<div class="empty">Nothing on the server.</div>';
    } else {
      var rows = runs.slice(0, 40).map(function (r) {
        var cls = r.state === "running" ? "run" : (r.phase === "queued_gpu" ? "wait" : "");
        var indent = r.depth ? '<span class="nest">' +
          new Array(r.depth + 1).join("\\u2007\\u2007\\u2514 ") + "</span>" : "";
        return "<tr><td>" + indent + (r.tool || "?") +
          '</td><td><span class="tag ' + cls + '">' + r.phase + "</span></td>" +
          '<td class="n">' + (r.fraction == null ? "" : Math.round(r.fraction * 100) + "%") +
          '</td><td class="n">' + ago(r.started_at) + "</td></tr>";
      }).join("");
      el("runbody").innerHTML =
        "<table><thead><tr><th>tool</th><th>phase</th><th></th><th></th></tr></thead>" +
        "<tbody>" + rows + "</tbody></table>";
    }

    var costs = d.costs || {};
    var names = Object.keys(costs).sort(function (x, y) {
      return costs[y].vram_bytes - costs[x].vram_bytes;
    });
    el("costbody").innerHTML = names.length
      ? "<table><thead><tr><th>tool</th><th>vram</th><th>ram</th><th>runs</th></tr></thead><tbody>" +
        names.map(function (n) {
          var c = costs[n];
          return "<tr><td>" + n + (c.input_dependent
            ? ' <span class="varies">memory varies \\u00d7' + c.vram_spread.toFixed(1) + "</span>"
            : "") + '</td><td class="n">' + gib(c.vram_bytes) +
            '</td><td class="n">' + gib(c.ram_bytes) +
            '</td><td class="n">' + c.samples + "</td></tr>";
        }).join("") + "</tbody></table>"
      : '<div class="empty">Nothing measured yet \\u2014 every tool runs alone until it has ' +
        "been measured once.</div>";

    ["budget", "runs", "costs"].forEach(function (id) { show(id, true); });
    show("gate", false);
    el("when").textContent = "updated " + new Date().toLocaleTimeString();
  }

  function seconds(v) { return v == null ? "" : v.toFixed(0) + "s"; }

  function drawCampaign(c) {
    if (!c) {
      show("bench", false);
      show("cover", false);
      return;
    }
    // Linked from here rather than from the header: this line already
    // identifies the campaign the new page expands, and `drawCampaign(null)`
    // hides this whole section -- so the link cannot appear on a deployment
    // with nothing to draw. Relative, so it resolves behind a proxy prefix.
    el("benchwhen").innerHTML =
      escaped(c.source) + " — " + escaped(c.hardware || "") +
      ' · <a href="benchmarks/view">see every run drawn</a>';

    el("benchbody").innerHTML = '<div class="scroll"><table><thead><tr>' +
      "<th>arm</th><th>load</th><th>clients</th><th>ok</th><th>wall</th>" +
      "<th>median</th><th>slowest</th><th>at once</th><th>speedup</th><th>peak vram</th>" +
      "</tr></thead><tbody>" +
      (c.arms || []).map(function (a) {
        var load = (a.tools || []).join("/") +
          (a.transfer === "upload" ? " (upload)" : "") +
          (a.detached === false ? " (blocking)" : "");
        return '<tr><td class="arm">' + a.arm + "</td><td>" + load +
          '</td><td class="n">' + a.clients +
          '</td><td class="n' + (a.ok < a.total ? " bad" : "") + '">' +
          a.ok + "/" + a.total +
          '</td><td class="n">' + seconds(a.wall_seconds) +
          '</td><td class="n">' + seconds(a.median_seconds) +
          '</td><td class="n">' + seconds(a.slowest_seconds) +
          '</td><td class="n">' + a.peak_running + " / " + a.mean_running.toFixed(1) +
          '</td><td class="n gain">' + (a.speedup ? a.speedup.toFixed(1) + "×" : "—") +
          '</td><td class="n">' + gib(a.peak_vram_bytes) + "</td></tr>";
      }).join("") + "</tbody></table></div>";
    show("bench", true);

    var cov = c.coverage || [];
    el("coverbody").innerHTML = cov.length
      ? '<div class="scroll"><table><thead><tr><th>tool</th><th>result</th>' +
        "<th>seconds</th><th>vram</th><th>ram</th></tr></thead><tbody>" +
        cov.map(function (t) {
          var ok = t.status === "ok";
          return "<tr><td>" + t.tool + '</td><td><span class="tag ' +
            (ok ? "run" : "wait") + '">' + t.status + "</span>" +
            (t.error ? ' <span class="bad">' + t.error.slice(0, 90) + "</span>" : "") +
            '</td><td class="n">' + seconds(t.seconds) +
            '</td><td class="n">' + (t.vram_bytes == null ? "" : gib(t.vram_bytes)) +
            '</td><td class="n">' + (t.ram_bytes == null ? "" : gib(t.ram_bytes)) +
            "</td></tr>";
        }).join("") + "</tbody></table></div>" +
        ((c.out_of_scope || []).length
          ? '<div class="empty">Not run, deliberately: ' +
            c.out_of_scope.join(", ") + ".</div>"
          : "")
      : "";
    show("cover", cov.length > 0);
  }

  function pollCampaign() {
    if (!token) { return; }
    // Its own request, and a failure here never disturbs the live view: a
    // deployment that ran no campaign is the normal case, not a broken one.
    fetch("benchmarks", { headers: { Authorization: "Bearer " + token } })
      .then(function (r) { return r.ok ? r.json() : { campaign: null }; })
      .then(function (d) { drawCampaign(d.campaign); })
      .catch(function () { drawCampaign(null); });
  }

  function poll() {
    if (!token) { show("gate", true); el("when").textContent = "waiting for a token"; return; }
    fetch("status", { headers: { Authorization: "Bearer " + token } })
      .then(function (r) {
        if (r.status === 401) { throw new Error("That token was refused."); }
        if (!r.ok) { throw new Error("The server answered " + r.status + "."); }
        return r.json();
      })
      .then(draw)
      .catch(function (e) {
        show("gate", true);
        el("gateerr").textContent = e.message;
        el("when").textContent = "not connected";
      });
  }

  el("save").addEventListener("click", function () {
    token = el("token").value.trim();
    try { window.localStorage.setItem(KEY, token); } catch (e) { /* private window */ }
    el("gateerr").textContent = "";
    poll();
    pollCampaign();
  });
  el("token").addEventListener("keydown", function (e) {
    if (e.key === "Enter") { el("save").click(); }
  });

  poll();
  pollCampaign();
  window.setInterval(poll, 2000);
  // A campaign does not change while anyone is watching; re-read it rarely, so
  // the page is not asking for a few kilobytes of history every two seconds.
  window.setInterval(pollCampaign, 60000);
}());
</script>
</body>
</html>
"""
