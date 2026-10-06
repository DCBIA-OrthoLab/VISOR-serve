"""The stylesheet every operator page shares: glass over a softly lit ground.

The admin panel set the look; the benchmark pages sit one click away from it
and read as a different product when they carry a palette of their own. So the
part that makes the look -- the colour tokens in three theme states, the lit
ground, the scrollbars, the controls -- is written once here and concatenated
into each page's `<style>`. What is particular to a page stays in that page.

Plain CSS with no backslash and no URL: it lands inside raw and non-raw Python
strings alike, and in pages that are tested for asking nothing of the network.
"""

from __future__ import annotations

GLASS_BASE = """
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
    --menu: #ffffff;
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
      --menu: #1f232b;
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
    --menu: #1f232b;
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
  .mono { font-variant-numeric: tabular-nums; }
  button, input, select { font: inherit; color: inherit; }
  button { background: var(--sunk); border: 1px solid var(--line); border-radius: 8px;
           padding: 6px 12px; cursor: pointer; transition: border-color .15s, background .15s; }
  button:hover { border-color: var(--soft); }
  button.ghost { border-color: transparent; background: transparent; }
  button.ghost:hover { background: var(--sunk); border-color: var(--line); }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
  button.primary:disabled { opacity: .45; cursor: default; }
  input { background: var(--sunk); border: 1px solid var(--line); border-radius: 9px;
          padding: 7px 11px; min-width: 220px; outline: none; }
  input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
  /* Drop-down menus. The control is glass like every other one: frosted,
     its chevron drawn in CSS (an icon file would be a request, and these
     pages make none). The list it opens is drawn by GLASS_SELECT_JS. */
  select { -webkit-appearance: none; appearance: none; cursor: pointer;
           color: var(--ink); background-color: var(--sunk);
           background-image: linear-gradient(45deg, transparent 50%, var(--soft) 50%),
                             linear-gradient(135deg, var(--soft) 50%, transparent 50%);
           background-position: calc(100% - 15px) 52%, calc(100% - 10px) 52%;
           background-size: 5px 5px, 5px 5px; background-repeat: no-repeat;
           border: 1px solid var(--line); border-radius: 9px; padding: 5px 30px 5px 10px;
           -webkit-backdrop-filter: var(--blur); backdrop-filter: var(--blur);
           box-shadow: 0 1px 2px rgba(0,0,0,.05), inset 0 1px 0 var(--panel-edge);
           transition: border-color .15s, box-shadow .15s; outline: none; }
  select:hover { border-color: var(--soft); }
  select:focus-visible, select.gopen { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
  select:disabled { opacity: .5; cursor: default; }
  .gmenu { position: fixed; z-index: 100; min-width: 160px; max-height: 340px; overflow: auto;
           padding: 5px; border-radius: 12px; background: var(--panel-strong);
           -webkit-backdrop-filter: var(--blur); backdrop-filter: var(--blur);
           border: 1px solid var(--panel-edge); box-shadow: var(--shadow-lg);
           font-size: 13px; opacity: 0; transform: translateY(-4px) scale(.98);
           transition: opacity .12s ease, transform .12s ease; }
  .gmenu.open { opacity: 1; transform: none; }
  .gmenu .ggrp { padding: 7px 10px 3px; font-size: 10.5px; letter-spacing: .06em; text-transform: uppercase;
                 color: var(--ghost); font-weight: 650; }
  .gmenu .gopt { display: flex; align-items: center; gap: 8px; padding: 6px 10px 6px 8px; border-radius: 7px;
                 cursor: pointer; color: var(--ink); white-space: nowrap; }
  .gmenu .gopt i { width: 14px; flex: none; font-style: normal; color: var(--accent); font-weight: 700; text-align: center; }
  .gmenu .gopt.on { color: var(--accent); font-weight: 600; }
  .gmenu .gopt.act { background: var(--accent-soft); }
  .gmenu .gopt.off { opacity: .45; cursor: default; }
  .navlink { color: var(--accent); text-decoration: none; font-size: 13px; font-weight: 600;
             padding: 6px 10px; border-radius: 8px; white-space: nowrap; }
  .navlink:hover { background: var(--accent-soft); }
"""


# The list a drop-down opens, drawn as glass. A native <select> paints its open
# list with the system's toolkit, which no stylesheet reaches -- so the page
# draws it instead, and the <select> stays the control: choosing sets its value
# and fires the same `change` event, so every handler written against a plain
# select keeps working. A select a redraw replaced while its list was open is
# found again by its id or its data attributes.
#
# No backslash anywhere, on purpose: it is concatenated into raw and non-raw
# Python strings alike.
GLASS_SELECT_JS = """<script>
(function () {
  "use strict";
  var menu = null, target = null, items = [], active = -1, typed = "", typedAt = 0;

  function key(node) {
    if (node.id) { return "#" + node.id; }
    var parts = [];
    Array.prototype.forEach.call(node.attributes, function (a) {
      if (a.name.indexOf("data-") === 0) { parts.push("[" + a.name + '="' + a.value.split('"').join("") + '"]'); }
    });
    return parts.length ? "select" + parts.join("") : null;
  }
  function live() {
    if (target && document.contains(target)) { return target; }
    var selector = target && target.getAttribute("data-gkey");
    var found = selector ? document.querySelector(selector) : null;
    return found || null;
  }
  function close() {
    if (!menu) { return; }
    var node = live();
    if (node) { node.classList.remove("gopen"); }
    menu.remove();
    menu = null; target = null; items = []; active = -1;
  }
  function mark(index) {
    items.forEach(function (it, i) { it.el.classList.toggle("act", i === index); });
    active = index;
    if (items[index]) { items[index].el.scrollIntoView({ block: "nearest" }); }
  }
  function choose(index) {
    var it = items[index], node = live();
    close();
    if (!it || it.disabled || !node) { return; }
    if (node.value !== it.value) {
      node.value = it.value;
      node.dispatchEvent(new Event("change", { bubbles: true }));
    }
    node.focus({ preventScroll: true });
  }
  function open(select) {
    close();
    target = select;
    var k = key(select);
    if (k) { select.setAttribute("data-gkey", k); }
    menu = document.createElement("div");
    menu.className = "gmenu";
    menu.setAttribute("role", "listbox");
    function add(option) {
      var row = document.createElement("div");
      row.className = "gopt" + (option.selected ? " on" : "") + (option.disabled ? " off" : "");
      row.setAttribute("role", "option");
      var tick = document.createElement("i");
      tick.textContent = option.selected ? "✓" : "";
      var text = document.createElement("span");
      text.textContent = option.textContent;
      row.appendChild(tick); row.appendChild(text);
      var index = items.length;
      row.addEventListener("mouseenter", function () { mark(index); });
      row.addEventListener("mousedown", function (e) { e.preventDefault(); choose(index); });
      items.push({ el: row, value: option.value, disabled: option.disabled, text: option.textContent.toLowerCase() });
      menu.appendChild(row);
    }
    Array.prototype.forEach.call(select.children, function (child) {
      if (child.tagName === "OPTGROUP") {
        var head = document.createElement("div");
        head.className = "ggrp";
        head.textContent = child.label;
        menu.appendChild(head);
        Array.prototype.forEach.call(child.children, add);
      } else if (child.tagName === "OPTION") {
        add(child);
      }
    });
    document.body.appendChild(menu);
    var r = select.getBoundingClientRect();
    menu.style.minWidth = Math.max(160, r.width) + "px";
    var below = window.innerHeight - r.bottom - 8, above = r.top - 8;
    var h = Math.min(menu.scrollHeight, 340);
    menu.style.left = Math.max(8, Math.min(r.left, window.innerWidth - menu.offsetWidth - 8)) + "px";
    if (below >= h || below >= above) {
      menu.style.top = (r.bottom + 4) + "px";
      menu.style.maxHeight = Math.max(120, Math.min(340, below)) + "px";
    } else {
      menu.style.top = Math.max(8, r.top - 4 - Math.min(h, above)) + "px";
      menu.style.maxHeight = Math.max(120, Math.min(340, above)) + "px";
    }
    select.classList.add("gopen");
    mark(Math.max(0, select.selectedIndex >= 0 ? items.findIndex(function (it) { return it.value === select.value; }) : 0));
    window.requestAnimationFrame(function () { if (menu) { menu.classList.add("open"); } });
  }
  function usable(node) {
    return node && node.tagName === "SELECT" && !node.disabled && !node.multiple && !(node.size > 1);
  }

  document.addEventListener("mousedown", function (e) {
    if (menu && menu.contains(e.target)) { return; }
    if (usable(e.target)) {
      e.preventDefault();
      var again = e.target === target;
      close();
      if (!again) { e.target.focus({ preventScroll: true }); open(e.target); }
      return;
    }
    close();
  }, true);
  document.addEventListener("keydown", function (e) {
    if (menu) {
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); var n = live(); close(); if (n) { n.focus(); } return; }
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        var step = e.key === "ArrowDown" ? 1 : -1, i = active;
        do { i += step; } while (items[i] && items[i].disabled);
        if (items[i]) { mark(i); }
        return;
      }
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(active); return; }
      if (e.key === "Tab") { close(); return; }
      if (e.key.length === 1) {
        typed = (Date.now() - typedAt < 700 ? typed : "") + e.key.toLowerCase();
        typedAt = Date.now();
        var hit = items.findIndex(function (it) { return !it.disabled && it.text.indexOf(typed) === 0; });
        if (hit >= 0) { mark(hit); }
      }
      return;
    }
    if (usable(e.target) && (e.key === "Enter" || e.key === " " || (e.key === "ArrowDown" && e.altKey))) {
      e.preventDefault();
      open(e.target);
    }
  }, true);
  window.addEventListener("resize", close);
  document.addEventListener("scroll", function (e) {
    if (menu && !menu.contains(e.target)) { close(); }
  }, true);
})();
</script>"""
