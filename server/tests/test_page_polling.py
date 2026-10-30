"""The two pages that poll this server ask again only once they have been answered.

`/` polled `/status` on `setInterval(poll, 2000)`, and the admin panel re-armed
its timer whether or not the last fetch had come back. Against a server that
answers slowly -- which is exactly when somebody opens the page to see why --
each tab stacked one more request every two seconds behind the ones already
waiting, and made the server slower still.

Each page's own script is run in node against a DOM that answers everything
and does nothing, a `fetch` that never settles, and fake timers advanced 20
seconds. Skipped when node is not installed: nothing else here needs it.
"""

import json
import re
import shutil
import subprocess

import pytest

from wire import debug_page, status_page

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

# A DOM in which every name exists and every call returns another such
# object, so a page script runs top to bottom without a browser; `fetch`
# never settles, or fails at once; the clock is advanced by hand. Prints how
# many fetches of `watched` were started.
_HARNESS = r"""
const vm = require("vm");
const fs = require("fs");
const [source, watched, seconds, answer] = [fs.readFileSync(process.argv[2], "utf8"),
                                           process.argv[3], Number(process.argv[4]), process.argv[5]];
function stub() {
  return new Proxy(function () {}, {
    get(target, key) {
      if (key === Symbol.toPrimitive) { return () => ""; }
      if (key === "then" || typeof key === "symbol") { return undefined; }
      return stub();
    },
    set() { return true; },
    apply() { return stub(); },
    construct() { return stub(); },
  });
}
let clock = 0, started = 0;
const timers = new Map();
let next = 1;
function arm(fn, ms, every) {
  const id = next++;
  timers.set(id, { at: clock + Math.max(0, Number(ms) || 0), fn, every });
  return id;
}
const own = {
  fetch(url) {
    if (String(url).indexOf(watched) !== -1) { started += 1; }
    if (answer === "refused") { return Promise.reject(new Error("refused")); }
    return new Promise(function () {});
  },
  setTimeout: (fn, ms) => arm(fn, ms, null),
  setInterval: (fn, ms) => arm(fn, ms, Math.max(1, Number(ms) || 0)),
  clearTimeout: (id) => timers.delete(id),
  clearInterval: (id) => timers.delete(id),
  localStorage: { getItem: () => "a-token", setItem() {}, removeItem() {} },
};
const scope = new Proxy({}, {
  has: () => true,
  get(target, key) {
    if (key === Symbol.unscopables) { return undefined; }
    if (key in own) { return own[key]; }
    if (key === "window" || key === "self" || key === "globalThis") { return scope; }
    if (key in globalThis) { return globalThis[key]; }
    return stub();
  },
  set(target, key, value) { own[key] = value; return true; },
});
const context = vm.createContext({ scope });
vm.runInContext("with (scope) {\n" + source + "\n}", context);
(async function () {
  const flush = () => new Promise((resolve) => setImmediate(resolve));
  await flush();
  const end = seconds * 1000;
  for (;;) {
    let due = null;
    for (const [id, timer] of timers) {
      if (timer.at <= end && (due === null || timer.at < timers.get(due).at)) { due = id; }
    }
    if (due === null) { break; }
    const timer = timers.get(due);
    clock = timer.at;
    if (timer.every) { timer.at += timer.every; } else { timers.delete(due); }
    timer.fn();
    await flush();
  }
  process.stdout.write(JSON.stringify({ started }));
})();
"""


def _scripts(page: str) -> str:
    return "\n;\n".join(re.findall(r"<script>(.*?)</script>", page, flags=re.S))


def _fetches_started(tmp_path, page: str, watched: str, answer: str = "never",
                     seconds: int = 20) -> int:
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS)
    script = tmp_path / "page.js"
    script.write_text(_scripts(page))
    # A guard, not a measurement: a page script that loops for ever must fail
    # this test rather than hang the suite.
    finished = subprocess.run([NODE, str(harness), str(script), watched, str(seconds), answer],
                              capture_output=True, text=True, timeout=30)
    assert finished.returncode == 0, finished.stderr
    return json.loads(finished.stdout)["started"]


def test_the_status_page_waits_for_its_answer_before_asking_again(tmp_path):
    started = _fetches_started(tmp_path, status_page.STATUS_PAGE, "status")
    assert started <= 2


def test_the_admin_panel_waits_for_its_answer_before_asking_again(tmp_path):
    started = _fetches_started(tmp_path, debug_page.DEBUG_PAGE, "admin-panel.json")
    assert started <= 2


@pytest.mark.parametrize("page, watched", [
    (status_page.STATUS_PAGE, "status"),
    (debug_page.DEBUG_PAGE, "admin-panel.json"),
], ids=["status page", "admin panel"])
def test_a_page_answered_at_once_keeps_polling(tmp_path, page, watched):
    """Waiting for the answer must not stop the polling: a refused fetch has
    settled, and the next one goes two seconds later."""
    started = _fetches_started(tmp_path, page, watched, answer="refused")
    assert started >= 10
