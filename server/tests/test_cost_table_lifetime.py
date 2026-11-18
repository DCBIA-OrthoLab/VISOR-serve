"""How long the learned cost table lives, and the only two things that end it.

The table is the one piece of server state nothing can regenerate: it is what
every run this deployment ever did measured itself to need. Losing it puts every
tool back to "unmeasured", and an unmeasured tool reserves the whole budget and
runs ALONE -- so a table that forgot itself on every update made each update a
silent loss of all the parallelism the server had learned.

So it outlives a restart, a recreated container, a server update and a tools
update. It is dropped for exactly two reasons, both of them written down by
whoever released the change:

- a tool's `[tool.sadt] cost_epoch` moved, which drops that tool's entry alone;
- the server's `costs.TABLE_EPOCH` moved, which drops the whole table once.
"""

import json
import logging
import os
import posixpath
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

os.environ.setdefault("API_TOKEN", "test-token")

import config
from config import settings
from execution import costs, runner

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SERVER_DIR)
COMPOSE = os.path.join(REPO_ROOT, "docker-compose.yml")
# Where the compose services mount the checkout's server/, and therefore what
# config.py's `_SERVER_DIR` is inside them.
CONTAINER_SERVER_DIR = "/workspace/server"


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    # COST_TABLE_PATH is pinned to "" by conftest, so the table follows this.
    monkeypatch.setattr(config.settings, "SCHEMA_CACHE_DIR", str(tmp_path / "schema-cache"))
    return tmp_path


def _raw() -> dict:
    with open(costs.table_path(), encoding="utf-8") as handle:
        return json.load(handle)


def _figures(name: str):
    cost = costs.cost_of(name)
    return None if cost is None else (cost.vram_bytes, cost.ram_bytes, cost.samples)


# ----------------------------------------------------------------------
# Where the production services keep it
# ----------------------------------------------------------------------

def _services() -> dict:
    yaml = pytest.importorskip("yaml")
    with open(COMPOSE, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _environment(service: dict) -> dict:
    declared = service.get("environment") or {}
    if isinstance(declared, list):
        declared = dict(item.split("=", 1) for item in declared)
    return declared


def _effective_table(environment: dict) -> str:
    """The table a service's server reads, by the rule `costs.table_path` applies:
    its own setting when set, beside the schema cache otherwise."""
    explicit = environment.get("COST_TABLE_PATH")
    if explicit:
        return explicit
    cache = environment.get("SCHEMA_CACHE_DIR") or posixpath.join(CONTAINER_SERVER_DIR, ".schema-cache")
    return posixpath.join(cache, "tool_costs.json")


# `source:target[:mode]`, for the mounts that name a plain path. The ones built
# from `${VAR:-default}` are the read-only tool mounts and never match.
_MOUNT = re.compile(r"^(?P<source>[^:$]+):(?P<target>/[^:]*)(?::(?P<mode>\w+))?$")


def _mount_holding(service: dict, path: str):
    """The mount `path` lies on, the deepest one when several nest."""
    best = None
    for volume in service.get("volumes") or ():
        found = _MOUNT.match(str(volume))
        if not found:
            continue
        target = found.group("target").rstrip("/")
        if path == target or path.startswith(target + "/"):
            if best is None or len(target) > len(best.group("target")):
                best = found
    return best


@pytest.mark.skipif(not os.path.isfile(COMPOSE), reason="docker-compose.yml is not beside server/")
def test_the_production_services_keep_the_table_where_a_recreate_cannot_reach():
    """`/tmp` inside the container is recreated with it, and the update agent
    recreates the container on every update that touches the compose file, the
    API's requirements or the image. The table has to sit on a mount that
    outlives the container, writable, and -- when that mount is the checkout --
    ignored by git, or the clone reads as dirty and `server_ctl up` refuses to
    fast-forward it."""
    compose = _services()
    services = compose["services"]
    named = set((compose.get("volumes") or {}).keys())
    for name in ("inference", "inference-cpu"):
        table = _effective_table(_environment(services[name]))
        assert not table.startswith("/tmp/"), f"{name} keeps its cost table in the container's /tmp"
        mount = _mount_holding(services[name], posixpath.dirname(table))
        assert mount is not None, f"{name}: {table} is on no mount, so it dies with the container"
        assert mount.group("mode") != "ro", f"{name}: {table} is on a read-only mount"
        source = mount.group("source")
        assert source.startswith("./") or source in named, f"{name}: {source} is not persistent"
        if source.startswith("./") and shutil.which("git") and os.path.exists(os.path.join(REPO_ROOT, ".git")):
            on_host = os.path.join(REPO_ROOT, source[2:],
                                   posixpath.relpath(table, mount.group("target")))
            ignored = subprocess.run(["git", "-C", REPO_ROOT, "check-ignore", "-q", "--no-index",
                                      os.path.relpath(on_host, REPO_ROOT)])
            assert ignored.returncode == 0, f"{name}: {on_host} is not ignored by git"


@pytest.mark.skipif(not os.path.isfile(COMPOSE), reason="docker-compose.yml is not beside server/")
def test_the_test_services_never_write_the_production_table():
    """They mount the same checkout read-write; a suite run must not teach the
    deployment's table what a stubbed tool cost."""
    services = _services()["services"]
    production = _effective_table(_environment(services["inference"]))
    for name in ("test", "test-gpu"):
        assert "COST_TABLE_PATH" not in _environment(services[name])
        assert _effective_table(_environment(services[name])) != production


# ----------------------------------------------------------------------
# Its own setting
# ----------------------------------------------------------------------

def test_the_table_follows_its_own_setting_and_defaults_beside_the_schema_cache(tmp_path, monkeypatch):
    """Unset, it stays where every existing deployment already has it -- the
    venvs image's cache volume, a checkout's `.schema-cache` -- so nothing moves
    for them. Set, it is independent of a schema cache that may be disposable."""
    assert costs.table_path() == os.path.join(settings.SCHEMA_CACHE_DIR, "tool_costs.json")
    elsewhere = tmp_path / "kept" / "nested" / "tool_costs.json"
    monkeypatch.setattr(settings, "COST_TABLE_PATH", str(elsewhere))
    assert costs.table_path() == str(elsewhere)
    costs.record("AMASSS", 2 << 30, 6 << 30)
    assert elsewhere.is_file()
    assert not os.path.exists(os.path.join(settings.SCHEMA_CACHE_DIR, "tool_costs.json"))
    assert _figures("AMASSS") == (2 << 30, 6 << 30, 1)


def test_a_tool_process_reads_the_table_the_server_writes(tmp_path, monkeypatch):
    """The nested runner reads the file by path; it must be the same file."""
    from execution import concurrency, dispatch

    monkeypatch.setattr(settings, "COST_TABLE_PATH", str(tmp_path / "kept" / "tool_costs.json"))
    environment = dispatch._child_environment("job-1", str(tmp_path / "job-1"))
    assert environment[concurrency.COST_TABLE_ENV] == settings.COST_TABLE_PATH


def test_a_table_that_cannot_be_written_says_so_once(monkeypatch, caplog):
    """Still never a failed run -- but a table that silently stops persisting is
    exactly the defect this file exists for, so it is said, once."""
    monkeypatch.setattr(settings, "COST_TABLE_PATH", "/proc/nonexistent/costs/tool_costs.json")
    monkeypatch.setattr(costs, "_unwritable_said", False, raising=False)
    with caplog.at_level(logging.WARNING, logger="inference_server"):
        costs.record("AMASSS", 1, 1)
        costs.record("AMASSS", 2, 2)
    said = [r.getMessage() for r in caplog.records
            if r.levelno >= logging.WARNING and "/proc/nonexistent/costs/tool_costs.json" in r.getMessage()]
    assert len(said) == 1


# ----------------------------------------------------------------------
# A restart, a recreated container
# ----------------------------------------------------------------------

_WRITE = """
from execution import costs
costs.record("AMASSS", 2 << 30, 6 << 30, channels=1)
costs.record("AMASSS", 3 << 30, 7 << 30, channels=3)
costs.record("ALI_CBCT", 600 << 20, 1 << 30)
"""

_READ = """
import json
from execution import costs
print(json.dumps({name: [cost.vram_bytes, cost.vram_marginal, cost.ram_bytes,
                         cost.samples, cost.widths]
                  for name, cost in costs.known().items()}))
"""


def _new_process(code: str, environment: dict) -> str:
    done = subprocess.run([sys.executable, "-c", code], cwd=SERVER_DIR, env=environment,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout


def _container(tmp_path, name: str) -> dict:
    """A server process as the compose service starts one: a /tmp of its own,
    gone with the container, and the table on the persistent mount."""
    environment = {key: value for key, value in os.environ.items()
                   if key not in ("COST_TABLE_PATH", "SCHEMA_CACHE_DIR")}
    environment.update(
        API_TOKEN="test-token",
        PYTHONDONTWRITEBYTECODE="1",
        SCHEMA_CACHE_DIR=str(tmp_path / name / "tmp" / "schema-cache"),
        COST_TABLE_PATH=str(tmp_path / "checkout" / "server" / ".costs" / "tool_costs.json"),
    )
    return environment


def test_a_recreated_container_keeps_every_tool_s_figures(tmp_path):
    _new_process(_WRITE, _container(tmp_path, "before-the-update"))
    learned = json.loads(_new_process(_READ, _container(tmp_path, "before-the-update")))
    assert sorted(learned) == ["ALI_CBCT", "AMASSS"]

    after = json.loads(_new_process(_READ, _container(tmp_path, "after-the-update")))

    assert after == learned


# ----------------------------------------------------------------------
# A tool's own epoch: [tool.sadt] cost_epoch
# ----------------------------------------------------------------------

def test_a_tool_update_that_keeps_its_cost_epoch_keeps_everything():
    costs.reconcile({"AMASSS": 1, "ALI_CBCT": 0})
    costs.record("AMASSS", 2 << 30, 6 << 30)
    costs.record("ALI_CBCT", 600 << 20, 1 << 30)
    before = (_figures("AMASSS"), _figures("ALI_CBCT"), _raw())

    # The next start, after a release of both tools that did not touch the key.
    costs.reconcile({"AMASSS": 1, "ALI_CBCT": 0})

    assert (_figures("AMASSS"), _figures("ALI_CBCT"), _raw()) == before
    assert _raw()["AMASSS"][costs.COST_EPOCH_KEY] == 1


def test_a_bumped_cost_epoch_forgets_that_tool_and_no_other(caplog):
    costs.reconcile({"AMASSS": 1, "ALI_CBCT": 0})
    costs.record("AMASSS", 2 << 30, 6 << 30)
    costs.record("ALI_CBCT", 600 << 20, 1 << 30)
    untouched = (_figures("ALI_CBCT"), _raw()["ALI_CBCT"])

    with caplog.at_level(logging.INFO, logger="inference_server"):
        costs.reconcile({"AMASSS": 2, "ALI_CBCT": 0})

    assert costs.cost_of("AMASSS") is None
    assert (_figures("ALI_CBCT"), _raw()["ALI_CBCT"]) == untouched
    # Gone from the FILE, not only from this process's reading of it: the
    # nested runner reads the file itself.
    assert "AMASSS" not in _raw()
    said = [r.getMessage() for r in caplog.records if "AMASSS" in r.getMessage()]
    assert len(said) == 1 and "cost_epoch 1 -> 2" in said[0]

    # It relearns under the new epoch, and the next start keeps that.
    costs.record("AMASSS", 3 << 30, 5 << 30)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="inference_server"):
        costs.reconcile({"AMASSS": 2, "ALI_CBCT": 0})
    assert _figures("AMASSS") == (3 << 30, 5 << 30, 1)
    assert _raw()["AMASSS"][costs.COST_EPOCH_KEY] == 2
    assert not [r for r in caplog.records if "AMASSS" in r.getMessage()]


def test_an_entry_learned_before_epochs_existed_is_epoch_zero(tmp_path):
    """Every table on every deployment today. Kept, not reset, by the release
    that introduces the key."""
    os.makedirs(os.path.dirname(costs.table_path()), exist_ok=True)
    legacy = {
        "AMASSS": {"vram_bytes": 2 << 30, "ram_bytes": 6 << 30, "samples": 4,
                   "recent": [[2 << 30, 6 << 30, 1, 0.0]]},
        "ALI_CBCT": {"vram_bytes": 600 << 20, "ram_bytes": 1 << 30, "samples": 2,
                     "recent": [[600 << 20, 1 << 30, 1, 0.0]]},
    }
    with open(costs.table_path(), "w", encoding="utf-8") as handle:
        json.dump(legacy, handle)

    costs.reconcile({"AMASSS": 0, "ALI_CBCT": 0})

    assert _figures("AMASSS") == (2 << 30, 6 << 30, 4)
    assert _figures("ALI_CBCT") == (600 << 20, 1 << 30, 2)

    # And a first declared epoch is a change from that zero.
    costs.reconcile({"AMASSS": 1, "ALI_CBCT": 0})
    assert costs.cost_of("AMASSS") is None
    assert _figures("ALI_CBCT") == (600 << 20, 1 << 30, 2)


def test_a_tool_that_did_not_load_this_time_keeps_its_entry():
    """Nothing declared means nothing to compare: a tool that failed to start,
    or whose epoch could not be read, is not a tool whose release asked."""
    costs.reconcile({"AMASSS": 3})
    costs.record("AMASSS", 2 << 30, 6 << 30)

    costs.reconcile({})
    assert _figures("AMASSS") == (2 << 30, 6 << 30, 1)
    costs.reconcile({"AMASSS": None})
    assert _figures("AMASSS") == (2 << 30, 6 << 30, 1)
    assert _raw()["AMASSS"][costs.COST_EPOCH_KEY] == 3


def test_startup_hands_every_readable_epoch_to_the_table():
    import main

    tools = {
        "AMASSS": SimpleNamespace(cost_epoch=2),
        "ALI_CBCT": SimpleNamespace(cost_epoch=0),
        # Declared, and not an integer: the registry said so and left it None.
        "CLIC": SimpleNamespace(cost_epoch=None),
    }
    assert main._declared_cost_epochs(tools) == {"AMASSS": 2, "ALI_CBCT": 0}


def test_the_server_start_drops_what_a_tool_s_release_declared_stale(monkeypatch, probe_tool):
    """End to end through the lifespan: the entry is gone before the first
    request can be admitted on it."""
    from fastapi.testclient import TestClient

    import registry
    from main import app

    costs.record(probe_tool.name, 2 << 30, 6 << 30)
    raw = _raw()
    raw[probe_tool.name][costs.COST_EPOCH_KEY] = 0
    with open(costs.table_path(), "w", encoding="utf-8") as handle:
        json.dump(raw, handle)
    monkeypatch.setattr(probe_tool, "cost_epoch", 1, raising=False)
    monkeypatch.setitem(registry.TOOLS, probe_tool.name, probe_tool)

    with TestClient(app):
        pass

    assert costs.cost_of(probe_tool.name) is None
    assert probe_tool.name not in _raw()


# ----------------------------------------------------------------------
# The server's own epoch: costs.TABLE_EPOCH
# ----------------------------------------------------------------------

def test_bumping_the_table_epoch_discards_the_whole_table_once(monkeypatch, caplog):
    costs.reconcile({"AMASSS": 0, "ALI_CBCT": 0})
    costs.record("AMASSS", 2 << 30, 6 << 30)
    costs.record("ALI_CBCT", 600 << 20, 1 << 30)

    monkeypatch.setattr(costs, "TABLE_EPOCH", costs.TABLE_EPOCH + 1)
    # Even before the start-up pass, nothing of the old epoch is admitted on.
    assert costs.cost_of("AMASSS") is None
    with caplog.at_level(logging.INFO, logger="inference_server"):
        costs.reconcile({"AMASSS": 0, "ALI_CBCT": 0})

    assert costs.known() == {}
    said = [r.getMessage() for r in caplog.records if "TABLE_EPOCH" in r.getMessage()]
    assert len(said) == 1
    assert "{} -> {}".format(costs.TABLE_EPOCH - 1, costs.TABLE_EPOCH) in said[0]
    assert _raw() == {costs.TABLE_EPOCH_KEY: costs.TABLE_EPOCH}

    # Once: it then learns normally, and the next start keeps what it learned.
    costs.record("AMASSS", 3 << 30, 5 << 30)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="inference_server"):
        costs.reconcile({"AMASSS": 0, "ALI_CBCT": 0})
    assert _figures("AMASSS") == (3 << 30, 5 << 30, 1)
    assert not [r for r in caplog.records if "TABLE_EPOCH" in r.getMessage()]


def test_a_table_with_no_marker_belongs_to_the_first_epoch():
    os.makedirs(os.path.dirname(costs.table_path()), exist_ok=True)
    with open(costs.table_path(), "w", encoding="utf-8") as handle:
        json.dump({"AMASSS": {"vram_bytes": 5, "ram_bytes": 5, "samples": 1}}, handle)
    costs.reconcile({})
    assert _figures("AMASSS") == (5, 5, 1)


def test_the_epoch_marker_is_never_taken_for_a_tool(monkeypatch):
    from fastapi.testclient import TestClient

    from main import app

    costs.record("AMASSS", 2 << 30, 6 << 30)
    raw = _raw()
    assert costs.TABLE_EPOCH_KEY in raw
    # Not a dict, so even a server from before the marker -- a rollback --
    # reads it as "no measurement" rather than as a tool costing nothing.
    assert not isinstance(raw[costs.TABLE_EPOCH_KEY], dict)

    assert list(costs.known()) == ["AMASSS"]
    banner = costs.banner()
    assert "1 tool(s) measured" in banner
    assert costs.TABLE_EPOCH_KEY not in banner

    # The nested runner, which reads the file with nothing but json.
    monkeypatch.setenv(runner.COST_TABLE_ENV, costs.table_path())
    assert runner._measured_cost(costs.TABLE_EPOCH_KEY) is None
    assert runner._measured_cost("AMASSS") is not None

    # The admin panel, and /status -- which is all a benchmark harness or any
    # other client ever sees of the table.
    client = TestClient(app)
    panel = client.get("/admin-panel.json", headers={"X-Admin-Token": settings.ADMIN_TOKEN})
    assert panel.status_code == 200
    assert list(panel.json()["costs"]) == ["AMASSS"]
    status = client.get("/status", headers={"Authorization": f"Bearer {settings.API_TOKEN}"})
    assert list(status.json()["costs"]) == ["AMASSS"]
    tool = client.get("/admin-panel/tools/%s.json" % costs.TABLE_EPOCH_KEY,
                      headers={"X-Admin-Token": settings.ADMIN_TOKEN})
    assert tool.status_code != 200 or tool.json()["cost"] is None


def test_nothing_else_resets_the_table(monkeypatch):
    """Not a new schema, not a new image, not a new venv, not a restart: none of
    those is a statement about what a tool costs."""
    costs.reconcile({"AMASSS": 1})
    costs.record("AMASSS", 2 << 30, 6 << 30)
    before = _raw()
    # The table pinned where it is, and a schema cache that starts empty -- what
    # a recreated container with a regenerated schema looks like.
    monkeypatch.setattr(settings, "COST_TABLE_PATH", costs.table_path())
    monkeypatch.setattr(settings, "SCHEMA_CACHE_DIR", settings.SCHEMA_CACHE_DIR + "-regenerated")
    for _restart in range(3):
        costs.reconcile({"AMASSS": 1})
    assert _raw() == before
