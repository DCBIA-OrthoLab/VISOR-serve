"""The updater's decisions, without a network, a docker or a server.

Three of them carry the whole design, and each is asserted here:

* what a pull can and cannot deliver. Pulling a lockfile change onto a
  deployment whose virtualenvs live in the image produces a tool that imports
  a package nobody installed -- broken in a way no log line explains;
* that a run is never interrupted. Every wait in here ends by leaving the
  server alone, never by killing anything;
* that a paused run does not block an update, and a ghost does not either.
"""

import importlib.util
import os
import sys

import pytest

_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts",
)


def _load_server_ctl():
    """Import scripts/server_ctl.py by path: scripts/ is not a package."""
    path = os.path.join(_SCRIPTS, "server_ctl.py")
    if not os.path.isfile(path):
        pytest.skip("scripts/server_ctl.py is not mounted here")
    spec = importlib.util.spec_from_file_location("server_ctl_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ctl = _load_server_ctl()


# ---------------------------------------------------------------------------
# What a pull can deliver
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "tools/AMASSS/uv.lock",
    "tools/ALI/ALI_IOS/pyproject.toml",
    "pyproject.toml",
    "uv.lock",
    "server/requirements.txt",
    "server/requirements-api.txt",
    "docker/Dockerfile",
])
def test_a_dependency_change_cannot_travel_by_pull(path):
    """The virtualenvs are built into the image and git never touches them, so
    landing a source that needs a new package is landing a broken tool."""
    assert ctl.needs_image([path]) == [path]


@pytest.mark.parametrize("path", [
    "tools/ALI/ALI_IOS/src/sadt_ali_ios/engine.py",
    "server/main.py",
    "server/wire/maintenance.py",
    "CLAUDE.md",
    "server/deployment.toml",
    "docs/SERVER_CONTRACT.md",
])
def test_a_source_change_travels_by_pull(path):
    assert ctl.needs_image([path]) == []


def test_the_refusal_names_every_path_it_is_refusing_over():
    """Returned rather than answered yes/no: "an image is needed" sends someone
    reading a diff, "because tools/AMASSS/uv.lock changed" sends them to the
    commit."""
    changed = [
        "tools/AMASSS/src/x.py",
        "tools/AMASSS/uv.lock",
        "server/main.py",
        "docker/Dockerfile",
    ]
    assert ctl.needs_image(changed) == ["tools/AMASSS/uv.lock", "docker/Dockerfile"]


def test_a_file_merely_named_like_one_is_not_mistaken_for_it():
    """`uv.lock` is a whole file name, not a suffix: a source file that ends in
    those letters must not stop an update."""
    assert ctl.needs_image(["tools/X/src/my_uv.lock.py"]) == []
    assert ctl.needs_image(["tools/X/src/pyproject.toml.j2"]) == []


# ---------------------------------------------------------------------------
# The drain
# ---------------------------------------------------------------------------

def _load(running=0, waiting=0, busy=(), paused=()):
    return {"running": running, "waiting": waiting, "busy": list(busy),
            "paused": list(paused), "accepting": True}


def test_an_idle_server_is_ready_at_once(monkeypatch):
    monkeypatch.setattr(ctl, "server_load", lambda url, token: _load())
    assert ctl.wait_until_idle("http://x", "t") is True


def test_a_paused_run_does_not_block_an_update(monkeypatch):
    """It holds no reservation, no thread and no subprocess -- it is a record on
    disk plus a staged cohort, and a restart that keeps TEMP_DIR keeps both.
    Waiting for one would mean waiting out its four-hour idle timeout for a
    clinician who may have gone home, and the only thing that would end that
    wait is the reaper deleting the very work the wait was protecting."""
    paused = [{"run_id": "r1", "tool": "AMASSS", "state": "paused"}]
    monkeypatch.setattr(ctl, "server_load", lambda url, token: _load(paused=paused))
    assert ctl.wait_until_idle("http://x", "t") is True


def test_a_busy_server_is_left_alone_rather_than_interrupted(monkeypatch):
    """The timeout gives up. It does not kill, it does not force, it does not
    close the door -- the next poll tries again."""
    busy = [{"run_id": "r1", "tool": "AMASSS", "state": "running", "phase": "running"}]
    monkeypatch.setattr(ctl, "server_load", lambda url, token: _load(busy=busy))
    monkeypatch.setattr(ctl.time, "sleep", lambda _s: None)

    assert ctl.wait_until_idle("http://x", "t", timeout=0) is False


def test_a_server_that_will_not_answer_is_not_updated(monkeypatch):
    """Not updating while the state is unknown is the safe direction: the
    alternative is restarting a server that may be mid-cohort."""
    monkeypatch.setattr(ctl, "server_load", lambda url, token: None)
    assert ctl.wait_until_idle("http://x", "t") is False


def test_the_wait_ends_once_the_work_does(monkeypatch):
    """Busy, then busy, then idle -- and it must notice the third."""
    answers = iter([_load(busy=[{"tool": "ASO", "state": "running"}]),
                    _load(running=1),
                    _load()])
    monkeypatch.setattr(ctl, "server_load", lambda url, token: next(answers))
    monkeypatch.setattr(ctl.time, "sleep", lambda _s: None)
    assert ctl.wait_until_idle("http://x", "t") is True


# ---------------------------------------------------------------------------
# What a ghost is, and what it is not
# ---------------------------------------------------------------------------

def test_a_run_whose_client_vanished_does_not_hold_an_update_for_ever(monkeypatch):
    """Its record sits at `pending` until the server's reaper collects it, up
    to RUN_TTL_SECONDS later. Waiting behind it would stall every update on
    this server for fifteen minutes for a client that is gone."""
    import time as real_time

    stale = real_time.time() - (ctl.GHOST_AFTER_SECONDS + 60)
    status = {
        "admission": {"running": 0, "waiting": 0},
        "runs": [{"run_id": "ghost", "tool": "ALI_CBCT", "state": "pending",
                  "updated_at": stale}],
        "maintenance": {"accepting": True},
    }
    monkeypatch.setattr(ctl, "_api", lambda *a, **k: status)

    assert ctl.server_load("http://x", "t")["busy"] == []


def test_a_run_that_is_merely_slow_still_holds_the_update(monkeypatch):
    """The mirror image, and the one that matters clinically: a cohort that has
    been quiet for a minute is working, not gone."""
    import time as real_time

    status = {
        "admission": {"running": 1, "waiting": 0},
        "runs": [{"run_id": "real", "tool": "AMASSS", "state": "running",
                  "updated_at": real_time.time() - 60}],
        "maintenance": {"accepting": True},
    }
    monkeypatch.setattr(ctl, "_api", lambda *a, **k: status)

    assert len(ctl.server_load("http://x", "t")["busy"]) == 1


# ---------------------------------------------------------------------------
# One poll, end to end, with git and the server replaced by records
# ---------------------------------------------------------------------------

class _Git:
    """Records every git command, and answers the few that matter."""

    def __init__(self, local, changed):
        self.calls = []
        self.local = local
        self.changed = changed

    def __call__(self, args, timeout=120):
        self.calls.append(list(args))
        if args[:2] == ["rev-parse", "HEAD"]:
            return 0, self.local, ""
        if args[0] == "diff":
            return 0, "\n".join(self.changed), ""
        return 0, "", ""

    @property
    def pulled(self):
        return any(call[0] == "pull" for call in self.calls)


class _Args:
    branch = None
    url = None
    device = None
    poll = 30
    drain_timeout = None
    timeout = 60
    once = True


def test_a_change_needing_an_image_is_refused_without_pulling(monkeypatch):
    """The property the whole two-tier design rests on. A pull here would land
    a source whose dependencies are not installed, and every run of that tool
    would fail with an ImportError nobody can act on."""
    git = _Git(local="aaaaaaa", changed=["tools/AMASSS/uv.lock", "tools/AMASSS/src/x.py"])
    monkeypatch.setattr(ctl, "_git", git)
    monkeypatch.setattr(ctl, "remote_head", lambda *a, **k: "bbbbbbb")
    monkeypatch.setattr(ctl, "wait_until_idle", lambda *a, **k: pytest.fail(
        "it must refuse before it ever asks the server to go idle"))

    outcome = ctl._watch_once("deploy", "http://x", "inference", "token", _Args())

    assert outcome["needs_image"]["paths"] == ["tools/AMASSS/uv.lock"]
    assert not git.pulled, "a tier-two change must never be pulled"


def test_a_busy_server_is_not_pulled_over(monkeypatch):
    """The drain said no. Nothing is pulled, nothing is restarted, and the door
    is never even closed."""
    git = _Git(local="aaaaaaa", changed=["server/main.py"])
    closed = []
    monkeypatch.setattr(ctl, "_git", git)
    monkeypatch.setattr(ctl, "remote_head", lambda *a, **k: "bbbbbbb")
    monkeypatch.setattr(ctl, "wait_until_idle", lambda *a, **k: False)
    monkeypatch.setattr(ctl, "set_accepting", lambda *a, **k: closed.append(a) or {})

    outcome = ctl._watch_once("deploy", "http://x", "inference", "token", _Args())

    assert not git.pulled
    assert closed == [], "the door must not be shut for a server that stays busy"
    assert "applied" not in outcome


def test_nothing_happens_when_the_remote_has_not_moved(monkeypatch):
    """The common case, thirty times a minute across a fleet: it must fetch
    nothing and say nothing."""
    git = _Git(local="aaaaaaa", changed=[])
    monkeypatch.setattr(ctl, "_git", git)
    monkeypatch.setattr(ctl, "remote_head", lambda *a, **k: "aaaaaaa")

    outcome = ctl._watch_once("deploy", "http://x", "inference", "token", _Args())

    assert outcome == {"head": "aaaaaaa"}
    assert git.calls == [["rev-parse", "HEAD"]], "a poll that changes nothing must fetch nothing"


def test_a_run_that_slips_in_while_the_door_closes_stops_the_update(monkeypatch):
    """The race the door exists to close. Between "idle" and "shut" a POST can
    still land, and pulling on top of it would restart the server under a run
    that had just been accepted."""
    git = _Git(local="aaaaaaa", changed=["server/main.py"])
    monkeypatch.setattr(ctl, "_git", git)
    monkeypatch.setattr(ctl, "remote_head", lambda *a, **k: "bbbbbbb")
    monkeypatch.setattr(ctl, "wait_until_idle", lambda *a, **k: True)
    monkeypatch.setattr(ctl, "set_accepting", lambda *a, **k: {"accepting": False})
    monkeypatch.setattr(ctl, "server_load", lambda *a, **k: _load(running=1))

    outcome = ctl._watch_once("deploy", "http://x", "inference", "token", _Args())

    assert not git.pulled
    assert "applied" not in outcome


def test_the_door_is_reopened_even_when_the_pull_fails(monkeypatch):
    """A refusal must not leave a clinic behind a shut door waiting for the
    deadman. The lease is the backstop for this process being killed, not for
    it forgetting."""
    class _FailingGit(_Git):
        def __call__(self, args, timeout=120):
            if args[0] == "pull":
                self.calls.append(list(args))
                return 1, "", "the branch has diverged"
            return super().__call__(args, timeout)

    git = _FailingGit(local="aaaaaaa", changed=["server/main.py"])
    reopened = []
    monkeypatch.setattr(ctl, "_git", git)
    monkeypatch.setattr(ctl, "remote_head", lambda *a, **k: "bbbbbbb")
    monkeypatch.setattr(ctl, "wait_until_idle", lambda *a, **k: True)
    monkeypatch.setattr(ctl, "server_load", lambda *a, **k: _load())
    monkeypatch.setattr(
        ctl, "set_accepting",
        lambda url, token, accepting, *a, **k: reopened.append(accepting) or {"accepting": accepting},
    )

    ctl._watch_once("deploy", "http://x", "inference", "token", _Args())

    assert reopened == [False, True], "the door must be reopened on the failure path too"


# ---------------------------------------------------------------------------
# Whether this deployment follows its branch at all
# ---------------------------------------------------------------------------

def test_a_deployment_follows_nothing_unless_it_was_told_to(monkeypatch):
    """The default is off, and that is a safety property rather than a
    convenience one: a machine holding patient imaging must not begin executing
    code from the internet because somebody installed a timer."""
    monkeypatch.delenv("SADT_AUTO_UPDATE", raising=False)
    monkeypatch.setattr(ctl, "read_env", lambda *a, **k: {})
    assert ctl.auto_update_mode() == ctl.AUTO_UPDATE_OFF


@pytest.mark.parametrize("raw,expected", [
    ("apply", ctl.AUTO_UPDATE_APPLY),
    ("NOTIFY", ctl.AUTO_UPDATE_NOTIFY),
    ("  off  ", ctl.AUTO_UPDATE_OFF),
])
def test_the_mode_is_read_from_the_environment(monkeypatch, raw, expected):
    monkeypatch.setenv("SADT_AUTO_UPDATE", raw)
    assert ctl.auto_update_mode() == expected


def test_a_value_nobody_recognises_is_treated_as_off(monkeypatch):
    """The safe direction. A typo must not be read as consent to self-update."""
    monkeypatch.setenv("SADT_AUTO_UPDATE", "yes")
    monkeypatch.setattr(ctl, "read_env", lambda *a, **k: {})
    assert ctl.auto_update_mode() == ctl.AUTO_UPDATE_OFF


def test_off_asks_the_remote_nothing_at_all(monkeypatch):
    """Inert means inert: no network, no git, no server. A unit installed on
    every machine and switched off costs that machine nothing."""
    monkeypatch.setattr(ctl, "remote_head", lambda *a, **k: pytest.fail("it polled"))
    monkeypatch.setattr(ctl, "_git", lambda *a, **k: pytest.fail("it ran git"))

    args = _Args()
    args.mode = ctl.AUTO_UPDATE_OFF
    assert ctl.cmd_watch(args)["mode"] == ctl.AUTO_UPDATE_OFF


def test_notify_classifies_but_never_touches_the_clone(monkeypatch):
    """What a site runs while it decides whether to trust the mechanism: it
    reports exactly what it would have done, and does none of it."""
    git = _Git(local="aaaaaaa", changed=["server/main.py"])
    monkeypatch.setattr(ctl, "_git", git)
    monkeypatch.setattr(ctl, "remote_head", lambda *a, **k: "bbbbbbb")
    monkeypatch.setattr(ctl, "wait_until_idle", lambda *a, **k: pytest.fail(
        "notify must not drain a server"))

    outcome = ctl._watch_once("deploy", "http://x", "inference", "t", _Args(), apply=False)

    assert outcome["would_apply"]["sha"] == "bbbbbbb"
    assert not git.pulled
