"""The server's own "Check for updates": read-only, and honest about what it
cannot know.

Built on real git repositories in a temp folder: a remote, and a checkout of
it that falls behind when a commit is pushed from elsewhere.
"""

import os
import subprocess

import pytest
from fastapi.testclient import TestClient

import main
from config import settings
from wire import release_diff, update_check

client = TestClient(main.app)


def _git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd)] + list(args), check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})


@pytest.fixture
def repos(tmp_path):
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    writer = tmp_path / "writer"
    _git(tmp_path, "clone", "-q", str(origin), str(writer))
    _git(writer, "checkout", "-q", "-b", "deploy")
    (writer / "server").mkdir()
    (writer / "server" / "main.py").write_text("v1")
    _git(writer, "add", "-A"); _git(writer, "commit", "-q", "-m", "ADD: start")
    _git(writer, "push", "-q", "origin", "HEAD:deploy")
    checkout = tmp_path / "checkout"
    _git(tmp_path, "clone", "-q", "-b", "deploy", str(origin), str(checkout))
    update_check.reset()
    yield writer, checkout
    update_check.reset()


def _push(writer, path, text, message):
    full = writer / path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(text)
    _git(writer, "add", "-A"); _git(writer, "commit", "-q", "-m", message)
    _git(writer, "push", "-q", "origin", "HEAD:deploy")


def test_an_up_to_date_checkout_has_nothing_waiting(repos):
    _writer, checkout = repos
    info = update_check.check_repo(str(checkout), "server")
    assert info["error"] == "" and info["behind"] == 0 and info["branch"] == "deploy"
    assert info["current"]["subject"] == "ADD: start"


def test_a_waiting_update_is_seen_even_where_its_details_cannot_be_read(repos):
    writer, checkout = repos
    _push(writer, "server/main.py", "v2", "FIX: x")
    info = update_check.check_repo(str(checkout), "server")
    assert info["behind"] is None and "update is waiting" in info["error"]


def test_with_github_the_commits_and_their_effect_are_listed(repos, monkeypatch):
    writer, checkout = repos
    _push(writer, "docker/Dockerfile", "FROM x", "UPDATE: image")
    monkeypatch.setattr(update_check, "_compare", lambda url, base, head: {
        "ahead_by": 1, "behind_by": 0,
        "commits": [{"sha": "abcdef1234", "commit": {"message": "UPDATE: image\n\nbody",
                                                     "author": {"name": "Jules", "date": "2026-10-05T10:00:00Z"}}}],
        "files": [{"filename": "docker/Dockerfile"}],
    })
    info = update_check.check_repo(str(checkout), "server")
    assert info["behind"] == 1
    assert info["commits"][0] == {"sha": "abcdef123", "subject": "UPDATE: image", "author": "Jules",
                                  "at": 1791194400}
    assert info["changes"]["action"] == release_diff.IMAGE
    assert any("image" in r for r in update_check.blockers(info))


def test_a_folder_that_is_no_checkout_says_so(tmp_path):
    assert "not visible" in update_check.check_repo(str(tmp_path), "tools")["error"]


def test_the_endpoint_checks_both_checkouts_and_the_panel_carries_it(repos, monkeypatch):
    _writer, checkout = repos
    monkeypatch.setattr(settings, "SERVER_REPO", str(checkout))
    monkeypatch.setattr(update_check, "tools_repo", lambda: str(checkout))
    admin = {"X-Admin-Token": settings.ADMIN_TOKEN}
    result = client.post("/admin/updates/check", headers=admin).json()
    assert result["server"]["behind"] == 0 and result["tools"]["kind"] == "tools"
    panel = client.get("/admin-panel/updates.json", headers=admin).json()
    assert panel["check"]["server"]["local"] == result["server"]["local"]
    assert client.post("/admin/updates/check", headers={"Authorization": f"Bearer {settings.API_TOKEN}"}).status_code == 401
