"""scripts/update_agent.py, against real git repositories in a temp folder.

Each test builds a remote and a checkout that follows it, commits to the
remote, and lets the agent look. What must hold:

* it counts and lists what is waiting, and classifies it -- a restart, a
  recreated container, or an image it must refuse; per tool, code or
  environment;
* it never touches a checkout without a request, and a request withdrawn while
  it waits for runs changes nothing;
* an update pulls, restarts, and ALWAYS ends with the door open.
"""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts")


@pytest.fixture(scope="module")
def agent_module():
    path = os.path.join(_SCRIPTS, "update_agent.py")
    if not os.path.isfile(path):
        pytest.skip("scripts/update_agent.py is not mounted here")
    sys.path.insert(0, _SCRIPTS)
    spec = importlib.util.spec_from_file_location("update_agent_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd)] + list(args), check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})


def _commit(repo, files, message):
    for path, text in files.items():
        full = repo / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    _git(repo, "push", "-q", "origin", "HEAD:deploy")


@pytest.fixture
def repos(tmp_path):
    """`(writer, checkout)`: commits made in `writer` reach `checkout` only
    through its remote, as on a deployment."""
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    writer = tmp_path / "writer"
    _git(tmp_path, "clone", "-q", str(origin), str(writer))
    _git(writer, "checkout", "-q", "-b", "deploy")
    _commit(writer, {"server/main.py": "v1", "tools/AMASSS/src/a.py": "v1",
                     "tools/AREG/AREG_CBCT/uv.lock": "v1"}, "ADD: start")
    checkout = tmp_path / "checkout"
    _git(tmp_path, "clone", "-q", "-b", "deploy", str(origin), str(checkout))
    return writer, checkout


def test_an_up_to_date_checkout_has_nothing_waiting(agent_module, repos):
    _writer, checkout = repos
    info = agent_module.inspect(str(checkout), "server")
    assert info["behind"] == 0 and info["commits"] == [] and info["error"] == ""
    assert info["branch"] == "deploy"


def test_server_commits_are_listed_and_classified(agent_module, repos):
    writer, checkout = repos
    _commit(writer, {"server/main.py": "v2"}, "FIX: a server fix")
    info = agent_module.inspect(str(checkout), "server")
    assert info["behind"] == 1
    assert info["commits"][0]["subject"] == "FIX: a server fix"
    assert info["changes"]["action"] == agent_module.RESTART
    _commit(writer, {"server/requirements.txt": "fastapi"}, "UPDATE: deps")
    assert agent_module.inspect(str(checkout), "server")["changes"]["action"] == agent_module.RECREATE
    _commit(writer, {"docker/Dockerfile": "FROM x"}, "UPDATE: image")
    info = agent_module.inspect(str(checkout), "server")
    assert info["changes"]["action"] == agent_module.IMAGE
    assert any("image" in reason for reason in agent_module.blockers(info))


def test_tools_commits_name_each_tool_and_whether_its_environment_changes(agent_module, repos):
    writer, checkout = repos
    _commit(writer, {"tools/AMASSS/src/a.py": "v2", "tools/AREG/AREG_CBCT/uv.lock": "v2",
                     "scripts/data-manifest.yml": "x", "tools/_template/x": "y"}, "UPDATE: tools")
    changes = agent_module.inspect(str(checkout), "tools")["changes"]
    by_tool = {t["tool"]: t for t in changes["tools"]}
    assert set(by_tool) == {"AMASSS", "AREG"}
    assert by_tool["AMASSS"]["environment"] is False
    assert by_tool["AREG"]["environment"] is True
    assert changes["environments"] == ["tools/AREG/AREG_CBCT"]
    assert "scripts/data-manifest.yml" in changes["other"]


def test_local_changes_block_an_update(agent_module, repos):
    writer, checkout = repos
    _commit(writer, {"server/main.py": "v2"}, "FIX: x")
    (checkout / "server" / "main.py").write_text("edited here")
    assert any("uncommitted" in r for r in agent_module.blockers(agent_module.inspect(str(checkout), "server")))


class _FakeServer:
    def __init__(self, busy_polls=0):
        self.door_calls = []
        self.busy_polls = busy_polls

    def install(self, agent):
        agent.door = lambda accepting, reason="": self.door_calls.append(accepting)
        agent.healthy = lambda: True

        def in_flight():
            if self.busy_polls > 0:
                self.busy_polls -= 1
                return [{"tool": "AMASSS", "state": "running"}]
            return []
        agent.in_flight = in_flight


def _agent(agent_module, checkout, tmp_path, **extra):
    args = agent_module.argparse.Namespace(
        server_repo=str(checkout), tools_repo="", update_dir=str(tmp_path / "update"),
        url="http://127.0.0.1:1", token="t", service=None, restart_cmd="true", poll=60,
        data_dir=str(tmp_path / "DATA"),
        health_timeout=5, no_sync=True, once=True)
    for key, value in extra.items():
        setattr(args, key, value)
    return agent_module.Agent(args)


def _request(agent, target="all"):
    os.makedirs(agent.update_dir, exist_ok=True)
    request = {"id": "r1", "target": target}
    with open(os.path.join(agent.update_dir, "request.json"), "w") as handle:
        json.dump(request, handle)
    return request


def test_an_update_drains_pulls_restarts_and_reopens(agent_module, repos, tmp_path, monkeypatch):
    writer, checkout = repos
    _commit(writer, {"server/main.py": "v2"}, "FIX: x")
    monkeypatch.setattr(agent_module, "REQUEST_POLL_SECONDS", 0)
    agent = _agent(agent_module, checkout, tmp_path)
    server = _FakeServer(busy_polls=2)
    server.install(agent)
    agent.apply(_request(agent))
    assert (checkout / "server" / "main.py").read_text() == "v2"
    assert server.door_calls[0] is False and server.door_calls[-1] is True
    last = json.load(open(os.path.join(agent.update_dir, "status.json")))["last"]
    assert last["ok"] is True and "server" in last["message"]
    assert not os.path.exists(os.path.join(agent.update_dir, "request.json"))
    assert any("Waiting for 1 run(s)" in line for line in last["log"])


def test_a_request_withdrawn_while_runs_finish_changes_nothing(agent_module, repos, tmp_path, monkeypatch):
    writer, checkout = repos
    _commit(writer, {"server/main.py": "v2"}, "FIX: x")
    monkeypatch.setattr(agent_module, "REQUEST_POLL_SECONDS", 0)
    agent = _agent(agent_module, checkout, tmp_path)
    server = _FakeServer(busy_polls=1000)
    server.install(agent)
    request = _request(agent)

    real_in_flight = agent.in_flight

    def withdraw_then_report():
        os.remove(os.path.join(agent.update_dir, "request.json"))
        return real_in_flight()
    agent.in_flight = withdraw_then_report
    agent.apply(request)
    assert (checkout / "server" / "main.py").read_text() == "v1"
    assert server.door_calls[-1] is True
    last = json.load(open(os.path.join(agent.update_dir, "status.json")))["last"]
    assert last["ok"] is False and "Withdrawn" in last["message"]


def test_an_image_change_is_refused_without_closing_the_door(agent_module, repos, tmp_path):
    writer, checkout = repos
    _commit(writer, {"docker/Dockerfile": "FROM x"}, "UPDATE: image")
    agent = _agent(agent_module, checkout, tmp_path)
    server = _FakeServer()
    server.install(agent)
    agent.apply(_request(agent))
    last = json.load(open(os.path.join(agent.update_dir, "status.json")))["last"]
    assert last["ok"] is False and "image" in last["message"]
    assert False not in server.door_calls
    assert not os.path.exists(checkout / "docker" / "Dockerfile")


def test_a_failed_restart_still_reopens_the_door(agent_module, repos, tmp_path, monkeypatch):
    writer, checkout = repos
    _commit(writer, {"server/main.py": "v2"}, "FIX: x")
    agent = _agent(agent_module, checkout, tmp_path, restart_cmd="false")
    server = _FakeServer()
    server.install(agent)
    agent.apply(_request(agent))
    last = json.load(open(os.path.join(agent.update_dir, "status.json")))["last"]
    assert last["ok"] is False and "restart failed" in last["message"]
    assert server.door_calls[-1] is True


def _manifest_repo(tmp_path):
    repo = tmp_path / "tools-lib"
    (repo / "scripts").mkdir(parents=True)
    engine = os.path.join(_SCRIPTS, "fetch_data.py")
    (repo / "scripts" / "fetch_data.py").write_text(open(engine).read())
    (repo / "scripts" / "data-manifest.yml").write_text(
        "tools:\n"
        "  AMASSS:\n"
        "    models:\n"
        "      - name: weights.zip\n"
        "        dest: AMASSS_Models\n"
        "        url: https://example.invalid/w.zip\n"
        "        size: 1000\n"
        "        extract: true\n"
        "    testfiles:\n"
        "      - name: scan.nii.gz\n"
        "        url: https://example.invalid/scan.nii.gz\n"
        "        size: 20\n")
    return repo


def test_the_survey_says_which_manifest_entries_are_on_disk(agent_module, tmp_path):
    repo = _manifest_repo(tmp_path)
    data = tmp_path / "DATA"
    (data / "AMASSS" / "models" / "AMASSS_Models").mkdir(parents=True)
    summary = agent_module.data_summary(str(repo), str(data))
    entries = {e["name"]: e for e in summary["tools"]["AMASSS"]["entries"]}
    assert entries["AMASSS_Models"]["present"] is True and entries["AMASSS_Models"]["kind"] == "models"
    assert entries["scan.nii.gz"]["present"] is False and entries["scan.nii.gz"]["size"] == 20


def test_a_data_request_runs_the_download_for_that_tool_only(agent_module, repos, tmp_path):
    _writer, checkout = repos
    repo = _manifest_repo(tmp_path)
    agent = _agent(agent_module, checkout, tmp_path, tools_repo=str(repo))
    os.makedirs(agent.update_dir, exist_ok=True)
    agent.apply_data({"id": "d1", "kind": "data", "tool": "AMASSS", "force": False})
    last = json.load(open(os.path.join(agent.update_dir, "status.json")))["last"]
    # example.invalid cannot be reached, so the download fails -- and says so,
    # with the engine's own output in the log.
    assert last["kind"] == "data" and last["tool"] == "AMASSS" and last["ok"] is False
    assert any("AMASSS" in line for line in last["log"])


def test_a_change_to_a_shared_package_rebuilds_every_environment_that_copies_it(agent_module, repos):
    """A package installed from a folder of the library is a COPY in each
    dependent environment: changing it must reinstall it there."""
    writer, checkout = repos
    _commit(writer, {
        "tools/Shared/common/pyproject.toml": "[project]\nname = 'shared-common'\n",
        "tools/User/pyproject.toml": "[project]\nname='user'\n[tool.uv.sources]\n"
                                     "shared-common = { path = \"../Shared/common\" }\n",
    }, "ADD: a shared package")
    agent_module.git(str(checkout), ["pull", "-q", "--ff-only"])
    _commit(writer, {"tools/Shared/common/src/x.py": "v2"}, "FIX: shared")
    changes = agent_module.inspect(str(checkout), "tools")["changes"]
    assert "tools/User" in changes["environments"]
    assert changes["reinstall"] == {"tools/User": ["shared-common"]}
    assert {t["tool"]: t["environment"] for t in changes["tools"]}["User"] is True
