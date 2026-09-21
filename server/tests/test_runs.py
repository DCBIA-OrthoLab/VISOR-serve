"""Run progress and cancellation (wire/runs.py + its endpoints + dispatch).

No GPU, no weights, no network. The one test that needs a real tool process
builds a one-file tool with a symlinked virtualenv, the way test_supervisor.py
does -- what is under test there is precisely that a signal reaches a process
this server does not hold a handle on.

Run with: cd server && ./venv/bin/pytest tests/test_runs.py
"""

import json
import os
import secrets
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

os.environ.setdefault("API_TOKEN", "test-token")

import pytest
from fastapi.testclient import TestClient

import config
import registry
from base import ArgSpec, Tool
from config import settings
from execution import dispatch
from main import app
from wire import runs

client = TestClient(app)
TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _new_id() -> str:
    """Minted the way the client mints one, so the shape under test is real."""
    return secrets.token_urlsafe(24)


@pytest.fixture
def run_id():
    """A registered run, removed whatever the test does to it."""
    identifier = runs.register(_new_id())
    yield identifier
    runs.discard(identifier)


def _read_stream(url: str, timeout: float = 20.0) -> list:
    """Consume an SSE response to its end and return the events it carried.

    The stream ending is half of what these tests assert, so there is
    deliberately no early exit: a stream that never ends fails on the read
    timeout rather than hanging the suite.
    """
    events = []
    with client.stream("GET", url, headers=AUTH, timeout=timeout) as response:
        assert response.status_code == 200, response.status_code
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return events


# ----------------------------------------------------------------------
# A client that says nothing gets exactly what it always got
# ----------------------------------------------------------------------

def test_the_two_halves_agree_on_the_progress_variable():
    """`runs.py` sets it, `runner.py` reads it, and nothing else pins them equal.

    The duplication is deliberate: `runner.py` is executed by a TOOL's
    interpreter and is stdlib-only by contract, so it cannot import from the
    server package that names the variable. What the contract does not give is
    a failure when one side is renamed -- progress would simply stop arriving,
    silently, for every tool at once. This is that failure.
    """
    from execution import runner

    assert runner.PROGRESS_FILE_ENV == runs.PROGRESS_FILE_ENV


def test_a_run_without_a_run_id_creates_no_run_state():
    """The hard half of the contract: this feature is optional in both
    directions, so a client that has not been released yet must not pay a byte
    for it."""
    root = os.path.join(settings.TEMP_DIR, "runs")
    before = set(os.listdir(root)) if os.path.isdir(root) else set()

    response = client.post(
        "/run/Test_Tool", headers=AUTH, data={"text_1": "hello", "text_2": "world"}
    )

    assert response.status_code == 200
    assert response.json() == {"result": "hello world"}
    after = set(os.listdir(root)) if os.path.isdir(root) else set()
    assert after == before


def test_the_run_endpoints_require_a_token():
    identifier = _new_id()
    assert client.get(f"/runs/{identifier}").status_code == 401
    assert client.get(f"/runs/{identifier}/events").status_code == 401
    assert client.delete(f"/runs/{identifier}").status_code == 401


# ----------------------------------------------------------------------
# The id
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "identifier",
    ["short", "../../etc/passwd", "has spaces in it", "a" * 65, "with/slash/inside"],
)
def test_a_malformed_run_id_is_refused_before_a_path_is_built_from_it(identifier):
    response = client.post(
        "/run/Test_Tool",
        headers=dict(AUTH, **{"X-Run-Id": identifier}),
        data={"text_1": "a", "text_2": "b"},
    )
    assert response.status_code == 400
    assert "Malformed" in response.json()["detail"]


def test_a_run_id_already_in_use_is_a_409(run_id):
    """Two runs sharing a directory would interleave their events and let
    either one cancel the other."""
    response = client.post(
        "/run/Test_Tool",
        headers=dict(AUTH, **{"X-Run-Id": run_id}),
        data={"text_1": "a", "text_2": "b"},
    )
    assert response.status_code == 409
    assert run_id in response.json()["detail"]


def test_an_unknown_run_is_a_404_everywhere():
    identifier = _new_id()
    assert client.get(f"/runs/{identifier}", headers=AUTH).status_code == 404
    assert client.get(f"/runs/{identifier}/events", headers=AUTH).status_code == 404
    assert client.delete(f"/runs/{identifier}", headers=AUTH).status_code == 404


# ----------------------------------------------------------------------
# The events, and the two ways of reading them
# ----------------------------------------------------------------------

def test_a_watcher_that_attaches_late_is_never_behind(run_id):
    """The reason the stream starts from the beginning of the file rather than
    from the moment of connection: the client opens it from a second thread
    while the first is already blocked inside the POST, and the two cannot be
    ordered."""
    runs.append(run_id, runs.PHASE_RECEIVED)
    runs.append(run_id, runs.PHASE_STAGING, message="input 1 of 2")
    runs.append(run_id, runs.PHASE_RUNNING, fraction=0.5, message="patient 14 of 40")
    runs.finish(run_id, runs.PHASE_DONE)

    streamed = _read_stream(f"/runs/{run_id}/events")
    snapshot = client.get(f"/runs/{run_id}", headers=AUTH).json()

    phases = [runs.PHASE_RECEIVED, runs.PHASE_STAGING, runs.PHASE_RUNNING, runs.PHASE_DONE]
    assert [event["phase"] for event in streamed] == phases
    assert [event["phase"] for event in snapshot["events"]] == phases
    # Both readers derive the same numbering from the same file, which is the
    # whole reason seq is the line number rather than something written.
    assert [event["seq"] for event in streamed] == [0, 1, 2, 3]
    assert [event["seq"] for event in snapshot["events"]] == [0, 1, 2, 3]


def test_the_snapshot_reports_where_the_run_stands(run_id):
    runs.append(run_id, runs.PHASE_RECEIVED)
    runs.append(run_id, runs.PHASE_RUNNING, fraction=0.35, message="patient 14 of 40")

    snapshot = client.get(f"/runs/{run_id}", headers=AUTH).json()

    assert snapshot["run_id"] == run_id
    assert snapshot["state"] == runs.STATE_RUNNING
    assert snapshot["phase"] == runs.PHASE_RUNNING
    assert snapshot["fraction"] == 0.35
    assert snapshot["message"] == "patient 14 of 40"
    assert snapshot["depth"] == 0
    assert snapshot["updated_at"] >= snapshot["started_at"]


def test_the_stream_ends_on_a_terminal_event(run_id):
    """Without this a watcher hangs on a run that finished half an hour ago."""
    runs.append(run_id, runs.PHASE_RUNNING)
    runs.finish(run_id, runs.PHASE_FAILED)
    runs.append(run_id, runs.PHASE_RUNNING, message="written after the end")

    streamed = _read_stream(f"/runs/{run_id}/events")

    assert [event["phase"] for event in streamed] == [
        runs.PHASE_RUNNING, runs.PHASE_FAILED
    ]
    assert streamed[-1]["state"] == runs.STATE_FAILED


def test_the_stream_delivers_events_that_arrive_while_it_is_open(run_id):
    """The tail, which is what the endpoint exists for: the interesting events
    are the ones written while the client is watching."""
    runs.append(run_id, runs.PHASE_RECEIVED)

    def write_the_rest():
        time.sleep(settings.RUN_EVENT_POLL_SECONDS * 2)
        runs.append(run_id, runs.PHASE_RUNNING, fraction=0.5, message="halfway")
        runs.finish(run_id, runs.PHASE_DONE)

    writer = threading.Thread(target=write_the_rest)
    writer.start()
    try:
        streamed = _read_stream(f"/runs/{run_id}/events")
    finally:
        writer.join(timeout=10)

    assert [event["phase"] for event in streamed] == [
        runs.PHASE_RECEIVED, runs.PHASE_RUNNING, runs.PHASE_DONE
    ]


def test_the_stream_ends_when_the_run_directory_goes_away(run_id):
    """A run whose request died between the last event and the cleanup. The
    watcher must stop rather than poll a directory that will never change."""
    runs.append(run_id, runs.PHASE_RUNNING)

    def remove_it():
        time.sleep(settings.RUN_EVENT_POLL_SECONDS * 2)
        runs.discard(run_id)

    remover = threading.Thread(target=remove_it)
    remover.start()
    try:
        streamed = _read_stream(f"/runs/{run_id}/events")
    finally:
        remover.join(timeout=10)

    assert [event["phase"] for event in streamed] == [runs.PHASE_RUNNING]


# ----------------------------------------------------------------------
# What a tool writes is untrusted
# ----------------------------------------------------------------------

def _append_raw(run_id: str, record: dict) -> None:
    """Append exactly what a tool would append, bypassing runs.append -- these
    lines come from another repository's code, which is the point."""
    path = os.path.join(settings.TEMP_DIR, "runs", run_id, runs.EVENTS_FILE)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def test_a_message_is_truncated_to_two_hundred_characters(run_id):
    runs.append(run_id, runs.PHASE_RUNNING, message="x" * 500)
    _append_raw(run_id, {"fraction": 0.1, "message": "y" * 500})

    events = runs.read_events(run_id)

    assert len(events[0]["message"]) == runs.MAX_MESSAGE_CHARS
    assert len(events[1]["message"]) == runs.MAX_MESSAGE_CHARS


def test_a_tool_cannot_claim_a_phase_a_state_or_a_sequence(run_id):
    """A tool's line is data. It says how far along it is; it does not get to
    say the run is done, nor where its event sits in the order."""
    _append_raw(
        run_id,
        {"seq": 99, "state": "done", "phase": "done", "fraction": 0.5, "message": "no"},
    )

    event = runs.read_events(run_id)[0]

    assert event["seq"] == 0
    assert event["state"] == runs.STATE_RUNNING
    assert event["phase"] == runs.PHASE_RUNNING


@pytest.mark.parametrize(
    "fraction, expected",
    [(2.5, 1.0), (-1.0, 0.0), ("half", None), (None, None), (float("nan"), None)],
)
def test_a_fraction_outside_the_range_is_clamped_or_dropped(run_id, fraction, expected):
    """A progress bar believes what it is given, so a nonsense fraction becomes
    "unknown" rather than a number."""
    _append_raw(run_id, {"fraction": fraction, "message": "x"})

    assert runs.read_events(run_id)[0]["fraction"] == expected


def test_a_malformed_line_costs_that_line_and_nothing_else(run_id):
    path = os.path.join(settings.TEMP_DIR, "runs", run_id, runs.EVENTS_FILE)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("{not json at all\n")
    runs.append(run_id, runs.PHASE_RUNNING, message="still here")

    events = runs.read_events(run_id)

    assert [event["message"] for event in events] == ["still here"]
    # The line still counted: seq is the position in the file, so a dropped
    # line leaves a gap rather than shifting everything after it.
    assert events[0]["seq"] == 1


def test_a_half_written_line_is_read_on_the_next_poll(run_id):
    """Records are appended with one write() under PIPE_BUF, so this cannot
    happen on a local filesystem -- but the reader is what makes that a
    guarantee rather than an assumption."""
    directory = os.path.join(settings.TEMP_DIR, "runs", run_id)
    reader = runs.EventReader(directory)
    path = os.path.join(directory, runs.EVENTS_FILE)

    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"fraction": 0.5, "message": "part')
    assert reader.read() == []

    with open(path, "a", encoding="utf-8") as handle:
        handle.write('ial"}\n')
    assert [event["message"] for event in reader.read()] == ["partial"]


def test_a_chatty_tool_is_capped_but_the_run_can_still_end(run_id, monkeypatch):
    """MAX_RUN_EVENTS, and the one exception to it. A cohort of 500 patients
    reporting per slice must not fill TEMP_DIR -- and a stream that could not
    deliver its terminal event would leave every watcher hanging."""
    monkeypatch.setattr(settings, "MAX_RUN_EVENTS", 5)
    for index in range(20):
        _append_raw(run_id, {"fraction": index / 20, "message": f"step {index}"})
    runs.finish(run_id, runs.PHASE_DONE)

    events = runs.read_events(run_id)

    assert [event["message"] for event in events[:5]] == [f"step {i}" for i in range(5)]
    assert "not reported" in events[5]["message"]
    assert events[-1]["phase"] == runs.PHASE_DONE
    # One notice, not fifteen.
    assert len(events) == 7


# ----------------------------------------------------------------------
# The phases the server emits with no tool cooperation at all
# ----------------------------------------------------------------------

def test_a_run_reports_its_phases_from_received_to_done(monkeypatch):
    """What makes this useful on day one: no tool was edited for any of it."""
    identifier = _new_id()
    recorded = []

    class WatchingTool(Tool):
        name = "phase_probe_tool"
        arguments = {"text": ArgSpec(type=str)}
        output_kind = "text"

        def run(self, text: str) -> str:
            # Read from inside the worker thread, which is where dispatch
            # reads it too: the context copy is what carries it there.
            recorded.extend(runs.read_events(runs.CURRENT_RUN.get()))
            return text

    monkeypatch.setitem(registry.TOOLS, WatchingTool.name, WatchingTool())

    with TestClient(app) as shared:
        response = shared.post(
            f"/run/{WatchingTool.name}",
            headers=dict(AUTH, **{"X-Run-Id": identifier}),
            data={"text": "x"},
        )

    assert response.status_code == 200
    assert [event["phase"] for event in recorded] == [runs.PHASE_RECEIVED]
    # And the directory went with the request rather than waiting out its TTL:
    # a progress message is written by a tool and can name a file.
    assert client.get(f"/runs/{identifier}", headers=AUTH).status_code == 404


def test_staging_is_reported_per_input_and_never_names_one(monkeypatch):
    """The message says where in the batch, never which file: it travels to a
    panel and is written to disk, and an input's name is the patient's."""
    identifier = _new_id()
    seen = []

    class StagingProbeTool(Tool):
        name = "staging_probe_tool"
        arguments = {"input": ArgSpec(type="csv_file"), "label": ArgSpec(type=str)}
        output_kind = "text"

        def run(self, input, label: str) -> str:
            seen.extend(runs.read_events(runs.CURRENT_RUN.get()))
            return label

    monkeypatch.setitem(registry.TOOLS, StagingProbeTool.name, StagingProbeTool())

    with TestClient(app) as shared:
        response = shared.post(
            f"/run/{StagingProbeTool.name}",
            headers=dict(AUTH, **{"X-Run-Id": identifier}),
            data={"label": "x"},
            files={"input": ("patient_0042.csv", b"col\n1\n", "text/csv")},
        )

    assert response.status_code == 200
    staging = [event for event in seen if event["phase"] == runs.PHASE_STAGING]
    assert [event["message"] for event in staging] == ["input 1 of 1"]
    assert not any("patient_0042" in event["message"] for event in seen)


def test_the_run_directory_is_gone_after_a_failed_run(monkeypatch):
    identifier = _new_id()

    class FailingTool(Tool):
        name = "failing_probe_tool"
        arguments = {"text": ArgSpec(type=str)}
        output_kind = "text"

        def run(self, text: str) -> str:
            raise RuntimeError("boom")

    monkeypatch.setitem(registry.TOOLS, FailingTool.name, FailingTool())

    with TestClient(app, raise_server_exceptions=False) as shared:
        response = shared.post(
            f"/run/{FailingTool.name}",
            headers=dict(AUTH, **{"X-Run-Id": identifier}),
            data={"text": "x"},
        )

    assert response.status_code == 500
    assert client.get(f"/runs/{identifier}", headers=AUTH).status_code == 404


def test_an_unknown_tool_still_cleans_up_the_run_it_registered():
    """The tool is resolved AFTER the run is registered -- it has to be, since
    registering first is what stops the client's watcher getting a 404 -- so a
    404 for the tool now has a directory of its own to take down."""
    identifier = _new_id()

    response = client.post(
        "/run/NoSuchTool",
        headers=dict(AUTH, **{"X-Run-Id": identifier}),
        data={"text": "x"},
    )

    assert response.status_code == 404
    assert client.get(f"/runs/{identifier}", headers=AUTH).status_code == 404


# ----------------------------------------------------------------------
# Expiry
# ----------------------------------------------------------------------

def _age(directory: str, seconds: float) -> None:
    stamp = time.time() - seconds
    os.utime(directory, (stamp, stamp))


def test_a_run_still_reporting_is_never_reaped_under_itself(run_id):
    """RUN_TTL_SECONDS is an IDLE timeout: a cohort legitimately runs for
    hours, and an age limit would take its progress away mid-run."""
    directory = os.path.join(settings.TEMP_DIR, "runs", run_id)
    _age(directory, settings.RUN_TTL_SECONDS + 60)

    runs.append(run_id, runs.PHASE_RUNNING, message="still going")
    runs.reap_expired()

    assert os.path.isdir(directory)


def test_an_abandoned_run_expires(run_id):
    directory = os.path.join(settings.TEMP_DIR, "runs", run_id)
    _age(directory, settings.RUN_TTL_SECONDS + 60)

    assert runs.reap_expired() >= 1
    assert not os.path.isdir(directory)


def test_the_run_reaper_runs_on_the_same_timer_as_the_transfer_one():
    """The hole this closes: a client that vanished mid-POST leaves a run
    directory nothing else will ever remove, and an idle server is exactly
    where it sits longest."""
    identifier = runs.register(_new_id())
    directory = os.path.join(settings.TEMP_DIR, "runs", identifier)
    _age(directory, settings.RUN_TTL_SECONDS + 60)

    original = settings.TRANSFER_SWEEP_SECONDS
    settings.TRANSFER_SWEEP_SECONDS = 0.05
    try:
        with TestClient(app):
            deadline = time.monotonic() + 10
            while os.path.isdir(directory) and time.monotonic() < deadline:
                time.sleep(0.05)
    finally:
        settings.TRANSFER_SWEEP_SECONDS = original
        runs.discard(identifier)

    assert not os.path.isdir(directory), "the timed sweep never reached the runs"


# ----------------------------------------------------------------------
# Cancellation
# ----------------------------------------------------------------------

def test_deleting_a_run_is_idempotent(run_id):
    assert client.delete(f"/runs/{run_id}", headers=AUTH).status_code == 204
    assert client.delete(f"/runs/{run_id}", headers=AUTH).status_code == 204
    assert runs.is_cancelled(run_id)


def test_a_cancel_before_the_process_starts_spawns_nothing(
    probe_tool, probe_python, tracked_scratch_dirs, run_id, monkeypatch
):
    """The first of dispatch's four checks. It is what covers the window with
    no process in it -- inputs staged, job file about to be written -- where
    signalling a process group would have nothing to signal."""
    started = []
    monkeypatch.setattr(
        dispatch, "_execute", lambda *args, **kwargs: started.append(args) or 0
    )
    runs.request_cancel(run_id)

    token = runs.CURRENT_RUN.set(run_id)
    try:
        with pytest.raises(dispatch.RunCancelled):
            dispatch.dispatch(probe_tool, {"a": 1, "b": 1})
    finally:
        runs.CURRENT_RUN.reset(token)

    assert started == [], "a tool process was started for a run already cancelled"


def test_a_cancelled_run_answers_the_post_with_499(monkeypatch):
    """499 rather than 500, and that is the whole reason it exists: the client
    has to tell "I stopped this" from "this broke" without reading a message."""
    identifier = _new_id()

    class CancellingTool(Tool):
        name = "cancelling_probe_tool"
        arguments = {"text": ArgSpec(type=str)}
        output_kind = "text"

        def run(self, text: str) -> str:
            raise dispatch.RunCancelled("The client cancelled this run.")

    monkeypatch.setitem(registry.TOOLS, CancellingTool.name, CancellingTool())

    with TestClient(app, raise_server_exceptions=False) as shared:
        response = shared.post(
            f"/run/{CancellingTool.name}",
            headers=dict(AUTH, **{"X-Run-Id": identifier}),
            data={"text": "x"},
        )

    assert response.status_code == 499
    assert response.json() == {"detail": "Run cancelled by the client."}


def test_a_cancel_is_read_as_one_even_when_the_tool_exited_cleanly(
    probe_tool, probe_python, tracked_scratch_dirs, run_id
):
    """A DELETE served by another uvicorn worker kills the process group and
    tells this one nothing. Without the check after the wait, the run would be
    reported as a tool that died with -15 -- a 500, and an error dialog for
    something the user asked for."""
    token = runs.CURRENT_RUN.set(run_id)
    try:
        runs.request_cancel(run_id)
        with pytest.raises(dispatch.RunCancelled):
            dispatch._wait(
                subprocess.Popen([sys.executable, "-c", "pass"]),
                None, "_dispatch_probe", run_id,
            )
    finally:
        runs.CURRENT_RUN.reset(token)


def test_the_timeout_message_is_unchanged_by_the_polling_wait(run_id):
    """The wait is broken into slices so a cancel can interrupt it. Everything
    else about a timeout -- the budget, the kill, the message naming both knobs
    -- has to be exactly what it was."""
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        with pytest.raises(dispatch.ToolExecutionError) as raised:
            dispatch._wait(process, 0.5, "Sleeper", run_id)
    finally:
        process.kill()
        process.wait(timeout=10)

    message = str(raised.value)
    assert "did not finish within its timeout (0.5s)" in message
    assert "[tools.Sleeper] timeout_seconds in deployment.toml" in message
    assert "TOOL_TIMEOUT_SECONDS" in message


def test_an_implausible_process_group_is_never_signalled(caplog):
    """`os.killpg(0, ...)` signals the CALLER's group, which is this server. A
    truncated pgid file must not be able to take the API down on a cancel."""
    for pgid in (0, 1, -1, None, "12"):
        dispatch.kill_process_group(pgid)
    assert "implausible process group" in caplog.text


def test_a_pgid_file_that_says_zero_is_read_as_no_pgid(run_id):
    directory = os.path.join(settings.TEMP_DIR, "runs", run_id)
    with open(os.path.join(directory, runs.PGID_FILE), "w", encoding="utf-8") as handle:
        handle.write("0")

    assert runs.get_pgid(run_id) is None
    assert runs.request_cancel(run_id) is None


# ----------------------------------------------------------------------
# Cancelling a tool that is really running, in a real process group
# ----------------------------------------------------------------------

SLEEPER = """
    import os
    import subprocess
    import sys
    import time


    def run(pid_file: str) -> str:
        \"\"\"Fork a worker, publish both pids, and hang.

        The grandchild is the point: nnUNet, torch's DataLoader and shapeaxi all
        fork workers, and killing only the pid the server knows about leaves
        those holding VRAM on a card nothing can be attributed to any more.
        \"\"\"
        worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        with open(pid_file, "w") as handle:
            handle.write("{} {}".format(os.getpid(), worker.pid))
        time.sleep(300)
        return "never"
"""


class SleeperTool(Tool):
    """Declared here rather than under tools/: it must stay invisible to the
    registry, like the dispatch probe."""

    name = "Sleeper"
    arguments = {"pid_file": ArgSpec(type=str)}
    output_kind = "text"

    def run(self, pid_file: str) -> str:
        raise AssertionError("run() must not be called on the subprocess path")


def _install_sleeper(tools_dir: Path) -> None:
    """A runnable tool: src/sadt_sleeper/ plus a venv that is a symlink to this
    interpreter. The runner derives the tool folder from sys.prefix, so the
    interpreter only has to LOOK like it lives there."""
    package = tools_dir / SleeperTool.name / "src" / "sadt_sleeper"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(textwrap.dedent(SLEEPER), encoding="utf-8")
    binaries = tools_dir / SleeperTool.name / ".venv" / "bin"
    binaries.mkdir(parents=True)
    (binaries / "python").symlink_to(sys.executable)
    (tools_dir / SleeperTool.name / ".venv" / "pyvenv.cfg").write_text(
        "home = {}\ninclude-system-site-packages = true\n".format(
            os.path.dirname(sys.executable)
        ),
        encoding="utf-8",
    )


def _is_alive(pid: int) -> bool:
    """os.kill(pid, 0) signals nothing and raises once the pid is gone."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def test_deleting_a_run_kills_the_whole_process_group(tmp_path, monkeypatch):
    """The half of cancellation that actually stops a two-hour nnUNet.

    Everything here is real: a tool in its own interpreter, a worker it forked,
    a POST blocked in another thread, and a DELETE that holds no handle on any
    of it -- the process group id travelled through the run directory, which is
    why a uvicorn worker that never started the run can still stop it.
    """
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    _install_sleeper(tools_dir)
    pid_file = tmp_path / "pids"

    monkeypatch.setattr(settings, "TOOLS_DIR", str(tools_dir))
    monkeypatch.setattr(settings, "SADT_DISPATCH_MODE", config.DISPATCH_SUBPROCESS)
    # A bound, so a failure to kill fails the test instead of hanging the suite.
    monkeypatch.setattr(settings, "TOOL_TIMEOUT_SECONDS", 60)
    monkeypatch.setitem(registry.TOOLS, SleeperTool.name, SleeperTool())

    identifier = _new_id()
    answers = []

    with TestClient(app) as shared:
        def post():
            answers.append(
                shared.post(
                    f"/run/{SleeperTool.name}",
                    headers=dict(AUTH, **{"X-Run-Id": identifier}),
                    data={"pid_file": str(pid_file)},
                )
            )

        caller = threading.Thread(target=post)
        caller.start()
        try:
            deadline = time.monotonic() + 30
            while not pid_file.is_file() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert pid_file.is_file(), "the tool never started, so this proves nothing"
            tool_pid, worker_pid = (int(part) for part in pid_file.read_text().split())

            assert shared.delete(f"/runs/{identifier}", headers=AUTH).status_code == 204
            caller.join(timeout=60)
        finally:
            for pid in locals().get("tool_pid", 0), locals().get("worker_pid", 0):
                if pid and _is_alive(pid):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass

    assert not caller.is_alive(), "the run never returned after the cancel"
    assert answers and answers[0].status_code == 499
    assert answers[0].json() == {"detail": "Run cancelled by the client."}

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and (_is_alive(tool_pid) or _is_alive(worker_pid)):
        time.sleep(0.1)
    assert not _is_alive(tool_pid), "the tool process survived the cancel"
    assert not _is_alive(worker_pid), "a forked worker survived: killpg missed the group"


# ----------------------------------------------------------------------
# The two phases dispatch owns
# ----------------------------------------------------------------------

def test_the_gpu_queue_is_announced_only_when_the_run_wants_the_card(
    probe_tool, probe_python, tracked_scratch_dirs, run_id, monkeypatch
):
    """A multi-minute wait for the card is today indistinguishable from a tool
    that is running, which is the most confusing thing a panel can show. A
    tabular prediction that never queues must not claim it did."""
    monkeypatch.setattr(dispatch, "_execute", lambda *args, **kwargs: 0)
    monkeypatch.setattr(dispatch, "_read_result", lambda *args, **kwargs: "ok")
    monkeypatch.setattr(settings, "DEVICE", "cpu")

    class CpuTool(Tool):
        name = "_dispatch_probe"
        arguments = {"device": ArgSpec(type=str, required=False, initial="cpu")}
        output_kind = "text"

        def run(self, device: str = "cpu"):
            raise AssertionError("dispatched, not run")

    token = runs.CURRENT_RUN.set(run_id)
    try:
        dispatch.dispatch(probe_tool, {"a": 1, "b": 1})
        with_card = [event["phase"] for event in runs.read_events(run_id)]
        dispatch.dispatch(CpuTool(), {"device": "cpu"})
        both = [event["phase"] for event in runs.read_events(run_id)]
    finally:
        runs.CURRENT_RUN.reset(token)

    # Stronger than it used to be. The phase was announced before the wait, so
    # a run that was admitted instantly still claimed to have queued; it is now
    # emitted only when the run actually had to wait for room, and neither of
    # these did.
    assert with_card == []
    assert both == [], "a run that never waited said it was queueing"


def test_a_tool_is_told_where_to_report_its_progress(run_id):
    """An absolute path, and only when the run has a directory to hold it."""
    environment = dispatch._child_environment(
        "job1", "/jobs/job1", None, runs.progress_file(run_id)
    )

    path = environment[runs.PROGRESS_FILE_ENV]
    assert os.path.isabs(path)
    assert path == os.path.join(settings.TEMP_DIR, "runs", run_id, runs.EVENTS_FILE)


def test_a_run_nobody_is_watching_carries_no_progress_file(monkeypatch):
    """A stale value inherited from the server's own environment would send a
    tool's progress into a file belonging to nothing."""
    monkeypatch.setenv(runs.PROGRESS_FILE_ENV, "/somewhere/stale/events.jsonl")

    environment = dispatch._child_environment("job1", "/jobs/job1", None, None)

    assert runs.PROGRESS_FILE_ENV not in environment


def test_a_nested_marker_names_the_tool_it_called(run_id):
    """The supervisor's marker is what makes a chain visible.

    ASO drives ALI_CBCT for most of its wall clock, and before this the only
    trace was a depth-0 log MESSAGE -- which anything reading a run's events
    drops, messages being free text a tool wrote. Measured on the campaign's
    two chain arms: 56 events across seven runs, every one at depth 0.
    """
    directory = os.path.join(settings.TEMP_DIR, "runs", run_id)
    path = os.path.join(directory, runs.EVENTS_FILE)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"depth": 1, "tool": "ALI_CBCT"}) + "\n")

    event = runs.read_events(run_id)[-1]

    assert event["depth"] == 1
    assert event["tool"] == "ALI_CBCT"
    # Still a tool's line, so still the server's vocabulary for the rest of it:
    # a marker says WHICH tool and HOW DEEP, never what phase the run is in.
    assert event["phase"] == runs.PHASE_RUNNING


def test_a_tool_name_that_is_not_an_identifier_is_dropped_whole(run_id):
    """This field travels where a message does not.

    The benchmark payload drops messages precisely because a tool writes them
    and its free text may name a patient's file; it keeps `tool` so a chain's
    bar can say which tool ran. That trade only holds while the field cannot
    BE free text -- so anything unlike a tool folder's name is dropped rather
    than truncated, a truncated file name being still a file name.
    """
    directory = os.path.join(settings.TEMP_DIR, "runs", run_id)
    path = os.path.join(directory, runs.EVENTS_FILE)
    refused = [
        "/DATA/ALI/testfiles/patient_0042.nii.gz",
        "ALI_CBCT on patient_0042",
        "../../etc/passwd",
        "A" * 65,
        "",
        None,
        {"tool": "ALI_CBCT"},
    ]
    with open(path, "a", encoding="utf-8") as handle:
        for name in refused:
            handle.write(json.dumps({"depth": 1, "tool": name}) + "\n")

    events = runs.read_events(run_id)[-len(refused):]

    assert [event.get("tool") for event in events] == [None] * len(refused)
    # Dropped, not the run: a marker nobody can read is still a nested call.
    assert all(event["depth"] == 1 for event in events)


def test_a_run_reports_what_it_itself_cost(run_id):
    """Per-RUN, where the cost table is per-TOOL.

    With six runs sharing one card, a trace of the card lines a peak up with
    every run that could have caused it and attributes it to none. This is the
    figure for one run, which is what a reader of one run can actually use.
    """
    runs.append(run_id, runs.PHASE_RUNNING, measured={
        "vram_bytes": 1 << 30, "ram_bytes": 2 << 30,
        "cpu_cores": 3.5, "channels": 4})

    event = runs.read_events(run_id)[-1]

    assert event["measured"] == {"vram_bytes": 1 << 30, "ram_bytes": 2 << 30,
                                 "cpu_cores": 3.5, "channels": 4}


def test_a_measurement_a_tool_wrote_itself_is_refused(run_id):
    """A tool able to write its own cost could tell the budget it is free.

    Same rule as `phase` and `result`: only a record this server wrote may
    carry one, and a tool appends to the very same file.
    """
    directory = os.path.join(settings.TEMP_DIR, "runs", run_id)
    with open(os.path.join(directory, runs.EVENTS_FILE), "a", encoding="utf-8") as h:
        h.write(json.dumps({"measured": {"vram_bytes": 0, "ram_bytes": 0}}) + "\n")

    assert "measured" not in runs.read_events(run_id)[-1]


def test_a_measurement_carries_four_numbers_and_nothing_else(run_id):
    """Whitelisted rather than passed through: this rides an event a browser
    renders, so the shape is stated rather than inherited from a caller."""
    runs.append(run_id, runs.PHASE_RUNNING, measured={
        "vram_bytes": 1 << 30,
        "scan": "/DATA/ALI/testfiles/patient_0042.nii.gz",
        "ram_bytes": float("inf"),
        "cpu_cores": -1,
        "channels": True,
    })

    measured = runs.read_events(run_id)[-1]["measured"]

    # The one usable number survives; an infinity, a negative, a bool dressed
    # as an int and a path are each dropped rather than clamped.
    assert measured == {"vram_bytes": 1 << 30}


def test_nothing_measurable_is_not_an_empty_measurement(run_id):
    """"Not measured" and "measured as empty" must not render the same."""
    runs.append(run_id, runs.PHASE_RUNNING, measured={"scan": "patient.nii.gz"})
    assert "measured" not in runs.read_events(run_id)[-1]
