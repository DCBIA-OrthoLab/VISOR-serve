"""Fixtures for the dispatch path: a tool folder laid out the way the
deployment image lays one out, with its own interpreter.

The probe is COPIED out of the repo and given its venv there rather than in
tools/_dispatch_probe/.venv. Two reasons, both of which have teeth: the docker
`test` service mounts server/ from the host, so a venv built inside it would
leave root-owned files behind and point at an interpreter that does not exist
on the other side of the mount; and a venv is a build artifact, which the
source tree of a repo cloned onto a second machine should not carry.

TOOLS_DIR is a setting precisely so the tool folders can live somewhere else,
which is what these fixtures use -- the layout being exercised is exactly the
real one.
"""

import json
import os
import shutil
import subprocess
import venv

import pytest

# Set before anything builds config.Settings().
os.environ.setdefault("API_TOKEN", "test-token")

import config
import file_utils
import schema_tool
from base import ArgSpec, Tool
from config import settings

PROBE_NAME = "_dispatch_probe"
PROBE_SOURCE = os.path.join(settings.TOOLS_DIR, PROBE_NAME)


class ProbeTool(Tool):
    """Declared here rather than under tools/: the probe must stay invisible to
    the registry, or it would appear in GET /tools."""

    name = PROBE_NAME
    arguments = {
        "a": ArgSpec(type=int, description="First addend"),
        "b": ArgSpec(type=int, description="Second addend"),
        "fail": ArgSpec(type=bool, required=False, description="Raise instead of running"),
    }

    def run(self, a: int, b: int, fail: bool = False):
        raise AssertionError("run() must not be called on the subprocess path")


def _build_venv(venv_dir: str) -> None:
    """uv when it is installed -- what the deployment image uses -- otherwise
    the standard library's venv without pip. The probe imports nothing, so
    there is nothing to install and no network to reach."""
    if shutil.which("uv"):
        subprocess.run(["uv", "venv", venv_dir], check=True, capture_output=True)
    else:
        venv.EnvBuilder(with_pip=False, symlinks=True).create(venv_dir)


def _is_usable(interpreter: str) -> bool:
    if not os.path.isfile(interpreter):
        return False
    probe = subprocess.run(
        [interpreter, "-c", "import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)"],
        capture_output=True,
    )
    return probe.returncode == 0


@pytest.fixture(scope="session")
def probe_tools_dir(tmp_path_factory) -> str:
    """A TOOLS_DIR holding one tool: <root>/_dispatch_probe/{src,.venv}."""
    root = tmp_path_factory.mktemp("tools")
    destination = os.path.join(root, PROBE_NAME)
    shutil.copytree(
        PROBE_SOURCE, destination, ignore=shutil.ignore_patterns(".venv", "__pycache__")
    )
    venv_dir = os.path.join(destination, ".venv")
    try:
        _build_venv(venv_dir)
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        pytest.skip(f"Cannot create the probe virtualenv: {exc}")

    interpreter = os.path.join(venv_dir, "bin", "python")
    if not _is_usable(interpreter):
        pytest.skip(f"The probe virtualenv at {venv_dir} has no working interpreter")
    return str(root)


@pytest.fixture
def probe_tool_dir(probe_tools_dir) -> str:
    return os.path.join(probe_tools_dir, PROBE_NAME)


@pytest.fixture
def probe_python(probe_tools_dir, monkeypatch) -> str:
    """The probe's interpreter, with the server pointed at the folder it
    belongs to."""
    monkeypatch.setattr(settings, "TOOLS_DIR", probe_tools_dir)
    return os.path.join(probe_tools_dir, PROBE_NAME, ".venv", "bin", "python")


@pytest.fixture
def probe_schema() -> dict:
    """The probe's own .schema.json, as shipped."""
    with open(os.path.join(PROBE_SOURCE, schema_tool.SCHEMA_FILE), encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def make_tool_folder(tmp_path, probe_schema):
    """Install a tool folder the way the deployment image installs one:
    <TOOLS_DIR>/<name>/{.schema.json,src/}.

    The source is the probe's, so `source_hash` is a real hash of a real tree
    and a test that wants a MISMATCH has to actually change something. No
    virtualenv: discovery never needs one, and building one per test would
    dominate the suite.
    """
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir(exist_ok=True)

    def install(folder_name: str, **schema_overrides) -> str:
        folder = tools_dir / folder_name
        shutil.copytree(
            os.path.join(PROBE_SOURCE, schema_tool.SRC_DIR_NAME),
            folder / schema_tool.SRC_DIR_NAME,
            # Compiled artifacts are excluded from the hash, so carrying them
            # into a fresh install would only make the fixture lie about what a
            # packaged tool contains.
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        schema = dict(probe_schema, name=folder_name)
        schema.update(schema_overrides)
        (folder / schema_tool.SCHEMA_FILE).write_text(json.dumps(schema))
        return str(folder)

    install.root = str(tools_dir)
    return install


@pytest.fixture
def probe_tool() -> ProbeTool:
    return ProbeTool()


@pytest.fixture
def probe_name() -> str:
    """Handed out as a fixture rather than imported: `tests` is not a package,
    so `from tests.conftest import ...` resolves only under some of pytest's
    import modes -- it works locally and fails in the docker test service."""
    return PROBE_NAME


@pytest.fixture
def subprocess_mode(monkeypatch):
    monkeypatch.setattr(settings, "SADT_DISPATCH_MODE", config.DISPATCH_SUBPROCESS)


@pytest.fixture
def tracked_scratch_dirs():
    """Stand in for the request handler, which is what removes a job directory
    once the response has been streamed."""
    directories = file_utils.track_scratch_dirs()
    yield directories
    for directory in directories:
        shutil.rmtree(directory, ignore_errors=True)
