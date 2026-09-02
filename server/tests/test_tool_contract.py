"""The contract with SADT-VISOR, tested against what that repository actually
emits rather than against the prose describing it.

Every case here comes from a real difference found by reading the packaged
tools: a hash computed the other way round, a package named `sadt_<tool>`, an
`output_dir` no client can supply, `Literal[...]` options that arrive as
`choices`, a `device` argument that used to be a setting, and an error whose
class NAME is the only thing that says which status code to answer.
"""

import json
import os

import pytest

os.environ.setdefault("API_TOKEN", "test-token")

from registry import deployment
from execution import dispatch
import main
from registry import schema_hash
from registry import schema_tool
from base import CHOICE_TYPE, MULTICHOICE_TYPE, Selection
from config import settings
from registry.deployment import DeploymentConfig, ToolDeployment


# ----------------------------------------------------------------------
# What the generator emits
# ----------------------------------------------------------------------

def test_the_hash_matches_the_algorithm_that_generates_it():
    """Ported from SADT-VISOR's scripts/describe.py. The two sides disagreeing
    by one separator means every tool looks stale -- which is what happened,
    and was only caught by hashing a real tool both ways.

    The digest is: <relative posix path>\\0<the file's bytes>\\0 per file, in
    PATH order, sha256 over the lot.
    """
    import hashlib
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        src = pathlib.Path(root)
        (src / "pkg").mkdir()
        (src / "pkg" / "__init__.py").write_bytes(b"def run():\n    pass\n")
        (src / "pkg" / "helper.py").write_bytes(b"X = 1\n")
        (src / "top.py").write_bytes(b"# top\n")

        expected = hashlib.sha256()
        for path in sorted(p for p in src.rglob("*") if p.is_file()):
            expected.update(path.relative_to(src).as_posix().encode())
            expected.update(b"\0")
            expected.update(path.read_bytes())
            expected.update(b"\0")

        assert schema_hash.hash_source_tree(str(src)) == expected.hexdigest()


def test_literal_options_become_a_picker(make_tool_folder):
    """`Literal[...]` in the signature is published as `choices`, and the two
    widgets fall out of whether it is a list. Without this a clinician gets a
    free-text field where AMASSS used to show nine check boxes."""
    folder = make_tool_folder(
        "picky",
        arguments={
            "structures": {
                "type": "list[str]",
                "required": False,
                "default": ["MAND", "MAX"],
                "choices": ["MAND", "MAX", "CB", "SKIN"],
            },
            "device": {
                "type": "str",
                "required": False,
                "default": "cuda",
                "choices": ["cuda", "cpu"],
            },
        },
    )

    tool = schema_tool.load_tool(folder, DeploymentConfig({}))

    structures = tool.arguments["structures"]
    assert structures.types[0] == MULTICHOICE_TYPE
    assert structures.choices == {"MAND": True, "MAX": True, "CB": False, "SKIN": False}

    device = tool.arguments["device"]
    assert device.types[0] == CHOICE_TYPE
    assert device.choices == {"cuda": True, "cpu": False}


def test_a_selection_reaches_the_tool_as_the_list_it_declared(make_tool_folder):
    """An imported tool's run() took a Selection -- every option mapped to
    true/false. A packaged one declares `list[Literal[...]]`, so it takes the
    enabled options and nothing else; sending the mapping would hand a dict to
    a parameter annotated as a list."""
    folder = make_tool_folder(
        "listy_choice",
        arguments={
            "structures": {
                "type": "list[str]",
                "required": False,
                "default": ["MAND"],
                "choices": ["MAND", "MAX", "CB"],
            }
        },
    )
    tool = schema_tool.load_tool(folder, DeploymentConfig({}))

    cleaned = tool.validate({"structures": "MAND,CB"})
    assert isinstance(cleaned["structures"], Selection)

    assert tool.for_the_wire(cleaned) == {"structures": ["MAND", "CB"]}


def test_the_return_of_a_packaged_tool_is_always_archived(make_tool_folder):
    """`path` is the output DIRECTORY the tool was given, `dict[str, path]` is
    several named files. Both go back as an archive; neither is one file to
    stream."""
    for declared in ("path", "dict[str, path]"):
        folder = make_tool_folder(f"ret_{declared.replace('[', '').replace(']', '').replace(', ', '_')}",
                                  returns=declared)
        assert schema_tool.load_tool(folder, DeploymentConfig({})).output_kind == "files"


# ----------------------------------------------------------------------
# output_dir
# ----------------------------------------------------------------------

def test_the_output_directory_is_never_published(make_tool_folder):
    """A client offered a file picker for a directory on the server would send
    something meaningless; offered nothing, every run would be a 422 for a
    missing required argument."""
    folder = make_tool_folder(
        "writes",
        arguments={
            "scans": {"type": "path", "required": True},
            "output_dir": {"type": "path", "required": True},
        },
    )

    tool = schema_tool.load_tool(folder, DeploymentConfig({}))

    assert "output_dir" not in tool.arguments
    assert tool.wants_output_dir is True


def test_the_output_directory_is_filled_in_by_the_server(probe_tool, probe_python, tmp_path):
    job_dir = str(tmp_path)
    os.makedirs(os.path.join(job_dir, "output"), exist_ok=True)
    probe_tool.wants_output_dir = True

    filled = dispatch._server_provided(probe_tool, {"a": 1}, job_dir)

    assert filled["output_dir"] == os.path.join(job_dir, "output")
    assert os.path.isdir(filled["output_dir"])


def test_a_tool_that_takes_no_output_directory_is_not_given_one(probe_tool, tmp_path):
    filled = dispatch._server_provided(probe_tool, {"a": 1}, str(tmp_path))

    assert "output_dir" not in filled


# ----------------------------------------------------------------------
# The GPU, which nothing serialises any more
# ----------------------------------------------------------------------

def _gpu_tool(make_tool_folder, name, default="cuda"):
    folder = make_tool_folder(
        name,
        arguments={
            "scans": {"type": "path", "required": True},
            "device": {
                "type": "str",
                "required": False,
                "default": default,
                "choices": ["cuda", "cpu"],
            },
        },
    )
    return schema_tool.load_tool(folder, DeploymentConfig({}))


def test_a_tool_declaring_a_device_is_gpu_work(make_tool_folder):
    """Every tool used to hold its own semaphore, which worked only because
    they shared one process. A packaged tool is its own process, so the server
    holds the limit -- and it has to be one counter across tools, since an
    AMASSS run and a CrownSeg run want the same card."""
    tool = _gpu_tool(make_tool_folder, "gpu_tool")

    assert dispatch.uses_the_gpu(tool, {"scans": "/x"}) is True
    assert dispatch.uses_the_gpu(tool, {"scans": "/x", "device": "cuda:1"}) is True
    assert dispatch.uses_the_gpu(tool, {"scans": "/x", "device": "cpu"}) is False


def test_a_tool_that_says_nothing_is_assumed_to_use_the_card(probe_tool):
    """The safe default is the strict one: a tool that imports torch without
    declaring `device` -- and that was every tool until they were packaged --
    would otherwise never queue, and two of them would meet on the card."""
    assert dispatch.uses_the_gpu(probe_tool, {"a": 1}) is True


def test_the_deployment_decides_the_device_when_the_caller_does_not(
    make_tool_folder, monkeypatch, tmp_path
):
    """`settings.DEVICE` used to be read inside each tool. A tool that no
    longer reads the environment would otherwise always run on its own default
    -- cuda -- on a server configured for CPU."""
    monkeypatch.setattr(settings, "DEVICE", "cpu")
    tool = _gpu_tool(make_tool_folder, "device_tool")

    filled = dispatch._server_provided(tool, {"scans": "/x"}, str(tmp_path))

    assert filled["device"] == "cpu"
    # And THIS is how a run opts out of the queue: by saying where it runs.
    assert dispatch.uses_the_gpu(tool, filled) is False


def test_a_device_the_caller_picked_is_kept(make_tool_folder, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEVICE", "cpu")
    tool = _gpu_tool(make_tool_folder, "explicit_device")

    filled = dispatch._server_provided(tool, {"scans": "/x", "device": "cuda"}, str(tmp_path))

    assert filled["device"] == "cuda"


# ----------------------------------------------------------------------
# Errors, mapped by class name
# ----------------------------------------------------------------------

def test_the_error_map_covers_what_the_tools_raise():
    """There is no shared exception type, because there is no shared package.
    These four names are the convention SADT-VISOR documents."""
    assert main.TOOL_ERROR_STATUS["ToolInputError"] == 422
    assert main.TOOL_ERROR_STATUS["ValueError"] == 422
    assert main.TOOL_ERROR_STATUS["FileNotFoundError"] == 422
    assert main.TOOL_ERROR_STATUS["ToolUnavailableError"] == 503


@pytest.mark.parametrize(
    "error_type, expected_status, message_travels",
    [
        ("ToolInputError", 422, True),
        ("FileNotFoundError", 422, True),
        ("ToolUnavailableError", 503, True),
        ("KeyError", 500, False),
        ("RuntimeError", 500, False),
    ],
)
def test_a_tool_failure_becomes_the_right_status(
    probe_tool, probe_name, probe_python, subprocess_mode, monkeypatch,
    error_type, expected_status, message_travels,
):
    """A message written to be read by whoever sent the request travels; a
    crash inside a tool does not, because it can name server-side paths."""
    from fastapi.testclient import TestClient

    import registry
    from main import app

    monkeypatch.setitem(registry.TOOLS, probe_name, probe_tool)
    secret = "the scan at /DATA/patients/x.nii.gz is unreadable"
    monkeypatch.setattr(
        dispatch,
        "dispatch",
        lambda *args, **kwargs: (_ for _ in ()).throw(dispatch.ToolFailure(error_type, secret)),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            f"/run/{probe_name}",
            headers={"Authorization": "Bearer test-token"},
            data={"a": "1", "b": "1"},
        )

    assert response.status_code == expected_status
    if message_travels:
        assert response.json()["detail"] == secret
    else:
        assert response.json() == {"detail": "Tool execution failed."}
        assert secret not in response.text


def test_the_runner_records_the_exception_class(probe_python, probe_name, tmp_path):
    """End to end: the tool raises, the runner writes the class name, and
    dispatch turns it into something main.py can map."""
    job_dir = tmp_path / "job"
    (job_dir / "output").mkdir(parents=True)
    (job_dir / "result.json").write_text(
        json.dumps({"error": {"type": "ToolInputError", "message": "Unknown structure code"}})
    )

    with pytest.raises(dispatch.ToolFailure) as failure:
        dispatch._read_result(str(job_dir), probe_name)

    assert failure.value.error_type == "ToolInputError"
    assert failure.value.message == "Unknown structure code"


# ----------------------------------------------------------------------
# Where the data lives
# ----------------------------------------------------------------------

def test_the_data_folder_can_be_named_differently_from_the_tool():
    """Packaged tools are lowercase (`amasss`); the bundles staged under DATA/
    are not (`AMASSS/`). A case-insensitive lookup would be a guess -- on a
    case-sensitive filesystem both can exist."""
    config = DeploymentConfig({"amasss": ToolDeployment(data_dir="AMASSS")})

    assert config.data_slug("amasss") == "AMASSS"
    assert config.data_slug("crownseg") == "crownseg"


def test_the_data_folder_is_validated_at_startup(tmp_path):
    path = tmp_path / "deployment.toml"
    path.write_text("[tools.amasss]\ndata_dir = 42\n")

    with pytest.raises(deployment.DeploymentConfigError, match="data_dir"):
        deployment.load(str(path))


# ----------------------------------------------------------------------
# Archives: no tool unpacks one any more
# ----------------------------------------------------------------------

def test_a_zip_sent_for_a_path_argument_is_unpacked():
    """Each tool used to carry its own extraction, zip-bomb cap and scratch
    directory for it. They have all dropped it, so the server unpacks before
    run() is called -- and a `path` argument is what a packaged tool declares
    for every input."""
    assert main._unpacks_to_a_folder("path", ".zip") is True
    assert main._unpacks_to_a_folder("path", ".nii.gz") is False
    assert main._unpacks_to_a_folder("folder", ".zip") is True
    # A generic "file" argument is NOT a packaged tool's path: an imported tool
    # declaring one may legitimately want the archive itself.
    assert main._unpacks_to_a_folder("file", ".zip") is False


# ----------------------------------------------------------------------
# Against the real repository, when it is here
# ----------------------------------------------------------------------

SADT_TOOLS_REPO = os.environ.get("SADT_TOOLS_REPO", os.path.expanduser("~/code/SADT-VISOR"))


def _packaged_tools() -> list:
    tools = os.path.join(SADT_TOOLS_REPO, "tools")
    if not os.path.isdir(tools) or not os.path.isfile(
        os.path.join(SADT_TOOLS_REPO, "scripts", "describe.py")
    ):
        return []
    return [
        name
        for name in sorted(os.listdir(tools))
        if not name.startswith("_")
        # Only the ones whose virtualenv exists: describe.py has to run with
        # the tool's own interpreter, and a tool that has not been synced yet
        # is not something this server could serve either.
        and os.path.isfile(os.path.join(tools, name, ".venv", "bin", "python"))
    ]


@pytest.mark.skipif(not _packaged_tools(), reason="the SADT-VISOR checkout is not here")
@pytest.mark.parametrize("tool_name", _packaged_tools())
def test_a_really_packaged_tool_loads(tool_name, tmp_path, monkeypatch):
    """The one test that would have caught every difference found by reading
    that repository: the hash algorithm, the `sadt_<name>` package, `choices`,
    `output_dir`, `dict[str, path]`.

    It runs THEIR generator with THAT tool's interpreter -- exactly what the
    image does at build time -- and loads the result the way the registry does.
    Skipped wherever the checkout is absent, so it never fails a clean clone.
    """
    import shutil
    import subprocess

    source = os.path.join(SADT_TOOLS_REPO, "tools", tool_name)
    described = subprocess.run(
        [
            os.path.join(source, ".venv", "bin", "python"),
            os.path.join(SADT_TOOLS_REPO, "scripts", "describe.py"),
            source,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert described.returncode == 0, described.stderr

    folder = tmp_path / tool_name
    shutil.copytree(
        os.path.join(source, "src"),
        folder / "src",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (folder / ".schema.json").write_text(described.stdout)
    monkeypatch.setattr(settings, "TOOLS_DIR", str(tmp_path))

    tool = schema_tool.load_tool(str(folder), DeploymentConfig({}))

    assert tool.name == tool_name
    # Everything a packaged tool has in common, whatever it computes.
    assert tool.wants_output_dir, "every tool takes output_dir, and the server fills it in"
    assert "output_dir" not in tool.arguments, "and never publishes it"
    assert tool.output_kind == "files"
    # And the schema the server checked is the one that source produces.
    assert json.loads(described.stdout)["source_hash"] == schema_hash.hash_source_tree(
        str(folder / "src")
    )
