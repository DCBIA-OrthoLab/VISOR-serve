"""Splitting a cohort into batches: what the server tells a client to send.

A folder of 20 CBCTs is ~2 GB in one archive. The card waits for its last byte,
nothing survives a connection dropped at 95%, and it is over MAX_UPLOAD_MB
anyway. The fix is that the client sends it as several runs -- and everything
in this file is the server's half of that: which argument holds the cohort, and
how much of it goes in one batch.

Nothing here executes anything. A batch is an ordinary run of an ordinary tool;
`/run` does not know this feature exists, and that is deliberate.
"""

import json
import os

import pytest

os.environ.setdefault("API_TOKEN", "test-token")

from fastapi.testclient import TestClient

import main
from config import settings
from registry import conventions, deployment, facade, schema_tool
from registry.deployment import DeploymentConfig, DeploymentConfigError, ToolDeployment

client = TestClient(main.app)

DEFAULTS = (settings.BATCH_MAX_MB, settings.BATCH_MAX_FILES)


def _write(tmp_path, text: str) -> str:
    path = tmp_path / "deployment.toml"
    path.write_text(text)
    return str(path)


def _plan(arguments: dict, declared=None, defaults=DEFAULTS):
    return conventions.batch_plan(arguments, declared or ToolDeployment(), defaults)


# ----------------------------------------------------------------------
# Which argument holds the cohort
# ----------------------------------------------------------------------

def test_the_one_required_folder_a_tool_takes_is_the_axis_a_cohort_splits_on():
    """AMASSS's own declarations, verbatim from its .schema.json.

    **Written from the RAW schema and not from what GET /tools publishes**,
    which is the shape this rule actually reads -- and where `output_dir` and
    `model` are still required `path` arguments like any other. A first version
    of these tests used the published shape, passed, and shipped a rule that
    excluded AMASSS from batching and nominated another tool's OUTPUT DIRECTORY
    as the thing to divide.
    """
    arguments = {
        "scans": {"type": "path", "required": True},
        "model": {"type": "path", "required": True},
        "output_dir": {"type": "path", "required": True},
        "structures": {"type": "list[str]", "required": False},
        "device": {"type": "str", "required": False},
    }

    assert conventions.batch_axis_for(arguments) == "scans"


def test_the_output_directory_is_never_the_thing_that_gets_divided():
    """The server fills it in with the job's own output/ and no caller supplies
    it. A tool declaring nothing else required has no cohort, not a cohort made
    of its results."""
    arguments = {
        "output_dir": {"type": "path", "required": True},
        "catalog_file": {"type": "path", "required": False},
    }

    assert conventions.batch_axis_for(arguments) is None


def test_a_hosted_model_is_not_a_cohort():
    """`model` and `*_reference` are published as a NAME picked from this
    server's DATA/ -- never a folder a client uploads, so never one it splits."""
    arguments = {
        "meshes": {"type": "path", "required": True},
        "model": {"type": "path", "required": True},
        "atlas_reference": {"type": "path", "required": True},
        "output_dir": {"type": "path", "required": True},
    }

    assert conventions.batch_axis_for(arguments) == "meshes"


def test_a_tool_pairing_two_folders_per_patient_is_never_split_by_convention():
    """The rule that matters, and the reason it is "exactly one" rather than
    "the first".

    AREG takes `t1` and `t2`, AutoCrop3D `scans` and `roi`, AutoMatrix `files`
    and `transforms`, GreedyReg `t1` and `t2`. Each pairs them per patient.
    Splitting one side without splitting the other by the same key registers a
    patient against somebody else's baseline -- and that does not raise, it
    returns a plausible volume nobody can tell is wrong.

    So no tool of this shape is ever offered for batching, and a new one is
    excluded the day it is added without anyone having had to notice.
    """
    for first, second in (("t1", "t2"), ("scans", "roi"), ("files", "transforms")):
        arguments = {
            first: {"type": "path", "required": True},
            second: {"type": "path", "required": True},
            "output_dir": {"type": "path", "required": True},
        }

        assert conventions.batch_axis_for(arguments) is None
        assert _plan(arguments) is None


def test_an_optional_second_folder_does_not_stop_a_cohort_being_split():
    """ASO's `landmarks`, GreedyReg's `masks`, FlexReg's `reference`.

    The client sends the axis in batches and every other argument whole with
    each one, so a tool matching landmarks to scans by patient name still finds
    the patient it is looking at. That costs re-sending the secondary folder
    once per batch, which is why it is only safe for the OPTIONAL ones: the
    convention admits a single required path, and what sits beside it is
    landmarks, masks and references -- kilobytes against a volume's megabytes.
    """
    arguments = {
        "input": {"type": "path", "required": True},
        "landmarks": {"type": "path", "required": False},
    }

    assert conventions.batch_axis_for(arguments) == "input"


def test_a_tool_that_takes_no_folder_at_all_is_not_split():
    """Test_Tool and Example_Tool. There is no cohort to divide."""
    assert conventions.batch_axis_for({"text_1": {"type": "str", "required": True}}) is None
    assert conventions.batch_axis_for({}) is None


def test_a_tool_whose_only_paths_are_optional_is_not_split():
    """Nothing says the run is ABOUT that file, so nothing says dividing it
    divides the work rather than breaking it."""
    arguments = {"catalog_file": {"type": "path", "required": False}}

    assert conventions.batch_axis_for(arguments) is None


# ----------------------------------------------------------------------
# How much goes in one batch
# ----------------------------------------------------------------------

def test_the_plan_carries_the_axis_and_both_caps():
    plan = _plan({"scans": {"type": "path", "required": True}}, defaults=(400, 25))

    assert plan == {"axis": "scans", "max_mb": 400, "max_files": 25}


def test_zero_on_both_caps_is_how_a_deployment_turns_batching_off():
    """One switch for the whole server, without a tool table to edit."""
    assert _plan({"scans": {"type": "path", "required": True}}, defaults=(0, 0)) is None


def test_one_cap_alone_is_still_a_usable_plan():
    """A deployment that only cares about bytes, or only about count."""
    arguments = {"scans": {"type": "path", "required": True}}

    assert _plan(arguments, defaults=(400, 0))["max_mb"] == 400
    assert _plan(arguments, defaults=(0, 25))["max_files"] == 25


# ----------------------------------------------------------------------
# deployment.toml, the exceptions
# ----------------------------------------------------------------------

def test_a_deployment_can_turn_batching_off_for_one_tool(tmp_path):
    """CNE's case, and the one input to this decision that is genuinely the
    tool's: it loads a 4.4 GB GGUF per call. Five batches of notes are five
    loads of 4.4 GB to save nothing, the notes themselves being kilobytes. The
    server cannot see that in a schema, so it is declared."""
    config = deployment.load(_write(tmp_path, '[tools.CNE]\nbatch = false\n'))

    assert config.for_tool("CNE").batch_enabled is False
    assert conventions.batch_plan(
        {"notes": {"type": "path", "required": True}},
        config.for_tool("CNE"),
        DEFAULTS,
    ) is None


def test_a_deployment_can_name_an_axis_no_convention_would_derive(tmp_path):
    """The escape hatch for a paired-folder tool, deliberately manual: whoever
    writes this line has decided how the pairing survives the split."""
    config = deployment.load(
        _write(tmp_path, '[tools.AutoCrop3D]\nbatch = { axis = "scans" }\n')
    )
    arguments = {
        "scans": {"type": "path", "required": True},
        "roi": {"type": "path", "required": True},
    }

    plan = conventions.batch_plan(arguments, config.for_tool("AutoCrop3D"), DEFAULTS)

    assert plan["axis"] == "scans"


def test_a_deployment_sizes_batches_for_the_server_and_not_per_tool(tmp_path):
    """[batch] at the top level, because what a batch protects -- bandwidth,
    disk, the upload limit -- is the machine's and not any tool's.

    In deployment.toml as well as in config.py because this file is MOUNTED:
    tuning the number here costs no container recreate, so nothing running is
    dropped to try a different value.
    """
    config = deployment.load(_write(tmp_path, "[batch]\nmax_mb = 120\nmax_files = 6\n"))

    assert config.batch_defaults == (120, 6)


def test_the_servers_own_numbers_apply_when_the_file_says_nothing(tmp_path):
    config = deployment.load(_write(tmp_path, "[tools.AMASSS]\ndata_dir = 'AMASSS'\n"))

    assert config.batch_defaults == (settings.BATCH_MAX_MB, settings.BATCH_MAX_FILES)


def test_a_tool_may_override_either_cap(tmp_path):
    config = deployment.load(
        _write(tmp_path, '[tools.AMASSS]\nbatch = { max_mb = 200 }\n')
    )

    plan = conventions.batch_plan(
        {"scans": {"type": "path", "required": True}},
        config.for_tool("AMASSS"),
        (400, 25),
    )

    assert plan == {"axis": "scans", "max_mb": 200, "max_files": 25}


# ----------------------------------------------------------------------
# A file that has moved ahead of the server reading it
# ----------------------------------------------------------------------

def test_an_unknown_batch_key_says_the_file_may_be_newer_than_the_server(tmp_path):
    """Same treatment as every other unknown value in this file: the mounted
    config can outrun the image, and an operator reading a traceback cannot
    guess that. The repair is a rebuild, not an edit."""
    path = _write(tmp_path, '[tools.AMASSS]\nbatch = { axis = "scans", size = 5 }\n')

    with pytest.raises(DeploymentConfigError) as raised:
        deployment.load(path)

    assert deployment.CONFIG_AHEAD_MARKER in str(raised.value)
    assert "size" in str(raised.value)


def test_an_unknown_key_in_the_global_table_says_the_same(tmp_path):
    path = _write(tmp_path, "[batch]\nmax_megabytes = 400\n")

    with pytest.raises(DeploymentConfigError) as raised:
        deployment.load(path)

    assert deployment.CONFIG_AHEAD_MARKER in str(raised.value)


@pytest.mark.parametrize(
    "text",
    [
        '[tools.AMASSS]\nbatch = { max_mb = -1 }\n',
        '[tools.AMASSS]\nbatch = { max_files = -5 }\n',
        '[tools.AMASSS]\nbatch = { axis = "" }\n',
        '[tools.AMASSS]\nbatch = "yes"\n',
        "[batch]\nmax_mb = -1\n",
    ],
)
def test_a_batch_declaration_that_cannot_be_honoured_is_refused_at_startup(tmp_path, text):
    """Validated when the file is read, not on the first cohort: a bad number
    here is otherwise a client quietly sending the wrong thing for weeks."""
    with pytest.raises(DeploymentConfigError):
        deployment.load(_write(tmp_path, text))


# ----------------------------------------------------------------------
# What GET /tools publishes
# ----------------------------------------------------------------------

def test_a_tool_that_cannot_be_split_publishes_no_batch_field_at_all():
    """Omitted rather than null, like every other optional hint.

    tests/golden/tools_response.json pins the published shape byte for byte
    because the Slicer client builds its whole panel from it. A tool with
    nothing to say about batching must publish exactly what it published before
    this field existed.
    """
    served = client.get("/tools").json()

    assert served, "the two in-process demos should always be served"
    for tool in served:
        assert "batch" not in tool, f"{tool['name']} takes no folder and should say nothing"


def test_the_published_plan_says_what_to_split_and_how_much_to_send(make_tool_folder):
    folder = make_tool_folder(
        "cohort_tool",
        arguments={
            "scans": {"type": "path", "required": True},
            "suffix": {"type": "str", "required": False, "default": "seg"},
        },
    )

    tool = schema_tool.load_tool(folder, DeploymentConfig({}))

    assert tool.batch == {
        "axis": "scans",
        "max_mb": settings.BATCH_MAX_MB,
        "max_files": settings.BATCH_MAX_FILES,
    }


def test_a_paired_tool_built_from_a_real_schema_carries_no_plan(make_tool_folder):
    folder = make_tool_folder(
        "paired_tool",
        arguments={
            "t1": {"type": "path", "required": True},
            "t2": {"type": "path", "required": True},
        },
    )

    assert schema_tool.load_tool(folder, DeploymentConfig({})).batch is None


# ----------------------------------------------------------------------
# Facades
# ----------------------------------------------------------------------

def test_a_facade_splits_only_where_every_mode_splits_the_same_way(make_tool_folder):
    """ALI's two engines both take one required `input`, so ALI inherits their
    plan. AREG's take `t1` and `t2` and have none, so AREG has none either.

    The mode is chosen at run time and the client splits before the run starts,
    so a facade offering a plan that holds for only one of its engines would be
    offering it for a run that has not picked an engine yet.
    """
    one_folder = {"input": {"type": "path", "required": True}}
    two_folders = {"t1": {"type": "path", "required": True},
                   "t2": {"type": "path", "required": True}}
    config = DeploymentConfig({})
    registry = {
        name: schema_tool.load_tool(make_tool_folder(name, arguments=arguments), config)
        for name, arguments in (
            ("Engine_CBCT", one_folder),
            ("Engine_IOS", one_folder),
            ("Paired_CBCT", two_folders),
        )
    }

    splittable = facade.compose(
        "Splittable", {"CBCT": "Engine_CBCT", "IOS": "Engine_IOS"}, registry)
    mixed = facade.compose(
        "Mixed", {"CBCT": "Engine_CBCT", "Paired": "Paired_CBCT"}, registry)

    assert splittable.batch["axis"] == "input"
    assert mixed.batch is None
