"""Tool discovery: one broken tool must never take the whole server down.

With 15+ tools planned, a single missing dependency or typo used to make the
server refuse to boot -- every other model unavailable with it. Discovery is
lenient instead: the failure is logged loudly, recorded in FAILED_TOOLS, and
the tool is skipped.
"""

import contextlib
import importlib
import os
import shutil
import sys

import pytest

# Set before importing main, so config.Settings() picks up a known token
# regardless of whatever is in the developer's local .env.
os.environ.setdefault("API_TOKEN", "test-token")

import registry
import tools as tools_package

TOOLS_DIR = tools_package.__path__[0]


@pytest.fixture(autouse=True)
def restore_failed_tools():
    """_build_registry() writes to the module-level FAILED_TOOLS, so a test
    rebuilding the registry would otherwise leak into the next one."""
    saved = dict(registry.FAILED_TOOLS)
    yield
    registry.FAILED_TOOLS.clear()
    registry.FAILED_TOOLS.update(saved)


@contextlib.contextmanager
def temporary_tool(folder: str, source: str = None):
    """Drop a real tool folder into tools/ for the duration of the test.

    Discovery walks the filesystem and imports by module name, so there is no
    honest way to exercise it without a folder actually being there.
    `source=None` creates the folder WITHOUT its <folder>.py, which is its own
    failure mode.
    """
    path = os.path.join(TOOLS_DIR, folder)
    os.makedirs(path)
    try:
        open(os.path.join(path, "__init__.py"), "w").close()
        if source is not None:
            with open(os.path.join(path, f"{folder}.py"), "w") as handle:
                handle.write(source)
        importlib.invalidate_caches()
        yield
    finally:
        shutil.rmtree(path, ignore_errors=True)
        prefix = f"{tools_package.__name__}.{folder}"
        for module in [name for name in sys.modules if name.startswith(prefix)]:
            del sys.modules[module]
        importlib.invalidate_caches()


def test_a_broken_import_does_not_stop_the_other_tools(caplog):
    with temporary_tool("zz_broken_probe", "import a_module_that_does_not_exist\n"):
        rebuilt = registry._build_registry()

    assert "zz_broken_probe" in registry.FAILED_TOOLS
    assert "ModuleNotFoundError" in registry.FAILED_TOOLS["zz_broken_probe"]
    # The healthy tools are all still registered.
    assert {"Test_Tool", "Example_Tool"} <= set(rebuilt)
    # And the failure is impossible to miss in the console.
    assert "TOOL FAILED TO LOAD" in caplog.text
    assert "zz_broken_probe" in caplog.text
    assert "a_module_that_does_not_exist" in caplog.text  # the traceback is there too


def test_a_tool_raising_at_import_is_skipped():
    source = 'raise RuntimeError("model weights not found")\n'
    with temporary_tool("zz_raising_probe", source):
        rebuilt = registry._build_registry()

    assert "model weights not found" in registry.FAILED_TOOLS["zz_raising_probe"]
    assert "Test_Tool" in rebuilt


def test_an_invalid_schema_is_skipped_not_fatal():
    """check_schema() runs at startup; before, it aborted the whole registry."""
    source = (
        "from base import ArgSpec, Tool\n"
        "\n"
        "class BadSchemaTool(Tool):\n"
        "    name = 'zz_bad_schema_probe'\n"
        "    arguments = {'opt': ArgSpec(type='choice')}\n"  # no choices declared
        "\n"
        "    def run(self):\n"
        "        return None\n"
    )
    with temporary_tool("zz_bad_schema_probe", source):
        rebuilt = registry._build_registry()

    assert "ToolSchemaError" in registry.FAILED_TOOLS["zz_bad_schema_probe"]
    assert "zz_bad_schema_probe" not in rebuilt
    assert "Test_Tool" in rebuilt


def test_a_folder_without_its_module_is_skipped_not_fatal():
    with temporary_tool("zz_no_module_probe", source=None):
        rebuilt = registry._build_registry()

    assert "is missing its" in registry.FAILED_TOOLS["zz_no_module_probe"]
    assert "Test_Tool" in rebuilt


def test_a_folder_git_could_not_delete_is_not_a_tool_that_failed():
    """The exact state a rename leaves on an existing checkout.

    tools/example_tool/ became tools/Example_Tool/, and git removed the tracked
    files but not the directory, because pytest had left a __pycache__/ inside
    it. Reported as a failure, that is a startup banner claiming two tools are
    unavailable on a deployment where every tool loaded -- and a red suite for
    anyone who pulls, which the pre-push hook turns into a blocked push.
    """
    path = os.path.join(TOOLS_DIR, "zz_leftover_probe")
    os.makedirs(os.path.join(path, "__pycache__"))
    try:
        open(os.path.join(path, "__pycache__", "example_tool.cpython-310.pyc"), "w").close()
        rebuilt = registry._build_registry()
    finally:
        shutil.rmtree(path, ignore_errors=True)

    assert "zz_leftover_probe" not in registry.FAILED_TOOLS
    assert "Test_Tool" in rebuilt


def test_get_tool_distinguishes_a_failed_tool_from_an_unknown_one():
    """"Unknown tool" on something that exists in the source tree reads like a
    typo and sends the client developer looking in the wrong place."""
    registry.FAILED_TOOLS["zz_failed_probe"] = "ImportError: no module named 'torch'"

    with pytest.raises(KeyError, match="failed to load at server startup"):
        registry.get_tool("zz_failed_probe")
    with pytest.raises(KeyError, match="Unknown tool"):
        registry.get_tool("zz_never_existed")


def test_running_a_failed_tool_is_404_with_a_useful_message():
    from fastapi.testclient import TestClient

    from main import app

    registry.FAILED_TOOLS["zz_failed_probe"] = "ImportError: no module named 'torch'"
    response = TestClient(app).post(
        "/run/zz_failed_probe",
        headers={"Authorization": f"Bearer {os.environ['API_TOKEN']}"},
        data={"anything": "1"},
    )

    assert response.status_code == 404
    assert "failed to load" in response.json()["detail"]


def test_every_tool_in_the_repo_actually_loads():
    """The counterpart to lenient discovery: since a broken tool no longer
    stops the server, nothing else would notice it went missing. This is what
    turns a silent skip back into a red build.

    A failure here with a ModuleNotFoundError usually means the environment is
    stale -- `pip install -r requirements.txt` -- not that the tool is broken.
    """
    assert registry.FAILED_TOOLS == {}, "tool(s) failed to load: " + "; ".join(
        f"{folder} ({reason})" for folder, reason in sorted(registry.FAILED_TOOLS.items())
    )


# ---------------------------------------------------------------------------
# A name written two ways


def test_a_tool_is_found_under_a_different_spelling_of_its_name():
    """`SurgMovPred` became `Surg_Mov_Pred` when the tool was packaged.

    Every Slicer module naming the old spelling got a 404 reading "Unknown
    tool", which is what a typo returns -- so a cosmetic rename on this side
    looked, from the panel, like the client asking for something that never
    existed. CLAUDE.md already described names as compared case- and
    separator-insensitively; the code did neither, and nothing tested it.
    """
    name = next(iter(registry.TOOLS))
    spelled_differently = name.replace("_", "").lower()

    assert registry.get_tool(spelled_differently) is registry.TOOLS[name]


def test_a_name_that_matches_nothing_is_still_unknown():
    """Forgiving about separators, not about the name."""
    with pytest.raises(KeyError, match="Unknown tool"):
        registry.get_tool("Nonexistent_Tool")


def test_two_spellings_of_one_name_cannot_both_be_served():
    """The other half of the same rule, and the reason the lookup is safe: at
    most one tool can match a canonical key, so there is nothing to
    disambiguate at request time."""
    served = {"Batch_Dental_Seg": object()}

    with pytest.raises(RuntimeError, match="same tool written two ways"):
        registry._reject_duplicate("BatchDentalSeg", served)


def test_a_genuinely_different_name_is_not_a_duplicate():
    served = {"ALI_CBCT": object()}
    registry._reject_duplicate("ALI_IOS", served)  # must not raise
