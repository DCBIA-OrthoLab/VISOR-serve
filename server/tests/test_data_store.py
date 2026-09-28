"""Unit tests for LocalDataStore: server-side models/test files are listed
and resolved whether they are plain files (e.g. a zip archive) or whole
folders (e.g. an unpacked model directory served directly to the tool).

Run with: cd server && pytest tests/test_data_store.py
"""

import os

os.environ.setdefault("API_TOKEN", "test-token")

import pytest

from data_store import DataNotFoundError, LocalDataStore


@pytest.fixture
def store(tmp_path):
    models = tmp_path / "SomeTool" / "models"
    models.mkdir(parents=True)
    # A packaged model (single file)...
    (models / "packaged_model.zip").write_bytes(b"PK\x03\x04")
    # ...and an unpacked one (folder), the usual layout under DATA/.
    folder = models / "folder_model"
    folder.mkdir()
    (folder / "stacking_package.pkl").write_bytes(b"pkl")
    # Junk left behind by macOS archives: must never be listed as a model.
    (models / ".DS_Store").write_bytes(b"junk")
    (models / "__MACOSX").mkdir()
    return LocalDataStore(str(tmp_path))


def test_list_models_includes_folders_and_files_but_not_junk(store):
    assert store.list_models("SomeTool") == ["folder_model", "packaged_model.zip"]


def test_list_models_unknown_tool_is_empty(store):
    assert store.list_models("no_such_tool") == []


def test_resolve_model_returns_folder_path(store):
    resolved = store.resolve_model("SomeTool", "folder_model")
    assert os.path.isdir(resolved.path)
    assert not resolved.is_temporary


def test_resolve_model_returns_file_path(store):
    resolved = store.resolve_model("SomeTool", "packaged_model.zip")
    assert os.path.isfile(resolved.path)


def test_resolve_model_rejects_path_traversal(store):
    with pytest.raises(DataNotFoundError):
        store.resolve_model("SomeTool", "../models")


def test_resolve_unknown_model_raises(store):
    with pytest.raises(DataNotFoundError):
        store.resolve_model("SomeTool", "missing")


# --- what a name cannot say ------------------------------------------------
#
# The client downloads what a user picks now, so the picker has to show what
# the click costs -- and a folder used to look like a file whose extension was
# missing.

def test_describe_says_file_or_folder(tmp_path, monkeypatch):
    root = tmp_path / "DATA"
    (root / "Probe" / "testfiles" / "a_folder").mkdir(parents=True)
    (root / "Probe" / "testfiles" / "a_folder" / "inner.nii.gz").write_bytes(b"x" * 10)
    (root / "Probe" / "testfiles" / "a_file.nii.gz").write_bytes(b"y" * 7)

    store = LocalDataStore(str(root))
    described = {row["name"]: row for row in store.describe("Probe", "testfiles")}

    assert described["a_folder"]["kind"] == "folder"
    assert described["a_file.nii.gz"]["kind"] == "file"


def test_describe_sizes_a_folder_by_what_it_holds(tmp_path):
    root = tmp_path / "DATA"
    cohort = root / "Probe" / "testfiles" / "cohort"
    (cohort / "nested").mkdir(parents=True)
    (cohort / "one.nii.gz").write_bytes(b"x" * 100)
    (cohort / "nested" / "two.nii.gz").write_bytes(b"y" * 250)

    store = LocalDataStore(str(root))
    described = {row["name"]: row for row in store.describe("Probe", "testfiles")}

    # What a download costs is everything inside, not the directory's own inode.
    assert described["cohort"]["size"] == 350


def test_describe_returns_the_same_names_as_the_plain_listing(tmp_path):
    root = tmp_path / "DATA"
    (root / "Probe" / "testfiles").mkdir(parents=True)
    for name in ("b.nii.gz", "a.nii.gz"):
        (root / "Probe" / "testfiles" / name).write_bytes(b"x")

    store = LocalDataStore(str(root))

    assert [row["name"] for row in store.describe("Probe", "testfiles")] == \
        store.list_testfiles("Probe")


def test_describe_of_a_tool_with_no_data_is_empty(tmp_path):
    store = LocalDataStore(str(tmp_path))
    assert store.describe("Absent", "testfiles") == []
    assert store.describe("Absent", "models") == []


# ---------------------------------------------------------------------------
# Scoped hosted files: one folder per modality, one argument drawing from one
# ---------------------------------------------------------------------------
@pytest.fixture
def scoped(tmp_path):
    """`testfiles/{CBCT,IOS}/`, the shape a tool serving two modalities needs.

    AREG registers CBCT volumes, intraoral surfaces, and one onto the other.
    Flat, its whole catalogue was offered to every argument: the CBCT baseline
    picker listed the intraoral meshes, which cannot be a baseline.
    """
    root = tmp_path / "AREG" / "testfiles"
    (root / "CBCT").mkdir(parents=True)
    (root / "IOS").mkdir(parents=True)
    (root / "CBCT" / "CBCT_FullyAuto").mkdir()
    (root / "IOS" / "IOS_test_scans").mkdir()
    # An entry left at the top level: a deployment mid-migration still works.
    (root / "loose_set").mkdir()
    return LocalDataStore(str(tmp_path))


def test_a_scope_lists_only_its_own_folder(scoped):
    assert scoped.list_testfiles("AREG", "CBCT") == ["CBCT_FullyAuto"]
    assert scoped.list_testfiles("AREG", "IOS") == ["IOS_test_scans"]


def test_no_scope_still_lists_the_folder_itself(scoped):
    """Unscoped arguments, and every existing deployment, are unchanged."""
    assert scoped.list_testfiles("AREG") == ["CBCT", "IOS", "loose_set"]


def test_a_name_resolves_inside_its_scope(scoped):
    resolved = scoped.resolve_testfile("AREG", "CBCT_FullyAuto", "CBCT")
    assert resolved.path.endswith(os.path.join("testfiles", "CBCT", "CBCT_FullyAuto"))


def test_a_name_from_another_scope_is_not_found(scoped):
    """The whole point: what one argument may pick, another may not."""
    with pytest.raises(DataNotFoundError):
        scoped.resolve_testfile("AREG", "IOS_test_scans", "CBCT")


def test_a_scope_cannot_be_escaped_by_the_name(scoped):
    """The scope comes from deployment.toml and the name from the request, so
    the name is the attacker-controlled half and stays bare."""
    with pytest.raises(DataNotFoundError):
        scoped.resolve_testfile("AREG", "../IOS/IOS_test_scans", "CBCT")
