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
