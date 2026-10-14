"""A test file's scope is a folder this deployment declares, never a path."""

import pytest
from fastapi.testclient import TestClient

import main
from config import settings
from data_store import DataNotFoundError, LocalDataStore

client = TestClient(main.app)
AUTH = {"Authorization": f"Bearer {settings.API_TOKEN}"}


@pytest.fixture
def store(tmp_path):
    (tmp_path / "DATA" / "Tool" / "testfiles" / "Cohort").mkdir(parents=True)
    (tmp_path / "DATA" / "Tool" / "testfiles" / "Cohort" / "scan.nii.gz").write_bytes(b"ok")
    (tmp_path / "DATA" / "Tool" / "models").mkdir()
    (tmp_path / "DATA" / "Tool" / "models" / "weights.pth").write_bytes(b"model")
    (tmp_path / "secret.txt").write_text("secret")
    return LocalDataStore(str(tmp_path / "DATA"))


def test_a_declared_scope_resolves(store):
    assert store.resolve_testfile("Tool", "scan.nii.gz", "Cohort").path.endswith("scan.nii.gz")


@pytest.mark.parametrize("scope", ["../../..", "..", ".", "../models", "Cohort/../..", "/etc"])
def test_a_scope_cannot_leave_the_tools_folder(store, scope):
    for name in ("secret.txt", "weights.pth", "passwd"):
        with pytest.raises(DataNotFoundError):
            store.resolve_testfile("Tool", name, scope)


@pytest.mark.parametrize("scope", ["../../..", "../models", "../../../../../../../../etc"])
def test_the_download_route_refuses_an_undeclared_scope(scope):
    response = client.get(f"/tools/Test_Tool/testfiles/passwd?scope={scope}", headers=AUTH)
    assert response.status_code == 404
    assert "root:" not in response.text
