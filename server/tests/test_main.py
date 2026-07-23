"""Smoke tests for the tool-registry server.

Run with: cd server && ./venv/bin/pytest
(requires requirements-dev.txt: pip install -r requirements-dev.txt)
"""

import os

# Set before importing main, so config.Settings() picks up a known token
# regardless of whatever is in the developer's local .env.
os.environ["API_TOKEN"] = "test-token"

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)
TOKEN = "test-token"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_tools_lists_test_tool():
    response = client.get("/tools")
    assert response.status_code == 200
    names = [tool["name"] for tool in response.json()]
    assert "test_tool" in names


def test_run_without_token_is_401():
    response = client.post("/run/test_tool", data={"text_1": "a", "text_2": "b"})
    assert response.status_code == 401


def test_run_unknown_tool_is_404():
    response = client.post(
        "/run/does_not_exist",
        headers={"Authorization": f"Bearer {TOKEN}"},
        data={"text_1": "a", "text_2": "b"},
    )
    assert response.status_code == 404


def test_run_missing_argument_is_422():
    response = client.post(
        "/run/test_tool",
        headers={"Authorization": f"Bearer {TOKEN}"},
        data={"text_1": "a"},
    )
    assert response.status_code == 422


def test_run_unexpected_argument_is_422():
    response = client.post(
        "/run/test_tool",
        headers={"Authorization": f"Bearer {TOKEN}"},
        data={"text_1": "a", "text_2": "b", "text_3": "c"},
    )
    assert response.status_code == 422


def test_run_test_tool_happy_path():
    response = client.post(
        "/run/test_tool",
        headers={"Authorization": f"Bearer {TOKEN}"},
        data={"text_1": "hello", "text_2": "world"},
    )
    assert response.status_code == 200
    assert response.json() == {"result": "hello world"}


def test_run_example_tool_with_file(tmp_path):
    fake_volume = tmp_path / "volume.nii.gz"
    fake_volume.write_bytes(b"fake nifti content")

    with open(fake_volume, "rb") as file_obj:
        response = client.post(
            "/run/example_tool",
            headers={"Authorization": f"Bearer {TOKEN}"},
            data={"label": "case_1", "threshold": "0.5"},
            files={"file": ("volume.nii.gz", file_obj, "application/gzip")},
        )

    assert response.status_code == 200
    assert response.json()["result"].startswith("label=case_1")
