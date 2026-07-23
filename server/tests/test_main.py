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


def test_run_tool_with_two_named_files(monkeypatch):
    """A tool can declare more than one "file"-typed argument; each uploaded
    file is matched to the tool argument with the same field name."""
    import base
    import registry

    class TwoFileTestTool(base.Tool):
        name = "two_file_test_tool"
        arguments = {
            "fixed_image": base.ArgSpec(type="file", required=True),
            "moving_image": base.ArgSpec(type="file", required=True),
        }
        output_kind = "text"

        def run(self, fixed_image: str, moving_image: str) -> str:
            return f"{os.path.getsize(fixed_image)}:{os.path.getsize(moving_image)}"

    monkeypatch.setitem(registry.TOOLS, "two_file_test_tool", TwoFileTestTool())

    response = client.post(
        "/run/two_file_test_tool",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files={
            "fixed_image": ("a.nii.gz", b"aaa", "application/gzip"),
            "moving_image": ("b.nii.gz", b"bbbbb", "application/gzip"),
        },
    )

    assert response.status_code == 200
    assert response.json() == {"result": "3:5"}


def test_run_tool_with_two_named_files_missing_one_is_422(monkeypatch):
    import base
    import registry

    class TwoFileTestTool(base.Tool):
        name = "two_file_test_tool_2"
        arguments = {
            "fixed_image": base.ArgSpec(type="file", required=True),
            "moving_image": base.ArgSpec(type="file", required=True),
        }
        output_kind = "text"

        def run(self, fixed_image: str, moving_image: str) -> str:
            return "unused"

    monkeypatch.setitem(registry.TOOLS, "two_file_test_tool_2", TwoFileTestTool())

    response = client.post(
        "/run/two_file_test_tool_2",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files={"fixed_image": ("a.nii.gz", b"aaa", "application/gzip")},
    )

    assert response.status_code == 422
