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


def test_run_rejects_upload_for_scalar_argument():
    """SurgMovPred's "model" is a server-side-only selection (type str +
    server_selectable): sending it as a file upload must be rejected outright,
    not silently passed through as a temp path."""
    response = client.post(
        "/run/SurgMovPred",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files={"model": ("model.zip", b"PK\x03\x04fake", "application/zip")},
    )

    assert response.status_code == 400
    assert "model" in response.json()["detail"]


def test_run_unknown_server_model_name_is_404():
    response = client.post(
        "/run/SurgMovPred",
        headers={"Authorization": f"Bearer {TOKEN}"},
        data={"model": "does_not_exist", "input": "does_not_exist.zip"},
    )

    assert response.status_code == 404


def test_run_resolves_scalar_server_selectable_model_by_name(monkeypatch, tmp_path):
    """A str-typed server_selectable argument sent as a plain form value (the
    model's name) is resolved through data_store; run() receives a local path."""
    import base
    import main
    import registry
    from data_store import ResolvedFile

    model_file = tmp_path / "stacking_v2.zip"
    model_file.write_bytes(b"model-bytes")

    class ServerModelTestTool(base.Tool):
        name = "server_model_test_tool"
        arguments = {
            "model": base.ArgSpec(type=str, required=True, server_selectable="model"),
        }
        output_kind = "text"

        def run(self, model: str) -> str:
            with open(model, "rb") as fh:
                return f"{os.path.basename(model)}:{len(fh.read())}"

    monkeypatch.setitem(registry.TOOLS, "server_model_test_tool", ServerModelTestTool())
    monkeypatch.setattr(
        main.data_store,
        "resolve_model",
        lambda tool_name, filename: ResolvedFile(path=str(model_file)),
    )

    response = client.post(
        "/run/server_model_test_tool",
        headers={"Authorization": f"Bearer {TOKEN}"},
        data={"model": "stacking_v2.zip"},
    )

    assert response.status_code == 200
    assert response.json() == {"result": "stacking_v2.zip:11"}


def test_concurrent_requests_run_in_parallel(monkeypatch):
    """Two /run requests must execute their tools at the same time, not one
    after the other. Each run() blocks on a 2-party barrier: it can only pass
    if BOTH requests are inside run() simultaneously. If tool execution ever
    goes back to blocking the event loop (serial execution), the barrier
    times out and the test fails instead of deadlocking forever."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import base
    import registry

    barrier = threading.Barrier(2, timeout=15)

    class ParallelProbeTool(base.Tool):
        name = "parallel_probe_tool"
        arguments = {"text": base.ArgSpec(type=str, required=True)}
        output_kind = "text"

        def run(self, text: str) -> str:
            barrier.wait()
            return text

    monkeypatch.setitem(registry.TOOLS, "parallel_probe_tool", ParallelProbeTool())

    # A single TestClient entered as a context manager routes every request
    # through ONE shared event loop -- exactly like uvicorn in production.
    # (Two bare client.post calls from two threads would each spin up their
    # own loop, which would pass even with a blocking server.)
    with TestClient(app) as shared_client:

        def call(index: int):
            return shared_client.post(
                "/run/parallel_probe_tool",
                headers={"Authorization": f"Bearer {TOKEN}"},
                data={"text": f"req_{index}"},
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(call, range(2)))

    assert [response.status_code for response in responses] == [200, 200]
    assert {response.json()["result"] for response in responses} == {"req_0", "req_1"}


def test_run_tool_rejects_wrong_extension_for_specific_file_type(monkeypatch):
    """A "zip_file"-typed argument only accepts .zip, regardless of the
    generic config.ALLOWED_EXTENSIONS fallback list."""
    import base
    import registry

    class ZipOnlyTestTool(base.Tool):
        name = "zip_only_test_tool"
        arguments = {
            "archive": base.ArgSpec(type="zip_file", required=True),
        }
        output_kind = "text"

        def run(self, archive: str) -> str:
            return "unused"

    monkeypatch.setitem(registry.TOOLS, "zip_only_test_tool", ZipOnlyTestTool())

    response = client.post(
        "/run/zip_only_test_tool",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files={"archive": ("data.csv", b"a,b\n1,2", "text/csv")},
    )

    assert response.status_code == 400
