"""Smoke tests for the tool-registry server.

Run with: cd server && ./venv/bin/pytest
(requires requirements-dev.txt: pip install -r requirements-dev.txt)
"""

import io
import json
import os
import zipfile

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


def test_run_example_tool_with_single_csv(tmp_path):
    """The "input" argument declares type=("csv_file", "folder"): a plain .csv
    resolves to kind="csv_file" and the tool returns two files, zipped."""
    csv_file = tmp_path / "measures.csv"
    csv_file.write_text("a,b\n1,2\n3,4\n")

    with open(csv_file, "rb") as file_obj:
        response = client.post(
            "/run/example_tool",
            headers={"Authorization": f"Bearer {TOKEN}"},
            data={"label": "case_1", "threshold": "0.5"},
            files={"input": ("measures.csv", file_obj, "text/csv")},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "example_tool_output.zip" in response.headers["content-disposition"]

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert sorted(archive.namelist()) == ["preview.csv", "summary.txt"]
        summary = archive.read("summary.txt").decode()
    assert "label=case_1" in summary
    assert "input_kind=csv_file" in summary
    assert "rows=2 columns=2" in summary


def test_run_example_tool_with_zipped_folder(tmp_path):
    """The same argument accepts a folder: the client zips it, the server
    extracts it, and run() sees kind="folder" with a real directory."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        # A single root folder, as any OS produces when zipping a directory --
        # extraction strips it so the tool sees the files directly.
        archive.writestr("measures/first.csv", "a,b\n1,2\n")
        archive.writestr("measures/second.csv", "a,b\n3,4\n")
    buffer.seek(0)

    response = client.post(
        "/run/example_tool",
        headers={"Authorization": f"Bearer {TOKEN}"},
        data={"label": "case_2", "threshold": "2.5"},
        files={"input": ("measures.zip", buffer, "application/zip")},
    )

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        summary = archive.read("summary.txt").decode()
    assert "input_kind=folder" in summary
    assert "folder, 2 entries" in summary
    assert "rows=2 columns=2" in summary  # both CSVs concatenated
    assert "values_above_threshold=2" in summary  # 3 and 4 are > 2.5


def test_run_example_tool_rejects_unsupported_extension():
    """type=("csv_file", "folder") accepts .csv and .zip -- nothing else."""
    response = client.post(
        "/run/example_tool",
        headers={"Authorization": f"Bearer {TOKEN}"},
        data={"label": "case_3", "threshold": "0.5"},
        files={"input": ("volume.nii.gz", b"not tabular", "application/gzip")},
    )

    assert response.status_code == 400


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


def test_tools_reports_every_declared_type():
    """GET /tools keeps a single-string "type" for older clients and adds
    "types" with the full list, so a client can filter its file picker."""
    tools = {tool["name"]: tool for tool in client.get("/tools").json()}

    example_input = tools["example_tool"]["arguments"]["input"]
    assert example_input["type"] == "csv_file"
    assert example_input["types"] == ["csv_file", "folder"]

    single_typed = tools["test_tool"]["arguments"]["text_1"]
    assert single_typed["type"] == "str"
    assert single_typed["types"] == ["str"]


def test_tools_publishes_the_extensions_of_every_file_type():
    """A type name does not reliably spell out its extensions ("nifti_file" is
    .nii/.nii.gz, "volume_or_zip_file" is seven of them), so /tools sends
    FILE_TYPES' entry for each declared file type. Without it a client has to
    keep a copy of that table, and drifts every time it changes here."""
    tools = {tool["name"]: tool for tool in client.get("/tools").json()}

    example_input = tools["example_tool"]["arguments"]["input"]
    # Keyed by type, not flattened: "folder"'s .zip is what a zipped folder may
    # be uploaded as, not something a file picker should offer.
    assert example_input["extensions"] == {"csv_file": [".csv"], "folder": [".zip"]}

    # An argument taking no file at all says so.
    assert tools["example_tool"]["arguments"]["label"]["extensions"] is None

    for tool in tools.values():
        for argument in tool["arguments"].values():
            for type_name, extensions in (argument["extensions"] or {}).items():
                # null = declared but deliberately unrestricted (the generic
                # "file", which falls back to ALLOWED_EXTENSIONS at upload time).
                assert extensions is None or all(e.startswith(".") for e in extensions), type_name


def test_run_declaration_order_decides_zip_vs_folder(monkeypatch):
    """("zip_file", "folder") hands the archive over untouched; ("folder",
    "zip_file") extracts it first. Same upload, different kind."""
    import base
    import registry

    class OrderTestTool(base.Tool):
        name = "order_test_tool"
        arguments = {"data": base.ArgSpec(type=("zip_file", "folder"), required=True)}
        output_kind = "text"

        def run(self, data: str) -> str:
            return f"{data.kind}:{os.path.isdir(data)}"

    monkeypatch.setitem(registry.TOOLS, "order_test_tool", OrderTestTool())

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a.txt", "content")
    payload = buffer.getvalue()

    response = client.post(
        "/run/order_test_tool",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files={"data": ("data.zip", io.BytesIO(payload), "application/zip")},
    )
    assert response.json() == {"result": "zip_file:False"}

    OrderTestTool.arguments = {"data": base.ArgSpec(type=("folder", "zip_file"), required=True)}
    monkeypatch.setitem(registry.TOOLS, "order_test_tool", OrderTestTool())

    response = client.post(
        "/run/order_test_tool",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files={"data": ("data.zip", io.BytesIO(payload), "application/zip")},
    )
    assert response.json() == {"result": "folder:True"}


def test_run_rejects_zip_slip_archive(monkeypatch):
    """A folder argument extracts an untrusted archive: a member escaping the
    extraction directory must be refused, not written outside it."""
    import base
    import registry

    class FolderTestTool(base.Tool):
        name = "folder_slip_test_tool"
        arguments = {"data": base.ArgSpec(type="folder", required=True)}
        output_kind = "text"

        def run(self, data: str) -> str:
            return "should not run"

    monkeypatch.setitem(registry.TOOLS, "folder_slip_test_tool", FolderTestTool())

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../../escaped.txt", "pwned")
    buffer.seek(0)

    response = client.post(
        "/run/folder_slip_test_tool",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files={"data": ("evil.zip", buffer, "application/zip")},
    )

    assert response.status_code == 400
    assert "points outside" in response.json()["detail"]


def test_run_rejects_oversized_archive(monkeypatch):
    """MAX_EXTRACTED_MB caps the uncompressed size: a small upload that would
    expand to far more is refused before anything is written."""
    import base
    import main
    import registry

    class FolderTestTool(base.Tool):
        name = "folder_bomb_test_tool"
        arguments = {"data": base.ArgSpec(type="folder", required=True)}
        output_kind = "text"

        def run(self, data: str) -> str:
            return "should not run"

    monkeypatch.setitem(registry.TOOLS, "folder_bomb_test_tool", FolderTestTool())
    monkeypatch.setattr(main, "_MAX_EXTRACTED_BYTES", 1024)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("big.txt", "0" * 100_000)  # compresses to a few hundred bytes
    buffer.seek(0)

    response = client.post(
        "/run/folder_bomb_test_tool",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files={"data": ("bomb.zip", buffer, "application/zip")},
    )

    assert response.status_code == 400
    assert "expands to more than" in response.json()["detail"]


def test_run_returning_a_directory_zips_its_contents(monkeypatch):
    """output_kind="files" also accepts a single directory: its contents land
    at the root of the archive, named after the folder."""
    import base
    import file_utils
    import registry

    class DirOutputTool(base.Tool):
        name = "dir_output_test_tool"
        arguments = {"text": base.ArgSpec(type=str, required=True)}
        output_kind = "files"

        def run(self, text: str) -> str:
            output_dir = os.path.join(file_utils.make_scratch_dir("dir_output_"), "results")
            os.makedirs(os.path.join(output_dir, "nested"))
            with open(os.path.join(output_dir, "a.txt"), "w") as handle:
                handle.write(text)
            with open(os.path.join(output_dir, "nested", "b.txt"), "w") as handle:
                handle.write(text)
            return output_dir

    monkeypatch.setitem(registry.TOOLS, "dir_output_test_tool", DirOutputTool())

    response = client.post(
        "/run/dir_output_test_tool",
        headers={"Authorization": f"Bearer {TOKEN}"},
        data={"text": "hello"},
    )

    assert response.status_code == 200
    assert "results.zip" in response.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert sorted(archive.namelist()) == ["a.txt", "nested/b.txt"]


def test_run_files_output_cleans_up_the_tool_scratch_dir(monkeypatch):
    """Nothing may be left behind under TEMP_DIR once the response is sent."""
    import base
    import file_utils
    import registry
    from config import settings

    produced = {}

    class ScratchOutputTool(base.Tool):
        name = "scratch_output_test_tool"
        arguments = {"text": base.ArgSpec(type=str, required=True)}
        output_kind = "files"

        def run(self, text: str) -> list:
            output_dir = file_utils.make_scratch_dir("scratch_output_")
            produced["dir"] = output_dir
            paths = []
            for name in ("one.txt", "two.txt"):
                path = os.path.join(output_dir, name)
                with open(path, "w") as handle:
                    handle.write(text)
                paths.append(path)
            return paths

    monkeypatch.setitem(registry.TOOLS, "scratch_output_test_tool", ScratchOutputTool())

    before = set(os.listdir(settings.TEMP_DIR))
    response = client.post(
        "/run/scratch_output_test_tool",
        headers={"Authorization": f"Bearer {TOKEN}"},
        data={"text": "hello"},
    )

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert sorted(archive.namelist()) == ["one.txt", "two.txt"]
    assert not os.path.exists(produced["dir"])
    assert set(os.listdir(settings.TEMP_DIR)) == before


def test_run_files_output_rejects_duplicate_basenames(monkeypatch):
    """Two outputs sharing a base name would collide inside the archive and
    silently return fewer files than the tool produced."""
    import base
    import file_utils
    import registry

    class CollidingOutputTool(base.Tool):
        name = "colliding_output_test_tool"
        arguments = {"text": base.ArgSpec(type=str, required=True)}
        output_kind = "files"

        def run(self, text: str) -> list:
            output_dir = file_utils.make_scratch_dir("colliding_")
            paths = []
            for folder in ("left", "right"):
                os.makedirs(os.path.join(output_dir, folder))
                path = os.path.join(output_dir, folder, "result.csv")
                with open(path, "w") as handle:
                    handle.write(text)
                paths.append(path)
            return paths

    monkeypatch.setitem(registry.TOOLS, "colliding_output_test_tool", CollidingOutputTool())

    response = client.post(
        "/run/colliding_output_test_tool",
        headers={"Authorization": f"Bearer {TOKEN}"},
        data={"text": "hello"},
    )

    assert response.status_code == 500


def test_server_selectable_folder_argument(monkeypatch, tmp_path):
    """A "folder" argument can also be satisfied server-side: a real folder in
    the data store is passed straight through, an archive is extracted first.
    Either way run() sees a directory -- the tool can't tell the routes apart."""
    import base
    import main
    import registry
    from data_store import ResolvedFile

    class ServerFolderTool(base.Tool):
        name = "server_folder_test_tool"
        arguments = {
            "dataset": base.ArgSpec(type="folder", required=True, server_selectable="testfile"),
        }
        output_kind = "text"

        def run(self, dataset: str) -> str:
            return f"{dataset.kind}:{sorted(os.listdir(dataset))}"

    monkeypatch.setitem(registry.TOOLS, "server_folder_test_tool", ServerFolderTool())

    stored_folder = tmp_path / "cohort"
    stored_folder.mkdir()
    (stored_folder / "a.csv").write_text("a\n1\n")

    stored_archive = tmp_path / "cohort.zip"
    with zipfile.ZipFile(stored_archive, "w") as archive:
        archive.writestr("cohort/b.csv", "b\n2\n")

    available = {"cohort": str(stored_folder), "cohort.zip": str(stored_archive)}
    monkeypatch.setattr(
        main.data_store,
        "resolve_testfile",
        lambda tool_name, filename: ResolvedFile(path=available[filename]),
    )

    for filename, expected in (("cohort", "a.csv"), ("cohort.zip", "b.csv")):
        response = client.post(
            "/run/server_folder_test_tool",
            headers={"Authorization": f"Bearer {TOKEN}"},
            data={"dataset": filename},
        )
        assert response.status_code == 200
        assert response.json() == {"result": f"folder:['{expected}']"}

    # The read-only data store must be left exactly as it was.
    assert stored_archive.exists()
    assert sorted(os.listdir(stored_folder)) == ["a.csv"]


def test_invalid_schema_is_rejected_at_startup():
    """check_schema() runs when the registry is built, so a malformed tool
    fails on boot instead of on the first request that reaches it."""
    import base

    class UnknownTypeTool(base.Tool):
        name = "unknown_type_tool"
        arguments = {"data": base.ArgSpec(type="pdf_file", required=True)}

        def run(self, data: str) -> str:
            return "unused"

    class MixedTypeTool(base.Tool):
        name = "mixed_type_tool"
        arguments = {"data": base.ArgSpec(type=(str, "folder"), required=True)}

        def run(self, data: str) -> str:
            return "unused"

    for tool, expected in ((UnknownTypeTool(), "unknown type"), (MixedTypeTool(), "combined")):
        try:
            tool.check_schema()
        except base.ToolSchemaError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"{type(tool).__name__} should have been rejected")


def test_tools_exposes_choices():
    """GET /tools ships the options and their initial state, so the client can
    render check boxes / a combo box with no tool-specific code."""
    tools = {tool["name"]: tool for tool in client.get("/tools").json()}
    arguments = tools["example_tool"]["arguments"]

    assert arguments["outputs"]["type"] == "multichoice"
    assert arguments["outputs"]["choices"] == {
        "summary": True,
        "preview": True,
        "columns": False,
    }
    assert arguments["preview_format"]["type"] == "choice"
    assert arguments["preview_format"]["choices"] == {"csv": True, "json": False}
    # Every other argument reports choices=None.
    assert arguments["threshold"]["choices"] is None


def test_tools_publishes_the_initial_value_of_scalar_arguments():
    """A scalar argument's `initial` is what a client pre-fills its widget with.

    It matters because a form sends every widget it rendered: a spin box left at
    Qt's own 0 sends 0, and the tool's Python default never applies. Publishing
    `initial` is what keeps the widget and run()'s signature agreeing.
    """
    tools = {tool["name"]: tool for tool in client.get("/tools").json()}

    amasss = tools.get("AMASSS")
    if amasss is not None:  # skipped when the nnUNet stack isn't installed
        assert amasss["arguments"]["surface_smoothing"]["initial"] == 5
        assert amasss["arguments"]["generate_surface"]["initial"] is False

    # None when the tool declares none -- example_tool's `iterations` means
    # "unset", and must not be coerced to 0 client-side.
    assert tools["example_tool"]["arguments"]["iterations"]["initial"] is None


    # None when the tool declares none -- example_tool's `iterations` means
    # "unset", and must not be coerced to 0 client-side.
    assert tools["example_tool"]["arguments"]["iterations"]["initial"] is None


def _run_example(**extra):
    return client.post(
        "/run/example_tool",
        headers={"Authorization": f"Bearer {TOKEN}"},
        data={"label": "choices", "threshold": "0", **extra},
        files={"input": ("m.csv", b"a,b\n1,2\n", "text/csv")},
    )


def test_choice_arguments_fall_back_to_their_declared_defaults():
    """Omitting an optional choice argument yields what `choices` declared --
    the tool never repeats those defaults in run()'s signature."""
    response = _run_example()

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert sorted(archive.namelist()) == ["preview.csv", "summary.txt"]
        assert "outputs=summary,preview" in archive.read("summary.txt").decode()


def test_multichoice_accepts_json_and_comma_shorthand():
    """Both wire formats describe the COMPLETE selection: an option that isn't
    mentioned is off, whatever its declared default."""
    as_json = _run_example(outputs='{"summary": false, "preview": true, "columns": true}')
    with zipfile.ZipFile(io.BytesIO(as_json.content)) as archive:
        assert sorted(archive.namelist()) == ["columns.txt", "preview.csv"]

    as_list = _run_example(outputs="preview,columns")
    with zipfile.ZipFile(io.BytesIO(as_list.content)) as archive:
        assert sorted(archive.namelist()) == ["columns.txt", "preview.csv"]

    # "summary" is declared True by default, but isn't in this selection.
    partial = _run_example(outputs='{"columns": true}')
    with zipfile.ZipFile(io.BytesIO(partial.content)) as archive:
        assert archive.namelist() == ["columns.txt"]


def test_choice_argument_switches_behaviour():
    response = _run_example(preview_format="json")

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert "preview.json" in archive.namelist()
        assert json.loads(archive.read("preview.json").decode()) == [{"a": 1, "b": 2}]


def test_unknown_option_is_422():
    assert _run_example(preview_format="xml").status_code == 422
    assert "unknown option 'xml'" in _run_example(preview_format="xml").json()["detail"]

    multi = _run_example(outputs="summary,does_not_exist")
    assert multi.status_code == 422
    assert "does_not_exist" in multi.json()["detail"]

    malformed = _run_example(outputs='{"summary": tru')
    assert malformed.status_code == 422
    assert "invalid JSON" in malformed.json()["detail"]


def test_tool_can_reject_an_invalid_combination_with_422():
    """A cross-argument rule can't live in the schema; a tool raising
    ToolArgumentError gets the same 422 treatment as a schema violation."""
    response = _run_example(outputs="")

    assert response.status_code == 422
    assert response.json()["detail"] == "Select at least one output to produce."


def test_choice_schema_is_validated_at_startup():
    import base

    def spec(**kwargs):
        return {"opt": base.ArgSpec(**kwargs)}

    invalid = {
        "no choices declared": spec(type="choice"),
        "choices on a plain type": spec(type=str, choices={"a": True}),
        "empty choices": spec(type="multichoice", choices={}),
        "non-boolean state": spec(type="multichoice", choices={"a": "yes"}),
        "two defaults for a combo box": spec(type="choice", choices={"a": True, "b": True}),
        "no default for a combo box": spec(type="choice", choices={"a": False}),
        # `choices` already carries the initial state; a second declaration
        # would be the two-places-for-one-default problem `initial` removes.
        "initial alongside choices": spec(
            type="choice", choices={"a": True, "b": False}, initial="a"
        ),
        "initial on a file argument": spec(type="csv_file", initial="/tmp/x.csv"),
    }

    for reason, arguments in invalid.items():
        tool = type("T", (base.Tool,), {"name": "t", "arguments": arguments, "run": lambda self: None})()
        try:
            tool.check_schema()
        except base.ToolSchemaError:
            continue
        raise AssertionError(f"should have been rejected: {reason}")

    valid = type(
        "T",
        (base.Tool,),
        {
            "name": "t",
            "arguments": spec(type="multichoice", choices={"a": True, "b": False}),
            "run": lambda self: None,
        },
    )()
    valid.check_schema()


def test_selection_helper():
    """run() gets every declared option, so no .get(name, False) is ever needed."""
    import base

    class ChoiceTool(base.Tool):
        name = "selection_probe"
        arguments = {
            "structures": base.ArgSpec(
                type="multichoice",
                required=True,
                choices={"mandible": True, "maxilla": False, "skull": False},
            )
        }

        def run(self, structures):
            return structures

    selection = ChoiceTool().validate({"structures": "skull,mandible"})["structures"]

    assert selection == {"mandible": True, "maxilla": False, "skull": True}
    assert selection.selected == ("mandible", "skull")  # declaration order, not input order
    assert selection["maxilla"] is False


def test_scratch_dir_is_removed_when_run_raises(monkeypatch):
    """A tool crashing mid-inference must not leave its scratch dir -- and the
    patient data in it -- behind. main.py only ever learns about that folder
    through the path run() returns, which a crash never produces."""
    import base
    import file_utils
    import registry
    from config import settings

    produced = {}

    class CrashingTool(base.Tool):
        name = "crashing_scratch_tool"
        arguments = {"text": base.ArgSpec(type=str, required=True)}
        output_kind = "file"

        def run(self, text: str) -> str:
            produced["dir"] = file_utils.make_scratch_dir("crashing_")
            with open(os.path.join(produced["dir"], "patient.csv"), "w") as handle:
                handle.write("confidential")
            raise RuntimeError("inference blew up")

    monkeypatch.setitem(registry.TOOLS, "crashing_scratch_tool", CrashingTool())

    before = set(os.listdir(settings.TEMP_DIR))
    response = client.post(
        "/run/crashing_scratch_tool",
        headers={"Authorization": f"Bearer {TOKEN}"},
        data={"text": "hello"},
    )

    assert response.status_code == 500
    assert not os.path.exists(produced["dir"])
    assert set(os.listdir(settings.TEMP_DIR)) == before


def test_concurrent_requests_do_not_share_scratch_tracking(monkeypatch):
    """The scratch-dir registry is per-request: one request finishing must
    never delete a folder another one is still writing into."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import base
    import file_utils
    import registry
    from config import settings

    both_inside = threading.Barrier(2, timeout=10)
    seen = {}

    class SlowScratchTool(base.Tool):
        name = "concurrent_scratch_tool"
        arguments = {"tag": base.ArgSpec(type=str, required=True)}
        output_kind = "text"

        def run(self, tag: str) -> str:
            scratch_dir = file_utils.make_scratch_dir(f"concurrent_{tag}_")
            marker = os.path.join(scratch_dir, "marker")
            with open(marker, "w") as handle:
                handle.write(tag)
            # Hold both requests inside run() at once, then look at the
            # registry this request will be cleaned up from: it must list this
            # request's folder and nothing else, even though the other request
            # has already recorded its own.
            both_inside.wait()
            tracked = file_utils._scratch_dirs.get()
            seen[tag] = (scratch_dir, os.path.exists(marker), list(tracked))
            return tag

    monkeypatch.setitem(registry.TOOLS, "concurrent_scratch_tool", SlowScratchTool())

    before = set(os.listdir(settings.TEMP_DIR))
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(
            pool.map(
                lambda tag: client.post(
                    "/run/concurrent_scratch_tool",
                    headers={"Authorization": f"Bearer {TOKEN}"},
                    data={"tag": tag},
                ),
                ("a", "b"),
            )
        )

    assert [response.status_code for response in responses] == [200, 200]
    assert seen["a"][0] != seen["b"][0]  # each request got its own folder
    assert seen["a"][1] and seen["b"][1]  # neither was wiped by the other
    # The registry is per-request, not global: had it been shared, each list
    # would hold both folders and either request could delete the other's.
    assert seen["a"][2] == [seen["a"][0]]
    assert seen["b"][2] == [seen["b"][0]]
    assert set(os.listdir(settings.TEMP_DIR)) == before  # both cleaned up after


def test_served_request_is_logged_with_both_sizes(caplog, tmp_path):
    """One log line per served request, carrying the bytes in AND the bytes out.

    `sent` is what makes a run diagnosable after the fact: `received` alone
    says nothing about a segmentation that came back empty. It is measured on
    the response file, so for output_kind="files" it is the size of the
    archive actually streamed, not of what run() produced before zipping.
    """
    csv_file = tmp_path / "measures.csv"
    csv_file.write_text("a,b\n1,2\n3,4\n")

    with caplog.at_level("INFO", logger="inference_server"):
        with open(csv_file, "rb") as file_obj:
            response = client.post(
                "/run/example_tool",
                headers={"Authorization": f"Bearer {TOKEN}"},
                data={"label": "case_1", "threshold": "0.5"},
                files={"input": ("measures.csv", file_obj, "text/csv")},
            )

    assert response.status_code == 200
    line = next(
        message for message in caplog.messages if message.startswith("endpoint=/run/example_tool")
    )
    assert f"received={csv_file.stat().st_size}B" in line
    assert f"sent={len(response.content)}B" in line
    assert "(12 B)" in line  # the human-readable form sits beside the exact one
    assert "status=200" in line


def test_text_output_is_logged_without_a_sent_size(caplog):
    """A "text" tool's result travels as JSON: there is no output file to
    measure, so the field is omitted rather than reported as a bogus zero."""
    with caplog.at_level("INFO", logger="inference_server"):
        response = client.post(
            "/run/test_tool",
            headers={"Authorization": f"Bearer {TOKEN}"},
            data={"text_1": "hello", "text_2": "world"},
        )

    assert response.status_code == 200
    line = next(
        message for message in caplog.messages if message.startswith("endpoint=/run/test_tool")
    )
    assert "received=0B (0 B)" in line
    assert "sent=" not in line


def test_log_line_carries_no_file_name_or_argument_value(caplog, tmp_path):
    """The server handles confidential medical data: logs stay limited to
    timestamp, endpoint, tool, status, duration and sizes (see main.py's
    header). A file name or an argument value in there would be a leak."""
    csv_file = tmp_path / "patient_ident_0042.csv"
    csv_file.write_text("a,b\n1,2\n")

    with caplog.at_level("INFO", logger="inference_server"):
        with open(csv_file, "rb") as file_obj:
            response = client.post(
                "/run/example_tool",
                headers={"Authorization": f"Bearer {TOKEN}"},
                data={"label": "secret_patient_label", "threshold": "0.5"},
                files={"input": (csv_file.name, file_obj, "text/csv")},
            )

    assert response.status_code == 200
    line = next(
        message for message in caplog.messages if message.startswith("endpoint=/run/example_tool")
    )
    assert "patient_ident_0042" not in line
    assert "secret_patient_label" not in line
