"""Unit tests for the CrownSeg tool.

shapeaxi's segmentation is stubbed, so everything around it -- input
discovery, the already-segmented bypass, the output tree, the run report, and
the schema -- runs for real. The network itself cannot run here: shapeaxi and
pytorch3d are absent from the current deployment image (pytorch3d has no PyPI
distribution at all), which is the whole reason both are imported lazily.

Run just this module:
    cd server && python -m pytest tools/CrownSeg/test/
"""

import json
import os
import zipfile

os.environ.setdefault("API_TOKEN", "test-token")

import pytest

from base import ToolArgumentError
from tools.CrownSeg.CrownSeg import CrownSegTool
from tools.CrownSeg.src import CrownSegLogic


def write_surface(path, labelled=False, array_name="Universal_ID"):
    vtk = pytest.importorskip("vtk")

    points = vtk.vtkPoints()
    for coordinates in ((0, 0, 0), (1, 0, 0), (0, 1, 0)):
        points.InsertNextPoint(*coordinates)
    polys = vtk.vtkCellArray()
    polys.InsertNextCell(3)
    for point_id in range(3):
        polys.InsertCellPoint(point_id)

    surface = vtk.vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(polys)

    if labelled:
        labels = vtk.vtkIntArray()
        labels.SetName(array_name)
        for _ in range(3):
            labels.InsertNextValue(8)
        surface.GetPointData().AddArray(labels)

    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.Write()
    return str(path)


@pytest.fixture
def stub_shapeaxi(monkeypatch, tmp_path):
    """Write a labelled mesh wherever shapeaxi would have written one."""
    calls = []

    def fake_run(csv_path, output_dir, model_path, input_root, array_name, suffix, device, fdi):
        calls.append(
            {
                "csv": csv_path,
                "out": output_dir,
                "model": model_path,
                "root": input_root,
                "device": device,
                "fdi": fdi,
            }
        )
        csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
        with open(csv_path, encoding="utf-8") as handle:
            meshes = [line.strip() for line in handle.read().splitlines()[1:] if line.strip()]
        for mesh in meshes:
            destination = CrownSegLogic._predicted_path(
                output_dir, csv_stem, suffix, mesh, input_root
            )
            write_surface(destination, labelled=True, array_name=array_name)

    monkeypatch.setattr(CrownSegLogic, "_run_shapeaxi", fake_run)
    monkeypatch.setattr(CrownSegLogic, "resolve_device", lambda requested=None: "cpu")
    monkeypatch.setattr(
        CrownSegLogic, "default_model_path", lambda: str(tmp_path / "model.pth")
    )
    return calls


# ---------------------------------------------------------------------------
# Already-segmented detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("array_name", CrownSegLogic.LABEL_ARRAY_NAMES)
def test_every_known_label_array_counts_as_segmented(tmp_path, array_name):
    mesh = write_surface(tmp_path / "arch.vtk", labelled=True, array_name=array_name)
    assert CrownSegLogic.is_segmented(mesh) is True


def test_a_raw_mesh_is_not_segmented(tmp_path):
    assert CrownSegLogic.is_segmented(write_surface(tmp_path / "arch.vtk")) is False


def test_an_stl_can_never_be_segmented(tmp_path):
    """STL carries no point data by construction. The Slicer module copied a
    "segmented" .stl into its bypass folder and its CLI then globbed for .vtk
    only, so those meshes were silently never processed at all."""
    vtk = pytest.importorskip("vtk")

    source = write_surface(tmp_path / "arch.vtk")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(source)
    reader.Update()
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(tmp_path / "arch.stl"))
    writer.SetInputData(reader.GetOutput())
    writer.Write()

    assert CrownSegLogic.is_segmented(str(tmp_path / "arch.stl")) is False


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_discovery_is_recursive(tmp_path):
    write_surface(tmp_path / "cohort" / "a" / "one.vtk")
    write_surface(tmp_path / "cohort" / "b" / "two.vtk")

    meshes = CrownSegLogic.discover_meshes(str(tmp_path / "cohort"), str(tmp_path / "scratch"))
    assert len(meshes) == 2


def test_a_zip_is_unpacked(tmp_path):
    write_surface(tmp_path / "cohort" / "one.vtk")
    archive = tmp_path / "cohort.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(tmp_path / "cohort" / "one.vtk", "one.vtk")

    meshes = CrownSegLogic.discover_meshes(str(archive), str(tmp_path / "scratch"))
    assert [os.path.basename(path) for path in meshes] == ["one.vtk"]


def test_a_file_of_the_wrong_kind_is_a_422(tmp_path):
    (tmp_path / "notes.txt").write_text("not a mesh")
    with pytest.raises(ToolArgumentError):
        CrownSegLogic.discover_meshes(str(tmp_path / "notes.txt"), str(tmp_path / "scratch"))


def test_an_empty_folder_is_a_422(tmp_path):
    (tmp_path / "cohort").mkdir()
    with pytest.raises(ToolArgumentError):
        CrownSegLogic.discover_meshes(str(tmp_path / "cohort"), str(tmp_path / "scratch"))


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def test_a_raw_mesh_is_segmented_and_reported(tmp_path, stub_shapeaxi):
    write_surface(tmp_path / "cohort" / "arch.vtk")

    run = CrownSegLogic.segment_crowns(
        input_path=str(tmp_path / "cohort"), scratch_dir=str(tmp_path / "scratch")
    )

    assert run.report["summary"] == {
        "total": 1, "segmented": 1, "already_segmented": 0, "failed": 0
    }
    assert len(run.meshes) == 1
    assert CrownSegLogic.is_segmented(run.meshes[0])


def test_an_already_segmented_mesh_is_passed_through(tmp_path, stub_shapeaxi):
    """Minutes of GPU time per mesh, for an array that is already there."""
    write_surface(tmp_path / "cohort" / "arch.vtk", labelled=True)

    run = CrownSegLogic.segment_crowns(
        input_path=str(tmp_path / "cohort"), scratch_dir=str(tmp_path / "scratch")
    )

    assert stub_shapeaxi == []  # the network was never invoked
    assert run.report["summary"]["already_segmented"] == 1
    assert CrownSegLogic.is_segmented(run.meshes[0])


def test_skip_segmented_false_re_runs_the_network(tmp_path, stub_shapeaxi):
    write_surface(tmp_path / "cohort" / "arch.vtk", labelled=True)

    run = CrownSegLogic.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        scratch_dir=str(tmp_path / "scratch"),
        skip_segmented=False,
    )
    assert len(stub_shapeaxi) == 1
    assert run.report["summary"]["segmented"] == 1


def test_a_batch_keeps_its_tree(tmp_path, stub_shapeaxi):
    write_surface(tmp_path / "cohort" / "siteA" / "arch.vtk")
    write_surface(tmp_path / "cohort" / "siteB" / "arch.vtk")

    run = CrownSegLogic.segment_crowns(
        input_path=str(tmp_path / "cohort"), scratch_dir=str(tmp_path / "scratch")
    )

    # Two patients with the same base name. Flattened, one would overwrite the
    # other and the caller would silently get half its cohort back.
    assert len(run.meshes) == 2
    assert len(set(run.meshes)) == 2
    assert sorted(run.report["meshes"]) == [
        os.path.join("siteA", "arch.vtk"), os.path.join("siteB", "arch.vtk")
    ]


def test_the_input_csv_is_written_to_the_scratch_dir(tmp_path, stub_shapeaxi):
    """The Slicer module wrote it into the extension's own source folder,
    which breaks a read-only install and leaves behind a file pointing at the
    patient's data."""
    write_surface(tmp_path / "cohort" / "arch.vtk")

    CrownSegLogic.segment_crowns(
        input_path=str(tmp_path / "cohort"), scratch_dir=str(tmp_path / "scratch")
    )
    assert stub_shapeaxi[0]["csv"].startswith(str(tmp_path / "scratch"))


def test_the_run_report_lands_beside_the_results(tmp_path, stub_shapeaxi):
    write_surface(tmp_path / "cohort" / "arch.vtk")

    run = CrownSegLogic.segment_crowns(
        input_path=str(tmp_path / "cohort"), scratch_dir=str(tmp_path / "scratch")
    )
    written = json.loads(
        open(os.path.join(run.output_dir, "run_report.json"), encoding="utf-8").read()
    )
    assert written["numbering"] == "Universal"
    assert written["array_name"] == CrownSegLogic.DEFAULT_ARRAY_NAME


def test_fdi_numbering_is_passed_through(tmp_path, stub_shapeaxi):
    write_surface(tmp_path / "cohort" / "arch.vtk")

    run = CrownSegLogic.segment_crowns(
        input_path=str(tmp_path / "cohort"), scratch_dir=str(tmp_path / "scratch"), fdi=True
    )
    assert stub_shapeaxi[0]["fdi"] is True
    assert run.report["numbering"] == "FDI"


def test_a_missing_default_model_says_how_to_fetch_it(tmp_path, monkeypatch):
    """Never a silent download: shapeaxi's own fallback pulls the checkpoint
    from GitHub mid-request, and this server does not make outbound calls
    while holding patient data."""
    from data_store import DataNotFoundError

    monkeypatch.setattr(
        CrownSegLogic.data_store,
        "resolve_model",
        lambda tool, name: (_ for _ in ()).throw(DataNotFoundError("nope")),
    )
    with pytest.raises(FileNotFoundError) as raised:
        CrownSegLogic.default_model_path()
    assert "setup-models.sh" in str(raised.value)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_the_schema_is_valid():
    CrownSegTool().check_schema()


def test_the_model_is_optional_so_another_tool_can_just_ask_for_segmentation():
    """ALI's IOS mode calls segment_crowns() without naming a model: crown
    segmentation is a service, not a file the caller has to know about."""
    model = CrownSegTool().arguments["model"]
    assert not model.required
    assert model.server_selectable == "model"
    assert not model.is_file  # name only; the weights never travel


def test_the_input_accepts_a_mesh_or_a_zipped_folder():
    spec = CrownSegTool().arguments["input"]
    assert spec.types == ("surface_or_zip_file", "folder")
    assert set(spec.extensions) == {".vtk", ".stl", ".zip"}
