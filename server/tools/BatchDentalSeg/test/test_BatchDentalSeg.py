"""BatchDentalSeg unit tests.

`nnunet_runner.predict_folder` is stubbed, so no GPU, no weights and no network
are needed. Everything around inference runs for real: discovery, the NIfTI
conversion, geometry matching, the output tree, the per-segment split and the
report.
"""

import json
import os
import zipfile

import numpy as np
import pytest
import SimpleITK as sitk

from base import ToolArgumentError
from tools.BatchDentalSeg.BatchDentalSeg import BatchDentalSegTool
from tools.BatchDentalSeg.src import BatchDentalSegLogic, catalogs, nnunet_runner


@pytest.fixture(autouse=True)
def _temp_dir(tmp_path, monkeypatch):
    """Point TEMP_DIR at the test's own directory.

    `file_utils.make_scratch_dir` registers what it hands out with the request
    being served and main.py is what deletes it, so a test calling the logic
    directly has no request and nothing cleans up -- without this the suite
    leaves one scratch directory per call in the server's real TEMP_DIR.
    """
    from config import settings

    monkeypatch.setattr(settings, "TEMP_DIR", str(tmp_path / "server_tmp"))


def _write_scan(path: str, size=(8, 8, 8), value: int = 40) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    array = np.full(size[::-1], value, dtype=np.int16)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 0.5, 0.5))
    image.SetOrigin((-10.0, -20.0, 30.0))
    sitk.WriteImage(image, path)
    return path


def _model_bundle(root: str, folder: str) -> str:
    """A bundle laid out the way nnUNet needs: the two json files beside a
    fold_0 holding the checkpoint."""
    base = os.path.join(root, folder, "nnUNetTrainer__nnUNetPlans__3d_fullres")
    os.makedirs(os.path.join(base, "fold_0"), exist_ok=True)
    for name in ("dataset.json", "plans.json"):
        with open(os.path.join(base, name), "w", encoding="utf-8") as handle:
            json.dump({}, handle)
    open(os.path.join(base, "fold_0", "checkpoint_final.pth"), "wb").close()
    return base


def _stub_prediction(labels_present):
    """Stand in for nnUNet: write one label volume per input case.

    Honours `on_case_start`/`on_case_done` like the real runner does, so the
    streamed path is exercised end to end rather than only its callers.
    """

    def predict_folder(model_folder, input_dir, output_dir, device,
                       on_case_start=None, on_case_done=None):
        os.makedirs(output_dir, exist_ok=True)
        names = [n for n in sorted(os.listdir(input_dir)) if n.endswith("_0000.nii.gz")]
        for index, name in enumerate(names):
            case_id = name[: -len("_0000.nii.gz")]
            if on_case_start is not None:
                on_case_start(case_id, index, len(names))
            reference = sitk.ReadImage(os.path.join(input_dir, name))
            array = np.zeros(sitk.GetArrayViewFromImage(reference).shape, dtype=np.uint8)
            # One slice per label, so every requested value is present.
            for slice_index, value in enumerate(labels_present):
                array[slice_index] = value
            mask = sitk.GetImageFromArray(array)
            mask.CopyInformation(reference)
            sitk.WriteImage(mask, os.path.join(output_dir, f"{case_id}.nii.gz"))
            if on_case_done is not None:
                on_case_done(case_id, index, len(names))

    return predict_folder


@pytest.fixture
def stub_nnunet(monkeypatch):
    def _install(labels_present=(1, 2, 3)):
        monkeypatch.setattr(nnunet_runner, "predict_folder", _stub_prediction(labels_present))
        monkeypatch.setattr(nnunet_runner, "check_dependencies", lambda: None)
        monkeypatch.setattr(nnunet_runner, "resolve_device", lambda requested=None: "cpu")

    return _install



# Marching cubes needs a real volume: at 8x8x8 (the default above, which is all
# the segmentation tests need) vtkNrrdReader hands VTK a single slice and every
# mesh comes out empty. Measured: it reads the file correctly from 16^3 up.
# Meshing is the ONE thing here that depends on the data being three-
# dimensional, so the mesh tests carry their own scan.
_MESH_SCAN_SIZE = (24, 24, 24)


def _write_mesh_scan(path: str) -> str:
    return _write_scan(path, size=_MESH_SCAN_SIZE)


def _blocks(labels_present):
    """Stand in for nnUNet with SOLID BLOCKS rather than single slices, so each
    label has a closed surface to contour."""

    def predict_folder(model_folder, input_dir, output_dir, device,
                       on_case_start=None, on_case_done=None):
        os.makedirs(output_dir, exist_ok=True)
        names = [n for n in sorted(os.listdir(input_dir)) if n.endswith("_0000.nii.gz")]
        for case_index, name in enumerate(names):
            case_id = name[: -len("_0000.nii.gz")]
            if on_case_start is not None:
                on_case_start(case_id, case_index, len(names))
            reference = sitk.ReadImage(os.path.join(input_dir, name))
            shape = sitk.GetArrayViewFromImage(reference).shape
            array = np.zeros(shape, dtype=np.uint8)
            depth = shape[0] // (len(labels_present) + 1)
            for index, value in enumerate(labels_present):
                start = 2 + index * depth
                array[start: start + depth - 1, 2:-2, 2:-2] = value
            mask = sitk.GetImageFromArray(array)
            mask.CopyInformation(reference)
            sitk.WriteImage(mask, os.path.join(output_dir, f"{case_id}.nii.gz"))
            if on_case_done is not None:
                on_case_done(case_id, case_index, len(names))

    return predict_folder


@pytest.fixture
def stub_nnunet_meshable(monkeypatch):
    def _install(labels_present=(1, 2)):
        monkeypatch.setattr(nnunet_runner, "predict_folder", _blocks(labels_present))
        monkeypatch.setattr(nnunet_runner, "check_dependencies", lambda: None)
        monkeypatch.setattr(nnunet_runner, "resolve_device", lambda requested=None: "cpu")

    return _install


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

def test_the_catalog_is_keyed_by_the_folder_the_manifest_downloads():
    """The bundle's directory name IS the model: data-manifest.yml writes each
    one into a folder of that name, and the client picks that name from
    GET /tools/BatchDentalSeg/data. A key that drifts from the manifest makes
    an installed model unselectable.

    Skipped rather than failed when scripts/ is out of reach: the test
    container mounts server/ only, and that is not a reason to fail a push.
    """
    manifest = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "data-manifest.yml")
    )
    if not os.path.isfile(manifest):
        pytest.skip("scripts/data-manifest.yml is not mounted in this environment")

    with open(manifest, encoding="utf-8") as handle:
        text = handle.read()
    block = text[text.index("  BatchDentalSeg:"):]
    block = block[: block.index("\n\n  #")]
    for name in catalogs.MODEL_NAMES:
        assert f"{name}/" in block or f"dest: {name}" in block, name


def test_label_values_are_unique_within_a_model():
    """Two names sharing one integer would make the split silently write the
    same mask twice under different anatomy."""
    for model in catalogs.MODELS.values():
        values = list(model.labels.values())
        assert len(values) == len(set(values)), model.name


def test_the_universal_model_labels_every_permanent_tooth():
    """1-32 in Universal numbering, which is what downstream tools index by."""
    universal = catalogs.get("UniversalLab")
    assert set(range(1, 33)) <= set(universal.labels.values())
    assert universal.labels["Mandibular canal"] == 55


def test_naso_maxilla_separates_the_maxilla_and_shifts_the_rest():
    """The whole point of that model, and the reason its table is its own: an
    off-by-one here labels the canal as teeth."""
    naso = catalogs.get("NasoMaxillaDentSeg")
    five = catalogs.get("DentalSegmentator")
    assert naso.labels["Maxilla"] == 3
    assert naso.labels["Mandibular canal"] == 6
    assert five.labels["Mandibular canal"] == 5
    assert "Maxilla" not in five.labels


# ---------------------------------------------------------------------------
# Model bundle discovery
# ---------------------------------------------------------------------------

def test_a_bundle_is_found_however_deeply_the_archive_nested_it(tmp_path):
    """DentalSegmentator arrives as a zip with its own Dataset<n>/ tree, the
    other three as flat files: one discovery rule has to cover both."""
    root = str(tmp_path / "models")
    expected = _model_bundle(root, "DentalSegmentator")
    assert nnunet_runner.find_model_folder(os.path.join(root, "DentalSegmentator")) == expected


def test_a_bundle_missing_its_checkpoint_is_not_accepted(tmp_path):
    """A half-downloaded bundle must report 'not installed', not fail inside
    nnUNet's loader."""
    base = _model_bundle(str(tmp_path / "models"), "DentalSegmentator")
    os.remove(os.path.join(base, "fold_0", "checkpoint_final.pth"))
    assert nnunet_runner.find_model_folder(str(tmp_path / "models" / "DentalSegmentator")) is None


def test_an_unusable_bundle_is_an_argument_error_naming_the_setup_command(tmp_path):
    """422, not 500: the request is fine, the deployment's data is not."""
    os.makedirs(str(tmp_path / "models" / "DentalSegmentator"), exist_ok=True)
    with pytest.raises(ToolArgumentError, match="setup-models.sh"):
        BatchDentalSegLogic.resolve_model(str(tmp_path / "models" / "DentalSegmentator"))


def test_a_bundle_whose_name_is_not_a_known_model_is_refused(tmp_path):
    """The label table comes from the bundle's name, so an unrecognised one
    must stop the run: guessing a table would name every structure wrong."""
    _model_bundle(str(tmp_path / "models"), "SomeOtherBundle")
    with pytest.raises(ToolArgumentError, match="not a BatchDentalSeg model"):
        BatchDentalSegLogic.resolve_model(str(tmp_path / "models" / "SomeOtherBundle"))


def test_the_bundle_name_selects_the_label_table(tmp_path):
    """Picking the NasoMaxilla bundle must bring NasoMaxilla's six labels, not
    the five-label table -- they disagree from value 3 onwards."""
    _model_bundle(str(tmp_path / "models"), "NasoMaxillaDentSeg")
    model, _folder = BatchDentalSegLogic.resolve_model(
        str(tmp_path / "models" / "NasoMaxillaDentSeg")
    )
    assert model.name == "NasoMaxillaDentSeg"
    assert model.labels["Maxilla"] == 3


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_discovery_is_recursive(tmp_path):
    _write_scan(str(tmp_path / "in" / "a" / "scan1.nii.gz"))
    _write_scan(str(tmp_path / "in" / "b" / "deep" / "scan2.nii.gz"))
    found = BatchDentalSegLogic.discover_scans(str(tmp_path / "in"), "Seg", str(tmp_path / "s"))
    assert len(found) == 2


def test_a_previous_run_is_not_re_ingested(tmp_path):
    """`scan_Seg.nii.gz` sorts before `scan.nii.gz`, so without this a second
    run would segment the first run's output."""
    _write_scan(str(tmp_path / "in" / "scan.nii.gz"))
    _write_scan(str(tmp_path / "in" / "scan_Seg.nii.gz"))
    found = BatchDentalSegLogic.discover_scans(str(tmp_path / "in"), "Seg", str(tmp_path / "s"))
    assert [os.path.basename(path) for path in found] == ["scan.nii.gz"]


def test_an_input_with_no_scan_is_an_argument_error(tmp_path, stub_nnunet):
    stub_nnunet()
    os.makedirs(str(tmp_path / "in"), exist_ok=True)
    with open(str(tmp_path / "in" / "notes.txt"), "w") as handle:
        handle.write("nothing here")
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    with pytest.raises(ToolArgumentError, match="No scan found"):
        BatchDentalSegLogic.segment(
            input_path=str(tmp_path / "in"), model_path=str(tmp_path / "models" / "DentalSegmentator")
        )


# ---------------------------------------------------------------------------
# A run
# ---------------------------------------------------------------------------

def test_a_run_writes_one_segmentation_per_scan_and_a_report(tmp_path, stub_nnunet):
    stub_nnunet()
    _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _write_scan(str(tmp_path / "in" / "p2.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    run = BatchDentalSegLogic.segment(
        input_path=str(tmp_path / "in"), model_path=str(tmp_path / "models" / "DentalSegmentator")
    )

    produced = sorted(os.path.basename(path) for path in run.segmentation_files)
    assert produced == ["p1_Seg.nii.gz", "p2_Seg.nii.gz"]
    assert run.report["summary"] == "2/2 scan(s) segmented"
    assert os.path.isfile(os.path.join(run.output_dir, "BatchDentalSeg_report.json"))


def test_the_report_carries_the_label_table(tmp_path, stub_nnunet):
    """The output is a label volume; without the table its integers mean
    nothing to whoever opens it, and the four models disagree on them."""
    stub_nnunet()
    _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "NasoMaxillaDentSeg")

    run = BatchDentalSegLogic.segment(
        input_path=str(tmp_path / "in"),
        model_path=str(tmp_path / "models" / "NasoMaxillaDentSeg"),
    )
    assert run.report["model"] == "NasoMaxillaDentSeg"
    assert run.report["labels"]["Maxilla"] == 3


def test_the_output_mirrors_the_input_tree(tmp_path, stub_nnunet):
    """Two patients whose scans share a file name must stay apart -- keying on
    the base name would collapse them into one output file."""
    stub_nnunet()
    _write_scan(str(tmp_path / "in" / "subjectA" / "scan.nii.gz"))
    _write_scan(str(tmp_path / "in" / "subjectB" / "scan.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    run = BatchDentalSegLogic.segment(
        input_path=str(tmp_path / "in"), model_path=str(tmp_path / "models" / "DentalSegmentator")
    )

    relative = sorted(os.path.relpath(path, run.output_dir) for path in run.segmentation_files)
    assert relative == [
        os.path.join("subjectA", "scan_Seg.nii.gz"),
        os.path.join("subjectB", "scan_Seg.nii.gz"),
    ]


def test_the_segmentation_lands_on_the_input_scan_geometry(tmp_path, stub_nnunet):
    """A mask whose origin differs from its scan opens offset from the anatomy
    it describes, and nothing in the report would say so."""
    stub_nnunet()
    scan = _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    run = BatchDentalSegLogic.segment(
        input_path=str(tmp_path / "in"), model_path=str(tmp_path / "models" / "DentalSegmentator")
    )

    reference = sitk.ReadImage(scan)
    produced = sitk.ReadImage(run.segmentation_files[0])
    assert produced.GetSize() == reference.GetSize()
    assert np.allclose(produced.GetOrigin(), reference.GetOrigin())
    assert np.allclose(produced.GetSpacing(), reference.GetSpacing())


def test_separate_segments_writes_only_the_labels_actually_present(tmp_path, stub_nnunet):
    """A UniversalLab run would otherwise write 55 files per patient, most of
    them empty -- and an empty mask reads like a structure the model missed."""
    stub_nnunet(labels_present=(1, 3))
    _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    run = BatchDentalSegLogic.segment(
        input_path=str(tmp_path / "in"),
        model_path=str(tmp_path / "models" / "DentalSegmentator"),
        separate_segments=True,
    )

    names = sorted(os.path.basename(path) for path in run.segmentation_files)
    assert names == ["p1_Seg.nii.gz", "p1_Seg_Upper-Skull.nii.gz", "p1_Seg_Upper-Teeth.nii.gz"]


def test_a_separate_segment_holds_only_its_own_label(tmp_path, stub_nnunet):
    stub_nnunet(labels_present=(1, 2))
    _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    run = BatchDentalSegLogic.segment(
        input_path=str(tmp_path / "in"),
        model_path=str(tmp_path / "models" / "DentalSegmentator"),
        separate_segments=True,
    )

    mandible = next(path for path in run.segmentation_files if path.endswith("Mandible.nii.gz"))
    # GetArrayFromImage, not GetArrayViewFromImage: a VIEW borrows the image's
    # buffer, and reading it off a temporary the expression drops reads freed
    # memory -- which looks exactly like a corrupt mask.
    values = set(np.unique(sitk.GetArrayFromImage(sitk.ReadImage(mandible))).tolist())
    assert values <= {0, 1}, "a per-segment file is binary"
    assert 1 in values


def test_an_unreadable_scan_does_not_lose_the_others(tmp_path, stub_nnunet, monkeypatch):
    """One corrupt patient in a cohort of forty must not cost the other
    thirty-nine.

    The scan is unreadable at CONVERSION time, which happens before inference:
    an unguarded loop there aborts the whole run before a single scan has been
    segmented, which is what this pins.
    """
    stub_nnunet()
    _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    # A file with a scan extension and no valid volume in it -- exactly what a
    # truncated upload or a mislabelled file looks like.
    with open(str(tmp_path / "in" / "p2.nii.gz"), "wb") as handle:
        handle.write(b"not a volume")
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    run = BatchDentalSegLogic.segment(
        input_path=str(tmp_path / "in"), model_path=str(tmp_path / "models" / "DentalSegmentator")
    )

    statuses = {entry["input"]: entry["status"] for entry in run.report["scans"]}
    assert statuses == {"p1.nii.gz": "ok", "p2.nii.gz": "failed"}
    assert run.report["summary"] == "1/2 scan(s) segmented"
    assert len(run.segmentation_files) == 1


def test_a_batch_of_only_unreadable_scans_is_an_argument_error(tmp_path, stub_nnunet):
    """Nothing to infer on: say so rather than hand nnUNet an empty folder and
    report a successful run of zero scans."""
    stub_nnunet()
    os.makedirs(str(tmp_path / "in"), exist_ok=True)
    with open(str(tmp_path / "in" / "p1.nii.gz"), "wb") as handle:
        handle.write(b"not a volume")
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    with pytest.raises(ToolArgumentError, match="could be read"):
        BatchDentalSegLogic.segment(
            input_path=str(tmp_path / "in"), model_path=str(tmp_path / "models" / "DentalSegmentator")
        )


def test_a_zip_input_is_unpacked(tmp_path, stub_nnunet):
    """A batch reaches run() as an archive, because the schema declares the
    volume type first so the client's file picker stays a file picker."""
    import zipfile

    stub_nnunet()
    _write_scan(str(tmp_path / "src" / "p1.nii.gz"))
    archive = str(tmp_path / "batch.zip")
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(str(tmp_path / "src" / "p1.nii.gz"), "cohort/p1.nii.gz")
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    run = BatchDentalSegLogic.segment(
        input_path=archive, model_path=str(tmp_path / "models" / "DentalSegmentator")
    )
    assert len(run.segmentation_files) == 1


def test_nnunet_case_ids_are_positional_not_patient_names(tmp_path, stub_nnunet, monkeypatch):
    """nnUNet writes its output under the id it was given, so deriving the id
    from the file name would make two `scan.nii.gz` overwrite each other."""
    seen = {}

    def capture(model_folder, input_dir, output_dir, device):
        seen["inputs"] = sorted(os.listdir(input_dir))
        _stub_prediction((1,))(model_folder, input_dir, output_dir, device)

    monkeypatch.setattr(nnunet_runner, "predict_folder", capture)
    monkeypatch.setattr(nnunet_runner, "check_dependencies", lambda: None)
    monkeypatch.setattr(nnunet_runner, "resolve_device", lambda requested=None: "cpu")

    _write_scan(str(tmp_path / "in" / "a" / "scan.nii.gz"))
    _write_scan(str(tmp_path / "in" / "b" / "scan.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    BatchDentalSegLogic.segment(
        input_path=str(tmp_path / "in"), model_path=str(tmp_path / "models" / "DentalSegmentator")
    )
    assert seen["inputs"] == ["case_0000_0000.nii.gz", "case_0001_0000.nii.gz"]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_schema_is_valid():
    BatchDentalSegTool().check_schema()


def test_input_alone_is_not_a_complete_request():
    """`model` names the hosted bundle and is required: without it there is
    nothing to run."""
    tool = BatchDentalSegTool()
    with pytest.raises(ToolArgumentError, match="model"):
        tool.validate({"input": "/tmp/scan.nii.gz"})


def test_the_model_is_a_name_not_an_upload():
    """`model` is a scalar `server_selectable`, so the weights are selected by
    name and never travel: main.py refuses an upload for it with a 400."""
    spec = BatchDentalSegTool().arguments["model"]
    assert spec.server_selectable == "model"
    assert not spec.is_file


def test_unexpected_arguments_are_refused():
    tool = BatchDentalSegTool()
    with pytest.raises(ToolArgumentError, match="Unexpected argument"):
        tool.validate(
            {"input": "/tmp/scan.nii.gz", "model": "bundle", "dental_model": "NotAModel"}
        )


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

def test_a_structure_keeps_its_colour_across_models():
    """NasoMaxillaDentSeg separates the maxilla, so the upper teeth are 3 under
    one model and 4 under another. The palette is keyed by NAME for that
    reason: indexed by value it would recolour the teeth on that one model."""
    five = catalogs.get("DentalSegmentator").label_colors
    naso = catalogs.get("NasoMaxillaDentSeg").label_colors
    assert five["Upper Teeth"] == naso["Upper Teeth"]
    assert five["Mandible"] == naso["Mandible"]


def test_every_structure_gets_a_colour_including_the_unnamed_ones():
    """UniversalLab's 52 teeth are in no colour table; they are generated, and
    every one of them must come out as a real hex colour."""
    colors = catalogs.get("UniversalLab").label_colors
    assert len(colors) == 55
    assert all(len(value) == 7 and value.startswith("#") for value in colors.values())


def test_consecutive_generated_colours_are_distinguishable():
    """Adjacent teeth are what a clinician has to tell apart."""
    for value in range(1, 33):
        here = catalogs.rgb_of(catalogs.color_of(f"tooth {value}", value))
        following = catalogs.rgb_of(catalogs.color_of(f"tooth {value + 1}", value + 1))
        assert sum(abs(a - b) for a, b in zip(here, following)) > 60


# ---------------------------------------------------------------------------
# Mesh export
# ---------------------------------------------------------------------------

def test_no_export_format_selected_writes_no_mesh(tmp_path, stub_nnunet_meshable):
    """The default. Meshing a UniversalLab scan is 55 surfaces, minutes of CPU
    and hundreds of MB nobody asked for."""
    stub_nnunet_meshable()
    _write_mesh_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    run = BatchDentalSegLogic.segment(
        input_path=str(tmp_path / "in"),
        model_path=str(tmp_path / "models" / "DentalSegmentator"),
    )
    assert run.surface_files == []
    assert run.report["export_formats"] == []
    # Reported as None rather than as their defaults: a number recorded for a
    # run that built nothing reads as a setting that was applied.
    assert run.report["surface_smoothing"] is None


@pytest.mark.parametrize(
    "selection, extension",
    [({"VTK": True}, ".vtk"), ({"STL": True}, ".stl"), ({"OBJ": True}, ".obj")],
)
def test_each_format_writes_one_file_per_present_structure(
    tmp_path, stub_nnunet_meshable, selection, extension
):
    pytest.importorskip("vtk")
    stub_nnunet_meshable(labels_present=(1, 2))
    _write_mesh_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    run = BatchDentalSegLogic.segment(
        input_path=str(tmp_path / "in"),
        model_path=str(tmp_path / "models" / "DentalSegmentator"),
        export_formats=selection,
    )

    produced = sorted(os.path.basename(path) for path in run.surface_files)
    # Only the labels PRESENT in this scan: an empty mesh is indistinguishable
    # from a structure the model failed on.
    assert produced == [f"p1_Seg_Mandible{extension}", f"p1_Seg_Upper-Skull{extension}"]
    assert all(os.path.getsize(path) > 0 for path in run.surface_files)


def test_the_merged_format_is_one_file_for_the_whole_scan(tmp_path, stub_nnunet_meshable):
    pytest.importorskip("vtk")
    stub_nnunet_meshable(labels_present=(1, 2, 3))
    _write_mesh_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    run = BatchDentalSegLogic.segment(
        input_path=str(tmp_path / "in"),
        model_path=str(tmp_path / "models" / "DentalSegmentator"),
        export_formats={"VTK (merged)": True},
    )
    assert [os.path.basename(path) for path in run.surface_files] == ["p1_Seg_merged.vtk"]


def test_several_formats_can_be_asked_for_at_once(tmp_path, stub_nnunet_meshable):
    pytest.importorskip("vtk")
    stub_nnunet_meshable(labels_present=(1,))
    _write_mesh_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    run = BatchDentalSegLogic.segment(
        input_path=str(tmp_path / "in"),
        model_path=str(tmp_path / "models" / "DentalSegmentator"),
        export_formats={"VTK": True, "STL": True, "OBJ": False, "VTK (merged)": True},
    )
    produced = sorted(os.path.basename(path) for path in run.surface_files)
    assert produced == ["p1_Seg_Upper-Skull.stl", "p1_Seg_Upper-Skull.vtk", "p1_Seg_merged.vtk"]
    # An option left unchecked is not a selection: what is sent IS the choice.
    assert run.report["export_formats"] == ["VTK", "STL", "VTK (merged)"]


def test_the_segmentation_is_still_written_alongside_the_meshes(tmp_path, stub_nnunet_meshable):
    """The label volume is what every downstream tool consumes; a mesh is an
    addition to it, never a replacement."""
    pytest.importorskip("vtk")
    stub_nnunet_meshable(labels_present=(1,))
    _write_mesh_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    run = BatchDentalSegLogic.segment(
        input_path=str(tmp_path / "in"),
        model_path=str(tmp_path / "models" / "DentalSegmentator"),
        export_formats={"STL": True},
    )
    assert [os.path.basename(path) for path in run.segmentation_files] == ["p1_Seg.nii.gz"]


def test_a_binary_vtk_is_written_not_ascii(tmp_path, stub_nnunet_meshable):
    """vtkPolyDataWriter defaults to ASCII, and that default is what made
    AMASSS's merged surface 848.5 MB against 6.4 MB for its segmentations.
    Binary is also the more accurate of the two: it round-trips the float32
    vertices exactly."""
    pytest.importorskip("vtk")
    stub_nnunet_meshable(labels_present=(1,))
    _write_mesh_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    run = BatchDentalSegLogic.segment(
        input_path=str(tmp_path / "in"),
        model_path=str(tmp_path / "models" / "DentalSegmentator"),
        export_formats={"VTK": True},
    )
    with open(run.surface_files[0], "rb") as handle:
        header = handle.read(200)
    assert b"BINARY" in header and b"ASCII" not in header


def test_a_requested_format_without_vtk_fails_before_inference(tmp_path, stub_nnunet, monkeypatch):
    """Discovering it after an hour of inference would waste the whole run."""
    stub_nnunet()
    _write_mesh_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")
    monkeypatch.setattr(BatchDentalSegLogic.surface_export, "is_available", lambda: False)

    called = []
    monkeypatch.setattr(
        nnunet_runner, "predict_folder", lambda *a, **k: called.append(True)
    )
    with pytest.raises(ToolArgumentError, match="VTK"):
        BatchDentalSegLogic.segment(
            input_path=str(tmp_path / "in"),
            model_path=str(tmp_path / "models" / "DentalSegmentator"),
            export_formats={"STL": True},
        )
    assert called == []


def test_the_selection_survives_the_schema_round_trip():
    """`validate()` turns a multichoice into a Selection (every option mapped
    to a boolean). _requested_formats has to read that, a plain list, and a
    single name -- a server-side caller must not have to build a Selection."""
    tool = BatchDentalSegTool()
    cleaned = tool.validate(
        {"input": "/tmp/scan.nii.gz", "model": "DentalSegmentator", "export_formats": "STL,VTK"}
    )
    assert BatchDentalSegLogic._requested_formats(cleaned["export_formats"]) == ["VTK", "STL"]
    assert BatchDentalSegLogic._requested_formats(["OBJ"]) == ["OBJ"]
    assert BatchDentalSegLogic._requested_formats("VTK") == ["VTK"]
    assert BatchDentalSegLogic._requested_formats(None) == []


def test_an_unknown_export_format_is_refused_by_the_schema():
    tool = BatchDentalSegTool()
    with pytest.raises(ToolArgumentError, match="gltf"):
        tool.validate(
            {"input": "/tmp/scan.nii.gz", "model": "DentalSegmentator", "export_formats": "gltf"}
        )


def test_a_structure_that_meshes_to_nothing_produces_no_file(tmp_path, stub_nnunet):
    """A label too thin to contour must leave NO file, not an empty one.

    The three writers disagree on what an empty mesh means: vtkPolyDataWriter
    writes a valid .vtk with a header and no triangle, while vtkSTLWriter and
    vtkOBJWriter write nothing at all and only complain on the console. Neither
    raises. So without an explicit check, the same run would ship an empty .vtk
    as a success -- a structure the clinician believes was found and cannot
    see -- and list .stl paths in the report that are not on disk.

    The default fixture's 8^3 volume is what produces this, and not by
    accident: vtkNrrdReader hands VTK a single slice below 16^3, so marching
    cubes returns nothing. It is the cheapest way to reach the empty-mesh path.
    """
    pytest.importorskip("vtk")
    stub_nnunet(labels_present=(1, 2))
    _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    run = BatchDentalSegLogic.segment(
        input_path=str(tmp_path / "in"),
        model_path=str(tmp_path / "models" / "DentalSegmentator"),
        export_formats={"VTK": True, "STL": True, "OBJ": True, "VTK (merged)": True},
    )

    assert run.surface_files == []
    # Every path the report DOES carry has to exist: that is the whole point.
    assert all(os.path.isfile(path) for path in run.surface_files)
    # And the run is still a success -- the segmentation is the deliverable.
    assert run.report["summary"] == "1/1 scan(s) segmented"
    assert [os.path.basename(path) for path in run.segmentation_files] == ["p1_Seg.nii.gz"]


# ---------------------------------------------------------------------------
# Streamed runs
# ---------------------------------------------------------------------------

def _collect(events: list):
    return lambda event: events.append(event)


def test_a_streamed_run_reports_every_scan_twice(tmp_path, stub_nnunet):
    """Once as it starts and once as it ends: a queue row that only appears
    when a scan is finished shows nothing at all during the minutes it takes."""
    stub_nnunet()
    _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _write_scan(str(tmp_path / "in" / "p2.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    events = []
    BatchDentalSegLogic.segment(
        input_path=str(tmp_path / "in"),
        model_path=str(tmp_path / "models" / "DentalSegmentator"),
        emit=_collect(events),
    )

    items = [e for e in events if e["event"] == "item"]
    assert [(e["index"], e["status"]) for e in items] == [
        (1, "running"), (1, "ok"), (2, "running"), (2, "ok"),
    ]
    assert all(e["total"] == 2 for e in items)


def test_a_streamed_run_hands_over_each_scan_as_it_finishes(tmp_path, stub_nnunet):
    """The artifact for scan 1 is emitted BEFORE scan 2 even starts -- which is
    the whole reason for the mechanism."""
    stub_nnunet()
    _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _write_scan(str(tmp_path / "in" / "p2.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    events = []
    BatchDentalSegLogic.segment(
        input_path=str(tmp_path / "in"),
        model_path=str(tmp_path / "models" / "DentalSegmentator"),
        emit=_collect(events),
    )

    kinds = [(e["event"], e.get("status"), e.get("name")) for e in events]
    first_artifact = next(i for i, e in enumerate(events) if e["event"] == "artifact")
    second_start = next(
        i for i, e in enumerate(events)
        if e["event"] == "item" and e.get("index") == 2 and e["status"] == "running"
    )
    assert first_artifact < second_start, kinds

    artifacts = [e for e in events if e["event"] == "artifact"]
    assert len(artifacts) == 2
    for artifact in artifacts:
        assert os.path.isfile(artifact["path"])
        with zipfile.ZipFile(artifact["path"]) as archive:
            assert archive.namelist()


def test_a_streamed_artifact_says_where_it_belongs_in_the_tree(tmp_path, stub_nnunet):
    """Two patients whose scans share a file name must not collide on the
    client either, so each artifact carries its directory relative to the
    output root -- never a path on this server."""
    stub_nnunet()
    _write_scan(str(tmp_path / "in" / "subjectA" / "scan.nii.gz"))
    _write_scan(str(tmp_path / "in" / "subjectB" / "scan.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    events = []
    BatchDentalSegLogic.segment(
        input_path=str(tmp_path / "in"),
        model_path=str(tmp_path / "models" / "DentalSegmentator"),
        emit=_collect(events),
    )
    relatives = sorted(e["relative_dir"] for e in events if e["event"] == "artifact")
    assert relatives == ["subjectA", "subjectB"]


def test_an_unreadable_scan_is_announced_and_the_others_still_ship(
    tmp_path, stub_nnunet, monkeypatch
):
    """A scan that never reaches the GPU would otherwise sit in the client's
    queue as pending forever, with the run ending on a row that never moved."""
    stub_nnunet()
    _write_scan(str(tmp_path / "in" / "good.nii.gz"))
    broken = tmp_path / "in" / "broken.nii.gz"
    broken.write_bytes(b"not a volume")
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    real_read = BatchDentalSegLogic.sitk.ReadImage

    def failing_read(path, *args, **kwargs):
        if str(path).endswith("broken.nii.gz"):
            raise RuntimeError("unreadable")
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(BatchDentalSegLogic.sitk, "ReadImage", failing_read)

    events = []
    run = BatchDentalSegLogic.segment(
        input_path=str(tmp_path / "in"),
        model_path=str(tmp_path / "models" / "DentalSegmentator"),
        emit=_collect(events),
    )

    failed = [e for e in events if e["event"] == "item" and e["status"] == "failed"]
    assert [e["name"] for e in failed] == ["broken.nii.gz"]
    assert [e["name"] for e in events if e["event"] == "artifact"] == ["good.nii.gz"]
    assert run.report["summary"] == "1/2 scan(s) segmented"


def test_a_streamed_run_produces_the_same_files_as_a_blocking_one(tmp_path, stub_nnunet):
    """The events are additional, not an alternative: the output tree, the
    report and the archive are identical either way. Two implementations would
    drift on exactly the details that decide what a clinician gets."""
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")
    produced = {}
    for label, emit in (("blocking", None), ("streamed", _collect([]))):
        stub_nnunet()
        root = tmp_path / label
        _write_scan(str(root / "in" / "p1.nii.gz"))
        _write_scan(str(root / "in" / "sub" / "p2.nii.gz"))
        run = BatchDentalSegLogic.segment(
            input_path=str(root / "in"),
            model_path=str(tmp_path / "models" / "DentalSegmentator"),
            emit=emit,
        )
        produced[label] = sorted(
            os.path.relpath(path, run.output_dir) for path in run.segmentation_files
        )
        assert run.report["summary"] == "2/2 scan(s) segmented"

    assert produced["blocking"] == produced["streamed"]


def test_the_report_is_written_whether_or_not_anyone_was_listening(tmp_path, stub_nnunet):
    stub_nnunet()
    _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    run = BatchDentalSegLogic.segment(
        input_path=str(tmp_path / "in"),
        model_path=str(tmp_path / "models" / "DentalSegmentator"),
        emit=_collect([]),
    )
    assert os.path.isfile(os.path.join(run.output_dir, "BatchDentalSeg_report.json"))


def test_a_runner_that_ignores_the_callbacks_still_produces_a_full_report(
    tmp_path, stub_nnunet, monkeypatch
):
    """Defensive, and cheap: an older runner copy (or a stub) that never calls
    back would otherwise report a run with no scans in it at all."""
    stub_nnunet()
    _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    silent = _stub_prediction((1, 2, 3))

    def ignores_callbacks(model_folder, input_dir, output_dir, device, **_kwargs):
        silent(model_folder, input_dir, output_dir, device)

    monkeypatch.setattr(nnunet_runner, "predict_folder", ignores_callbacks)

    events = []
    run = BatchDentalSegLogic.segment(
        input_path=str(tmp_path / "in"),
        model_path=str(tmp_path / "models" / "DentalSegmentator"),
        emit=_collect(events),
    )
    assert run.report["summary"] == "1/1 scan(s) segmented"


def test_the_per_scan_call_hands_nnunet_a_truncated_output_path(tmp_path, monkeypatch):
    """nnUNet APPENDS its own file ending to the output path it is given.

    This is the one thing the stubs above cannot catch, because they stand in
    for nnUNet entirely -- and it is what made the first streamed run report
    "nnUNet produced no output for this scan" for every scan: the path carried
    `.nii.gz`, nnUNet wrote `case_0000.nii.gz.nii.gz`, and the caller looked
    for `case_0000.nii.gz`. So the boundary is pinned here instead.
    """
    calls = []

    class FakePredictor:
        def initialize_from_trained_model_folder(self, *args, **kwargs):
            pass

        def predict_from_files(self, inputs, outputs, **kwargs):
            calls.append((inputs, outputs))

    monkeypatch.setattr(nnunet_runner, "_build_predictor", lambda device: FakePredictor())

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "case_0000_0000.nii.gz").write_bytes(b"")

    nnunet_runner.predict_folder(
        str(tmp_path / "model"), str(input_dir), str(tmp_path / "out"), "cpu",
        on_case_start=lambda *a: None, on_case_done=lambda *a: None,
    )

    (_inputs, outputs), = calls
    assert outputs == [os.path.join(str(tmp_path / "out"), "case_0000")]
    assert not outputs[0].endswith(".nii.gz")


def test_the_batch_call_still_hands_nnunet_the_folder(tmp_path, monkeypatch):
    """The blocking path is unchanged: nnUNet builds the names itself there."""
    calls = []

    class FakePredictor:
        def initialize_from_trained_model_folder(self, *args, **kwargs):
            pass

        def predict_from_files(self, inputs, outputs, **kwargs):
            calls.append((inputs, outputs))

    monkeypatch.setattr(nnunet_runner, "_build_predictor", lambda device: FakePredictor())
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "case_0000_0000.nii.gz").write_bytes(b"")

    nnunet_runner.predict_folder(
        str(tmp_path / "model"), str(input_dir), str(tmp_path / "out"), "cpu"
    )
    (inputs, outputs), = calls
    assert inputs == str(input_dir)
    assert outputs == str(tmp_path / "out")


# ---------------------------------------------------------------------------
# GPU resampling
# ---------------------------------------------------------------------------

class _FakeLabelManager:
    def __init__(self, heads):
        self.num_segmentation_heads = heads


class _FakePredictor:
    def __init__(self, heads=55):
        self.label_manager = _FakeLabelManager(heads)


def _free_vram(monkeypatch, free_bytes):
    """Stand in for torch.cuda.mem_get_info without a card."""
    class _Device:
        def __init__(self, name):
            pass

    fake_torch = type("torch", (), {
        "cuda": type("cuda", (), {"mem_get_info": staticmethod(lambda device: (free_bytes, free_bytes))}),
        "device": _Device,
    })
    monkeypatch.setattr(nnunet_runner, "_import_torch", lambda: fake_torch)


def test_gpu_resampling_is_refused_when_it_would_not_fit(tmp_path, monkeypatch):
    """UniversalLab emits 55 classes, so the resampled array is
    (55, Z, Y, X) float32 -- around 20 GiB on a 512x512x365 CBCT. On a 16 GB
    card that is a CUDA OOM minutes into the run; falling back to the slow but
    correct scipy path is always better than failing."""
    _free_vram(monkeypatch, 16 * 2**30)
    monkeypatch.setattr(nnunet_runner, "_largest_scan_voxels", lambda d: 512 * 512 * 365)
    assert not nnunet_runner._gpu_resampling_fits(_FakePredictor(55), "cuda", str(tmp_path))


def test_gpu_resampling_is_accepted_on_the_card_it_was_measured_on(tmp_path, monkeypatch):
    """47 GiB free, 55 classes, 512x512x365: this is the exact configuration
    the 5.83x was measured on, so the guard must say yes to it. An estimate
    that refuses the card the measurement came from is worse than no guard."""
    _free_vram(monkeypatch, 47 * 2**30)
    monkeypatch.setattr(nnunet_runner, "_largest_scan_voxels", lambda d: 512 * 512 * 365)
    assert nnunet_runner._gpu_resampling_fits(_FakePredictor(55), "cuda", str(tmp_path))


def test_a_five_class_model_fits_where_the_universal_one_does_not(tmp_path, monkeypatch):
    """The guard is about CLASSES, not about the card: the same scan on
    DentalSegmentator needs an eleventh of the memory."""
    _free_vram(monkeypatch, 16 * 2**30)
    monkeypatch.setattr(nnunet_runner, "_largest_scan_voxels", lambda d: 512 * 512 * 365)
    assert nnunet_runner._gpu_resampling_fits(_FakePredictor(5), "cuda", str(tmp_path))


def test_the_largest_scan_of_the_batch_is_what_decides(tmp_path):
    """A cohort is sized by its biggest member: one large scan among small ones
    would OOM on its own."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    _write_scan(str(input_dir / "small_0000.nii.gz"), size=(8, 8, 8))
    _write_scan(str(input_dir / "big_0000.nii.gz"), size=(24, 24, 24))
    assert nnunet_runner._largest_scan_voxels(str(input_dir)) == 24 * 24 * 24


def test_a_bundle_pinning_its_own_resampler_opts_itself_out(monkeypatch):
    """We have no idea what a non-default resampler was trained to expect."""
    class _ConfigurationManager:
        configuration = {"resampling_fn_data": "something_custom",
                         "resampling_fn_probabilities": "something_custom"}

    predictor = _FakePredictor()
    predictor.configuration_manager = _ConfigurationManager()
    monkeypatch.setattr(nnunet_runner, "_import_torch", lambda: __import__("types"))
    assert not nnunet_runner._enable_gpu_resampling(predictor, "cuda")


def test_gpu_resampling_never_applies_on_cpu():
    assert not nnunet_runner._enable_gpu_resampling(_FakePredictor(), "cpu")
