"""BatchDentalSeg unit tests.

`nnunet_runner.predict_folder` is stubbed, so no GPU, no weights and no network
are needed. Everything around inference runs for real: discovery, the NIfTI
conversion, geometry matching, the output tree, the per-segment split and the
report.
"""

import json
import os

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
    """Stand in for nnUNet: write one label volume per input case."""

    def predict_folder(model_folder, input_dir, output_dir, device):
        os.makedirs(output_dir, exist_ok=True)
        for name in sorted(os.listdir(input_dir)):
            if not name.endswith("_0000.nii.gz"):
                continue
            case_id = name[: -len("_0000.nii.gz")]
            reference = sitk.ReadImage(os.path.join(input_dir, name))
            array = np.zeros(sitk.GetArrayViewFromImage(reference).shape, dtype=np.uint8)
            # One slice per label, so every requested value is present.
            for index, value in enumerate(labels_present):
                array[index] = value
            mask = sitk.GetImageFromArray(array)
            mask.CopyInformation(reference)
            sitk.WriteImage(mask, os.path.join(output_dir, f"{case_id}.nii.gz"))

    return predict_folder


@pytest.fixture
def stub_nnunet(monkeypatch):
    def _install(labels_present=(1, 2, 3)):
        monkeypatch.setattr(nnunet_runner, "predict_folder", _stub_prediction(labels_present))
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
