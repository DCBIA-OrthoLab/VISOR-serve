"""Unit tests for the AMASSS tool logic.

No GPU and no real nnUNet models are needed: `nnunet_runner.predict_folder`
is monkeypatched with a stub that writes synthetic masks, so everything
around the inference itself -- input discovery, output filtering, model
resolution, format conversion, label merging, file naming, the report -- is
exercised for real.

Run just this module:
    cd server && ./venv/bin/pytest tools/AMASSS/test/
"""

import json
import os
import zipfile

# Set before anything imports config.Settings() (file_utils does, via the
# tool logic), so the suite runs regardless of the local environment.
os.environ.setdefault("API_TOKEN", "test-token")

import numpy as np
import pytest
import SimpleITK as sitk

from base import SELECTION_TYPE, ArgSpec, Tool, ToolArgumentError
from tools.AMASSS.src import AMASSSLogic, nnunet_runner


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_scan(path, size=(8, 8, 8), value=100):
    array = np.full(size[::-1], value, dtype=np.int16)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 0.5, 0.5))
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    sitk.WriteImage(image, str(path))
    return str(path)


def _make_model_bundle(root, codes):
    """A bundle laid out exactly as find_model_folder expects."""
    for code in codes:
        plans = root / code / "Dataset001_X" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        (plans / "fold_0").mkdir(parents=True)
        (plans / "fold_0" / nnunet_runner.CHECKPOINT_NAME).write_bytes(b"fake checkpoint")
    return str(root)


@pytest.fixture
def stub_predictor(monkeypatch):
    """Replace nnUNet inference with a deterministic synthetic mask writer."""

    def fake_predict_folder(model_folder, input_dir, output_dir, device):
        os.makedirs(output_dir, exist_ok=True)
        for name in sorted(os.listdir(input_dir)):
            if not name.endswith("_0000.nii.gz"):
                continue
            case_id = name[: -len("_0000.nii.gz")]
            reference = sitk.ReadImage(os.path.join(input_dir, name))
            array = np.zeros(sitk.GetArrayFromImage(reference).shape, dtype=np.uint8)
            array[2:5, 2:5, 2:5] = 1
            mask = sitk.GetImageFromArray(array)
            mask.CopyInformation(reference)
            sitk.WriteImage(mask, os.path.join(output_dir, f"{case_id}.nii.gz"))

    monkeypatch.setattr(nnunet_runner, "predict_folder", fake_predict_folder)
    monkeypatch.setattr(nnunet_runner, "resolve_device", lambda requested: "cpu")


# ---------------------------------------------------------------------------
# split_scan_extension / is_previous_output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filename, expected",
    [
        ("scan.nii.gz", ("scan", ".nii.gz")),
        ("scan.nii", ("scan", ".nii")),
        ("scan.nrrd", ("scan", ".nrrd")),
        ("scan.gipl.gz", ("scan", ".gipl.gz")),
        ("a.b.nii.gz", ("a.b", ".nii.gz")),
    ],
)
def test_split_scan_extension(filename, expected):
    assert AMASSSLogic.split_scan_extension(filename) == expected


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("patient01.nii.gz", False),
        ("patient01_Pred_MAND.nii.gz", True),
        ("patient01_Pred_MERGED.nii.gz", True),
        ("patient01_Pred_CBMASK.nii.gz", True),
        # A legitimate scan whose name merely contains MASK is NOT an output:
        # the original CLI's blunt `'MASK' not in f` filter dropped it.
        ("MASKED_patient.nii.gz", False),
    ],
)
def test_is_previous_output(filename, expected):
    assert AMASSSLogic.is_previous_output(filename, "Pred") is expected


def test_discover_scans_excludes_previous_outputs(tmp_path):
    """Regression: running twice on the same folder must not re-ingest run 1."""
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    _write_scan(tmp_path / "input" / "patient01_Pred_MAND.nii.gz")
    _write_scan(tmp_path / "input" / "patient01_Pred_MERGED.nii.gz")

    scans = AMASSSLogic.discover_scans(str(tmp_path / "input"), "Pred", str(tmp_path / "scratch"))

    assert [os.path.basename(path) for path in scans] == ["patient01.nii.gz"]


def test_discover_scans_is_recursive(tmp_path):
    """The Slicer UI always counted recursively; the CLI did not. It does now."""
    _write_scan(tmp_path / "input" / "a.nii.gz")
    _write_scan(tmp_path / "input" / "nested" / "deeper" / "b.nii.gz")

    scans = AMASSSLogic.discover_scans(str(tmp_path / "input"), "Pred", str(tmp_path / "scratch"))

    assert sorted(os.path.basename(path) for path in scans) == ["a.nii.gz", "b.nii.gz"]


def test_discover_scans_accepts_zip(tmp_path):
    scan = _write_scan(tmp_path / "src" / "patient01.nii.gz")
    zip_path = tmp_path / "input.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(scan, "patient01.nii.gz")

    scans = AMASSSLogic.discover_scans(str(zip_path), "Pred", str(tmp_path / "scratch"))

    assert [os.path.basename(path) for path in scans] == ["patient01.nii.gz"]


def test_discover_scans_raises_instead_of_exiting(tmp_path):
    """The original called sys.exit(1); a server needs a catchable exception."""
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        AMASSSLogic.discover_scans(str(tmp_path / "empty"), "Pred", str(tmp_path / "scratch"))


# ---------------------------------------------------------------------------
# model resolution
# ---------------------------------------------------------------------------

def test_resolve_models_reports_missing_structures(tmp_path):
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND"])

    available, missing = AMASSSLogic.resolve_models(
        bundle, ("MAND", "MAX"), str(tmp_path / "scratch")
    )

    assert list(available) == ["MAND"]
    assert missing == ["MAX"]


def test_resolve_models_ignores_model_without_checkpoint(tmp_path):
    """A half-copied bundle degrades to 'unavailable', it does not crash later."""
    plans = tmp_path / "bundle" / "MAND" / "D1" / "t__nnUNetPlans__3d_fullres"
    plans.mkdir(parents=True)
    _make_model_bundle(tmp_path / "bundle", ["MAX"])

    available, missing = AMASSSLogic.resolve_models(
        str(tmp_path / "bundle"), ("MAND", "MAX"), str(tmp_path / "scratch")
    )

    assert list(available) == ["MAX"]
    assert missing == ["MAND"]


def test_resolve_models_raises_when_nothing_found(tmp_path):
    (tmp_path / "bundle").mkdir()
    with pytest.raises(nnunet_runner.ModelNotFoundError):
        AMASSSLogic.resolve_models(str(tmp_path / "bundle"), ("MAND",), str(tmp_path / "scratch"))


def test_resolve_models_descends_into_single_wrapper_folder(tmp_path):
    """A zip of 'AMASSS_Models/' rather than of its contents still resolves."""
    _make_model_bundle(tmp_path / "bundle" / "AMASSS_Models", ["MAND"])

    available, _missing = AMASSSLogic.resolve_models(
        str(tmp_path / "bundle"), ("MAND",), str(tmp_path / "scratch")
    )

    assert list(available) == ["MAND"]


# ---------------------------------------------------------------------------
# format conversion
# ---------------------------------------------------------------------------

def test_nrrd_input_is_really_converted(tmp_path):
    """The original renamed NRRD to .nii.gz without converting it."""
    source = _write_scan(tmp_path / "scan.nrrd")
    destination = str(tmp_path / "p_000_0000.nii.gz")

    AMASSSLogic._convert_to_nifti(source, destination)

    reread = sitk.ReadImage(destination)
    assert reread.GetSize() == (8, 8, 8)
    # Really NIfTI, not an NRRD wearing a .nii.gz name.
    with open(destination, "rb") as handle:
        assert handle.read(2) == b"\x1f\x8b"  # gzip magic
    assert "NRRD" not in sitk.ReadImage(destination).GetMetaDataKeys()


# ---------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------

def test_segment_batch_merged_and_separate(tmp_path, stub_predictor):
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    _write_scan(tmp_path / "input" / "patient02.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND", "MAX"])

    run = AMASSSLogic.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        structures=("MAND", "MAX"),
        merge=("MERGED", "SEPARATE"),
        prediction_ID="Pred",
        scratch_dir=str(tmp_path / "scratch"),
    )

    assert run.report["summary"] == {"total": 2, "processed": 2, "failed": 0}
    names = sorted(os.path.basename(path) for path in run.segmentation_files)
    assert names == [
        "patient01_Pred_MAND.nii.gz",
        "patient01_Pred_MAX.nii.gz",
        "patient01_Pred_MERGED.nii.gz",
        "patient02_Pred_MAND.nii.gz",
        "patient02_Pred_MAX.nii.gz",
        "patient02_Pred_MERGED.nii.gz",
    ]
    assert os.path.isfile(os.path.join(run.output_dir, "AMASSS_report.json"))


def test_merged_volume_uses_the_documented_labels(tmp_path, stub_predictor):
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND", "CB"])

    run = AMASSSLogic.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        structures=("MAND", "CB"),
        merge=("MERGED",),
        scratch_dir=str(tmp_path / "scratch"),
    )

    merged = next(p for p in run.segmentation_files if p.endswith("_MERGED.nii.gz"))
    values = set(np.unique(sitk.GetArrayFromImage(sitk.ReadImage(merged))).tolist())
    # The stub gives both structures the same voxels, so MAND (painted last
    # per MERGING_ORDER) wins -- which is exactly the documented behavior.
    assert values == {0, AMASSSLogic.LABELS["MAND"]}


def test_single_structure_is_written_separately_even_in_merged_mode(tmp_path, stub_predictor):
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND"])

    run = AMASSSLogic.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        structures=("MAND",),
        merge=("MERGED",),
        scratch_dir=str(tmp_path / "scratch"),
    )

    assert [os.path.basename(p) for p in run.segmentation_files] == ["patient01_Pred_MAND.nii.gz"]


def test_report_lists_structures_without_a_model(tmp_path, stub_predictor):
    """Regression: a structure with no model used to vanish silently."""
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND"])

    run = AMASSSLogic.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        structures=("MAND", "MAX", "SKIN"),
        scratch_dir=str(tmp_path / "scratch"),
    )

    assert run.report["structures_without_model"] == ["MAX", "SKIN"]
    assert run.report["predicted_structures"] == ["MAND"]


def test_segment_rejects_unknown_structure(tmp_path):
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    with pytest.raises(ValueError, match="Unknown structure"):
        AMASSSLogic.segment(
            input_path=str(tmp_path / "input"),
            model_path=str(tmp_path),
            structures=("RC",),
            scratch_dir=str(tmp_path / "scratch"),
        )


def test_main_returns_a_zip_of_the_outputs(tmp_path, stub_predictor):
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND", "MAX"])

    zip_path = AMASSSLogic.main(
        input=str(tmp_path / "input"),
        model=bundle,
        structures=("MAND", "MAX"),
        merge=("MERGED",),
    )

    assert zipfile.is_zipfile(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert "AMASSS_report.json" in names
        assert any(name.endswith("patient01_Pred_MERGED.nii.gz") for name in names)
        report = json.loads(zf.read("AMASSS_report.json"))
    assert report["summary"]["processed"] == 1


# ---------------------------------------------------------------------------
# schema: the grouped selection argument
# ---------------------------------------------------------------------------

class _SelectionProbe(Tool):
    name = "_selection_probe"
    arguments = {
        "structures": ArgSpec(
            type=SELECTION_TYPE,
            required=True,
            multiple=True,
            choices=AMASSSLogic.STRUCTURE_CODES,
            choice_groups=AMASSSLogic.STRUCTURE_GROUPS,
        ),
    }

    def run(self, structures):
        return structures


@pytest.mark.parametrize(
    "sent",
    [
        "MAND,MAX",
        "MAND MAX",
        '["MAND", "MAX"]',
        '{"MAND": true, "MAX": true, "CB": false}',
        # The shape a grouped-checkbox UI produces naturally: display names,
        # exactly as the server published them in choice_groups.
        '{"Mandible": true, "Maxilla": true, "Cranial base": false}',
        '{"Mandible": "true", "Maxilla": "1", "Skin": "0"}',
    ],
)
def test_selection_accepts_every_wire_shape(sent):
    assert _SelectionProbe().validate({"structures": sent}) == {"structures": ["MAND", "MAX"]}


def test_selection_result_follows_declared_order():
    """Deterministic downstream behavior, whatever order the client sent."""
    cleaned = _SelectionProbe().validate({"structures": "SKIN,MAND,CV"})
    assert cleaned["structures"] == ["MAND", "CV", "SKIN"]


def test_selection_rejects_unknown_value():
    with pytest.raises(ToolArgumentError, match="MANDD"):
        _SelectionProbe().validate({"structures": "MANDD"})


def test_selection_rejects_empty_required_selection():
    with pytest.raises(ToolArgumentError, match="at least one"):
        _SelectionProbe().validate({"structures": '{"Mandible": false}'})


def test_amasss_schema_publishes_groups_and_defaults():
    """The client builds its checkbox groups from this and nothing else."""
    from tools.AMASSS.AMASSS import AMASSSTool

    spec = AMASSSTool.arguments["structures"]
    assert spec.choice_groups["Bones"]["Mandible"] == "MAND"
    assert set(spec.default) <= set(spec.choices)
    # Structures with no shipped model must not be offered at all.
    assert not {"RC", "TEETH", "MCAN"} & set(spec.choices)
