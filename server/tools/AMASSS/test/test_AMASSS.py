"""Unit tests for the AMASSS tool logic.

No GPU and no real nnUNet models are needed: `nnunet_runner.predict_folder`
is monkeypatched with a stub that writes synthetic masks, so everything
around the inference itself -- input discovery, output filtering, model
resolution, format conversion, label merging, file naming, the report -- is
exercised for real.

Run just this module:
    cd server && ./venv/bin/pytest tools/AMASSS/test/
"""

import glob
from functools import lru_cache
import json
import os
import zipfile

# Set before anything imports config.Settings() (file_utils does, via the
# tool logic), so the suite runs regardless of the local environment.
os.environ.setdefault("API_TOKEN", "test-token")

import numpy as np
import pytest
import SimpleITK as sitk

from base import Selection, ToolArgumentError
from config import settings
from tools.AMASSS.AMASSS import AMASSSTool
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


def test_main_returns_the_output_directory(tmp_path, stub_predictor, monkeypatch):
    """output_kind = "files": run() hands back a folder and main.py zips it."""
    from config import settings

    monkeypatch.setattr(settings, "TEMP_DIR", str(tmp_path / "server_tmp"))
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND", "MAX"])

    output_dir = AMASSSLogic.main(
        input=str(tmp_path / "input"),
        model=bundle,
        structures=("MAND", "MAX"),
        merge=("MERGED",),
    )

    assert os.path.isdir(output_dir)
    # main.py names the streamed archive after this folder, so it must carry
    # the run's identity rather than a generic "output".
    assert os.path.basename(output_dir) == "AMASSS_Pred"

    produced = os.listdir(output_dir)
    assert "AMASSS_report.json" in produced
    assert os.path.isfile(
        os.path.join(output_dir, "patient01_Pred_SegOut", "patient01_Pred_MERGED.nii.gz")
    )
    with open(os.path.join(output_dir, "AMASSS_report.json")) as handle:
        assert json.load(handle)["summary"]["processed"] == 1

    # The bulky nnUNet intermediates are dropped before the response streams.
    scratch_dir = os.path.dirname(output_dir)
    assert not os.path.exists(os.path.join(scratch_dir, "nnunet_input"))
    assert not glob.glob(os.path.join(scratch_dir, "pred_*"))


# ---------------------------------------------------------------------------
# schema: the multichoice arguments
# ---------------------------------------------------------------------------

def test_schema_is_valid():
    """registry.py runs this at startup; a malformed declaration must not be
    discovered by the first request that happens to hit the tool."""
    AMASSSTool().check_schema()


def test_structure_choices_are_display_names_with_declared_defaults():
    spec = AMASSSTool.arguments["structures"]
    assert spec.choices["Mandible"] is True
    assert spec.choices["Skin"] is False
    # Structures with no shipped model must not be offered at all.
    assert not {"Teeth", "Root canal", "Mandibular canal"} & set(spec.choices)
    # Declaration order is what the client renders, and it follows the
    # anatomical grouping (Bones, Soft tissue, Masks).
    assert list(spec.choices)[:2] == ["Mandible", "Maxilla"]
    assert list(spec.choices)[-1] == "Maxilla (Mask)"


@pytest.mark.parametrize(
    "sent, expected",
    [
        ('{"Mandible": true, "Maxilla": true}', ("MAND", "MAX")),
        ("Mandible,Maxilla", ("MAND", "MAX")),
        # Whatever arrives is the complete selection: an option left out is
        # off, whatever its declared default.
        ('{"Skin": true}', ("SKIN",)),
    ],
)
def test_structures_reach_run_as_codes(sent, expected):
    cleaned = AMASSSTool().validate(
        {"input": "/fake/scan.nii.gz", "model": "bundle", "structures": sent}
    )
    assert AMASSSLogic.structure_codes(cleaned["structures"]) == expected


def test_validate_rejects_an_unknown_structure():
    with pytest.raises(ToolArgumentError, match="Mandibule"):
        AMASSSTool().validate(
            {"input": "/fake/scan.nii.gz", "model": "bundle", "structures": "Mandibule"}
        )


def test_selection_maps_to_codes_in_declaration_order():
    """Deterministic downstream behavior, whatever order the client sent."""
    selection = Selection({"Skin": True, "Mandible": True, "Cervical vertebra": True})
    assert AMASSSLogic.structure_codes(selection) == ("SKIN", "MAND", "CV")


def test_omitted_optional_merge_falls_back_to_the_default():
    """`merge` is optional; None must not silently disable merging."""
    assert AMASSSLogic.merge_modes(None) == AMASSSLogic.DEFAULT_MERGE_MODES
    assert AMASSSLogic.merge_modes(Selection(AMASSSLogic.MERGE_CHOICES)) == ("MERGED",)


def test_segment_rejects_an_empty_selection(tmp_path):
    """Every box unchecked is a valid Selection, so the schema cannot catch it."""
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    with pytest.raises(ToolArgumentError, match="at least one structure"):
        AMASSSLogic.segment(
            input_path=str(tmp_path / "input"),
            model_path=str(tmp_path),
            structures=(),
            scratch_dir=str(tmp_path / "scratch"),
        )


# ---------------------------------------------------------------------------
# GPU resampling (see settings.AMASSS_GPU_RESAMPLING)
# ---------------------------------------------------------------------------

class _FakeConfigurationManager:
    """Stands in for nnUNet's, reproducing the one detail that can bite.

    The real `resampling_fn_data` / `resampling_fn_probabilities` are
    `@property @lru_cache`, so a value read before the swap outlives it unless
    the cache is cleared. Declaring them the same way here means a version of
    `_enable_gpu_resampling` that forgot `cache_clear()` fails the test below
    instead of passing it.
    """

    def __init__(self, configuration):
        self.configuration = configuration

    @property
    @lru_cache(maxsize=1)  # noqa: B019 - deliberately mirrors nnUNet's own shape
    def resampling_fn_data(self):
        return self.configuration["resampling_fn_data"]

    @property
    @lru_cache(maxsize=1)  # noqa: B019
    def resampling_fn_probabilities(self):
        return self.configuration["resampling_fn_probabilities"]


def _fake_predictor(**overrides):
    configuration = {
        "resampling_fn_data": "resample_data_or_seg_to_shape",
        "resampling_fn_data_kwargs": {"is_seg": False, "order": 3},
        "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
        "resampling_fn_probabilities_kwargs": {"is_seg": False, "order": 1},
    }
    configuration.update(overrides)

    class _Predictor:
        pass

    predictor = _Predictor()
    predictor.configuration_manager = _FakeConfigurationManager(configuration)
    return predictor


def test_gpu_resampling_is_skipped_on_cpu():
    """No CUDA, nothing to move: the scipy resamplers must stay untouched."""
    predictor = _fake_predictor()
    assert nnunet_runner._enable_gpu_resampling(predictor, "cpu") is False
    assert (
        predictor.configuration_manager.configuration["resampling_fn_data"]
        == "resample_data_or_seg_to_shape"
    )


def test_gpu_resampling_leaves_a_non_default_resampler_alone():
    """A bundle pinning its own resampler configured its geometry on purpose."""
    predictor = _fake_predictor(resampling_fn_data="no_resampling")
    assert nnunet_runner._enable_gpu_resampling(predictor, "cuda") is False
    assert predictor.configuration_manager.configuration["resampling_fn_data"] == "no_resampling"


def test_gpu_resampling_redirects_both_ends_and_drops_the_memoized_value():
    """Both resamplers move to the GPU, and the cached property does not survive."""
    pytest.importorskip("torch")
    predictor = _fake_predictor()
    manager = predictor.configuration_manager

    # Read one through the cache first: this is what the swap has to invalidate.
    assert manager.resampling_fn_data == "resample_data_or_seg_to_shape"

    assert nnunet_runner._enable_gpu_resampling(predictor, "cuda") is True

    for key in ("resampling_fn_data", "resampling_fn_probabilities"):
        assert manager.configuration[key] == "resample_torch_fornnunet"
        kwargs = manager.configuration[f"{key}_kwargs"]
        assert kwargs["mode"] == "linear"
        assert str(kwargs["device"]).startswith("cuda")
        # The scipy-only spline order must be gone, not merely overridden.
        assert "order" not in kwargs

    assert manager.resampling_fn_data == "resample_torch_fornnunet"


def test_report_records_the_settings_that_move_the_masks(tmp_path, stub_predictor):
    """GPU resampling and the step size both change the output, so a mask is
    only reproducible next to the values that produced it."""
    _write_scan(tmp_path / "input" / "patient01.nii.gz")
    bundle = _make_model_bundle(tmp_path / "bundle", ["MAND"])

    run = AMASSSLogic.segment(
        input_path=str(tmp_path / "input"),
        model_path=bundle,
        structures=("MAND",),
        scratch_dir=str(tmp_path / "scratch"),
    )

    # resolve_device is stubbed to "cpu", where the GPU path cannot apply.
    assert run.report["gpu_resampling"] is False
    assert run.report["tile_step_size"] == float(settings.AMASSS_TILE_STEP_SIZE)


def test_converted_input_keeps_its_voxel_type(tmp_path):
    """Casting to float32 doubled the bytes gzipped per scan and bought nothing."""
    source = _write_scan(tmp_path / "scan.nrrd")
    destination = str(tmp_path / "p_000_0000.nii.gz")

    AMASSSLogic._convert_to_nifti(source, destination)

    assert sitk.ReadImage(destination).GetPixelID() == sitk.ReadImage(source).GetPixelID()
