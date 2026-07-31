"""Unit tests for the ASO tool logic.

No GPU, no real models and no HTTP: the landmark seam (`ali_client`) is
monkeypatched, and everything else -- discovery, pairing, recentring,
registration, output naming, the report -- runs for real against synthetic
volumes and meshes.

Every test docstring names the defect of the original CLIs it prevents coming
back.

Run just this module:
    cd server && python -m pytest tools/ASO/test/
"""

import json
import os

# Set before anything imports config.Settings() (file_utils does, via the tool
# logic), so the suite runs regardless of the local environment.
os.environ.setdefault("API_TOKEN", "test-token")

import numpy as np
import pytest
import SimpleITK as sitk
import vtk

from base import Selection, ToolArgumentError
from config import settings
from tools.ASO.ASO import ASOTool
from tools.ASO.src import ASOLogic, ali_client, catalogs, geometry, markups
from tools.ASO.src.cbct import icp as cbct_icp
from tools.ASO.src.cbct import pipeline as cbct_pipeline
from tools.ASO.src.ios import icp as ios_icp
from tools.ASO.src.ios import pipeline as ios_pipeline
from tools.ASO.src.ios import surfaces


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# Seven points in a plausible skull-ish arrangement: no three are collinear, so
# the coarse alignment can always find a usable triplet.
_REFERENCE_POINTS = {
    "Ba": np.array([0.0, -30.0, -20.0]),
    "S": np.array([0.0, -10.0, 5.0]),
    "N": np.array([0.0, 25.0, 10.0]),
    "RPo": np.array([-35.0, -20.0, -5.0]),
    "LPo": np.array([35.0, -20.0, -5.0]),
    "ROr": np.array([-25.0, 20.0, 0.0]),
    "LOr": np.array([25.0, 20.0, 0.0]),
}


def _rotate(points: dict, axis, degrees: float, offset=(0.0, 0.0, 0.0)) -> dict:
    matrix = geometry.rotation_matrix(np.array(axis, dtype=float), np.radians(degrees))
    return {
        name: matrix @ point + np.array(offset, dtype=float)
        for name, point in points.items()
    }


def _write_scan(path, size=(16, 16, 16), spacing=(1.0, 1.0, 1.0)):
    array = np.zeros(size[::-1], dtype=np.int16)
    array[4:12, 4:12, 4:12] = 800
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    sitk.WriteImage(image, str(path), useCompression=True)
    return str(path)


def _write_markups(path, landmarks: dict):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    return markups.write_landmarks(landmarks, str(path))


def _write_mesh(path, centroids: dict, array_name="Universal_ID", points_per_tooth=6):
    """A mesh with `points_per_tooth` points clustered around each centroid,
    labelled with that tooth's universal id."""
    points = vtk.vtkPoints()
    labels = vtk.vtkIntArray()
    labels.SetName(array_name)
    vertices = vtk.vtkCellArray()

    offsets = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0]],
        dtype=float,
    )
    for tooth_id, centroid in centroids.items():
        # Symmetric offsets, so the cluster's mean IS the centroid.
        for offset in offsets[:points_per_tooth]:
            point_id = points.InsertNextPoint(*(np.asarray(centroid) + offset))
            labels.InsertNextValue(int(tooth_id))
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(point_id)

    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetVerts(vertices)
    mesh.GetPointData().AddArray(labels)
    mesh.GetPointData().SetActiveScalars(array_name)

    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    return surfaces.write_surface(mesh, str(path))


_UPPER_CENTROIDS = {
    3: np.array([-20.0, 0.0, 0.0]),    # UR6
    9: np.array([0.0, 25.0, 2.0]),     # UL1
    14: np.array([20.0, 0.0, 0.0]),    # UL6
}
_LOWER_CENTROIDS = {
    19: np.array([18.0, -1.0, -12.0]),   # LL6
    25: np.array([0.0, 24.0, -10.0]),    # LR1
    30: np.array([-18.0, -1.0, -12.0]),  # LR6
}


def _cbct_case(root, key="patient1", points=None, extension=".nii.gz"):
    """One CBCT patient plus its landmarks, rotated away from the reference."""
    points = _REFERENCE_POINTS if points is None else points
    moved = _rotate(points, (0.2, 1.0, 0.3), 17.0, offset=(12.0, -8.0, 5.0))
    scan = _write_scan(os.path.join(str(root), f"{key}_scan{extension}"))
    _write_markups(os.path.join(str(root), f"{key}_lm.mrk.json"), moved)
    return scan


def _cbct_reference(root):
    path = os.path.join(str(root), "gold", "reference_lm.mrk.json")
    _write_markups(path, _REFERENCE_POINTS)
    return os.path.join(str(root), "gold")


def _ios_reference(root, with_surfaces=True, with_markups=False):
    folder = os.path.join(str(root), "gold_ios")
    for jaw, centroids in (("U", _UPPER_CENTROIDS), ("L", _LOWER_CENTROIDS)):
        if with_surfaces:
            _write_mesh(os.path.join(folder, f"Gold_{jaw}_Seg.vtk"), centroids)
        if with_markups:
            _write_markups(
                os.path.join(folder, f"Gold_{jaw}_lm.mrk.json"),
                _ios_landmarks(centroids),
            )
    return folder


def _ios_landmarks(centroids: dict) -> dict:
    """One 'O' landmark per tooth, at its centroid."""
    return {f"{catalogs.TOOTH_NAMES[tooth]}O": point for tooth, point in centroids.items()}


def _ios_case(root, key="P1", array_name="Universal_ID", with_markups=False, jaws=("U", "L")):
    moved = {}
    for jaw, centroids in (("U", _UPPER_CENTROIDS), ("L", _LOWER_CENTROIDS)):
        if jaw not in jaws:
            continue
        rotated = {
            tooth: geometry.rotation_matrix(np.array([0.1, 0.2, 1.0]), np.radians(12.0))
            @ point
            + np.array([5.0, -3.0, 2.0])
            for tooth, point in centroids.items()
        }
        moved[jaw] = rotated
        _write_mesh(
            os.path.join(str(root), f"{key}_{jaw}_Seg.vtk"), rotated, array_name=array_name
        )
        if with_markups:
            _write_markups(
                os.path.join(str(root), f"{key}_{jaw}_lm.mrk.json"), _ios_landmarks(rotated)
            )
    return moved


@pytest.fixture(autouse=True)
def scratch_under_tmp(tmp_path, monkeypatch):
    """Keep every scratch directory the tool makes inside the test's tmp_path."""
    monkeypatch.setattr(settings, "TEMP_DIR", str(tmp_path / "server_tmp"))


def _report(output_dir: str) -> dict:
    with open(os.path.join(output_dir, ASOLogic.REPORT_NAME)) as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

def test_schema_is_valid():
    """Every tool must survive the check registry.py runs at startup."""
    ASOTool().check_schema()


def test_mode_specific_arguments_are_optional():
    """A required argument belonging to the inactive mode would block every
    request in the other one -- the schema cannot say 'only in mode X'."""
    optional = (
        "landmark_models", "cbct_landmarks", "ios_teeth", "ios_landmark_types",
        "ios_jaws", "ios_occlusion", "dicom_input", "output_suffix",
    )
    for name in optional:
        assert not ASOTool.arguments[name].required, name
    for name in ("modality", "automation", "input", "reference"):
        assert ASOTool.arguments[name].required, name


def test_input_declares_a_file_type_first():
    """GET /tools publishes types[0] as `type`, and the Slicer client keys its
    file picker off it: leading with "folder" makes the argument look
    non-file client-side."""
    assert ASOTool.arguments["input"].types[0] == "volume_or_zip_file"
    assert "folder" in ASOTool.arguments["input"].types
    assert ASOTool.arguments["reference"].types[0] == "zip_file"


def test_catalogs_have_no_duplicates_and_valid_defaults():
    """ALI shipped a UI saying 'UR3OI' against a CLI expecting 'UR3OIP'. One
    vocabulary, defined once, with defaults that are members of it."""
    assert len(catalogs.CBCT_LANDMARKS) == len(set(catalogs.CBCT_LANDMARKS))
    assert len(catalogs.TOOTH_IDS) == 32
    assert sorted(catalogs.TOOTH_IDS.values()) == list(range(1, 33))
    for default in catalogs.DEFAULT_CBCT_LANDMARKS:
        assert default in catalogs.CBCT_LANDMARK_CHOICES
    for default in catalogs.DEFAULT_TEETH:
        assert default in catalogs.TOOTH_CHOICES
    # Three teeth per jaw: what the coarse alignment needs, out of the box.
    per_jaw = catalogs.split_by_jaw(catalogs.DEFAULT_TEETH)
    assert len(per_jaw["Upper"]) == 3 and len(per_jaw["Lower"]) == 3


@pytest.mark.parametrize(
    "wire",
    ["Ba,S,N,RPo,LPo", {"Ba": True, "S": True, "N": True, "RPo": True, "LPo": True},
     '{"Ba": true, "S": true, "N": true, "RPo": true, "LPo": true}'],
)
def test_multichoice_wire_formats(wire):
    """A multichoice arrives as a comma list, a dict or JSON; whatever arrives
    is the COMPLETE selection, read back in DECLARATION order (not the order it
    was sent in, and never sorted)."""
    cleaned = ASOTool().validate(
        {
            "modality": "CBCT", "automation": "Semi-Automated",
            "input": "/tmp/x.nii.gz", "reference": "/tmp/ref.zip",
            "cbct_landmarks": wire,
        }
    )
    selected = cleaned["cbct_landmarks"].selected
    assert set(selected) == {"Ba", "S", "N", "RPo", "LPo"}
    assert list(selected) == [
        name for name in catalogs.CBCT_LANDMARK_CHOICES if name in set(selected)
    ]


def test_omitted_multichoice_falls_back_to_the_declared_defaults():
    """An absent optional choice argument gets `choices`'s own defaults -- the
    trap a `default` FIELD on ArgSpec silently turns into None."""
    cleaned = ASOTool().validate(
        {
            "modality": "CBCT", "automation": "Semi-Automated",
            "input": "/tmp/x.nii.gz", "reference": "/tmp/ref.zip",
        }
    )
    assert set(cleaned["cbct_landmarks"].selected) == set(catalogs.DEFAULT_CBCT_LANDMARKS)


def test_initial_matches_the_python_default():
    """A form always sends every widget, so a spin box starting at Qt's 0
    overrides run()'s default -- which is how AMASSS shipped unsmoothed
    surfaces."""
    assert ASOTool.arguments["output_suffix"].initial == "Or"
    assert ASOTool.arguments["dicom_input"].initial is False


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def test_angle_is_clamped_at_both_ends():
    """np.arccos of a dot product rounding to 1.0000000002 is NaN, which then
    propagates through the rotation matrix into a scan full of zeros. The
    original clamped the upper end only."""
    for vector in ([1.0, 2.0, 3.0], [3.0, 1.0, 2.0]):
        angle, _ = geometry.angle_and_axis(np.array(vector), np.array(vector))
        assert np.isfinite(angle)
    angle, _ = geometry.angle_and_axis(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))
    assert np.isfinite(angle)


def test_rotation_matrix_survives_a_degenerate_axis():
    """Two already-parallel vectors give a zero cross product; dividing by its
    norm produced a matrix of NaN."""
    matrix = geometry.rotation_matrix(np.zeros(3), 0.5)
    assert np.allclose(matrix, np.eye(3))


def test_triplet_search_is_deterministic_and_leaves_numpy_alone():
    """The search drew from the process-global numpy generator, so the same
    request gave a different orientation each run and two concurrent requests
    consumed each other's state."""
    source = _rotate(_REFERENCE_POINTS, (1.0, 0.4, 0.2), 15.0)
    np.random.seed(1234)
    before = np.random.get_state()[1][0]
    first = geometry.best_triplet(source, _REFERENCE_POINTS, False, 2500, 0)
    second = geometry.best_triplet(source, _REFERENCE_POINTS, False, 2500, 0)
    assert first == second
    assert np.random.get_state()[1][0] == before


# ---------------------------------------------------------------------------
# CBCT: discovery and pairing
# ---------------------------------------------------------------------------

def test_same_named_scans_in_different_folders_stay_apart(tmp_path):
    """`GetPatients` keyed on the base name, so two `scan.nii.gz` in different
    subfolders became one patient -- in the working dict AND in the flat output
    folder."""
    _cbct_case(tmp_path / "input" / "siteA", key="scan")
    _cbct_case(tmp_path / "input" / "siteB", key="scan")
    found = cbct_pipeline.discover(str(tmp_path / "input"))
    assert sorted(found) == [os.path.join("siteA", "scan"), os.path.join("siteB", "scan")]


def test_timepoints_are_two_patients(tmp_path):
    """`GetPatients` stripped `_T1`/`_T2`, collapsing two timepoints of one
    subject into a single patient: the second scan was dropped and both
    landmark sets were merged onto the survivor."""
    _cbct_case(tmp_path / "input", key="P1_T1")
    _cbct_case(tmp_path / "input", key="P1_T2")
    assert sorted(cbct_pipeline.discover(str(tmp_path / "input"))) == ["P1_T1", "P1_T2"]


def test_a_previous_run_is_not_re_ingested(tmp_path):
    """`patient1_Or.nii.gz` sorts BEFORE `patient1_scan.nii.gz`, so running
    twice on the same folder would orient the first run's output; and the first
    run's `patient1_lm_Or.mrk.json` would overwrite the caller's landmarks
    during the merge. AMASSS had the same class of bug."""
    root = tmp_path / "input"
    _cbct_case(root)
    _write_scan(root / "patient1_Or.nii.gz")
    _write_markups(root / "patient1_lm_Or.mrk.json", {"Ba": np.full(3, 99.0)})

    found = cbct_pipeline.discover(str(root), "Or")
    assert found["patient1"]["scan"].endswith("patient1_scan.nii.gz")
    assert [os.path.basename(path) for path in found["patient1"]["markups"]] == [
        "patient1_lm.mrk.json"
    ]


def test_a_folder_of_only_previous_outputs_still_works(tmp_path):
    """Skipping previous outputs outright would refuse a folder someone
    deliberately re-orients."""
    root = tmp_path / "input"
    _write_scan(root / "patient1_Or.nii.gz")
    _write_markups(root / "patient1_lm_Or.mrk.json", _REFERENCE_POINTS)
    found = cbct_pipeline.discover(str(root), "Or")
    assert found["patient1"]["scan"].endswith("patient1_Or.nii.gz")


def test_ios_previous_outputs_are_set_aside_too(tmp_path):
    root = tmp_path / "input"
    _ios_case(root, key="P1", jaws=("U",))
    _write_mesh(root / "P1_U_Seg_Or.vtk", _UPPER_CENTROIDS)
    found = ios_pipeline.discover(str(root), "Or")
    assert found["P1"]["Upper"]["surface"].endswith("P1_U_Seg.vtk")


def test_landmark_files_of_one_patient_are_merged_in_memory(tmp_path):
    """`MergeJson` merged a patient's per-group files by writing the result INTO
    the caller's input folder and deleting the sources."""
    root = tmp_path / "input"
    _write_scan(root / "P1_scan.nii.gz")
    _write_markups(root / "P1_lm_Pred_CB.mrk.json", {"Ba": np.zeros(3), "S": np.ones(3)})
    _write_markups(root / "P1_lm_Pred_U.mrk.json", {"N": np.full(3, 2.0)})

    found = cbct_pipeline.discover(str(root))
    merged = cbct_pipeline.load_landmarks(found["P1"]["markups"])
    assert sorted(merged) == ["Ba", "N", "S"]
    assert sorted(os.listdir(root)) == [
        "P1_lm_Pred_CB.mrk.json", "P1_lm_Pred_U.mrk.json", "P1_scan.nii.gz",
    ]


# ---------------------------------------------------------------------------
# CBCT: registration
# ---------------------------------------------------------------------------

def test_registration_needs_no_transform_file(tmp_path):
    """`SEMI_ASO_CBCT` read `data["tfm"]` unconditionally, but only the
    fully-automated chain ever produced one -- so every patient of a
    semi-automated run died on a KeyError caught 90 lines above."""
    source = _rotate(_REFERENCE_POINTS, (1.0, 0.3, 0.2), 12.0)
    registration = cbct_icp.register(
        source, _REFERENCE_POINTS, list(_REFERENCE_POINTS), pre_transform=None
    )
    assert registration.used == sorted(_REFERENCE_POINTS)
    assert registration.dropped == {}


def test_a_landmark_absent_from_the_reference_is_reported_not_fatal():
    """`GetDistDifference` indexed the reference's table with the INPUT's keys,
    so one extra landmark raised a KeyError that lost the whole patient."""
    source = dict(_rotate(_REFERENCE_POINTS, (1.0, 0.0, 0.0), 10.0))
    source["Extra"] = np.array([100.0, 100.0, 100.0])
    registration = cbct_icp.register(
        source, _REFERENCE_POINTS, list(source), pre_transform=None
    )
    assert registration.dropped["Extra"] == "not in the reference landmark file"
    assert "Ba" in registration.used


def test_too_few_landmarks_says_why():
    """`ICP()` returned (None, None, None) and the caller logged only 'ICP
    registration returned None'."""
    source = _rotate(_REFERENCE_POINTS, (1.0, 0.0, 0.0), 5.0)
    with pytest.raises(cbct_icp.RegistrationError) as raised:
        cbct_icp.register(source, _REFERENCE_POINTS, ["Ba", "S"], pre_transform=None)
    assert "2 usable landmark" in str(raised.value)


def test_registration_recovers_the_reference_frame():
    """The whole point: a rotated landmark set comes back onto the reference."""
    source = _rotate(_REFERENCE_POINTS, (0.3, 1.0, 0.2), 20.0)
    registration = cbct_icp.register(
        source, _REFERENCE_POINTS, list(_REFERENCE_POINTS), pre_transform=None
    )
    residual = geometry.mean_distance(registration.landmarks, _REFERENCE_POINTS)
    assert residual < 1.0, f"registration left {residual:.2f} mm of error"


def test_recentring_puts_the_volume_centre_on_the_origin(tmp_path):
    """All of what PRE_ASO_CBCT still does, once its learned orientation step
    was removed and its model_folder/SmallFOV/temp_folder arguments became
    dead."""
    path = _write_scan(tmp_path / "scan.nii.gz", size=(16, 16, 16), spacing=(0.5, 0.5, 0.5))
    image = sitk.ReadImage(path)
    original_center = np.array(
        image.TransformContinuousIndexToPhysicalPoint(
            (np.array(image.GetSize()) / 2.0).tolist()
        )
    )
    centered, translation = cbct_pipeline.recenter(image)
    new_center = np.array(
        centered.TransformContinuousIndexToPhysicalPoint(
            (np.array(centered.GetSize()) / 2.0).tolist()
        )
    )
    assert np.allclose(new_center, np.zeros(3), atol=1e-6)
    # The reported translation is what moved it there, which is what
    # center_landmarks applies to the points and what the .tfm composes back.
    assert np.allclose(np.array(translation.GetOffset()), -original_center, atol=1e-6)
    assert centered.GetSize() == image.GetSize()


def test_the_written_landmarks_overlay_the_written_volume(tmp_path):
    """The one thing that must be exact. The volume is moved by resampling and
    the landmarks by a matrix, through two different code paths -- if they
    disagree, the markups file opens in Slicer floating beside the scan it
    belongs to, and nothing in the report would say so."""
    path = _write_scan(tmp_path / "p1_scan.nii.gz", size=(24, 24, 24), spacing=(0.4, 0.4, 0.4))
    centered, pre = cbct_pipeline.prepare(path)
    source = cbct_pipeline.center_landmarks(
        _rotate(_REFERENCE_POINTS, (0.2, 1.0, 0.3), 17.0, offset=(12.0, -8.0, 5.0)), pre
    )
    registration = cbct_icp.register(source, _REFERENCE_POINTS, list(_REFERENCE_POINTS), pre)

    # resample_transform maps an OUTPUT point back to an INPUT point, so its
    # inverse is the forward move the voxels actually underwent.
    forward = registration.resample_transform.GetInverse()
    for name, point in source.items():
        assert np.allclose(
            forward.TransformPoint(point.tolist()), registration.landmarks[name], atol=1e-6
        ), name


def test_the_transform_file_maps_the_result_back_to_the_original(tmp_path):
    """The .tfm is what lets a clinician carry a measurement made on the
    oriented scan back onto the acquisition. Its direction is ORIENTED ->
    ORIGINAL, recentring included -- worth asserting rather than assuming, since
    getting it backwards is silent and the file still loads."""
    root = tmp_path / "input"
    original = _rotate(_REFERENCE_POINTS, (0.2, 1.0, 0.3), 17.0, offset=(12.0, -8.0, 5.0))
    _write_scan(root / "p1_scan.nii.gz", size=(24, 24, 24), spacing=(0.4, 0.4, 0.4))
    _write_markups(root / "p1_lm.mrk.json", original)

    run = ASOLogic.orient(
        input_path=str(root),
        reference_path=_cbct_reference(tmp_path),
        modality=catalogs.MODALITY_CBCT,
        automation=catalogs.AUTOMATION_SEMI,
        cbct_landmarks=list(_REFERENCE_POINTS),
        scratch_dir=str(tmp_path / "scratch"),
    )
    oriented = markups.load_landmarks(os.path.join(run.output_dir, "p1_lm_Or.mrk.json"))
    transform = sitk.ReadTransform(os.path.join(run.output_dir, "p1_Or_transform.tfm"))

    for name, point in oriented.items():
        assert np.allclose(
            transform.TransformPoint(point.tolist()), original[name], atol=1e-6
        ), name


def test_landmarks_follow_the_recentring(tmp_path):
    """Recentring the volume without moving the landmarks compares a centred
    scan against uncentred points -- the state the semi-automated CLI ran in,
    since it recentred nothing and then read a .tfm it never produced."""
    path = _write_scan(tmp_path / "scan.nii.gz")
    image = sitk.ReadImage(path)
    _, translation = cbct_pipeline.recenter(image)
    moved = cbct_pipeline.center_landmarks({"Ba": np.zeros(3)}, translation)
    assert np.allclose(moved["Ba"], np.array(translation.GetOffset()))


# ---------------------------------------------------------------------------
# CBCT: end to end
# ---------------------------------------------------------------------------

def test_semi_automated_cbct_end_to_end(tmp_path):
    """The mode that must work on day one, with no server-side model at all."""
    _cbct_case(tmp_path / "input")
    run = ASOLogic.orient(
        input_path=str(tmp_path / "input"),
        reference_path=_cbct_reference(tmp_path),
        modality=catalogs.MODALITY_CBCT,
        automation=catalogs.AUTOMATION_SEMI,
        cbct_landmarks=list(_REFERENCE_POINTS),
        scratch_dir=str(tmp_path / "scratch"),
    )
    entry = run.report["patients"]["patient1"]
    assert entry["status"] == "ok"
    assert set(entry["outputs"]) == {
        "patient1_Or.nii.gz", "patient1_lm_Or.mrk.json", "patient1_Or_transform.tfm",
    }
    assert run.report["summary"] == {"patients": 1, "oriented": 1, "failed": 0}
    for name in entry["outputs"]:
        assert os.path.getsize(os.path.join(run.output_dir, name)) > 0


def test_the_input_tree_is_untouched(tmp_path):
    """The original wrote a merged JSON into the input folder, deleted the
    sources, and converted DICOM into `<input>/NIFTI/` -- which a later run then
    re-discovered as input scans."""
    root = tmp_path / "input"
    _cbct_case(root)
    before = {
        name: os.path.getsize(os.path.join(root, name)) for name in sorted(os.listdir(root))
    }
    ASOLogic.orient(
        input_path=str(root),
        reference_path=_cbct_reference(tmp_path),
        modality=catalogs.MODALITY_CBCT,
        automation=catalogs.AUTOMATION_SEMI,
        cbct_landmarks=list(_REFERENCE_POINTS),
        scratch_dir=str(tmp_path / "scratch"),
    )
    after = {
        name: os.path.getsize(os.path.join(root, name)) for name in sorted(os.listdir(root))
    }
    assert after == before


@pytest.mark.parametrize(
    "given,expected", [(".nii", ".nii.gz"), (".nii.gz", ".nii.gz"), (".nrrd", ".nrrd")]
)
def test_output_keeps_the_input_format_and_is_compressed(tmp_path, given, expected):
    """A scan sent uncompressed must not come back as a 191 MB file per patient;
    and ITK has no ".nrrd.gz" writer at all, so NRRD keeps its own extension."""
    _cbct_case(tmp_path / "input", extension=given)
    run = ASOLogic.orient(
        input_path=str(tmp_path / "input"),
        reference_path=_cbct_reference(tmp_path),
        modality=catalogs.MODALITY_CBCT,
        automation=catalogs.AUTOMATION_SEMI,
        cbct_landmarks=list(_REFERENCE_POINTS),
        scratch_dir=str(tmp_path / "scratch"),
    )
    assert f"patient1_Or{expected}" in run.report["patients"]["patient1"]["outputs"]


def test_a_patient_without_landmarks_fails_alone(tmp_path):
    """One bad case must not take the batch down: the original wrote a
    `<name>Error.txt` into the OUTPUT folder and carried on silently."""
    root = tmp_path / "input"
    _cbct_case(root, key="good")
    _write_scan(root / "orphan_scan.nii.gz")
    run = ASOLogic.orient(
        input_path=str(root),
        reference_path=_cbct_reference(tmp_path),
        modality=catalogs.MODALITY_CBCT,
        automation=catalogs.AUTOMATION_SEMI,
        cbct_landmarks=list(_REFERENCE_POINTS),
        scratch_dir=str(tmp_path / "scratch"),
    )
    assert run.report["patients"]["good"]["status"] == "ok"
    assert run.report["patients"]["orphan"]["status"] == "failed"
    assert "no landmark file" in run.report["patients"]["orphan"]["reason"]
    assert run.report["summary"] == {"patients": 2, "oriented": 1, "failed": 1}


def test_the_output_tree_mirrors_the_input(tmp_path):
    """Two patients in different folders must land in different folders."""
    _cbct_case(tmp_path / "input" / "siteA", key="scan")
    _cbct_case(tmp_path / "input" / "siteB", key="scan")
    run = ASOLogic.orient(
        input_path=str(tmp_path / "input"),
        reference_path=_cbct_reference(tmp_path),
        modality=catalogs.MODALITY_CBCT,
        automation=catalogs.AUTOMATION_SEMI,
        cbct_landmarks=list(_REFERENCE_POINTS),
        scratch_dir=str(tmp_path / "scratch"),
    )
    assert os.path.isfile(os.path.join(run.output_dir, "siteA", "scan_Or.nii.gz"))
    assert os.path.isfile(os.path.join(run.output_dir, "siteB", "scan_Or.nii.gz"))


def test_a_reference_of_landmarks_alone_is_enough(tmp_path):
    """`ExtractFilesFromFolder(..., gold=True)` took `scan_files[0]`, so a
    reference bundle holding only the landmarks it registers against died on an
    IndexError -- while the scan it demanded was read and never used."""
    landmarks = cbct_pipeline.load_reference(_cbct_reference(tmp_path))
    assert sorted(landmarks) == sorted(_REFERENCE_POINTS)


def test_the_run_is_reproducible(tmp_path):
    """Same input, same seed, same transform. Not true while the triplet search
    drew from the global numpy generator."""
    transforms = []
    for index in (0, 1):
        _cbct_case(tmp_path / f"input{index}")
        run = ASOLogic.orient(
            input_path=str(tmp_path / f"input{index}"),
            reference_path=_cbct_reference(tmp_path),
            modality=catalogs.MODALITY_CBCT,
            automation=catalogs.AUTOMATION_SEMI,
            cbct_landmarks=list(_REFERENCE_POINTS),
            scratch_dir=str(tmp_path / f"scratch{index}"),
        )
        with open(os.path.join(run.output_dir, "patient1_lm_Or.mrk.json")) as handle:
            transforms.append(handle.read())
    assert transforms[0] == transforms[1]


# ---------------------------------------------------------------------------
# CBCT: fully automated
# ---------------------------------------------------------------------------

def test_fully_automated_cbct_says_the_landmark_tool_is_missing(tmp_path):
    """Until ALI is deployed this mode must fail with an explanation and a way
    forward, not a 404 from somewhere inside the server."""
    with pytest.raises(ToolArgumentError) as raised:
        ASOLogic.main(
            input=str(tmp_path), modality="CBCT", automation="Fully-Automated",
            reference=str(tmp_path), landmark_models="CBCT_landmark_models_ALI",
        )
    message = str(raised.value)
    assert settings.ASO_LANDMARK_TOOL in message
    assert "Semi-Automated" in message


def test_the_missing_tool_is_reported_before_the_missing_model(tmp_path):
    """With the landmark tool absent, naming a model cannot help -- so that is
    what the caller is told, whether or not they picked one. The empty dropdown
    is the common case, and "name a model" would send them looking for a bundle
    that changes nothing."""
    with pytest.raises(ToolArgumentError) as raised:
        ASOLogic.main(
            input=str(tmp_path), modality="CBCT", automation="Fully-Automated",
            reference=str(tmp_path),
        )
    assert settings.ASO_LANDMARK_TOOL in str(raised.value)


def test_fully_automated_cbct_needs_a_model_name(tmp_path, monkeypatch):
    """Once the tool IS there, the model is the next thing checked -- still
    before the input is extracted, since minutes of work would not supply it."""
    monkeypatch.setattr(ali_client, "is_available", lambda tool_name: True)
    with pytest.raises(ToolArgumentError) as raised:
        ASOLogic.main(
            input=str(tmp_path), modality="CBCT", automation="Fully-Automated",
            reference=str(tmp_path),
        )
    assert "landmark_models" in str(raised.value)


def test_fully_automated_cbct_runs_once_the_tool_is_there(tmp_path, monkeypatch):
    """The seam is wired end to end: with the landmark tool present, predictions
    replace the caller's markups and nothing else changes."""
    root = tmp_path / "input"
    _write_scan(root / "patient1_scan.nii.gz")
    predicted = _rotate(_REFERENCE_POINTS, (0.2, 1.0, 0.3), 14.0)

    monkeypatch.setattr(ali_client, "is_available", lambda tool_name: True)
    monkeypatch.setattr(
        ali_client,
        "predict_landmarks",
        lambda input_dir, tool_name, model_name, landmarks, work_dir: {
            "patient1": predicted
        },
    )
    output_dir = ASOLogic.main(
        input=str(root),
        modality="CBCT",
        automation="Fully-Automated",
        reference=_cbct_reference(tmp_path),
        landmark_models="CBCT_landmark_models_ALI",
        cbct_landmarks=Selection({name: True for name in _REFERENCE_POINTS}),
    )
    assert _report(output_dir)["patients"]["patient1"]["status"] == "ok"


def test_the_landmark_tool_is_never_imported_at_module_level():
    """registry.py imports every tool at startup; a module-level import of the
    registry would be a cycle, and one of torch would stop the server booting."""
    source = open(ali_client.__file__).read()
    header = source.split("def ", 1)[0]
    assert "from registry" not in header and "import torch" not in header


# ---------------------------------------------------------------------------
# CBCT: DICOM input
# ---------------------------------------------------------------------------

def _write_dicom_series(directory, size=(8, 8, 6)):
    """A minimal but genuinely readable CT series."""
    os.makedirs(str(directory), exist_ok=True)
    array = np.zeros(size[::-1], dtype=np.int16)
    array[2:5, 2:6, 2:6] = 500
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 0.5, 1.0))

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    series_uid = "1.2.826.0.1.3680043.2.1125.1234567890"
    for index in range(image.GetDepth()):
        slice_image = image[:, :, index]
        position = "\\".join(
            str(value) for value in image.TransformIndexToPhysicalPoint((0, 0, index))
        )
        for tag, value in (
            ("0008|0060", "CT"), ("0020|000e", series_uid), ("0020|0032", position),
            ("0020|0013", str(index)), ("0028|0030", "0.5\\0.5"), ("0018|0050", "1.0"),
            ("0020|0037", "1\\0\\0\\0\\1\\0"),
        ):
            slice_image.SetMetaData(tag, value)
        writer.SetFileName(os.path.join(str(directory), f"slice{index:03d}.dcm"))
        writer.Execute(slice_image)
    return str(directory)


def test_dicom_conversion_never_writes_into_the_input(tmp_path):
    """`convertdicom2nifti` wrote `<input>/NIFTI/` -- into the caller's own
    data, which a later run then re-discovered as input scans."""
    root = tmp_path / "input"
    _write_dicom_series(root / "patientA")
    _write_markups(root / "patientA_lm.mrk.json", _REFERENCE_POINTS)
    before = sorted(os.listdir(root))

    converted = dicom_module().convert_tree(str(root), str(tmp_path / "nifti"))
    assert sorted(os.listdir(root)) == before
    assert os.path.isfile(os.path.join(converted, "patientA.nii.gz"))
    # Markups travel with the scans, which is what lets Semi-Automated take DICOM.
    assert os.path.isfile(os.path.join(converted, "patientA_lm.mrk.json"))


def test_nested_dicom_exports_keep_their_tree(tmp_path):
    """The original listed one level of subfolders, so a nested export was
    invisible -- and two patients with the same folder name under different
    parents would have collided in its flat NIFTI/ output."""
    root = tmp_path / "input"
    _write_dicom_series(root / "siteA" / "scan")
    _write_dicom_series(root / "siteB" / "scan")
    converted = dicom_module().convert_tree(str(root), str(tmp_path / "nifti"))
    assert os.path.isfile(os.path.join(converted, "siteA", "scan.nii.gz"))
    assert os.path.isfile(os.path.join(converted, "siteB", "scan.nii.gz"))


def test_input_without_a_dicom_series_says_so(tmp_path):
    root = tmp_path / "input"
    _write_scan(root / "patient1.nii.gz")
    with pytest.raises(RuntimeError) as raised:
        dicom_module().convert_tree(str(root), str(tmp_path / "nifti"))
    assert "dicom_input" in str(raised.value)


def test_semi_automated_cbct_from_dicom_end_to_end(tmp_path):
    """The whole DICOM path, through main()'s own argument."""
    root = tmp_path / "input"
    _write_dicom_series(root / "patient1")
    _write_markups(
        root / "patient1_lm.mrk.json",
        _rotate(_REFERENCE_POINTS, (0.2, 1.0, 0.3), 15.0, offset=(6.0, -4.0, 3.0)),
    )
    output_dir = ASOLogic.main(
        input=str(root),
        modality="CBCT",
        automation="Semi-Automated",
        reference=_cbct_reference(tmp_path),
        cbct_landmarks=Selection({name: True for name in _REFERENCE_POINTS}),
        dicom_input=True,
    )
    assert _report(output_dir)["patients"]["patient1"]["status"] == "ok"
    assert os.path.isfile(os.path.join(output_dir, "patient1_Or.nii.gz"))


def dicom_module():
    """Imported through a function so the DICOM tests state the seam explicitly."""
    from tools.ASO.src.cbct import dicom

    return dicom


# ---------------------------------------------------------------------------
# IOS
# ---------------------------------------------------------------------------

def test_a_file_that_does_not_name_its_jaw_is_refused(tmp_path):
    """`UpperOrLower` DEFAULTED TO LOWER, so a maxillary scan named
    `patient1.vtk` was registered against the mandibular reference and returned
    as a success."""
    with pytest.raises(ios_pipeline.JawError):
        ios_pipeline.patient_and_jaw("patient1.vtk")
    assert ios_pipeline.patient_and_jaw("P1_U_Seg.vtk") == ("P1", "Upper")
    assert ios_pipeline.patient_and_jaw("P1_Lower.vtk") == ("P1", "Lower")


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("Upper_gold.vtk", ("gold", "Upper")),
        ("Lower_gold.vtk", ("gold", "Lower")),
        ("Upper_gold.json", ("gold", "Upper")),
    ],
)
def test_the_published_reference_bundle_is_readable(filename, expected):
    """The IOS reference published as HUTIN1/ASO v1.0.0 Gold_file.zip names the
    jaw FIRST -- `Upper_gold.vtk`. Requiring an identifier before the jaw token
    rejected the entire bundle with "no mesh whose name says which jaw it is",
    which would have made Fully-Automated IOS unusable with the only reference
    anyone ships. Verified against the real archive."""
    assert ios_pipeline.patient_and_jaw(filename) == expected


def test_a_gold_bundle_pairs_both_jaws_under_one_case(tmp_path):
    """Both files of the published bundle have to land on the same patient key,
    or load_reference would see two half-cases instead of one reference."""
    root = tmp_path / "gold"
    _write_mesh(root / "Upper_gold.vtk", _UPPER_CENTROIDS, array_name="PredictedID")
    _write_mesh(root / "Lower_gold.vtk", _LOWER_CENTROIDS, array_name="PredictedID")
    found = ios_pipeline.discover(str(root))
    assert list(found) == ["gold"]
    assert sorted(found["gold"]) == ["Lower", "Upper"]

    reference = ios_pipeline.load_reference(str(root), need_surfaces=True)
    assert reference["Upper"]["surface"].endswith("Upper_gold.vtk")
    assert reference["Lower"]["surface"].endswith("Lower_gold.vtk")


def test_patients_are_paired_by_exact_stem(tmp_path):
    """`Files_vtk_json.organise` paired with `vtk_name in json_name`, so patient
    `1` matched patient `10` -- and padded its list with a literal sentinel
    string to keep a loop non-empty."""
    root = tmp_path / "input"
    _ios_case(root, key="P1", with_markups=True, jaws=("U",))
    _ios_case(root, key="P10", with_markups=True, jaws=("U",))
    found = ios_pipeline.discover(str(root))
    assert sorted(found) == ["P1", "P10"]
    assert found["P1"]["Upper"]["markups"].endswith("P1_U_lm.mrk.json")


@pytest.mark.parametrize("array_name", ["Universal_ID", "PredictedID", "UniversalID"])
def test_every_accepted_label_array_name_works(tmp_path, array_name):
    """Slicer's tools have used three names over time and all three are in the
    wild."""
    _write_mesh(tmp_path / "P1_U_Seg.vtk", _UPPER_CENTROIDS, array_name=array_name)
    mesh = surfaces.read_surface(str(tmp_path / "P1_U_Seg.vtk"))
    assert surfaces.label_array_name(mesh) == array_name


def test_fully_automated_ios_end_to_end(tmp_path):
    """Tooth centroids in, both jaws oriented, one transform each."""
    _ios_case(tmp_path / "input")
    run = ASOLogic.orient(
        input_path=str(tmp_path / "input"),
        reference_path=_ios_reference(tmp_path),
        modality=catalogs.MODALITY_IOS,
        automation=catalogs.AUTOMATION_FULLY,
        ios_teeth=catalogs.DEFAULT_TEETH,
        ios_jaws=["Upper", "Lower"],
        scratch_dir=str(tmp_path / "scratch"),
    )
    entry = run.report["patients"]["P1"]
    assert entry["status"] == "ok"
    assert entry["jaws"]["Upper"]["status"] == "ok"
    assert entry["jaws"]["Lower"]["status"] == "ok"
    # Per jaw: the original wrote `<patient>_SegOr.tfm` for both, so the second
    # silently overwrote the first.
    assert "P1_Upper_Or.tfm" in entry["outputs"]
    assert "P1_Lower_Or.tfm" in entry["outputs"]


def test_semi_automated_ios_end_to_end(tmp_path):
    """Landmarks in, meshes AND landmarks come back oriented."""
    _ios_case(tmp_path / "input", with_markups=True)
    run = ASOLogic.orient(
        input_path=str(tmp_path / "input"),
        reference_path=_ios_reference(tmp_path, with_surfaces=False, with_markups=True),
        modality=catalogs.MODALITY_IOS,
        automation=catalogs.AUTOMATION_SEMI,
        ios_teeth=catalogs.DEFAULT_TEETH,
        ios_landmark_types=["O"],
        ios_jaws=["Upper", "Lower"],
        scratch_dir=str(tmp_path / "scratch"),
    )
    entry = run.report["patients"]["P1"]
    assert entry["status"] == "ok"
    assert "P1_U_Seg_Or.vtk" in entry["outputs"]
    assert "P1_U_lm_Or.mrk.json" in entry["outputs"]


def test_unsegmented_meshes_are_refused_with_the_array_names(tmp_path):
    """A 422 must say what is missing and where to go instead."""
    mesh = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    for offset in range(4):
        points.InsertNextPoint(offset, offset, 0)
    mesh.SetPoints(points)
    surfaces.write_surface(mesh, str(tmp_path / "input" / "P1_U_Seg.vtk"))

    with pytest.raises(ToolArgumentError) as raised:
        ASOLogic.orient(
            input_path=str(tmp_path / "input"),
            reference_path=_ios_reference(tmp_path),
            modality=catalogs.MODALITY_IOS,
            automation=catalogs.AUTOMATION_FULLY,
            ios_teeth=catalogs.DEFAULT_TEETH,
            ios_jaws=["Upper"],
            scratch_dir=str(tmp_path / "scratch"),
        )
    for name in markups.LABEL_ARRAY_NAMES:
        assert name in str(raised.value)


def test_a_mixed_batch_processes_what_it_can(tmp_path):
    """'None of your meshes is segmented' is a 422; 'one of your forty was bad'
    is a report entry. Returning nothing for the second is the failure the
    original made routine."""
    root = tmp_path / "input"
    _ios_case(root, key="good", jaws=("U",))
    bad = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    for offset in range(4):
        points.InsertNextPoint(offset, 0, 0)
    bad.SetPoints(points)
    surfaces.write_surface(bad, str(root / "bad_U_Seg.vtk"))

    run = ASOLogic.orient(
        input_path=str(root),
        reference_path=_ios_reference(tmp_path),
        modality=catalogs.MODALITY_IOS,
        automation=catalogs.AUTOMATION_FULLY,
        ios_teeth=catalogs.DEFAULT_TEETH,
        ios_jaws=["Upper"],
        scratch_dir=str(tmp_path / "scratch"),
    )
    assert run.report["patients"]["good"]["status"] == "ok"
    assert run.report["patients"]["bad"]["status"] == "failed"
    assert "tooth labels" in run.report["patients"]["bad"]["jaws"]["Upper"]["reason"]


def test_occlusion_moves_both_jaws_with_one_transform(tmp_path):
    """The point of the mode: two meshes in occlusion stay in occlusion."""
    _ios_case(tmp_path / "input")
    run = ASOLogic.orient(
        input_path=str(tmp_path / "input"),
        reference_path=_ios_reference(tmp_path),
        modality=catalogs.MODALITY_IOS,
        automation=catalogs.AUTOMATION_FULLY,
        ios_teeth=catalogs.DEFAULT_TEETH,
        ios_jaws=["Upper", "Lower"],
        ios_occlusion=catalogs.OCCLUSION_UPPER_DRIVES,
        scratch_dir=str(tmp_path / "scratch"),
    )
    entry = run.report["patients"]["P1"]
    assert entry["jaws"]["Lower"]["registered_on"] == "the Upper jaw's transform"
    upper = sitk.ReadTransform(os.path.join(run.output_dir, "P1_Upper_Or.tfm"))
    lower = sitk.ReadTransform(os.path.join(run.output_dir, "P1_Lower_Or.tfm"))
    assert np.allclose(upper.GetParameters(), lower.GetParameters())


def test_meshes_are_written_binary(tmp_path):
    """vtkPolyDataWriter defaults to ASCII: bigger, ~130x slower to parse, and
    LESS accurate (it prints about six significant digits)."""
    path = _write_mesh(tmp_path / "P1_U_Seg.vtk", _UPPER_CENTROIDS)
    with open(path, "rb") as handle:
        assert b"BINARY" in handle.read(256)


def test_a_missing_tooth_fails_that_jaw_with_its_name(tmp_path):
    """`SameNumberPoint` silently subsampled the longer point set with the
    global numpy generator and re-keyed both by index, so a patient missing one
    tooth was registered against a random correspondence."""
    mesh = surfaces.read_surface(
        _write_mesh(tmp_path / "P1_U_Seg.vtk", _UPPER_CENTROIDS)
    )
    with pytest.raises(ios_icp.RegistrationError) as raised:
        ios_icp.mean_teeth(mesh, (3, 9, 14, 16), "Universal_ID")
    assert "tooth 16" in str(raised.value)


# ---------------------------------------------------------------------------
# Cross-argument rules
# ---------------------------------------------------------------------------

def test_fewer_than_three_cbct_landmarks_is_refused(tmp_path):
    with pytest.raises(ToolArgumentError) as raised:
        ASOLogic.main(
            input=str(tmp_path), modality="CBCT", automation="Semi-Automated",
            reference=str(tmp_path), cbct_landmarks={"Ba": True, "S": True},
        )
    assert "at least 3 landmarks" in str(raised.value)


def test_fully_automated_ios_needs_three_or_four_teeth_per_jaw(tmp_path):
    with pytest.raises(ToolArgumentError) as raised:
        ASOLogic.main(
            input=str(tmp_path), modality="IOS", automation="Fully-Automated",
            reference=str(tmp_path),
            ios_teeth={"UR6": True, "UL6": True},
            ios_jaws={"Upper": True, "Lower": False},
        )
    assert "3 or 4 teeth" in str(raised.value)


def test_a_driving_jaw_must_be_selected(tmp_path):
    """The original carried an `occlusion` boolean and a `jaw` string that could
    contradict each other; one argument cannot."""
    with pytest.raises(ToolArgumentError) as raised:
        ASOLogic.main(
            input=str(tmp_path), modality="IOS", automation="Fully-Automated",
            reference=str(tmp_path),
            ios_jaws={"Upper": False, "Lower": True},
            ios_occlusion=catalogs.OCCLUSION_UPPER_DRIVES,
        )
    assert "not selected in 'ios_jaws'" in str(raised.value)


def test_output_suffix_cannot_be_a_path(tmp_path):
    """It becomes part of every output file name."""
    with pytest.raises(ToolArgumentError):
        ASOLogic.main(
            input=str(tmp_path), modality="CBCT", automation="Semi-Automated",
            reference=str(tmp_path), output_suffix="../escape",
        )


def test_a_selection_the_reference_cannot_support_is_refused_once(tmp_path):
    """The two published references carry DISJOINT landmark sets: Frankfurt
    Horizontal has Ba/S/N/RPo/LPo/ROr/LOr (the schema's defaults), Occlusal has
    ANS/IF/PNS/UL6O/UR1O/UR6O. Picking the second one and leaving the defaults
    alone would drop every landmark and fail all forty patients separately, for
    one wrong choice made in one place."""
    occlusal = os.path.join(str(tmp_path), "occlusal")
    _write_markups(
        os.path.join(occlusal, "UP01_Or.mrk.json"),
        {name: np.array([float(index), 1.0, 2.0]) for index, name in
         enumerate(("ANS", "IF", "PNS", "UL6O", "UR1O", "UR6O"))},
    )
    _cbct_case(tmp_path / "input")

    with pytest.raises(ToolArgumentError) as raised:
        ASOLogic.orient(
            input_path=str(tmp_path / "input"),
            reference_path=occlusal,
            modality=catalogs.MODALITY_CBCT,
            automation=catalogs.AUTOMATION_SEMI,
            cbct_landmarks=list(catalogs.DEFAULT_CBCT_LANDMARKS),
            scratch_dir=str(tmp_path / "scratch"),
        )
    message = str(raised.value)
    # It names what the reference actually offers, so the fix is one step away.
    for name in ("ANS", "IF", "PNS", "UL6O", "UR1O", "UR6O"):
        assert name in message


def test_the_matching_selection_runs_against_the_same_reference(tmp_path):
    """The other half of the rule: selecting the reference's own landmarks
    works, so the check cannot be blocking a legitimate request."""
    points = {
        "ANS": np.array([0.0, 30.0, 5.0]),
        "IF": np.array([0.0, 20.0, -5.0]),
        "PNS": np.array([0.0, -10.0, 0.0]),
        "UL6O": np.array([20.0, 5.0, -10.0]),
        "UR1O": np.array([0.0, 28.0, -8.0]),
        "UR6O": np.array([-20.0, 5.0, -10.0]),
    }
    occlusal = os.path.join(str(tmp_path), "occlusal")
    _write_markups(os.path.join(occlusal, "UP01_Or.mrk.json"), points)
    _cbct_case(tmp_path / "input", points=points)

    run = ASOLogic.orient(
        input_path=str(tmp_path / "input"),
        reference_path=occlusal,
        modality=catalogs.MODALITY_CBCT,
        automation=catalogs.AUTOMATION_SEMI,
        cbct_landmarks=list(points),
        scratch_dir=str(tmp_path / "scratch"),
    )
    assert run.report["patients"]["patient1"]["status"] == "ok"
    assert run.report["reference_landmarks"] == sorted(points)


def test_an_empty_input_says_what_was_expected(tmp_path):
    (tmp_path / "input").mkdir()
    with pytest.raises(ToolArgumentError) as raised:
        ASOLogic.orient(
            input_path=str(tmp_path / "input"),
            reference_path=_cbct_reference(tmp_path),
            modality=catalogs.MODALITY_CBCT,
            automation=catalogs.AUTOMATION_SEMI,
            scratch_dir=str(tmp_path / "scratch"),
        )
    assert ".nii.gz" in str(raised.value)


# ---------------------------------------------------------------------------
# main() / the archive
# ---------------------------------------------------------------------------

def test_main_returns_a_directory_with_the_report_and_no_intermediates(tmp_path):
    """main.py zips whatever comes back and names the archive after it, so the
    working copies must be gone by then."""
    _cbct_case(tmp_path / "input")
    output_dir = ASOLogic.main(
        input=str(tmp_path / "input"),
        modality="CBCT",
        automation="Semi-Automated",
        reference=_cbct_reference(tmp_path),
        cbct_landmarks=Selection({name: True for name in _REFERENCE_POINTS}),
        output_suffix="Or",
    )
    assert os.path.basename(output_dir) == "ASO_Or"
    assert os.path.isfile(os.path.join(output_dir, ASOLogic.REPORT_NAME))
    scratch = os.path.dirname(output_dir)
    for intermediate in ("input_extracted", "reference_extracted", "centered"):
        assert not os.path.exists(os.path.join(scratch, intermediate))


def test_a_zipped_input_is_extracted_by_the_tool(tmp_path):
    """The `input` argument leads with a FILE type, so a .zip reaches run() as
    an archive -- exactly like AMASSS's."""
    import zipfile

    _cbct_case(tmp_path / "cohort")
    archive = str(tmp_path / "cohort.zip")
    with zipfile.ZipFile(archive, "w") as zf:
        for name in os.listdir(tmp_path / "cohort"):
            zf.write(os.path.join(tmp_path / "cohort", name), os.path.join("cohort", name))

    run = ASOLogic.orient(
        input_path=archive,
        reference_path=_cbct_reference(tmp_path),
        modality=catalogs.MODALITY_CBCT,
        automation=catalogs.AUTOMATION_SEMI,
        cbct_landmarks=list(_REFERENCE_POINTS),
        scratch_dir=str(tmp_path / "scratch"),
    )
    assert run.report["patients"]["patient1"]["status"] == "ok"


def test_a_single_uploaded_file_does_not_drag_in_its_neighbours(tmp_path):
    """main.py streams every upload of a request into ONE work directory, so
    treating the file's parent as the input root would make the reference
    archive part of the input."""
    work_dir = tmp_path / "work"
    _write_scan(work_dir / "input.nii.gz")
    (work_dir / "reference.zip").write_bytes(b"not a real archive")

    root = ASOLogic._as_directory(
        str(work_dir / "input.nii.gz"), str(tmp_path / "scratch" / "input_extracted")
    )
    assert sorted(os.listdir(root)) == ["input.nii.gz"]
