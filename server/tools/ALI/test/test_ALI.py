"""Unit tests for the ALI tool.

No GPU, no real model weights and no network inference: the agent that walks
the volume is stubbed, so everything AROUND it -- mode detection, DICOM
recognition, input discovery, weight discovery, the landmark vocabulary,
output naming and tree preservation, the run report, and every cross-argument
rule -- is exercised for real.

The parts that genuinely cannot run here are skipped rather than faked: the
IOS engine needs pytorch3d, which has no PyPI distribution and must be
compiled into the deployment image. What does not need it (the model naming
rule, the tooth vocabulary, the network selection) is tested regardless, since
that is where the original's defects were.

Run just this module:
    cd server && python -m pytest tools/ALI/test/
"""

import json
import os
import zipfile

# Set before anything imports config.Settings(), so the suite runs regardless
# of the local environment.
os.environ.setdefault("API_TOKEN", "test-token")

import numpy as np
import pytest
import SimpleITK as sitk

from base import Selection, ToolArgumentError, ToolUnavailableError
from tools.ALI.ALI import ALITool
from tools.ALI.src import ALILogic
from tools.ALI.src import markups
from tools.ALI.src.ALI_CBCT import landmarks as cbct_catalog
from tools.ALI.src.ALI_IOS import landmarks as ios_catalog
from tools.ALI.src.ALI_IOS import engine as ios_engine


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def write_volume(path, size=(16, 16, 16)):
    """A small synthetic CBCT with a bright cube in it."""
    array = np.zeros(size[::-1], dtype=np.int16)
    array[4:12, 4:12, 4:12] = 800
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 0.5, 0.5))
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    sitk.WriteImage(image, str(path))
    return str(path)


def write_surface(path, labelled=True):
    """A minimal .vtk polydata, optionally carrying a tooth-label array."""
    vtk = pytest.importorskip("vtk")

    points = vtk.vtkPoints()
    for coordinates in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)):
        points.InsertNextPoint(*coordinates)

    polys = vtk.vtkCellArray()
    for triangle in ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)):
        polys.InsertNextCell(3)
        for point_id in triangle:
            polys.InsertCellPoint(point_id)

    surface = vtk.vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(polys)

    if labelled:
        labels = vtk.vtkIntArray()
        labels.SetName("Universal_ID")
        for value in (8, 8, 8, 8):
            labels.InsertNextValue(value)
        surface.GetPointData().AddArray(labels)

    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.Write()
    return str(path)


def write_cbct_bundle(root, landmarks_by_region):
    """A CBCT bundle in the <region>/<landmark>/<scale>/*.pth layout."""
    for region, labels in landmarks_by_region.items():
        for label in labels:
            for scale in cbct_catalog.SCALE_KEYS:
                folder = root / region / label / scale
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"{label}_Net_{scale}.pth").write_bytes(b"fake checkpoint")
    return str(root)


def write_ios_bundle(root, names):
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"fake checkpoint")
    return str(root)


@pytest.fixture
def stub_agent(monkeypatch):
    """Replace the deep-RL search with a deterministic voxel position.

    Every landmark whose name starts with "X" is reported as not found, so the
    report's two failure kinds can be told apart in a test.
    """
    from tools.ALI.src.ALI_CBCT import agent as agent_module
    from tools.ALI.src.ALI_CBCT import engine as cbct_engine

    class StubBrain:
        def __init__(self, scale_keys, device, out_channels=6):
            self.scale_keys = scale_keys

        def load(self, weights_per_scale):
            assert set(weights_per_scale) == set(self.scale_keys)

        def release(self):
            pass

    class StubAgent:
        def __init__(self, target, scale_keys, brain, environment, **_kwargs):
            self.target = target
            self.environment = environment

        def search(self, max_seconds):
            if self.target.startswith("X"):
                raise agent_module.NotFound("stubbed failure")
            return np.array([3.0, 4.0, 5.0])

    monkeypatch.setattr(cbct_engine, "Brain", StubBrain)
    monkeypatch.setattr(cbct_engine, "Agent", StubAgent)
    monkeypatch.setattr(cbct_engine, "resolve_device", lambda requested=None: "cpu")


# ---------------------------------------------------------------------------
# The landmark vocabulary
# ---------------------------------------------------------------------------

def test_impacted_canine_spellings_both_resolve():
    """The defect that lost a whole patient's landmarks.

    The Slicer UI called them UR3OI/UL3OI/UR3RI/UL3RI and the CLI's tables
    called them UR3OIP/.../UL3RIP. `LABEL_GROUPS[landmark]` was then indexed
    with no guard inside the save loop, so one unknown name raised a KeyError
    that was caught far above -- and NOTHING was written for that scan,
    including every landmark that had been found correctly.
    """
    for ui_name, canonical_name in (
        ("UR3OI", "UR3OIP"),
        ("UL3OI", "UL3OIP"),
        ("UR3RI", "UR3RIP"),
        ("UL3RI", "UL3RIP"),
    ):
        assert cbct_catalog.canonical(ui_name) == canonical_name
        assert cbct_catalog.group_of(ui_name) == "CI"
        assert cbct_catalog.group_of(canonical_name) == "CI"


def test_group_of_never_raises_on_an_unknown_name():
    assert cbct_catalog.group_of("NoSuchLandmark") == cbct_catalog.UNGROUPED


def test_scale_keys_match_the_shipped_weight_folders():
    assert cbct_catalog.SCALE_KEYS == ("1", "0-3")


def test_region_choices_are_all_on_by_default():
    assert cbct_catalog.REGION_CHOICES == {
        "Cranial base": True, "Upper": True, "Lower": True, "Impacted canine": True
    }


def test_region_codes_from_a_selection():
    selection = Selection(
        {"Cranial base": True, "Upper": False, "Lower": True, "Impacted canine": False}
    )
    assert cbct_catalog.region_codes(selection) == ("CB", "L")
    # An omitted optional argument must fall back to every region, not to none.
    assert cbct_catalog.region_codes(None) == cbct_catalog.REGION_CODES


def test_ios_offers_only_landmark_types_a_model_predicts():
    """R, RIP and OIP were selectable in the Slicer UI and predicted by
    nothing: no network produced them and no label table contained them.
    Ticking them did literally nothing."""
    offered = {lm_type for types in ios_catalog.NETWORKS.values() for lm_type in types}
    assert offered == {"O", "MB", "DB", "CL", "CB"}
    assert not offered & {"R", "RIP", "OIP"}


def test_ios_tooth_numbering_matches_the_shipped_label_tables():
    assert ios_catalog.UNIVERSAL_NUMBERS["Upper"]["UL7"] == 15
    assert ios_catalog.UNIVERSAL_NUMBERS["Upper"]["UR7"] == 2
    assert ios_catalog.UNIVERSAL_NUMBERS["Lower"]["LL7"] == 18
    assert ios_catalog.UNIVERSAL_NUMBERS["Lower"]["LR7"] == 31
    # Tooth 8 is UR1: the occlusal network's three channels, in channel order.
    assert ios_catalog.LABELS["O"]["8"] == ["UR1O", "UR1MB", "UR1DB"]
    assert ios_catalog.LABELS["C"]["8"] == ["UR1CL", "UR1CB"]


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

def test_a_single_volume_is_cbct(tmp_path):
    scan = write_volume(tmp_path / "in" / "patient01.nii.gz")
    detected = ALILogic.detect(scan, str(tmp_path / "scratch"))
    assert detected.mode == ALILogic.CBCT
    assert detected.scans == [(scan, "patient01.nii.gz")]


def test_a_single_surface_is_ios(tmp_path):
    mesh = write_surface(tmp_path / "in" / "arch.vtk")
    detected = ALILogic.detect(mesh, str(tmp_path / "scratch"))
    assert detected.mode == ALILogic.IOS


def test_a_zip_of_volumes_is_cbct_and_keeps_its_tree(tmp_path):
    """A .zip says nothing about which engine applies -- which is exactly why
    there is no `mode` argument for the caller to get wrong."""
    write_volume(tmp_path / "cohort" / "siteA" / "patient01.nii.gz")
    write_volume(tmp_path / "cohort" / "siteB" / "patient01.nii.gz")

    archive = tmp_path / "cohort.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for root, _dirs, files in os.walk(tmp_path / "cohort"):
            for name in files:
                path = os.path.join(root, name)
                zf.write(path, os.path.relpath(path, tmp_path / "cohort"))

    detected = ALILogic.detect(str(archive), str(tmp_path / "scratch"))
    assert detected.mode == ALILogic.CBCT
    # Two patients with the SAME base name in different folders. Keyed by
    # relative path, they stay distinct; the original keyed by file.name and
    # one silently replaced the other.
    assert sorted(key for _path, key in detected.scans) == [
        os.path.join("siteA", "patient01.nii.gz"),
        os.path.join("siteB", "patient01.nii.gz"),
    ]


def test_a_mixed_input_is_refused_rather_than_guessed(tmp_path):
    write_volume(tmp_path / "mixed" / "patient01.nii.gz")
    write_surface(tmp_path / "mixed" / "arch.vtk")

    with pytest.raises(ToolArgumentError) as raised:
        ALILogic.detect(str(tmp_path / "mixed"), str(tmp_path / "scratch"))
    assert "mixes" in str(raised.value)


def test_an_input_with_nothing_recognizable_says_what_it_wanted(tmp_path):
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "notes.txt").write_text("nothing here")

    with pytest.raises(ToolArgumentError) as raised:
        ALILogic.detect(str(tmp_path / "empty"), str(tmp_path / "scratch"))
    assert ".vtk" in str(raised.value) and ".nii.gz" in str(raised.value)


def test_detection_is_recursive(tmp_path):
    write_volume(tmp_path / "deep" / "a" / "b" / "c" / "patient.nii.gz")
    detected = ALILogic.detect(str(tmp_path / "deep"), str(tmp_path / "scratch"))
    assert len(detected.scans) == 1


def test_a_folder_of_dicom_is_not_mistaken_for_an_empty_input(tmp_path, monkeypatch):
    """DICOM slices carry no extension, so only GDCM can recognize them --
    which is why the client offers a folder picker and why detection probes."""
    from tools.ALI.src.ALI_CBCT import preprocess

    series = tmp_path / "cohort" / "patient01"
    series.mkdir(parents=True)
    for index in range(3):
        (series / f"IM{index:06d}").write_bytes(b"not really dicom")

    monkeypatch.setattr(preprocess, "is_dicom_series", lambda directory: True)
    monkeypatch.setattr(
        preprocess,
        "convert_dicom_series",
        lambda directory, destination: write_volume(destination),
    )

    detected = ALILogic.detect(str(tmp_path / "cohort"), str(tmp_path / "scratch"))
    assert detected.mode == ALILogic.CBCT
    assert detected.converted_dicom == 1
    # Converted into the SCRATCH dir, never into the user's own folder -- the
    # original wrote <input>/NIFTI/ and then re-ingested it on the next run.
    converted_path = detected.scans[0][0]
    assert str(tmp_path / "scratch") in converted_path
    assert not (tmp_path / "cohort" / "NIFTI").exists()


def test_a_folder_already_holding_volumes_is_not_probed_for_dicom(tmp_path, monkeypatch):
    from tools.ALI.src.ALI_CBCT import preprocess

    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    probed = []
    monkeypatch.setattr(
        preprocess, "is_dicom_series", lambda directory: probed.append(directory) or False
    )

    ALILogic.detect(str(tmp_path / "cohort"), str(tmp_path / "scratch"))
    assert probed == []


# ---------------------------------------------------------------------------
# CBCT model bundles
# ---------------------------------------------------------------------------

def test_weight_discovery_reads_the_folder_tree(tmp_path):
    from tools.ALI.src.ALI_CBCT import engine as cbct_engine

    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba", "S"]})
    weights = cbct_engine.discover_weights(bundle)

    assert sorted(weights) == ["Ba", "S"]
    assert sorted(weights["Ba"]) == sorted(cbct_catalog.SCALE_KEYS)


def test_a_landmark_missing_one_scale_is_not_offered(tmp_path):
    """Half a bundle must be reported up front, not fail mid-run: the agent
    walks the coarse scale and then the fine one, and needs both."""
    from tools.ALI.src.ALI_CBCT import engine as cbct_engine

    folder = tmp_path / "bundle" / "Ba" / "1"
    folder.mkdir(parents=True)
    (folder / "Ba_Net_1.pth").write_bytes(b"fake")

    assert cbct_engine.discover_weights(str(tmp_path / "bundle")) == {}


def test_requested_landmarks_separates_missing_from_unknown(tmp_path):
    from tools.ALI.src.ALI_CBCT import engine as cbct_engine

    weights = {"Ba": {}, "S": {}, "Mystery": {}}
    runnable, without_model, ungrouped = cbct_engine.requested_landmarks(weights, ["CB"])

    assert runnable == ("Ba", "S")
    # Every other cranial-base landmark the catalog knows about: the client
    # renders this as "not in the selected model bundle", i.e. "use another".
    assert "N" in without_model and "Ba" not in without_model
    # Weights for a landmark this catalog has never heard of are surfaced,
    # not silently ignored.
    assert ungrouped == ["Mystery"]


def test_a_bundle_using_the_ui_spelling_resolves(tmp_path):
    from tools.ALI.src.ALI_CBCT import engine as cbct_engine

    bundle = write_cbct_bundle(tmp_path / "bundle", {"Canine": ["UR3OI"]})
    weights = cbct_engine.discover_weights(bundle)

    assert list(weights) == ["UR3OIP"]
    runnable, _missing, ungrouped = cbct_engine.requested_landmarks(weights, ["CI"])
    assert runnable == ("UR3OIP",)
    assert ungrouped == []


# ---------------------------------------------------------------------------
# IOS model bundles
# ---------------------------------------------------------------------------

def test_ios_weight_discovery_reads_the_published_names(tmp_path):
    """The real ALIDDM bundle: Upper_O_model.pth, Lower_C_model.pth, ..."""
    bundle = write_ios_bundle(
        tmp_path / "bundle",
        ["Upper_O_model.pth", "Lower_O_model.pth", "Upper_C_model.pth", "Lower_C_model.pth"],
    )
    weights, unrecognized = ios_engine.discover_weights(bundle)

    assert weights["O"].keys() == {"Upper", "Lower"}
    assert weights["C"].keys() == {"Upper", "Lower"}
    assert unrecognized == []


def test_an_ios_checkpoint_with_no_jaw_token_is_reported_not_assumed_upper(tmp_path):
    """The original treated every file not containing "Lower" as upper-jaw
    weights, so a bundle missing its mandibular model quietly predicted the
    lower arch with the maxillary one."""
    bundle = write_ios_bundle(tmp_path / "bundle", ["model_O.pth", "Upper_C_model.pth"])
    weights, unrecognized = ios_engine.discover_weights(bundle)

    assert unrecognized == ["model_O.pth"]
    assert "O" not in weights
    assert weights["C"] == {"Upper": str(tmp_path / "bundle" / "Upper_C_model.pth")}


def test_ios_network_codes_from_a_selection():
    selection = Selection({"Occlusal": True, "Cervical": False})
    assert ios_catalog.network_codes(selection) == ("O",)
    assert ios_catalog.network_codes(None) == ios_catalog.NETWORK_CODES


# ---------------------------------------------------------------------------
# Model bundle auto-selection (the `model` argument is optional)
# ---------------------------------------------------------------------------

class FakeStore:
    """A data_store stub mapping name -> (path, is_temporary)."""

    def __init__(self, entries: dict):
        self.entries = entries

    def list_models(self, tool_name):
        assert tool_name == "ALI"
        return sorted(self.entries)

    def resolve_model(self, tool_name, name):
        path, is_temporary = self.entries[name]
        return ALILogic.ResolvedFile(path=str(path), is_temporary=is_temporary)


def test_the_bundle_is_picked_from_the_detected_mode(tmp_path, monkeypatch):
    """No `model` in the request: the bundle whose LAYOUT matches the mode runs."""
    cbct = write_cbct_bundle(tmp_path / "ALI_CBCT_Models", {"Cranial_Base": ["Ba"]})
    ios = write_ios_bundle(
        tmp_path / "ALI_IOS_Models", ["Upper_O_model.pth", "Lower_O_model.pth"]
    )
    monkeypatch.setattr(
        ALILogic,
        "data_store",
        FakeStore({"ALI_CBCT_Models": (cbct, False), "ALI_IOS_Models": (ios, False)}),
    )

    assert ALILogic.select_bundle(ALILogic.CBCT).path == cbct
    assert ALILogic.select_bundle(ALILogic.IOS).path == ios


def test_no_hosted_bundle_is_a_422_naming_the_setup_script(monkeypatch):
    monkeypatch.setattr(ALILogic, "data_store", FakeStore({}))
    with pytest.raises(ToolArgumentError, match="setup-models.sh"):
        ALILogic.select_bundle(ALILogic.IOS)


def test_a_mode_with_no_matching_bundle_is_a_422_naming_the_mode(tmp_path, monkeypatch):
    cbct = write_cbct_bundle(tmp_path / "ALI_CBCT_Models", {"Cranial_Base": ["Ba"]})
    monkeypatch.setattr(
        ALILogic, "data_store", FakeStore({"ALI_CBCT_Models": (cbct, False)})
    )
    with pytest.raises(ToolArgumentError, match="IOS"):
        ALILogic.select_bundle(ALILogic.IOS)


def test_two_matching_bundles_are_a_422_not_a_silent_pick(tmp_path, monkeypatch):
    """Which model vintage ran must never be a surprise."""
    first = write_ios_bundle(tmp_path / "ALIDDM_2023", ["Upper_O_model.pth"])
    second = write_ios_bundle(tmp_path / "ALIDDM_2026", ["Upper_O_model.pth"])
    monkeypatch.setattr(
        ALILogic,
        "data_store",
        FakeStore({"ALIDDM_2023": (first, False), "ALIDDM_2026": (second, False)}),
    )
    with pytest.raises(ToolArgumentError) as excinfo:
        ALILogic.select_bundle(ALILogic.IOS)
    assert "ALIDDM_2023" in str(excinfo.value) and "ALIDDM_2026" in str(excinfo.value)


def test_a_temporary_probe_copy_is_removed(tmp_path, monkeypatch):
    """A backend temp copy probed and not selected must not leak on disk."""
    cbct = write_cbct_bundle(tmp_path / "cbct", {"Cranial_Base": ["Ba"]})
    ios = write_ios_bundle(tmp_path / "ios", ["Upper_O_model.pth"])
    monkeypatch.setattr(
        ALILogic, "data_store", FakeStore({"cbct": (cbct, True), "ios": (ios, False)})
    )

    selected = ALILogic.select_bundle(ALILogic.IOS)

    assert selected.path == ios
    assert not os.path.exists(cbct)


def test_a_bundle_of_the_wrong_kind_is_an_argument_error(tmp_path, cbct_environment):
    """The mismatch that used to be a 500: naming the IOS bundle for a CBCT
    run (or vice versa) must answer 422, whose message Slicer shows verbatim
    -- a 500 buries it in the server log."""
    from tools.ALI.src.ALI_CBCT import engine as cbct_engine

    ios_bundle = write_ios_bundle(tmp_path / "ios", ["Upper_O_model.pth"])
    with pytest.raises(ToolArgumentError, match="No CBCT landmark weights"):
        cbct_engine.predict_landmarks(
            scans=[],
            model_path=ios_bundle,
            regions=None,
            prediction_ID="Pred",
            output_dir=str(tmp_path / "out"),
            scratch_dir=str(tmp_path / "scratch"),
        )


# ---------------------------------------------------------------------------
# The markups writer
# ---------------------------------------------------------------------------

def test_markups_are_written_as_a_slicer_file(tmp_path):
    path = markups.write(
        {"Ba": np.array([1.5, 2.5, 3.5]), "S": np.array([4.0, 5.0, 6.0])},
        str(tmp_path / "out" / f"scan_lm_Pred{markups.MARKUPS_EXTENSION}"),
    )
    assert path.endswith(".mrk.json")

    content = json.loads(open(path, encoding="utf-8").read())
    node = content["markups"][0]
    assert node["coordinateSystem"] == "LPS"
    assert [point["label"] for point in node["controlPoints"]] == ["Ba", "S"]
    # numpy scalars are cast to float on the way in: json.dump cannot
    # serialize them, and the failure would land after all the inference.
    assert node["controlPoints"][0]["position"] == [1.5, 2.5, 3.5]


def test_both_engines_write_the_same_extension():
    """CBCT wrote .mrk.json and IOS wrote .json for byte-identical content,
    and Slicer only associates the first with a markups node."""
    assert markups.MARKUPS_EXTENSION == ".mrk.json"


def test_the_display_node_is_visible(tmp_path):
    """Regression, and the nastiest kind: the file is perfectly valid, Slicer
    loads it, the node appears in the Markups module -- and nothing is drawn.

    Both original CLIs wrote `display.visibility: false`. That is independent
    of each control point's own `visibility`, so the points below are visible
    inside a node that is not displayed. Dropping a result file into Slicer
    showed an empty scene.
    """
    path = markups.write({"Ba": (1.0, 2.0, 3.0)}, str(tmp_path / "scan.mrk.json"))
    node = json.loads(open(path, encoding="utf-8").read())["markups"][0]

    assert node["display"]["visibility"] is True
    assert node["controlPoints"][0]["visibility"] is True
    # Off by design, and not the same kind of statement as the two above: a
    # point shows on the slice it is on, nothing more. Projecting 113 of them
    # onto neighbouring slices was tried and crowds the view used to judge
    # placement. It stays one checkbox away in the Markups module.
    assert node["display"]["sliceProjection"] is False


def test_positions_are_plain_json_floats(tmp_path):
    """numpy scalars are not JSON-serializable, and both engines produce
    coordinates as numpy arrays. The cast has to happen in the writer."""
    path = markups.write(
        {"Ba": np.array([1.5, 2.5, 3.5], dtype=np.float32)},
        str(tmp_path / "scan.mrk.json"),
    )
    position = json.loads(open(path, encoding="utf-8").read())[
        "markups"
    ][0]["controlPoints"][0]["position"]

    assert all(type(value) is float for value in position)


# ---------------------------------------------------------------------------
# End to end, agent stubbed
# ---------------------------------------------------------------------------

@pytest.fixture
def cbct_environment():
    pytest.importorskip("monai")
    pytest.importorskip("itk")
    pytest.importorskip("torch")


def test_a_cbct_run_writes_one_file_per_scan(tmp_path, stub_agent, cbct_environment):
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba", "S", "N"]})

    report = ALILogic.identify(
        input_path=str(tmp_path / "cohort"),
        model_path=bundle,
        cbct_regions=Selection(
            {"Cranial base": True, "Upper": False, "Lower": False, "Impacted canine": False}
        ),
        scratch_dir=str(tmp_path / "scratch"),
    )

    assert report["mode"] == "CBCT"
    assert report["summary"] == {"total": 1, "processed": 1, "failed": 0}

    produced = [
        os.path.join(root, name)
        for root, _dirs, files in os.walk(report["output_dir"])
        for name in files
        if name.endswith(".mrk.json")
    ]
    # ONE file holding every region, not one per anatomical group -- which is
    # what every downstream tool (ASO, AREG, AutoMatrix) had to recombine.
    assert len(produced) == 1
    assert os.path.basename(produced[0]) == "patient01_lm_Pred.mrk.json"

    content = json.loads(open(produced[0], encoding="utf-8").read())
    assert sorted(point["label"] for point in content["markups"][0]["controlPoints"]) == [
        "Ba", "N", "S"
    ]


def test_a_cbct_run_with_no_model_uses_the_matching_bundle(
    tmp_path, monkeypatch, stub_agent, cbct_environment
):
    """End to end without `model`: the data picks the engine, the engine's
    layout picks the bundle, and the report says which one ran."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    cbct = write_cbct_bundle(tmp_path / "ALI_CBCT_Models", {"Cranial_Base": ["Ba"]})
    ios = write_ios_bundle(tmp_path / "ALI_IOS_Models", ["Upper_O_model.pth"])
    monkeypatch.setattr(
        ALILogic,
        "data_store",
        FakeStore({"ALI_CBCT_Models": (cbct, False), "ALI_IOS_Models": (ios, False)}),
    )

    report = ALILogic.identify(
        input_path=str(tmp_path / "cohort"),
        cbct_regions=Selection(
            {"Cranial base": True, "Upper": False, "Lower": False, "Impacted canine": False}
        ),
        scratch_dir=str(tmp_path / "scratch"),
    )

    assert report["summary"]["processed"] == 1
    assert report["model_bundle"] == "ALI_CBCT_Models"


def test_a_batch_keeps_its_tree_so_homonyms_cannot_collide(
    tmp_path, stub_agent, cbct_environment
):
    write_volume(tmp_path / "cohort" / "siteA" / "patient01.nii.gz")
    write_volume(tmp_path / "cohort" / "siteB" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})

    report = ALILogic.identify(
        input_path=str(tmp_path / "cohort"),
        model_path=bundle,
        cbct_regions=Selection(
            {"Cranial base": True, "Upper": False, "Lower": False, "Impacted canine": False}
        ),
        scratch_dir=str(tmp_path / "scratch"),
    )

    produced = sorted(
        os.path.relpath(os.path.join(root, name), report["output_dir"])
        for root, _dirs, files in os.walk(report["output_dir"])
        for name in files
        if name.endswith(".mrk.json")
    )
    assert produced == [
        os.path.join("siteA", "patient01_lm_Pred.mrk.json"),
        os.path.join("siteB", "patient01_lm_Pred.mrk.json"),
    ]
    assert report["summary"]["processed"] == 2


def test_the_report_tells_a_missing_model_from_a_failed_search(
    tmp_path, stub_agent, cbct_environment
):
    """The two failures look identical in the Slicer scene -- a landmark that
    is simply not there -- and need opposite fixes: another bundle, or another
    scan. The report is the only place they are distinguishable."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    # "Xanadu" is not in the catalog but IS in the bundle, so the stub agent
    # runs it and fails it; "S" is in the catalog but not in the bundle.
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})
    for scale in cbct_catalog.SCALE_KEYS:
        folder = tmp_path / "bundle" / "Cranial_Base" / "XBa" / scale
        folder.mkdir(parents=True)
        (folder / "x.pth").write_bytes(b"fake")

    report = ALILogic.identify(
        input_path=str(tmp_path / "cohort"),
        model_path=bundle,
        cbct_regions=Selection(
            {"Cranial base": True, "Upper": False, "Lower": False, "Impacted canine": False}
        ),
        scratch_dir=str(tmp_path / "scratch"),
    )

    scan = report["scans"]["patient01.nii.gz"]
    assert scan["landmarks_found"] == ["Ba"]
    assert "S" in report["landmarks_without_model"]
    # XBa has weights, is unknown to the catalog, and is therefore never run.
    assert report["landmarks_ungrouped"] == ["XBa"]
    assert scan["landmarks_failed"] == {}


def test_a_landmark_that_never_converges_does_not_cost_the_others(
    tmp_path, stub_agent, cbct_environment
):
    """The unguarded `LABEL_GROUPS[landmark]` lookup meant one bad landmark
    lost the whole patient, including the ones already found."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})
    # The stub fails any landmark starting with "X". "XN" is aliased into the
    # catalog below so it is genuinely requested and genuinely runs.
    for scale in cbct_catalog.SCALE_KEYS:
        folder = tmp_path / "bundle" / "Cranial_Base" / "XN" / scale
        folder.mkdir(parents=True)
        (folder / "x.pth").write_bytes(b"fake")
    cbct_catalog.LABEL_GROUPS["XN"] = "CB"
    try:
        report = ALILogic.identify(
            input_path=str(tmp_path / "cohort"),
            model_path=bundle,
            cbct_regions=Selection(
                {"Cranial base": True, "Upper": False, "Lower": False, "Impacted canine": False}
            ),
            scratch_dir=str(tmp_path / "scratch"),
        )
    finally:
        del cbct_catalog.LABEL_GROUPS["XN"]

    scan = report["scans"]["patient01.nii.gz"]
    assert scan["status"] == "ok"
    assert scan["landmarks_found"] == ["Ba"]
    assert "XN" in scan["landmarks_failed"]
    assert scan["files"]


def test_the_run_report_lands_beside_the_results(tmp_path, stub_agent, cbct_environment):
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})

    report = ALILogic.identify(
        input_path=str(tmp_path / "cohort"),
        model_path=bundle,
        scratch_dir=str(tmp_path / "scratch"),
    )

    # The Slicer module reads exactly this file, by this name, from the
    # unpacked archive.
    on_disk = os.path.join(report["output_dir"], ALILogic.REPORT_NAME)
    assert os.path.isfile(on_disk)
    written = json.loads(open(on_disk, encoding="utf-8").read())
    assert written["mode"] == "CBCT"
    assert written["regions"]


def test_prediction_id_reaches_the_file_names(tmp_path, stub_agent, cbct_environment):
    """`SaveId` existed in the Slicer UI and was read by nothing; the suffix
    was hardcoded in both CLIs."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})

    report = ALILogic.identify(
        input_path=str(tmp_path / "cohort"),
        model_path=bundle,
        prediction_ID="T1",
        scratch_dir=str(tmp_path / "scratch"),
    )
    produced = [
        name
        for _root, _dirs, files in os.walk(report["output_dir"])
        for name in files
        if name.endswith(".mrk.json")
    ]
    assert produced == ["patient01_lm_T1.mrk.json"]


# ---------------------------------------------------------------------------
# A missing dependency belongs to the server, not to a scan
# ---------------------------------------------------------------------------

def test_a_missing_dependency_fails_before_any_scan_is_touched(tmp_path, monkeypatch):
    """Regression: `itk` absent from the deployment produced one identical
    failure PER SCAN -- each only after a full histogram correction -- and the
    run then ended on "produced no landmarks for any scan", which buried the
    one line saying what to install. It is a property of the server; it has to
    be raised once, before the loop."""
    from tools.ALI.src.ALI_CBCT import engine as cbct_engine
    from tools.ALI.src.ALI_CBCT import preprocess

    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    write_volume(tmp_path / "cohort" / "patient02.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})

    corrected = []
    monkeypatch.setattr(
        preprocess, "import_itk",
        lambda: (_ for _ in ()).throw(ToolUnavailableError("needs itk (missing: itk)")),
    )
    monkeypatch.setattr(
        preprocess, "correct_histogram",
        lambda *args, **kwargs: corrected.append(args) or args[1],
    )

    # ToolUnavailableError, not RuntimeError: main.py maps it to 501 with the
    # message, so the caller reads "install itk" instead of a blank 500.
    with pytest.raises(ToolUnavailableError) as raised:
        cbct_engine.predict_landmarks(
            scans=[(str(tmp_path / "cohort" / "patient01.nii.gz"), "patient01.nii.gz"),
                   (str(tmp_path / "cohort" / "patient02.nii.gz"), "patient02.nii.gz")],
            model_path=bundle,
            regions=("CB",),
            output_dir=str(tmp_path / "out"),
            scratch_dir=str(tmp_path / "scratch"),
        )

    # The install message itself, not a summary that hides it.
    assert "missing: itk" in str(raised.value)
    assert "produced no landmarks" not in str(raised.value)
    # And not one scan was preprocessed before finding out.
    assert corrected == []


# ---------------------------------------------------------------------------
# Cross-argument rules -- the 422s
# ---------------------------------------------------------------------------

def test_an_empty_cbct_selection_on_cbct_input_names_the_argument(tmp_path):
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")

    with pytest.raises(ToolArgumentError) as raised:
        ALILogic.identify(
            input_path=str(tmp_path / "cohort"),
            model_path=str(tmp_path),
            cbct_regions=Selection(
                {"Cranial base": False, "Upper": False, "Lower": False, "Impacted canine": False}
            ),
            scratch_dir=str(tmp_path / "scratch"),
        )
    message = str(raised.value)
    assert "CBCT" in message and "cbct_regions" in message
    # The 422 lists what to tick, so a mode mismatch explains itself.
    for name in cbct_catalog.REGION_NAMES:
        assert name in message


def test_an_empty_ios_selection_on_ios_input_names_the_argument(tmp_path):
    write_surface(tmp_path / "cohort" / "arch.vtk")

    with pytest.raises(ToolArgumentError) as raised:
        ALILogic.identify(
            input_path=str(tmp_path / "cohort"),
            model_path=str(tmp_path),
            ios_networks=Selection({"Occlusal": False, "Cervical": False}),
            scratch_dir=str(tmp_path / "scratch"),
        )
    message = str(raised.value)
    assert "ios_networks" in message and "Occlusal" in message


def test_the_inactive_modes_empty_selection_is_ignored(tmp_path, stub_agent, cbct_environment):
    """Both groups are always rendered by the client and one is always inert.
    Unticking the inert one must not fail the run."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})

    report = ALILogic.identify(
        input_path=str(tmp_path / "cohort"),
        model_path=bundle,
        ios_networks=Selection({"Occlusal": False, "Cervical": False}),
        scratch_dir=str(tmp_path / "scratch"),
    )
    assert report["mode"] == "CBCT"


# ---------------------------------------------------------------------------
# The schema itself -- the contract the Slicer client is written against
# ---------------------------------------------------------------------------

def test_the_schema_is_valid_and_has_no_mode_argument():
    tool = ALITool()
    tool.check_schema()
    assert sorted(tool.arguments) == [
        "cbct_regions", "input", "ios_networks", "model", "prediction_ID"
    ]
    assert "mode" not in tool.arguments


def test_input_accepts_both_kinds_and_the_model_is_name_only():
    tool = ALITool()
    assert tool.arguments["input"].types == ("volume_or_zip_file", "surface_or_zip_file")
    assert set(tool.arguments["input"].extensions) == {
        ".nii", ".nii.gz", ".nrrd", ".nrrd.gz", ".gipl", ".gipl.gz", ".vtk", ".stl", ".zip"
    }
    # ".zip" is declared by both types and must appear once, or the client's
    # file dialog repeats it in its filter string.
    assert len(tool.arguments["input"].extensions) == len(set(tool.arguments["input"].extensions))

    model = tool.arguments["model"]
    assert model.type is str and model.server_selectable == "model"
    assert not model.is_file


def test_both_selections_are_optional_so_neither_blocks_the_other_mode():
    tool = ALITool()
    assert not tool.arguments["cbct_regions"].required
    assert not tool.arguments["ios_networks"].required
    for name in ("cbct_regions", "ios_networks"):
        assert name.split("_")[0].upper() in tool.arguments[name].description


def test_the_output_is_a_bundle_of_files():
    assert ALITool().output_kind == "files"
