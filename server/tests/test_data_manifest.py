"""The download manifest, and the names a client is allowed to use for it.

`scripts/data-manifest.yml` keys on the DATA/ folder names, which stopped being
the tool names when ALI became ALI_CBCT/ALI_IOS and AREG became three. A client
reads its names from `GET /tools`, so the two vocabularies have to meet
somewhere; they meet in `fetch_data.resolve_tools`, and this is what pins it.

Found by running `fetch_data.py --tool Crown_Seg`: it printed the whole 30 GB
manifest. `--list` ignored `--tool` entirely, so asking about one tool answered
with every other one's bundles and nothing said the filter had been dropped.
"""

import importlib.util
import os
import sys

import pytest

_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts",
)


def _load_fetch_data():
    """Import scripts/fetch_data.py by path: scripts/ is not a package."""
    path = os.path.join(_SCRIPTS, "fetch_data.py")
    if not os.path.isfile(path):
        pytest.skip("scripts/fetch_data.py is not mounted here")
    spec = importlib.util.spec_from_file_location("fetch_data_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fetch_data():
    return _load_fetch_data()


@pytest.fixture(scope="module")
def manifest(fetch_data):
    path = os.path.join(_SCRIPTS, "data-manifest.yml")
    if not os.path.isfile(path):
        pytest.skip("scripts/data-manifest.yml is not mounted here")
    return fetch_data._parse_manifest(path)


# ---------------------------------------------------------------------------
# The names a client actually has


@pytest.mark.parametrize(
    "served, expected",
    [
        # Renames that only moved underscores: resolved by normalizing, the same
        # rule the registry uses when it refuses two spellings of one tool.
        # Harmonised 2026-09-10: a DATA folder is now named exactly as the tool
        # it belongs to, so a name resolves to itself. The run-together spellings
        # are kept as INPUTS -- an older deployment still has those folders, and
        # `data_slug` still falls back to them.
        ("Crown_Seg", "Crown_Seg"),
        ("CrownSeg", "Crown_Seg"),
        ("Batch_Dental_Seg", "Batch_Dental_Seg"),
        ("Surg_Mov_Pred", "Surg_Mov_Pred"),
        # Splits: one bundle now feeds several tools, which no naming rule can
        # derive. The manifest says so with `provides:`.
        ("ALI_CBCT", "ALI"),
        ("ALI_IOS", "ALI"),
        ("AREG_CBCT", "AREG"),
        ("AREG_IOS", "AREG"),
        ("AREG_IOSCBCT", "AREG"),
    ],
)

def test_a_served_tool_name_resolves_to_its_bundle(fetch_data, manifest, served, expected):
    assert fetch_data.resolve_tools(manifest, [served]) == [expected]


def test_every_provided_name_resolves_to_the_entry_that_declares_it(fetch_data, manifest):
    """`provides:` is only useful if it round-trips.

    A name listed under the wrong entry would download the wrong bundle and
    report success, which is the failure this whole seam exists to avoid.
    """
    for key, entry in manifest.items():
        for served in entry.get("provides") or ():
            assert fetch_data.resolve_tools(manifest, [served]) == [key], served


def test_an_unknown_tool_is_refused_rather_than_ignored(fetch_data, manifest):
    """Silently widening to everything is how a 12 GB bundle arrives for the
    wrong tool. The message has to name what IS available, including the served
    spellings, since those are the names the caller was reading."""
    with pytest.raises(fetch_data.ManifestError) as caught:
        fetch_data.resolve_tools(manifest, ["Nonexistent_Tool"])

    message = str(caught.value)
    assert "Nonexistent_Tool" in message
    assert "ALI_CBCT" in message, "the served spellings belong in the known list"


def test_no_argument_means_every_entry(fetch_data, manifest):
    assert fetch_data.resolve_tools(manifest, []) == sorted(manifest)
    assert fetch_data.resolve_tools(manifest, None) == sorted(manifest)


def test_asking_twice_for_one_bundle_downloads_it_once(fetch_data, manifest):
    """ALI_CBCT and ALI_IOS are one bundle. A caller selecting both in a panel
    must not fetch 12 GB twice."""
    assert fetch_data.resolve_tools(manifest, ["ALI_CBCT", "ALI_IOS"]) == ["ALI"]


# ---------------------------------------------------------------------------
# The listing


def test_listing_honours_the_filter(fetch_data, manifest, capsys):
    fetch_data._list_manifest(manifest, ["Crown_Seg"])
    printed = capsys.readouterr().out

    assert "Crown_Seg" in printed
    assert "AMASSS" not in printed, "--tool was dropped and everything was listed"


def test_listing_without_a_filter_shows_everything(fetch_data, manifest, capsys):
    fetch_data._list_manifest(manifest)
    printed = capsys.readouterr().out

    for key in manifest:
        assert key in printed, key


def test_aso_and_ali_landmark_weights_come_from_different_publications(manifest):
    """They are two trainings of the same landmarks, not two copies of one.

    An earlier note in the manifest said they were the same and suggested
    hard-linking one at the other to save disk. Measured 2026-09-10: every
    checkpoint differs, at both scales, for every landmark compared. ASO's
    published reference planes were built with ASO's weights, so swapping in
    ALI's would move the landmarks a little, move the registration with them,
    and produce oriented scans nobody could tell were wrong.

    Pinned by SOURCE rather than by bytes, because the bytes are not in the
    repository and this has to fail on a laptop with an empty DATA/.
    """
    def urls(tool, prefix):
        entries = manifest.get(tool, {}).get("models", [])
        return {e["url"] for e in entries if str(e.get("dest", "")).startswith(prefix)}

    aso = urls("ASO", "CBCT_landmark_models/")
    ali = urls("ALI", "ALI_CBCT_Models/")
    assert aso, "ASO declares no landmark weights"
    assert ali, "ALI declares no landmark weights"
    assert not (aso & ali), (
        "ASO and ALI now share a landmark weight archive. They are separate "
        "trainings; sharing one silently changes what ASO registers on."
    )


# ---------------------------------------------------------------------------
# A picker scope has to name a folder the manifest builds


def _deployment_scopes():
    """`(data folder, scope)` for every `testfile:<scope>` in deployment.toml."""
    from registry.deployment import tomllib

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "deployment.toml")
    with open(path, "rb") as handle:
        tools = tomllib.load(handle).get("tools", {})
    scopes = set()
    for name, entry in tools.items():
        for value in (entry.get("server_selectable") or {}).values():
            if isinstance(value, str) and value.startswith("testfile:"):
                scopes.add((entry.get("data_dir", name), value.split(":", 1)[1]))
    return sorted(scopes)


@pytest.mark.parametrize("data_dir, scope", _deployment_scopes())
def test_every_testfile_scope_is_a_folder_the_manifest_builds(manifest, data_dir, scope):
    """Found on a fresh deployment: AREG's pickers were scoped to `T1`/`T2`,
    folders built by hand on one machine and by no script, so every other
    server listed nothing. A scope is either the top folder of an entry's
    `dest`, or a view an entry's `split` publishes."""
    entries = manifest.get(data_dir, {}).get("testfiles", [])
    tops = {str(e.get("dest") or e["name"]).split("/")[0] for e in entries}
    views = {e["split"] for e in entries if e.get("split")}
    assert scope in tops or any(scope.startswith(f"{prefix}_") for prefix in views), (
        f"deployment.toml scopes a {data_dir} picker to '{scope}', "
        f"which no entry of the manifest stages"
    )


def _cohort(root, tool, dest):
    for timepoint in ("T1", "T2"):
        folder = os.path.join(root, tool, "testfiles", *dest.split("/"), timepoint)
        os.makedirs(os.path.join(folder, "nested"))
        with open(os.path.join(folder, f"scan_{timepoint}.nii.gz"), "w") as handle:
            handle.write(timepoint)


def test_split_publishes_each_timepoint_as_a_hardlinked_view(fetch_data, tmp_path):
    entry = {"tool": "AREG", "kind": "testfiles", "name": "FullyAuto.zip",
             "dest": "CBCT/CBCT_FullyAuto", "split": "CBCT", "extract": True}
    _cohort(str(tmp_path), "AREG", entry["dest"])

    # Already present, which is the deployment this repairs: nothing is
    # downloaded, and the views are built all the same.
    assert fetch_data._fetch(entry, str(tmp_path), False, None) == "skipped"

    for timepoint in ("T1", "T2"):
        view = tmp_path / "AREG" / "testfiles" / f"CBCT_{timepoint}" / "CBCT_FullyAuto"
        scan = view / f"scan_{timepoint}.nii.gz"
        assert scan.read_text() == timepoint
        assert (view / "nested").is_dir()
        assert os.stat(scan).st_nlink == 2, "a view must cost no data"
    assert not (tmp_path / "AREG" / "testfiles" / "CBCT_T1" / "CBCT_FullyAuto"
                / "scan_T2.nii.gz").exists()


def test_force_rebuilds_a_view_rather_than_keeping_what_it_held(fetch_data, tmp_path):
    entry = {"tool": "AREG", "kind": "testfiles", "name": "a.zip",
             "dest": "IOS/scans", "split": "IOS", "extract": True}
    _cohort(str(tmp_path), "AREG", entry["dest"])
    fetch_data._split(entry, str(tmp_path), False)
    stale = tmp_path / "AREG" / "testfiles" / "IOS_T1" / "scans" / "stale"
    stale.write_text("left over")

    fetch_data._split(entry, str(tmp_path), False)
    assert stale.exists(), "an existing view is kept, which is what makes a re-run cheap"
    fetch_data._split(entry, str(tmp_path), True)
    assert not stale.exists()


@pytest.mark.parametrize("split, extract", [("../out", True), ("a/b", True), ("CBCT", False)])
def test_a_split_that_could_escape_or_cannot_apply_is_refused(fetch_data, split, extract):
    manifest = {"AREG": {"testfiles": [
        {"name": "a.zip", "url": "https://example.invalid/a.zip",
         "split": split, "extract": extract},
    ]}}
    with pytest.raises(fetch_data.ManifestError):
        fetch_data._entries(manifest, "testfiles", ["AREG"])
