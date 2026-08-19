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
        ("Crown_Seg", "CrownSeg"),
        ("CrownSeg", "CrownSeg"),
        ("Batch_Dental_Seg", "BatchDentalSeg"),
        ("Surg_Mov_Pred", "SurgMovPred"),
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

    assert "CrownSeg" in printed
    assert "AMASSS" not in printed, "--tool was dropped and everything was listed"


def test_listing_without_a_filter_shows_everything(fetch_data, manifest, capsys):
    fetch_data._list_manifest(manifest)
    printed = capsys.readouterr().out

    for key in manifest:
        assert key in printed, key
