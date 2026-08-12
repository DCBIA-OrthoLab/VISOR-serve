"""The comparison that has to happen before an imported tool is deleted.

parity.py is meant for real tools on real data, where a run takes minutes. What
is tested here is the comparison itself, on the one tool that exists in both
forms in milliseconds: `_dispatch_probe`, packaged with its own `.schema.json`
and virtualenv, against an in-process twin written here.

Both directions, because a check that cannot fail proves nothing: agreement is
reported as agreement, and a twin that drifts -- in what it returns, or in the
bytes of the file it writes -- is caught.
"""

import os

import pytest

os.environ.setdefault("API_TOKEN", "test-token")

import file_utils
import parity
from base import ArgSpec, Tool
from config import settings

# What the probe reports about WHERE it ran, which differs between the two
# forms by construction: one is the server's own interpreter, the other is the
# tool's. A real ported tool returns none of this; the probe returns it because
# proving the interpreter differs is its whole job.
WHERE_IT_RAN = ("executable", "cwd", "sadt_api", "sees_api_token")
IGNORED = parity.DEFAULT_IGNORED_KEYS + WHERE_IT_RAN


class InProcessProbe(Tool):
    """What `_dispatch_probe` would look like if it had never been packaged:
    the same arithmetic, the same file, in this process."""

    name = "_dispatch_probe"
    arguments = {
        "a": ArgSpec(type=int),
        "b": ArgSpec(type=int),
        "out_name": ArgSpec(type=str, required=False),
    }
    output_kind = "text"

    total_offset = 0
    written_suffix = ""

    def run(self, a: int, b: int, out_name: str = "probe.txt"):
        total = a + b + self.total_offset
        output_dir = file_utils.make_scratch_dir(prefix="parity_")
        output_path = os.path.join(output_dir, out_name)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(str(a + b) + self.written_suffix)
        return {"outputs": {"probe": output_path}, "total": total, "tags": []}


class DriftingResult(InProcessProbe):
    """Same file, different answer."""

    total_offset = 1


class DriftingFile(InProcessProbe):
    """Same answer, different file."""

    written_suffix = "\n"


@pytest.fixture
def packaged_probe(probe_python, probe_name):
    """The probe as the server sees it in production: read from its
    .schema.json, run in its own interpreter, never imported."""
    return parity.packaged_tool(os.path.join(settings.TOOLS_DIR, probe_name))


def _compare(imported_tool, packaged, arguments=None):
    arguments = arguments or {"a": 2, "b": 3}
    imported = parity.run_once(imported_tool, arguments, IGNORED)
    other = parity.run_once(packaged, arguments, IGNORED)
    return imported, other, parity.compare(imported, other)


def test_the_two_forms_of_a_tool_agree(packaged_probe):
    imported, packaged, report = _compare(InProcessProbe(), packaged_probe)

    assert report.ok, report.differing or report.result_imported
    assert report.identical == ["probe.txt"]
    # Two different directories, one artifact name: this is what makes the
    # comparison possible at all.
    assert imported.artifacts == packaged.artifacts
    assert report.result_imported == {
        "outputs": {"probe": "<artifact:probe.txt>"},
        "total": 5,
        "tags": [],
    }


def test_a_different_answer_is_caught(packaged_probe):
    _, _, report = _compare(DriftingResult(), packaged_probe)

    assert not report.ok
    assert not report.results_match
    assert report.result_imported["total"] != report.result_packaged["total"]


def test_a_different_file_is_caught(packaged_probe):
    """The returned value is identical here; only the bytes on disk moved --
    which is precisely the failure a port introduces and a smoke test misses."""
    _, _, report = _compare(DriftingFile(), packaged_probe)

    assert not report.ok
    assert report.differing == ["probe.txt"]
    assert report.results_match


def test_a_missing_artifact_is_named(packaged_probe):
    """The two runs are asked for different file names, so neither produces
    the other's: the report says which side is missing what rather than just
    'they differ'."""
    imported = parity.run_once(InProcessProbe(), {"a": 1, "b": 1, "out_name": "left.txt"}, IGNORED)
    packaged = parity.run_once(packaged_probe, {"a": 1, "b": 1, "out_name": "right.txt"}, IGNORED)

    report = parity.compare(imported, packaged)

    assert report.only_imported == ["left.txt"]
    assert report.only_packaged == ["right.txt"]


def test_a_parity_run_leaves_nothing_behind(packaged_probe):
    """A parity run on a cohort produces as much data as a real one, and it is
    patient data."""
    before = set(os.listdir(settings.TEMP_DIR))

    _compare(InProcessProbe(), packaged_probe)

    assert set(os.listdir(settings.TEMP_DIR)) == before


def test_an_imported_tool_is_loaded_by_path():
    """Both forms share a name, so the registry only ever holds one of them.
    Comparing them means loading the imported one directly."""
    folder = os.path.join(settings.TOOLS_DIR, "test_tool")

    tool = parity.imported_tool(folder)

    assert tool.name == "test_tool"
    assert tool.invoke({"text_1": "hello", "text_2": "world"}) == "hello world"


def test_noisy_keys_are_left_out_of_the_comparison():
    """A duration and a job id differ between two runs of the SAME code."""
    assert "duration_seconds" in parity.DEFAULT_IGNORED_KEYS
    assert "job_id" in parity.DEFAULT_IGNORED_KEYS

    normalized = parity._normalize(
        {"duration_seconds": 12.4, "job_id": "abc", "kept": 1}, [], parity.DEFAULT_IGNORED_KEYS
    )

    assert normalized == {"kept": 1}
