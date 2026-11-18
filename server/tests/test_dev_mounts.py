"""Which tools the dev override serves from the checkout, and which from the image.

The list in `docker-compose.dev.yml` is hand-written, and nothing has ever
compared it to the tools that exist: on 2026-09-29 it named **ten of eighteen**,
and the eight it left out included all three AREG engines -- the ones being
worked on that week. The failure is silent and worse than a crash, because the
tool still answers: a fix is committed, pulled, reported as deployed, and the
container goes on running the copy baked into the image.

So every tool is named here with a decision, and the decision has a reason. A
tool added to `SADT-VISOR` and to no list fails this file rather than being
quietly frozen.
"""

import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMPOSE_DEV = os.path.join(REPO_ROOT, "docker-compose.dev.yml")
TOOLS_CHECKOUT = os.path.join(os.path.dirname(REPO_ROOT), "SADT-VISOR", "tools")

# Served from the checkout: an edit under `src/` is live on the next run, with
# no rebuild. Only for a tool whose SOURCE is ahead of the image -- the `.venv`
# stays the image's, so a change that needs a new dependency is not covered by
# a mount and must not be listed here to pretend otherwise.
MOUNTED = {
    "ALI/ALI_CBCT", "ALI/ALI_IOS", "AMASSS", "ASO", "AutoCrop3D", "AutoMatrix",
    "Batch_Dental_Seg", "CLIC", "Crown_Seg", "GreedyReg",
    "AREG/AREG_CBCT", "AREG/AREG_IOS", "AREG/AREG_IOSCBCT",
}

# Served from the image, on purpose. Not an oversight -- each of these is
# either stable or developed elsewhere, so mounting it would only make the
# container follow a working tree nobody is editing.
FROM_THE_IMAGE = {
    "Agent": "developed in its own repository; the checkout here is a mirror",
    "CNE": "stable, and its llama-cpp build is the image's business",
    "DOCShapeAXI": "auxiliary, pinned to shapeaxi 1.x, not under change",
    "FlexReg": "developed in parallel; its source is not what this deployment runs",
    "Surg_Mov_Pred": "stable, tabular, unchanged since the port",
}


def _mounted_in_compose():
    with open(COMPOSE_DEV) as handle:
        text = handle.read()
    return set(re.findall(r"\.\./SADT-VISOR/tools/(.+?)/src:", text))


def test_the_override_mounts_exactly_what_this_file_declares():
    assert _mounted_in_compose() == MOUNTED


def test_no_tool_is_in_both_lists():
    assert not (MOUNTED & set(FROM_THE_IMAGE))


@pytest.mark.skipif(not os.path.isdir(TOOLS_CHECKOUT),
                    reason="no SADT-VISOR checkout beside this one")
def test_every_tool_that_exists_is_decided_one_way_or_the_other():
    """The check that would have caught it: a tool nobody listed.

    A `[tool.sadt]` marker is what the registry, the Dockerfile and CI all use
    to mean "this is a tool", so it is what this counts too -- rather than a
    folder name, which would also catch `common/` and `_template/`.
    """
    found = set()
    for root, directories, files in os.walk(TOOLS_CHECKOUT):
        if ".venv" in root:
            directories[:] = []
            continue
        if "pyproject.toml" not in files:
            continue
        with open(os.path.join(root, "pyproject.toml")) as handle:
            if "[tool.sadt]" not in handle.read():
                continue
        found.add(os.path.relpath(root, TOOLS_CHECKOUT))

    undecided = found - MOUNTED - set(FROM_THE_IMAGE)
    assert not undecided, (
        f"tools neither mounted nor declared image-served: {sorted(undecided)}. "
        "Add each to MOUNTED (and to docker-compose.dev.yml) or to "
        "FROM_THE_IMAGE with the reason."
    )

    vanished = (MOUNTED | set(FROM_THE_IMAGE)) - found
    assert not vanished, f"declared but no longer a tool: {sorted(vanished)}"
