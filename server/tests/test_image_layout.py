"""Guards on the deployment image's two invariants, checked from the test
suite because neither of them fails visibly.

An API process that quietly grows a dependency on numpy still works -- right
up to the day two tools want different numpys and the server is back to being
pinned to whatever they can agree on. A build fixture whose schema no longer
matches its source still builds an image; it just serves no tools.
"""

import os

import pytest

os.environ.setdefault("API_TOKEN", "test-token")

import schema_hash
import schema_tool

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCKER_DIR = os.path.join(REPO_ROOT, "docker")
API_REQUIREMENTS = os.path.join(REPO_ROOT, "server", "requirements-api.txt")

# What the API process is allowed to need. Everything heavy belongs to a tool,
# in the tool's own virtualenv, one exec() away.
ALLOWED_API_REQUIREMENTS = {
    "fastapi",
    "uvicorn[standard]",
    "python-multipart",
    "pydantic-settings",
}


def _requirements(path: str) -> set:
    with open(path) as handle:
        return {
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        }


def test_the_api_requirements_stay_slim():
    """The whole migration is this list staying short. Adding to it is a
    decision, not a detail -- so it is spelled out here rather than only in a
    comment."""
    declared = _requirements(API_REQUIREMENTS)

    assert declared == ALLOWED_API_REQUIREMENTS, (
        f"server/requirements-api.txt changed to {sorted(declared)}. This file is what the "
        f"API process installs: it may only grow a dependency the SERVER itself needs "
        f"(routing, validation, moving bytes). Anything a tool needs goes in that tool's "
        f"pyproject.toml. If this is deliberate, update ALLOWED_API_REQUIREMENTS."
    )


def test_no_heavy_dependency_creeps_into_the_api():
    """Spelled out separately from the exact-set check above, because these are
    the names that undo the architecture rather than merely widen it."""
    declared = " ".join(_requirements(API_REQUIREMENTS)).lower()

    for heavy in ("torch", "numpy", "pandas", "simpleitk", "vtk", "nnunet", "monai", "itk"):
        assert heavy not in declared, (
            f"'{heavy}' must not be a dependency of the API process: it pins the server's "
            f"Python to what the tools can agree on, which is the problem the per-tool "
            f"virtualenvs solve."
        )


def _fixture_folders() -> list:
    fixtures = os.path.join(DOCKER_DIR, "fixtures")
    if not os.path.isdir(fixtures):
        return []
    return [
        os.path.join(fixtures, entry)
        for entry in sorted(os.listdir(fixtures))
        if os.path.isdir(os.path.join(fixtures, entry))
    ]


@pytest.mark.skipif(not os.path.isdir(DOCKER_DIR), reason="docker/ is not mounted here")
def test_the_build_fixtures_describe_their_own_source():
    """The image's fixtures are packaged exactly like a real tool, so they are
    subject to the same rule: a schema that no longer matches its src/ refuses
    to start the server -- which for these would mean an image that builds and
    serves nothing."""
    folders = _fixture_folders()
    assert folders, "the build fixtures are missing"

    for folder in folders:
        schema = schema_tool.read_schema(folder)
        actual = schema_hash.hash_source_tree(os.path.join(folder, "src"))
        assert schema["source_hash"] == actual, (
            f"{os.path.basename(folder)}: .schema.json no longer matches its src/. "
            f"Regenerate it with `python server/schema_hash.py "
            f"docker/fixtures/{os.path.basename(folder)}/src`."
        )


@pytest.mark.skipif(not os.path.isdir(DOCKER_DIR), reason="docker/ is not mounted here")
def test_the_fixtures_pin_incompatible_dependencies_on_purpose():
    """They are not decoration: two tools that cannot share an interpreter, in
    one image, is the claim the layout rests on. If they ever agree on a numpy,
    the image proves nothing."""
    pins = {}
    for folder in _fixture_folders():
        with open(os.path.join(folder, "pyproject.toml")) as handle:
            pins[os.path.basename(folder)] = handle.read()

    assert 'numpy<2' in pins.get("numpy_old", "")
    assert 'numpy>=2' in pins.get("numpy_new", "")
    # And one that must resolve to the SAME wheel as another, or
    # docker/verify_dedup.py has nothing to compare and cannot fail.
    assert 'numpy>=2' in pins.get("numpy_twin", "")
