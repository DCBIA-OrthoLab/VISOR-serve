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

from registry import schema_hash
from registry import schema_tool

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
    fixtures = os.path.join(DOCKER_DIR, "fixtures", "tools")
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


# ---------------------------------------------------------------------------
# What an in-process tool may import
# ---------------------------------------------------------------------------
# The suite runs on a fat interpreter (conda: pandas, numpy, torch); the
# deployed API runs on `/opt/sadt/.venv`, which holds the four packages above
# and nothing else. So a module in `server/` can import pandas, pass every
# test, and raise `ModuleNotFoundError` on the first real request -- which is
# exactly what `Example_Tool` did, for months, while two tests that CALL it
# went on passing. This walks the imports instead of trusting the interpreter.

import ast
import sys

SERVER_DIR = os.path.join(REPO_ROOT, "server")

# Third-party distributions the API venv provides, as the module names they
# install under.
API_MODULES = {
    "fastapi", "starlette", "uvicorn", "multipart", "pydantic", "pydantic_settings",
    "anyio", "sniffio", "idna", "click", "h11", "typing_extensions", "annotated_types",
    "dotenv", "yaml", "certifi", "httpx", "httpcore", "watchfiles", "websockets",
    "httptools", "uvloop", "colorama",
}
# The server's own top-level modules, importable because `server/` is on the path.
SERVER_MODULES = {
    os.path.splitext(name)[0]
    for name in os.listdir(SERVER_DIR)
    if name.endswith(".py") or os.path.isdir(os.path.join(SERVER_DIR, name))
}


def _catches_missing_module(handlers) -> bool:
    """Whether a try block's handlers cover an absent module."""
    names = set()
    for handler in handlers:
        node = handler.type
        for part in (node.elts if isinstance(node, ast.Tuple) else [node]):
            if isinstance(part, ast.Name):
                names.add(part.id)
    return bool(names & {"ImportError", "ModuleNotFoundError", "Exception"})


def _imported_roots(node, guarded: bool = False) -> set:
    """Every top-level module name imported WITHOUT a fallback, lazily or not.

    An import inside `try: ... except ModuleNotFoundError:` is deliberate and
    safe -- `tomllib` with a `tomli` fallback for 3.10 is the case in the tree
    -- so it does not count. An unguarded one is a hard dependency wherever it
    sits, function body included.
    """
    roots = set()
    if isinstance(node, ast.Import) and not guarded:
        roots.update(alias.name.split(".")[0] for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and not guarded and node.level == 0 and node.module:
        roots.add(node.module.split(".")[0])

    if isinstance(node, ast.Try):
        covered = _catches_missing_module(node.handlers)
        # Only the BODY is protected. An import in `else:` runs after the body
        # succeeded and in `finally:` runs regardless -- in both, the handler
        # is already past, so a missing module raises out of the try. The
        # handlers themselves are the fallback and are protected with the body.
        for child in node.body + node.handlers:
            roots |= _imported_roots(child, guarded or covered)
        for child in node.orelse + node.finalbody:
            roots |= _imported_roots(child, guarded)
        return roots

    for child in ast.iter_child_nodes(node):
        roots |= _imported_roots(child, guarded)
    return roots


def _file_imports(path: str) -> set:
    with open(path, encoding="utf-8") as handle:
        return _imported_roots(ast.parse(handle.read(), filename=path))


def _server_python_files() -> list:
    found = []
    for directory, _, names in os.walk(SERVER_DIR):
        parts = set(directory.split(os.sep))
        if parts & {"tests", "venv", ".venv", "__pycache__"}:
            continue
        found.extend(
            os.path.join(directory, name) for name in names if name.endswith(".py")
        )
    return found


def test_the_server_imports_nothing_the_api_venv_lacks():
    """Including from an in-process tool, and including a lazy import inside a
    function -- a `ModuleNotFoundError` raised there is a 500 on a real
    request, not a failure anyone sees at startup."""
    allowed = API_MODULES | SERVER_MODULES | set(sys.stdlib_module_names)
    offenders = {}
    for path in _server_python_files():
        extra = sorted(_file_imports(path) - allowed)
        if extra:
            offenders[os.path.relpath(path, REPO_ROOT)] = extra

    assert not offenders, (
        f"These modules under server/ import packages the API virtualenv does not have: "
        f"{offenders}. The suite's interpreter is fatter than the deployed one, so this "
        f"passes here and answers 500 in production. An in-process tool may use the "
        f"standard library and the server's own modules; anything else belongs to a "
        f"packaged tool with its own venv."
    )


def test_an_import_in_the_else_of_a_guarded_try_still_counts():
    """`else:` runs after the body succeeded, so the handler is already past
    and a missing module raises out of the try. Protecting it would let a
    hard dependency into the API venv unnoticed."""
    source = (
        "try:\n"
        "    import tomllib\n"
        "except ModuleNotFoundError:\n"
        "    import tomli as tomllib\n"
        "else:\n"
        "    import numpy\n"
    )
    assert "numpy" in _imported_roots(ast.parse(source))
    assert "tomllib" not in _imported_roots(ast.parse(source))


def test_an_import_in_the_finally_of_a_guarded_try_still_counts():
    source = (
        "try:\n"
        "    import tomllib\n"
        "except ModuleNotFoundError:\n"
        "    pass\n"
        "finally:\n"
        "    import pandas\n"
    )
    assert "pandas" in _imported_roots(ast.parse(source))


def test_the_fallback_import_in_the_handler_is_still_protected():
    source = (
        "try:\n"
        "    import tomllib\n"
        "except ModuleNotFoundError:\n"
        "    import tomli as tomllib\n"
    )
    assert _imported_roots(ast.parse(source)) == set()


def test_a_try_that_catches_the_wrong_error_protects_nothing():
    source = (
        "try:\n"
        "    import numpy\n"
        "except ValueError:\n"
        "    pass\n"
    )
    assert "numpy" in _imported_roots(ast.parse(source))
