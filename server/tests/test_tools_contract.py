"""GET /tools is a frozen wire contract.

The Slicer client (SlicerAutomatedDentalToolsCloud) builds its whole UI from
this response: every field it reads -- types, extensions, choices, labels,
sections, visible_when, ui, groups -- comes from here and from nowhere else. A
change to the shape is a client release, not a server detail.

tests/golden/tools_response.json is the response as the server produced it
while tools still ran IN-PROCESS. It is the reference the subprocess-dispatch
work is measured against: the point of that work is that a client cannot tell
the difference.

IF THIS TEST FAILS, the client breaks. Do not regenerate the fixture to make it
pass -- report the difference instead. It is regenerated deliberately, and only
together with a client release:

    cd server && API_TOKEN=x python -c "import json, os; \
        os.environ['API_TOKEN']='x'; \
        from fastapi.testclient import TestClient; from main import app; \
        json.dump(TestClient(app).get('/tools').json(), \
                  open('tests/golden/tools_response.json','w'), indent=2)"
"""

import json
import os

# Set before importing main, so config.Settings() picks up a known token
# regardless of whatever is in the developer's local .env.
os.environ.setdefault("API_TOKEN", "test-token")

from fastapi.testclient import TestClient

import registry
from main import app

client = TestClient(app)

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "tools_response.json")


def _golden() -> dict:
    with open(GOLDEN_PATH) as handle:
        return {tool["name"]: tool for tool in json.load(handle)}


def _served() -> dict:
    response = client.get("/tools")
    assert response.status_code == 200
    return {tool["name"]: tool for tool in response.json()}


def _absent_for_a_load_failure(names: set) -> set:
    """The golden tools missing from this run because they failed to load.

    Discovery is lenient (see registry.py), so an environment without torch
    registers fewer tools -- that is a dependency problem, not a wire-shape
    one, and it must not be reported as a broken contract. FAILED_TOOLS is
    keyed by FOLDER name, which is the tool name for every tool here; a tool
    whose two names differ falls through as a real failure, which is the safe
    direction.
    """
    return names & set(registry.FAILED_TOOLS)


def test_no_tool_gained_or_lost_its_entry():
    served, golden = _served(), _golden()

    unexpected = set(served) - set(golden)
    assert not unexpected, (
        f"Tool(s) {sorted(unexpected)} appear in GET /tools but not in the golden fixture. "
        f"A new tool is a client-visible addition: regenerate the fixture deliberately."
    )

    missing = set(golden) - set(served)
    assert not (missing - _absent_for_a_load_failure(missing)), (
        f"Tool(s) {sorted(missing)} vanished from GET /tools without a load failure to "
        f"explain them. The client would lose their panels."
    )


def test_every_argument_schema_is_byte_identical():
    """Compared per tool rather than as one blob, so a failure names the tool
    and the argument instead of dumping 1500 lines of diff."""
    served, golden = _served(), _golden()

    for name, tool in sorted(served.items()):
        expected = golden[name]
        assert set(tool["arguments"]) == set(expected["arguments"]), (
            f"Tool '{name}': the set of arguments changed "
            f"(now {sorted(tool['arguments'])})."
        )
        for arg_name, spec in tool["arguments"].items():
            assert spec == expected["arguments"][arg_name], (
                f"Tool '{name}', argument '{arg_name}': the published schema changed."
            )
        assert tool["output_kind"] == expected["output_kind"], (
            f"Tool '{name}': output_kind changed."
        )
        # Nothing else may appear either: an added key is a shape change, and
        # the client's form generator iterates what it is given.
        assert set(tool) == set(expected), f"Tool '{name}': the top-level keys changed."


def test_the_tools_are_served_in_the_same_order():
    """The client renders them in the order it receives, so the order is part
    of what the user sees."""
    response = client.get("/tools")
    served = [tool["name"] for tool in response.json()]
    with open(GOLDEN_PATH) as handle:
        golden = [tool["name"] for tool in json.load(handle)]

    assert served == [name for name in golden if name in set(served)]
