"""Inputs a tool's default mode needs although its schema calls them optional.

A tool with modes cannot publish "required, depending on another argument", so
an input only some modes need goes out as `required: false` with a
`visible_when` naming those modes. The "each-tool-solo" preset sent two such
tools their required arguments alone, and both refused before doing anything:

* FlexReg: "'Patch and register' registers onto a reference surface: name one
  in 'reference'." -- its default mode needs `reference`;
* VFACE: "Registering needs masks around the regions measured, and
  'segmentation_model' names the bundle that makes them."

The schemas below are shaped like the ones those two tools publish, projected
the way `main._benchmark_schemas` projects them, so the resolver is exercised
with dictionaries and no tool.
"""

import benchmark_presets as presets


def argument(required=False, selectable=None, kind="path", choices=None,
             initial=None, visible_when=None, scope=None, model_named=False):
    return {"required": required, "server_selectable": selectable,
            "initial": initial, "choices": choices, "type": kind,
            "visible_when": visible_when, "selectable_scope": scope,
            "model_named": model_named}


def flexreg_shaped() -> dict:
    return {"arguments": {
        "surfaces": argument(required=True, selectable="testfile"),
        "mode": argument(kind="choice", choices={
            "Patch": False, "Register": False, "Patch and register": True}),
        "patch": argument(kind="choice", choices={
            "Palate (butterfly)": True, "Mucogingival line": False}),
        "reference": argument(
            selectable="testfile",
            visible_when={"mode": ["Register", "Patch and register"]}),
        "anterior_right": argument(
            kind="vec2", initial=[0.5, 0.0],
            visible_when={"patch": "Palate (butterfly)"}),
        "shift": argument(
            kind="vec2", initial=[0.0, 0.0],
            visible_when={"patch": "Palate (butterfly)"}),
        "output_suffix": argument(kind="str", initial="_Reg"),
    }}


def vface_shaped() -> dict:
    """Every bundle opted out of the hosted convention, as the deployment
    declares them: the tool resolves its own from its data folder."""
    measuring = ["Measurements and heat maps", "Measurements and classification"]
    return {"arguments": {
        "t1": argument(required=True, selectable="testfile", scope="Scans"),
        "mode": argument(kind="choice", choices={
            "Full pipeline": True, "File already Oriented": False,
            "File already Registered": False}),
        "study": argument(kind="choice", choices={
            "Asymmetry assessment": True, "Longitudinal study": False}),
        "outputs": argument(kind="choice", choices={
            "Measurements and classification": True,
            "Measurements and heat maps": False, "Heat maps": False}),
        "t2": argument(visible_when={"study": "Longitudinal study"}),
        "registration_transforms": argument(
            visible_when={"mode": "File already Registered"}),
        "segmentation_model": argument(
            model_named=True,
            visible_when={"mode": ["Full pipeline", "File already Oriented"]}),
        "cranial_base_reference": argument(
            model_named=True, visible_when={"mode": "Full pipeline"}),
        "mirror_reference": argument(
            model_named=True, visible_when={"study": "Asymmetry assessment"}),
        "measurements": argument(visible_when={"outputs": measuring}),
        "classifier_model": argument(
            model_named=True, visible_when={"outputs": measuring}),
        "surface_model": argument(
            model_named=True,
            visible_when={"outputs": ["Measurements and heat maps", "Heat maps"]}),
    }}


# ---------------------------------------------------------------------------
# A hosted input the default mode needs
# ---------------------------------------------------------------------------

def test_the_reference_the_default_mode_registers_onto_is_sent():
    resolved = presets.resolve_arguments(
        flexreg_shaped(), {"testfiles": ["Arches"]})

    assert resolved["params"] == {"surfaces": "Arches", "reference": "Arches"}
    assert resolved["missing"] == []


def test_a_mode_input_nothing_hosted_can_fill_makes_the_tool_unrunnable():
    """Reported up front rather than sent: the alternative is the 422 the
    battery collected and showed as the tool failing."""
    schema = flexreg_shaped()
    schema["arguments"]["reference"]["selectable_scope"] = "References"
    resolved = presets.resolve_arguments(
        schema, {"testfiles": ["Arches"], "testfiles_by_scope": {"References": []}})

    assert resolved["params"] == {"surfaces": "Arches"}
    assert resolved["missing"] == ["reference"]


def test_an_input_of_a_mode_that_is_not_on_is_still_not_sent():
    schema = flexreg_shaped()
    schema["arguments"]["mode"]["choices"] = {
        "Patch": True, "Register": False, "Patch and register": False}
    resolved = presets.resolve_arguments(schema, {"testfiles": ["Arches"]})

    assert resolved["params"] == {"surfaces": "Arches"}
    assert resolved["missing"] == []


def test_a_hosted_model_the_default_mode_needs_is_sent_by_name():
    """The same rule for a bundle as for a test file, drawn from the models."""
    resolved = presets.resolve_arguments(
        {"arguments": {
            "scans": argument(required=True, selectable="testfile"),
            "mode": argument(kind="choice", choices={"Segment": True, "Crop": False}),
            "segmentation_model": argument(
                selectable="model", model_named=True,
                visible_when={"mode": "Segment"}),
        }},
        {"testfiles": ["Scan"], "models": ["Bundle"]})

    assert resolved["params"] == {"scans": "Scan", "segmentation_model": "Bundle"}
    assert resolved["missing"] == []


def test_a_facade_sent_its_mode_alone_is_reported_instead():
    """Before this, a facade whose mode inputs had nothing staged was sent its
    mode alone and answered `422 Missing required argument 't1'`."""
    resolved = presets.resolve_arguments(
        {"arguments": {
            "mode": argument(required=True, kind="choice",
                             choices={"CBCT": True, "IOS": False, "IOS to CBCT": False}),
            "t1": argument(selectable="testfile", scope="CBCT_T1",
                           visible_when={"mode": ["CBCT", "IOS"]}),
            "ios": argument(selectable="testfile", scope="IOSCBCT",
                            visible_when={"mode": ["IOS to CBCT"]}),
        }},
        {"testfiles_by_scope": {"CBCT_T1": [], "IOSCBCT": ["Cohort"]}})

    assert resolved["params"] == {"mode": "CBCT"}
    assert resolved["missing"] == ["t1"]


# ---------------------------------------------------------------------------
# A bundle the tool resolves itself
# ---------------------------------------------------------------------------

def test_a_bundle_the_tool_resolves_itself_is_neither_sent_nor_missing():
    """Opted out of the hosted convention, so there is no name to send -- and
    with bundles staged the tool finds its own, so the tool stays runnable."""
    resolved = presets.resolve_arguments(
        vface_shaped(), {"testfiles_by_scope": {"Scans": ["Scan"]},
                         "models": ["Bundle_a", "Bundle_b"]})

    assert resolved["params"] == {"t1": "Scan"}
    assert resolved["missing"] == []


def test_a_tool_hosting_no_bundle_for_its_default_mode_is_unrunnable():
    """What the server CAN see of an opted-out bundle: nothing staged at all,
    so the tool's own resolution has nothing to find. Only the bundles the
    default values put in force are named."""
    resolved = presets.resolve_arguments(
        vface_shaped(), {"testfiles_by_scope": {"Scans": ["Scan"]}, "models": []})

    assert resolved["params"] == {"t1": "Scan"}
    assert resolved["missing"] == ["classifier_model", "cranial_base_reference",
                                   "mirror_reference", "segmentation_model"]


def test_an_upload_only_input_is_never_reported_missing():
    """`measurements` and `registration_transforms` are not named like
    bundles; whether the tool fills them is the tool's to say."""
    resolved = presets.resolve_arguments(
        vface_shaped(), {"testfiles_by_scope": {"Scans": ["Scan"]}, "models": []})

    assert "measurements" not in resolved["missing"]
    assert "registration_transforms" not in resolved["missing"]


def test_a_tool_that_cannot_run_is_left_out_of_the_solo_preset():
    resolved = presets.runnable_tools(
        {"FlexReg": flexreg_shaped(), "VFACE": vface_shaped()},
        lambda name: {"testfiles": ["Arches"], "models": [],
                      "testfiles_by_scope": {"Scans": ["Scan"]}})

    plan = presets.build_plan("each-tool-solo", resolved)

    assert [run["tool"] for run in plan["arms"][0]["runs"]] == ["FlexReg"]
    assert plan["arms"][0]["runs"][0]["params"] == {
        "surfaces": "Arches", "reference": "Arches"}


# ---------------------------------------------------------------------------
# What the resolution still leaves alone
# ---------------------------------------------------------------------------

def test_a_mode_with_one_input_is_a_requirement_not_a_derivation():
    """A value showing the ONLY input its setting governs needs that input;
    it is a value showing every one of several that picks its path from them."""
    assert presets._derives_mode(
        flexreg_shaped()["arguments"], "mode", "Patch and register") is False
    assert presets._derives_mode(
        {"a": argument(visible_when={"automation": ["Auto", "A"]}),
         "b": argument(visible_when={"automation": ["Auto", "B"]})},
        "automation", "Auto") is True
    assert presets._derives_mode(
        {"a": argument(visible_when={"automation": ["Auto", "A"]}),
         "b": argument(visible_when={"automation": ["B"]})},
        "automation", "Auto") is False
