"""What a preset would run here, and the four ways that can be wrong.

`benchmark_presets` decides whether a tool is playable on this deployment and
with which arguments. It is worth pinning because every one of its mistakes
arrives wearing somebody else's clothes:

* a required argument left out reaches the tool as a 422, and reads on the
  launcher as "that tool failed";
* a required argument filled from the wrong pool runs the tool against the
  wrong bundle and succeeds, which is worse;
* a cap that does not hold turns one button into forty GPU runs;
* a tool reported runnable when nothing is staged for it kills a battery
  halfway through, after the arms before it have already been measured.

None of these needs a server, a card or a tool: the resolver takes a schema and
a hosting listing, so the whole surface is exercised with dictionaries.
"""

import pytest

import benchmark_presets as presets


def schema(**arguments) -> dict:
    return {"arguments": arguments}


def argument(required=True, selectable=None, initial=None, choices=None,
             kind="path") -> dict:
    return {"required": required, "server_selectable": selectable,
            "initial": initial, "choices": choices, "type": kind}


# ---------------------------------------------------------------------------
# Resolving one tool
# ---------------------------------------------------------------------------

def test_a_hosted_input_is_filled_by_name():
    resolved = presets.resolve_arguments(
        schema(scans=argument(selectable="testfile"),
               model=argument(selectable="model")),
        {"testfiles": ["MG_test_scan.nii.gz"], "models": ["AMASSS_Models"]})
    assert resolved["params"] == {"scans": "MG_test_scan.nii.gz",
                                  "model": "AMASSS_Models"}
    assert resolved["missing"] == []


def test_a_model_argument_is_never_filled_from_the_test_files():
    """The two pools are not interchangeable. Filling `model` from a test file
    would run the tool against something that is not a bundle -- and a tool
    handed the wrong weights can succeed, which is the failure nobody sees."""
    resolved = presets.resolve_arguments(
        schema(model=argument(selectable="model")),
        {"testfiles": ["a_scan.nii.gz"], "models": []})
    assert resolved["params"] == {}
    assert resolved["missing"] == ["model"]


def test_a_hosted_input_with_nothing_staged_is_reported_missing():
    resolved = presets.resolve_arguments(
        schema(scans=argument(selectable="testfile")), {"testfiles": []})
    assert resolved["missing"] == ["scans"]


def test_the_first_hosted_name_is_taken_so_two_batteries_compare():
    """Deterministic on purpose: picking at random would make two runs of one
    preset on one deployment incomparable with each other."""
    hosted = {"testfiles": ["alpha", "beta", "gamma"], "models": []}
    first = presets.resolve_arguments(schema(scans=argument(selectable="testfile")), hosted)
    second = presets.resolve_arguments(schema(scans=argument(selectable="testfile")), hosted)
    assert first["params"] == second["params"] == {"scans": "alpha"}


def test_a_required_scalar_with_no_default_is_sent_rather_than_skipped():
    """The bug this file exists for. The schema calls a default `initial`, and
    reading `default` instead made every required scalar look answered -- so the
    battery posted a form with the field missing and collected a 422 it
    reported as the tool failing."""
    resolved = presets.resolve_arguments(
        schema(text_1=argument(kind="str"), text_2=argument(kind="str")), {})
    assert resolved["params"] == {"text_1": "benchmark", "text_2": "benchmark"}
    assert resolved["missing"] == []


def test_a_required_scalar_the_tool_already_defaults_is_left_alone():
    """So the run exercises the tool's own default rather than a value this
    module invented."""
    resolved = presets.resolve_arguments(
        schema(device=argument(kind="str", initial="cuda")), {})
    assert resolved["params"] == {}
    assert resolved["missing"] == []


def test_an_optional_argument_is_never_sent():
    resolved = presets.resolve_arguments(
        schema(extra=argument(required=False, selectable="testfile")),
        {"testfiles": ["something"]})
    assert resolved["params"] == {}


def test_an_input_a_mode_requires_is_sent_although_the_facade_calls_it_optional():
    """The case a facade cannot express, and the one this exists for.

    `AREG` publishes ONE required argument -- its mode -- and every input as
    optional, because `t1` means something for its CBCT modes and `ios` only
    for CBCT-to-IOS. A facade has no way to say "required, depending on another
    argument", so a launcher that trusts `required` sends the mode alone and
    the tool the facade dispatched to refuses. Measured on 2026-09-28: six
    concurrent `POST /run/AREG` answered `422 Missing required argument 't1'
    for tool 'AREG_CBCT'` in twenty milliseconds, reported as AREG failing.

    `visible_when` is what the facade DOES publish, and it is the reason the
    argument is optional rather than a mere fact that it is.
    """
    resolved = presets.resolve_arguments(
        schema(
            mode=argument(kind="choice", choices={"CBCT": True, "IOS": False}),
            t1=dict(argument(required=False, selectable="testfile"),
                    visible_when={"mode": ["CBCT", "IOS"]}),
            ios=dict(argument(required=False, selectable="testfile"),
                     visible_when={"mode": ["CBCT to IOS"]}),
        ),
        {"testfiles": ["AREG_test_scans"]})

    # `t1` applies to the mode that is on; `ios` belongs to a mode that is not.
    # And the mode itself travels -- see the test below for why.
    assert resolved["params"] == {"mode": "CBCT", "t1": "AREG_test_scans"}
    assert resolved["missing"] == []


def test_an_optional_input_is_left_alone_when_the_tool_already_has_one():
    """The guard against helping too much, and it is not hypothetical.

    ASO declares an optional `landmarks` folder, shown in every mode -- and
    supplying it is what makes ASO register on the caller's points INSTEAD of
    asking the landmark tool for them. Filling it would turn a benchmark of the
    ASO-to-ALI chain into a benchmark of ASO alone, and the number would read
    as an improvement. Optional everywhere means the tool has its own answer.
    """
    resolved = presets.resolve_arguments(
        schema(
            input=argument(selectable="testfile"),
            landmarks=argument(required=False, selectable="testfile"),
        ),
        {"testfiles": ["CBCT_FullyAuto"]})

    assert resolved["params"] == {"input": "CBCT_FullyAuto"}
    assert resolved["missing"] == []


def test_an_input_a_mode_derives_from_is_left_alone():
    """The other way of helping too much: a value that picks its path from
    what it is given.

    AREG's engines default `automation` to "From the data", which shows every
    input the setting governs because any of them may decide the path. Filling
    `cbct_landmarks` there would CHOOSE the registration-only path and skip the
    landmark prediction the benchmark is meant to time.
    """
    resolved = presets.resolve_arguments(
        schema(
            cbct=argument(selectable="testfile"),
            automation=argument(
                required=False, kind="choice",
                choices={"From the data": True, "Registration": False,
                         "Fully-Automated": False}),
            cbct_landmarks=dict(
                argument(required=False, selectable="testfile"),
                visible_when={"automation": ["From the data", "Registration"]}),
            reference=dict(
                argument(required=False, selectable="model"),
                visible_when={"automation": ["From the data", "Fully-Automated"]}),
        ),
        {"testfiles": ["IOSCBCT"], "models": ["Frankfurt"]})

    assert resolved["params"] == {"cbct": "IOSCBCT"}
    assert resolved["missing"] == []


def test_an_input_belonging_to_another_mode_is_not_sent():
    """Sending it would be an unknown-for-this-mode argument, which is the
    other way to collect a 422."""
    resolved = presets.resolve_arguments(
        schema(
            mode=argument(kind="choice", choices={"CBCT": True, "IOS": False}),
            ios=dict(argument(required=False, selectable="testfile"),
                     visible_when={"mode": ["IOS"]}),
        ),
        {"testfiles": ["something"]})

    assert resolved["params"] == {"mode": "CBCT"}
    assert "ios" not in resolved["params"]


def test_a_required_choice_travels_because_a_ticked_flag_is_not_a_default():
    """`{option: True}` says which option a PANEL pre-selects. It is not a
    value the server applies -- `initial` is, and a required choice that has
    none has to be sent.

    This test asserted the opposite until 2026-09-28, when a six-client AREG
    arm collected six `422 Missing required argument 'mode' for 'AREG'`. A
    facade needs its mode BEFORE it can dispatch, so it refuses at a point
    where nothing has had the chance to fill a default in. The three arguments
    in this catalogue with this shape are `ALI.mode`, `AREG.mode` -- both
    facades -- and `CNE.notes_type`, for which sending the ticked option is
    what its default would have been anyway.
    """
    resolved = presets.resolve_arguments(
        schema(mode=argument(kind="choice", choices={"CBCT": True, "IOS": False})), {})
    assert resolved["params"] == {"mode": "CBCT"}


def test_a_required_choice_the_tool_defaults_is_still_left_alone():
    """`initial` IS a server-side default, so it is still answered for."""
    resolved = presets.resolve_arguments(
        schema(mode=argument(kind="choice", initial="IOS",
                             choices={"CBCT": True, "IOS": False})), {})
    assert resolved["params"] == {}


def test_a_multichoice_already_ticked_is_left_alone():
    """The half of the old rule that survives, and it matters: sending one
    option of a several-of argument NARROWS what the tool ticked, and a run of
    AMASSS's one structure is not a run of its five."""
    resolved = presets.resolve_arguments(
        schema(structures=argument(kind="multichoice",
                                   choices={"MAND": True, "MAX": True, "SKIN": False})), {})
    assert resolved["params"] == {}


def test_a_choice_with_nothing_on_picks_one_so_the_run_has_a_branch():
    resolved = presets.resolve_arguments(
        schema(mode=argument(kind="choice", choices={"IOS": False, "CBCT": False})), {})
    assert resolved["params"] == {"mode": "CBCT"}


# ---------------------------------------------------------------------------
# Building a plan
# ---------------------------------------------------------------------------

def _resolved(*names, missing=()) -> dict:
    report = {name: {"params": {"input": "x"}, "missing": []} for name in names}
    for name in missing:
        report[name] = {"params": {}, "missing": ["input"]}
    return report


def test_a_sequential_preset_runs_each_tool_once_at_width_one():
    plan = presets.build_plan("each-tool-solo", _resolved("A", "B", "C"))
    assert plan["total_runs"] == 3
    arm = plan["arms"][0]
    assert arm["concurrency"] == 1
    assert [run["tool"] for run in arm["runs"]] == ["A", "B", "C"]


def test_a_parallel_preset_cycles_its_tools_up_to_the_count():
    plan = presets.build_plan("six-at-once", _resolved("CLIC"))
    arm = plan["arms"][0]
    assert len(arm["runs"]) == 6
    assert {run["tool"] for run in arm["runs"]} == {"CLIC"}
    assert arm["concurrency"] == 6


def test_a_ladder_becomes_one_arm_per_step():
    plan = presets.build_plan("ramp", _resolved("CLIC"))
    assert [arm["concurrency"] for arm in plan["arms"]] == [1, 2, 4]
    assert plan["total_runs"] == 7


def test_a_tool_that_cannot_be_resolved_is_left_out_rather_than_run():
    plan = presets.build_plan("mixed-load",
                              _resolved("Test_Tool", "CLIC", "Crown_Seg",
                                        "Surg_Mov_Pred", missing=["CLIC"]))
    tools = {run["tool"] for arm in plan["arms"] for run in arm["runs"]}
    assert "CLIC" not in tools
    assert "Test_Tool" in tools


def test_a_preset_nothing_can_satisfy_refuses_and_names_what_is_missing():
    """Refused, not silently shrunk: a battery that ran four of the six runs
    somebody asked for, and said so nowhere, is worse than one that stopped."""
    with pytest.raises(presets.PresetError) as raised:
        presets.build_plan("six-at-once", _resolved("CLIC", missing=["CLIC"]))
    assert "CLIC" in str(raised.value)


def test_an_unknown_preset_is_refused_by_name():
    with pytest.raises(presets.PresetError):
        presets.build_plan("no-such-thing", _resolved("A"))


# ---------------------------------------------------------------------------
# The caps, which are the difference between a tool and a foot-gun
# ---------------------------------------------------------------------------

def test_the_run_ceiling_holds_against_a_deployment_with_many_tools():
    many = _resolved(*[f"Tool_{index}" for index in range(presets.MAX_RUNS + 5)])
    with pytest.raises(presets.PresetError) as raised:
        presets.build_plan("each-tool-solo", many)
    assert str(presets.MAX_RUNS) in str(raised.value)


def test_a_caller_cannot_ask_for_more_concurrency_than_the_ceiling():
    plan = presets.build_plan("six-at-once", _resolved("CLIC"),
                              concurrency=presets.MAX_CONCURRENCY + 40)
    assert plan["arms"][0]["concurrency"] == presets.MAX_CONCURRENCY


def test_a_caller_can_narrow_the_tool_list():
    plan = presets.build_plan("each-tool-solo", _resolved("A", "B", "C"),
                              tools=["B"])
    assert {run["tool"] for arm in plan["arms"] for run in arm["runs"]} == {"B"}


# ---------------------------------------------------------------------------
# The catalogue the launcher renders
# ---------------------------------------------------------------------------

def test_the_catalogue_says_what_each_preset_would_do_here():
    listing = {entry["id"]: entry for entry in presets.catalogue(_resolved("CLIC"))}
    assert listing["six-at-once"]["runs"] == 6
    assert listing["six-at-once"]["blocked"] is None


def test_the_catalogue_reports_a_blocked_preset_instead_of_raising():
    """The launcher draws every preset, including the ones it cannot offer:
    a card saying why is what stops somebody hunting for a missing button."""
    listing = {entry["id"]: entry for entry in presets.catalogue({})}
    assert listing["six-at-once"]["runs"] == 0
    assert listing["six-at-once"]["blocked"]


def test_runnable_tools_survives_a_listing_it_cannot_read():
    """One tool whose hosting cannot be listed must not take the catalogue
    down with it."""
    def hosted(name):
        raise OSError("DATA is not mounted")

    report = presets.runnable_tools({"A": schema(x=argument(selectable="testfile"))},
                                    hosted)
    assert report["A"]["missing"] == ["x"]
