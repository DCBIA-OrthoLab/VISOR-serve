"""A tool that is a choice between other tools.

The point of composing rather than writing one: change a target and the facade
publishes the change, with nothing to keep in sync. Most of what is worth
testing here is that nothing is ever copied.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("API_TOKEN", "test-token")

from base import ArgSpec, Tool
from registry import facade


class _Target(Tool):
    output_kind = "files"

    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

    def run(self, **kwargs):
        return None


def _registry(**tools):
    return {name: _Target(name, arguments) for name, arguments in tools.items()}


# ---------------------------------------------------------------------------
# Composition


def test_the_facade_publishes_a_mode_choice_over_its_targets():
    registry = _registry(
        A={"shared": ArgSpec(type="path")},
        B={"shared": ArgSpec(type="path")},
    )
    composed = facade.compose("F", {"first": "A", "second": "B"}, registry)

    mode = composed.arguments[facade.MODE_ARGUMENT]
    assert mode.type == "choice"
    assert mode.required
    assert list(mode.choices) == ["first", "second"]
    # First, because every other field's visibility is a condition on it.
    assert list(composed.arguments)[0] == facade.MODE_ARGUMENT


def test_an_argument_only_one_mode_has_is_shown_only_for_that_mode():
    registry = _registry(
        A={"common": ArgSpec(type="path"), "only_a": ArgSpec(type="int", required=False)},
        B={"common": ArgSpec(type="path")},
    )
    composed = facade.compose("F", {"a": "A", "b": "B"}, registry)

    assert composed.arguments["only_a"].visible_when == {facade.MODE_ARGUMENT: ["a"]}
    # Applies everywhere, so no condition to evaluate on every keystroke.
    assert not composed.arguments["common"].visible_when


def test_an_argument_two_modes_share_names_both():
    registry = _registry(
        A={"t1": ArgSpec(type="path")},
        B={"t1": ArgSpec(type="path")},
        C={"other": ArgSpec(type="path")},
    )
    composed = facade.compose("F", {"a": "A", "b": "B", "c": "C"}, registry)

    assert composed.arguments["t1"].visible_when == {facade.MODE_ARGUMENT: ["a", "b"]}


def test_a_required_argument_of_one_mode_is_not_required_of_the_others():
    """The panel would otherwise refuse to apply in a mode that never wanted it.

    Validation still happens, on the target, which is the one that knows.
    """
    registry = _registry(
        A={"needed": ArgSpec(type="path", required=True)},
        B={"other": ArgSpec(type="path", required=True)},
    )
    composed = facade.compose("F", {"a": "A", "b": "B"}, registry)

    assert composed.arguments["needed"].required is False


def test_a_choice_whose_options_differ_per_mode_narrows():
    """AREG's `automation` offers three modes for CBCT and two for IOS."""
    registry = _registry(
        A={"automation": ArgSpec(type="choice", choices={"Semi": True, "Full": False, "Oriented": False})},
        B={"automation": ArgSpec(type="choice", choices={"Semi": True, "Full": False})},
    )
    composed = facade.compose("F", {"a": "A", "b": "B"}, registry)

    narrowed = composed.arguments["automation"].options_when[facade.MODE_ARGUMENT]
    assert sorted(narrowed["a"]) == ["Full", "Oriented", "Semi"]
    assert sorted(narrowed["b"]) == ["Full", "Semi"]


def test_the_default_of_a_narrowed_choice_is_the_first_modes():
    """Measured on AREG, 2026-09-25: `automation` defaults to Fully-Automated
    for CBCT and for IOS, and to Registration for IOSCBCT. Merged last-wins,
    the facade published Registration as the default of the whole tool and
    turned Fully-Automated off -- and Registration is a value CBCT does not
    have, so every CBCT panel opened on the first option left to it,
    Semi-Automated. That is the one mode with no segmentation boxes and with a
    masks field, so the panel read as having lost half of itself.

    The first mode's default, because the first mode is what the panel opens
    on (the mode box is built `index == 0`)."""
    registry = _registry(
        A={"automation": ArgSpec(type="choice", choices={"Semi": False, "Full": True})},
        B={"automation": ArgSpec(type="choice",
                                 choices={"Semi": False, "Full": False, "Reg": True})},
    )
    composed = facade.compose("F", {"a": "A", "b": "B"}, registry)

    choices = composed.arguments["automation"].choices
    assert sorted(choices) == ["Full", "Reg", "Semi"], "the union is still offered"
    assert [option for option, on in choices.items() if on] == ["Full"]


def test_an_option_only_a_later_mode_has_is_offered_but_not_the_default():
    """Its own mode turns it on: `options_when` narrows the list there, and a
    combo that cannot show the current value re-selects within what is left."""
    registry = _registry(
        A={"automation": ArgSpec(type="choice", choices={"Full": True})},
        B={"automation": ArgSpec(type="choice", choices={"Full": False, "Reg": True})},
    )
    composed = facade.compose("F", {"a": "A", "b": "B"}, registry)

    assert composed.arguments["automation"].choices["Reg"] is False
    assert "Reg" in composed.arguments["automation"].options_when[facade.MODE_ARGUMENT]["b"]


def test_a_multichoice_keeps_every_mode_s_own_defaults():
    """Not a combo: several options may legitimately be on, and each mode's
    own default has to survive the union."""
    registry = _registry(
        A={"keep": ArgSpec(type="multichoice", choices={"x": True, "y": False})},
        B={"keep": ArgSpec(type="multichoice", choices={"y": False, "z": True})},
    )
    composed = facade.compose("F", {"a": "A", "b": "B"}, registry)

    choices = composed.arguments["keep"].choices
    assert [option for option, on in choices.items() if on] == ["x", "z"]


def test_a_choice_offering_the_same_options_everywhere_is_left_alone():
    registry = _registry(
        A={"pick": ArgSpec(type="choice", choices={"x": True, "y": False})},
        B={"pick": ArgSpec(type="choice", choices={"x": True, "y": False})},
    )
    composed = facade.compose("F", {"a": "A", "b": "B"}, registry)

    assert not composed.arguments["pick"].options_when


# ---------------------------------------------------------------------------
# Nothing is copied


def test_a_target_gaining_an_argument_changes_what_the_facade_publishes():
    """The whole reason this is composed and not written.

    A packaged dispatcher would restate its targets' arguments in its own run()
    signature, and the day one of them gains an argument the dispatcher stops
    forwarding it -- silently, which is the failure this repository keeps
    finding.
    """
    registry = _registry(A={"one": ArgSpec(type="path")}, B={"one": ArgSpec(type="path")})
    before = facade.compose("F", {"a": "A", "b": "B"}, registry)
    assert "added_later" not in before.arguments

    registry["A"].arguments["added_later"] = ArgSpec(type="int", required=False)
    after = facade.compose("F", {"a": "A", "b": "B"}, registry)

    assert after.arguments["added_later"].visible_when == {facade.MODE_ARGUMENT: ["a"]}


def test_composing_does_not_mutate_the_targets():
    """`visible_when` is written onto the facade's copy, never the tool's: the
    target is still served directly and must not grow a condition on a `mode`
    argument it does not have."""
    registry = _registry(A={"only_a": ArgSpec(type="path")}, B={"other": ArgSpec(type="path")})
    facade.compose("F", {"a": "A", "b": "B"}, registry)

    assert not registry["A"].arguments["only_a"].visible_when


# ---------------------------------------------------------------------------
# What it refuses


def test_one_name_meaning_two_things_cannot_be_published():
    """There is one entry per name, so a union that disagrees is unpublishable.
    Refusing names the two tools rather than silently taking the last one."""
    registry = _registry(
        A={"input": ArgSpec(type="path")},
        B={"input": ArgSpec(type="int")},
    )
    with pytest.raises(facade.FacadeError, match="One name has to mean one thing"):
        facade.compose("F", {"a": "A", "b": "B"}, registry)


def test_a_target_this_server_does_not_serve_is_refused():
    with pytest.raises(facade.FacadeError, match="does not serve"):
        facade.compose("F", {"a": "Absent"}, _registry(A={}))


def test_a_facade_over_absent_targets_is_a_warning_not_a_failure(caplog):
    """A deployment that does not serve AREG_CBCT genuinely cannot offer AREG.

    Not fatal, and not a failed tool either: FAILED_TOOLS is for a tool that is
    on disk and would not load, and this one never existed here.
    """
    built = facade.build_facades(
        _registry(Real={}),
        configured=("F",),
        for_tool=lambda name: type("_", (), {"dispatch": {"a": "Absent"}})(),
    )
    assert built == {}


def test_a_facade_named_after_a_served_tool_is_fatal():
    """deployment.toml contradicting the registry: both would answer the same
    /run path with no way to say which one did."""
    with pytest.raises(facade.FacadeError, match="both served as a tool"):
        facade.build_facades(
            _registry(Taken={}, Other={}),
            configured=("Taken",),
            for_tool=lambda name: type("_", (), {"dispatch": {"a": "Other"}})(),
        )


def test_an_unknown_mode_names_the_ones_that_exist():
    composed = facade.compose("F", {"a": "A"}, _registry(A={}))
    with pytest.raises(Exception, match="is not a mode of F"):
        composed.target_for("nope")
