"""A tool that is a choice between other tools, composed rather than written.

`ALI` became `ALI_CBCT` and `ALI_IOS`; `AREG` became three. The split was about
dependency isolation -- two engines that cannot share a torch version cannot
share a virtualenv -- and it was right. What it cost is the caller who used to
send data and let the tool work out which engine it needed, and every Slicer
module that named the old tool: those answered "Unknown tool 'AREG'", which is
what a typo answers.

A facade puts that back without giving any of it up. It declares nothing of its
own:

    [tools.AREG]
    dispatch = { CBCT = "AREG_CBCT", IOS = "AREG_IOS", "CBCT to IOS" = "AREG_IOSCBCT" }

and its published schema is COMPOSED from those three at startup -- a `mode`
choice in front, then every argument of every target, each shown only for the
modes that have it. Running it reads `mode`, drops it, and runs that target.

**Composed, never copied**, and that is the whole point. A packaged dispatcher
would have to restate its targets' arguments in its own `run()` signature, and
the day `AREG_IOS` gains an argument the dispatcher stops forwarding it, in
silence. Here there is nothing to keep in sync: change a target, restart, and
the facade publishes the change. The only thing it can get wrong is naming a
tool that is not served, which is a startup error.

Nothing here is dental. The server learns "this name is a choice between those
names", which is deployment configuration in exactly the sense `data_dir` is.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

from base import ArgSpec, Tool, ToolArgumentError

logger = logging.getLogger("inference_server")

# The argument the facade adds, and the only one it owns.
MODE_ARGUMENT = "mode"


class FacadeError(Exception):
    """Raised at startup when a facade cannot be composed."""


class FacadeTool(Tool):
    """A `Tool` whose arguments are its targets', and whose run is theirs.

    `invoke` is never reached: `main.py` resolves the mode and dispatches to the
    target, so the run that happens is an ordinary run of an ordinary tool with
    an ordinary timeout, GPU slot and job directory. Keeping it that way is what
    stops a facade from being a second execution path to maintain.
    """

    def __init__(self, name: str, targets: dict, arguments: dict,
                 output_kind: str, description: str = ""):
        self.name = name
        self.targets = dict(targets)
        self.arguments = arguments
        self.output_kind = output_kind
        self.description = description
        # A facade runs nothing itself, so it has no folder and no interpreter.
        self.folder = None

    def target_for(self, mode: str) -> str:
        try:
            return self.targets[mode]
        except KeyError:
            raise ToolArgumentError(
                "'{}' is not a mode of {}. Choose one of: {}.".format(
                    mode, self.name, ", ".join(sorted(self.targets))
                )
            )

    def run(self, **kwargs):  # pragma: no cover - never called, see class docstring
        raise RuntimeError(
            f"{self.name} is a facade; main.py dispatches it to one of "
            f"{sorted(self.targets)} instead of running it."
        )


def _identity(spec: ArgSpec) -> tuple:
    """What has to agree for one argument name to mean one thing.

    Deliberately not the whole ArgSpec: `default`, `description` and the
    presentation keys may differ between two tools without the caller being
    asked for anything different. Type, what it accepts and whether it is
    required are what a client renders and validates against.
    """
    return (spec.type, tuple(spec.accepts or ()), spec.required)


def compose(name: str, targets: dict, registry: dict) -> FacadeTool:
    """Build the facade's published schema from the tools it dispatches to.

    Raises FacadeError on anything that would make the published schema a lie:
    a target this server does not serve, or one argument name meaning two
    different things across modes -- which cannot be published, there being one
    entry per name.
    """
    missing = sorted(target for target in targets.values() if target not in registry)
    if missing:
        raise FacadeError(
            "'{}' dispatches to {}, which this server does not serve. Either "
            "deploy them or fix the names in deployment.toml.".format(
                name, ", ".join(missing))
        )

    if name in registry:
        raise FacadeError(
            f"'{name}' is both served as a tool and declared as a facade. A "
            f"facade needs a name of its own."
        )

    modes = list(targets)
    arguments: dict = {}
    # {argument name: [modes that have it]}, so an argument shared by two modes
    # is shown for both rather than the last one composed.
    seen_in: dict = {}
    identity: dict = {}
    # {argument name: {mode: its choices in that mode}}, for a choice whose
    # options differ -- AREG's `automation` offers three modes for CBCT, two for
    # IOS and three different ones for IOSCBCT.
    choices_by_mode: dict = {}

    for mode, target in targets.items():
        tool = registry[target]
        for argument_name, spec in tool.arguments.items():
            here = _identity(spec)
            if argument_name in identity and identity[argument_name] != here:
                raise FacadeError(
                    "'{}' cannot publish '{}': {} declares it as {} and {} as "
                    "{}. One name has to mean one thing.".format(
                        name, argument_name, targets[seen_in[argument_name][0]],
                        identity[argument_name], target, here)
                )
            identity[argument_name] = here
            seen_in.setdefault(argument_name, []).append(mode)
            if argument_name not in arguments:
                arguments[argument_name] = copy.deepcopy(spec)
            if spec.choices:
                choices_by_mode.setdefault(argument_name, {})[mode] = dict(spec.choices)

    for argument_name, spec in arguments.items():
        appears_in = seen_in[argument_name]
        # Only when it does NOT apply everywhere: a visible_when naming every
        # mode is noise a client has to evaluate on every keystroke.
        if len(appears_in) < len(modes):
            spec.visible_when = dict(spec.visible_when or {})
            spec.visible_when[MODE_ARGUMENT] = list(appears_in)
            # An argument only some modes have cannot be required of the others,
            # and validation happens on the TARGET anyway -- it is the one that
            # knows. Publishing it as required would make the panel refuse to
            # apply in a mode that never wanted it.
            spec.required = False
        per_mode = choices_by_mode.get(argument_name, {})
        if len({tuple(sorted(options)) for options in per_mode.values()}) > 1:
            spec.options_when = {
                MODE_ARGUMENT: {mode: list(options) for mode, options in per_mode.items()}
            }

    mode_spec = ArgSpec(
        type="choice",
        required=True,
        description="Which of this tool's engines to run.",
        choices={mode: index == 0 for index, mode in enumerate(modes)},
        label="Mode",
    )
    # First, because every other field's visibility depends on it.
    composed = {MODE_ARGUMENT: mode_spec}
    composed.update(arguments)

    kinds = {registry[target].output_kind for target in targets.values()}
    output_kind = kinds.pop() if len(kinds) == 1 else "files"

    logger.info(
        "Facade '%s' composed over %s: %d argument(s)",
        name, ", ".join(f"{mode}={target}" for mode, target in targets.items()),
        len(arguments),
    )
    return FacadeTool(name, targets, composed, output_kind)


def build_facades(registry: dict, configured, for_tool, on_failure=None) -> dict:
    """Every facade deployment.toml declares, composed against `registry`.

    A facade whose targets are absent is a WARNING and is skipped -- not fatal,
    and not a failed tool either. It was both in earlier versions of this
    function, and neither was right:

    - fatal repeats a question this repository already settled the other way.
      With 15+ tools, one missing must never take the rest down, and a
      deployment that does not serve AREG_CBCT genuinely cannot offer AREG.
    - FAILED_TOOLS is for a tool that is ON DISK and would not load. A facade
      over tools this installation does not carry never existed; recording it
      there puts a configuration entry in a list of broken tools, and it is the
      same shape as an unknown [tools.X] section, which this file already
      answers with a warning.

    A facade whose NAME is already served stays FATAL: that is deployment.toml
    contradicting the registry rather than a missing dependency, and both would
    answer the same /run path with no way to say which one did.
    """
    facades = {}
    for name in configured:
        targets = for_tool(name).dispatch
        if not targets:
            continue
        if name in registry:
            raise FacadeError(
                f"'{name}' is both served as a tool and declared as a facade in "
                f"deployment.toml. A facade needs a name of its own."
            )
        try:
            facades[name] = compose(name, targets, registry)
        except FacadeError as error:
            logger.warning(
                "Facade '%s' is not offered by this deployment: %s", name, error)
    return facades
