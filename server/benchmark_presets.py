"""Pre-built load patterns, and the arguments they run with.

`benchmarks/` is the instrument of record: five campaigns, a config file, a
protocol client that speaks chunked transfer and ranged GETs, and a paper to
answer to. Nothing here replaces it. This is the other thing somebody wants on
a server -- **press a button and see what happens** -- and the difference in
purpose is what makes it a different mechanism rather than a second front end
onto that one.

Two decisions carry the design.

**A preset names tools and a shape, never arguments.** Every argument is
resolved at launch from what THIS deployment hosts: `GET /tools` says what a
tool requires, `GET /tools/{tool}/data` says which of those it can satisfy from
`DATA/`, and the two together fill the form exactly as a clinician's panel
fills it. So a preset is portable -- it says "run each tool once" and means
that on any deployment -- and a tool whose test files are absent is reported as
unrunnable rather than failing halfway with a 422.

**Nothing is uploaded.** Every input a preset uses is `server_selectable`, so it
travels as a file NAME and the bytes never move. That makes a battery cheap
enough to press a button for, and it means these numbers measure dispatch,
admission and the tools -- NOT the transfer path. Which is exactly the division
of labour with `benchmarks/`: B2 is the instrument for the wire, and a preset
that quietly re-measured it worse would be the worst of both.

The presets themselves are deliberately few and obvious. A battery nobody can
predict the cost of is a battery nobody presses.
"""

from __future__ import annotations

import copy
from typing import Optional

# How a battery is shaped. `parallel` fires its runs at once and is what
# exercises admission; `sequential` fires them one after another and is what
# gives a clean per-tool duration with nothing else on the card.
PARALLEL = "parallel"
SEQUENTIAL = "sequential"

# What a preset lets a reader change before pressing it. `ONE` swaps the single
# tool it runs; `MANY` narrows the set. `None` means the preset IS its tool --
# the smoke test is Test_Tool by definition, and offering to run it against
# AMASSS would make it a different question with the same name.
ONE = "one"
MANY = "many"

# The ceiling on any single battery, whatever a preset or a caller asks for.
# Not a tuning knob: a page with a button that can start forty CBCT
# segmentations is a page that can fill a disk and hold a card for a day, and
# the cap is the difference between a tool somebody uses and a foot-gun.
MAX_RUNS = 24
MAX_CONCURRENCY = 8


# What a required scalar is given when its tool names no default. Deliberately
# dull: a benchmark's job is to make the tool run, and a value that looked like
# real input would be the one thing on this page that was not obviously fake.
_NEUTRAL = {"str": "benchmark", "int": 1, "float": 1.0, "bool": False}


class PresetError(ValueError):
    """A preset that cannot be built here, with the reason for a caller."""


# --- the catalogue ---------------------------------------------------------
#
# `tools` is either an explicit list or the string "all", which means every
# tool this server can resolve arguments for -- so the catalogue does not have
# to be edited when a tool is added.

PRESETS = {
    "smoke": {
        "label": "Smoke test",
        "about": "Test_Tool once. Proves the whole round trip with no GPU and "
                 "no data -- the first thing to press when something looks wrong.",
        "tools": ["Test_Tool"],
        "shape": SEQUENTIAL,
        "repeats": 1,
        "choose": None,
    },
    "each-tool-solo": {
        "label": "Every tool, one at a time",
        "about": "One run of each tool this deployment can resolve, in "
                 "sequence. Nothing competes, so each duration is the tool's "
                 "own -- this is the baseline every other preset is read against.",
        "tools": "all",
        "shape": SEQUENTIAL,
        "repeats": 1,
        "choose": MANY,
    },
    "six-at-once": {
        "label": "Six of the same tool at once",
        "about": "Six concurrent runs of one tool. What admission does when "
                 "six clients want the same card is the question, and the "
                 "answer is the gap between the slowest run and the fastest.",
        "tools": ["CLIC"],
        "shape": PARALLEL,
        "concurrency": 6,
        "repeats": 6,
        "choose": ONE,
    },
    "mixed-load": {
        "label": "Different tools together",
        "about": "Four different tools at once. Unlike six of one, this asks "
                 "whether a cheap tool can still get through while an "
                 "expensive one holds the card.",
        "tools": ["Test_Tool", "CLIC", "Crown_Seg", "Surg_Mov_Pred"],
        "shape": PARALLEL,
        "concurrency": 4,
        "repeats": 1,
        "choose": MANY,
    },
    "ramp": {
        "label": "One, then two, then four",
        "about": "The same tool at rising concurrency, back to back. The shape "
                 "of the curve says whether this machine is bound by the card "
                 "or by the request path.",
        "tools": ["CLIC"],
        "shape": PARALLEL,
        "ladder": [1, 2, 4],
        "repeats": 1,
        "choose": ONE,
    },
    "chain": {
        "label": "A supervised chain",
        "about": "One ASO run, which calls ALI mid-run through the supervisor. "
                 "The nesting is the point: one client, several tools, and the "
                 "depth visible in the run's own phases.",
        "tools": ["ASO"],
        "shape": SEQUENTIAL,
        "repeats": 1,
        "choose": ONE,
    },
}


# --- resolving a tool's arguments ------------------------------------------

# Which hosted name a given argument should be filled with, where the first one
# alphabetically is the WRONG KIND rather than merely a different cohort.
#
# `pool[0]` is deterministic on purpose -- two batteries of one preset have to
# be comparable -- but determinism is not correctness: a pool can hold several
# kinds and nothing in the schema separates them, since none of these arguments
# declares `extensions`. Measured, every one of these refused in under a second
# with a message naming exactly this:
#
#   AutoCrop3D  "'MG_test_scan.nii.gz' is not a Slicer ROI."
#   ALI_IOS     "'MG_test_scan.nii.gz' is a CBCT volume. This tool places
#                landmarks on intraoral surfaces; run ALI_CBCT on volumes."
#   AREG_IOS    a CBCT cohort handed to the intraoral engine
#   GreedyReg   one folder holding both timepoints, which it refuses by design
#
# A name here is a PREFERENCE, not a requirement: it is used only when the pool
# actually offers it, so a deployment staging different cohorts falls back to
# the first-alphabetically rule and stays runnable.
PREFERRED = {
    "AutoCrop3D": {"roi": "ROI_box"},
    "ALI_IOS": {"input": "T1_01_U_segmented.vtk"},
    "AREG_IOS": {"t1": "IOS_test_scans", "t2": "IOS_test_scans"},
    "GreedyReg": {"t1": "T1", "t2": "T2"},
}


def _pool_for(declaration: dict, hosted: dict) -> list:
    """The hosted names an argument may be filled from.

    A SCOPED argument draws from one subfolder, not from the tool's whole
    catalogue, and the difference is not cosmetic: `AREG_CBCT.t1` is scoped to
    `T1`, whose siblings are `T2` and `IOSCBCT`. Reading the catalogue made
    `pool[0]` the alphabetically first name -- `IOSCBCT` -- which `t1` cannot
    resolve, so all four AREG arms answered 404 in under ten milliseconds and
    were reported as the tools failing.
    """
    kind = declaration.get("server_selectable")
    scope = declaration.get("selectable_scope")
    if kind == "model":
        if scope:
            return list((hosted.get("models_by_scope") or {}).get(scope) or ())
        return list(hosted.get("models") or ())
    if kind == "testfile":
        if scope:
            return list((hosted.get("testfiles_by_scope") or {}).get(scope) or ())
        return list(hosted.get("testfiles") or ())
    return []


def _effective(arguments: dict, params: dict) -> dict:
    """What each argument's value will BE when the run is sent.

    `params` holds only what has to travel; everything else falls back to what
    the schema declares. The distinction matters for exactly the arguments the
    second pass below exists for: a facade's `mode` is a choice whose default
    is already on, so NOTHING is sent for it, and reading `params` alone would
    make every mode-specific input look inapplicable.
    """
    effective = {}
    for name, declaration in arguments.items():
        if not isinstance(declaration, dict):
            continue
        if name in params:
            effective[name] = params[name]
            continue
        choices = declaration.get("choices")
        if isinstance(choices, dict):
            # A `choice` publishes {option: on by default}. What the server
            # will apply is whatever is already on.
            on = [option for option, value in choices.items() if value]
            if on:
                effective[name] = on[0] if len(on) == 1 else on
            continue
        if declaration.get("initial") is not None:
            effective[name] = declaration["initial"]
    return effective


def _applies(declaration: dict, effective: dict) -> bool:
    """Whether an argument's `visible_when` holds for the values chosen.

    The same rule the Slicer panel applies in `formgen.is_visible`, so the
    launcher and the clinician's form agree about which fields are in force:
    every entry must match, a list means any-of, and a controlling argument
    absent from the values counts as NOT matching.
    """
    condition = declaration.get("visible_when")
    if not condition:
        return True
    for other, wanted in condition.items():
        if other not in effective:
            return False
        allowed = wanted if isinstance(wanted, (list, tuple)) else [wanted]
        have = effective[other]
        haves = have if isinstance(have, (list, tuple)) else [have]
        if not any(value in allowed for value in haves):
            return False
    return True


def _preferred(tool: str, argument: str, pool: list) -> Optional[str]:
    """The name `PREFERRED` asks for, if this deployment hosts it."""
    wanted = (PREFERRED.get(tool) or {}).get(argument)
    return wanted if wanted in pool else None


def resolve_arguments(schema: dict, hosted: dict, tool: str = "") -> dict:
    """{params, missing} for one tool, filled from what this server hosts.

    `params` carries only what has to be SENT: a required argument that the
    tool already defaults is left out, so the run exercises the tool's own
    default rather than a value invented here.

    `missing` names every required argument nothing could satisfy. A tool with
    anything in it is not run -- reporting it as unrunnable up front is the
    whole reason this function exists, since the alternative is a battery that
    dies on its third arm with a 422 and no plan.
    """
    arguments = schema.get("arguments") or {}
    params, missing = {}, []
    for name, declaration in sorted(arguments.items()):
        if not isinstance(declaration, dict) or not declaration.get("required"):
            continue
        pool = _pool_for(declaration, hosted)
        chosen = _preferred(tool, name, pool)
        if chosen:
            params[name] = chosen
            continue
        if pool:
            # The first hosted name, deterministically: a battery re-run on the
            # same deployment must be comparable with the one before it, and
            # picking at random would make two runs of one preset incomparable.
            params[name] = pool[0]
            continue
        if declaration.get("server_selectable"):
            missing.append(name)
            continue
        # `initial`, which is what the schema calls a default -- NOT `default`.
        # Reading the wrong key made every required scalar look answered, and a
        # preset then posted a form with the field missing and collected a 422
        # it reported as the tool failing.
        if declaration.get("initial") is not None:
            continue  # the tool answers for it
        choices = declaration.get("choices")
        if choices:
            # A `choice` publishes {option: on by default}, and that flag is a
            # hint about which option a PANEL pre-selects -- not a value the
            # server applies. `initial` is the server-side default, and it was
            # checked above; a required choice that reached here has none, so
            # something has to be sent.
            #
            # Leaving it out because an option looked "already on" is what made
            # a six-client AREG arm collect six `422 Missing required argument
            # 'mode' for 'AREG'` on 2026-09-28. A facade needs its mode BEFORE
            # it can dispatch, so it refuses at a point where nothing has had a
            # chance to fill anything in.
            #
            # `multichoice` keeps the old behaviour: sending one option of a
            # several-of argument would NARROW a selection the tool ticked, and
            # a run of AMASSS's one structure is not a run of its five.
            if isinstance(choices, dict) and declaration.get("type") == "choice":
                on = [option for option, value in choices.items() if value]
                params[name] = on[0] if on else sorted(choices)[0]
            elif isinstance(choices, dict):
                on = [option for option, value in choices.items() if value]
                if not on:
                    params[name] = sorted(choices)[0]
            continue
        kind = declaration.get("type")
        if kind in _NEUTRAL:
            # Required, scalar, and the tool names no default: something HAS to
            # be sent or the run is a 422 before it starts. A neutral value,
            # never anything that reads as data.
            params[name] = _NEUTRAL[kind]
            continue
        missing.append(name)

    # Second pass: the inputs a FACADE needs and cannot declare as required.
    #
    # A facade over several tools publishes ONE required argument -- its mode --
    # and every input as `required: false`, because `t1` only means something
    # for AREG's CBCT modes and `ios` only for its CBCT-to-IOS one. A facade
    # cannot say "required, depending on another argument", so the refusal
    # arrives later, from the tool the facade dispatched to. Measured on
    # 2026-09-28: a six-client AREG arm, six `POST /run/AREG` answered
    # `422 Missing required argument 't1' for tool 'AREG_CBCT'` in 20
    # milliseconds, and the battery reported it as AREG failing.
    #
    # What the facade DOES publish is `visible_when`, which says which mode an
    # input belongs to. So an input that applies to the mode being run and that
    # this deployment hosts is filled, whether or not it is marked required.
    #
    # **Not added to `missing` when nothing can fill it**, and that asymmetry is
    # deliberate: the facade does not publish which of its mode-specific inputs
    # the concrete tool REQUIRES, so `t1` (required by AREG_CBCT) and
    # `t1_masks` (optional to it) are indistinguishable here. Marking either
    # missing would report a runnable tool as unrunnable, which is the worse
    # error -- a battery that refuses to run something that works.
    #
    # Two guards, and each one is a mistake this made before it was narrowed.
    #
    # **Only an input whose `visible_when` explains why it is optional.** An
    # argument that is simply optional stays unsent, which is the rule the pass
    # above exists to keep: a benchmark exercises the tool's own defaults. What
    # `visible_when` adds is a REASON -- `t1` is not optional, it is required
    # for two of three modes -- and that is the only case where sending it is
    # restoring an intent rather than inventing one.
    #
    # **And only for a tool that got no input at all above.** ASO declares an
    # optional `landmarks` folder behind a `visible_when` too, and the first
    # hosted name satisfies it -- but supplying it is what makes ASO use the
    # caller's points INSTEAD of asking the landmark tool for them. Filling it
    # would quietly turn a benchmark of the ASO-to-ALI chain into a benchmark
    # of ASO alone, and the number would look like an improvement. A tool that
    # already has its inputs needs nothing more from here; one that has none
    # cannot run at all.
    if not any(
        isinstance(arguments.get(name), dict)
        and arguments[name].get("server_selectable") == "testfile"
        for name in params
    ):
        effective = _effective(arguments, params)
        for name, declaration in sorted(arguments.items()):
            if name in params or not isinstance(declaration, dict):
                continue
            if declaration.get("server_selectable") != "testfile":
                continue
            if not declaration.get("visible_when"):
                continue
            if not _applies(declaration, effective):
                continue
            pool = _pool_for(declaration, hosted)
            if pool:
                params[name] = pool[0]

    return {"params": params, "missing": missing}


def runnable_tools(schemas: dict, hosted_for) -> dict:
    """{tool: {params, missing}} for every published tool.

    `hosted_for(tool)` returns that tool's `{models, testfiles}`; passed in
    rather than imported so this module touches neither the data store nor the
    filesystem and stays testable with a dict.
    """
    report = {}
    for name, schema in schemas.items():
        try:
            hosted = hosted_for(name) or {}
        except Exception:  # noqa: BLE001 - one unreadable tool is not a failure
            hosted = {}
        report[name] = resolve_arguments(schema, hosted, name)
    return report


# --- turning a preset into a plan ------------------------------------------

def _chosen_tools(preset: dict, resolved: dict, asked=None) -> list:
    if asked:
        wanted = [name for name in asked if name in resolved]
    elif preset["tools"] == "all":
        wanted = sorted(resolved)
    else:
        wanted = [name for name in preset["tools"] if name in resolved]
    return [name for name in wanted if not resolved[name]["missing"]]


def build_plan(preset_id: str, resolved: dict, tools=None,
               concurrency: Optional[int] = None) -> dict:
    """A preset plus this deployment's resolution, as the arms to execute.

    Raises `PresetError` rather than silently shrinking: a battery that ran
    four of the six runs somebody asked for, and said so nowhere, is worse than
    one that refused and named the tool it could not resolve.
    """
    preset = PRESETS.get(preset_id)
    if preset is None:
        raise PresetError(f"No preset called '{preset_id}'.")

    chosen = _chosen_tools(preset, resolved, tools)
    if not chosen:
        blocked = sorted(
            f"{name} (needs {', '.join(resolved[name]['missing'])})"
            for name in resolved if resolved[name]["missing"]
        )
        raise PresetError(
            "Nothing in this preset can run on this deployment. "
            + ("Unresolved: " + "; ".join(blocked[:6]) if blocked else
               "No tool matched.")
        )

    ladder = preset.get("ladder")
    width = concurrency or preset.get("concurrency") or 1
    width = max(1, min(int(width), MAX_CONCURRENCY))
    repeats = max(1, int(preset.get("repeats", 1)))

    arms = []
    if ladder:
        for step in ladder:
            step = max(1, min(int(step), MAX_CONCURRENCY))
            arms.append(_arm(f"{preset_id}-x{step}", chosen, step, step,
                             PARALLEL, resolved))
    elif preset["shape"] == PARALLEL:
        total = max(repeats, len(chosen))
        arms.append(_arm(preset_id, chosen, total, width, PARALLEL, resolved))
    else:
        arms.append(_arm(preset_id, chosen, len(chosen) * repeats, 1,
                         SEQUENTIAL, resolved))

    total_runs = sum(len(arm["runs"]) for arm in arms)
    if total_runs > MAX_RUNS:
        raise PresetError(
            f"That would start {total_runs} runs; this page stops at {MAX_RUNS}. "
            "Narrow the tool list, or run it from `benchmarks/` where a long "
            "campaign belongs."
        )
    return {
        "preset": preset_id,
        "label": preset["label"],
        "about": preset["about"],
        "arms": arms,
        "total_runs": total_runs,
    }


def _arm(name: str, tools: list, count: int, width: int, shape: str,
         resolved: dict) -> dict:
    """`count` runs over `tools`, cycled, at `width` at a time."""
    runs = []
    for index in range(count):
        tool = tools[index % len(tools)]
        runs.append({
            "index": index,
            "tool": tool,
            "params": copy.deepcopy(resolved[tool]["params"]),
        })
    return {"arm": name, "shape": shape, "concurrency": width,
            "tools": sorted(set(tools)), "runs": runs}


def catalogue(resolved: dict) -> list:
    """Every preset, with what it would do HERE, for the launcher to render."""
    listing = []
    ready = sorted(name for name, entry in resolved.items()
                   if not entry["missing"])
    for preset_id, preset in PRESETS.items():
        entry = {"id": preset_id, "label": preset["label"],
                 "about": preset["about"], "shape": preset["shape"],
                 "ladder": preset.get("ladder"),
                 "choose": preset.get("choose"),
                 # What the preset would pick left alone, so a control can open
                 # on the preset's own answer rather than on the first name in
                 # an alphabetical list.
                 "default_tools": ([] if preset["tools"] == "all"
                                   else list(preset["tools"])),
                 "candidates": ready}
        try:
            plan = build_plan(preset_id, resolved)
            entry["runs"] = plan["total_runs"]
            entry["tools"] = sorted({run["tool"]
                                     for arm in plan["arms"] for run in arm["runs"]})
            entry["blocked"] = None
        except PresetError as exc:
            entry["runs"] = 0
            entry["tools"] = []
            entry["blocked"] = str(exc)
        listing.append(entry)
    return listing


# --- a battery somebody composed -------------------------------------------
#
# The presets answer "press and see". A custom battery answers the question an
# operator actually has after a change: THIS tool, with THESE arguments, on
# THIS case, N times -- and, more often than not, against a second
# configuration of the same tool, alternated so the two sides share whatever
# the machine was doing. A single run read as a baseline once overstated a gain
# sevenfold; alternating is what made the measurement honest, and here it is
# one button rather than a script.
#
# What it may send is checked HERE, before anything is spawned, against the
# same projection of the schemas the presets read: an argument the tool does
# not declare, a hosted name this deployment does not host, or a required
# argument left out is a refusal naming it -- not a battery that dies on its
# third run with a 422 it reports as the tool failing.

# Higher than a preset's, because a custom battery is aimed: somebody chose
# every number in it. Still a ceiling, for the reason MAX_RUNS gives.
CUSTOM_MAX_RUNS = 60
CUSTOM_MAX_CONCURRENCY = 16
CUSTOM_MAX_CONFIGS = 4
CUSTOM_MAX_LADDER = 6
CUSTOM_MAX_STAGGER = 600

# The argument types a value is typed into. Anything else is a file input, and
# a file input is filled from what this server hosts -- a test file, a model,
# or a bench entry -- never from text.
_VALUE_TYPES = {"str", "int", "float", "bool", "choice", "multichoice", "list[str]"}
_LABELS = "ABCD"


def _wire(declaration: dict, value):
    """One value as the form field `/run` reads, or a PresetError."""
    kind = declaration.get("type")
    choices = declaration.get("choices") if isinstance(declaration.get("choices"), dict) else None
    if kind == "multichoice":
        picked = value if isinstance(value, list) else [value]
        picked = [str(item) for item in picked]
        unknown = [item for item in picked if choices is not None and item not in choices]
        if unknown:
            raise PresetError(f"'{unknown[0]}' is not one of its options.")
        # An empty selection is sent as nothing: the tool's own default then
        # applies, which is what an empty multichoice means on the wire.
        return ",".join(picked) if picked else None
    if isinstance(value, (list, dict)):
        raise PresetError("takes a single value.")
    if kind == "choice":
        if choices is not None and str(value) not in choices:
            raise PresetError(f"'{value}' is not one of its options.")
        return str(value)
    if kind == "bool":
        if isinstance(value, bool):
            return "true" if value else "false"
        if str(value).lower() in ("true", "false", "1", "0", "yes", "no"):
            return str(value).lower()
        raise PresetError("takes true or false.")
    if kind == "int":
        try:
            return str(int(str(value).strip()))
        except ValueError:
            raise PresetError("takes a whole number.")
    if kind == "float":
        try:
            return repr(float(str(value).strip()))
        except ValueError:
            raise PresetError("takes a number.")
    text = str(value)
    if len(text) > 2000:
        raise PresetError("is too long.")
    return text


def _config(index: int, raw: dict, schemas: dict, hosted_for, bench_for) -> dict:
    """One configuration, checked, as {label, tool, params, bench}."""
    if not isinstance(raw, dict):
        raise PresetError(f"Configuration {index + 1} is not an object.")
    tool = raw.get("tool")
    if tool not in schemas:
        raise PresetError(f"{_LABELS[index]}: no tool called '{tool}' on this server.")
    label = _LABELS[index]
    where = f"{label} ({tool})"
    arguments = schemas[tool].get("arguments") or {}
    hosted = hosted_for(tool) or {}
    bench_names = {entry["name"] for entry in (bench_for(tool) or [])}

    params, bench = {}, {}
    for name, value in (raw.get("params") or {}).items():
        declaration = arguments.get(name)
        if not isinstance(declaration, dict):
            raise PresetError(f"{where}: '{tool}' has no argument '{name}'.")
        if value is None or value == "":
            continue
        if declaration.get("server_selectable"):
            pool = _pool_for(declaration, hosted)
            if str(value) not in pool:
                raise PresetError(f"{where}: '{value}' is not hosted for '{name}'.")
            params[name] = str(value)
            continue
        if declaration.get("type") not in _VALUE_TYPES:
            raise PresetError(f"{where}: '{name}' is a file; pick it from what this server hosts.")
        try:
            wired = _wire(declaration, value)
        except PresetError as exc:
            raise PresetError(f"{where}: '{name}' {exc}")
        if wired is not None:
            params[name] = wired

    for name, entry in (raw.get("bench") or {}).items():
        declaration = arguments.get(name)
        if not isinstance(declaration, dict):
            raise PresetError(f"{where}: '{tool}' has no argument '{name}'.")
        if declaration.get("type") in _VALUE_TYPES:
            raise PresetError(f"{where}: '{name}' is not a file input.")
        if entry not in bench_names:
            raise PresetError(f"{where}: no bench input '{entry}' staged for this tool.")
        if name in params:
            raise PresetError(f"{where}: '{name}' is given twice.")
        bench[name] = entry

    # A required argument nothing fills is a 422 on every run. Said now.
    for name, declaration in sorted(arguments.items()):
        if not isinstance(declaration, dict) or not declaration.get("required"):
            continue
        if name in params or name in bench or declaration.get("initial") is not None:
            continue
        raise PresetError(f"{where}: '{name}' is required.")
    return {"label": label, "tool": tool, "params": params, "bench": bench}


def _int(spec: dict, key: str, default: int, low: int, high: int) -> int:
    try:
        value = int(spec.get(key, default) if spec.get(key) is not None else default)
    except (TypeError, ValueError):
        raise PresetError(f"'{key}' takes a whole number.")
    if not low <= value <= high:
        raise PresetError(f"'{key}' is {value}; it goes from {low} to {high}.")
    return value


def build_custom_plan(spec: dict, schemas: dict, hosted_for, bench_for) -> dict:
    """A composed battery as the arms to execute, or a PresetError saying why not.

    `spec` is what the launcher sends:

        {"label": str, "shape": "sequential" | "parallel",
         "repeats": int, "concurrency": int, "ladder": [int], "stagger": seconds,
         "configs": [{"tool": str, "params": {name: value}, "bench": {name: entry}}]}

    `sequential` runs one at a time, the configurations ALTERNATED -- A, B, A,
    B -- `repeats` times each; with two configurations that is the A/B
    comparison. `parallel` keeps `concurrency` runs in flight for `repeats`
    rounds, the configurations cycled across them, and a `ladder` repeats that
    at each width in turn. `stagger` spaces the starts.

    Bench entries leave here as names only; the caller turns them into paths
    for the battery process, so this module never touches the filesystem.
    """
    if not isinstance(spec, dict):
        raise PresetError("A custom battery is an object.")
    raw_configs = spec.get("configs") or []
    if not isinstance(raw_configs, list) or not raw_configs:
        raise PresetError("Add at least one configuration.")
    if len(raw_configs) > CUSTOM_MAX_CONFIGS:
        raise PresetError(f"At most {CUSTOM_MAX_CONFIGS} configurations in one battery.")
    configs = [_config(i, raw, schemas, hosted_for, bench_for)
               for i, raw in enumerate(raw_configs)]

    shape = spec.get("shape") or SEQUENTIAL
    if shape not in (SEQUENTIAL, PARALLEL):
        raise PresetError(f"Unknown shape '{shape}'.")
    repeats = _int(spec, "repeats", 1, 1, CUSTOM_MAX_RUNS)
    stagger = spec.get("stagger") or 0
    try:
        stagger = float(stagger)
    except (TypeError, ValueError):
        raise PresetError("'stagger' takes a number of seconds.")
    if not 0 <= stagger <= CUSTOM_MAX_STAGGER:
        raise PresetError(f"'stagger' goes from 0 to {CUSTOM_MAX_STAGGER} seconds.")

    label = " ".join(str(spec.get("label") or "").split())[:80]
    if not label:
        label = " vs ".join(dict.fromkeys(c["tool"] for c in configs))

    def runs(count: int) -> list:
        return [{"index": i, "config": configs[i % len(configs)]["label"],
                 "tool": configs[i % len(configs)]["tool"],
                 "params": dict(configs[i % len(configs)]["params"]),
                 "bench": dict(configs[i % len(configs)]["bench"])}
                for i in range(count)]

    def arm(name: str, count: int, width: int, arm_shape: str) -> dict:
        said = (f"{width} at once" if arm_shape == PARALLEL else
                "one at a time" + (", alternated" if len(configs) > 1 else ""))
        return {"arm": name, "label": f"{label} \u00b7 {said}", "shape": arm_shape,
                "concurrency": width, "stagger": stagger,
                "tools": sorted({c["tool"] for c in configs}), "runs": runs(count)}

    arms = []
    if shape == SEQUENTIAL:
        arms.append(arm("custom", repeats * len(configs), 1, SEQUENTIAL))
    else:
        ladder = spec.get("ladder") or []
        if not isinstance(ladder, list):
            raise PresetError("'ladder' is a list of widths.")
        if len(ladder) > CUSTOM_MAX_LADDER:
            raise PresetError(f"At most {CUSTOM_MAX_LADDER} steps on a ladder.")
        widths = []
        for step in ladder or [spec.get("concurrency") or 1]:
            try:
                step = int(step)
            except (TypeError, ValueError):
                raise PresetError("A width is a whole number.")
            if not 1 <= step <= CUSTOM_MAX_CONCURRENCY:
                raise PresetError(f"A width goes from 1 to {CUSTOM_MAX_CONCURRENCY}.")
            widths.append(step)
        for width in widths:
            name = f"custom-x{width}" if len(widths) > 1 else "custom"
            arms.append(arm(name, width * repeats, width, PARALLEL))

    total = sum(len(a["runs"]) for a in arms)
    if total > CUSTOM_MAX_RUNS:
        raise PresetError(f"That would start {total} runs; a custom battery stops at "
                          f"{CUSTOM_MAX_RUNS}.")
    return {
        "preset": "custom",
        "label": label,
        "about": "A battery composed in the launcher.",
        "configs": [{"label": c["label"], "tool": c["tool"],
                     "params": dict(c["params"]), "bench": sorted(c["bench"])}
                    for c in configs],
        "arms": arms,
        "total_runs": total,
    }
