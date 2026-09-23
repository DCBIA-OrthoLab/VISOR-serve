"""What a step said it did, read back.

A tool writes a run report beside its output: which cases it handled, what it
was given for each, and what it produced. This reads that back, so a resumed
run can be narrowed to the cases a clinician marked instead of redoing the
whole cohort.

**Why the tool is asked rather than the file names.** Working out which
results belong to one patient by stripping markers off their names means the
server holding a table of what `_Or`, `_lm_Pred` and `_MERGED` mean -- that is
knowing the dental tools, which this server does not and must not. The report
does not deduce it. It states it, in the tool's own words, as the tool wrote
the files.

**The vocabulary is a contract**, published by every tool of the catalogue:

    {"cases": {"<id>": {"input": "<what it was given>",
                        "produced": ["<every file written for it>"]}}}

**Nothing here is load-bearing for correctness.** A report that is missing,
unreadable, or written in an older vocabulary yields nothing, and a caller
that learns nothing narrows nothing -- so the replay is the whole cohort,
which is slower and never wrong. Losing work to a report we could not parse
would be the worse failure by far, so every path here fails towards doing
more work rather than less.
"""

import json
import logging
import os

logger = logging.getLogger("inference_server")

# The name most tools use, and the pattern the rest follow
# (`ASO_report.json`, `CLIC_report.json`, ...). Tried in this order so a tool
# writing both is read by the general one.
_PREFERRED = "run_report.json"
_SUFFIX = "_report.json"

CASES = "cases"
INPUT = "input"
PRODUCED = "produced"


def find(directory: str):
    """The report a step wrote in `directory`, or None.

    Only the top level: a report belongs beside the results it describes, and
    a nested one is a CALLEE's, which the caller's report does not speak for.
    """
    if not directory or not os.path.isdir(directory):
        return None
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return None
    if _PREFERRED in names:
        return os.path.join(directory, _PREFERRED)
    for name in names:
        if name.endswith(_SUFFIX):
            return os.path.join(directory, name)
    return None


def read(directory: str) -> dict:
    """The report as a dict, or `{}` when there is nothing readable.

    Unreadable is not an error here. A tool that wrote no report, or wrote
    one this cannot parse, simply says nothing -- and a caller that learns
    nothing does more work rather than the wrong work.
    """
    path = find(directory)
    if path is None:
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s: %s", os.path.basename(path), exc)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def cases(report: dict) -> dict:
    """`{case id: entry}`, or `{}` for a report that does not say.

    A report from before the vocabulary was agreed says nothing here rather
    than something wrong: its `scans`/`patients` key means the same thing,
    but reading it as if it were `cases` would be assuming a shape nobody
    promised.
    """
    found = (report or {}).get(CASES)
    if not isinstance(found, dict):
        return {}
    return {name: entry for name, entry in found.items()
            if isinstance(entry, dict)}


def inputs_for(report: dict, wanted) -> list:
    """What those cases were GIVEN, in the report's own order.

    This is what narrows a replay: the run is fed these instead of the whole
    cohort, so the cases nobody marked keep the results they already have.
    """
    keep = set(wanted or ())
    return [entry[INPUT] for name, entry in cases(report).items()
            if name in keep and isinstance(entry.get(INPUT), str)]


def produced_by(report: dict, wanted) -> list:
    """Every file those cases produced, in the report's own order.

    Duplicates are dropped -- a tool may name one file under two cases when a
    pair was registered together -- and the order of first mention is kept,
    because a reader looking at a log wants the order things happened in.
    """
    keep = set(wanted or ())
    found = []
    for name, entry in cases(report).items():
        if name not in keep:
            continue
        for path in entry.get(PRODUCED) or ():
            if isinstance(path, str) and path not in found:
                found.append(path)
    return found


def case_of(report: dict, path: str) -> str:
    """Which case a file belongs to, or "" if the report does not say.

    Matched on the file NAME rather than the whole path: a report records
    what a tool wrote in its own output directory, and the same file reaches
    this server through a job directory, a staging directory and an archive,
    under three different absolute paths and one unchanging name.
    """
    target = os.path.basename(path or "")
    if not target:
        return ""
    for name, entry in cases(report).items():
        for produced in entry.get(PRODUCED) or ():
            if isinstance(produced, str) and os.path.basename(produced) == target:
                return name
        if os.path.basename(entry.get(INPUT) or "") == target:
            return name
    return ""


def cases_of(report: dict, paths) -> list:
    """The distinct cases a set of files belongs to, in first-seen order.

    What a resume is narrowed BY: the client sends back the files a reader
    corrected, and these are the cases those files are about.
    """
    found = []
    for path in paths or ():
        name = case_of(report, path)
        if name and name not in found:
            found.append(name)
    return found
