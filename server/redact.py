"""Make a tool's own words safe to show an operator, and to keep.

A tool's failure message or log line is free text, and free text written by
code that handles patient data names patient data sooner or later: `No scan
found in /tmp/job/P05_T1_Or.nii.gz`, a GreedyReg error listing every pair it
skipped by patient key. The operator needs the SENTENCE -- "no scan found",
"3 of 7 landmarks" -- and never the name it was said about.

So everything that can carry a name is replaced by what it was:

- anything holding a path separator      -> `<path>`
- a file name with an imaging/data suffix -> `<file>`
- an e-mail address                       -> `<email>`
- an IPv4 address                         -> `<address>`
- a token mixing letters and two or more digits (`P05`, `C_0001`, `case12`)
                                          -> `<id>`

It over-redacts on purpose -- `float32` becomes `<id>` -- because the cost of
a word too many is a less readable line, and the cost of a word too few is a
patient's identifier in a persisted history. Measurements survive: a number on
its own is not a name.

Applied wherever a tool's text crosses to the operator: the admin console, the
run ledger, and the server's own log line for a failure. NOT applied to what
goes back to the client that started the run -- that is the requester reading
about their own data, under the rule progress messages already follow.
"""

import re

MAX_REDACTED_CHARS = 300

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Any whitespace-delimited token holding a separator, minus a ratio like 3/40
# which is a position in a batch and the most useful thing a line can say.
_PATH = re.compile(r"""[^\s'"()\[\]{}<>,;]*[/\\][^\s'"()\[\]{}<>,;]*""")
_RATIO = re.compile(r"\d+/\d+[:.,;]?")
_FILE = re.compile(
    r"[\w.+-]+\.(?:nii|gz|nrrd|nhdr|mha|mhd|dcm|dicom|vtk|vtp|stl|obj|ply|off|"
    r"json|csv|tsv|xlsx|xls|txt|pkl|pickle|pth|pt|ckpt|npy|npz|zip|tar|tgz|"
    r"png|jpe?g|tiff?|bmp|h5|hdf5|mrk|fcsv|mrml|seg|xml|yaml|yml|log)\b",
    re.IGNORECASE,
)
_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_IDENTIFIER = re.compile(
    r"\b(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*\d[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{3,}\b"
)
# A number with its unit -- `4.21s`, `512MB`, `0.5mm`, `16GiB` -- is a
# measurement, the thing a log line is most often FOR. Kept whole; the
# identifier rule above would otherwise read `35s` as a patient code.
_MEASURE = re.compile(
    r"\d+(?:\.\d+)?(?:ms|s|min|h|mm|cm|m|[KMGT]i?B|B|px|vox|%|x)\b|\d+(?:\.\d+)?(?=s\b)"
)


def _identifier(match) -> str:
    text = match.group(0)
    return text if _MEASURE.fullmatch(text) else "<id>"


def _path_or_ratio(match) -> str:
    text = match.group(0)
    return text if _RATIO.fullmatch(text) else "<path>"


def scrub(text) -> str:
    """`text` with every name in it replaced by its kind, on one line, bounded."""
    if text is None:
        return ""
    value = str(text).replace("\r", " ").replace("\n", " ")
    value = _EMAIL.sub("<email>", value)
    value = _PATH.sub(_path_or_ratio, value)
    value = _FILE.sub("<file>", value)
    value = _IPV4.sub("<address>", value)
    value = _IDENTIFIER.sub(_identifier, value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:MAX_REDACTED_CHARS]
