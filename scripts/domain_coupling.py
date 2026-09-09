#!/usr/bin/env python3
"""Count application-domain identifiers in the server's own modules.

This server serves dental and craniofacial tools, and knows nothing about
either. That is a claim, and a claim about absence is worth exactly as much as
the measurement behind it -- so this script is the measurement, and the word
list it counts is published here rather than described in prose.

    python3 scripts/domain_coupling.py            # human-readable table
    python3 scripts/domain_coupling.py --json     # one JSON object on stdout
    python3 scripts/domain_coupling.py --max 2    # exit 1 above the budget

Standard library only, same rule as fetch_data.py and server_ctl.py.

## What is counted, and why it is counted that way

A grep over these files reports many hits and means little: the modules are
commented in the vocabulary of the project they were written for, and a
docstring saying "e.g. an AMASSS run" is documentation, not coupling. Coupling
is the server *behaving* differently because a tool is dental.

So the count is taken from the parsed syntax tree, not from the text:

* **Comments never reach the AST.** They are excluded by construction, which
  is the honest reason -- not a filter someone chose.
* **Docstrings are located and excluded explicitly**, module, class and
  function alike.
* What remains is executable code: names, attribute accesses, argument names,
  and string literals that are not docstrings.

A string literal in executable code is included even when it is only an
example inside an error message, because excluding it would need a judgement
this script does not get to make. Two such examples are the entire current
count; `--max` exists so that a third one has to be argued for rather than
merged unnoticed.

Read the number as an upper bound on coupling, never as proof of its absence:
a server could branch on a tool name without ever spelling a dental word.
What this measures is the vocabulary, which is where coupling shows first.
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

# The published word list. Anatomy, modality, and the names of the tools this
# deployment happens to serve. Deliberately broad -- a list narrow enough to
# return zero would be a list chosen to return zero.
WORDS = [
    # anatomy
    "dental", "dentistry", "tooth", "teeth", "mandible", "maxilla",
    "craniofacial", "cranial", "airway", "cervical", "vertebra", "vertebrae",
    "condyle", "tmj", "skull", "jaw", "occlusal", "orthodont", "dentition",
    # imaging modality and domain objects
    "cbct", "ios", "intraoral", "landmark", "cephalometric",
    # the served catalogue, by name
    "amasss", "ali_cbct", "ali_ios", "areg", "aso", "crown_seg",
    "surg_mov", "batch_dental", "flexreg", "docshapeaxi", "medx", "mri2cbct",
]
PATTERN = re.compile(r"\b(" + "|".join(WORDS) + r")\b", re.I)

# The server's own modules. `tools/` is excluded on purpose: a tool is allowed
# -- required, even -- to be about its domain. So are the fixtures under
# docker/, which exist to be tools. Tests are excluded because a test names
# what it tests.
CORE = [
    "server/main.py",
    "server/base.py",
    "server/config.py",
    "server/data_store.py",
    "server/file_utils.py",
    "server/transfer.py",
    "server/registry",
    "server/execution",
    "server/wire",
]


def modules(root):
    """Every core .py file, tests excluded, in a stable order."""
    for spec in CORE:
        path = root / spec
        if path.is_dir():
            yield from sorted(f for f in path.rglob("*.py")
                              if "test" not in f.name)
        elif path.is_file():
            yield path


def docstrings_of(tree):
    """Every docstring in the module, as the exact string objects the AST holds.

    Compared by identity below rather than by value: two functions may share a
    docstring, and a string that happens to equal one is still code.
    """
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                found.add(id(node.body[0].value))
    return found


def count(path):
    """Domain-word occurrences in one module's executable code.

    Returns (hits, matched_words). A syntax error is reported rather than
    swallowed: a core module that does not parse is a bigger problem than
    whatever this script was asked to measure.
    """
    source = path.read_text(encoding="utf8", errors="replace")
    tree = ast.parse(source, filename=str(path))
    docs = docstrings_of(tree)

    hits, words = 0, set()

    def record(text):
        nonlocal hits
        found = PATTERN.findall(text)
        hits += len(found)
        words.update(w.lower() for w in found)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            record(node.id)
        elif isinstance(node, ast.Attribute):
            record(node.attr)
        elif isinstance(node, ast.arg):
            record(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            record(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docs:
                record(node.value)

    return hits, sorted(words)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true",
                        help="one JSON object on stdout, nothing else")
    parser.add_argument("--max", type=int, metavar="N",
                        help="exit 1 if the total exceeds N (for CI)")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    per_module, total = [], 0
    for path in modules(root):
        hits, words = count(path)
        total += hits
        per_module.append({
            "module": str(path.relative_to(root)),
            "hits": hits,
            "words": words,
        })

    if args.json:
        json.dump({"total": total, "word_list": WORDS,
                   "modules": per_module}, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for entry in per_module:
            note = "  <- " + ", ".join(entry["words"]) if entry["words"] else ""
            print(f"{entry['hits']:4d}  {entry['module']}{note}")
        print(f"\n{total} domain-word occurrence(s) in executable code, "
              f"across {len(per_module)} core modules.")
        print("Comments and docstrings are excluded; see this file's docstring.")

    if args.max is not None and total > args.max:
        # stderr, so --json still produces exactly one object on stdout.
        print(f"domain coupling: {total} occurrence(s), budget is {args.max}. "
              f"A new one is a design decision -- argue for it or remove it.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
