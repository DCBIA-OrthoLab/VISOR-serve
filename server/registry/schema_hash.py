#!/usr/bin/env python3
"""The `source_hash` in a tool's `.schema.json`, and how it is computed.

The schema declares the signature `run()` is validated against; the hash says
which source tree that signature was read from. If the two drift, the server
is checking requests against a function that no longer exists -- an argument
the tool has since renamed passes validation and then fails inside the tool,
and one the tool has since added can never be sent at all.

**This file is the reference implementation, and it is executable**:

    python server/schema_hash.py <path to a tool's src/>

so the generator in `sadt-tools` can be checked against it byte for byte
rather than against a description of it. Both sides have to agree exactly --
if they do not, every tool refuses to start.

**This is a port of `scripts/describe.py::source_hash` in the sadt-tools
repository, and it has to stay byte-for-byte equivalent to it.** That side
generates the hash; this side only checks it. The two disagreeing by one
separator means every tool looks stale and nothing loads -- which is exactly
what happened the first time, before the two were compared on a real tool.

The rule: sha256 over `<relative posix path>\\0<the file's bytes>\\0`, for
every file under src/ in sorted order, with compiled-Python artifacts left out.

- **Relative POSIX paths**, so the hash does not depend on where the tool
  happens to be checked out, or on which OS wrote it.
- **Sorted as PATHS, not as strings.** `sorted(src.rglob("*"))` orders by path
  parts, so `a/b.py` sorts before `a.py`; sorting the relative strings instead
  puts them the other way round, because "." is 0x2E and "/" is 0x2F. Same
  files, different digest.
- **The path is hashed as well as the content**, so renaming a file changes
  the hash. A tool that moved run() into another module has a different
  signature even when every byte is otherwise the same.
- **`__pycache__` and `*.pyc`/`*.pyo` excluded.** They are generated, they
  differ between interpreter versions, and importing the tool once would
  otherwise change its own hash.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import sys

# Everything a Python interpreter generates on its own. Kept deliberately
# short: every entry here is a rule the other side has to implement too.
IGNORED_DIRECTORIES = ("__pycache__",)
IGNORED_SUFFIXES = (".pyc", ".pyo")

_READ_CHUNK = 1024 * 1024


def source_files(src_dir: str) -> list:
    """Every file that counts, in the order the digest consumes them.

    `pathlib.Path` ordering, not string ordering: see the note in the module
    docstring. Kept in this shape because the generating side is written this
    way, and the two have to agree.
    """
    root = pathlib.Path(src_dir)
    return [
        path
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
        if "__pycache__" not in path.parts and path.suffix not in IGNORED_SUFFIXES
    ]


def hash_source_tree(src_dir: str) -> str:
    """The `source_hash` of a tool's src/ directory, as a hex string."""
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"No such source directory: {src_dir}")

    root = pathlib.Path(src_dir)
    digest = hashlib.sha256()
    for path in source_files(src_dir):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        # Read in chunks rather than whole: the digest is identical either way,
        # and a tool is free to keep a large file under src/.
        with open(path, "rb") as handle:
            while chunk := handle.read(_READ_CHUNK):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(f"usage: {os.path.basename(__file__)} <tool src/ directory>", file=sys.stderr)
        return 2
    print(hash_source_tree(argv[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
