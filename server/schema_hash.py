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

The rule, in one sentence: sha256 over one line per file, `<relative path>\\0
<sha256 of its bytes>\\n`, for every file under src/ in sorted order, with
compiled-Python artifacts left out.

Every part of that is chosen to be reproducible somewhere else:

- **Relative POSIX paths**, so the hash does not depend on where the tool
  happens to be checked out, or on which OS wrote it.
- **Sorted**, so it does not depend on the order the filesystem hands entries
  back -- which differs between machines for the same tree.
- **The path is hashed as well as the content**, so renaming a file changes
  the hash. A tool that moved run() into another module has a different
  signature even when every byte is otherwise the same.
- **Per-file digests rather than concatenated content**, so no file's bytes
  can be read as another's (`a.py` + `b.py` and one file holding both).
- **`__pycache__` and `*.pyc`/`*.pyo` excluded.** They are generated, they
  differ between interpreter versions, and importing the tool once would
  otherwise change its own hash.
"""

from __future__ import annotations

import hashlib
import os
import sys

# Everything a Python interpreter generates on its own. Kept deliberately
# short: every entry here is a rule the other side has to implement too.
IGNORED_DIRECTORIES = ("__pycache__",)
IGNORED_SUFFIXES = (".pyc", ".pyo")

_READ_CHUNK = 1024 * 1024


def _file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_READ_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(src_dir: str) -> list:
    """Every file that counts, as relative POSIX paths, sorted."""
    found = []
    for dir_path, dir_names, file_names in os.walk(src_dir):
        # Pruned in place, which is what stops os.walk descending into them.
        dir_names[:] = sorted(name for name in dir_names if name not in IGNORED_DIRECTORIES)
        for file_name in file_names:
            if file_name.endswith(IGNORED_SUFFIXES):
                continue
            absolute = os.path.join(dir_path, file_name)
            found.append(os.path.relpath(absolute, src_dir).replace(os.sep, "/"))
    return sorted(found)


def hash_source_tree(src_dir: str) -> str:
    """The `source_hash` of a tool's src/ directory, as a hex string."""
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"No such source directory: {src_dir}")

    digest = hashlib.sha256()
    for relative_path in source_files(src_dir):
        absolute = os.path.join(src_dir, relative_path)
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_digest(absolute).encode("ascii"))
        digest.update(b"\n")
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
