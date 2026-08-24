#!/usr/bin/env python3
"""Is the deduplication between tool virtualenvs actually working?

    docker run --rm <image> /opt/sadt/.venv/bin/python /opt/sadt/verify_dedup.py

uv installs a package by HARDLINKING it out of its cache. When that works, the
same wheel installed into eleven virtualenvs occupies the disk once. When it
silently falls back to copying -- UV_CACHE_DIR on another filesystem, or the
syncs split across image layers so overlayfs copies the files up -- everything
still works, every test still passes, and the image is 37 GB larger.

Nothing observable fails, which is exactly why this is a script and not a
comment.

**What counts as the same file.** Two virtualenvs holding numpy 1.26 and numpy
2.5 have hundreds of identically-named files, and a few dozen with identical
bytes (test fixtures that never changed). Those come from different wheels and
uv could not share them if it wanted to -- its cache is keyed per wheel. So the
comparison is restricted to files belonging to the same distribution AT THE
SAME VERSION, which each venv's `.dist-info/RECORD` states precisely. Within
such a group the bytes are identical by construction, so a second inode means
a copy, with no hashing needed to prove it.

Exits non-zero if any such file exists more than once. Standard library only,
so it runs in the API venv, in a tool's venv, or on the host.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

DEFAULT_TOOLS_DIR = "/tools"

# The plan's own check, kept because it is the one to run on a real image:
#   stat -c '%h %n' /tools/*/.venv/lib/python*/site-packages/torch/lib/libtorch_cuda.so
# A link count of 1 everywhere means deduplication is broken.
HEADLINE_PACKAGE = "torch"

# Files an installer WRITES rather than unpacks, which therefore cannot be
# hardlinked out of a shared cache and are not evidence of anything:
#
#   RECORD/INSTALLER/REQUESTED/direct_url.json  written per installation
#   anything reached through "../"              a console script in the venv's
#                                               own bin/, whose shebang names
#                                               that venv's interpreter
GENERATED_NAMES = ("RECORD", "INSTALLER", "REQUESTED", "direct_url.json")


def _is_generated(relative_path: str) -> bool:
    return relative_path.startswith("..") or os.path.basename(relative_path) in GENERATED_NAMES


def site_packages_dirs(tools_dir: str) -> dict:
    """{tool name: its site-packages directory}, for every installed tool."""
    found = {}
    for tool in sorted(os.listdir(tools_dir)) if os.path.isdir(tools_dir) else []:
        lib = os.path.join(tools_dir, tool, ".venv", "lib")
        if not os.path.isdir(lib):
            continue
        for python_dir in sorted(os.listdir(lib)):
            candidate = os.path.join(lib, python_dir, "site-packages")
            if os.path.isdir(candidate):
                found[tool] = candidate
    return found


def installed_files(site_packages: str) -> dict:
    """{path relative to site-packages: "<distribution> <version>"}.

    Read from each .dist-info/RECORD, which is the only authority on which
    wheel a file came out of. A file no RECORD claims (a .pyc, something
    written after installation) is left out: it cannot be attributed, so it
    cannot be compared.
    """
    owners = {}
    for entry in os.listdir(site_packages):
        if not entry.endswith(".dist-info"):
            continue
        distribution = entry[: -len(".dist-info")]
        record = os.path.join(site_packages, entry, "RECORD")
        try:
            with open(record, encoding="utf-8", newline="") as handle:
                for row in csv.reader(handle):
                    if row:
                        owners[row[0]] = distribution
        except OSError:
            continue
    return owners


def index(site_packages: dict) -> dict:
    """{(distribution, relative path): [(tool, inode, size), ...]}."""
    files = defaultdict(list)
    for tool, root in site_packages.items():
        owners = installed_files(root)
        for relative_path, distribution in owners.items():
            if _is_generated(relative_path):
                continue
            absolute = os.path.join(root, relative_path)
            try:
                stat = os.stat(absolute, follow_symlinks=False)
            except OSError:
                continue
            if not os.path.isfile(absolute) or os.path.islink(absolute):
                continue
            files[(distribution, relative_path)].append((tool, stat.st_ino, stat.st_size))
    return files


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024


def _headline(site_packages: dict) -> None:
    """Link counts for the package that decides whether an image is 26 GB or
    63 GB, in the shape the plan asks for."""
    installed = [
        (tool, os.path.join(root, HEADLINE_PACKAGE))
        for tool, root in site_packages.items()
        if os.path.isdir(os.path.join(root, HEADLINE_PACKAGE))
    ]
    if not installed:
        return
    print(f"\n{HEADLINE_PACKAGE}, in {len(installed)} tool(s) -- link count is what matters:")
    for tool, package_dir in installed:
        for dir_path, _, file_names in os.walk(package_dir):
            shared_objects = sorted(name for name in file_names if ".so" in name)
            if not shared_objects:
                continue
            stat = os.stat(os.path.join(dir_path, shared_objects[0]))
            print(
                f"  links={stat.st_nlink:<4} inode={stat.st_ino:<12} {tool:<20} "
                f"{os.path.relpath(os.path.join(dir_path, shared_objects[0]), package_dir)}"
            )
            break


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-dir", default=DEFAULT_TOOLS_DIR)
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=0,
        help="Only report duplicated files at least this large (all still count).",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="exit 0 when fewer than two tools are installed",
    )
    arguments = parser.parse_args(argv)

    site_packages = site_packages_dirs(arguments.tools_dir)
    if len(site_packages) < 2:
        print(
            f"Fewer than two tools installed under {arguments.tools_dir} "
            f"({', '.join(site_packages) or 'none'}): nothing to share, nothing to check."
        )
        # Not a pass. An image built with no virtualenvs reaches this line, and
        # returning 0 made CI green on exactly the build this script exists to
        # reject: the fixtures lost their [tool.sadt] marker, uv sync skipped
        # all three, and nothing said so for twelve days.
        return 0 if arguments.allow_empty else 1

    print(f"Tools: {', '.join(site_packages)}")

    shared = 0
    saved = 0
    duplicated = []
    for (distribution, relative_path), entries in index(site_packages).items():
        if len(entries) < 2:
            continue
        shared += 1
        inodes = {inode for _, inode, _ in entries}
        size = entries[0][2]
        if len(inodes) == 1:
            saved += size * (len(entries) - 1)
        elif size >= arguments.min_bytes:
            duplicated.append((size, distribution, relative_path, entries))

    print(f"\nFiles of the same package and version in several virtualenvs: {shared}")
    print(f"Disk NOT spent, because they are one inode                   : {_human(saved)}")

    _headline(site_packages)

    if not duplicated:
        print("\nOK: every shared file is a single inode. Deduplication is working.")
        return 0

    duplicated.sort(reverse=True)
    wasted = sum(size * (len(entries) - 1) for size, _, _, entries in duplicated)
    print(
        f"\nBROKEN: {len(duplicated)} file(s) of the same package and version exist as "
        f"separate copies, wasting {_human(wasted)}."
    )
    print(
        "Causes, in order of likelihood: UV_CACHE_DIR on a different filesystem from the "
        "virtualenvs (a BuildKit cache mount pointed straight at it), or the `uv sync` calls "
        "split across image layers."
    )
    for size, distribution, relative_path, entries in duplicated[:10]:
        print(
            f"  {_human(size):>10}  {distribution}/{relative_path} "
            f"({', '.join(tool for tool, _, _ in entries)})"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
