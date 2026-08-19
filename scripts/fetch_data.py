#!/usr/bin/env python3
"""Populate DATA/<tool>/{models,testfiles}/ from data-manifest.yml.

This is the engine behind scripts/setup-models.sh and scripts/setup-testfiles.sh;
it is equally usable on its own:

    python3 scripts/fetch_data.py --kind models
    python3 scripts/fetch_data.py --kind testfiles --tool AMASSS
    python3 scripts/fetch_data.py --list

The layout it writes is the one server/data_store.py reads, so a file landing
here shows up in GET /tools/<tool>/data with no further step.

Standard library only, on purpose: this runs on a bare GPU host before any
requirements.txt is installed, so it cannot import yaml, requests, or tqdm.
The manifest parser below therefore covers the small YAML subset the manifest
actually uses (see _parse_manifest), not YAML at large.
"""

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

KINDS = ("models", "testfiles")

_CHUNK = 1024 * 1024
# GitHub answers 403 to a request with no User-Agent.
_HEADERS = {"User-Agent": "slicer-remote-tool-server-fetch/1.0"}

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MANIFEST = os.path.join(_SCRIPT_DIR, "data-manifest.yml")


class ManifestError(Exception):
    """The manifest is malformed, or names something that cannot be fetched."""


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _parse_scalar(text: str):
    """YAML scalars as the manifest uses them: quoted or bare, plus booleans."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    lowered = text.lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    return text


def _split_key(line: str):
    """Split "key: value" on the FIRST ": " only.

    Splitting on the first colon alone would cut "https://..." in half; a colon
    followed by a space is what actually separates a key from its value here.
    """
    if line.endswith(":"):
        return line[:-1].strip(), None
    marker = line.find(": ")
    if marker == -1:
        raise ManifestError(f"Not a 'key: value' line: {line!r}")
    key = line[:marker].strip()
    value = _parse_scalar(line[marker + 2:])
    # Only `size` is numeric. Converting every all-digit scalar instead would
    # silently turn an all-digit sha256 into the integer 0 -- which reads as
    # "no checksum declared" and skips verification altogether.
    if key == "size" and isinstance(value, str):
        if not value.isdigit():
            raise ManifestError(f"'size' must be a whole number of bytes, got {value!r}")
        value = int(value)
    return key, value


def _parse_manifest(path: str) -> dict:
    """Read the manifest into {tool: {kind: [entry, ...]}}.

    Deliberately NOT a YAML implementation -- it understands exactly the shape
    documented at the top of data-manifest.yml (two nesting levels of mappings,
    then a list of flat mappings) and rejects anything else with the line
    number, which is more useful here than partial support for the rest of YAML.
    """
    if not os.path.isfile(path):
        raise ManifestError(f"Manifest not found: {path}")

    tools: dict = {}
    tool = kind = entry = None
    seen_root = False

    with open(path, encoding="utf-8") as manifest:
        for number, raw in enumerate(manifest, start=1):
            line = raw.split("#", 1)[0].rstrip() if not _in_value(raw) else raw.rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()

            try:
                if indent == 0:
                    if stripped != "tools:":
                        raise ManifestError(f"Expected 'tools:' at the root, got {stripped!r}")
                    seen_root = True
                    continue
                if not seen_root:
                    raise ManifestError("Manifest must start with a 'tools:' block")

                if indent == 2:  # tool name
                    name, value = _split_key(stripped)
                    if value is not None:
                        raise ManifestError(f"Tool {name!r} must be followed by its kinds")
                    tool, kind, entry = name, None, None
                    tools.setdefault(tool, {})
                elif indent == 4:  # models: / testfiles:
                    name, value = _split_key(stripped)
                    # `provides: [A, B]` is a property of the tool, not a kind: the
                    # served tool names this one bundle covers, for a folder whose
                    # name stopped being a tool name when ALI and AREG each became
                    # several. Inline, so it cannot be read as downloadable items.
                    if name == "provides":
                        tools[tool]["provides"] = [
                            part.strip() for part in (value or "").strip("[]").split(",")
                            if part.strip()
                        ]
                        kind, entry = None, None
                        continue
                    if name not in KINDS:
                        raise ManifestError(f"Unknown section {name!r}, expected one of {KINDS} or 'provides'")
                    kind, entry = name, None
                    tools[tool].setdefault(kind, [])
                elif indent == 6 and stripped.startswith("- "):  # first key of an entry
                    entry = {}
                    tools[tool][kind].append(entry)
                    key, value = _split_key(stripped[2:])
                    entry[key] = value
                elif indent == 8:  # subsequent keys of the same entry
                    if entry is None:
                        raise ManifestError("Continuation line before any '- name:' entry")
                    key, value = _split_key(stripped)
                    entry[key] = value
                else:
                    raise ManifestError(f"Unexpected indentation ({indent} spaces)")
            except ManifestError as exc:
                raise ManifestError(f"{path}:{number}: {exc}") from None

    return tools


def _in_value(raw: str) -> bool:
    """True when a '#' on this line is part of a URL fragment, not a comment."""
    marker = raw.find(": ")
    return marker != -1 and "#" in raw[marker:] and "://" in raw[marker:]


def _entries(manifest: dict, kind: str, only_tools) -> list:
    """Flatten the manifest into the download list for one kind."""
    selected = []
    for tool in sorted(manifest):
        if only_tools and tool not in only_tools:
            continue
        for index, entry in enumerate(manifest[tool].get(kind, [])):
            where = f"tool '{tool}', {kind} entry #{index + 1}"
            if "name" not in entry or "url" not in entry:
                raise ManifestError(f"{where}: both 'name' and 'url' are required")
            if os.path.basename(entry["name"]) != entry["name"] or entry["name"] in ("", ".", ".."):
                # The name becomes a path under DATA/: a bare file name is the
                # only thing that cannot escape it.
                raise ManifestError(f"{where}: 'name' must be a bare file name")
            destination = entry.get("dest")
            if destination is not None:
                if os.path.isabs(destination) or ".." in destination.split("/"):
                    raise ManifestError(
                        f"{where}: 'dest' must be a relative path inside the tool's folder"
                    )
            selected.append({**entry, "tool": tool, "kind": kind})
    return selected


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _human(size) -> str:
    if size is None:
        return "unknown size"
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024


def _target_path(data_dir: str, entry: dict) -> str:
    """Where this entry ends up on disk once fetched (and extracted).

    `dest` overrides the file name and may name a subfolder, which is what
    makes several nnUNet bundles co-exist: they all ship a file literally
    called checkpoint_final.pth, so without it the second one downloaded would
    overwrite the first.
    """
    directory = os.path.join(data_dir, entry["tool"], entry["kind"])
    destination = entry.get("dest")
    if destination:
        return os.path.join(directory, *destination.split("/"))
    name = entry["name"]
    if entry.get("extract"):
        name = name[:-4] if name.lower().endswith(".zip") else name
    return os.path.join(directory, name)


# How progress is reported while a file downloads.
#   "tty"  one line rewritten in place with \r -- what a terminal wants
#   "line" one new line every _LINE_PROGRESS_SECONDS -- what a PIPE wants,
#          because \r is invisible to a reader collecting whole lines and a
#          12 GB bundle would otherwise print nothing at all for an hour
#   None   silent
_LINE_PROGRESS_SECONDS = 3


def _download(url: str, destination: str, progress) -> str:
    """Stream `url` to `destination`, returning the content's sha256."""
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers=_HEADERS)
    last_report = 0.0
    with urllib.request.urlopen(request) as response:  # noqa: S310 - https URLs from the manifest
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with open(destination, "wb") as out:
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress == "tty":
                    percent = f" {done * 100 // total:3d}%" if total else ""
                    print(f"\r      {_human(done)}{percent}", end="", flush=True)
                elif progress == "line" and time.monotonic() - last_report >= _LINE_PROGRESS_SECONDS:
                    last_report = time.monotonic()
                    of_total = f" / {_human(total)} ({done * 100 // total:3d}%)" if total else ""
                    print(f"      {_human(done)}{of_total}", flush=True)
    if progress == "tty":
        print("\r" + " " * 32 + "\r", end="", flush=True)
    return digest.hexdigest()


def _safe_extract(archive: str, destination: str) -> None:
    """Unpack `archive` into `destination`, refusing members that escape it.

    These archives come from known GitHub releases rather than from a client,
    but a path-traversal member would write outside DATA/ as whatever user ran
    this -- cheap to rule out, so it is ruled out.
    """
    root = os.path.realpath(destination)
    os.makedirs(root, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            resolved = os.path.realpath(os.path.join(root, member.filename))
            if resolved != root and os.path.commonpath([root, resolved]) != root:
                raise ManifestError(
                    f"Refusing to extract '{member.filename}': it points outside {destination}"
                )
        zf.extractall(root)

    # A zip of "AMASSS_Models/" rather than of its contents would otherwise
    # nest as AMASSS_Models/AMASSS_Models/. Descend into a lone root folder so
    # both packaging styles land identically on disk.
    entries = [name for name in os.listdir(root) if name != "__MACOSX" and not name.startswith(".")]
    if len(entries) == 1 and os.path.isdir(os.path.join(root, entries[0])):
        inner = os.path.join(root, entries[0])
        staging = root + ".unwrap"
        os.rename(inner, staging)
        shutil.rmtree(root)
        os.rename(staging, root)


def _fetch(entry: dict, data_dir: str, force: bool, progress) -> str:
    """Fetch one entry. Returns "skipped", "fetched", or raises."""
    target = _target_path(data_dir, entry)
    label = os.path.relpath(target, data_dir)

    if os.path.exists(target) and not force:
        print(f"  = {label} (already present)")
        return "skipped"

    os.makedirs(os.path.dirname(target), exist_ok=True)
    print(f"  + {label}")

    # Downloaded into a scratch folder beside the target and only moved into
    # place once complete and verified. An interrupted run therefore leaves
    # nothing at `target` -- rather than a truncated model the next run would
    # happily report as "already present".
    staging = tempfile.mkdtemp(dir=os.path.dirname(target), prefix=".fetch_")
    try:
        archive = os.path.join(staging, entry["name"])
        actual = _download(entry["url"], archive, progress)

        expected = entry.get("sha256")
        if expected and actual != expected:
            raise ManifestError(
                f"{label}: checksum mismatch\n"
                f"      expected {expected}\n"
                f"      got      {actual}\n"
                f"      The file was NOT installed. Either the download was corrupted, "
                f"or the release asset changed and the manifest needs updating."
            )
        if not expected:
            print(f"      no sha256 in the manifest; got {actual}")

        if entry.get("extract"):
            unpacked = os.path.join(staging, "unpacked")
            _safe_extract(archive, unpacked)
            os.remove(archive)
            ready = unpacked
        else:
            ready = archive

        if os.path.exists(target):
            shutil.rmtree(target) if os.path.isdir(target) else os.remove(target)
        os.rename(ready, target)
        return "fetched"
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    """Case- and separator-insensitive, the rule the registry already uses.

    `Crown_Seg` and `CrownSeg` are one tool written two ways; the server says so
    when it refuses duplicates, and a client reading GET /tools has no way to
    know which spelling this file happens to use.
    """
    return "".join(character for character in name.lower()
                   if character.isalnum())


def resolve_tools(manifest: dict, wanted) -> list:
    """Manifest entries for the tool names a caller asked for.

    A caller reads names from `GET /tools` and gets `Crown_Seg`, `ALI_CBCT`,
    `AREG_IOSCBCT`. This file predates the splits and still keys on `CrownSeg`,
    `ALI`, `AREG`, because those are the DATA/ folder names and moving them
    would break every deployment that already downloaded them. So the names are
    matched rather than renamed:

    - normalized equality, which resolves every rename that only moved
      underscores (`Crown_Seg`, `Batch_Dental_Seg`, `Surg_Mov_Pred`);
    - `provides:`, which an entry uses to name the tools it serves when one
      bundle feeds several (`ALI` feeds ALI_CBCT and ALI_IOS).

    Returns the manifest keys, in manifest order. Raises ManifestError naming
    what is unresolvable, because the alternative -- silently downloading
    something else -- is how a 12 GB bundle arrives for the wrong tool.
    """
    if not wanted:
        return sorted(manifest)

    index = {}
    for key, entry in manifest.items():
        index.setdefault(_normalize(key), key)
        for served in entry.get("provides", ()) or ():
            index.setdefault(_normalize(served), key)

    resolved, unknown = [], []
    for name in wanted:
        key = index.get(_normalize(name))
        if key is None:
            unknown.append(name)
        elif key not in resolved:
            resolved.append(key)
    if unknown:
        served = sorted({served for entry in manifest.values()
                         for served in (entry.get("provides") or ())} | set(manifest))
        raise ManifestError(
            "no such tool in the manifest: {}. Known: {}".format(
                ", ".join(sorted(unknown)), ", ".join(served)))
    return resolved


def _list_manifest(manifest: dict, wanted=None) -> None:
    grand_total = 0
    for tool in resolve_tools(manifest, wanted):
        entries = [entry for kind in KINDS for entry in manifest[tool].get(kind, [])]
        total = sum(entry.get("size") or 0 for entry in entries)
        grand_total += total
        print(f"\n{tool}  ({len(entries)} item(s), {_human(total)})")
        for kind in KINDS:
            for entry in manifest[tool].get(kind, []):
                shown = entry.get("dest") or entry["name"]
                print(f"  {kind:<10} {shown:<48} {_human(entry.get('size'))}")
    print(f"\nEverything: {_human(grand_total)}\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the server's models / test files into DATA/.",
        epilog="Files already present are skipped, so re-running is cheap and resumable.",
    )
    parser.add_argument(
        "--kind", choices=KINDS, action="append",
        help="What to fetch. Repeatable. Default: both.",
    )
    parser.add_argument(
        "--tool", action="append",
        help="Restrict to this tool (repeatable). Default: every tool in the manifest.",
    )
    parser.add_argument(
        "--data-dir", default=os.environ.get("DATA_DIR", "DATA"),
        help="Destination root, laid out as <tool>/<kind>/. Default: ./DATA",
    )
    parser.add_argument("--manifest", default=_DEFAULT_MANIFEST, help="Manifest to read.")
    parser.add_argument(
        "--progress", choices=("auto", "always", "never"), default="auto",
        help="Per-file download progress. 'auto' redraws one line on a terminal and stays "
             "silent in a pipe; 'always' prints a new line every few seconds, which is what "
             "a GUI streaming this output needs. Default: auto.",
    )
    parser.add_argument("--list", action="store_true", help="Show the manifest and exit.")
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    args = parser.parse_args(argv)

    try:
        manifest = _parse_manifest(args.manifest)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Resolved BEFORE listing: `--tool X --list` used to print the whole
    # manifest, so asking about one tool answered with 12 GB of another one's
    # bundles and no indication the filter had been dropped.
    try:
        selected = resolve_tools(manifest, args.tool)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.list:
        _list_manifest(manifest, args.tool)
        return 0

    kinds = args.kind or list(KINDS)
    try:
        work = [entry for kind in kinds for entry in _entries(manifest, kind, selected)]
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not work:
        print("Nothing to do: the manifest has no matching entry.")
        return 0

    data_dir = os.path.abspath(args.data_dir)
    # Sizes come from the manifest, so the figure covers items already on disk
    # too -- it is what the selection weighs, not what will travel. Worth
    # printing anyway: the full set runs to tens of GB, most of it one tool's
    # per-landmark models, and that is better known before the run than during.
    missing = [entry for entry in work if not os.path.exists(_target_path(data_dir, entry))]
    print(f"Destination: {data_dir}")
    print(f"Selected: {len(work)} item(s) ({_human(sum(e.get('size') or 0 for e in work))}), "
          f"of which {len(missing)} not yet present "
          f"({_human(sum(e.get('size') or 0 for e in missing))} to download)")
    print(f"Kinds: {', '.join(kinds)}\n")

    if args.progress == "never":
        progress = None
    elif args.progress == "always":
        progress = "line"
    else:
        progress = "tty" if sys.stdout.isatty() else None

    counts = {"fetched": 0, "skipped": 0}
    failures = []
    for entry in work:
        try:
            counts[_fetch(entry, data_dir, args.force, progress)] += 1
        except (ManifestError, urllib.error.URLError, OSError, zipfile.BadZipFile) as exc:
            # One unreachable asset must not abandon the other 40: report it at
            # the end and keep going, so a single dead release URL still leaves
            # a usable DATA/ behind.
            print(f"  ! FAILED {entry['tool']}/{entry['kind']}/{entry['name']}: {exc}")
            failures.append(entry)

    print(f"\n{counts['fetched']} fetched, {counts['skipped']} already present, {len(failures)} failed.")
    if failures:
        print("\nFailed items (re-run to retry just those):")
        for entry in failures:
            print(f"  {entry['tool']}/{entry['kind']}/{entry['name']}  <- {entry['url']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
