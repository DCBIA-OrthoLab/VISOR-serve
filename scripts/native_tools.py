#!/usr/bin/env python3
"""Build each SADT tool's virtualenv on a machine the Linux lock cannot resolve on.

Every torch tool's pyproject says `torch = { index = "pytorch-cu128" }`, with no
platform marker, and that index holds no macOS arm64 wheel: `uv sync --frozen`
cannot succeed on a Mac. Fixing it in each pyproject would touch twelve lock
files of a Linux deployment that works. This recipe leaves `pyproject.toml` and
`uv.lock` exactly as they are and builds the Mac venv from them instead:

  1. `uv venv` with the interpreter the pyproject asks for;
  2. `uv export --frozen`: the locked, pinned set, read-only;
  3. drop what names a CUDA build (torch, torchvision, pytorch3d `+cu128`,
     `nvidia-*`, triton, the CPU llama-cpp wheel) and install the rest with
     `--no-deps`, so nothing is re-resolved: the versions are the locked ones;
  4. install torch and torchvision from PyPI at the versions in
     `native_tools.json`, pytorch3d (CPU) where the tool imports it, the Metal
     llama-cpp wheel for CNE, and the per-tool extras (embreex for the IOS tools);
  5. `uv pip check`, and import torch, reporting whether MPS is available.

The tool itself is NOT installed as a package. The Docker image does the same
(`uv sync --no-install-project`): `runner.py` puts the tool's `src/` on
`sys.path`, so an editable install would add nothing and its metadata would
make a checkout write into itself.

Supply-chain risk, known and NOT fixed here (security audit, finding 3). The
recipe exports the lock with `--no-hashes` and installs with `uv pip install`
without `--require-hashes`; torch, torchvision, embreex, pytorch3d (`+pt280cpu`)
and llama-cpp-python are taken by version only, and pytorch3d and llama-cpp come
from third-party indexes listed in `native_tools.json` (`--extra-index-url`).
`run-native.sh` installs `server/requirements-api.txt` (fastapi, uvicorn, ...)
unpinned and unhashed. Whoever controls one of those indexes, or publishes a new
wheel or release for a name above, gets code running with the user's rights.
Left alone on purpose: changing the recipe or the API requirements risks the
Docker image and the Mac build. To fix it: (a) keep the hashes in the export and
add `--require-hashes` to each `pip install`; (b) put a sha256 per wheel in
`native_tools.json` and install those through a `pkg==v --hash=sha256:...` file
with `--require-hashes`; (c) produce a `requirements-api.lock` with
`uv pip compile --generate-hashes` and use it in `run-native.sh`; (d) mirror the
two third-party wheel indexes or vendor the wheels; (e) test with a fake `uv`
that records its argv and asserts `--require-hashes` on every install.

Standard library only, Python 3.9+, like `server_ctl.py`: it is called from
there, and that runs inside Slicer's interpreter.

    python3 scripts/native_tools.py list    --sadt ~/code/SADT-VISOR
    python3 scripts/native_tools.py install --sadt ~/code/SADT-VISOR --tool AMASSS --tool ASO
    python3 scripts/native_tools.py install --sadt ~/code/SADT-VISOR --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_HERE, "native_tools.json")

# Written into each venv once it is complete, holding a fingerprint of every
# input. A venv without it, or with another one, is rebuilt; one with the same
# is left alone, which is what makes `update` cheap when nothing moved.
STAMP_NAME = ".visor-native-recipe"


class RecipeError(Exception):
    """A step the user can act on: uv missing, a lock that does not export, a wheel that does not exist."""


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# What is where
# ---------------------------------------------------------------------------

def find_uv():
    found = os.environ.get("UV") or shutil.which("uv")
    if found:
        return found
    for candidate in (os.path.expanduser("~/.local/bin/uv"), "/opt/homebrew/bin/uv", "/usr/local/bin/uv"):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def default_torch_home() -> str:
    """<state>/torch_home with the state folder run-native.sh and server_ctl_native use."""
    state = os.environ.get("VISOR_STATE_DIR")
    if not state:
        state = (os.path.expanduser("~/Library/Application Support/VISOR/native") if sys.platform == "darwin"
                 else os.path.expanduser("~/.local/state/visor/native"))
    return os.path.join(state, "torch_home")


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _read_pyproject(path: str) -> dict:
    """`{requires_python, tool_name, is_tool}` from a pyproject.toml.

    `tomllib` where it exists (3.11+), a narrow regex reader where it does not
    (a Mac's system Python is 3.9). Only three facts are read, all of them
    single-line strings in these repositories.
    """
    with open(path, "rb") as handle:
        raw = handle.read()
    try:
        import tomllib  # type: ignore[import-not-found]
        data = tomllib.loads(raw.decode("utf-8"))
        sadt = data.get("tool", {}).get("sadt")
        return {
            "requires_python": data.get("project", {}).get("requires-python"),
            "tool_name": (sadt or {}).get("name"),
            "is_tool": bool(sadt and sadt.get("tool")),
        }
    except ImportError:
        pass
    text = raw.decode("utf-8")
    match = re.search(r'^requires-python\s*=\s*"([^"]+)"', text, re.M)
    section = re.search(r"^\[tool\.sadt\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    name = None
    is_tool = False
    if section:
        body = section.group(1)
        found = re.search(r'^name\s*=\s*"([^"]+)"', body, re.M)
        name = found.group(1) if found else None
        is_tool = bool(re.search(r"^tool\s*=\s*true", body, re.M))
    return {"requires_python": match.group(1) if match else None, "tool_name": name, "is_tool": is_tool}


def discover_tools(sadt_root: str) -> dict:
    """`{api name: tool directory}`, two levels deep, gated on `[tool.sadt]`.

    The same rule the Dockerfile and the server's registry use, so a shared
    package (`tools/ALI/common`) never gets a venv and a tool in a grouping
    folder (`tools/ALI/ALI_CBCT`) is found.
    """
    root = os.path.join(sadt_root, "tools")
    found = {}
    if not os.path.isdir(root):
        return found
    for first in sorted(os.listdir(root)):
        for candidate in (os.path.join(root, first),) + tuple(
                os.path.join(root, first, second)
                for second in (sorted(os.listdir(os.path.join(root, first)))
                               if os.path.isdir(os.path.join(root, first)) else [])):
            pyproject = os.path.join(candidate, "pyproject.toml")
            if not os.path.isfile(pyproject) or not os.path.isfile(os.path.join(candidate, "uv.lock")):
                continue
            info = _read_pyproject(pyproject)
            if not info["is_tool"]:
                continue
            name = info["tool_name"] or os.path.basename(candidate)
            found[name] = candidate
    return found


# ---------------------------------------------------------------------------
# Reading the lock
# ---------------------------------------------------------------------------

_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(==|@|;|$|>=|<=|~=|!=)")


def _normal(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def export_lock(uv: str, tool_dir: str, python=None) -> list:
    """The locked requirement lines of a tool, as `uv export --frozen` prints them.

    Read-only: `--frozen` never touches uv.lock. Every extra is included, for
    the reason the Dockerfile gives (an extra is where a tool keeps an engine it
    does not want CI to build, and a deployment needs it).
    """
    command = [uv, "export", "--frozen", "--no-dev", "--no-emit-project", "--no-hashes",
               "--no-annotate", "--all-extras", "--quiet"]
    done = subprocess.run(command, cwd=tool_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if done.returncode != 0:
        raise RecipeError("uv export failed in %s:\n%s" % (tool_dir, done.stderr.strip()))
    return [line for line in done.stdout.splitlines() if line.strip() and not line.startswith("#")]


def split_requirements(lines, config: dict, tool_dir: str):
    """`(kept lines, dropped {name: version})`, paths made absolute.

    A path dependency (`sadt-naming`, `sadt-areg-common`) is written by uv
    relative to the tool, and `uv pip install -r` resolves it against the
    CURRENT directory. Made absolute here so the install does not depend on
    where this script was started.
    """
    drop = config["drop_from_lock"]
    names = {_normal(name) for name in drop["names"]}
    prefixes = tuple(drop["prefixes"])
    markers = tuple(drop["version_contains"])
    kept, dropped = [], {}
    for line in lines:
        match = _REQ_NAME.match(line)
        if not match:
            # A path or an editable: `../../naming`, `-e ../x`.
            stripped = line.strip()
            editable = stripped.startswith("-e ")
            target = stripped[3:].strip() if editable else stripped
            if target.startswith((".", "/")):
                absolute = os.path.normpath(os.path.join(tool_dir, target.split(" ;")[0].strip()))
                kept.append(("-e " if editable else "") + absolute)
            else:
                kept.append(line)
            continue
        name = _normal(match.group(1))
        version = re.search(r"==\s*([^\s;]+)", line)
        version = version.group(1) if version else ""
        if name in names or name.startswith(prefixes) or any(mark in version for mark in markers):
            dropped[name] = version
            continue
        kept.append(line)
    return kept, dropped


# ---------------------------------------------------------------------------
# The plan for one tool
# ---------------------------------------------------------------------------

def plan_for(name: str, tool_dir: str, config: dict, lock_lines: list) -> dict:
    """What would be installed, decided without installing anything."""
    info = _read_pyproject(os.path.join(tool_dir, "pyproject.toml"))
    kept, dropped = split_requirements(lock_lines, config, tool_dir)
    locked_torch = dropped.get("torch")
    wants_pytorch3d = "pytorch3d" in dropped
    wants_llama = name in config["metal_llama_cpp"]["tools"]

    torch = None
    if locked_torch:
        torch = dict(config["default_torch"])
        torch.update({key: value for key, value in config["torch_by_tool"].get(name, {}).items()
                      if not key.startswith("_")})
        if "torchvision" not in dropped:
            torch.pop("torchvision", None)
        elif "torchvision" not in torch:
            torch["torchvision"] = config["default_torch"]["torchvision"]
    note = None
    pytorch3d = None
    if wants_pytorch3d:
        pytorch3d = config["pytorch3d"]["version"]
        required = config["pytorch3d"]["requires"]
        if torch is None:
            raise RecipeError("%s imports pytorch3d but its lock holds no torch" % name)
        if torch.get("torch") != required["torch"]:
            note = ("pytorch3d %s is a CPU wheel built against torch %s: %s is held to torch %s "
                    "instead of %s" % (pytorch3d, required["torch"], name, required["torch"], torch.get("torch")))
        torch = dict(required)
        if "torchvision" not in dropped:
            torch.pop("torchvision", None)
    if torch is not None and "torchaudio" in dropped:
        # Same release train as torch: 2.8.0 goes with 2.8.0.
        torch["torchaudio"] = torch["torch"]

    return {
        "name": name,
        "tool_dir": tool_dir,
        "python": info["requires_python"],
        "requirements": kept,
        "dropped": dropped,
        "torch": torch,
        "pytorch3d": pytorch3d,
        "pytorch3d_index": config["indexes"]["pytorch3d"] if pytorch3d else None,
        "llama_cpp": config["metal_llama_cpp"]["requirement"] if wants_llama else None,
        "llama_cpp_index": config["indexes"]["llama_cpp_metal"] if wants_llama else None,
        "extras": list(config["extra_by_tool"].get(name, [])),
        "note": note,
    }


_SKIP_DIRS = {"__pycache__", ".git", ".venv", "build", "dist", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def path_packages(requirements) -> list:
    """The non-editable local packages of a plan's requirements (`sadt-areg-common`, `sadt-naming`).

    split_requirements made them absolute directories. An editable one (`-e`)
    is not listed: it is linked, so the venv always sees its current source.
    """
    return [line for line in requirements
            if not line.startswith("-e ") and os.path.isabs(line) and os.path.isdir(line)]


def _hash_tree(digest, root: str) -> None:
    """Fold every source file under `root` (names and bytes, sorted) into `digest`."""
    for folder, subfolders, files in os.walk(root):
        subfolders[:] = sorted(d for d in subfolders if d not in _SKIP_DIRS and not d.endswith(".egg-info"))
        for name in sorted(files):
            if name.endswith((".pyc", ".pyo")):
                continue
            path = os.path.join(folder, name)
            digest.update(os.path.relpath(path, root).encode("utf-8", "replace") + b"\0")
            try:
                with open(path, "rb") as handle:
                    digest.update(handle.read())
            except OSError:
                digest.update(b"?")


def fingerprint(plan: dict, tool_dir: str) -> str:
    """Hash of everything the venv was built from, so a change anywhere rebuilds it.

    The source of the non-editable path packages is part of it: uv builds those
    into a wheel and caches it by directory, not by content, so a venv whose
    lock and recipe did not move would otherwise keep a stale `common` forever.
    """
    digest = hashlib.sha256()
    for name in ("pyproject.toml", "uv.lock"):
        with open(os.path.join(tool_dir, name), "rb") as handle:
            digest.update(handle.read())
    digest.update(json.dumps({key: plan[key] for key in
                              ("torch", "pytorch3d", "llama_cpp", "extras", "python")},
                             sort_keys=True).encode())
    digest.update(json.dumps(plan["requirements"]).encode())
    for package in path_packages(plan["requirements"]):
        digest.update(b"\0path:" + package.encode())
        _hash_tree(digest, package)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Doing it
# ---------------------------------------------------------------------------

def _run(command, cwd=None, env=None, attempts: int = 2) -> None:
    """Run a command, narrating its output to stderr. Retried once: the two
    indexes that are not PyPI (github.io pages) reset connections now and then,
    which was measured here on the first full install and not on the second."""
    for attempt in range(1, attempts + 1):
        log("$ " + " ".join(command))
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        with process:
            for line in process.stdout:
                log("  | " + line.rstrip())
        if process.returncode == 0:
            return
        if attempt < attempts:
            log("%s exited with %d; trying once more in 5 s" % (command[0], process.returncode))
            time.sleep(5)
    raise RecipeError("%s exited with %d" % (command[0], process.returncode))


def install_one(uv: str, plan: dict, force: bool = False) -> dict:
    """Build (or keep) one tool's `.venv`. Returns what was done."""
    tool_dir = plan["tool_dir"]
    venv = os.path.join(tool_dir, ".venv")
    python = os.path.join(venv, "bin", "python")
    want = fingerprint(plan, tool_dir)
    stamp = os.path.join(venv, STAMP_NAME)
    try:
        with open(stamp, encoding="utf-8") as handle:
            have = handle.read().strip()
    except OSError:
        have = None
    if not force and have == want and os.path.isfile(python):
        log("%s: venv is current, left alone" % plan["name"])
        return {"tool": plan["name"], "action": "kept"}

    log("%s: building %s" % (plan["name"], venv))
    if os.path.isdir(venv):
        shutil.rmtree(venv)
    command = [uv, "venv", "--python", plan["python"] or "3.11", venv]
    _run(command, cwd=tool_dir)

    local = path_packages(plan["requirements"])
    registry = [line for line in plan["requirements"] if line not in local]
    if registry:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", prefix="visor-%s-" % plan["name"],
                                         delete=False, encoding="utf-8") as handle:
            handle.write("\n".join(registry) + "\n")
            requirements = handle.name
        try:
            # --no-deps: the export is the whole locked set, pinned. Letting uv
            # resolve again would be how a Mac venv drifts from the Linux one.
            _run([uv, "pip", "install", "--python", python, "--no-deps", "-r", requirements], cwd=tool_dir)
        finally:
            os.unlink(requirements)
    if local:
        # --no-cache --reinstall: uv keys the wheel it built from a local directory on the
        # directory, not on its files, so the cached one can predate the checkout.
        _run([uv, "pip", "install", "--python", python, "--no-deps", "--reinstall", "--no-cache"] + local,
             cwd=tool_dir)

    if plan["torch"]:
        specs = ["%s==%s" % (package, version) for package, version in plan["torch"].items()
                 if not package.startswith("_")]
        _run([uv, "pip", "install", "--python", python] + specs, cwd=tool_dir)
    if plan["pytorch3d"]:
        # A local version (`+pt280cpu`) exists only on this index. --no-deps:
        # its dependencies (iopath, fvcore) are in the lock export already.
        _run([uv, "pip", "install", "--python", python, "--no-deps",
              "--extra-index-url", plan["pytorch3d_index"], "pytorch3d==%s" % plan["pytorch3d"]],
             cwd=tool_dir)
    warnings = []
    if plan["llama_cpp"]:
        # Not fatal, and not stamped: the rest of CNE (documents, text) is
        # usable without the model engine, and the next `update` tries again.
        # Measured 2026-11-17: the 0.3.35 Metal wheel on that index is a
        # corrupt archive (bad zip offsets), which uv rejects.
        try:
            _run([uv, "pip", "install", "--python", python, "--no-deps",
                  "--extra-index-url", plan["llama_cpp_index"], plan["llama_cpp"]], cwd=tool_dir)
        except RecipeError as exc:
            warnings.append("%s was not installed: %s" % (plan["llama_cpp"], exc))
            log("%s: WARNING %s" % (plan["name"], warnings[-1]))
    if plan["extras"]:
        _run([uv, "pip", "install", "--python", python] + plan["extras"], cwd=tool_dir)

    report = verify_one(uv, plan)
    if warnings:
        report["warnings"] = warnings
    if report["pip_check_ok"] and not warnings:
        with open(stamp, "w", encoding="utf-8") as handle:
            handle.write(want + "\n")
    report.update({"tool": plan["name"], "action": "built"})
    return report


_PROBE = (
    "import json,sys\n"
    "out={'python':sys.version.split()[0]}\n"
    "try:\n"
    "    import torch\n"
    "    out['torch']=torch.__version__\n"
    "    b=getattr(torch.backends,'mps',None)\n"
    "    out['mps_available']=bool(b and b.is_available())\n"
    "except ImportError:\n"
    "    out['torch']=None\n"
    "print(json.dumps(out))\n"
)


def _unmet(check_output: str, ignore) -> list:
    """The incompatibilities `uv pip check` lists, minus the ones the config says cannot be met here."""
    found = []
    for line in check_output.splitlines():
        if line.startswith("The package "):
            if any(("requires `%s" % name) in line for name in ignore):
                continue
            found.append(line)
    return found


def verify_one(uv: str, plan: dict, config=None) -> dict:
    python = os.path.join(plan["tool_dir"], ".venv", "bin", "python")
    check = subprocess.run([uv, "pip", "check", "--python", python], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
    probe = subprocess.run([python, "-c", _PROBE], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    facts = {}
    if probe.returncode == 0:
        try:
            facts = json.loads(probe.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            facts = {}
    ignore = ((config or load_config()).get("pip_check_ignore", {}).get("requires", []))
    unmet = _unmet(check.stdout, ignore) if check.returncode != 0 else []
    if check.returncode != 0:
        log("%s: uv pip check reported:\n%s" % (plan["name"], check.stdout.strip()))
    return {"pip_check_ok": not unmet, "pip_check": check.stdout.strip(), "probe": facts}


def with_requirements(wanted, config: dict, found: dict) -> list:
    """`wanted` plus the tools the server needs to serve them, each after the tool that needs it.

    Only the left-hand tools of `serve_requires` pull anything in, so asking for
    every tool, or for tools nobody depends on, changes nothing.
    """
    needs = config.get("serve_requires", {})
    out = []

    def add(name, because=None):
        if name in out:
            return
        if name not in found:
            raise RecipeError(f"{name} is needed to serve {because} but is not in this checkout")
        out.append(name)
        for dependency in needs.get(name, []):
            add(dependency, name)

    for name in wanted:
        add(name)
    for name in out:
        if name not in wanted:
            callers = ", ".join(sorted(k for k in out if name in needs.get(k, []))) or "its callers"
            log(f"{name}: added, the server refuses to serve {callers} without it")
    return out


def _download(url: str, destination: str) -> None:
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=120) as response, open(destination, "wb") as handle:
            shutil.copyfileobj(response, handle)
        return
    except Exception as exc:  # noqa: BLE001 - python.org builds of macOS Python often lack CA roots
        log(f"download with urllib failed ({type(exc).__name__}); trying curl")
    curl = shutil.which("curl")
    if not curl:
        raise RecipeError(f"could not download {url} (no curl to fall back on)")
    _run([curl, "-fsSL", "--max-time", "300", "-o", destination, url], attempts=1)


def _sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_backbones(names, torch_home: str, config: dict, dry_run: bool = False) -> list:
    """Put each needed backbone in `<torch_home>/hub/checkpoints`, verified by sha256.

    A file already there with the right digest is kept; a wrong one is
    replaced. The download goes to a temporary name and is moved only after the
    digest matches, so a truncated or substituted file never sits under the
    real name where the tool would load it.
    """
    report = []
    for name in names:
        spec = config.get("backbones", {}).get(name)
        if not spec:
            continue
        directory = os.path.join(torch_home, "hub", "checkpoints")
        target = os.path.join(directory, spec["file"])
        entry = {"tool": name, "file": spec["file"], "path": target}
        if os.path.isfile(target) and _sha256_of(target) == spec["sha256"]:
            entry["action"] = "kept"
        elif dry_run:
            entry["action"] = "plan"
        else:
            os.makedirs(directory, exist_ok=True)
            partial = target + ".part"
            try:
                log(f"{name}: staging {spec['file']}")
                _download(spec["url"], partial)
                got = _sha256_of(partial)
                if got != spec["sha256"]:
                    raise RecipeError(f"{spec['file']} has sha256 {got}, expected {spec['sha256']}")
                os.replace(partial, target)
                entry["action"] = "staged"
            except (RecipeError, OSError) as exc:
                entry.update({"action": "failed", "error": str(exc)})
                log(f"{name}: backbone FAILED: {exc}")
            finally:
                if os.path.exists(partial):
                    os.unlink(partial)
        report.append(entry)
    return report


def run(sadt_root: str, tools=None, dry_run: bool = False, force: bool = False, config=None,
        torch_home=None) -> dict:
    uv = find_uv()
    if not uv:
        raise RecipeError("uv was not found. Install it (https://docs.astral.sh/uv/) and run this again.")
    config = config or load_config()
    found = discover_tools(sadt_root)
    if not found:
        raise RecipeError("No tool with a pyproject.toml and a uv.lock under %s/tools" % sadt_root)
    wanted = list(tools) if tools else sorted(found)
    unknown = [name for name in wanted if name not in found]
    if unknown:
        raise RecipeError("Unknown tool(s): %s. Known: %s" % (", ".join(unknown), ", ".join(sorted(found))))
    wanted = with_requirements(wanted, config, found)

    results = []
    for name in wanted:
        tool_dir = found[name]
        lines = export_lock(uv, tool_dir)
        plan = plan_for(name, tool_dir, config, lines)
        if plan["note"]:
            log("%s: NOTE %s" % (name, plan["note"]))
        if dry_run:
            results.append({
                "tool": name, "action": "plan", "python": plan["python"], "torch": plan["torch"],
                "pytorch3d": plan["pytorch3d"], "llama_cpp": plan["llama_cpp"], "extras": plan["extras"],
                "locked_packages_kept": len(plan["requirements"]),
                "dropped_from_lock": sorted(plan["dropped"]),
                "note": plan["note"],
            })
            continue
        try:
            results.append(install_one(uv, plan, force=force))
        except RecipeError as exc:
            log("%s: FAILED: %s" % (name, exc))
            results.append({"tool": name, "action": "failed", "error": str(exc)})
    failed = [r["tool"] for r in results if r["action"] == "failed"]
    backbones = []
    if torch_home:
        backbones = stage_backbones(wanted, torch_home, config, dry_run=dry_run)
        failed += [f"{b['tool']} (backbone)" for b in backbones if b["action"] == "failed"]
    return {"sadt": sadt_root, "tools": results, "backbones": backbones, "torch_home": torch_home, "failed": failed}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("list", "install"))
    parser.add_argument("--sadt", required=True, help="A SADT-VISOR checkout (the folder holding tools/).")
    parser.add_argument("--tool", action="append", help="Only this tool (repeatable). Default: all.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan, install nothing.")
    parser.add_argument("--force", action="store_true", help="Rebuild even when the venv is current.")
    parser.add_argument("--torch-home", default=None,
                        help="Where backbones are staged (default: <VISOR_STATE_DIR>/torch_home, as run-native.sh).")
    args = parser.parse_args(argv)
    sadt = os.path.abspath(os.path.expanduser(args.sadt))
    try:
        if args.command == "list":
            result = {"tools": discover_tools(sadt)}
        else:
            result = run(sadt, args.tool, dry_run=args.dry_run, force=args.force,
                         torch_home=args.torch_home or default_torch_home())
    except RecipeError as exc:
        log("error: %s" % exc)
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 1 if result.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
