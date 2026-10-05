"""What a set of changed paths asks of a deployment -- shared by the server's own
"Check for updates" and by the host's update agent.

Standard library only, and nothing imported from the server: the agent on the
host loads this file by path, before anything is installed, so the panel and
the agent can never classify the same commits differently.
"""

import os
import re

RESTART, RECREATE, IMAGE = "restart", "recreate", "image"
_RECREATE_PATHS = ("docker-compose.yml",)
_RECREATE_PREFIXES = ("server/requirements",)
_IMAGE_PREFIXES = ("docker/",)


def classify_server(paths) -> dict:
    """`{action, heavy}` for the server repository: the heaviest of restart,
    recreate and image that any changed path asks for."""
    action = RESTART if paths else None
    heavy = []
    for path in paths:
        if path.startswith(_IMAGE_PREFIXES):
            action = IMAGE
            heavy.append(path)
        elif path in _RECREATE_PATHS or path.startswith(_RECREATE_PREFIXES):
            if action != IMAGE:
                action = RECREATE
            heavy.append(path)
    return {"action": action, "heavy": heavy}


_SOURCE_PATH = re.compile(r'^\s*([A-Za-z0-9_.\-]+)\s*=\s*\{[^}]*\bpath\s*=\s*"([^"]+)"')
_EDITABLE = re.compile(r"\beditable\s*=\s*true\b")


def local_dependencies(repo: str) -> dict:
    """`{environment folder: {package: its folder}}` for every environment in
    the library that installs another folder of it as a package.

    Read from each `pyproject.toml`'s `[tool.uv.sources]` (`name = { path =
    "..." }`), so nothing here knows which tool depends on what. It matters
    because such a package is installed by COPY: a commit that changes it
    changes nothing in a dependent's environment until that environment is
    synced again with the package reinstalled.
    """
    found = {}
    tools_root = os.path.join(repo, "tools")
    for directory, subdirs, files in os.walk(tools_root):
        subdirs[:] = [d for d in subdirs if not d.startswith(".") and d not in ("node_modules", "__pycache__")]
        if "pyproject.toml" not in files:
            continue
        try:
            with open(os.path.join(directory, "pyproject.toml"), encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            continue
        sources, in_sources = {}, False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("["):
                in_sources = stripped == "[tool.uv.sources]"
                continue
            if in_sources:
                match = _SOURCE_PATH.match(line)
                # An editable install reads the folder live: nothing to redo.
                if match and not _EDITABLE.search(line):
                    sources[match.group(1)] = os.path.normpath(os.path.join(directory, match.group(2)))
        if sources:
            environment = os.path.relpath(directory, repo).replace(os.sep, "/")
            found[environment] = {name: os.path.relpath(path, repo).replace(os.sep, "/")
                                  for name, path in sources.items()}
    return found


def classify_tools(paths, dependencies=None) -> dict:
    """Per tool, whether only its code changed or its environment must be
    rebuilt -- and which environments, as the folders to `uv sync` in.

    A tool is `tools/<Name>/`; an environment is any folder holding a changed
    `pyproject.toml` or `uv.lock`, which for AREG's engines is a level deeper
    (`tools/AREG/AREG_CBCT/`). Anything outside `tools/` -- the data manifest,
    the scripts -- is listed separately: it changes no running tool.
    """
    tools, environments, other = {}, set(), []
    for path in paths:
        parts = path.split("/")
        if len(parts) < 3 or parts[0] != "tools" or parts[1].startswith(("_", ".")):
            other.append(path)
            continue
        entry = tools.setdefault(parts[1], {"tool": parts[1], "files": 0, "environment": False})
        entry["files"] += 1
        if parts[-1] in ("pyproject.toml", "uv.lock"):
            entry["environment"] = True
            environments.add("/".join(parts[:-1]))
    # A changed folder another environment installs by copy: that environment
    # is rebuilt with the package reinstalled, and its tool listed as changed.
    reinstall = {}
    for environment, packages in (dependencies or {}).items():
        for package, folder in packages.items():
            if any(path == folder or path.startswith(folder + "/") for path in paths):
                reinstall.setdefault(environment, []).append(package)
    for environment in reinstall:
        environments.add(environment)
        parts = environment.split("/")
        if len(parts) >= 2 and parts[0] == "tools":
            entry = tools.setdefault(parts[1], {"tool": parts[1], "files": 0, "environment": False})
            entry["environment"] = True
    return {
        "tools": sorted(tools.values(), key=lambda t: t["tool"]),
        "environments": sorted(environments),
        "reinstall": {env: sorted(set(names)) for env, names in reinstall.items()},
        "other": other[:20],
    }


def classify(kind: str, paths, repo: str = "") -> dict:
    if kind == "server":
        return classify_server(paths)
    return classify_tools(paths, local_dependencies(repo) if repo else None)
