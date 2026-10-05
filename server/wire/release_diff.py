"""What a set of changed paths asks of a deployment -- shared by the server's own
"Check for updates" and by the host's update agent.

Standard library only, and nothing imported from the server: the agent on the
host loads this file by path, before anything is installed, so the panel and
the agent can never classify the same commits differently.
"""

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


def classify_tools(paths) -> dict:
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
    return {
        "tools": sorted(tools.values(), key=lambda t: t["tool"]),
        "environments": sorted(environments),
        "other": other[:20],
    }


def classify(kind: str, paths) -> dict:
    return classify_server(paths) if kind == "server" else classify_tools(paths)
