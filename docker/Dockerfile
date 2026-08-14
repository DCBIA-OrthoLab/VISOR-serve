# syntax=docker/dockerfile:1.7
#
# One image, one container, N virtualenvs.
#
#     /opt/sadt/{.venv,server,runner.py}   the API: newest Python, no torch
#     /tools/<name>/{.venv,.schema.json,src}
#     /DATA/<name>/{models,testfiles}      read-only mount, outside the image
#     /jobs/<uuid>/                        ephemeral
#
# NOT one image per tool, and not docker-compose orchestrating tools: a tool is
# a virtualenv, not a service. Fifteen images would mean fifteen copies of the
# same CUDA stack, fifteen things to schedule, and a network hop between a tool
# and the file it was handed.
#
# Build:
#     docker buildx build -f docker/Dockerfile --build-context tools=<dir> -t sadt .
#
# where <dir> holds one folder per tool, each with pyproject.toml, uv.lock,
# .schema.json and src/ -- what the sadt-tools repository produces. Without
# that context the image builds with no tools at all, which is a valid (if
# empty) server: the `tools` stage below is what gets overridden.

# ----------------------------------------------------------------------
# The tools, from another repository
# ----------------------------------------------------------------------
# A named build context overrides a stage of the same name, so this empty
# stage is the default and `--build-context tools=<dir>` is the real input.
FROM scratch AS tools

# ----------------------------------------------------------------------
# Manifests only
# ----------------------------------------------------------------------
# Editing a tool's SOURCE must not reinstall its dependencies. Docker's COPY
# cannot glob a directory structure (`COPY tools/*/uv.lock` flattens), so the
# manifests are extracted in a stage of their own: this stage re-runs on any
# change, but its OUTPUT is content-addressed, so as long as no pyproject.toml
# or uv.lock changed, the layer digest is identical and the expensive `uv sync`
# below stays cached.
FROM busybox:1.36 AS manifests
COPY --from=tools . /ctx/
RUN mkdir -p /manifests && cd /ctx && \
    for manifest in $(find . -name pyproject.toml -o -name uv.lock); do \
        mkdir -p "/manifests/$(dirname "$manifest")" && cp "$manifest" "/manifests/$manifest"; \
    done

# ----------------------------------------------------------------------
# The image
# ----------------------------------------------------------------------
# One recent base for everything. Deliberately NOT one CUDA base per torch
# version: a torch wheel bundles its own CUDA runtime and the driver lives on
# the host, and the nvidia-* wheels are `py3-none-manylinux` -- no Python ABI
# tag -- so they deduplicate across virtualenvs even when the tools pin
# different Pythons. uv installs each tool's interpreter itself.
FROM python:3.13-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /usr/local/bin/

# UV_CACHE_DIR is on the IMAGE's filesystem, and that is load-bearing: uv
# installs a package by hardlinking it out of its cache, link() fails with
# EXDEV across filesystems, and uv then falls back to copying -- silently, for
# every package of every tool. A BuildKit cache mount is a different
# filesystem, so it is used below as transport only, never as UV_CACHE_DIR.
ENV UV_CACHE_DIR=/opt/uv-cache \
    UV_LINK_MODE=hardlink \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PYTHON_DOWNLOADS=automatic

# ---- the API process -------------------------------------------------
# fastapi, uvicorn, pydantic, python-multipart. No torch, no numpy, no pandas:
# this process validates requests and moves files, and everything heavy lives
# in a tool's own virtualenv one exec() away.
COPY server/requirements-api.txt /opt/sadt/requirements-api.txt
RUN uv venv --python 3.13 /opt/sadt/.venv && \
    uv pip install --python /opt/sadt/.venv/bin/python -r /opt/sadt/requirements-api.txt && \
    uv cache prune --ci

# ---- the tools -------------------------------------------------------
COPY --from=manifests /manifests/ /tools/

# EVERY `uv sync` in ONE layer, and this is not stylistic. overlayfs copies a
# file up when a later layer touches it, which breaks the hardlinks between
# tools installed in different layers -- eleven torch tools go from ~26 GB to
# ~63 GB. It also means the cache must be created AND removed here: a cache
# deleted in a later layer frees nothing, it only adds a whiteout.
#
# Deleting it costs nothing: the venvs' files are hardlinks to the same inodes,
# so removing the cache's copy drops a link count, not the data.
RUN --mount=type=cache,target=/cache/uv,sharing=locked \
    set -eu; \
    mkdir -p "$UV_CACHE_DIR"; \
    cp -a /cache/uv/. "$UV_CACHE_DIR"/ 2>/dev/null || true; \
    for tool in /tools/*/; do \
        [ -f "$tool/pyproject.toml" ] || continue; \
        echo "==> uv sync $(basename "$tool")"; \
        ( cd "$tool" && uv sync --frozen --no-install-project --no-dev ); \
    done; \
    cp -a "$UV_CACHE_DIR"/. /cache/uv/ 2>/dev/null || true; \
    uv cache prune --ci >/dev/null 2>&1 || true; \
    rm -rf "$UV_CACHE_DIR"

# The sources, last: a tool's code changes far more often than its lock file,
# and this is the only layer that has to be rebuilt when it does. `--frozen
# --no-install-project` above is why the venvs could be built without them --
# a tool's code is not installed as a package, runner.py puts src/ on sys.path.
COPY --from=tools . /tools/

# ---- the server ------------------------------------------------------
# server/tools/ is excluded in .dockerignore: this image serves tools from
# /tools, and importing the in-process ones would need the very stack the API
# venv is built to do without.
COPY server/ /opt/sadt/server/
# Beside the server rather than inside it. runner.py is executed BY a tool's
# interpreter and is handed over by path, so it is never installed into a tool
# venv -- runner and server are then always the same version.
COPY server/runner.py /opt/sadt/runner.py
COPY docker/verify_dedup.py /opt/sadt/verify_dedup.py

ENV SADT_DISPATCH_MODE=subprocess \
    TOOLS_DIR=/tools \
    RUNNER_PATH=/opt/sadt/runner.py \
    DEPLOYMENT_CONFIG=/opt/sadt/deployment.toml \
    DATA_DIR=/DATA \
    TEMP_DIR=/jobs \
    PYTHONDONTWRITEBYTECODE=1

# Third-party code from fifteen upstreams runs in this container, on
# confidential imaging. It does not run as root.
RUN useradd --system --create-home --home-dir /home/sadt sadt && \
    install -d -o sadt -g sadt /jobs && \
    install -d -o sadt -g sadt /DATA

USER sadt
WORKDIR /opt/sadt/server
EXPOSE 8000

# No --reload: this image is the artifact, not a mount of a working tree.
CMD ["/opt/sadt/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
