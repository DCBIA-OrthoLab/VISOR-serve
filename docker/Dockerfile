# syntax=docker/dockerfile:1.7-labs
#
# One image, one container, N virtualenvs.
#
#     /opt/sadt/{.venv,server}   the API: newest Python, no torch
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
# where <dir> is the sadt-tools repository: `tools/<name>/` with pyproject.toml,
# uv.lock and src/, and `scripts/describe.py`, which is what turns a tool's
# run() signature into the schema this server publishes. Without that context
# the image builds with no tools at all, which is a valid (if empty) server:
# the `tools` stage below is what gets overridden.

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
# --exclude for the same reason the sources are copied with it: the tools
# context is a working checkout, where every tool already has a .venv. Without
# it the find below walks INTO those venvs and copies the pyproject.toml of
# every installed package, which recreates a `.venv/` directory holding no
# interpreter -- and `uv sync` then refuses the tool outright.
COPY --from=tools --exclude=**/.venv --exclude=**/__pycache__ tools /ctx/
RUN mkdir -p /manifests && cd /ctx && \
    for manifest in $(find . -name pyproject.toml -o -name uv.lock); do \
        mkdir -p "/manifests/$(dirname "$manifest")" && cp "$manifest" "/manifests/$manifest"; \
    done && \
    # A SHARED package comes over whole, sources included, not just its
    # manifest. `uv sync` runs before the tool sources are copied -- which is
    # what keeps a source edit from rebuilding every virtualenv -- but a path
    # dependency has to EXIST at that point. tools/ALI/common/ and
    # tools/AREG/common/ are dependencies of the tools beside them, so a
    # manifest alone leaves `uv sync` with nothing to install and the tool fails
    # at import with "No module named sadt_ali_common".
    #
    # They are identified the same way the server identifies them: a
    # pyproject.toml with no [tool.sadt]. Cheap to copy early, being stdlib-only
    # and a few hundred lines, and they change far less often than tool sources.
    for manifest in $(find . -name pyproject.toml); do \
        grep -q "^\[tool.sadt\]" "$manifest" && continue; \
        package="$(dirname "$manifest")"; \
        mkdir -p "/manifests/$package" && cp -a "$package/." "/manifests/$package/"; \
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

# The one system library the tools need and no lockfile can provide. libgomp is
# the GNU OpenMP runtime: lightgbm dlopens it, and so do torch and
# scikit-learn. A tool's uv.lock pins its PYTHON dependencies exactly and says
# nothing about the C libraries they load, so without this every SurgMovPred
# model fails to unpickle -- "None of the 112 model package(s) could be loaded"
# -- and the cause, `libgomp.so.1: cannot open shared object file`, is three
# frames deep inside joblib.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

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

# Two levels, and gated on [tool.sadt], for the same reason the registry is: a
# tool may live inside a GROUPING folder (tools/ALI/ALI_CBCT), and a folder with
# a pyproject.toml but no [tool.sadt] is a shared package rather than a tool --
# tools/ALI/common/ must not get a virtualenv of its own. A one-level loop built
# six venvs of eleven and the five nested tools were simply absent from the
# image, which nothing reported until the server tried to run one.
#
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
    for tool in /tools/*/ /tools/*/*/; do \
        [ -f "$tool/pyproject.toml" ] || continue; \
        grep -q "^\[tool.sadt\]" "$tool/pyproject.toml" || continue; \
        echo "==> uv sync $(basename "$tool")"; \
        # --all-extras, because an extra is where a tool puts an engine it does
        # not want CI to build on every PR -- Crown_Seg's shapeaxi, behind
        # `segmentation`. Skipping them builds a virtualenv that imports and then
        # answers 503 on the first real request. There is nothing optional about
        # them in a deployment image: this is the machine that runs the tool.
        ( cd "$tool" && uv sync --frozen --no-install-project --no-dev --all-extras ); \
    done; \
    cp -a "$UV_CACHE_DIR"/. /cache/uv/ 2>/dev/null || true; \
    uv cache prune --ci >/dev/null 2>&1 || true; \
    rm -rf "$UV_CACHE_DIR"

# The sources, last: a tool's code changes far more often than its lock file,
# and this is the only layer that has to be rebuilt when it does. `--frozen
# --no-install-project` above is why the venvs could be built without them --
# a tool's code is not installed as a package, runner.py puts src/ on sys.path.
# --exclude, and it is not an optimisation: the tools context is a working
# checkout, where every tool already has a .venv of its own -- 7.2-7.7 GB each
# at cu128. Copying those in would bury the deduplicated ones this image just
# built, under a mountain of layers.
COPY --from=tools --exclude=**/.venv --exclude=**/__pycache__ tools /tools/
# The schema generator, from the same repository as the tools. It reads the
# schema out of `run()`'s signature, so the two cannot drift; deployed beside
# the tools rather than inside the server because it belongs to them.
COPY --from=tools scripts /tools/scripts/

# ---- the schemas -----------------------------------------------------
# Generated at build time, each with ITS OWN tool's interpreter: importing a
# tool needs the tool's dependencies, which is the whole reason the server does
# not do it. The result is a cache -- `source_hash` is what lets the server
# notice one has gone stale and regenerate it -- so a failure here is a tool
# that will be described at startup instead, or reported as not loadable.
RUN set -eu; \
    [ -f /tools/scripts/describe.py ] || { echo "no describe.py: tools must ship .schema.json"; exit 0; }; \
    for tool in /tools/*/ /tools/*/*/; do \
        name=$(basename "$tool"); \
        [ -x "$tool/.venv/bin/python" ] || continue; \
        if "$tool/.venv/bin/python" /tools/scripts/describe.py "$tool" > "$tool/.schema.json.tmp"; then \
            mv "$tool/.schema.json.tmp" "$tool/.schema.json"; \
            echo "==> described $name"; \
        else \
            rm -f "$tool/.schema.json.tmp"; \
            echo "!!! $name could not describe itself; it will not be served" >&2; \
        fi; \
    done

# ---- the server ------------------------------------------------------
# server/tools/ is excluded in .dockerignore: this image serves tools from
# /tools, and importing the in-process ones would need the very stack the API
# venv is built to do without.
# runner.py rides along in server/execution/. It is executed BY a tool's
# interpreter and handed over by path, so it is never installed into a tool
# venv -- runner and server are then always the same version.
COPY server/ /opt/sadt/server/
COPY docker/verify_dedup.py /opt/sadt/verify_dedup.py

ENV SADT_DISPATCH_MODE=subprocess \
    TOOLS_DIR=/tools \
    RUNNER_PATH=/opt/sadt/server/execution/runner.py \
    DESCRIBE_PATH=/tools/scripts/describe.py \
    SCHEMA_CACHE_DIR=/var/cache/sadt \
    # Next to the server, which is where COPY server/ actually puts it and
    # what config.py defaults to. Pointed at /opt/sadt/ it named a file that
    # never existed, so every packaged tool fell back to its own name for
    # the DATA lookup: `amasss` instead of `AMASSS`, no models, no test
    # files, and a `model` argument the client could not fill in.
    DEPLOYMENT_CONFIG=/opt/sadt/server/deployment.toml \
    DATA_DIR=/DATA \
    TEMP_DIR=/jobs \
    PYTHONDONTWRITEBYTECODE=1

# Third-party code from fifteen upstreams runs in this container, on
# confidential imaging. It does not run as root.
RUN useradd --system --create-home --home-dir /home/sadt sadt && \
    install -d -o sadt -g sadt /jobs && \
    install -d -o sadt -g sadt /DATA && \
    # Writable, because a schema regenerated at startup is cached here.
    # /tools stays root-owned and read-only to the server: third-party code
    # must not be able to rewrite its own virtualenv.
    install -d -o sadt -g sadt /var/cache/sadt

USER sadt
WORKDIR /opt/sadt/server
EXPOSE 8000

# No --reload: this image is the artifact, not a mount of a working tree.
CMD ["/opt/sadt/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
