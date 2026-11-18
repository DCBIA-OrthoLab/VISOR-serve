#!/usr/bin/env bash
# Run the server NATIVELY on this machine: no Docker, so on a Mac the tools can
# reach the Apple GPU (Docker Desktop, OrbStack and Apple's `container` cannot
# give a container access to MPS).
#
# Not a replacement for run-local.sh, which is untouched: that one is the
# developer's quick CUDA server on port 8001 with a venv created by hand and a
# hard-coded ~/code/SADT-VISOR. This one creates what it needs, binds
# loopback, and is what `server_ctl.py up --native` launches.
#
#   scripts/run-native.sh               # foreground, until Ctrl-C
#   scripts/run-native.sh --setup-only  # create/refresh the API venv, then stop
#
# Inputs, all from the environment (server_ctl.py sets them from .env):
#   API_TOKEN     required unless --setup-only. Never printed.
#   SADT_TOOLS    a SADT-VISOR checkout (default: ../SADT-VISOR next to this repo)
#   DEVICE        mps on an Apple-silicon Mac, cuda if nvidia-smi exists, else cpu
#   HOST_PORT     default 8000
#   BIND_ADDR     default 127.0.0.1. Plain HTTP unless VISOR_TLS_* is set: keep it on loopback.
#   VISOR_TLS_CERT, VISOR_TLS_KEY
#                 PEM files (both or neither). Set: uvicorn serves https, and only
#                 on 127.0.0.1. The key is passed by path, never read or printed here.
#   DATA_DIR      default <repo>/DATA
#   VISOR_STATE_DIR  where the API venv and the schema cache live
#                 (macOS: ~/Library/Application Support/VISOR/native,
#                  elsewhere: ~/.local/state/visor/native)
#
# bash 3.2 compatible: that is what a Mac ships.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
OS="$(uname -s)"

# Rosetta trap: a shell started from an x86_64 process (a Slicer under Rosetta,
# an old terminal) hands x86_64 to every child, and PyTorch's MPS needs arm64.
if [ "$OS" = "Darwin" ] && [ "$(uname -m)" != "arm64" ] \
        && [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = "1" ]; then
    exec arch -arm64 "$0" "$@"
fi

if [ -z "${VISOR_STATE_DIR:-}" ]; then
    if [ "$OS" = "Darwin" ]; then
        VISOR_STATE_DIR="$HOME/Library/Application Support/VISOR/native"
    else
        VISOR_STATE_DIR="$HOME/.local/state/visor/native"
    fi
fi
mkdir -p "$VISOR_STATE_DIR"

UV="${UV:-$(command -v uv || true)}"
if [ -z "$UV" ]; then
    for candidate in "$HOME/.local/bin/uv" /opt/homebrew/bin/uv /usr/local/bin/uv; do
        [ -x "$candidate" ] && UV="$candidate" && break
    done
fi
if [ -z "$UV" ]; then
    echo "uv was not found. Install it (https://docs.astral.sh/uv/) and run this again." >&2
    exit 1
fi

# --- the API venv: the newest Python, fastapi and nothing else ---------------
API_VENV="$VISOR_STATE_DIR/api-venv"
REQS="$REPO/server/requirements-api.txt"
STAMP="$API_VENV/.requirements.sha"
want="$(shasum -a 256 "$REQS" | cut -d' ' -f1)"
if [ ! -x "$API_VENV/bin/python" ] || [ "$(cat "$STAMP" 2>/dev/null || true)" != "$want" ]; then
    echo "Preparing the API environment in $API_VENV ..." >&2
    [ -x "$API_VENV/bin/python" ] || "$UV" venv --python 3.13 "$API_VENV" >&2
    "$UV" pip install --python "$API_VENV/bin/python" -r "$REQS" >&2
    echo "$want" > "$STAMP"
fi
if [ "${1:-}" = "--setup-only" ]; then
    echo "$API_VENV"
    exit 0
fi

# --- the server's environment ------------------------------------------------
: "${API_TOKEN:?API_TOKEN must be set (server_ctl.py up --native sets it)}"
SADT_TOOLS="${SADT_TOOLS:-$(dirname "$REPO")/SADT-VISOR}"
if [ ! -d "$SADT_TOOLS/tools" ]; then
    echo "No tools folder at $SADT_TOOLS/tools. Set SADT_TOOLS to a SADT-VISOR checkout." >&2
    exit 1
fi
PORT="${HOST_PORT:-8000}"
BIND="${BIND_ADDR:-127.0.0.1}"

# Loopback only, with or without TLS: plain HTTP on a routable address would
# give the token and the scans to the LAN.
if [ "$BIND" != "127.0.0.1" ]; then
    echo "The native server listens on 127.0.0.1 only (BIND_ADDR is '$BIND'), with or without TLS." >&2
    exit 1
fi

# Local TLS: both files or neither, and only on loopback (the certificate the
# client pins names 127.0.0.1, ::1 and localhost, nothing routable).
SCHEME=http
TLS_ARGS=()
if [ -n "${VISOR_TLS_CERT:-}${VISOR_TLS_KEY:-}" ]; then
    if [ -z "${VISOR_TLS_CERT:-}" ] || [ -z "${VISOR_TLS_KEY:-}" ]; then
        echo "VISOR_TLS_CERT and VISOR_TLS_KEY go together." >&2
        exit 1
    fi
    if [ "$BIND" != "127.0.0.1" ]; then
        echo "TLS is only served on 127.0.0.1 (BIND_ADDR is '$BIND')." >&2
        exit 1
    fi
    if [ ! -f "$VISOR_TLS_CERT" ] || [ ! -f "$VISOR_TLS_KEY" ]; then
        echo "The TLS certificate or key file is missing." >&2
        exit 1
    fi
    SCHEME=https
    TLS_ARGS=(--ssl-certfile "$VISOR_TLS_CERT" --ssl-keyfile "$VISOR_TLS_KEY")
fi
# The tools inherit this environment and have no use for the key's location.
unset VISOR_TLS_CERT VISOR_TLS_KEY

if [ -z "${DEVICE:-}" ]; then
    if [ "$OS" = "Darwin" ]; then DEVICE=mps
    elif command -v nvidia-smi >/dev/null 2>&1; then DEVICE=cuda
    else DEVICE=cpu; fi
fi

export API_TOKEN DEVICE
export TOOLS_DIR="$SADT_TOOLS/tools"
export DESCRIBE_PATH="$SADT_TOOLS/scripts/describe.py"
export DATA_DIR="${DATA_DIR:-$REPO/DATA}"
export SCHEMA_CACHE_DIR="${SCHEMA_CACHE_DIR:-$VISOR_STATE_DIR/schema-cache}"
# config.py defaults TEMP_DIR to /tmp/inference_server, which holds the runs/ registry:
# two native servers of one user (two states, two ports) would share and reap each
# other's runs. Keep it with the rest of this instance's state.
export TEMP_DIR="${TEMP_DIR:-$VISOR_STATE_DIR/tmp}"
# config.py defaults to `inprocess`, which imports every tool into the API
# process: the Dockerfile forces `subprocess` and so does this.
export SADT_DISPATCH_MODE=subprocess
export SADT_API="$SCHEME://127.0.0.1:$PORT"
# The tools' own downloads (torch hub: the EfficientNet backbone DOCShapeAXI
# needs) are staged by `install-tools` under the state folder, not in ~/.cache.
export TORCH_HOME="$VISOR_STATE_DIR/torch_home"
export PYTHONUNBUFFERED=1
mkdir -p "$DATA_DIR" "$SCHEMA_CACHE_DIR" "$TEMP_DIR"

if [ "$DEVICE" = "mps" ]; then
    # Some 3D ops are missing on MPS (adaptive pooling with an output above 1,
    # and on torch 2.8 max/avg pool and grid_sample): fall back to the CPU for
    # that op instead of failing the run. Measured on the M4 Max.
    export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
fi

# macOS gives a process 256 open files. The server holds one per upload chunk
# and per log, and a tool opens a file per case. Raised for this process and
# inherited by every tool it starts; the hard limit is the ceiling.
ulimit -S -n 10240 2>/dev/null || ulimit -S -n "$(ulimit -Hn)" 2>/dev/null || true

echo "tools   : $TOOLS_DIR" >&2
echo "device  : $DEVICE" >&2
echo "listen  : $SCHEME://$BIND:$PORT" >&2
echo "open files limit: $(ulimit -n)" >&2

cd "$REPO/server"
# ${arr[@]+...}: an empty array is an unbound variable to bash 3.2 under `set -u`.
exec "$API_VENV/bin/python" -m uvicorn main:app --host "$BIND" --port "$PORT" \
    ${TLS_ARGS[@]+"${TLS_ARGS[@]}"}
