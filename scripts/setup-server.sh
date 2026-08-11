#!/bin/sh
# Clone this repository, make sure docker is usable, and start the server.
#
# From a machine with nothing checked out:
#
#   curl -fsSL https://raw.githubusercontent.com/Jules-GP/slicer-remote-tool-server/main/scripts/setup-server.sh | sh
#
# Options reach this script through `sh -s --` when piping:
#
#   curl -fsSL .../setup-server.sh | sh -s -- --tool AMASSS --tool ALI
#
# Options:
#   --dir DIR       where to clone (default: ./slicer-remote-tool-server, or $INSTALL_DIR)
#   --tool NAME     also download this tool's models and test files (repeatable).
#                   Nothing is downloaded by default: the full set is ~29 GB, and
#                   which tools a given site uses is not something to assume.
#   --device gpu|cpu   force the compose service instead of detecting it
#   --bind ADDR     host address the port is published on (default: 127.0.0.1)
#   --port N        host port to publish on (default: 8000). Only needed when
#                   something else already holds it; it is remembered in .env.
#   --no-start      clone and check prerequisites, but do not start anything
#
# Environment:
#   REPO/REF        fork / branch to clone (default: this repo, main)
#   REPO_URL        the clone URL outright, when REPO's github.com/<owner>/<name>
#                   shape does not fit (a mirror, an ssh remote, a local path)
#   INSTALL_DIR     same as --dir
#
# Re-running is safe: an existing clone is updated rather than re-cloned, and
# the API token already in .env is kept, so clients configured against this
# server keep working.

set -eu

REPO="${REPO:-Jules-GP/slicer-remote-tool-server}"
REF="${REF:-main}"
REPO_URL="${REPO_URL:-https://github.com/${REPO}.git}"
INSTALL_DIR="${INSTALL_DIR:-./slicer-remote-tool-server}"
TOOLS=""
DEVICE="auto"
BIND="127.0.0.1"
PORT=""
START=1

while [ $# -gt 0 ]; do
    case "$1" in
        --dir) INSTALL_DIR="$2"; shift 2 ;;
        --tool) TOOLS="$TOOLS --tool $2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --bind) BIND="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --no-start) START=0; shift ;;
        -h|--help) sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "setup-server: unknown option '$1'" >&2; exit 2 ;;
    esac
done

for tool in git python3; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "setup-server: $tool is required but was not found in PATH." >&2
        echo "  Debian/Ubuntu: sudo apt-get install -y $tool" >&2
        echo "  Fedora/RHEL:   sudo dnf install -y $tool" >&2
        exit 1
    fi
done

# --- the clone -----------------------------------------------------------
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "Updating the existing clone in $INSTALL_DIR ..."
    git -C "$INSTALL_DIR" fetch --quiet origin
    # --ff-only, never a merge: a clone someone has edited locally must fail
    # loudly here rather than have this script invent a merge commit in it.
    if ! git -C "$INSTALL_DIR" pull --ff-only; then
        echo "setup-server: could not fast-forward $INSTALL_DIR." >&2
        echo "  It has local commits or uncommitted changes. Resolve them, then re-run." >&2
        exit 1
    fi
elif [ -e "$INSTALL_DIR" ]; then
    echo "setup-server: $INSTALL_DIR exists and is not a git clone. Move it aside" >&2
    echo "  or pass --dir with somewhere else." >&2
    exit 1
else
    echo "Cloning ${REPO_URL}@${REF} into $INSTALL_DIR ..."
    git clone --branch "$REF" "$REPO_URL" "$INSTALL_DIR"
fi

CTL="$INSTALL_DIR/scripts/server_ctl.py"
if [ ! -f "$CTL" ]; then
    echo "setup-server: $CTL is missing -- ${REPO}@${REF} does not carry it." >&2
    exit 1
fi

# --- docker --------------------------------------------------------------
if ! docker info >/dev/null 2>&1; then
    echo
    echo "Docker is not installed, or its daemon is not reachable by this user."
    echo "Install it with:"
    echo
    echo "    sudo sh $INSTALL_DIR/scripts/install-docker.sh"
    echo
    echo "then log out and back in (group membership only applies to a new session)"
    echo "and re-run this script. It will pick up where it left off."
    exit 1
fi

# --- models --------------------------------------------------------------
# Before starting the server, not after: the tools that have no weights on
# disk answer 422 rather than failing mysteriously, so it is better to know
# what is missing while someone is still watching the terminal.
if [ -n "$TOOLS" ]; then
    # shellcheck disable=SC2086 -- $TOOLS is a deliberately word-split option list
    python3 "$CTL" models $TOOLS
fi

# --- start ---------------------------------------------------------------
if [ "$START" -eq 0 ]; then
    python3 "$CTL" status --device "$DEVICE"
    exit 0
fi

if [ -n "$PORT" ]; then
    python3 "$CTL" up --device "$DEVICE" --bind "$BIND" --port "$PORT"
else
    python3 "$CTL" up --device "$DEVICE" --bind "$BIND"
fi

echo
echo "Point the Slicer client at this server with:"
echo "    URL    http://localhost:${PORT:-8000}"
echo "    token  $(python3 "$CTL" token)"
echo
echo "Or open the 'Slicer Cloud' module in Slicer, which does all of the above"
echo "(clone, start, update, model selection) from a panel."
echo
echo "This deployment listens on ${BIND} over plain HTTP. That is fine for"
echo "localhost; putting it on a network address requires a TLS terminator in"
echo "front -- see SECURITY.md."
