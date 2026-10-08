#!/bin/sh
# Clone this repository, make sure docker is usable, and start the server.
#
# From a machine with nothing checked out:
#
#   curl -fsSL https://raw.githubusercontent.com/DCBIA-OrthoLab/VISOR-serve/deploy/scripts/setup-server.sh | sh
#
# Options reach this script through `sh -s --` when piping:
#
#   curl -fsSL .../setup-server.sh | sh -s -- --tool AMASSS --tool ALI
#
# Options:
#   --dir DIR       where to clone (default: ./VISOR-serve, or $INSTALL_DIR)
#   --tool NAME     also download this tool's models and test files (repeatable).
#                   Nothing is downloaded by default: the full set is ~29 GB, and
#                   which tools a given site uses is not something to assume.
#   --device gpu|cpu   force the compose service instead of detecting it
#   --bind ADDR     host address the port is published on (default: 127.0.0.1)
#   --port N        host port to publish on (default: 8000). Only needed when
#                   something else already holds it; it is remembered in .env.
#   --no-start      clone and check prerequisites, but do not start anything
#   --full          download EVERY tool's models and test files (~29 GB). The
#                   same as naming every tool with --tool, and the honest
#                   default for a site that does not yet know what it uses.
#   --token VALUE   set the API token instead of keeping or generating one
#   --auto-update MODE   off | notify | apply. Written to .env as
#                   SADT_AUTO_UPDATE, which `server_ctl.py watch` reads. See
#                   scripts/visor-update.service for the unit that runs it.
#   --branch NAME   the branch this deployment follows (default: deploy). Both
#                   what is cloned and what `watch` fast-forwards to. `deploy`
#                   is a commit of `main` whose CI went green and which was
#                   then promoted on purpose, so a deployment never picks up
#                   whatever landed on `main` an hour ago. Pass `main` to
#                   follow the tip instead. A fork carrying no such branch
#                   falls back to its default one, saying so.
#   --tools URL     also clone a TOOLS repository and point this deployment at
#                   it, so the server serves real tools instead of the two
#                   built-in demos. A git URL, and there is deliberately no
#                   default: this server imports nothing from the tools and
#                   knows no dental tool, so naming one repository in here
#                   would be the one piece of that knowledge it holds. The
#                   operator says which tools this deployment serves.
#   --tools-dir DIR where to clone it (default: beside the server clone)
#   --tools-ref REF which branch of the tools repository to clone (default:
#                   the same name as --branch if that repository has it, its
#                   own default branch otherwise). The counterpart of
#                   --branch, and defaulted to the same name on purpose: the
#                   two sides are promoted together, and a server following
#                   `deploy` while the tools follow their tip is how a schema
#                   and the server that reads it come apart.
#   --build-tools   build each tool's virtualenv after cloning, which is what
#                   makes them actually RUN. ~25 GB and well over an hour, it
#                   downloads two CUDA torch runtimes. Implied by --full.
#                   Installs `uv` from astral.sh if it is not on PATH.
#   --tailscale     also put the server on a Tailscale network, reachable as
#                   https://<name>.<tailnet>.ts.net from the tailnet's machines
#                   and from nowhere else (scripts/setup-tailscale.sh; needs
#                   sudo). Forces --bind 127.0.0.1: Tailscale is then the only
#                   way in, and it brings the TLS. Off unless asked for.
#   --tailscale-key-file PATH
#                   the Tailscale auth key, read from this file (a file holding
#                   only the key). Without it the key is asked for at the
#                   prompt, hidden, or taken from TS_AUTHKEY. Never accepted
#                   on the command line, where `ps` and the history keep it.
#   --yes           never ask anything; take the defaults and the options given
#
# Environment:
#   REPO/REF        fork / branch to clone (default: this repo, deploy)
#   REPO_URL        the clone URL outright, when REPO's github.com/<owner>/<name>
#                   shape does not fit (a mirror, an ssh remote, a local path)
#   INSTALL_DIR     same as --dir
#   TS_AUTHKEY      the Tailscale auth key, for an unattended --tailscale
#
# Re-running is safe: an existing clone is updated rather than re-cloned, and
# the API token already in .env is kept, so clients configured against this
# server keep working.

set -eu

REPO="${REPO:-DCBIA-OrthoLab/VISOR-serve}"
REF="${REF:-deploy}"
REPO_URL="${REPO_URL:-https://github.com/${REPO}.git}"
INSTALL_DIR="${INSTALL_DIR:-./VISOR-serve}"
TOOLS=""
DEVICE="auto"
BIND="127.0.0.1"
PORT=""
START=1
FULL=0
TOKEN=""
AUTO_UPDATE=""
ASK=1
TOOLS_REPO=""
TOOLS_DIR_OPT=""
TOOLS_REF="${TOOLS_REF:-}"
BUILD_TOOLS=0
TAILSCALE=0
TS_KEY_FILE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --dir) INSTALL_DIR="$2"; shift 2 ;;
        --tool) TOOLS="$TOOLS --tool $2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --bind) BIND="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --no-start) START=0; shift ;;
        --full) FULL=1; shift ;;
        --token) TOKEN="$2"; shift 2 ;;
        --auto-update) AUTO_UPDATE="$2"; shift 2 ;;
        --branch) REF="$2"; shift 2 ;;
        --tools) TOOLS_REPO="$2"; shift 2 ;;
        --tools-dir) TOOLS_DIR_OPT="$2"; shift 2 ;;
        --tools-ref) TOOLS_REF="$2"; shift 2 ;;
        --build-tools) BUILD_TOOLS=1; shift ;;
        --tailscale) TAILSCALE=1; shift ;;
        --tailscale-key-file) TAILSCALE=1; TS_KEY_FILE="$2"; shift 2 ;;
        --yes|-y|--non-interactive) ASK=0; shift ;;
        -h|--help) sed -n '2,76p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "setup-server: unknown option '$1'" >&2; exit 2 ;;
    esac
done

# --- the questions -------------------------------------------------------
# Only on a terminal, and read from /dev/tty rather than stdin: the documented
# way to run this is `curl ... | sh`, where stdin IS the script and a `read`
# would swallow the rest of it. A machine, a CI job or --yes takes the
# defaults and never blocks.
ask() {
    # ask VARIABLE "question" "default"
    eval "_current=\${$1}"
    if [ -n "$_current" ] || [ "$ASK" -eq 0 ] || [ ! -r /dev/tty ] || [ ! -t 1 ]; then
        [ -n "$_current" ] || eval "$1=\$3"
        return 0
    fi
    printf '%s [%s]: ' "$2" "$3" > /dev/tty
    read -r _answer < /dev/tty || _answer=""
    [ -n "$_answer" ] || _answer="$3"
    eval "$1=\$_answer"
}

if [ "$ASK" -eq 1 ] && [ -r /dev/tty ] && [ -t 1 ]; then
    echo
    echo "Answer, or press Enter to take the default in brackets."
    echo
fi

ask PORT "Host port to publish the server on" "8000"

# Tailscale decides the address, so it is settled first. Only --tailscale turns
# it on: an operator who wants it says so, and nothing is asked otherwise.
if [ "$TAILSCALE" -eq 1 ]; then
    # Anything else would leave the same port open in plain HTTP beside the
    # tailnet, which is the one thing Tailscale was brought in to prevent.
    if [ "$BIND" != "127.0.0.1" ]; then
        echo "setup-server: --tailscale publishes through Tailscale only; binding 127.0.0.1, not $BIND." >&2
    fi
    BIND="127.0.0.1"
else
    ask BIND "Host address to publish it on (127.0.0.1 keeps it off the network)" "127.0.0.1"
fi
ask REF  "Branch this deployment follows" "$REF"
ask AUTO_UPDATE "Follow that branch automatically? off | notify | apply" "off"

# Without a tools repository the server starts with its two demo tools and
# nothing else, which reads as a small deployment rather than an unfinished
# one. Asked as a URL rather than offered as a default: see --tools.
if [ -z "$TOOLS_REPO" ] && [ "$ASK" -eq 1 ] && [ -r /dev/tty ] && [ -t 1 ]; then
    printf 'Git URL of the tools repository to serve (empty: the demos only): ' > /dev/tty
    read -r TOOLS_REPO < /dev/tty || TOOLS_REPO=""
fi

case "$AUTO_UPDATE" in
    off|notify|apply) ;;
    *) echo "setup-server: --auto-update must be off, notify or apply (got '$AUTO_UPDATE')." >&2; exit 2 ;;
esac

# The token is what the Slicer client and the dashboard authenticate with.
# Left empty, server_ctl.py generates one and a re-run keeps it, so configured
# clients keep working -- which is why this does not default to a new value.
if [ -z "$TOKEN" ] && [ "$ASK" -eq 1 ] && [ -r /dev/tty ] && [ -t 1 ]; then
    printf 'API token for Slicer and the dashboard [keep existing, else generate]: ' > /dev/tty
    read -r TOKEN < /dev/tty || TOKEN=""
fi

# What to download. Asked as one question rather than per tool: the full set is
# ~29 GB and naming eighteen tools at a prompt is not a thing anyone does.
if [ "$FULL" -eq 0 ] && [ -z "$TOOLS" ] && [ "$ASK" -eq 1 ] && [ -r /dev/tty ] && [ -t 1 ]; then
    printf 'Download models and test files for every tool now? ~29 GB [y/N]: ' > /dev/tty
    read -r _all < /dev/tty || _all=""
    case "$_all" in y|Y|yes|YES) FULL=1 ;; esac
fi

# A server whose tools have no virtualenv reports every one of them as
# unloadable, which reads as a broken deployment rather than an unbuilt one.
[ "$FULL" -eq 1 ] && BUILD_TOOLS=1
if [ "$BUILD_TOOLS" -eq 0 ] && [ -n "$TOOLS_REPO" ] && [ "$ASK" -eq 1 ] && [ -r /dev/tty ] && [ -t 1 ]; then
    printf 'Build each tool virtualenv now? Needed to run them. ~25 GB, over an hour [y/N]: ' > /dev/tty
    read -r _bt < /dev/tty || _bt=""
    case "$_bt" in y|Y|yes|YES) BUILD_TOOLS=1 ;; esac
fi

# A branch name is a DEFAULT here, not a demand: this script is curl-piped at
# forks and mirrors, and `git clone --branch deploy` against one that promotes
# nothing exits 128 with the whole install undone. Asking the remote first
# turns that into one printed line and the default branch.
remote_branch_or_default() {
    # $1 remote URL, $2 wanted branch. Echoes the branch to clone, or nothing
    # for "let git pick", and explains itself on stderr when it gives up.
    if [ -z "$2" ]; then return 0; fi
    if git ls-remote --exit-code --heads "$1" "$2" >/dev/null 2>&1; then
        echo "$2"
    else
        echo "setup-server: $1 has no '$2' branch; taking its default one." >&2
    fi
}

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
    CLONE_REF="$(remote_branch_or_default "$REPO_URL" "$REF")"
    if [ -n "$CLONE_REF" ]; then
        echo "Cloning ${REPO_URL}@${CLONE_REF} into $INSTALL_DIR ..."
        git clone --branch "$CLONE_REF" "$REPO_URL" "$INSTALL_DIR"
    else
        echo "Cloning ${REPO_URL} into $INSTALL_DIR ..."
        git clone "$REPO_URL" "$INSTALL_DIR"
        REF="$(git -C "$INSTALL_DIR" rev-parse --abbrev-ref HEAD)"
    fi
fi

CTL="$INSTALL_DIR/scripts/server_ctl.py"
if [ ! -f "$CTL" ]; then
    echo "setup-server: $CTL is missing -- ${REPO}@${REF} does not carry it." >&2
    exit 1
fi

# --- how far behind the tip this deployment is ---------------------------
# A deployment branch is behind `main` on purpose; the failure mode is silence
# about HOW far. A `deploy` nobody promoted for three months reads exactly
# like one promoted this morning, and the operator who has to decide whether a
# fix is in their deployment has no way to tell. Reported, never enforced:
# whether to promote is the maintainer's call, not this script's.
if [ "$REF" != "main" ]; then
    git -C "$INSTALL_DIR" fetch --quiet origin main 2>/dev/null || true
    BEHIND="$(git -C "$INSTALL_DIR" rev-list --count HEAD..FETCH_HEAD 2>/dev/null || true)"
    TIP_DATE="$(git -C "$INSTALL_DIR" log -1 --format=%cs FETCH_HEAD 2>/dev/null || true)"
    case "$BEHIND" in
        ''|*[!0-9]*) ;;
        0) echo "Following '${REF}', which is level with main." ;;
        *) echo "Following '${REF}', ${BEHIND} commit(s) behind main (tip ${TIP_DATE})." ;;
    esac
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

# --- settings that outlive this run --------------------------------------
# Merged line by line, never rewritten: an operator's own additions to .env
# must survive, which is the same rule server_ctl.py's write_env follows.
set_env() {
    _file="$INSTALL_DIR/.env"
    [ -f "$_file" ] || { printf '%s\n' "# Written by scripts/setup-server.sh." > "$_file"; }
    if grep -q "^$1=" "$_file" 2>/dev/null; then
        _tmp="$_file.tmp.$$"
        sed "s|^$1=.*|$1=$2|" "$_file" > "$_tmp" && mv "$_tmp" "$_file"
    else
        printf '%s=%s\n' "$1" "$2" >> "$_file"
    fi
}

# --- the tools -----------------------------------------------------------
# A separate repository on purpose: this server imports nothing from it and
# knows no dental tool. It is cloned, mounted at its own path, and served.
TOOLS_ROOT=""
if [ -n "$TOOLS_REPO" ]; then
    if [ -n "$TOOLS_DIR_OPT" ]; then
        TOOLS_ROOT="$TOOLS_DIR_OPT"
    else
        # Named after the repository, never after a repository this script
        # knows: `--tools .../Foo.git` clones into ./Foo beside the server.
        _tools_name="$(basename "$TOOLS_REPO")"
        _tools_name="${_tools_name%.git}"
        TOOLS_ROOT="$(dirname "$INSTALL_DIR")/${_tools_name}"
    fi
    if [ -d "$TOOLS_ROOT/.git" ]; then
        echo "Updating the tools clone in $TOOLS_ROOT ..."
        git -C "$TOOLS_ROOT" fetch --quiet origin || true
        git -C "$TOOLS_ROOT" pull --ff-only || {
            echo "setup-server: could not fast-forward $TOOLS_ROOT; leaving it as it is." >&2
        }
    elif [ -e "$TOOLS_ROOT" ]; then
        echo "setup-server: $TOOLS_ROOT exists and is not a git clone." >&2
        echo "  Move it aside or pass --tools-dir with somewhere else." >&2
        exit 1
    else
        echo "Cloning the tools from $TOOLS_REPO into $TOOLS_ROOT ..."
        # Unset means "follow the server", not "follow the tip".
        TOOLS_CLONE_REF="$(remote_branch_or_default "$TOOLS_REPO" "${TOOLS_REF:-$REF}")"
        if [ -n "$TOOLS_CLONE_REF" ]; then
            git clone --branch "$TOOLS_CLONE_REF" "$TOOLS_REPO" "$TOOLS_ROOT"
        else
            git clone "$TOOLS_REPO" "$TOOLS_ROOT"
        fi
    fi
    TOOLS_ROOT="$(cd "$TOOLS_ROOT" && pwd)"
fi

# --- models and test files -----------------------------------------------
# Before starting the server, not after: a tool with no weights on disk
# answers 422 rather than failing mysteriously, so it is better to know what
# is missing while someone is still watching the terminal.
#
# Delegated to the TOOLS repository's own script: the manifest listing which
# bundle belongs to which tool is dental knowledge and lives beside the tools.
# Falls back to this repo's copy while both exist, so an older tools checkout
# still works.
fetch_data() {
    if [ -n "$TOOLS_ROOT" ] && [ -f "$TOOLS_ROOT/scripts/setup-data.sh" ]; then
        ( cd "$TOOLS_ROOT" && sh scripts/setup-data.sh --data-dir "$(cd "$INSTALL_DIR" && pwd)/DATA" "$@" )
    else
        # shellcheck disable=SC2086
        python3 "$CTL" models "$@"
    fi
}

if [ "$FULL" -eq 1 ]; then
    echo
    echo "Downloading every tool's models and test files (~31 GB)."
    echo "Whatever is already on disk is skipped, so this is resumable."
    fetch_data
elif [ -n "$TOOLS" ]; then
    # shellcheck disable=SC2086 -- $TOOLS is a deliberately word-split option list
    fetch_data $TOOLS
fi

# --- the tool virtualenvs ------------------------------------------------
# One per tool, built from its own committed lockfile. They are what makes a
# tool runnable: TOOLS_DIR pointing at a tree with no `.venv` registers
# nothing. A tool that fails to build is REPORTED and the others carry on --
# the same rule the registry follows, because with eighteen tools one missing
# dependency must not cost the other seventeen.
if [ "$BUILD_TOOLS" -eq 1 ] && [ -n "$TOOLS_ROOT" ]; then
    if ! command -v uv >/dev/null 2>&1; then
        echo
        echo "Installing uv (the tools' package manager) from astral.sh ..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Its installer puts uv here and prints a line about restarting the
        # shell, which a non-interactive run cannot do.
        PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        export PATH
    fi
    if ! command -v uv >/dev/null 2>&1; then
        echo "setup-server: uv is still not on PATH; cannot build the tool virtualenvs." >&2
        echo "  Install it yourself (https://docs.astral.sh/uv/) and re-run with --build-tools." >&2
    else
        echo
        echo "Building each tool's virtualenv. This downloads two CUDA torch"
        echo "runtimes and lands at roughly 25 GB; it is resumable, so a"
        echo "re-run only does what is missing."
        _failed=""
        for _pyproject in "$TOOLS_ROOT"/tools/*/pyproject.toml "$TOOLS_ROOT"/tools/*/*/pyproject.toml; do
            [ -f "$_pyproject" ] || continue
            grep -q '^\[tool.sadt\]' "$_pyproject" || continue   # a tool, not a shared package
            _tool_dir="$(dirname "$_pyproject")"
            _tool_name="$(basename "$_tool_dir")"
            printf '  %-20s ' "$_tool_name"
            # --all-extras, as the image builds them: Crown_Seg's segmentation
            # engine is an extra, and without it every mesh that is not already
            # labelled is refused (AREG IOS included).
            if ( cd "$_tool_dir" && uv sync --frozen --all-extras --quiet ) >/dev/null 2>&1; then
                echo "built"
            else
                echo "FAILED"
                _failed="$_failed $_tool_name"
            fi
        done
        if [ -n "$_failed" ]; then
            echo
            echo "These tools did not build and will not be served:$_failed"
            echo "  Re-run the one that matters from its own folder to see why:"
            echo "      cd $TOOLS_ROOT/tools/<name> && uv sync --frozen --all-extras"
        fi
    fi
fi

# --- point the deployment at the tools -----------------------------------
if [ -n "$TOOLS_ROOT" ]; then
    set_env SADT_TOOLS "$TOOLS_ROOT"
    # The interpreters those virtualenvs symlink to. uv installs its own
    # CPython outside the checkout, so without this every `.venv/bin/python`
    # inside the container is a dangling link.
    for candidate in "${UV_PYTHON_INSTALL_DIR:-}" "$HOME/.local/share/uv/python"; do
        [ -n "$candidate" ] && [ -d "$candidate" ] && { set_env UV_PYTHON_STORE "$candidate"; break; }
    done
fi

[ -n "$TOKEN" ] && set_env API_TOKEN "$TOKEN"
[ -n "$AUTO_UPDATE" ] && set_env SADT_AUTO_UPDATE "$AUTO_UPDATE"

# --- start ---------------------------------------------------------------
if [ "$START" -eq 0 ]; then
    python3 "$CTL" status --device "$DEVICE" --branch "$REF"
    exit 0
fi

if [ -n "$PORT" ]; then
    python3 "$CTL" up --device "$DEVICE" --bind "$BIND" --port "$PORT" --branch "$REF"
else
    python3 "$CTL" up --device "$DEVICE" --bind "$BIND" --branch "$REF"
fi

echo
echo "Point the Slicer client at this server with:"
echo "    URL    http://localhost:${PORT:-8000}"
echo "    token  $(python3 "$CTL" token)"
echo
echo "Or open the 'Slicer Cloud' module in Slicer, which does all of the above"
echo "(clone, start, update, model selection) from a panel."
echo
echo "Dashboard   http://localhost:${PORT:-8000}/server-debug   (the same token)"
echo "Benchmarks  http://localhost:${PORT:-8000}/benchmark"

if [ "$AUTO_UPDATE" != "off" ] && [ -n "$AUTO_UPDATE" ]; then
    echo
    echo "SADT_AUTO_UPDATE is '${AUTO_UPDATE}', which only takes effect once the"
    echo "updater is running. It is a systemd unit, not part of this script:"
    echo
    echo "    sudo cp $INSTALL_DIR/scripts/visor-update.service /etc/systemd/system/"
    echo "    sudo systemctl daemon-reload"
    echo "    sudo systemctl enable --now visor-update"
    echo
    echo "It polls '${REF}' and never interrupts a run in flight; 'notify' reports"
    echo "what it would do, 'apply' fast-forwards and restarts between runs."
fi

if [ "$TAILSCALE" -eq 1 ]; then
    echo
    echo "--- Tailscale ---"
    _ts_args="--port ${PORT:-8000} --dir $INSTALL_DIR"
    [ "$ASK" -eq 0 ] && _ts_args="$_ts_args --yes"
    [ -n "$TS_KEY_FILE" ] && _ts_args="$_ts_args --auth-key-file $TS_KEY_FILE"
    # The server is installed and running by now. A Tailscale step that stops
    # -- no key on an unattended run, certificates not enabled on the tailnet,
    # a device waiting for approval -- must not turn that into a failed
    # install: it says what is missing, and re-running the script finishes it.
    _ts_ok=1
    if [ "$(id -u)" -eq 0 ]; then
        # shellcheck disable=SC2086 -- a deliberately word-split option list
        sh "$INSTALL_DIR/scripts/setup-tailscale.sh" $_ts_args || _ts_ok=0
    else
        echo "Installing Tailscale needs root; sudo may ask for your password."
        # TS_AUTHKEY is kept by NAME, so an unattended install works under
        # sudo, which would otherwise drop it -- and the key itself never
        # becomes an argument `ps` could show.
        # shellcheck disable=SC2086
        sudo --preserve-env=TS_AUTHKEY sh "$INSTALL_DIR/scripts/setup-tailscale.sh" $_ts_args || _ts_ok=0
    fi
    if [ "$_ts_ok" -eq 0 ]; then
        echo
        echo "The server is installed and listens on 127.0.0.1:${PORT:-8000}, but Tailscale"
        echo "is not finished (see above). Once that is fixed, finish it with:"
        echo
        echo "    sudo sh $INSTALL_DIR/scripts/setup-tailscale.sh --port ${PORT:-8000} --dir $INSTALL_DIR"
    fi
else
    echo
    echo "This deployment listens on ${BIND} over plain HTTP. That is fine for"
    echo "localhost; putting it on a network address requires a TLS terminator in"
    echo "front -- see SECURITY.md, or re-run with --tailscale."
fi
