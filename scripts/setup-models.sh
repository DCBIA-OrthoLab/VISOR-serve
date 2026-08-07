#!/bin/sh
# Download every AI model listed in scripts/data-manifest.yml into ./DATA/.
#
# From anywhere, without cloning:
#
#   curl -fsSL https://raw.githubusercontent.com/Jules-GP/slicer-remote-tool-server/main/scripts/setup-models.sh | sh
#
# To fetch one tool's models only, pass arguments through `sh -s --`:
#
#   curl -fsSL .../setup-models.sh | sh -s -- --tool AMASSS
#
# Environment:
#   DATA_DIR   destination root (default: ./DATA)
#   REPO/REF   where to fetch the engine + manifest from (default: this repo, main)
#
# Files already present are skipped, so re-running only fetches what is
# missing. The result is the layout server/data_store.py reads:
# DATA/<tool>/models/<name>.
#
# To see what is already on disk before choosing, and to stand the server
# itself up, see scripts/server_ctl.py (catalog / up / update) and
# scripts/setup-server.sh.

set -eu

REPO="${REPO:-Jules-GP/slicer-remote-tool-server}"
REF="${REF:-main}"
RAW="https://raw.githubusercontent.com/${REPO}/${REF}/scripts"

if ! command -v python3 >/dev/null 2>&1; then
    echo "setup-models: python3 is required but was not found in PATH." >&2
    exit 1
fi

# Run from a clone when there is one (so a local edit to the manifest is what
# takes effect); otherwise pull both files down to a temp dir.
if [ -f "./scripts/fetch_data.py" ] && [ -f "./scripts/data-manifest.yml" ]; then
    exec python3 ./scripts/fetch_data.py --kind models "$@"
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "setup-models: curl is required but was not found in PATH." >&2
    exit 1
fi

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT INT TERM

fetch() {
    if ! curl -fsSL "${RAW}/$1" -o "${work_dir}/$1"; then
        echo "setup-models: could not download $1 from ${REPO}@${REF}." >&2
        echo "  Check that the branch exists and carries scripts/$1," >&2
        echo "  or point elsewhere with: REF=<branch> REPO=<owner/repo>" >&2
        exit 1
    fi
}

echo "Fetching the download engine from ${REPO}@${REF}..."
fetch fetch_data.py
fetch data-manifest.yml

python3 "${work_dir}/fetch_data.py" --kind models "$@"
