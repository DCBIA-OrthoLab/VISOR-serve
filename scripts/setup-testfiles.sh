#!/bin/sh
# Download every reference test file listed in scripts/data-manifest.yml
# into ./DATA/.
#
# From anywhere, without cloning:
#
#   curl -fsSL https://raw.githubusercontent.com/DCBIA-OrthoLab/VISOR-serve/main/scripts/setup-testfiles.sh | sh
#
# To fetch one tool's test files only, pass arguments through `sh -s --`:
#
#   curl -fsSL .../setup-testfiles.sh | sh -s -- --tool AMASSS
#
# Environment:
#   DATA_DIR   destination root (default: ./DATA)
#   REPO/REF   where to fetch the engine + manifest from (default: this repo, main)
#
# Files already present are skipped, so re-running only fetches what is
# missing. The result is the layout server/data_store.py reads:
# DATA/<tool>/testfiles/<name>.

set -eu

REPO="${REPO:-DCBIA-OrthoLab/VISOR-serve}"
REF="${REF:-main}"
RAW="https://raw.githubusercontent.com/${REPO}/${REF}/scripts"

if ! command -v python3 >/dev/null 2>&1; then
    echo "setup-testfiles: python3 is required but was not found in PATH." >&2
    exit 1
fi

# Run from a clone when there is one (so a local edit to the manifest is what
# takes effect); otherwise pull both files down to a temp dir.
if [ -f "./scripts/fetch_data.py" ] && [ -f "./scripts/data-manifest.yml" ]; then
    exec python3 ./scripts/fetch_data.py --kind testfiles "$@"
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "setup-testfiles: curl is required but was not found in PATH." >&2
    exit 1
fi

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT INT TERM

fetch() {
    if ! curl -fsSL "${RAW}/$1" -o "${work_dir}/$1"; then
        echo "setup-testfiles: could not download $1 from ${REPO}@${REF}." >&2
        echo "  Check that the branch exists and carries scripts/$1," >&2
        echo "  or point elsewhere with: REF=<branch> REPO=<owner/repo>" >&2
        exit 1
    fi
}

echo "Fetching the download engine from ${REPO}@${REF}..."
fetch fetch_data.py
fetch data-manifest.yml

python3 "${work_dir}/fetch_data.py" --kind testfiles "$@"
