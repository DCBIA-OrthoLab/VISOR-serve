#!/usr/bin/env bash
# Local test server, serving the PACKAGED tools from a sadt-tools checkout.
#
# Port 8001 on purpose: 8000 is the docker deployment, and this must not fight
# it. Nothing here touches that container -- it reads a different TOOLS_DIR and
# writes its schema cache somewhere else.
set -euo pipefail

SADT_TOOLS="${SADT_TOOLS:-$HOME/code/SADT-VISOR}"
HERE="$(cd "$(dirname "$0")" && pwd)"

export API_TOKEN="${API_TOKEN:-local-dev-token}"
export TOOLS_DIR="$SADT_TOOLS/tools"          # <-- the link to sadt-tools
export DESCRIBE_PATH="$SADT_TOOLS/scripts/describe.py"
export DATA_DIR="${DATA_DIR:-$HERE/DATA}"
export SCHEMA_CACHE_DIR="${SCHEMA_CACHE_DIR:-$HERE/.schema-cache}"
export DEVICE="${DEVICE:-cuda}"

LOG="${LOG:-$HERE/local-server.log}"

echo "tools   : $TOOLS_DIR"
echo "data    : $DATA_DIR"
echo "token   : $API_TOKEN"
echo "url     : http://127.0.0.1:8001"
echo "log     : $LOG"
echo

# The venv was created at another path, so its console scripts carry a dead
# shebang; invoking uvicorn as a module sidesteps that.
cd "$HERE/server"
# Logged as well as shown: a tool failure is a traceback in here, and it is the
# only place it exists -- the client gets a status code and a one-line message.
exec "$HERE/server/venv/bin/python" -m uvicorn main:app \
    --host 127.0.0.1 --port 8001 "$@" 2>&1 | tee -a "$LOG"
