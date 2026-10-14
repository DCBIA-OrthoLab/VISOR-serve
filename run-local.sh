#!/usr/bin/env bash
# Local test server, serving the PACKAGED tools from a SADT-VISOR checkout.
#
# Port 8001 on purpose: 8000 is the docker deployment, and this must not fight
# it. Nothing here touches that container -- it reads a different TOOLS_DIR and
# writes its schema cache somewhere else.
set -euo pipefail

SADT_TOOLS="${SADT_TOOLS:-$HOME/code/SADT-VISOR}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# A random token, kept across restarts so a configured client keeps working.
# Never a fixed one: serve.py refuses every token this repository has published.
TOKEN_FILE="$HERE/server/.local-token"
if [ -z "${API_TOKEN:-}" ]; then
    if [ ! -s "$TOKEN_FILE" ]; then
        (umask 077 && python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > "$TOKEN_FILE")
    fi
    API_TOKEN="$(cat "$TOKEN_FILE")"
fi
export API_TOKEN
export TOOLS_DIR="$SADT_TOOLS/tools"          # <-- the link to SADT-VISOR
export DESCRIBE_PATH="$SADT_TOOLS/scripts/describe.py"
export DATA_DIR="${DATA_DIR:-$HERE/DATA}"
export SCHEMA_CACHE_DIR="${SCHEMA_CACHE_DIR:-$HERE/.schema-cache}"
# Where a campaign left its summaries. The default is the container's mount
# point, which does not exist outside it -- so `/benchmarks/view` served from a
# checkout rendered an empty page rather than the campaigns sitting right here.
export SADT_BENCHMARK_DIR="${SADT_BENCHMARK_DIR:-$HERE/benchmarks/results/summary}"
export DEVICE="${DEVICE:-cuda}"

LOG="${LOG:-$HERE/local-server.log}"

echo "tools   : $TOOLS_DIR"
echo "data    : $DATA_DIR"
# Where the token is, never its value: this output lands in terminals, logs
# and bug reports.
if [ -s "$TOKEN_FILE" ] && [ "$API_TOKEN" = "$(cat "$TOKEN_FILE")" ]; then
    echo "token   : in $TOKEN_FILE"
else
    echo "token   : from the API_TOKEN environment variable"
fi
echo "url     : https://localhost:8001  (self-signed: server/.tls/cert.pem)"
echo "log     : $LOG"
echo

# serve.py rather than `uvicorn`: it is the only way to start the server, and
# it only ever listens over HTTPS (server/tls.py), here with a self-signed
# certificate kept in server/.tls/.
cd "$HERE/server"
# Logged as well as shown: a tool failure is a traceback in here, and it is the
# only place it exists -- the client gets a status code and a one-line message.
exec "$HERE/server/venv/bin/python" serve.py \
    --host 127.0.0.1 --port 8001 "$@" 2>&1 | tee -a "$LOG"
