#!/bin/sh
# Put this server on a Tailscale network, reachable over HTTPS and nowhere else.
# Needs root.
#
#   sudo sh scripts/setup-tailscale.sh
#   curl -fsSL .../setup-tailscale.sh | sudo sh -s -- --port 8000
#
# What it does, each step skipped when already done, so re-running is safe:
#
#   1. installs Tailscale (https://tailscale.com/install.sh, their own script),
#   2. joins the tailnet with an auth key, asked for without echoing it,
#   3. checks the tailnet has HTTPS certificates turned on,
#   4. publishes http://127.0.0.1:PORT as https://<name>.<tailnet>.ts.net with
#      `tailscale serve`, which fetches and renews the certificate itself,
#   5. checks the server answers there, and prints the URL to hand out.
#
# `tailscale serve` is the TLS terminator SECURITY.md asks for, in place of
# nginx: the server keeps speaking plain HTTP on loopback, and the only way in
# from another machine is the encrypted tailnet. That only holds if the port
# is NOT also published on a network address, which is why this warns when
# .env has BIND_ADDR set to anything but 127.0.0.1.
#
# The auth key comes from the tailnet's admin console (Settings -> Keys ->
# Generate auth key). For a server: NOT reusable, tagged (tag:server), so the
# machine belongs to no person and its login never expires. The key is never
# accepted on the command line, where it would land in the shell history and
# in `ps` for every user to read. Give it, in this order of preference:
#
#   - at the prompt (the default),
#   - --auth-key-file PATH, a file holding only the key,
#   - TS_AUTHKEY in the environment, for unattended installs.
#
# Options:
#   --port N            local port the server listens on (default: HOST_PORT
#                       from .env, else 8000)
#   --hostname NAME     the machine's name on the tailnet, and therefore in its
#                       URL (default: visor). It is PUBLIC: every certificate
#                       is written to the Certificate Transparency logs, so the
#                       name and the tailnet's are visible to anyone. Keep it
#                       neutral -- no person, patient, project or site name.
#   --auth-key-file P   read the auth key from this file
#   --dir DIR           the server clone, to read .env from (default: the
#                       clone this script sits in, else ./VISOR-serve)
#   --no-auto-update    do not turn on Tailscale's own automatic updates
#   --yes               never ask anything; fail instead of prompting
#
# Linux only: the server runs on Linux, and this is the server's side. The
# researchers' laptops install the Tailscale app from tailscale.com/download.

set -eu

PORT=""
TS_HOSTNAME="visor"
KEY_FILE=""
INSTALL_DIR=""
AUTO_UPDATE=1
ASK=1

while [ $# -gt 0 ]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --hostname) TS_HOSTNAME="$2"; shift 2 ;;
        --auth-key-file) KEY_FILE="$2"; shift 2 ;;
        --dir) INSTALL_DIR="$2"; shift 2 ;;
        --no-auto-update) AUTO_UPDATE=0; shift ;;
        --yes|-y|--non-interactive) ASK=0; shift ;;
        -h|--help) sed -n '2,48p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        --auth-key|--auth-key=*)
            echo "setup-tailscale: the key is not accepted on the command line, where" >&2
            echo "  the shell history and \`ps\` would keep it. Use the prompt," >&2
            echo "  --auth-key-file, or TS_AUTHKEY." >&2
            exit 2 ;;
        *) echo "setup-tailscale: unknown option '$1'" >&2; exit 2 ;;
    esac
done

if [ "$(uname -s)" != "Linux" ]; then
    echo "setup-tailscale: this sets up the SERVER side, which runs on Linux." >&2
    echo "  On a laptop, install the app from https://tailscale.com/download" >&2
    exit 1
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "setup-tailscale: this needs root. Re-run it as:" >&2
    echo "  sudo sh $0 $*" >&2
    exit 1
fi

can_ask() {
    [ "$ASK" -eq 1 ] && [ -r /dev/tty ] && [ -w /dev/tty ]
}

# --- the deployment's settings ------------------------------------------
# Read, never written: where the server listens is server_ctl.py's business.
if [ -z "$INSTALL_DIR" ]; then
    _here="$(cd "$(dirname "$0")" 2>/dev/null && pwd || true)"
    if [ -n "$_here" ] && [ -f "$_here/server_ctl.py" ]; then
        INSTALL_DIR="$(dirname "$_here")"
    else
        INSTALL_DIR="./VISOR-serve"
    fi
fi

env_value() {
    # The last assignment wins, as it does for compose.
    [ -f "$INSTALL_DIR/.env" ] || return 0
    sed -n "s/^$1=//p" "$INSTALL_DIR/.env" | tail -n 1
}

[ -n "$PORT" ] || PORT="$(env_value HOST_PORT)"
[ -n "$PORT" ] || PORT=8000
case "$PORT" in
    ''|*[!0-9]*) echo "setup-tailscale: --port must be a number (got '$PORT')." >&2; exit 2 ;;
esac

# --- 1. install ----------------------------------------------------------
if ! command -v tailscale >/dev/null 2>&1; then
    echo "Installing Tailscale ..."
    curl -fsSL https://tailscale.com/install.sh | sh
fi
if ! tailscale version >/dev/null 2>&1; then
    echo "setup-tailscale: tailscale is installed but does not run; see" >&2
    echo "  https://tailscale.com/download/linux" >&2
    exit 1
fi
# The installer enables the daemon on systemd hosts; this covers a host where
# it was installed by hand and never started.
if command -v systemctl >/dev/null 2>&1; then
    systemctl enable --now tailscaled >/dev/null 2>&1 || true
fi

status_field() {
    # status_field EXPR -- one value out of `tailscale status --json`, where
    # EXPR is a Python expression over `s`. python3 is already required by
    # setup-server.sh, and parsing JSON with sed is how scripts break.
    tailscale status --json 2>/dev/null | python3 -c "
import json, sys
try:
    s = json.load(sys.stdin)
    print($1)
except Exception:
    pass
"
}

# --- 2. join the tailnet -------------------------------------------------
STATE="$(status_field 's.get("BackendState", "")')"
case "$STATE" in
    Running)
        echo "Already on the tailnet as $(status_field 's["Self"]["DNSName"].rstrip(".")')."
        ;;
    Stopped)
        # Logged in, only switched off: no key needed.
        echo "Reconnecting to the tailnet ..."
        tailscale up
        ;;
    *)
        # The key goes to tailscale as a FILE (its `--auth-key=file:` form), so
        # it never appears in an argument list. 0600, removed on any exit.
        _tmp_key="$(umask 077 && mktemp)"
        trap 'rm -f "$_tmp_key"' EXIT INT TERM
        if [ -n "$KEY_FILE" ]; then
            [ -r "$KEY_FILE" ] || { echo "setup-tailscale: cannot read $KEY_FILE." >&2; exit 1; }
            tr -d ' \t\r\n' < "$KEY_FILE" > "$_tmp_key"
        elif [ -n "${TS_AUTHKEY:-}" ]; then
            printf '%s' "$TS_AUTHKEY" > "$_tmp_key"
        elif can_ask; then
            echo "Paste the server's auth key (Settings -> Keys in the admin console)."
            echo "It is not shown as you type."
            printf 'Auth key: ' > /dev/tty
            stty -echo < /dev/tty 2>/dev/null || true
            read -r _key < /dev/tty || _key=""
            stty echo < /dev/tty 2>/dev/null || true
            printf '\n' > /dev/tty
            printf '%s' "$_key" > "$_tmp_key"
            _key=""
        else
            echo "setup-tailscale: no auth key. Give one with --auth-key-file or TS_AUTHKEY." >&2
            exit 1
        fi
        if ! grep -q '^tskey-' "$_tmp_key"; then
            echo "setup-tailscale: that does not look like an auth key (they start with 'tskey-')." >&2
            exit 1
        fi
        echo "Joining the tailnet as '$TS_HOSTNAME' ..."
        if ! tailscale up --auth-key="file:$_tmp_key" --hostname="$TS_HOSTNAME"; then
            echo "setup-tailscale: the tailnet refused the key. It may have expired, or" >&2
            echo "  been single-use and already spent: generate a new one and re-run." >&2
            exit 1
        fi
        rm -f "$_tmp_key"
        ;;
esac

# A tailnet with device approval holds a new machine until an admin lets it in,
# and until then it has no name and no certificate to serve.
STATE="$(status_field 's.get("BackendState", "")')"
if [ "$STATE" = "NeedsMachineAuth" ]; then
    echo
    echo "This machine joined, but the tailnet approves devices by hand. Approve it"
    echo "in the admin console (Machines), then re-run this script."
    exit 1
elif [ "$STATE" != "Running" ]; then
    echo "setup-tailscale: Tailscale is in state '${STATE:-unknown}', not Running." >&2
    echo "  \`tailscale status\` says why." >&2
    exit 1
fi

FQDN="$(status_field 's["Self"]["DNSName"].rstrip(".")')"

# --- 3. HTTPS certificates ----------------------------------------------
# Off by default on a new tailnet, and `tailscale serve` without them stops to
# print a URL and wait for someone to click it -- which, in a script, is a hang.
if [ -z "$(status_field '",".join(s.get("CertDomains") or [])')" ]; then
    echo
    echo "HTTPS certificates are not enabled on this tailnet. In the admin console,"
    echo "open DNS, make sure MagicDNS is on, and enable 'HTTPS Certificates'."
    echo "Then re-run this script; everything already done is skipped."
    exit 1
fi

# --- 4. publish ----------------------------------------------------------
# Kept across reboots by tailscaled itself; nothing has to run at start-up.
echo "Publishing http://127.0.0.1:$PORT as https://$FQDN ..."
tailscale serve --bg "http://127.0.0.1:$PORT" >/dev/null

if [ "$AUTO_UPDATE" -eq 1 ]; then
    # A machine holding medical images should not sit on a stale VPN client.
    # Not fatal: some package setups do not support it.
    tailscale set --auto-update >/dev/null 2>&1 \
        || echo "Note: could not turn on Tailscale's automatic updates; keep it updated with your package manager."
fi

# --- 5. check ------------------------------------------------------------
# The first request makes tailscaled fetch the certificate, which takes a few
# seconds; a failure here is reported, not fatal, because the server itself
# may simply not be started yet.
echo "Checking https://$FQDN/health ..."
_ok=0
for _try in 1 2 3 4 5 6; do
    if curl -fsS --max-time 20 "https://$FQDN/health" >/dev/null 2>&1; then
        _ok=1
        break
    fi
    sleep 5
done

BIND_ADDR_VALUE="$(env_value BIND_ADDR)"

echo
if [ "$_ok" -eq 1 ]; then
    echo "Done. The server answers at:"
else
    echo "Tailscale is set up, but the server did not answer at /health yet."
    echo "If it is not started, start it (python3 scripts/server_ctl.py up), then open:"
fi
echo
echo "    https://$FQDN"
echo
echo "Only machines on the tailnet can reach it. That is the URL to put in the"
echo "Slicer client, with the same API token as before."

if [ -f "$INSTALL_DIR/.env" ] && [ "$BIND_ADDR_VALUE" != "127.0.0.1" ]; then
    echo
    echo "WARNING: .env publishes the port on '${BIND_ADDR_VALUE:-every interface}', so the"
    echo "server is ALSO reachable in plain HTTP outside the tailnet. Keep it on"
    echo "loopback, where only Tailscale reaches it:"
    echo
    echo "    python3 $INSTALL_DIR/scripts/server_ctl.py up --bind 127.0.0.1"
fi
