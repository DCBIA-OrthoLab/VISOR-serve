"""Start the server: `python serve.py [--host H] [--port P] [--reload]`.

The only supported way to start it, because it is the one that cannot start it
without TLS: the certificate is resolved (or generated) first, the listener is
handed it, and a failure to do either exits non-zero rather than falling back
to anything. `uvicorn main:app` typed by hand is refused by main.py's lifespan
(see tls.require_active).

`FORWARDED_ALLOW_IPS` is read from the environment by uvicorn itself, exactly
as the `uvicorn` command did.
"""

import argparse
import os
import secrets
import sys
import urllib.parse

import uvicorn

import tls
from config import settings


# Every token that has ever been written down as a default in this repository.
# A server holding one of them is open to anyone who has read the repository.
KNOWN_DEFAULT_TOKENS = frozenset({
    "dev-token", "local-dev-token", "test-token", "change-me",
    "change-me-to-a-long-random-secret",
})
MIN_TOKEN_LENGTH = 24


def _weak_secrets() -> str:
    """Why the configured tokens may not guard patient data, or ""."""
    for name, value, required in (("API_TOKEN", settings.API_TOKEN, True),
                                  ("ADMIN_TOKEN", settings.ADMIN_TOKEN, False)):
        if not value and not required:
            continue
        if value in KNOWN_DEFAULT_TOKENS or len(value) < MIN_TOKEN_LENGTH:
            return (f"{name} is empty, a published default, or shorter than "
                    f"{MIN_TOKEN_LENGTH} characters. Generate one: "
                    "python3 -c 'import secrets; print(secrets.token_urlsafe(32))' "
                    "(scripts/server_ctl.py up does it for you).")
    return ""


EVENT_KEY_FILE = ".event-signing-key"


def event_signing_key(temp_dir: str) -> str:
    """The key run events are signed with, the same one across restarts.

    Kept beside the runs it signs, in TEMP_DIR, which `main` closes to this
    user. A key drawn afresh at each start made every event written before a
    restart fail its check: the reader then took a paused run's `paused`, and
    a finished run's `done`, for a tool's unsigned word and showed `running`.
    A run directory and its key go together -- whoever wipes TEMP_DIR wipes
    both.
    """
    path = os.path.join(temp_dir, EVENT_KEY_FILE)
    try:
        with open(path, encoding="utf-8") as handle:
            key = handle.read().strip()
        if key:
            return key
    except FileNotFoundError:
        pass
    key = secrets.token_urlsafe(32)
    # Written whole under a temporary name then renamed, so a reader never
    # sees half a key; 0600 from the first byte.
    staging = f"{path}.{os.getpid()}"
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(key)
    os.replace(staging, path)
    return key


def _exposure_problem(cert: str) -> str:
    """Why this server may not be published where it is, or "".

    On loopback it may listen with anything: only a front end on this machine
    reaches it, and that front end presents the public certificate. Published
    on a network address, clients reach THIS listener directly, so its own
    certificate must be one they can verify: publicly signed, full chain, and
    covering PUBLIC_URL's host. The self-signed one is never published.
    """
    public = settings.PUBLIC_URL.strip()
    if public and urllib.parse.urlsplit(public).scheme != "https":
        return f"PUBLIC_URL must be an https:// URL; {public!r} is not."
    if tls.is_loopback(settings.PUBLISH_ADDR):
        return ""
    if not settings.TLS_CERT_FILE.strip():
        return (f"PUBLISH_ADDR={settings.PUBLISH_ADDR} puts this server on the network, "
                "which needs a publicly signed certificate: set TLS_CERT_FILE (full chain) "
                "and TLS_KEY_FILE, or publish on 127.0.0.1 behind tailscale serve or a reverse proxy.")
    hostname = urllib.parse.urlsplit(settings.PUBLIC_URL).hostname or ""
    if not hostname:
        return "PUBLISH_ADDR is a network address: set PUBLIC_URL to the https URL clients use."
    problem = tls.public_certificate_problem(cert, hostname)
    if problem:
        return (f"TLS_CERT_FILE is not a publicly signed certificate for {hostname} "
                f"({problem}). Clients would refuse it.")
    return ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true",
                        help="restart on a code change (development only)")
    parser.add_argument("--check", action="store_true",
                        help="probe this server's own /health instead of starting it")
    args = parser.parse_args(argv)
    if args.check:
        return check(args.port)

    weak = _weak_secrets()
    if weak:
        print(f"serve.py: {weak}", file=sys.stderr)
        return 1
    try:
        cert, key = tls.ensure_certificate()
    except tls.TLSConfigurationError as exc:
        print(f"serve.py: {exc}", file=sys.stderr)
        return 1
    exposure = _exposure_problem(cert)
    if exposure:
        print(f"serve.py: {exposure}", file=sys.stderr)
        return 1
    # Every patient file the server holds -- uploads, job directories, results
    # kept by reference, run state -- lives under TEMP_DIR, created by this
    # user with whatever umask the host has (0002 on a lab workstation, so
    # group- or world-readable). One closed door at the root covers all of it.
    os.makedirs(settings.TEMP_DIR, mode=0o700, exist_ok=True)
    os.chmod(settings.TEMP_DIR, 0o700)
    # Inherited by --reload's worker process, which is the one that runs the
    # lifespan check.
    os.environ[tls.ACTIVE_ENV] = "1"
    # One signing key for the whole server, so every worker agrees on which run
    # events are the server's own; inherited by the workers, stripped from the
    # tools (execution/dispatch.SERVER_SECRETS). Set only if absent, so a
    # --reload worker keeps the parent's.
    os.environ.setdefault("SADT_EVENT_SIGNING_KEY", event_signing_key(settings.TEMP_DIR))
    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload,
                ssl_certfile=cert, ssl_keyfile=key)
    return 0


def check(port: int) -> int:
    """The container's healthcheck: 0 when this server answers on its loopback.

    The one probe that does not go through the public URL, because it runs
    where that URL may not exist yet (a first start, before `tailscale serve`).
    It never leaves the container, sends no token, and verifies the server
    against the exact certificate file it listens with -- nothing is trusted on
    sight.
    """
    import urllib.request

    try:
        cert, _key = tls.certificate_paths()
        # Fixed loopback https URL.
        with urllib.request.urlopen(f"https://127.0.0.1:{port}/health", timeout=5,  # nosec B310
                                    context=tls.loopback_context(cert)) as response:
            return 0 if response.status == 200 else 1
    except Exception:  # noqa: BLE001 - a probe answers 1, it never raises
        return 1


if __name__ == "__main__":
    sys.exit(main())
