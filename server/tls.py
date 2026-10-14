"""The certificate this server listens with. There is no plain-HTTP mode.

Patient imaging and the bearer token travel on every request, so the server
never opens a socket that speaks HTTP in the clear -- not on loopback, not
behind a proxy, not for a developer. An `http://` request fails at the TLS
handshake, before a byte of it is read. There is deliberately no setting that
turns this off: the previous design delegated TLS to a terminator in front of
uvicorn, and nothing checked the terminator was there -- a deployment with no
`tailscale serve` and the port published on every interface served medical
images in the clear while every document said TLS was mandatory.

Which certificate:

* `TLS_CERT_FILE` + `TLS_KEY_FILE`, when the operator has one (an
  institutional certificate, `tailscale cert`). Both or neither.
* Otherwise a self-signed one, generated once into `TLS_DIR` and kept, so a
  client that pinned it keeps trusting it across restarts. Its names are the
  loopback ones: a local install's client reaches it as `https://localhost`
  and pins what that address presents, and `tailscale serve` reaches it as
  `https+insecure://127.0.0.1`, re-encrypting a hop that never leaves the
  machine. A remote client is meant to come in through Tailscale's publicly
  signed certificate, never through this one.

Generated with the `openssl` command rather than a Python package: it is in
the image and on every Linux host, and the API's requirements stay what
`tests/test_api_requirements.py` says they are.
"""

import os
import shutil
import ssl
import subprocess
import tempfile

from config import settings

CERT_NAME = "cert.pem"
KEY_NAME = "key.pem"

# Set by serve.py once the listener is configured with the certificate, and
# checked by main.py's lifespan. It is a tripwire against `uvicorn main:app`
# typed by hand -- the one way left to start this server without TLS -- not a
# security boundary: someone who sets it by hand has chosen to.
ACTIVE_ENV = "SADT_TLS_ACTIVE"

# Ten years. The certificate is pinned, never checked against an authority, so
# an expiry only ever breaks clients; it is renewed a month before it lapses.
VALID_DAYS = 3650
RENEW_SECONDS = 30 * 24 * 3600

SUBJECT_ALT_NAMES = "DNS:localhost,IP:127.0.0.1,IP:::1"


class TLSConfigurationError(RuntimeError):
    """No usable certificate. The server does not start without one."""


def _generated_paths() -> tuple:
    return (os.path.join(settings.TLS_DIR, CERT_NAME),
            os.path.join(settings.TLS_DIR, KEY_NAME))


def certificate_paths() -> tuple:
    """(cert, key) the listener uses, whether supplied or generated."""
    cert, key = settings.TLS_CERT_FILE.strip(), settings.TLS_KEY_FILE.strip()
    if cert or key:
        if not (cert and key):
            raise TLSConfigurationError(
                "TLS_CERT_FILE and TLS_KEY_FILE go together: set both, or neither "
                "to let the server generate its own certificate.")
        return cert, key
    return _generated_paths()


def _openssl() -> str:
    found = shutil.which("openssl")
    if not found:
        raise TLSConfigurationError(
            "The `openssl` command is needed to generate this server's certificate "
            "and is not installed. Install it, or set TLS_CERT_FILE and TLS_KEY_FILE.")
    return found


def _still_valid(cert: str) -> bool:
    if not os.path.isfile(cert):
        return False
    probe = subprocess.run(
        [_openssl(), "x509", "-checkend", str(RENEW_SECONDS), "-noout", "-in", cert],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return probe.returncode == 0


def _generate(cert: str, key: str) -> None:
    directory = os.path.dirname(cert)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    os.chmod(directory, 0o700)
    # Generated beside its destination and renamed into place, so a crash
    # halfway leaves the previous pair rather than a key with no certificate.
    staging = tempfile.mkdtemp(prefix=".tls-", dir=directory)
    try:
        new_cert = os.path.join(staging, CERT_NAME)
        new_key = os.path.join(staging, KEY_NAME)
        made = subprocess.run(
            [_openssl(), "req", "-x509", "-newkey", "ec",
             "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes",
             "-days", str(VALID_DAYS), "-subj", "/CN=visor-serve",
             "-addext", f"subjectAltName={SUBJECT_ALT_NAMES}",
             "-addext", "basicConstraints=critical,CA:FALSE",
             "-addext", "keyUsage=critical,digitalSignature",
             "-addext", "extendedKeyUsage=serverAuth",
             "-keyout", new_key, "-out", new_cert],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
        if made.returncode != 0:
            raise TLSConfigurationError(
                "openssl could not generate the server certificate: "
                + made.stderr.decode("utf-8", "replace").strip()[-500:])
        os.chmod(new_key, 0o600)
        os.chmod(new_cert, 0o644)
        os.replace(new_key, key)
        os.replace(new_cert, cert)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def ensure_certificate() -> tuple:
    """Return (cert, key), generating the self-signed pair when it is needed.

    Raises TLSConfigurationError rather than returning anything the listener
    could start without.
    """
    cert, key = certificate_paths()
    if not (settings.TLS_CERT_FILE.strip() or _still_valid(cert)):
        _generate(cert, key)
    # Loaded here, once, so a key that does not match its certificate, or a
    # file the server cannot read, is one clear line at startup instead of a
    # handshake failure on every request.
    try:
        ssl.create_default_context(ssl.Purpose.CLIENT_AUTH).load_cert_chain(cert, key)
    except (OSError, ssl.SSLError) as exc:
        raise TLSConfigurationError(f"The TLS certificate cannot be used: {exc}") from exc
    return cert, key


def require_active() -> None:
    """Refuse to serve unless serve.py configured the listener with TLS."""
    if os.environ.get(ACTIVE_ENV) != "1":
        raise TLSConfigurationError(
            "This server only listens over HTTPS and must be started with "
            "`python serve.py`, never `uvicorn main:app` directly.")


def loopback_context(cafile: str) -> ssl.SSLContext:
    """A client context for this server's own loopback, trusting `cafile` only.

    For the callers that live beside the server (the benchmark battery, a tool
    reading `SADT_API`). The certificate is pinned rather than looked up, and
    the host name is not checked: an operator's certificate names the public
    address, never 127.0.0.1, and a pinned certificate already says which key
    may answer.
    """
    context = ssl.create_default_context(cafile=cafile)
    context.check_hostname = False
    # A supplied certificate is a leaf issued by an authority; trusting the
    # leaf alone needs partial chains, which Python only enables from 3.13.
    context.verify_flags |= getattr(ssl, "VERIFY_X509_PARTIAL_CHAIN", 0)
    return context
