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
import sys

import uvicorn

import tls


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true",
                        help="restart on a code change (development only)")
    args = parser.parse_args(argv)

    try:
        cert, key = tls.ensure_certificate()
    except tls.TLSConfigurationError as exc:
        print(f"serve.py: {exc}", file=sys.stderr)
        return 1
    # Inherited by --reload's worker process, which is the one that runs the
    # lifespan check.
    os.environ[tls.ACTIVE_ENV] = "1"
    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload,
                ssl_certfile=cert, ssl_keyfile=key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
