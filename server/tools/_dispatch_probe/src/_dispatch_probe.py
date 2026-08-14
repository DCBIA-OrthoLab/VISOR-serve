"""The smallest thing that can prove a tool ran in another interpreter.

Standard library only, no inference, no models: everything this returns is
about the run itself, so a failure here is a failure of the dispatch path and
of nothing else.

The leading underscore on the folder keeps registry.py from discovering it --
this is a test fixture, and it must never appear in GET /tools.
"""

import os
import sys


def run(a, b, out_name="probe.txt", fail=False, tags=None):
    """Add two numbers, write a file into the job's output/, describe the run.

    `fail` raises instead, which is the other half of the contract: the runner
    must then write no result.json and exit non-zero. `tags` is here for one
    reason: it is the only argument shape a .schema.json can declare that the
    server had no type for, so something has to carry it across the wire.
    """
    if fail:
        raise RuntimeError("_dispatch_probe was asked to fail")

    job_dir = os.environ["SADT_JOB_DIR"]
    total = a + b
    output_path = os.path.join(job_dir, "output", out_name)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(str(total))

    return {
        # The canonical return: outputs named, so a caller -- and the parity
        # harness -- finds the file without guessing which string is a path.
        "outputs": {"probe": output_path},
        "total": total,
        "tags": list(tags or []),
        # The proof: the server compares this against its own sys.executable.
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "job_id": os.environ.get("SADT_JOB_ID"),
        "sadt_api": os.environ.get("SADT_API"),
        # The server's bearer token has no business in a tool's environment.
        "sees_api_token": "API_TOKEN" in os.environ,
    }
