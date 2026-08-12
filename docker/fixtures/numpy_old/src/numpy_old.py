"""Build fixture: report the numpy this tool was installed with.

Not a tool anyone runs for its output. It exists so the image can prove, from
the inside, that two tools with incompatible pins are both installed and both
runnable -- which is the claim the whole per-tool-virtualenv layout rests on.
"""

import os
import sys

import numpy


def run(size=3):
    matrix = numpy.eye(int(size))
    return {
        "outputs": {},
        "numpy": numpy.__version__,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "trace": float(numpy.trace(matrix)),
        "job_id": os.environ.get("SADT_JOB_ID"),
    }
