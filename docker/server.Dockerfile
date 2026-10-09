# syntax=docker/dockerfile:1.7
#
# The server alone, ready to run: the API, its four Python dependencies, and
# the system libraries tool virtualenvs load from the image they run in.
#
#     docker build -f docker/server.Dockerfile -t visor-serve .
#
# NO tool is inside. Tools come from tool libraries -- SADT-VISOR today,
# others tomorrow -- each mounted with its own virtualenvs, exactly as
# docker-compose.yml mounts SADT_TOOLS. A library's tools may count on the
# libraries installed below and on nothing else from this image.
#
# docker/Dockerfile is the other shape: one image holding the server AND one
# library's virtualenvs. This one replaces the general-purpose lab image the
# compose file used to run, a 32 GB conda environment of which the server read
# Python and FastAPI.

FROM python:3.11-slim-bookworm

# What the tool virtualenvs dlopen from the system, measured by running `ldd`
# over every shared object in every SADT-VISOR virtualenv inside the previous
# image and keeping what resolved outside the venvs:
#   libgl1 libglib2.0-0 libx11-6 libxext6 libxrender1   VTK, OpenCV
#   libgomp1                                            torch, lightgbm, scikit-learn
# git is the server's own: "Check for updates" reads the mounted checkout.
# The NVIDIA driver is NOT here: the container runtime mounts the host's.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 libgomp1 libx11-6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# uid 1000, as in the image this replaces: a deployment's mounted checkout and
# job folders were written by that uid, and a different one could not read or
# clean them.
RUN useradd --uid 1000 --create-home lab

COPY server/requirements-api.txt /tmp/requirements-api.txt
RUN pip install --no-cache-dir -r /tmp/requirements-api.txt && rm /tmp/requirements-api.txt

# Shipped so the image runs on its own; a deployment that follows its
# checkout mounts server/ over it, as docker-compose.yml does. scripts/ is
# host-side tooling and stays out (.dockerignore).
COPY --chown=lab:lab server/ /workspace/server/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
USER lab
WORKDIR /workspace/server
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
