#!/bin/sh
# Install Docker Engine and the compose plugin, then let the current user
# drive them. Needs root.
#
#   sudo sh scripts/install-docker.sh
#   sudo sh scripts/install-docker.sh --nvidia    # also the GPU container toolkit
#
# Linux only, deliberately. On macOS and Windows the supported way to get a
# docker daemon is Docker Desktop, which is a GUI installer and cannot be
# driven from a shell script -- this prints the link and stops rather than
# pretending otherwise.
#
# What it runs on Linux is https://get.docker.com, Docker's own convenience
# script: it detects the distribution, adds Docker's apt/dnf repository and
# its signing key, and installs docker-ce plus docker-compose-plugin. That is
# the upstream-recommended path for a non-production host. If piping a remote
# script into root is not acceptable in your environment, follow the manual
# per-distribution instructions at https://docs.docker.com/engine/install/
# instead and skip this file -- server_ctl.py only cares that `docker` and
# `docker compose` work afterwards.

set -eu

WITH_NVIDIA=0
for arg in "$@"; do
    case "$arg" in
        --nvidia) WITH_NVIDIA=1 ;;
        -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "install-docker: unknown option '$arg'" >&2; exit 2 ;;
    esac
done

case "$(uname -s)" in
    Linux) ;;
    Darwin)
        echo "install-docker: on macOS, install Docker Desktop:" >&2
        echo "  https://docs.docker.com/desktop/install/mac-install/" >&2
        exit 1
        ;;
    *)
        echo "install-docker: on Windows, install Docker Desktop with the WSL2 backend:" >&2
        echo "  https://docs.docker.com/desktop/install/windows-install/" >&2
        exit 1
        ;;
esac

if [ "$(id -u)" -ne 0 ]; then
    echo "install-docker: this needs root. Re-run it as:" >&2
    echo "  sudo sh $0 $*" >&2
    exit 1
fi

# Whoever called sudo is who should end up able to run docker -- not root,
# who already can.
TARGET_USER="${SUDO_USER:-${USER:-root}}"

if command -v docker >/dev/null 2>&1; then
    echo "Docker is already installed: $(docker --version)"
else
    echo "Installing Docker Engine via https://get.docker.com ..."
    if ! command -v curl >/dev/null 2>&1; then
        echo "install-docker: curl is required but was not found in PATH." >&2
        exit 1
    fi
    tmp="$(mktemp)"
    trap 'rm -f "$tmp"' EXIT INT TERM
    curl -fsSL https://get.docker.com -o "$tmp"
    sh "$tmp"
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "install-docker: docker is installed but 'docker compose' is not." >&2
    echo "  Install the plugin for your distribution (docker-compose-plugin)," >&2
    echo "  see https://docs.docker.com/compose/install/linux/" >&2
    exit 1
fi

# Start it now and on every boot, so the server survives a reboot without
# anyone remembering to bring it back.
if command -v systemctl >/dev/null 2>&1; then
    systemctl enable --now docker >/dev/null 2>&1 || true
fi

if [ "$TARGET_USER" != "root" ]; then
    if ! id -nG "$TARGET_USER" | tr ' ' '\n' | grep -qx docker; then
        echo "Adding '$TARGET_USER' to the 'docker' group..."
        groupadd -f docker
        usermod -aG docker "$TARGET_USER"
        NEEDS_RELOGIN=1
    fi
fi

if [ "$WITH_NVIDIA" -eq 1 ]; then
    # Only the container toolkit -- the GPU DRIVER is not installed here. A
    # driver install can require a reboot and a specific kernel package, which
    # is not something to do silently behind a "start my server" button.
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "install-docker: --nvidia was passed but nvidia-smi is missing, so this host" >&2
        echo "  has no working GPU driver yet. Install the driver first; the server runs" >&2
        echo "  on CPU in the meantime." >&2
        exit 1
    fi
    # Already installed: go straight to the registration below. Re-running the
    # install would add an apt/dnf repository and can UPGRADE the toolkit --
    # a change nobody asked for, on the very component being repaired, when
    # what is usually missing is only the one-line runtime registration. This
    # is the common case now that the toolkit ships CDI specs by itself: the
    # card works, docker just has no runtime entry pointing at it.
    if command -v nvidia-ctk >/dev/null 2>&1; then
        echo "The NVIDIA Container Toolkit is already installed: $(nvidia-ctk --version | head -n1)"
    elif command -v apt-get >/dev/null 2>&1; then
        echo "Installing the NVIDIA Container Toolkit..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
            | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
            > /etc/apt/sources.list.d/nvidia-container-toolkit.list
        apt-get update
        apt-get install -y nvidia-container-toolkit
    elif command -v dnf >/dev/null 2>&1; then
        echo "Installing the NVIDIA Container Toolkit..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
            -o /etc/yum.repos.d/nvidia-container-toolkit.repo
        dnf install -y nvidia-container-toolkit
    else
        echo "install-docker: no apt-get or dnf here. Install the toolkit by hand:" >&2
        echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" >&2
        exit 1
    fi
    echo "Registering the 'nvidia' runtime with docker..."
    nvidia-ctk runtime configure --runtime=docker
    if command -v systemctl >/dev/null 2>&1; then
        # This STOPS every running container, the tool server included. Callers
        # that may have one up have to say so before asking for a password.
        echo "Restarting the docker daemon -- running containers will stop."
        systemctl restart docker
    fi
    echo "Done. 'docker info' now lists an 'nvidia' runtime."
fi

echo
echo "Docker is ready: $(docker --version), $(docker compose version)"
if [ "${NEEDS_RELOGIN:-0}" -eq 1 ]; then
    echo
    echo "IMPORTANT: '$TARGET_USER' was just added to the 'docker' group, which only takes"
    echo "effect on a NEW login session. Log out and back in (or reboot) before starting"
    echo "the server -- until then every docker command answers 'permission denied'."
fi
