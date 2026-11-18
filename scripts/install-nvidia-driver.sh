#!/bin/sh
# Install the NVIDIA driver Ubuntu recommends for this host's card, replacing
# an older one. Needs root, and a reboot afterwards to load it.
#
#   sudo sh scripts/install-nvidia-driver.sh
#
# Ubuntu (and its derivatives) only: `ubuntu-drivers` matches the card's PCI ID
# against the driver packages of the release, so a new card -- an RTX 50xx --
# gets a branch that knows it (570 or newer, open kernel modules) without this
# file having to list cards. Elsewhere it prints NVIDIA's page and stops.
#
# With Secure Boot on, the driver's kernel module must be signed by a key the
# firmware trusts: apt asks for a password here, and the next boot shows a blue
# "MOK management" screen where that password enrolls the key. Skipping that
# screen leaves the driver unloaded.

set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "install-nvidia-driver: run it as root (sudo)." >&2
    exit 1
fi
if ! command -v apt-get >/dev/null 2>&1; then
    echo "install-nvidia-driver: only Ubuntu-like systems are handled here; install the driver from" >&2
    echo "  https://www.nvidia.com/Download/index.aspx" >&2
    exit 1
fi

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ubuntu-drivers-common >/dev/null
recommended=$(ubuntu-drivers devices 2>/dev/null | awk '/recommended/ {print $3}' | head -n1)
if [ -z "$recommended" ]; then
    echo "install-nvidia-driver: Ubuntu recommends no driver for this card; install one from" >&2
    echo "  https://www.nvidia.com/Download/index.aspx" >&2
    exit 1
fi
echo "Installing $recommended..."
apt-get install -y "$recommended"
echo "Installed $recommended. Reboot to load it."
