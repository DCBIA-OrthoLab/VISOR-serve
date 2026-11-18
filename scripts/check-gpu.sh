#!/bin/sh
# Can this host's NVIDIA GPU run the tools? Prints one line saying why, and
# exits:
#   0  usable
#   1  no NVIDIA GPU
#   2  a card the tools cannot use: older than Turing
#   3  a usable card whose driver is missing, broken or too old: installing
#      the current driver (scripts/install-nvidia-driver.sh) fixes it
#
#     sh scripts/check-gpu.sh
#
# The tools' virtualenvs carry CUDA 12.8 builds of PyTorch. CUDA 12 runs on any
# driver from the 525 branch on (minor-version compatibility); older drivers
# cannot load it at all. And those builds hold kernels for compute capability
# 7.5 and newer only -- Turing (RTX 20xx, T4, Quadro RTX) onwards -- so an older
# card has nothing to run them with. Blackwell cards (RTX 50xx, compute
# capability 10.0 and up) are in those builds too, but their driver support
# starts with the 570 branch. Either way a run would fail at its first CUDA
# call; knowing it at install time lets the server start on CPU instead.

MIN_DRIVER=525
MIN_DRIVER_BLACKWELL=570
MIN_COMPUTE=75      # 7.5, without the dot
BLACKWELL_COMPUTE=100

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    # No driver to ask, so ask the PCI bus: a card sitting there without a
    # driver is the common case on a freshly installed machine.
    card=$(lspci 2>/dev/null | grep -iE 'vga|3d controller|display' | grep -i nvidia | head -n1 | sed 's/^[^ ]* //; s/^[^:]*: *//')
    if [ -n "$card" ]; then
        echo "$card: no working NVIDIA driver"
        exit 3
    fi
    echo "no NVIDIA GPU"
    exit 1
fi

line=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n1)
name=$(echo "$line" | cut -d, -f1 | sed 's/^ *//; s/ *$//')
driver=$(echo "$line" | cut -d, -f2 | sed 's/^ *//; s/ *$//')
major=${driver%%.*}
case "$major" in ''|*[!0-9]*) major=0 ;; esac
if [ "$major" -lt "$MIN_DRIVER" ]; then
    echo "$name, driver $driver: too old for the tools' CUDA 12.8 (needs $MIN_DRIVER or newer)"
    exit 3
fi

# compute_cap is a query field from driver 510 on, which every driver past the
# check above has. Unreadable means unknown, and unknown is not refused.
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' .')
case "$cap" in
    ''|*[!0-9]*) ;;
    *)
        if [ "$cap" -lt "$MIN_COMPUTE" ]; then
            echo "$name (compute capability ${cap%?}.${cap#?}): older than Turing, which the tools' CUDA builds need (7.5 or newer)"
            exit 2
        fi
        if [ "$cap" -ge "$BLACKWELL_COMPUTE" ] && [ "$major" -lt "$MIN_DRIVER_BLACKWELL" ]; then
            echo "$name, driver $driver: a Blackwell card needs driver $MIN_DRIVER_BLACKWELL or newer"
            exit 3
        fi
        ;;
esac

echo "$name, driver $driver: usable"
exit 0
