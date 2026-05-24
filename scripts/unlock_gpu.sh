#!/usr/bin/env bash
#
# Restore GPU defaults after bench — PLAYBOOK § 0.1 counterpart.
#
# Releases the clock lock and re-enables auto-boost. Persistence mode
# is left on (it's harmless and avoids re-paying the cold-load tax
# on subsequent runs in the same pod session).
#
# Usage:
#   sudo scripts/unlock_gpu.sh
#   sudo scripts/unlock_gpu.sh -g 0

set -euo pipefail

GPU_INDEX=0

while getopts ":g:h" opt; do
    case "$opt" in
        g) GPU_INDEX="$OPTARG" ;;
        h)
            sed -n '2,11p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "usage: $0 [-g gpu_idx]" >&2
            exit 1
            ;;
    esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found." >&2
    exit 1
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: must run as root." >&2
    echo "  hint: sudo $0 $*" >&2
    exit 1
fi

nvidia-smi -i "$GPU_INDEX" -rgc >/dev/null
nvidia-smi -i "$GPU_INDEX" --auto-boost-default=1 >/dev/null 2>&1 || true

ACTUAL=$(nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=clocks.gr,clocks.max.gr \
    --format=csv,noheader,nounits)

echo "✓ GPU $GPU_INDEX unlocked. graphics clock now: $(echo "$ACTUAL" | cut -d, -f1 | tr -d ' ') MHz"
