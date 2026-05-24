#!/usr/bin/env bash
#
# GPU hardware lock-down — PLAYBOOK § 0.1.
#
# Locks GPU clocks, sets power limit, enables persistence mode, and
# disables auto-boost. Without this, run-to-run variance routinely
# exceeds the regression signal we're trying to measure.
#
# Usage:
#   sudo scripts/lock_gpu.sh              # 4090 defaults (350W, 2520MHz)
#   sudo scripts/lock_gpu.sh -p 400 -c 2700
#
# Run unlock_gpu.sh to restore default behaviour at end of bench.

set -euo pipefail

POWER_LIMIT=350      # watts; 4090 max
CLOCK_LOCK=2520      # MHz; 4090 base
GPU_INDEX=0          # which GPU (single-GPU pods are always 0)

while getopts ":p:c:g:h" opt; do
    case "$opt" in
        p) POWER_LIMIT="$OPTARG" ;;
        c) CLOCK_LOCK="$OPTARG" ;;
        g) GPU_INDEX="$OPTARG" ;;
        h)
            sed -n '2,12p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "usage: $0 [-p power_W] [-c clock_MHz] [-g gpu_idx]" >&2
            exit 1
            ;;
    esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found — this is a CUDA-only script." >&2
    echo "  macOS users: see PLAYBOOK § 0.2 for Metal mitigation (caffeinate)." >&2
    exit 1
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: must run as root (nvidia-smi -pm/-pl/-lgc require sudo)." >&2
    echo "  hint: sudo $0 $*" >&2
    exit 1
fi

echo "Locking GPU $GPU_INDEX: power=${POWER_LIMIT}W, clock=${CLOCK_LOCK}MHz"

# 1. Persistence mode — keeps driver loaded between probes so cold-start
#    latency doesn't pollute the first bench cell.
nvidia-smi -i "$GPU_INDEX" -pm 1 >/dev/null

# 2. Power limit — pinning at max prevents thermal-induced steps.
nvidia-smi -i "$GPU_INDEX" -pl "$POWER_LIMIT" >/dev/null

# 3. Disable auto-boost (defensive; not all SKUs support this).
nvidia-smi -i "$GPU_INDEX" --auto-boost-default=0 >/dev/null 2>&1 || true

# 4. Lock graphics clock at a specific MHz — the big one.
nvidia-smi -i "$GPU_INDEX" -lgc "$CLOCK_LOCK","$CLOCK_LOCK" >/dev/null

# Verify state — fail-fast if anything didn't take.
ACTUAL=$(nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=clocks.gr,clocks.max.gr,power.limit,persistence_mode \
    --format=csv,noheader,nounits)
GR_CLOCK=$(echo "$ACTUAL" | cut -d, -f1 | tr -d ' ')
PERSIST=$(echo "$ACTUAL" | cut -d, -f4 | tr -d ' ')

echo "  graphics clock:    ${GR_CLOCK} MHz  (locked to ${CLOCK_LOCK})"
echo "  power limit:       $(echo "$ACTUAL" | cut -d, -f3 | tr -d ' ') W  (set ${POWER_LIMIT})"
echo "  persistence mode:  $PERSIST"

# Clock should be within 5% of target (some variance is allowed; if
# driver refuses to lock, GR_CLOCK will be way off).
TOLERANCE=$(echo "$CLOCK_LOCK * 0.05" | bc -l)
DIFF=$(echo "$GR_CLOCK - $CLOCK_LOCK" | bc -l | tr -d '-')
if (( $(echo "$DIFF > $TOLERANCE" | bc -l) )); then
    echo "ERROR: clock lock did not take (actual ${GR_CLOCK} vs target ${CLOCK_LOCK})." >&2
    echo "  pod's GPU may not support clock locking under this driver." >&2
    exit 1
fi
if [ "$PERSIST" != "Enabled" ]; then
    echo "ERROR: persistence mode not Enabled." >&2
    exit 1
fi

echo
echo "✓ GPU $GPU_INDEX locked. Run unlock_gpu.sh at end of bench."
