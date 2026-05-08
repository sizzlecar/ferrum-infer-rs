#!/bin/bash
# Sweep fused_moe_w4a16 tile sizes — for each tile, regen PTX, then bench.
# Run on Linux+CUDA pod.
set -uo pipefail

FERRUM_DIR="${FERRUM_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
TILES=(
    "16x64x32"
    "16x128x32"
    "16x128x64"
    "16x256x64"
    "16x64x64"
    "32x64x32"
    "32x128x32"
    "32x128x64"
    "32x256x64"
    "64x128x32"
    "64x128x64"
    "64x256x64"
)

cd "$FERRUM_DIR"

for tile in "${TILES[@]}"; do
    echo
    echo "════ tile=$tile ════"
    bash scripts/regen-fused-moe-tile-sweep.sh "$tile" || { echo "regen failed for $tile"; continue; }
    FERRUM_FUSED_MOE_TILE_BENCH="$tile" \
      cargo test -p ferrum-kernels --features cuda,triton-kernels --release \
        --test triton_fused_moe_tile_sweep -- --ignored --nocapture 2>&1 \
        | grep -E "RESULT|panicked|error:" | tail -3
done

echo
echo "Marlin baseline (Stage 12.1 fused gate_up at c=32): ~183 µs/layer"
