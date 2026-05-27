#!/usr/bin/env bash
#
# Build + run moe_marlin_perf microbench using ferrum's already-compiled
# CUDA objects. Run AFTER cargo build with `--features vllm-moe-marlin`
# has finished (otherwise the .o files don't exist).
#
# Usage:
#   bash scripts/microbenches/build_and_run_moe_marlin_perf.sh
#     [target/release/build/ferrum-kernels-<hash>/out]

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/ferrum-infer-rs}"
cd "$REPO_ROOT"

# Auto-detect the ferrum-kernels build out dir (cargo names it with a content hash)
OUT_DIR="${1:-}"
if [ -z "$OUT_DIR" ]; then
    OUT_DIR=$(find target/release/build -maxdepth 2 -type d -name 'ferrum-kernels-*' \
        -exec test -d {}/out \; -print 2>/dev/null | head -1)
    if [ -n "$OUT_DIR" ]; then OUT_DIR="$OUT_DIR/out"; fi
fi
if [ -z "$OUT_DIR" ] || [ ! -d "$OUT_DIR" ]; then
    echo "ERROR: ferrum-kernels OUT_DIR not found. Build ferrum first:" >&2
    echo "  cargo build --release -p ferrum-cli --features cuda,vllm-moe-marlin" >&2
    exit 1
fi
echo "[microbench] using OUT_DIR=$OUT_DIR"

# Required objects (extern C symbols we link against)
NEEDED=(
    "$OUT_DIR/vllm_moe_ops.o"                        # ferrum_vllm_marlin_moe_f16
    "$OUT_DIR/vllm_moe_kernel_instantiations.o"      # Marlin<...> template specializations
    "$OUT_DIR/gptq_marlin_repack.o"                  # ferrum_vllm_gptq_marlin_repack
)
for f in "${NEEDED[@]}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f — vllm-moe-marlin feature didn't compile" >&2
        exit 1
    fi
done

# All sm89_kernel_*.o and sm80_kernel_*.o objects from libvllm_marlin —
# they contain the actual Marlin specializations the moe wrapper calls.
KERNEL_OBJS=( "$OUT_DIR"/sm89_kernel_*.o "$OUT_DIR"/sm80_kernel_*.o )

# Compile + link the microbench. nvcc handles .cu compile and .o link in one pass.
NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_89}"
SRC="scripts/microbenches/moe_marlin_perf.cu"
OUT_BIN="/tmp/moe_marlin_perf"

echo "[microbench] compiling $SRC + linking ${#NEEDED[@]} ferrum objs + ${#KERNEL_OBJS[@]} Marlin instantiations"
"$NVCC" -O3 -arch="$ARCH" -std=c++17 \
    "$SRC" \
    "${NEEDED[@]}" \
    "${KERNEL_OBJS[@]}" \
    -lcuda -lcudart \
    -o "$OUT_BIN" 2>&1 | tail -25

if [ ! -x "$OUT_BIN" ]; then
    echo "ERROR: nvcc did not produce binary" >&2
    exit 1
fi

ls -la "$OUT_BIN"
echo
echo "[microbench] running $OUT_BIN"
"$OUT_BIN" 2>&1
