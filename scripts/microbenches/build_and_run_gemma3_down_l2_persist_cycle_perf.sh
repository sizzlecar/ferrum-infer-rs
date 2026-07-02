#!/usr/bin/env bash
#
# Build + run the standalone Gemma3 down_proj L2 persistence cycle probe.
#
# This bypasses Cargo and model loading. It compiles only the probe plus the
# product CUDA kernels needed by the MLP producer chain:
#   fused_silu_mul.cu and marlin_cuda_kernel.cu.
#
# Usage:
#   bash scripts/microbenches/build_and_run_gemma3_down_l2_persist_cycle_perf.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/ferrum-infer-rs}"
cd "$REPO_ROOT"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_89}"
SRC="scripts/microbenches/gemma3_down_l2_persist_cycle_perf.cu"
OUT_BIN="${OUT_BIN:-/tmp/gemma3_down_l2_persist_cycle_perf}"
KERNEL_SRCS=(
    "crates/ferrum-kernels/kernels/fused_silu_mul.cu"
    "crates/ferrum-kernels/kernels/marlin_cuda_kernel.cu"
)

echo "[microbench] compiling $SRC"
"$NVCC" -O3 -arch="$ARCH" -std=c++17 \
    "$SRC" \
    "${KERNEL_SRCS[@]}" \
    -lcuda -lcudart \
    -o "$OUT_BIN"

ls -la "$OUT_BIN"
echo
echo "[microbench] running $OUT_BIN"
"$OUT_BIN"
