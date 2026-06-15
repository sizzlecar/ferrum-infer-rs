#!/usr/bin/env bash
#
# Build + run the standalone Gemma3 tail-MLP native-CUDA chain probe.
#
# This bypasses Cargo and model loading. It compiles only the probe plus the
# product CUDA kernels needed by the tail MLP chain:
#   sandwich_norm.cu, fused_silu_mul.cu, and marlin_cuda_kernel.cu.
#
# Usage:
#   bash scripts/microbenches/build_and_run_gemma3_tail_mlp_chain_perf.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/ferrum-infer-rs}"
cd "$REPO_ROOT"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_89}"
SRC="scripts/microbenches/gemma3_tail_mlp_chain_perf.cu"
OUT_BIN="${OUT_BIN:-/tmp/gemma3_tail_mlp_chain_perf}"
KERNEL_SRCS=(
    "crates/ferrum-kernels/kernels/sandwich_norm.cu"
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
