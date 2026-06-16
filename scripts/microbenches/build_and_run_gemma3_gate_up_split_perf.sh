#!/usr/bin/env bash
#
# Build + run the standalone Gemma3 gate_up split-vs-fused native-CUDA probe.
#
# This bypasses Cargo and product model loading:
#   - compiles the probe plus Marlin and GeGLU CUDA kernels;
#   - compares product fused gate_up projection against split gate/up variants;
#   - keeps an 8-layer weight cycle so warm single-layer rows do not overstate
#     product relevance.
#
# Usage:
#   bash scripts/microbenches/build_and_run_gemma3_gate_up_split_perf.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/ferrum-infer-rs}"
cd "$REPO_ROOT"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_89}"
SRC="scripts/microbenches/gemma3_gate_up_split_perf.cu"
OUT_BIN="${OUT_BIN:-/tmp/gemma3_gate_up_split_perf}"
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
