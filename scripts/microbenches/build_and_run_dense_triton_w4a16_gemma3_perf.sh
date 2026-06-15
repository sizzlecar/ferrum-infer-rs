#!/usr/bin/env bash
#
# Build + run the standalone Gemma3 dense GPTQ Marlin vs Triton W4A16 probe.
#
# This bypasses Cargo and product model loading:
#   - compiles the C++/CUDA harness plus Marlin's CUDA kernel;
#   - loads the committed Triton W4A16 PTX through the CUDA Driver API;
#   - compares Gemma3 tail-MLP gate_up/down projection shapes.
#
# Usage:
#   bash scripts/microbenches/build_and_run_dense_triton_w4a16_gemma3_perf.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/ferrum-infer-rs}"
cd "$REPO_ROOT"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_89}"
SRC="scripts/microbenches/dense_triton_w4a16_gemma3_perf.cu"
KERNEL_SRC="crates/ferrum-kernels/kernels/marlin_cuda_kernel.cu"
PTX="${PTX:-crates/ferrum-kernels/triton_ptx/w4a16_gptq_f16.ptx}"
OUT_BIN="${OUT_BIN:-/tmp/dense_triton_w4a16_gemma3_perf}"

echo "[microbench] compiling $SRC + $KERNEL_SRC"
"$NVCC" -O3 -arch="$ARCH" -std=c++17 \
    "$SRC" \
    "$KERNEL_SRC" \
    -lcuda -lcudart \
    -o "$OUT_BIN"

ls -la "$OUT_BIN"
echo
echo "[microbench] running $OUT_BIN $PTX"
"$OUT_BIN" "$PTX"
