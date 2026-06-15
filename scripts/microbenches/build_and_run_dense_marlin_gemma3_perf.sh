#!/usr/bin/env bash
#
# Build + run the standalone dense Marlin Gemma3 native-CUDA probe.
#
# This intentionally bypasses Cargo and model loading:
#   - compiles only the probe plus marlin_cuda_kernel.cu with nvcc;
#   - allocates synthetic buffers for Gemma3-27B GPTQ projection shapes;
#   - reports per-call hot, host-sync, cold-cache, and multi-weight-cycle timing.
#
# Usage:
#   bash scripts/microbenches/build_and_run_dense_marlin_gemma3_perf.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/ferrum-infer-rs}"
cd "$REPO_ROOT"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_89}"
SRC="scripts/microbenches/dense_marlin_gemma3_perf.cu"
KERNEL_SRC="crates/ferrum-kernels/kernels/marlin_cuda_kernel.cu"
OUT_BIN="${OUT_BIN:-/tmp/dense_marlin_gemma3_perf}"

echo "[microbench] compiling $SRC + $KERNEL_SRC"
"$NVCC" -O3 -arch="$ARCH" -std=c++17 \
    "$SRC" \
    "$KERNEL_SRC" \
    -lcuda -lcudart \
    -o "$OUT_BIN"

ls -la "$OUT_BIN"
echo
echo "[microbench] running $OUT_BIN"
"$OUT_BIN"
