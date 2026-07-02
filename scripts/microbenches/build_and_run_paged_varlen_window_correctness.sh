#!/usr/bin/env bash
#
# Build + run the standalone paged-varlen sliding-window correctness probe.
#
# This intentionally bypasses Cargo and model loading:
#   - compiles only the probe plus paged_varlen_attention.cu with nvcc;
#   - compares sliding_window=0 and sliding_window=3 against CPU reference;
#   - checks both one-pass and split-K varlen attention kernels.
#
# Usage:
#   bash scripts/microbenches/build_and_run_paged_varlen_window_correctness.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/ferrum-infer-rs}"
cd "$REPO_ROOT"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_89}"
SRC="scripts/microbenches/paged_varlen_window_correctness.cu"
KERNEL_SRC="crates/ferrum-kernels/kernels/paged_varlen_attention.cu"
OUT_BIN="${OUT_BIN:-/tmp/paged_varlen_window_correctness}"

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
