#!/usr/bin/env bash
#
# Build + run the standalone paged-varlen split-QKV + attention correctness probe.
#
# This intentionally bypasses Cargo and model loading:
#   - compiles only the probe plus the two CUDA kernels with nvcc;
#   - compares varlen split_qkv_norm_rope_into_paged_cache Q/K/V output against CPU;
#   - feeds the produced paged Q/K/V buffers into paged_varlen_attention and checks CPU parity;
#   - covers qk_mode=1 (QK-norm + half-split RoPE), qk_mode=2, qk_mode=3,
#     and sliding_window=0/3.
#
# Usage:
#   bash scripts/microbenches/build_and_run_paged_varlen_split_qkv_correctness.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/ferrum-infer-rs}"
cd "$REPO_ROOT"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_89}"
SRC="scripts/microbenches/paged_varlen_split_qkv_correctness.cu"
SPLIT_SRC="crates/ferrum-kernels/kernels/split_qkv_norm_rope_into_paged_cache.cu"
ATTN_SRC="crates/ferrum-kernels/kernels/paged_varlen_attention.cu"
OUT_BIN="${OUT_BIN:-/tmp/paged_varlen_split_qkv_correctness}"

echo "[microbench] compiling $SRC + $SPLIT_SRC + $ATTN_SRC"
"$NVCC" -O3 -arch="$ARCH" -std=c++17 \
    -Icrates/ferrum-kernels/kernels \
    "$SRC" \
    "$SPLIT_SRC" \
    "$ATTN_SRC" \
    -lcuda -lcudart \
    -o "$OUT_BIN"

ls -la "$OUT_BIN"
echo
echo "[microbench] running $OUT_BIN"
"$OUT_BIN"
