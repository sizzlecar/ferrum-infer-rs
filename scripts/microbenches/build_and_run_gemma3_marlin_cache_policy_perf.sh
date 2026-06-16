#!/usr/bin/env bash
#
# Build + run a native-CUDA A/B for Marlin B-weight cp.async cache policy.
#
# This compiles the existing Gemma3 tail-MLP chain probe twice:
#   1. baseline: current product Marlin kernel
#   2. evict_first: FERRUM_MARLIN_CP_ASYNC_EVICT_FIRST=1
#
# The variant is compile-time only and does not change default product behavior.
#
# Usage:
#   bash scripts/microbenches/build_and_run_gemma3_marlin_cache_policy_perf.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/ferrum-infer-rs}"
cd "$REPO_ROOT"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_89}"
SRC="scripts/microbenches/gemma3_tail_mlp_chain_perf.cu"
BASE_BIN="${BASE_BIN:-/tmp/gemma3_tail_mlp_cache_policy_baseline}"
EVICT_BIN="${EVICT_BIN:-/tmp/gemma3_tail_mlp_cache_policy_evict_first}"
KERNEL_SRCS=(
    "crates/ferrum-kernels/kernels/sandwich_norm.cu"
    "crates/ferrum-kernels/kernels/fused_silu_mul.cu"
    "crates/ferrum-kernels/kernels/marlin_cuda_kernel.cu"
)

compile_variant() {
    local label="$1"
    local out_bin="$2"
    shift 2
    local extra_flags=("$@")

    echo "[microbench] compiling ${label}: ${SRC}"
    "$NVCC" -O3 -arch="$ARCH" -std=c++17 \
        "${extra_flags[@]}" \
        "$SRC" \
        "${KERNEL_SRCS[@]}" \
        -lcuda -lcudart \
        -o "$out_bin"
    ls -la "$out_bin"
}

run_variant() {
    local label="$1"
    local bin="$2"
    echo
    echo "### variant=${label}"
    "$bin"
}

compile_variant baseline "$BASE_BIN"
compile_variant evict_first "$EVICT_BIN" -DFERRUM_MARLIN_CP_ASYNC_EVICT_FIRST=1

run_variant baseline "$BASE_BIN"
run_variant evict_first "$EVICT_BIN"

echo
echo "VERDICT: gemma3 Marlin cache-policy native CUDA probe complete"
