#!/usr/bin/env bash
#
# Build + run a native-CUDA A/B for Marlin B-weight cp.async cache policy.
#
# This compiles the existing Gemma3 tail-MLP chain probe twice:
#   1. legacy_plain: FERRUM_MARLIN_CP_ASYNC_PLAIN=1
#   2. product_default: current product Marlin kernel
#
# Use this after changing the product default to ensure the default remains the
# measured fast path.
#
# Usage:
#   bash scripts/microbenches/build_and_run_gemma3_marlin_cache_policy_perf.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/ferrum-infer-rs}"
cd "$REPO_ROOT"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_89}"
SRC="scripts/microbenches/gemma3_tail_mlp_chain_perf.cu"
LEGACY_BIN="${LEGACY_BIN:-/tmp/gemma3_tail_mlp_cache_policy_legacy_plain}"
DEFAULT_BIN="${DEFAULT_BIN:-/tmp/gemma3_tail_mlp_cache_policy_product_default}"
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

compile_variant legacy_plain "$LEGACY_BIN" -DFERRUM_MARLIN_CP_ASYNC_PLAIN=1
compile_variant product_default "$DEFAULT_BIN"

run_variant legacy_plain "$LEGACY_BIN"
run_variant product_default "$DEFAULT_BIN"

echo
echo "VERDICT: gemma3 Marlin cache-policy native CUDA probe complete"
