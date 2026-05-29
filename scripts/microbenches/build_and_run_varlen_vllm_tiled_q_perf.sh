#!/usr/bin/env bash
#
# Build + run the standalone vLLM-layout varlen attention tiled-Q microbench.
#
# Usage from the repo root on an sm_89 GPU host:
#   bash scripts/microbenches/build_and_run_varlen_vllm_tiled_q_perf.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
cd "$REPO_ROOT"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_89}"
SRC="scripts/microbenches/varlen_vllm_tiled_q_perf.cu"
OUT_BIN="/tmp/varlen_vllm_tiled_q_perf"

echo "[microbench] compiling $SRC"
"$NVCC" -O3 -arch="$ARCH" -std=c++17 "$SRC" -o "$OUT_BIN"

echo "[microbench] running $OUT_BIN"
"$OUT_BIN"
