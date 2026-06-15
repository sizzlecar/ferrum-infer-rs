#!/usr/bin/env bash
#
# Build + run the standalone Gemma3 device-shadow CUDA graph probe.
#
# This intentionally bypasses Cargo and model loading:
#   - compiles only scripts/microbenches/gemma3_shadow_graph_bench.cu with nvcc;
#   - simulates a 62-layer Gemma3-style device F32 residual shadow decode step;
#   - compares eager launches with CUDA graph replay before product graph edits.
#
# Usage:
#   bash scripts/microbenches/build_and_run_gemma3_shadow_graph_bench.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
cd "$REPO_ROOT"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_89}"
SRC="scripts/microbenches/gemma3_shadow_graph_bench.cu"
OUT_BIN="${OUT_BIN:-/tmp/gemma3_shadow_graph_bench}"

echo "[microbench] compiling $SRC"
"$NVCC" -O3 -arch="$ARCH" -std=c++17 \
    "$SRC" \
    -lcuda -lcudart \
    -o "$OUT_BIN"

ls -la "$OUT_BIN"
echo
echo "[microbench] running $OUT_BIN"
"$OUT_BIN"
