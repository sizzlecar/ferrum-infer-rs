#!/usr/bin/env bash
#
# Build and run the native monolithic-vs-segmented CUDA graph probe.
#
# This bypasses Cargo and model loading. It is intended for quick validation
# before changing Ferrum's product CUDA graph capture granularity.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
cd "$REPO_ROOT"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_89}"
SRC="scripts/microbenches/cuda_graph_segment_probe.cu"
OUT_BIN="${OUT_BIN:-/tmp/cuda_graph_segment_probe}"

echo "[microbench] compiling $SRC"
"$NVCC" -O3 -arch="$ARCH" -std=c++17 \
    "$SRC" \
    -lcuda -lcudart \
    -o "$OUT_BIN"

ls -la "$OUT_BIN"
echo
echo "[microbench] running $OUT_BIN $*"
"$OUT_BIN" "$@"
