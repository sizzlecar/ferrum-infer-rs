#!/bin/bash
# Run the native CUDA graph reproducer under gdb.
# Usage: bash run_graph_repro.sh [output_bin]
#
# Expected outcomes:
#   A) Binary prints all [repro] lines and exits 0 → graph works; our Rust
#      code has a trigger the bare C++ path doesn't hit. Look for deltas in:
#      event tracking / cuMemPool state / cuCtxSetCurrent / cudarc internals.
#   B) SIGSEGV inside libcuda.so at cuGraphLaunch → driver bug confirmed
#      independent of our code. Nothing we can fix at app layer.

set -u
cd "$(dirname "$0")/.."

OUT="${1:-/tmp/graph_repro}"
SRC="scripts/graph_repro.cu"

# Find nvcc — prefer CUDA 13 since that's where we see the crash.
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
if ! command -v "$NVCC" > /dev/null 2>&1; then
    NVCC="/usr/local/cuda-13.0/bin/nvcc"
fi

echo "== compile =="
"$NVCC" -o "$OUT" "$SRC" -lcuda

echo
echo "== run plain =="
"$OUT" || echo "(plain run exited non-zero, will re-run under gdb for trace)"

echo
echo "== run under gdb =="
gdb \
    -batch \
    -ex "set pagination off" \
    -ex "set confirm off" \
    -ex "handle SIGSEGV stop print" \
    -ex run \
    -ex "bt 15" \
    -ex "info registers rip rsp" \
    --args "$OUT" 2>&1 | tail -60
