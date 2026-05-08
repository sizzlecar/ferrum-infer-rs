#!/bin/bash
# Regen fused_moe_w4a16 PTX for one specific tile size.
# Used by the bench-sweep loop — emits PTX/JSON named after the tile,
# loaded individually by the bench harness.
#
# Usage:
#   bash scripts/regen-fused-moe-tile-sweep.sh 16x128x64
#
# Writes:
#   crates/ferrum-kernels/triton_ptx/fused_moe_w4a16_f16_<tile>.ptx
#   crates/ferrum-kernels/triton_ptx/fused_moe_w4a16_f16_<tile>.json
#
# Run on Linux+CUDA box with sibling triton-rs clone.

set -euo pipefail

TILE="${1:?tile arg required, e.g. 16x128x64}"
FERRUM_DIR="${FERRUM_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
TRITONRS_DIR="${TRITONRS_DIR:-$(cd "$FERRUM_DIR/../triton-rs" && pwd)}"
ARCH="${ARCH:-89}"
OUT_DIR="$FERRUM_DIR/crates/ferrum-kernels/triton_ptx"

LLVM_REV=$(head -c 8 "$TRITONRS_DIR/crates/triton-sys/vendor/triton/cmake/llvm-hash.txt")
export TRITON_LLVM_SYSPATH="${TRITON_LLVM_SYSPATH:-$HOME/.cache/triton-rs/llvm/llvm-${LLVM_REV}-ubuntu-x64}"
export TRITON_LIBDEVICE_PATH="${TRITON_LIBDEVICE_PATH:-$TRITONRS_DIR/crates/triton-sys/vendor/triton/third_party/nvidia/backend/lib/libdevice.10.bc}"

cd "$TRITONRS_DIR"
work=$(mktemp -d)
echo "== fused_moe_w4a16 tile=$TILE =="

FERRUM_FUSED_MOE_TILE="$TILE" cargo run --quiet --example ferrum_fused_moe_w4a16 -p triton-dsl > "$work/kernel.mlir"
cargo run --quiet --release -p triton-sys --features compile-triton \
    --example compile_mlir -- "$work/kernel.mlir" "$work" --arch "$ARCH"

# Suffix-name based on tile so multiple variants can coexist.
out_base="fused_moe_w4a16_f16_${TILE//x/_}"
cp "$work/kernel.ptx"  "$OUT_DIR/${out_base}.ptx"
cp "$work/kernel.json" "$OUT_DIR/${out_base}.json"
rm -rf "$work"
echo "wrote $OUT_DIR/${out_base}.{ptx,json}"
