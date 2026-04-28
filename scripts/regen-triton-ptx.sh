#!/bin/bash
# Regenerate ferrum-kernels/triton_ptx/ from triton-rs DSL kernels.
#
# Run on a Linux + CUDA box with sibling clones of ferrum-infer-rs and
# triton-rs. First run takes ~30 min (cmake-builds vendored Triton C++);
# subsequent runs are minutes.
#
# Outputs:
#   crates/ferrum-kernels/triton_ptx/<kernel>.ptx     — PTX text
#   crates/ferrum-kernels/triton_ptx/<kernel>.json    — kernel metadata
#
# Required env (or auto-detected):
#   FERRUM_DIR     default: this repo
#   TRITONRS_DIR   default: ../triton-rs relative to FERRUM_DIR
#   ARCH           default: 89 (sm_89 PTX; driver JITs forward on Blackwell)

set -euo pipefail

FERRUM_DIR="${FERRUM_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
TRITONRS_DIR="${TRITONRS_DIR:-$(cd "$FERRUM_DIR/../triton-rs" && pwd)}"
ARCH="${ARCH:-89}"
OUT_DIR="$FERRUM_DIR/crates/ferrum-kernels/triton_ptx"

if [ ! -d "$TRITONRS_DIR" ]; then
    echo "error: triton-rs not found at $TRITONRS_DIR — set TRITONRS_DIR or clone it." >&2
    exit 1
fi

# Resolve LLVM sysroot and libdevice (auto-discovered or via env override).
LLVM_REV=$(head -c 8 "$TRITONRS_DIR/crates/triton-sys/vendor/triton/cmake/llvm-hash.txt" 2>/dev/null || echo "")
if [ -z "$LLVM_REV" ]; then
    echo "warning: triton-rs vendor not yet fetched — will be done by build.rs on first cargo invocation."
fi
export TRITON_LLVM_SYSPATH="${TRITON_LLVM_SYSPATH:-$HOME/.cache/triton-rs/llvm/llvm-${LLVM_REV}-ubuntu-x64}"
export TRITON_LIBDEVICE_PATH="${TRITON_LIBDEVICE_PATH:-$TRITONRS_DIR/crates/triton-sys/vendor/triton/third_party/nvidia/backend/lib/libdevice.10.bc}"

echo "FERRUM_DIR     = $FERRUM_DIR"
echo "TRITONRS_DIR   = $TRITONRS_DIR"
echo "ARCH           = $ARCH"
echo "OUT_DIR        = $OUT_DIR"
echo

mkdir -p "$OUT_DIR"

# Map: <triton-dsl example>:<output basename>:<MLIR func name>
KERNELS=(
    "ferrum_rms_norm:rms_norm_f32:rms_norm_f32"
    # Add more here as kernels migrate. Each must have a triton-dsl example
    # and a matching ferrum-kernels Rust dispatcher.
)

cd "$TRITONRS_DIR"

for entry in "${KERNELS[@]}"; do
    IFS=: read -r ex base mlir_fn <<< "$entry"
    work=$(mktemp -d)
    echo "== $base =="
    cargo run --quiet --example "$ex" -p triton-dsl > "$work/kernel.mlir"
    cargo run --quiet --release -p triton-sys --features compile-triton \
        --example compile_mlir -- "$work/kernel.mlir" "$work" --arch "$ARCH"
    cp "$work/kernel.ptx"  "$OUT_DIR/${base}.ptx"
    cp "$work/kernel.json" "$OUT_DIR/${base}.json"
    rm -rf "$work"
    echo "   wrote $OUT_DIR/${base}.{ptx,json}"
done

echo
echo "done. commit the changes under $OUT_DIR/ to lock in the triton-rs binaries."
