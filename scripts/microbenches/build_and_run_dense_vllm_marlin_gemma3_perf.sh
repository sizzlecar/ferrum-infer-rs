#!/usr/bin/env bash
#
# Build + run the standalone dense vLLM Marlin Gemma3 native-CUDA probe.
#
# This intentionally bypasses Cargo and product model loading. It builds a
# temporary, selector-enabled copy of the vendored vLLM dense Marlin source
# with only the FP16/U4B8 selector branches needed by Gemma3 m=16/23/32 shapes.
#
# Usage:
#   bash scripts/microbenches/build_and_run_dense_vllm_marlin_gemma3_perf.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
cd "$REPO_ROOT"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_89}"
BUILD_DIR="${BUILD_DIR:-/tmp/ferrum_dense_vllm_marlin_probe_build}"
OUT_BIN="${OUT_BIN:-/tmp/dense_vllm_marlin_gemma3_perf}"
SRC="scripts/microbenches/dense_vllm_marlin_gemma3_perf.cu"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cp -R crates/ferrum-kernels/vllm_marlin "$BUILD_DIR/vllm_marlin"

cat > "$BUILD_DIR/vllm_marlin/kernel_selector.h" <<'EOF'
// Minimal selector for dense_vllm_marlin_gemma3_perf.cu.
// Covers FP16/U4B8/FP16/FP16, group_size=128 -> group_blocks=8.
if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 &&
    c_type == vllm::kFloat16 && s_type == vllm::kFloat16 &&
    threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 &&
    thread_k_blocks == 8 && m_block_size_8 == false && stages == 4 &&
    group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(),
                  vllm::kFloat16.id(), 256, 1, 8, 8, false, 4, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 &&
         c_type == vllm::kFloat16 && s_type == vllm::kFloat16 &&
         threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 &&
         thread_k_blocks == 4 && m_block_size_8 == false && stages == 4 &&
         group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(),
                  vllm::kFloat16.id(), 256, 2, 16, 4, false, 4, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 &&
         c_type == vllm::kFloat16 && s_type == vllm::kFloat16 &&
         threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 &&
         thread_k_blocks == 8 && m_block_size_8 == false && stages == 4 &&
         group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(),
                  vllm::kFloat16.id(), 128, 2, 4, 8, false, 4, 8, false>;
EOF

perl -0pi -e 's@// #include "kernel_selector\\.h"@#include "kernel_selector.h"@' \
  "$BUILD_DIR/vllm_marlin/marlin.cu"

echo "[microbench] compiling selector-enabled vLLM dense Marlin objects"
"$NVCC" -O3 -arch="$ARCH" -std=c++17 --use_fast_math \
  --expt-relaxed-constexpr --expt-extended-lambda \
  -I"$BUILD_DIR/vllm_marlin" -DMARLIN_NAMESPACE_NAME=marlin \
  -Xcompiler -fPIC -Xcompiler -fvisibility=default \
  -c "$BUILD_DIR/vllm_marlin/marlin.cu" \
  -o "$BUILD_DIR/marlin.o"

"$NVCC" -O3 -arch="$ARCH" -std=c++17 --use_fast_math \
  --expt-relaxed-constexpr --expt-extended-lambda \
  -I"$BUILD_DIR/vllm_marlin" -DMARLIN_NAMESPACE_NAME=marlin \
  -Xcompiler -fPIC -Xcompiler -fvisibility=default \
  -c "$BUILD_DIR/vllm_marlin/sm80_kernel_float16_u4b8_float16.cu" \
  -o "$BUILD_DIR/sm80_kernel_float16_u4b8_float16.o"

echo "[microbench] compiling $SRC"
"$NVCC" -O3 -arch="$ARCH" -std=c++17 \
  "$SRC" \
  "$BUILD_DIR/marlin.o" \
  "$BUILD_DIR/sm80_kernel_float16_u4b8_float16.o" \
  -lcuda -lcudart \
  -o "$OUT_BIN"

ls -la "$OUT_BIN"
echo
echo "[microbench] running $OUT_BIN"
"$OUT_BIN"
