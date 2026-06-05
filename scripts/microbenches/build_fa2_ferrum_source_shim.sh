#!/usr/bin/env bash
set -euo pipefail

# Build the FlashAttention-2 C ABI diagnostic source shim for Ferrum.
#
# This is deliberately different from build_fa2_ferrum_shim.sh: it does not
# link vLLM's _vllm_fa2_C Torch extension and does not require libtorch,
# libpython, or vLLM at runtime. It compiles the two Qwen3 M3 forward templates
# we need directly from a FlashAttention source checkout:
#   - hdim128 fp16 split-K, causal=false
#   - hdim128 fp16 split-K, causal=true
#
# This script is not used by product or release builds. FA_SRC_DIR must point at
# an explicit FlashAttention checkout when running this legacy diagnostic.
# CUTLASS_INCLUDE_DIR must point at a CUTLASS 3.x include tree.

FA_SRC_DIR="${FA_SRC_DIR:-}"
FA_GIT_URL="${FA_GIT_URL:-https://github.com/vllm-project/flash-attention.git}"
FA_GIT_REV="${FA_GIT_REV:-f5bc33cfc02c744d24a2e9d50e6db656de40611c}"
CUTLASS_INCLUDE_DIR="${CUTLASS_INCLUDE_DIR:-}"
SRC="${SRC:-scripts/microbenches/fa2_ferrum_source_shim.cu}"
OUT_SO="${OUT_SO:-/workspace/libferrum_fa2_source_shim.so}"
BUILD_DIR="${BUILD_DIR:-/tmp/ferrum-fa2-source-shim-build}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-89}"
NVCC_THREADS="${FERRUM_NVCC_THREADS:-0}"

if [[ -z "$FA_SRC_DIR" ]]; then
  echo "FA_SRC_DIR must point at a FlashAttention source checkout for this legacy diagnostic" >&2
  exit 1
fi

if [[ ! -d "$FA_SRC_DIR/.git" ]]; then
  echo "[fa2-source-shim] cloning FlashAttention source to $FA_SRC_DIR"
  git clone --filter=blob:none "$FA_GIT_URL" "$FA_SRC_DIR"
fi
git -C "$FA_SRC_DIR" checkout -q "$FA_GIT_REV"

if [[ -z "$CUTLASS_INCLUDE_DIR" ]]; then
  for candidate in \
    /workspace/vllm-venv/lib/python3.12/site-packages/flashinfer/data/cutlass/include \
    /workspace/vllm-venv/lib/python3.12/site-packages/tilelang/3rdparty/cutlass/include \
    /workspace/vllm-venv/lib/python3.12/site-packages/vllm/third_party/deep_gemm/include; do
    if [[ -f "$candidate/cute/tensor.hpp" && -f "$candidate/cutlass/cutlass.h" ]]; then
      CUTLASS_INCLUDE_DIR="$candidate"
      break
    fi
  done
fi

if [[ -z "$CUTLASS_INCLUDE_DIR" || ! -f "$CUTLASS_INCLUDE_DIR/cute/tensor.hpp" ]]; then
  echo "CUTLASS_INCLUDE_DIR must point at CUTLASS/CUTE headers" >&2
  exit 1
fi

NVCC="$CUDA_ROOT/bin/nvcc"
if [[ ! -x "$NVCC" ]]; then
  NVCC="$(command -v nvcc || true)"
fi
if [[ -z "$NVCC" || ! -x "$NVCC" ]]; then
  echo "nvcc not found; set CUDA_ROOT or PATH" >&2
  exit 1
fi

STUB_DIR="$BUILD_DIR/stubs"
mkdir -p "$STUB_DIR/ATen/cuda/detail" "$STUB_DIR/c10/cuda" "$(dirname "$OUT_SO")"

cat > "$STUB_DIR/ferrum_fa2_prelude.h" <<'HDR'
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <tuple>
HDR

cat > "$STUB_DIR/ATen/cuda/CUDAGeneratorImpl.h" <<'HDR'
#pragma once
#include <cstdint>

namespace at {

struct PhiloxCudaState {
    PhiloxCudaState() = default;

    union Payload {
        uint64_t val;
        int64_t *ptr;
    };

    Payload seed_{};
    Payload offset_{};
    uint64_t offset_intragraph_ = 0;
    bool captured_ = false;
};

}  // namespace at
HDR

cat > "$STUB_DIR/ATen/cuda/detail/UnpackRaw.cuh" <<'HDR'
#pragma once
#include <cstdint>
#include <tuple>

namespace at::cuda::philox {

__host__ __device__ __forceinline__ std::tuple<uint64_t, uint64_t>
unpack(at::PhiloxCudaState arg) {
    if (arg.captured_) {
        return std::make_tuple(static_cast<uint64_t>(*arg.seed_.ptr),
                               static_cast<uint64_t>(*(arg.offset_.ptr) +
                                                     arg.offset_intragraph_));
    }
    return std::make_tuple(arg.seed_.val, arg.offset_.val);
}

}  // namespace at::cuda::philox
HDR

cat > "$STUB_DIR/c10/cuda/CUDAException.h" <<'HDR'
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define C10_CUDA_CHECK(EXPR)                                                   \
    do {                                                                       \
        cudaError_t ferrum_cuda_status_ = (EXPR);                              \
        if (ferrum_cuda_status_ != cudaSuccess) {                              \
            std::fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(ferrum_cuda_status_));   \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaPeekAtLastError())
HDR

COMMON_FLAGS=(
  "-O3"
  "-std=c++17"
  "--use_fast_math"
  "--expt-relaxed-constexpr"
  "--expt-extended-lambda"
  "-gencode=arch=compute_${CUDA_COMPUTE_CAP},code=sm_${CUDA_COMPUTE_CAP}"
  "--threads=$NVCC_THREADS"
  "-Xcompiler=-fPIC"
  "-Xcompiler=-fvisibility=hidden"
  "-I$STUB_DIR"
  "-I$FA_SRC_DIR/csrc/flash_attn/src"
  "-I$CUTLASS_INCLUDE_DIR"
  "-I$CUDA_ROOT/include"
  "-include=$STUB_DIR/ferrum_fa2_prelude.h"
)

SOURCES=(
  "$SRC"
  "$FA_SRC_DIR/csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_sm80.cu"
  "$FA_SRC_DIR/csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_causal_sm80.cu"
)

echo "[fa2-source-shim] compiling $OUT_SO"
echo "[fa2-source-shim] fa_src=$FA_SRC_DIR"
echo "[fa2-source-shim] cutlass=$CUTLASS_INCLUDE_DIR"
"$NVCC" "${COMMON_FLAGS[@]}" -shared "${SOURCES[@]}" -o "$OUT_SO"

echo "[fa2-source-shim] wrote $OUT_SO"
ldd "$OUT_SO" | sed 's/^/[fa2-source-shim] /'
nm -D "$OUT_SO" | grep ferrum_fa2_paged_varlen_fwd | sed 's/^/[fa2-source-shim] /'
