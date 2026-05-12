// FP8 quantization stub for ferrum's vendored vLLM paged-attention v2.
//
// vLLM's `attention_kernels.cuh` includes
// `../quantization/w8a8/fp8/nvidia/quant_utils.cuh` to get `fp8::scaled_convert`
// — used when `KV_DTYPE != kAuto`. Ferrum's single instantiation
// (HEAD=128, BLOCK=16, FP16, KV_DTYPE=kAuto) never reaches those paths, so we
// provide a minimal stub that satisfies the include without pulling in the
// whole w8a8 vendored subtree.
//
// If you later add FP8 KV cache, replace this stub with the real vendored file
// from vllm/csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh.
#pragma once

#include <cuda_fp16.h>

namespace vllm {
namespace fp8 {

// Identity-conversion stubs — referenced inside `if constexpr` guarded by
// `KV_DTYPE != kAuto`. The compiler still has to parse the template body even
// when the branch is dead, so the signatures must exist. Returning the input
// unchanged keeps semantics sound if someone accidentally instantiates with
// FP8 KV without rebuilding the real header.
template <typename Tout, typename Tin, int KV_DTYPE>
__inline__ __device__ Tout scaled_convert(const Tin& x, const float scale) {
    return static_cast<Tout>(x);
}

}  // namespace fp8
}  // namespace vllm
