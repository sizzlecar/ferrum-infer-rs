// Element-wise GELU activation (erf-based, matches PyTorch default):
//   gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
//
// Used by Bert/Clip/Whisper MLPs (LLM path uses silu-mul).
//
// Launch: grid = ((len+255)/256, 1, 1), block = (256, 1, 1).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" __global__ void gelu_f16(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    const int len
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    float v = __half2float(x[idx]);
    // 1/sqrt(2) ≈ 0.7071067811865475
    float g = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
    out[idx] = __float2half(g);
}
