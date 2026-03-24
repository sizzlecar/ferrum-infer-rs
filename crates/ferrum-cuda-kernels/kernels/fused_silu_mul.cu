// Fused SiLU activation + elementwise multiply kernel.
//
// Replaces 2 kernel launches (silu + mul) with 1.
// Used in MLP gate projection: output = silu(gate) * up
//
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

#include <cuda_fp16.h>

// FP16 version: output[i] = silu(gate[i]) * up[i]
extern "C" __global__ void fused_silu_mul_f16(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        float silu_g = g / (1.0f + expf(-g));
        output[idx] = __float2half(silu_g * u);
    }
}

// FP32 version
extern "C" __global__ void fused_silu_mul_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        float silu_g = g / (1.0f + expf(-g));
        output[idx] = silu_g * up[idx];
    }
}
