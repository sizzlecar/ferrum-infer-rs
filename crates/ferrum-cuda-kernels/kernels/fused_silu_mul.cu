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

// Batched variant for interleaved gate+up GEMM output.
// gate_up layout: [batch, 2*inter] = [gate_0, up_0, gate_1, up_1, ...]
// output layout: [batch, inter] = [act_0, act_1, ...] (contiguous for down GEMM)
extern "C" __global__ void fused_silu_mul_interleaved_f16(
    const __half* __restrict__ gate_up,  // [batch * 2 * inter]
    __half* __restrict__ output,         // [batch * inter]
    const int inter,                     // intermediate_size
    const int total                      // = batch * inter
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int b = idx / inter;
        int i = idx % inter;
        float g = __half2float(gate_up[b * 2 * inter + i]);
        float u = __half2float(gate_up[b * 2 * inter + inter + i]);
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
