// Standalone RMS normalization kernel (no residual add).
//
// output = rms_norm(input, weight, eps)
//        = input / sqrt(mean(input^2) + eps) * weight
//
// Supports both flat [num_tokens, hidden_size] and per-head [num_heads, head_dim]
// layouts — the kernel just sees num_rows × row_size.

#include "common.cuh"

// Grid:  (num_rows,)   — one block per row
// Block: (min(row_size, 1024),)
//
// input:  [num_rows, row_size] fp16
// weight: [row_size] fp16
// output: [num_rows, row_size] fp16
extern "C" __global__ void rms_norm_f16(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    const int row_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int offset = row * row_size;

    float variance = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        float x = __half2float(input[offset + i]);
        variance += x * x;
    }

    variance = block_reduce_sum(variance);
    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(variance / (float)row_size + eps);
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        float val = __half2float(input[offset + i]);
        float w = __half2float(weight[i]);
        output[offset + i] = __float2half(val * inv_rms * w);
    }
}

extern "C" __global__ void rms_norm_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int row_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int offset = row * row_size;

    float variance = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        float x = input[offset + i];
        variance += x * x;
    }

    variance = block_reduce_sum(variance);
    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(variance / (float)row_size + eps);
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        output[offset + i] = input[offset + i] * inv_rms * weight[i];
    }
}

// vNext F32 master activations retain their precision through normalization.
// The declared norm weight remains F16, independently of the activation type.
template<typename Output>
__device__ void vnext_rms_norm_master(
    const float* input, const __half* weight, Output* output,
    const int row_size, const float eps
) {
    const size_t offset = static_cast<size_t>(blockIdx.x) * row_size;
    float variance = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float value = input[offset + i];
        variance += value * value;
    }
    variance = block_reduce_sum(variance);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(variance / static_cast<float>(row_size) + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float value = input[offset + i] * inverse_rms * __half2float(weight[i]);
        output[offset + i] = static_cast<Output>(value);
    }
}

extern "C" __global__ void vnext_rms_norm_f32_to_f16(
    const float* input, const __half* weight, __half* output,
    const int row_size, const float eps
) {
    vnext_rms_norm_master(input, weight, output, row_size, eps);
}

extern "C" __global__ void vnext_rms_norm_f32(
    const float* input, const __half* weight, float* output,
    const int row_size, const float eps
) {
    vnext_rms_norm_master(input, weight, output, row_size, eps);
}
