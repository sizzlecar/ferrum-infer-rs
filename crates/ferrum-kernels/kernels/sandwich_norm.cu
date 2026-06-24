// Gemma-style sandwich-norm helpers for a device-side F32 residual shadow.
//
// The main CUDA activation path is FP16 because quantized GEMM kernels consume
// FP16 activations. Gemma3 sandwich norms can overflow FP16 on the residual
// stream, so the model path keeps the residual shadow in FP32 and materializes
// only normalized activations back to FP16 for the next projection.

#include "common.cuh"
#include <cuda_fp16.h>

extern "C" __global__ void activation_to_f32_shadow_f16(
    const __half* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __half2float(input[idx]);
    }
}

extern "C" __global__ void activation_add_to_f32_shadow_f16(
    const __half* __restrict__ input,
    float* __restrict__ residual,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        residual[idx] += __half2float(input[idx]);
    }
}

extern "C" __global__ void f32_to_activation_f16(
    const float* __restrict__ input,
    __half* __restrict__ output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(input[idx]);
    }
}

extern "C" __global__ void rms_norm_f16_to_f32(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    float* __restrict__ output,
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
        float x = __half2float(input[offset + i]);
        float w = __half2float(weight[i]);
        output[offset + i] = x * inv_rms * w;
    }
}

extern "C" __global__ void rms_norm_f16_add_to_f32(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    float* __restrict__ residual,
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
        float x = __half2float(input[offset + i]);
        float w = __half2float(weight[i]);
        residual[offset + i] += x * inv_rms * w;
    }
}

extern "C" __global__ void rms_norm_f32_to_f16(
    const float* __restrict__ input,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
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
        float w = __half2float(weight[i]);
        output[offset + i] = __float2half(input[offset + i] * inv_rms * w);
    }
}
