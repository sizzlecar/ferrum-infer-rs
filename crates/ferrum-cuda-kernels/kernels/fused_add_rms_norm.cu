// Fused residual-add + RMS normalization kernel.
//
// Replaces 3 separate kernel launches (add, variance, normalize) with 1.
// Reads input+residual once, writes normalized output + updated residual once.
//
// Reference: vLLM csrc/layernorm_kernels.cu (Apache 2.0)

#include <cuda_fp16.h>

// Warp-level reduction for sum
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Block-level reduction for sum using shared memory
__inline__ __device__ float block_reduce_sum(float val) {
    __shared__ float shared[32]; // Max 32 warps per block
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

// Fused: residual_out = input + residual; output = rms_norm(residual_out, weight, eps)
// All in FP16, internal computation in FP32.
//
// Grid:  (num_tokens,)   — one block per token
// Block: (min(hidden_size, 1024),)
//
// input:        [num_tokens, hidden_size] fp16  (read-only)
// residual:     [num_tokens, hidden_size] fp16  (read-only)
// weight:       [hidden_size] fp16              (read-only)
// output:       [num_tokens, hidden_size] fp16  (normalized result)
// residual_out: [num_tokens, hidden_size] fp16  (input + residual)
extern "C" __global__ void fused_add_rms_norm_f16(
    const __half* __restrict__ input,
    const __half* __restrict__ residual,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    __half* __restrict__ residual_out,
    const int hidden_size,
    const float eps
) {
    const int token_idx = blockIdx.x;
    const int offset = token_idx * hidden_size;

    // Step 1: Add residual and compute variance in one pass
    float variance = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float x = __half2float(input[offset + i]);
        float r = __half2float(residual[offset + i]);
        float sum = x + r;
        residual_out[offset + i] = __float2half(sum);
        variance += sum * sum;
    }

    // Reduce variance across block
    variance = block_reduce_sum(variance);
    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(variance / (float)hidden_size + eps);
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    // Step 2: Normalize and write output
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __half2float(residual_out[offset + i]);
        float w = __half2float(weight[i]);
        output[offset + i] = __float2half(val * inv_rms * w);
    }
}

// FP32 version
extern "C" __global__ void fused_add_rms_norm_f32(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    float* __restrict__ output,
    float* __restrict__ residual_out,
    const int hidden_size,
    const float eps
) {
    const int token_idx = blockIdx.x;
    const int offset = token_idx * hidden_size;

    float variance = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float sum = input[offset + i] + residual[offset + i];
        residual_out[offset + i] = sum;
        variance += sum * sum;
    }

    variance = block_reduce_sum(variance);
    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(variance / (float)hidden_size + eps);
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        output[offset + i] = residual_out[offset + i] * inv_rms * weight[i];
    }
}
