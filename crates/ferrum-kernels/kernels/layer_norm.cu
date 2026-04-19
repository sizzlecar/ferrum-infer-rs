// Full LayerNorm (mean + variance + affine):
//   out[r, c] = ((x[r, c] - mean) / sqrt(var + eps)) * gamma[c] + beta[c]
// where mean, var reduce over the last dim (cols).
//
// Used by Bert/Clip/Whisper encoders (LLM path uses rms_norm).
//
// Launch: grid = (rows, 1, 1), block = (warpSize=32, 1, 1).
// Warp-cooperative two-pass reduce — no shared memory needed.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

extern "C" __global__ void layer_norm_f16(
    const __half* __restrict__ x,        // [tokens, dim]
    const __half* __restrict__ gamma,    // [dim]
    const __half* __restrict__ beta,     // [dim]
    __half* __restrict__ out,            // [tokens, dim]
    const int dim,
    const float eps
) {
    const int r = blockIdx.x;
    const int lane = threadIdx.x;
    const __half* row = x + r * dim;
    __half* orow = out + r * dim;

    // Pass 1: sum and sum_sq for Welford-style reduce (simple two-pass is fine).
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        float v = __half2float(row[i]);
        sum += v;
        sum_sq += v * v;
    }
    sum = warp_sum(sum);
    sum_sq = warp_sum(sum_sq);

    const float mean = sum / (float)dim;
    const float var  = sum_sq / (float)dim - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    // Pass 2: affine.
    for (int i = lane; i < dim; i += 32) {
        float v = (__half2float(row[i]) - mean) * inv_std;
        v = v * __half2float(gamma[i]) + __half2float(beta[i]);
        orow[i] = __float2half(v);
    }
}
