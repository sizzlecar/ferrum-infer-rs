// Softmax over last dimension: out[i] = exp(x[i] - max) / sum(exp(x[j] - max))
// Used by TTS flash attention and speaker encoder.
// Grid: one block per row. Block: 256 threads.

#include "common.cuh"
#include <cfloat>

extern "C" __global__ void softmax_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x = input + row * cols;
    float* o = output + row * cols;

    // Phase 1: find max
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }

    // Warp reduction
    local_max = warp_reduce_max(local_max);

    // Block reduction via shared memory
    __shared__ float smem[32];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) smem[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < (blockDim.x + 31) / 32) ? smem[lane] : -FLT_MAX;
        v = warp_reduce_max(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    float row_max = smem[0];

    // Phase 2: exp and sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(x[i] - row_max);
        o[i] = e;  // store exp temporarily
        local_sum += e;
    }

    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) smem[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    float inv_sum = 1.0f / smem[0];

    // Phase 3: normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        o[i] *= inv_sum;
    }
}
