// Shared CUDA reduction primitives.
// Include this header instead of duplicating warp/block reduce in every kernel.

#pragma once

#include <cuda_fp16.h>

// ── Warp-level reductions (32 threads) ─────────────────────────────────

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

// ── Block-level reductions (up to 256 threads = 8 warps) ───────────────

__inline__ __device__ float block_reduce_sum(float val) {
    __shared__ float shared[8];
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

__inline__ __device__ float block_reduce_max(float val) {
    __shared__ float shared[8];
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;
    val = warp_reduce_max(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -1e20f;
    if (wid == 0) val = warp_reduce_max(val);
    return val;
}
