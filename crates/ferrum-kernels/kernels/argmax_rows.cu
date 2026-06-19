// Argmax per row for FP16 logits.
//
// One block per row of a [m, n] matrix. Each thread strides through the
// row computing a local max + index; warp-level shfl_xor + shared-memory
// final reduction picks the global argmax. Writes one i32 index per row.
//
// Used by the greedy-decode fast path in Qwen3MoeModel /
// LlamaFamilyModel: replaces the per-iter D2H of `m * vocab * 2` bytes
// (e.g. 19.5 MB at c=32, vocab=152064) + CPU argmax (~5 ms at c=32)
// with one kernel + tiny D2H of `m * 4` bytes (128 bytes at c=32).
//
// Launch: grid=(m, 1, 1), block=(256, 1, 1). 32 KB shmem is plenty.

#include <cuda_fp16.h>

extern "C" __global__ void apply_repetition_penalties_sparse_f16(
    __half* __restrict__ logits,                 // [m, n] row-major
    int n,                                       // vocab size
    const unsigned int* __restrict__ row_offsets, // [m + 1]
    const unsigned int* __restrict__ token_ids,   // [total_token_ids]
    const float* __restrict__ repetition_penalties, // [m]
    int total_token_ids
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    float penalty = repetition_penalties[row];
    if (penalty == 1.0f) {
        return;
    }

    unsigned int start = row_offsets[row];
    unsigned int end = row_offsets[row + 1];
    if (start > (unsigned int)total_token_ids) {
        start = (unsigned int)total_token_ids;
    }
    if (end > (unsigned int)total_token_ids) {
        end = (unsigned int)total_token_ids;
    }

    __half* row_ptr = logits + (size_t)row * (size_t)n;
    for (unsigned int entry = start + tid; entry < end; entry += block_size) {
        unsigned int token = token_ids[entry];
        if (token >= (unsigned int)n) {
            continue;
        }
        float v = __half2float(row_ptr[token]);
        if (!isfinite(v)) {
            continue;
        }
        v = (v > 0.0f) ? (v / penalty) : (v * penalty);
        row_ptr[token] = __float2half_rn(v);
    }
}

extern "C" __global__ void argmax_rows_f16(
    const __half* __restrict__ logits,  // [m, n] row-major
    int n,                              // vocab size
    int* __restrict__ out_idx           // [m]
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const __half* row_ptr = logits + (size_t)row * (size_t)n;

    // Step 1: each thread scans its strided portion, tracking
    // (max_val, max_idx). Tie-break by lower index so result is
    // deterministic across thread layouts.
    float local_max = -INFINITY;
    int local_idx = 0;
    for (int i = tid; i < n; i += block_size) {
        float v = __half2float(row_ptr[i]);
        if (v > local_max || (v == local_max && i < local_idx)) {
            local_max = v;
            local_idx = i;
        }
    }

    // Step 2: warp-level reduction (32 lanes per warp). Each iteration
    // halves the active span. `__shfl_xor_sync` lets every lane in the
    // warp see the same merged result.
    const unsigned mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_xor_sync(mask, local_max, offset);
        int   other_idx = __shfl_xor_sync(mask, local_idx, offset);
        if (other_max > local_max || (other_max == local_max && other_idx < local_idx)) {
            local_max = other_max;
            local_idx = other_idx;
        }
    }

    // Step 3: warp leaders → shared memory. block_size=256 → 8 warps.
    __shared__ float s_max[32];
    __shared__ int   s_idx[32];
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0) {
        s_max[warp_id] = local_max;
        s_idx[warp_id] = local_idx;
    }
    __syncthreads();

    // Step 4: warp 0 reduces the per-warp leaders.
    if (warp_id == 0) {
        int num_warps = (block_size + 31) / 32;
        local_max = (lane < num_warps) ? s_max[lane] : -INFINITY;
        local_idx = (lane < num_warps) ? s_idx[lane] : 0;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_max = __shfl_xor_sync(mask, local_max, offset);
            int   other_idx = __shfl_xor_sync(mask, local_idx, offset);
            if (other_max > local_max || (other_max == local_max && other_idx < local_idx)) {
                local_max = other_max;
                local_idx = other_idx;
            }
        }
        if (lane == 0) {
            out_idx[row] = local_idx;
        }
    }
}

extern "C" __global__ void argmax_rows_f16_masked(
    const __half* __restrict__ logits,      // [m, n] row-major
    int n,                                  // vocab size
    const signed char* __restrict__ valid,  // [mask_len], nonzero = selectable
    int mask_len,
    int* __restrict__ out_idx               // [m]
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const __half* row_ptr = logits + (size_t)row * (size_t)n;

    float local_max = -INFINITY;
    int local_idx = -1;
    for (int i = tid; i < n; i += block_size) {
        if (i >= mask_len || valid[i] == 0) {
            continue;
        }
        float v = __half2float(row_ptr[i]);
        if (!isfinite(v)) {
            continue;
        }
        if (local_idx < 0 || v > local_max || (v == local_max && i < local_idx)) {
            local_max = v;
            local_idx = i;
        }
    }

    const unsigned mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_xor_sync(mask, local_max, offset);
        int   other_idx = __shfl_xor_sync(mask, local_idx, offset);
        if (other_idx >= 0 && (local_idx < 0 || other_max > local_max ||
                               (other_max == local_max && other_idx < local_idx))) {
            local_max = other_max;
            local_idx = other_idx;
        }
    }

    __shared__ float s_max[32];
    __shared__ int   s_idx[32];
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0) {
        s_max[warp_id] = local_max;
        s_idx[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (block_size + 31) / 32;
        local_max = (lane < num_warps) ? s_max[lane] : -INFINITY;
        local_idx = (lane < num_warps) ? s_idx[lane] : -1;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_max = __shfl_xor_sync(mask, local_max, offset);
            int   other_idx = __shfl_xor_sync(mask, local_idx, offset);
            if (other_idx >= 0 && (local_idx < 0 || other_max > local_max ||
                                   (other_max == local_max && other_idx < local_idx))) {
                local_max = other_max;
                local_idx = other_idx;
            }
        }
        if (lane == 0) {
            out_idx[row] = local_idx;
        }
    }
}
