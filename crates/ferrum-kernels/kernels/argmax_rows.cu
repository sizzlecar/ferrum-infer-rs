// Argmax over rows of f16 matrix [n_rows, vocab].
// Output: i32[n_rows], each value is the column index of the max in that row.
//
// Used to skip the heavy device→host transfer of full logits when the
// engine knows sampling is greedy (T=0). Saves ~1ms / iter at c=16
// vocab=128k (8 MB of D→H per iter → 64 bytes).
//
// One block per row, 1024 threads. Each thread reduces vocab/1024 ≈ 125
// elements; then shared-mem reduction picks the row's argmax.

#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>

#define BLOCK 1024

extern "C" __global__ void argmax_rows_f16(
    const __half* __restrict__ input,   // [n_rows, vocab]
    int32_t* __restrict__ output,       // [n_rows]
    int n_rows,
    int vocab
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    const __half* row_ptr = input + (size_t)row * vocab;

    // Per-thread max scan.
    float local_max = -FLT_MAX;
    int   local_idx = 0;
    for (int i = threadIdx.x; i < vocab; i += BLOCK) {
        float v = __half2float(row_ptr[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    // Block reduction in shared mem.
    __shared__ float s_val[BLOCK];
    __shared__ int   s_idx[BLOCK];
    s_val[threadIdx.x] = local_max;
    s_idx[threadIdx.x] = local_idx;
    __syncthreads();

    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float other_v = s_val[threadIdx.x + stride];
            if (other_v > s_val[threadIdx.x]) {
                s_val[threadIdx.x] = other_v;
                s_idx[threadIdx.x] = s_idx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[row] = s_idx[0];
    }
}
