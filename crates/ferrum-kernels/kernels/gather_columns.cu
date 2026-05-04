// Column gather for f16 row-major matrices.
// output[m, j] = input[m, perm[j]]
//
// Used by perm-aware GPTQ desc_act path: before INT4 GEMM, gather
// activation columns to match the qweight rows that were permuted by
// `argsort(g_idx)` at load time. After gather + standard Marlin GEMM,
// the result equals the original (un-permuted) GEMM output.
//
// Grid: (M, ceil(K/512), 1). Block: (512, 1, 1).
// Each block handles up to 512 columns of one row.

#include <cuda_fp16.h>
#include <cstdint>

extern "C" __global__ void gather_columns_f16(
    const __half* __restrict__ input,   // [M, K]
    const int32_t* __restrict__ perm,   // [K]
    __half* __restrict__ output,        // [M, K]
    int M,
    int K
) {
    int m = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (m >= M || j >= K) return;

    int src_col = perm[j];
    output[m * K + j] = input[m * K + src_col];
}
