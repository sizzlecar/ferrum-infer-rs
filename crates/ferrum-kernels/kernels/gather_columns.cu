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

// Byte-exact program-binding transport. The host validates all packet and
// arena extents, arithmetic, and destination overlap before launch. These
// checks do not replace Core resource authority.
using ScatterU64 = unsigned long long;
static_assert(sizeof(ScatterU64) == 8, "scatter descriptor fields require u64");

__device__ __forceinline__ ScatterU64 scatter_read_u64_le(
    const unsigned char* bytes) {
    ScatterU64 value = 0;
#pragma unroll
    for (unsigned int byte = 0; byte < 8; ++byte) {
        value |= static_cast<ScatterU64>(bytes[byte]) << (byte * 8);
    }
    return value;
}

// ABI v1: five little-endian u64 fields per 40-byte descriptor:
// src_offset (absolute within packet), dst_offset, row_bytes, rows, dst_pitch.
// Grid: one CTA per descriptor. Block: (256, 1, 1).
// Source rows are tightly packed; destination row gaps remain untouched.
extern "C" __global__ void ferrum_program_binding_scatter_v1(
    const unsigned char* packet, unsigned char* arena, unsigned int count) {
    if (blockIdx.x >= count) {
        return;
    }
    const unsigned char* descriptor =
        packet + static_cast<ScatterU64>(blockIdx.x) * 40;
    const ScatterU64 src_offset = scatter_read_u64_le(descriptor);
    const ScatterU64 dst_offset = scatter_read_u64_le(descriptor + 8);
    const ScatterU64 row_bytes = scatter_read_u64_le(descriptor + 16);
    const ScatterU64 rows = scatter_read_u64_le(descriptor + 24);
    const ScatterU64 dst_pitch = scatter_read_u64_le(descriptor + 32);
    const ScatterU64 total_bytes = row_bytes * rows;
    for (ScatterU64 linear = threadIdx.x; linear < total_bytes;
         linear += blockDim.x) {
        const ScatterU64 row = linear / row_bytes;
        const ScatterU64 column = linear % row_bytes;
        arena[dst_offset + row * dst_pitch + column] =
            packet[src_offset + linear];
    }
}
