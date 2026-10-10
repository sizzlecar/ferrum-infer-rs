// Test-only transport primitive. The host validates all packet and arena
// extents, arithmetic, and destination overlap before launching this kernel.
// These checks do not replace Core resource authority.

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

// Each 40-byte descriptor contains five little-endian u64 fields:
// src_offset (absolute within packet), dst_offset, row_bytes, rows, dst_pitch.
// Source rows are tightly packed; destination row gaps remain untouched.
extern "C" __global__ void ferrum_test_scatter(
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
