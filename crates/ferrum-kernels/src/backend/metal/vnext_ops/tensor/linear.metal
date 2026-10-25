#include <metal_simdgroup_matrix>

struct LinearParams { uint rows; uint inputs; uint outputs; uint reserved; };

// Each SIMD group owns a 16 x 32 rectangle in a 32 x 64 output tile.
// BF16 values are decoded directly to FP32; no intermediate FP16 boundary.
// The 12 KiB threadgroup storage is on-chip, with no implicit device buffer.
kernel void ferrum_tensor_linear_bf16(
    device const ushort* input [[buffer(0)]],
    device const ushort* weight [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant LinearParams& p [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float workspace[3072];
    threadgroup float* a = workspace;
    threadgroup float* b = workspace + 1024;
    ulong row_start = ulong(group.x) * 32;
    ulong col_start = ulong(group.y) * 64;
    uint local_row = (sid / 2) * 16;
    uint local_col = (sid % 2) * 32;
    simdgroup_float8x8 sums[8];
    for (uint i = 0; i < 8; ++i) sums[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (ulong k = 0; k < ulong(p.inputs); k += 32) {
        for (uint i = tid; i < 1024; i += 128) {
            ulong row = row_start + i / 32;
            ulong col = k + i % 32;
            a[i] = row < p.rows && col < p.inputs ? decode_bf16(input[row * p.inputs + col]) : 0.0f;
        }
        for (uint i = tid; i < 2048; i += 128) {
            ulong row = col_start + i % 64;
            ulong col = k + i / 64;
            b[i] = row < p.outputs && col < p.inputs ? decode_bf16(weight[row * p.inputs + col]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint chunk = 0; chunk < 32; chunk += 8) {
            simdgroup_float8x8 av[2], bv[4];
            for (uint m = 0; m < 2; ++m)
                simdgroup_load(av[m], a + (local_row + m * 8) * 32 + chunk, 32, 0, false);
            for (uint n = 0; n < 4; ++n)
                simdgroup_load(bv[n], b + chunk * 64 + local_col + n * 8, 64, 0, false);
            for (uint m = 0; m < 2; ++m)
                for (uint n = 0; n < 4; ++n)
                    simdgroup_multiply_accumulate(sums[m * 4 + n], av[m], bv[n], sums[m * 4 + n]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Reuse the retired B tile for FP32 results before the final BF16 RNE.
    for (uint m = 0; m < 2; ++m)
        for (uint n = 0; n < 4; ++n)
            simdgroup_store(sums[m * 4 + n], b + (local_row + m * 8) * 64 + local_col + n * 8, 64, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 2048; i += 128) {
        ulong row = row_start + i / 64;
        ulong col = col_start + i % 64;
        if (row < p.rows && col < p.outputs) output[row * p.outputs + col] = encode_bf16(b[i]);
    }
}
