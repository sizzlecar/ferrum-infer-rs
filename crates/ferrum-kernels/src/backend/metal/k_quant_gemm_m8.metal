// Compiled after k_quant_gemm.metal; dequantization and parameter ABI are shared.
// Same half dequantization and four K8 MMA steps per K32 as production.
// Four SIMD groups own 16 output columns each, all for the same eight rows.
template <
    typename block_q,
    short dequant_tiles_per_block,
    void (*dequantize)(device const block_q *, short, thread half4x4 &)
>
static inline void gemm_f16a_quant_m8(
    device const half * input,
    device const block_q * weight,
    device half * output,
    constant KQuantGemmParams & p,
    threadgroup char * shmem,
    uint3 position,
    ushort tid,
    ushort sg
) {
    threadgroup half * weight_tile = (threadgroup half *)shmem;
    threadgroup half * input_tile = (threadgroup half *)(shmem + 4096);
    const uint output_start = position.y * 64;
    const uint input_start = position.x * 8;
    const short output_count = short(min(p.out_features - output_start, 64u));
    const short input_count = short(min(p.rows - input_start, 8u));
    const short weight_row = min(short(tid / 2), short(output_count - 1));
    const short dequant_tile0 = short(tid % 2);
    short dequant_tile = dequant_tile0;
    const uint blocks_per_row = p.in_features / (16 * dequant_tiles_per_block);
    device const block_q * x = weight
        + (output_start + weight_row) * blocks_per_row;
    const short input_row = min(short(tid / 4), short(input_count - 1));
    device const half * y = input
        + ulong(input_start + input_row) * p.in_features + 8 * (tid % 4);
    simdgroup_half8x8 weight_matrices[2];
    simdgroup_half8x8 input_matrix;
    simdgroup_float8x8 accumulators[2];
    FOR_UNROLL (short i = 0; i < 2; ++i) {
        accumulators[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    for (uint k = 0; k < p.in_features; k += 32) {
        half4x4 dequantized;
        dequantize(x, dequant_tile, dequantized);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // The complete 64-column half-weight tile is byte-for-byte the old layout.
        FOR_UNROLL (short i = 0; i < 16; ++i) {
            const short tile_x = 2 * dequant_tile0 + i / 8;
            const short tile_y = short(tid / 2) / 8;
            const short local_x = short(tid / 2) % 8;
            const short local_y = i % 8;
            weight_tile[64 * (8 * tile_x + tile_y) + 8 * local_y + local_x]
                = dequantized[i / 4][i % 4];
        }
        // Only one M8 fragment is needed. No thread skips either CTA barrier.
        if (tid < 32) {
            half2x4 values;
            FOR_UNROLL (short i = 0; i < 8; ++i) {
                values[i / 4][i % 4] = y[i];
            }
            *(threadgroup half2x4 *)(input_tile + 64 * (tid % 4) + 8 * (tid / 4))
                = values;
        }
        dequant_tile = dequant_tile + 2 < dequant_tiles_per_block
            ? dequant_tile + 2 : dequant_tile % 2;
        if (dequant_tile < 2) x += 1;
        y += 32;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half * lhs = weight_tile + 2 * 64 * sg;
        threadgroup const half * rhs = input_tile;
        FOR_UNROLL (short chunk = 0; chunk < 4; ++chunk) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; ++i) {
                simdgroup_load(weight_matrices[i], lhs + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(input_matrix, rhs, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; ++i) {
                simdgroup_multiply_accumulate(
                    accumulators[i], input_matrix, weight_matrices[i], accumulators[i]
                );
            }
            lhs += 8 * 64;
            rhs += 64;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * result = (threadgroup float *)shmem;
    FOR_UNROLL (short i = 0; i < 2; ++i) {
        simdgroup_store(accumulators[i], result + 16 * sg + 8 * i, 64, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = tid; index < 8 * 64; index += 128) {
        const uint row = index / 64;
        const uint column = index % 64;
        if (row < uint(input_count) && column < uint(output_count)) {
            output[ulong(input_start + row) * p.output_stride
                + p.output_column_offset + output_start + column] = half(result[index]);
        }
    }
}

#define M8_QUANT_KERNEL(name, block, tiles, dequant) \
kernel void name( \
    device const half * input [[buffer(0)]], \
    device const block * weight [[buffer(1)]], \
    device half * output [[buffer(2)]], \
    constant KQuantGemmParams & p [[buffer(3)]], \
    threadgroup char * shmem [[threadgroup(0)]], \
    uint3 position [[threadgroup_position_in_grid]], \
    ushort tid [[thread_index_in_threadgroup]], \
    ushort sg [[simdgroup_index_in_threadgroup]]) { \
    gemm_f16a_quant_m8<block, tiles, dequant>( \
        input, weight, output, p, shmem, position, tid, sg); \
}
M8_QUANT_KERNEL(gemm_f16a_q4kw_m8, block_q4_K, 16, dequantize_q4_K)
M8_QUANT_KERNEL(gemm_f16a_q5kw_m8, block_q5_K, 16, dequantize_q5_K)
M8_QUANT_KERNEL(gemm_f16a_q6kw_m8, block_q6_K, 16, dequantize_q6_K)
