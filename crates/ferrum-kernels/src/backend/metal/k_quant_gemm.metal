// Shared tiled prefill GEMM for GGUF K-quant weights and F16 activations.
// The 64x32 output tile and simdgroup matrix layout follow llama.cpp's
// kernel_mul_mm family; the ABI is Ferrum's backend-neutral LinearParams.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define QK_K 256
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

struct block_q4_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[QK_K / 2];
};

struct block_q5_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qh[QK_K / 8];
    uchar qs[QK_K / 2];
};

struct block_q6_K {
    uchar ql[QK_K / 2];
    uchar qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    half d;
};

struct KQuantGemmParams {
    uint rows;
    uint in_features;
    uint out_features;
    uint output_stride;
    uint output_column_offset;
};

static inline uchar2 get_scale_min_k4(int j, int k, device const uchar * q) {
    if (j < 4) {
        return uchar2(q[j + k] & 63, q[j + 4 + k] & 63);
    }
    return uchar2(
        (q[j + 4 + k] & 0x0f) | ((q[j - 4 + k] & 0xc0) >> 2),
        (q[j + 4 + k] >> 4) | ((q[j + k] & 0xc0) >> 2)
    );
}

template <typename type4x4>
static inline void dequantize_q4_K(
    device const block_q4_K * xb,
    short il,
    thread type4x4 & reg
) {
    device const uchar * q = xb->qs + 32 * (il / 4) + 16 * (il & 1);
    const short is = (il / 4) * 2;
    il &= 3;
    const uchar2 sc = get_scale_min_k4(is, il / 2, xb->scales);
    const float d = il < 2 ? float(xb->d) : float(xb->d) / 16.f;
    const float dl = d * float(sc[0]);
    const float ml = float(xb->dmin) * float(sc[1]);
    const ushort mask = il < 2 ? 0x0f : 0xf0;
    FOR_UNROLL (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = dl * float(q[i] & mask) - ml;
    }
}

template <typename type4x4>
static inline void dequantize_q5_K(
    device const block_q5_K * xb,
    short il,
    thread type4x4 & reg
) {
    device const uchar * q = xb->qs + 32 * (il / 4) + 16 * (il & 1);
    device const uchar * qh = xb->qh + 16 * (il & 1);
    const short is = (il / 4) * 2;
    const uchar high_mask = uchar(1u << (il / 2));
    il &= 3;
    const uchar2 sc = get_scale_min_k4(is, il / 2, xb->scales);
    const float d = il < 2 ? float(xb->d) : float(xb->d) / 16.f;
    const float dl = d * float(sc[0]);
    const float ml = float(xb->dmin) * float(sc[1]);
    const ushort low_mask = il < 2 ? 0x0f : 0xf0;
    const float high_value = il < 2 ? 16.f : 256.f;
    FOR_UNROLL (int i = 0; i < 16; ++i) {
        const float value = float(q[i] & low_mask)
            + ((qh[i] & high_mask) != 0 ? high_value : 0.f);
        reg[i / 4][i % 4] = dl * value - ml;
    }
}

template <typename type4x4>
static inline void dequantize_q6_K(
    device const block_q6_K * xb,
    short il,
    thread type4x4 & reg
) {
    device const uchar * ql = xb->ql + 64 * (il / 8)
        + 32 * ((il / 2) & 1) + 16 * (il & 1);
    device const uchar * qh = xb->qh + 32 * (il / 8) + 16 * (il & 1);
    const float scale = float(xb->scales[(il % 2) + 2 * (il / 2)]);
    il = (il / 2) & 3;
    const ushort high_mask = il > 1 ? (il > 2 ? 192 : 48) : (il > 0 ? 12 : 3);
    const ushort low_mask = il > 1 ? 0xf0 : 0x0f;
    const float coefficient = il > 1 ? 1.f / 16.f : 1.f;
    const float dl = float(xb->d) * scale * coefficient;
    const float ml = float(xb->d) * scale * 32.f;
    FOR_UNROLL (int i = 0; i < 16; ++i) {
        const ushort value = (il & 1) != 0
            ? ushort((ql[i] & low_mask) | ((qh[i] & high_mask) << 2))
            : ushort((ql[i] & low_mask) | ((qh[i] & high_mask) << 4));
        reg[i / 4][i % 4] = dl * float(value) - ml;
    }
}

constant short TILE_OUTPUT_ROWS = 64;
constant short TILE_INPUT_ROWS = 32;
constant short TILE_K = 32;
constant short WEIGHT_LOADERS_PER_ROW = 2;
constant short INPUT_LOADERS_PER_ROW = 4;
constant short DEQUANT_TILES_PER_BLOCK = 16;

template <typename block_q, void (*dequantize)(device const block_q *, short, thread half4x4 &)>
static inline void gemm_f16a_kquant_tiled(
    device const half * input,
    device const block_q * weight,
    device half * output,
    constant KQuantGemmParams & p,
    threadgroup char * shmem,
    uint3 threadgroup_position,
    ushort thread_index,
    ushort simdgroup_index
) {
    threadgroup half * weight_tile = (threadgroup half *)shmem;
    threadgroup half * input_tile = (threadgroup half *)(shmem + 4096);

    const int output_start = int(threadgroup_position.y) * TILE_OUTPUT_ROWS;
    const int input_start = int(threadgroup_position.x) * TILE_INPUT_ROWS;
    const short output_count = min(short(p.out_features - uint(output_start)), TILE_OUTPUT_ROWS);
    const short input_count = min(short(p.rows - uint(input_start)), TILE_INPUT_ROWS);
    const short weight_row = min(
        short(thread_index / WEIGHT_LOADERS_PER_ROW), short(output_count - 1)
    );
    const short input_row = min(
        short(thread_index / INPUT_LOADERS_PER_ROW), short(input_count - 1)
    );
    const short dequant_tile0 = short(thread_index) % WEIGHT_LOADERS_PER_ROW;
    short dequant_tile = dequant_tile0;

    const int blocks_per_row = int(p.in_features / QK_K);
    device const block_q * x = weight
        + (output_start + weight_row) * blocks_per_row
        + dequant_tile0 / DEQUANT_TILES_PER_BLOCK;
    const short input_column = 8 * (short(thread_index) % INPUT_LOADERS_PER_ROW);
    device const half * y = input
        + ulong(input_start + input_row) * p.in_features + input_column;

    simdgroup_half8x8 weight_matrices[4];
    simdgroup_half8x8 input_matrices[2];
    simdgroup_float8x8 accumulators[8];
    FOR_UNROLL (short i = 0; i < 8; ++i) {
        accumulators[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (uint k = 0; k < p.in_features; k += TILE_K) {
        half4x4 dequantized;
        dequantize(x, dequant_tile, dequantized);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; ++i) {
            const short tile_x = 2 * dequant_tile0 + i / 8;
            const short tile_y = (short(thread_index) / WEIGHT_LOADERS_PER_ROW) / 8;
            const short local_x = (short(thread_index) / WEIGHT_LOADERS_PER_ROW) % 8;
            const short local_y = i % 8;
            const short block = 8 * tile_x + tile_y;
            weight_tile[64 * block + 8 * local_y + local_x] = dequantized[i / 4][i % 4];
        }

        const short input_tile_x = short(thread_index) % INPUT_LOADERS_PER_ROW;
        const short input_tile_y = (short(thread_index) / INPUT_LOADERS_PER_ROW) / 8;
        const short input_local_y = (short(thread_index) / INPUT_LOADERS_PER_ROW) % 8;
        const short input_block = 4 * input_tile_x + input_tile_y;
        half2x4 input_values;
        FOR_UNROLL (short i = 0; i < 8; ++i) {
            input_values[i / 4][i % 4] = y[i];
        }
        *(threadgroup half2x4 *)(input_tile + 64 * input_block + 8 * input_local_y) =
            input_values;

        dequant_tile = dequant_tile + 2 < DEQUANT_TILES_PER_BLOCK
            ? dequant_tile + 2
            : dequant_tile % 2;
        if (dequant_tile < 2) {
            x += 1;
        }
        y += TILE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half * lhs = weight_tile + 4 * 64 * (simdgroup_index % 2);
        threadgroup const half * rhs = input_tile + 2 * 64 * (simdgroup_index / 2);
        FOR_UNROLL (short chunk = 0; chunk < TILE_K / 8; ++chunk) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; ++i) {
                simdgroup_load(weight_matrices[i], lhs + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; ++i) {
                simdgroup_load(input_matrices[i], rhs + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(
                    accumulators[i], input_matrices[i / 4], weight_matrices[i % 4], accumulators[i]
                );
            }
            lhs += 8 * 64;
            rhs += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * result_tile = ((threadgroup float *)shmem)
        + 32 * (simdgroup_index & 1)
        + 16 * (simdgroup_index >> 1) * TILE_OUTPUT_ROWS;
    FOR_UNROLL (short i = 0; i < 8; ++i) {
        simdgroup_store(
            accumulators[i],
            result_tile + 8 * (i % 4) + 8 * TILE_OUTPUT_ROWS * (i / 4),
            TILE_OUTPUT_ROWS,
            0,
            false
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index == 0) {
        for (int row = thread_index; row < input_count; row += TILE_INPUT_ROWS) {
            device half * destination = output
                + ulong(input_start + row) * p.output_stride
                + p.output_column_offset + output_start;
            threadgroup float * source = ((threadgroup float *)shmem)
                + row * TILE_OUTPUT_ROWS;
            for (int column = 0; column < output_count; ++column) {
                destination[column] = half(source[column]);
            }
        }
    }
}

kernel void gemm_f16a_q4kw_tiled(
    device const half * input [[buffer(0)]],
    device const block_q4_K * weight [[buffer(1)]],
    device half * output [[buffer(2)]],
    constant KQuantGemmParams & p [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 position [[threadgroup_position_in_grid]],
    ushort thread_index [[thread_index_in_threadgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]) {
    gemm_f16a_kquant_tiled<block_q4_K, dequantize_q4_K>(
        input, weight, output, p, shmem, position, thread_index, simdgroup_index
    );
}

kernel void gemm_f16a_q5kw_tiled(
    device const half * input [[buffer(0)]],
    device const block_q5_K * weight [[buffer(1)]],
    device half * output [[buffer(2)]],
    constant KQuantGemmParams & p [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 position [[threadgroup_position_in_grid]],
    ushort thread_index [[thread_index_in_threadgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]) {
    gemm_f16a_kquant_tiled<block_q5_K, dequantize_q5_K>(
        input, weight, output, p, shmem, position, thread_index, simdgroup_index
    );
}

kernel void gemm_f16a_q6kw_tiled(
    device const half * input [[buffer(0)]],
    device const block_q6_K * weight [[buffer(1)]],
    device half * output [[buffer(2)]],
    constant KQuantGemmParams & p [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 position [[threadgroup_position_in_grid]],
    ushort thread_index [[thread_index_in_threadgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]) {
    gemm_f16a_kquant_tiled<block_q6_K, dequantize_q6_K>(
        input, weight, output, p, shmem, position, thread_index, simdgroup_index
    );
}
