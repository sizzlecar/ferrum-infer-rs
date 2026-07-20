#include <metal_stdlib>
using namespace metal;

constant uint THREADS_PER_GROUP = 256;
constant uint QK_K = 256;

struct EmbeddingParams {
    uint token_count;
    uint hidden_size;
    uint vocabulary_size;
};

struct RmsNormParams {
    uint rows;
    uint hidden_size;
    float epsilon;
};

struct ResidualAddParams {
    uint elements;
};

struct block_q6_K {
    uchar ql[QK_K / 2];
    uchar qh[QK_K / 4];
    char scales[QK_K / 16];
    half d;
};

static inline float q6_k_value(device const block_q6_K & block, uint index) {
    const uint half_block = index / 128;
    const uint position = index % 128;
    const uint lane = position % 32;
    const uint ql_base = half_block * 64;
    const uint qh_base = half_block * 32;
    const uint scale_base = half_block * 8;
    const uchar high = block.qh[qh_base + lane];

    uint low;
    uint upper;
    uint scale;
    if (position < 32) {
        low = block.ql[ql_base + lane] & 0x0f;
        upper = (high >> 0) & 0x03;
        scale = scale_base + lane / 16;
    } else if (position < 64) {
        low = block.ql[ql_base + lane + 32] & 0x0f;
        upper = (high >> 2) & 0x03;
        scale = scale_base + lane / 16 + 2;
    } else if (position < 96) {
        low = block.ql[ql_base + lane] >> 4;
        upper = (high >> 4) & 0x03;
        scale = scale_base + lane / 16 + 4;
    } else {
        low = block.ql[ql_base + lane + 32] >> 4;
        upper = (high >> 6) & 0x03;
        scale = scale_base + lane / 16 + 6;
    }
    const int quantized = int(low | (upper << 4)) - 32;
    return float(block.d) * float(block.scales[scale]) * float(quantized);
}

kernel void vnext_embedding_dense_f16(
    device const half * table [[buffer(0)]],
    device const uint * token_ids [[buffer(1)]],
    device half * output [[buffer(2)]],
    constant EmbeddingParams & params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint token = group.y;
    const uint column = group.x * THREADS_PER_GROUP + lane;
    if (token >= params.token_count || column >= params.hidden_size) {
        return;
    }
    const uint token_id = token_ids[token];
    const ulong output_index = ulong(token) * params.hidden_size + column;
    output[output_index] = token_id < params.vocabulary_size
        ? table[ulong(token_id) * params.hidden_size + column]
        : half(0.0h);
}

kernel void vnext_embedding_q6_k_f16(
    device const block_q6_K * table [[buffer(0)]],
    device const uint * token_ids [[buffer(1)]],
    device half * output [[buffer(2)]],
    constant EmbeddingParams & params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint token = group.y;
    const uint column = group.x * THREADS_PER_GROUP + lane;
    if (token >= params.token_count || column >= params.hidden_size) {
        return;
    }
    const uint token_id = token_ids[token];
    const ulong output_index = ulong(token) * params.hidden_size + column;
    if (token_id >= params.vocabulary_size) {
        output[output_index] = half(0.0h);
        return;
    }
    const uint blocks_per_row = params.hidden_size / QK_K;
    const ulong block_index = ulong(token_id) * blocks_per_row + column / QK_K;
    output[output_index] = half(q6_k_value(table[block_index], column % QK_K));
}

kernel void vnext_rms_norm_f16(
    device const half * input [[buffer(0)]],
    device const half * weight [[buffer(1)]],
    device half * output [[buffer(2)]],
    constant RmsNormParams & params [[buffer(3)]],
    threadgroup float * partial [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row = group.x;
    if (row >= params.rows) {
        return;
    }
    const ulong base = ulong(row) * params.hidden_size;
    float sum = 0.0f;
    for (uint column = lane; column < params.hidden_size; column += THREADS_PER_GROUP) {
        const float value = float(input[base + column]);
        sum += value * value;
    }
    sum = simd_sum(sum);
    if (simd_lane == 0) {
        partial[simd_group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        const uint simdgroups = THREADS_PER_GROUP / 32;
        float total = simd_lane < simdgroups ? partial[simd_lane] : 0.0f;
        total = simd_sum(total);
        if (simd_lane == 0) {
            partial[0] = rsqrt(total / float(params.hidden_size) + params.epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inverse_rms = partial[0];
    for (uint column = lane; column < params.hidden_size; column += THREADS_PER_GROUP) {
        output[base + column] = half(
            float(input[base + column]) * inverse_rms * float(weight[column])
        );
    }
}

kernel void vnext_residual_add_f16(
    device const half * left [[buffer(0)]],
    device const half * right [[buffer(1)]],
    device half * output [[buffer(2)]],
    constant ResidualAddParams & params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < params.elements) {
        output[index] = half(float(left[index]) + float(right[index]));
    }
}
