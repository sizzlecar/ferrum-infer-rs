#include <metal_stdlib>
using namespace metal;

constant uint THREADS_PER_GROUP = 256;
constant uint QK_K = 256;
constant uint QK8_0 = 32;

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

struct LastTokenMaskedArgmaxParams {
    uint vocabulary_size;
    uint repetition_capacity;
};

struct block_q4_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[QK_K / 2];
};

struct block_q6_K {
    uchar ql[QK_K / 2];
    uchar qh[QK_K / 4];
    char scales[QK_K / 16];
    half d;
};

struct block_q8_0 {
    half d;
    char qs[QK8_0];
};

static inline float q4_k_value(device const block_q4_K & block, uint index) {
    const uint subblock = index / 32;
    uchar scale;
    uchar minimum;
    if (subblock < 4) {
        scale = block.scales[subblock] & 63;
        minimum = block.scales[subblock + 4] & 63;
    } else {
        scale = (block.scales[subblock + 4] & 0x0f)
            | ((block.scales[subblock - 4] >> 6) << 4);
        minimum = (block.scales[subblock + 4] >> 4)
            | ((block.scales[subblock] >> 6) << 4);
    }
    const uint packed_index = (subblock / 2) * 32 + index % 32;
    const uchar packed = block.qs[packed_index];
    const uint quantized = (subblock & 1) == 0 ? packed & 0x0f : packed >> 4;
    return float(block.d) * float(scale) * float(quantized)
        - float(block.dmin) * float(minimum);
}

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

kernel void vnext_embedding_q4_k_f16(
    device const block_q4_K * table [[buffer(0)]],
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
    output[output_index] = half(q4_k_value(table[block_index], column % QK_K));
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

kernel void vnext_embedding_q8_0_f16(
    device const block_q8_0 * table [[buffer(0)]],
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
    const uint blocks_per_row = params.hidden_size / QK8_0;
    const ulong block_index = ulong(token_id) * blocks_per_row + column / QK8_0;
    const block_q8_0 block = table[block_index];
    output[output_index] = half(float(block.d) * float(block.qs[column % QK8_0]));
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

kernel void vnext_last_token_masked_argmax_f16(
    device const half * logits [[buffer(0)]],
    device half * scratch [[buffer(1)]],
    device const uchar * valid_mask [[buffer(2)]],
    device const uint * repetition_token_ids [[buffer(3)]],
    device const uint * repetition_offsets [[buffer(4)]],
    device const float * repetition_penalty [[buffer(5)]],
    device uint * output [[buffer(6)]],
    constant LastTokenMaskedArgmaxParams & params [[buffer(7)]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const float penalty = repetition_penalty[0];
    const uint repetition_start = min(repetition_offsets[0], params.repetition_capacity);
    const uint repetition_end = min(repetition_offsets[1], params.repetition_capacity);
    const bool apply_penalty = penalty != 1.0f && repetition_start < repetition_end;
    if (apply_penalty) {
        for (uint token = lane; token < params.vocabulary_size; token += THREADS_PER_GROUP) {
            scratch[token] = logits[token];
        }
        threadgroup_barrier(mem_flags::mem_device);
        for (uint entry = repetition_start + lane;
             entry < repetition_end;
             entry += THREADS_PER_GROUP) {
            const uint token = repetition_token_ids[entry];
            if (token >= params.vocabulary_size) {
                continue;
            }
            const float value = float(scratch[token]);
            if (!isfinite(value)) {
                continue;
            }
            scratch[token] = half(value > 0.0f ? value / penalty : value * penalty);
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    float local_maximum = -INFINITY;
    int local_index = -1;
    for (uint token = lane; token < params.vocabulary_size; token += THREADS_PER_GROUP) {
        if (valid_mask[token] == 0) {
            continue;
        }
        const float value = float(apply_penalty ? scratch[token] : logits[token]);
        if (!isfinite(value)) {
            continue;
        }
        if (local_index < 0 || value > local_maximum ||
            (value == local_maximum && int(token) < local_index)) {
            local_maximum = value;
            local_index = int(token);
        }
    }

    const float simd_maximum = simd_max(local_maximum);
    const int simd_index = simd_min(
        local_index >= 0 && local_maximum == simd_maximum ? local_index : 0x7fffffff
    );
    threadgroup float partial_maximum[THREADS_PER_GROUP / 32];
    threadgroup int partial_index[THREADS_PER_GROUP / 32];
    if (simd_lane == 0) {
        partial_maximum[simd_group] = simd_maximum;
        partial_index[simd_group] = simd_index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float maximum = -INFINITY;
        int index = 0x7fffffff;
        for (uint group = 0; group < THREADS_PER_GROUP / 32; ++group) {
            const float candidate_maximum = partial_maximum[group];
            const int candidate_index = partial_index[group];
            if (candidate_index != 0x7fffffff &&
                (index == 0x7fffffff || candidate_maximum > maximum ||
                 (candidate_maximum == maximum && candidate_index < index))) {
                maximum = candidate_maximum;
                index = candidate_index;
            }
        }
        output[0] = index == 0x7fffffff ? 0xffffffffu : uint(index);
    }
}

kernel void vnext_embedding_dense_f32(
    device const half * table [[buffer(0)]],
    device const uint * token_ids [[buffer(1)]],
    device float * output [[buffer(2)]],
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
        ? float(table[ulong(token_id) * params.hidden_size + column])
        : 0.0f;
}

kernel void vnext_embedding_q4_k_f32(
    device const block_q4_K * table [[buffer(0)]],
    device const uint * token_ids [[buffer(1)]],
    device float * output [[buffer(2)]],
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
        output[output_index] = 0.0f;
        return;
    }
    const uint blocks_per_row = params.hidden_size / QK_K;
    const ulong block_index = ulong(token_id) * blocks_per_row + column / QK_K;
    output[output_index] = q4_k_value(table[block_index], column % QK_K);
}

kernel void vnext_embedding_q6_k_f32(
    device const block_q6_K * table [[buffer(0)]],
    device const uint * token_ids [[buffer(1)]],
    device float * output [[buffer(2)]],
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
        output[output_index] = 0.0f;
        return;
    }
    const uint blocks_per_row = params.hidden_size / QK_K;
    const ulong block_index = ulong(token_id) * blocks_per_row + column / QK_K;
    output[output_index] = q6_k_value(table[block_index], column % QK_K);
}

kernel void vnext_embedding_q8_0_f32(
    device const block_q8_0 * table [[buffer(0)]],
    device const uint * token_ids [[buffer(1)]],
    device float * output [[buffer(2)]],
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
        output[output_index] = 0.0f;
        return;
    }
    const uint blocks_per_row = params.hidden_size / QK8_0;
    const ulong block_index = ulong(token_id) * blocks_per_row + column / QK8_0;
    const block_q8_0 block = table[block_index];
    output[output_index] = float(block.d) * float(block.qs[column % QK8_0]);
}

kernel void vnext_rms_norm_f32_to_f16(
    device const float * input [[buffer(0)]],
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
        const float value = input[base + column];
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
        output[base + column] = half(input[base + column] * inverse_rms * float(weight[column]));
    }
}

kernel void vnext_rms_norm_f32(
    device const float * input [[buffer(0)]],
    device const half * weight [[buffer(1)]],
    device float * output [[buffer(2)]],
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
        const float value = input[base + column];
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
        output[base + column] = input[base + column] * inverse_rms * float(weight[column]);
    }
}

kernel void vnext_residual_add_f32_f16(
    device const float * left [[buffer(0)]],
    device const half * right [[buffer(1)]],
    device float * output [[buffer(2)]],
    constant ResidualAddParams & params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < params.elements) {
        output[index] = left[index] + float(right[index]);
    }
}

kernel void vnext_last_token_masked_argmax_f32(
    device const float * logits [[buffer(0)]],
    device float * scratch [[buffer(1)]],
    device const uchar * valid_mask [[buffer(2)]],
    device const uint * repetition_token_ids [[buffer(3)]],
    device const uint * repetition_offsets [[buffer(4)]],
    device const float * repetition_penalty [[buffer(5)]],
    device uint * output [[buffer(6)]],
    constant LastTokenMaskedArgmaxParams & params [[buffer(7)]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const float penalty = repetition_penalty[0];
    const uint repetition_start = min(repetition_offsets[0], params.repetition_capacity);
    const uint repetition_end = min(repetition_offsets[1], params.repetition_capacity);
    const bool apply_penalty = penalty != 1.0f && repetition_start < repetition_end;
    if (apply_penalty) {
        for (uint token = lane; token < params.vocabulary_size; token += THREADS_PER_GROUP) {
            scratch[token] = logits[token];
        }
        threadgroup_barrier(mem_flags::mem_device);
        for (uint entry = repetition_start + lane;
             entry < repetition_end;
             entry += THREADS_PER_GROUP) {
            const uint token = repetition_token_ids[entry];
            if (token >= params.vocabulary_size) {
                continue;
            }
            const float value = scratch[token];
            if (!isfinite(value)) {
                continue;
            }
            scratch[token] = value > 0.0f ? value / penalty : value * penalty;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    float local_maximum = -INFINITY;
    int local_index = -1;
    for (uint token = lane; token < params.vocabulary_size; token += THREADS_PER_GROUP) {
        if (valid_mask[token] == 0) {
            continue;
        }
        const float value = apply_penalty ? scratch[token] : logits[token];
        if (!isfinite(value)) {
            continue;
        }
        if (local_index < 0 || value > local_maximum ||
            (value == local_maximum && int(token) < local_index)) {
            local_maximum = value;
            local_index = int(token);
        }
    }

    const float simd_maximum = simd_max(local_maximum);
    const int simd_index = simd_min(
        local_index >= 0 && local_maximum == simd_maximum ? local_index : 0x7fffffff
    );
    threadgroup float partial_maximum[THREADS_PER_GROUP / 32];
    threadgroup int partial_index[THREADS_PER_GROUP / 32];
    if (simd_lane == 0) {
        partial_maximum[simd_group] = simd_maximum;
        partial_index[simd_group] = simd_index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float maximum = -INFINITY;
        int index = 0x7fffffff;
        for (uint group = 0; group < THREADS_PER_GROUP / 32; ++group) {
            const float candidate_maximum = partial_maximum[group];
            const int candidate_index = partial_index[group];
            if (candidate_index != 0x7fffffff &&
                (index == 0x7fffffff || candidate_maximum > maximum ||
                 (candidate_maximum == maximum && candidate_index < index))) {
                maximum = candidate_maximum;
                index = candidate_index;
            }
        }
        output[0] = index == 0x7fffffff ? 0xffffffffu : uint(index);
    }
}
