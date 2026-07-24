#include <metal_stdlib>
using namespace metal;

constant uint THREADS_PER_GROUP = 256;
constant uint VALUE_TILE = 16;
constant uint SIMD_DELTA_WIDTH = 32;
constant uint SIMD_DELTA_ROWS_PER_GROUP = 4;
constant uint SIMD_DELTA_MAX_COLUMNS_PER_LANE = 4;
constant uint GATED_DELTA_CHUNK_SIZE = 64;
constant uint GATED_DELTA_CHUNK_KEY_DIM_LIMIT = 128;
constant bool GATED_DELTA_CARRY_KEY_DIM_128 [[function_constant(0)]];
constant bool GATED_DELTA_KKT_PRECOMPUTED_GRAM [[function_constant(1)]];

struct GatedDeltaParams {
    uint tokens;
    uint hidden_size;
    uint key_heads;
    uint value_heads;
    uint key_dim;
    uint value_dim;
    uint qkv_features;
    uint value_features;
    uint qkvz_features;
    uint ba_features;
    uint conv_kernel;
    float epsilon;
    float scale;
    uint decay_parameterization;
    uint value_head_mapping;
};

static inline float ferrum_silu(float value) {
    return value / (1.0f + exp(-value));
}

static inline float ferrum_softplus(float value) {
    return value > 20.0f ? value : (value < -20.0f ? exp(value) : log(1.0f + exp(value)));
}

kernel void vnext_gated_delta_prepare_conv_f16(
    device const half * mixed_qkvz [[buffer(0)]],
    device const half * conv_weight [[buffer(1)]],
    device const half * initial_state [[buffer(2)]],
    device float * query [[buffer(3)]],
    device float * key [[buffer(4)]],
    device float * value [[buffer(5)]],
    device half * z [[buffer(6)]],
    constant GatedDeltaParams & params [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    const uint total = params.tokens * params.qkvz_features;
    if (index >= total) {
        return;
    }
    const uint token = index / params.qkvz_features;
    const uint channel = index - token * params.qkvz_features;
    if (channel >= params.qkv_features) {
        z[ulong(token) * params.value_features + channel - params.qkv_features] =
            mixed_qkvz[index];
        return;
    }
    const uint state_width = params.conv_kernel - 1;
    float sum = 0.0f;
    for (uint kernel_index = 0; kernel_index < params.conv_kernel; ++kernel_index) {
        const int source = int(token) + int(kernel_index) - int(state_width);
        const float activation = source >= 0
            ? float(mixed_qkvz[ulong(source) * params.qkvz_features + channel])
            : float(initial_state[ulong(channel) * state_width + uint(int(state_width) + source)]);
        sum += activation
            * float(conv_weight[ulong(channel) * params.conv_kernel + kernel_index]);
    }
    const float mixed = ferrum_silu(sum);
    const uint qk_features = params.key_heads * params.key_dim;
    if (channel < qk_features) {
        query[ulong(token) * qk_features + channel] = mixed;
    } else if (channel < 2 * qk_features) {
        key[ulong(token) * qk_features + channel - qk_features] = mixed;
    } else {
        value[ulong(token) * params.value_features + channel - 2 * qk_features] = mixed;
    }
}

kernel void vnext_gated_delta_prepare_gates_f16(
    device const half * ba_raw [[buffer(0)]],
    device const float * a_log [[buffer(1)]],
    device const float * dt_bias [[buffer(2)]],
    device float * g [[buffer(3)]],
    device float * beta [[buffer(4)]],
    constant GatedDeltaParams & params [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    const uint total = params.tokens * params.value_heads;
    if (index >= total) {
        return;
    }
    const uint token = index / params.value_heads;
    const uint head = index % params.value_heads;
    const ulong base = ulong(token) * params.ba_features;
    const float b = float(ba_raw[base + head]);
    const float a = float(ba_raw[base + params.value_heads + head]) + dt_bias[head];
    const float decay_rate = params.decay_parameterization == 0
        ? -exp(a_log[head])
        : a_log[head];
    g[index] = decay_rate * ferrum_softplus(a);
    beta[index] = 1.0f / (1.0f + exp(-b));
}

kernel void vnext_gated_delta_collect_conv_state_f16(
    device const half * mixed_qkvz [[buffer(0)]],
    device const half * initial_state [[buffer(1)]],
    device half * final_state [[buffer(2)]],
    constant GatedDeltaParams & params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const uint state_width = params.conv_kernel - 1;
    const uint total = params.qkv_features * state_width;
    if (index >= total) {
        return;
    }
    const uint channel = index / state_width;
    const uint position = index - channel * state_width;
    const int source = int(params.tokens) + int(position) - int(state_width);
    final_state[index] = source >= 0
        ? mixed_qkvz[ulong(source) * params.qkvz_features + channel]
        : initial_state[ulong(channel) * state_width + uint(int(state_width) + source)];
}

kernel void vnext_gated_delta_copy_f16(
    device const half * input [[buffer(0)]],
    device half * output [[buffer(1)]],
    constant uint & elements [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < elements) {
        output[index] = input[index];
    }
}

kernel void vnext_gated_delta_qk_norm_f32(
    device float * query [[buffer(0)]],
    device float * key [[buffer(1)]],
    constant GatedDeltaParams & params [[buffer(2)]],
    threadgroup float * partial [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row = group.x;
    const uint rows = params.tokens * params.key_heads;
    if (row >= rows) {
        return;
    }
    const ulong base = ulong(row) * params.key_dim;
    float query_sum = 0.0f;
    float key_sum = 0.0f;
    for (uint column = lane; column < params.key_dim; column += THREADS_PER_GROUP) {
        const float q = query[base + column];
        const float k = key[base + column];
        query_sum += q * q;
        key_sum += k * k;
    }
    query_sum = simd_sum(query_sum);
    key_sum = simd_sum(key_sum);
    if (simd_lane == 0) {
        partial[simd_group] = query_sum;
        partial[8 + simd_group] = key_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        const float q = simd_lane < 8 ? partial[simd_lane] : 0.0f;
        const float k = simd_lane < 8 ? partial[8 + simd_lane] : 0.0f;
        const float q_total = simd_sum(q);
        const float k_total = simd_sum(k);
        if (simd_lane == 0) {
            partial[0] = rsqrt(q_total + 1.0e-6f);
            partial[1] = rsqrt(k_total + 1.0e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float query_inverse = partial[0];
    const float key_inverse = partial[1];
    for (uint column = lane; column < params.key_dim; column += THREADS_PER_GROUP) {
        query[base + column] *= query_inverse;
        key[base + column] *= key_inverse;
    }
}

kernel void vnext_gated_delta_rule_tiled16_f32_state(
    device const float * query [[buffer(0)]],
    device const float * key [[buffer(1)]],
    device const float * value [[buffer(2)]],
    device const float * g [[buffer(3)]],
    device const float * beta [[buffer(4)]],
    device float * state [[buffer(5)]],
    device float * output [[buffer(6)]],
    constant GatedDeltaParams & params [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint value_start = group.x * VALUE_TILE;
    const uint value_head = group.y;
    if (value_head >= params.value_heads || value_start >= params.value_dim) {
        return;
    }
    const uint repeat = params.value_heads / params.key_heads;
    const uint key_head = params.value_head_mapping == 0
        ? value_head / repeat
        : value_head % params.key_heads;
    threadgroup float partial[VALUE_TILE][THREADS_PER_GROUP];
    threadgroup float delta[VALUE_TILE];
    float local[VALUE_TILE];

    for (uint token = 0; token < params.tokens; ++token) {
        const uint gate_index = token * params.value_heads + value_head;
        const float decay = exp(g[gate_index]);
        const float beta_value = beta[gate_index];
        for (uint local_value = 0; local_value < VALUE_TILE; ++local_value) {
            local[local_value] = 0.0f;
        }
        for (uint index = lane; index < VALUE_TILE * params.key_dim;
             index += THREADS_PER_GROUP) {
            const uint local_value = index / params.key_dim;
            const uint key_column = index - local_value * params.key_dim;
            const uint value_column = value_start + local_value;
            if (value_column >= params.value_dim) {
                continue;
            }
            const ulong state_index =
                (ulong(value_head) * params.value_dim + value_column) * params.key_dim
                + key_column;
            const ulong qk_index =
                (ulong(token) * params.key_heads + key_head) * params.key_dim + key_column;
            const float decayed = state[state_index] * decay;
            state[state_index] = decayed;
            local[local_value] += decayed * key[qk_index];
        }
        for (uint local_value = 0; local_value < VALUE_TILE; ++local_value) {
            partial[local_value][lane] = local[local_value];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                for (uint local_value = 0; local_value < VALUE_TILE; ++local_value) {
                    partial[local_value][lane] += partial[local_value][lane + stride];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            for (uint local_value = 0; local_value < VALUE_TILE; ++local_value) {
                const uint value_column = value_start + local_value;
                if (value_column < params.value_dim) {
                    const ulong value_index =
                        (ulong(token) * params.value_heads + value_head) * params.value_dim
                        + value_column;
                    delta[local_value] =
                        (value[value_index] - partial[local_value][0]) * beta_value;
                } else {
                    delta[local_value] = 0.0f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint local_value = 0; local_value < VALUE_TILE; ++local_value) {
            local[local_value] = 0.0f;
        }
        for (uint index = lane; index < VALUE_TILE * params.key_dim;
             index += THREADS_PER_GROUP) {
            const uint local_value = index / params.key_dim;
            const uint key_column = index - local_value * params.key_dim;
            const uint value_column = value_start + local_value;
            if (value_column >= params.value_dim) {
                continue;
            }
            const ulong state_index =
                (ulong(value_head) * params.value_dim + value_column) * params.key_dim
                + key_column;
            const ulong qk_index =
                (ulong(token) * params.key_heads + key_head) * params.key_dim + key_column;
            const float updated = state[state_index] + delta[local_value] * key[qk_index];
            state[state_index] = updated;
            local[local_value] += updated * query[qk_index] * params.scale;
        }
        for (uint local_value = 0; local_value < VALUE_TILE; ++local_value) {
            partial[local_value][lane] = local[local_value];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                for (uint local_value = 0; local_value < VALUE_TILE; ++local_value) {
                    partial[local_value][lane] += partial[local_value][lane + stride];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            for (uint local_value = 0; local_value < VALUE_TILE; ++local_value) {
                const uint value_column = value_start + local_value;
                if (value_column < params.value_dim) {
                    const ulong output_index =
                        (ulong(token) * params.value_heads + value_head) * params.value_dim
                        + value_column;
                    output[output_index] = partial[local_value][0];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void vnext_gated_delta_rule_simd_f32_state(
    device const float * query [[buffer(0)]],
    device const float * key [[buffer(1)]],
    device const float * value [[buffer(2)]],
    device const float * g [[buffer(3)]],
    device const float * beta [[buffer(4)]],
    device float * state [[buffer(5)]],
    device float * output [[buffer(6)]],
    constant GatedDeltaParams & params [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint value_column = group.x * SIMD_DELTA_ROWS_PER_GROUP + simd_group;
    const uint value_head = group.y;
    if (value_head >= params.value_heads || value_column >= params.value_dim
        || params.key_dim == 0 || params.key_dim > 128
        || params.key_dim % SIMD_DELTA_WIDTH != 0) {
        return;
    }
    const uint repeat = params.value_heads / params.key_heads;
    const uint key_head = params.value_head_mapping == 0
        ? value_head / repeat
        : value_head % params.key_heads;
    const uint columns_per_lane = params.key_dim / SIMD_DELTA_WIDTH;
    const ulong state_base =
        (ulong(value_head) * params.value_dim + value_column) * params.key_dim;
    float local[SIMD_DELTA_MAX_COLUMNS_PER_LANE];
    for (uint column = 0; column < columns_per_lane; ++column) {
        local[column] = state[state_base + simd_lane * columns_per_lane + column];
    }

    for (uint token = 0; token < params.tokens; ++token) {
        const uint gate_index = token * params.value_heads + value_head;
        const float decay = exp(g[gate_index]);
        const ulong qk_base =
            (ulong(token) * params.key_heads + key_head) * params.key_dim;
        float predicted = 0.0f;
        for (uint column = 0; column < columns_per_lane; ++column) {
            const uint key_column = simd_lane * columns_per_lane + column;
            local[column] *= decay;
            predicted += local[column] * key[qk_base + key_column];
        }
        predicted = simd_sum(predicted);
        const ulong value_index =
            (ulong(token) * params.value_heads + value_head) * params.value_dim
            + value_column;
        const float delta = (value[value_index] - predicted) * beta[gate_index];
        float result = 0.0f;
        for (uint column = 0; column < columns_per_lane; ++column) {
            const uint key_column = simd_lane * columns_per_lane + column;
            local[column] += key[qk_base + key_column] * delta;
            result += local[column] * query[qk_base + key_column];
        }
        result = simd_sum(result);
        if (simd_lane == 0) {
            output[value_index] = result * params.scale;
        }
    }

    for (uint column = 0; column < columns_per_lane; ++column) {
        state[state_base + simd_lane * columns_per_lane + column] = local[column];
    }
}

kernel void vnext_gated_delta_chunk_k_gram_c64_k128(
    device const float * key [[buffer(0)]],
    device float * gram [[buffer(1)]],
    constant GatedDeltaParams & params [[buffer(2)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint chunk_start = group.x * GATED_DELTA_CHUNK_SIZE;
    const uint key_head = group.y;
    if (chunk_start >= params.tokens || key_head >= params.key_heads
        || params.key_dim != GATED_DELTA_CHUNK_KEY_DIM_LIMIT) {
        return;
    }
    const uint chunk_tokens = min(GATED_DELTA_CHUNK_SIZE, params.tokens - chunk_start);
    for (uint index = lane;
         index < GATED_DELTA_CHUNK_SIZE * GATED_DELTA_CHUNK_SIZE;
         index += THREADS_PER_GROUP) {
        const uint row = index / GATED_DELTA_CHUNK_SIZE;
        const uint column = index - row * GATED_DELTA_CHUNK_SIZE;
        if (row >= chunk_tokens) {
            continue;
        }
        float dot = 0.0f;
        if (column < row) {
            const uint row_token = chunk_start + row;
            const uint column_token = chunk_start + column;
            const ulong row_base =
                (ulong(row_token) * params.key_heads + key_head)
                    * GATED_DELTA_CHUNK_KEY_DIM_LIMIT;
            const ulong column_base =
                (ulong(column_token) * params.key_heads + key_head)
                    * GATED_DELTA_CHUNK_KEY_DIM_LIMIT;
            for (uint key_column = 0;
                 key_column < GATED_DELTA_CHUNK_KEY_DIM_LIMIT;
                 ++key_column) {
                dot += key[row_base + key_column] * key[column_base + key_column];
            }
        }
        gram[(ulong(chunk_start + row) * params.key_heads + key_head)
            * GATED_DELTA_CHUNK_SIZE + column] = dot;
    }
}

kernel void vnext_gated_delta_chunk_kkt_inverse_c64(
    device const float * key_or_gram [[buffer(0)]],
    device float * g [[buffer(1)]],
    device const float * beta [[buffer(2)]],
    device half * inverse [[buffer(3)]],
    constant GatedDeltaParams & params [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint chunk_start = group.x * GATED_DELTA_CHUNK_SIZE;
    const uint value_head = group.y;
    if (chunk_start >= params.tokens || value_head >= params.value_heads
        || (GATED_DELTA_KKT_PRECOMPUTED_GRAM
            && params.key_dim != GATED_DELTA_CHUNK_KEY_DIM_LIMIT)) {
        return;
    }
    const uint chunk_tokens = min(GATED_DELTA_CHUNK_SIZE, params.tokens - chunk_start);
    const uint repeat = params.value_heads / params.key_heads;
    const uint key_head = params.value_head_mapping == 0
        ? value_head / repeat
        : value_head % params.key_heads;
    threadgroup float kkt[GATED_DELTA_CHUNK_SIZE * GATED_DELTA_CHUNK_SIZE];

    if (lane == 0) {
        float cumulative = 0.0f;
        for (uint row = 0; row < chunk_tokens; ++row) {
            const ulong gate_index =
                ulong(chunk_start + row) * params.value_heads + value_head;
            cumulative += g[gate_index];
            g[gate_index] = cumulative;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint index = lane;
         index < GATED_DELTA_CHUNK_SIZE * GATED_DELTA_CHUNK_SIZE;
         index += THREADS_PER_GROUP) {
        const uint row = index / GATED_DELTA_CHUNK_SIZE;
        const uint column = index - row * GATED_DELTA_CHUNK_SIZE;
        float coefficient = 0.0f;
        if (row < chunk_tokens && column < row) {
            const uint row_token = chunk_start + row;
            const uint column_token = chunk_start + column;
            float dot = 0.0f;
            if (GATED_DELTA_KKT_PRECOMPUTED_GRAM) {
                dot = key_or_gram[
                    (ulong(row_token) * params.key_heads + key_head)
                        * GATED_DELTA_CHUNK_SIZE
                    + column];
            } else {
                const ulong row_base =
                    (ulong(row_token) * params.key_heads + key_head) * params.key_dim;
                const ulong column_base =
                    (ulong(column_token) * params.key_heads + key_head) * params.key_dim;
                for (uint key_column = 0; key_column < params.key_dim; ++key_column) {
                    dot += key_or_gram[row_base + key_column]
                        * key_or_gram[column_base + key_column];
                }
            }
            const float row_g =
                g[ulong(row_token) * params.value_heads + value_head];
            const float column_g =
                g[ulong(column_token) * params.value_heads + value_head];
            coefficient =
                beta[ulong(row_token) * params.value_heads + value_head]
                * exp(row_g - column_g) * dot;
        }
        kkt[index] = coefficient;
        if (row < chunk_tokens) {
            const ulong inverse_index =
                (ulong(chunk_start + row) * params.value_heads + value_head)
                    * GATED_DELTA_CHUNK_SIZE
                + column;
            inverse[inverse_index] = half(row == column ? 1.0f : 0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    // Invert the unit-lower matrix row by row. Previous rows are deliberately
    // stored as half, matching the FLA contract's solve output precision.
    for (uint row = 1; row < chunk_tokens; ++row) {
        if (lane < row) {
            const uint column = lane;
            float sum = 0.0f;
            for (uint inner = column; inner < row; ++inner) {
                const ulong previous_index =
                    (ulong(chunk_start + inner) * params.value_heads + value_head)
                        * GATED_DELTA_CHUNK_SIZE
                    + column;
                sum += kkt[row * GATED_DELTA_CHUNK_SIZE + inner]
                    * float(inverse[previous_index]);
            }
            const ulong output_index =
                (ulong(chunk_start + row) * params.value_heads + value_head)
                    * GATED_DELTA_CHUNK_SIZE
                + column;
            inverse[output_index] = half(-sum);
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

kernel void vnext_gated_delta_chunk_uw_c64(
    device const float * key [[buffer(0)]],
    device const float * value [[buffer(1)]],
    device const float * g [[buffer(2)]],
    device const float * beta [[buffer(3)]],
    device const half * inverse [[buffer(4)]],
    device half * uw [[buffer(5)]],
    constant GatedDeltaParams & params [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    const uint row_width = params.value_dim + params.key_dim;
    const ulong total =
        ulong(params.tokens) * params.value_heads * row_width;
    if (ulong(index) >= total) {
        return;
    }
    const uint component = index % row_width;
    const uint row = index / row_width;
    const uint value_head = row % params.value_heads;
    const uint token = row / params.value_heads;
    const uint chunk_start =
        (token / GATED_DELTA_CHUNK_SIZE) * GATED_DELTA_CHUNK_SIZE;
    const uint local_row = token - chunk_start;
    const uint repeat = params.value_heads / params.key_heads;
    const uint key_head = params.value_head_mapping == 0
        ? value_head / repeat
        : value_head % params.key_heads;
    float result = 0.0f;
    for (uint local_column = 0; local_column <= local_row; ++local_column) {
        const uint source_token = chunk_start + local_column;
        const float solve = float(inverse[
            (ulong(token) * params.value_heads + value_head)
                * GATED_DELTA_CHUNK_SIZE
            + local_column]);
        const float source_beta =
            beta[ulong(source_token) * params.value_heads + value_head];
        if (component < params.value_dim) {
            result += solve * source_beta * value[
                (ulong(source_token) * params.value_heads + value_head)
                    * params.value_dim
                + component];
        } else {
            const uint key_column = component - params.value_dim;
            const float source_g =
                g[ulong(source_token) * params.value_heads + value_head];
            result += solve * source_beta * exp(source_g) * key[
                (ulong(source_token) * params.key_heads + key_head) * params.key_dim
                + key_column];
        }
    }
    uw[index] = half(result);
}

kernel void vnext_gated_delta_chunk_uw_c64_k128_v128(
    device const float * key [[buffer(0)]],
    device const float * value [[buffer(1)]],
    device const float * g [[buffer(2)]],
    device const float * beta [[buffer(3)]],
    device const half * inverse [[buffer(4)]],
    device half * uw [[buffer(5)]],
    constant GatedDeltaParams & params [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint token = group.x;
    const uint value_head = group.y;
    if (token >= params.tokens || value_head >= params.value_heads
        || params.key_dim != GATED_DELTA_CHUNK_KEY_DIM_LIMIT
        || params.value_dim != GATED_DELTA_CHUNK_KEY_DIM_LIMIT) {
        return;
    }

    const uint chunk_start =
        (token / GATED_DELTA_CHUNK_SIZE) * GATED_DELTA_CHUNK_SIZE;
    const uint local_row = token - chunk_start;
    const uint repeat = params.value_heads / params.key_heads;
    const uint key_head = params.value_head_mapping == 0
        ? value_head / repeat
        : value_head % params.key_heads;
    threadgroup float value_weights[GATED_DELTA_CHUNK_SIZE];
    threadgroup float key_weights[GATED_DELTA_CHUNK_SIZE];

    if (lane <= local_row) {
        const uint source_token = chunk_start + lane;
        const float solve = float(inverse[
            (ulong(token) * params.value_heads + value_head)
                * GATED_DELTA_CHUNK_SIZE
            + lane]);
        const float weight =
            solve * beta[ulong(source_token) * params.value_heads + value_head];
        value_weights[lane] = weight;
        key_weights[lane] = weight
            * exp(g[ulong(source_token) * params.value_heads + value_head]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float result = 0.0f;
    if (lane < GATED_DELTA_CHUNK_KEY_DIM_LIMIT) {
        for (uint local_column = 0; local_column <= local_row; ++local_column) {
            const uint source_token = chunk_start + local_column;
            result += value_weights[local_column] * value[
                (ulong(source_token) * params.value_heads + value_head)
                    * GATED_DELTA_CHUNK_KEY_DIM_LIMIT
                + lane];
        }
    } else {
        const uint key_column = lane - GATED_DELTA_CHUNK_KEY_DIM_LIMIT;
        for (uint local_column = 0; local_column <= local_row; ++local_column) {
            const uint source_token = chunk_start + local_column;
            result += key_weights[local_column] * key[
                (ulong(source_token) * params.key_heads + key_head)
                    * GATED_DELTA_CHUNK_KEY_DIM_LIMIT
                + key_column];
        }
    }
    uw[(ulong(token) * params.value_heads + value_head)
        * (2 * GATED_DELTA_CHUNK_KEY_DIM_LIMIT) + lane] = half(result);
}

kernel void vnext_gated_delta_chunk_qk_c64(
    device const float * query [[buffer(0)]],
    device const float * key [[buffer(1)]],
    device half * raw_qk [[buffer(2)]],
    constant GatedDeltaParams & params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const ulong total = ulong(params.tokens) * params.key_heads * GATED_DELTA_CHUNK_SIZE;
    if (ulong(index) >= total) {
        return;
    }
    const uint local_column = index % GATED_DELTA_CHUNK_SIZE;
    const uint row = index / GATED_DELTA_CHUNK_SIZE;
    const uint key_head = row % params.key_heads;
    const uint token = row / params.key_heads;
    const uint chunk_start =
        (token / GATED_DELTA_CHUNK_SIZE) * GATED_DELTA_CHUNK_SIZE;
    const uint local_row = token - chunk_start;
    float dot = 0.0f;
    if (local_column <= local_row) {
        const uint source_token = chunk_start + local_column;
        const ulong query_base =
            (ulong(token) * params.key_heads + key_head) * params.key_dim;
        const ulong key_base =
            (ulong(source_token) * params.key_heads + key_head) * params.key_dim;
        for (uint key_column = 0; key_column < params.key_dim; ++key_column) {
            dot += query[query_base + key_column] * key[key_base + key_column];
        }
    }
    raw_qk[index] = half(dot);
}

kernel void vnext_gated_delta_chunk_carry_c64(
    device const float * query [[buffer(0)]],
    device const float * key [[buffer(1)]],
    device const float * g [[buffer(2)]],
    device half * uw [[buffer(3)]],
    device float * state [[buffer(4)]],
    device float * output [[buffer(5)]],
    constant GatedDeltaParams & params [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint value_start = group.x * VALUE_TILE;
    const uint value_head = group.y;
    if (value_start >= params.value_dim || value_head >= params.value_heads
        || params.key_dim > GATED_DELTA_CHUNK_KEY_DIM_LIMIT) {
        return;
    }
    const uint repeat = params.value_heads / params.key_heads;
    const uint key_head = params.value_head_mapping == 0
        ? value_head / repeat
        : value_head % params.key_heads;
    const uint uw_width = params.value_dim + params.key_dim;
    threadgroup float state_tile[VALUE_TILE * GATED_DELTA_CHUNK_KEY_DIM_LIMIT];

    for (uint index = lane; index < VALUE_TILE * params.key_dim;
         index += THREADS_PER_GROUP) {
        const uint local_value = index / params.key_dim;
        const uint key_column = index - local_value * params.key_dim;
        const uint value_column = value_start + local_value;
        state_tile[index] = value_column < params.value_dim
            ? state[(ulong(value_head) * params.value_dim + value_column) * params.key_dim
                + key_column]
            : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint chunk_start = 0; chunk_start < params.tokens;
         chunk_start += GATED_DELTA_CHUNK_SIZE) {
        const uint chunk_tokens = min(GATED_DELTA_CHUNK_SIZE, params.tokens - chunk_start);
        for (uint item = lane; item < chunk_tokens * VALUE_TILE;
             item += THREADS_PER_GROUP) {
            const uint local_token = item / VALUE_TILE;
            const uint local_value = item - local_token * VALUE_TILE;
            const uint value_column = value_start + local_value;
            if (value_column >= params.value_dim) {
                continue;
            }
            const uint token = chunk_start + local_token;
            const ulong uw_base =
                (ulong(token) * params.value_heads + value_head) * uw_width;
            const ulong q_base =
                (ulong(token) * params.key_heads + key_head) * params.key_dim;
            float state_prediction = 0.0f;
            float state_output = 0.0f;
            const uint state_base = local_value * params.key_dim;
            for (uint key_column = 0; key_column < params.key_dim; ++key_column) {
                const float current = state_tile[state_base + key_column];
                state_prediction +=
                    float(uw[uw_base + params.value_dim + key_column]) * current;
                state_output += query[q_base + key_column] * current;
            }
            uw[uw_base + value_column] =
                half(float(uw[uw_base + value_column]) - state_prediction);
            output[(ulong(token) * params.value_heads + value_head) * params.value_dim
                + value_column] = exp(g[ulong(token) * params.value_heads + value_head])
                * state_output * params.scale;
        }
        threadgroup_barrier(mem_flags::mem_device);

        const uint final_token = chunk_start + chunk_tokens - 1;
        const float final_g =
            g[ulong(final_token) * params.value_heads + value_head];
        const float final_decay = exp(final_g);
        if (GATED_DELTA_CARRY_KEY_DIM_128
            && params.key_dim == GATED_DELTA_CHUNK_KEY_DIM_LIMIT) {
            // At key_dim=128 each lane owns eight value rows with one shared
            // key column. Keep those rows in registers so every token decay
            // and key load is consumed eight times instead of recomputed.
            constexpr uint ROWS_PER_LANE =
                (VALUE_TILE * GATED_DELTA_CHUNK_KEY_DIM_LIMIT) / THREADS_PER_GROUP;
            float updated[ROWS_PER_LANE];
            bool valid[ROWS_PER_LANE];
            const uint key_column = lane % GATED_DELTA_CHUNK_KEY_DIM_LIMIT;
            for (uint slot = 0; slot < ROWS_PER_LANE; ++slot) {
                const uint index = lane + slot * THREADS_PER_GROUP;
                const uint local_value = index / GATED_DELTA_CHUNK_KEY_DIM_LIMIT;
                const uint value_column = value_start + local_value;
                valid[slot] = value_column < params.value_dim;
                updated[slot] = valid[slot] ? final_decay * state_tile[index] : 0.0f;
            }
            for (uint local_token = 0; local_token < chunk_tokens; ++local_token) {
                const uint token = chunk_start + local_token;
                const ulong uw_base =
                    (ulong(token) * params.value_heads + value_head) * uw_width;
                const ulong key_base =
                    (ulong(token) * params.key_heads + key_head) * params.key_dim;
                const float token_g =
                    g[ulong(token) * params.value_heads + value_head];
                const float weighted_key =
                    exp(final_g - token_g) * key[key_base + key_column];
                for (uint slot = 0; slot < ROWS_PER_LANE; ++slot) {
                    if (valid[slot]) {
                        const uint index = lane + slot * THREADS_PER_GROUP;
                        const uint local_value =
                            index / GATED_DELTA_CHUNK_KEY_DIM_LIMIT;
                        const uint value_column = value_start + local_value;
                        updated[slot] += float(uw[uw_base + value_column]) * weighted_key;
                    }
                }
            }
            for (uint slot = 0; slot < ROWS_PER_LANE; ++slot) {
                const uint index = lane + slot * THREADS_PER_GROUP;
                if (valid[slot]) {
                    state_tile[index] = updated[slot];
                }
            }
        } else {
            for (uint index = lane; index < VALUE_TILE * params.key_dim;
                 index += THREADS_PER_GROUP) {
                const uint local_value = index / params.key_dim;
                const uint key_column = index - local_value * params.key_dim;
                const uint value_column = value_start + local_value;
                if (value_column >= params.value_dim) {
                    continue;
                }
                float updated = final_decay * state_tile[index];
                for (uint local_token = 0; local_token < chunk_tokens; ++local_token) {
                    const uint token = chunk_start + local_token;
                    const ulong uw_base =
                        (ulong(token) * params.value_heads + value_head) * uw_width;
                    const ulong key_base =
                        (ulong(token) * params.key_heads + key_head) * params.key_dim;
                    const float token_g =
                        g[ulong(token) * params.value_heads + value_head];
                    updated += exp(final_g - token_g)
                        * float(uw[uw_base + value_column])
                        * key[key_base + key_column];
                }
                state_tile[index] = updated;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint index = lane; index < VALUE_TILE * params.key_dim;
         index += THREADS_PER_GROUP) {
        const uint local_value = index / params.key_dim;
        const uint key_column = index - local_value * params.key_dim;
        const uint value_column = value_start + local_value;
        if (value_column < params.value_dim) {
            state[(ulong(value_head) * params.value_dim + value_column) * params.key_dim
                + key_column] = state_tile[index];
        }
    }
}

kernel void vnext_gated_delta_chunk_output_c64(
    device const half * raw_qk [[buffer(0)]],
    device const float * g [[buffer(1)]],
    device const half * uw [[buffer(2)]],
    device float * output [[buffer(3)]],
    constant GatedDeltaParams & params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const ulong total = ulong(params.tokens) * params.value_heads * params.value_dim;
    if (ulong(index) >= total) {
        return;
    }
    const uint value_column = index % params.value_dim;
    const uint row = index / params.value_dim;
    const uint value_head = row % params.value_heads;
    const uint token = row / params.value_heads;
    const uint repeat = params.value_heads / params.key_heads;
    const uint key_head = params.value_head_mapping == 0
        ? value_head / repeat
        : value_head % params.key_heads;
    const uint chunk_start =
        (token / GATED_DELTA_CHUNK_SIZE) * GATED_DELTA_CHUNK_SIZE;
    const uint local_row = token - chunk_start;
    const uint uw_width = params.value_dim + params.key_dim;
    const float token_g = g[ulong(token) * params.value_heads + value_head];
    float block_output = 0.0f;
    const ulong qk_base =
        (ulong(token) * params.key_heads + key_head) * GATED_DELTA_CHUNK_SIZE;
    for (uint local_column = 0; local_column <= local_row; ++local_column) {
        const uint source_token = chunk_start + local_column;
        const float source_g =
            g[ulong(source_token) * params.value_heads + value_head];
        const ulong uw_base =
            (ulong(source_token) * params.value_heads + value_head) * uw_width;
        block_output += exp(token_g - source_g) * float(raw_qk[qk_base + local_column])
            * float(uw[uw_base + value_column]);
    }
    output[index] += block_output * params.scale;
}

kernel void vnext_gated_delta_gated_norm_f16(
    device const float * core [[buffer(0)]],
    device const half * z [[buffer(1)]],
    device const float * weight [[buffer(2)]],
    device half * output [[buffer(3)]],
    constant GatedDeltaParams & params [[buffer(4)]],
    threadgroup float * partial [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row = group.x;
    const uint rows = params.tokens * params.value_heads;
    if (row >= rows) {
        return;
    }
    const ulong base = ulong(row) * params.value_dim;
    float sum = 0.0f;
    for (uint column = lane; column < params.value_dim; column += THREADS_PER_GROUP) {
        const float current = core[base + column];
        sum += current * current;
    }
    sum = simd_sum(sum);
    if (simd_lane == 0) {
        partial[simd_group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        const float value = simd_lane < 8 ? partial[simd_lane] : 0.0f;
        const float total = simd_sum(value);
        if (simd_lane == 0) {
            partial[0] = rsqrt(total / float(params.value_dim) + params.epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inverse_rms = partial[0];
    for (uint column = lane; column < params.value_dim; column += THREADS_PER_GROUP) {
        output[base + column] = half(
            core[base + column] * inverse_rms * weight[column]
            * ferrum_silu(float(z[base + column]))
        );
    }
}
