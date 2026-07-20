#include <metal_stdlib>
using namespace metal;

constant uint THREADS_PER_GROUP = 256;
constant uint VALUE_TILE = 16;

struct GatedDeltaParams {
    uint tokens;
    uint hidden_size;
    uint key_heads;
    uint value_heads;
    uint key_dim;
    uint value_dim;
    uint qkv_features;
    uint value_features;
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
    device const half * mixed_qkv [[buffer(0)]],
    device const half * conv_weight [[buffer(1)]],
    device const half * initial_state [[buffer(2)]],
    device float * query [[buffer(3)]],
    device float * key [[buffer(4)]],
    device float * value [[buffer(5)]],
    constant GatedDeltaParams & params [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    const uint total = params.tokens * params.qkv_features;
    if (index >= total) {
        return;
    }
    const uint token = index / params.qkv_features;
    const uint channel = index - token * params.qkv_features;
    const uint state_width = params.conv_kernel - 1;
    float sum = 0.0f;
    for (uint kernel_index = 0; kernel_index < params.conv_kernel; ++kernel_index) {
        const int source = int(token) + int(kernel_index) - int(state_width);
        const float activation = source >= 0
            ? float(mixed_qkv[ulong(source) * params.qkv_features + channel])
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
    device const half * a_raw [[buffer(0)]],
    device const half * b_raw [[buffer(1)]],
    device const float * a_log [[buffer(2)]],
    device const float * dt_bias [[buffer(3)]],
    device float * g [[buffer(4)]],
    device float * beta [[buffer(5)]],
    constant GatedDeltaParams & params [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    const uint total = params.tokens * params.value_heads;
    if (index >= total) {
        return;
    }
    const uint head = index % params.value_heads;
    const float a = float(a_raw[index]) + dt_bias[head];
    const float b = float(b_raw[index]);
    const float decay_rate = params.decay_parameterization == 0
        ? -exp(a_log[head])
        : a_log[head];
    g[index] = decay_rate * ferrum_softplus(a);
    beta[index] = 1.0f / (1.0f + exp(-b));
}

kernel void vnext_gated_delta_collect_conv_state_f16(
    device const half * mixed_qkv [[buffer(0)]],
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
        ? mixed_qkv[ulong(source) * params.qkv_features + channel]
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
