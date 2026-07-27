#include <metal_stdlib>

using namespace metal;

#define VNEXT_MAX_KV_PAGES 16384
#define VNEXT_SIMD_WIDTH 32
#define VNEXT_MAX_HEAD_CHUNKS 8

struct VNextKvPageTable {
    array<device half *, VNEXT_MAX_KV_PAGES> pages [[id(0)]];
};

struct VNextCausalAttentionParams {
    uint page_elements;
    uint page_count;
    uint position_start;
    uint tokens;
    uint query_heads;
    uint key_value_heads;
    uint head_dim;
    uint rope_dim;
    uint query_projection_stride;
    uint query_head_stride;
    uint kv_projection_stride;
    uint output_gate;
    uint rope_interleaved;
    uint attention_simdgroups;
    float epsilon;
    float rope_theta;
};

inline ulong vnext_kv_element_index(
    uint token,
    uint kind,
    uint head,
    uint dim,
    constant VNextCausalAttentionParams& params) {
    return (((ulong)token * 2ul + (ulong)kind) *
                (ulong)params.key_value_heads +
            (ulong)head) *
               (ulong)params.head_dim +
           (ulong)dim;
}

inline device half *vnext_paged_element(
    device VNextKvPageTable& table,
    constant VNextCausalAttentionParams& params,
    ulong logical_element) {
    const ulong page = logical_element / (ulong)params.page_elements;
    if (page >= (ulong)params.page_count) {
        return nullptr;
    }
    const ulong offset = logical_element - page * (ulong)params.page_elements;
    return table.pages[page] + offset;
}

inline void vnext_store_kv(
    device VNextKvPageTable& table,
    constant VNextCausalAttentionParams& params,
    uint token,
    uint kind,
    uint head,
    uint dim,
    half value) {
    device half *destination = vnext_paged_element(
        table,
        params,
        vnext_kv_element_index(token, kind, head, dim, params));
    if (destination != nullptr) {
        *destination = value;
    }
}

inline float vnext_load_kv(
    device VNextKvPageTable& table,
    constant VNextCausalAttentionParams& params,
    uint token,
    uint kind,
    uint head,
    uint dim) {
    device half *source = vnext_paged_element(
        table,
        params,
        vnext_kv_element_index(token, kind, head, dim, params));
    return source == nullptr ? 0.0f : float(*source);
}

inline void vnext_store_prepared_value(
    device half *query,
    device VNextKvPageTable& table,
    constant VNextCausalAttentionParams& params,
    bool is_query,
    uint token,
    uint absolute_position,
    uint head,
    uint dim,
    float value) {
    const half converted = half(value);
    if (is_query) {
        query[((ulong)token * (ulong)params.query_heads + (ulong)head) *
                  (ulong)params.head_dim +
              (ulong)dim] = converted;
    } else {
        vnext_store_kv(
            table, params, absolute_position, 0, head, dim, converted);
    }
}

kernel void vnext_causal_prepare_f16(
    const device half *query_raw [[buffer(0)]],
    const device half *key_raw [[buffer(1)]],
    const device half *value_raw [[buffer(2)]],
    const device half *query_norm_weight [[buffer(3)]],
    const device half *key_norm_weight [[buffer(4)]],
    device half *query [[buffer(5)]],
    device VNextKvPageTable& page_table [[buffer(6)]],
    constant VNextCausalAttentionParams& params [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint token = group.x;
    const uint combined_head = group.y;
    const uint combined_heads =
        params.query_heads + 2u * params.key_value_heads;
    if (token >= params.tokens || combined_head >= combined_heads ||
        lane >= VNEXT_SIMD_WIDTH) {
        return;
    }

    const bool is_query = combined_head < params.query_heads;
    const bool is_key = !is_query &&
                        combined_head < params.query_heads + params.key_value_heads;
    const uint head = is_query
                          ? combined_head
                          : (is_key
                                 ? combined_head - params.query_heads
                                 : combined_head - params.query_heads -
                                       params.key_value_heads);
    const uint absolute_position = params.position_start + token;

    if (!is_query && !is_key) {
        const device half *source =
            value_raw + (ulong)token * (ulong)params.kv_projection_stride +
            (ulong)head * (ulong)params.head_dim;
        for (uint dim = lane; dim < params.head_dim; dim += VNEXT_SIMD_WIDTH) {
            vnext_store_kv(
                page_table,
                params,
                absolute_position,
                1,
                head,
                dim,
                source[dim]);
        }
        return;
    }

    const device half *source = is_query
                                    ? query_raw +
                                          (ulong)token *
                                              (ulong)params.query_projection_stride +
                                          (ulong)head *
                                              (ulong)params.query_head_stride
                                    : key_raw +
                                          (ulong)token *
                                              (ulong)params.kv_projection_stride +
                                          (ulong)head * (ulong)params.head_dim;
    const device half *weight = is_query ? query_norm_weight : key_norm_weight;
    float sum_squares = 0.0f;
    for (uint dim = lane; dim < params.head_dim; dim += VNEXT_SIMD_WIDTH) {
        const float value = float(source[dim]);
        sum_squares += value * value;
    }
    sum_squares = simd_sum(sum_squares);
    const float norm_scale =
        rsqrt(sum_squares / float(params.head_dim) + params.epsilon);
    const uint half_rope = params.rope_dim / 2u;

    if (params.rope_interleaved != 0u) {
        for (uint pair = lane; pair < half_rope; pair += VNEXT_SIMD_WIDTH) {
            const uint low = 2u * pair;
            const uint high = low + 1u;
            const float x0 =
                float(source[low]) * norm_scale * float(weight[low]);
            const float x1 =
                float(source[high]) * norm_scale * float(weight[high]);
            const float exponent = -(2.0f * float(pair)) / float(params.rope_dim);
            const float angle =
                float(absolute_position) * powr(params.rope_theta, exponent);
            const float sine = sin(angle);
            const float cosine = cos(angle);
            vnext_store_prepared_value(
                query,
                page_table,
                params,
                is_query,
                token,
                absolute_position,
                head,
                low,
                x0 * cosine - x1 * sine);
            vnext_store_prepared_value(
                query,
                page_table,
                params,
                is_query,
                token,
                absolute_position,
                head,
                high,
                x1 * cosine + x0 * sine);
        }
    } else {
        for (uint pair = lane; pair < half_rope; pair += VNEXT_SIMD_WIDTH) {
            const uint low = pair;
            const uint high = pair + half_rope;
            const float x0 =
                float(source[low]) * norm_scale * float(weight[low]);
            const float x1 =
                float(source[high]) * norm_scale * float(weight[high]);
            const float exponent = -(2.0f * float(pair)) / float(params.rope_dim);
            const float angle =
                float(absolute_position) * powr(params.rope_theta, exponent);
            const float sine = sin(angle);
            const float cosine = cos(angle);
            vnext_store_prepared_value(
                query,
                page_table,
                params,
                is_query,
                token,
                absolute_position,
                head,
                low,
                x0 * cosine - x1 * sine);
            vnext_store_prepared_value(
                query,
                page_table,
                params,
                is_query,
                token,
                absolute_position,
                head,
                high,
                x1 * cosine + x0 * sine);
        }
    }

    for (uint dim = params.rope_dim + lane; dim < params.head_dim;
         dim += VNEXT_SIMD_WIDTH) {
        const float value =
            float(source[dim]) * norm_scale * float(weight[dim]);
        vnext_store_prepared_value(
            query,
            page_table,
            params,
            is_query,
            token,
            absolute_position,
            head,
            dim,
            value);
    }
}

kernel void vnext_causal_attention_f16(
    const device half *query [[buffer(0)]],
    const device half *query_raw [[buffer(1)]],
    device half *output [[buffer(2)]],
    device VNextKvPageTable& page_table [[buffer(3)]],
    constant VNextCausalAttentionParams& params [[buffer(4)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint token = group.x;
    const uint query_head = group.y;
    if (token >= params.tokens || query_head >= params.query_heads ||
        lane >= VNEXT_SIMD_WIDTH) {
        return;
    }

    const uint kv_head =
        query_head / (params.query_heads / params.key_value_heads);
    const uint absolute_position = params.position_start + token;
    float query_values[VNEXT_MAX_HEAD_CHUNKS];
    float accumulated[VNEXT_MAX_HEAD_CHUNKS];

    for (uint chunk = 0; chunk < VNEXT_MAX_HEAD_CHUNKS; ++chunk) {
        const uint dim = lane + chunk * VNEXT_SIMD_WIDTH;
        query_values[chunk] =
            dim < params.head_dim
                ? float(query[((ulong)token * (ulong)params.query_heads +
                               (ulong)query_head) *
                                  (ulong)params.head_dim +
                              (ulong)dim])
                : 0.0f;
        accumulated[chunk] = 0.0f;
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    const float attention_scale = rsqrt(float(params.head_dim));
    for (uint key_position = simdgroup; key_position <= absolute_position;
         key_position += params.attention_simdgroups) {
        float partial_dot = 0.0f;
        for (uint chunk = 0; chunk < VNEXT_MAX_HEAD_CHUNKS; ++chunk) {
            const uint dim = lane + chunk * VNEXT_SIMD_WIDTH;
            if (dim < params.head_dim) {
                partial_dot += query_values[chunk] *
                               vnext_load_kv(
                                   page_table,
                                   params,
                                   key_position,
                                   0,
                                   kv_head,
                                   dim);
            }
        }
        const float score = simd_sum(partial_dot) * attention_scale;
        const float next_max = max(running_max, score);
        const float previous_scale =
            isinf(running_max) ? 0.0f : exp(running_max - next_max);
        const float value_scale = exp(score - next_max);
        running_sum = running_sum * previous_scale + value_scale;
        for (uint chunk = 0; chunk < VNEXT_MAX_HEAD_CHUNKS; ++chunk) {
            const uint dim = lane + chunk * VNEXT_SIMD_WIDTH;
            if (dim < params.head_dim) {
                const float value = vnext_load_kv(
                    page_table,
                    params,
                    key_position,
                    1,
                    kv_head,
                    dim);
                accumulated[chunk] = accumulated[chunk] * previous_scale +
                                     value * value_scale;
            }
        }
        running_max = next_max;
    }

    threadgroup float *partial_outputs = shared;
    threadgroup float *partial_maxima =
        partial_outputs + params.attention_simdgroups * params.head_dim;
    threadgroup float *partial_sums =
        partial_maxima + params.attention_simdgroups;
    threadgroup float *partial_scales =
        partial_sums + params.attention_simdgroups;
    for (uint chunk = 0; chunk < VNEXT_MAX_HEAD_CHUNKS; ++chunk) {
        const uint dim = lane + chunk * VNEXT_SIMD_WIDTH;
        if (dim < params.head_dim) {
            partial_outputs[simdgroup * params.head_dim + dim] =
                accumulated[chunk];
        }
    }
    if (lane == 0) {
        partial_maxima[simdgroup] = running_max;
        partial_sums[simdgroup] = running_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        const bool active = lane < params.attention_simdgroups;
        const float local_maximum =
            active ? partial_maxima[lane] : -INFINITY;
        const float global_maximum = simd_max(local_maximum);
        const float scale =
            active && !isinf(local_maximum)
                ? exp(local_maximum - global_maximum)
                : 0.0f;
        const float scaled_sum =
            active ? partial_sums[lane] * scale : 0.0f;
        const float global_sum = simd_sum(scaled_sum);
        if (active) {
            partial_scales[lane] = scale;
        }
        if (lane == 0) {
            partial_sums[0] = global_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup != 0) {
        return;
    }
    const float inverse_sum = 1.0f / partial_sums[0];
    for (uint chunk = 0; chunk < VNEXT_MAX_HEAD_CHUNKS; ++chunk) {
        const uint dim = lane + chunk * VNEXT_SIMD_WIDTH;
        if (dim < params.head_dim) {
            float value = 0.0f;
            for (uint partial = 0; partial < params.attention_simdgroups;
                 ++partial) {
                value += partial_outputs
                             [partial * params.head_dim + dim] *
                         partial_scales[partial];
            }
            value *= inverse_sum;
            if (params.output_gate != 0u) {
                const ulong gate_index =
                    (ulong)token * (ulong)params.query_projection_stride +
                    (ulong)query_head * (2ul * (ulong)params.head_dim) +
                    (ulong)params.head_dim + (ulong)dim;
                const float gate = float(query_raw[gate_index]);
                value *= 1.0f / (1.0f + exp(-gate));
            }
            output[((ulong)token * (ulong)params.query_heads +
                    (ulong)query_head) *
                       (ulong)params.head_dim +
                   (ulong)dim] = half(value);
        }
    }
}
