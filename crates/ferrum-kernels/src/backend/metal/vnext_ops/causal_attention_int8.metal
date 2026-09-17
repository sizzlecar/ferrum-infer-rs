#include <metal_stdlib>
using namespace metal;

#define VNEXT_MAX_KV_PAGES 16384
#define VNEXT_SIMD_WIDTH 32
#define VNEXT_MAX_HEAD_CHUNKS 8
#define VNEXT_SCALE_PAGE_ELEMENTS 16384

// Both arrays contain retained fixed 64 KiB pages; their token capacities differ.
struct VNextKvPageTable {
    array<device char *, VNEXT_MAX_KV_PAGES> pages [[id(0)]];
    array<device float *, VNEXT_MAX_KV_PAGES> scales [[id(VNEXT_MAX_KV_PAGES)]];
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


inline ulong vnext_int8_head_index(
    uint token, uint kind, uint head,
    constant VNextCausalAttentionParams& params) {
    return ((ulong)token * 2ul + kind) * params.key_value_heads + head;
}

inline device float *vnext_int8_scale(
    device VNextKvPageTable& table, ulong head_index) {
    return table.scales[head_index / VNEXT_SCALE_PAGE_ELEMENTS] +
        head_index % VNEXT_SCALE_PAGE_ELEMENTS;
}

inline device char *vnext_int8_element(
    device VNextKvPageTable& table,
    constant VNextCausalAttentionParams& params,
    ulong element) {
    return table.pages[element / params.page_elements] + element % params.page_elements;
}

inline float vnext_load_int8_kv(
    device VNextKvPageTable& table,
    constant VNextCausalAttentionParams& params,
    uint token, uint kind, uint head, uint dim) {
    const ulong head_index = vnext_int8_head_index(token, kind, head, params);
    const float scale = *vnext_int8_scale(table, head_index);
    const char value = *vnext_int8_element(table, params, head_index * params.head_dim + dim);
    return float(value) * scale;
}

// Quantize exactly the F16 values that the unquantized prepare would store.
// Q keeps its previous norm/RoPE/F16 boundary; the output gate is untouched.
kernel void vnext_causal_prepare_int8(
    const device half *query_raw [[buffer(0)]],
    const device half *key_raw [[buffer(1)]],
    const device half *value_raw [[buffer(2)]],
    const device half *query_norm_weight [[buffer(3)]],
    const device half *key_norm_weight [[buffer(4)]],
    device half *query [[buffer(5)]],
    device VNextKvPageTable& page_table [[buffer(6)]],
    constant VNextCausalAttentionParams& params [[buffer(7)]],
    device atomic_uint *error_flag [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint token = group.x;
    const uint combined_head = group.y;
    if (token >= params.tokens ||
        combined_head >= params.query_heads + 2u * params.key_value_heads ||
        lane >= VNEXT_SIMD_WIDTH) {
        return;
    }
    const bool is_query = combined_head < params.query_heads;
    const bool is_key = !is_query && combined_head < params.query_heads + params.key_value_heads;
    const uint head = is_query ? combined_head :
        combined_head - params.query_heads - (is_key ? 0u : params.key_value_heads);
    const uint position = params.position_start + token;
    const device half *source = is_query ?
        query_raw + (ulong)token * params.query_projection_stride + (ulong)head * params.query_head_stride :
        (is_key ? key_raw : value_raw) + (ulong)token * params.kv_projection_stride +
            (ulong)head * params.head_dim;
    const device half *weight = is_query ? query_norm_weight : key_norm_weight;
    float norm_scale = 1.0f;
    if (is_query || is_key) {
        float squares = 0.0f;
        for (uint dim = lane; dim < params.head_dim; dim += VNEXT_SIMD_WIDTH) {
            const float value = float(source[dim]);
            squares += value * value;
        }
        norm_scale = rsqrt(simd_sum(squares) / float(params.head_dim) + params.epsilon);
    }
    float prepared[VNEXT_MAX_HEAD_CHUNKS];
    float maximum = 0.0f;
    bool invalid = false;
    for (uint chunk = 0; chunk < VNEXT_MAX_HEAD_CHUNKS; ++chunk) {
        const uint dim = lane + chunk * VNEXT_SIMD_WIDTH;
        float value = 0.0f;
        if (dim < params.head_dim) {
            value = float(source[dim]);
            if (is_query || is_key) {
                value = value * norm_scale * float(weight[dim]);
                if (dim < params.rope_dim) {
                    const uint half_rope = params.rope_dim / 2u;
                    const uint low = params.rope_interleaved ? (dim & ~1u) : (dim % half_rope);
                    const uint high = low + (params.rope_interleaved ? 1u : half_rope);
                    const uint pair = params.rope_interleaved ? low / 2u : low;
                    const float x0 = float(source[low]) * norm_scale * float(weight[low]);
                    const float x1 = float(source[high]) * norm_scale * float(weight[high]);
                    const float exponent = -(2.0f * float(pair)) / float(params.rope_dim);
                    const float angle = float(position) * powr(params.rope_theta, exponent);
                    const float sine = sin(angle);
                    const float cosine = cos(angle);
                    value = dim == low ? x0 * cosine - x1 * sine : x1 * cosine + x0 * sine;
                }
            }
            value = float(half(value));
            invalid = invalid || !isfinite(value);
            maximum = max(maximum, abs(value));
            if (is_query) {
                query[((ulong)token * params.query_heads + head) * params.head_dim + dim] = half(value);
            }
        }
        prepared[chunk] = value;
    }
    if (is_query) {
        if (simd_any(invalid) && lane == 0) {
            atomic_fetch_or_explicit(error_flag, 1u, memory_order_relaxed);
        }
        return;
    }
    const bool invalid_head = simd_any(invalid);
    maximum = simd_max(maximum);
    const float scale = invalid_head || maximum == 0.0f ? 1.0f : maximum / 127.0f;
    const ulong head_index = vnext_int8_head_index(position, is_key ? 0u : 1u, head, params);
    if (lane == 0) {
        *vnext_int8_scale(page_table, head_index) = scale;
        if (invalid_head) {
            atomic_fetch_or_explicit(error_flag, 1u, memory_order_relaxed);
        }
    }
    for (uint chunk = 0; chunk < VNEXT_MAX_HEAD_CHUNKS; ++chunk) {
        const uint dim = lane + chunk * VNEXT_SIMD_WIDTH;
        if (dim < params.head_dim) {
            // rint is nearest with ties to even. Never cast a non-finite float.
            const float encoded = invalid_head ? 0.0f :
                clamp(rint(prepared[chunk] / scale), -127.0f, 127.0f);
            *vnext_int8_element(page_table, params, head_index * params.head_dim + dim) = char(encoded);
        }
    }
}

kernel void vnext_causal_attention_int8(
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
                               vnext_load_int8_kv(
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
                const float value = vnext_load_int8_kv(
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
