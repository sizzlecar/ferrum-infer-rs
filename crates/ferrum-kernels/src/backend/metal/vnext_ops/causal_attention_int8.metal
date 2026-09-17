#include <metal_stdlib>
#include <metal_simdgroup_matrix>
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

// Decode resolves each contiguous quantized row once and reuses both scales.
kernel void vnext_causal_attention_decode_direct_int8(
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
    const uint token_elements =
        2u * params.key_value_heads * params.head_dim;
    const uint tokens_per_page = params.page_elements / token_elements;
    const float attention_scale = rsqrt(float(params.head_dim));
    float query_values[VNEXT_MAX_HEAD_CHUNKS];
    float accumulated[VNEXT_MAX_HEAD_CHUNKS];

    for (uint chunk = 0; chunk < VNEXT_MAX_HEAD_CHUNKS; ++chunk) {
        const uint dim = lane + chunk * VNEXT_SIMD_WIDTH;
        query_values[chunk] =
            dim < params.head_dim
                ? float(query[((ulong)token * (ulong)params.query_heads +
                               (ulong)query_head) *
                                  (ulong)params.head_dim +
                              (ulong)dim]) *
                      attention_scale
                : 0.0f;
        accumulated[chunk] = 0.0f;
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    for (uint key_position = simdgroup; key_position <= absolute_position;
         key_position += params.attention_simdgroups) {
        const uint page = key_position / tokens_per_page;
        if (page >= params.page_count) {
            continue;
        }
        const uint token_in_page =
            key_position - page * tokens_per_page;
        device char *page_base = page_table.pages[page];
        const device char *key_row =
            page_base +
            (token_in_page * 2u * params.key_value_heads + kv_head) *
                params.head_dim;
        const device char *value_row =
            page_base +
            (token_in_page * 2u * params.key_value_heads +
             params.key_value_heads + kv_head) *
                params.head_dim;

        const float key_scale = *vnext_int8_scale(page_table,
            vnext_int8_head_index(key_position, 0, kv_head, params));
        const float value_scale_factor = *vnext_int8_scale(page_table,
            vnext_int8_head_index(key_position, 1, kv_head, params));
        float partial_dot = 0.0f;
        for (uint chunk = 0; chunk < VNEXT_MAX_HEAD_CHUNKS; ++chunk) {
            const uint dim = lane + chunk * VNEXT_SIMD_WIDTH;
            if (dim < params.head_dim) {
                partial_dot += query_values[chunk] * (float(key_row[dim]) * key_scale);
            }
        }
        const float score = simd_sum(partial_dot);
        const float next_max = max(running_max, score);
        const float previous_scale =
            isinf(running_max) ? 0.0f : exp(running_max - next_max);
        const float value_scale = exp(score - next_max);
        running_sum = running_sum * previous_scale + value_scale;
        for (uint chunk = 0; chunk < VNEXT_MAX_HEAD_CHUNKS; ++chunk) {
            const uint dim = lane + chunk * VNEXT_SIMD_WIDTH;
            if (dim < params.head_dim) {
                accumulated[chunk] = accumulated[chunk] * previous_scale +
                                     (float(value_row[dim]) * value_scale_factor) * value_scale;
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
    if (lane == 0u) {
        partial_maxima[simdgroup] = running_max;
        partial_sums[simdgroup] = running_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
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
        if (lane == 0u) {
            partial_sums[0] = global_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup != 0u) {
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


inline float vnext_load_int8_kv_half(
    device VNextKvPageTable& table,
    constant VNextCausalAttentionParams& params,
    uint token, uint kind, uint head, uint dim) {
    return float(half(vnext_load_int8_kv(table, params, token, kind, head, dim)));
}

// SIMDgroup prefill stages at most 32 K or V rows. It never expands the history.
#define VNEXT_PREFILL_QUERY_TILE 8
#define VNEXT_PREFILL_KEY_TILE 32
#define VNEXT_PREFILL_SIMDGROUPS 4
template <uint HEAD_TILES, uint SIMD_GROUP_COUNT, uint KEY_TILE>
inline void vnext_causal_attention_prefill_tiled_int8_body(
    const device half *query,
    const device half *query_raw,
    device half *output,
    device VNextKvPageTable& page_table,
    constant VNextCausalAttentionParams& params,
    threadgroup half *shared_half,
    threadgroup float *shared_float,
    uint query_start,
    uint query_head_start,
    uint kv_head,
    uint thread_index,
    uint simdgroup,
    uint lane) {
    constexpr uint query_rows = HEAD_TILES * VNEXT_PREFILL_QUERY_TILE;
    constexpr uint rows_per_simdgroup = query_rows / SIMD_GROUP_COUNT;
    const uint query_elements = query_rows * params.head_dim;
    threadgroup half *query_tile = shared_half;
    threadgroup half *probabilities =
        query_tile + query_elements;
    // K and V reuse this bounded tile in separate phases.
    threadgroup half *kv_tile = probabilities + query_rows * KEY_TILE;
    threadgroup float *accumulated_output = shared_float;
    threadgroup float *scores =
        accumulated_output + query_elements;

    for (uint element = thread_index;
         element < query_elements;
         element += SIMD_GROUP_COUNT * VNEXT_SIMD_WIDTH) {
        const uint logical_row = element / params.head_dim;
        const uint dim = element - logical_row * params.head_dim;
        const uint head_slot = logical_row / VNEXT_PREFILL_QUERY_TILE;
        const uint query_row =
            logical_row - head_slot * VNEXT_PREFILL_QUERY_TILE;
        const uint token = query_start + query_row;
        const uint query_head = query_head_start + head_slot;
        query_tile[element] =
            token < params.tokens
                ? query[((ulong)token * (ulong)params.query_heads +
                         (ulong)query_head) *
                            (ulong)params.head_dim +
                        (ulong)dim]
                : half(0.0h);
        accumulated_output[element] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint query_end = min(
        query_start + VNEXT_PREFILL_QUERY_TILE,
        params.tokens);
    const uint maximum_key_end = params.position_start + query_end;
    const uint key_value_row_stride =
        params.head_dim;
    const float attention_scale = rsqrt(float(params.head_dim));

    float running_max[rows_per_simdgroup];
    float running_sum[rows_per_simdgroup];
    for (uint row = 0; row < rows_per_simdgroup; ++row) {
        running_max[row] = -INFINITY;
        running_sum[row] = 0.0f;
    }

    for (uint key_start = 0; key_start < maximum_key_end;
         key_start += KEY_TILE) {
        for (uint element = thread_index; element < KEY_TILE * params.head_dim;
             element += SIMD_GROUP_COUNT * VNEXT_SIMD_WIDTH) {
            const uint position = key_start + element / params.head_dim;
            const uint dim = element % params.head_dim;
            kv_tile[element] = position < maximum_key_end ?
                half(vnext_load_int8_kv(page_table, params, position, 0, kv_head, dim)) : half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simdgroup < KEY_TILE / 8u) {
            simdgroup_float8x8 score_matrices[2];
            for (uint head_slot = 0; head_slot < HEAD_TILES; ++head_slot) {
                score_matrices[head_slot] =
                    make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
            const uint key_block_start = key_start + simdgroup * 8u;
            const uint key_block_rows =
                key_block_start < maximum_key_end
                    ? min(8u, maximum_key_end - key_block_start)
                    : 0u;
            if (key_block_rows == 8u) {
                threadgroup half *key_block =
                    kv_tile + (key_block_start - key_start) * params.head_dim;
                if (key_block != nullptr) {
                    for (uint dim = 0; dim < params.head_dim; dim += 8) {
                        simdgroup_half8x8 key_matrix;
                        simdgroup_load(
                            key_matrix,
                            key_block + dim,
                            key_value_row_stride,
                            ulong2(0, 0),
                            true);
                        for (uint head_slot = 0; head_slot < HEAD_TILES;
                             ++head_slot) {
                            simdgroup_half8x8 query_matrix;
                            simdgroup_load(
                                query_matrix,
                                query_tile +
                                    head_slot * VNEXT_PREFILL_QUERY_TILE *
                                        params.head_dim +
                                    dim,
                                params.head_dim,
                                ulong2(0, 0),
                                false);
                            simdgroup_multiply_accumulate(
                                score_matrices[head_slot],
                                query_matrix,
                                key_matrix,
                                score_matrices[head_slot]);
                        }
                    }
                }
                for (uint head_slot = 0; head_slot < HEAD_TILES;
                     ++head_slot) {
                    simdgroup_store(
                        score_matrices[head_slot],
                        scores +
                            head_slot * VNEXT_PREFILL_QUERY_TILE *
                                KEY_TILE +
                            simdgroup * 8,
                        KEY_TILE,
                        ulong2(0, 0),
                        false);
                }
            } else {
                // State pages intentionally use StateInitialization::None.
                // Avoid speculative matrix reads from unused final-page rows:
                // NaN slack would otherwise survive the later P@V product.
                if (lane == 0u) {
                    for (uint logical_row = 0; logical_row < query_rows;
                         ++logical_row) {
                        for (uint key_row = 0; key_row < 8u; ++key_row) {
                            scores[logical_row * KEY_TILE +
                                   simdgroup * 8u + key_row] = 0.0f;
                        }
                    }
                }
                for (uint query_row = 0;
                     query_row < VNEXT_PREFILL_QUERY_TILE;
                     ++query_row) {
                    for (uint key_row = 0; key_row < key_block_rows;
                         ++key_row) {
                        float partial_dots[2] = {0.0f, 0.0f};
                        for (uint dim = lane; dim < params.head_dim;
                             dim += VNEXT_SIMD_WIDTH) {
                            const float key_value = vnext_load_int8_kv_half(
                                page_table,
                                params,
                                key_block_start + key_row,
                                0,
                                kv_head,
                                dim);
                            for (uint head_slot = 0; head_slot < HEAD_TILES;
                                 ++head_slot) {
                                const uint logical_row =
                                    head_slot * VNEXT_PREFILL_QUERY_TILE +
                                    query_row;
                                partial_dots[head_slot] +=
                                    float(query_tile
                                              [logical_row * params.head_dim +
                                               dim]) *
                                    key_value;
                            }
                        }
                        for (uint head_slot = 0; head_slot < HEAD_TILES;
                             ++head_slot) {
                            const float dot =
                                simd_sum(partial_dots[head_slot]);
                            if (lane == 0u) {
                                const uint logical_row =
                                    head_slot * VNEXT_PREFILL_QUERY_TILE +
                                    query_row;
                                scores[logical_row * KEY_TILE +
                                       simdgroup * 8u + key_row] = dot;
                            }
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint row_slot = 0; row_slot < rows_per_simdgroup;
             ++row_slot) {
            const uint logical_row =
                row_slot * SIMD_GROUP_COUNT + simdgroup;
            const uint query_row = logical_row % VNEXT_PREFILL_QUERY_TILE;
            const uint token = query_start + query_row;
            constexpr uint columns_per_lane =
                KEY_TILE / VNEXT_SIMD_WIDTH;
            float tile_scores[columns_per_lane];
            bool tile_keeps[columns_per_lane];
            float tile_maximum = -INFINITY;
            for (uint column_slot = 0; column_slot < columns_per_lane;
                 ++column_slot) {
                const uint key_column =
                    lane + column_slot * VNEXT_SIMD_WIDTH;
                const uint key_position = key_start + key_column;
                const bool keep =
                    token < params.tokens &&
                    key_position <= params.position_start + token &&
                    key_position < maximum_key_end;
                const float score =
                    keep
                        ? scores[logical_row * KEY_TILE + key_column] *
                              attention_scale
                        : -INFINITY;
                tile_scores[column_slot] = score;
                tile_keeps[column_slot] = keep;
                tile_maximum =
                    max(tile_maximum, simd_max(score));
            }
            const float next_maximum =
                max(running_max[row_slot], tile_maximum);
            const float previous_scale =
                isinf(running_max[row_slot])
                    ? 0.0f
                    : exp(running_max[row_slot] - next_maximum);
            float tile_sum = 0.0f;
            for (uint column_slot = 0; column_slot < columns_per_lane;
                 ++column_slot) {
                const float probability =
                    tile_keeps[column_slot]
                        ? exp(tile_scores[column_slot] - next_maximum)
                        : 0.0f;
                tile_sum += simd_sum(probability);
                const uint key_column =
                    lane + column_slot * VNEXT_SIMD_WIDTH;
                probabilities[logical_row * KEY_TILE + key_column] =
                    half(probability);
            }
            running_sum[row_slot] =
                running_sum[row_slot] * previous_scale +
                tile_sum;
            running_max[row_slot] = next_maximum;
            for (uint dim = lane; dim < params.head_dim;
                 dim += VNEXT_SIMD_WIDTH) {
                accumulated_output
                    [logical_row * params.head_dim + dim] *=
                    previous_scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint element = thread_index; element < KEY_TILE * params.head_dim;
             element += SIMD_GROUP_COUNT * VNEXT_SIMD_WIDTH) {
            const uint position = key_start + element / params.head_dim;
            const uint dim = element % params.head_dim;
            kv_tile[element] = position < maximum_key_end ?
                half(vnext_load_int8_kv(page_table, params, position, 1, kv_head, dim)) : half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 output_matrices[8];
        const uint output_tiles_per_head =
            params.head_dim / (SIMD_GROUP_COUNT * 8);
        for (uint head_slot = 0; head_slot < HEAD_TILES; ++head_slot) {
            for (uint output_tile = 0;
                 output_tile < output_tiles_per_head;
                 ++output_tile) {
                const uint matrix_index =
                    head_slot * output_tiles_per_head + output_tile;
                const uint output_column =
                    simdgroup * 8 +
                    output_tile * SIMD_GROUP_COUNT * 8;
                simdgroup_load(
                    output_matrices[matrix_index],
                    accumulated_output +
                        head_slot * VNEXT_PREFILL_QUERY_TILE *
                            params.head_dim +
                        output_column,
                    params.head_dim,
                    ulong2(0, 0),
                    false);
            }
        }
        // Stage two adjacent key tiles and two independent output tiles at a
        // time. Each output accumulator still observes key tiles in the same
        // order, while the independent matrix loads and multiplies give the
        // GPU enough work to overlap their latency. Resolve each value tile
        // through the page table independently: adjacent eight-token tiles
        // are not guaranteed to occupy the same physical page.
        for (uint key_tile = 0;
             key_tile < KEY_TILE / 8;
             key_tile += 2) {
            simdgroup_half8x8 probability_matrices[4];
            for (uint staged_key = 0; staged_key < 2; ++staged_key) {
                for (uint head_slot = 0; head_slot < HEAD_TILES;
                     ++head_slot) {
                    const uint probability_index =
                        staged_key * HEAD_TILES + head_slot;
                    simdgroup_load(
                        probability_matrices[probability_index],
                        probabilities +
                            head_slot * VNEXT_PREFILL_QUERY_TILE *
                                KEY_TILE +
                            (key_tile + staged_key) * 8,
                        KEY_TILE,
                        ulong2(0, 0),
                        false);
                }
            }

            threadgroup half *value_blocks[2] = {nullptr, nullptr};
            for (uint staged_key = 0; staged_key < 2; ++staged_key) {
                const uint value_block_start =
                    key_start + (key_tile + staged_key) * 8u;
                const uint value_block_rows =
                    value_block_start < maximum_key_end
                        ? min(8u, maximum_key_end - value_block_start)
                        : 0u;
                if (value_block_rows == 8u) {
                    value_blocks[staged_key] =
                        kv_tile + (value_block_start - key_start) * params.head_dim;
                }
            }

            for (uint output_tile = 0;
                 output_tile < output_tiles_per_head;
                 output_tile += 2) {
                const uint output_column_0 =
                    simdgroup * 8 +
                    output_tile * SIMD_GROUP_COUNT * 8;
                const uint output_column_1 =
                    output_column_0 + SIMD_GROUP_COUNT * 8;
                if (value_blocks[0] != nullptr &&
                    value_blocks[1] != nullptr) {
                    simdgroup_half8x8 value_matrices[4];
                    simdgroup_load(
                        value_matrices[0],
                        value_blocks[0] + output_column_0,
                        key_value_row_stride,
                        ulong2(0, 0),
                        false);
                    simdgroup_load(
                        value_matrices[1],
                        value_blocks[0] + output_column_1,
                        key_value_row_stride,
                        ulong2(0, 0),
                        false);
                    simdgroup_load(
                        value_matrices[2],
                        value_blocks[1] + output_column_0,
                        key_value_row_stride,
                        ulong2(0, 0),
                        false);
                    simdgroup_load(
                        value_matrices[3],
                        value_blocks[1] + output_column_1,
                        key_value_row_stride,
                        ulong2(0, 0),
                        false);
                    for (uint head_slot = 0; head_slot < HEAD_TILES;
                         ++head_slot) {
                        const uint matrix_index =
                            head_slot * output_tiles_per_head + output_tile;
                        simdgroup_multiply_accumulate(
                            output_matrices[matrix_index],
                            probability_matrices[head_slot],
                            value_matrices[0],
                            output_matrices[matrix_index]);
                        simdgroup_multiply_accumulate(
                            output_matrices[matrix_index + 1],
                            probability_matrices[head_slot],
                            value_matrices[1],
                            output_matrices[matrix_index + 1]);
                        simdgroup_multiply_accumulate(
                            output_matrices[matrix_index],
                            probability_matrices[HEAD_TILES + head_slot],
                            value_matrices[2],
                            output_matrices[matrix_index]);
                        simdgroup_multiply_accumulate(
                            output_matrices[matrix_index + 1],
                            probability_matrices[HEAD_TILES + head_slot],
                            value_matrices[3],
                            output_matrices[matrix_index + 1]);
                    }
                } else {
                    for (uint staged_key = 0; staged_key < 2;
                         ++staged_key) {
                        if (value_blocks[staged_key] == nullptr) {
                            continue;
                        }
                        simdgroup_half8x8 value_matrices[2];
                        simdgroup_load(
                            value_matrices[0],
                            value_blocks[staged_key] + output_column_0,
                            key_value_row_stride,
                            ulong2(0, 0),
                            false);
                        simdgroup_load(
                            value_matrices[1],
                            value_blocks[staged_key] + output_column_1,
                            key_value_row_stride,
                            ulong2(0, 0),
                            false);
                        for (uint head_slot = 0; head_slot < HEAD_TILES;
                             ++head_slot) {
                            const uint matrix_index =
                                head_slot * output_tiles_per_head +
                                output_tile;
                            const uint probability_index =
                                staged_key * HEAD_TILES + head_slot;
                            simdgroup_multiply_accumulate(
                                output_matrices[matrix_index],
                                probability_matrices[probability_index],
                                value_matrices[0],
                                output_matrices[matrix_index]);
                            simdgroup_multiply_accumulate(
                                output_matrices[matrix_index + 1],
                                probability_matrices[probability_index],
                                value_matrices[1],
                                output_matrices[matrix_index + 1]);
                        }
                    }
                }
            }
        }
        for (uint head_slot = 0; head_slot < HEAD_TILES; ++head_slot) {
            for (uint output_tile = 0;
                 output_tile < output_tiles_per_head;
                 ++output_tile) {
                const uint matrix_index =
                    head_slot * output_tiles_per_head + output_tile;
                const uint output_column =
                    simdgroup * 8 +
                    output_tile * SIMD_GROUP_COUNT * 8;
                simdgroup_store(
                    output_matrices[matrix_index],
                    accumulated_output +
                        head_slot * VNEXT_PREFILL_QUERY_TILE *
                            params.head_dim +
                        output_column,
                    params.head_dim,
                    ulong2(0, 0),
                    false);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Finish the one partial eight-token block without reading page slack.
        // Every thread owns disjoint output elements, so no atomic is needed.
        const uint tail_rows = maximum_key_end % 8u;
        const uint tail_start = maximum_key_end - tail_rows;
        if (tail_rows != 0u && tail_start >= key_start &&
            tail_start < key_start + KEY_TILE) {
            const uint probability_column = tail_start - key_start;
            for (uint element = thread_index;
                 element < VNEXT_PREFILL_QUERY_TILE * params.head_dim;
                 element += SIMD_GROUP_COUNT * VNEXT_SIMD_WIDTH) {
                const uint query_row = element / params.head_dim;
                const uint dim = element - query_row * params.head_dim;
                float tail_values[2] = {0.0f, 0.0f};
                for (uint key_row = 0; key_row < tail_rows; ++key_row) {
                    const float value = vnext_load_int8_kv_half(
                        page_table,
                        params,
                        tail_start + key_row,
                        1,
                        kv_head,
                        dim);
                    for (uint head_slot = 0; head_slot < HEAD_TILES;
                         ++head_slot) {
                        const uint logical_row =
                            head_slot * VNEXT_PREFILL_QUERY_TILE + query_row;
                        tail_values[head_slot] +=
                            float(probabilities
                                      [logical_row *
                                           KEY_TILE +
                                       probability_column + key_row]) *
                            value;
                    }
                }
                for (uint head_slot = 0; head_slot < HEAD_TILES;
                     ++head_slot) {
                    const uint logical_element =
                        head_slot * VNEXT_PREFILL_QUERY_TILE *
                            params.head_dim +
                        element;
                    accumulated_output[logical_element] +=
                        tail_values[head_slot];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint row_slot = 0; row_slot < rows_per_simdgroup; ++row_slot) {
        const uint logical_row =
            row_slot * SIMD_GROUP_COUNT + simdgroup;
        const uint head_slot = logical_row / VNEXT_PREFILL_QUERY_TILE;
        const uint query_row = logical_row % VNEXT_PREFILL_QUERY_TILE;
        const uint query_head = query_head_start + head_slot;
        const uint token = query_start + query_row;
        if (token >= params.tokens) {
            continue;
        }
        const float inverse_sum = 1.0f / running_sum[row_slot];
        for (uint dim = lane; dim < params.head_dim;
             dim += VNEXT_SIMD_WIDTH) {
            float value =
                accumulated_output
                    [logical_row * params.head_dim + dim] *
                inverse_sum;
            if (params.output_gate != 0u) {
                const ulong gate_index =
                    (ulong)token * (ulong)params.query_projection_stride +
                    (ulong)query_head *
                        (2ul * (ulong)params.head_dim) +
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

kernel void vnext_causal_attention_prefill_tiled_int8(
    const device half *query [[buffer(0)]],
    const device half *query_raw [[buffer(1)]],
    device half *output [[buffer(2)]],
    device VNextKvPageTable& page_table [[buffer(3)]],
    constant VNextCausalAttentionParams& params [[buffer(4)]],
    threadgroup half *shared_half [[threadgroup(0)]],
    threadgroup float *shared_float [[threadgroup(1)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint query_start = group.x * VNEXT_PREFILL_QUERY_TILE;
    const uint query_head = group.y;
    if (query_start >= params.tokens || query_head >= params.query_heads ||
        params.key_value_heads == 0u) {
        return;
    }
    const uint query_heads_per_kv_head =
        params.query_heads / params.key_value_heads;
    const uint kv_head = query_head / query_heads_per_kv_head;
    vnext_causal_attention_prefill_tiled_int8_body<
        1,
        VNEXT_PREFILL_SIMDGROUPS,
        VNEXT_PREFILL_KEY_TILE>(
        query,
        query_raw,
        output,
        page_table,
        params,
        shared_half,
        shared_float,
        query_start,
        query_head,
        kv_head,
        thread_index,
        simdgroup,
        lane);
}

// The GQA reader has D=256 and the provider binds fixed 64 KiB payload pages.
// Every four-dimensional vector is four-byte aligned and stays within its
// head/page, even when a complete token row spans two physical pages. Resolve
// payload and scale independently; a scale page covers a different frontier.
inline half4 vnext_load_int8_gqa_kv4(
    device VNextKvPageTable& page_table,
    constant VNextCausalAttentionParams& params,
    uint token, uint kind, uint head, uint dim) {
    const ulong head_index = vnext_int8_head_index(token, kind, head, params);
    const ulong element = head_index * 256ul + dim;
    const device char *payload = page_table.pages[element >> 16u] + (element & 65535ul);
    const char4 quantized = *reinterpret_cast<const device char4 *>(payload);
    const float scale = *vnext_int8_scale(page_table, head_index);
    return half4(float4(quantized) * scale);
}

// Two query heads share each dequantized K/V slab. Keeping all Q and output
// rows resident but only 64 K/V dimensions at a time uses 31 KiB, including
// scores and probabilities, on devices with a 32 KiB threadgroup limit.
// The existing single-head kernel remains the fallback and decode is separate.
kernel void vnext_causal_attention_prefill_gqa_tiled_int8(
    const device half *query [[buffer(0)]],
    const device half *query_raw [[buffer(1)]],
    device half *output [[buffer(2)]],
    device VNextKvPageTable& page_table [[buffer(3)]],
    constant VNextCausalAttentionParams& params [[buffer(4)]],
    threadgroup half *shared_half [[threadgroup(0)]],
    threadgroup float *shared_float [[threadgroup(1)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint HEADS = 2;
    constexpr uint QUERY_TILE = 8;
    constexpr uint KEY_TILE = 32;
    constexpr uint DIM = 256;
    constexpr uint DIM_SLAB = 64;
    constexpr uint SIMD_GROUPS = 8;
    constexpr uint QUERY_ROWS = HEADS * QUERY_TILE;
    const uint query_start = group.x * QUERY_TILE;
    const uint query_head_start = group.y * HEADS;
    // All conditions are uniform across the threadgroup, before any barrier.
    if (query_start >= params.tokens ||
        query_head_start + HEADS > params.query_heads ||
        params.head_dim != DIM || params.key_value_heads == 0u) {
        return;
    }
    const uint heads_per_kv = params.query_heads / params.key_value_heads;
    if (heads_per_kv < HEADS || heads_per_kv % HEADS != 0u ||
        params.query_heads % params.key_value_heads != 0u) {
        return;
    }
    const uint kv_head = query_head_start / heads_per_kv;
    const uint maximum_key_end = params.position_start +
        min(query_start + QUERY_TILE, params.tokens);
    threadgroup half *query_tile = shared_half;
    threadgroup half *probabilities = query_tile + QUERY_ROWS * DIM;
    threadgroup half *kv_slab = probabilities + QUERY_ROWS * KEY_TILE;
    threadgroup float *accumulated_output = shared_float;
    threadgroup float *scores = accumulated_output + QUERY_ROWS * DIM;

    for (uint element = thread_index; element < QUERY_ROWS * DIM;
         element += SIMD_GROUPS * VNEXT_SIMD_WIDTH) {
        const uint row = element / DIM;
        const uint dim = element % DIM;
        const uint token = query_start + row % QUERY_TILE;
        const uint head = query_head_start + row / QUERY_TILE;
        query_tile[element] = token < params.tokens
            ? query[((ulong)token * params.query_heads + head) * DIM + dim]
            : half(0.0h);
        accumulated_output[element] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float running_max[HEADS] = {-INFINITY, -INFINITY};
    float running_sum[HEADS] = {0.0f, 0.0f};
    const float attention_scale = rsqrt(float(DIM));

    for (uint key_start = 0; key_start < maximum_key_end; key_start += KEY_TILE) {
        // All eight SIMD groups compute: four key blocks for each of the
        // two heads. Both heads consume the same cooperatively gathered slab.
        const uint score_head = simdgroup / (KEY_TILE / 8u);
        const uint score_block = simdgroup % (KEY_TILE / 8u);
        simdgroup_float8x8 score_matrix = make_filled_simdgroup_matrix<float, 8>(0.0f);
        const uint key_block_start = key_start + score_block * 8u;
        const uint key_block_rows = key_block_start < maximum_key_end
            ? min(8u, maximum_key_end - key_block_start) : 0u;
        for (uint dim_start = 0; dim_start < DIM; dim_start += DIM_SLAB) {
#if defined(VNEXT_INT8_GQA_SCALAR_GATHER_REFERENCE)
            // Rust's opt-in device diagnostic compiles the preceding scalar
            // gather unchanged for a paired measurement. Production never
            // defines this macro and has no runtime selector for it.
            for (uint element = thread_index; element < KEY_TILE * DIM_SLAB;
                 element += SIMD_GROUPS * VNEXT_SIMD_WIDTH) {
                const uint position = key_start + element / DIM_SLAB;
                const uint dim = dim_start + element % DIM_SLAB;
                kv_slab[element] = position < maximum_key_end
                    ? half(vnext_load_int8_kv(page_table, params, position, 0, kv_head, dim))
                    : half(0.0h);
            }
#else
            for (uint vector_index = thread_index; vector_index < KEY_TILE * DIM_SLAB / 4u;
                 vector_index += SIMD_GROUPS * VNEXT_SIMD_WIDTH) {
                const uint element = vector_index * 4u;
                const uint position = key_start + element / DIM_SLAB;
                const uint dim = dim_start + element % DIM_SLAB;
                *reinterpret_cast<threadgroup half4 *>(kv_slab + element) =
                    position < maximum_key_end
                        ? vnext_load_int8_gqa_kv4(page_table, params, position, 0, kv_head, dim)
                        : half4(0.0h);
            }
#endif
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (key_block_rows == 8u) {
                for (uint dim = 0; dim < DIM_SLAB; dim += 8u) {
                    simdgroup_half8x8 key_matrix;
                    simdgroup_load(key_matrix, kv_slab + score_block * 8u * DIM_SLAB + dim,
                                   DIM_SLAB, ulong2(0, 0), true);
                    simdgroup_half8x8 query_matrix;
                    simdgroup_load(query_matrix,
                                   query_tile + score_head * QUERY_TILE * DIM + dim_start + dim,
                                   DIM, ulong2(0, 0), false);
                    simdgroup_multiply_accumulate(score_matrix, query_matrix, key_matrix, score_matrix);
                }
            }
            // All matrix consumers finish before the next dimension slab.
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (key_block_rows == 8u) {
            simdgroup_store(score_matrix,
                            scores + score_head * QUERY_TILE * KEY_TILE + score_block * 8u,
                            KEY_TILE, ulong2(0, 0), false);
        } else {
            // Retain the scalar tail's summation order and never read the
            // uninitialized remainder of the last physical state page.
            if (lane == 0u) {
                for (uint row = 0; row < QUERY_TILE; ++row) {
                    for (uint key_row = 0; key_row < 8u; ++key_row) {
                        scores[(score_head * QUERY_TILE + row) * KEY_TILE +
                               score_block * 8u + key_row] = 0.0f;
                    }
                }
            }
            for (uint query_row = 0; query_row < QUERY_TILE; ++query_row) {
                for (uint key_row = 0; key_row < key_block_rows; ++key_row) {
                    float partial_dot = 0.0f;
                    for (uint dim = lane; dim < DIM; dim += VNEXT_SIMD_WIDTH) {
                        const float key_value = vnext_load_int8_kv_half(
                            page_table, params, key_block_start + key_row, 0, kv_head, dim);
                        partial_dot +=
                            float(query_tile[(score_head * QUERY_TILE + query_row) * DIM + dim]) *
                            key_value;
                    }
                    const float dot = simd_sum(partial_dot);
                    if (lane == 0u) {
                        scores[(score_head * QUERY_TILE + query_row) * KEY_TILE +
                               score_block * 8u + key_row] = dot;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each SIMD group owns the same query row in the two heads.
        for (uint head = 0; head < HEADS; ++head) {
            const uint row = head * QUERY_TILE + simdgroup;
            const uint token = query_start + simdgroup;
            const uint key_position = key_start + lane;
            const bool keep = token < params.tokens &&
                key_position <= params.position_start + token && key_position < maximum_key_end;
            const float score = keep ? scores[row * KEY_TILE + lane] * attention_scale : -INFINITY;
            const float next_maximum = max(running_max[head], simd_max(score));
            const float previous_scale = isinf(running_max[head])
                ? 0.0f : exp(running_max[head] - next_maximum);
            const float probability = keep ? exp(score - next_maximum) : 0.0f;
            probabilities[row * KEY_TILE + lane] = half(probability);
            running_sum[head] = running_sum[head] * previous_scale + simd_sum(probability);
            running_max[head] = next_maximum;
            for (uint dim = lane; dim < DIM; dim += VNEXT_SIMD_WIDTH) {
                accumulated_output[row * DIM + dim] *= previous_scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint dim_start = 0; dim_start < DIM; dim_start += DIM_SLAB) {
#if defined(VNEXT_INT8_GQA_SCALAR_GATHER_REFERENCE)
            for (uint element = thread_index; element < KEY_TILE * DIM_SLAB;
                 element += SIMD_GROUPS * VNEXT_SIMD_WIDTH) {
                const uint position = key_start + element / DIM_SLAB;
                const uint dim = dim_start + element % DIM_SLAB;
                kv_slab[element] = position < maximum_key_end
                    ? half(vnext_load_int8_kv(page_table, params, position, 1, kv_head, dim))
                    : half(0.0h);
            }
#else
            for (uint vector_index = thread_index; vector_index < KEY_TILE * DIM_SLAB / 4u;
                 vector_index += SIMD_GROUPS * VNEXT_SIMD_WIDTH) {
                const uint element = vector_index * 4u;
                const uint position = key_start + element / DIM_SLAB;
                const uint dim = dim_start + element % DIM_SLAB;
                *reinterpret_cast<threadgroup half4 *>(kv_slab + element) =
                    position < maximum_key_end
                        ? vnext_load_int8_gqa_kv4(page_table, params, position, 1, kv_head, dim)
                        : half4(0.0h);
            }
#endif
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // One eight-column matrix per head/SIMD group fits this slab.
            simdgroup_float8x8 output_matrices[HEADS];
            const uint output_column = dim_start + simdgroup * 8u;
            for (uint head = 0; head < HEADS; ++head) {
                simdgroup_load(output_matrices[head],
                               accumulated_output + head * QUERY_TILE * DIM + output_column,
                               DIM, ulong2(0, 0), false);
            }
            for (uint key_block = 0; key_block < KEY_TILE / 8u; ++key_block) {
                if (key_start + (key_block + 1u) * 8u > maximum_key_end) {
                    continue;
                }
                simdgroup_half8x8 value_matrix;
                simdgroup_load(value_matrix, kv_slab + key_block * 8u * DIM_SLAB + simdgroup * 8u,
                               DIM_SLAB, ulong2(0, 0), false);
                for (uint head = 0; head < HEADS; ++head) {
                    simdgroup_half8x8 probability_matrix;
                    simdgroup_load(probability_matrix,
                                   probabilities + head * QUERY_TILE * KEY_TILE + key_block * 8u,
                                   KEY_TILE, ulong2(0, 0), false);
                    simdgroup_multiply_accumulate(output_matrices[head], probability_matrix,
                                                 value_matrix, output_matrices[head]);
                }
            }
            for (uint head = 0; head < HEADS; ++head) {
                simdgroup_store(output_matrices[head],
                                accumulated_output + head * QUERY_TILE * DIM + output_column,
                                DIM, ulong2(0, 0), false);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const uint tail_rows = maximum_key_end % 8u;
        const uint tail_start = maximum_key_end - tail_rows;
        if (tail_rows != 0u && tail_start >= key_start && tail_start < key_start + KEY_TILE) {
            const uint probability_column = tail_start - key_start;
            for (uint element = thread_index; element < QUERY_TILE * DIM;
                 element += SIMD_GROUPS * VNEXT_SIMD_WIDTH) {
                const uint query_row = element / DIM;
                const uint dim = element % DIM;
                float tail_values[HEADS] = {0.0f, 0.0f};
                for (uint key_row = 0; key_row < tail_rows; ++key_row) {
                    const float value = vnext_load_int8_kv_half(
                        page_table, params, tail_start + key_row, 1, kv_head, dim);
                    for (uint head = 0; head < HEADS; ++head) {
                        tail_values[head] += float(probabilities[
                            (head * QUERY_TILE + query_row) * KEY_TILE +
                            probability_column + key_row]) * value;
                    }
                }
                for (uint head = 0; head < HEADS; ++head) {
                    accumulated_output[head * QUERY_TILE * DIM + element] += tail_values[head];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint head = 0; head < HEADS; ++head) {
        const uint row = head * QUERY_TILE + simdgroup;
        const uint token = query_start + simdgroup;
        const uint query_head = query_head_start + head;
        if (token >= params.tokens) {
            continue;
        }
        const float inverse_sum = 1.0f / running_sum[head];
        for (uint dim = lane; dim < DIM; dim += VNEXT_SIMD_WIDTH) {
            float value = accumulated_output[row * DIM + dim] * inverse_sum;
            if (params.output_gate != 0u) {
                const ulong gate_index = (ulong)token * params.query_projection_stride +
                    (ulong)query_head * (2ul * DIM) + DIM + dim;
                value *= 1.0f / (1.0f + exp(-float(query_raw[gate_index])));
            }
            output[((ulong)token * params.query_heads + query_head) * DIM + dim] = half(value);
        }
    }
}
