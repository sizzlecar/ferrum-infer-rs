#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

#define VNEXT_MAX_KV_PAGES 16384
#define VNEXT_SIMD_WIDTH 32
#define VNEXT_MAX_HEAD_CHUNKS 8
#define VNEXT_PREFILL_QUERY_TILE 8
#define VNEXT_PREFILL_KEY_TILE 32
#define VNEXT_GQA_PREFILL_KEY_TILE 64
#define VNEXT_PREFILL_SIMDGROUPS 4
#define VNEXT_DECODE_PARTITIONS 8

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

// Decode hot path for fixed pages that contain whole K/V token rows. The
// general kernel resolves a logical page address for every head element. At a
// long context that repeats 64-bit division and argument-buffer lookup 16
// times per key position for head_dim=256. Decode only needs one query row, so
// resolve the token page once and address its contiguous K/V head rows
// directly. Unsupported page geometry remains on the general kernel.
kernel void vnext_causal_attention_decode_direct_f16(
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
        device half *page_base = page_table.pages[page];
        const device half *key_row =
            page_base +
            (token_in_page * 2u * params.key_value_heads + kv_head) *
                params.head_dim;
        const device half *value_row =
            page_base +
            (token_in_page * 2u * params.key_value_heads +
             params.key_value_heads + kv_head) *
                params.head_dim;

        float partial_dot = 0.0f;
        for (uint chunk = 0; chunk < VNEXT_MAX_HEAD_CHUNKS; ++chunk) {
            const uint dim = lane + chunk * VNEXT_SIMD_WIDTH;
            if (dim < params.head_dim) {
                partial_dot += query_values[chunk] * float(key_row[dim]);
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
                                     float(value_row[dim]) * value_scale;
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

// First half of split-K GQA decode. Query heads that share one KV head are
// rows of the same 8x8 SIMDgroup matrices, so K/V is fetched once for the
// whole group. Eight context partitions keep one threadgroup resident per
// GPU core on a 32-core device; the reduction kernel below merges their
// independent online-softmax states.
kernel void vnext_causal_attention_decode_grouped_partial_f16(
    const device half *query [[buffer(0)]],
    const device half *query_raw [[buffer(1)]],
    device float *partials [[buffer(2)]],
    device VNextKvPageTable& page_table [[buffer(3)]],
    constant VNextCausalAttentionParams& params [[buffer(4)]],
    threadgroup half *shared_half [[threadgroup(0)]],
    threadgroup float *shared_float [[threadgroup(1)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint partition = group.x;
    const uint kv_head = group.y;
    const uint token = 0u;
    if (partition >= VNEXT_DECODE_PARTITIONS || params.tokens != 1u ||
        kv_head >= params.key_value_heads) {
        return;
    }

    const uint query_heads_per_kv_head =
        params.query_heads / params.key_value_heads;
    const uint query_head_start = kv_head * query_heads_per_kv_head;
    const uint query_elements =
        VNEXT_PREFILL_QUERY_TILE * params.head_dim;
    threadgroup half *query_tile = shared_half;
    threadgroup half *probabilities =
        query_tile + query_elements;
    threadgroup float *accumulated_output = shared_float;
    threadgroup float *scores =
        accumulated_output + query_elements;

    for (uint element = thread_index;
         element < query_elements;
         element += VNEXT_PREFILL_SIMDGROUPS * VNEXT_SIMD_WIDTH) {
        const uint query_row = element / params.head_dim;
        const uint dim = element - query_row * params.head_dim;
        const uint query_head = query_head_start + query_row;
        query_tile[element] =
            query_row < query_heads_per_kv_head
                ? query[((ulong)token * (ulong)params.query_heads +
                         (ulong)query_head) *
                            (ulong)params.head_dim +
                        (ulong)dim]
                : half(0.0h);
        accumulated_output[element] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint maximum_key_end = params.position_start + token + 1u;
    const uint key_value_row_stride =
        2u * params.key_value_heads * params.head_dim;
    const float attention_scale = rsqrt(float(params.head_dim));
    float running_max[VNEXT_PREFILL_QUERY_TILE / VNEXT_PREFILL_SIMDGROUPS];
    float running_sum[VNEXT_PREFILL_QUERY_TILE / VNEXT_PREFILL_SIMDGROUPS];
    for (uint row = 0;
         row < VNEXT_PREFILL_QUERY_TILE / VNEXT_PREFILL_SIMDGROUPS;
         ++row) {
        running_max[row] = -INFINITY;
        running_sum[row] = 0.0f;
    }

    for (uint key_start = partition * VNEXT_PREFILL_KEY_TILE;
         key_start < maximum_key_end;
         key_start += VNEXT_DECODE_PARTITIONS * VNEXT_PREFILL_KEY_TILE) {
        simdgroup_float8x8 score_matrix =
            make_filled_simdgroup_matrix<float, 8>(0.0f);
        const uint key_block_start = key_start + simdgroup * 8u;
        const uint key_block_rows =
            key_block_start < maximum_key_end
                ? min(8u, maximum_key_end - key_block_start)
                : 0u;
        if (key_block_rows == 8u) {
            device half *key_block = vnext_paged_element(
                page_table,
                params,
                vnext_kv_element_index(
                    key_block_start, 0, kv_head, 0, params));
            if (key_block != nullptr) {
                for (uint dim = 0; dim < params.head_dim; dim += 8u) {
                    simdgroup_half8x8 query_matrix;
                    simdgroup_half8x8 key_matrix;
                    simdgroup_load(
                        query_matrix,
                        query_tile + dim,
                        params.head_dim,
                        ulong2(0, 0),
                        false);
                    simdgroup_load(
                        key_matrix,
                        key_block + dim,
                        key_value_row_stride,
                        ulong2(0, 0),
                        true);
                    simdgroup_multiply_accumulate(
                        score_matrix,
                        query_matrix,
                        key_matrix,
                        score_matrix);
                }
            }
            simdgroup_store(
                score_matrix,
                scores + simdgroup * 8u,
                VNEXT_PREFILL_KEY_TILE,
                ulong2(0, 0),
                false);
        } else {
            // Never issue matrix loads for the partial final block: physical
            // page slack is intentionally uninitialized.
            if (lane == 0u) {
                for (uint query_row = 0;
                     query_row < VNEXT_PREFILL_QUERY_TILE;
                     ++query_row) {
                    for (uint key_row = 0; key_row < 8u; ++key_row) {
                        scores[query_row * VNEXT_PREFILL_KEY_TILE +
                               simdgroup * 8u + key_row] = 0.0f;
                    }
                }
            }
            for (uint query_row = 0;
                 query_row < query_heads_per_kv_head;
                 ++query_row) {
                for (uint key_row = 0; key_row < key_block_rows;
                     ++key_row) {
                    float partial_dot = 0.0f;
                    for (uint dim = lane; dim < params.head_dim;
                         dim += VNEXT_SIMD_WIDTH) {
                        partial_dot +=
                            float(query_tile
                                      [query_row * params.head_dim + dim]) *
                            vnext_load_kv(
                                page_table,
                                params,
                                key_block_start + key_row,
                                0,
                                kv_head,
                                dim);
                    }
                    const float dot = simd_sum(partial_dot);
                    if (lane == 0u) {
                        scores[query_row * VNEXT_PREFILL_KEY_TILE +
                               simdgroup * 8u + key_row] = dot;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint row_slot = 0;
             row_slot < VNEXT_PREFILL_QUERY_TILE / VNEXT_PREFILL_SIMDGROUPS;
             ++row_slot) {
            const uint query_row =
                row_slot * VNEXT_PREFILL_SIMDGROUPS + simdgroup;
            const uint key_position = key_start + lane;
            const bool keep =
                query_row < query_heads_per_kv_head &&
                key_position < maximum_key_end;
            const float score =
                keep
                    ? scores[query_row * VNEXT_PREFILL_KEY_TILE + lane] *
                          attention_scale
                    : -INFINITY;
            const float tile_maximum = simd_max(score);
            const float next_maximum =
                max(running_max[row_slot], tile_maximum);
            const float previous_scale =
                isinf(running_max[row_slot])
                    ? 0.0f
                    : exp(running_max[row_slot] - next_maximum);
            const float probability =
                keep ? exp(score - next_maximum) : 0.0f;
            running_sum[row_slot] =
                running_sum[row_slot] * previous_scale +
                simd_sum(probability);
            running_max[row_slot] = next_maximum;
            probabilities[query_row * VNEXT_PREFILL_KEY_TILE + lane] =
                half(probability);
            for (uint dim = lane; dim < params.head_dim;
                 dim += VNEXT_SIMD_WIDTH) {
                accumulated_output
                    [query_row * params.head_dim + dim] *=
                    previous_scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 output_matrices[8];
        const uint output_tiles_per_simdgroup =
            params.head_dim / (VNEXT_PREFILL_SIMDGROUPS * 8u);
        for (uint output_tile = 0;
             output_tile < output_tiles_per_simdgroup;
             ++output_tile) {
            const uint output_column =
                simdgroup * 8u +
                output_tile * VNEXT_PREFILL_SIMDGROUPS * 8u;
            simdgroup_load(
                output_matrices[output_tile],
                accumulated_output + output_column,
                params.head_dim,
                ulong2(0, 0),
                false);
        }
        for (uint key_tile = 0;
             key_tile < VNEXT_PREFILL_KEY_TILE / 8u;
             ++key_tile) {
            simdgroup_half8x8 probability_matrix;
            simdgroup_load(
                probability_matrix,
                probabilities + key_tile * 8u,
                VNEXT_PREFILL_KEY_TILE,
                ulong2(0, 0),
                false);
            const uint value_block_start = key_start + key_tile * 8u;
            const uint value_block_rows =
                value_block_start < maximum_key_end
                    ? min(8u, maximum_key_end - value_block_start)
                    : 0u;
            if (value_block_rows == 8u) {
                device half *value_block = vnext_paged_element(
                    page_table,
                    params,
                    vnext_kv_element_index(
                        value_block_start, 1, kv_head, 0, params));
                if (value_block != nullptr) {
                    for (uint output_tile = 0;
                         output_tile < output_tiles_per_simdgroup;
                         ++output_tile) {
                        const uint output_column =
                            simdgroup * 8u +
                            output_tile * VNEXT_PREFILL_SIMDGROUPS * 8u;
                        simdgroup_half8x8 value_matrix;
                        simdgroup_load(
                            value_matrix,
                            value_block + output_column,
                            key_value_row_stride,
                            ulong2(0, 0),
                            false);
                        simdgroup_multiply_accumulate(
                            output_matrices[output_tile],
                            probability_matrix,
                            value_matrix,
                            output_matrices[output_tile]);
                    }
                }
            }
        }
        for (uint output_tile = 0;
             output_tile < output_tiles_per_simdgroup;
             ++output_tile) {
            const uint output_column =
                simdgroup * 8u +
                output_tile * VNEXT_PREFILL_SIMDGROUPS * 8u;
            simdgroup_store(
                output_matrices[output_tile],
                accumulated_output + output_column,
                params.head_dim,
                ulong2(0, 0),
                false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint tail_rows = maximum_key_end % 8u;
        const uint tail_start = maximum_key_end - tail_rows;
        if (tail_rows != 0u && tail_start >= key_start &&
            tail_start < key_start + VNEXT_PREFILL_KEY_TILE) {
            const uint probability_column = tail_start - key_start;
            for (uint element = thread_index;
                 element < query_elements;
                 element += VNEXT_PREFILL_SIMDGROUPS * VNEXT_SIMD_WIDTH) {
                const uint query_row = element / params.head_dim;
                const uint dim = element - query_row * params.head_dim;
                float tail_value = 0.0f;
                for (uint key_row = 0; key_row < tail_rows; ++key_row) {
                    tail_value +=
                        float(probabilities
                                  [query_row * VNEXT_PREFILL_KEY_TILE +
                                   probability_column + key_row]) *
                        vnext_load_kv(
                            page_table,
                            params,
                            tail_start + key_row,
                            1,
                            kv_head,
                            dim);
                }
                accumulated_output[element] += tail_value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint partial_stride = params.head_dim + 2u;
    for (uint row_slot = 0;
         row_slot < VNEXT_PREFILL_QUERY_TILE / VNEXT_PREFILL_SIMDGROUPS;
         ++row_slot) {
        const uint query_row =
            row_slot * VNEXT_PREFILL_SIMDGROUPS + simdgroup;
        if (query_row >= query_heads_per_kv_head) {
            continue;
        }
        const uint query_head = query_head_start + query_row;
        const ulong partial_base =
            ((ulong)partition * (ulong)params.query_heads +
             (ulong)query_head) *
            (ulong)partial_stride;
        if (lane == 0u) {
            partials[partial_base] = running_max[row_slot];
            partials[partial_base + 1ul] = running_sum[row_slot];
        }
        for (uint dim = lane; dim < params.head_dim;
             dim += VNEXT_SIMD_WIDTH) {
            partials[partial_base + 2ul + (ulong)dim] =
                accumulated_output
                    [query_row * params.head_dim + dim];
        }
    }
}

kernel void vnext_causal_attention_decode_grouped_reduce_f16(
    const device float *partials [[buffer(0)]],
    const device half *query_raw [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant VNextCausalAttentionParams& params [[buffer(4)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint query_head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (params.tokens != 1u || query_head >= params.query_heads ||
        lane >= VNEXT_SIMD_WIDTH) {
        return;
    }

    const uint partial_stride = params.head_dim + 2u;
    const bool active = lane < VNEXT_DECODE_PARTITIONS;
    const ulong partial_base =
        ((ulong)lane * (ulong)params.query_heads + (ulong)query_head) *
        (ulong)partial_stride;
    const float local_maximum =
        active ? partials[partial_base] : -INFINITY;
    const float global_maximum = simd_max(local_maximum);
    const float scale =
        active && !isinf(local_maximum)
            ? exp(local_maximum - global_maximum)
            : 0.0f;
    const float scaled_sum =
        active ? partials[partial_base + 1ul] * scale : 0.0f;
    const float global_sum = simd_sum(scaled_sum);
    if (active) {
        shared[lane] = scale;
    }
    if (lane == 0u) {
        shared[VNEXT_DECODE_PARTITIONS] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inverse_sum = 1.0f / shared[VNEXT_DECODE_PARTITIONS];
    for (uint dim = lane; dim < params.head_dim;
         dim += VNEXT_SIMD_WIDTH) {
        float value = 0.0f;
        for (uint partition = 0; partition < VNEXT_DECODE_PARTITIONS;
             ++partition) {
            const ulong base =
                ((ulong)partition * (ulong)params.query_heads +
                 (ulong)query_head) *
                (ulong)partial_stride;
            value += partials[base + 2ul + (ulong)dim] *
                     shared[partition];
        }
        value *= inverse_sum;
        if (params.output_gate != 0u) {
            const ulong gate_index =
                (ulong)query_head * (ulong)params.query_head_stride +
                (ulong)params.head_dim + (ulong)dim;
            const float gate = float(query_raw[gate_index]);
            value *= 1.0f / (1.0f + exp(-gate));
        }
        output[(ulong)query_head * (ulong)params.head_dim +
               (ulong)dim] = half(value);
    }
}

// Full-attention prefill hot path for the head_dim=128 and head_dim=256 shapes
// used by the vNext Qwen3 and Qwen3.5 families. The scalar kernel above gives
// one threadgroup to every query row and rereads the same K/V history for each
// row. This kernel keeps eight query rows together and uses Apple SIMDgroup
// matrix operations for QK^T and P@V. The fixed-page dispatch contract keeps
// every eight-token K/V matrix inside one page, so matrix loads can read the
// paged cache directly instead of copying every element through threadgroup
// memory. The online softmax keeps memory linear in the context length.
//
// The operation still performs the exact O(q_len * kv_len) causal-attention
// math; tiling removes redundant device-memory traffic and replaces scalar
// dot products with matrix instructions. Other head shapes stay on the
// general kernel.
template <uint HEAD_TILES, uint SIMD_GROUP_COUNT, uint KEY_TILE>
inline void vnext_causal_attention_prefill_tiled_body(
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
        2u * params.key_value_heads * params.head_dim;
    const float attention_scale = rsqrt(float(params.head_dim));

    float running_max[rows_per_simdgroup];
    float running_sum[rows_per_simdgroup];
    for (uint row = 0; row < rows_per_simdgroup; ++row) {
        running_max[row] = -INFINITY;
        running_sum[row] = 0.0f;
    }

    for (uint key_start = 0; key_start < maximum_key_end;
         key_start += KEY_TILE) {
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
                device half *key_block = vnext_paged_element(
                    page_table,
                    params,
                    vnext_kv_element_index(
                        key_block_start, 0, kv_head, 0, params));
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
                            const float key_value = vnext_load_kv(
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

            device half *value_blocks[2] = {nullptr, nullptr};
            for (uint staged_key = 0; staged_key < 2; ++staged_key) {
                const uint value_block_start =
                    key_start + (key_tile + staged_key) * 8u;
                const uint value_block_rows =
                    value_block_start < maximum_key_end
                        ? min(8u, maximum_key_end - value_block_start)
                        : 0u;
                if (value_block_rows == 8u) {
                    value_blocks[staged_key] = vnext_paged_element(
                        page_table,
                        params,
                        vnext_kv_element_index(
                            value_block_start, 1, kv_head, 0, params));
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
                    const float value = vnext_load_kv(
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

kernel void vnext_causal_attention_prefill_tiled_f16(
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
    vnext_causal_attention_prefill_tiled_body<
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

kernel void vnext_causal_attention_prefill_gqa_tiled_f16(
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
    if (query_start >= params.tokens || params.key_value_heads == 0u ||
        params.query_heads % params.key_value_heads != 0u) {
        return;
    }
    const uint query_heads_per_kv_head =
        params.query_heads / params.key_value_heads;
    if (query_heads_per_kv_head < 2u ||
        query_heads_per_kv_head % 2u != 0u ||
        group.y >= params.query_heads / 2u) {
        return;
    }
    const uint pairs_per_kv_head = query_heads_per_kv_head / 2u;
    const uint kv_head = group.y / pairs_per_kv_head;
    const uint pair_in_kv_head = group.y % pairs_per_kv_head;
    const uint query_head_start =
        kv_head * query_heads_per_kv_head + pair_in_kv_head * 2u;
    vnext_causal_attention_prefill_tiled_body<
        2,
        8,
        VNEXT_GQA_PREFILL_KEY_TILE>(
        query,
        query_raw,
        output,
        page_table,
        params,
        shared_half,
        shared_float,
        query_start,
        query_head_start,
        kv_head,
        thread_index,
        simdgroup,
        lane);
}
