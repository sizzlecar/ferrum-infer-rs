#include <metal_simdgroup_matrix>

constant uint TENSOR_ATTENTION_HEAD [[function_constant(101)]];
constant bool cache_query_128 = is_function_constant_defined(TENSOR_ATTENTION_HEAD)
    ? TENSOR_ATTENTION_HEAD == 128 : false;
// MPP reads Q directly for complete visible tiles. Only the independent SIMD
// library retains full-head Q fragments; MPP fallbacks stage one 32-wide slice.
// Output fragments and their shared-memory layout still use cache_query_128.
#ifdef FERRUM_ATTENTION_MPP_QK
constant bool cache_q_fragments = false;
#else
constant bool cache_q_fragments = cache_query_128;
#endif

struct AttentionParams {
    ulong jobs, grid_groups, query_tiles, sequence, query_heads, key_value_heads;
    #ifdef FERRUM_ATTENTION_CROSS
    ulong key_sequence;
    #endif
    uint head_dim;
    float scale;
};
struct AttentionMask {
    uint isolated_segments;
    uint segment_count;
    uint segment_ends[32];
    uint bidirectional_mask;
    uint key_run_counts[16];
    uint key_run_ends[512];
    uint key_valid_masks[16];
    uint first_valid_keys[16];
};
inline uint attention_visible_start(constant AttentionMask& mask, uint query) {
    if (!mask.isolated_segments) return 0;
    uint start = 0;
    for (uint i = 0; i < mask.segment_count; ++i) {
        if (query < mask.segment_ends[i]) return start;
        start = mask.segment_ends[i];
    }
    return start;
}
inline uint attention_visible_end(constant AttentionMask& mask, uint query) {
    for (uint i = 0; i < mask.segment_count; ++i)
        if (query < mask.segment_ends[i])
            return (mask.bidirectional_mask & (1u << i)) ? mask.segment_ends[i] : query + 1;
    return 0;
}
inline bool attention_key_valid(constant AttentionMask& mask, uint batch, ulong key) {
    for (uint i = 0; i < mask.key_run_counts[batch]; ++i)
        if (key < mask.key_run_ends[batch * 32 + i])
            return (mask.key_valid_masks[batch] & (1u << i)) != 0;
    return false;
}

// Four SIMD groups own 8x16 rectangles in each 16x32 matrix product.
// All intermediates stay FP32 in device-bounded threadgroup memory. No S*S
// allocation, BF16 probability rounding, or repeated GQA K/V allocation.
kernel void ferrum_tensor_attention_bf16(
    device const ushort* query [[buffer(0)]],
    device const ushort* key [[buffer(1)]],
    device const ushort* value [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant AttentionParams& p [[buffer(4)]],
    constant AttentionMask& mask [[buffer(5)]],
    threadgroup float* workspace [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float* a = workspace;
    threadgroup float* b = workspace + 512;
    threadgroup float* scores = workspace + 1536;
    // The cached path writes its final register accumulator over the retired
    // Q/K/P/V staging area; row state stays outside that reused interval.
    threadgroup float* accum = cache_query_128 ? workspace : workspace + 2048;
    threadgroup float* row_max = workspace + 2048 + (cache_query_128 ? 0 : 16 * p.head_dim);
    threadgroup float* row_sum = row_max + 16;
    threadgroup float* alpha = row_sum + 16;
    threadgroup uint* end = reinterpret_cast<threadgroup uint*>(alpha + 16);
    threadgroup uint* poisoned = end + 16;
    threadgroup uint* start = poisoned + 16;
    threadgroup atomic_uint nonfinite_value;
    const uint local_row = (sid / 2) * 8;
    const uint local_col = (sid % 2) * 16;
    #ifdef FERRUM_ATTENTION_CROSS
    const ulong key_sequence = p.key_sequence;
    #else
    const ulong key_sequence = p.sequence;
    #endif
    for (ulong job = group; job < p.jobs; job += p.grid_groups) {
        const ulong q_start = (job % p.query_tiles) * 16;
        const ulong q_head = (job / p.query_tiles) % p.query_heads;
        const uint batch = uint(job / (p.query_tiles * p.query_heads));
        const ulong kv_head = q_head / (p.query_heads / p.key_value_heads);
        const ulong q_base = (ulong(batch) * p.sequence * p.query_heads + q_head) * p.head_dim;
        const ulong kv_base = (ulong(batch) * key_sequence * p.key_value_heads + kv_head) * p.head_dim;
        #ifdef FERRUM_ATTENTION_CROSS
        const uint max_end = uint(key_sequence), min_start = 0;
        #else
        const uint max_end = attention_visible_end(mask, uint(min(q_start + 15, p.sequence - 1)));
        const uint min_start = attention_visible_start(mask, uint(q_start));
        #endif
        if (tid < 16) {
            #ifdef FERRUM_ATTENTION_CROSS
            end[tid] = q_start + tid < p.sequence ? uint(key_sequence) : 0;
            start[tid] = 0;
            #else
            end[tid] = q_start + tid < p.sequence ? attention_visible_end(mask, uint(q_start + tid)) : 0;
            start[tid] = q_start + tid < p.sequence ? attention_visible_start(mask, uint(q_start + tid)) : 0;
            #endif
            row_max[tid] = -INFINITY;
            row_sum[tid] = 0.0f;
            alpha[tid] = 1.0f;
            poisoned[tid] = 0;
        }
        for (uint i = tid; i < 16 * p.head_dim; i += 128) accum[i] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // No valid key in this query tile: do not read any Q/K/V data.
        if (mask.first_valid_keys[batch] < max_end) {
            // Compile away Q fragment retention in the MPP library without
            // disabling the independent cached output or row-coordinate state.
            simdgroup_float8x8 cached_query[16];
            simdgroup_float8x8 cached_output[8], row_coordinates;
            if (cache_query_128) {
                // Obtain the fragment row mapping through Metal's own load,
                // without assuming a particular SIMD lane/element layout.
                for (uint i = tid; i < 512; i += 128) a[i] = float(i / 32);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(row_coordinates, a + local_row * 32 + local_col, 32, 0, false);
                for (uint i = 0; i < 8; ++i)
                    cached_output[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (cache_q_fragments) {
                #pragma unroll
                for (uint d = 0; d < 128; d += 32) {
                    for (uint i = tid; i < 512; i += 128) {
                        uint r = i / 32, c = d + i % 32;
                        a[i] = end[r] > mask.first_valid_keys[batch]
                            ? decode_bf16(query[q_base + (q_start + r) * p.query_heads * p.head_dim + c]) : 0.0f;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    #pragma unroll
                    for (uint chunk = 0; chunk < 32; chunk += 8)
                        simdgroup_load(cached_query[(d + chunk) / 8], a + local_row * 32 + chunk, 32, 0, false);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            for (ulong k_start = min_start; k_start < max_end; k_start += 32) {
                #ifdef FERRUM_ATTENTION_MPP_QK
                // Every thread takes the same collective branch. Borrowed
                // tensor views may read only complete, visible Q/K tiles;
                // tails, mask holes and differing row ranges use the original
                // QK path below. No softmax/PV boundary changes.
                bool mpp_tile = cache_query_128 && q_start + 16 <= p.sequence
                    && k_start + 32 <= key_sequence;
                if (mpp_tile) {
                    for (uint r = 0; r < 16; ++r)
                        mpp_tile = mpp_tile && k_start >= start[r] && k_start + 32 <= end[r];
                    for (uint j = 0; j < 32; ++j)
                        mpp_tile = mpp_tile && attention_key_valid(mask, batch, k_start + j);
                }
                if (mpp_tile) {
                    ferrum_attention_mpp_qk(query, key,
                        q_base + q_start * p.query_heads * 128,
                        kv_base + k_start * p.key_value_heads * 128,
                        int(p.query_heads * 128), int(p.key_value_heads * 128), scores);
                } else
                #endif
                {
                simdgroup_float8x8 dot[2];
                for (uint i = 0; i < 2; ++i) dot[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
                #pragma unroll
                for (uint d = 0; d < (cache_query_128 ? 128 : p.head_dim); d += 32) {
                    if (!cache_q_fragments) for (uint i = tid; i < 512; i += 128) {
                        uint r = i / 32, c = d + i % 32;
                        a[i] = end[r] > mask.first_valid_keys[batch] && !poisoned[r] && c < p.head_dim
                            ? decode_bf16(query[q_base + (q_start + r) * p.query_heads * p.head_dim + c]) : 0.0f;
                    }
                    for (uint i = tid; i < 1024; i += 128) {
                        ulong k = k_start + i % 32;
                        uint c = d + i / 32;
                        b[i] = k < max_end && c < p.head_dim && attention_key_valid(mask, batch, k)
                            ? decode_bf16(key[kv_base + k * p.key_value_heads * p.head_dim + c]) : 0.0f;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    #pragma unroll
                    for (uint chunk = 0; chunk < 32; chunk += 8) {
                        simdgroup_float8x8 av, bv;
                        if (cache_q_fragments) av = cached_query[(d + chunk) / 8];
                        else simdgroup_load(av, a + local_row * 32 + chunk, 32, 0, false);
                        for (uint n = 0; n < 2; ++n) {
                            simdgroup_load(bv, b + chunk * 32 + local_col + n * 8, 32, 0, false);
                            simdgroup_multiply_accumulate(dot[n], av, bv, dot[n]);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                for (uint n = 0; n < 2; ++n)
                    simdgroup_store(dot[n], scores + local_row * 32 + local_col + n * 8, 32, 0, false);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                // A SIMD group handles one row's keys in parallel. Only the
                // reduction order changes; scores, exp, sums and rescaling all
                // remain FP32 with no probability quantization boundary.
                const uint lane = tid % 32;
                for (uint r = sid; r < 16; r += 4) {
                    ulong k = k_start + lane;
                    bool visible = k >= start[r] && k < end[r] && attention_key_valid(mask, batch, k);
                    float score = visible ? scores[r * 32 + lane] * p.scale : 0.0f;
                    bool bad = simd_any(visible && !isfinite(score)) || poisoned[r];
                    bool present = simd_any(visible);
                    float maximum = max(row_max[r], simd_max(visible && isfinite(score) ? score : -INFINITY));
                    float rescale = 1.0f, total = 0.0f;
                    if (present && !bad)
                        rescale = row_sum[r] == 0.0f ? 0.0f : exp(row_max[r] - maximum);
                    float probability = visible && !bad ? exp(score - maximum) : 0.0f;
                    scores[r * 32 + lane] = probability;
                    total = simd_sum(probability);
                    if (lane == 0) {
                        poisoned[r] = uint(bad);
                        alpha[r] = rescale;
                        row_sum[r] = row_sum[r] * rescale + total;
                        row_max[r] = maximum;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                #pragma unroll
                for (uint d = 0; d < (cache_query_128 ? 128 : p.head_dim); d += 32) {
                    if (tid == 0) atomic_store_explicit(&nonfinite_value, 0, memory_order_relaxed);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint i = tid; i < 1024; i += 128) {
                        ulong k = k_start + i / 32;
                        uint c = d + i % 32;
                        float v = k < max_end && c < p.head_dim && attention_key_valid(mask, batch, k)
                            ? decode_bf16(value[kv_base + k * p.key_value_heads * p.head_dim + c]) : 0.0f;
                        b[i] = v;
                        if (!isfinite(v)) atomic_store_explicit(&nonfinite_value, 1, memory_order_relaxed);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    simdgroup_float8x8 pv[2];
                    if (atomic_load_explicit(&nonfinite_value, memory_order_relaxed) == 0) {
                        #ifdef FERRUM_ATTENTION_MPP_PV
                        // QK has finished consuming its a staging tile, which
                        // is disjoint from b and scores; the cached final
                        // accumulator is stored there only after all key tiles.
                        // Reuse a for the finite FP32 PV result without
                        // another buffer.
                        ferrum_attention_mpp_pv(scores, b, a);
                        #else
                        for (uint n = 0; n < 2; ++n) pv[n] = make_filled_simdgroup_matrix<float, 8>(0.0f);
                        for (uint chunk = 0; chunk < 32; chunk += 8) {
                            simdgroup_float8x8 av, bv;
                            simdgroup_load(av, scores + local_row * 32 + chunk, 32, 0, false);
                            for (uint n = 0; n < 2; ++n) {
                                simdgroup_load(bv, b + chunk * 32 + local_col + n * 8, 32, 0, false);
                                simdgroup_multiply_accumulate(pv[n], av, bv, pv[n]);
                            }
                        }
                        if (!cache_query_128) for (uint n = 0; n < 2; ++n)
                            simdgroup_store(pv[n], a + local_row * 32 + local_col + n * 8, 32, 0, false);
                        #endif
                    } else {
                        // Dense P*V would evaluate masked 0*NaN/Inf. Preserve
                        // key-only masking with a visible-only IEEE fallback.
                        for (uint i = tid; i < 512; i += 128) {
                            uint r = i / 32, c = i % 32;
                            float sum = 0.0f;
                            if (!poisoned[r]) for (uint j = 0; j < 32; ++j) {
                                ulong k = k_start + j;
                                if (k >= start[r] && k < end[r] && attention_key_valid(mask, batch, k))
                                    sum = sum + scores[r * 32 + j] * b[j * 32 + c];
                            }
                            a[i] = sum;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (cache_query_128) {
                        #pragma unroll
                        for (uint n = 0; n < 2; ++n) {
                            #ifdef FERRUM_ATTENTION_MPP_PV
                            // Both MPP and the nonfinite fallback wrote a.
                            // The preceding group barrier publishes the tile
                            // before restoring the original SIMD accumulator.
                            simdgroup_load(pv[n], a + local_row * 32 + local_col + n * 8, 32, 0, false);
                            #else
                            if (atomic_load_explicit(&nonfinite_value, memory_order_relaxed) != 0)
                                simdgroup_load(pv[n], a + local_row * 32 + local_col + n * 8, 32, 0, false);
                            #endif
                            #pragma unroll
                            for (uint e = 0; e < 2; ++e) {
                                uint row = uint(row_coordinates.thread_elements()[e]);
                                uint index = (d / 32) * 2 + n;
                                cached_output[index].thread_elements()[e] =
                                    cached_output[index].thread_elements()[e] * alpha[row]
                                    + pv[n].thread_elements()[e];
                            }
                        }
                    } else for (uint i = tid; i < 512; i += 128) {
                        uint r = i / 32, c = d + i % 32;
                        if (c < p.head_dim) {
                            uint index = r * p.head_dim + c;
                            accum[index] = accum[index] * alpha[r] + a[i];
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            if (cache_query_128) {
                #pragma unroll
                for (uint d = 0; d < 128; d += 32)
                    for (uint n = 0; n < 2; ++n)
                        simdgroup_store(cached_output[(d / 32) * 2 + n], accum + local_row * 128 + d + local_col + n * 8, 128, 0, false);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 16 * p.head_dim; i += 128) {
            uint r = i / p.head_dim, c = i % p.head_dim;
            if (q_start + r < p.sequence) {
                ushort result = end[r] <= mask.first_valid_keys[batch] ? ushort(0)
                    : poisoned[r] ? ushort(0x7fc0) : encode_bf16(accum[i] * (1.0f / row_sum[r]));
                output[q_base + (q_start + r) * p.query_heads * p.head_dim + c] = result;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
