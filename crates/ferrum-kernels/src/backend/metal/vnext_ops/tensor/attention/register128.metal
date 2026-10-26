// Both domains share the same arithmetic and register-only P/O. Their explicit
// 64-byte packet does not change either legacy kernel's parameter ABI.
struct RegisterAttentionParams {
    ulong jobs, grid_groups, query_tiles, query_sequence, query_heads, key_value_heads, key_sequence;
    uint head_dim;
    float scale;
};

constant uint ATTENTION_REGISTER_GATED [[function_constant(106)]];
constant bool attention_register_gated = is_function_constant_defined(ATTENTION_REGISTER_GATED)
    ? ATTENTION_REGISTER_GATED != 0 : false;

constant uint ATTENTION_REGISTER_DOMAIN [[function_constant(105)]];
constant bool attention_register_cross = is_function_constant_defined(ATTENTION_REGISTER_DOMAIN)
    ? ATTENTION_REGISTER_DOMAIN != 0 : false;

inline uint attention_register_visible_start(constant AttentionMask& mask, uint query) {
    return attention_register_cross ? 0 : attention_visible_start(mask, query);
}
inline uint attention_register_visible_end(
    constant AttentionMask& mask, uint query, constant RegisterAttentionParams& p) {
    // Cross masks intentionally have no segments. Do not infer their range
    // through the structured helper, whose empty segment list returns zero.
    return attention_register_cross ? uint(p.key_sequence) : attention_visible_end(mask, query);
}

constant uint ATTENTION_REGISTER_ALL_KEYS_VALID [[function_constant(104)]];
constant bool attention_register_all_keys_valid =
    is_function_constant_defined(ATTENTION_REGISTER_ALL_KEYS_VALID)
        ? ATTENTION_REGISTER_ALL_KEYS_VALID != 0 : false;

inline bool attention_register_key_valid(constant AttentionMask& mask, uint batch, ulong key) {
    // Only the host's validated, all-batch canonical key-run proof enables this.
    // Every caller still checks the row range or sequence tail independently.
    return attention_register_all_keys_valid || attention_key_valid(mask, batch, key);
}

// Finite V above 2^93 disables in-place accumulation for the rest of this job:
// the largest allowed sequence is below 2^32, so its nominal unnormalized
// absolute sum at unit probability is below 2^125, below FP32's 2^128 range.
// Do not rely on that real-arithmetic bound alone: observe actual O after each
// update and guard P/alpha too. |O|<=2^124, P/alpha<=2 and |V|<=2^93 leave
// ample range for a 16-key MMA even including FP32 rounding. The fallback is
// the original arithmetic, not a claim that it solves every finite overflow.
constant float ATTENTION_REGISTER_FINITE_V_LIMIT = 0x1p93f;
constant float ATTENTION_REGISTER_ACCUMULATOR_LIMIT = 0x1p124f;

// Values are ordered P[0].e0, P[0].e1, P[1].e0, P[1].e1. The fast mapping is
// proved from a coordinate tile loaded by Metal; no lane-layout bit formula is
// assumed. Unknown fragment layouts use the full coordinate-filtered shuffle.
inline bool attention_register_has_key(
    constant AttentionMask& mask, uint batch, uint begin, uint end) {
    if (attention_register_all_keys_valid) return begin < end;
    uint previous = 0;
    for (uint run = 0; run < mask.key_run_counts[batch]; ++run) {
        uint next = mask.key_run_ends[batch * 32 + run];
        if ((mask.key_valid_masks[batch] & (1u << run)) != 0
            && max(begin, previous) < min(end, next)) return true;
        previous = next;
    }
    return false;
}

inline float attention_register_reduce(
    thread const float* values, uint2 rows, uint target_row,
    thread const uint* peers, bool four_peers, bool xor_peers, bool sum) {
    float result = sum ? 0.0f : -INFINITY;
    if (four_peers) {
        float local = sum ? 0.0f : -INFINITY;
        for (uint i = 0; i < 4; ++i)
            local = sum ? local + values[i] : max(local, values[i]);
        if (xor_peers) {
            float other = simd_shuffle_xor(local, ushort(1));
            float pair = sum ? local + other : max(local, other);
            other = simd_shuffle_xor(pair, ushort(8));
            result = sum ? pair + other : max(pair, other);
        } else {
            for (uint i = 0; i < 4; ++i) {
                float other = simd_shuffle(local, ushort(peers[i]));
                result = sum ? result + other : max(result, other);
            }
        }
    } else {
        for (ushort lane = 0; lane < 32; ++lane) {
            uint2 source_rows = uint2(simd_shuffle(rows.x, lane), simd_shuffle(rows.y, lane));
            for (uint i = 0; i < 4; ++i) {
                float other = simd_shuffle(values[i], lane);
                if (source_rows[i % 2] == target_row)
                    result = sum ? result + other : max(result, other);
            }
        }
    }
    return result;
}

kernel void ferrum_tensor_attention_register128_bf16(
    device const ushort* query [[buffer(0)]],
    device const ushort* key [[buffer(1)]],
    device const ushort* value [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant RegisterAttentionParams& p [[buffer(4)]],
    constant AttentionMask& mask [[buffer(5)]],
    device const uint* job_flags [[buffer(6)]],
    threadgroup uchar* workspace [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sid [[simdgroup_index_in_threadgroup]]) {
    threadgroup atomic_uint* nonfinite_value = reinterpret_cast<threadgroup atomic_uint*>(workspace);
    threadgroup ushort* q_tile = reinterpret_cast<threadgroup ushort*>(workspace + 16);
    threadgroup ushort* kv_tile = q_tile + 32 * 136;
    threadgroup float* rare_probability = reinterpret_cast<threadgroup float*>(kv_tile + 16 * 136);
    threadgroup float* coordinates = reinterpret_cast<threadgroup float*>(kv_tile);
    uint2 rows, columns;
    uint peers[4];
    uint peer_count, column_mask, peer_mask;
    bool four_peers, xor_peers;
    attention_register_calibrate(coordinates, tid, lane, rows, columns, peers,
        peer_count, column_mask, peer_mask, four_peers, xor_peers);

    for (ulong job = group; job < p.jobs; job += p.grid_groups) {
        // Full original grid/pitch is retained. Only a successfully completed
        // dense tile is skipped; prefix and rejected suffix jobs remain strict.
        if (attention_register_gated && job_flags[job] != 0) continue;
        const ulong q_start = (job % p.query_tiles) * 32;
        const ulong q_head = (job / p.query_tiles) % p.query_heads;
        const uint batch = uint(job / (p.query_tiles * p.query_heads));
        const ulong kv_head = q_head / (p.query_heads / p.key_value_heads);
        const ulong q_base = (ulong(batch) * p.query_sequence * p.query_heads + q_head) * 128;
        const ulong kv_base = (ulong(batch) * p.key_sequence * p.key_value_heads + kv_head) * 128;
        const uint min_start = attention_register_visible_start(mask, uint(q_start));
        const uint max_end = attention_register_visible_end(mask, uint(min(q_start + 31, p.query_sequence - 1)), p);
        uint2 row_start, row_end;
        for (uint e = 0; e < 2; ++e) {
            ulong q = q_start + sid * 8 + rows[e];
            row_start[e] = q < p.query_sequence ? attention_register_visible_start(mask, uint(q)) : 0;
            row_end[e] = q < p.query_sequence ? attention_register_visible_end(mask, uint(q), p) : 0;
        }
        float2 row_max = float2(-INFINITY), row_sum = float2(0.0f);
        bool2 poisoned = bool2(false), seen = bool2(false);
        bool wide_value_seen = false;
        bool accumulator_risk = false;
        simdgroup_float8x8 accumulated[16];
        #pragma unroll
        for (uint d = 0; d < 16; ++d)
            accumulated[d] = make_filled_simdgroup_matrix<float, 8>(0.0f);

        // Use one otherwise unused padding entry per row. Empty rows do not
        // read Q even when the batch has valid keys outside this row's range.
        if (tid < 32) {
            ulong q = q_start + tid;
            q_tile[tid * 136 + 128] = ushort(q < p.query_sequence
                && attention_register_has_key(mask, batch,
                    attention_register_visible_start(mask, uint(q)),
                    attention_register_visible_end(mask, uint(q), p)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 32 * 128; i += 128) {
            uint r = i / 128, c = i % 128;
            ulong q = q_start + r;
            q_tile[r * 136 + c] = q_tile[r * 136 + 128] != 0
                ? query[q_base + q * p.query_heads * 128 + c] : ushort(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ulong k_start = min_start; k_start < max_end; k_start += 16) {
            if (tid == 0) atomic_store_explicit(nonfinite_value, 0, memory_order_relaxed);
            // Each SIMD transposes one 8-dimension x 8-key block per round.
            // Four adjacent lanes read adjacent dimension pairs from one key;
            // two shuffles turn these into adjacent key pairs for the same D.
            for (uint tile_round = 0; tile_round < 8; ++tile_round) {
                uint tile = 8 * sid + tile_round;
                uint d_base = (tile / 2) * 8, c_base = (tile % 2) * 8;
                uint read_d = d_base + 2 * (lane % 4);
                ulong k = k_start + c_base + lane / 4;
                uint packed = 0;
                if (k < max_end && attention_register_key_valid(mask, batch, k)) {
                    ulong offset = kv_base + k * p.key_value_heads * 128 + read_d;
                    // The input contract guarantees only ushort alignment.
                    packed = uint(key[offset]) | (uint(key[offset + 1]) << 16);
                }
                ushort source = ushort(8 * (lane % 4) + lane / 8);
                uint first = simd_shuffle(packed, source);
                uint second = simd_shuffle(packed, ushort(source + 4));
                uint shift = 16 * ((lane / 4) % 2);
                uint write_d = d_base + lane / 4, write_c = c_base + 2 * (lane % 4);
                kv_tile[write_d * 24 + write_c] = ushort(first >> shift);
                kv_tile[write_d * 24 + write_c + 1] = ushort(second >> shift);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_float8x8 probability[2];
            for (uint n = 0; n < 2; ++n)
                probability[n] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (uint d = 0; d < 128; d += 8) {
                simdgroup_float8x8 q, k;
                #pragma clang loop unroll(full)
                for (uint e = 0; e < 2; ++e)
                    q.thread_elements()[e] = decode_bf16(q_tile[(sid * 8 + rows[e]) * 136 + d + columns[e]]);
                #pragma clang loop unroll(full)
                for (uint n = 0; n < 2; ++n) {
                    #pragma clang loop unroll(full)
                    for (uint e = 0; e < 2; ++e)
                        k.thread_elements()[e] = decode_bf16(kv_tile[(d + rows[e]) * 24 + n * 8 + columns[e]]);
                    simdgroup_multiply_accumulate(probability[n], q, k, probability[n]);
                }
            }
            // All QK consumers finish before K storage becomes V storage.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < 16 * 128; i += 128) {
                uint r = i / 128, c = i % 128;
                ulong k = k_start + r;
                ushort bits = k < max_end && attention_register_key_valid(mask, batch, k)
                    ? value[kv_base + k * p.key_value_heads * 128 + c] : ushort(0);
                kv_tile[r * 136 + c] = bits;
                float v = decode_bf16(bits);
                uint risk = !isfinite(v) ? 1u
                    : fabs(v) > ATTENTION_REGISTER_FINITE_V_LIMIT ? 2u : 0u;
                if (risk != 0) atomic_fetch_or_explicit(nonfinite_value, risk, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scores[4];
            bool visible[4];
            for (uint n = 0; n < 2; ++n) for (uint e = 0; e < 2; ++e) {
                uint i = n * 2 + e;
                ulong k = k_start + n * 8 + columns[e];
                visible[i] = k >= row_start[e] && k < row_end[e]
                    && attention_register_key_valid(mask, batch, k);
                scores[i] = visible[i] ? probability[n].thread_elements()[e] * p.scale : 0.0f;
            }
            float2 alpha, maximum;
            if (four_peers) {
                // Calibration proves both elements belong to the same row.
                // -Inf marks no visible score; every visible nonfinite score,
                // including -Inf, becomes +Inf and permanently poisons the row.
                float classified[4];
                for (uint i = 0; i < 4; ++i)
                    classified[i] = !visible[i] ? -INFINITY
                        : isfinite(scores[i]) ? scores[i] : INFINITY;
                float tile_max = attention_register_reduce(classified, rows, rows.x, peers, four_peers, xor_peers, false);
                bool present = tile_max != -INFINITY;
                bool bad = tile_max == INFINITY;
                // Finite rows keep the original max tree. A poisoned row may
                // carry +Inf internally; its externally visible result remains
                // canonical NaN regardless of all later tiles.
                maximum = float2(max(row_max.x, tile_max));
                poisoned = bool2(poisoned.x || bad);
                alpha = float2(present && !poisoned.x
                    ? (row_sum.x == 0.0f ? 0.0f : exp(row_max.x - maximum.x)) : 1.0f);
                seen = bool2(seen.x || present);
            } else {
                // Unknown layouts retain independent per-element row states
                // and the original three classification reductions.
                float maxima[4], bad_values[4], present_values[4];
                for (uint i = 0; i < 4; ++i) {
                    maxima[i] = visible[i] && isfinite(scores[i]) ? scores[i] : -INFINITY;
                    bad_values[i] = visible[i] && !isfinite(scores[i]) ? 1.0f : 0.0f;
                    present_values[i] = visible[i] ? 1.0f : 0.0f;
                }
                for (uint e = 0; e < 2; ++e) {
                    bool present = attention_register_reduce(present_values, rows, rows[e], peers, four_peers, xor_peers, false) > 0.0f;
                    bool bad = attention_register_reduce(bad_values, rows, rows[e], peers, four_peers, xor_peers, false) > 0.0f;
                    maximum[e] = max(row_max[e], attention_register_reduce(maxima, rows, rows[e], peers, four_peers, xor_peers, false));
                    poisoned[e] = poisoned[e] || bad;
                    alpha[e] = present && !poisoned[e]
                        ? (row_sum[e] == 0.0f ? 0.0f : exp(row_max[e] - maximum[e])) : 1.0f;
                    seen[e] = seen[e] || present;
                }
            }
            float probabilities[4];
            for (uint n = 0; n < 2; ++n) for (uint e = 0; e < 2; ++e) {
                uint i = n * 2 + e;
                float v = visible[i] && !poisoned[e] ? exp(scores[i] - maximum[e]) : 0.0f;
                probabilities[i] = v;
                probability[n].thread_elements()[e] = v;
            }
            if (four_peers) {
                float sum = attention_register_reduce(probabilities, rows, rows.x, peers, four_peers, xor_peers, true);
                row_sum = float2(row_sum.x * alpha.x + sum);
                row_max = maximum;
            } else {
                for (uint e = 0; e < 2; ++e) {
                    float sum = attention_register_reduce(probabilities, rows, rows[e], peers, four_peers, xor_peers, true);
                    row_sum[e] = row_sum[e] * alpha[e] + sum;
                    row_max[e] = maximum[e];
                }
            }

            // The flag is uniform across all four SIMD groups. Only this rare
            // IEEE path materializes P, disjoint from V in the same KV region.
            uint value_flags = atomic_load_explicit(nonfinite_value, memory_order_relaxed);
            bool nonfinite = (value_flags & 1u) != 0;
            wide_value_seen = wide_value_seen || (value_flags & 2u) != 0;
            bool scale_or_probability_risk = !(alpha.x >= 0.0f && alpha.x <= 2.0f
                && alpha.y >= 0.0f && alpha.y <= 2.0f);
            for (uint i = 0; i < 4; ++i)
                scale_or_probability_risk = scale_or_probability_risk
                    || !(probabilities[i] >= 0.0f && probabilities[i] <= 2.0f);
            // SIMD-uniform decision for collective MMA. O risk is sticky, so
            // a previous Inf/NaN or very large O cannot silently re-enter the
            // new accumulation order when a later tile contains small values.
            bool inplace = !wide_value_seen
                && !simd_any(accumulator_risk || scale_or_probability_risk);
            if (nonfinite) {
                for (uint n = 0; n < 2; ++n)
                    simdgroup_store(probability[n], rare_probability + sid * 8 * 16 + n * 8, 16, 0, false);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            #pragma unroll
            for (uint d = 0; d < 128; d += 8) {
                if (!nonfinite && inplace) {
                    // A distinct numerical profile: round the online rescale
                    // in FP32, then use O as the MMA C accumulator. P and V
                    // remain FP32; there is no intermediate zero-based PV sum.
                    for (uint e = 0; e < 2; ++e)
                        accumulated[d / 8].thread_elements()[e] =
                            accumulated[d / 8].thread_elements()[e] * alpha[e];
                    for (uint n = 0; n < 2; ++n) {
                        simdgroup_float8x8 v;
                        for (uint e = 0; e < 2; ++e)
                            v.thread_elements()[e] = decode_bf16(kv_tile[(n * 8 + rows[e]) * 136 + d + columns[e]]);
                        simdgroup_multiply_accumulate(accumulated[d / 8], probability[n], v, accumulated[d / 8]);
                    }
                } else {
                    // Extreme finite values retain the original zero-based
                    // two-MMA PV and separate add. Nonfinite V retains the
                    // original visible-only scalar IEEE path, including 0*Inf.
                    simdgroup_float8x8 pv = make_filled_simdgroup_matrix<float, 8>(0.0f);
                    if (!nonfinite) {
                        for (uint n = 0; n < 2; ++n) {
                            simdgroup_float8x8 v;
                            for (uint e = 0; e < 2; ++e)
                                v.thread_elements()[e] = decode_bf16(kv_tile[(n * 8 + rows[e]) * 136 + d + columns[e]]);
                            simdgroup_multiply_accumulate(pv, probability[n], v, pv);
                        }
                    } else {
                        for (uint e = 0; e < 2; ++e) {
                            float sum = 0.0f;
                            if (!poisoned[e]) for (uint j = 0; j < 16; ++j) {
                                ulong k = k_start + j;
                                if (k >= row_start[e] && k < row_end[e] && attention_register_key_valid(mask, batch, k))
                                    sum = sum + rare_probability[(sid * 8 + rows[e]) * 16 + j]
                                        * decode_bf16(kv_tile[j * 136 + d + columns[e]]);
                            }
                            pv.thread_elements()[e] = sum;
                        }
                    }
                    for (uint e = 0; e < 2; ++e)
                        accumulated[d / 8].thread_elements()[e] =
                            accumulated[d / 8].thread_elements()[e] * alpha[e] + pv.thread_elements()[e];
                }
                for (uint e = 0; e < 2; ++e)
                    accumulator_risk = accumulator_risk
                        || !(fabs(accumulated[d / 8].thread_elements()[e]) <= ATTENTION_REGISTER_ACCUMULATOR_LIMIT);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        #pragma unroll
        for (uint d = 0; d < 128; d += 8) for (uint e = 0; e < 2; ++e) {
            ulong q = q_start + sid * 8 + rows[e];
            if (q < p.query_sequence) {
                ushort result = poisoned[e] ? ushort(0x7fc0) : !seen[e] ? ushort(0)
                    : encode_bf16(accumulated[d / 8].thread_elements()[e] * (1.0f / row_sum[e]));
                output[q_base + q * p.query_heads * 128 + d + columns[e]] = result;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
