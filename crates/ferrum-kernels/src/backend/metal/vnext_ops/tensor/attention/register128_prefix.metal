// Test-only QK -> online-softmax state, copied from the R267 register path.
// Production params/calibration/reduce/mask helpers precede this source.
// V/PV/O are absent; resource geometry and unconditional barriers are retained.
// Different register liveness and IO mean full minus prefix is not PV time.
kernel void ferrum_tensor_attention_qk_prefix_f32(
    device const ushort* query [[buffer(0)]],
    device const ushort* key [[buffer(1)]],
    device float* output [[buffer(3)]],
    constant RegisterAttentionParams& p [[buffer(4)]],
    constant AttentionMask& mask [[buffer(5)]],
    threadgroup uchar* workspace [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sid [[simdgroup_index_in_threadgroup]]) {
    threadgroup ushort* q_tile = reinterpret_cast<threadgroup ushort*>(workspace + 16);
    threadgroup ushort* kv_tile = q_tile + 32 * 136;
    threadgroup float* coordinates = reinterpret_cast<threadgroup float*>(kv_tile);
    uint2 rows, columns;
    uint peers[4];
    uint peer_count, column_mask, peer_mask;
    bool four_peers, xor_peers;
    attention_register_calibrate(coordinates, tid, lane, rows, columns, peers,
        peer_count, column_mask, peer_mask, four_peers, xor_peers);

    for (ulong job = group; job < p.jobs; job += p.grid_groups) {
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
            // Retain the production V-publish synchronization point, but no V
            // is loaded by this independently typed QK/softmax diagnostic.
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

            // Retain the tile-end barrier before reusing the K staging area.
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // Coordinate calibration gives one owner of column zero per row.
        // Write both state components to observable F32 output (no DCE).
        for (uint e = 0; e < 2; ++e) {
            ulong q = q_start + sid * 8 + rows[e];
            if (q < p.query_sequence && columns[e] == 0) {
                ulong row = (ulong(batch) * p.query_sequence + q) * p.query_heads + q_head;
                float poison = as_type<float>(0x7fc00000u);
                output[row * 2] = poisoned[e] ? poison : row_max[e];
                output[row * 2 + 1] = poisoned[e] ? poison : row_sum[e];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
