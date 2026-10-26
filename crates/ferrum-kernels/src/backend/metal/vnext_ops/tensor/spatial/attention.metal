#include <metal_simdgroup_matrix>
struct SpatialAttentionParams { ulong jobs, grid_groups, query_tiles, spatial; uint channels; float scale; };
// The 16x32 score tile stays on chip. The C-wide online numerator lives in
// core-admitted invocation scratch, allowing C up to 2048 without S*S storage
// or an on-chip allocation proportional to 16*C. Each query has one owner.
kernel void ferrum_tensor_spatial_attention_bf16(
    device const ushort* qkv [[buffer(0)]], device float* accum [[buffer(1)]],
    device ushort* output [[buffer(2)]], constant SpatialAttentionParams& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float workspace[2112];
    threadgroup float* a = workspace;
    threadgroup float* b = workspace + 512;
    threadgroup float* scores = workspace + 1536;
    threadgroup float* row_max = workspace + 2048;
    threadgroup float* row_sum = row_max + 16;
    threadgroup float* alpha = row_sum + 16;
    threadgroup uint* poisoned = reinterpret_cast<threadgroup uint*>(alpha + 16);
    uint local_row = (sid / 2) * 8, local_col = (sid % 2) * 16;
    for (ulong job = group; job < p.jobs; job += p.grid_groups) {
        ulong q_start = (job % p.query_tiles) * 16, batch = job / p.query_tiles;
        ulong q_base = batch * 3 * p.channels * p.spatial;
        ulong k_base = q_base + p.channels * p.spatial, v_base = k_base + p.channels * p.spatial;
        ulong accum_base = (batch * p.spatial + q_start) * p.channels;
        if (tid < 16) { row_max[tid] = -INFINITY; row_sum[tid] = 0.0f; alpha[tid] = 1.0f; poisoned[tid] = 0; }
        for (uint i = tid; i < 16 * p.channels; i += 128)
            if (q_start + i / p.channels < p.spatial) accum[accum_base + i] = 0.0f;
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        for (ulong k_start = 0; k_start < p.spatial; k_start += 32) {
            simdgroup_float8x8 dot[2];
            for (uint i = 0; i < 2; ++i) dot[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            for (uint d = 0; d < p.channels; d += 32) {
                for (uint i = tid; i < 512; i += 128) {
                    uint r = i / 32, c = d + i % 32;
                    a[i] = q_start + r < p.spatial && c < p.channels && !poisoned[r]
                        ? decode_bf16(qkv[q_base + ulong(c) * p.spatial + q_start + r]) : 0.0f;
                }
                for (uint i = tid; i < 1024; i += 128) {
                    ulong k = k_start + i % 32; uint c = d + i / 32;
                    b[i] = k < p.spatial && c < p.channels ? decode_bf16(qkv[k_base + ulong(c) * p.spatial + k]) : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint chunk = 0; chunk < 32; chunk += 8) {
                    simdgroup_float8x8 av, bv;
                    simdgroup_load(av, a + local_row * 32 + chunk, 32, 0, false);
                    for (uint n = 0; n < 2; ++n) {
                        simdgroup_load(bv, b + chunk * 32 + local_col + n * 8, 32, 0, false);
                        simdgroup_multiply_accumulate(dot[n], av, bv, dot[n]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint n = 0; n < 2; ++n) simdgroup_store(dot[n], scores + local_row * 32 + local_col + n * 8, 32, 0, false);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Cooperatively reduce one spatial row per SIMD group, retaining
            // FP32 probabilities and the existing managed accumulator.
            const uint lane = tid % 32;
            for (uint r = sid; r < 16; r += 4) {
                bool visible = q_start + r < p.spatial && k_start + lane < p.spatial;
                float score = visible ? scores[r * 32 + lane] * p.scale : 0.0f;
                bool bad = simd_any(visible && !isfinite(score)) || poisoned[r];
                float maximum = max(row_max[r], simd_max(visible && isfinite(score) ? score : -INFINITY));
                float rescale = 1.0f;
                if (q_start + r < p.spatial && !bad)
                    rescale = row_sum[r] == 0.0f ? 0.0f : exp(row_max[r] - maximum);
                float probability = visible && !bad ? exp(score - maximum) : 0.0f;
                scores[r * 32 + lane] = probability;
                float total = simd_sum(probability);
                if (lane == 0) {
                    poisoned[r] = uint(bad);
                    alpha[r] = rescale;
                    row_sum[r] = row_sum[r] * rescale + total;
                    row_max[r] = maximum;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint d = 0; d < p.channels; d += 32) {
                for (uint i = tid; i < 1024; i += 128) {
                    ulong k = k_start + i / 32; uint c = d + i % 32;
                    b[i] = k < p.spatial && c < p.channels ? decode_bf16(qkv[v_base + ulong(c) * p.spatial + k]) : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_float8x8 pv[2];
                for (uint n = 0; n < 2; ++n) pv[n] = make_filled_simdgroup_matrix<float, 8>(0.0f);
                for (uint chunk = 0; chunk < 32; chunk += 8) {
                    simdgroup_float8x8 av, bv;
                    simdgroup_load(av, scores + local_row * 32 + chunk, 32, 0, false);
                    for (uint n = 0; n < 2; ++n) {
                        simdgroup_load(bv, b + chunk * 32 + local_col + n * 8, 32, 0, false);
                        simdgroup_multiply_accumulate(pv[n], av, bv, pv[n]);
                    }
                }
                for (uint n = 0; n < 2; ++n) simdgroup_store(pv[n], a + local_row * 32 + local_col + n * 8, 32, 0, false);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = tid; i < 512; i += 128) {
                    uint r = i / 32, c = d + i % 32;
                    if (q_start + r < p.spatial && c < p.channels) {
                        ulong index = accum_base + ulong(r) * p.channels + c;
                        accum[index] = accum[index] * alpha[r] + a[i];
                    }
                }
                threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            }
        }
        for (uint i = tid; i < 16 * p.channels; i += 128) {
            uint r = i / p.channels, c = i % p.channels;
            if (q_start + r < p.spatial) output[(batch * p.channels + c) * p.spatial + q_start + r]
                = poisoned[r] ? ushort(0x7fc0) : encode_bf16(accum[accum_base + i] * (1.0f / row_sum[r]));
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}
