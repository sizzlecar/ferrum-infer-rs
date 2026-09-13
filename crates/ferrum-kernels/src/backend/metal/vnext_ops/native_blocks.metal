// Native GGUF block layouts follow ggml. Copyright (c) 2023-2026 The ggml
// authors. MIT license: ../../../gguf_blocks/LICENSE.ggml.
// IQ tables are injected from the same source constants as the CPU decoder.

struct NativeBlockParams { uint format; uint values; uint bytes; };
struct NativeLinearParams {
    uint rows; uint in_features; uint out_features;
    uint output_stride; uint output_column_offset;
};

static inline float native_half(device const uchar * b, uint offset) {
    const ushort bits = ushort(b[offset]) | (ushort(b[offset + 1]) << 8);
    return float(as_type<half>(bits));
}

static inline uint2 native_scale_min(device const uchar * s, uint group) {
    if (group < 4) return uint2(s[group] & 63, s[group + 4] & 63);
    return uint2((s[group + 4] & 15) | ((s[group - 4] >> 6) << 4),
        (s[group + 4] >> 4) | ((s[group] >> 6) << 4));
}

static inline float native_block_value(device const uchar * b, uint i, uint format) {
    switch (format) {
        case 11: {
            const uint group = i / 16;
            const uint lo = (b[96 + group % 8] >> (4 * (group / 8))) & 15;
            const uint hi = (b[104 + group % 4] >> (2 * (group / 4))) & 3;
            const int scale = int(lo | (hi << 4)) - 32;
            const uint q = (b[32 + (i / 128) * 32 + i % 32] >> (2 * ((i % 128) / 32))) & 3;
            const int offset = (b[i % 32] & (1 << (i / 32))) ? 0 : 4;
            return (native_half(b, 108) * float(scale)) * float(int(q) - offset);
        }
        case 12:
        case 13: {
            const uint group = i / 32;
            const uint2 sm = native_scale_min(b + 4, group);
            const uint start = format == 13 ? 48 : 16;
            uint q = (b[start + (i / 64) * 32 + i % 32] >> (4 * ((i % 64) / 32))) & 15;
            if (format == 13 && (b[16 + i % 32] & (1 << group))) q += 16;
            return (native_half(b, 0) * float(sm.x)) * float(q) - native_half(b, 2) * float(sm.y);
        }
        case 14: {
            const uint group = (i % 128) / 32;
            const uint lo = (b[(i / 128) * 64 + (group % 2) * 32 + i % 32] >> (4 * (group / 2))) & 15;
            const uint hi = (b[128 + (i / 128) * 32 + i % 32] >> (2 * group)) & 3;
            return (native_half(b, 208) * float(as_type<char>(b[192 + i / 16]))) * float(int(lo | (hi << 4)) - 32);
        }
        case 8: return native_half(b, 0) * float(as_type<char>(b[2 + i]));
        case 20: {
            const uint q = (b[2 + i % 16] >> (4 * (i / 16))) & 15;
            return native_half(b, 0) * float(iq4_nl_values[q]);
        }
        case 23: {
            const uint group = i / 32;
            const uint scales_h = uint(b[2]) | (uint(b[3]) << 8);
            const uint lo = (b[4 + group / 2] >> (4 * (group % 2))) & 15;
            const uint hi = (scales_h >> (2 * group)) & 3;
            const int scale = int(lo | (hi << 4)) - 32;
            const uint q = (b[8 + group * 16 + i % 16] >> (4 * ((i % 32) / 16))) & 15;
            return (native_half(b, 0) * float(scale)) * float(iq4_nl_values[q]);
        }
        case 21: {
            const uint group = i / 32;
            const uint lo = b[2 + i / 4];
            const uint hi = (b[66 + group] >> ((i % 32) / 4)) & 1;
            const float q = float((iq3_s_grid[lo | (hi << 8)] >> (8 * (i % 4))) & 255);
            const uint scale = 1 + 2 * ((b[106 + group / 2] >> (4 * (group % 2))) & 15);
            const float sign = (b[74 + i / 8] & (1 << (i % 8))) ? -1.0f : 1.0f;
            return ((native_half(b, 0) * float(scale)) * q) * sign;
        }
    }
    return NAN;
}

template<typename T>
static inline void native_linear(
    device const T * input, device const uchar * weight, device T * output,
    constant NativeLinearParams & p, constant NativeBlockParams & block,
    uint3 group, uint lane, uint subgroup) {
    const uint row = group.y;
    const uint first = group.x * 4 + subgroup * 2;
    const uint blocks_per_row = p.in_features / block.values;
    float sums[2] = {0.0f, 0.0f};
    for (uint col = lane; col < p.in_features; col += 32) {
        const float x = float(input[ulong(row) * p.in_features + col]);
        for (uint part = 0; part < 2; ++part) {
            const uint out_col = first + part;
            if (out_col < p.out_features) {
                const ulong offset = (ulong(out_col) * blocks_per_row + col / block.values) * block.bytes;
                sums[part] += x * native_block_value(weight + offset, col % block.values, block.format);
            }
        }
    }
    for (uint part = 0; part < 2; ++part) {
        const float value = simd_sum(sums[part]);
        const uint out_col = first + part;
        if (lane == 0 && out_col < p.out_features) {
            output[ulong(row) * p.output_stride + p.output_column_offset + out_col] = T(value);
        }
    }
}

kernel void vnext_native_block_linear_f16(
    device const half * x [[buffer(0)]], device const uchar * w [[buffer(1)]], device half * y [[buffer(2)]],
    constant NativeLinearParams & p [[buffer(3)]], constant NativeBlockParams & b [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint subgroup [[simdgroup_index_in_threadgroup]]) {
    native_linear(x, w, y, p, b, group, lane, subgroup);
}

kernel void vnext_native_block_linear_f32(
    device const float * x [[buffer(0)]], device const uchar * w [[buffer(1)]], device float * y [[buffer(2)]],
    constant NativeLinearParams & p [[buffer(3)]], constant NativeBlockParams & b [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint subgroup [[simdgroup_index_in_threadgroup]]) {
    native_linear(x, w, y, p, b, group, lane, subgroup);
}

// One 32-token x 64-output tile. Decode directly into float so the native
// weight values do not acquire a half rounding before multiplication.
// MMA changes the reduction grouping relative to native_linear's lane sums.
kernel void vnext_native_block_gemm_f16_f32(
    device const half * input [[buffer(0)]],
    device const uchar * weight [[buffer(1)]],
    device half * output [[buffer(2)]],
    constant NativeLinearParams & p [[buffer(3)]],
    constant NativeBlockParams & block [[buffer(4)]],
    threadgroup float * workspace [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]) {
    threadgroup float * input_tile = workspace;
    threadgroup float * weight_tile = workspace + 32 * 32;
    const ulong input_start = ulong(group.x) * 32;
    const ulong output_start = ulong(group.y) * 64;
    const ulong blocks_per_row = ulong(p.in_features) / block.values;
    const uint matrix_row = (simdgroup_index / 2) * 16;
    const uint matrix_column = (simdgroup_index % 2) * 32;
    simdgroup_float8x8 accumulators[8];
    for (uint i = 0; i < 8; ++i) {
        accumulators[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (ulong k = 0; k < ulong(p.in_features); k += 32) {
        for (uint i = thread_index; i < 32 * 32; i += 128) {
            const ulong row = input_start + i / 32;
            const ulong column = k + i % 32;
            input_tile[i] = row < ulong(p.rows) && column < ulong(p.in_features)
                ? float(input[row * ulong(p.in_features) + column]) : 0.0f;
        }
        // The physical weight is output-major; the shared tile is K x N.
        // The decoder supports both 32-value and 256-value native blocks.
        for (uint i = thread_index; i < 32 * 64; i += 128) {
            // Neighboring lanes decode consecutive K values of one source row.
            const uint local_row = i / 32;
            const uint local_column = i % 32;
            const ulong row = output_start + local_row;
            const ulong column = k + local_column;
            float value = 0.0f;
            if (row < ulong(p.out_features) && column < ulong(p.in_features)) {
                const ulong offset = (row * blocks_per_row + column / block.values)
                    * ulong(block.bytes);
                value = native_block_value(
                    weight + offset, uint(column % block.values), block.format);
            }
            weight_tile[local_column * 64 + local_row] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint chunk = 0; chunk < 32; chunk += 8) {
            simdgroup_float8x8 input_matrices[2];
            simdgroup_float8x8 weight_matrices[4];
            for (uint m = 0; m < 2; ++m) {
                simdgroup_load(input_matrices[m],
                    input_tile + (matrix_row + m * 8) * 32 + chunk, 32, 0, false);
            }
            for (uint n = 0; n < 4; ++n) {
                simdgroup_load(weight_matrices[n],
                    weight_tile + chunk * 64 + matrix_column + n * 8, 64, 0, false);
            }
            for (uint m = 0; m < 2; ++m) {
                for (uint n = 0; n < 4; ++n) {
                    simdgroup_multiply_accumulate(accumulators[m * 4 + n],
                        input_matrices[m], weight_matrices[n], accumulators[m * 4 + n]);
                }
            }
        }
        // No thread can overwrite the next K tile while another SIMD reads it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // All K tiles are consumed. Reuse the 8 KiB weight tile for the 32 x 64
    // float result; every SIMD owns a disjoint 16 x 32 rectangle.
    for (uint m = 0; m < 2; ++m) {
        for (uint n = 0; n < 4; ++n) {
            simdgroup_store(accumulators[m * 4 + n],
                weight_tile + (matrix_row + m * 8) * 64 + matrix_column + n * 8,
                64, 0, false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = thread_index; i < 32 * 64; i += 128) {
        const ulong row = input_start + i / 64;
        const ulong column = output_start + i % 64;
        if (row < ulong(p.rows) && column < ulong(p.out_features)) {
            output[row * ulong(p.output_stride) + ulong(p.output_column_offset) + column]
                = half(weight_tile[i]);
        }
    }
}

kernel void vnext_native_block_decode(
    device const uchar * input [[buffer(0)]], device float * output [[buffer(1)]],
    constant NativeBlockParams & b [[buffer(2)]], constant uint & count [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i < count) output[i] = native_block_value(input + ulong(i / b.values) * b.bytes, i % b.values, b.format);
}
