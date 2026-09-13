// Native GGUF block layouts follow ggml. Copyright (c) 2023-2026 The ggml
// authors. MIT license: ../../../gguf_blocks/LICENSE.ggml.
// IQ tables are injected from the same source constants as the CPU decoder.

struct NativeBlockParams { uint format; uint values; uint bytes; };
struct NativeLinearParams {
    uint rows; uint in_features; uint out_features;
    uint output_stride; uint output_column_offset;
};

// Zero retains the runtime decoder only for test PSOs. Production GEMV PSOs
// bind a supported GgufBlockFormat; prefill and block decoding do not use this.
constant uint NATIVE_GEMV_FORMAT [[function_constant(0)]];

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

// Decode one aligned 16-value fragment directly into the unchanged K x N tile.
// A fragment stays within one scale group for these four native formats. Keep
// scale products and IQ3 signs in the scalar decoder's FP32 evaluation order.
// Byte loads also support the two-byte-aligned 110- and 18-byte block layouts.
static inline void native_gemm_fragment16(
    device const uchar * b, uint first, uint format, threadgroup float * tile) {
    switch (format) {
        case 11: {
            const uint group = first / 16;
            const uint lo = (b[96 + group % 8] >> (4 * (group / 8))) & 15;
            const uint hi = (b[104 + group % 4] >> (2 * (group / 4))) & 3;
            const float scale = native_half(b, 108) * float(int(lo | (hi << 4)) - 32);
            const uint q_base = 32 + (first / 128) * 32 + first % 32;
            const uint q_shift = 2 * ((first % 128) / 32);
            const uint h_base = first % 32;
            const uint h_mask = 1 << (first / 32);
            for (uint j = 0; j < 16; ++j) {
                const uint q = (b[q_base + j] >> q_shift) & 3;
                const int offset = (b[h_base + j] & h_mask) ? 0 : 4;
                tile[j * 64] = scale * float(int(q) - offset);
            }
            return;
        }
        case 20:
        case 23: {
            float scale = native_half(b, 0);
            uint q_base = 2;
            if (format == 23) {
                const uint group = first / 32;
                const uint scales_h = uint(b[2]) | (uint(b[3]) << 8);
                const uint lo = (b[4 + group / 2] >> (4 * (group % 2))) & 15;
                const uint hi = (scales_h >> (2 * group)) & 3;
                scale = scale * float(int(lo | (hi << 4)) - 32);
                q_base = 8 + group * 16;
            }
            const uint shift = 4 * ((first % 32) / 16);
            for (uint j = 0; j < 16; j += 4) {
                const uint packed = uint(b[q_base + j])
                    | (uint(b[q_base + j + 1]) << 8)
                    | (uint(b[q_base + j + 2]) << 16)
                    | (uint(b[q_base + j + 3]) << 24);
                for (uint component = 0; component < 4; ++component) {
                    const uint q = (packed >> (8 * component + shift)) & 15;
                    tile[(j + component) * 64] = scale * float(iq4_nl_values[q]);
                }
            }
            return;
        }
        case 21: {
            const uint group = first / 32;
            const uint packed_high = b[66 + group];
            const uint scale_bits = 1 + 2 * ((b[106 + group / 2] >> (4 * (group % 2))) & 15);
            const float scale = native_half(b, 0) * float(scale_bits);
            for (uint j = 0; j < 16; j += 4) {
                const uint i = first + j;
                const uint lo = b[2 + i / 4];
                const uint hi = (packed_high >> ((i % 32) / 4)) & 1;
                const uint grid = iq3_s_grid[lo | (hi << 8)];
                const uint signs = b[74 + i / 8];
                for (uint component = 0; component < 4; ++component) {
                    const float q = float((grid >> (8 * component)) & 255);
                    const float sign = (signs & (1 << ((i + component) % 8))) ? -1.0f : 1.0f;
                    tile[(j + component) * 64] = (scale * q) * sign;
                }
            }
            return;
        }
        default:
            for (uint j = 0; j < 16; ++j) {
                tile[j * 64] = native_block_value(b, first + j, format);
            }
    }
}

// IQ4_XS has one half scale and six packed scale bytes per 256 values.
// Keep only this header live; each g still evaluates (d * scale) * q in FP32.
struct NativeIq4XsHeader { float d; uint scales_h; uint scales_l; };

static inline NativeIq4XsHeader native_iq4xs_header(device const uchar * b) {
    return {native_half(b, 0), uint(b[2]) | (uint(b[3]) << 8),
        uint(b[4]) | (uint(b[5]) << 8) | (uint(b[6]) << 16) | (uint(b[7]) << 24)};
}

static inline float native_iq4xs_lane_value(
    device const uchar * b, thread const NativeIq4XsHeader & header, uint g, uint lane) {
    const uint lo = (header.scales_l >> (4 * g)) & 15;
    const uint hi = (header.scales_h >> (2 * g)) & 3;
    const int scale = int(lo | (hi << 4)) - 32;
    const uint q = (b[8 + g * 16 + (lane & 15)] >> (4 * (lane >> 4))) & 15;
    return (header.d * float(scale)) * float(iq4_nl_values[q]);
}

template<typename T>
static inline void native_linear(
    device const T * input, device const uchar * weight, device T * output,
    constant NativeLinearParams & p, constant NativeBlockParams & block,
    uint3 group, uint lane, uint subgroup) {
    const uint row = group.y;
    const uint first = group.x * 4 + subgroup * 2;
    const uint format = NATIVE_GEMV_FORMAT == 0 ? block.format : NATIVE_GEMV_FORMAT;
    // NativeBlockParams admits only 32- or 256-value blocks. Keep the lane
    // traversal and ulong byte addresses unchanged while sharing this index
    // calculation between the two output columns.
    const uint block_shift = block.values == 256 ? 8 : 5;
    const uint blocks_per_row = p.in_features >> block_shift;
    float sums[2] = {0.0f, 0.0f};
    if (NATIVE_GEMV_FORMAT == 23) {
        // The typed IQ4_XS binding guarantees K % 256 == 0 and 136 bytes/block.
        // g=0..7 is exactly the old per-lane col+=32 order. In particular, keep
        // both outputs inside g so their input load remains shared.
        for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
            NativeIq4XsHeader headers[2] = {};
            #pragma clang loop unroll(full)
            for (uint part = 0; part < 2; ++part) {
                const uint out_col = first + part;
                if (out_col < p.out_features) {
                    const ulong offset = (ulong(out_col) * blocks_per_row + block_index) * block.bytes;
                    headers[part] = native_iq4xs_header(weight + offset);
                }
            }
            for (uint g = 0; g < 8; ++g) {
                const uint col = block_index * 256 + g * 32 + lane;
                const float x = float(input[ulong(row) * p.in_features + col]);
                #pragma clang loop unroll(full)
                for (uint part = 0; part < 2; ++part) {
                    const uint out_col = first + part;
                    if (out_col < p.out_features) {
                        const ulong offset = (ulong(out_col) * blocks_per_row + block_index) * block.bytes;
                        sums[part] += x * native_iq4xs_lane_value(weight + offset, headers[part], g, lane);
                    }
                }
            }
        }
    } else {
        // FORMAT0 remains the original scalar decoder, including IQ4_XS.
        for (uint col = lane; col < p.in_features; col += 32) {
            const float x = float(input[ulong(row) * p.in_features + col]);
            const uint block_index = col >> block_shift;
            const uint in_block = col & (block.values - 1);
            for (uint part = 0; part < 2; ++part) {
                const uint out_col = first + part;
                if (out_col < p.out_features) {
                    const ulong offset = (ulong(out_col) * blocks_per_row + block_index) * block.bytes;
                    sums[part] += x * native_block_value(weight + offset, in_block, format);
                }
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

// Decode one coefficient for all B independent rows without rounding the
// weight to half. The typed selector restricts this path to small batches.
// Each row retains native_linear's lane/column order and FP32 SIMD reduction.
template<typename T, ushort B>
static inline void native_shared_linear(
    device const T * input, device const uchar * weight, device T * output,
    constant NativeLinearParams & p, constant NativeBlockParams & block,
    uint group, uint lane, uint subgroup) {
    const uint first = group * 4 + subgroup * 2;
    const uint block_shift = block.values == 256 ? 8 : 5;
    const uint blocks_per_row = p.in_features >> block_shift;
    // Complete each independent output before starting the next, keeping only
    // B independent accumulator variables live across the column loop.
    #pragma clang loop unroll(disable)
    for (ushort part = 0; part < 2; ++part) {
        const uint out_col = first + part;
        if (out_col >= p.out_features) continue;
        float sums[B] = {};
        if (NATIVE_GEMV_FORMAT == 23) {
            for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
                const ulong offset = (ulong(out_col) * blocks_per_row + block_index) * block.bytes;
                device const uchar * b = weight + offset;
                const NativeIq4XsHeader header = native_iq4xs_header(b);
                for (uint g = 0; g < 8; ++g) {
                    const uint col = block_index * 256 + g * 32 + lane;
                    const float w = native_iq4xs_lane_value(b, header, g, lane);
                    #pragma clang loop unroll(full)
                    for (ushort batch = 0; batch < B; ++batch) {
                        const float x = float(input[ulong(batch) * p.in_features + col]);
                        sums[batch] += x * w;
                    }
                }
            }
        } else {
            for (uint col = lane; col < p.in_features; col += 32) {
                const uint block_index = col >> block_shift;
                const uint in_block = col & (block.values - 1);
                const ulong offset = (ulong(out_col) * blocks_per_row + block_index) * block.bytes;
                const float w = native_block_value(weight + offset, in_block, NATIVE_GEMV_FORMAT);
                #pragma clang loop unroll(full)
                for (ushort batch = 0; batch < B; ++batch) {
                    const float x = float(input[ulong(batch) * p.in_features + col]);
                    sums[batch] += x * w;
                }
            }
        }
        #pragma clang loop unroll(full)
        for (ushort batch = 0; batch < B; ++batch) {
            const float value = simd_sum(sums[batch]);
            if (lane == 0) {
                output[ulong(batch) * p.output_stride + p.output_column_offset + out_col] = T(value);
            }
        }
    }
}

#define NATIVE_SHARED_LINEAR(T, SUFFIX, B) \
kernel void vnext_native_shared_linear_##SUFFIX##_b##B( \
    device const T * x [[buffer(0)]], device const uchar * w [[buffer(1)]], device T * y [[buffer(2)]], \
    constant NativeLinearParams & p [[buffer(3)]], constant NativeBlockParams & b [[buffer(4)]], \
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint subgroup [[simdgroup_index_in_threadgroup]]) { \
    if (p.rows != B) return; \
    native_shared_linear<T, B>(x, w, y, p, b, group.x, lane, subgroup); \
}
NATIVE_SHARED_LINEAR(half, f16, 2)
NATIVE_SHARED_LINEAR(half, f16, 3)
NATIVE_SHARED_LINEAR(half, f16, 4)
NATIVE_SHARED_LINEAR(float, f32, 2)
NATIVE_SHARED_LINEAR(float, f32, 3)
NATIVE_SHARED_LINEAR(float, f32, 4)
#undef NATIVE_SHARED_LINEAR

// A 32- or 64-token x 64-output tile. Decode directly into float so the native
// weight values do not acquire a half rounding before multiplication.
// MMA changes the reduction grouping relative to native_linear's lane sums.
template<uint ROW_TILE>
static inline void native_tiled_gemm(
    device const half * input, device const uchar * weight, device half * output,
    constant NativeLinearParams & p, constant NativeBlockParams & block,
    threadgroup float * workspace, uint3 group, uint thread_index, uint simdgroup_index) {
    constexpr uint THREADS = ROW_TILE * 4;
    threadgroup float * input_tile = workspace;
    threadgroup float * weight_tile = workspace + ROW_TILE * 32;
    const ulong input_start = ulong(group.x) * ROW_TILE;
    const ulong output_start = ulong(group.y) * 64;
    // Native GGUF blocks contain 32 or 256 values. A K32 tile cannot cross
    // a block boundary, so compute its block address once per K iteration.
    const uint block_shift = block.values == 256 ? 8 : 5;
    const ulong blocks_per_row = ulong(p.in_features >> block_shift);
    const uint matrix_row = (simdgroup_index / 2) * 16;
    const uint matrix_column = (simdgroup_index % 2) * 32;
    simdgroup_float8x8 accumulators[8];
    for (uint i = 0; i < 8; ++i) {
        accumulators[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (ulong k = 0; k < ulong(p.in_features); k += 32) {
        const ulong block_index = ulong(uint(k) >> block_shift);
        const uint in_block_base = uint(k) & (block.values - 1);
        for (uint i = thread_index; i < ROW_TILE * 32; i += THREADS) {
            const ulong row = input_start + i / 32;
            const ulong column = k + i % 32;
            input_tile[i] = row < ulong(p.rows) && column < ulong(p.in_features)
                ? float(input[row * ulong(p.in_features) + column]) : 0.0f;
        }
        // Two threads own the two 16-value fragments of each physical row.
        // Reuse decoded headers within a thread while keeping the K x N tile.
        // Additional M64 SIMD groups reuse this same weight tile. Every thread
        // still reaches both barriers, including those without a weight load.
        if (thread_index < 128) {
            const uint local_row = thread_index / 2;
            const uint local_column = (thread_index % 2) * 16;
            const ulong row = output_start + local_row;
            const ulong column = k + local_column;
            threadgroup float * weight_fragment = weight_tile + local_column * 64 + local_row;
            if (row < ulong(p.out_features) && column + 16 <= ulong(p.in_features)) {
                const ulong offset = (row * blocks_per_row + block_index) * ulong(block.bytes);
                native_gemm_fragment16(weight + offset, in_block_base + local_column,
                    block.format, weight_fragment);
            } else {
                // Keep every consumed tile element initialized, including tails.
                for (uint j = 0; j < 16; ++j) {
                    float value = 0.0f;
                    if (row < ulong(p.out_features) && column + j < ulong(p.in_features)) {
                        const ulong offset = (row * blocks_per_row + block_index) * ulong(block.bytes);
                        value = native_block_value(weight + offset,
                            in_block_base + local_column + j, block.format);
                    }
                    weight_fragment[j * 64] = value;
                }
            }
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

    // The last K barrier retires all reads. M32 retains its 8 KiB weight-tile
    // reuse; M64 needs the entire 16 KiB workspace for its 64 x 64 result.
    // Every SIMD owns a disjoint 16 x 32 rectangle in either case.
    threadgroup float * result_tile = ROW_TILE == 32 ? weight_tile : workspace;
    for (uint m = 0; m < 2; ++m) {
        for (uint n = 0; n < 4; ++n) {
            simdgroup_store(accumulators[m * 4 + n],
                result_tile + (matrix_row + m * 8) * 64 + matrix_column + n * 8,
                64, 0, false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = thread_index; i < ROW_TILE * 64; i += THREADS) {
        const ulong row = input_start + i / 64;
        const ulong column = output_start + i % 64;
        if (row < ulong(p.rows) && column < ulong(p.out_features)) {
            output[row * ulong(p.output_stride) + ulong(p.output_column_offset) + column]
                = half(result_tile[i]);
        }
    }
}

#define NATIVE_TILED_GEMM(NAME, ROW_TILE) \
kernel void NAME( \
    device const half * input [[buffer(0)]], device const uchar * weight [[buffer(1)]], \
    device half * output [[buffer(2)]], constant NativeLinearParams & p [[buffer(3)]], \
    constant NativeBlockParams & block [[buffer(4)]], threadgroup float * workspace [[threadgroup(0)]], \
    uint3 group [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]], \
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]) { \
    native_tiled_gemm<ROW_TILE>(input, weight, output, p, block, workspace, group, thread_index, simdgroup_index); \
}

NATIVE_TILED_GEMM(vnext_native_block_gemm_f16_f32, 32)
NATIVE_TILED_GEMM(vnext_native_block_gemm_f16_f32_m64, 64)
#undef NATIVE_TILED_GEMM

kernel void vnext_native_block_decode(
    device const uchar * input [[buffer(0)]], device float * output [[buffer(1)]],
    constant NativeBlockParams & b [[buffer(2)]], constant uint & count [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i < count) output[i] = native_block_value(input + ulong(i / b.values) * b.bytes, i % b.values, b.format);
}
