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

kernel void vnext_native_block_decode(
    device const uchar * input [[buffer(0)]], device float * output [[buffer(1)]],
    constant NativeBlockParams & b [[buffer(2)]], constant uint & count [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i < count) output[i] = native_block_value(input + ulong(i / b.values) * b.bytes, i % b.values, b.format);
}
