// Native GGUF layouts follow ggml; MIT license: ../../../gguf_blocks/LICENSE.ggml.
// Shared by token and fixed-tensor providers. IQ tables precede this source.
static inline float native_half(device const uchar * b, uint offset) {
    const ushort bits = ushort(b[offset]) | (ushort(b[offset + 1]) << 8);
    return float(as_type<half>(bits));
}

// The typed block ABI admits 32, 128, or 256 values per block.
static inline uint native_block_shift(uint values) {
    return values == 256 ? 8 : (values == 128 ? 7 : 5);
}

static inline uint2 native_scale_min(device const uchar * s, uint group) {
    if (group < 4) return uint2(s[group] & 63, s[group + 4] & 63);
    return uint2((s[group + 4] & 15) | ((s[group - 4] >> 6) << 4),
        (s[group + 4] >> 4) | ((s[group] >> 6) << 4));
}

static inline float native_block_value(device const uchar * b, uint i, uint format) {
    switch (format) {
        case 142: {
            const uint q = (b[2 + i / 4] >> (2 * (i % 4))) & 3;
            return native_half(b, 0) * float(int(q) - 1);
        }
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
