// Native GGML block layouts. Copyright (c) 2023-2026 The ggml authors.
// MIT license: ../src/gguf_blocks/LICENSE.ggml.
#pragma once
#include <cuda_fp16.h>
#include <math.h>

__device__ __forceinline__ float ferrum_gguf_half(const unsigned char* b, unsigned offset) {
    return __half2float(__ushort_as_half(
        static_cast<unsigned short>(b[offset] | (unsigned(b[offset + 1]) << 8))));
}

// Decode is shared by token-native and fixed-tensor providers. Callers bind
// the exact format, bytes/block and logical values/block before launching.
__device__ __forceinline__ float ferrum_gguf_matrix_value(const unsigned char* b, unsigned i, unsigned format) {
    switch (format) {
        case 12:
        case 13: {
            const unsigned group = i / 32;
            const unsigned char* s = b + 4;
            const unsigned scale = group < 4 ? s[group] & 63
                : (s[group + 4] & 15) | ((s[group - 4] >> 6) << 4);
            const unsigned minimum = group < 4 ? s[group + 4] & 63
                : (s[group + 4] >> 4) | ((s[group] >> 6) << 4);
            const unsigned start = format == 13 ? 48 : 16;
            unsigned q = (b[start + (i / 64) * 32 + i % 32] >> (4 * ((i % 64) / 32))) & 15;
            if (format == 13 && (b[16 + i % 32] & (1 << group))) q += 16;
            return (ferrum_gguf_half(b, 0) * float(scale)) * float(q) - ferrum_gguf_half(b, 2) * float(minimum);
        }
        case 14: {
            const unsigned group = (i % 128) / 32;
            const unsigned lo = (b[(i / 128) * 64 + (group % 2) * 32 + i % 32] >> (4 * (group / 2))) & 15;
            const unsigned hi = (b[128 + (i / 128) * 32 + i % 32] >> (2 * group)) & 3;
            return (ferrum_gguf_half(b, 208) * float(static_cast<signed char>(b[192 + i / 16])))
                * float(int(lo | (hi << 4)) - 32);
        }
        case 8:
            return ferrum_gguf_half(b, 0) * float(static_cast<signed char>(b[2 + i]));
    }
    return NAN;
}
