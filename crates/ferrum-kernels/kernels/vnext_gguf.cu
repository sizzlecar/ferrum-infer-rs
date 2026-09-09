// Native GGUF layouts follow ggml. Copyright (c) 2023-2026 The ggml authors.
// MIT license: ../src/gguf_blocks/LICENSE.ggml. IQ tables share Rust sources.
#include <cuda_fp16.h>
#include <math.h>
#include "vnext_gguf_codebooks.cuh"

using byte = unsigned char;

__device__ __forceinline__ float native_half(const byte* b, unsigned offset) {
    return __half2float(__ushort_as_half(
        static_cast<unsigned short>(b[offset] | (unsigned(b[offset + 1]) << 8))));
}

__device__ __forceinline__ float native_block_value(const byte* b, unsigned i, unsigned format) {
    switch (format) {
        case 11: {
            const unsigned group = i / 16;
            const unsigned lo = (b[96 + group % 8] >> (4 * (group / 8))) & 15;
            const unsigned hi = (b[104 + group % 4] >> (2 * (group / 4))) & 3;
            const int scale = int(lo | (hi << 4)) - 32;
            const unsigned q = (b[32 + (i / 128) * 32 + i % 32] >> (2 * ((i % 128) / 32))) & 3;
            const int offset = (b[i % 32] & (1 << (i / 32))) ? 0 : 4;
            return (native_half(b, 108) * float(scale)) * float(int(q) - offset);
        }
        case 12:
        case 13: {
            const unsigned group = i / 32;
            const byte* s = b + 4;
            const unsigned scale = group < 4 ? s[group] & 63
                : (s[group + 4] & 15) | ((s[group - 4] >> 6) << 4);
            const unsigned minimum = group < 4 ? s[group + 4] & 63
                : (s[group + 4] >> 4) | ((s[group] >> 6) << 4);
            const unsigned start = format == 13 ? 48 : 16;
            unsigned q = (b[start + (i / 64) * 32 + i % 32] >> (4 * ((i % 64) / 32))) & 15;
            if (format == 13 && (b[16 + i % 32] & (1 << group))) q += 16;
            return (native_half(b, 0) * float(scale)) * float(q) - native_half(b, 2) * float(minimum);
        }
        case 14: {
            const unsigned group = (i % 128) / 32;
            const unsigned lo = (b[(i / 128) * 64 + (group % 2) * 32 + i % 32] >> (4 * (group / 2))) & 15;
            const unsigned hi = (b[128 + (i / 128) * 32 + i % 32] >> (2 * group)) & 3;
            return (native_half(b, 208) * float(static_cast<signed char>(b[192 + i / 16])))
                * float(int(lo | (hi << 4)) - 32);
        }
        case 8:
            return native_half(b, 0) * float(static_cast<signed char>(b[2 + i]));
        case 20: {
            const unsigned q = (b[2 + i % 16] >> (4 * (i / 16))) & 15;
            return native_half(b, 0) * float(iq4_nl_values[q]);
        }
        case 23: {
            const unsigned group = i / 32;
            const unsigned scales_h = unsigned(b[2]) | (unsigned(b[3]) << 8);
            const unsigned lo = (b[4 + group / 2] >> (4 * (group % 2))) & 15;
            const unsigned hi = (scales_h >> (2 * group)) & 3;
            const int scale = int(lo | (hi << 4)) - 32;
            const unsigned q = (b[8 + group * 16 + i % 16] >> (4 * ((i % 32) / 16))) & 15;
            return (native_half(b, 0) * float(scale)) * float(iq4_nl_values[q]);
        }
        case 21: {
            const unsigned group = i / 32;
            const unsigned lo = b[2 + i / 4];
            const unsigned hi = (b[66 + group] >> ((i % 32) / 4)) & 1;
            const float q = float((iq3_s_grid[lo | (hi << 8)] >> (8 * (i % 4))) & 255);
            const unsigned scale = 1 + 2 * ((b[106 + group / 2] >> (4 * (group % 2))) & 15);
            const float sign = (b[74 + i / 8] & (1 << (i % 8))) ? -1.0f : 1.0f;
            return ((native_half(b, 0) * float(scale)) * q) * sign;
        }
    }
    return NAN;
}

template<typename T>
__device__ void native_linear(const T* x, const byte* w, T* y,
    unsigned rows, unsigned inputs, unsigned outputs, unsigned stride, unsigned offset,
    unsigned format, unsigned block_values, unsigned block_bytes) {
    const unsigned row = blockIdx.y;
    const unsigned lane = threadIdx.x % 32;
    const unsigned column = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= rows || column >= outputs) return;
    const size_t row_bytes = size_t(inputs / block_values) * block_bytes;
    float sum = 0.0f;
    for (unsigned i = lane; i < inputs; i += 32) {
        const byte* block = w + size_t(column) * row_bytes + size_t(i / block_values) * block_bytes;
        sum += float(x[size_t(row) * inputs + i]) * native_block_value(block, i % block_values, format);
    }
    for (unsigned step = 16; step != 0; step /= 2) sum += __shfl_down_sync(0xffffffff, sum, step);
    if (lane == 0) y[size_t(row) * stride + offset + column] = T(sum);
}

template<typename T>
__device__ void native_embedding(const unsigned* tokens, const byte* w, T* y,
    unsigned count, unsigned width, unsigned vocabulary,
    unsigned format, unsigned block_values, unsigned block_bytes) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(count) * width) return;
    const unsigned token = tokens[index / width];
    if (token >= vocabulary) { y[index] = T(NAN); return; }
    const unsigned col = index % width;
    const size_t block = size_t(token) * (width / block_values) + col / block_values;
    y[index] = T(native_block_value(w + block * block_bytes, col % block_values, format));
}

#define NATIVE_LINEAR(T, suffix) \
extern "C" __global__ void vnext_gguf_linear_##suffix(const T* x, const byte* w, T* y, \
    unsigned rows, unsigned inputs, unsigned outputs, unsigned stride, unsigned offset, \
    unsigned format, unsigned values, unsigned bytes) { \
    native_linear(x, w, y, rows, inputs, outputs, stride, offset, format, values, bytes); \
}
NATIVE_LINEAR(half, f16)
NATIVE_LINEAR(float, f32)

#define NATIVE_EMBEDDING(T, suffix) \
extern "C" __global__ void vnext_gguf_embedding_##suffix(const unsigned* tokens, const byte* w, T* y, \
    unsigned count, unsigned width, unsigned vocabulary, unsigned format, unsigned values, unsigned bytes) { \
    native_embedding(tokens, w, y, count, width, vocabulary, format, values, bytes); \
}
NATIVE_EMBEDDING(half, f16)
NATIVE_EMBEDDING(float, f32)

extern "C" __global__ void vnext_gguf_decode(const byte* input, float* output,
    unsigned count, unsigned format, unsigned values, unsigned bytes) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = native_block_value(input + size_t(i / values) * bytes, i % values, format);
}
