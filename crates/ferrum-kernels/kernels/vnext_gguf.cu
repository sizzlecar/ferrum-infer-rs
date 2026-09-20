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
        case 1:
            return native_half(b, 2 * i);
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
        case 142: {
            // PQ2_0: one F16 scale and 128 values packed low slot first.
            // All four physical codes are defined, including code 3 = +2.
            const unsigned q = (b[2 + i / 4] >> (2 * (i % 4))) & 3;
            return native_half(b, 0) * float(int(q) - 1);
        }
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

template<typename Input, typename Output, unsigned RowTile>
__device__ void native_linear(const Input* x, const byte* w, Output* y,
    unsigned rows, unsigned inputs, unsigned outputs, unsigned stride, unsigned offset,
    unsigned format, unsigned block_values, unsigned block_bytes) {
    const unsigned row = blockIdx.y * RowTile;
    const unsigned lane = threadIdx.x % 32;
    const unsigned column = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= rows || column >= outputs) return;
    const size_t row_bytes = size_t(inputs / block_values) * block_bytes;
    // Reuse each decoded weight across a bounded set of activation rows.
    // Each row retains the original lane-strided FP32 sum and warp reduction;
    // neither expanded weights nor a lower-precision accumulation is needed.
    float sums[RowTile] = {};
    for (unsigned i = lane; i < inputs; i += 32) {
        const byte* block = w + size_t(column) * row_bytes + size_t(i / block_values) * block_bytes;
        const float weight = native_block_value(block, i % block_values, format);
        #pragma unroll
        for (unsigned r = 0; r < RowTile; ++r) {
            if (row + r < rows) sums[r] += float(x[size_t(row + r) * inputs + i]) * weight;
        }
    }
    #pragma unroll
    for (unsigned r = 0; r < RowTile; ++r) {
        for (unsigned step = 16; step != 0; step /= 2)
            sums[r] += __shfl_down_sync(0xffffffff, sums[r], step);
        if (lane == 0 && row + r < rows)
            y[size_t(row + r) * stride + offset + column] = Output(sums[r]);
    }
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

// Q4_K keeps the same reconstruction and lane-strided FP32 accumulation as
// native_linear. Fix the physical block ABI at compilation so each coefficient
// does not require runtime block division/remainder and format selection.
template<unsigned RowTile>
__device__ void native_q4k_linear(const half* x, const byte* w, half* y,
    unsigned rows, unsigned inputs, unsigned outputs, unsigned stride, unsigned offset) {
    const unsigned row = blockIdx.y * RowTile;
    const unsigned lane = threadIdx.x % 32;
    const unsigned column = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= rows || column >= outputs) return;
    const unsigned blocks = inputs / 256;
    const byte* weights = w + size_t(column) * blocks * 144;
    float sums[RowTile] = {};
    for (unsigned block = 0; block < blocks; ++block) {
        const byte* source = weights + size_t(block) * 144;
        #pragma unroll
        for (unsigned group = 0; group < 8; ++group) {
            const unsigned local = group * 32 + lane;
            const unsigned i = block * 256 + local;
            const float weight = native_block_value(source, local, 12);
            #pragma unroll
            for (unsigned r = 0; r < RowTile; ++r) {
                if (row + r < rows) sums[r] += float(x[size_t(row + r) * inputs + i]) * weight;
            }
        }
    }
    #pragma unroll
    for (unsigned r = 0; r < RowTile; ++r) {
        for (unsigned step = 16; step != 0; step /= 2)
            sums[r] += __shfl_down_sync(0xffffffff, sums[r], step);
        if (lane == 0 && row + r < rows)
            y[size_t(row + r) * stride + offset + column] = half(sums[r]);
    }
}

#define NATIVE_LINEAR(Input, Output, suffix, tile) \
extern "C" __global__ void vnext_gguf_linear_##suffix(const Input* x, const byte* w, Output* y, \
    unsigned rows, unsigned inputs, unsigned outputs, unsigned stride, unsigned offset, \
    unsigned format, unsigned values, unsigned bytes) { \
    native_linear<Input, Output, tile>(x, w, y, rows, inputs, outputs, stride, offset, format, values, bytes); \
}
NATIVE_LINEAR(half, half, f16, 1)
NATIVE_LINEAR(float, float, f32, 1)
NATIVE_LINEAR(half, half, tiled_f16, 8)
NATIVE_LINEAR(float, float, tiled_f32, 8)
NATIVE_LINEAR(float, half, f32_f16, 1)
NATIVE_LINEAR(float, half, tiled_f32_f16, 8)

// Keep the generic exports available for all other formats/precision modes,
// and as an independent control for specialized-kernel conformance and timing.
#define NATIVE_Q4K_LINEAR(suffix, tile) \
extern "C" __global__ void vnext_gguf_linear_q4k_##suffix(const half* x, const byte* w, half* y, \
    unsigned rows, unsigned inputs, unsigned outputs, unsigned stride, unsigned offset, \
    unsigned format, unsigned values, unsigned bytes) { \
    if (format != 12 || values != 256 || bytes != 144 || inputs % 256 != 0) return; \
    native_q4k_linear<tile>(x, w, y, rows, inputs, outputs, stride, offset); \
}
NATIVE_Q4K_LINEAR(f16, 1)
NATIVE_Q4K_LINEAR(tiled_f16, 8)

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

// Every block owns one whole Sylvester block. All threads execute every barrier,
// including lanes beyond the transform width. The complete feature permutation
// precedes signs and H; inverse lookup applies signs after H instead.
template<typename Input, typename Output>
__device__ void native_hadamard(const Input* input, Output* output, const float* signs,
    unsigned width, unsigned block_size, unsigned inverse,
    unsigned inner, unsigned first, unsigned second) {
    extern __shared__ float values[];
    const size_t row = blockIdx.y;
    const unsigned base = blockIdx.x * block_size;
    for (unsigned i = threadIdx.x; i < block_size; i += blockDim.x) {
        const unsigned feature = base + i;
        unsigned source = feature;
        if (inner != 0) {
            const unsigned a = feature % inner;
            const unsigned b = (feature / inner) % second;
            const unsigned c = feature / (inner * second);
            source = a + inner * (c + first * b);
        }
        float value = float(input[row * width + source]);
        if (!inverse && signs) value *= signs[feature];
        values[i] = value;
    }
    __syncthreads();
    for (unsigned step = 1; step < block_size; step *= 2) {
        for (unsigned pair = threadIdx.x; pair < block_size / 2; pair += blockDim.x) {
            const unsigned lo = (pair / step) * (2 * step) + pair % step;
            const float a = values[lo];
            const float b = values[lo + step];
            values[lo] = a + b;
            values[lo + step] = a - b;
        }
        __syncthreads();
    }
    const float scale = 1.0f / sqrtf(float(block_size));
    for (unsigned i = threadIdx.x; i < block_size; i += blockDim.x) {
        float value = values[i] * scale;
        if (inverse && signs) value *= signs[base + i];
        output[row * width + base + i] = Output(value);
    }
}

#define NATIVE_HADAMARD(Input, Output, suffix) \
extern "C" __global__ void vnext_gguf_hadamard_##suffix( \
    const Input* input, Output* output, const float* signs, unsigned width, \
    unsigned block_size, unsigned inverse, unsigned inner, unsigned first, unsigned second) { \
    native_hadamard(input, output, signs, width, block_size, inverse, inner, first, second); \
}
NATIVE_HADAMARD(half, float, f16_f32)
NATIVE_HADAMARD(float, float, f32_f32)
NATIVE_HADAMARD(float, half, f32_f16)
