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

// A register-only selection from the same signed-byte codebook. Use the lower
// three bits for byte selection, and the GGUF code's fourth bit to select the
// upper half of the sixteen-entry table separately.
__device__ __forceinline__ int native_iq4_register_value(unsigned code) {
    const unsigned low = __byte_perm(IQ4_NL_PACKED_0, IQ4_NL_PACKED_1, code & 7u);
    const unsigned high = __byte_perm(IQ4_NL_PACKED_2, IQ4_NL_PACKED_3, code & 7u);
    const unsigned selected = (code & 8u) ? high : low;
    return int(selected & 255u) - int((selected & 128u) << 1);
}

__device__ __forceinline__ float native_iq4xs_register_block_value(const byte* b, unsigned i) {
    // Match case 23's byte-safe loads and FP32 expression exactly. Only the
    // integer codebook lookup changes; generic/embedding/control stay intact.
    const unsigned group = i / 32;
    const unsigned scales_h = unsigned(b[2]) | (unsigned(b[3]) << 8);
    const unsigned lo = (b[4 + group / 2] >> (4 * (group % 2))) & 15;
    const unsigned hi = (scales_h >> (2 * group)) & 3;
    const int scale = int(lo | (hi << 4)) - 32;
    const unsigned q = (b[8 + group * 16 + i % 16] >> (4 * ((i % 32) / 16))) & 15;
    return (native_half(b, 0) * float(scale)) * float(native_iq4_register_value(q));
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

// Q8_0 has one signed byte per lane and one F16 scale per 32-value block.
// Fix the physical stride without changing coefficient reconstruction, each
// lane's K order, cross-row reuse, or the final warp reduction.
template<unsigned RowTile>
__device__ void native_q8_0_linear(const half* x, const byte* w, half* y,
    unsigned rows, unsigned inputs, unsigned outputs, unsigned stride, unsigned offset) {
    const unsigned row = blockIdx.y * RowTile;
    const unsigned lane = threadIdx.x % 32;
    const unsigned column = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= rows || column >= outputs) return;
    const unsigned blocks = inputs / 32;
    const byte* weights = w + size_t(column) * blocks * 34;
    float sums[RowTile] = {};
    for (unsigned block = 0; block < blocks; ++block) {
        const byte* source = weights + size_t(block) * 34;
        const unsigned i = block * 32 + lane;
        // Keep the byte-safe scale load: matrix views can have odd offsets.
        const float weight = native_half(source, 0)
            * float(static_cast<signed char>(source[2 + lane]));
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
            y[size_t(row + r) * stride + offset + column] = half(sums[r]);
    }
}

#define NATIVE_Q8_0_LINEAR(suffix, tile) \
extern "C" __global__ void vnext_gguf_linear_q8_0_##suffix(const half* x, const byte* w, half* y, \
    unsigned rows, unsigned inputs, unsigned outputs, unsigned stride, unsigned offset, \
    unsigned format, unsigned values, unsigned bytes) { \
    if (format != 8 || values != 32 || bytes != 34 || inputs % 32 != 0) return; \
    native_q8_0_linear<tile>(x, w, y, rows, inputs, outputs, stride, offset); \
}
NATIVE_Q8_0_LINEAR(f16, 1)
NATIVE_Q8_0_LINEAR(tiled_f16, 8)
#undef NATIVE_Q8_0_LINEAR

// Fix a 256-value block's format and byte stride without changing reconstructed
// F32 coefficients, each lane's K order, or the final warp reduction. Byte-safe
// coefficient loads preserve odd-offset views. F32 operands remain F32: the
// Q6_K output head must not acquire an intermediate half rounding here.
template<typename Input, typename Output, unsigned RowTile, unsigned Format, unsigned BlockBytes,
    bool RegisterIq4 = false>
__device__ void native_fixed256_linear(const Input* x, const byte* w, Output* y,
    unsigned rows, unsigned inputs, unsigned outputs, unsigned stride, unsigned offset) {
    const unsigned row = blockIdx.y * RowTile;
    const unsigned lane = threadIdx.x % 32;
    const unsigned column = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= rows || column >= outputs) return;
    const unsigned blocks = inputs / 256;
    const byte* weights = w + size_t(column) * blocks * BlockBytes;
    float sums[RowTile] = {};
    for (unsigned block = 0; block < blocks; ++block) {
        const byte* source = weights + size_t(block) * BlockBytes;
        #pragma unroll
        for (unsigned group = 0; group < 8; ++group) {
            const unsigned local = group * 32 + lane;
            const unsigned i = block * 256 + local;
            float weight;
            if constexpr (Format == 23 && RegisterIq4) {
                weight = native_iq4xs_register_block_value(source, local);
            } else {
                weight = native_block_value(source, local, Format);
            }
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
            y[size_t(row + r) * stride + offset + column] = Output(sums[r]);
    }
}

#define NATIVE_FIXED256_LINEAR(Input, Output, name, Format, BlockBytes, suffix, tile, register_iq4) \
extern "C" __global__ void vnext_gguf_linear_##name##_##suffix(const Input* x, const byte* w, Output* y, \
    unsigned rows, unsigned inputs, unsigned outputs, unsigned stride, unsigned offset, \
    unsigned format, unsigned values, unsigned bytes) { \
    if (format != Format || values != 256 || bytes != BlockBytes || inputs % 256 != 0) return; \
    native_fixed256_linear<Input, Output, tile, Format, BlockBytes, register_iq4>(x, w, y, rows, inputs, outputs, stride, offset); \
}
NATIVE_FIXED256_LINEAR(half, half, q5k, 13, 176, f16, 1, false)
NATIVE_FIXED256_LINEAR(half, half, q5k, 13, 176, tiled_f16, 8, false)
NATIVE_FIXED256_LINEAR(half, half, iq4xs, 23, 136, f16, 1, true)
NATIVE_FIXED256_LINEAR(half, half, iq4xs, 23, 136, tiled_f16, 8, true)
// Same-binary controls reproduce the previous fixed-format IQ4_XS route.
NATIVE_FIXED256_LINEAR(half, half, iq4xs_constant, 23, 136, f16, 1, false)
NATIVE_FIXED256_LINEAR(half, half, iq4xs_constant, 23, 136, tiled_f16, 8, false)
NATIVE_FIXED256_LINEAR(half, half, q6k, 14, 210, f16, 1, false)
NATIVE_FIXED256_LINEAR(half, half, q6k, 14, 210, tiled_f16, 8, false)
NATIVE_FIXED256_LINEAR(float, float, q6k, 14, 210, f32, 1, false)
NATIVE_FIXED256_LINEAR(float, float, q6k, 14, 210, tiled_f32, 8, false)
#undef NATIVE_FIXED256_LINEAR

// Diagnostic entrypoint checks reconstruction before multiplication/half
// rounding can hide a wrong codebook entry or a tiny signed-scale result.
extern "C" __global__ void vnext_gguf_iq4xs_register_decode(
    const byte* input, float* output, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        output[i] = native_iq4xs_register_block_value(input + size_t(i / 256) * 136, i % 256);
}

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

// Explicit RN-F16 fragment operand ABI v1. Selected only by its typed FFN operation.

// Cold N16/K32 cells retain original quant codes plus exact F32 scale metadata.
// Each warp constructs four MMA A half2 registers; no F16 weight shared tile.
__device__ __forceinline__ unsigned rn_frag_pair(half lo, half hi) {
    return unsigned(__half_as_ushort(lo)) | (unsigned(__half_as_ushort(hi)) << 16);
}
__device__ __forceinline__ unsigned rn_frag_word(const byte* p) {
    if ((reinterpret_cast<size_t>(p) & 3) == 0)
        return *reinterpret_cast<const unsigned*>(p);
    return unsigned(p[0]) | (unsigned(p[1]) << 8)
        | (unsigned(p[2]) << 16) | (unsigned(p[3]) << 24);
}
__device__ __forceinline__ unsigned rn_frag_step(unsigned format) {
    return 128 + (format - 12) * 32;
}
__device__ __forceinline__ bool rn_frag_valid(
    unsigned k, unsigned n, unsigned format, unsigned long long bytes, unsigned abi) {
    if (abi != 0x524e4631u || format < 12 || format > 14 || !k || k % 256 || !n)
        return false;
    const unsigned long long expected = ((static_cast<unsigned long long>(n) + 15) / 16)
        * (k / 32) * (128 + 2 * rn_frag_step(format));
    return bytes == expected;
}
__device__ __forceinline__ unsigned rn_frag_code(
    unsigned low, const byte* high, unsigned lane, unsigned slot, unsigned format) {
    unsigned q = (low >> (4 * slot)) & 15;
    if (format >= 13) q |= ((unsigned(high[lane]) >> slot) & 1) << 4;
    if (format == 14) q |= ((unsigned(high[32 + lane]) >> slot) & 1) << 5;
    return q;
}
__device__ __forceinline__ half rn_frag_half(float a, float z, unsigned q, unsigned format) {
    // Exactly the materializer's F32 reconstruction before its RN-even half.
    const float value = format == 14 ? __fmul_rn(a, float(int(q) - 32))
        : __fsub_rn(__fmul_rn(a, float(q)), z);
    return __float2half_rn(value);
}
extern "C" __global__ void vnext_rn_fragment_coefficients(
    const byte* packed, unsigned long long bytes, half* output,
    unsigned k, unsigned n, unsigned format, unsigned abi) {
    if (!rn_frag_valid(k, n, format, bytes, abi)) return;
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(n) * k) return;
    const unsigned column = i / k, kk = i % k, local = column % 16;
    const unsigned fragment = (kk % 32) / 16, step = rn_frag_step(format);
    const byte* cell = packed + (size_t(column / 16) * (k / 32) + kk / 32) * (128 + 2 * step);
    const byte* codes = cell + 128 + fragment * step;
    const unsigned lane = (local % 8) * 4 + (kk % 8) / 2;
    const unsigned slot = (kk % 16 >= 8 ? 4 : 0) + (local >= 8 ? 2 : 0) + kk % 2;
    const float a = __uint_as_float(rn_frag_word(cell + (format == 14 ? fragment * 64 : 0) + local * 4));
    const float z = format == 14 ? 0.0f : __uint_as_float(rn_frag_word(cell + 64 + local * 4));
    output[i] = rn_frag_half(a, z, rn_frag_code(rn_frag_word(codes + lane * 4), codes + 128, lane, slot, format), format);
}
extern "C" __global__ void vnext_rn_fragment_mma(
    const half* x, const byte* packed, unsigned long long bytes, half* y,
    unsigned rows, unsigned k, unsigned n, unsigned stride, unsigned offset,
    unsigned format, unsigned abi) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 || gridDim.z != 1
        || !rows || offset > stride || n > stride - offset
        || !rn_frag_valid(k, n, format, bytes, abi)) return;
    __shared__ float partial[8][128]; // Only F32 final partials: exactly 4096 B.
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned g = lane / 4, t = lane % 4;
    const unsigned first_col = blockIdx.x * 16, first_row = blockIdx.y * 8;
    if (blockIdx.x >= (static_cast<unsigned long long>(n) + 15) / 16
        || blockIdx.y >= (static_cast<unsigned long long>(rows) + 7) / 8) return;
    const unsigned step = rn_frag_step(format), cell_bytes = 128 + 2 * step;
    float c[4] = {};
    // K32 stripes are disjoint. No atomics/global scratch/fixup; the final
    // eight-warp F32 sum changes reduction order relative to vendor GEMM.
    for (unsigned group = warp; group < k / 32; group += 8) {
        const byte* cell = packed + (size_t(blockIdx.x) * (k / 32) + group) * cell_bytes;
        #pragma unroll
        for (unsigned h = 0; h < 2; ++h) {
            const byte* codes = cell + 128 + h * step;
            const unsigned low = rn_frag_word(codes + lane * 4);
            const unsigned scale_offset = format == 14 ? h * 64 : 0;
            const float al = __uint_as_float(rn_frag_word(cell + scale_offset + g * 4));
            const float ah = __uint_as_float(rn_frag_word(cell + scale_offset + (g + 8) * 4));
            const float zl = format == 14 ? 0.0f : __uint_as_float(rn_frag_word(cell + 64 + g * 4));
            const float zh = format == 14 ? 0.0f : __uint_as_float(rn_frag_word(cell + 64 + (g + 8) * 4));
            unsigned a[4];
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const float scale = j % 2 ? ah : al, zero = j % 2 ? zh : zl;
                const half lo = rn_frag_half(scale, zero, rn_frag_code(low, codes + 128, lane, 2 * j, format), format);
                const half hi = rn_frag_half(scale, zero, rn_frag_code(low, codes + 128, lane, 2 * j + 1, format), format);
                a[j] = rn_frag_pair(lo, hi);
            }
            unsigned b0 = 0, b1 = 0;
            if (first_row + g < rows) {
                const size_t base = size_t(first_row + g) * k + group * 32 + h * 16 + 2 * t;
                b0 = rn_frag_pair(x[base], x[base + 1]);
                b1 = rn_frag_pair(x[base + 8], x[base + 9]);
            }
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
        }
    }
    #pragma unroll
    for (unsigned j = 0; j < 4; ++j) partial[warp][lane * 4 + j] = c[j];
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            float sum = partial[0][lane * 4 + j];
            #pragma unroll
            for (unsigned w = 1; w < 8; ++w) sum = __fadd_rn(sum, partial[w][lane * 4 + j]);
            const unsigned row = first_row + 2 * t + j % 2;
            const unsigned column = first_col + g + (j / 2) * 8;
            if (row < rows && column < n) y[size_t(row) * stride + offset + column] = __float2half_rn(sum);
        }
    }
// A stale/misbound module must never turn a successful launch into no output.
#else
    asm volatile("trap;");
#endif
}

// Selected only for Q6 by the typed per-projection host plan. The generic
// format arithmetic is retained to preserve the qualified instruction schedule.
// PTX cp.async groups are per issuing thread. Every lane commits, including
// inactive copy lanes; consumers wait and synchronize the warp before reading.
__device__ __forceinline__ void rn_frag_prefetch_copy(
    byte* destination, const byte* source, unsigned cell_bytes, unsigned lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (lane * 16 < cell_bytes) {
        const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(destination + lane * 16));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
            :: "r"(shared), "l"(source + lane * 16) : "memory");
    }
    // Reconverge before the uniform commit; no lane exits the warp protocol.
    __syncwarp();
    asm volatile("cp.async.commit_group;" ::: "memory");
#else
    asm volatile("trap;");
#endif
}
extern "C" __global__ void vnext_rn_fragment_q6_prefetch_mma(
    const half* x, const byte* packed, unsigned long long bytes, half* y,
    unsigned rows, unsigned k, unsigned n, unsigned stride, unsigned offset,
    unsigned format, unsigned abi) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 || gridDim.z != 1
        || !rows || offset > stride || n > stride - offset
        || !rn_frag_valid(k, n, format, bytes, abi)) return;
    __shared__ float partial[8][128]; // Original F32 final partials: 4096 B.
    __shared__ __align__(16) byte staged[8][2][512]; // Two exact packet slots per warp.
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned g = lane / 4, t = lane % 4;
    const unsigned first_col = blockIdx.x * 16, first_row = blockIdx.y * 8;
    if (blockIdx.x >= (static_cast<unsigned long long>(n) + 15) / 16
        || blockIdx.y >= (static_cast<unsigned long long>(rows) + 7) / 8) return;
    const unsigned step = rn_frag_step(format), cell_bytes = 128 + 2 * step;
    float c[4] = {};
    // The original U8 ABI permits unaligned physical views. They retain the
    // original global reads below and issue no asynchronous copies.
    const bool prefetch = (reinterpret_cast<size_t>(packed) & 15) == 0;
    unsigned stage = 0;
    if (prefetch) {
        const byte* first = packed + (size_t(blockIdx.x) * (k / 32) + warp) * cell_bytes;
        rn_frag_prefetch_copy(staged[warp][stage], first, cell_bytes, lane);
    }
    // K32 stripes are disjoint. No atomics/global scratch/fixup; the final
    // eight-warp F32 sum changes reduction order relative to vendor GEMM.
    for (unsigned group = warp; group < k / 32; group += 8) {
        const byte* cell = packed + (size_t(blockIdx.x) * (k / 32) + group) * cell_bytes;
        if (prefetch) {
            // Each lane waits for its own copy, then the complete warp may read
            // metadata/codes copied by other lanes. One group at most is pending.
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncwarp();
            cell = staged[warp][stage];
            if (group + 8 < k / 32) {
                const byte* next = packed + (size_t(blockIdx.x) * (k / 32) + group + 8) * cell_bytes;
                rn_frag_prefetch_copy(staged[warp][stage ^ 1], next, cell_bytes, lane);
            }
        }
        #pragma unroll
        for (unsigned h = 0; h < 2; ++h) {
            const byte* codes = cell + 128 + h * step;
            const unsigned low = rn_frag_word(codes + lane * 4);
            const unsigned scale_offset = format == 14 ? h * 64 : 0;
            const float al = __uint_as_float(rn_frag_word(cell + scale_offset + g * 4));
            const float ah = __uint_as_float(rn_frag_word(cell + scale_offset + (g + 8) * 4));
            const float zl = format == 14 ? 0.0f : __uint_as_float(rn_frag_word(cell + 64 + g * 4));
            const float zh = format == 14 ? 0.0f : __uint_as_float(rn_frag_word(cell + 64 + (g + 8) * 4));
            unsigned a[4];
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const float scale = j % 2 ? ah : al, zero = j % 2 ? zh : zl;
                const half lo = rn_frag_half(scale, zero, rn_frag_code(low, codes + 128, lane, 2 * j, format), format);
                const half hi = rn_frag_half(scale, zero, rn_frag_code(low, codes + 128, lane, 2 * j + 1, format), format);
                a[j] = rn_frag_pair(lo, hi);
            }
            unsigned b0 = 0, b1 = 0;
            if (first_row + g < rows) {
                const size_t base = size_t(first_row + g) * k + group * 32 + h * 16 + 2 * t;
                b0 = rn_frag_pair(x[base], x[base + 1]);
                b1 = rn_frag_pair(x[base + 8], x[base + 9]);
            }
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
        }
        if (prefetch) {
            // No lane may overwrite this slot until every lane consumed it.
            __syncwarp();
            stage ^= 1;
        }
    }
    if (prefetch) {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncwarp();
    }
    #pragma unroll
    for (unsigned j = 0; j < 4; ++j) partial[warp][lane * 4 + j] = c[j];
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            float sum = partial[0][lane * 4 + j];
            #pragma unroll
            for (unsigned w = 1; w < 8; ++w) sum = __fadd_rn(sum, partial[w][lane * 4 + j]);
            const unsigned row = first_row + 2 * t + j % 2;
            const unsigned column = first_col + g + (j / 2) * 8;
            if (row < rows && column < n) y[size_t(row) * stride + offset + column] = __float2half_rn(sum);
        }
    }
// A stale/misbound module must never turn a successful launch into no output.
#else
    asm volatile("trap;");
#endif
}
