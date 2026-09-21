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

// Q4_K/Q5_K/Q6_K x Q8 projections for the explicit Q8/F32scale policy.
// The strict native provider does not select these exports. Quantizing activations
// changes the numeric policy: this is not
// a strict-equivalent implementation of native_linear or llama's Q8_1 format.
//
// Pack ABI: x[rows][inputs] F16; scales[rows][inputs/32] F32;
// qwords[rows][inputs/32][8] u32. Each word contains four signed int8 values,
// with the lower K index in its least-significant byte. Inputs must be a
// multiple of 32. Launch 128 threads and ceil(rows*(inputs/32)/4) blocks.
// Typed pointers retain their natural alignment; caller owns all extents.
extern "C" __global__ void vnext_gguf_q8_f32scale_pack_f16_prototype(
    const half* x, float* scales, unsigned* qwords, unsigned rows, unsigned inputs) {
    if (blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1
        || gridDim.y != 1 || gridDim.z != 1 || inputs == 0 || inputs % 32 != 0) return;
    const unsigned lane = threadIdx.x % 32;
    const size_t group = size_t(blockIdx.x) * 4 + threadIdx.x / 32;
    if (group >= size_t(rows) * (inputs / 32)) return; // Whole-warp exit.
    const float value = __half2float(x[group * 32 + lane]);
    const bool finite = isfinite(value);
    const bool invalid_group = __ballot_sync(0xffffffff, !finite) != 0;
    float maximum = finite ? fabsf(value) : 0.0f;
    for (unsigned step = 16; step != 0; step /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, step));
    // F16's smallest nonzero magnitude / 127 remains a normal F32 number.
    // Explicit RN division and roundf define scale rounding and ties-away
    // integer rounding independently of the device's default conversion mode.
    const float scale = invalid_group ? NAN
        : maximum == 0.0f ? 0.0f : __fdiv_rn(maximum, 127.0f);
    int q = 0;
    if (!invalid_group && maximum != 0.0f) {
        const float rounded = roundf(__fdiv_rn(value, scale));
        q = int(fmaxf(-127.0f, fminf(127.0f, rounded)));
    }
    if (lane == 0) scales[group] = scale;
    // Every lane participates in the shuffle; only the four-lane leaders store.
    const unsigned q0 = unsigned(q) & 255;
    const unsigned q1 = __shfl_down_sync(0xffffffff, q0, 1, 4);
    const unsigned q2 = __shfl_down_sync(0xffffffff, q0, 2, 4);
    const unsigned q3 = __shfl_down_sync(0xffffffff, q0, 3, 4);
    if (lane % 4 == 0)
        qwords[group * 8 + lane / 4] = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
}

__device__ __forceinline__ int prototype_signed_dot4(unsigned a, unsigned b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(static_cast<int>(a), static_cast<int>(b), 0);
#else
    // Keep older production PTX targets buildable. The SM89 experiment uses
    // the DP4A branch; this fallback makes no performance capability claim.
    int result = 0;
    #pragma unroll
    for (unsigned byte_index = 0; byte_index < 4; ++byte_index) {
        int av = int((a >> (8 * byte_index)) & 255);
        int bv = int((b >> (8 * byte_index)) & 255);
        if (av >= 128) av -= 256;
        if (bv >= 128) bv -= 256;
        result += av * bv;
    }
    return result;
#endif
}

template<unsigned Format>
__device__ __forceinline__ unsigned prototype_qk_block_bytes() {
    static_assert(Format == 12 || Format == 13 || Format == 14, "prototype K format");
    return Format == 12 ? 144 : Format == 13 ? 176 : 210;
}

// Read four coefficients without requiring any alignment of the native block.
// Q4/Q5 codes are nonnegative; Q6 codes are signed bytes in [-32,31].
template<unsigned Format>
__device__ __forceinline__ unsigned prototype_qk_word(
    const byte* block, unsigned group, unsigned word) {
    unsigned packed = 0;
    #pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        const unsigned within = word * 4 + j;
        unsigned code;
        if constexpr (Format == 14) {
            const unsigned i = group * 32 + within;
            const unsigned local_group = (i % 128) / 32;
            const unsigned lo = (block[(i / 128) * 64 + (local_group % 2) * 32 + within]
                >> (4 * (local_group / 2))) & 15;
            const unsigned hi = (block[128 + (i / 128) * 32 + within]
                >> (2 * local_group)) & 3;
            code = unsigned(int(lo | (hi << 4)) - 32) & 255;
        } else {
            constexpr unsigned start = Format == 13 ? 48 : 16;
            code = (block[start + (group / 2) * 32 + within]
                >> (4 * (group % 2))) & 15;
            if constexpr (Format == 13) {
                if (block[16 + within] & (1u << group)) code += 16;
            }
        }
        packed |= code << (j * 8);
    }
    return packed;
}

// Q4/Q5: positive scale and minimum, each rounded once to F32.
// Q6: the two independent K16 scales within one K32 activation scale group.
template<unsigned Format>
__device__ __forceinline__ void prototype_qk_coefficients(
    const byte* block, unsigned group, float& first, float& second) {
    if constexpr (Format == 14) {
        const float d = native_half(block, 208);
        first = d * float(static_cast<signed char>(block[192 + 2 * group]));
        second = d * float(static_cast<signed char>(block[193 + 2 * group]));
    } else {
        const byte* s = block + 4;
        const unsigned scale6 = group < 4 ? s[group] & 63
            : (s[group + 4] & 15) | ((s[group - 4] >> 6) << 4);
        const unsigned min6 = group < 4 ? s[group + 4] & 63
            : (s[group + 4] >> 4) | ((s[group] >> 6) << 4);
        first = native_half(block, 0) * float(scale6);
        second = native_half(block, 2) * float(min6);
    }
}

// Matmul ABI: packed buffers above, native K weights[outputs][inputs/256]
// (Q4:144, Q5:176, Q6:210 bytes/block), and y[rows][stride] F16 at offset.
// Launch (ceil(outputs/4), ceil(rows/RowTile), 1) blocks of (128,1,1).
// A warp owns one output column; its four 8-lane subgroups process four K32
// groups. Each lane loads four byte-safe coefficients and one packed Q8 word.
// Each lane applies F32 scaling before the final warp reduction. Weights are
// shared across RowTile activation rows, without per-group integer shuffles.
template<unsigned Format, unsigned RowTile>
__device__ void prototype_qk_q8_linear(const float* scales, const unsigned* qwords,
    const byte* w, half* y, unsigned rows, unsigned inputs, unsigned outputs,
    unsigned stride, unsigned offset) {
    const size_t first_row = size_t(blockIdx.y) * RowTile;
    const size_t column = size_t(blockIdx.x) * 4 + threadIdx.x / 32;
    const unsigned lane = threadIdx.x % 32;
    const unsigned subgroup = lane / 8;
    const unsigned word = lane % 8;
    if (first_row >= rows || column >= outputs) return; // Whole-warp exit.
    const unsigned groups = inputs / 32;
    const byte* weights = w + column * size_t(inputs / 256) * prototype_qk_block_bytes<Format>();
    float sums[RowTile] = {};
    for (unsigned base = 0; base < groups; base += 4) {
        const unsigned group = base + subgroup; // inputs%256: no partial group.
        const unsigned local_group = group % 8;
        const byte* block = weights + size_t(group / 8) * prototype_qk_block_bytes<Format>();
        const unsigned packed_weight = prototype_qk_word<Format>(block, local_group, word);
        float a, b;
        prototype_qk_coefficients<Format>(block, local_group, a, b);
        #pragma unroll
        for (unsigned r = 0; r < RowTile; ++r) {
            if (first_row + r < rows) { // Uniform condition within each warp.
                const size_t packed_group = (first_row + r) * groups + group;
                const unsigned packed_x = qwords[packed_group * 8 + word];
                const int dot = prototype_signed_dot4(packed_weight, packed_x);
                // Four activations: |dot|<=4*32*127 for all three formats.
                const float d = scales[packed_group];
                // Independent approximate policy, no unquantized sum
                // correction. The source is built with --fmad=false.
                if constexpr (Format == 14) {
                    // Each word stays entirely within one K16 weight scale.
                    sums[r] += d * ((word < 4 ? a : b) * float(dot));
                } else {
                    const int sum = prototype_signed_dot4(0x01010101u, packed_x);
                    sums[r] += d * (a * float(dot) - b * float(sum));
                }
            }
        }
    }
    #pragma unroll
    for (unsigned r = 0; r < RowTile; ++r) {
        for (unsigned step = 16; step != 0; step /= 2)
            sums[r] += __shfl_down_sync(0xffffffff, sums[r], step);
        if (lane == 0 && first_row + r < rows)
            y[(first_row + r) * stride + offset + column] = half(sums[r]);
    }
}

#define PROTOTYPE_QK_Q8_LINEAR(name, format, suffix, tile) \
extern "C" __global__ void vnext_gguf_##name##_q8_f32scale_dp4a_##suffix##_prototype( \
    const float* scales, const unsigned* qwords, const byte* w, half* y, \
    unsigned rows, unsigned inputs, unsigned outputs, unsigned stride, unsigned offset) { \
    if (blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 || gridDim.z != 1 \
        || inputs == 0 || inputs % 256 != 0 || offset > stride || outputs > stride - offset) return; \
    prototype_qk_q8_linear<format, tile>(scales, qwords, w, y, rows, inputs, outputs, stride, offset); \
}
PROTOTYPE_QK_Q8_LINEAR(q4k, 12, lane_f16, 1)
PROTOTYPE_QK_Q8_LINEAR(q4k, 12, lane_tiled_f16, 8)
PROTOTYPE_QK_Q8_LINEAR(q5k, 13, lane_f16, 1)
PROTOTYPE_QK_Q8_LINEAR(q5k, 13, lane_tiled_f16, 8)
PROTOTYPE_QK_Q8_LINEAR(q6k, 14, lane_f16, 1)
PROTOTYPE_QK_Q8_LINEAR(q6k, 14, lane_tiled_f16, 8)

// Same diagnostic arithmetic policy, now a 16-output x 32-token CTA. Four
// warps share a K256 decoded weight tile, each computing 16 outputs x 8 tokens.
// Integer MMA is restricted to a single K32 scale group; its C registers are
// reset before every instruction, then rescaled in F32. This export is not
// selected by the strict provider and requires SM80+ to execute.
template<unsigned Format>
__device__ void prototype_qk_q8_mma(
    const float* scales, const unsigned* qwords, const byte* w, half* y,
    unsigned rows, unsigned inputs, unsigned outputs, unsigned stride, unsigned offset) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 || gridDim.z != 1
        || inputs == 0 || inputs % 256 != 0 || offset > stride || outputs > stride - offset) return;
    // Each row has eight live words and four padding words. A fragment load
    // uses (12*g+t) mod 32, g=0..7/t=0..3: one address in each shared bank.
    // Codes: 6144 bytes; affine metadata: 1024 bytes; total: 7168 bytes/CTA.
    __shared__ unsigned codes[8][16][12];
    __shared__ float coefficients[8][16];
    // Q4/Q5 store the minimum; Q6 stores the second K16 coefficient.
    __shared__ float second_coefficients[8][16];
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid % 32;
    const unsigned warp = tid / 32;
    const unsigned g = lane / 4;
    const unsigned t = lane % 4;
    const unsigned load_column = tid / 8;
    const unsigned load_word = tid % 8;
    const size_t first_column = size_t(blockIdx.x) * 16;
    const size_t first_row = size_t(blockIdx.y) * 32 + warp * 8;
    const unsigned blocks = inputs / 256;
    const unsigned groups = inputs / 32;
    // Four interleaved chains preserve the existing subgroup policy's bounded
    // serial-sum length, rather than silently changing its F32 error bound.
    float partial[4][4] = {};
    for (unsigned block_index = 0; block_index < blocks; ++block_index) {
        const bool valid_column = first_column + load_column < outputs;
        const byte* block = valid_column
            ? w + ((first_column + load_column) * blocks + block_index)
                * prototype_qk_block_bytes<Format>() : w;
        #pragma unroll
        for (unsigned group = 0; group < 8; ++group) {
            unsigned packed_weight = 0;
            if (valid_column)
                packed_weight = prototype_qk_word<Format>(block, group, load_word);
            codes[group][load_column][load_word] = packed_weight;
            if (load_word == 0) {
                float a = 0.0f, b = 0.0f;
                if (valid_column) prototype_qk_coefficients<Format>(block, group, a, b);
                coefficients[group][load_column] = a;
                second_coefficients[group][load_column] = b;
            }
        }
        __syncthreads();
        #pragma unroll
        for (unsigned group = 0; group < 8; ++group) {
            // m16n8k32 row.col s8: A rows are output channels, B columns are
            // token rows. Every lane supplies four A words and two B words.
            const unsigned a0 = codes[group][g][t];
            const unsigned a1 = codes[group][g + 8][t];
            const unsigned a2 = codes[group][g][t + 4];
            const unsigned a3 = codes[group][g + 8][t + 4];
            unsigned b0 = 0, b1 = 0;
            float activation_scale = 0.0f;
            if (first_row + g < rows) {
                const size_t packed_group = (first_row + g) * groups + block_index * 8 + group;
                b0 = qwords[packed_group * 8 + t];
                b1 = qwords[packed_group * 8 + t + 4];
                activation_scale = scales[packed_group];
            }
            const float d0 = __shfl_sync(0xffffffff, activation_scale, 8 * t);
            const float d1 = __shfl_sync(0xffffffff, activation_scale, 8 * t + 4);
            int dot0 = 0, dot1 = 0, dot2 = 0, dot3 = 0;
            const float a_lower = coefficients[group][g];
            const float a_upper = coefficients[group][g + 8];
            const float b_lower = second_coefficients[group][g];
            const float b_upper = second_coefficients[group][g + 8];
            if constexpr (Format == 14) {
                // Q6 has two weight scales per K32 activation group. A/B
                // lower words form K16, with the same C ownership as K32;
                // upper words form the second K16. Each |dot|<=16*32*127.
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                    "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                    : "+r"(dot0), "+r"(dot1), "+r"(dot2), "+r"(dot3)
                    : "r"(a0), "r"(a1), "r"(b0));
                int upper0 = 0, upper1 = 0, upper2 = 0, upper3 = 0;
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                    "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                    : "+r"(upper0), "+r"(upper1), "+r"(upper2), "+r"(upper3)
                    : "r"(a2), "r"(a3), "r"(b1));
                // Coefficients were rounded to F32 before the integer dot.
                // Both K16 halves share delta; combine once per K32, then
                // accumulate with the same four partial chains as Q4/Q5.
                partial[group % 4][0] += d0 * (a_lower * float(dot0) + b_lower * float(upper0));
                partial[group % 4][1] += d1 * (a_lower * float(dot1) + b_lower * float(upper1));
                partial[group % 4][2] += d0 * (a_upper * float(dot2) + b_upper * float(upper2));
                partial[group % 4][3] += d1 * (a_upper * float(dot3) + b_upper * float(upper3));
            } else {
                int activation_sum = prototype_signed_dot4(0x01010101u, b0)
                    + prototype_signed_dot4(0x01010101u, b1);
                activation_sum += __shfl_xor_sync(0xffffffff, activation_sum, 1, 4);
                activation_sum += __shfl_xor_sync(0xffffffff, activation_sum, 2, 4);
                const int sum0 = __shfl_sync(0xffffffff, activation_sum, 8 * t);
                const int sum1 = __shfl_sync(0xffffffff, activation_sum, 8 * t + 4);
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+r"(dot0), "+r"(dot1), "+r"(dot2), "+r"(dot3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                partial[group % 4][0] += d0 * (a_lower * float(dot0) - b_lower * float(sum0));
                partial[group % 4][1] += d1 * (a_lower * float(dot1) - b_lower * float(sum1));
                partial[group % 4][2] += d0 * (a_upper * float(dot2) - b_upper * float(sum0));
                partial[group % 4][3] += d1 * (a_upper * float(dot3) - b_upper * float(sum1));
            }
        }
        // Invalid token/output tails still participate in both barriers and
        // every warp instruction; only their final stores are suppressed.
        __syncthreads();
    }
    #pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        const size_t row = first_row + 2 * t + j % 2;
        const size_t column = first_column + g + (j / 2) * 8;
        const float sum = (partial[0][j] + partial[2][j]) + (partial[1][j] + partial[3][j]);
        if (row < rows && column < outputs)
            y[row * stride + offset + column] = half(sum);
    }
#endif
}

#define PROTOTYPE_QK_Q8_MMA(name, format) \
extern "C" __global__ void vnext_gguf_##name##_q8_f32scale_mma_f16_prototype( \
    const float* scales, const unsigned* qwords, const byte* w, half* y, \
    unsigned rows, unsigned inputs, unsigned outputs, unsigned stride, unsigned offset) { \
    prototype_qk_q8_mma<format>(scales, qwords, w, y, rows, inputs, outputs, stride, offset); \
}
PROTOTYPE_QK_Q8_MMA(q4k, 12)
PROTOTYPE_QK_Q8_MMA(q5k, 13)
PROTOTYPE_QK_Q8_MMA(q6k, 14)

// A prefill tile reuses activations across output columns as well as decoded
// weights across token rows. Decode directly into bounded shared F32 storage;
// no expanded persistent weights, F16 weight rounding, or tensor math mode.
// The sum now follows input-column order instead of a warp reduction, while
// multiplication and accumulation remain F32 (this source uses --fmad=false).
template<typename Input, typename Output, unsigned Format, unsigned Values, unsigned Bytes>
__device__ void native_block_gemm(const Input* x, const byte* w, Output* y,
    unsigned rows, unsigned inputs, unsigned outputs, unsigned stride, unsigned offset,
    float (&activations)[64][33], float (&weights)[32][65]) {
    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const unsigned tid = ty * 16 + tx;
    const unsigned first_row = blockIdx.y * 64;
    const unsigned first_column = blockIdx.x * 64;
    const size_t row_bytes = size_t(inputs / Values) * Bytes;
    float sums[4][4] = {};
    for (unsigned base = 0; base < inputs; base += 32) {
        // Consecutive lanes read consecutive coefficients. Padding avoids
        // shared-memory bank conflicts when transposing the weight tile.
        for (unsigned element = tid; element < 64 * 32; element += 256) {
            const unsigned local = element / 32;
            const unsigned k = element % 32;
            const unsigned input_column = base + k;
            activations[local][k] = first_row + local < rows && input_column < inputs
                ? float(x[size_t(first_row + local) * inputs + input_column]) : 0.0f;
            float weight = 0.0f;
            if (first_column + local < outputs && input_column < inputs) {
                const byte* block = w + size_t(first_column + local) * row_bytes
                    + size_t(input_column / Values) * Bytes;
                weight = native_block_value(block, input_column % Values, Format);
            }
            weights[k][local] = weight;
        }
        __syncthreads();
        #pragma unroll
        for (unsigned k = 0; k < 32; ++k) {
            float a[4], b[4];
            #pragma unroll
            for (unsigned r = 0; r < 4; ++r) a[r] = activations[ty + r * 16][k];
            #pragma unroll
            for (unsigned c = 0; c < 4; ++c) b[c] = weights[k][tx + c * 16];
            #pragma unroll
            for (unsigned r = 0; r < 4; ++r) {
                #pragma unroll
                for (unsigned c = 0; c < 4; ++c) sums[r][c] += a[r] * b[c];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned r = 0; r < 4; ++r) {
        const unsigned row = first_row + ty + r * 16;
        #pragma unroll
        for (unsigned c = 0; c < 4; ++c) {
            const unsigned column = first_column + tx + c * 16;
            if (row < rows && column < outputs)
                y[size_t(row) * stride + offset + column] = Output(sums[r][c]);
        }
    }
}

#define NATIVE_BLOCK_GEMM(Input, Output, suffix, Format, Values, Bytes) \
extern "C" __global__ void vnext_gguf_gemm_##suffix(const Input* x, const byte* w, Output* y, \
    unsigned rows, unsigned inputs, unsigned outputs, unsigned stride, unsigned offset, \
    unsigned format, unsigned values, unsigned bytes) { \
    if (format != Format || values != Values || bytes != Bytes || inputs % Values != 0) return; \
    __shared__ float activations[64][33]; \
    __shared__ float weights[32][65]; \
    native_block_gemm<Input, Output, Format, Values, Bytes>(x, w, y, rows, inputs, outputs, \
        stride, offset, activations, weights); \
}
NATIVE_BLOCK_GEMM(half, half, q4k_f16, 12, 256, 144)
NATIVE_BLOCK_GEMM(float, float, q4k_f32, 12, 256, 144)
NATIVE_BLOCK_GEMM(float, half, q4k_f32_f16, 12, 256, 144)
NATIVE_BLOCK_GEMM(half, half, q5k_f16, 13, 256, 176)
NATIVE_BLOCK_GEMM(float, float, q5k_f32, 13, 256, 176)
NATIVE_BLOCK_GEMM(float, half, q5k_f32_f16, 13, 256, 176)
NATIVE_BLOCK_GEMM(half, half, q6k_f16, 14, 256, 210)
NATIVE_BLOCK_GEMM(float, float, q6k_f32, 14, 256, 210)
NATIVE_BLOCK_GEMM(float, half, q6k_f32_f16, 14, 256, 210)

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
