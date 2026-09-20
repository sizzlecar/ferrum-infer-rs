// Test-only experiment based on native_pq2_linear_f32_complete.
// The original eight-output tiling follows ggml's MIT-licensed PQ2 Metal GEMV.
// See crates/ferrum-kernels/src/gguf_blocks/LICENSE.ggml in the Ferrum repository.
//
// Host contract: positive M, positive K divisible by 128, positive N divisible
// by 16, valid input/weight/output spans, grid (N / 16, M, 1), threads (32, 2, 1),
// SIMD width 32. Compile with fast_math_enabled(false), exactly as production.
// Buffer ABI is the production F32 Hadamard workspace -> PQ2 -> F16/F32 ABI.
// F32 operands remain wide until the final output store. No reassociation,
// vector input loads, or full-block predecode.
#include <metal_stdlib>
using namespace metal;

struct InterleavedEightParams {
    uint rows; uint in_features; uint out_features;
    uint output_stride; uint output_column_offset;
};

template<typename Output>
static inline void pq2_interleaved_eight_complete(
    device const float * input, device const uchar * weight, device Output * output,
    constant InterleavedEightParams & p, uint3 group, uint lane, uint subgroup) {
    const uint first_output = group.x * 16 + subgroup * 8;
    const uint blocks_per_row = p.in_features / 128;
    const uint first_in_block = (lane % 8) * 16;
    device const float * input_row = input + ulong(group.y) * p.in_features + first_in_block;
    const ulong row_bytes = ulong(blocks_per_row) * 34;
    device const uchar * weight_rows[8];
    #pragma clang loop unroll(full)
    for (uint part = 0; part < 8; ++part)
        weight_rows[part] = weight + ulong(first_output + part) * row_bytes;
    ulong input_block_offset = ulong(lane / 8) * 128;
    ulong weight_block_offset = ulong(lane / 8) * 34;
    float sums[8] = {};
    for (uint block_index = lane / 8; block_index < blocks_per_row; block_index += 4) {
        float values[16];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 16; ++i) values[i] = input_row[input_block_offset + i];

        // Retain production's row bases and block offsets. Each scale is read
        // once per output and K step, rather than once per interleaved update.
        device const uchar * blocks[8];
        float scales[8];
        #pragma clang loop unroll(full)
        for (uint part = 0; part < 8; ++part) {
            blocks[part] = weight_rows[part] + weight_block_offset;
            const ushort bits = ushort(blocks[part][0]) | (ushort(blocks[part][1]) << 8);
            scales[part] = float(as_type<half>(bits));
        }

        #pragma clang loop unroll(full)
        for (uint byte = 0; byte < 4; ++byte) {
            // Only this packed byte's four coefficients are live per output.
            // Packed-byte and scale load counts match the complete8 control.
            float codes[8][4];
            #pragma clang loop unroll(full)
            for (uint part = 0; part < 8; ++part) {
                const float packed = float(blocks[part][2 + first_in_block / 4 + byte]);
                const float top = floor(packed * (1.0f / 64.0f));
                const float middle = floor(packed * (1.0f / 16.0f));
                const float bottom = floor(packed * (1.0f / 4.0f));
                codes[part][0] = packed - 4.0f * bottom - 1.0f;
                codes[part][1] = bottom - 4.0f * middle - 1.0f;
                codes[part][2] = middle - 4.0f * top - 1.0f;
                codes[part][3] = top - 1.0f;
            }
            // Each sums[part] still receives component 0..3 of byte 0..3,
            // followed by the same next K step. Only independent outputs are
            // interleaved; keep the scale product before the activation product.
            #pragma clang loop unroll(full)
            for (uint component = 0; component < 4; ++component) {
                #pragma clang loop unroll(full)
                for (uint part = 0; part < 8; ++part) {
                    const float value = scales[part] * codes[part][component];
                    sums[part] += values[byte * 4 + component] * value;
                }
            }
        }
        input_block_offset += 4 * 128;
        weight_block_offset += 4 * 34;
    }
    #pragma clang loop unroll(full)
    for (uint part = 0; part < 8; ++part) {
        const float value = simd_sum(sums[part]);
        if (lane == 0)
            output[ulong(group.y) * p.output_stride + p.output_column_offset + first_output + part]
                = Output(value);
    }
}

#define INTERLEAVED_EIGHT(OUTPUT, SUFFIX) \
kernel void pq2_interleaved_eight_##SUFFIX##_complete( \
    device const float * x [[buffer(0)]], device const uchar * w [[buffer(1)]], \
    device OUTPUT * y [[buffer(2)]], constant InterleavedEightParams & p [[buffer(3)]], \
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
    uint subgroup [[simdgroup_index_in_threadgroup]]) { \
    pq2_interleaved_eight_complete(x, w, y, p, group, lane, subgroup); \
}
INTERLEAVED_EIGHT(float, f32)
INTERLEAVED_EIGHT(half, f16)
#undef INTERLEAVED_EIGHT
