// Test-only output tiles for production's complete-address PQ2 GEMV.
// Preserve its F32 products, per-output accumulation, and SIMD reduction order.
// Only the independent outputs per SIMD change from eight to four or sixteen.
// The original eight-output tiling follows ggml's MIT implementation.
// License: ../../../../../gguf_blocks/LICENSE.ggml.
#include <metal_stdlib>
using namespace metal;

struct OutputTileParams {
    uint rows; uint in_features; uint out_features;
    uint output_stride; uint output_column_offset;
};

template<typename Output, uint OUTPUTS_PER_SIMD, bool COMPLETE>
static inline void pq2_output_tile(
    device const float * input, device const uchar * weight, device Output * output,
    constant OutputTileParams & p, uint3 group, uint lane, uint subgroup) {
    const uint first_output = group.x * (2 * OUTPUTS_PER_SIMD) + subgroup * OUTPUTS_PER_SIMD;
    const uint blocks_per_row = p.in_features / 128;
    const uint first_in_block = (lane % 8) * 16;
    device const float * input_row = input + ulong(group.y) * p.in_features + first_in_block;
    const ulong row_bytes = ulong(blocks_per_row) * 34;
    device const uchar * weight_rows[OUTPUTS_PER_SIMD];
    #pragma clang loop unroll(full)
    for (uint part = 0; part < OUTPUTS_PER_SIMD; ++part) {
        // Do not form an out-of-range row pointer for an inactive tail lane.
        weight_rows[part] = weight;
        if (COMPLETE || first_output + part < p.out_features)
            weight_rows[part] += ulong(first_output + part) * row_bytes;
    }
    ulong input_block_offset = ulong(lane / 8) * 128;
    ulong weight_block_offset = ulong(lane / 8) * 34;
    float sums[OUTPUTS_PER_SIMD] = {};
    for (uint block_index = lane / 8; block_index < blocks_per_row; block_index += 4) {
        float values[16];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 16; ++i) values[i] = input_row[input_block_offset + i];
        #pragma clang loop unroll(full)
        for (uint part = 0; part < OUTPUTS_PER_SIMD; ++part) {
            if (!COMPLETE && first_output + part >= p.out_features) continue;
            device const uchar * block = weight_rows[part] + weight_block_offset;
            const ushort bits = ushort(block[0]) | (ushort(block[1]) << 8);
            const float scale = float(as_type<half>(bits));
            #pragma clang loop unroll(full)
            for (uint byte = 0; byte < 4; ++byte) {
                const float packed = float(block[2 + first_in_block / 4 + byte]);
                const float top = floor(packed * (1.0f / 64.0f));
                const float middle = floor(packed * (1.0f / 16.0f));
                const float bottom = floor(packed * (1.0f / 4.0f));
                const float codes[4] = {
                    packed - 4.0f * bottom - 1.0f,
                    bottom - 4.0f * middle - 1.0f,
                    middle - 4.0f * top - 1.0f,
                    top - 1.0f,
                };
                #pragma clang loop unroll(full)
                for (uint component = 0; component < 4; ++component) {
                    const float value = scale * codes[component];
                    sums[part] += values[byte * 4 + component] * value;
                }
            }
        }
        input_block_offset += 4 * 128;
        weight_block_offset += 4 * 34;
    }
    #pragma clang loop unroll(full)
    for (uint part = 0; part < OUTPUTS_PER_SIMD; ++part) {
        const float value = simd_sum(sums[part]);
        if (lane == 0 && (COMPLETE || first_output + part < p.out_features))
            output[ulong(group.y) * p.output_stride + p.output_column_offset + first_output + part]
                = Output(value);
    }
}

#define OUTPUT_TILE(OUTPUT, TILE, SUFFIX, COUNT, COMPLETE) \
kernel void pq2_output_tile_##TILE##_##SUFFIX( \
    device const float * x [[buffer(0)]], device const uchar * w [[buffer(1)]], \
    device OUTPUT * y [[buffer(2)]], constant OutputTileParams & p [[buffer(3)]], \
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
    uint subgroup [[simdgroup_index_in_threadgroup]]) { \
    pq2_output_tile<OUTPUT, COUNT, COMPLETE>(x, w, y, p, group, lane, subgroup); \
}
OUTPUT_TILE(float, four, f32_complete, 4, true)
OUTPUT_TILE(half, four, f16_complete, 4, true)
OUTPUT_TILE(float, four, f32_guarded, 4, false)
OUTPUT_TILE(half, four, f16_guarded, 4, false)
OUTPUT_TILE(float, sixteen, f32_complete, 16, true)
OUTPUT_TILE(half, sixteen, f16_complete, 16, true)
OUTPUT_TILE(float, sixteen, f32_guarded, 16, false)
OUTPUT_TILE(half, sixteen, f16_guarded, 16, false)
#undef OUTPUT_TILE
