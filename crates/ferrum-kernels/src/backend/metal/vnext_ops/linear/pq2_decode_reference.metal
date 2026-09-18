// Test-only original direct-bit decoder for numerical and GPU timing controls.
// The eight-output tiling is informed by ggml's MIT implementation.
// License: ../../../../gguf_blocks/LICENSE.ggml.
#include <metal_stdlib>
using namespace metal;

struct ReferenceParams {
    uint rows; uint in_features; uint out_features;
    uint output_stride; uint output_column_offset;
};

template<typename Output>
static inline void pq2_direct_bits(
    device const float * input, device const uchar * weight, device Output * output,
    constant ReferenceParams & p, uint3 group, uint lane, uint subgroup) {
    const uint first_output = group.x * 16 + subgroup * 8;
    const uint blocks_per_row = p.in_features / 128;
    const uint first_in_block = (lane % 8) * 16;
    float sums[8] = {};
    for (uint block_index = lane / 8; block_index < blocks_per_row; block_index += 4) {
        const ulong input_start = ulong(group.y) * p.in_features
            + ulong(block_index) * 128 + first_in_block;
        float values[16];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 16; ++i) values[i] = input[input_start + i];
        #pragma clang loop unroll(full)
        for (uint part = 0; part < 8; ++part) {
            const uint out_col = first_output + part;
            if (out_col >= p.out_features) continue;
            device const uchar * block = weight
                + (ulong(out_col) * blocks_per_row + block_index) * 34;
            const ushort bits = ushort(block[0]) | (ushort(block[1]) << 8);
            const float scale = float(as_type<half>(bits));
            #pragma clang loop unroll(full)
            for (uint byte = 0; byte < 4; ++byte) {
                const uint packed = block[2 + first_in_block / 4 + byte];
                #pragma clang loop unroll(full)
                for (uint component = 0; component < 4; ++component) {
                    const float value = scale * float(int((packed >> (2 * component)) & 3) - 1);
                    sums[part] += values[byte * 4 + component] * value;
                }
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint part = 0; part < 8; ++part) {
        const float value = simd_sum(sums[part]);
        const uint out_col = first_output + part;
        if (lane == 0 && out_col < p.out_features)
            output[ulong(group.y) * p.output_stride + p.output_column_offset + out_col]
                = Output(value);
    }
}

#define REFERENCE(OUTPUT, SUFFIX) \
kernel void pq2_direct_bits_##SUFFIX( \
    device const float * x [[buffer(0)]], device const uchar * w [[buffer(1)]], \
    device OUTPUT * y [[buffer(2)]], constant ReferenceParams & p [[buffer(3)]], \
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
    uint subgroup [[simdgroup_index_in_threadgroup]]) { \
    pq2_direct_bits(x, w, y, p, group, lane, subgroup); \
}
REFERENCE(float, f32)
REFERENCE(half, f16)
#undef REFERENCE
