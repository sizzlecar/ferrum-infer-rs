#include <metal_stdlib>
using namespace metal;

constant uint QK_K = 256;
constant uint QK8_0 = 32;

struct LinearParams {
    uint rows;
    uint in_features;
    uint out_features;
    uint output_stride;
    uint output_column_offset;
};

struct SwiGluParams {
    uint rows;
    uint intermediate_size;
    uint gate_up_stride;
};

struct block_q4_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[QK_K / 2];
};

struct block_q5_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qh[QK_K / 8];
    uchar qs[QK_K / 2];
};

struct block_q6_K {
    uchar ql[QK_K / 2];
    uchar qh[QK_K / 4];
    char scales[QK_K / 16];
    half d;
};

struct block_q8_0 {
    half d;
    char qs[QK8_0];
};

static inline uchar2 qk_scale_min(uint index, device const uchar * scales) {
    if (index < 4) {
        return uchar2(scales[index] & 63, scales[index + 4] & 63);
    }
    return uchar2(
        (scales[index + 4] & 0x0f) | ((scales[index - 4] & 0xc0) >> 2),
        (scales[index + 4] >> 4) | ((scales[index] & 0xc0) >> 2)
    );
}

static inline float q4_k_value(device const block_q4_K & block, uint index) {
    const uint subblock = index / 32;
    const uint group = subblock / 2;
    const uchar packed = block.qs[group * 32 + index % 32];
    const uint quantized = (subblock & 1) == 0 ? packed & 0x0f : packed >> 4;
    const uchar2 scale_min = qk_scale_min(subblock, block.scales);
    return float(block.d) * float(scale_min.x) * float(quantized)
        - float(block.dmin) * float(scale_min.y);
}

static inline float q5_k_value(device const block_q5_K & block, uint index) {
    const uint subblock = index / 32;
    const uint group = subblock / 2;
    const uint lane = index % 32;
    const uchar packed = block.qs[group * 32 + lane];
    const uint low = (subblock & 1) == 0 ? packed & 0x0f : packed >> 4;
    const uint high_mask = 1u << (2 * group + (subblock & 1));
    const uint quantized = low + ((block.qh[lane] & high_mask) != 0 ? 16 : 0);
    const uchar2 scale_min = qk_scale_min(subblock, block.scales);
    return float(block.d) * float(scale_min.x) * float(quantized)
        - float(block.dmin) * float(scale_min.y);
}

static inline float q6_k_value(device const block_q6_K & block, uint index) {
    const uint half_block = index / 128;
    const uint position = index % 128;
    const uint lane = position % 32;
    const uint ql_base = half_block * 64;
    const uint qh_base = half_block * 32;
    const uint scale_base = half_block * 8;
    const uchar high = block.qh[qh_base + lane];

    uint low;
    uint upper;
    uint scale;
    if (position < 32) {
        low = block.ql[ql_base + lane] & 0x0f;
        upper = (high >> 0) & 0x03;
        scale = scale_base + lane / 16;
    } else if (position < 64) {
        low = block.ql[ql_base + lane + 32] & 0x0f;
        upper = (high >> 2) & 0x03;
        scale = scale_base + lane / 16 + 2;
    } else if (position < 96) {
        low = block.ql[ql_base + lane] >> 4;
        upper = (high >> 4) & 0x03;
        scale = scale_base + lane / 16 + 4;
    } else {
        low = block.ql[ql_base + lane + 32] >> 4;
        upper = (high >> 6) & 0x03;
        scale = scale_base + lane / 16 + 6;
    }
    const int quantized = int(low | (upper << 4)) - 32;
    return float(block.d) * float(block.scales[scale]) * float(quantized);
}

static inline float q8_0_value(device const block_q8_0 & block, uint index) {
    return float(block.d) * float(block.qs[index]);
}

kernel void vnext_linear_dense_f16(
    device const half * input [[buffer(0)]],
    device const half * weight [[buffer(1)]],
    device half * output [[buffer(2)]],
    constant LinearParams & params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row = group.y;
    const uint first_output = group.x * 4 + simd_group * 2;
    float sums[2] = {0.0f, 0.0f};
    const ulong input_base = ulong(row) * params.in_features;
    for (uint column = simd_lane; column < params.in_features; column += 32) {
        const float activation = float(input[input_base + column]);
        for (uint local_output = 0; local_output < 2; ++local_output) {
            const uint output_column = first_output + local_output;
            if (output_column < params.out_features) {
                sums[local_output] += activation
                    * float(weight[ulong(output_column) * params.in_features + column]);
            }
        }
    }
    for (uint local_output = 0; local_output < 2; ++local_output) {
        const uint output_column = first_output + local_output;
        const float total = simd_sum(sums[local_output]);
        if (simd_lane == 0 && output_column < params.out_features) {
            output[ulong(row) * params.output_stride
                + params.output_column_offset + output_column] = half(total);
        }
    }
}

kernel void vnext_linear_q4_k_f16(
    device const half * input [[buffer(0)]],
    device const block_q4_K * weight [[buffer(1)]],
    device half * output [[buffer(2)]],
    constant LinearParams & params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row = group.y;
    const uint first_output = group.x * 4 + simd_group * 2;
    const uint blocks_per_row = params.in_features / QK_K;
    float sums[2] = {0.0f, 0.0f};
    const ulong input_base = ulong(row) * params.in_features;
    for (uint column = simd_lane; column < params.in_features; column += 32) {
        const float activation = float(input[input_base + column]);
        for (uint local_output = 0; local_output < 2; ++local_output) {
            const uint output_column = first_output + local_output;
            if (output_column < params.out_features) {
                const ulong weight_base = ulong(output_column) * blocks_per_row;
                sums[local_output] += activation * q4_k_value(
                    weight[weight_base + column / QK_K], column % QK_K
                );
            }
        }
    }
    for (uint local_output = 0; local_output < 2; ++local_output) {
        const uint output_column = first_output + local_output;
        const float total = simd_sum(sums[local_output]);
        if (simd_lane == 0 && output_column < params.out_features) {
            output[ulong(row) * params.output_stride
                + params.output_column_offset + output_column] = half(total);
        }
    }
}

kernel void vnext_linear_q5_k_f16(
    device const half * input [[buffer(0)]],
    device const block_q5_K * weight [[buffer(1)]],
    device half * output [[buffer(2)]],
    constant LinearParams & params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row = group.y;
    const uint first_output = group.x * 4 + simd_group * 2;
    const uint blocks_per_row = params.in_features / QK_K;
    float sums[2] = {0.0f, 0.0f};
    const ulong input_base = ulong(row) * params.in_features;
    for (uint column = simd_lane; column < params.in_features; column += 32) {
        const float activation = float(input[input_base + column]);
        for (uint local_output = 0; local_output < 2; ++local_output) {
            const uint output_column = first_output + local_output;
            if (output_column < params.out_features) {
                const ulong weight_base = ulong(output_column) * blocks_per_row;
                sums[local_output] += activation * q5_k_value(
                    weight[weight_base + column / QK_K], column % QK_K
                );
            }
        }
    }
    for (uint local_output = 0; local_output < 2; ++local_output) {
        const uint output_column = first_output + local_output;
        const float total = simd_sum(sums[local_output]);
        if (simd_lane == 0 && output_column < params.out_features) {
            output[ulong(row) * params.output_stride
                + params.output_column_offset + output_column] = half(total);
        }
    }
}

kernel void vnext_linear_q6_k_f16(
    device const half * input [[buffer(0)]],
    device const block_q6_K * weight [[buffer(1)]],
    device half * output [[buffer(2)]],
    constant LinearParams & params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row = group.y;
    const uint first_output = group.x * 4 + simd_group * 2;
    const uint blocks_per_row = params.in_features / QK_K;
    float sums[2] = {0.0f, 0.0f};
    const ulong input_base = ulong(row) * params.in_features;
    for (uint column = simd_lane; column < params.in_features; column += 32) {
        const float activation = float(input[input_base + column]);
        for (uint local_output = 0; local_output < 2; ++local_output) {
            const uint output_column = first_output + local_output;
            if (output_column < params.out_features) {
                const ulong weight_base = ulong(output_column) * blocks_per_row;
                sums[local_output] += activation * q6_k_value(
                    weight[weight_base + column / QK_K], column % QK_K
                );
            }
        }
    }
    for (uint local_output = 0; local_output < 2; ++local_output) {
        const uint output_column = first_output + local_output;
        const float total = simd_sum(sums[local_output]);
        if (simd_lane == 0 && output_column < params.out_features) {
            output[ulong(row) * params.output_stride
                + params.output_column_offset + output_column] = half(total);
        }
    }
}

kernel void vnext_linear_q8_0_f16(
    device const half * input [[buffer(0)]],
    device const block_q8_0 * weight [[buffer(1)]],
    device half * output [[buffer(2)]],
    constant LinearParams & params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row = group.y;
    const uint first_output = group.x * 4 + simd_group * 2;
    const uint blocks_per_row = params.in_features / QK8_0;
    float sums[2] = {0.0f, 0.0f};
    const ulong input_base = ulong(row) * params.in_features;
    for (uint column = simd_lane; column < params.in_features; column += 32) {
        const float activation = float(input[input_base + column]);
        for (uint local_output = 0; local_output < 2; ++local_output) {
            const uint output_column = first_output + local_output;
            if (output_column < params.out_features) {
                const ulong weight_base = ulong(output_column) * blocks_per_row;
                sums[local_output] += activation * q8_0_value(
                    weight[weight_base + column / QK8_0], column % QK8_0
                );
            }
        }
    }
    for (uint local_output = 0; local_output < 2; ++local_output) {
        const uint output_column = first_output + local_output;
        const float total = simd_sum(sums[local_output]);
        if (simd_lane == 0 && output_column < params.out_features) {
            output[ulong(row) * params.output_stride
                + params.output_column_offset + output_column] = half(total);
        }
    }
}

kernel void vnext_swiglu_f16(
    device const half * gate_up [[buffer(0)]],
    device half * activation [[buffer(1)]],
    constant SwiGluParams & params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    const uint elements = params.rows * params.intermediate_size;
    if (index >= elements) {
        return;
    }
    const uint row = index / params.intermediate_size;
    const uint column = index % params.intermediate_size;
    const ulong base = ulong(row) * params.gate_up_stride;
    const float gate = float(gate_up[base + column]);
    const float up = float(gate_up[base + params.intermediate_size + column]);
    activation[index] = half((gate / (1.0f + exp(-gate))) * up);
}
