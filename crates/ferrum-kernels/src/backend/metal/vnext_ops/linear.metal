#include <metal_stdlib>
using namespace metal;

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

struct block_q8_0 {
    half d;
    char qs[QK8_0];
};

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

// Narrow dense projections expose too few groups when each SIMD computes two
// outputs. Give one output a whole group and partition K across its SIMDs.
// Inputs and weights retain their F16 ABI; all products and both reductions
// remain F32 until the single final output conversion.
template<uint THREADS>
static inline void linear_dense_narrow_f16(
    device const half * input, device const half * weight, device half * output,
    constant LinearParams & params, uint3 group, uint thread_index,
    uint simd_lane, uint simd_group, threadgroup float * partials) {
    const uint row = group.y;
    const uint output_column = group.x;
    // Uniform across the group, including every thread reaching the barrier.
    if (row >= params.rows || output_column >= params.out_features) return;
    const ulong input_base = ulong(row) * params.in_features;
    const ulong weight_base = ulong(output_column) * params.in_features;
    float sum = 0.0f;
    for (ulong column = thread_index; column < params.in_features; column += THREADS) {
        sum += float(input[input_base + column]) * float(weight[weight_base + column]);
    }
    const float simd_total = simd_sum(sum);
    if (simd_lane == 0) partials[simd_group] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        const float partial = simd_lane < THREADS / 32 ? partials[simd_lane] : 0.0f;
        const float total = simd_sum(partial);
        if (simd_lane == 0) {
            output[ulong(row) * params.output_stride
                + params.output_column_offset + output_column] = half(total);
        }
    }
}

#define NARROW_DENSE_F16(THREADS) \
kernel void vnext_linear_dense_narrow_f16_##THREADS( \
    device const half * input [[buffer(0)]], device const half * weight [[buffer(1)]], \
    device half * output [[buffer(2)]], constant LinearParams & params [[buffer(3)]], \
    uint3 group [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float partials[THREADS / 32]; \
    linear_dense_narrow_f16<THREADS>(input, weight, output, params, group, \
        thread_index, simd_lane, simd_group, partials); \
}
NARROW_DENSE_F16(128)
NARROW_DENSE_F16(256)
#undef NARROW_DENSE_F16

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

template<typename Output>
static inline void linear_dense_f32(
    device const float * input, device const half * weight, device Output * output,
    constant LinearParams & params, uint3 group, uint simd_lane, uint simd_group) {
    const uint row = group.y;
    const uint first_output = group.x * 4 + simd_group * 2;
    float sums[2] = {0.0f, 0.0f};
    const ulong input_base = ulong(row) * params.in_features;
    for (uint column = simd_lane; column < params.in_features; column += 32) {
        const float activation = input[input_base + column];
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
                + params.output_column_offset + output_column] = Output(total);
        }
    }
}

kernel void vnext_linear_dense_f32(
    device const float * input [[buffer(0)]], device const half * weight [[buffer(1)]],
    device float * output [[buffer(2)]], constant LinearParams & params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    linear_dense_f32(input, weight, output, params, group, simd_lane, simd_group);
}

kernel void vnext_linear_dense_f32_f16(
    device const float * input [[buffer(0)]], device const half * weight [[buffer(1)]],
    device half * output [[buffer(2)]], constant LinearParams & params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    linear_dense_f32(input, weight, output, params, group, simd_lane, simd_group);
}

kernel void vnext_linear_q8_0_f32(
    device const float * input [[buffer(0)]],
    device const block_q8_0 * weight [[buffer(1)]],
    device float * output [[buffer(2)]],
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
        const float activation = input[input_base + column];
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
                + params.output_column_offset + output_column] = total;
        }
    }
}
