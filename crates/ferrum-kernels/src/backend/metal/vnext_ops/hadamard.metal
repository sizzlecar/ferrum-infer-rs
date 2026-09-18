#include <metal_stdlib>
using namespace metal;

struct HadamardParams {
    uint rows;
    uint width;
    uint block_size;
    uint input_stride;
    uint output_stride;
    uint has_signs;
    uint inverse;
    uint inner_extent;
    uint first_outer_extent;
    uint second_outer_extent;
};

template<typename Input, typename Output>
static inline void normalized_hadamard(
    device const Input * input, device const float * signs, device Output * output,
    constant HadamardParams & p, threadgroup float * values,
    uint3 group, uint lane, uint threads) {
    const uint row = group.y;
    const uint first = group.x * p.block_size;
    if (row >= p.rows || first >= p.width) return;
    for (uint i = lane; i < p.block_size; i += threads) {
        const uint destination = first + i;
        uint source = destination;
        if (p.inner_extent != 0) {
            // Fastest axis first: [inner,first,second] -> [inner,second,first].
            const uint inner = destination % p.inner_extent;
            const uint second = (destination / p.inner_extent) % p.second_outer_extent;
            const uint first_outer = destination / (p.inner_extent * p.second_outer_extent);
            source = inner + p.inner_extent * (first_outer + p.first_outer_extent * second);
        }
        float value = float(input[ulong(row) * p.input_stride + source]);
        if (p.has_signs != 0 && p.inverse == 0) value *= signs[destination];
        values[i] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // One thread owns both sides of a butterfly. All threads reach every
    // barrier, including when a block contains fewer butterflies than threads.
    for (uint stride = 1; stride < p.block_size; stride *= 2) {
        for (uint pair = lane; pair < p.block_size / 2; pair += threads) {
            const uint a = (pair / stride) * (2 * stride) + pair % stride;
            const float left = values[a];
            const float right = values[a + stride];
            values[a] = left + right;
            values[a + stride] = left - right;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float normalization = 1.0f / sqrt(float(p.block_size));
    for (uint i = lane; i < p.block_size; i += threads) {
        const uint destination = first + i;
        float value = values[i] * normalization;
        if (p.has_signs != 0 && p.inverse != 0) value *= signs[destination];
        output[ulong(row) * p.output_stride + destination] = Output(value);
    }
}

#define HADAMARD_KERNEL(INPUT, OUTPUT, SUFFIX) \
kernel void vnext_hadamard_##SUFFIX( \
    device const INPUT * input [[buffer(0)]], device const float * signs [[buffer(1)]], \
    device OUTPUT * output [[buffer(2)]], constant HadamardParams & p [[buffer(3)]], \
    threadgroup float * values [[threadgroup(0)]], \
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]], \
    uint3 threads [[threads_per_threadgroup]]) { \
    normalized_hadamard(input, signs, output, p, values, group, lane, threads.x); \
}

HADAMARD_KERNEL(half, float, f16_f32)
HADAMARD_KERNEL(float, float, f32_f32)
HADAMARD_KERNEL(float, half, f32_f16)
