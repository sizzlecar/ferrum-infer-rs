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

// Fixed 1024-point Sylvester transform: 256 threads retain four values each.
// Stages 1..16 stay inside a 32-lane SIMD group, stages 32..128 retain the
// generic one-owner-per-butterfly shared-memory algorithm, and stages 256/512
// stay in registers. Every stage preserves the generic left +/- right order.
template<typename Input, typename Output>
static inline void normalized_hadamard_1024(
    device const Input * input, device const float * signs, device Output * output,
    constant HadamardParams & p, threadgroup float * values,
    uint3 group, uint lane, uint subgroup) {
    const uint row = group.y;
    const uint first = group.x * 1024;
    if (row >= p.rows || first >= p.width) return;
    const uint thread_index = subgroup * 32 + lane;
    float local[4];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 4; ++j) {
        const uint destination = first + thread_index + j * 256;
        uint source = destination;
        if (p.inner_extent != 0) {
            const uint inner = destination % p.inner_extent;
            const uint second = (destination / p.inner_extent) % p.second_outer_extent;
            const uint first_outer = destination / (p.inner_extent * p.second_outer_extent);
            source = inner + p.inner_extent * (first_outer + p.first_outer_extent * second);
        }
        float value = float(input[ulong(row) * p.input_stride + source]);
        if (p.has_signs != 0 && p.inverse == 0) value *= signs[destination];
        local[j] = value;
    }
    #pragma clang loop unroll(full)
    for (uint stride = 1; stride < 32; stride *= 2) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) {
            const float own = local[j];
            const float peer = simd_shuffle_xor(own, stride);
            // The high lane owns the right operand: peer - own, never the
            // negation of own - peer. Addition also retains left-first order.
            local[j] = (lane & stride) == 0 ? own + peer : peer - own;
        }
    }
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 4; ++j) values[thread_index + j * 256] = local[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint stride = 32; stride <= 128; stride *= 2) {
        #pragma clang loop unroll(full)
        for (uint pair = thread_index; pair < 512; pair += 256) {
            const uint a = (pair / stride) * (2 * stride) + pair % stride;
            const float left = values[a];
            const float right = values[a + stride];
            values[a] = left + right;
            values[a + stride] = left - right;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 4; ++j) local[j] = values[thread_index + j * 256];
    // Stride 256. Keep both original inputs alive until both outputs exist.
    const float sum01 = local[0] + local[1];
    const float difference01 = local[0] - local[1];
    const float sum23 = local[2] + local[3];
    const float difference23 = local[2] - local[3];
    // Stride 512, followed by exactly the generic normalization/sign steps.
    local[0] = sum01 + sum23;
    local[2] = sum01 - sum23;
    local[1] = difference01 + difference23;
    local[3] = difference01 - difference23;
    const float normalization = 1.0f / sqrt(float(p.block_size));
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 4; ++j) {
        const uint destination = first + thread_index + j * 256;
        float value = local[j] * normalization;
        if (p.has_signs != 0 && p.inverse != 0) value *= signs[destination];
        output[ulong(row) * p.output_stride + destination] = Output(value);
    }
}

#define HADAMARD_1024_KERNEL(INPUT, OUTPUT, SUFFIX) \
kernel void vnext_hadamard_1024_##SUFFIX( \
    device const INPUT * input [[buffer(0)]], device const float * signs [[buffer(1)]], \
    device OUTPUT * output [[buffer(2)]], constant HadamardParams & p [[buffer(3)]], \
    threadgroup float * values [[threadgroup(0)]], \
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
    uint subgroup [[simdgroup_index_in_threadgroup]]) { \
    normalized_hadamard_1024(input, signs, output, p, values, group, lane, subgroup); \
}

HADAMARD_1024_KERNEL(half, float, f16_f32)
HADAMARD_1024_KERNEL(float, float, f32_f32)
HADAMARD_1024_KERNEL(float, half, f32_f16)
