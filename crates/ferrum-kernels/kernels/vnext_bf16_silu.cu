// Fixed-tensor BF16 pointwise bundle. Precise libdevice expf/tanhf and
// separate RN32 operations retain every declared BF16 boundary. This is not
// the CPU libm numerical profile and does not certify deployment parity.
// build.rs compiles this translation unit with --ftz=false --prec-div=true
// --prec-sqrt=true --fmad=false and without --use_fast_math.
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

__device__ __forceinline__ float ferrum_silu_exp(float value) {
    return expf(value);
}

__device__ __forceinline__ uint16_t ferrum_bf16_rne(float value) {
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        // NaN classification is contractual; payload preservation is not.
        return static_cast<uint16_t>((bits >> 16) | 0x0040U);
    }
    const uint32_t rounded = bits + 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<uint16_t>(rounded >> 16);
}

__device__ __forceinline__ float ferrum_bf16_decode(uint16_t value) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

__device__ __forceinline__ uint16_t ferrum_silu_bf16_value(float x) {
    const float exponential = ferrum_silu_exp(-x);
    const float denominator = __fadd_rn(1.0f, exponential);
    return ferrum_bf16_rne(__fdiv_rn(x, denominator));
}

__device__ __forceinline__ float ferrum_tanh(float x) {
    return tanhf(x);
}

extern "C" __global__ void ferrum_vnext_silu_bf16(
    const uint16_t* input, uint16_t* output, uint64_t elements) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = first; index < elements; index += stride) {
        output[index] = ferrum_silu_bf16_value(ferrum_bf16_decode(input[index]));
    }
}

extern "C" __global__ void ferrum_vnext_silu_mul_bf16(
    const uint16_t* gate, const uint16_t* up, uint16_t* output, uint64_t elements) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = first; index < elements; index += stride) {
        const uint16_t activated = ferrum_silu_bf16_value(ferrum_bf16_decode(gate[index]));
        output[index] = ferrum_bf16_rne(__fmul_rn(
            ferrum_bf16_decode(activated), ferrum_bf16_decode(up[index])));
    }
}

extern "C" __global__ void ferrum_vnext_tanh_bf16(
    const uint16_t* input, uint16_t* output, uint64_t elements) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = first; index < elements; index += stride) {
        output[index] = ferrum_bf16_rne(ferrum_tanh(ferrum_bf16_decode(input[index])));
    }
}

extern "C" __global__ void ferrum_vnext_modulation_scale_bf16(
    const uint16_t* input, const uint16_t* scale, uint16_t* output,
    uint64_t elements, uint64_t width, uint32_t row_broadcast) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = first; index < elements; index += stride) {
        const uint64_t parameter = row_broadcast ? index % width : index;
        const uint16_t multiplier = ferrum_bf16_rne(__fadd_rn(1.0f, ferrum_bf16_decode(scale[parameter])));
        output[index] = ferrum_bf16_rne(__fmul_rn(
            ferrum_bf16_decode(input[index]), ferrum_bf16_decode(multiplier)));
    }
}

extern "C" __global__ void ferrum_vnext_gated_residual_bf16(
    const uint16_t* input, const uint16_t* gate, const uint16_t* update, uint16_t* output,
    uint64_t elements, uint64_t width, uint32_t row_broadcast) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = first; index < elements; index += stride) {
        const uint64_t parameter = row_broadcast ? index % width : index;
        const uint16_t scaled_update = ferrum_bf16_rne(__fmul_rn(
            ferrum_bf16_decode(gate[parameter]), ferrum_bf16_decode(update[index])));
        output[index] = ferrum_bf16_rne(__fadd_rn(
            ferrum_bf16_decode(input[index]), ferrum_bf16_decode(scaled_update)));
    }
}

// Private qualification probe, not a model operation. It shares the exact
// production expf call and compile options; the Rust test checks its FP32 bits
// independently before checking the composed BF16 output.
extern "C" __global__ void ferrum_vnext_silu_exp_probe(
    const uint16_t* input, uint32_t* output, uint64_t elements) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = first; index < elements; index += stride) {
        const float x = __uint_as_float(static_cast<uint32_t>(input[index]) << 16);
        output[index] = __float_as_uint(ferrum_silu_exp(-x));
    }
}

// Independent tanh qualification can inspect this exact production primitive.
extern "C" __global__ void ferrum_vnext_tanh_probe(
    const uint16_t* input, uint32_t* output, uint64_t elements) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = first; index < elements; index += stride) {
        output[index] = __float_as_uint(ferrum_tanh(ferrum_bf16_decode(input[index])));
    }
}
