// Fixed-tensor SiLU profile: precise libdevice expf, separate RN32 add/div,
// then integer BF16 RN-even. This is not the CPU libm numerical profile.
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

extern "C" __global__ void ferrum_vnext_silu_bf16(
    const uint16_t* input, uint16_t* output, uint64_t elements) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = first; index < elements; index += stride) {
        const float x = __uint_as_float(static_cast<uint32_t>(input[index]) << 16);
        const float exponential = ferrum_silu_exp(-x);
        const float denominator = __fadd_rn(1.0f, exponential);
        output[index] = ferrum_bf16_rne(__fdiv_rn(x, denominator));
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
