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

// Power-of-two adjacent-pair tree. Each stage writes only the left endpoint
// of a disjoint subtree, so no thread overwrites another thread's input.
// The provider validates 1 <= width <= 4096 and launch shared bytes=4*width.
__device__ __forceinline__ void ferrum_norm_pairwise(float* values, uint32_t width) {
    for (uint32_t stride = 1; stride < width; stride *= 2) {
        for (uint32_t left = threadIdx.x * (2 * stride); left < width;
             left += blockDim.x * (2 * stride)) {
            values[left] = __fadd_rn(values[left], values[left + stride]);
        }
        __syncthreads();
    }
}

// Two-pass population LayerNorm, no affine weights. Shared memory is bounded
// CTA storage (4*width dynamic bytes + two static scalars), not a global
// workspace. Nonfinite data follows IEEE arithmetic; preflight is structural.
extern "C" __global__ void ferrum_vnext_layer_norm_bf16(
    const uint16_t* input, uint16_t* output,
    uint64_t rows, uint32_t width, float epsilon) {
    extern __shared__ float values[];
    __shared__ float mean;
    __shared__ float inverse;
    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const uint64_t start = row * width;
        for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
            values[column] = ferrum_bf16_decode(input[start + column]);
        }
        __syncthreads();
        ferrum_norm_pairwise(values, width);
        if (threadIdx.x == 0) {
            mean = __fdiv_rn(values[0], static_cast<float>(width));
        }
        __syncthreads();
        for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
            const float centered = __fsub_rn(ferrum_bf16_decode(input[start + column]), mean);
            values[column] = __fmul_rn(centered, centered);
        }
        __syncthreads();
        ferrum_norm_pairwise(values, width);
        if (threadIdx.x == 0) {
            const float variance = __fdiv_rn(values[0], static_cast<float>(width));
            const float root = __fsqrt_rn(__fadd_rn(variance, epsilon));
            inverse = __fdiv_rn(1.0f, root);
        }
        __syncthreads();
        for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
            const float centered = __fsub_rn(ferrum_bf16_decode(input[start + column]), mean);
            output[start + column] = ferrum_bf16_rne(__fmul_rn(centered, inverse));
        }
        // All readers finish before a block reuses shared storage for its next row.
        __syncthreads();
    }
}

// Contiguous [rows, columns] is divided into adjacent heads of width D.
// gamma[D] is shared across heads, and the normalized BF16 boundary precedes
// the affine multiplication. This is not zero-centered RMSNorm or RoPE.
extern "C" __global__ void ferrum_vnext_head_rms_norm_bf16(
    const uint16_t* input, const uint16_t* gamma, uint16_t* output,
    uint64_t groups, uint32_t width, float epsilon) {
    extern __shared__ float values[];
    __shared__ float inverse;
    for (uint64_t group = blockIdx.x; group < groups; group += gridDim.x) {
        const uint64_t start = group * width;
        for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
            const float value = ferrum_bf16_decode(input[start + column]);
            values[column] = __fmul_rn(value, value);
        }
        __syncthreads();
        ferrum_norm_pairwise(values, width);
        if (threadIdx.x == 0) {
            const float mean_square = __fdiv_rn(values[0], static_cast<float>(width));
            const float root = __fsqrt_rn(__fadd_rn(mean_square, epsilon));
            inverse = __fdiv_rn(1.0f, root);
        }
        __syncthreads();
        for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
            const uint16_t normalized = ferrum_bf16_rne(__fmul_rn(
                ferrum_bf16_decode(input[start + column]), inverse));
            output[start + column] = ferrum_bf16_rne(__fmul_rn(
                ferrum_bf16_decode(normalized), ferrum_bf16_decode(gamma[column])));
        }
        __syncthreads();
    }
}

// Explicit FMA is intentional even with --fmad=false: that option prevents
// unintended contraction, not the declared fused intrinsic. Frequencies are
// [sequence, head_dim/2, 2] F32, shared across batches and heads. No in-place
// output is accepted by the provider and no frequency table is synthesized.
extern "C" __global__ void ferrum_vnext_adjacent_rope_bf16(
    const uint16_t* input, const float* frequencies, uint16_t* output,
    uint64_t pairs, uint64_t columns, uint64_t sequence, uint64_t head_dim) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    const uint64_t pairs_per_row = columns / 2;
    const uint64_t pairs_per_head = head_dim / 2;
    for (uint64_t pair = first; pair < pairs; pair += stride) {
        const uint64_t row = pair / pairs_per_row;
        const uint64_t local_pair = (pair % pairs_per_row) % pairs_per_head;
        const uint64_t frequency = (row % sequence) * head_dim + 2 * local_pair;
        const float a = ferrum_bf16_decode(input[2 * pair]);
        const float b = ferrum_bf16_decode(input[2 * pair + 1]);
        const float c = frequencies[frequency];
        const float s = frequencies[frequency + 1];
        const float negative_product = -__fmul_rn(b, s);
        const float positive_product = __fmul_rn(a, s);
        output[2 * pair] = ferrum_bf16_rne(__fmaf_rn(a, c, negative_product));
        output[2 * pair + 1] = ferrum_bf16_rne(__fmaf_rn(b, c, positive_product));
    }
}
