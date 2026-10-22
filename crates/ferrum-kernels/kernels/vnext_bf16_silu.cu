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

// Mirrors Rust NativeMask. Bounded kernel parameters, not an unaccounted
// device metadata allocation. All fields are validated before submission.
struct FerrumStructuredAttentionMask {
    uint32_t segment_count;
    uint32_t segment_ends[32];
    uint32_t bidirectional_mask;
    uint32_t key_run_counts[16];
    uint32_t key_run_ends[16 * 32];
    uint32_t key_valid_masks[16];
};
static_assert(sizeof(FerrumStructuredAttentionMask) == 2312, "attention mask ABI");

// One CTA per (batch, query, head). Each thread owns one query scalar and one
// output accumulator. There is no S*S score matrix or per-thread D-array.
// Finite scores use explicit separate FP32 operations and precise expf.
// Any visible nonfinite score poisons that query's whole head with NaNs;
// visible V values otherwise propagate IEEE arithmetic component-wise.
__device__ __forceinline__ void ferrum_structured_attention_body(
    const uint16_t* query, const uint16_t* key, const uint16_t* value,
    uint16_t* output, uint64_t groups, uint64_t query_heads, uint64_t key_value_heads, uint32_t sequence,
    uint32_t width, float scale, const FerrumStructuredAttentionMask& mask) {
    extern __shared__ float dot_values[];
    __shared__ uint32_t visible_end, seen, poisoned, first_key;
    __shared__ float maximum, denominator, correction, weight, inverse;
    for (uint64_t group = blockIdx.x; group < groups; group += gridDim.x) {
        const uint64_t head = group % query_heads;
        const uint64_t kv_head = head / (query_heads / key_value_heads);
        const uint64_t row = group / query_heads;
        const uint32_t batch = static_cast<uint32_t>(row / sequence);
        const uint32_t q = static_cast<uint32_t>(row % sequence);
        const uint64_t columns = query_heads * width;
        const uint64_t kv_columns = key_value_heads * width;
        const uint64_t query_offset = row * columns + head * width + threadIdx.x;
        float q_scalar = 0.0f;
        float accumulator = 0.0f;
        if (threadIdx.x == 0) {
            visible_end = q + 1;
            for (uint32_t segment = 0; segment < mask.segment_count; ++segment) {
                if (q < mask.segment_ends[segment]) {
                    if ((mask.bidirectional_mask >> segment) & 1U) {
                        visible_end = mask.segment_ends[segment];
                    }
                    break;
                }
            }
            seen = 0;
            poisoned = 0;
        }
        __syncthreads();
        uint32_t run = 0;
        for (uint32_t k = 0; k < visible_end; ++k) {
            while (run + 1 < mask.key_run_counts[batch]
                   && k >= mask.key_run_ends[batch * 32 + run]) {
                ++run;
            }
            // Uniform within the CTA. Excluded keys are not loaded, so a
            // masked NaN/Inf cannot contaminate either scores or values.
            if (((mask.key_valid_masks[batch] >> run) & 1U) == 0) continue;
            // No Q/K/V load for an empty visible set. Q is loaded only at
            // the first visible key and retained in one scalar register.
            if (!seen) q_scalar = ferrum_bf16_decode(query[query_offset]);
            const uint64_t kv_offset = (static_cast<uint64_t>(batch) * sequence + k)
                * kv_columns + kv_head * width + threadIdx.x;
            dot_values[threadIdx.x] = __fmul_rn(q_scalar, ferrum_bf16_decode(key[kv_offset]));
            __syncthreads();
            ferrum_norm_pairwise(dot_values, width);
            if (threadIdx.x == 0) {
                const float score = __fmul_rn(dot_values[0], scale);
                if (!isfinite(score)) {
                    poisoned = 1;
                } else if (!seen) {
                    first_key = 1;
                    seen = 1;
                    maximum = score;
                    denominator = 1.0f;
                } else {
                    first_key = 0;
                    const float next_maximum = fmaxf(maximum, score);
                    correction = expf(__fsub_rn(maximum, next_maximum));
                    weight = expf(__fsub_rn(score, next_maximum));
                    denominator = __fadd_rn(__fmul_rn(denominator, correction), weight);
                    maximum = next_maximum;
                }
            }
            __syncthreads();
            if (poisoned) break;
            const float v = ferrum_bf16_decode(value[kv_offset]);
            accumulator = first_key ? v : __fadd_rn(
                __fmul_rn(accumulator, correction), __fmul_rn(weight, v));
            __syncthreads();
        }
        if (threadIdx.x == 0 && seen && !poisoned) {
            inverse = __fdiv_rn(1.0f, denominator);
        }
        __syncthreads();
        output[query_offset] = poisoned ? 0x7fc0U :
            (seen ? ferrum_bf16_rne(__fmul_rn(accumulator, inverse)) : 0U);
        __syncthreads();
    }
}

// The original equal-head entry preserves its ABI and calls the same body.
extern "C" __global__ void ferrum_vnext_structured_attention_bf16(
    const uint16_t* query, const uint16_t* key, const uint16_t* value,
    uint16_t* output, uint64_t groups, uint64_t heads, uint32_t sequence,
    uint32_t width, float scale, const FerrumStructuredAttentionMask mask) {
    ferrum_structured_attention_body(query, key, value, output, groups,
        heads, heads, sequence, width, scale, mask);
}

// GQA has a separate semantic contract, but does not repeat the KV tensors.
extern "C" __global__ void ferrum_vnext_structured_gqa_attention_bf16(
    const uint16_t* query, const uint16_t* key, const uint16_t* value,
    uint16_t* output, uint64_t groups, uint64_t query_heads, uint64_t key_value_heads,
    uint32_t sequence, uint32_t width, float scale, const FerrumStructuredAttentionMask mask) {
    ferrum_structured_attention_body(query, key, value, output, groups,
        query_heads, key_value_heads, sequence, width, scale, mask);
}

// Whole-row zero-centered RMS. Weight is stored BF16; weight+1 and x*inverse
// remain FP32 until the sole final BF16 conversion. This is deliberately not
// the two-BF16-boundary head RMS implementation.
extern "C" __global__ void ferrum_vnext_zero_centered_rms_norm_bf16(
    const uint16_t* input, const uint16_t* weight, uint16_t* output,
    uint64_t rows, uint32_t width, float epsilon) {
    extern __shared__ float values[];
    __shared__ float inverse;
    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const uint64_t start = row * width;
        for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
            const float x = ferrum_bf16_decode(input[start + column]);
            values[column] = __fmul_rn(x, x);
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
            const float normalized = __fmul_rn(
                ferrum_bf16_decode(input[start + column]), inverse);
            const float scale = __fadd_rn(ferrum_bf16_decode(weight[column]), 1.0f);
            output[start + column] = ferrum_bf16_rne(__fmul_rn(normalized, scale));
        }
        __syncthreads();
    }
}

// GELU(tanh) has one final BF16 boundary. The explicit FP32 evaluation order
// and constant bits form this profile; libdevice's computed FP32 tanh domain
// is not covered by earlier BF16-input tanh qualification. Nonfinite data
// follows IEEE propagation rather than being rejected after submission.
extern "C" __global__ void ferrum_vnext_gelu_tanh_bf16(
    const uint16_t* input, uint16_t* output, uint64_t elements) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    const float beta = __uint_as_float(0x3f4c422aU);
    const float kappa = __uint_as_float(0x3d372713U);
    for (uint64_t index = first; index < elements; index += stride) {
        const float x = ferrum_bf16_decode(input[index]);
        const float xx = __fmul_rn(x, x);
        const float cube = __fmul_rn(xx, x);
        const float polynomial = __fadd_rn(x, __fmul_rn(kappa, cube));
        const float argument = __fmul_rn(beta, polynomial);
        const float activation = __fadd_rn(1.0f, ferrum_tanh(argument));
        const float half_x = __fmul_rn(0.5f, x);
        output[index] = ferrum_bf16_rne(__fmul_rn(half_x, activation));
    }
}

// Pure BF16 bit transport. Metadata is passed by value, not allocated on device.
struct FerrumRowGatherRuns {
    uint64_t run_ends[32];
    uint64_t source_starts[32];
    uint64_t source_batch_strides[32];
    uint64_t source_row_strides[32];
    uint64_t rows_per_batch;
    uint64_t source_width;
    uint64_t column_start;
    uint64_t column_count;
    uint32_t run_count;
    uint32_t padding;
};
static_assert(sizeof(FerrumRowGatherRuns) == 1064, "gather run ABI");

extern "C" __global__ void ferrum_vnext_row_gather_bf16(
    const uint16_t* source, uint16_t* output, uint64_t elements,
    const FerrumRowGatherRuns runs) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = first; index < elements; index += stride) {
        const uint64_t row = index / runs.column_count;
        const uint64_t batch = row / runs.rows_per_batch;
        const uint64_t local_row = row % runs.rows_per_batch;
        uint32_t run = 0;
        while (run + 1 < runs.run_count && local_row >= runs.run_ends[run]) ++run;
        const uint64_t run_start = run == 0 ? 0 : runs.run_ends[run - 1];
        const uint64_t source_row = runs.source_starts[run]
            + batch * runs.source_batch_strides[run]
            + (local_row - run_start) * runs.source_row_strides[run];
        const uint64_t source_column = runs.column_start + index % runs.column_count;
        output[index] = source[source_row * runs.source_width + source_column];
    }
}

extern "C" __global__ void ferrum_vnext_batched_row_concat_bf16(
    const uint16_t* left, const uint16_t* right, uint16_t* output,
    uint64_t elements, uint64_t left_rows, uint64_t right_rows, uint64_t columns) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    const uint64_t rows_per_batch = left_rows + right_rows;
    for (uint64_t index = first; index < elements; index += stride) {
        const uint64_t row = index / columns;
        const uint64_t batch = row / rows_per_batch;
        const uint64_t local_row = row % rows_per_batch;
        const uint64_t column = index % columns;
        output[index] = local_row < left_rows
            ? left[(batch * left_rows + local_row) * columns + column]
            : right[(batch * right_rows + local_row - left_rows) * columns + column];
    }
}

// Plain residual and half-split BF16 text rotary use the same bounded module.
extern "C" __global__ void ferrum_vnext_plain_residual_add_bf16(
    const uint16_t* lhs, const uint16_t* rhs, uint16_t* output, uint64_t elements) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = first; index < elements; index += stride) {
        output[index] = ferrum_bf16_rne(__fadd_rn(
            ferrum_bf16_decode(lhs[index]), ferrum_bf16_decode(rhs[index])));
    }
}

// One thread per half-split pair. Cos and sin are BF16 [R,D], independently
// specified for both halves and shared only across heads of a given row.
// Each multiplication is explicitly rounded to BF16 before the final sum.
extern "C" __global__ void ferrum_vnext_half_split_rope_bf16(
    const uint16_t* input, const uint16_t* cosine, const uint16_t* sine,
    uint16_t* output, uint64_t pairs, uint64_t columns, uint64_t head_dim) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    const uint64_t half = head_dim / 2;
    const uint64_t pairs_per_row = columns / 2;
    for (uint64_t pair = first; pair < pairs; pair += stride) {
        const uint64_t row = pair / pairs_per_row;
        const uint64_t row_pair = pair % pairs_per_row;
        const uint64_t head = row_pair / half;
        const uint64_t channel = row_pair % half;
        const uint64_t left = row * columns + head * head_dim + channel;
        const uint64_t right = left + half;
        const uint64_t frequency = row * head_dim + channel;
        const float a = ferrum_bf16_decode(input[left]);
        const float b = ferrum_bf16_decode(input[right]);
        const uint16_t ac = ferrum_bf16_rne(__fmul_rn(a, ferrum_bf16_decode(cosine[frequency])));
        const uint16_t bs = ferrum_bf16_rne(__fmul_rn(-b, ferrum_bf16_decode(sine[frequency])));
        const uint16_t bc = ferrum_bf16_rne(__fmul_rn(b, ferrum_bf16_decode(cosine[frequency + half])));
        const uint16_t as = ferrum_bf16_rne(__fmul_rn(a, ferrum_bf16_decode(sine[frequency + half])));
        output[left] = ferrum_bf16_rne(__fadd_rn(ferrum_bf16_decode(ac), ferrum_bf16_decode(bs)));
        output[right] = ferrum_bf16_rne(__fadd_rn(ferrum_bf16_decode(bc), ferrum_bf16_decode(as)));
    }
}
