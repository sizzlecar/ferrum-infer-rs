// GPT-OSS routed-MoE auxiliaries.
//
// The expert GEMMs themselves are the native BF16 x MXFP4 Marlin entrypoint.
// These kernels implement the model-specific boundaries around those GEMMs:
//   * F16 activations -> BF16 (the native expert ABI),
//   * BF16 router weight/bias with F16 input and deterministic F32 reduction,
//   * top-K followed by softmax over only the selected logits,
//   * the interleaved, clamped GPT-OSS SwiGLU variant, and
//   * BF16 expert-slot reduction into the public F16 operation output.

#include <cfloat>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

__device__ __forceinline__ float block_sum(float value) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
    }
    return value;
}

template <bool emit_single_token_marlin_blocks>
__device__ __forceinline__ void selected_topk_softmax(
    const __half* __restrict__ logits,
    int32_t* __restrict__ route_ids,
    float* __restrict__ route_weights,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_block_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int tokens,
    int expert_count,
    int top_k,
    int moe_block_size) {
    const int token = static_cast<int>(blockIdx.x);
    if (token >= tokens) {
        return;
    }
    if constexpr (emit_single_token_marlin_blocks) {
        if (tokens != 1 || moe_block_size <= 0) {
            return;
        }
    }

    extern __shared__ float candidates[];
    for (int expert = threadIdx.x; expert < expert_count; expert += blockDim.x) {
        candidates[expert] = __half2float(logits[token * expert_count + expert]);
    }
    __syncthreads();

    // One lane deliberately owns the selection order. Expert counts are at
    // most 256 and K at most 32, so this is a small deterministic tail after
    // the router GEMM. Ascending scans give the required lower-id tie break.
    if (threadIdx.x == 0) {
        float selected[32];
        for (int rank = 0; rank < top_k; ++rank) {
            int best_id = 0;
            float best = -FLT_MAX;
            for (int expert = 0; expert < expert_count; ++expert) {
                const float value = candidates[expert];
                if (value > best) {
                    best = value;
                    best_id = expert;
                }
            }
            route_ids[token * top_k + rank] = best_id;
            selected[rank] = best;
            candidates[best_id] = -FLT_MAX;
        }

        // GPT-OSS applies softmax after top-K, not full softmax followed by a
        // second normalization. The selected values are in descending order.
        const float maximum = selected[0];
        float denominator = 0.0f;
        for (int rank = 0; rank < top_k; ++rank) {
            selected[rank] = expf(selected[rank] - maximum);
            denominator += selected[rank];
        }
        const float inverse = denominator > 0.0f ? 1.0f / denominator : 1.0f / top_k;
        for (int rank = 0; rank < top_k; ++rank) {
            route_weights[token * top_k + rank] = selected[rank] * inverse;
        }

        if constexpr (emit_single_token_marlin_blocks) {
            const int padded_pair_count = top_k * moe_block_size;
            for (int rank = 0; rank < top_k; ++rank) {
                const int expert = route_ids[rank];
                int expert_rank = 0;
                for (int other_rank = 0; other_rank < top_k; ++other_rank) {
                    const int other = route_ids[other_rank];
                    expert_rank += other < expert || (other == expert && other_rank < rank);
                }
                expert_block_ids[expert_rank] = expert;
                const int block_start = expert_rank * moe_block_size;
                sorted_token_ids[block_start] = rank;
                for (int offset = 1; offset < moe_block_size; ++offset) {
                    sorted_token_ids[block_start + offset] = top_k;
                }
            }
            total_tokens_post_pad[0] = padded_pair_count;
        }
    }
}

}  // namespace

extern "C" __global__ void gpt_oss_f16_to_bf16(
    const __half* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int64_t elements) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = __float2bfloat16(__half2float(input[index]));
    }
}

extern "C" __global__ void gpt_oss_router_logits_f16_bf16(
    const __half* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __half* __restrict__ logits,
    int tokens,
    int expert_count,
    int hidden_size) {
    const int problem = static_cast<int>(blockIdx.x);
    const int token = problem / expert_count;
    const int expert = problem - token * expert_count;
    if (token >= tokens) {
        return;
    }
    float partial = 0.0f;
    const int64_t input_base = static_cast<int64_t>(token) * hidden_size;
    const int64_t weight_base = static_cast<int64_t>(expert) * hidden_size;
    for (int hidden = threadIdx.x; hidden < hidden_size; hidden += blockDim.x) {
        partial += __half2float(input[input_base + hidden]) *
                   __bfloat162float(weight[weight_base + hidden]);
    }
    const float total = block_sum(partial);
    if (threadIdx.x == 0) {
        logits[problem] = __float2half(total + __bfloat162float(bias[expert]));
    }
}

extern "C" __global__ void gpt_oss_router_topk_selected_softmax_f16(
    const __half* __restrict__ logits,
    int32_t* __restrict__ route_ids,
    float* __restrict__ route_weights,
    int tokens,
    int expert_count,
    int top_k) {
    selected_topk_softmax<false>(
        logits,
        route_ids,
        route_weights,
        nullptr,
        nullptr,
        nullptr,
        tokens,
        expert_count,
        top_k,
        0);
}

extern "C" __global__ void gpt_oss_router_topk_selected_softmax_f16_single_token_marlin(
    const __half* __restrict__ logits,
    int32_t* __restrict__ route_ids,
    float* __restrict__ route_weights,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_block_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int tokens,
    int expert_count,
    int top_k,
    int moe_block_size) {
    selected_topk_softmax<true>(
        logits,
        route_ids,
        route_weights,
        sorted_token_ids,
        expert_block_ids,
        total_tokens_post_pad,
        tokens,
        expert_count,
        top_k,
        moe_block_size);
}

extern "C" __global__ void gpt_oss_clamped_swiglu_interleaved_bf16(
    const __nv_bfloat16* __restrict__ gate_up,
    __nv_bfloat16* __restrict__ output,
    int logical_intermediate_size,
    int physical_intermediate_size,
    int64_t elements,
    float limit) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int64_t row = index / physical_intermediate_size;
    const int feature = static_cast<int>(index - row * physical_intermediate_size);
    if (feature >= logical_intermediate_size) {
        output[index] = __float2bfloat16(0.0f);
        return;
    }
    const int64_t base = row * (2LL * logical_intermediate_size) + 2LL * feature;
    const float gate = fminf(__bfloat162float(gate_up[base]), limit);
    const float up = fminf(fmaxf(__bfloat162float(gate_up[base + 1]), -limit), limit);
    const float glu = gate / (1.0f + expf(-1.702f * gate));
    output[index] = __float2bfloat16((up + 1.0f) * glu);
}

extern "C" __global__ void gpt_oss_weighted_sum_bf16_to_f16(
    const __nv_bfloat16* __restrict__ slots,
    const float* __restrict__ route_weights,
    __half* __restrict__ output,
    int tokens,
    int top_k,
    int hidden_size,
    int64_t elements) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int token = static_cast<int>(index / hidden_size);
    const int hidden = static_cast<int>(index - static_cast<int64_t>(token) * hidden_size);
    float sum = 0.0f;
    for (int rank = 0; rank < top_k; ++rank) {
        const int64_t slot = (static_cast<int64_t>(token) * top_k + rank) * hidden_size + hidden;
        sum += route_weights[token * top_k + rank] * __bfloat162float(slots[slot]);
    }
    output[index] = __float2half(sum);
}
