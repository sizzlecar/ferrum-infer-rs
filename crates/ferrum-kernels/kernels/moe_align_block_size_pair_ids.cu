// vLLM-native MoE align-block-size.
//
// Unlike moe_align_block_size.cu, sorted_token_ids stores the original
// flattened pair id p = token * top_k + slot. The vLLM marlin_moe kernel can
// then read A[p / top_k] directly for gate_up, matching vLLM's data flow and
// avoiding Ferrum's pre-gathered x_packed buffer on the gate_up phase.

#include <cstdint>
#include <cuda_runtime.h>

#define MAX_NUM_EXPERTS 256

extern "C" __global__ void moe_align_block_size_pair_ids_f32(
    const int32_t* __restrict__ expert_ids_per_pair,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ block_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int batch_x_topk,
    int num_experts,
    int block_size,
    int sorted_max_size
) {
    __shared__ int counts[MAX_NUM_EXPERTS];
    __shared__ int counts_padded[MAX_NUM_EXPERTS];
    __shared__ int offsets[MAX_NUM_EXPERTS + 1];
    __shared__ int cursors[MAX_NUM_EXPERTS];

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    for (int e = tid; e < num_experts; e += nthreads) {
        counts[e] = 0;
    }
    __syncthreads();

    for (int i = tid; i < sorted_max_size; i += nthreads) {
        sorted_token_ids[i] = batch_x_topk;
    }

    for (int p = tid; p < batch_x_topk; p += nthreads) {
        int e = expert_ids_per_pair[p];
        if (e >= 0 && e < num_experts) {
            atomicAdd(&counts[e], 1);
        }
    }
    __syncthreads();

    if (tid < num_experts) {
        int c = counts[tid];
        counts_padded[tid] = ((c + block_size - 1) / block_size) * block_size;
    }
    __syncthreads();

    if (tid == 0) {
        int acc = 0;
        for (int e = 0; e < num_experts; e++) {
            offsets[e] = acc;
            acc += counts_padded[e];
        }
        offsets[num_experts] = acc;
        total_tokens_post_pad[0] = acc;
    }
    __syncthreads();

    for (int e = tid; e < num_experts; e += nthreads) {
        cursors[e] = offsets[e];
    }
    __syncthreads();

    for (int p = tid; p < batch_x_topk; p += nthreads) {
        int e = expert_ids_per_pair[p];
        if (e >= 0 && e < num_experts) {
            int slot = atomicAdd(&cursors[e], 1);
            sorted_token_ids[slot] = p;
        }
    }
    __syncthreads();

    int total_blocks = total_tokens_post_pad[0] / block_size;
    for (int b = tid; b < total_blocks; b += nthreads) {
        int row = b * block_size;
        int e = 0;
        for (int ei = 0; ei < num_experts; ei++) {
            if (offsets[ei] <= row && row < offsets[ei + 1]) {
                e = ei;
                break;
            }
        }
        block_ids[b] = e;
    }
}
