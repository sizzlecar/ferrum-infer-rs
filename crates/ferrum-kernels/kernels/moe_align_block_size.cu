// MoE align-block-size: GPU-side prep for the (future) fused Marlin
// MoE kernel. Takes the per-pair expert assignments from Stage 8's
// `route_topk_softmax` (`expert_ids[batch * top_k]`) and produces:
//
//   sorted_token_ids[N_padded]  — for each expert e's padded region,
//                                  contains the UNPADDED packed_row
//                                  indices that belong to e (i.e.
//                                  unpadded_offsets[e] + 0, +1, ...).
//                                  Remaining slots filled with sentinel
//                                  = batch * top_k. The vLLM marlin MoE
//                                  kernel reads `a[sorted[i] / top_k]`;
//                                  ferrum passes pre-gathered x_packed
//                                  with top_k=1 so the kernel reads
//                                  x_packed[packed_row]. The value here
//                                  MUST be the packed_row (matching
//                                  moe_build_pairs.cu output), NOT the
//                                  pair index — see Pass 3 comment.
//   expert_ids_per_block[N_padded / block_size]
//                                — which expert each block_size-row
//                                  tile of sorted_token_ids belongs
//                                  to. The fused Marlin kernel reads
//                                  this to know which expert weights
//                                  to load.
//   total_tokens_post_pad[1]    — actual padded token count (used by
//                                  the fused kernel's grid_y dim).
//
// Algorithm: single block, ≤ 1024 threads.
//   Pass 1: count pairs per expert (atomic into shared mem).
//   Pass 2: ceil-div to block_size, prefix-sum into BOTH
//           padded `offsets` AND `unpadded_offsets`.
//   Pass 3: walk pairs, atomic-claim padded slot in expert's region,
//           write unpadded_offsets[e] + per_expert_pos.
//   Pass 4: fill expert_ids_per_block[block_idx] = expert_id for
//           each block_size-row tile.
//   Pass 5: thread 0 writes total_tokens_post_pad.
//
// Single-block design works for small num_experts × top_k (Qwen3-MoE
// has 128 × 8 = 1024 pairs, all fits comfortably). For larger configs
// (DeepSeek 256 × 8 = 2048 pairs) consider splitting across blocks.

#include <cstdint>
#include <cuda_runtime.h>

#define MAX_NUM_EXPERTS 256

extern "C" __global__ void moe_align_block_size_f32(
    const int32_t* __restrict__ expert_ids_per_pair,  // [B*K] i32
    int32_t*       __restrict__ sorted_token_ids,      // [N_padded] i32
    int32_t*       __restrict__ block_ids,             // [N_padded / block_size] i32
    int32_t*       __restrict__ total_tokens_post_pad, // [1] i32
    int batch_x_topk,           // numel = batch * top_k
    int num_experts,
    int block_size,             // e.g. 16 for Marlin's M_BLOCK
    int sorted_max_size         // = N_padded sentinel (num_experts * ceil(numel / block_size) * block_size, upper bound)
) {
    __shared__ int counts[MAX_NUM_EXPERTS];
    __shared__ int counts_padded[MAX_NUM_EXPERTS];
    __shared__ int offsets[MAX_NUM_EXPERTS + 1];
    __shared__ int cursors[MAX_NUM_EXPERTS];
    // Unpadded prefix-sum offsets — needed so the value written into
    // sorted_token_ids matches the packed_row produced by
    // moe_build_pairs.cu (which uses the same unpadded offsets via
    // moe_build_pairs_by_token::expert_offsets). The vLLM marlin MoE
    // kernel reads A[sorted_token_ids[i] / top_k], and A (= ferrum's
    // x_packed) is gathered in unpadded-packed-row order.
    __shared__ int unpadded_offsets[MAX_NUM_EXPERTS];

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Init counts to 0.
    for (int e = tid; e < num_experts; e += nthreads) {
        counts[e] = 0;
    }
    __syncthreads();

    // Init sorted_token_ids to sentinel (= batch_x_topk; the fused
    // Marlin kernel treats indices ≥ batch_x_topk as "no row, skip").
    for (int i = tid; i < sorted_max_size; i += nthreads) {
        sorted_token_ids[i] = batch_x_topk;
    }

    // Pass 1: per-expert count.
    for (int p = tid; p < batch_x_topk; p += nthreads) {
        int e = expert_ids_per_pair[p];
        if (e >= 0 && e < num_experts) {
            atomicAdd(&counts[e], 1);
        }
    }
    __syncthreads();

    // Pass 2: round counts up to block_size. Single thread does the
    // exclusive prefix-sum (num_experts ≤ 256 — trivial cost).
    if (tid < num_experts) {
        int c = counts[tid];
        int padded = ((c + block_size - 1) / block_size) * block_size;
        counts_padded[tid] = padded;
    }
    __syncthreads();

    if (tid == 0) {
        int acc_pad = 0;
        int acc_unpad = 0;
        for (int e = 0; e < num_experts; e++) {
            offsets[e] = acc_pad;
            unpadded_offsets[e] = acc_unpad;
            acc_pad += counts_padded[e];
            acc_unpad += counts[e];
        }
        offsets[num_experts] = acc_pad;
        total_tokens_post_pad[0] = acc_pad;
    }
    __syncthreads();

    // Init cursors[e] = offsets[e] (the next slot to fill for expert e).
    for (int e = tid; e < num_experts; e += nthreads) {
        cursors[e] = offsets[e];
    }
    __syncthreads();

    // Pass 3: scatter UNPADDED packed_row into expert e's padded slot.
    //
    // sorted_token_ids[padded_slot] = unpadded_offsets[e] + per_expert_pos
    //
    // This matches the value moe_build_pairs.cu writes into pairs_by_token
    // and the index into packed_token_idx[] that embedding_lookup_dev uses
    // to gather x into x_packed. The vLLM marlin MoE kernel then reads
    // A[sorted_token_ids[i] / top_k] (= x_packed[packed_row]) — correct
    // only when the value here is the packed_row, NOT the pair_index.
    for (int p = tid; p < batch_x_topk; p += nthreads) {
        int e = expert_ids_per_pair[p];
        if (e >= 0 && e < num_experts) {
            int slot = atomicAdd(&cursors[e], 1);
            int per_expert_pos = slot - offsets[e];
            sorted_token_ids[slot] = unpadded_offsets[e] + per_expert_pos;
        }
    }
    __syncthreads();

    // Pass 4: fill block_ids[]. Each block of `block_size` rows in
    // sorted_token_ids belongs to ONE expert (because counts are
    // padded to block_size). For tile i in [0, total_blocks):
    //   block_ids[i] = expert e such that offsets[e] <= i*block_size
    //                                       < offsets[e+1].
    int total_blocks = total_tokens_post_pad[0] / block_size;
    for (int b = tid; b < total_blocks; b += nthreads) {
        int row = b * block_size;
        // Linear scan over experts (≤ 256). Cheap.
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
