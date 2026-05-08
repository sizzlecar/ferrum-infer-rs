// MoE align-block-size: GPU-side prep for the (future) fused Marlin
// MoE kernel. Takes the per-pair expert assignments from Stage 8's
// `route_topk_softmax` (`expert_ids[batch * top_k]`) and produces:
//
//   sorted_token_ids[N_padded]  — flat list of (batch * top_k) pair
//                                  indices, sorted by their assigned
//                                  expert and padded with `numel`
//                                  sentinel inside each expert's
//                                  group up to a `block_size` boundary.
//   expert_ids_per_block[N_padded / block_size]
//                                — which expert each block_size-row
//                                  tile of sorted_token_ids belongs
//                                  to. The fused Marlin kernel reads
//                                  this to know which expert weights
//                                  to load.
//   total_tokens_post_pad[1]    — actual padded token count (used by
//                                  the fused kernel's grid_y dim).
//
// Layout matches vLLM's marlin_moe_wna16 kernel input expectation
// (sorted_token_ids holds pair indices in [0, batch * top_k), the
// fused kernel reads `a[sorted[i] / top_k]` to recover the input row).
//
// Algorithm: single block, ≤ 1024 threads.
//   Pass 1: count pairs per expert (atomic into shared mem).
//   Pass 2: ceil-div to block_size, prefix-sum → expert_offsets.
//   Pass 3: walk pairs, atomic-claim slot in their expert's region,
//           write sorted_token_ids[slot] = pair_idx.
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
        int acc = 0;
        for (int e = 0; e < num_experts; e++) {
            offsets[e] = acc;
            acc += counts_padded[e];
        }
        offsets[num_experts] = acc;
        total_tokens_post_pad[0] = acc;
    }
    __syncthreads();

    // Init cursors[e] = offsets[e] (the next slot to fill for expert e).
    for (int e = tid; e < num_experts; e += nthreads) {
        cursors[e] = offsets[e];
    }
    __syncthreads();

    // Pass 3: scatter pair indices into their expert's slot.
    for (int p = tid; p < batch_x_topk; p += nthreads) {
        int e = expert_ids_per_pair[p];
        if (e >= 0 && e < num_experts) {
            int slot = atomicAdd(&cursors[e], 1);
            sorted_token_ids[slot] = p;
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
