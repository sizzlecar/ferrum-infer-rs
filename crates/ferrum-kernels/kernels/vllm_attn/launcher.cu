// vLLM paged_attention_v2 — ferrum extern "C" launcher.
//
// Wraps `vllm::paged_attention_v2_kernel` + its companion reduce kernel
// from `attention_kernels.cuh`, exporting a torch-free entry point we
// can call from cudarc. Instantiates the FP16 block_size=16 shapes needed
// by the CUDA model lanes: head_size=128 for Qwen3-30B-A3B and
// head_size=256 for Qwen3.5 full-attention layers.
//
// Replaces ferrum's `paged_batched_flash_decode_attn_f16` for the c=32
// decode hot path identified in nsys 2026-05-12
// (`docs/bench/v0.2-cuda/results_nsys_20260512_105538`): that kernel
// averaged 115 µs / call × 36 calls/iter = 4.3 ms per iter at c=32 and
// is ~91 MB read-bound, so the v2 kernel's explicit multi-partition
// pattern is where vLLM's edge comes from at the c=32 cliff.

#include "attention_kernels.cuh"
#include <cuda_runtime.h>

extern "C" void ferrum_vllm_paged_attention_v1_f16_h128_b16(
    __half* __restrict__ out,           // [num_seqs, num_heads, 128]
    const __half* __restrict__ query,   // [num_seqs, num_heads, 128]
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int num_seqs,
    const int num_heads,
    const int max_num_blocks_per_seq,
    const int q_stride,                 // query.stride(0)
    const int kv_block_stride,          // key_cache.stride(0)
    const int kv_head_stride,           // key_cache.stride(1)
    const int max_seq_len,
    cudaStream_t stream
) {
    constexpr int HEAD_SIZE = 128;
    constexpr int BLOCK_SIZE = 16;
    constexpr int NUM_THREADS = 128;
    constexpr bool IS_BLOCK_SPARSE = false;
    constexpr vllm::Fp8KVCacheDataType KV_DTYPE = vllm::Fp8KVCacheDataType::kAuto;
    const float* k_scale_ptr = nullptr;
    const float* v_scale_ptr = nullptr;
    const int tp_rank = 0;
    const int blocksparse_local_blocks = 0;
    const int blocksparse_vert_stride = 0;
    const int blocksparse_block_size = 0;
    const int blocksparse_head_sliding_step = 0;
    const float* alibi_slopes_ptr = nullptr;

    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    int padded_max_seq_len = ((max_seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    int logits_size = padded_max_seq_len * sizeof(float);
    int outputs_size = (NUM_WARPS / 2) * HEAD_SIZE * sizeof(float);
    int shared_mem_size =
        logits_size > outputs_size ? logits_size : outputs_size;

    dim3 grid(num_heads, num_seqs, 1);
    dim3 block(NUM_THREADS);
    vllm::paged_attention_v1_kernel<
        uint16_t, uint16_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, KV_DTYPE,
        IS_BLOCK_SPARSE>
        <<<grid, block, shared_mem_size, stream>>>(
            reinterpret_cast<uint16_t*>(out),
            reinterpret_cast<const uint16_t*>(query),
            reinterpret_cast<const uint16_t*>(key_cache),
            reinterpret_cast<const uint16_t*>(value_cache), num_kv_heads,
            scale, block_tables, seq_lens, max_num_blocks_per_seq,
            alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,
            k_scale_ptr, v_scale_ptr, tp_rank, blocksparse_local_blocks,
            blocksparse_vert_stride, blocksparse_block_size,
            blocksparse_head_sliding_step);
}

extern "C" void ferrum_vllm_paged_attention_v1_f16_h256_b16(
    __half* __restrict__ out,           // [num_seqs, num_heads, 256]
    const __half* __restrict__ query,   // [num_seqs, num_heads, 256]
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int num_seqs,
    const int num_heads,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const int max_seq_len,
    cudaStream_t stream
) {
    constexpr int HEAD_SIZE = 256;
    constexpr int BLOCK_SIZE = 16;
    constexpr int NUM_THREADS = 128;
    constexpr bool IS_BLOCK_SPARSE = false;
    constexpr vllm::Fp8KVCacheDataType KV_DTYPE = vllm::Fp8KVCacheDataType::kAuto;
    const float* k_scale_ptr = nullptr;
    const float* v_scale_ptr = nullptr;
    const int tp_rank = 0;
    const int blocksparse_local_blocks = 0;
    const int blocksparse_vert_stride = 0;
    const int blocksparse_block_size = 0;
    const int blocksparse_head_sliding_step = 0;
    const float* alibi_slopes_ptr = nullptr;

    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    int padded_max_seq_len = ((max_seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    int logits_size = padded_max_seq_len * sizeof(float);
    int outputs_size = (NUM_WARPS / 2) * HEAD_SIZE * sizeof(float);
    int shared_mem_size =
        logits_size > outputs_size ? logits_size : outputs_size;

    dim3 grid(num_heads, num_seqs, 1);
    dim3 block(NUM_THREADS);
    vllm::paged_attention_v1_kernel<
        uint16_t, uint16_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, KV_DTYPE,
        IS_BLOCK_SPARSE>
        <<<grid, block, shared_mem_size, stream>>>(
            reinterpret_cast<uint16_t*>(out),
            reinterpret_cast<const uint16_t*>(query),
            reinterpret_cast<const uint16_t*>(key_cache),
            reinterpret_cast<const uint16_t*>(value_cache), num_kv_heads,
            scale, block_tables, seq_lens, max_num_blocks_per_seq,
            alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,
            k_scale_ptr, v_scale_ptr, tp_rank, blocksparse_local_blocks,
            blocksparse_vert_stride, blocksparse_block_size,
            blocksparse_head_sliding_step);
}

extern "C" void ferrum_vllm_paged_attention_v2_f16_h128_b16(
    __half* __restrict__ out,           // [num_seqs, num_heads, 128]
    float* __restrict__ exp_sums,       // [num_seqs, num_heads, max_partitions]
    float* __restrict__ max_logits,     // [num_seqs, num_heads, max_partitions]
    __half* __restrict__ tmp_out,       // [num_seqs, num_heads, max_partitions, 128]
    const __half* __restrict__ query,   // [num_seqs, num_heads, 128]
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int num_seqs,
    const int num_heads,
    const int max_num_blocks_per_seq,
    const int q_stride,                 // query.stride(0)
    const int kv_block_stride,          // key_cache.stride(0)
    const int kv_head_stride,           // key_cache.stride(1)
    const int max_seq_len,
    cudaStream_t stream
) {
    constexpr int HEAD_SIZE = 128;
    constexpr int BLOCK_SIZE = 16;
    constexpr int NUM_THREADS = 128;
    constexpr int PARTITION_SIZE = 512;
    constexpr bool IS_BLOCK_SPARSE = false;
    constexpr vllm::Fp8KVCacheDataType KV_DTYPE = vllm::Fp8KVCacheDataType::kAuto;
    // Dummy fp8 scale ptrs — kernel only reads them inside the (kAuto-false)
    // FP8 dequant arm, so they can stay null for our FP16 path.
    const float* k_scale_ptr = nullptr;
    const float* v_scale_ptr = nullptr;
    // Block-sparse params unused when IS_BLOCK_SPARSE=false.
    const int tp_rank = 0;
    const int blocksparse_local_blocks = 0;
    const int blocksparse_vert_stride = 0;
    const int blocksparse_block_size = 0;
    const int blocksparse_head_sliding_step = 0;
    // ALiBi not used by Qwen3 — pass nullptr to disable.
    const float* alibi_slopes_ptr = nullptr;

    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    int max_num_partitions =
        (max_seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
    int logits_size = PARTITION_SIZE * sizeof(float);
    int outputs_size = (NUM_WARPS / 2) * HEAD_SIZE * sizeof(float);
    int shared_mem_size =
        logits_size > outputs_size ? logits_size : outputs_size;

    dim3 grid(num_heads, num_seqs, max_num_partitions);
    dim3 block(NUM_THREADS);
    // vLLM's dtype machinery (`Vec<uint16_t,N>::Type`, `from_float(uint16_t&, float)`)
    // is specialized for `uint16_t` as the half-precision scalar — never
    // `__half` directly. Cast our __half pointers to uint16_t* for the kernel
    // call. Bitwise identical, just template-matching plumbing.
    vllm::paged_attention_v2_kernel<
        uint16_t, uint16_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, KV_DTYPE,
        IS_BLOCK_SPARSE, PARTITION_SIZE>
        <<<grid, block, shared_mem_size, stream>>>(
            exp_sums, max_logits, reinterpret_cast<uint16_t*>(tmp_out),
            reinterpret_cast<const uint16_t*>(query),
            reinterpret_cast<const uint16_t*>(key_cache),
            reinterpret_cast<const uint16_t*>(value_cache), num_kv_heads,
            scale, block_tables, seq_lens, max_num_blocks_per_seq,
            alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,
            k_scale_ptr, v_scale_ptr, tp_rank, blocksparse_local_blocks,
            blocksparse_vert_stride, blocksparse_block_size,
            blocksparse_head_sliding_step);

    dim3 reduce_grid(num_heads, num_seqs);
    int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);
    vllm::paged_attention_v2_reduce_kernel<
        uint16_t, HEAD_SIZE, NUM_THREADS, PARTITION_SIZE>
        <<<reduce_grid, block, reduce_shared_mem_size, stream>>>(
            reinterpret_cast<uint16_t*>(out), exp_sums, max_logits,
            reinterpret_cast<const uint16_t*>(tmp_out), seq_lens,
            max_num_partitions);
}

extern "C" void ferrum_vllm_paged_attention_v2_f16_h256_b16(
    __half* __restrict__ out,           // [num_seqs, num_heads, 256]
    float* __restrict__ exp_sums,       // [num_seqs, num_heads, max_partitions]
    float* __restrict__ max_logits,     // [num_seqs, num_heads, max_partitions]
    __half* __restrict__ tmp_out,       // [num_seqs, num_heads, max_partitions, 256]
    const __half* __restrict__ query,   // [num_seqs, num_heads, 256]
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int num_seqs,
    const int num_heads,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const int max_seq_len,
    cudaStream_t stream
) {
    constexpr int HEAD_SIZE = 256;
    constexpr int BLOCK_SIZE = 16;
    constexpr int NUM_THREADS = 128;
    constexpr int PARTITION_SIZE = 512;
    constexpr bool IS_BLOCK_SPARSE = false;
    constexpr vllm::Fp8KVCacheDataType KV_DTYPE = vllm::Fp8KVCacheDataType::kAuto;
    const float* k_scale_ptr = nullptr;
    const float* v_scale_ptr = nullptr;
    const int tp_rank = 0;
    const int blocksparse_local_blocks = 0;
    const int blocksparse_vert_stride = 0;
    const int blocksparse_block_size = 0;
    const int blocksparse_head_sliding_step = 0;
    const float* alibi_slopes_ptr = nullptr;

    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    int max_num_partitions =
        (max_seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
    int logits_size = PARTITION_SIZE * sizeof(float);
    int outputs_size = (NUM_WARPS / 2) * HEAD_SIZE * sizeof(float);
    int shared_mem_size =
        logits_size > outputs_size ? logits_size : outputs_size;

    dim3 grid(num_heads, num_seqs, max_num_partitions);
    dim3 block(NUM_THREADS);
    vllm::paged_attention_v2_kernel<
        uint16_t, uint16_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, KV_DTYPE,
        IS_BLOCK_SPARSE, PARTITION_SIZE>
        <<<grid, block, shared_mem_size, stream>>>(
            exp_sums, max_logits, reinterpret_cast<uint16_t*>(tmp_out),
            reinterpret_cast<const uint16_t*>(query),
            reinterpret_cast<const uint16_t*>(key_cache),
            reinterpret_cast<const uint16_t*>(value_cache), num_kv_heads,
            scale, block_tables, seq_lens, max_num_blocks_per_seq,
            alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,
            k_scale_ptr, v_scale_ptr, tp_rank, blocksparse_local_blocks,
            blocksparse_vert_stride, blocksparse_block_size,
            blocksparse_head_sliding_step);

    dim3 reduce_grid(num_heads, num_seqs);
    int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);
    vllm::paged_attention_v2_reduce_kernel<
        uint16_t, HEAD_SIZE, NUM_THREADS, PARTITION_SIZE>
        <<<reduce_grid, block, reduce_shared_mem_size, stream>>>(
            reinterpret_cast<uint16_t*>(out), exp_sums, max_logits,
            reinterpret_cast<const uint16_t*>(tmp_out), seq_lens,
            max_num_partitions);
}
