#include "attention_gqa2_test.cuh"
#include <cuda_runtime.h>

#ifdef FERRUM_TEST_CAUSAL_GQA2
// Independent test ABI. The normal indexed exports below are unchanged.
extern "C" int ferrum_test_vnext_vllm_paged_attention_v2_f16_h256_b16_gqa2_indexed_v1(
    __half* out, float* exp_sums, float* max_logits, __half* tmp_out,
    const __half* query, int num_kv_heads, float scale,
    const uint64_t* block_addresses, const int* seq_lens, const int* row_map,
    int num_seqs, int original_num_seqs, int num_heads,
    int max_num_blocks_per_seq, int q_stride, int kv_block_stride,
    int kv_head_stride, int max_seq_len, cudaStream_t stream) {
    // Narrow pilot domain: preserve the original addressed F16/B16 layout and
    // require both query heads of every pair to belong to the same KV head.
    if (!out || !exp_sums || !max_logits || !tmp_out || !query ||
        !block_addresses || !seq_lens || !row_map || num_seqs <= 0 ||
        num_seqs > original_num_seqs || num_heads <= 0 || num_kv_heads <= 0 ||
        num_heads % num_kv_heads != 0 || (num_heads / num_kv_heads) % 2 != 0 ||
        max_seq_len <= 512 || max_seq_len > 2048 ||
        max_num_blocks_per_seq < (max_seq_len + 15) / 16 ||
        q_stride != num_heads * 256 || kv_head_stride != 256 * 16 ||
        kv_block_stride != num_kv_heads * kv_head_stride) {
        return int(cudaErrorInvalidValue);
    }
    const int partitions = (max_seq_len + 511) / 512;
    vllm::paged_attention_v2_gqa2_f16_h256_b16_indexed_test_kernel
        <<<dim3(num_heads / 2, num_seqs, partitions), 128, 2 * 512 * sizeof(float), stream>>>(
            exp_sums, max_logits, reinterpret_cast<uint16_t*>(tmp_out),
            reinterpret_cast<const uint16_t*>(query), num_kv_heads, scale,
            block_addresses, seq_lens, row_map, original_num_seqs,
            max_num_blocks_per_seq, q_stride, kv_block_stride, kv_head_stride);
    const cudaError_t first = cudaPeekAtLastError();
    if (first != cudaSuccess) return int(first);
    // Exactly the existing per-head V2 reduction and its original scratch layout.
    vllm::paged_attention_v2_reduce_indexed_kernel<uint16_t, 256, 128, 512>
        <<<dim3(num_heads, num_seqs), 128, 2 * partitions * sizeof(float), stream>>>(
            reinterpret_cast<uint16_t*>(out), exp_sums, max_logits,
            reinterpret_cast<const uint16_t*>(tmp_out), seq_lens,
            partitions, row_map, original_num_seqs);
    return int(cudaPeekAtLastError());
}
#endif
