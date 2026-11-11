// Paged variable-length attention over vLLM paged_attention_v2 KV layout.
//
// This is the q_len>1 companion for Ferrum's vLLM-layout decode path:
// `split_qkv_norm_rope_into_paged_cache_vllm_*` writes K/V as
//   K: [num_blocks, num_kv_heads, head_dim/x, block_size, x]
//   V: [num_blocks, num_kv_heads, head_dim, block_size]
// while the existing `paged_varlen_attention.cu` reads Ferrum's legacy
// [num_blocks, block_size, num_kv_heads, head_dim] pool.
//
// The kernel is intentionally simple and non split-K. It is a correctness
// bridge for prefill/chunk prefill when `FERRUM_USE_VLLM_PAGED_ATTN=1`; the
// hot decode path still uses vLLM's `paged_attention_v2` implementation.

#include "common.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define WARP_SIZE 32
#define VLLM_X 8  // 16 / sizeof(half)
#define VNEXT_VLLM_BLOCK_SIZE 16

namespace ferrum_varlen_f16 {
using CausalScalar = __half;
__device__ __forceinline__ float causal_float(CausalScalar x) { return __half2float(x); }
__device__ __forceinline__ CausalScalar causal_round(float x) { return __float2half_rn(x); }
#include "paged_varlen_attention_vllm_impl.cuh"
}

namespace ferrum_varlen_bf16 {
using CausalScalar = __nv_bfloat16;
__device__ __forceinline__ float causal_float(CausalScalar x) { return __bfloat162float(x); }
__device__ __forceinline__ CausalScalar causal_round(float x) { return __float2bfloat16_rn(x); }
#define paged_varlen_attn_vllm_f16 paged_varlen_attn_vllm_bf16
#define vnext_paged_varlen_attn_vllm_addressed_f16 vnext_paged_varlen_attn_vllm_addressed_bf16
#define paged_varlen_attn_vllm_tiled_q4_f16 paged_varlen_attn_vllm_tiled_q4_bf16
#define vnext_paged_varlen_attn_vllm_tiled_q4_addressed_f16 vnext_paged_varlen_attn_vllm_tiled_q4_addressed_bf16
#include "paged_varlen_attention_vllm_impl.cuh"
#undef paged_varlen_attn_vllm_f16
#undef vnext_paged_varlen_attn_vllm_addressed_f16
#undef paged_varlen_attn_vllm_tiled_q4_f16
#undef vnext_paged_varlen_attn_vllm_tiled_q4_addressed_f16
}
