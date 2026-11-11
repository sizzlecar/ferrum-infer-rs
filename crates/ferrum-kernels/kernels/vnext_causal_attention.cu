// vNext causal attention over independently allocated fixed-size pages. The
// provider owns the address table and the core owns allocation, admission, and
// lifetime. The token-major kernels remain the generic fallback. The addressed
// kernels use vLLM's 16-token K/V block layout so decode can dispatch the
// existing tiled vLLM attention without changing core resource ownership.

#include "common.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

#define VNEXT_WARP_SIZE 32
#define VNEXT_MAX_HEAD_CHUNKS 16
#define VNEXT_VLLM_BLOCK_TOKENS 16
#define VNEXT_VLLM_K_PACK 8
#define VNEXT_GROUPED_DEFAULT_KV_TILE_TOKENS 16
#define VNEXT_GROUPED_GEMMA_LOCAL_KV_TILE_TOKENS 32

namespace ferrum_causal_f16 {
using CausalScalar = __half;
__device__ __forceinline__ float causal_float(CausalScalar value) { return __half2float(value); }
__device__ __forceinline__ CausalScalar causal_round(float value) { return __float2half_rn(value); }
#include "vnext_causal_attention_impl.cuh"
}

namespace ferrum_causal_bf16 {
#define grouped_kv_tile grouped_kv_tile_bf16
using CausalScalar = __nv_bfloat16;
__device__ __forceinline__ float causal_float(CausalScalar value) { return __bfloat162float(value); }
__device__ __forceinline__ CausalScalar causal_round(float value) { return __float2bfloat16_rn(value); }
#define vnext_causal_gather_decode_lengths vnext_causal_gather_decode_lengths_bf16
#define vnext_causal_gather_decode_lengths_indexed vnext_causal_gather_decode_lengths_indexed_bf16
#define vnext_causal_gather_decode_lengths_fixed_capacity vnext_causal_gather_decode_lengths_fixed_capacity_bf16
#define vnext_causal_gather_decode_lengths_unified_v2 vnext_causal_gather_decode_lengths_unified_v2_bf16
#define vnext_causal_prepare_f16 vnext_causal_prepare_bf16
#define vnext_causal_attention_f16 vnext_causal_attention_bf16
#define vnext_causal_prepare_int8_kv vnext_causal_prepare_int8_kv_bf16
#define vnext_causal_attention_int8_kv vnext_causal_attention_int8_kv_bf16
#define vnext_causal_attention_grouped_f16 vnext_causal_attention_grouped_bf16
#include "vnext_causal_attention_impl.cuh"
#undef vnext_causal_gather_decode_lengths
#undef vnext_causal_gather_decode_lengths_indexed
#undef vnext_causal_gather_decode_lengths_fixed_capacity
#undef vnext_causal_gather_decode_lengths_unified_v2
#undef vnext_causal_prepare_f16
#undef vnext_causal_attention_f16
#undef vnext_causal_prepare_int8_kv
#undef vnext_causal_attention_int8_kv
#undef vnext_causal_attention_grouped_f16
#undef grouped_kv_tile
}
