// Transpose [heads, tokens, head_dim] -> [tokens, heads, head_dim].
// Inverse of qk_norm_rope's transpose — runs after flash_attention to
// restore token-major layout for the O-proj GEMM.
//
// Launch: grid = ((total + 255)/256, 1, 1), block = (256, 1, 1).
// One thread per element.

#include <cuda_fp16.h>

extern "C" __global__ void transpose_head_to_token_f16(
    const __half* __restrict__ input,   // [heads, tokens, head_dim]
    __half* __restrict__ output,        // [tokens, heads, head_dim]
    const int tokens,
    const int heads,
    const int head_dim
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * heads * head_dim;
    if (tid >= total) return;

    const int hd = head_dim;

    // Decode flat index (in input layout) -> (head, tok, d)
    const int d    = tid % hd;
    const int tok  = (tid / hd) % tokens;
    const int head = tid / (tokens * hd);

    // output[tok, head, d] = input[head, tok, d]
    output[tok * heads * hd + head * hd + d] = input[head * tokens * hd + tok * hd + d];
}

// Inverse: [tokens, heads, head_dim] -> [heads, tokens, head_dim].
// Used by the CUDA paged_decode_attention wrapper to restore the
// head-major layout Qwen3MoeModel's `attn_head_major_out` expects after
// `paged_varlen_attention` (which is token-major native, designed for
// LlamaFamily's unified_forward).
extern "C" __global__ void transpose_token_to_head_f16(
    const __half* __restrict__ input,   // [tokens, heads, head_dim]
    __half* __restrict__ output,        // [heads, tokens, head_dim]
    const int tokens,
    const int heads,
    const int head_dim
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * heads * head_dim;
    if (tid >= total) return;

    const int hd = head_dim;

    // Decode flat index (in input/token-major layout) -> (tok, head, d)
    const int d    = tid % hd;
    const int head = (tid / hd) % heads;
    const int tok  = tid / (heads * hd);

    // output[head, tok, d] = input[tok, head, d]
    output[head * tokens * hd + tok * hd + d] = input[tok * heads * hd + head * hd + d];
}
