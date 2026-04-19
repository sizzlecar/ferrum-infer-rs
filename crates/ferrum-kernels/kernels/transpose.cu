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
