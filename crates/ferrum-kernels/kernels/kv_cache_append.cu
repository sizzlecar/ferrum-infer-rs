// Append new K (or V) into a pre-allocated head-major cache slot.
//
// Cache:   [nkv, capacity, hd]           (head-major, pre-allocated)
// New:     [nkv, new_tokens, hd]         (produced by qk_norm_rope)
// Writes cache[h, cache_len..cache_len+new_tokens, :] <- new[h, :, :]
//
// Launch: grid = ((total+255)/256, 1, 1), block = (256, 1, 1).
// Same kernel handles K and V — caller issues two launches.

#include <cuda_fp16.h>

extern "C" __global__ void kv_cache_append_head_major_f16(
    __half* __restrict__ cache,           // [nkv, capacity, hd]
    const __half* __restrict__ new_data,  // [nkv, new_tokens, hd]
    const int nkv,
    const int hd,
    const int cache_len,
    const int new_tokens,
    const int capacity
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nkv * new_tokens * hd;
    if (tid >= total) return;

    const int d   = tid % hd;
    const int tok = (tid / hd) % new_tokens;
    const int h   = tid / (new_tokens * hd);

    // Destination slot: [h, cache_len + tok, d]
    cache[h * capacity * hd + (cache_len + tok) * hd + d]
        = new_data[h * new_tokens * hd + tok * hd + d];
}
