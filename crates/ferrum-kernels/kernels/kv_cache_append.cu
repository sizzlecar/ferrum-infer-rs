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

// Batched per-cache append for the multi-seq decode path (q_len=1 per item).
// Writes M items' (K or V) head-major slices into M independent caches in
// a single launch.
//
// Per-cache layout: each `caches[i]` is a [nkv, capacity, hd] head-major
// buffer. Item i writes to `caches[i][h, cache_lens[i], d]` for each
// (h, d). new_data is [m, nkv, hd] item-major.
//
// Launch: grid = ((nkv*hd + 255)/256, m, 1), block = (256, 1, 1).
extern "C" __global__ void kv_cache_append_batched_per_cache_f16(
    __half** caches,                            // [m] device pointers to per-cache buffers
    const __half* __restrict__ new_data,        // [m, nkv, hd] item-major
    const int* __restrict__ cache_lens,         // [m] per-item current cache_len
    const int m,
    const int nkv,
    const int hd,
    const int capacity
) {
    const int item = blockIdx.y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int per_item = nkv * hd;
    if (item >= m || tid >= per_item) return;

    const int d = tid % hd;
    const int h = tid / hd;
    const int cache_len = cache_lens[item];
    __half* cache = caches[item];

    cache[h * capacity * hd + cache_len * hd + d]
        = new_data[item * per_item + h * hd + d];
}

// Device-state variant for graph capture. `cache_len` read from device slot.
// `new_tokens` stays scalar (always 1 on the decode path).
extern "C" __global__ void kv_cache_append_head_major_f16_dyn(
    __half* __restrict__ cache,
    const __half* __restrict__ new_data,
    const int nkv,
    const int hd,
    const int* __restrict__ cache_len_ptr,  // device: single int32
    const int new_tokens,
    const int capacity
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nkv * new_tokens * hd;
    if (tid >= total) return;

    const int d   = tid % hd;
    const int tok = (tid / hd) % new_tokens;
    const int h   = tid / (new_tokens * hd);
    const int cache_len = cache_len_ptr[0];

    cache[h * capacity * hd + (cache_len + tok) * hd + d]
        = new_data[h * new_tokens * hd + tok * hd + d];
}
