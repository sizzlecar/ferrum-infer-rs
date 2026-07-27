// Embedding lookup: output[i] = table[index * dim + i]
// Grid: ceil(dim / 256) blocks. Block: 256 threads.
// Supports batch: output[b * dim + i] = table[indices[b] * dim + i]

#include <cstdint>
#include <cuda_fp16.h>

extern "C" __global__ void embedding_lookup_f32(
    const float* __restrict__ table,   // [vocab_size, dim]
    const uint32_t* __restrict__ indices, // [batch]
    float* __restrict__ output,        // [batch, dim]
    int batch,
    int dim
) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch || i >= dim) return;

    uint32_t idx = indices[b];
    output[b * dim + i] = table[idx * dim + i];
}

// f16 variant used by the LLM hot path (Qwen3 / Llama etc run in fp16).
extern "C" __global__ void embedding_lookup_f16(
    const __half* __restrict__ table,     // [vocab_size, dim]
    const uint32_t* __restrict__ indices, // [batch]
    __half* __restrict__ output,          // [batch, dim]
    int batch,
    int dim
) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch || i >= dim) return;

    uint32_t idx = indices[b];
    output[b * dim + i] = table[idx * dim + i];
}

// vNext chunks long prefills at the provider boundary so this hot kernel keeps
// the division-free 2D indexing used by the validated legacy implementation.
// Invalid ids are zero-filled to prevent an out-of-bounds device read; the
// product tokenizer/input boundary remains responsible for rejecting them.
extern "C" __global__ void vnext_embedding_lookup_f16(
    const __half* __restrict__ table,      // [vocab_size, dim]
    const uint32_t* __restrict__ indices, // [tokens]
    __half* __restrict__ output,           // [tokens, dim]
    int batch,
    int dim,
    uint32_t vocab_size
) {
    int token = blockIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= batch || column >= dim) return;
    uint32_t index = indices[token];
    if (index >= vocab_size) {
        output[(unsigned long long)token * dim + column] = __float2half(0.0f);
        return;
    }
    output[(unsigned long long)token * dim + column] =
        table[(unsigned long long)index * dim + column];
}

// Device-state variant: token id read from a single device slot.
// Enables CUDA graph capture — the captured graph's scalar args are frozen,
// so dynamic values (token id that changes every decode step) must live in
// device memory and be updated via memcpy_htod_async before each replay.
// Grid: (ceil(dim/256), 1, 1). Batch hardcoded to 1 (decode path).
extern "C" __global__ void embedding_lookup_f16_dyn(
    const __half* __restrict__ table,        // [vocab, dim]
    const uint32_t* __restrict__ token_ptr,  // [1]
    __half* __restrict__ output,             // [dim]
    int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    uint32_t idx = token_ptr[0];
    output[i] = table[(int)idx * dim + i];
}

// Argmax: find index of maximum value in array.
// Single block, uses warp reduction.
// output[0] = argmax(input[0..n])

extern "C" __global__ void argmax_f32(
    const float* __restrict__ input,
    uint32_t* __restrict__ output,
    int n
) {
    float local_max = -1e30f;
    int local_idx = 0;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = input[i];
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
        if (other > local_max) {
            local_max = other;
            local_idx = other_idx;
        }
    }

    // Block reduction
    __shared__ float smem_val[32];
    __shared__ int smem_idx[32];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) {
        smem_val[warp_id] = local_max;
        smem_idx[warp_id] = local_idx;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float best = smem_val[0];
        int best_idx = smem_idx[0];
        int n_warps = (blockDim.x + 31) / 32;
        for (int w = 1; w < n_warps; w++) {
            if (smem_val[w] > best) {
                best = smem_val[w];
                best_idx = smem_idx[w];
            }
        }
        output[0] = (uint32_t)best_idx;
    }
}
