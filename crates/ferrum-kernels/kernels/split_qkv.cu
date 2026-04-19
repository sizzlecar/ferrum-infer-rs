// Split fused QKV [tokens, q_dim + 2*kv_dim] -> separate Q, K, V buffers.
//
// Input:  qkv[t, :] = [q | k | v]   where q is q_dim, k/v are each kv_dim
// Output: Q[t, :] = qkv[t, 0..q_dim]
//         K[t, :] = qkv[t, q_dim..q_dim+kv_dim]
//         V[t, :] = qkv[t, q_dim+kv_dim..q_dim+2*kv_dim]
//
// Launch: grid = (tokens, 1, 1), block = (256, 1, 1).

#include <cuda_fp16.h>

extern "C" __global__ void split_qkv_f16(
    const __half* __restrict__ qkv,   // [tokens, q_dim + 2*kv_dim]
    __half* __restrict__ q,           // [tokens, q_dim]
    __half* __restrict__ k,           // [tokens, kv_dim]
    __half* __restrict__ v,           // [tokens, kv_dim]
    const int tokens,
    const int q_dim,
    const int kv_dim
) {
    const int t = blockIdx.x;
    if (t >= tokens) return;
    const int tid = threadIdx.x;
    const int stride = q_dim + 2 * kv_dim;
    const __half* row = qkv + t * stride;

    for (int i = tid; i < q_dim; i += blockDim.x) {
        q[t * q_dim + i] = row[i];
    }
    for (int i = tid; i < kv_dim; i += blockDim.x) {
        k[t * kv_dim + i] = row[q_dim + i];
    }
    for (int i = tid; i < kv_dim; i += blockDim.x) {
        v[t * kv_dim + i] = row[q_dim + kv_dim + i];
    }
}
