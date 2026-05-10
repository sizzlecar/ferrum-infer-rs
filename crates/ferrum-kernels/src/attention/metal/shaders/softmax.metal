#include <metal_stdlib>
using namespace metal;

// ── Softmax Last Dim ────────────────────────────────────────────────────
// In-place softmax over the last dimension of a 2D [rows, cols] tensor.
// Each threadgroup handles one row. Uses simd_sum for parallel reduction.
//
// Matches candle's softmax_last_dim CPU implementation:
//   1. Find max (simd reduction)
//   2. Subtract max and exp
//   3. Sum exp values (simd reduction)
//   4. Divide by sum

struct SoftmaxParams {
    int rows;
    int cols;
};

kernel void softmax_last_dim_f32(
    device float* data          [[buffer(0)]],   // [rows, cols] in-place
    constant SoftmaxParams& p   [[buffer(1)]],
    uint  tgpig [[threadgroup_position_in_grid]],  // row index
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]],
    uint  ntg   [[threads_per_threadgroup]],
    threadgroup float* shmem [[threadgroup(0)]])
{
    const int row = tgpig;
    if (row >= p.rows) return;

    device float* x = data + row * p.cols;
    const int cols = p.cols;

    // 1. Find max (parallel reduction)
    float local_max = -INFINITY;
    for (int i = tiisg; i < cols; i += 32) {
        local_max = max(local_max, x[i]);
    }
    local_max = simd_max(local_max);

    // Cross-simdgroup reduction if needed
    if (ntg > 32) {
        if (sgitg == 0) shmem[tiisg] = -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) shmem[sgitg] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        local_max = shmem[tiisg < (ntg / 32) ? tiisg : 0];
        local_max = simd_max(local_max);
    }

    // 2. Exp and sum
    float local_sum = 0.0f;
    for (int i = tiisg; i < cols; i += 32) {
        float e = exp(x[i] - local_max);
        x[i] = e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);

    if (ntg > 32) {
        if (sgitg == 0) shmem[tiisg] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) shmem[sgitg] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        local_sum = shmem[tiisg < (ntg / 32) ? tiisg : 0];
        local_sum = simd_sum(local_sum);
    }

    // 3. Normalize
    float inv_sum = 1.0f / local_sum;
    for (int i = tiisg; i < cols; i += 32) {
        x[i] *= inv_sum;
    }
}

// Non-in-place version: reads from input, writes to output
kernel void softmax_last_dim_f32_out(
    device const float* input   [[buffer(0)]],
    device       float* output  [[buffer(1)]],
    constant SoftmaxParams& p   [[buffer(2)]],
    uint  tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]],
    uint  ntg   [[threads_per_threadgroup]],
    threadgroup float* shmem [[threadgroup(0)]])
{
    const int row = tgpig;
    if (row >= p.rows) return;

    device const float* x = input + row * p.cols;
    device float* y = output + row * p.cols;
    const int cols = p.cols;

    // 1. Find max
    float local_max = -INFINITY;
    for (int i = tiisg; i < cols; i += 32) {
        local_max = max(local_max, x[i]);
    }
    local_max = simd_max(local_max);

    if (ntg > 32) {
        if (sgitg == 0) shmem[tiisg] = -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) shmem[sgitg] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        local_max = shmem[tiisg < (ntg / 32) ? tiisg : 0];
        local_max = simd_max(local_max);
    }

    // 2. Exp and sum
    float local_sum = 0.0f;
    for (int i = tiisg; i < cols; i += 32) {
        float e = exp(x[i] - local_max);
        y[i] = e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);

    if (ntg > 32) {
        if (sgitg == 0) shmem[tiisg] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) shmem[sgitg] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        local_sum = shmem[tiisg < (ntg / 32) ? tiisg : 0];
        local_sum = simd_sum(local_sum);
    }

    // 3. Normalize
    float inv_sum = 1.0f / local_sum;
    for (int i = tiisg; i < cols; i += 32) {
        y[i] *= inv_sum;
    }
}
