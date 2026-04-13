#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ── RMS Norm ────────────────────────────────────────────────────────────
// out[i] = (x[i] / rms) * weight[i], where rms = sqrt(mean(x^2) + eps)
// One threadgroup per row. Uses simd_sum for parallel reduction.

struct RmsNormParams {
    int dim;      // hidden dimension
    float eps;
};

kernel void rms_norm_f32(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant RmsNormParams& p   [[buffer(3)]],
    uint  tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]],
    uint  ntg   [[threads_per_threadgroup]],
    threadgroup float* shmem [[threadgroup(0)]])
{
    const int row = tgpig;
    device const float* x = input + row * p.dim;
    device float* y = output + row * p.dim;

    // Parallel sum of squares
    float sum_sq = 0.0f;
    for (int i = tiisg; i < p.dim; i += 32) {
        float v = x[i];
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);

    // Cross-simdgroup reduction if ntg > 32
    if (ntg > 32) {
        if (sgitg == 0) shmem[tiisg] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) shmem[sgitg] = sum_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum_sq = shmem[tiisg];
        sum_sq = simd_sum(sum_sq);
    }

    const float scale = 1.0f / sqrt(sum_sq / float(p.dim) + p.eps);

    // Apply norm and weight
    for (int i = tiisg; i < p.dim; i += 32) {
        y[i] = x[i] * scale * weight[i];
    }
}

// ── SiLU × Gate (SwiGLU) ───────────────────────────────────────────────
// out[i] = silu(gate[i]) * up[i]
// Simple element-wise, one thread per element.

struct SiluMulParams {
    int n; // total elements
};

kernel void silu_mul_f32(
    device const float* gate    [[buffer(0)]],
    device const float* up      [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant SiluMulParams& p   [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= uint(p.n)) return;
    float g = gate[tid];
    float silu = g / (1.0f + exp(-g));
    output[tid] = silu * up[tid];
}

// ── Residual Add ────────────────────────────────────────────────────────
// out[i] = a[i] + b[i]

struct AddParams {
    int n;
};

kernel void add_f32(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant AddParams& p       [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= uint(p.n)) return;
    output[tid] = a[tid] + b[tid];
}

// ── Element-wise Multiply (broadcast scale) ─────────────────────────────
// out[i] = a[i] * scale[i % scale_len]
// Used for layer_scale: out = attn_out * scale_vector

struct MulScaleParams {
    int n;          // total elements in a
    int scale_len;  // length of scale vector (broadcasted)
};

kernel void mul_scale_f32(
    device const float* a       [[buffer(0)]],
    device const float* scale   [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant MulScaleParams& p  [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= uint(p.n)) return;
    output[tid] = a[tid] * scale[tid % p.scale_len];
}

// ── GEMM: C = A @ B^T ──────────────────────────────────────────────────
// A: [M, K], B: [N, K] (row-major), C: [M, N]
// Uses simdgroup_matrix for 8x8 tiles.
// One threadgroup computes one 8×8 tile of C.

struct GemmParams {
    int M;
    int N;
    int K;
};

kernel void gemm_f32(
    device const float* A       [[buffer(0)]],
    device const float* B       [[buffer(1)]],
    device       float* C       [[buffer(2)]],
    constant GemmParams& p      [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    threadgroup float* shmem_a  [[threadgroup(0)]],
    threadgroup float* shmem_b  [[threadgroup(1)]])
{
    // Each threadgroup handles one 8x8 output tile
    const int tile_row = tgpig.y * 8;
    const int tile_col = tgpig.x * 8;

    if (tile_row >= p.M || tile_col >= p.N) return;

    // For boundary tiles: use shared memory with zero-padding
    const int m_remain = min(8, p.M - tile_row);
    const int n_remain = min(8, p.N - tile_col);

    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (int kk = 0; kk < p.K; kk += 8) {
        const int k_remain = min(8, p.K - kk);

        // Load A tile into shared memory (zero-padded)
        for (int i = tiisg; i < 64; i += 32) {
            int r = i / 8, c = i % 8;
            shmem_a[i] = (tile_row + r < p.M && kk + c < p.K) ? A[(tile_row + r) * p.K + kk + c] : 0.0f;
        }
        // Load B tile (transposed) into shared memory
        for (int i = tiisg; i < 64; i += 32) {
            int r = i / 8, c = i % 8;  // r=K dim, c=N dim
            shmem_b[i] = (tile_col + c < p.N && kk + r < p.K) ? B[(tile_col + c) * p.K + kk + r] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 ma, mb;
        simdgroup_load(ma, shmem_a, 8);
        simdgroup_load(mb, shmem_b, 8);
        simdgroup_multiply_accumulate(acc, ma, mb, acc);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result (only valid elements)
    // Use shared memory to stage the write
    simdgroup_store(acc, shmem_a, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = tiisg; i < 64; i += 32) {
        int r = i / 8, c = i % 8;
        if (tile_row + r < p.M && tile_col + c < p.N) {
            C[(tile_row + r) * p.N + tile_col + c] = shmem_a[i];
        }
    }
}
