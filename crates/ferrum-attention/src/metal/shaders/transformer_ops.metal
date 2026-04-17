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

struct MulScaleParams {
    int n;
    int scale_len;
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

// ── Fused Scale-Add: out = a + b * scale ────────────────────────────────
// Single kernel replaces mul_scale + add (saves 1 dispatch + 1 encoder barrier)

kernel void fused_scale_add_f32(
    device const float* a       [[buffer(0)]],   // residual
    device const float* b       [[buffer(1)]],   // attn/mlp output
    device const float* scale   [[buffer(2)]],   // layer_scale vector
    device       float* output  [[buffer(3)]],
    constant MulScaleParams& p  [[buffer(4)]],   // reuse params struct
    uint tid [[thread_position_in_grid]])
{
    if (tid >= uint(p.n)) return;
    output[tid] = a[tid] + b[tid] * scale[tid % p.scale_len];
}

// ── Fused Residual-Add + RMSNorm ────────────────────────────────────────
// out_residual = a + b  (or a + b * scale if scale != NULL)
// out_norm = rms_norm(out_residual) * weight
// Saves 2 dispatches + 1 encoder barrier

struct FusedResNormParams {
    int tokens;
    int dim;
    float eps;
    int has_scale;   // 0 = no scale, 1 = apply scale to b before add
    int scale_len;
};

kernel void fused_residual_norm_f32(
    device const float* a       [[buffer(0)]],   // residual input
    device const float* b       [[buffer(1)]],   // attn/mlp output
    device const float* scale   [[buffer(2)]],   // layer_scale (or dummy if has_scale=0)
    device const float* weight  [[buffer(3)]],   // norm weight
    device       float* out_res [[buffer(4)]],   // residual output (a + b*scale)
    device       float* out_norm[[buffer(5)]],   // normalized output
    constant FusedResNormParams& p [[buffer(6)]],
    uint  tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    threadgroup float* shmem [[threadgroup(0)]])
{
    const int row = tgpig;
    if (row >= p.tokens) return;

    device const float* a_row = a + row * p.dim;
    device const float* b_row = b + row * p.dim;
    device float* res_row = out_res + row * p.dim;
    device float* norm_row = out_norm + row * p.dim;

    // Step 1: Residual add (with optional scale)
    float sum_sq = 0.0f;
    for (int i = tiisg; i < p.dim; i += 32) {
        float bv = b_row[i];
        if (p.has_scale) bv *= scale[i % p.scale_len];
        float r = a_row[i] + bv;
        res_row[i] = r;
        sum_sq += r * r;
    }

    // Step 2: RMSNorm reduction
    sum_sq = simd_sum(sum_sq);
    float inv = 1.0f / sqrt(sum_sq / float(p.dim) + p.eps);

    // Step 3: Apply norm + weight
    for (int i = tiisg; i < p.dim; i += 32) {
        norm_row[i] = res_row[i] * inv * weight[i];
    }
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

// ── Argmax ─────────────────────────────────────────────────────────────
// Find index of maximum value in a float array.
// Input: data[n], Output: result[0] = argmax index (as uint32)
// Uses simd reduction for parallel max-finding.

struct ArgmaxParams {
    int n;
};

kernel void argmax_f32(
    device const float* data   [[buffer(0)]],
    device uint*        result [[buffer(1)]],
    constant ArgmaxParams& p  [[buffer(2)]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]
) {
    // Each thread finds local max
    float local_max = -INFINITY;
    int   local_idx = 0;
    for (int i = tid; i < p.n; i += tg_size) {
        float v = data[i];
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    // Simd reduction to find global max
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = simd_shuffle_down(local_max, offset);
        int   other_idx = simd_shuffle_down(local_idx, offset);
        if (other_max > local_max) {
            local_max = other_max;
            local_idx = other_idx;
        }
    }

    // Threadgroup reduction (first thread of each simdgroup)
    threadgroup float tg_max[32];
    threadgroup int   tg_idx[32];
    int simd_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0) {
        tg_max[simd_id] = local_max;
        tg_idx[simd_id] = local_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by thread 0
    if (tid == 0) {
        float best = tg_max[0];
        int   best_idx = tg_idx[0];
        int n_simd = (tg_size + 31) / 32;
        for (int s = 1; s < n_simd; s++) {
            if (tg_max[s] > best) {
                best = tg_max[s];
                best_idx = tg_idx[s];
            }
        }
        result[0] = (uint)best_idx;
    }
}

// ── Embedding Lookup ───────────────────────────────────────────────────
// output[i] = table[index * dim + i]
// table: [vocab_size, dim], index: scalar, output: [dim]

struct EmbedParams {
    int dim;
};

kernel void embedding_lookup_f32(
    device const float* table  [[buffer(0)]],
    device const uint*  index  [[buffer(1)]],
    device float*       output [[buffer(2)]],
    constant EmbedParams& p    [[buffer(3)]],
    uint tid                   [[thread_position_in_grid]]
) {
    if ((int)tid < p.dim) {
        uint idx = index[0];
        output[tid] = table[idx * p.dim + tid];
    }
}

// ── Split fused QKV ────────────────────────────────────────────────────
// Input:  qkv [tokens, q_dim + 2*kv_dim] (row-major, Q|K|V interleaved per row)
// Output: q [tokens, q_dim], k [tokens, kv_dim], v [tokens, kv_dim]
// One thread per output element (dispatch q_dim + 2*kv_dim × tokens threads, one thread per element).

struct SplitQkvParams {
    int tokens;
    int q_dim;
    int kv_dim;
};

kernel void split_qkv_f32(
    device const float* qkv     [[buffer(0)]],
    device       float* q       [[buffer(1)]],
    device       float* k       [[buffer(2)]],
    device       float* v       [[buffer(3)]],
    constant SplitQkvParams& p  [[buffer(4)]],
    uint tid                    [[thread_position_in_grid]])
{
    const int qd  = p.q_dim;
    const int kd  = p.kv_dim;
    const int row_stride = qd + 2 * kd;
    const int total = p.tokens * row_stride;
    if ((int)tid >= total) return;

    const int t = (int)tid / row_stride;
    const int c = (int)tid % row_stride;
    const float val = qkv[(int)tid];

    if (c < qd) {
        q[t * qd + c] = val;
    } else if (c < qd + kd) {
        k[t * kd + (c - qd)] = val;
    } else {
        v[t * kd + (c - qd - kd)] = val;
    }
}

// ── GEMV: m=1 GEMM specialization ───────────────────────────────────────
// C[1, N] = A[1, K] @ B[N, K]^T
// One threadgroup (1 simdgroup = 32 threads) per output column.
// K-dim reduction via simd_sum.
//
// Note: tried tile_n=4 (A-vector reuse across 4 cols), turned out slower on
// Qwen3-0.6B (30.2 vs 32.5 tok/s). Bottleneck is not A-side DRAM traffic,
// it's B-side reads (vocab×hidden for lm_head dwarfs everything else).

kernel void gemv_f32(
    device const float* A       [[buffer(0)]],   // [1, K]
    device const float* B       [[buffer(1)]],   // [N, K]
    device       float* C       [[buffer(2)]],   // [1, N]
    constant GemmParams& p      [[buffer(3)]],
    uint  tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]])
{
    const int nj = (int)tgpig;
    if (nj >= p.N) return;

    device const float* b_row = B + nj * p.K;
    const int K = p.K;

    float acc = 0.0f;

    // Vectorized float4 path when K is aligned and large enough.
    // Each thread consumes 4 floats per step = 128-bit load, 4 FMAs.
    if ((K & 3) == 0) {
        device const float4* a4 = (device const float4*)A;
        device const float4* b4 = (device const float4*)b_row;
        const int K4 = K >> 2;
        for (int k4 = (int)tiisg; k4 < K4; k4 += 32) {
            float4 a = a4[k4];
            float4 b = b4[k4];
            acc += dot(a, b);
        }
    } else {
        for (int k = (int)tiisg; k < K; k += 32) {
            acc += A[k] * b_row[k];
        }
    }
    acc = simd_sum(acc);

    if (tiisg == 0) {
        C[nj] = acc;
    }
}

// ── Fused SiLU × Up with gate_up split ──────────────────────────────────
// Input:  gate_up [tokens, 2*im]   (gate = first im, up = second im per row)
// Output: out     [tokens, im]     out[t, i] = silu(gate[t, i]) * up[t, i]

struct SiluMulSplitParams {
    int tokens;
    int im;
};

kernel void silu_mul_split_f32(
    device const float* gate_up    [[buffer(0)]],
    device       float* out        [[buffer(1)]],
    constant SiluMulSplitParams& p [[buffer(2)]],
    uint tid                       [[thread_position_in_grid]])
{
    const int total = p.tokens * p.im;
    if ((int)tid >= total) return;
    const int t = (int)tid / p.im;
    const int i = (int)tid % p.im;
    const float g = gate_up[t * 2 * p.im + i];
    const float u = gate_up[t * 2 * p.im + p.im + i];
    const float silu = g / (1.0f + exp(-g));
    out[(int)tid] = silu * u;
}
