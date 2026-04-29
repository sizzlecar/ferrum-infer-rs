// Fused Q4_K_M dequant + GEMV — ferrum-native Metal kernel.
//
// Computes C[N] f32 = A[K] f32 @ W[N, K]^T  where W is stored as
// `block_q4_K[N * (K/256)]` (row-major super-blocks, K must be multiple
// of 256). Decodes Q4 weights *inline* inside the GEMV reduction loop;
// no transient fp16 buffer is materialised. Drops ~64 MB of intermediate
// memory traffic per 4K×4K matmul vs. the naive (dequant→write fp16
// transient→read fp16→GEMV) pipeline.
//
// Threadgroup layout:
//   - 1 threadgroup per output column (`tgpig.x` == col)
//   - 32 threads per threadgroup = exactly 1 SIMD group on Apple Silicon
//   - Each thread strides through K with offset `tiitg` and step 32, so
//     within one super-block of 256 weights all 32 threads share the
//     header (d, dmin, scales) and decode 8 distinct weights per
//     super-block (one per 32-weight sub-block).
//
// Block layout (matches GGML / candle's BlockQ4K, see q4_k_dequant.metal):
//   half d;                       // super-block scale-of-scales
//   half dmin;                    // super-block scale-of-mins
//   uchar scales[12];             // 8 sub-blocks × (6-bit scale + 6-bit min)
//   uchar qs[128];                // 256 weights × 4-bit (packed 2 per byte)

#include <metal_stdlib>
using namespace metal;

#define QK_K          256
#define K_SCALE_SIZE  12

struct block_q4_K {
    half  d;
    half  dmin;
    uchar scales[K_SCALE_SIZE];
    uchar qs[QK_K / 2];
};

struct GemvQ4KParams {
    int N;      // out_features
    int K;      // in_features (multiple of 256)
};

// 6-bit scale & 6-bit min unpacker. Same as candle's `get_scale_min_k4`.
static inline void get_scale_min_k4(
    int j,
    device const uchar * q,
    thread uchar       & sc,
    thread uchar       & mn
) {
    if (j < 4) {
        sc = q[j]     & 63;
        mn = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        mn = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

kernel void gemv_f32a_q4kw(
    device const float*       A      [[buffer(0)]],
    device const block_q4_K*  W      [[buffer(1)]],
    device       float*       C      [[buffer(2)]],
    constant GemvQ4KParams&   p      [[buffer(3)]],
    uint3  tgpig                     [[threadgroup_position_in_grid]],
    ushort tiitg                     [[thread_index_in_threadgroup]])
{
    const int col = tgpig.x;
    if (col >= p.N) return;

    const int n_blocks_per_row = p.K / QK_K;
    const device block_q4_K* W_row = W + col * n_blocks_per_row;

    float acc = 0.0f;

    for (int blk = 0; blk < n_blocks_per_row; blk++) {
        const device block_q4_K* B = W_row + blk;
        const float    d    = float(B->d);
        const float    dmin = float(B->dmin);
        device const uchar* qs     = B->qs;
        device const uchar* scales = B->scales;

        // Each thread loads 4 q-bytes covering 8 weights worth (2 per byte
        // × 4 bytes), one byte per pair of sub-blocks. Sub-block layout:
        //   sb 0 (low nibble of qs[0..32))
        //   sb 1 (high nibble of qs[0..32))
        //   sb 2 (low nibble of qs[32..64))
        //   sb 3 (high nibble of qs[32..64))
        //   sb 4 (low nibble of qs[64..96))
        //   sb 5 (high nibble of qs[64..96))
        //   sb 6 (low nibble of qs[96..128))
        //   sb 7 (high nibble of qs[96..128))
        const uchar q0 = qs[0  + tiitg];
        const uchar q1 = qs[32 + tiitg];
        const uchar q2 = qs[64 + tiitg];
        const uchar q3 = qs[96 + tiitg];

        const int a_base = blk * QK_K;

        uchar sc, mn;

        // sb 0: low nibble of q0
        get_scale_min_k4(0, scales, sc, mn);
        {
            float w = (d * float(sc)) * float(q0 & 0xF) - dmin * float(mn);
            acc += A[a_base + 0 * 32 + tiitg] * w;
        }
        // sb 1: high nibble of q0
        get_scale_min_k4(1, scales, sc, mn);
        {
            float w = (d * float(sc)) * float(q0 >> 4) - dmin * float(mn);
            acc += A[a_base + 1 * 32 + tiitg] * w;
        }
        // sb 2: low nibble of q1
        get_scale_min_k4(2, scales, sc, mn);
        {
            float w = (d * float(sc)) * float(q1 & 0xF) - dmin * float(mn);
            acc += A[a_base + 2 * 32 + tiitg] * w;
        }
        // sb 3: high nibble of q1
        get_scale_min_k4(3, scales, sc, mn);
        {
            float w = (d * float(sc)) * float(q1 >> 4) - dmin * float(mn);
            acc += A[a_base + 3 * 32 + tiitg] * w;
        }
        // sb 4: low nibble of q2
        get_scale_min_k4(4, scales, sc, mn);
        {
            float w = (d * float(sc)) * float(q2 & 0xF) - dmin * float(mn);
            acc += A[a_base + 4 * 32 + tiitg] * w;
        }
        // sb 5: high nibble of q2
        get_scale_min_k4(5, scales, sc, mn);
        {
            float w = (d * float(sc)) * float(q2 >> 4) - dmin * float(mn);
            acc += A[a_base + 5 * 32 + tiitg] * w;
        }
        // sb 6: low nibble of q3
        get_scale_min_k4(6, scales, sc, mn);
        {
            float w = (d * float(sc)) * float(q3 & 0xF) - dmin * float(mn);
            acc += A[a_base + 6 * 32 + tiitg] * w;
        }
        // sb 7: high nibble of q3
        get_scale_min_k4(7, scales, sc, mn);
        {
            float w = (d * float(sc)) * float(q3 >> 4) - dmin * float(mn);
            acc += A[a_base + 7 * 32 + tiitg] * w;
        }
    }

    acc = simd_sum(acc);
    if (tiitg == 0) {
        C[col] = acc;
    }
}
