// Q4_K_M MoE indirect-dispatch GEMV — batched over M tokens.
//
// One Metal dispatch covers `n_pairs = m * top_k` (token, expert) pairs
// instead of the existing single-token `gemv_q4kw_moe_id_f32`'s
// `n_selected = top_k` pairs. Grid Z dimension extends from `top_k` to
// `m * top_k`; everything else (Q4_K decode, simdgroup reduce) is
// byte-for-byte the same.
//
// The point: ferrum's m=1-only `gemv_q4kw_moe_id_f32` forced the
// engine to loop per-token at the call site (16 invocations / layer
// at c=16). llama.cpp's `kernel_mul_mv_id` already covers all
// (token, expert) pairs in one launch — that's the structural gap
// behind the c=16 30B-A3B 51 vs 95 tok/s difference. This kernel is
// the symmetric move: hold the FFN dispatch count flat as concurrency
// scales.
//
// Inputs:
//   src0 : [num_experts, N, K/256] Q4_K block bytes, expert stride
//          `nb02 = N * (K/256) * 144` bytes.
//   src1 : activation buffer. Indexed per (token_idx, slot_idx) pair as
//             y_offset = token_idx * src1_outer_stride
//                      + slot_idx  * src1_inner_stride
//          gate / up:  src1 = norm_out[m, K]
//                      outer = K, inner = 0  (slots in a token broadcast)
//          down:       src1 = silu_stacked[m, top_k, K]
//                      outer = top_k*K, inner = K  (per-pair row)
//   ids  : [n_pairs] selected expert IDs (i32), flat layout indexed
//          by pair_idx = token_idx * top_k + slot_idx.
//   dst  : [n_pairs, N] output rows.
//
// Grid: (ceil(N/4), 1, n_pairs) threadgroups.
// Threadgroup: (32, 2, 1) threads = 2 simdgroups × 32 threads.

#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define N_R0 2
#define N_SG 2
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

struct block_q4_K {
    half  d;
    half  dmin;
    uchar scales[12];
    uchar qs[QK_K / 2];
};

struct GemvQ4KMoeBatchedParams {
    int N;                   // out_features per expert
    int K;                   // in_features (multiple of 256)
    int nb01;                // src0 row stride per expert in BYTES = (K/256) * 144
    int nb02;                // src0 expert stride in BYTES = N * (K/256) * 144
    int top_k;               // selected experts per token
    int n_pairs;             // total pairs = m * top_k
    int src1_outer_stride;   // floats per token in src1 (gate/up: K, down: top_k*K)
    int src1_inner_stride;   // floats per slot within token (gate/up: 0, down: K)
};

kernel void gemv_q4kw_moe_id_batched_f32(
    device const block_q4_K * src0  [[buffer(0)]],
    device const float      * src1  [[buffer(1)]],
    device const int        * ids   [[buffer(2)]],
    device       float      * dst   [[buffer(3)]],
    constant GemvQ4KMoeBatchedParams & p [[buffer(4)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int pair_idx = tgpig.z;
    if (pair_idx >= p.n_pairs) return;

    const int token_idx = pair_idx / p.top_k;
    const int slot_idx  = pair_idx - token_idx * p.top_k;

    const int expert_id = ids[pair_idx];

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    const int nb = p.K / QK_K;
    const int r0 = tgpig.x;

    const int first_row = (r0 * N_SG + sgitg) * N_R0;
    if (first_row >= p.N) return;

    // src0: this expert's slab + this simdgroup's first row.
    device const block_q4_K * x = (device const block_q4_K *)(
        (device const char *)src0 + expert_id * p.nb02 + first_row * p.nb01
    );
    // src1: token_idx-major × slot_idx-minor offset (in floats).
    device const float * y = src1
        + token_idx * p.src1_outer_stride
        + slot_idx  * p.src1_inner_stride;

    float yl[16];
    float yh[16];
    float sumf[N_R0] = {0.f};

    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < N_R0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += dh[0] * (
                              (acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                              (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                              (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                              (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f
                          )
                        - dh[1] * (
                              sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                              sumy[2] * sc8[6] + sumy[3] * sc8[7]
                          );

            q1 += p.nb01 / 2;
            sc += p.nb01 / 2;
            dh += p.nb01 / 2;
        }

        y4 += 4 * QK_K;
    }

    device float * dst_pair = dst + pair_idx * p.N;
    for (short row = 0; row < N_R0 && (first_row + row) < p.N; ++row) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_pair[first_row + row] = sum_all;
        }
    }
}
