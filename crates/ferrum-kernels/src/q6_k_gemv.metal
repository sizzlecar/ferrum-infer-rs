// Q6_K GEMV — adapted from llama.cpp's kernel_mul_mv_q6_K_f32_impl
// (ggml/src/ggml-metal/ggml-metal.metal, MIT licensed).
//
// Q6_K block layout (256 weights per super-block, 210 bytes = 6.5 bits/w):
//   uint8_t  ql[128];        // lower 4 bits of each weight
//   uint8_t  qh[64];          // upper 2 bits (packed 4 weights per byte)
//   int8_t   scales[16];      // 16 sub-block scales (one per 16 weights)
//   half     d;               // super-block scale
//
// Each weight reconstructs as `int8(low4 | high2_shifted) - 32`, scaled by
// `d * scales[sub]`.
//
// Threadgroup: (32, N_SG, 1) — same N_R0=2, N_SG=2 layout as q4_k_gemv_v2.

#include <metal_stdlib>
using namespace metal;

#define QK_K  256
#define N_R0  2
#define N_SG  2
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

struct block_q6_K {
    uchar  ql[QK_K / 2];     // 128 bytes
    uchar  qh[QK_K / 4];     // 64 bytes
    int8_t scales[QK_K / 16]; // 16 int8 sub-block scales
    half   d;                 // super-block scale
};

struct GemvQ6KParams {
    int N;        // out_features
    int K;        // in_features (multiple of 256)
    int nb01;     // src0 row stride in BYTES = (K/256)*210
};

kernel void gemv_f32a_q6kw_v2(
    device const block_q6_K * src0 [[buffer(0)]],   // [N, K/256] super-blocks
    device const float      * src1 [[buffer(1)]],   // [K] activations
    device       float      * dst  [[buffer(2)]],   // [N] output
    constant GemvQ6KParams  & p    [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = p.K / QK_K;
    const int r0 = tgpig.x;
    const int first_row = (r0 * N_SG + sgitg) * N_R0;
    if (first_row >= p.N) return;

    device const block_q6_K * x = src0 + first_row * nb;
    device const float      * yy = src1;

    float sumf[N_R0] = { 0.f };
    float yl[16];

    const short tid = tiisg / 2;     // 0..15
    const short ix  = tiisg % 2;     // 0 or 1
    const short ip  = tid / 8;       // 0 or 1
    const short il  = tid % 8;       // 0..7
    const short l0  = 4 * il;        // 0, 4, 8, ...
    const short is  = 8 * ip + l0 / 16;

    const short y_offset   = 128 * ip + l0;
    const short q_offset_l =  64 * ip + l0;
    const short q_offset_h =  32 * ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uchar  * q1 = x[i].ql + q_offset_l;
        device const uchar  * q2 = q1 + 32;
        device const uchar  * qh = x[i].qh + q_offset_h;
        device const int8_t * sc = x[i].scales + is;
        device const half   * dh = &x[i].d;

        device const float * y = yy + i * QK_K + y_offset;

        FOR_UNROLL (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        for (short row = 0; row < N_R0; ++row) {
            float4 sums = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0x0F) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0x0F) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += dh[0] * (
                sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]
            );

            // Advance pointers by ONE row stride. nb01 is the byte stride of
            // src0 between consecutive output rows. q1, q2, qh are uchar*; sc
            // is int8*; dh is half*. nb01 is in bytes for all of these.
            q1 += p.nb01;
            q2 += p.nb01;
            qh += p.nb01;
            sc += p.nb01;
            dh += p.nb01 / 2;
        }
    }

    for (int row = 0; row < N_R0 && (first_row + row) < p.N; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[first_row + row] = sum_all;
        }
    }
}
