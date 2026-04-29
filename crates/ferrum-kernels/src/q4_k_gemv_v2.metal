// Q4_K_M GEMV — adapted from llama.cpp's kernel_mul_mv_q4_K_f32_impl
// (ggml/src/ggml-metal/ggml-metal.metal, MIT licensed).
//
// Two key wins over our v1 kernel:
//
// 1. **A reuse**: each simdgroup processes nr0=2 output rows, loading
//    activations from `A` once into registers and applying them to both
//    rows. Halves activation bandwidth.
//
// 2. **Better occupancy**: 2 simdgroups per threadgroup (64 threads),
//    4 output rows per threadgroup. M1 Max compute units schedule
//    multiple simdgroups per threadgroup more efficiently than 1.
//
// Grid: (ceil(N / 4), 1, 1) threadgroups.
// Threadgroup: (32, 2, 1) threads = 1 simdgroup of 32 × 2 simdgroups.

#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define N_R0 2     // rows per simdgroup
#define N_SG 2     // simdgroups per threadgroup
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

struct block_q4_K {
    half  d;
    half  dmin;
    uchar scales[12];
    uchar qs[QK_K / 2];
};

struct GemvQ4KV2Params {
    int N;        // out_features
    int K;        // in_features (multiple of 256)
    int nb01;     // src0 row stride in BYTES = (K/256)*144
};

kernel void gemv_f32a_q4kw_v2(
    device const block_q4_K * src0 [[buffer(0)]],   // [N, K/256] super-blocks
    device const float      * src1 [[buffer(1)]],   // [K] activations
    device       float      * dst  [[buffer(2)]],   // [N] output
    constant GemvQ4KV2Params & p   [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg/8;  // 0..3 — which super-block in a group of 4
    const short it = tiisg%8;  // 0..7 — thread within an 8-thread sub-group
    const short iq = it/4;     // 0 or 1 — which half (low/high nibble subblocks)
    const short ir = it%4;     // 0..3 — which sub-position

    const int nb = p.K / QK_K;
    const int r0 = tgpig.x;

    // First row this simdgroup handles. Each threadgroup spans
    // N_R0 * N_SG = 4 consecutive rows; this simdgroup gets rows
    // [first_row .. first_row + N_R0).
    const int first_row = (r0 * N_SG + sgitg) * N_R0;
    if (first_row >= p.N) return;

    // src0 base pointer for first_row.
    device const block_q4_K * x = src0 + first_row * nb;
    device const float      * y = src1;

    float yl[16];
    float yh[16];
    float sumf[N_R0] = {0.f};

    // Activation pointer: (ix selects which super-block in the group of 4),
    // 64*iq picks the high or low half of the super-block,
    // 8*ir picks the sub-position.
    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        // Load 32 floats of A into registers + accumulate row sum.
        // Layout: yl covers low-nibble half, yh covers high-nibble half.
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        // For each row in this simdgroup: locate that row's super-block ib,
        // unpack its scales/mins, compute partial dot product, accumulate.
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

            // Combine sub-block contributions weighted by sc8[*] (4 sub-block
            // scales) and the dmin term carried by sumy.
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

            // Advance pointers by one row stride (nb01 / sizeof(uint16_t) == nb01/2).
            q1 += p.nb01 / 2;
            sc += p.nb01 / 2;
            dh += p.nb01 / 2;
        }

        y4 += 4 * QK_K;
    }

    // Reduce across the simdgroup and write nr0 outputs.
    for (short row = 0; row < N_R0 && (first_row + row) < p.N; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[first_row + row] = sum_all;
        }
    }
}
