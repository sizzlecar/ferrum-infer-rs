// Q4_K_M MoE indirect-dispatch GEMV — adapted from q4_k_gemv_v2.metal
// (which is itself a port of llama.cpp's `kernel_mul_mv_q4_K_f32_impl`).
//
// One Metal dispatch handles ALL `n_selected` (token, expert) pairs for
// decode m=1, replacing the per-expert gemv loop in `moe_forward`. For
// Qwen3-30B-A3B with top_k=8 and 48 layers, this drops the gate-or-up
// dispatch count from 8 per layer to 1 per layer (-87%) — matching
// llama.cpp's `kernel_mul_mm_id` decode path.
//
// Inputs:
//   src0 : [num_experts, N, K/256] Q4_K block bytes, contiguous, with
//          stride `nb02` (= N * K/256 * 144 bytes) between experts.
//   src1 : [K] activations (single token).
//   ids  : [n_selected] selected expert IDs (i32).
//   dst  : [n_selected, N] output rows (one per selected expert).
//
// Grid: (ceil(N/4), 1, n_selected) threadgroups.
// Threadgroup: (32, 2, 1) threads = 2 simdgroups × 32 threads.
//
// The inner reduction is byte-for-byte identical to gemv_f32a_q4kw_v2 —
// the only changes are:
//   1. `tgpig.z` selects which expert slot to compute.
//   2. The src0 base pointer is offset by `ids[tgpig.z] * nb02`.
//   3. The dst base pointer is offset by `tgpig.z * N`.
// The arithmetic per thread is unchanged, so M1 Max occupancy /
// register pressure / instruction mix all carry over from v2 verbatim.

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

struct GemvQ4KMoeParams {
    int N;            // out_features per expert
    int K;            // in_features (multiple of 256)
    int nb01;         // src0 row stride per expert in BYTES = (K/256) * 144
    int nb02;         // src0 expert stride in BYTES = N * (K/256) * 144
    int n_selected;   // number of selected experts (= top_k for decode m=1)
    int src1_stride;  // src1 element stride per slot. 0 for broadcast
                      // (every slot reads the same activation row, e.g.
                      // gate / up), `K` for non-broadcast (each slot
                      // reads its own row, e.g. down where each expert
                      // sees its own silu(gate)·up output).
};

kernel void gemv_q4kw_moe_id_f32(
    device const block_q4_K * src0  [[buffer(0)]],
    device const float      * src1  [[buffer(1)]],
    device const int        * ids   [[buffer(2)]],   // [n_selected]
    device       float      * dst   [[buffer(3)]],   // [n_selected, N]
    constant GemvQ4KMoeParams & p   [[buffer(4)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int slot = tgpig.z;
    if (slot >= p.n_selected) return;

    const int expert_id = ids[slot];

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

    // src0 base for this expert + this simdgroup's first row.
    // Byte offsets: expert_id * nb02 picks the expert slab,
    //               first_row * nb01 advances inside that slab.
    device const block_q4_K * x = (device const block_q4_K *)(
        (device const char *)src0 + expert_id * p.nb02 + first_row * p.nb01
    );
    // Per-slot activation base. `src1_stride == 0` ⇒ all slots read the
    // same row (used for gate / up broadcasts). `src1_stride == K` ⇒
    // each slot reads its own row (used for down — each expert sees a
    // different silu(gate)·up).
    device const float * y = src1 + slot * p.src1_stride;

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

    // Write nr0 outputs into this slot's row band.
    device float * dst_slot = dst + slot * p.N;
    for (short row = 0; row < N_R0 && (first_row + row) < p.N; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_slot[first_row + row] = sum_all;
        }
    }
}
