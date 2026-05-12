// Q4_K_M MoE fused gate+up gemv with in-register SiLU·gate.
//
// Folds the three decode-m=1 dispatches that the stacked MoE FFN
// emitted per layer:
//   1) gemv_q4kw_moe_id_f32 (gate)  → gate_out_stacked[top_k, ffn]
//   2) gemv_q4kw_moe_id_f32 (up)    → up_out_stacked[top_k, ffn]
//   3) silu_mul_stacked_f32         → silu_stacked[top_k, ffn]
// into a single dispatch that writes silu_stacked directly. Both gate
// and up always broadcast their src1 across slots in the decode path,
// so this kernel hard-codes the broadcast (no `src1_stride` parameter)
// and reuses the per-thread `yl/yh` activation register file across
// the two weight matrices — saving one full activation read pass plus
// the entire round-trip through gate_out_stacked / up_out_stacked
// scratch memory.
//
// Bandwidth (per layer, top_k=8, ffn=expert_inter):
//   old: read activation 2× + read gate weights 1× + read up weights 1×
//        + write gate_out + write up_out + read gate_out + read up_out
//        + write silu_out
//   new: read activation 1× + read gate weights 1× + read up weights 1×
//        + write silu_out
//   saved = 1× activation + 4× [top_k, ffn] intermediate traffic
//
// Inputs:
//   gate_w : [num_experts, N, K/256] Q4_K block bytes, contiguous,
//            stride `nb02 = N * K/256 * 144` bytes between experts.
//   up_w   : same shape and stride as gate_w (must share N, K, nb02).
//   src1   : [K] activations (single token, broadcast across slots).
//   ids    : [n_selected] selected expert IDs (i32).
//   dst    : [n_selected, N] output rows — silu(gate · x) * (up · x).
//
// Grid: (ceil(N/4), 1, n_selected) threadgroups.
// Threadgroup: (32, 2, 1) threads = 2 simdgroups × 32 threads.
//
// Register pressure delta vs gemv_q4kw_moe_id_f32: +N_R0=2 floats per
// thread for the second accumulator (sumf_u). Inner acc1/acc2 buffers
// remain stack-local within each row loop. M1 Max headroom is ample.

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

struct GemvQ4KMoeFusedParams {
    int N;          // out_features per expert (= ffn = expert_inter)
    int K;          // in_features (multiple of 256, = hidden)
    int nb01;       // src0 row stride per expert in BYTES = (K/256) * 144
    int nb02;       // src0 expert stride in BYTES = N * (K/256) * 144
    int n_selected; // number of selected experts (= top_k)
};

kernel void gemv_q4kw_moe_id_gate_up_silu_f32(
    device const block_q4_K * gate_w [[buffer(0)]],
    device const block_q4_K * up_w   [[buffer(1)]],
    device const float      * src1   [[buffer(2)]],
    device const int        * ids    [[buffer(3)]],
    device       float      * dst    [[buffer(4)]],
    constant GemvQ4KMoeFusedParams & p [[buffer(5)]],
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

    // Two src0 base pointers — same expert slab in both stacks.
    device const block_q4_K * xg = (device const block_q4_K *)(
        (device const char *)gate_w + expert_id * p.nb02 + first_row * p.nb01
    );
    device const block_q4_K * xu = (device const block_q4_K *)(
        (device const char *)up_w   + expert_id * p.nb02 + first_row * p.nb01
    );
    // Activation is broadcast across slots — no per-slot offset.
    device const float * y = src1;

    float yl[16];
    float yh[16];
    float sumf_g[N_R0] = {0.f};
    float sumf_u[N_R0] = {0.f};

    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        // ── Activation read (shared across gate and up) ─────────────
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i +  160]; sumy[3] += yh[i + 8];
        }

        // ── Gate accumulation ───────────────────────────────────────
        {
            device const uint16_t * sc = (device const uint16_t *)xg[ib].scales + iq;
            device const uint16_t * q1 = (device const uint16_t *)xg[ib].qs + 16 * iq + 4 * ir;
            device const half     * dh = &xg[ib].d;

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

                sumf_g[row] += dh[0] * (
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
        }

        // ── Up accumulation (yl/yh and sumy reused from above) ──────
        {
            device const uint16_t * sc = (device const uint16_t *)xu[ib].scales + iq;
            device const uint16_t * q1 = (device const uint16_t *)xu[ib].qs + 16 * iq + 4 * ir;
            device const half     * dh = &xu[ib].d;

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

                sumf_u[row] += dh[0] * (
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
        }

        y4 += 4 * QK_K;
    }

    // ── Reduce + SiLU(g) * u → silu_stacked ─────────────────────────
    device float * dst_slot = dst + slot * p.N;
    for (short row = 0; row < N_R0 && (first_row + row) < p.N; ++row) {
        const float g = simd_sum(sumf_g[row]);
        const float u = simd_sum(sumf_u[row]);
        if (tiisg == 0) {
            // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
            const float silu_g = g / (1.0f + exp(-g));
            dst_slot[first_row + row] = silu_g * u;
        }
    }
}
