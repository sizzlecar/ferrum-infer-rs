// Q6_K MoE indirect-dispatch GEMV — adapted from q6_k_gemv.metal
// (which is itself a port of llama.cpp's `kernel_mul_mv_q6_K_f32_impl`).
//
// One Metal dispatch handles ALL `n_selected` (token, expert) pairs for
// decode m=1, replacing the per-expert down_proj gemv loop. See the
// matching q4_k_moe_id_gemv.metal for the broader rationale.
//
// Inputs:
//   src0 : [num_experts, N, K/256] Q6_K block bytes, contiguous, with
//          stride `nb02` (= N * K/256 * 210 bytes) between experts.
//   src1 : [K] activations.
//   ids  : [n_selected] selected expert IDs (i32).
//   dst  : [n_selected, N] output rows.
//
// Grid: (ceil(N/4), 1, n_selected). Threadgroup: (32, 2, 1).

#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define N_R0 2
#define N_SG 2
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

struct block_q6_K {
    uchar  ql[QK_K / 2];
    uchar  qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    half   d;
};

struct GemvQ6KMoeParams {
    int N;            // out_features per expert
    int K;            // in_features (multiple of 256)
    int nb01;         // src0 row stride per expert in BYTES = (K/256) * 210
    int nb02;         // src0 expert stride in BYTES = N * (K/256) * 210
    int n_selected;
    int src1_stride;  // 0 for broadcast (gate/up), K for per-slot (down)
};

kernel void gemv_q6kw_moe_id_f32(
    device const block_q6_K * src0  [[buffer(0)]],
    device const float      * src1  [[buffer(1)]],
    device const int        * ids   [[buffer(2)]],   // [n_selected]
    device       float      * dst   [[buffer(3)]],   // [n_selected, N]
    constant GemvQ6KMoeParams & p   [[buffer(4)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int slot = tgpig.z;
    if (slot >= p.n_selected) return;

    const int expert_id = ids[slot];

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = p.K / QK_K;
    const int r0 = tgpig.x;
    const int first_row = (r0 * N_SG + sgitg) * N_R0;
    if (first_row >= p.N) return;

    // src0 base: pick the expert slab, then the simdgroup's first row.
    device const block_q6_K * x = (device const block_q6_K *)(
        (device const char *)src0 + expert_id * p.nb02 + first_row * p.nb01
    );
    // Per-slot activation base. See q4_k_moe_id_gemv.metal for the
    // semantics of `src1_stride` (0 = broadcast, K = per-slot rows).
    device const float * yy = src1 + slot * p.src1_stride;

    float sumf[N_R0] = { 0.f };
    float yl[16];

    const short tid = tiisg / 2;
    const short ix  = tiisg % 2;
    const short ip  = tid / 8;
    const short il  = tid % 8;
    const short l0  = 4 * il;
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

            q1 += p.nb01;
            q2 += p.nb01;
            qh += p.nb01;
            sc += p.nb01;
            dh += p.nb01 / 2;
        }
    }

    device float * dst_slot = dst + slot * p.N;
    for (int row = 0; row < N_R0 && (first_row + row) < p.N; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_slot[first_row + row] = sum_all;
        }
    }
}
