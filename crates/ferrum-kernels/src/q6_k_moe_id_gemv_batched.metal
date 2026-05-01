// Q6_K MoE indirect-dispatch GEMV — **batched over M tokens**.
//
// Counterpart to `q4_k_moe_id_gemv_batched.metal` for Q6_K-quantized
// expert weights (used by the down projection in some Qwen3-MoE
// Q4_K_M GGUF mixes — the M variant keeps a few sensitive linears at
// Q6_K to bound quantization error).
//
// Inputs (same shape contract as the Q4_K batched kernel, just with
// Q6_K block layout in src0):
//   src0 : [num_experts, N, K/256] Q6_K block bytes, expert stride
//          `nb02 = N * (K/256) * 210` bytes.
//   src1 : per-pair row selected by
//             y_offset = (pair / top_k) * src1_outer_stride
//                      + (pair % top_k) * src1_inner_stride
//          gate/up : norm_out [m, K]; outer = K, inner = 0.
//          down    : silu_stacked [m, top_k, K]; outer = top_k*K, inner = K.
//   ids  : [n_pairs] selected expert IDs (i32).
//   dst  : [n_pairs, N] output rows.
//
// Grid: (ceil(N/4), 1, n_pairs). Threadgroup: (32, 2, 1).

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

struct GemvQ6KMoeBatchedParams {
    int N;
    int K;
    int nb01;
    int nb02;
    int top_k;
    int n_pairs;
    int src1_outer_stride;
    int src1_inner_stride;
};

kernel void gemv_q6kw_moe_id_batched_f32(
    device const block_q6_K * src0  [[buffer(0)]],
    device const float      * src1  [[buffer(1)]],
    device const int        * ids   [[buffer(2)]],
    device       float      * dst   [[buffer(3)]],
    constant GemvQ6KMoeBatchedParams & p [[buffer(4)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int pair_idx = tgpig.z;
    if (pair_idx >= p.n_pairs) return;

    const int token_idx = pair_idx / p.top_k;
    const int slot_idx  = pair_idx - token_idx * p.top_k;

    const int expert_id = ids[pair_idx];

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = p.K / QK_K;
    const int r0 = tgpig.x;
    const int first_row = (r0 * N_SG + sgitg) * N_R0;
    if (first_row >= p.N) return;

    device const block_q6_K * x = (device const block_q6_K *)(
        (device const char *)src0 + expert_id * p.nb02 + first_row * p.nb01
    );
    device const float * yy = src1
        + token_idx * p.src1_outer_stride
        + slot_idx  * p.src1_inner_stride;

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

    device float * dst_pair = dst + pair_idx * p.N;
    for (int row = 0; row < N_R0 && (first_row + row) < p.N; ++row) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_pair[first_row + row] = sum_all;
        }
    }
}
