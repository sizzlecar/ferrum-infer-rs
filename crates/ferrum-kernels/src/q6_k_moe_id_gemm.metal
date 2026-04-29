// Q6_K MoE 2-D GEMM with indirect dispatch — adapted from q6_k_gemm.metal
// (which is itself a port of llama.cpp's `kernel_mul_mm_q6_K_f32`).
//
// See `q4_k_moe_id_gemm.metal` for the design rationale (expert grid
// dim, `tpe[im]` early termination, indirect src1 read + dst write
// via the `ids` table). This file is the Q6_K analogue used for the
// MoE `down` projection (Qwen3-30B-A3B's down_exps tensor is Q6_K
// while gate / up are Q4_K).
//
// Note: for `down`, the input is the silu·gate output already laid
// out as `[batch, top_k, ffn]`. ne11 = top_k → each (token, slot)
// pair reads its own row, indexed via the same `ids` array.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define QK_K       256
#define QK_NL_Q6_K 16
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

struct block_q6_K {
    uchar  ql[QK_K / 2];
    uchar  qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    half   d;
};

struct GemmQ6KMoeIdParams {
    int M;
    int K;
    int nb01;
    int nb02;
    int ne11;          // 1 for broadcast, top_k for per-slot
    int top_k;
    int max_per_expert;
    int batch;
};

template <typename type4x4>
static inline void dequantize_q6_K_id(
    device const block_q6_K * xb,
    short il,
    thread type4x4 & reg
) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t   * scales = (device const int8_t *)xb->scales;

    ql = ql + 32 * (il / 8) + 16 * ((il / 2) & 1) + 8 * (il & 1);
    qh = qh + 16 * (il / 8) + 8 * (il & 1);
    float sc = scales[(il % 2) + 2 * (il / 2)];
    il = (il / 2) & 3;

    const uint32_t kmask1 = il > 1
        ? (il > 2 ? 0xC0C0C0C0u : 0x30303030u)
        : (il > 0 ? 0x0C0C0C0Cu : 0x03030303u);
    const uint32_t kmask2 = il > 1 ? 0xF0F0F0F0u : 0x0F0F0F0Fu;
    const float ml  = float(d_all) * sc * 32.f;
    const float dl0 = float(d_all) * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uchar shr_h = il > 2 ? 2 : 0;
    const uchar shl_h = il > 1 ? 0 : (il > 0 ? 2 : 4);
    const uchar shr_l = il > 1 ? 4 : 0;

    FOR_UNROLL (int i = 0; i < 4; ++i) {
        const uint32_t low = ((uint32_t)ql[2 * i] | ((uint32_t)ql[2 * i + 1] << 16)) & kmask2;
        const uint32_t high = ((uint32_t)qh[2 * i] | ((uint32_t)qh[2 * i + 1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = dl0 * float(q & 0x000000FFu) - ml;
        reg[i][1] = dl1 * float(q & 0x0000FF00u) - ml;
        reg[i][2] = dl2 * float(q & 0x00FF0000u) - ml;
        reg[i][3] = dl3 * float(q & 0xFF000000u) - ml;
    }
}

constant short NR0_Q6 = 64;
constant short NR1_Q6 = 32;
constant short NK_Q6  = 32;
constant short NL0_Q6 = NK_Q6 / 16;
constant short NL1_Q6 = NK_Q6 / 8;
constant short NL_BLOCK_Q6 = QK_NL_Q6_K;

kernel void gemm_q6kw_moe_id_f32(
    device const block_q6_K * src0  [[buffer(0)]],   // [num_experts, M, K/256]
    device const float      * src1  [[buffer(1)]],   // [batch, ne11, K]
    device const int        * ids   [[buffer(2)]],
    device const int        * tpe   [[buffer(3)]],
    device       float      * dst   [[buffer(4)]],   // [batch, top_k, M]
    constant GemmQ6KMoeIdParams & p [[buffer(5)]],
    threadgroup char        * shmem [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int im = tgpig.z;
    const int neh1 = tpe[im];
    const int r0 = tgpig.y * NR0_Q6;
    const int r1 = tgpig.x * NR1_Q6;

    if (r1 >= neh1) return;

    const short nr0 = (p.M - r0 < NR0_Q6) ? short(p.M - r0) : NR0_Q6;
    const short nr1 = (neh1 - r1 < NR1_Q6) ? short(neh1 - r1) : NR1_Q6;

    const short lr0 = (short(tiitg) / NL0_Q6) < nr0 ? (short(tiitg) / NL0_Q6) : (nr0 - 1);
    const short lr1 = (short(tiitg) / NL1_Q6) < nr1 ? (short(tiitg) / NL1_Q6) : (nr1 - 1);

    const short il0 = short(tiitg) % NL0_Q6;
    short il = il0;
    const short offset1 = il0 / NL_BLOCK_Q6;

    device const block_q6_K * x = (device const block_q6_K *)(
        (device const char *)src0 + p.nb02 * im + p.nb01 * (r0 + lr0)
    ) + offset1;

    const int  pair_id = ids[im * p.max_per_expert + r1 + lr1];
    const short i12_tile = pair_id / p.top_k;
    const short i11_tile = (pair_id % p.top_k) % p.ne11;
    const short iy = 8 * (short(tiitg) % NL1_Q6);
    device const float * y = src1
        + (i12_tile * p.ne11 + i11_tile) * p.K
        + iy;

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < p.K; loop_k += NK_Q6) {
        {
            half4x4 temp_a;
            dequantize_q6_K_id(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (short(tiitg) / NL0_Q6) / 8;
                const short lx = (short(tiitg) / NL0_Q6) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = short(tiitg) % NL1_Q6;
            const short sy = (short(tiitg) / NL1_Q6) / 8;
            const short ly = (short(tiitg) / NL1_Q6) % 8;
            const short ib = 4 * sx + sy;

            float4 v0 = float4(*((device const float4 *)(y + 0)));
            float4 v1 = float4(*((device const float4 *)(y + 4)));

            threadgroup half * dst8 = sb + 64 * ib + 8 * ly;
            dst8[0] = half(v0[0]);
            dst8[1] = half(v0[1]);
            dst8[2] = half(v0[2]);
            dst8[3] = half(v0[3]);
            dst8[4] = half(v1[0]);
            dst8[5] = half(v1[1]);
            dst8[6] = half(v1[2]);
            dst8[7] = half(v1[3]);
        }

        il = (il + 2 < NL_BLOCK_Q6) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + NL_BLOCK_Q6 - 1) / NL_BLOCK_Q6 : x;
        y += NK_Q6;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK_Q6 / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }

            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = ((threadgroup float *)shmem)
        + 32 * (sgitg & 1)
        + (16 * (sgitg >> 1)) * NR0_Q6;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0_Q6 * (i / 4), NR0_Q6, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int id = ids[im * p.max_per_expert + r1 + j];
        const short ide = id % p.top_k;
        const short idt = id / p.top_k;

        device float * D = dst + (idt * p.top_k + ide) * p.M + r0;
        threadgroup float * C = ((threadgroup float *)shmem) + j * NR0_Q6;

        for (short i = tiisg; i < nr0; i += 32) {
            D[i] = C[i];
        }
    }
}
