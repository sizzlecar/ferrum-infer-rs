// Q4_K_M MoE 2-D GEMM with indirect dispatch — adapted from q4_k_gemm.metal
// (which is itself a port of llama.cpp's `kernel_mul_mm_q4_K_f32`).
// Adds three things on top of the dense GEMM:
//
// 1. **Expert dimension on the grid (`tgpig.z = im`).** Each expert's
//    weight slab is at offset `im * nb02` bytes into the contiguous
//    stacked-experts buffer.
// 2. **`tpe[im]` early termination.** Each expert may have a different
//    number of (token, slot) pairs assigned. If this threadgroup's
//    column tile is past `tpe[im]`, return immediately.
// 3. **`ids` indirection on src1 read AND dst write.** Each pair index
//    `id ∈ [0, batch * top_k)` is encoded as `id = token_idx * top_k +
//    slot_within_token`. For src1, decode `(i12, i11)` and read the
//    token's activation row (broadcast for gate/up where `ne11=1`,
//    per-slot for down where `ne11=top_k`). For dst, write to the
//    natural `[token, slot, n]` layout — so silu_mul + down + final
//    weighted sum can use the same plain layout without any reshuffle.
//
// Closes the prefill MoE gap to llama.cpp on Qwen3-30B-A3B by replacing
// the per-token gemv loop (302 tokens × per-token launch) with one
// batched mul_mm dispatch covering all (token, expert) pairs in
// parallel — m grows from 1 to ~batch×top_k/num_experts (≈ 19 for
// Qwen3-30B-A3B prefill m=302), enabling the simdgroup_half8x8
// matrix-multiply fast path that's 5–10× faster per FLOP than gemv.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define QK_K       256
#define QK_NL_Q4_K 16
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

struct block_q4_K {
    half  d;
    half  dmin;
    uchar scales[12];
    uchar qs[QK_K / 2];
};

struct GemmQ4KMoeIdParams {
    int M;            // per-expert out_features
    int K;            // in_features (multiple of 256)
    int nb01;         // per-expert row stride in BYTES = (K/256) * 144
    int nb02;         // per-expert slab stride in BYTES = M * (K/256) * 144
    int ne11;         // src1 inner-batch dim: 1 for gate/up (broadcast), top_k for down
    int top_k;        // top-K experts per token
    int max_per_expert; // max ids array stride per expert
    int batch;        // num_tokens (= ne12; for output layout)
};

// Identical scale unpacker as q4_k_gemm.metal.
static inline uchar2 get_scale_min_k4_just2_id(
    int j, int k, device const uchar * q
) {
    if (j < 4) {
        return uchar2(q[j + 0 + k] & 63, q[j + 4 + k] & 63);
    } else {
        return uchar2(
            (q[j + 4 + k] & 0x0F) | ((q[j - 4 + k] & 0xC0) >> 2),
            (q[j + 4 + k] >> 4)   | ((q[j + 0 + k] & 0xC0) >> 2)
        );
    }
}

template <typename type4x4>
static inline void dequantize_q4_K_id(
    device const block_q4_K * xb,
    short il,
    thread type4x4 & reg
) {
    device const uchar * q = xb->qs;
    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2_id(is, il / 2, xb->scales);
    const float d   = il < 2 ? float(xb->d) : float(xb->d) / 16.f;
    const float minv = float(xb->dmin);
    const float dl = d * float(sc[0]);
    const float ml = minv * float(sc[1]);
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    FOR_UNROLL (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = dl * float(q[i] & mask) - ml;
    }
}

constant short NR0 = 64;
constant short NR1 = 32;
constant short NK  = 32;
constant short NL0 = NK / 16;
constant short NL1 = NK / 8;
constant short NL_BLOCK = QK_NL_Q4_K;

kernel void gemm_q4kw_moe_id_f32(
    device const block_q4_K * src0  [[buffer(0)]],   // [num_experts, M, K/256]
    device const float      * src1  [[buffer(1)]],   // [batch, ne11, K]
    device const int        * ids   [[buffer(2)]],   // [num_experts, max_per_expert]
    device const int        * tpe   [[buffer(3)]],   // [num_experts]
    device       float      * dst   [[buffer(4)]],   // [batch, top_k, M]
    constant GemmQ4KMoeIdParams & p [[buffer(5)]],
    threadgroup char        * shmem [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int im   = tgpig.z;                      // expert id
    const int neh1 = tpe[im];                       // pairs assigned to this expert
    const int r0   = tgpig.y * NR0;                 // weight-row tile start
    const int r1   = tgpig.x * NR1;                 // pair-row tile start within expert

    // Early exit: this threadgroup is past `neh1` (the expert has fewer
    // pairs than max_per_expert). Without this, we'd still compute a
    // full output tile and either write garbage or skip writes — the
    // up-front exit saves the GPU-side work entirely.
    if (r1 >= neh1) return;

    const short nr0 = (p.M - r0 < NR0) ? short(p.M - r0) : NR0;
    const short nr1 = (neh1 - r1 < NR1) ? short(neh1 - r1) : NR1;

    const short lr0 = (short(tiitg) / NL0) < nr0 ? (short(tiitg) / NL0) : (nr0 - 1);
    const short lr1 = (short(tiitg) / NL1) < nr1 ? (short(tiitg) / NL1) : (nr1 - 1);

    const short il0 = short(tiitg) % NL0;
    short il = il0;
    const short offset1 = il0 / NL_BLOCK;

    // Weight pointer: pick this expert's slab, then this thread's row.
    device const block_q4_K * x = (device const block_q4_K *)(
        (device const char *)src0 + p.nb02 * im + p.nb01 * (r0 + lr0)
    ) + offset1;

    // Activation pointer: indirect via ids.
    //   id = ids[im * max_per_expert + r1 + lr1]
    //   i12 = id / top_k        (token index)
    //   i11 = id % top_k % ne11 (slot within token; 0 for ne11=1, slot for ne11=top_k)
    // src1 layout: [batch, ne11, K] row-major → index (i12, i11, iy) is
    // src1[(i12 * ne11 + i11) * K + iy].
    const int  pair_id = ids[im * p.max_per_expert + r1 + lr1];
    const short i12_tile = pair_id / p.top_k;
    const short i11_tile = (pair_id % p.top_k) % p.ne11;
    const short iy = 8 * (short(tiitg) % NL1);
    device const float * y = src1
        + (i12_tile * p.ne11 + i11_tile) * p.K
        + iy;

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < p.K; loop_k += NK) {
        // Load A (weights) tile with inline dequant.
        {
            half4x4 temp_a;
            dequantize_q4_K_id(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (short(tiitg) / NL0) / 8;
                const short lx = (short(tiitg) / NL0) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        // Load B (activations) tile.
        // Vector store of 8 halves at once (matches llama.cpp's
        // `*(threadgroup S1_2x4 *) = (S1_2x4)(*((device T1_2x4 *) y))`).
        // Replaces the scalar half writes — same address layout, but
        // emits one threadgroup-store instead of 8, halving the load
        // phase shmem-write cost. (Critical for the inner-K loop where
        // this fires K/NK = 64 times per kernel.)
        {
            const short sx = short(tiitg) % NL1;
            const short sy = (short(tiitg) / NL1) / 8;
            const short ly = (short(tiitg) / NL1) % 8;
            const short ib = 4 * sx + sy;

            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                half2x4(*((device const float2x4 *) y));
        }

        il = (il + 2 < NL_BLOCK) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + NL_BLOCK - 1) / NL_BLOCK : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inner matmul: 4 simdgroups produce 8 8x8 output tiles each.
        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK / 8; ++ik) {
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

    // === Indirect store: write each accumulated row to its (token, slot) ===
    //
    // The mc[i] tiles hold a 64×32 output block in expert-local order.
    // To map back, we go through shmem and, for each row j ∈ [0, nr1),
    // look up `id = ids[im, r1+j]`, decompose to `(token=idt, slot=ide)`,
    // and copy the 64 floats to `dst[idt, ide, r0..r0+nr0)`.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = ((threadgroup float *)shmem)
        + 32 * (sgitg & 1)
        + (16 * (sgitg >> 1)) * NR0;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Distribute the nr1 rows across the 4 simdgroups (sgitg ∈ [0,4)).
    // Each simdgroup handles every 4th row starting at sgitg, all 32
    // threads collaborating on the cross-row copy.
    for (short j = sgitg; j < nr1; j += 4) {
        const int id = ids[im * p.max_per_expert + r1 + j];
        const short ide = id % p.top_k;        // slot within token
        const short idt = id / p.top_k;        // token index

        // dst[idt, ide, :] — layout is [batch, top_k, M] row-major.
        device float * D = dst + (idt * p.top_k + ide) * p.M + r0;
        device float4 * D4 = (device float4 *) D;
        threadgroup float * C = ((threadgroup float *)shmem) + j * NR0;
        threadgroup float4 * C4 = (threadgroup float4 *) C;

        // Vector copy: 32 threads × 4 floats per store = 128 floats per
        // simdgroup pass. Beats 32-thread scalar copy by 4× on the M1
        // memory pipeline (matches llama.cpp's writeback pattern).
        int i = tiisg;
        for (; i < nr0 / 4; i += 32) {
            D4[i] = C4[i];
        }
        // Tail: any rows past `(nr0/4)*4` get scalar copies.
        i = (4 * (nr0 / 4)) + tiisg;
        for (; i < nr0; i += 32) {
            D[i] = C[i];
        }
    }
}
