// Q4_K_M GEMM (m>1 prefill path) — adapted from llama.cpp's
// `kernel_mul_mm` template instantiated as `kernel_mul_mm_q4_K_f32`
// (ggml/src/ggml-metal/ggml-metal.metal, MIT licensed). Uses
// `simdgroup_half8x8` matrix multiply for the inner GEMM and inlines
// Q4_K dequantization into the threadgroup-memory load — no fp16
// transient buffer materialised.
//
// Tile layout (per threadgroup):
//   - 4 simdgroups × 32 threads = 128 threads
//   - Output tile: 64 weight rows × 32 activation rows = 2048 fp32 outputs
//   - K processed in chunks of NK = 32
//   - shmem: 4096 bytes for half-A + 4096 bytes for half-B = 8192 bytes
//
// Grid: `(N/NR0, M/NR1, batch)` threadgroups where N = weight rows
// (out_features), M = activation rows (m / batch tokens), K = in_features.
//
// All notation matches llama.cpp's kernel for direct line-by-line
// comparison: src0 = weights (block_q4_K), src1 = activations (float),
// dst = output (float).

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define QK_K       256
#define QK_NL_Q4_K 16   // 16 dequant tiles per super-block (16 weights/tile)

struct block_q4_K {
    half  d;
    half  dmin;
    uchar scales[12];
    uchar qs[QK_K / 2];
};

struct GemmQ4KParams {
    int M;       // weight rows (out_features)
    int N;       // activation rows (m / batch)
    int K;       // in_features (multiple of 256)
    int nb01;    // src0 row stride in BYTES = (K/256) * 144
    int strideC; // dst row stride (in elements) — usually M for col-major output
};

// Dequant one 16-element tile (`il` ∈ [0,16)) of a Q4_K super-block.
// Mirrors llama.cpp's `dequantize_q4_K` verbatim.
static inline uchar2 get_scale_min_k4_just2(
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
static inline void dequantize_q4_K(
    device const block_q4_K * xb,
    short il,
    thread type4x4 & reg
) {
    device const uchar * q = xb->qs;

    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il / 2, xb->scales);
    const float d   = il < 2 ? float(xb->d) : float(xb->d) / 16.f;
    const float minv = float(xb->dmin);
    const float dl = d * float(sc[0]);
    const float ml = minv * float(sc[1]);

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = dl * float(q[i] & mask) - ml;
    }
}

// Tile constants — match llama.cpp's legacy mul_mm path
constant short NR0 = 64;     // weight rows per threadgroup
constant short NR1 = 32;     // activation rows per threadgroup
constant short NK  = 32;     // K-chunk per outer loop iteration
constant short NL0 = NK / 16;  // = 2: 16-element dequant tiles per K-chunk
constant short NL1 = NK / 8;   // = 4: 8-element activation loads per K-chunk
constant short NL_BLOCK_Q4_K = QK_NL_Q4_K; // = 16 (alias for clarity)

kernel void gemm_q4kw_f32a_f32o(
    device const block_q4_K * src0  [[buffer(0)]],   // weights [M, K/256] super-blocks
    device const float      * src1  [[buffer(1)]],   // activations [N, K]
    device       float      * dst   [[buffer(2)]],   // output [N, M] — col-major
    constant GemmQ4KParams  & p     [[buffer(3)]],
    threadgroup char        * shmem [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int r0 = tgpig.y * NR0;  // weight row tile start
    const int r1 = tgpig.x * NR1;  // activation row tile start

    const short nr0 = (p.M - r0 < NR0) ? short(p.M - r0) : NR0;
    const short nr1 = (p.N - r1 < NR1) ? short(p.N - r1) : NR1;

    const short lr0 = (short(tiitg) / NL0) < nr0 ? (short(tiitg) / NL0) : (nr0 - 1);
    const short lr1 = (short(tiitg) / NL1) < nr1 ? (short(tiitg) / NL1) : (nr1 - 1);

    const short il0 = short(tiitg) % NL0;
    short il = il0;
    const short offset1 = il0 / NL_BLOCK_Q4_K;

    // Weight pointer for this thread's loaded row, stepping in QK_NL_Q4_K
    // tiles. nb01 is in bytes; super-block size is sizeof(block_q4_K) = 144.
    device const block_q4_K * x = (device const block_q4_K *)(
        (device const char *)src0 + p.nb01 * (r0 + lr0)
    ) + offset1;

    // Activation pointer for this thread's loaded row
    const short iy = 8 * (short(tiitg) % NL1);
    device const float * y = src1 + (r1 + lr1) * p.K + iy;

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < p.K; loop_k += NK) {
        // === Load A (weights) tile into shared memory with inline dequant ===
        {
            half4x4 temp_a;
            dequantize_q4_K(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (short(tiitg) / NL0) / 8;
                const short lx = (short(tiitg) / NL0) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        // === Load B (activations) tile into shared memory ===
        {
            const short sx = short(tiitg) % NL1;
            const short sy = (short(tiitg) / NL1) / 8;
            const short ly = (short(tiitg) / NL1) % 8;
            const short ib = 4 * sx + sy;

            // 8 floats from device memory → half2x4 in shared
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

        // Advance weight pointer for next K tile
        il = (il + 2 < NL_BLOCK_Q4_K) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + NL_BLOCK_Q4_K - 1) / NL_BLOCK_Q4_K : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Inner matmul: 4 simdgroups produce 8 8x8 output tiles each ===
        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);

        for (short ik = 0; ik < NK / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);

            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);

            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);

            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }

            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    // === Store the 64x32 output tile to device memory (col-major dst) ===
    if (r0 + NR0 <= p.M && r1 + NR1 <= p.N) {
        // Fast path: tile fully in bounds, direct simdgroup store.
        device float * C = dst
            + (r0 + 32 * (sgitg & 1))
            + (r1 + 16 * (sgitg >> 1)) * p.M;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * p.M * (i / 4), p.M, 0, false);
        }
    } else {
        // Tile straddles M/N edge — stage to threadgroup memory then
        // sgitg=0 writes the valid sub-rectangle.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *)shmem)
            + 32 * (sgitg & 1)
            + (16 * (sgitg >> 1)) * NR0;

        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float * D = dst + r0 + (r1 + j) * p.M;
                threadgroup float * C = ((threadgroup float *)shmem) + j * NR0;
                for (int i = 0; i < nr0; ++i) {
                    D[i] = C[i];
                }
            }
        }
    }
}
