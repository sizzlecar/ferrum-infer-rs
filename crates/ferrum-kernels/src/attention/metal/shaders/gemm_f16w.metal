#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Mixed-precision GEMM: C[M,N] f32 = A[M,K] f32 @ B[N,K]^T f16
//
// Same 64x32 tile structure as gemm_f32_v2, but B is loaded as half and
// upcast to float when staged into threadgroup memory. A stays float, the
// simdgroup matrix multiply runs in f32 (higher precision than an all-f16
// MAC, which would also require a separate half simdgroup path).
//
// Use when B holds big weight matrices stored as fp16 — halves the weight
// memory footprint while keeping accumulation precision.

struct GemmF16WParams {
    int M;
    int N;
    int K;
};

kernel void gemm_f32a_f16w_v2(
    device const float* A        [[buffer(0)]],
    device const half*  B        [[buffer(1)]],
    device       float* C        [[buffer(2)]],
    constant GemmF16WParams& p   [[buffer(3)]],
    threadgroup float* shmem     [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr short NR0 = 64;  // M tile
    constexpr short NR1 = 32;  // N tile
    constexpr short NK  = 32;  // K tile

    threadgroup float* sa = shmem;
    threadgroup float* sb = shmem + NR0 * NK;

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;
    const short sg_row = sgitg * 16;

    simdgroup_float8x8 acc[8];
    for (short i = 0; i < 8; i++) {
        acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (int kk = 0; kk < p.K; kk += NK) {
        // Load A tile [NR0, NK] from f32
        for (short i = tiitg; i < NR0 * NK; i += 128) {
            short row = i / NK;
            short col = i % NK;
            sa[row * NK + col] = (r0 + row < p.M && kk + col < p.K)
                ? A[(r0 + row) * p.K + kk + col] : 0.0f;
        }

        // Load B^T tile [NK, NR1] from f16 and upcast
        for (short i = tiitg; i < NK * NR1; i += 128) {
            short k_idx = i / NR1;
            short n_idx = i % NR1;
            sb[k_idx * NR1 + n_idx] = (r1 + n_idx < p.N && kk + k_idx < p.K)
                ? (float)B[(r1 + n_idx) * p.K + kk + k_idx] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short k = 0; k < NK; k += 8) {
            simdgroup_float8x8 ma0, ma1;
            simdgroup_load(ma0, sa + (sg_row + 0) * NK + k, NK);
            simdgroup_load(ma1, sa + (sg_row + 8) * NK + k, NK);

            simdgroup_float8x8 mb0, mb1, mb2, mb3;
            simdgroup_load(mb0, sb + k * NR1 + 0,  NR1);
            simdgroup_load(mb1, sb + k * NR1 + 8,  NR1);
            simdgroup_load(mb2, sb + k * NR1 + 16, NR1);
            simdgroup_load(mb3, sb + k * NR1 + 24, NR1);

            simdgroup_multiply_accumulate(acc[0], ma0, mb0, acc[0]);
            simdgroup_multiply_accumulate(acc[1], ma0, mb1, acc[1]);
            simdgroup_multiply_accumulate(acc[2], ma0, mb2, acc[2]);
            simdgroup_multiply_accumulate(acc[3], ma0, mb3, acc[3]);
            simdgroup_multiply_accumulate(acc[4], ma1, mb0, acc[4]);
            simdgroup_multiply_accumulate(acc[5], ma1, mb1, acc[5]);
            simdgroup_multiply_accumulate(acc[6], ma1, mb2, acc[6]);
            simdgroup_multiply_accumulate(acc[7], ma1, mb3, acc[7]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (short rb = 0; rb < 2; rb++) {
        for (short cb = 0; cb < 4; cb++) {
            int gr = r0 + sg_row + rb * 8;
            int gc = r1 + cb * 8;
            if (gr < p.M && gc < p.N) {
                if (gr + 8 <= p.M && gc + 8 <= p.N) {
                    simdgroup_store(acc[rb * 4 + cb], C + gr * p.N + gc, (ulong)p.N);
                } else {
                    threadgroup float* stage = sa + sgitg * 64;
                    simdgroup_store(acc[rb * 4 + cb], stage, 8);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    ushort lane = tiitg % 32;
                    for (short i = lane; i < 64; i += 32) {
                        short r = i / 8, c = i % 8;
                        if (gr + r < p.M && gc + c < p.N) {
                            C[(gr + r) * p.N + gc + c] = stage[i];
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
    }
}

// GEMV: C[N] f32 = A[K] f32 @ B[N, K]^T f16
// One threadgroup per output column. K reduction via simd_sum.
// Used when M=1 (decode) with f16-weight B.
kernel void gemv_f32a_f16w(
    device const float* A        [[buffer(0)]],
    device const half*  B        [[buffer(1)]],
    device       float* C        [[buffer(2)]],
    constant GemmF16WParams& p   [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]])
{
    const int col = tgpig.x;
    if (col >= p.N) return;

    float acc = 0.0f;
    // Each thread strides through K, summing A[k] * B[col, k]
    for (int k = tiitg; k < p.K; k += 32) {
        acc += A[k] * (float)B[col * p.K + k];
    }
    acc = simd_sum(acc);
    if (tiitg == 0) {
        C[col] = acc;
    }
}
