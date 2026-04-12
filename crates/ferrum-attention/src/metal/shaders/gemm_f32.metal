#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ── High-performance f32 GEMM: C[M,N] = A[M,K] @ B[N,K]^T ─────────────
// 64x32 output tiles, 4 simdgroups (128 threads).
// Each simdgroup: 16 rows x 32 cols via 2x4 = 8 accumulators of 8x8.
// K-dimension in tiles of 32.
// Shared memory: sa[64][32] for A tile, sb[32][32] for B^T tile.

struct GemmParams {
    int M;
    int N;
    int K;
};

kernel void gemm_f32_v2(
    device const float* A        [[buffer(0)]],
    device const float* B        [[buffer(1)]],
    device       float* C        [[buffer(2)]],
    constant GemmParams& p       [[buffer(3)]],
    threadgroup float* shmem     [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr short NR0 = 64;  // M tile
    constexpr short NR1 = 32;  // N tile
    constexpr short NK  = 32;  // K tile

    // sa: A tile [NR0, NK], sb: B^T tile [NK, NR1] (B stored transposed)
    threadgroup float* sa = shmem;              // 64 * 32 = 2048 floats
    threadgroup float* sb = shmem + NR0 * NK;   // 32 * 32 = 1024 floats

    const int r0 = tgpig.y * NR0;  // M offset
    const int r1 = tgpig.x * NR1;  // N offset

    // 4 simdgroups, each handles 16 rows of the 64-row tile
    const short sg_row = sgitg * 16;

    // 8 accumulators: 2 row-blocks x 4 col-blocks of 8x8
    simdgroup_float8x8 acc[8];
    for (short i = 0; i < 8; i++) {
        acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (int kk = 0; kk < p.K; kk += NK) {
        // ── Load A tile [NR0, NK] ──
        // A[M, K] row-major. Load rows [r0..r0+64], cols [kk..kk+32]
        for (short i = tiitg; i < NR0 * NK; i += 128) {
            short row = i / NK;
            short col = i % NK;
            sa[row * NK + col] = (r0 + row < p.M && kk + col < p.K)
                ? A[(r0 + row) * p.K + kk + col] : 0.0f;
        }

        // ── Load B^T tile [NK, NR1] ──
        // B[N, K] row-major. We want B^T[K, N] = sb[k][n].
        // sb[k * NR1 + n] = B[n, k] = B[(r1 + n) * K + (kk + k)]
        for (short i = tiitg; i < NK * NR1; i += 128) {
            short k_idx = i / NR1;  // row in B^T = K dimension
            short n_idx = i % NR1;  // col in B^T = N dimension
            sb[k_idx * NR1 + n_idx] = (r1 + n_idx < p.N && kk + k_idx < p.K)
                ? B[(r1 + n_idx) * p.K + kk + k_idx] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Multiply: this simdgroup's 16x32 sub-tile ──
        for (short k = 0; k < NK; k += 8) {
            // Load 2 A blocks: [8x8] from sa, rows [sg_row+0..+8] and [sg_row+8..+16]
            simdgroup_float8x8 ma0, ma1;
            simdgroup_load(ma0, sa + (sg_row + 0) * NK + k, NK);  // stride = NK = 32
            simdgroup_load(ma1, sa + (sg_row + 8) * NK + k, NK);

            // Load 4 B^T blocks: [8x8] from sb, cols [0..8], [8..16], [16..24], [24..32]
            // sb layout: [NK, NR1], stride = NR1 = 32
            simdgroup_float8x8 mb0, mb1, mb2, mb3;
            simdgroup_load(mb0, sb + k * NR1 + 0,  NR1);
            simdgroup_load(mb1, sb + k * NR1 + 8,  NR1);
            simdgroup_load(mb2, sb + k * NR1 + 16, NR1);
            simdgroup_load(mb3, sb + k * NR1 + 24, NR1);

            // C += A * B^T:
            // acc[rb*4 + cb] accumulates rows [sg_row+rb*8..+8] x cols [cb*8..+8]
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

    // ── Store results directly to device memory ──
    // Each simdgroup writes its 2x4 = 8 accumulator blocks (8x8 each)
    for (short rb = 0; rb < 2; rb++) {
        for (short cb = 0; cb < 4; cb++) {
            int gr = r0 + sg_row + rb * 8;
            int gc = r1 + cb * 8;
            if (gr < p.M && gc < p.N) {
                // Store 8x8 block, stride = p.N (output row stride)
                // Need bounds check — use shared mem for partial tiles
                if (gr + 8 <= p.M && gc + 8 <= p.N) {
                    // Full 8x8 block — direct store
                    simdgroup_store(acc[rb * 4 + cb], C + gr * p.N + gc, (ulong)p.N);
                } else {
                    // Partial tile — stage through threadgroup memory
                    // Each simdgroup uses its own 64-float section
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
