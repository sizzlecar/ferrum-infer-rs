#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ── Fused Flash Attention for f32 ──────────────────────────────────────
//
// Single kernel: QK^T * scale + causal_mask → online softmax → attn@V
// No intermediate buffers. All accumulation in registers/threadgroup memory.
//
// Layout:
//   Q: [batch, num_heads, q_len, head_dim]  (contiguous)
//   K: [batch, num_kv_heads, kv_len, head_dim]
//   V: [batch, num_kv_heads, kv_len, head_dim]
//   O: [batch, num_heads, q_len, head_dim]
//
// Grid: (q_len, num_heads, batch) — one threadgroup per query position per head
// Each threadgroup processes one row of the attention matrix.

struct FlashAttnParams {
    int batch;
    int num_heads;
    int num_kv_heads;
    int q_len;
    int kv_len;
    int head_dim;
    float scale;
    int causal;       // 0 or 1
    int pos_offset;
    int kv_seq_stride; // seq dimension stride for K/V (= kv_len for contiguous, = max_len for paged cache)
    int sliding_window; // 0 = full causal, >0 = attend only to last `w` KV positions (Mistral v0.1, Gemma)
};

// Block size for KV processing — process this many KV positions per iteration
constant int BLOCK_KV = 32;

kernel void flash_attn_f32(
    device const float* Q       [[buffer(0)]],
    device const float* K       [[buffer(1)]],
    device const float* V       [[buffer(2)]],
    device       float* O       [[buffer(3)]],
    constant FlashAttnParams& p [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],   // (q_pos, head, batch)
    uint  tiisg [[thread_index_in_simdgroup]],       // 0..31
    uint  sgitg [[simdgroup_index_in_threadgroup]])  // simdgroup index
{
    const int qi    = tgpig.x;  // which query position
    const int hi    = tgpig.y;  // which head
    const int bi    = tgpig.z;  // which batch

    const int kv_hi = hi / (p.num_heads / p.num_kv_heads); // GQA: KV head index
    const int d     = p.head_dim;
    const int sk    = p.kv_len;

    // Causal upper bound and optional sliding-window lower bound.
    const int attend_end = p.causal ? min(p.pos_offset + qi + 1, sk) : sk;
    const int attend_start = (p.causal && p.sliding_window > 0)
        ? max(0, attend_end - p.sliding_window)
        : 0;
    const int attend_len = attend_end;

    // Pointers
    device const float* q_row = Q + ((bi * p.num_heads + hi) * p.q_len + qi) * d;
    const int kv_stride = (p.kv_seq_stride > 0) ? p.kv_seq_stride : sk;
    device const float* k_base = K + (bi * p.num_kv_heads + kv_hi) * kv_stride * d;
    device const float* v_base = V + (bi * p.num_kv_heads + kv_hi) * kv_stride * d;
    device       float* o_row = O + ((bi * p.num_heads + hi) * p.q_len + qi) * d;

    // Online softmax state (per thread, then reduced)
    float M = -INFINITY;  // running max
    float S = 0.0f;       // running sum of exp

    // Output accumulator in registers (one float per head_dim element)
    // Each thread handles d/32 elements (32 threads in simdgroup)
    // For head_dim=128: each thread handles 4 elements
    const int elems_per_thread = d / 32;

    // Local output accumulator
    float acc[4] = {0, 0, 0, 0}; // supports up to head_dim=128 (4 per thread)

    // Process KV in blocks, starting at `attend_start` so sliding-window
    // positions that are too old don't contribute.
    for (int kv_start = attend_start; kv_start < attend_len; kv_start += BLOCK_KV) {
        int kv_end = min(kv_start + BLOCK_KV, attend_len);

        for (int ki = kv_start; ki < kv_end; ++ki) {
            device const float* k_row = k_base + ki * d;
            device const float* v_row = v_base + ki * d;

            // Compute dot product Q[qi] · K[ki] using simd reduction.
            // Vectorized: each thread loads 4 floats per iter (float4
            // load = 16-byte coalesced access vs 4× 4-byte scalar loads).
            // For d=128, head_dim divisible by 4: each thread does d/(32*4)
            // = 1 float4 load and 1 metal::dot. For odd head_dims fall
            // back to scalar tail.
            float dot_acc = 0.0f;
            const int d4 = d & ~3; // round down to multiple of 4
            for (int j = tiisg * 4; j < d4; j += 32 * 4) {
                float4 q_v = *((device const float4 *)(q_row + j));
                float4 k_v = *((device const float4 *)(k_row + j));
                dot_acc += metal::dot(q_v, k_v);
            }
            // Tail (head_dim not multiple of 4) — rare, kept for safety.
            for (int j = d4 + tiisg; j < d; j += 32) {
                dot_acc += q_row[j] * k_row[j];
            }
            float dot_v = simd_sum(dot_acc) * p.scale;

            // Online softmax update
            float old_M = M;
            M = max(M, dot_v);
            float exp_diff = exp(old_M - M);
            float exp_val = exp(dot_v - M);

            // Rescale existing accumulator and sum
            S = S * exp_diff + exp_val;

            // Update output: O = O * exp_diff + exp_val * V[ki]
            for (int j = 0; j < elems_per_thread; ++j) {
                int idx = tiisg + j * 32;
                if (idx < d) {
                    acc[j] = acc[j] * exp_diff + exp_val * v_row[idx];
                }
            }
        }
    }

    // Final normalization: O = O / S
    float inv_S = (S > 0.0f) ? (1.0f / S) : 0.0f;
    for (int j = 0; j < elems_per_thread; ++j) {
        int idx = tiisg + j * 32;
        if (idx < d) {
            o_row[idx] = acc[j] * inv_S;
        }
    }
}

// ── Q-tiled flash attention with simdgroup_matmul (head_dim=128, f32) ────
//
// Mirrors llama.cpp's kernel_flash_attn_ext_impl shape:
//   Q_TILE = 8 query rows per threadgroup
//   NSG    = 4 simdgroups per threadgroup (128 threads)
//   NQ     = Q_TILE / NSG = 2 query rows per simdgroup
//   C      = 32 KV columns per inner chunk (4 simdgroups × 8 cols each)
//   DK=DV  = 128 head dimension
//
// Each threadgroup processes one (q_tile, head, batch) and walks the
// full KV range for that head. The 4 simdgroups cooperate:
//   • QK^T   — each simdgroup computes one [8,8] tile via simdgroup_matmul
//   • softmax — each simdgroup handles its NQ rows
//   • P @ V   — 16 output [8,8] tiles split across 4 simdgroups (NO=4 each)
//
// Restrictions (caller picks the legacy kernel when violated):
//   • head_dim == 128
//   • sliding_window == 0
//   • num_heads % num_kv_heads == 0  (any GQA ratio works)
//   • q_len divisible by 8 *or* the trailing tile is padded with zero queries
//
// Total threadgroup memory: 8*128 (sq) + 8*128 (so) + 8*32 (ss) = 2304 f32
// = 9.0 KB — well within Apple7's 32 KB per-threadgroup limit.

constant int Q_TILE_R = 8;
constant int FA_NSG   = 4;
constant int FA_NQ    = 2;        // Q_TILE_R / FA_NSG
constant int FA_C     = 32;
constant int FA_DK    = 128;
constant int FA_DK8   = 16;       // FA_DK / 8
constant int FA_DV    = 128;
constant int FA_DV4   = 32;       // FA_DV / 4
constant int FA_DV8   = 16;       // FA_DV / 8
constant int FA_NO    = 4;        // FA_DV8 / FA_NSG

kernel void flash_attn_q_tiled_f32(
    device const float* Q       [[buffer(0)]],
    device const float* K       [[buffer(1)]],
    device const float* V       [[buffer(2)]],
    device       float* O       [[buffer(3)]],
    constant FlashAttnParams& p [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    const int qtile = int(tgpig.x);
    const int hi    = int(tgpig.y);
    const int bi    = int(tgpig.z);
    const int iq1   = qtile * Q_TILE_R;

    if (iq1 >= p.q_len) return;

    const int kv_hi    = hi / (p.num_heads / p.num_kv_heads);
    const int kv_stride = (p.kv_seq_stride > 0) ? p.kv_seq_stride : p.kv_len;

    device const float* q_base = Q + ((bi * p.num_heads + hi) * p.q_len + iq1) * FA_DK;
    device const float* k_base = K + (bi * p.num_kv_heads + kv_hi) * kv_stride * FA_DK;
    device const float* v_base = V + (bi * p.num_kv_heads + kv_hi) * kv_stride * FA_DV;
    device       float* o_base = O + ((bi * p.num_heads + hi) * p.q_len + iq1) * FA_DV;

    // Threadgroup memory — laid out contiguously
    threadgroup float sq[Q_TILE_R * FA_DK];   // queries
    threadgroup float so[Q_TILE_R * FA_DV];   // running output (post rescale)
    threadgroup float ss[Q_TILE_R * FA_C];    // attention scores / probabilities

    // 1. Load Q tile into shared memory; pad rows beyond q_len with zero.
    for (int jj = 0; jj < FA_NQ; ++jj) {
        const int j = jj * FA_NSG + int(sgitg);
        const int q_pos = iq1 + j;
        if (q_pos < p.q_len) {
            device const float4* q_row4 = (device const float4 *)(q_base + j * FA_DK);
            threadgroup float4* sq4 = (threadgroup float4 *)(sq + j * FA_DK);
            for (int i = int(tiisg); i < FA_DK / 4; i += 32) {
                sq4[i] = q_row4[i];
            }
        } else {
            threadgroup float4* sq4 = (threadgroup float4 *)(sq + j * FA_DK);
            for (int i = int(tiisg); i < FA_DK / 4; i += 32) {
                sq4[i] = float4(0.0f);
            }
        }
    }

    // 2. Zero output accumulator.
    for (int jj = 0; jj < FA_NQ; ++jj) {
        const int j = jj * FA_NSG + int(sgitg);
        threadgroup float4* so4 = (threadgroup float4 *)(so + j * FA_DV);
        for (int i = int(tiisg); i < FA_DV / 4; i += 32) {
            so4[i] = float4(0.0f);
        }
    }

    // Per-simdgroup running max and sum (covers FA_NQ rows).
    float M[FA_NQ];
    float S[FA_NQ];
    for (int jj = 0; jj < FA_NQ; ++jj) {
        M[jj] = -INFINITY;
        S[jj] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Upper bound for the chunk loop. Causal: can stop after the last row's
    // attend_end (= pos_offset + iq1 + Q_TILE_R since rows j increase).
    int attend_end_max = p.kv_len;
    if (p.causal) {
        attend_end_max = min(p.pos_offset + iq1 + Q_TILE_R, p.kv_len);
    }

    // 3. Walk KV in C=32 chunks.
    for (int ic = 0; ic < attend_end_max; ic += FA_C) {
        // ── 3a. QK^T: each simdgroup writes one [8,8] tile to ss. ──
        {
            device const float* pk = k_base + (ic + 8 * int(sgitg)) * FA_DK;
            threadgroup const float* pq = sq;
            threadgroup       float* ps = ss + 8 * int(sgitg);

            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

            for (int i = 0; i < FA_DK8; ++i) {
                simdgroup_float8x8 mk;
                simdgroup_float8x8 mq;
                simdgroup_load(mk, pk + 8 * i, FA_DK, ulong2(0, 0), true);
                simdgroup_load(mq, pq + 8 * i, FA_DK);
                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
            }

            simdgroup_store(mqk, ps, FA_C, ulong2(0, 0), false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── 3b. Online softmax: each simdgroup handles its FA_NQ rows. ──
        for (int jj = 0; jj < FA_NQ; ++jj) {
            const int j = jj * FA_NSG + int(sgitg);
            const int q_pos = iq1 + j;
            const int k_pos = ic + int(tiisg);

            // Load this lane's score (C == warp size, so one element per lane).
            float s = ss[j * FA_C + int(tiisg)];
            s *= p.scale;

            // Mask: out-of-range Q row, out-of-range K column, or causal.
            bool keep = (q_pos < p.q_len) && (k_pos < p.kv_len);
            if (p.causal) {
                const int row_end = min(p.pos_offset + q_pos + 1, p.kv_len);
                keep = keep && (k_pos < row_end);
            }
            if (!keep) {
                s = -INFINITY;
            }

            const float old_M = M[jj];
            const float row_max = simd_max(s);
            const float new_M = max(old_M, row_max);

            // Guard against the "all -INF" case (e.g. early causal rows).
            const float ms = isfinite(old_M) ? exp(old_M - new_M) : 0.0f;
            const float vs = isfinite(s)     ? exp(s - new_M)     : 0.0f;

            S[jj] = ms * S[jj] + simd_sum(vs);

            // Persist post-softmax probability for the P@V stage.
            ss[j * FA_C + int(tiisg)] = vs;

            // Rescale this row's running output by ms (each lane = FA_DV/32 elems).
            threadgroup float4* so4 = (threadgroup float4 *)(so + j * FA_DV);
            for (int i = int(tiisg); i < FA_DV / 4; i += 32) {
                so4[i] = so4[i] * ms;
            }

            M[jj] = new_M;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── 3c. O += P @ V via simdgroup_matmul. ──
        // Each simdgroup owns FA_NO=4 output [8,8] tiles at column offsets
        // 8*sgitg, 8*(sgitg+NSG), 8*(sgitg+2*NSG), 8*(sgitg+3*NSG). Loading
        // the running O tiles into registers, accumulating, then storing
        // back avoids touching the other simdgroups' data.
        {
            simdgroup_float8x8 lo[FA_NO];
            {
                threadgroup const float* pso = so + 8 * int(sgitg);
                for (int ii = 0; ii < FA_NO; ++ii) {
                    simdgroup_load(lo[ii], pso, FA_DV, ulong2(0, 0), false);
                    pso += 8 * FA_NSG;
                }
            }

            device const float* pv = v_base + ic * FA_DV;
            for (int cc = 0; cc < FA_C / 8; ++cc) {
                simdgroup_float8x8 vs;
                simdgroup_load(vs, ss + 8 * cc, FA_C, ulong2(0, 0), false);

                for (int ii = 0; ii < FA_NO; ++ii) {
                    simdgroup_float8x8 mv;
                    simdgroup_load(mv,
                                   pv + 8 * int(sgitg) + 8 * FA_NSG * ii,
                                   FA_DV, ulong2(0, 0), false);
                    simdgroup_multiply_accumulate(lo[ii], vs, mv, lo[ii]);
                }

                pv += 8 * FA_DV;
            }

            {
                threadgroup float* pso = so + 8 * int(sgitg);
                for (int ii = 0; ii < FA_NO; ++ii) {
                    simdgroup_store(lo[ii], pso, FA_DV, ulong2(0, 0), false);
                    pso += 8 * FA_NSG;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 4. Normalize and write O = so / S to global memory.
    for (int jj = 0; jj < FA_NQ; ++jj) {
        const int j = jj * FA_NSG + int(sgitg);
        const int q_pos = iq1 + j;
        if (q_pos >= p.q_len) continue;

        const float inv_S = (S[jj] > 0.0f) ? (1.0f / S[jj]) : 0.0f;
        device float4* o_row4 = (device float4 *)(o_base + j * FA_DV);
        threadgroup const float4* so4 = (threadgroup const float4 *)(so + j * FA_DV);
        for (int i = int(tiisg); i < FA_DV / 4; i += 32) {
            o_row4[i] = so4[i] * inv_S;
        }
    }
}

// ── SDPA vector decode (m=1, head_dim=128) — MLX-style wide threadgroup ──
//
// Ported from MLX's sdpa_vector kernel (the same kernel candle-metal-kernels
// uses; mistral.rs reaches it via candle's `ops::sdpa`). The legacy
// `flash_attn_f32` above uses 32 threads (one simdgroup) per
// (head, query) — for Llama-3.1-8B that's 32 active threads × 32 q-heads
// = 1024 active threads total, ~3% of M1 Max's ~32K-thread concurrent
// capacity. KV positions are walked sequentially within that single
// simdgroup, so most of the GPU sits idle during decode m=1.
//
// This kernel widens the threadgroup to 32 simdgroups × 32 threads =
// 1024 threads, one TG per (head, batch). The 32 simdgroups process
// distinct KV positions in parallel; each thread within a simdgroup
// owns elem_per_thread = head_dim/32 = 4 elements of Q/K/V/O. After
// the KV loop, simdgroups merge their partial (max, sumexp, output)
// via threadgroup memory using the same online-softmax rescaling
// trick the inner loop uses.
//
// Restrictions (caller picks legacy when violated):
//   • q_len == 1                    (decode hot path; prefill stays Q-tiled)
//   • head_dim == 128               (4 elements/thread × 32 threads)
//   • sliding_window == 0           (handled later if needed)
//   • num_heads % num_kv_heads == 0 (standard GQA)
//
// Threadgroup memory:
//   outputs[BN * head_dim] = 32 * 128 * 4 = 16 KB
//   max_scores[BN]         = 128 B
//   sum_exp_scores[BN]     = 128 B
//   total ≈ 16.25 KB — within Apple7's 32 KB cap.

constant int SDPA_BN = 32;        // simdgroups per threadgroup
constant int SDPA_BD = 32;        // simdgroup width (Apple GPU)
constant int SDPA_D  = 128;       // head_dim
constant int SDPA_EPT = SDPA_D / SDPA_BD; // = 4 elements per thread

kernel void flash_attn_decode_f32(
    device const float* Q       [[buffer(0)]],
    device const float* K       [[buffer(1)]],
    device const float* V       [[buffer(2)]],
    device       float* O       [[buffer(3)]],
    constant FlashAttnParams& p [[buffer(4)]],
    uint3  tgpig [[threadgroup_position_in_grid]],   // (1, head, batch)
    uint   sgitg [[simdgroup_index_in_threadgroup]], // 0..31 — which KV-stripe
    uint   tiisg [[thread_index_in_simdgroup]])      // 0..31 — which D slice
{
    const int hi    = int(tgpig.y);
    const int bi    = int(tgpig.z);
    const int kv_hi = hi / (p.num_heads / p.num_kv_heads);
    const int d     = p.head_dim;
    const int sk    = p.kv_len;

    // Causal upper bound for q_len=1: attend to positions [0, pos_offset+1)
    // (the new token can see itself once it's been written into the cache).
    const int attend_end = p.causal ? min(p.pos_offset + 1, sk) : sk;

    // Pointers — offset to the per-thread element slice. Each thread owns
    // SDPA_EPT contiguous elements at offset `tiisg * SDPA_EPT`.
    const int kv_stride = (p.kv_seq_stride > 0) ? p.kv_seq_stride : sk;
    device const float* q_row = Q + ((bi * p.num_heads    + hi   ) * p.q_len) * d
                                  + tiisg * SDPA_EPT;
    device const float* k_base = K + (bi * p.num_kv_heads + kv_hi) * kv_stride * d;
    device const float* v_base = V + (bi * p.num_kv_heads + kv_hi) * kv_stride * d;
    device       float* o_row = O + ((bi * p.num_heads    + hi   ) * p.q_len) * d;

    // Per-thread Q (pre-scaled), running output, scratch K/V.
    float q[SDPA_EPT];
    float o_acc[SDPA_EPT];
    float k_v[SDPA_EPT];
    for (int i = 0; i < SDPA_EPT; ++i) {
        q[i]     = p.scale * q_row[i];
        o_acc[i] = 0.0f;
    }

    float max_score = -INFINITY;
    float sum_exp   = 0.0f;

    // KV loop — each simdgroup walks positions { sgitg, sgitg+BN, sgitg+2*BN, ... }.
    for (int ki = int(sgitg); ki < attend_end; ki += SDPA_BN) {
        device const float* k_row = k_base + ki * d + tiisg * SDPA_EPT;
        device const float* v_row = v_base + ki * d + tiisg * SDPA_EPT;

        // Read this thread's slice of K and compute per-thread partial dot.
        float dot_acc = 0.0f;
        for (int j = 0; j < SDPA_EPT; ++j) {
            k_v[j]   = k_row[j];
            dot_acc += q[j] * k_v[j];
        }
        // Reduce across the 32 threads of this simdgroup — `score` is the
        // full Q·K dot product for KV position `ki`.
        const float score = simd_sum(dot_acc);

        // Online softmax update (Tri Dao). All 32 threads in this simdgroup
        // see the same `score` after simd_sum so they update identically.
        const float new_max = max(max_score, score);
        const float factor  = exp(max_score - new_max);
        const float exp_sc  = exp(score      - new_max);

        max_score = new_max;
        sum_exp   = sum_exp * factor + exp_sc;

        // Read this thread's slice of V and fold into accumulator.
        for (int j = 0; j < SDPA_EPT; ++j) {
            o_acc[j] = o_acc[j] * factor + exp_sc * v_row[j];
        }
    }

    // ── Cross-simdgroup combine ───────────────────────────────────────
    // Each simdgroup has its own partial (max_score, sum_exp, o_acc) over
    // the KV positions it walked. Merge across the 32 simdgroups using
    // threadgroup memory + the same online-softmax rescaling trick.

    threadgroup float outputs[SDPA_BN * SDPA_D];   // [BN][D]
    threadgroup float max_scores[SDPA_BN];
    threadgroup float sum_exp_scores[SDPA_BN];

    // Stash this simdgroup's partial output (one row of D floats per SG).
    threadgroup float* my_row = outputs + sgitg * SDPA_D + tiisg * SDPA_EPT;
    for (int j = 0; j < SDPA_EPT; ++j) {
        my_row[j] = o_acc[j];
    }
    // The leader of each simdgroup publishes its scalar (max, sum_exp).
    if (tiisg == 0) {
        max_scores[sgitg]     = max_score;
        sum_exp_scores[sgitg] = sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce the 32 (max_score, sum_exp) pairs to a single (M*, S*) using
    // online softmax. Simdgroup 0 does the reduction across all 32 lanes
    // by reading lane `tiisg` from threadgroup memory.
    if (sgitg == 0) {
        const float local_max = max_scores[tiisg];
        const float global_max = simd_max(local_max);
        const float local_factor = exp(local_max - global_max);
        const float local_sum_scaled = sum_exp_scores[tiisg] * local_factor;
        const float global_sum = simd_sum(local_sum_scaled);

        // Store the per-simdgroup rescale factor and the broadcast totals.
        max_scores[tiisg]     = local_factor;          // reused as factor
        sum_exp_scores[tiisg] = global_sum;            // every lane reads the same
        if (tiisg == 0) {
            // Sentinel slot to keep things simple: stash global_max in [0]
            // — no consumer reads it currently (we only need factor + sum)
            // but it's useful for debug. (no-op write, kept as comment)
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final write. Each thread owns SDPA_EPT output positions across D.
    // For each of those positions we need to combine the 32 simdgroup
    // partials. The element is at column `tiisg * SDPA_EPT + j` in the
    // [BN][D] outputs grid; thread `tiisg` of simdgroup `sgitg` walks the
    // 32 simdgroups.
    //
    // But we want exactly ONE thread to write each output element. Use
    // simdgroup 0 to do the writes — its 32 threads cover all D output
    // slots since SDPA_BD * SDPA_EPT = 128 = D.
    if (sgitg == 0) {
        const float inv_S = (sum_exp_scores[0] > 0.0f) ? (1.0f / sum_exp_scores[0]) : 0.0f;

        for (int j = 0; j < SDPA_EPT; ++j) {
            const int col = tiisg * SDPA_EPT + j;   // 0..127
            float total = 0.0f;
            for (int s = 0; s < SDPA_BN; ++s) {
                total += outputs[s * SDPA_D + col] * max_scores[s];
            }
            o_row[col] = total * inv_S;
        }
    }
}
