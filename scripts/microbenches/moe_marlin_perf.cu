// Standalone perf microbench for ferrum_vllm_marlin_moe_f16 at Qwen3-MoE
// production shape. Goal: answer "is marlin_moe near RTX 4090 roofline at
// the Qwen3-30B-A3B shape, or is there kernel-level headroom?"
//
// Output: per-cell µs/call + achieved TFLOPS vs theoretical peak. Reads
// no model weights — synthetic random INT4 with U4B8 + per-group scales.
//
// Production shape (Qwen3-30B-A3B-GPTQ-Int4, gate_up_proj):
//   K = 2048 (hidden_size)
//   N = 1536 (= 2 × moe_intermediate_size, gate + up packed)
//   num_experts = 128
//   top_k = 8
//   m_per_expert avg ~= batch × top_k / num_experts
//                     ≈ 0.0625 × c (for c=16 → 1 token/expert,
//                                    c=32 → 2 tokens/expert)
//
// We probe ferrum_vllm_marlin_moe_f16 with m_per_expert ∈ {1, 2, 4, 8, 16}
// (Marlin tile minimum is 16, so effective prob_m=16 for all small m;
// the kernel pads with zeros — measures cost of "wasted" Marlin tile).
//
// Build:
//   nvcc -O3 -arch=sm_89 -std=c++17 -I<ferrum-root>/crates/ferrum-kernels \
//        scripts/microbenches/moe_marlin_perf.cu \
//        crates/ferrum-kernels/kernels/vllm_marlin_moe/ops.cu \
//        crates/ferrum-kernels/kernels/vllm_marlin/gptq_marlin_repack.cu \
//        -lcuda -o moe_marlin_perf
//
// (The exact compile command lives in scripts/microbenches/README.md plus
// the build.rs of ferrum-kernels for the canonical set of source files
// the vllm-moe-marlin feature pulls in. If linking fails with undefined
// `marlin::Marlin<...>`, also pull in vllm_marlin_moe/marlin_kernels_*.cu
// — those are the per-shape template instantiations.)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" int ferrum_vllm_gptq_marlin_repack(
    const void* qweight_in, const void* perm_in, void* qweight_out,
    int size_k, int size_n, int num_bits, int has_perm,
    int dev, cudaStream_t stream);

extern "C" int ferrum_vllm_marlin_moe_f16(
    const void* A, const void* B, void* C, void* C_tmp,
    const void* b_scales, void* workspace,
    const int* sorted_token_ids, const int* expert_ids,
    const int* num_tokens_past_padded, const float* topk_weights,
    int moe_block_size, int top_k, int mul_topk_weights, int is_ep,
    int prob_m, int prob_n, int prob_k, int group_size,
    int dev, cudaStream_t stream,
    int use_atomic_add, int use_fp32_reduce);

#define CHK(stmt) do { cudaError_t e=(stmt); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(1);}} while(0)

constexpr int K            = 2048;   // hidden_size
constexpr int N            = 1536;   // 2 × moe_intermediate_size
constexpr int NUM_EXPERTS  = 128;
constexpr int TOP_K        = 8;       // unused by kernel (we pass top_k=1
                                       // since ferrum passes pre-gathered x_packed)
constexpr int GROUP_SIZE   = 128;
constexpr int MOE_BLOCK    = 16;       // ferrum's VLLM_MOE_BLOCK_SIZE
constexpr int WARMUP_ITERS = 8;
constexpr int TIMED_ITERS  = 200;

// RTX 4090 sm_89 published peak FP16 tensor TFLOPS (per spec sheet).
// Marlin's INT4-weight × FP16-act effectively runs at FP16 GEMM
// throughput (the INT4 → FP16 dequant happens in-flight). Per the
// Marlin paper, achieves 80-90% of dense FP16 peak on H100; sm_89
// numbers (4090) are similar in shape. Use 140 TFLOPS as the
// "near-peak" reference (≈ 85% of 165).
constexpr double RTX4090_FP16_TFLOPS_PEAK   = 165.0;
constexpr double RTX4090_FP16_TFLOPS_USEFUL = 140.0;

// Build a per-expert sorted_token_ids/expert_ids layout that distributes
// `total_pairs` pairs (= batch × top_k) across `active_experts` experts
// in round-robin order. Returns padded total.
static int build_routing(
    int total_pairs, int active_experts,
    std::vector<int>& sorted_token_ids,
    std::vector<int>& expert_ids,
    int& m_per_expert_max)
{
    // pairs_per_expert ≈ total_pairs / active_experts (rounded up)
    int per = (total_pairs + active_experts - 1) / active_experts;
    int per_padded = ((per + MOE_BLOCK - 1) / MOE_BLOCK) * MOE_BLOCK;
    int total_padded = active_experts * per_padded;
    int total_blocks = total_padded / MOE_BLOCK;

    int sentinel = total_pairs;
    sorted_token_ids.assign(total_padded, sentinel);
    expert_ids.assign(total_blocks, 0);

    int p = 0; // packed_row cursor
    for (int e = 0; e < active_experts; ++e) {
        int padded_off = e * per_padded;
        // Fill `per` packed_rows (or fewer for the last expert if not divisible)
        int m_e = std::min(per, total_pairs - p);
        if (m_e < 0) m_e = 0;
        for (int i = 0; i < m_e; ++i) {
            sorted_token_ids[padded_off + i] = p + i; // packed_row indexing
        }
        for (int b = 0; b < per_padded / MOE_BLOCK; ++b) {
            expert_ids[padded_off / MOE_BLOCK + b] = e;
        }
        p += m_e;
    }
    m_per_expert_max = per_padded;
    return total_padded;
}

// Synthetic GPTQ packing: same pattern as moe_marlin_correctness.cu but
// with random INT4 values (deterministic seed).
static std::vector<int32_t> build_random_qweight(int K_, int N_, uint32_t seed) {
    int blocks_k = K_ / 8;
    std::vector<int32_t> out(blocks_k * N_, 0);
    uint32_t s = seed;
    auto rnd = [&]() { s = s * 1664525u + 1013904223u; return s; };
    for (int bk = 0; bk < blocks_k; ++bk) {
        for (int n = 0; n < N_; ++n) {
            uint32_t v = 0;
            for (int i = 0; i < 8; ++i) {
                v |= (uint32_t)(rnd() & 0xF) << (i * 4);
            }
            out[bk * N_ + n] = (int32_t)v;
        }
    }
    return out;
}

// Repack stacked GPTQ INT4 into Marlin tile layout for `n_experts` experts.
// Each expert's repacked output goes into `out_marlin` at the per-expert offset.
static void repack_stack(
    const std::vector<std::vector<int32_t>>& gptq_per_expert,
    int K_, int N_,
    std::vector<int32_t>& out_marlin)
{
    int n_experts = (int)gptq_per_expert.size();
    int marlin_i32_per_expert = (N_ * K_) / 8; // 4 bits/elem
    out_marlin.assign(n_experts * marlin_i32_per_expert, 0);

    int32_t* d_in  = nullptr;
    int32_t* d_out = nullptr;
    CHK(cudaMalloc(&d_in,  (K_ / 8) * N_ * sizeof(int32_t)));
    CHK(cudaMalloc(&d_out, marlin_i32_per_expert * sizeof(int32_t)));
    for (int e = 0; e < n_experts; ++e) {
        CHK(cudaMemcpy(d_in, gptq_per_expert[e].data(),
                       (K_ / 8) * N_ * sizeof(int32_t), cudaMemcpyHostToDevice));
        int rc = ferrum_vllm_gptq_marlin_repack(
            d_in, nullptr, d_out, K_, N_, 4, 0, 0, 0);
        if (rc != 0) { fprintf(stderr, "repack expert %d failed rc=%d\n", e, rc); std::exit(1); }
        CHK(cudaMemcpy(&out_marlin[e * marlin_i32_per_expert], d_out,
                       marlin_i32_per_expert * sizeof(int32_t), cudaMemcpyDeviceToHost));
    }
    CHK(cudaFree(d_in));
    CHK(cudaFree(d_out));
}

struct CellResult {
    int   c;
    int   active_experts;
    int   m_per_expert_avg;
    int   prob_m_padded;
    double avg_us;
    double total_gflops;
    double achieved_tflops;
    double pct_of_useful_peak;
};

static CellResult run_cell(
    int c, int active_experts, int m_per_expert_avg,
    const std::vector<int32_t>& B_marlin_stacked,
    const std::vector<__half>&  scales_stacked,
    int n_experts_stacked)
{
    int total_pairs = c * TOP_K; // (= m_total)
    std::vector<int> sorted_tok, eids;
    int prob_m_padded = 0;
    int total_padded = build_routing(total_pairs, active_experts,
                                     sorted_tok, eids, prob_m_padded);
    int total_blocks = total_padded / MOE_BLOCK;
    std::vector<int> num_past_pad = { total_padded };

    // Allocate device buffers
    __half* d_A = nullptr;
    __half* d_C = nullptr;
    float*  d_C_tmp = nullptr;
    int32_t* d_B = nullptr;
    __half*  d_scales = nullptr;
    int*     d_workspace = nullptr;
    int*     d_sorted = nullptr;
    int*     d_eids   = nullptr;
    int*     d_npp    = nullptr;

    int A_count   = total_pairs * K;       // pre-gathered x_packed with top_k=1
    int C_count   = total_pairs * N;
    int Btile_count_per_e = (N * K) / 8;
    int scales_groups = K / GROUP_SIZE;
    int scales_count_per_e = scales_groups * N;
    // ferrum's per-expert workspace: (N/64)*16 i32 (matches vllm_marlin.rs:255)
    int ws_count_per_e = (N / 64) * 16;
    int sms = 128; // 4090 has 128 SMs; c_tmp sized per vLLM's torch wrapper rule:
                   // c_tmp = sms * 4 * MIN_THREAD_N * max_thread_m
                   // MIN_THREAD_N=128, max_thread_m=4 → sms*4*128*4 fp32
                   // = 128*4*128*4 = 262144 fp32 = 1 MB (covers any reasonable size).
                   // Ferrum's mod.rs:313 fixes 2 MB, also fine.
    int c_tmp_count = sms * 4 * 128 * 4;

    CHK(cudaMalloc(&d_A,        A_count * sizeof(__half)));
    CHK(cudaMalloc(&d_C,        C_count * sizeof(__half)));
    CHK(cudaMalloc(&d_C_tmp,    c_tmp_count * sizeof(float)));
    CHK(cudaMalloc(&d_B,        (size_t)n_experts_stacked * Btile_count_per_e * sizeof(int32_t)));
    CHK(cudaMalloc(&d_scales,   (size_t)n_experts_stacked * scales_count_per_e * sizeof(__half)));
    CHK(cudaMalloc(&d_workspace,(size_t)n_experts_stacked * ws_count_per_e * sizeof(int32_t)));
    CHK(cudaMalloc(&d_sorted,   total_padded * sizeof(int)));
    CHK(cudaMalloc(&d_eids,     total_blocks * sizeof(int)));
    CHK(cudaMalloc(&d_npp,      sizeof(int)));

    // Fill A with random FP16 (just for realistic memory traffic; we don't
    // verify output here)
    std::vector<__half> A_host(A_count);
    uint32_t s = 0xC0DEC0DE;
    auto rnd_f = [&]() { s = s * 1664525u + 1013904223u; return (float)((int)(s % 1000) - 500) / 500.0f; };
    for (int i = 0; i < A_count; ++i) A_host[i] = __float2half(rnd_f() * 0.5f);
    CHK(cudaMemcpy(d_A, A_host.data(), A_count * sizeof(__half), cudaMemcpyHostToDevice));

    // Copy B + scales (stacked)
    CHK(cudaMemcpy(d_B, B_marlin_stacked.data(),
                   B_marlin_stacked.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_scales, scales_stacked.data(),
                   scales_stacked.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_sorted, sorted_tok.data(), total_padded * sizeof(int), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_eids, eids.data(), total_blocks * sizeof(int), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_npp, num_past_pad.data(), sizeof(int), cudaMemcpyHostToDevice));
    // Zero workspace (Marlin reads + writes locks; must start zeroed)
    CHK(cudaMemset(d_workspace, 0, (size_t)n_experts_stacked * ws_count_per_e * sizeof(int32_t)));
    CHK(cudaMemset(d_C_tmp,     0, c_tmp_count * sizeof(float)));

    // Stream + events
    cudaStream_t stream;
    CHK(cudaStreamCreate(&stream));
    cudaEvent_t t0, t1;
    CHK(cudaEventCreate(&t0));
    CHK(cudaEventCreate(&t1));

    auto launch = [&]() {
        // Pass topk_weights = nullptr + mul_topk_weights=0 → kernel skips
        // the per-row weight multiply (we're only measuring GEMM cost here).
        // c_tmp != null + use_fp32_reduce=1 = ferrum production setting.
        return ferrum_vllm_marlin_moe_f16(
            d_A, d_B, d_C, d_C_tmp, d_scales, d_workspace,
            d_sorted, d_eids, d_npp, /*topk_weights=*/nullptr,
            MOE_BLOCK, /*top_k=*/1, /*mul_topk_weights=*/0, /*is_ep=*/0,
            prob_m_padded, N, K, GROUP_SIZE,
            0, stream,
            /*use_atomic_add=*/0, /*use_fp32_reduce=*/1);
    };

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        CHK(cudaMemsetAsync(d_workspace, 0,
                            (size_t)n_experts_stacked * ws_count_per_e * sizeof(int32_t),
                            stream));
        CHK(cudaMemsetAsync(d_C_tmp, 0, c_tmp_count * sizeof(float), stream));
        int rc = launch();
        if (rc != 0) { fprintf(stderr, "launch failed rc=%d (warmup)\n", rc); std::exit(1); }
    }
    CHK(cudaStreamSynchronize(stream));

    // Timed loop. Reset workspace + c_tmp inside the timed window so we
    // measure the full per-call cost including the small memset; this
    // mirrors ferrum's per-iter call shape.
    CHK(cudaEventRecord(t0, stream));
    for (int i = 0; i < TIMED_ITERS; ++i) {
        CHK(cudaMemsetAsync(d_workspace, 0,
                            (size_t)n_experts_stacked * ws_count_per_e * sizeof(int32_t),
                            stream));
        CHK(cudaMemsetAsync(d_C_tmp, 0, c_tmp_count * sizeof(float), stream));
        int rc = launch();
        if (rc != 0) { fprintf(stderr, "launch failed rc=%d (timed iter %d)\n", rc, i); std::exit(1); }
    }
    CHK(cudaEventRecord(t1, stream));
    CHK(cudaStreamSynchronize(stream));

    float total_ms = 0;
    CHK(cudaEventElapsedTime(&total_ms, t0, t1));
    double avg_us = (double)(total_ms * 1e3) / TIMED_ITERS;

    // Useful FLOPs per call:
    //   sum over active experts of 2 × m_e × N × K  ≈  2 × total_pairs × N × K
    double useful_flops = 2.0 * (double)total_pairs * N * K;
    // Total FLOPs including Marlin's padding (kernel runs prob_m_padded rows per expert):
    double total_flops  = 2.0 * (double)prob_m_padded * N * K * active_experts;
    double useful_tflops_avg = useful_flops / (avg_us * 1e-6) / 1e12;
    double total_tflops_avg  = total_flops  / (avg_us * 1e-6) / 1e12;

    CellResult cr;
    cr.c = c;
    cr.active_experts = active_experts;
    cr.m_per_expert_avg = m_per_expert_avg;
    cr.prob_m_padded = prob_m_padded;
    cr.avg_us = avg_us;
    cr.total_gflops = total_flops / 1e9;
    cr.achieved_tflops = useful_tflops_avg;
    cr.pct_of_useful_peak = 100.0 * useful_tflops_avg / RTX4090_FP16_TFLOPS_USEFUL;

    // Free
    cudaStreamDestroy(stream);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_A); cudaFree(d_C); cudaFree(d_C_tmp);
    cudaFree(d_B); cudaFree(d_scales); cudaFree(d_workspace);
    cudaFree(d_sorted); cudaFree(d_eids); cudaFree(d_npp);

    fprintf(stderr,
        "  c=%-2d active=%-3d m/e_avg=%-2d prob_m_pad=%-2d  %6.1f µs/call"
        "  useful=%5.1f TFLOPS (%4.1f%% of %.0f peak)  padded=%5.1f TFLOPS\n",
        c, active_experts, m_per_expert_avg, prob_m_padded,
        avg_us, useful_tflops_avg, cr.pct_of_useful_peak,
        RTX4090_FP16_TFLOPS_USEFUL, total_tflops_avg);

    return cr;
}

int main() {
    fprintf(stderr, "=== moe_marlin_perf — Qwen3-30B-A3B shape ===\n");
    fprintf(stderr, "K=%d N=%d num_experts=%d top_k=%d group_size=%d block=%d\n",
            K, N, NUM_EXPERTS, TOP_K, GROUP_SIZE, MOE_BLOCK);
    fprintf(stderr, "warmup=%d timed=%d\n\n", WARMUP_ITERS, TIMED_ITERS);

    // Build a stacked weight set: NUM_EXPERTS experts, each with a unique
    // random GPTQ qweight + scales. Repack once (out of the timed loop).
    fprintf(stderr, "▶ building synthetic GPTQ weights + repacking %d experts...\n", NUM_EXPERTS);
    std::vector<std::vector<int32_t>> gptq_per_expert(NUM_EXPERTS);
    for (int e = 0; e < NUM_EXPERTS; ++e) {
        gptq_per_expert[e] = build_random_qweight(K, N, 0xCAFE0000u ^ (uint32_t)e);
    }
    std::vector<int32_t> B_marlin_stacked;
    repack_stack(gptq_per_expert, K, N, B_marlin_stacked);
    fprintf(stderr, "  repacked: %.1f MB total\n",
            (double)B_marlin_stacked.size() * sizeof(int32_t) / 1e6);

    // Scales — random per (gk, n) per expert, all close to 0.05 so the
    // GEMM doesn't overflow FP16. We're not verifying numerics here so
    // exact values don't matter.
    int groups = K / GROUP_SIZE;
    std::vector<__half> scales_stacked((size_t)NUM_EXPERTS * groups * N);
    uint32_t s = 0xDECAFBAD;
    auto rnd = [&]() { s = s * 1664525u + 1013904223u; return (float)(s & 0xFFFF) / 65536.0f; };
    for (size_t i = 0; i < scales_stacked.size(); ++i) {
        scales_stacked[i] = __float2half(0.02f + 0.06f * rnd());
    }
    fprintf(stderr, "  scales: %.1f MB\n",
            (double)scales_stacked.size() * sizeof(__half) / 1e6);

    // Sweep cells covering bench c=1, 4, 16, 32 routed to all 128 experts.
    // (With top_k=8 and 128 experts, c=16 → m_per_expert avg = 1, c=32 → 2.)
    fprintf(stderr, "\n▶ probing per-c with all 128 experts active:\n");
    std::vector<CellResult> results;
    int CELLS[][2] = {
        // {c, active_experts}
        { 1,   8 },   // c=1 × top_k=8 → 8 active experts max
        { 4,  32 },   // c=4 × top_k=8 → 32 active
        { 16, 128 },  // c=16 saturates 128 experts (1/expert avg)
        { 32, 128 },  // c=32 → 2/expert avg
    };
    for (auto& cell : CELLS) {
        int c = cell[0];
        int active = cell[1];
        int total_pairs = c * TOP_K;
        int m_per_e_avg = (total_pairs + active - 1) / active;
        results.push_back(run_cell(c, active, m_per_e_avg,
                                   B_marlin_stacked, scales_stacked, NUM_EXPERTS));
    }

    fprintf(stderr, "\n▶ probing m_per_expert scaling (active=8, c=8 → c=128):\n");
    int M_SCALE[][2] = {
        // simulate per-expert m by feeding c such that m_per_expert ≈ {1,2,4,8,16}
        { 8,   8 },   // m_per_e ≈ 1
        { 16,  8 },   // m_per_e ≈ 2
        { 32,  8 },   // m_per_e ≈ 4
        { 64,  8 },   // m_per_e ≈ 8
        {128,  8 },   // m_per_e ≈ 16 (matches Marlin's tile minimum)
    };
    for (auto& cell : M_SCALE) {
        int c = cell[0];
        int active = cell[1];
        int total_pairs = c * TOP_K;
        int m_per_e_avg = (total_pairs + active - 1) / active;
        results.push_back(run_cell(c, active, m_per_e_avg,
                                   B_marlin_stacked, scales_stacked, NUM_EXPERTS));
    }

    fprintf(stderr, "\n=== summary ===\n");
    fprintf(stderr, "shape: K=%d N=%d num_experts_stacked=%d  RTX 4090 useful FP16 peak=%.0f TFLOPS\n\n",
            K, N, NUM_EXPERTS, RTX4090_FP16_TFLOPS_USEFUL);
    fprintf(stderr, "%-4s %-7s %-7s %-9s %-8s %-9s %-7s\n",
            "c", "active", "m/e", "prob_m_p", "µs/call", "TFLOPS", "%peak");
    for (auto& r : results) {
        fprintf(stderr, "%-4d %-7d %-7d %-9d %-8.1f %-9.1f %-6.1f%%\n",
                r.c, r.active_experts, r.m_per_expert_avg, r.prob_m_padded,
                r.avg_us, r.achieved_tflops, r.pct_of_useful_peak);
    }

    // VERDICT
    double max_pct = 0;
    for (auto& r : results) max_pct = std::max(max_pct, r.pct_of_useful_peak);
    fprintf(stderr, "\nVERDICT: peak achieved on this sweep = %.1f%% of %.0f TFLOPS roofline.\n",
            max_pct, RTX4090_FP16_TFLOPS_USEFUL);
    if (max_pct < 30.0) {
        fprintf(stderr, "  → Kernel runs FAR below roofline. Likely BW-bound (small per-expert m)\n"
                        "    or launch-overhead-bound. Kernel-level rewrite unlikely to help —\n"
                        "    focus on reducing launches (graph coverage) or per-iter call count.\n");
    } else if (max_pct < 60.0) {
        fprintf(stderr, "  → Mid-range. Could be tile heuristic suboptimal at small m, or\n"
                        "    inherent per-expert overhead. Worth a tile-bucket experiment.\n");
    } else {
        fprintf(stderr, "  → Close to roofline. No more headroom in marlin_moe — pursue\n"
                        "    other levers (attn, launch count, lm_head, scheduler).\n");
    }
    return 0;
}
