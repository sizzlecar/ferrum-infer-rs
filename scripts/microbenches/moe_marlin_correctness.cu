// Standalone correctness test for ferrum_vllm_marlin_moe_f16.
//
// First sweep: all values constant — proved the kernel + repack + template
// instantiation work end-to-end for the simplest case.
//
// Second sweep (this file): probes multi-expert dispatch + non-uniform
// scales + non-uniform weights — the cases the constant-fill couldn't
// detect.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
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

// Pack a slice of K-many INT4 values (LSB=k[0], MSB=k[7]) into one i32.
static inline uint32_t pack_int4_block(const uint8_t* k_vals) {
    uint32_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= ((uint32_t)(k_vals[i] & 0xF)) << (i * 4);
    }
    return v;
}

// Produce GPTQ on-disk qweight [K/8, N] given per-(k,n) INT4 values [0..15].
// w_int4[k * N + n] = stored value.
static std::vector<int32_t>
build_gptq_qweight(const std::vector<uint8_t>& w_int4, int K, int N) {
    int blocks_k = K / 8;
    std::vector<int32_t> out(blocks_k * N, 0);
    for (int bk = 0; bk < blocks_k; ++bk) {
        for (int n = 0; n < N; ++n) {
            uint8_t k_vals[8];
            for (int i = 0; i < 8; ++i) {
                k_vals[i] = w_int4[(bk * 8 + i) * N + n];
            }
            out[bk * N + n] = (int32_t)pack_int4_block(k_vals);
        }
    }
    return out;
}

// FP16 reference: C[m, n] = sum_k A[m,k] * dequant_kU4B8(W[k,n]) * scale[gk,n]
// dequant_kU4B8(stored) = (stored - 8)
static float ref_C(int m, int n, int K, int N,
                   const std::vector<float>& A_f32,
                   const std::vector<uint8_t>& w_int4,
                   const std::vector<float>& scales_f32, int group_size)
{
    float acc = 0;
    for (int k = 0; k < K; ++k) {
        int gk = k / group_size;
        float a = A_f32[m * K + k];
        int stored = (int)w_int4[k * N + n];
        float w = (float)(stored - 8);
        float s = scales_f32[gk * N + n];
        acc += a * w * s;
    }
    return acc;
}

struct TestResult { float max_diff; int nan_count; int sentinel_count; };

// c_tmp / atomic_add mode for a test. ferrum production calls with
// non-null c_tmp + use_fp32_reduce=1.
enum ReduceMode {
    ATOMIC_ADD,           // c_tmp = nullptr, use_atomic_add=1, use_fp32_reduce=0
    FP32_REDUCE_FERRUM,   // c_tmp = 2*1024*1024 fp32 (ferrum's mod.rs:313)
    FP32_REDUCE_VLLM,     // c_tmp = sms*4*block*max_thread_n fp32 (vLLM's torch wrapper)
};

enum WorkspaceMode {
    WS_VLLM,    // (n_per * 4) per expert — vllm FFI comment formula N/128*sms*4
    WS_FERRUM,  // (n_per / 64) * 16 per expert — ferrum vllm_marlin.rs:255
};

// Run one (multi-expert) test. expert_layouts[e] gives INT4 values for expert e.
// expert_scales[e] gives FP scales for expert e.
// token_to_expert: which expert each token routes to.
// Returns: max |output - reference|.
static TestResult run_one(const char* label,
                          int K, int N, int M, int top_k, int num_experts,
                          int group_size, int moe_block_size,
                          const std::vector<std::vector<uint8_t>>& expert_layouts,
                          const std::vector<std::vector<float>>& expert_scales,
                          const std::vector<int>& token_to_expert,
                          const std::vector<float>& A_f32,
                          ReduceMode reduce_mode = ATOMIC_ADD,
                          WorkspaceMode ws_mode = WS_VLLM)
{
    printf("\n=== %s ===\n", label);
    printf("  K=%d N=%d M=%d top_k=%d num_experts=%d group_size=%d block=%d\n",
           K, N, M, top_k, num_experts, group_size, moe_block_size);

    int num_groups = K / group_size;
    int qw_per_expert = (K / 8) * N;
    int sc_per_expert = num_groups * N;

    // Build per-expert GPTQ qweight, concat into stacked.
    std::vector<int32_t> h_B_gptq(num_experts * qw_per_expert);
    std::vector<__half>  h_scales(num_experts * sc_per_expert);
    for (int e = 0; e < num_experts; ++e) {
        auto qe = build_gptq_qweight(expert_layouts[e], K, N);
        std::memcpy(&h_B_gptq[e * qw_per_expert], qe.data(), qe.size() * sizeof(int32_t));
        for (int i = 0; i < sc_per_expert; ++i) {
            h_scales[e * sc_per_expert + i] = __float2half(expert_scales[e][i]);
        }
    }

    std::vector<__half> h_A(M * K);
    for (int i = 0; i < M * K; ++i) h_A[i] = __float2half(A_f32[i]);

    // moe_align outputs — built by hand for this test.
    // Each token routes to exactly one expert (top_k=1) for simplicity here.
    // Build per-expert token lists then pad to moe_block_size.
    std::vector<std::vector<int>> per_expert_tokens(num_experts);
    for (int t = 0; t < M * top_k; ++t) {
        per_expert_tokens[token_to_expert[t]].push_back(t);
    }
    std::vector<int32_t> h_sorted_ids;
    std::vector<int32_t> h_expert_ids;
    int sentinel = M * top_k;
    for (int e = 0; e < num_experts; ++e) {
        auto& toks = per_expert_tokens[e];
        if (toks.empty()) continue;
        int blocks = (toks.size() + moe_block_size - 1) / moe_block_size;
        for (int b = 0; b < blocks; ++b) {
            for (int i = 0; i < moe_block_size; ++i) {
                int idx = b * moe_block_size + i;
                h_sorted_ids.push_back(idx < (int)toks.size() ? toks[idx] : sentinel);
            }
            h_expert_ids.push_back(e);
        }
    }
    int32_t total_padded = (int32_t)h_sorted_ids.size();

    // Allocate device buffers
    __half *d_A, *d_C, *d_scales;
    int32_t *d_B_gptq, *d_B_packed, *d_workspace, *d_sorted_ids, *d_expert_ids, *d_npp;

    CHK(cudaMalloc(&d_A, sizeof(__half) * M * K));
    CHK(cudaMalloc(&d_C, sizeof(__half) * M * top_k * N));
    CHK(cudaMalloc(&d_scales, sizeof(__half) * h_scales.size()));
    CHK(cudaMalloc(&d_B_gptq, sizeof(int32_t) * h_B_gptq.size()));
    CHK(cudaMalloc(&d_B_packed, sizeof(int32_t) * h_B_gptq.size()));

    int sms = 128;
    int ws_per_expert_vllm   = ((N + 127) / 128) * sms * 4;
    int ws_per_expert_ferrum = ((N + 63) / 64) * 16;
    if (ws_per_expert_vllm   < 16) ws_per_expert_vllm   = 16;
    if (ws_per_expert_ferrum < 16) ws_per_expert_ferrum = 16;
    int ws_per_expert = (ws_mode == WS_VLLM) ? ws_per_expert_vllm : ws_per_expert_ferrum;
    int ws_size = num_experts * ws_per_expert;
    printf("  workspace=%s  ws_per_expert=%d  total=%d i32 (vllm-wants=%d)\n",
           ws_mode == WS_VLLM ? "VLLM" : "FERRUM",
           ws_per_expert, ws_size, ws_per_expert_vllm);
    CHK(cudaMalloc(&d_workspace, sizeof(int32_t) * ws_size));
    CHK(cudaMemset(d_workspace, 0, sizeof(int32_t) * ws_size));

    CHK(cudaMalloc(&d_sorted_ids, sizeof(int32_t) * total_padded));
    CHK(cudaMalloc(&d_expert_ids, sizeof(int32_t) * h_expert_ids.size()));
    CHK(cudaMalloc(&d_npp, sizeof(int32_t)));

    CHK(cudaMemcpy(d_A, h_A.data(), sizeof(__half) * M * K, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_B_gptq, h_B_gptq.data(), sizeof(int32_t) * h_B_gptq.size(), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_scales, h_scales.data(), sizeof(__half) * h_scales.size(), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_sorted_ids, h_sorted_ids.data(), sizeof(int32_t) * total_padded, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_expert_ids, h_expert_ids.data(), sizeof(int32_t) * h_expert_ids.size(), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_npp, &total_padded, sizeof(int32_t), cudaMemcpyHostToDevice));

    cudaStream_t stream = 0;

    // Repack each expert in place into d_B_packed (per-expert offset call).
    // This mirrors what load_stacked_gptq_vllm_marlin does in vllm_marlin.rs.
    for (int e = 0; e < num_experts; ++e) {
        int32_t* eoff_in  = d_B_gptq  + e * qw_per_expert;
        int32_t* eoff_out = d_B_packed + e * qw_per_expert;
        int ret = ferrum_vllm_gptq_marlin_repack(
            eoff_in, nullptr, eoff_out,
            K, N, /*num_bits=*/4, /*has_perm=*/0, /*dev=*/0, stream);
        if (ret != 0) {
            printf("  ERROR: repack expert %d returned %d\n", e, ret);
            return {NAN, 0, 0};
        }
    }
    CHK(cudaStreamSynchronize(stream));

    // Zero output sentinel
    std::vector<__half> sent(M * top_k * N, __float2half(-9999.0f));
    CHK(cudaMemcpy(d_C, sent.data(), sizeof(__half) * sent.size(), cudaMemcpyHostToDevice));

    // c_tmp + reduce mode setup
    float* d_c_tmp = nullptr;
    int use_atomic_add_flag, use_fp32_reduce_flag;
    size_t c_tmp_alloc_fp32 = 0;
    if (reduce_mode == ATOMIC_ADD) {
        d_c_tmp = nullptr;
        use_atomic_add_flag = 1;
        use_fp32_reduce_flag = 0;
    } else {
        if (reduce_mode == FP32_REDUCE_FERRUM) {
            c_tmp_alloc_fp32 = 2ULL * 1024 * 1024;  // ferrum mod.rs:313
        } else {  // FP32_REDUCE_VLLM
            size_t bound_a = (size_t)N * total_padded;
            size_t bound_b = (size_t)sms * 4 * moe_block_size * 256;
            if (moe_block_size == 8) bound_b *= 2;
            c_tmp_alloc_fp32 = bound_a < bound_b ? bound_a : bound_b;
        }
        CHK(cudaMalloc(&d_c_tmp, sizeof(float) * c_tmp_alloc_fp32));
        CHK(cudaMemset(d_c_tmp, 0, sizeof(float) * c_tmp_alloc_fp32));
        use_atomic_add_flag = 0;
        use_fp32_reduce_flag = 1;
    }
    printf("  reduce_mode=%s c_tmp_size=%zu fp32 (%.1f MB)\n",
           reduce_mode == ATOMIC_ADD ? "atomic_add" :
           (reduce_mode == FP32_REDUCE_FERRUM ? "fp32_reduce_FERRUM_2M" : "fp32_reduce_VLLM_full"),
           c_tmp_alloc_fp32, (c_tmp_alloc_fp32 * 4) / 1e6);

    int ret = ferrum_vllm_marlin_moe_f16(
        d_A, d_B_packed, d_C,
        d_c_tmp,
        d_scales, d_workspace,
        d_sorted_ids, d_expert_ids, d_npp,
        /*topk_weights=*/nullptr,
        moe_block_size, top_k,
        /*mul_topk_weights=*/0, /*is_ep=*/0,
        /*prob_m=*/M, /*prob_n=*/N, /*prob_k=*/K,
        group_size, /*dev=*/0, stream,
        use_atomic_add_flag, use_fp32_reduce_flag);
    if (ret != 0) {
        printf("  ERROR: marlin_moe_f16 returned %d\n", ret);
        return {NAN, 0, 0};
    }
    cudaError_t sync = cudaStreamSynchronize(stream);
    if (sync != cudaSuccess) {
        printf("  ERROR after kernel: %s\n", cudaGetErrorString(sync));
        return {NAN, 0, 0};
    }

    std::vector<__half> h_C(M * top_k * N);
    CHK(cudaMemcpy(h_C.data(), d_C, sizeof(__half) * h_C.size(), cudaMemcpyDeviceToHost));

    // Compare against reference.
    // For top_k=1: output row t corresponds to token t routed to expert token_to_expert[t].
    TestResult tr{0.0f, 0, 0};
    int show_count = 0;
    for (int t = 0; t < M * top_k; ++t) {
        int e = token_to_expert[t];
        for (int n = 0; n < N; ++n) {
            float got = __half2float(h_C[t * N + n]);
            if (isnan(got) || isinf(got)) { tr.nan_count++; continue; }
            if (got < -9000.0f) { tr.sentinel_count++; continue; }
            // Reference: A[token_id] @ dequant(expert_layouts[e]) * scales[e]
            // For our test we assume token id == row in A.
            int token_id = t / top_k;
            float exp = ref_C(token_id, n, K, N, A_f32, expert_layouts[e],
                              expert_scales[e], group_size);
            float diff = fabsf(got - exp);
            if (diff > tr.max_diff) tr.max_diff = diff;
            if (show_count < 4 && (n == 0 || n == 1 || n == 17 || n == N - 1)) {
                printf("  t=%d e=%d n=%d got=%.3f exp=%.3f diff=%.3f\n",
                       t, e, n, got, exp, diff);
                ++show_count;
            }
        }
    }

    printf("  max|diff|=%.4f  nan=%d  sentinel=%d  ",
           tr.max_diff, tr.nan_count, tr.sentinel_count);
    if (tr.nan_count > 0)        printf("VERDICT: FAIL (NaN/Inf)\n");
    else if (tr.sentinel_count>0)printf("VERDICT: FAIL (untouched)\n");
    else if (tr.max_diff > 1.0f) printf("VERDICT: FAIL (max diff %.4f)\n", tr.max_diff);
    else                          printf("VERDICT: PASS\n");

    cudaFree(d_A); cudaFree(d_C); cudaFree(d_scales);
    cudaFree(d_B_gptq); cudaFree(d_B_packed); cudaFree(d_workspace);
    cudaFree(d_sorted_ids); cudaFree(d_expert_ids); cudaFree(d_npp);
    if (d_c_tmp) cudaFree(d_c_tmp);
    return tr;
}

int main() {
    cudaDeviceProp prop; CHK(cudaGetDeviceProperties(&prop, 0));
    printf("Device 0: %s CC=%d.%d SMs=%d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    // ────────────────────────────────────────────────────────────
    // TEST 1: Single expert, NON-UNIFORM weights — exposes if repack
    //         actually permutes the bits correctly (the original all-9
    //         test missed this because all nibbles were identical).
    // ────────────────────────────────────────────────────────────
    {
        int K=128, N=128, group_size=128;
        std::vector<uint8_t> w(K * N);
        for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) {
            // Pattern: stored value alternates with k position. INT4 in [0..15].
            w[k * N + n] = (uint8_t)((k * 3 + n * 5) % 16);
        }
        std::vector<float> s(K / group_size * N, 1.0f);
        std::vector<float> A(K, 1.0f);
        run_one("T1: 1 expert, non-uniform weights, scales=1, A=1",
                K, N, /*M=*/1, /*top_k=*/1, /*num_experts=*/1, group_size, /*block=*/16,
                {w}, {s}, {0}, A);
    }

    // ────────────────────────────────────────────────────────────
    // TEST 2: Single expert, non-uniform SCALES per group (need K > group_size).
    // ────────────────────────────────────────────────────────────
    {
        int K=256, N=128, group_size=128;  // 2 groups
        std::vector<uint8_t> w(K * N);
        for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) {
            w[k * N + n] = 9;  // dequant +1
        }
        std::vector<float> s(K / group_size * N);
        // group 0 → scale 1.0, group 1 → scale 2.0 (per N varies slightly)
        for (int g = 0; g < 2; ++g) for (int n = 0; n < N; ++n) {
            s[g * N + n] = (g == 0) ? 1.0f : 2.0f;
        }
        std::vector<float> A(K, 1.0f);
        run_one("T2: 1 expert, scales differ per group",
                K, N, 1, 1, 1, group_size, 16, {w}, {s}, {0}, A);
    }

    // ────────────────────────────────────────────────────────────
    // TEST 3: 2 EXPERTS, each with distinct weights. Tokens 0&1 route
    //         to expert 0; tokens 2&3 to expert 1.
    // ────────────────────────────────────────────────────────────
    {
        int K=128, N=128, group_size=128;
        std::vector<uint8_t> w0(K * N), w1(K * N);
        for (int i = 0; i < K * N; ++i) {
            w0[i] = 9;   // expert 0: dequant +1
            w1[i] = 10;  // expert 1: dequant +2
        }
        std::vector<float> s(K / group_size * N, 1.0f);
        std::vector<float> A(4 * K, 1.0f);
        run_one("T3: 2 experts, different weights (e0=+1, e1=+2)",
                K, N, /*M=*/4, /*top_k=*/1, /*num_experts=*/2, group_size, 16,
                {w0, w1}, {s, s},
                {0, 0, 1, 1},  // tokens 0,1 → e0; 2,3 → e1
                A);
    }

    // ────────────────────────────────────────────────────────────
    // TEST 4: 2 experts, NON-UNIFORM weights per expert (real-world shape).
    //         This is the test that most resembles the actual Qwen3-MoE.
    // ────────────────────────────────────────────────────────────
    {
        int K=128, N=128, group_size=128;
        std::vector<uint8_t> w0(K * N), w1(K * N);
        for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) {
            w0[k * N + n] = (uint8_t)((k * 3 + n * 5) % 16);
            w1[k * N + n] = (uint8_t)((k * 7 + n * 2 + 4) % 16);  // different pattern
        }
        std::vector<float> s(K / group_size * N, 1.0f);
        std::vector<float> A(4 * K);
        for (size_t i = 0; i < A.size(); ++i) A[i] = 1.0f;
        run_one("T4: 2 experts, non-uniform weights, tokens split",
                K, N, 4, 1, 2, group_size, 16,
                {w0, w1}, {s, s},
                {0, 0, 1, 1},
                A);
    }

    // ────────────────────────────────────────────────────────────
    // TEST 6 / 7 / 8: Qwen3-MoE-shaped prefill — large total_padded
    //   - K=2048 N=1536 group_size=128 (Qwen3-30B-A3B gate_up dims)
    //   - 128 experts, M=64 tokens (prefill cohort), top_k=1
    //   - constant weights = +1, scales = 1, A = 1 → expected output K = 2048
    // T6: atomic_add path (c_tmp=null) — ferrum NEVER uses this in production
    // T7: fp32_reduce with ferrum's 2M c_tmp (the actual production case)
    // T8: fp32_reduce with vLLM's max-bound c_tmp (the correct size)
    // ────────────────────────────────────────────────────────────
    auto qwen_shape_test = [&](const char* tag, ReduceMode mode) {
        int K=2048, N=1536, group_size=128, num_experts=128;
        int M = 64;  // prefill chunk
        int moe_block_size = 16;
        std::vector<std::vector<uint8_t>> ws(num_experts);
        std::vector<std::vector<float>>  ss(num_experts);
        for (int e = 0; e < num_experts; ++e) {
            ws[e].assign(K * N, 9);  // dequant +1
            ss[e].assign(K / group_size * N, 1.0f);
        }
        std::vector<int> route(M);
        for (int t = 0; t < M; ++t) route[t] = t % num_experts;  // spread across experts
        std::vector<float> A(M * K, 1.0f);
        run_one(tag, K, N, M, 1, num_experts, group_size, moe_block_size,
                ws, ss, route, A, mode);
    };
    qwen_shape_test("T6: Qwen3-shape, ATOMIC_ADD, ws=VLLM", ATOMIC_ADD);
    qwen_shape_test("T7: Qwen3-shape, FP32_REDUCE FERRUM c_tmp, ws=VLLM", FP32_REDUCE_FERRUM);
    qwen_shape_test("T8: Qwen3-shape, FP32_REDUCE VLLM c_tmp, ws=VLLM", FP32_REDUCE_VLLM);

    // T9/T10/T11: same as T6/T7/T8 but with FERRUM's small workspace —
    // expected to FAIL if the workspace-size bug hypothesis is correct.
    auto qwen_shape_test_wsferrum = [&](const char* tag, ReduceMode mode) {
        int K=2048, N=1536, group_size=128, num_experts=128;
        int M = 64;
        int moe_block_size = 16;
        std::vector<std::vector<uint8_t>> ws(num_experts);
        std::vector<std::vector<float>>  ss(num_experts);
        for (int e = 0; e < num_experts; ++e) {
            ws[e].assign(K * N, 9);
            ss[e].assign(K / group_size * N, 1.0f);
        }
        std::vector<int> route(M);
        for (int t = 0; t < M; ++t) route[t] = t % num_experts;
        std::vector<float> A(M * K, 1.0f);
        run_one(tag, K, N, M, 1, num_experts, group_size, moe_block_size,
                ws, ss, route, A, mode, WS_FERRUM);
    };
    qwen_shape_test_wsferrum("T9 : Qwen3-shape, ATOMIC_ADD, ws=FERRUM (16x smaller)", ATOMIC_ADD);
    qwen_shape_test_wsferrum("T10: Qwen3-shape, FP32_REDUCE FERRUM c_tmp, ws=FERRUM", FP32_REDUCE_FERRUM);
    qwen_shape_test_wsferrum("T11: Qwen3-shape, FP32_REDUCE VLLM c_tmp, ws=FERRUM", FP32_REDUCE_VLLM);

    // ────────────────────────────────────────────────────────────
    // T12/T13: Production-scale stress — M=2048 to push total_padded
    // up to ~2048 (matches batched prefill at c=32 * 256 tokens).
    // Tests if ferrum's 2M c_tmp is enough when first-bound is active.
    // ────────────────────────────────────────────────────────────
    auto qwen_stress = [&](const char* tag, ReduceMode rm, WorkspaceMode wm) {
        int K=2048, N=1536, group_size=128, num_experts=128;
        int M = 2048;  // worst-case batch * top_k after pre-gather
        int moe_block_size = 16;
        std::vector<std::vector<uint8_t>> ws(num_experts);
        std::vector<std::vector<float>>  ss(num_experts);
        for (int e = 0; e < num_experts; ++e) {
            // Per-expert NON-UNIFORM weight pattern so output depends on routing.
            ws[e].assign(K * N, 0);
            for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) {
                ws[e][k * N + n] = (uint8_t)((e * 7 + k * 3 + n * 5) % 16);
            }
            ss[e].assign(K / group_size * N, 1.0f);
        }
        std::vector<int> route(M);
        for (int t = 0; t < M; ++t) route[t] = t % num_experts;  // 16 tokens per expert
        std::vector<float> A(M * K, 1.0f);
        run_one(tag, K, N, M, 1, num_experts, group_size, moe_block_size,
                ws, ss, route, A, rm, wm);
    };
    qwen_stress("T12: Qwen3 STRESS M=2048 + FERRUM c_tmp + VLLM ws", FP32_REDUCE_FERRUM, WS_VLLM);
    qwen_stress("T13: Qwen3 STRESS M=2048 + VLLM c_tmp + VLLM ws",  FP32_REDUCE_VLLM,   WS_VLLM);
    qwen_stress("T14: Qwen3 STRESS M=2048 + FERRUM c_tmp + FERRUM ws (production)",
                FP32_REDUCE_FERRUM, WS_FERRUM);

    // ────────────────────────────────────────────────────────────
    // Original T5: kept for regression
    // ────────────────────────────────────────────────────────────
    {
        int K=128, N=128, group_size=128, num_experts=8;
        std::vector<std::vector<uint8_t>> ws(num_experts);
        std::vector<std::vector<float>> ss(num_experts);
        for (int e = 0; e < num_experts; ++e) {
            ws[e].assign(K * N, (uint8_t)(8 + e));  // stored 8..15 → dequant 0..7
            ss[e].assign(K / group_size * N, 1.0f);
        }
        int M = num_experts;  // one token per expert
        std::vector<int> route(M);
        for (int t = 0; t < M; ++t) route[t] = t;
        std::vector<float> A(M * K, 1.0f);
        run_one("T5: 8 experts (stored 8..15), 1 token each",
                K, N, M, 1, num_experts, group_size, 16,
                ws, ss, route, A);
    }

    return 0;
}
