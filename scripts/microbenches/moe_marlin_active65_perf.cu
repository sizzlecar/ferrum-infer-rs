// Standalone Ferrum vLLM-Marlin-MoE raw-op benchmark for the real c=32 route
// class captured on 2026-05-29:
//   batch_x_topk=256, block_size=16, total_post_pad=1040,
//   active_blocks=65, unique_experts=61.
//
// Build after a release build with vllm-moe-marlin:
//   OUT=target/release/build/ferrum-kernels-<hash>/out
//   nvcc -O3 -arch=sm_89 -std=c++17 \
//     scripts/microbenches/moe_marlin_active65_perf.cu \
//     $OUT/vllm_moe_ops.o $OUT/vllm_moe_kernel_instantiations.o \
//     $OUT/gptq_marlin_repack.o $OUT/sm89_kernel_*.o $OUT/sm80_kernel_*.o \
//     -lcuda -lcudart -o /tmp/moe_marlin_active65_perf

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" int ferrum_vllm_gptq_marlin_repack(
    const void* qweight_in, const void* perm_in, void* qweight_out, int size_k,
    int size_n, int num_bits, int has_perm, int dev, cudaStream_t stream);

extern "C" int ferrum_vllm_marlin_moe_f16(
    const void* A, const void* B, void* C, void* C_tmp, const void* b_scales,
    void* workspace, const int* sorted_token_ids, const int* expert_ids,
    const int* num_tokens_past_padded, const float* topk_weights,
    int moe_block_size, int top_k, int mul_topk_weights, int is_ep, int prob_m,
    int prob_n, int prob_k, int group_size, int dev, cudaStream_t stream,
    int use_atomic_add, int use_fp32_reduce);

#define CHK(stmt)                                                           \
  do {                                                                      \
    cudaError_t err__ = (stmt);                                             \
    if (err__ != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,          \
                   cudaGetErrorString(err__));                              \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

constexpr int NUM_EXPERTS = 128;
constexpr int BLOCK = 16;
constexpr int GROUP_SIZE = 128;
constexpr int WARMUP_ITERS = 10;
constexpr int TIMED_ITERS = 200;

static std::vector<int32_t> build_random_qweight(int k, int n, uint32_t seed) {
  std::vector<int32_t> out((k / 8) * n, 0);
  uint32_t s = seed;
  auto rnd = [&]() {
    s = s * 1664525u + 1013904223u;
    return s;
  };
  for (int bk = 0; bk < k / 8; ++bk) {
    for (int col = 0; col < n; ++col) {
      uint32_t v = 0;
      for (int i = 0; i < 8; ++i) v |= (rnd() & 0xFu) << (i * 4);
      out[bk * n + col] = static_cast<int32_t>(v);
    }
  }
  return out;
}

static void repack_stack(int k, int n, std::vector<int32_t>& out_marlin) {
  const int per_expert = (n * k) / 8;
  out_marlin.assign(static_cast<size_t>(NUM_EXPERTS) * per_expert, 0);

  int32_t* d_in = nullptr;
  int32_t* d_out = nullptr;
  CHK(cudaMalloc(&d_in, (k / 8) * n * sizeof(int32_t)));
  CHK(cudaMalloc(&d_out, per_expert * sizeof(int32_t)));
  for (int e = 0; e < NUM_EXPERTS; ++e) {
    auto q = build_random_qweight(k, n, 0xA5110000u ^ static_cast<uint32_t>(e));
    CHK(cudaMemcpy(d_in, q.data(), q.size() * sizeof(int32_t),
                   cudaMemcpyHostToDevice));
    int rc =
        ferrum_vllm_gptq_marlin_repack(d_in, nullptr, d_out, k, n, 4, 0, 0, 0);
    if (rc != 0) {
      std::fprintf(stderr, "repack failed expert=%d rc=%d\n", e, rc);
      std::exit(1);
    }
    CHK(cudaMemcpy(&out_marlin[static_cast<size_t>(e) * per_expert], d_out,
                   per_expert * sizeof(int32_t), cudaMemcpyDeviceToHost));
  }
  CHK(cudaFree(d_in));
  CHK(cudaFree(d_out));
}

static void build_active65_route(int total_pairs, std::vector<int>& sorted,
                                 std::vector<int>& experts,
                                 std::vector<int>& npp) {
  int counts[65];
  for (int i = 0; i < 61; ++i) counts[i] = 4;
  for (int i = 61; i < 65; ++i) counts[i] = 3;

  sorted.assign(65 * BLOCK, total_pairs);
  experts.resize(65);
  int row = 0;
  for (int b = 0; b < 65; ++b) {
    for (int i = 0; i < counts[b]; ++i) sorted[b * BLOCK + i] = row++;
    experts[b] = b < 61 ? b : b - 61;
  }
  if (row != total_pairs) {
    std::fprintf(stderr, "bad route row count %d != %d\n", row, total_pairs);
    std::exit(1);
  }
  npp = {65 * BLOCK};
}

static double run_shape(const char* name, int m, int n, int k, int top_k) {
  const int total_pairs = m * top_k;
  std::vector<int> sorted_h, experts_h, npp_h;
  build_active65_route(total_pairs, sorted_h, experts_h, npp_h);

  std::vector<int32_t> b_h;
  repack_stack(k, n, b_h);

  const int groups = k / GROUP_SIZE;
  std::vector<__half> scales_h(static_cast<size_t>(NUM_EXPERTS) * groups * n);
  uint32_t s = 0xDEADBEEFu;
  for (auto& v : scales_h) {
    s = s * 1664525u + 1013904223u;
    float f = 0.02f + 0.06f * static_cast<float>(s & 0xffffu) / 65536.0f;
    v = __float2half(f);
  }

  std::vector<__half> a_h(static_cast<size_t>(m) * k);
  s = 0xC001D00Du;
  for (auto& v : a_h) {
    s = s * 1664525u + 1013904223u;
    float f = (static_cast<int>(s % 1000u) - 500) / 1000.0f;
    v = __float2half(f);
  }

  __half* d_a = nullptr;
  __half* d_c = nullptr;
  float* d_c_tmp = nullptr;
  int32_t* d_b = nullptr;
  __half* d_scales = nullptr;
  int* d_workspace = nullptr;
  int* d_sorted = nullptr;
  int* d_experts = nullptr;
  int* d_npp = nullptr;

  CHK(cudaMalloc(&d_a, a_h.size() * sizeof(__half)));
  CHK(cudaMalloc(&d_c, static_cast<size_t>(total_pairs) * n * sizeof(__half)));
  CHK(cudaMalloc(&d_c_tmp, 262144 * sizeof(float)));
  CHK(cudaMalloc(&d_b, b_h.size() * sizeof(int32_t)));
  CHK(cudaMalloc(&d_scales, scales_h.size() * sizeof(__half)));
  CHK(cudaMalloc(&d_workspace, static_cast<size_t>(NUM_EXPERTS) * (n / 64) * 16 *
                                      sizeof(int32_t)));
  CHK(cudaMalloc(&d_sorted, sorted_h.size() * sizeof(int)));
  CHK(cudaMalloc(&d_experts, experts_h.size() * sizeof(int)));
  CHK(cudaMalloc(&d_npp, sizeof(int)));

  CHK(cudaMemcpy(d_a, a_h.data(), a_h.size() * sizeof(__half),
                 cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(d_b, b_h.data(), b_h.size() * sizeof(int32_t),
                 cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(d_scales, scales_h.data(), scales_h.size() * sizeof(__half),
                 cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(d_sorted, sorted_h.data(), sorted_h.size() * sizeof(int),
                 cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(d_experts, experts_h.data(), experts_h.size() * sizeof(int),
                 cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(d_npp, npp_h.data(), sizeof(int), cudaMemcpyHostToDevice));
  CHK(cudaMemset(d_workspace, 0,
                 static_cast<size_t>(NUM_EXPERTS) * (n / 64) * 16 *
                     sizeof(int32_t)));
  CHK(cudaMemset(d_c_tmp, 0, 262144 * sizeof(float)));

  cudaStream_t stream;
  cudaEvent_t start, stop;
  CHK(cudaStreamCreate(&stream));
  CHK(cudaEventCreate(&start));
  CHK(cudaEventCreate(&stop));

  auto launch = [&]() {
    return ferrum_vllm_marlin_moe_f16(
        d_a, d_b, d_c, d_c_tmp, d_scales, d_workspace, d_sorted, d_experts,
        d_npp, nullptr, BLOCK, top_k, 0, 0, m, n, k, GROUP_SIZE, 0, stream, 0,
        1);
  };

  for (int i = 0; i < WARMUP_ITERS; ++i) {
    int rc = launch();
    if (rc != 0) {
      std::fprintf(stderr, "%s warmup rc=%d\n", name, rc);
      std::exit(1);
    }
  }
  CHK(cudaStreamSynchronize(stream));

  CHK(cudaEventRecord(start, stream));
  for (int i = 0; i < TIMED_ITERS; ++i) {
    int rc = launch();
    if (rc != 0) {
      std::fprintf(stderr, "%s timed rc=%d iter=%d\n", name, rc, i);
      std::exit(1);
    }
  }
  CHK(cudaEventRecord(stop, stream));
  CHK(cudaStreamSynchronize(stream));

  float elapsed_ms = 0.0f;
  CHK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  double us = elapsed_ms * 1000.0 / TIMED_ITERS;
  double useful_tflops =
      (2.0 * static_cast<double>(total_pairs) * n * k) / (us * 1e-6) / 1e12;
  std::fprintf(stderr,
               "%s m=%d n=%d k=%d top_k=%d active_blocks=65 npp=1040 "
               "time=%.3f us useful=%.2f TFLOPS\n",
               name, m, n, k, top_k, us, useful_tflops);

  CHK(cudaEventDestroy(start));
  CHK(cudaEventDestroy(stop));
  CHK(cudaStreamDestroy(stream));
  CHK(cudaFree(d_a));
  CHK(cudaFree(d_c));
  CHK(cudaFree(d_c_tmp));
  CHK(cudaFree(d_b));
  CHK(cudaFree(d_scales));
  CHK(cudaFree(d_workspace));
  CHK(cudaFree(d_sorted));
  CHK(cudaFree(d_experts));
  CHK(cudaFree(d_npp));
  return us;
}

int main() {
  std::fprintf(stderr,
               "Ferrum Marlin-MoE active65 raw benchmark, warmup=%d timed=%d\n",
               WARMUP_ITERS, TIMED_ITERS);
  double gate = run_shape("gate_up_active65", 32, 1536, 2048, 8);
  double down = run_shape("down_active65", 256, 2048, 768, 1);
  std::printf("{\"gate_up_us\":%.6f,\"down_us\":%.6f,"
              "\"route\":{\"batch_x_topk\":256,\"block_size\":16,"
              "\"total_post_pad\":1040,\"active_blocks\":65,"
              "\"unique_experts\":61}}\n",
              gate, down);
  return 0;
}
