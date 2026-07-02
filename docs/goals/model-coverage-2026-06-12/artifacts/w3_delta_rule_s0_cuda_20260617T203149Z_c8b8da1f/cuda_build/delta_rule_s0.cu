
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::cerr << "CUDA error " << cudaGetErrorString(err)                 \
                << " at " << __FILE__ << ":" << __LINE__ << "\n";         \
      return 10;                                                            \
    }                                                                       \
  } while (0)

__global__ void delta_rule_kernel(
    const float* q,
    const float* k,
    const float* v,
    const float* beta,
    float* out,
    int B,
    int H,
    int T,
    int K,
    int V) {
  int bh = blockIdx.x;
  if (threadIdx.x != 0) return;
  int b = bh / H;
  int h = bh % H;
  float state[4096];
  for (int i = 0; i < K * V; ++i) state[i] = 0.0f;
  for (int t = 0; t < T; ++t) {
    float bt = beta[(b * H + h) * T + t];
    for (int vv = 0; vv < V; ++vv) {
      float pred = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        int qki = (((b * H + h) * T + t) * K) + kk;
        pred += k[qki] * state[kk * V + vv];
      }
      int vi = (((b * H + h) * T + t) * V) + vv;
      float delta = bt * (v[vi] - pred);
      for (int kk = 0; kk < K; ++kk) {
        int qki = (((b * H + h) * T + t) * K) + kk;
        state[kk * V + vv] += k[qki] * delta;
      }
    }
    for (int vv = 0; vv < V; ++vv) {
      float acc = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        int qki = (((b * H + h) * T + t) * K) + kk;
        acc += q[qki] * state[kk * V + vv];
      }
      int oi = (((b * H + h) * T + t) * V) + vv;
      out[oi] = acc;
    }
  }
}

static bool read_floats(const char* path, std::vector<float>& data) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  in.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
  return static_cast<size_t>(in.gcount()) == data.size() * sizeof(float);
}

static bool write_floats(const char* path, const std::vector<float>& data) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
  return out.good();
}

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage: " << argv[0] << " <input.bin> <output.bin> B H T K V\n";
    return 2;
  }
  const char* input_path = argv[1];
  const char* output_path = argv[2];
  int B = std::atoi(argv[3]);
  int H = std::atoi(argv[4]);
  int T = std::atoi(argv[5]);
  int K = std::atoi(argv[6]);
  int V = std::atoi(argv[7]);
  size_t qk_count = static_cast<size_t>(B) * H * T * K;
  size_t value_count = static_cast<size_t>(B) * H * T * V;
  size_t beta_count = static_cast<size_t>(B) * H * T;
  std::vector<float> input(qk_count * 2 + value_count + beta_count);
  if (!read_floats(input_path, input)) {
    std::cerr << "failed to read input\n";
    return 3;
  }
  const float* q_host = input.data();
  const float* k_host = q_host + qk_count;
  const float* v_host = k_host + qk_count;
  const float* beta_host = v_host + value_count;
  std::vector<float> output(value_count, 0.0f);

  float *q, *k, *v, *beta, *out;
  CUDA_CHECK(cudaMalloc(&q, qk_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&k, qk_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&v, value_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&beta, beta_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&out, value_count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(q, q_host, qk_count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(k, k_host, qk_count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(v, v_host, value_count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(beta, beta_host, beta_count * sizeof(float), cudaMemcpyHostToDevice));
  delta_rule_kernel<<<B * H, 1>>>(q, k, v, beta, out, B, H, T, K, V);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), out, value_count * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(q));
  CUDA_CHECK(cudaFree(k));
  CUDA_CHECK(cudaFree(v));
  CUDA_CHECK(cudaFree(beta));
  CUDA_CHECK(cudaFree(out));
  if (!write_floats(output_path, output)) {
    std::cerr << "failed to write output\n";
    return 4;
  }
  return 0;
}
