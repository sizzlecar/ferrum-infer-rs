#include "abi.h"
#include <cmath>
#include <limits>
#include <cuda_runtime.h>
#ifdef HAS_PYTORCH
#error "The native operator must not depend on PyTorch"
#endif
#include "kernel_forward.h"

namespace {
using D64 = AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, false, false>;
using D72 = AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 128, false, false>;
static_assert(!D64::kNeedsOutputAccumulatorBuffer && !D72::kNeedsOutputAccumulatorBuffer,
              "native operator assumes no global scratch");

bool valid(const FerrumCutlassFusedF32AttentionRequestV1* r) {
  return r && r->abi == 1 && r->size == sizeof(*r) && r->reserved == 0 &&
      r->batches > 0 && r->batches <= 65535 && r->heads > 0 && r->heads <= 65535 &&
      r->sequence > 0 && r->sequence <= 65535 &&
      (r->dimension == 64 || r->dimension == 72) &&
      std::isfinite(r->scale) && r->scale > 0 &&
      uint64_t(r->sequence) * r->heads * r->dimension <= uint64_t(INT32_MAX);
}

bool valid_spans(const FerrumCutlassFusedF32AttentionRequestV1& r, const FerrumCutlassFusedF32AttentionBindingsV1& b) {
  const uint64_t count = uint64_t(r.batches) * r.sequence * r.heads * r.dimension;
  const FerrumCutlassFusedF32AttentionSpanV1 spans[] = {b.query, b.key, b.value, b.output};
  for (const auto& s : spans) {
    if (s.address == 0 || s.address % 16 != 0 || s.elements != count ||
        count > (UINT64_MAX - s.address) / sizeof(float)) return false;
  }
  const uint64_t bytes = count * sizeof(float);
  for (int i = 0; i < 3; ++i) {
    if (b.output.address < spans[i].address + bytes &&
        spans[i].address < b.output.address + bytes) return false;
  }
  return true;
}

template<class A>
int prepare(FerrumCutlassFusedF32AttentionInfoV1* out) {
  auto kernel = attention_kernel_batched_impl<A>;
  const int bytes = sizeof(typename A::SharedStorage);
  if (bytes > 0xc000) {
    const auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (status != cudaSuccess) return int(status);
  }
  *out = {1, sizeof(*out), uint32_t(bytes), A::kNumThreads,
          A::kQueriesPerBlock, A::kKeysPerBlock, 0};
  return 0;
}

template<class A>
int run(const FerrumCutlassFusedF32AttentionRequestV1& r, const FerrumCutlassFusedF32AttentionBindingsV1& b, void* stream) {
  typename A::Params p{};
  p.query_ptr = reinterpret_cast<float*>(b.query.address);
  p.key_ptr = reinterpret_cast<float*>(b.key.address);
  p.value_ptr = reinterpret_cast<float*>(b.value.address);
  p.output_ptr = reinterpret_cast<float*>(b.output.address);
  p.scale = r.scale;
  p.head_dim = p.head_dim_value = r.dimension;
  p.num_queries = p.num_keys = p.num_keys_absolute = r.sequence;
  p.num_heads = r.heads;
  p.num_batches = r.batches;
  p.q_strideH = p.k_strideH = p.v_strideH = r.dimension;
  p.q_strideM = p.k_strideM = p.v_strideM = p.o_strideM = r.heads * r.dimension;
  p.q_strideB = p.k_strideB = p.v_strideB = int64_t(r.sequence) * r.heads * r.dimension;
  p.custom_mask_type = A::NoCustomMask;
  p.use_dropout = false;
  if (!A::check_supported(p)) return -2;
  attention_kernel_batched_impl<A><<<p.getBlocksGrid(), p.getThreadsGrid(),
      sizeof(typename A::SharedStorage), reinterpret_cast<cudaStream_t>(stream)>>>(p);
  return int(cudaGetLastError());
}
}

extern "C" __attribute__((visibility("default"))) int
ferrum_cutlass_fused_f32_attention_prepare_v1(const FerrumCutlassFusedF32AttentionRequestV1* r, FerrumCutlassFusedF32AttentionInfoV1* out) noexcept {
  if (!valid(r) || !out) return -1;
  try { return r->dimension == 64 ? prepare<D64>(out) : prepare<D72>(out); }
  catch (...) { return -3; }
}

extern "C" __attribute__((visibility("default"))) int
ferrum_cutlass_fused_f32_attention_run_v1(const FerrumCutlassFusedF32AttentionRequestV1* r, const FerrumCutlassFusedF32AttentionBindingsV1* b,
                          void* stream) noexcept {
  if (!valid(r) || !b || !valid_spans(*r, *b)) return -1;
  try { return r->dimension == 64 ? run<D64>(*r, *b, stream) : run<D72>(*r, *b, stream); }
  catch (...) { return -3; }
}
