#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace ferrum_fp8 {
// 256 threads, one row per CTA. All payloads are supplied by the caller.
// fmin/fmax intentionally follow CUDA NaN selection semantics; the accuracy
// qualification domain is finite inputs, not idealized NaN propagation.
__global__ void quantize_rows(const __nv_bfloat16* x, uint8_t* a, float* scales,
                             uint32_t width) {
  __shared__ float reduction[256];
  const uint64_t base = uint64_t(blockIdx.x) * width;
  float maximum = 0.0f;
  for (uint32_t column = threadIdx.x; column < width; column += blockDim.x)
    maximum = fmaxf(maximum, fabsf(__bfloat162float(x[base + column])));
  reduction[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned stride = 128; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride)
      reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    reduction[0] = fmaxf(__fdiv_rn(reduction[0], 448.0f), 1.0f / (448.0f * 512.0f));
    scales[blockIdx.x] = reduction[0];
  }
  __syncthreads();
  const float scale = reduction[0];
  for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
    float value = __fdiv_rn(__bfloat162float(x[base + column]), scale);
    value = fmaxf(-448.0f, fminf(value, 448.0f));
    a[base + column] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
  }
}
} // namespace ferrum_fp8
