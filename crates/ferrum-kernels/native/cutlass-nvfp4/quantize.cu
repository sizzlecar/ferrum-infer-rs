#include "abi.h"
#include "boundary_checks.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstring>
#include <initializer_list>

namespace {
using namespace ferrum_nvfp4_boundary;

__device__ __forceinline__ float rcp_ftz(float value) {
  float result;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}
__device__ __forceinline__ float mul_ftz(float a, float b) {
  float result;
  asm volatile("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
  return result;
}
__device__ __forceinline__ bool zero_ftz(float value) {
  uint32_t result;
  // FlashInfer's fast-math profile also flushes the zero comparison. Merely
  // making multiplication/reciprocal FTZ gives a different subnormal branch.
  asm volatile("{ .reg .pred p; setp.eq.ftz.f32 p,%1,%2; selp.u32 %0,1,0,p; }"
    : "=r"(result) : "f"(value), "f"(0.0f));
  return result != 0;
}
__device__ __forceinline__ uint32_t pack8(const float* x) {
  uint32_t result;
  asm volatile("{ .reg .b8 a,b,c,d;\n"
    "cvt.rn.satfinite.e2m1x2.f32 a,%2,%1;\n"
    "cvt.rn.satfinite.e2m1x2.f32 b,%4,%3;\n"
    "cvt.rn.satfinite.e2m1x2.f32 c,%6,%5;\n"
    "cvt.rn.satfinite.e2m1x2.f32 d,%8,%7;\n"
    "mov.b32 %0,{a,b,c,d}; }" : "=r"(result) :
      "f"(x[0]),"f"(x[1]),"f"(x[2]),"f"(x[3]),"f"(x[4]),"f"(x[5]),"f"(x[6]),"f"(x[7]));
  return result;
}
// Independent group ownership avoids inter-CTA state or a reduction scratch.
// Two aligned vector loads per group preserve the BF16 input boundary.
__global__ void quantize(const __nv_bfloat16* input, const float* global_scale,
    uint32_t* packed, uint8_t* scale_output, uint32_t rows, uint32_t width) {
  const uint64_t groups = width / 16;
  const uint64_t scale_columns = (groups + 3) / 4 * 4;
  const uint64_t work = ((uint64_t(rows) + 127) / 128 * 128) * scale_columns;
  const float global = *global_scale;
  for (uint64_t item = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       item < work; item += uint64_t(gridDim.x) * blockDim.x) {
    const uint64_t row = item / scale_columns, group = item % scale_columns;
    const uint64_t sf_index = ((((row / 128) * (scale_columns / 4) + group / 4) * 32 + row % 32) * 4 + (row / 32) % 4) * 4 + group % 4;
    if (row >= rows || group >= groups) { scale_output[sf_index] = 0; continue; }
    union alignas(16) Vector { uint4 chunks[2]; __nv_bfloat162 pairs[8]; } values;
    const uint4* address = reinterpret_cast<const uint4*>(input + row * width + group * 16);
    values.chunks[0] = address[0];
    values.chunks[1] = address[1];
    __nv_bfloat162 maximum = __habs2(values.pairs[0]);
    #pragma unroll
    for (int i = 1; i < 8; ++i) maximum = __hmax2(maximum, __habs2(values.pairs[i]));
    const float amax = __bfloat162float(__hmax(maximum.x, maximum.y));
    const __nv_fp8_e4m3 scale(mul_ftz(global, mul_ftz(amax, rcp_ftz(6.0f))));
    scale_output[sf_index] = scale.__x;
    const float multiplier = zero_ftz(amax) ? 0.0f : rcp_ftz(mul_ftz(float(scale), rcp_ftz(global)));
    float result[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float2 x = __bfloat1622float2(values.pairs[i]);
      result[i * 2] = mul_ftz(x.x, multiplier);
      result[i * 2 + 1] = mul_ftz(x.y, multiplier);
    }
    const uint64_t output = (row * groups + group) * 2;
    packed[output] = pack8(result);
    packed[output + 1] = pack8(result + 8);
  }
}

int derive(const FerrumNvfp4QuantizeBf16RequestV1& r, FerrumNvfp4QuantizeBf16PlanV1& p) {
  if (r.abi != 1 || r.size != sizeof(r) || !r.rows || !r.inputs || !r.multiprocessors ||
      r.rows > INT32_MAX || r.inputs > INT32_MAX || r.multiprocessors > INT32_MAX || r.device_ordinal > INT32_MAX) return -1;
  if (r.architecture != 1200 || r.inputs % 16 || r.profile != 1) return -2;
  uint64_t ignored;
  if (!extent(r.rows, uint64_t(r.inputs) * 2, ignored) || scales(r.rows, r.inputs) > INT64_MAX) return -3;
  int status = device(r.device_ordinal, r.architecture, r.multiprocessors, 0);
  if (status) return status;
  uint64_t blocks = (scales(r.rows, r.inputs) + 127) / 128;
  if (blocks > uint64_t(r.multiprocessors) * 8) blocks = uint64_t(r.multiprocessors) * 8;
  if (blocks > 65535) blocks = 65535;
  p = {}; p.request = r; p.abi = 1; p.size = sizeof(p); p.kernel_id = 1; p.blocks = uint32_t(blocks); p.threads = 128;
  return 0;
}
int validate(const FerrumNvfp4QuantizeBf16PlanV1& p, const FerrumNvfp4QuantizeBf16BindingsV1& b) {
  FerrumNvfp4QuantizeBf16PlanV1 expected{};
  int status = derive(p.request, expected); if (status) return status;
  if (std::memcmp(&expected, &p, sizeof(p))) return -5;
  const auto& r = p.request;
  if (!span(b.input, uint64_t(r.rows) * r.inputs * 2, 16) || !span(b.global_scale, 4, 4) ||
      !span(b.packed, uint64_t(r.rows) * r.inputs / 2, 16) || !span(b.scales, scales(r.rows, r.inputs), 16)) return -4;
  for (auto output : {b.packed, b.scales}) for (auto input : {b.input, b.global_scale}) if (!disjoint(output, input)) return -4;
  return disjoint(b.packed, b.scales) ? 0 : -4;
}
}
extern "C" int ferrum_nvfp4_quantize_bf16_plan_v1(const FerrumNvfp4QuantizeBf16RequestV1* r, FerrumNvfp4QuantizeBf16PlanV1* p) noexcept {
  if (!r || !p) return -1;
  try { FerrumNvfp4QuantizeBf16PlanV1 result{}; int status = derive(*r, result); if (!status) *p = result; return status; } catch (...) { return -8; }
}
extern "C" int ferrum_nvfp4_quantize_bf16_validate_v1(const FerrumNvfp4QuantizeBf16PlanV1* p, const FerrumNvfp4QuantizeBf16BindingsV1* b) noexcept {
  if (!p || !b) return -1;
  try { return validate(*p, *b); } catch (...) { return -8; }
}
extern "C" int ferrum_nvfp4_quantize_bf16_run_v1(const FerrumNvfp4QuantizeBf16PlanV1* p, const FerrumNvfp4QuantizeBf16BindingsV1* b, void* raw) noexcept {
  if (!p || !b) return -1;
  try {
    int status = validate(*p, *b); if (status) return status;
    cudaStream_t launch_stream;
    status = stream(raw, launch_stream); if (status) return status;
    quantize<<<p->blocks, p->threads, 0, launch_stream>>>(reinterpret_cast<const __nv_bfloat16*>(b->input.address),
      reinterpret_cast<const float*>(b->global_scale.address), reinterpret_cast<uint32_t*>(b->packed.address),
      reinterpret_cast<uint8_t*>(b->scales.address), p->request.rows, p->request.inputs);
    return int(cudaPeekAtLastError());
  } catch (...) { return -8; }
}
