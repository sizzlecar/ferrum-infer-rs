#include "abi.h"
#include <cuda_runtime.h>
#include <cstring>
#include <limits>
#include "gemm.cuh"
#include "quantize.cuh"

namespace {
constexpr uint32_t ABI = 1;
#if defined(FERRUM_CUTLASS_FP8_SM89)
constexpr uint32_t ARCHITECTURE = 890, KERNEL = 3;
#else
constexpr uint32_t ARCHITECTURE = 1200, KERNEL = 1;
#endif
constexpr uint64_t ALIGNMENT = 256;
bool extent(uint64_t a, uint64_t b, uint64_t element, uint64_t& out) {
  if (!a || !b || a > uint64_t(INT64_MAX) / b || a * b > uint64_t(INT64_MAX) / element) return false;
  out = a * b * element;
  return true;
}
bool add(uint64_t a, uint64_t b, uint64_t& out) {
  if (a > uint64_t(INT64_MAX) || b > uint64_t(INT64_MAX) - a) return false;
  out = a + b; return true;
}
bool align(uint64_t x, uint64_t& out) {
  if (!add(x, ALIGNMENT - 1, out)) return false;
  out &= ~(ALIGNMENT - 1); return true;
}
int request_valid(const FerrumCutlassFp8RequestV1& r) {
  if (r.abi != ABI || r.size != sizeof(r) || !r.rows || !r.inputs || !r.outputs ||
      r.rows > INT32_MAX || r.inputs > INT32_MAX || r.outputs > INT32_MAX ||
      r.device_ordinal > INT32_MAX || !r.multiprocessors || r.multiprocessors > INT32_MAX ||
      !r.maximum_dynamic_shared_bytes || r.maximum_dynamic_shared_bytes > UINT32_MAX) return -1;
  if (r.architecture != ARCHITECTURE || r.inputs % 16 || r.outputs % 16) return -2;
  uint64_t ignored;
  if (!extent(r.rows,r.inputs,2,ignored) || !extent(r.outputs,r.inputs,1,ignored) ||
      !extent(r.rows,r.outputs,2,ignored)) return -3;
  return 0;
}
int device_valid(const FerrumCutlassFp8RequestV1& r) {
  int device=-1, major=0, minor=0, sm=0, shared=0;
  auto s = cudaGetDevice(&device); if (s != cudaSuccess) return int(s);
  if (device != int(r.device_ordinal)) return -2;
  const cudaDeviceAttr attrs[] = {cudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMinor,
      cudaDevAttrMultiProcessorCount, cudaDevAttrMaxSharedMemoryPerBlockOptin};
  int* values[] = {&major,&minor,&sm,&shared};
  for (int i=0;i<4;i++) { s=cudaDeviceGetAttribute(values[i],attrs[i],device); if(s!=cudaSuccess)return int(s); }
  return major*100+minor*10 == int(r.architecture) && sm == int(r.multiprocessors) &&
      shared >= 0 && r.maximum_dynamic_shared_bytes <= uint64_t(shared) ? 0 : -2;
}
int derive(const FerrumCutlassFp8RequestV1& r, FerrumCutlassFp8PlanV1& p) {
  int status=request_valid(r); if(status)return status;
  status=device_valid(r); if(status)return status;
  constexpr auto shared = sizeof(typename ferrum_fp8::Kernel::SharedStorage);
  auto args=ferrum_fp8::arguments(r,nullptr,nullptr,nullptr,nullptr,nullptr);
  if (!shared || shared > r.maximum_dynamic_shared_bytes || shared > UINT32_MAX) return -2;
  uint64_t workspace=ferrum_fp8::Gemm::get_workspace_size(args);
  p={}; p.request=r; p.abi=ABI; p.size=sizeof(p); p.kernel_id=KERNEL; p.shared_bytes=uint32_t(shared);
  uint64_t end;
  if (!extent(r.rows,r.inputs,1,p.quantized_activation_bytes) ||
      !align(p.quantized_activation_bytes,p.activation_scale_offset) ||
      !extent(r.rows,1,4,p.activation_scale_bytes) ||
      !add(p.activation_scale_offset,p.activation_scale_bytes,end) ||
      !align(end,p.gemm_workspace_offset) || !add(p.gemm_workspace_offset,workspace,p.scratch_bytes)) return -3;
  p.gemm_workspace_bytes=workspace;
  return 0;
}
bool span(FerrumCutlassFp8SpanV1 value, uint64_t bytes, uint64_t alignment) {
  return value.address && value.address % alignment == 0 && value.bytes == bytes &&
      value.bytes && value.address <= UINT64_MAX-value.bytes;
}
bool disjoint(FerrumCutlassFp8SpanV1 a, FerrumCutlassFp8SpanV1 b) {
  return a.address >= b.address+b.bytes || b.address >= a.address+a.bytes;
}
auto gemm_arguments(const FerrumCutlassFp8PlanV1& p, const FerrumCutlassFp8BindingsV1& b) {
  return ferrum_fp8::arguments(p.request,
      reinterpret_cast<const ferrum_fp8::Input*>(b.scratch.address),
      reinterpret_cast<const ferrum_fp8::Input*>(b.weight.address),
      reinterpret_cast<const float*>(b.scratch.address+p.activation_scale_offset),
      reinterpret_cast<const float*>(b.weight_scale.address),
      reinterpret_cast<ferrum_fp8::Output*>(b.output.address));
}
int validate(const FerrumCutlassFp8PlanV1& p, const FerrumCutlassFp8BindingsV1& b) {
  FerrumCutlassFp8PlanV1 expected{}; int status=derive(p.request,expected); if(status)return status;
  if (std::memcmp(&p,&expected,sizeof(p))) return -5;
  uint64_t in=0,w=0,out=0;
  extent(p.request.rows,p.request.inputs,2,in); extent(p.request.outputs,p.request.inputs,1,w);
  extent(p.request.rows,p.request.outputs,2,out);
  if (!span(b.input,in,16) || !span(b.weight,w,16) || !span(b.weight_scale,4,4) ||
      !span(b.output,out,16) || !span(b.scratch,p.scratch_bytes,ALIGNMENT)) return -4;
  const FerrumCutlassFp8SpanV1 reads[]={b.input,b.weight,b.weight_scale};
  for (const auto& read:reads) if(!disjoint(b.output,read)||!disjoint(b.scratch,read))return -4;
  if(!disjoint(b.output,b.scratch))return -4;
  return ferrum_fp8::Gemm::can_implement(gemm_arguments(p,b)) == cutlass::Status::kSuccess ? 0 : -6;
}
}

extern "C" int ferrum_cutlass_fp8_plan_v1(const FerrumCutlassFp8RequestV1* request, FerrumCutlassFp8PlanV1* plan) noexcept {
  if(!request||!plan)return -1;
  try { FerrumCutlassFp8PlanV1 result{}; int s=derive(*request,result); if(!s)*plan=result; return s; }
  catch(...){return -8;}
}
extern "C" int ferrum_cutlass_fp8_validate_v1(const FerrumCutlassFp8PlanV1* plan, const FerrumCutlassFp8BindingsV1* bindings) noexcept {
  if(!plan||!bindings)return -1;
  try { return validate(*plan,*bindings); } catch(...){return -8;}
}
extern "C" int ferrum_cutlass_fp8_run_v1(const FerrumCutlassFp8PlanV1* plan, const FerrumCutlassFp8BindingsV1* bindings, void* raw_stream) noexcept {
  if(!plan||!bindings||!raw_stream)return -1;
  try {
    const auto& p=*plan; const auto& b=*bindings;
    int status=validate(p,b); if(status)return status;
    auto stream=reinterpret_cast<cudaStream_t>(raw_stream);
    cudaStreamCaptureStatus capture;
    auto error=cudaStreamIsCapturing(stream,&capture); if(error!=cudaSuccess)return int(error);
    if(capture!=cudaStreamCaptureStatusNone)return -2;
    ferrum_fp8::quantize_rows<<<p.request.rows,256,0,stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(b.input.address), reinterpret_cast<uint8_t*>(b.scratch.address),
      reinterpret_cast<float*>(b.scratch.address+p.activation_scale_offset), p.request.inputs);
    error=cudaPeekAtLastError(); if(error!=cudaSuccess)return int(error);
    ferrum_fp8::Gemm gemm;
    void* workspace=p.gemm_workspace_bytes ? reinterpret_cast<void*>(b.scratch.address+p.gemm_workspace_offset) : nullptr;
#if defined(FERRUM_CUTLASS_FP8_SM89)
    const auto result=gemm(gemm_arguments(p,b),workspace,stream);
#else
    const auto result=gemm.run(gemm_arguments(p,b),workspace,stream);
#endif
    if(result!=cutlass::Status::kSuccess)return -7;
    error=cudaPeekAtLastError(); return error==cudaSuccess ? 0 : int(error);
  } catch(...){return -8;}
}
