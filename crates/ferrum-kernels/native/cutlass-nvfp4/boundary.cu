#include "abi.h"
#include "boundary_checks.cuh"
#include <cuda_runtime.h>
#include <cstring>
#include <limits>
#include "gemm.cuh"

namespace {
using namespace ferrum_nvfp4_boundary;
int request_valid(const FerrumCutlassNvfp4RequestV1& r) {
  if(r.abi!=1||r.size!=sizeof(r)||!r.rows||!r.inputs||!r.outputs||!r.multiprocessors||
      r.rows>INT32_MAX||r.inputs>INT32_MAX||r.outputs>INT32_MAX||r.multiprocessors>INT32_MAX||
      r.device_ordinal>INT32_MAX||!r.maximum_dynamic_shared_bytes||r.maximum_dynamic_shared_bytes>UINT32_MAX)return -1;
  if(r.architecture!=1200||r.inputs%128||r.outputs%8)return -2;
  uint64_t ignored;
  if(!extent(r.rows,r.inputs/2,ignored)||!extent(r.outputs,r.inputs/2,ignored)||
      !extent(r.rows,uint64_t(r.outputs)*2,ignored)||scales(r.rows,r.inputs)>INT64_MAX||
      scales(r.outputs,r.inputs)>INT64_MAX)return -3;
  return 0;
}
int device_valid(const FerrumCutlassNvfp4RequestV1& r) {
  return device(r.device_ordinal,r.architecture,r.multiprocessors,r.maximum_dynamic_shared_bytes);
}
int derive(const FerrumCutlassNvfp4RequestV1& r,FerrumCutlassNvfp4PlanV1& p) {
  int status=request_valid(r);if(status)return status;
  status=device_valid(r);if(status)return status;
  constexpr auto shared=sizeof(typename ferrum_nvfp4::Kernel::SharedStorage);
  if(!shared||shared>r.maximum_dynamic_shared_bytes||shared>UINT32_MAX)return -2;
  auto args=ferrum_nvfp4::arguments(r,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr);
  uint64_t workspace=ferrum_nvfp4::Gemm::get_workspace_size(args);
  if(workspace>INT64_MAX)return -3;
  p={};p.request=r;p.abi=1;p.size=sizeof(p);p.kernel_id=1;p.shared_bytes=uint32_t(shared);p.workspace_bytes=workspace;
  return 0;
}
auto arguments(const FerrumCutlassNvfp4PlanV1& p,const FerrumCutlassNvfp4BindingsV1& b) {
  return ferrum_nvfp4::arguments(p.request,reinterpret_cast<const ferrum_nvfp4::Input*>(b.input.address),
      reinterpret_cast<const ferrum_nvfp4::Input*>(b.weight.address),
      reinterpret_cast<const ferrum_nvfp4::Scale*>(b.input_scales.address),
      reinterpret_cast<const ferrum_nvfp4::Scale*>(b.weight_scales.address),
      reinterpret_cast<const float*>(b.alpha.address),reinterpret_cast<ferrum_nvfp4::Output*>(b.output.address));
}
int validate(const FerrumCutlassNvfp4PlanV1& p,const FerrumCutlassNvfp4BindingsV1& b) {
  FerrumCutlassNvfp4PlanV1 expected{};int status=derive(p.request,expected);if(status)return status;
  if(std::memcmp(&p,&expected,sizeof(p)))return -5;
  const auto& r=p.request;
  if(!span(b.input,uint64_t(r.rows)*r.inputs/2,16)||!span(b.weight,uint64_t(r.outputs)*r.inputs/2,16)||
      !span(b.input_scales,scales(r.rows,r.inputs),16)||!span(b.weight_scales,scales(r.outputs,r.inputs),16)||
      !span(b.alpha,4,4)||!span(b.output,uint64_t(r.rows)*r.outputs*2,16)||!span(b.workspace,p.workspace_bytes,256))return -4;
  const FerrumCutlassNvfp4SpanV1 reads[]={b.input,b.weight,b.input_scales,b.weight_scales,b.alpha};
  for(const auto& read:reads)if(!disjoint(b.output,read)||!disjoint(b.workspace,read))return -4;
  if(!disjoint(b.output,b.workspace))return -4;
  return ferrum_nvfp4::Gemm::can_implement(arguments(p,b))==cutlass::Status::kSuccess?0:-6;
}
int run(const FerrumCutlassNvfp4PlanV1& p,const FerrumCutlassNvfp4BindingsV1& b,void* raw_stream) {
  int status=validate(p,b);if(status)return status;
  cudaStream_t launch_stream;
  status=stream(raw_stream,launch_stream);if(status)return status;
  ferrum_nvfp4::Gemm gemm;
  auto args=arguments(p,b);
  void* workspace=reinterpret_cast<void*>(b.workspace.address);
  // This overload initializes the caller-owned workspace exactly once.
  auto result=gemm.run(args,workspace,launch_stream,nullptr,false);if(result!=cutlass::Status::kSuccess)return -7;
  auto error=cudaPeekAtLastError();return error==cudaSuccess?0:int(error);
}
}
extern "C" int ferrum_cutlass_nvfp4_plan_v1(const FerrumCutlassNvfp4RequestV1* r,FerrumCutlassNvfp4PlanV1* p) noexcept {
  if(!r||!p)return -1;
  try{FerrumCutlassNvfp4PlanV1 result{};int status=derive(*r,result);if(!status)*p=result;return status;}catch(...){return -8;}
}
extern "C" int ferrum_cutlass_nvfp4_validate_v1(const FerrumCutlassNvfp4PlanV1* p,const FerrumCutlassNvfp4BindingsV1* b) noexcept {
  if(!p||!b)return -1;
  try{return validate(*p,*b);}catch(...){return -8;}
}
extern "C" int ferrum_cutlass_nvfp4_run_v1(const FerrumCutlassNvfp4PlanV1* p,const FerrumCutlassNvfp4BindingsV1* b,void* stream) noexcept {
  if(!p||!b)return -1;
  try{return run(*p,*b,stream);}catch(...){return -8;}
}
