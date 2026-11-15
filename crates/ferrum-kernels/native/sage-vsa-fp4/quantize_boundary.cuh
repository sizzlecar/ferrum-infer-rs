#pragma once
#include "quantize_abi.h"
#include "quantize_core.cuh"
#include "boundary_checks.cuh"
#include <cstring>

namespace ferrum_sage_vsa_quantization {
inline int quantize_request(const FerrumSageVsaFp4QuantizeRequestV1& r) {
  if (r.abi != 1 || r.size != sizeof(r) || r.reserved[0] || r.reserved[1] ||
      r.device_ordinal > INT32_MAX) return -1;
  if (r.profile != 1 || r.architecture != 1200 || r.head_dim != 128 ||
      !r.batches || !r.heads || !r.query_rows || !r.key_rows ||
      r.query_rows % 128 || r.key_rows % 128) return -2;
  // The unchanged kernel computes all strides and element offsets as int32.
  // This also bounds grid.x and every K permutation/transpose intermediate.
  for (uint32_t rows : {r.query_rows,r.key_rows}) {
    uint64_t elements;
    if (!ferrum_sage_vsa::extent({r.batches,rows,r.heads,128},elements) ||
        elements > INT32_MAX) return -3;
  }
  const uint64_t width = uint64_t(r.heads) * 128;
  const uint64_t expected[][3] = {{uint64_t(r.query_rows)*width,128,width},
      {uint64_t(r.key_rows)*width,128,width},{uint64_t(r.key_rows)*width,128,width}};
  const uint64_t* actual[] = {r.query_strides,r.key_strides,r.value_strides};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) if (actual[i][j] != expected[i][j]) return -2;
  return 0;
}
inline int quantize_device(const FerrumSageVsaFp4QuantizeRequestV1& r) {
  int current = -1;
  auto error = cudaGetDevice(&current);
  if (error != cudaSuccess) return int(error);
  if (current != int(r.device_ordinal)) return -2;
  const cudaDeviceAttr attrs[] = {cudaDevAttrComputeCapabilityMajor,cudaDevAttrComputeCapabilityMinor,
      cudaDevAttrMaxThreadsPerBlock,cudaDevAttrMaxBlockDimX,cudaDevAttrMaxGridDimX,
      cudaDevAttrMaxGridDimY,cudaDevAttrMaxGridDimZ,cudaDevAttrMaxSharedMemoryPerBlock};
  int values[8]{};
  for (int i = 0; i < 8; ++i) {
    error = cudaDeviceGetAttribute(&values[i],attrs[i],current);
    if (error != cudaSuccess) return int(error);
  }
  return values[0] * 100 + values[1] * 10 == int(r.architecture) &&
      values[2] >= 1024 && values[3] >= 1024 &&
      values[4] >= int(r.query_rows/128) && values[4] >= int(r.key_rows/128) &&
      values[5] >= int(r.batches) && values[6] >= int(r.heads) &&
      values[7] >= 128*128*int(sizeof(nv_bfloat16)) ? 0 : -2;
}
inline int quantize_derive(const FerrumSageVsaFp4QuantizeRequestV1& r,
    FerrumSageVsaFp4QuantizePlanV1& p) {
  int status = quantize_request(r); if (status) return status;
  status = quantize_device(r); if (status) return status;
  // Verify the compiled functions, not just the device-wide thread limit.
  // These queries enqueue nothing and do not change launch configuration.
  const auto inspect = [](auto kernel, size_t shared) -> int {
    cudaFuncAttributes attrs{};
    auto error = cudaFuncGetAttributes(&attrs,kernel);
    if (error != cudaSuccess) return int(error);
    if (attrs.maxThreadsPerBlock < 1024 || attrs.sharedSizeBytes != shared) return -5;
    int active_blocks = 0;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,kernel,1024,0);
    if (error != cudaSuccess) return int(error);
    return active_blocks > 0 ? 0 : -2;
  };
  status = inspect(scaled_fp4_quant_kernel<128,128,false,nv_bfloat16>,0);
  if (status) return status;
  status = inspect(scaled_fp4_quant_kernel<128,128,true,nv_bfloat16>,0);
  if (status) return status;
  status = inspect(scaled_fp4_quant_trans_kernel<128,128,nv_bfloat16>,128*128*sizeof(nv_bfloat16));
  if (status) return status;
  p = {}; p.request = r; p.abi = 1; p.size = sizeof(p);
  p.query_grid_x = r.query_rows / 128; p.key_grid_x = r.key_rows / 128;
  p.grid_y = r.batches; p.grid_z = r.heads; p.threads = 1024; p.launch_count = 3;
  p.value_shared_bytes = 128*128*sizeof(nv_bfloat16);
  // Request validation already bounds these products to <= INT32_MAX elements.
  const uint64_t bh = uint64_t(r.batches) * r.heads, q = r.query_rows, k = r.key_rows;
  p.bytes = {bh*q*128*2,bh*k*128*2,bh*k*128*2,
      bh*q*64,bh*k*64,bh*128*(k/2),bh*q*8,bh*k*8,bh*128*(k/16)};
  return 0;
}
inline int quantize_validate(const FerrumSageVsaFp4QuantizePlanV1& p,
    const FerrumSageVsaFp4QuantizeBindingsV1& b) {
  FerrumSageVsaFp4QuantizePlanV1 expected{};
  int status = quantize_derive(p.request,expected); if (status) return status;
  if (std::memcmp(&p,&expected,sizeof(p))) return -5;
  const auto& s = p.bytes;
  const FerrumSageVsaFp4SpanV1 spans[] = {b.query_input,b.key_input,b.value_input,
      b.query,b.key,b.value,b.query_scales,b.key_scales,b.value_scales};
  const uint64_t bytes[] = {s.query_input,s.key_input,s.value_input,
      s.query,s.key,s.value,s.query_scales,s.key_scales,s.value_scales};
  constexpr uint64_t input_align = alignof(PackedVec<nv_bfloat16>);
  const uint64_t align[] = {input_align,input_align,input_align,8,8,8,1,1,1};
  for (int i = 0; i < 9; ++i) {
    if (!ferrum_sage_vsa::span(spans[i],bytes[i],align[i])) return -4;
    status = ferrum_sage_vsa::pointer_kind(spans[i],cudaMemoryTypeDevice,p.request.device_ordinal);
    if (status) return status;
    if (i >= 3) for (int j = 0; j < i; ++j)
      if (!ferrum_sage_vsa::disjoint(spans[i],spans[j])) return -4;
  }
  return 0;
}
inline int quantize_run(const FerrumSageVsaFp4QuantizePlanV1& p,
    const FerrumSageVsaFp4QuantizeBindingsV1& b, void* raw_stream) {
  int status = quantize_validate(p,b); if (status) return status;
  cudaStream_t stream;
  status = ferrum_sage_vsa::stream_valid(raw_stream,stream); if (status) return status;
  const auto& r = p.request;
  const auto input = [](FerrumSageVsaFp4SpanV1 s) { return reinterpret_cast<const nv_bfloat16*>(uintptr_t(s.address)); };
  const auto output = [](FerrumSageVsaFp4SpanV1 s) { return reinterpret_cast<uint8_t*>(uintptr_t(s.address)); };
  const int h = int(r.heads), q = int(r.query_rows), k = int(r.key_rows);
  scaled_fp4_quant_kernel<128,128,false,nv_bfloat16>
      <<<dim3(p.query_grid_x,p.grid_y,p.grid_z),p.threads,0,stream>>>(
      input(b.query_input),output(b.query),output(b.query_scales),int(r.batches),h,q,
      int(r.query_strides[0]),int(r.query_strides[1]),int(r.query_strides[2]),
      h*q*64,q*64,64,h*q*8,q*8,8);
  auto error = cudaGetLastError(); if (error != cudaSuccess) return int(error);
  scaled_fp4_quant_kernel<128,128,true,nv_bfloat16>
      <<<dim3(p.key_grid_x,p.grid_y,p.grid_z),p.threads,0,stream>>>(
      input(b.key_input),output(b.key),output(b.key_scales),int(r.batches),h,k,
      int(r.key_strides[0]),int(r.key_strides[1]),int(r.key_strides[2]),
      h*k*64,k*64,64,h*k*8,k*8,8);
  error = cudaGetLastError(); if (error != cudaSuccess) return int(error);
  scaled_fp4_quant_trans_kernel<128,128,nv_bfloat16>
      <<<dim3(p.key_grid_x,p.grid_y,p.grid_z),p.threads,0,stream>>>(
      input(b.value_input),output(b.value),output(b.value_scales),int(r.batches),h,k,
      int(r.value_strides[0]),int(r.value_strides[1]),int(r.value_strides[2]),
      h*128*(k/2),128*(k/2),k/2,h*128*(k/16),128*(k/16),k/16);
  return int(cudaGetLastError());
}
} // namespace ferrum_sage_vsa_quantization

extern "C" int ferrum_sage_vsa_fp4_quantize_plan_v1(const FerrumSageVsaFp4QuantizeRequestV1* r,
    FerrumSageVsaFp4QuantizePlanV1* p) noexcept {
  if (!r || !p) return -1;
  try { FerrumSageVsaFp4QuantizePlanV1 value{}; const int s = ferrum_sage_vsa_quantization::quantize_derive(*r,value); if (!s) *p = value; return s; }
  catch (...) { return -8; }
}
extern "C" int ferrum_sage_vsa_fp4_quantize_validate_v1(const FerrumSageVsaFp4QuantizePlanV1* p,
    const FerrumSageVsaFp4QuantizeBindingsV1* b) noexcept {
  if (!p || !b) return -1;
  try { return ferrum_sage_vsa_quantization::quantize_validate(*p,*b); } catch (...) { return -8; }
}
extern "C" int ferrum_sage_vsa_fp4_quantize_run_v1(const FerrumSageVsaFp4QuantizePlanV1* p,
    const FerrumSageVsaFp4QuantizeBindingsV1* b, void* stream) noexcept {
  if (!p || !b) return -1;
  try { return ferrum_sage_vsa_quantization::quantize_run(*p,*b,stream); } catch (...) { return -8; }
}
