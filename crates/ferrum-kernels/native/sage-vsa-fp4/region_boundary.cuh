#pragma once
#include "region_abi.h"
#include "region_select.cuh"

namespace {
int region_derive(const FerrumSageVsaFp4RegionRequestV1& r, FerrumSageVsaFp4RegionPlanV1& p) {
  p = {};
  int status = derive(r.attention,p.attention); if (status) return status;
  const auto& a = r.attention;
  if (r.selection_profile != 1 || !r.region_count || r.region_count > a.query_blocks ||
      a.query_blocks >= a.key_blocks) return -2;
  p.region_count = r.region_count; p.selection_profile = r.selection_profile;
  p.selection_threads = 128;
  p.upload_count = 2; p.memset_count = 1;
  p.selection_grid_blocks = uint32_t(uint64_t(a.batches) * a.heads * a.query_blocks);
  if (!ferrum_sage_vsa::extent({a.batches,a.heads,a.key_blocks,a.key_blocks,4},p.scores_bytes)) return -3;
  p.regions_bytes = uint64_t(r.region_count) * 12;
  return 0;
}
int region_validate(const FerrumSageVsaFp4RegionPlanV1& p,
    const FerrumSageVsaFp4RegionBindingsV1& b, const FerrumSageVsaFp4RegionHostV1& h,
    bool pinned = true) {
  FerrumSageVsaFp4RegionPlanV1 expected{};
  const FerrumSageVsaFp4RegionRequestV1 request{p.attention.request,p.region_count,p.selection_profile};
  int status = region_derive(request,expected); if (status) return status;
  if (std::memcmp(&p,&expected,sizeof(p))) return -5;
  status = validate_bindings(p.attention,b.attention); if (status) return status;
  const auto& a = b.attention;
  const auto& r = p.attention.request;
  if (!span(b.scores,p.scores_bytes,4) || !span(b.regions,p.regions_bytes,4) ||
      !span(h.regions,p.regions_bytes,4) || !span(h.half_valid,p.attention.bytes.half_valid,4)) return -4;
  for (auto device : {b.scores,b.regions}) {
    status = pointer_kind(device,cudaMemoryTypeDevice,r.device_ordinal); if (status) return status;
  }
  const FerrumSageVsaFp4SpanV1 reads[] = {a.query,a.key,a.value,a.query_scales,a.key_scales,a.value_scales,b.scores};
  const FerrumSageVsaFp4SpanV1 writes[] = {a.output,a.lse,a.zero_delta,a.indices,a.counts,a.half_valid,b.regions};
  for (int i = 0; i < 7; ++i) {
    for (auto read : reads) if (!disjoint(writes[i],read)) return -4;
    for (int j = 0; j < i; ++j) if (!disjoint(writes[i],writes[j])) return -4;
  }
  for (auto host : {h.regions,h.half_valid}) {
    if (pinned) {
      status = pointer_kind(host,cudaMemoryTypeHost,r.device_ordinal); if (status) return status;
    } else {
      cudaPointerAttributes attrs{};
      const auto error = cudaPointerGetAttributes(&attrs,pointer(host));
      if (error != cudaSuccess) return int(error);
      if (attrs.type != cudaMemoryTypeHost && attrs.type != cudaMemoryTypeUnregistered) return -4;
    }
    for (auto write : writes) if (!disjoint(host,write)) return -4;
  }
  // Raw caller owes readable immutable Host storage for the complete checked
  // extents. The pinned entry retains it through DMA; the separate kernel-
  // argument entry copies its values during Run before returning.
  const auto regions = reinterpret_cast<const int32_t*>(uintptr_t(h.regions.address));
  const auto half = reinterpret_cast<const int32_t*>(uintptr_t(h.half_valid.address));
  int end = int(r.key_blocks - r.query_blocks);
  for (uint32_t i = 0; i < p.region_count; ++i) {
    const int start = regions[3*i], next = regions[3*i+1], keep = regions[3*i+2];
    if (start != end || next <= start || next > int(r.key_blocks) || keep < 1 || keep > next - start) return -6;
    end = next;
  }
  if (end != int(r.key_blocks)) return -6;
  for (uint32_t i = 0; i < r.key_blocks; ++i) {
    const int x = half[2*i], y = half[2*i+1];
    if (x < 1 || x > 64 || y < 0 || y > 64 || (x < 64 && y)) return -6;
  }
  return 0;
}
int region_selection_and_attention(const FerrumSageVsaFp4RegionPlanV1& p,
    const FerrumSageVsaFp4RegionBindingsV1& b, cudaStream_t stream) {
  auto error = cudaMemsetAsync(pointer(b.attention.zero_delta),0,b.attention.zero_delta.bytes,stream);
  if (error != cudaSuccess) return int(error);
  const auto& r = p.attention.request;
  ferrum_sage_vsa::region_select<<<p.selection_grid_blocks,p.selection_threads,0,stream>>>(
      reinterpret_cast<const uint32_t*>(uintptr_t(b.scores.address)),
      reinterpret_cast<const int32_t*>(uintptr_t(b.regions.address)),
      reinterpret_cast<int32_t*>(uintptr_t(b.attention.indices.address)),
      reinterpret_cast<int32_t*>(uintptr_t(b.attention.counts.address)),
      int(r.key_blocks),int(r.query_blocks),int(p.region_count));
  error = cudaGetLastError(); if (error != cudaSuccess) return int(error);
  auto params = parameters(r,b.attention);
  return launch(params,r.multiprocessors,stream);
}
int region_run(const FerrumSageVsaFp4RegionPlanV1& p,
    const FerrumSageVsaFp4RegionBindingsV1& b, const FerrumSageVsaFp4RegionHostV1& h,
    void* raw_stream) {
  int status = region_validate(p,b,h); if (status) return status;
  cudaStream_t stream;
  status = stream_valid(raw_stream,stream); if (status) return status;
  auto error = cudaMemcpyAsync(pointer(b.regions),pointer(h.regions),h.regions.bytes,cudaMemcpyHostToDevice,stream);
  if (error != cudaSuccess) return int(error);
  error = cudaMemcpyAsync(pointer(b.attention.half_valid),pointer(h.half_valid),h.half_valid.bytes,cudaMemcpyHostToDevice,stream);
  if (error != cudaSuccess) return int(error);
  return region_selection_and_attention(p,b,stream);
}
}
extern "C" int ferrum_sage_vsa_fp4_region_plan_v1(const FerrumSageVsaFp4RegionRequestV1* r,
    FerrumSageVsaFp4RegionPlanV1* p) noexcept {
  if (!r || !p) return -1;
  try { FerrumSageVsaFp4RegionPlanV1 value{}; const int s = region_derive(*r,value); if (!s) *p = value; return s; }
  catch (...) { return -8; }
}
extern "C" int ferrum_sage_vsa_fp4_region_validate_v1(const FerrumSageVsaFp4RegionPlanV1* p,
    const FerrumSageVsaFp4RegionBindingsV1* b, const FerrumSageVsaFp4RegionHostV1* h) noexcept {
  if (!p || !b || !h) return -1;
  try { return region_validate(*p,*b,*h); } catch (...) { return -8; }
}
extern "C" int ferrum_sage_vsa_fp4_region_run_v1(const FerrumSageVsaFp4RegionPlanV1* p,
    const FerrumSageVsaFp4RegionBindingsV1* b, const FerrumSageVsaFp4RegionHostV1* h, void* stream) noexcept {
  if (!p || !b || !h) return -1;
  try { return region_run(*p,*b,*h,stream); } catch (...) { return -8; }
}
