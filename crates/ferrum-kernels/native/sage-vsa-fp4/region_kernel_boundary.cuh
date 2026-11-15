#pragma once
#include "region_boundary.cuh"

// Separate transport ABI: the numerical geometry is unchanged. The original
// region entry continues to require pinned Host storage for asynchronous DMA.
struct FerrumSageVsaFp4RegionKernelPlanV1 {
  FerrumSageVsaFp4RegionPlanV1 geometry;
  uint32_t upload_kernel_count, reserved;
};
static_assert(sizeof(FerrumSageVsaFp4RegionKernelPlanV1) == 240, "kernel geometry plan ABI drift");

// A named namespace also keeps NVCC's generated kernel stubs unambiguous
// beside the imported CuTe anonymous namespaces.
namespace ferrum_sage_vsa_geometry_upload {
constexpr uint64_t region_packet_words = 512;
struct RegionGeometryPacket {
  uint64_t destination;
  uint32_t words;
  int32_t values[region_packet_words];
};
// Below even CUDA's older 4096-byte kernel parameter limit. Every packet is
// passed by value; no asynchronous command retains a pageable Host pointer.
static_assert(sizeof(RegionGeometryPacket) <= 4096, "geometry packet exceeds kernel argument limit");
__global__ void region_upload_packet(const __grid_constant__ RegionGeometryPacket packet) {
  auto destination = reinterpret_cast<int32_t*>(uintptr_t(packet.destination));
  for (uint32_t i = threadIdx.x; i < packet.words; i += blockDim.x) destination[i] = packet.values[i];
}
uint64_t region_packets(uint64_t bytes) {
  return (bytes + region_packet_words * 4 - 1) / (region_packet_words * 4);
}
int region_kernel_derive(const FerrumSageVsaFp4RegionRequestV1& r,
    FerrumSageVsaFp4RegionKernelPlanV1& p) {
  p = {};
  int status = region_derive(r,p.geometry); if (status) return status;
  const uint64_t n = region_packets(p.geometry.regions_bytes) +
      region_packets(p.geometry.attention.bytes.half_valid);
  if (n > UINT32_MAX) return -3;
  p.upload_kernel_count = uint32_t(n);
  return 0;
}
int region_kernel_validate(const FerrumSageVsaFp4RegionKernelPlanV1& p,
    const FerrumSageVsaFp4RegionBindingsV1& b, const FerrumSageVsaFp4RegionHostV1& h) {
  FerrumSageVsaFp4RegionKernelPlanV1 expected{};
  const auto& g = p.geometry;
  int status = region_kernel_derive({g.attention.request,g.region_count,g.selection_profile},expected);
  if (status) return status;
  if (std::memcmp(&p,&expected,sizeof(p))) return -5;
  return region_validate(g,b,h,false);
}
int region_upload_geometry(FerrumSageVsaFp4SpanV1 destination,
    FerrumSageVsaFp4SpanV1 source, cudaStream_t stream) {
  for (uint64_t offset = 0; offset < source.bytes; offset += region_packet_words * 4) {
    RegionGeometryPacket packet{};
    packet.destination = destination.address + offset;
    const uint64_t remaining = (source.bytes - offset) / 4;
    packet.words = uint32_t(remaining < region_packet_words ? remaining : region_packet_words);
    std::memcpy(packet.values,reinterpret_cast<const void*>(uintptr_t(source.address + offset)),packet.words * 4);
    region_upload_packet<<<1,128,0,stream>>>(packet);
    const auto error = cudaGetLastError(); if (error != cudaSuccess) return int(error);
  }
  return 0;
}
int region_kernel_run(const FerrumSageVsaFp4RegionKernelPlanV1& p,
    const FerrumSageVsaFp4RegionBindingsV1& b, const FerrumSageVsaFp4RegionHostV1& h,
    void* raw_stream) {
  int status = region_kernel_validate(p,b,h); if (status) return status;
  cudaStream_t stream;
  status = stream_valid(raw_stream,stream); if (status) return status;
  status = region_upload_geometry(b.regions,h.regions,stream); if (status) return status;
  status = region_upload_geometry(b.attention.half_valid,h.half_valid,stream); if (status) return status;
  return region_selection_and_attention(p.geometry,b,stream);
}
}
extern "C" int ferrum_sage_vsa_fp4_region_kernel_plan_v1(const FerrumSageVsaFp4RegionRequestV1* r,
    FerrumSageVsaFp4RegionKernelPlanV1* p) noexcept {
  if (!r || !p) return -1;
  try { FerrumSageVsaFp4RegionKernelPlanV1 value{}; const int s = ferrum_sage_vsa_geometry_upload::region_kernel_derive(*r,value); if (!s) *p = value; return s; }
  catch (...) { return -8; }
}
extern "C" int ferrum_sage_vsa_fp4_region_kernel_validate_v1(const FerrumSageVsaFp4RegionKernelPlanV1* p,
    const FerrumSageVsaFp4RegionBindingsV1* b, const FerrumSageVsaFp4RegionHostV1* h) noexcept {
  if (!p || !b || !h) return -1;
  try { return ferrum_sage_vsa_geometry_upload::region_kernel_validate(*p,*b,*h); } catch (...) { return -8; }
}
extern "C" int ferrum_sage_vsa_fp4_region_kernel_run_v1(const FerrumSageVsaFp4RegionKernelPlanV1* p,
    const FerrumSageVsaFp4RegionBindingsV1* b, const FerrumSageVsaFp4RegionHostV1* h, void* stream) noexcept {
  if (!p || !b || !h) return -1;
  try { return ferrum_sage_vsa_geometry_upload::region_kernel_run(*p,*b,*h,stream); } catch (...) { return -8; }
}
