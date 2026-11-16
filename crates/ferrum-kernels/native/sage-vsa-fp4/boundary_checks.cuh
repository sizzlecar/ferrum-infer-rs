#pragma once
#include "abi.h"
#include <cuda_runtime.h>
#include <initializer_list>
#include <limits>

namespace ferrum_sage_vsa {
inline bool extent(std::initializer_list<uint64_t> factors, uint64_t& out) {
  out = 1;
  for (auto n : factors) {
    if (!n || out > uint64_t(INT64_MAX) / n) return false;
    out *= n;
  }
  return true;
}
inline int sizes(const FerrumSageVsaFp4RequestV1& r, FerrumSageVsaFp4BufferSizesV1& s) {
  const uint64_t b = r.batches, h = r.heads, q = uint64_t(r.query_blocks) * 128,
      k = uint64_t(r.key_blocks) * 128;
  return extent({b,h,q,64},s.query) && extent({b,h,k,64},s.key) && extent({b,h,128,k/2},s.value) &&
      extent({b,h,q,8},s.query_scales) && extent({b,h,k,8},s.key_scales) && extent({b,h,128,k/16},s.value_scales) &&
      extent({b,h,q,128,2},s.output) && extent({b,h,q,4},s.lse) && extent({b,h,k,4},s.zero_delta) &&
      extent({b,h,r.query_blocks,r.key_blocks,4},s.indices) && extent({b,h,r.query_blocks,4},s.counts) &&
      extent({r.key_blocks,2,4},s.half_valid) ? 0 : -3;
}
inline int request_valid(const FerrumSageVsaFp4RequestV1& r) {
  const uint32_t positive[] = {r.batches,r.heads,r.query_blocks,r.key_blocks,r.multiprocessors};
  if (r.abi != 1 || r.size != sizeof(r) || r.reserved || r.device_ordinal > INT32_MAX ||
      !r.maximum_dynamic_shared_bytes || r.maximum_dynamic_shared_bytes > UINT32_MAX) return -1;
  for (auto n : positive) if (!n || n > INT32_MAX) return -1;
  // Check IEEE bits without fast-math finite assumptions; exclude -0 and NaN.
  if (r.softmax_scale_bits == 0 || (r.softmax_scale_bits & 0x80000000u) ||
      (r.softmax_scale_bits & 0x7f800000u) == 0x7f800000u) return -1;
  if (r.architecture != 1200 || r.head_dim != 128 || r.tile_rows != 128 ||
      r.profile != 1 || r.query_blocks > r.key_blocks) return -2;
  uint64_t work;
  if (uint64_t(r.query_blocks) * 128 > INT32_MAX || uint64_t(r.key_blocks) * 128 > INT32_MAX ||
      uint64_t(r.heads) * r.query_blocks * 128 > INT32_MAX ||
      !extent({r.batches,r.heads,r.query_blocks},work) ||
      work > uint64_t(INT32_MAX) - r.multiprocessors + 1) return -3;
  // CuTe's unchanged LayoutSF/LayoutDS shape/stride intermediates are int32.
  if (!extent({r.batches,r.heads,r.key_blocks,128,128},work) || work > INT32_MAX) return -3;
  FerrumSageVsaFp4BufferSizesV1 ignored{};
  return sizes(r, ignored);
}
inline int device_valid(const FerrumSageVsaFp4RequestV1& r) {
  int current = -1, major = 0, minor = 0, sms = 0, shared = 0;
  auto status = cudaGetDevice(&current);
  if (status != cudaSuccess) return int(status);
  if (current != int(r.device_ordinal)) return -2;
  const cudaDeviceAttr attrs[] = {cudaDevAttrComputeCapabilityMajor,cudaDevAttrComputeCapabilityMinor,
      cudaDevAttrMultiProcessorCount,cudaDevAttrMaxSharedMemoryPerBlockOptin};
  int* values[] = {&major,&minor,&sms,&shared};
  for (int i = 0; i < 4; ++i) {
    status = cudaDeviceGetAttribute(values[i], attrs[i], current);
    if (status != cudaSuccess) return int(status);
  }
  return major * 100 + minor * 10 == int(r.architecture) && sms == int(r.multiprocessors) &&
      shared > 0 && r.maximum_dynamic_shared_bytes <= uint64_t(shared) ? 0 : -2;
}
inline bool span(FerrumSageVsaFp4SpanV1 s, uint64_t bytes, uint64_t align) {
  return bytes && s.address && s.address % align == 0 && s.bytes == bytes &&
      s.address <= uint64_t(UINTPTR_MAX) - s.bytes;
}
inline bool disjoint(FerrumSageVsaFp4SpanV1 a, FerrumSageVsaFp4SpanV1 b) {
  return a.address >= b.address + b.bytes || b.address >= a.address + a.bytes;
}
inline int pointer_kind(FerrumSageVsaFp4SpanV1 s, cudaMemoryType kind, int device) {
  cudaPointerAttributes attrs{};
  const auto status = cudaPointerGetAttributes(&attrs, reinterpret_cast<const void*>(uintptr_t(s.address)));
  if (status != cudaSuccess) return int(status);
  // Managed/pageable/mapped-Host aliases are not accepted as device spans.
  return attrs.type == kind && (kind != cudaMemoryTypeDevice || attrs.device == device) ? 0 : -4;
}
inline int stream_valid(void* raw, cudaStream_t& stream) {
  if (!raw || raw == reinterpret_cast<void*>(cudaStreamLegacy) || raw == reinterpret_cast<void*>(cudaStreamPerThread)) return -1;
  stream = reinterpret_cast<cudaStream_t>(raw);
  cudaStreamCaptureStatus capture;
  const auto status = cudaStreamIsCapturing(stream, &capture);
  if (status != cudaSuccess) return int(status);
  return capture == cudaStreamCaptureStatusNone ? 0 : -2;
}
// Only call after checking pinned Host kinds and full span metadata. The raw
// caller must still supply readable immutable allocations of these extents.
inline bool list_contents(const FerrumSageVsaFp4RequestV1& r, const FerrumSageVsaFp4HostListsV1& h) {
  const auto idx = reinterpret_cast<const int32_t*>(uintptr_t(h.indices.address));
  const auto num = reinterpret_cast<const int32_t*>(uintptr_t(h.counts.address));
  const auto valid = reinterpret_cast<const int32_t*>(uintptr_t(h.half_valid.address));
  for (uint64_t n = 0; n < r.key_blocks; ++n) {
    const int a = valid[2*n], b = valid[2*n+1];
    if (a < 1 || a > 64 || b < 0 || b > 64 || (a < 64 && b)) return false;
  }
  const uint64_t rows = uint64_t(r.batches) * r.heads * r.query_blocks;
  for (uint64_t m = 0; m < rows; ++m) {
    const int count = num[m];
    if (count < 1 || uint32_t(count) > r.key_blocks) return false;
    const int32_t* row = idx + m * r.key_blocks;
    if (row[count-1] != 0) return false;
    for (int i = 0; i < count; ++i)
      if (row[i] < 0 || uint32_t(row[i]) >= r.key_blocks || (i && row[i-1] <= row[i])) return false;
  }
  return true;
}
} // namespace ferrum_sage_vsa
