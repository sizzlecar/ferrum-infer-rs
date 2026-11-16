#pragma once
#include "abi.h"
#include <cuda_runtime.h>
#include <limits>

namespace ferrum_nvfp4_boundary {
inline bool extent(uint64_t a, uint64_t b, uint64_t& out) {
  if (!a || !b || a > uint64_t(INT64_MAX) / b) return false;
  out = a * b;
  return true;
}
inline uint64_t scales(uint32_t rows, uint32_t inputs) {
  return ((uint64_t(rows) + 127) / 128 * 128) * ((uint64_t(inputs) / 16 + 3) / 4 * 4);
}
inline bool span(FerrumCutlassNvfp4SpanV1 s, uint64_t bytes, uint64_t align) {
  if (!bytes) return !s.address && !s.bytes;
  return s.address && s.address % align == 0 && s.bytes == bytes && s.address <= UINT64_MAX - s.bytes;
}
inline bool disjoint(FerrumCutlassNvfp4SpanV1 a, FerrumCutlassNvfp4SpanV1 b) {
  return !a.bytes || !b.bytes || a.address >= b.address + b.bytes || b.address >= a.address + a.bytes;
}
inline int device(uint32_t ordinal, uint32_t architecture, uint32_t multiprocessors, uint64_t shared_limit) {
  int current = -1, major = 0, minor = 0, sm = 0, shared = 0;
  auto status = cudaGetDevice(&current);
  if (status != cudaSuccess) return int(status);
  if (current != int(ordinal)) return -2;
  const cudaDeviceAttr attrs[] = {cudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMinor,
      cudaDevAttrMultiProcessorCount, cudaDevAttrMaxSharedMemoryPerBlockOptin};
  int* values[] = {&major, &minor, &sm, &shared};
  for (int i = 0; i < (shared_limit ? 4 : 3); ++i) {
    status = cudaDeviceGetAttribute(values[i], attrs[i], current);
    if (status != cudaSuccess) return int(status);
  }
  return major * 100 + minor * 10 == int(architecture) && sm == int(multiprocessors) &&
      shared >= 0 && shared_limit <= uint64_t(shared) ? 0 : -2;
}
inline int stream(void* raw, cudaStream_t& result) {
  if (!raw || raw == reinterpret_cast<void*>(cudaStreamLegacy) || raw == reinterpret_cast<void*>(cudaStreamPerThread)) return -1;
  result = reinterpret_cast<cudaStream_t>(raw);
  cudaStreamCaptureStatus capture;
  auto status = cudaStreamIsCapturing(result, &capture);
  if (status != cudaSuccess) return int(status);
  return capture == cudaStreamCaptureStatusNone ? 0 : -2;
}
} // namespace ferrum_nvfp4_boundary
