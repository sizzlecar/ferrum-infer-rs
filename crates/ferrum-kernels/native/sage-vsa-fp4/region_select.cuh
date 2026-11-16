#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace ferrum_sage_vsa {
// Integer comparison makes the specified NaN/subnormal/signed-zero behavior
// independent of the original attention TU's --use_fast_math setting.
__device__ __forceinline__ uint32_t selection_key(uint32_t bits) {
  const uint32_t magnitude = bits & 0x7fffffffu;
  // Torch 7661cd9c6b84 SortingRadixSelect.cuh TopKTypeConfig<float>:
  // NaNs rank above +inf; preserve the distinct -0/+0 radix keys.
  if (magnitude > 0x7f800000u) return 0xffffffffu;
  return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
}

// One CTA owns one complete [batch,head,video-query-block] row. Rank tests
// are quadratic within each region, intentionally simple for first correctness
// qualification; no performance claim or hidden approximate selection.
__global__ void region_select(const uint32_t* scores, const int32_t* regions,
    int32_t* indices, int32_t* counts, int keys, int queries, int region_count) {
  const uint64_t row = blockIdx.x;
  const int query = int(row % uint64_t(queries)) + keys - queries;
  const uint64_t bh = row / uint64_t(queries);
  const uint32_t* values = scores + (bh * keys + query) * keys;
  int32_t* output = indices + row * keys;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ int offsets[4];
  __shared__ int carried;
  for (int key = tid; key < keys; key += 128) output[key] = 0;
  if (tid == 0) carried = 0;
  __syncthreads();
  for (int base = 0; base < keys; base += 128) {
    const int key = keys - 1 - base - tid;
    bool keep = key >= 0 && key < keys - queries;
    if (key >= keys - queries) {
      for (int r = 0; r < region_count; ++r) {
        const int start = regions[3*r], end = regions[3*r+1], budget = regions[3*r+2];
        if (key < start || key >= end) continue;
        if (budget == end - start) { keep = true; break; }
        const uint32_t value = selection_key(values[key]);
        int rank = 0;
        for (int other = start; other < end; ++other) {
          const uint32_t peer = selection_key(values[other]);
          rank += peer > value || (peer == value && other < key);
        }
        keep = rank < budget;
        break;
      }
    }
    const unsigned mask = __ballot_sync(0xffffffffu, keep);
    if (lane == 0) offsets[warp] = __popc(mask);
    __syncthreads();
    if (tid == 0) {
      int end = carried;
      for (int w = 0; w < 4; ++w) { const int n = offsets[w]; offsets[w] = end; end += n; }
      carried = end;
    }
    __syncthreads();
    if (keep) output[offsets[warp] + __popc(mask & ((1u << lane) - 1u))] = key;
    __syncthreads();
  }
  if (tid == 0) counts[row] = carried;
}
} // namespace ferrum_sage_vsa
