#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include <cuda_bf16.h>

namespace ferrum_fp8 {
using Input = cutlass::float_e4m3_t;
using Output = cutlass::bfloat16_t;
// Ada has FP8 MMA but no SM120 collective/TMA epilogue. Keep the FP32
// accumulator in caller-owned scratch, then apply the exact row/scalar scales.
// No split-K, atomic reduction, hidden allocation or BF16 GEMM fallback.
// Group nearby N tiles so they can reuse the same activation tile in cache.
// Identity8 changes CTA placement only; warp geometry, K128 promotion and
// the separate FP32 scaling epilogue remain unchanged.
using Gemm = cutlass::gemm::device::GemmUniversal<
    Input, cutlass::layout::RowMajor, Input, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 64, 128>,
    cutlass::gemm::GemmShape<64, 32, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 3,
    16, 16, cutlass::arch::OpMultiplyAdd>;

inline Gemm::Arguments arguments(const FerrumCutlassFp8RequestV1& r,
    const Input* a, const Input* b, float* accumulator) {
  return {cutlass::gemm::GemmUniversalMode::kGemm,
    {int(r.rows), int(r.outputs), int(r.inputs)}, 1, {1.0f, 0.0f},
    a, b, accumulator, accumulator,
    int64_t(0), int64_t(0), int64_t(0), int64_t(0),
    int64_t(r.inputs), int64_t(r.inputs), int64_t(r.outputs), int64_t(r.outputs)};
}

inline uint64_t accumulator_bytes(const FerrumCutlassFp8RequestV1& r) {
  return (uint64_t(r.rows) * r.outputs * sizeof(float) + 255) & ~uint64_t(255);
}

__global__ void scale_output(const float* accumulator, const float* row_scale,
    const float* weight_scale, __nv_bfloat16* output, uint64_t elements, uint32_t columns) {
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < elements; i += uint64_t(blockDim.x) * gridDim.x) {
    const float weighted = __fmul_rn(*weight_scale, accumulator[i]);
    output[i] = __float2bfloat16_rn(__fmul_rn(weighted, row_scale[i / columns]));
  }
}
} // namespace ferrum_fp8
