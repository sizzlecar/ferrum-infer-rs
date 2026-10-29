#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

namespace ferrum_fp8 {
using Input = cutlass::float_e4m3_t;
using Output = cutlass::bfloat16_t;
using Tile = cutlass::gemm::GemmShape<128, 64, 128>;
using Warp = cutlass::gemm::GemmShape<64, 32, 128>;
namespace epi = cutlass::epilogue::threadblock;

// Keep Ada's MMA geometry, K128 staged promotion and Identity8 tile placement.
// The visitor applies weight then row scale in F32, with BF16 RNE only at the
// final store. No FP32 output scratch, split-K or extra reduction is needed.
using ThreadMap = epi::OutputTileThreadLayout<Tile, Warp, Output, 8>;
using RowScale = epi::VisitorColBroadcast<ThreadMap, float,
    cute::Stride<cute::_1, cute::_0, cute::_0>, false>;
using WeightScale = epi::VisitorScalarBroadcast<float>;
using Multiply = epi::VisitorCompute<cutlass::multiplies, float, float,
    cutlass::FloatRoundStyle::round_to_nearest>;
using Weighted = epi::Sm80EVT<Multiply, WeightScale, epi::VisitorAccFetch>;
using Scaled = epi::Sm80EVT<Multiply, Weighted, RowScale>;
using Store = epi::VisitorAuxStore<ThreadMap, Output,
    cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<int64_t, cute::_1, cute::_0>>;
using Epilogue = epi::Sm80EVT<Store, Scaled>;
using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    Input, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,
    Input, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,
    Output, cutlass::layout::RowMajor, 8, float, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    Tile, Warp, cutlass::gemm::GemmShape<16, 8, 32>, Epilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 3,
    cutlass::arch::OpMultiplyAdd>::GemmKernel;
using Gemm = cutlass::gemm::device::GemmUniversalBase<Kernel>;

inline Kernel::Arguments arguments(const FerrumCutlassFp8RequestV1& r,
    const Input* activation, const Input* weights, const float* row_scale,
    const float* weight_scale, Output* output) {
  WeightScale::Arguments ws{};
  ws.scalar_ptrs[0] = weight_scale;
  Weighted::Arguments weighted{ws, {}, {}};
  RowScale::Arguments rs{row_scale, 0.0f, {}};
  Scaled::Arguments scaled{weighted, rs, {}};
  Store::Arguments store{output, {int64_t(r.outputs), cute::_1{}, cute::_0{}}};
  Epilogue::Arguments epilogue{scaled, store};
  return {cutlass::gemm::GemmUniversalMode::kGemm,
    {int(r.rows), int(r.outputs), int(r.inputs)}, 1, epilogue,
    activation, weights, nullptr, output,
    int64_t(0), int64_t(0), int64_t(0), int64_t(0),
    int64_t(r.inputs), int64_t(r.inputs), int64_t(r.outputs), int64_t(r.outputs)};
}
} // namespace ferrum_fp8
