#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cute/tensor.hpp"

namespace ferrum_fp8 {
using Tile = cute::Shape<cute::_128, cute::_128, cute::_128>;
using Cluster = cute::Shape<cute::_1, cute::_1, cute::_1>;
using Input = cutlass::float_e4m3_t;
using Output = cutlass::bfloat16_t;
using RowScale = cutlass::epilogue::fusion::Sm90ColBroadcast<0, Tile, float, float,
    cute::Stride<cute::_1, cute::_0, cute::_0>, 4, false>;
using WeightScale = cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>;
using MultiplyWeight = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies,
    float, float, cutlass::FloatRoundStyle::round_to_nearest>;
using ScaledAccumulator = cutlass::epilogue::fusion::Sm90EVT<MultiplyWeight,
    WeightScale, cutlass::epilogue::fusion::Sm90AccFetch>;
using MultiplyRow = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies,
    Output, float, cutlass::FloatRoundStyle::round_to_nearest>;
using Final = cutlass::epilogue::fusion::Sm90EVT<MultiplyRow, RowScale, ScaledAccumulator>;

using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, Tile, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    void, cutlass::layout::RowMajor, 8, Output, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Final>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    Input, cutlass::layout::RowMajor, 16,
    Input, cutlass::layout::ColumnMajor, 16, float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
// Explicit persistent tiled scheduling, not split-K/Stream-K. Every output
// tile has one reduction owner; no nondeterministic atomic output reduction.
using Kernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int,int>,
    Mainloop, Epilogue, cutlass::gemm::PersistentScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

inline Kernel::Arguments arguments(const FerrumCutlassFp8RequestV1& r,
    const Input* a, const Input* b, const float* as, const float* ws, Output* y) {
  const auto shape = cute::make_shape(int(r.rows), int(r.outputs), int(r.inputs), 1);
  typename Kernel::StrideA stride_a = cute::make_stride(int64_t(r.inputs), cute::_1{}, cute::_0{});
  typename Kernel::StrideB stride_b = cute::make_stride(int64_t(r.inputs), cute::_1{}, cute::_0{});
  typename Kernel::StrideC stride_c = cute::make_stride(int64_t(r.outputs), cute::_1{}, cute::_0{});
  typename Kernel::StrideD stride_d = cute::make_stride(int64_t(r.outputs), cute::_1{}, cute::_0{});
  WeightScale::Arguments ws_args{};
  ws_args.scalar_ptrs[0] = ws;
  ScaledAccumulator::Arguments scaled_args{ws_args, {}, {}};
  Final::Arguments final_args{RowScale::Arguments{as, 0.0f, {}}, scaled_args, {}};
  typename Kernel::MainloopArguments main{a, stride_a, b, stride_b};
  typename Kernel::EpilogueArguments epi{final_args, nullptr, stride_c, y, stride_d};
  cutlass::KernelHardwareInfo hardware{};
  hardware.device_id = int(r.device_ordinal);
  hardware.sm_count = int(r.multiprocessors);
  return {cutlass::gemm::GemmUniversalMode::kGemm, shape, main, epi, hardware, {}};
}
} // namespace ferrum_fp8
