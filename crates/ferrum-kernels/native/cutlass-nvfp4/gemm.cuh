#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cute/tensor.hpp"

namespace ferrum_nvfp4 {
using Tile = cute::Shape<cute::_128, cute::_128, cute::_128>;
using Cluster = cute::Shape<cute::_1, cute::_1, cute::_1>;
using Packed = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using Input = cutlass::float_e2m1_t;
using Scale = cutlass::float_ue4m3_t;
using Output = cutlass::bfloat16_t;
using Fusion = cutlass::epilogue::fusion::LinearCombination<Output, float, void, float>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, Tile, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    void, cutlass::layout::RowMajor, 8, Output, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::TmaWarpSpecialized, Fusion>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    Packed, cutlass::layout::RowMajor, 32, Packed, cutlass::layout::ColumnMajor, 32,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
// One output tile/reduction owner. No split-K, atomics or data-dependent tuning.
using Kernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int,int>,
    Mainloop, Epilogue, cutlass::gemm::StaticPersistentScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

inline Kernel::Arguments arguments(const FerrumCutlassNvfp4RequestV1& r,
    const Input* a, const Input* b, const Scale* sa, const Scale* sb,
    const float* alpha, Output* y) {
  Kernel::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = cute::make_shape(int(r.rows), int(r.outputs), int(r.inputs), 1);
  args.mainloop.ptr_A = a;
  args.mainloop.ptr_B = b;
  args.mainloop.dA = cute::make_stride(int64_t(r.inputs), cute::_1{}, int64_t(0));
  args.mainloop.dB = cute::make_stride(int64_t(r.inputs), cute::_1{}, int64_t(0));
  args.mainloop.ptr_SFA = sa;
  args.mainloop.ptr_SFB = sb;
  using ScaleConfig = typename Mainloop::Sm1xxBlkScaledConfig;
  args.mainloop.layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(args.problem_shape);
  args.mainloop.layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(args.problem_shape);
  args.epilogue.thread.alpha_ptr = alpha;
  args.epilogue.ptr_C = nullptr;
  args.epilogue.ptr_D = y;
  args.epilogue.dC = cute::make_stride(int64_t(r.outputs), cute::_1{}, int64_t(0));
  args.epilogue.dD = args.epilogue.dC;
  args.hw_info.device_id = int(r.device_ordinal);
  args.hw_info.sm_count = int(r.multiprocessors);
  args.hw_info.cluster_shape = dim3(1,1,1);
  args.hw_info.cluster_shape_fallback = dim3(1,1,1);
  return args;
}
} // namespace ferrum_nvfp4
