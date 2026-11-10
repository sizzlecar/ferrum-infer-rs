// Independent INT8 GEMM output qualification. No allocation, cache or routing.
// EVT composition follows the public CUTLASS visitor API also used by the
// existing Ferrum SM89 FP8 boundary; scale order matches ConvRot exactly.
#include "abi.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

namespace {
using Output = cutlass::bfloat16_t;
using Tile = cutlass::gemm::GemmShape<128, 128, 64>;
using Warp = cutlass::gemm::GemmShape<64, 64, 64>;
namespace epi = cutlass::epilogue::threadblock;
using Map = epi::OutputTileThreadLayout<Tile, Warp, Output, 8>;
using Row = epi::VisitorColBroadcast<Map, float,
    cute::Stride<cute::_1, cute::_0, cute::_0>, false>;
using Column = epi::VisitorRowBroadcast<Map, float,
    cute::Stride<cute::_0, cute::_1, cute::_0>>;
using Multiply = epi::VisitorCompute<cutlass::multiplies, float, float,
    cutlass::FloatRoundStyle::round_to_nearest>;
using RowScaled = epi::Sm80EVT<Multiply, epi::VisitorAccFetch, Row>;
using Scaled = epi::Sm80EVT<Multiply, RowScaled, Column>;
using Store = epi::VisitorAuxStore<Map, Output,
    cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<int64_t, cute::_1, cute::_0>>;
using Epilogue = epi::Sm80EVT<Store, Scaled>;
using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,
    int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,
    Output, cutlass::layout::RowMajor, 8, int32_t, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    Tile, Warp, cutlass::gemm::GemmShape<16, 8, 32>, Epilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 3,
    cutlass::arch::OpMultiplyAddSaturate>::GemmKernel;
using Gemm = cutlass::gemm::device::GemmUniversalBase<Kernel>;

bool valid(const FerrumInt8OutputShape* s) {
  return s && s->reserved == 0 && s->rows && s->outputs && s->inputs &&
    s->rows <= INT32_MAX && s->outputs <= INT32_MAX &&
    s->inputs <= INT32_MAX / (128 * 128) &&
    s->inputs % 16 == 0 && s->outputs % 8 == 0;
}
Kernel::Arguments arguments(const FerrumInt8OutputShape& s,
    const FerrumInt8OutputBindings& b) {
  Row::Arguments rs{reinterpret_cast<const float*>(b.row_scale), 0.f, {}};
  Column::Arguments cs{reinterpret_cast<const float*>(b.channel_scale), 0.f, {}};
  RowScaled::Arguments r{{}, rs, {}};
  Scaled::Arguments scaled{r, cs, {}};
  auto* output = reinterpret_cast<Output*>(b.output);
  Store::Arguments store{output, {int64_t(s.outputs), cute::_1{}, cute::_0{}}};
  Epilogue::Arguments epilogue{scaled, store};
  return {cutlass::gemm::GemmUniversalMode::kGemm,
    {int(s.rows), int(s.outputs), int(s.inputs)}, 1, epilogue,
    reinterpret_cast<const int8_t*>(b.activation),
    reinterpret_cast<const int8_t*>(b.weight), nullptr, output,
    int64_t(0), int64_t(0), int64_t(0), int64_t(0),
    int64_t(s.inputs), int64_t(s.inputs), int64_t(s.outputs), int64_t(s.outputs)};
}
__global__ void reference_epilogue(const int32_t* acc, const float* rows,
    const float* channels, __nv_bfloat16* out, uint64_t count, uint32_t n) {
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += uint64_t(blockDim.x) * gridDim.x) {
    const float scaled = __fmul_rn(__int2float_rn(acc[i]), rows[i / n]);
    out[i] = __float2bfloat16_rn(__fmul_rn(scaled, channels[i % n]));
  }
}
__global__ void fill(uint16_t* out, uint64_t count, uint16_t bits) {
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += uint64_t(blockDim.x) * gridDim.x) out[i] = bits;
}
__global__ void compare(const uint16_t* expected, const uint16_t* actual,
    uint64_t count, FerrumInt8OutputComparison* result) {
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += uint64_t(blockDim.x) * gridDim.x) {
    const float a = __uint_as_float(uint32_t(expected[i]) << 16);
    const float b = __uint_as_float(uint32_t(actual[i]) << 16);
    if (expected[i] != actual[i]) atomicAdd(
      reinterpret_cast<unsigned long long*>(&result->mismatches), 1ULL);
    if (!isfinite(a) || !isfinite(b)) atomicAdd(
      reinterpret_cast<unsigned long long*>(&result->nonfinite), 1ULL);
    else {
      const float error = fabsf(a - b);
      // The correct case has hundreds of millions of equal outputs. Avoid
      // serializing their zero maxima on one global address.
      if (error > 0.f) atomicMax(&result->maximum_absolute_error_bits, __float_as_uint(error));
    }
  }
}
unsigned blocks(uint64_t elements) {
  return unsigned((elements + 255) / 256 < 65535 ? (elements + 255) / 256 : 65535);
}
}

extern "C" int ferrum_int8_fused_output_prepare_v1(
    const FerrumInt8OutputShape* s, FerrumInt8OutputInfo* info) {
  if (!valid(s) || !info) return -1;
  // get_workspace_size depends on geometry, not the placeholder addresses.
  FerrumInt8OutputBindings b{256, 256, 256, 256, 256};
  const auto args = arguments(*s, b);
  info->workspace_bytes = Gemm::get_workspace_size(args);
  info->tile_m = 128; info->tile_n = 128; info->tile_k = 64; info->threads = 128;
  return info->workspace_bytes == 0 ? 0 : -2;
}
extern "C" int ferrum_int8_fused_output_run_v1(
    const FerrumInt8OutputShape* s, const FerrumInt8OutputBindings* b, void* stream) {
  if (!valid(s) || !b || !b->activation || !b->weight || !b->row_scale ||
      !b->channel_scale || !b->output || b->activation % 16 || b->weight % 16 ||
      b->output % 16 || b->row_scale % 4 || b->channel_scale % 4) return -1;
  Gemm gemm;
  const auto args = arguments(*s, *b);
  if (Gemm::get_workspace_size(args) != 0) return -2;
  auto status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) return 100 + int(status);
  status = gemm.initialize(args, nullptr, static_cast<cudaStream_t>(stream));
  if (status != cutlass::Status::kSuccess) return 200 + int(status);
  status = gemm.run(static_cast<cudaStream_t>(stream));
  return status == cutlass::Status::kSuccess ? 0 : 300 + int(status);
}
extern "C" int ferrum_int8_output_reference_epilogue_v1(
    const FerrumInt8OutputShape* s, const FerrumInt8OutputBindings* b,
    uint64_t acc, void* stream) {
  if (!valid(s) || !b || !acc || !b->output || !b->row_scale || !b->channel_scale) return -1;
  const uint64_t count = uint64_t(s->rows) * s->outputs;
  reference_epilogue<<<blocks(count),256,0,static_cast<cudaStream_t>(stream)>>>(
    reinterpret_cast<const int32_t*>(acc), reinterpret_cast<const float*>(b->row_scale),
    reinterpret_cast<const float*>(b->channel_scale), reinterpret_cast<__nv_bfloat16*>(b->output), count, s->outputs);
  return int(cudaGetLastError());
}
extern "C" int ferrum_int8_output_fill_v1(uint64_t out, uint64_t count,
    uint16_t bits, void* stream) {
  if (!out || !count) return -1;
  fill<<<blocks(count),256,0,static_cast<cudaStream_t>(stream)>>>(reinterpret_cast<uint16_t*>(out), count, bits);
  return int(cudaGetLastError());
}
extern "C" int ferrum_int8_output_compare_v1(uint64_t reference, uint64_t actual,
    uint64_t count, uint64_t result, void* stream) {
  if (!reference || !actual || !count || !result) return -1;
  compare<<<blocks(count),256,0,static_cast<cudaStream_t>(stream)>>>(
    reinterpret_cast<const uint16_t*>(reference), reinterpret_cast<const uint16_t*>(actual),
    count, reinterpret_cast<FerrumInt8OutputComparison*>(result));
  return int(cudaGetLastError());
}
