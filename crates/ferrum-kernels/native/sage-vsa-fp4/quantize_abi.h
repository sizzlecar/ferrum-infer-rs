#pragma once
#include "abi.h"

// Independent QKV preparation boundary. Profile 1 is BF16 sequence-major
// [B, rows, H*128], tile128, original Sage E2M1/E4M3 group16 numerics. Q and K
// row counts may differ. K is ALREADY centered by an explicit preceding
// operation; this entry neither centers K nor slices prefix/video queries.
// Strides are in BF16 elements, in [batch, head, sequence] order. The first
// profile accepts only the exact contiguous sequence-major strides.
struct FerrumSageVsaFp4QuantizeRequestV1 {
  uint32_t abi, size, batches, heads, query_rows, key_rows, head_dim, profile;
  uint32_t device_ordinal, architecture, reserved[2];
  uint64_t query_strides[3], key_strides[3], value_strides[3];
};
struct FerrumSageVsaFp4QuantizeBytesV1 {
  uint64_t query_input, key_input, value_input;
  uint64_t query, key, value, query_scales, key_scales, value_scales;
};
struct FerrumSageVsaFp4QuantizePlanV1 {
  FerrumSageVsaFp4QuantizeRequestV1 request;
  uint32_t abi, size, query_grid_x, key_grid_x, grid_y, grid_z, threads, launch_count;
  uint32_t query_shared_bytes, key_shared_bytes, value_shared_bytes, reserved;
  FerrumSageVsaFp4QuantizeBytesV1 bytes;
};
struct FerrumSageVsaFp4QuantizeBindingsV1 {
  FerrumSageVsaFp4SpanV1 query_input, key_input, value_input;
  // Six pairwise-disjoint device writes, disjoint from all three inputs.
  // Q/K packed storage: [B,H,rows,64]. SFQ/SFK: physical [B,H,rows,8].
  // K has the original within-32-row permutation. Scale bytes have original
  // 64-row/4-column swizzling, not the GEMM provider's 128-row scale atoms.
  // V: [B,H,128,key_rows/2], SFV: physical [B,H,128,key_rows/16].
  FerrumSageVsaFp4SpanV1 query, key, value, query_scales, key_scales, value_scales;
};
static_assert(sizeof(FerrumSageVsaFp4QuantizeRequestV1) == 120, "quant request ABI drift");
static_assert(offsetof(FerrumSageVsaFp4QuantizeRequestV1, query_strides) == 48, "quant strides offset drift");
static_assert(sizeof(FerrumSageVsaFp4QuantizeBytesV1) == 72, "quant bytes ABI drift");
static_assert(sizeof(FerrumSageVsaFp4QuantizePlanV1) == 240, "quant plan ABI drift");
static_assert(offsetof(FerrumSageVsaFp4QuantizePlanV1, bytes) == 168, "quant bytes offset drift");
static_assert(sizeof(FerrumSageVsaFp4QuantizeBindingsV1) == 144, "quant binding ABI drift");

// Plan/Validate enqueue nothing. Run revalidates exact extents, device kinds,
// non-aliasing and explicit stream, then enqueues Q, permuted K and transposed
// V quantization. There is no global workspace, allocation, Host readback,
// default stream, device switch or synchronization. Any error after a launch
// may follow partial submission: the caller retains every input/output/code
// owner until device completion or quarantine. Pointer attributes cannot
// establish allocation extent or ownership; these remain unsafe caller duties.
extern "C" int ferrum_sage_vsa_fp4_quantize_plan_v1(const FerrumSageVsaFp4QuantizeRequestV1*, FerrumSageVsaFp4QuantizePlanV1*) noexcept;
extern "C" int ferrum_sage_vsa_fp4_quantize_validate_v1(const FerrumSageVsaFp4QuantizePlanV1*, const FerrumSageVsaFp4QuantizeBindingsV1*) noexcept;
extern "C" int ferrum_sage_vsa_fp4_quantize_run_v1(const FerrumSageVsaFp4QuantizePlanV1*, const FerrumSageVsaFp4QuantizeBindingsV1*, void*) noexcept;
