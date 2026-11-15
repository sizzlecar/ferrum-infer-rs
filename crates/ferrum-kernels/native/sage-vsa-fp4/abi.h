#pragma once
#include <stdint.h>
#include <stddef.h>

// Caller owns the current context, explicit stream, code and every Host/device
// span. No CUDA/CUTLASS/Torch types cross this boundary. Profile 1 is SM120,
// D128, tile128, noncausal, unsmoothed Q, two-level P and BF16 output.
struct FerrumSageVsaFp4RequestV1 {
  uint32_t abi, size, batches, heads, query_blocks, key_blocks, head_dim, tile_rows;
  uint32_t profile, softmax_scale_bits, device_ordinal, architecture, multiprocessors, reserved;
  uint64_t maximum_dynamic_shared_bytes;
};
struct FerrumSageVsaFp4BufferSizesV1 {
  uint64_t query, key, value, query_scales, key_scales, value_scales;
  uint64_t output, lse, zero_delta, indices, counts, half_valid;
};
struct FerrumSageVsaFp4PlanV1 {
  FerrumSageVsaFp4RequestV1 request;
  uint32_t abi, size, kernel_id, shared_bytes, threads, grid_blocks, upload_count, memset_count;
  FerrumSageVsaFp4BufferSizesV1 bytes;
};
struct FerrumSageVsaFp4SpanV1 { uint64_t address, bytes; };
struct FerrumSageVsaFp4BindingsV1 {
  FerrumSageVsaFp4SpanV1 query, key, value, query_scales, key_scales, value_scales;
  // All six following spans are writes, pairwise disjoint and disjoint from
  // QKV/scales. Run always uploads CPU lists; arbitrary device lists are not inputs.
  FerrumSageVsaFp4SpanV1 output, lse, zero_delta, indices, counts, half_valid;
};
struct FerrumSageVsaFp4HostListsV1 {
  FerrumSageVsaFp4SpanV1 indices, counts, half_valid;
};
static_assert(sizeof(FerrumSageVsaFp4RequestV1) == 64, "request ABI drift");
static_assert(offsetof(FerrumSageVsaFp4RequestV1, maximum_dynamic_shared_bytes) == 56, "request offset drift");
static_assert(sizeof(FerrumSageVsaFp4BufferSizesV1) == 96, "sizes ABI drift");
static_assert(sizeof(FerrumSageVsaFp4PlanV1) == 192, "plan ABI drift");
static_assert(offsetof(FerrumSageVsaFp4PlanV1, bytes) == 96, "plan offset drift");
static_assert(sizeof(FerrumSageVsaFp4BindingsV1) == 192, "bindings ABI drift");
static_assert(sizeof(FerrumSageVsaFp4HostListsV1) == 48, "Host lists ABI drift");

// 0 success; positive CUDA error; negative contract/launch failure.
// Plan/Validate never enqueue. Validate/Run dereference the Host lists after
// checking exact lengths, alignment and CUDA pinned-Host pointer kind. The
// unsafe caller must ensure each full Host extent is readable and immutable,
// and all Host/device/code owners remain alive until completion. A pointer-kind
// query cannot prove allocation extent or ownership.
// Run first revalidates, then enqueues three H2D copies, a zero-delta memset,
// and attention. Any later error may follow submission; no error code licenses
// early release. There is no allocator, pin/unpin, default stream, stream sync,
// device-mask bypass, or process exit in this boundary.
extern "C" int ferrum_sage_vsa_fp4_plan_v1(const FerrumSageVsaFp4RequestV1*, FerrumSageVsaFp4PlanV1*) noexcept;
extern "C" int ferrum_sage_vsa_fp4_validate_v1(const FerrumSageVsaFp4PlanV1*, const FerrumSageVsaFp4BindingsV1*, const FerrumSageVsaFp4HostListsV1*) noexcept;
extern "C" int ferrum_sage_vsa_fp4_run_v1(const FerrumSageVsaFp4PlanV1*, const FerrumSageVsaFp4BindingsV1*, const FerrumSageVsaFp4HostListsV1*, void* stream) noexcept;
