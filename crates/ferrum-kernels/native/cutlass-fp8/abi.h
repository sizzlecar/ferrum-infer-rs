#pragma once
#include <stdint.h>
#include <stddef.h>

// ABI 1, architecture-bound SM89/SM120 W8A8 kernels. No CUDA/CUTLASS types cross FFI.
struct FerrumCutlassFp8RequestV1 {
  uint32_t abi, size, rows, inputs, outputs, device_ordinal, architecture, multiprocessors;
  uint64_t maximum_dynamic_shared_bytes;
};
struct FerrumCutlassFp8PlanV1 {
  FerrumCutlassFp8RequestV1 request;
  uint32_t abi, size, kernel_id, shared_bytes;
  uint64_t quantized_activation_bytes, activation_scale_offset, activation_scale_bytes;
  uint64_t gemm_workspace_offset, gemm_workspace_bytes, scratch_bytes;
};
struct FerrumCutlassFp8SpanV1 { uint64_t address, bytes; };
struct FerrumCutlassFp8BindingsV1 {
  FerrumCutlassFp8SpanV1 input, weight, weight_scale, output, scratch;
};
static_assert(sizeof(FerrumCutlassFp8RequestV1) == 40, "request ABI drift");
static_assert(sizeof(FerrumCutlassFp8PlanV1) == 104, "plan ABI drift");
static_assert(offsetof(FerrumCutlassFp8PlanV1, quantized_activation_bytes) == 56, "plan offset drift");
static_assert(sizeof(FerrumCutlassFp8BindingsV1) == 80, "bindings ABI drift");

// 0 success, positive CUDA error, negative contract/CUTLASS error.
// Plan/validate use the caller's already-current context; never cudaSetDevice.
// Run may return an error AFTER enqueue, hence never implies NotSubmitted.
extern "C" int ferrum_cutlass_fp8_plan_v1(const FerrumCutlassFp8RequestV1*, FerrumCutlassFp8PlanV1*) noexcept;
extern "C" int ferrum_cutlass_fp8_validate_v1(const FerrumCutlassFp8PlanV1*, const FerrumCutlassFp8BindingsV1*) noexcept;
extern "C" int ferrum_cutlass_fp8_run_v1(const FerrumCutlassFp8PlanV1*, const FerrumCutlassFp8BindingsV1*, void* stream) noexcept;

// Two or three equal-shape projections share exact input/scratch spans. All
// outputs are disjoint from every read and from one another. Validation of the
// entire group precedes enqueue. One row quantization feeds independent GEMMs.
extern "C" int ferrum_cutlass_fp8_validate_group_v1(const FerrumCutlassFp8PlanV1*, const FerrumCutlassFp8BindingsV1*, uint32_t count) noexcept;
extern "C" int ferrum_cutlass_fp8_run_group_v1(const FerrumCutlassFp8PlanV1*, const FerrumCutlassFp8BindingsV1*, uint32_t count, void* stream) noexcept;

// Opt-in diagnostic only. The caller creates and retains three distinct timing
// events on the same context through completion. Records start, after quant,
// and after all GEMMs; never creates events, synchronizes or allocates payloads.
// Recorded intervals include any host enqueue gaps. Normal run_group has no
// event calls. Errors may follow partial enqueue, including event recording.
extern "C" int ferrum_cutlass_fp8_run_group_profiled_v1(const FerrumCutlassFp8PlanV1*, const FerrumCutlassFp8BindingsV1*, uint32_t count, void* stream, void* start, void* after_quant, void* after_gemm) noexcept;
