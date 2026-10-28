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
