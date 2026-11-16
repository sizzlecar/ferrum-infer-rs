#pragma once
#include <stdint.h>
#include <stddef.h>

// Caller owns all device spans, current context, stream and module capacity.
// No CUDA/CUTLASS type crosses this boundary. Zero workspace is exactly {0,0}.
struct FerrumCutlassNvfp4RequestV1 {
  uint32_t abi, size, rows, inputs, outputs, device_ordinal, architecture, multiprocessors;
  uint64_t maximum_dynamic_shared_bytes;
};
struct FerrumCutlassNvfp4PlanV1 {
  FerrumCutlassNvfp4RequestV1 request;
  uint32_t abi, size, kernel_id, shared_bytes;
  uint64_t workspace_bytes;
};
struct FerrumCutlassNvfp4SpanV1 { uint64_t address, bytes; };
struct FerrumCutlassNvfp4BindingsV1 {
  FerrumCutlassNvfp4SpanV1 input, weight, input_scales, weight_scales, alpha, output, workspace;
};
static_assert(sizeof(FerrumCutlassNvfp4RequestV1) == 40, "request ABI drift");
static_assert(sizeof(FerrumCutlassNvfp4PlanV1) == 64, "plan ABI drift");
static_assert(offsetof(FerrumCutlassNvfp4PlanV1, workspace_bytes) == 56, "plan offset drift");
static_assert(sizeof(FerrumCutlassNvfp4BindingsV1) == 112, "bindings ABI drift");

// 0 success; positive CUDA error; negative contract/CUTLASS error.
// Plan/validate do not enqueue or read device payload. Run errors may follow
// enqueue and never justify releasing resources without completion/quarantine.
extern "C" int ferrum_cutlass_nvfp4_plan_v1(const FerrumCutlassNvfp4RequestV1*, FerrumCutlassNvfp4PlanV1*) noexcept;
extern "C" int ferrum_cutlass_nvfp4_validate_v1(const FerrumCutlassNvfp4PlanV1*, const FerrumCutlassNvfp4BindingsV1*) noexcept;
extern "C" int ferrum_cutlass_nvfp4_run_v1(const FerrumCutlassNvfp4PlanV1*, const FerrumCutlassNvfp4BindingsV1*, void* stream) noexcept;

struct FerrumNvfp4QuantizeBf16RequestV1 {
  uint32_t abi, size, rows, inputs, device_ordinal, architecture, multiprocessors, profile;
};
struct FerrumNvfp4QuantizeBf16PlanV1 {
  FerrumNvfp4QuantizeBf16RequestV1 request;
  uint32_t abi, size, kernel_id, blocks, threads, reserved[3];
};
struct FerrumNvfp4QuantizeBf16BindingsV1 {
  FerrumCutlassNvfp4SpanV1 input, global_scale, packed, scales;
};
static_assert(sizeof(FerrumNvfp4QuantizeBf16RequestV1) == 32, "quantize request ABI drift");
static_assert(sizeof(FerrumNvfp4QuantizeBf16PlanV1) == 64, "quantize plan ABI drift");
static_assert(offsetof(FerrumNvfp4QuantizeBf16PlanV1, threads) == 48, "quantize plan offset drift");
static_assert(sizeof(FerrumNvfp4QuantizeBf16BindingsV1) == 64, "quantize bindings ABI drift");
extern "C" int ferrum_nvfp4_quantize_bf16_plan_v1(const FerrumNvfp4QuantizeBf16RequestV1*, FerrumNvfp4QuantizeBf16PlanV1*) noexcept;
extern "C" int ferrum_nvfp4_quantize_bf16_validate_v1(const FerrumNvfp4QuantizeBf16PlanV1*, const FerrumNvfp4QuantizeBf16BindingsV1*) noexcept;
extern "C" int ferrum_nvfp4_quantize_bf16_run_v1(const FerrumNvfp4QuantizeBf16PlanV1*, const FerrumNvfp4QuantizeBf16BindingsV1*, void* stream) noexcept;
