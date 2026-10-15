// Independent Q6_K F32 D4/MMQ ABI v1. No CUDA types cross this boundary.
#pragma once
#include <stdint.h>
#include <stddef.h>
struct FerrumUpstreamQ6F32RequestV1 {
    uint32_t abi, size, format, layout, rows, inputs, outputs, cc, sm_count, reserved;
    uint64_t shared_limit;
};
struct FerrumUpstreamQ6F32PlanV1 {
    FerrumUpstreamQ6F32RequestV1 request;
    uint32_t abi, size, algorithm, pack_abi, padded_inputs, padded_outputs, guard_blocks;
    uint32_t j, i, nthreads, shared_bytes, blocks, tiles_y, fixup;
    uint32_t ncols, channels, nwarps, rows_per_block, small_k, reserved;
    uint64_t weight_bytes, converted_bytes, packed_bytes, output_bytes, fixup_bytes;
};
static_assert(sizeof(FerrumUpstreamQ6F32RequestV1)==48, "request ABI drift");
static_assert(sizeof(FerrumUpstreamQ6F32PlanV1)==168, "plan ABI drift");
// Fixed format14/layout0/algorithm1/pack1; actual M1..32, K multiple256.
// F32 input, packed D4 activation and F32 output. Row/leaf flags propagate
// canonical F32 NaN; no F16 intermediate boundary or before-dot rejection.
// Status matches Ferrum's upstream ABI: positive CUDA error, negative contract
// error. plan may inspect/configure the device; launches allocate/sync nothing.
extern "C" int ferrum_upstream_q6_f32_plan_v1(const FerrumUpstreamQ6F32RequestV1 *, FerrumUpstreamQ6F32PlanV1 *) noexcept;
extern "C" int ferrum_upstream_q6_f32_pack_v1(const FerrumUpstreamQ6F32PlanV1 *, const void *, uint32_t, void *, void *, void *, void *) noexcept;
extern "C" int ferrum_upstream_q6_f32_dot_v1(const FerrumUpstreamQ6F32PlanV1 *, const void *, const void *, void *, void *, void *) noexcept;
extern "C" int ferrum_upstream_q6_f32_check_weights_v1(const FerrumUpstreamQ6F32PlanV1 *, const void *, void *, void *) noexcept;
extern "C" int ferrum_upstream_q6_f32_publish_v1(const FerrumUpstreamQ6F32PlanV1 *, const void *, void *, uint32_t, const void *, const void *, void *) noexcept;

// Diagnostic F16 boundary, deliberately distinct from the F32 ABI above.
// Export version v1 is its first interface; wire domain 2 prevents either
// boundary from accepting the other's request/plan. It is not a product route.
struct FerrumUpstreamQ6F16RequestV1 {
    uint32_t abi, size, format, layout, rows, inputs, outputs, cc, sm_count, reserved;
    uint64_t shared_limit;
};
struct FerrumUpstreamQ6F16PlanV1 {
    FerrumUpstreamQ6F16RequestV1 request;
    uint32_t abi, size, algorithm, pack_abi, padded_inputs, padded_outputs, guard_blocks;
    uint32_t j, i, nthreads, shared_bytes, blocks, tiles_y, fixup;
    uint32_t ncols, channels, nwarps, rows_per_block, small_k, reserved;
    uint64_t weight_bytes, converted_bytes, packed_bytes, output_bytes, fixup_bytes;
};
static_assert(sizeof(FerrumUpstreamQ6F16RequestV1)==48, "F16 request ABI drift");
static_assert(sizeof(FerrumUpstreamQ6F16PlanV1)==168, "F16 plan ABI drift");
static_assert(offsetof(FerrumUpstreamQ6F16PlanV1, weight_bytes)==128, "F16 plan field drift");
// Fixed request/plan abi2, format14/layout0/algorithm1/pack1. Actual M1..32,
// K multiple256. Input/output strides count F16 elements, permitting an aligned
// sub-leaf base within a wider row. converted_bytes/output_bytes describe F32
// scratch, NOT the final F16 output span. The caller validates all live spans.
// pack clears row flags and the owned packed guard, converts F16 exactly to F32,
// and uses the shared Q6 D4 quantizer. check_weights clears/scans the leaf flag.
// cast rounds finite raw F32 to F16 RN; invalid row/leaf, nonfinite raw or F16
// overflow publishes canonical F16 NaN 0x7e00. No allocation or synchronization.
extern "C" int ferrum_upstream_q6_f16_plan_v1(const FerrumUpstreamQ6F16RequestV1 *, FerrumUpstreamQ6F16PlanV1 *) noexcept;
extern "C" int ferrum_upstream_q6_f16_pack_v1(const FerrumUpstreamQ6F16PlanV1 *, const void *, uint32_t, void *, void *, void *, void *) noexcept;
extern "C" int ferrum_upstream_q6_f16_dot_v1(const FerrumUpstreamQ6F16PlanV1 *, const void *, const void *, void *, void *, void *) noexcept;
extern "C" int ferrum_upstream_q6_f16_check_weights_v1(const FerrumUpstreamQ6F16PlanV1 *, const void *, void *, void *) noexcept;
extern "C" int ferrum_upstream_q6_f16_cast_v1(const FerrumUpstreamQ6F16PlanV1 *, const void *, void *, uint32_t, const void *, const void *, void *) noexcept;
