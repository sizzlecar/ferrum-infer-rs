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
