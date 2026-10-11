// Ferrum upstream quantized linear ABI v1. No CUDA types cross this boundary.
#pragma once
#include <stdint.h>
#include <stddef.h>
struct FerrumUpstreamLinearRequestV1 {
    uint32_t abi, size, format, layout, rows, inputs, outputs, cc, sm_count, reserved;
    uint64_t shared_limit;
};
struct FerrumUpstreamLinearPlanV1 {
    FerrumUpstreamLinearRequestV1 request;
    uint32_t abi, size, algorithm, pack_abi, padded_inputs, padded_outputs, guard_blocks;
    uint32_t j, i, nthreads, shared_bytes, blocks, tiles_y, fixup;
    uint32_t ncols, channels, nwarps, rows_per_block, small_k, reserved;
    uint64_t weight_bytes, converted_bytes, packed_bytes, output_bytes, fixup_bytes;
};
static_assert(sizeof(FerrumUpstreamLinearRequestV1)==48, "request ABI drift");
static_assert(sizeof(FerrumUpstreamLinearPlanV1)==168, "plan ABI drift");
// Algorithms: 1 MMQ, 2 MMVQ. Layouts: 0 Columns, 1 Channels.
// Packs: 1 MMQ D4, 2 MMQ DS4, 3 ordinary row-major Q8_1.
// Status: zero success, positive CUDA runtime status; negative validated contract
// rejection (-1 argument, -2 unsupported geometry, -3 extent, -4 dispatch,
// -5 span/alignment, -6 internal upstream invariant, -7 unexpected exception).

// V1 launch geometry is immutable; V2 changes only declared marker semantics.
// Pointer extents/leases are checked by the Rust adapter. No function allocates
// device storage or synchronizes. plan may query/configure the current device.
#define FERRUM_UPSTREAM_ABI_DECL(prefix) \
extern "C" int prefix##_plan_v1(const FerrumUpstreamLinearRequestV1 *,FerrumUpstreamLinearPlanV1 *) noexcept; \
extern "C" int prefix##_pack_v1(const FerrumUpstreamLinearPlanV1 *,const void *,uint32_t,void *,void *,void *) noexcept; \
extern "C" int prefix##_dot_v1(const FerrumUpstreamLinearPlanV1 *,const void *,const void *,void *,void *,void *) noexcept; \
extern "C" int prefix##_cast_v1(const FerrumUpstreamLinearPlanV1 *,const void *,void *,uint32_t,void *) noexcept; \
extern "C" int prefix##_pack_v2(const FerrumUpstreamLinearPlanV1 *,const void *,uint32_t,void *,void *,void *,void *) noexcept; \
extern "C" int prefix##_check_weights_v2(const FerrumUpstreamLinearPlanV1 *,const void *,void *,void *) noexcept; \
extern "C" int prefix##_cast_v2(const FerrumUpstreamLinearPlanV1 *,const void *,void *,uint32_t,const void *,const void *,void *) noexcept;
FERRUM_UPSTREAM_ABI_DECL(ferrum_upstream_mmq)
FERRUM_UPSTREAM_ABI_DECL(ferrum_upstream_mmvq)
#undef FERRUM_UPSTREAM_ABI_DECL
// Explicit dense Columns M33..2048 admission, using fixed MMQ J32/I128.
// Old MMQ plan retains its <=32 domain; launch ABI and numerical rules are shared.
extern "C" int ferrum_upstream_mmq_prefill_plan_v1(const FerrumUpstreamLinearRequestV1 *,
        FerrumUpstreamLinearPlanV1 *) noexcept;
