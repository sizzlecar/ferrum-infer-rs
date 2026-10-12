// Independent test-only Q8_0 MMQ MarkerV2 exports in the upstream-linear artifact.
// Original three-format entrypoints keep their original admission and math.
#pragma once
#include "abi.h"
extern "C" int ferrum_upstream_q8_mmq_plan_v1(const FerrumUpstreamLinearRequestV1 *,FerrumUpstreamLinearPlanV1 *) noexcept;
extern "C" int ferrum_upstream_q8_mmq_pack_v2(const FerrumUpstreamLinearPlanV1 *,const void *,uint32_t,void *,void *,void *,void *) noexcept;
extern "C" int ferrum_upstream_q8_mmq_dot_v1(const FerrumUpstreamLinearPlanV1 *,const void *,const void *,void *,void *,void *) noexcept;
extern "C" int ferrum_upstream_q8_mmq_check_weights_v2(const FerrumUpstreamLinearPlanV1 *,const void *,void *,void *) noexcept;
extern "C" int ferrum_upstream_q8_mmq_cast_v2(const FerrumUpstreamLinearPlanV1 *,const void *,void *,uint32_t,const void *,const void *,void *) noexcept;
