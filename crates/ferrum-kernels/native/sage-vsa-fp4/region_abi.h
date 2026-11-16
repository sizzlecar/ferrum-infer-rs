#pragma once
#include "abi.h"

// Separate declared GPU producer profile. Original CPU-list ABI remains
// unchanged. The only metadata inputs are checked immutable geometry; device
// indices/counts are private output spans, always overwritten by this command.
struct FerrumSageVsaFp4RegionRequestV1 {
  FerrumSageVsaFp4RequestV1 attention;
  uint32_t region_count, selection_profile;
};
struct FerrumSageVsaFp4RegionPlanV1 {
  FerrumSageVsaFp4PlanV1 attention;
  uint32_t region_count, selection_profile, selection_threads, selection_grid_blocks;
  uint32_t upload_count, memset_count;
  uint64_t scores_bytes, regions_bytes;
};
struct FerrumSageVsaFp4RegionBindingsV1 {
  FerrumSageVsaFp4BindingsV1 attention;
  FerrumSageVsaFp4SpanV1 scores, regions;
};
struct FerrumSageVsaFp4RegionHostV1 {
  FerrumSageVsaFp4SpanV1 regions, half_valid;
};
static_assert(sizeof(FerrumSageVsaFp4RegionRequestV1) == 72, "region request ABI drift");
static_assert(sizeof(FerrumSageVsaFp4RegionPlanV1) == 232, "region plan ABI drift");
static_assert(sizeof(FerrumSageVsaFp4RegionBindingsV1) == 224, "region binding ABI drift");
static_assert(sizeof(FerrumSageVsaFp4RegionHostV1) == 32, "region Host ABI drift");

// Scores are full contiguous F32[B,H,Kblocks,Kblocks]. Video queries start at
// Kblocks-Qblocks. Region triples [start,end,keep] cover exactly this suffix;
// all prefix keys remain visible. Profile 1 orders score descending/index
// ascending; +0 ranks above -0, NaNs rank first, infinities remain legal.
// This is an explicit profile, not a claim of Torch's unspecified tie choice.
//
// Plan/Validate enqueue nothing. Run revalidates the readable immutable pinned
// Host geometry, then uploads two tables, zeroes delta, produces lists on GPU
// and calls the SAME attention kernel on the caller's explicit stream. The
// caller retains all spans/code until completion or quarantine, including
// errors after partial submission. No allocation or Host score readback.
extern "C" int ferrum_sage_vsa_fp4_region_plan_v1(const FerrumSageVsaFp4RegionRequestV1*, FerrumSageVsaFp4RegionPlanV1*) noexcept;
extern "C" int ferrum_sage_vsa_fp4_region_validate_v1(const FerrumSageVsaFp4RegionPlanV1*, const FerrumSageVsaFp4RegionBindingsV1*, const FerrumSageVsaFp4RegionHostV1*) noexcept;
extern "C" int ferrum_sage_vsa_fp4_region_run_v1(const FerrumSageVsaFp4RegionPlanV1*, const FerrumSageVsaFp4RegionBindingsV1*, const FerrumSageVsaFp4RegionHostV1*, void*) noexcept;
