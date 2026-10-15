// Independent test-only TU; production continues to compile mmq.cu.
#define FERRUM_TEST_MMQ_FULL_TILE_STREAM_K 1
#include "mmq.cu"

// Private helper for the independent I32 TU. Validate the complete original
// public plan without calling plan_v1's device queries/attribute setup inside
// graph capture or timing. No alternate public plan is issued.
extern "C" int ferrum_test_upstream_mmq_validate_i32_full_tile_plan_v1(
        const FerrumUpstreamLinearPlanV1 * plan) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmqPlan native{};
        if (!upstream_mmq_valid(plan, native)) return -1;
        if (native.rows != 8 || native.j != 8 || native.i != 128 ||
                native.nthreads != 256 || plan->request.cc < GGML_CUDA_CC_AMPERE) return -2;
        return 0;
    });
}
