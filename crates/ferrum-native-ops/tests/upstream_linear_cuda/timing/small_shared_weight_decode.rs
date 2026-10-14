//! Partial batches use MMQ today except M4. Compare whole projections against
//! that actual route before considering any product numerical policy change.
use super::*;

unsafe extern "C" {
    fn ferrum_test_upstream_mmvq_small_shared_weight_decode_dot_v1(
        plan: *const UpstreamLinearPlanV1,
        weights: *const c_void,
        packed: *const c_void,
        output: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
}

#[test]
#[ignore = "exclusive CUDA partial-batch shared weight complete-projection diagnostic"]
fn partial_batch_shared_weight_complete_projection_pairs() {
    let dot: MmvqDot = ferrum_test_upstream_mmvq_small_shared_weight_decode_dot_v1;
    for format in [UpstreamLinearFormat::Q4K, UpstreamLinearFormat::Q5K] {
        for m in 2..=8 {
            shared_weight_decode::assert_small_shared_markers(format, m, dot);
        }
        // M2, M3 and M7 account for 524/2194 observed decode waves in the
        // retained C8 diagnostic. M4 is an existing-MMVQ comparison control.
        for m in [2, 3, 4, 7] {
            let current = if m == 4 {
                UpstreamLinearAlgorithm::Mmvq
            } else {
                UpstreamLinearAlgorithm::Mmq
            };
            for (k, n) in [(5120, 10240), (6144, 5120), (5120, 17408), (17408, 5120)] {
                complete_projection::paired_projection_ring_for_rows(
                    "partial_batch_shared_weight_complete_projection",
                    format,
                    m,
                    current,
                    k,
                    n,
                    dot,
                    "mmvq_small_shared_test",
                );
            }
        }
    }
}
