//! Explicit IQ4 shared-dot experiment; the product FFN route remains MMQ.
use super::*;

unsafe extern "C" {
    fn ferrum_test_upstream_mmvq_iq4_shared_weight_decode_dot_v1(
        plan: *const UpstreamLinearPlanV1,
        weights: *const c_void,
        packed: *const c_void,
        output: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
}

#[test]
#[ignore = "exclusive CUDA IQ4 shared-dot complete-projection diagnostic"]
fn iq4_shared_weight_decode_ffn_complete_projection_pairs() {
    let format = UpstreamLinearFormat::Iq4Xs;
    let dot: MmvqDot = ferrum_test_upstream_mmvq_iq4_shared_weight_decode_dot_v1;
    // Full shared/baseline buffers, canonical markers, odd input/output views,
    // guard/padding and rejected ABI boundaries are outside measured intervals.
    shared_weight_decode::assert_marker_bits_and_boundaries(
        format,
        [ffi::ferrum_upstream_mmvq_dot_v1, dot],
        &[(256, 17), (512, 33), (768, 17)],
    );
    for algorithm in [UpstreamLinearAlgorithm::Mmq, UpstreamLinearAlgorithm::Mmvq] {
        for input in [
            Input::Finite,
            Input::Zero,
            Input::Subnormal,
            Input::Nonfinite,
            Input::SumOverflow,
        ] {
            check_geometry(
                algorithm,
                format,
                UpstreamLinearLayout::Columns,
                [8, 768, 17],
                input,
                false,
                true,
            );
        }
        check_geometry(
            algorithm,
            format,
            UpstreamLinearLayout::Columns,
            [8, 768, 17],
            Input::Finite,
            true,
            true,
        );
    }
    for (k, n) in [(5120, 17408), (17408, 5120)] {
        complete_projection::paired_projection_ring(
            "iq4_shared_m8_ffn_complete_projection",
            format,
            k,
            n,
            dot,
            "mmvq_iq4_shared_m8_test",
        );
    }
}
