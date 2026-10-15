//! Native projection diagnostic: actual M8 attention shapes, MMQ versus shared MMVQ.
//! Reuses timing::Route, its actual-packed-input F64 oracle and graph helpers.
//! Different activation pack contracts are compared; no cross-route bit claim.
use super::*;

#[test]
#[ignore = "exclusive CUDA current attention MMQ/shared-MMVQ diagnostic"]
fn marker_v2_m8_attention_mmq_vs_shared_mmvq_complete_projection_pairs() {
    for (format, k, n) in [
        (UpstreamLinearFormat::Q4K, 5120, 10240),
        (UpstreamLinearFormat::Q5K, 6144, 5120),
    ] {
        // Qualified helper checks retain per-algorithm zero/marker semantics.
        // These checks and all cold setup are outside measured intervals.
        shared_weight_decode::validate_selected_markers(format);
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
        complete_projection::paired_projection_ring(
            "attention_m8_current_complete_projection",
            format,
            k,
            n,
            shared_weight_decode::selected_dot(format),
            "mmvq_shared_m8_current",
        );
    }
}
