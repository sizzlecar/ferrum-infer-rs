//! Selected production GDN query, physical attribution and output/state parity.
use super::attention_cost_route::{compare, ExpectedRoute};
use super::*;

#[test]
fn selected_metal_gated_delta_cost_route_matches_actual_scalar_f32_master() {
    compare(AttentionKind::GatedDelta, &[4], ExpectedRoute::GatedDelta);
}

#[test]
fn selected_metal_gated_delta_cost_route_matches_actual_packed_multiple_rows() {
    compare(
        AttentionKind::GatedDelta,
        &[2, 3],
        ExpectedRoute::GatedDelta,
    );
}

#[test]
fn selected_metal_gated_delta_cost_route_unknown_hadamard_still_executes() {
    for kind in [
        AttentionKind::GatedDeltaHadamardF16,
        AttentionKind::GatedDeltaHadamardF32,
    ] {
        compare(kind, &[2, 3], ExpectedRoute::GatedDelta);
    }
}
