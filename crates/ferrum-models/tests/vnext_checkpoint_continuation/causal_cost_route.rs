//! Real selected causal providers, including independent INT8 payload/scales.
use super::attention_cost_route::{compare, ExpectedRoute};
use super::*;

#[test]
fn selected_metal_causal_cost_route_matches_scalar_multitoken_f16_kv() {
    compare(
        AttentionKind::Causal,
        &[8],
        ExpectedRoute::Causal { int8: false },
    );
}

#[test]
fn selected_metal_causal_cost_route_matches_packed_heterogeneous_f16_kv() {
    compare(
        AttentionKind::Causal,
        &[2, 9],
        ExpectedRoute::Causal { int8: false },
    );
}

#[test]
fn selected_metal_causal_cost_route_int8_keeps_reset_count_and_payload_scale_state() {
    for lengths in [&[9][..], &[129, 2][..]] {
        // The 129-token row crosses the payload's 64 KiB page while the
        // independent scale page remains separate. No grouping/alias guess.
        compare(
            AttentionKind::CausalInt8,
            lengths,
            ExpectedRoute::Causal { int8: true },
        );
    }
}
