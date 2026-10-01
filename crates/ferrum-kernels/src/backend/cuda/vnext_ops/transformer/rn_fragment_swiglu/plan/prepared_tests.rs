//! CPU-only equivalence and lifetime boundaries of prepared RN metadata.
use super::*;
use ferrum_interfaces::execution_cost::{AlgorithmWorkKindV1, SelectedReplayAlgorithmTemplateV1};

fn prepared(gate: RnF16FragmentSourceFormatV1, down: RnF16FragmentSourceFormatV1) -> PreparedShape {
    PreparedShape {
        hidden: 256,
        intermediate: 512,
        weights: PreparedWeights::from_axes(256, 512, gate, down)
            .unwrap()
            .prepare_algorithms()
            .unwrap(),
    }
}

fn counts() -> [usize; 3] {
    PREPARATION_COUNTS.with(std::cell::Cell::get)
}

fn selected(shape: Shape) -> SelectedCommandCostEvidenceV1 {
    shape
        .selected(
            SloStructuredCostCapture::HostSettledV1,
            Some(CublasHandleApiIdentity::fixture_identity()),
        )
        .unwrap()
}

fn assert_same_shape(left: Shape, right: Shape) {
    let tokens = left.tokens;
    assert_eq!(
        (
            left.tokens,
            left.rows,
            left.hidden,
            left.intermediate,
            left.activation_elements,
            left.gate_up_bytes,
            left.scratch_bytes,
        ),
        (
            right.tokens,
            right.rows,
            right.hidden,
            right.intermediate,
            right.activation_elements,
            right.gate_up_bytes,
            right.scratch_bytes,
        ),
    );
    assert_eq!(left.gate_weight, right.gate_weight);
    assert_eq!(left.down_weight, right.down_weight);
    // Evidence also covers both dynamic Gemm plans above the fragment boundary.
    let left = selected(left);
    let right = selected(right);
    assert_eq!(left, right);
    assert_eq!(left.algorithm_work(), right.algorithm_work());
    assert_eq!(
        left.independent_attention_family_v2(),
        right.independent_attention_family_v2(),
    );
    SelectedReplayAlgorithmTemplateV1::from_selected(&left, tokens, 3, 0)
        .unwrap()
        .validate_binding(&right)
        .unwrap();
}

#[test]
fn prepared_weights_preserve_current_extent_library_boundary_and_physical_axes() {
    use RnF16FragmentSourceFormatV1::{Q4K, Q6K};
    let mut prepared = prepared(Q4K, Q6K);
    for tokens in [1, 8, 9, 17] {
        let shape = prepared.for_tokens(tokens).unwrap();
        assert_eq!(shape.tokens, tokens);
        assert_eq!(shape.scratch_bytes, tokens * 512 * 6);
        let evidence = selected(shape);
        evidence.validate_command(tokens, 3, 0).unwrap();
        assert_eq!(
            evidence
                .algorithm_work()
                .unwrap()
                .unwrap()
                .entries()
                .iter()
                .filter(|work| work.kind() == AlgorithmWorkKindV1::LibraryCall)
                .count(),
            if tokens <= 8 { 0 } else { 2 },
        );
        assert_eq!(
            shape
                .selected(SloStructuredCostCapture::HostSettledV1, None)
                .is_some(),
            tokens <= 8,
        );
        assert!(shape
            .selected(
                SloStructuredCostCapture::Disabled,
                Some(CublasHandleApiIdentity::fixture_identity()),
            )
            .is_none());
    }
    for invalid in [0, u64::MAX, i32::MAX as u64 + 1] {
        assert!(prepared.for_tokens(invalid).is_err());
    }
    prepared.hidden = 512;
    assert!(prepared.for_tokens(1).is_err());
    prepared.hidden = 256;
    prepared.intermediate = 256;
    assert!(prepared.for_tokens(1).is_err());
}

#[test]
fn prepared_static_builds_are_reused_but_search_and_replay_rebuild_dynamic_instances() {
    use RnF16FragmentSourceFormatV1::{Q4K, Q5K, Q6K};
    for (gate, down) in [(Q4K, Q5K), (Q5K, Q6K), (Q6K, Q4K)] {
        PREPARATION_COUNTS.with(|counts| counts.set([0; 3]));
        let prepared = prepared(gate, down);
        assert_eq!(counts(), [2, 2, 0]);
        for tokens in [1, 2, 8, 9, 17] {
            let before = counts();
            let candidate = prepared.for_tokens(tokens).unwrap();
            let replay = prepared.for_tokens(tokens).unwrap();
            assert_same_shape(candidate, replay);
            // Each query creates a new dynamic shape. Neither query recreates
            // the two static weight plans or hashes the two algorithm classes.
            assert_eq!(counts(), [before[0], before[1], before[2] + 2]);

            let before_cold = counts();
            let cold = Shape::new(tokens, 256, 512, gate, down).unwrap();
            assert_same_shape(candidate, cold);
            assert_eq!(
                counts(),
                [
                    before_cold[0] + 2,
                    before_cold[1] + if tokens <= 8 { 2 } else { 0 },
                    before_cold[2] + 1,
                ],
            );
        }
    }
}

#[test]
fn prepared_static_classes_keep_original_identity_and_fresh_replay_geometry() {
    use RnF16FragmentSourceFormatV1::{Q4K, Q5K, Q6K};
    for format in [Q4K, Q5K, Q6K] {
        let prepared = prepared(format, format);
        // Original class recipe, independent of the new prepared helper.
        let plan = prepared.weights.gate;
        let kernel = FragmentKernel::for_format(format);
        let mut layout = Sha256::new();
        layout.update(kernel.layout_domain());
        layout.update(plan.packing_abi().to_le_bytes());
        layout.update(format_code(format).to_le_bytes());
        let original = SelectedAlgorithmClassV1::new(
            kernel.entry(),
            1,
            Sha256::digest(crate::ptx::VNEXT_GGUF.as_bytes()).into(),
            layout.finalize().into(),
        )
        .unwrap();
        assert_eq!(prepared.weights.algorithms, Some([original; 2]));

        let first = selected(prepared.for_tokens(8).unwrap());
        let template = SelectedReplayAlgorithmTemplateV1::from_selected(&first, 8, 3, 0).unwrap();
        let same = selected(prepared.for_tokens(8).unwrap());
        template.validate_binding(&same).unwrap();
        for changed_tokens in [1, 9] {
            let changed_shape = prepared.for_tokens(changed_tokens).unwrap();
            let changed = selected(changed_shape);
            assert_ne!(first, changed);
            assert!(template.validate_binding(&changed).is_err());
            assert!(changed_shape
                .project(8, Some(CublasHandleApiIdentity::fixture_identity()))
                .is_none());
        }
        let library_shape = prepared.for_tokens(9).unwrap();
        assert!(library_shape
            .selected(SloStructuredCostCapture::HostSettledV1, None)
            .is_none());
    }
}

#[test]
fn unprepared_encoding_keeps_static_evidence_lazy_when_capture_is_disabled() {
    use RnF16FragmentSourceFormatV1::{Q4K, Q6K};
    for tokens in [1, 8, 9] {
        PREPARATION_COUNTS.with(|counts| counts.set([0; 3]));
        let shape = Shape::new(tokens, 256, 512, Q4K, Q6K).unwrap();
        assert_eq!(counts(), [2, 0, 1]);
        assert!(shape
            .selected(
                SloStructuredCostCapture::Disabled,
                Some(CublasHandleApiIdentity::fixture_identity()),
            )
            .is_none());
        assert_eq!(counts(), [2, 0, 1]);
        selected(shape);
        assert_eq!(counts(), [2, if tokens <= 8 { 2 } else { 0 }, 1]);
    }
}
