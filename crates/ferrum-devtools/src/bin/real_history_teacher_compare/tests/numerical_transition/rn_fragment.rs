//! Evidence protocol only; actual model logits still require native captures.
use super::*;

const PROFILE: &str = "qwen3_5.f32-master.gguf-f16-projections.ffn-rn-fragment-m1to8";

fn fragment_pair(pair: Pair) -> Pair {
    let mut pair = full_pair_from(pair);
    let arm = &mut pair.candidate;
    arm.manifest.identity.as_mut().unwrap().numerical_profile = PROFILE.into();
    arm.manifest.configuration["numerical_execution"] = json!({"require": PROFILE});
    edit_arm_receipts(arm, |batch| {
        let ffn = &mut batch.nodes[0];
        assert_eq!(ffn.operation_id, DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID);
        ffn.operation_id = DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_OPERATION_ID.into();
        ffn.provider_id = "provider.rn-fragment.ffn-m1to8".into();
        ffn.provider_implementation_fingerprint = sha256(ffn.provider_id.as_bytes());
    });
    pair
}

fn fragment_declaration(pair: &Pair) -> NumericalTransition {
    let mut declaration = declaration(pair);
    declaration.scope = NumericalScope::Qwen35GgufF16ProjectionsRnFragmentFfnM1to8V1;
    declaration
}

#[test]
fn rn_fragment_scope_requires_strict_reference_and_every_projection_class() {
    for serial in [false, true] {
        let pair = fragment_pair(if serial {
            Pair::with_serial_width(3)
        } else {
            Pair::new()
        });
        let declaration = fragment_declaration(&pair);
        assert_eq!(
            serde_json::to_value(declaration.scope).unwrap(),
            json!("qwen35_gguf_f16_projections_rn_fragment_ffn_m1to8_v1")
        );
        let mut options = args(&pair, &declaration);
        if serial {
            options.comparison_shape = ComparisonShape::SerialToSerial;
        }
        let (report, code) = compare(&options);
        assert_eq!(code, 0, "{report:#}");
        assert_eq!(report["release_approved"], false);
        assert_eq!(report["provenance"]["same_binary"], true);
        assert_eq!(
            report["provenance"]["execution_qualification"]["actual_route_coverage"],
            "requires_independent_completed_wave_node_evidence"
        );
        for group in ["all", "prefill", "decode"] {
            assert_eq!(report["summary"][group]["quality_passed"], true);
        }
    }
    let pair = fragment_pair(Pair::new());
    let declaration = fragment_declaration(&pair);
    for index in 0..declaration.node_changes.len() {
        let mut incomplete = declaration.clone();
        incomplete.node_changes.remove(index);
        let (report, code) = compare(&args(&pair, &incomplete));
        assert_insufficient(&report, code, "requires FFN, GDN and causal");
    }
    let mut wrong_reference = declaration;
    wrong_reference.reference.numerical_profile =
        "qwen3_5.f32-master.gguf-f16-projections".parse().unwrap();
    assert_ne!(compare(&args(&pair, &wrong_reference)).1, 0);
}

#[test]
fn rn_fragment_scope_rejects_old_scope_other_binary_and_undeclared_head_change() {
    let mut pair = fragment_pair(Pair::new());
    let declaration = fragment_declaration(&pair);
    let mut old_scope = declaration.clone();
    old_scope.scope = NumericalScope::Qwen35GgufF16ProjectionsV1;
    let (report, code) = compare(&args(&pair, &old_scope));
    assert_insufficient(&report, code, "outside the declared numerical scope");
    let mut other_binary = declaration.clone();
    other_binary.candidate.execution.binary_sha256 = sha256(b"other executable");
    let (report, code) = compare(&args(&pair, &other_binary));
    assert_insufficient(&report, code, "same executable");
    edit_arm_receipts(&mut pair.candidate, |batch| {
        batch.nodes[3].provider_implementation_fingerprint = sha256(b"undeclared head");
    });
    let (report, code) = compare(&args(&pair, &declaration));
    assert_insufficient(&report, code, "undeclared numerical transition");
}

#[test]
fn rn_fragment_scope_keeps_prefill_decode_and_full_vocabulary_quality_limits() {
    for decision in [0, 1] {
        let mut pair = fragment_pair(Pair::new());
        let declaration = fragment_declaration(&pair);
        pair.candidate
            .set_logits("owner-0", decision, &[0.0, 2.0, 1.0, -8.0]);
        let (report, code) = compare(&args(&pair, &declaration));
        assert_eq!(code, 1, "{report:#}");
        assert_eq!(report["evidence_complete"], true);
        assert_eq!(report["quality_passed"], false);
        assert_eq!(report["summary"]["all"]["quality_passed"], false);
        let phase = if decision == 0 { "prefill" } else { "decode" };
        assert_eq!(report["summary"][phase]["quality_passed"], false);
    }
}
