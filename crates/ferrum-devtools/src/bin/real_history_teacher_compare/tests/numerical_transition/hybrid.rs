//! This is evidence-protocol coverage, never a measured model qualification.
use super::*;

fn hybrid_pair(pair: Pair) -> Pair {
    let mut pair = full_pair_from(pair);
    let arm = &mut pair.candidate;
    let profile = "qwen3_5.f32-master.gguf-f16-attention.q8-residual2-ffn-m2to8";
    arm.manifest.identity.as_mut().unwrap().numerical_profile = profile.into();
    arm.manifest.configuration["numerical_execution"] = json!({"require": profile});
    edit_arm_receipts(arm, |batch| {
        let ffn = &mut batch.nodes[0];
        assert_eq!(ffn.operation_id, DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID);
        ffn.operation_id = DENSE_SWIGLU_Q8_RESIDUAL2_FFN_M2_TO8_OPERATION_ID.into();
        ffn.provider_id = "provider.residual2.ffn-m2to8".into();
        ffn.provider_implementation_fingerprint = sha256(ffn.provider_id.as_bytes());
    });
    pair
}
fn hybrid_declared(pair: &Pair) -> NumericalTransition {
    let mut d = declaration(pair);
    d.scope = NumericalScope::Qwen35GgufF16AttentionQ8Residual2FfnM2to8V1;
    d
}

#[test]
fn hybrid_rn_attention_teacher_serial_and_batched_require_all_three_classes() {
    for serial in [false, true] {
        let pair = hybrid_pair(if serial {
            Pair::with_serial_width(3)
        } else {
            Pair::new()
        });
        let d = hybrid_declared(&pair);
        assert_eq!(
            serde_json::to_value(d.scope).unwrap(),
            json!("qwen35_gguf_f16_attention_q8_residual2_ffn_m2to8_v1")
        );
        let mut options = args(&pair, &d);
        if serial {
            options.comparison_shape = ComparisonShape::SerialToSerial;
        }
        let (report, code) = compare(&options);
        assert_eq!(code, 0, "{report:#}");
        assert_eq!(report["release_approved"], false);
        assert_eq!(report["provenance"]["same_binary"], true);
        assert_eq!(
            report["provenance"]["execution_qualification"]["changed_node_count"],
            3
        );
        assert_eq!(
            report["provenance"]["execution_qualification"]["actual_route_coverage"],
            "requires_independent_completed_wave_node_evidence"
        );
        assert_eq!(
            report["provenance"]["execution_qualification"]["scope"],
            "declared_rn_f16_attention_residual2_ffn_fixed_history_full_vocabulary_quality_only"
        );
    }
    let pair = hybrid_pair(Pair::new());
    let d = hybrid_declared(&pair);
    for index in 0..3 {
        let mut incomplete = d.clone();
        incomplete.node_changes.remove(index);
        let (report, code) = compare(&args(&pair, &incomplete));
        assert_insufficient(&report, code, "requires FFN, GDN and causal");
    }
}

#[test]
fn hybrid_rn_attention_teacher_rejects_full_rn_scope_other_elf_and_head_change() {
    let mut pair = hybrid_pair(Pair::new());
    let d = hybrid_declared(&pair);
    let mut wrong_scope = d.clone();
    wrong_scope.scope = NumericalScope::Qwen35GgufF16ProjectionsV1;
    let (report, code) = compare(&args(&pair, &wrong_scope));
    assert_insufficient(&report, code, "outside the declared numerical scope");
    let mut other_binary = d.clone();
    other_binary.candidate.execution.binary_sha256 = sha256(b"other executable");
    let (report, code) = compare(&args(&pair, &other_binary));
    assert_insufficient(&report, code, "same executable");
    edit_arm_receipts(&mut pair.candidate, |batch| {
        batch.nodes[3].provider_implementation_fingerprint = sha256(b"unrelated head change");
    });
    let (report, code) = compare(&args(&pair, &d));
    assert_insufficient(&report, code, "undeclared numerical transition");
}

#[test]
fn hybrid_rn_attention_teacher_still_rejects_original_prefill_and_decode_quality_failures() {
    for decision in [0, 1] {
        let mut pair = hybrid_pair(Pair::new());
        let d = hybrid_declared(&pair);
        pair.candidate
            .set_logits("owner-0", decision, &[0.0, 2.0, 1.0, -8.0]);
        let (report, code) = compare(&args(&pair, &d));
        assert_eq!(code, 1, "{report:#}");
        assert_eq!(report["evidence_complete"], true);
        assert_eq!(report["quality_passed"], false);
        assert_eq!(report["summary"]["all"]["quality_passed"], false);
        let group = if decision == 0 { "prefill" } else { "decode" };
        assert_eq!(report["summary"][group]["quality_passed"], false);
        assert_eq!(
            report["summary"][group]["checks"].as_array().unwrap().len(),
            4
        );
    }
}
