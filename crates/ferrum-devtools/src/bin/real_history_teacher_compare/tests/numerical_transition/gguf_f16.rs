//! Typed evidence-protocol fixtures, not real model quality measurements.
use super::*;
use ferrum_interfaces::vnext::*;

const CANDIDATE: &str = "qwen3_5.f32-master.gguf-f16-projections";
const PAIRS: [(&str, &str); 3] = [
    (
        DENSE_SWIGLU_OPERATION_ID,
        DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID,
    ),
    (
        GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
        GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID,
    ),
    (
        CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID,
        CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID,
    ),
];
fn full_pair() -> Pair {
    full_pair_from(Pair::new())
}
fn full_pair_from(pair: Pair) -> Pair {
    let mut pair = migrate_pair(pair, 1);
    for (arm, candidate) in [(&mut pair.reference, false), (&mut pair.candidate, true)] {
        let profile = if candidate {
            CANDIDATE
        } else {
            "qwen3_5.f32-master"
        };
        arm.manifest.identity.as_mut().unwrap().numerical_profile = profile.into();
        arm.manifest.configuration["numerical_execution"] = json!({"require":profile});
        edit_arm_receipts(arm, |batch| {
            let template = batch.nodes[0].clone();
            let mut head = batch.nodes[1].clone();
            head.operation_id = LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID.into();
            let mut nodes = Vec::new();
            for (index, (strict, rounded)) in PAIRS.iter().enumerate() {
                let mut node = template.clone();
                node.node_id = format!("node.projection.{index}");
                node.operation_id = if candidate { *rounded } else { *strict }.into();
                node.provider_id = format!(
                    "provider.{}.projection.{index}",
                    if candidate { "rn-f16" } else { "strict" }
                );
                node.provider_implementation_fingerprint = sha256(node.provider_id.as_bytes());
                for part in &mut node.participants {
                    part.identity.node_invocation_id =
                        Some(serde_json::from_value(json!(100 + index as u64)).unwrap());
                    part.identity.span_id = SpanId::new(format!(
                        "span.projection.{index}.{}",
                        part.participant_index
                    ))
                    .unwrap();
                }
                nodes.push(node);
            }
            nodes.push(head); // Original node.1 remains the exact output readback binding.
            batch.nodes = nodes;
        });
    }
    pair
}

#[test]
fn gguf_f16_serial_to_serial_keeps_exact_scope_and_original_quality_gates() {
    let mut pair = full_pair_from(Pair::with_serial_width(3));
    let declaration = declared(&pair);
    let mut options = args(&pair, &declaration);
    options.comparison_shape = ComparisonShape::SerialToSerial;
    let (report, code) = compare(&options);
    assert_eq!(code, 0, "{report:#}");
    assert_eq!(
        report["provenance"]["logical_decode_width"],
        json!({"reference":1,"candidate":1})
    );
    assert_eq!(
        report["provenance"]["execution_qualification"]["changed_node_count"],
        3
    );
    assert_eq!(report["release_approved"], false);
    for group in ["all", "prefill", "decode"] {
        assert_eq!(
            report["summary"][group]["checks"].as_array().unwrap().len(),
            4
        );
    }
    pair.candidate
        .set_logits("owner-0", 1, &[0.0, 2.0, 1.0, -8.0]);
    let (report, code) = compare(&options);
    assert_eq!(code, 1, "{report:#}");
    assert_eq!(report["evidence_complete"], true);
    assert_eq!(report["summary"]["decode"]["quality_passed"], false);
    let mut wrong = declaration;
    wrong.node_changes.pop();
    let mut options = args(&pair, &wrong);
    options.comparison_shape = ComparisonShape::SerialToSerial;
    let (report, code) = compare(&options);
    assert_insufficient(&report, code, "requires FFN, GDN and causal");
}
fn declared(pair: &Pair) -> NumericalTransition {
    let mut d = declaration(pair);
    d.scope = NumericalScope::Qwen35GgufF16ProjectionsV1;
    d
}

#[test]
fn gguf_f16_transition_checks_every_projection_and_preserves_quality_gates() {
    let pair = full_pair();
    let d = declared(&pair);
    assert_eq!(
        serde_json::to_value(d.scope).unwrap(),
        json!("qwen35_gguf_f16_projections_v1")
    );
    let (report, code) = compare(&args(&pair, &d));
    assert_eq!(code, 0, "{report:#}");
    assert_eq!(
        report["provenance"]["execution_qualification"]["changed_node_count"],
        3
    );
    assert_eq!(
        report["provenance"]["execution_qualification"]["ordered_node_count"],
        4
    );
    assert_eq!(report["release_approved"], false);
    assert_eq!(report["provenance"]["same_binary"], true);
    assert_eq!(
        report["provenance"]["execution_qualification"]["actual_route_coverage"],
        "requires_independent_completed_wave_node_evidence"
    );
    for group in ["all", "prefill", "decode"] {
        assert_eq!(
            report["summary"][group]["checks"].as_array().unwrap().len(),
            4
        );
        assert_eq!(report["summary"][group]["quality_passed"], true);
    }
}

#[test]
fn gguf_f16_transition_cannot_hide_any_strict_projection_or_reverse_a_pair() {
    let mut pair = full_pair();
    for arm in [&mut pair.reference, &mut pair.candidate] {
        edit_arm_receipts(arm, |batch| {
            let mut extra = batch.nodes[0].clone();
            extra.node_id = "node.extra.strict-ffn".into();
            extra.operation_id = DENSE_SWIGLU_OPERATION_ID.into();
            extra.provider_id = "provider.unchanged.strict-ffn".into();
            extra.provider_implementation_fingerprint = sha256(b"extra unchanged strict FFN");
            for part in &mut extra.participants {
                part.identity.node_invocation_id =
                    Some(serde_json::from_value(json!(500)).unwrap());
                part.identity.span_id =
                    SpanId::new(format!("span.extra.{}", part.participant_index)).unwrap();
            }
            batch.nodes.push(extra);
        });
    }
    let (report, code) = compare(&args(&pair, &declared(&pair)));
    assert_insufficient(&report, code, "omitted or changed a strict projection");
    let pair = full_pair();
    let base = declared(&pair);
    for index in 0..3 {
        let mut d = base.clone();
        d.node_changes.remove(index);
        let (report, code) = compare(&args(&pair, &d));
        assert_insufficient(&report, code, "requires FFN, GDN and causal");
    }
    let mut d = base;
    d.node_changes[1].candidate.operation_id =
        CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID.into();
    let (report, code) = compare(&args(&pair, &d));
    assert_insufficient(
        &report,
        code,
        "outside dense SwiGLU Q8 scope or the explicit all-projection scope",
    );
}

#[test]
fn gguf_f16_transition_rejects_untouched_node_changes_wrong_elf_and_wrong_profile() {
    let mut pair = full_pair();
    let original = declared(&pair);
    edit_arm_receipts(&mut pair.candidate, |batch| {
        batch.nodes[3].provider_implementation_fingerprint = sha256(b"head implementation changed");
    });
    let (report, code) = compare(&args(&pair, &original));
    assert_insufficient(&report, code, "undeclared numerical transition");
    let (report, code) = compare(&args(&pair, &declared(&pair)));
    assert_insufficient(
        &report,
        code,
        "outside dense SwiGLU Q8 scope or the explicit all-projection scope",
    );
    let pair = full_pair();
    let mut d = declared(&pair);
    d.candidate.execution.binary_sha256 = sha256(b"another ELF");
    let (report, code) = compare(&args(&pair, &d));
    assert_insufficient(&report, code, "same executable");
    let mut d = declared(&pair);
    d.candidate.numerical_profile = "qwen3_5.f32-master.q8-swiglu".parse().unwrap();
    let (report, code) = compare(&args(&pair, &d));
    assert_insufficient(&report, code, "outside the declared numerical scope");
}

#[test]
fn gguf_f16_transition_still_fails_original_prefill_decode_kl_nll_budgets() {
    for decision in [0, 1] {
        let mut pair = full_pair();
        let d = declared(&pair);
        pair.candidate
            .set_logits("owner-0", decision, &[0.0, 2.0, 1.0, -8.0]);
        let (report, code) = compare(&args(&pair, &d));
        assert_eq!(code, 1, "{report:#}");
        assert_eq!(report["evidence_complete"], true);
        assert_eq!(report["quality_passed"], false);
        let group = if decision == 0 { "prefill" } else { "decode" };
        assert_eq!(
            report["summary"][group]["checks"].as_array().unwrap().len(),
            4
        );
        assert_eq!(report["summary"][group]["quality_passed"], false);
        assert_eq!(report["summary"]["all"]["quality_passed"], false);
    }
}

#[path = "hybrid.rs"]
mod hybrid;

#[path = "rn_fragment.rs"]
mod rn_fragment;
