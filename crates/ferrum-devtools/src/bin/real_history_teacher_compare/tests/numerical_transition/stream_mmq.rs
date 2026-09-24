use super::*;

const PROFILE: &str = "qwen3_5.f32-master.q8-gate-up-stream-mmq";
const OPERATION: &str = "operation.dense_swiglu.q8-gate-up-stream-mmq-f32scale";

fn stream_pair() -> Pair {
    let mut pair = Pair::with_width(8);
    for (arm, candidate) in [(&mut pair.reference, false), (&mut pair.candidate, true)] {
        let profile = if candidate {
            PROFILE
        } else {
            "qwen3_5.f32-master"
        };
        let identity = arm.manifest.identity.as_mut().unwrap();
        identity.numerical_profile = profile.into();
        if candidate {
            identity.family_fingerprint = sha256(b"stream-mmq-family");
            identity.program_fingerprint = sha256(b"stream-mmq-program");
            identity.resolved_plan_fingerprint = sha256(b"stream-mmq-resolved");
        }
        arm.manifest.configuration["numerical_execution"] = json!({"require":profile});
        edit_arm_receipts(arm, |batch| {
            if candidate {
                batch.plan_id = "plan.stream-mmq".into();
                batch.plan_hash = sha256(b"stream-mmq-plan");
            }
            let node = &mut batch.nodes[0];
            node.operation_id = if candidate {
                OPERATION
            } else {
                "operation.dense_swiglu"
            }
            .into();
            node.provider_id = if candidate {
                "provider.stream-mmq"
            } else {
                "provider.ffn"
            }
            .into();
            node.provider_implementation_fingerprint = sha256(node.provider_id.as_bytes());
        });
    }
    pair
}

fn stream_declaration(pair: &Pair) -> NumericalTransition {
    NumericalTransition {
        scope: NumericalScope::DenseSwiGluQ8GateUpStreamMmqV1,
        ..declaration(pair)
    }
}

#[test]
fn stream_mmq_requires_exact_separate_scope_and_preserves_full_eight_owner_evidence() {
    let pair = stream_pair();
    let (report, code) = pair.compare();
    assert_insufficient(&report, code, "model/numerical/KV");
    let d = stream_declaration(&pair);
    assert_eq!(
        d.reference.execution.binary_sha256,
        d.candidate.execution.binary_sha256
    );
    let (report, code) = compare(&args(&pair, &d));
    assert_eq!(code, 0, "{report:#}");
    assert_eq!(report["release_approved"], false);
    assert_eq!(
        report["provenance"]["execution_qualification"]["declaration"]["content"]["scope"],
        "dense_swi_glu_q8_gate_up_stream_mmq_v1"
    );
    assert_eq!(report["summary"]["prefill"]["decision_count"], 8);
    assert_eq!(report["summary"]["decode"]["decision_count"], 16);
    // This fixture checks identity/readback contracts, not CUDA path coverage.
    assert_eq!(
        report["provenance"]["execution_qualification"]["actual_route_coverage"],
        "requires_independent_completed_wave_node_evidence"
    );
}

#[test]
fn stream_mmq_rejects_old_policy_scope_operation_and_wrong_pin() {
    let pair = stream_pair();
    for scope in [
        NumericalScope::DenseSwiGluQ8F32ScaleV1,
        NumericalScope::DenseSwiGluQ8F32ScaleInputSumV1,
    ] {
        let d = NumericalTransition {
            scope,
            ..stream_declaration(&pair)
        };
        let (report, code) = compare(&args(&pair, &d));
        assert_insufficient(&report, code, "profile pair is outside");
    }
    let mut d = stream_declaration(&pair);
    d.node_changes[0].candidate.operation_id = "operation.dense_swiglu.q8-f32scale".into();
    let (report, code) = compare(&args(&pair, &d));
    assert_insufficient(&report, code, "node transition is outside");
    let mut d = stream_declaration(&pair);
    d.candidate.program_fingerprint = sha256(b"wrong program");
    let (report, code) = compare(&args(&pair, &d));
    assert_eq!(code, 2, "{report:#}");
    assert_eq!(report["evidence_complete"], false);
}

#[test]
fn stream_mmq_does_not_authorize_undeclared_non_ffn_provider_changes() {
    let mut pair = stream_pair();
    let d = stream_declaration(&pair);
    edit_arm_receipts(&mut pair.candidate, |batch| {
        let node = batch.nodes.last_mut().unwrap();
        node.provider_implementation_fingerprint = sha256(b"unrelated provider changed");
    });
    let (report, code) = compare(&args(&pair, &d));
    assert_insufficient(&report, code, "undeclared numerical transition");
    // Even a pinned declaration cannot authorize a different operation scope.
    let d = stream_declaration(&pair);
    let (report, code) = compare(&args(&pair, &d));
    assert_insufficient(&report, code, "node transition is outside");
}

#[test]
fn stream_mmq_strict_prefill_cannot_dilute_decode_quality_failure() {
    let mut pair = stream_pair();
    let d = stream_declaration(&pair);
    pair.candidate
        .set_logits("owner-7", 2, &[0.0, 2.0, 1.0, -8.0]);
    let (report, code) = compare(&args(&pair, &d));
    assert_eq!(code, 1, "{report:#}");
    assert_eq!(report["evidence_complete"], true);
    assert_eq!(report["summary"]["prefill"]["quality_passed"], true);
    assert_eq!(report["summary"]["decode"]["quality_passed"], false);
    assert_eq!(report["quality_passed"], false);
}
