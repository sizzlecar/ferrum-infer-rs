use super::*;

fn input_sum_pair() -> Pair {
    let mut pair = migrated_pair(1);
    let profile = "qwen3_5.f32-master.q8-swiglu-input-sum";
    let identity = pair.candidate.manifest.identity.as_mut().unwrap();
    identity.numerical_profile = profile.into();
    identity.family_fingerprint = sha256(b"q8 input sum family");
    identity.program_fingerprint = sha256(b"q8 input sum program");
    identity.resolved_plan_fingerprint = sha256(b"q8 input sum resolved");
    pair.candidate.manifest.configuration["numerical_execution"] = json!({"require": profile});
    edit_arm_receipts(&mut pair.candidate, |batch| {
        batch.plan_id = "plan.q8-input-sum-ffn".into();
        batch.plan_hash = sha256(b"q8 input sum plan");
        let node = &mut batch.nodes[0];
        node.operation_id = "operation.dense_swiglu.q8-f32scale-input-sum".into();
        node.provider_id = "provider.q8-input-sum-ffn".into();
        node.provider_implementation_fingerprint = sha256(node.provider_id.as_bytes());
    });
    pair
}

fn input_sum_declaration(pair: &Pair) -> NumericalTransition {
    NumericalTransition {
        scope: NumericalScope::DenseSwiGluQ8F32ScaleInputSumV1,
        ..declaration(pair)
    }
}

#[test]
fn input_sum_requires_its_own_exact_scope_and_preserves_default_identity_checks() {
    let pair = input_sum_pair();
    let (report, code) = pair.compare();
    assert_insufficient(&report, code, "model/numerical/KV");
    let d = input_sum_declaration(&pair);
    let (report, code) = compare(&args(&pair, &d));
    assert_eq!(code, 0, "{report:#}");
    assert_eq!(report["evidence_complete"], true);
    assert_eq!(report["release_approved"], false);
    assert_eq!(
        report["provenance"]["execution_qualification"]["declaration"]["content"]["scope"],
        "dense_swi_glu_q8_f32_scale_input_sum_v1"
    );
}

#[test]
fn input_sum_and_quantized_sum_scopes_cannot_authorize_each_others_profiles_or_operations() {
    let old = migrated_pair(1);
    let new = input_sum_pair();
    for (pair, mut d) in [
        (&new, declaration(&new)),
        (&old, input_sum_declaration(&old)),
    ] {
        let a = args(pair, &d);
        let error = crate::numerical_transition::DeclaredNumericalTransition::read(
            a.numerical_profile_transition.as_deref().unwrap(),
        )
        .err()
        .unwrap();
        assert!(format!("{error:#}").contains("profile pair is outside"));

        // A matching profile cannot conceal the other operation's arithmetic.
        d.scope = if pair
            .candidate
            .manifest
            .identity
            .as_ref()
            .unwrap()
            .numerical_profile
            == "qwen3_5.f32-master.q8-swiglu-input-sum"
        {
            d.node_changes[0].candidate.operation_id = "operation.dense_swiglu.q8-f32scale".into();
            NumericalScope::DenseSwiGluQ8F32ScaleInputSumV1
        } else {
            d.node_changes[0].candidate.operation_id =
                "operation.dense_swiglu.q8-f32scale-input-sum".into();
            NumericalScope::DenseSwiGluQ8F32ScaleV1
        };
        let a = args(pair, &d);
        let error = crate::numerical_transition::DeclaredNumericalTransition::read(
            a.numerical_profile_transition.as_deref().unwrap(),
        )
        .err()
        .unwrap();
        assert!(format!("{error:#}").contains("node transition is outside"));
    }
}

#[test]
fn input_sum_scope_does_not_relax_the_original_prefill_quality_budget() {
    let mut pair = input_sum_pair();
    let d = input_sum_declaration(&pair);
    pair.candidate
        .set_logits("owner-0", 0, &[0.0, 2.0, 1.0, -8.0]);
    let (report, code) = compare(&args(&pair, &d));
    assert_eq!(code, 1, "{report:#}");
    assert_eq!(report["evidence_complete"], true);
    assert_eq!(report["summary"]["prefill"]["quality_passed"], false);
    assert_eq!(report["quality_passed"], false);
}
