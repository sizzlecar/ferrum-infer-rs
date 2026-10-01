//! Actual independently qualified sources may overlap incidentally. A narrower
//! successor keeps the original child while adding its new complete populations.
use super::*;

fn ordinary_population(session: &CalibrationSession) -> StructuredPreparedOwnerBlockDeclarationV8 {
    let mut declared = full_population(session);
    // A pair of ordinary cohorts offers prefill at Length and before Length,
    // followed by one incidental terminal decode. Eight pairs fill each of
    // discovery/Fit/Residual/Qualification's original 24-offer blocks.
    declared.cohort_plan.phases = std::array::from_fn(|phase| {
        (0..[32, 16, 16][phase])
            .map(|ordinal| CohortV2 {
                manifest_case: ordinal as u32,
                repetition: 0,
                requests: vec![
                    CohortRequestV2 {
                        manifest_prompt: 0,
                        maximum_output: 1 + (ordinal % 2) as u64,
                    };
                    2
                ],
            })
            .collect()
    });
    declared.prefix_plan.phases =
        std::array::from_fn(|phase| vec![None; declared.cohort_plan.phases[phase].len()]);
    declared.cohort_manifest_payload = serde_json::value::to_raw_value(&serde_json::json!({
        "prompt":"test", "outputs":["cli_text","completions_sse"],
        "maximum_outputs":[1,2], "prefix":"ordinary",
        "sampling":"installed_full_vocabulary_top_k"
    }))
    .unwrap();
    declared
}

#[tokio::test]
async fn startup_series_keeps_broad_original_decode_and_appends_qualified_prefill() {
    let (mut session, executor) = automatic_session().await;
    executor
        .native_structured_submission
        .store(true, Ordering::Release);
    executor.enable_structured_query_route();
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    let mut clean = full_population(&session);
    for prefix in clean.prefix_plan.phases.iter_mut().flatten().flatten() {
        for slot in &mut prefix.slots {
            slot.token_ids = vec![TokenId::new(10)];
            slot.token_bytes = vec![b"a".to_vec()];
        }
    }
    clean.cohort_manifest_payload = serde_json::value::to_raw_value(&serde_json::json!({
        "prompt":"test", "maximum_output":3, "prefix_cases":["empty"],
        "outputs":["cli_text","completions_sse"],
        "sampling":"installed_full_vocabulary_top_k"
    }))
    .unwrap();
    let declarations = [
        full_population(&session),
        ordinary_population(&session),
        clean,
    ];
    let requests: usize = declarations
        .iter()
        .flat_map(|d| &d.cohort_plan.phases)
        .flatten()
        .map(|c| c.requests.len())
        .sum();
    let waves: usize = declarations.iter().map(|d| d.maximum_offered_waves).sum();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(45);
    session
        .begin_startup_owner_series(declarations.len(), deadline)
        .unwrap();
    let mut budget = ProbeExecutionBudget::new(
        deadline,
        NonZeroUsize::new(requests).unwrap(),
        NonZeroUsize::new(waves).unwrap(),
    );
    let options = ProbeCohortSettings {
        prefill_plan:
            crate::continuous_engine::inner::calibration::cohort_driver::ProbePrefillPlan::Joint,
        prefill_chunk: NonZeroU32::MIN,
        decode_route: CalibrationDecodeRoute::FullLogits,
        reset_token_policy: false,
    };
    let mut original_decode = None;
    let mut installed_epoch = 0;
    for (source, declared) in declarations.into_iter().enumerate() {
        let maximums: [Vec<_>; 3] = std::array::from_fn(|phase| {
            declared.cohort_plan.phases[phase]
                .iter()
                .map(|cohort| cohort.requests[0].maximum_output as usize)
                .collect()
        });
        session
            .begin_prepared_owner_source(declared, CostProfileLoadLimits::default())
            .await
            .unwrap();
        for (pass, maximums) in maximums.iter().enumerate() {
            for (ordinal, &maximum) in maximums.iter().enumerate() {
                session.begin_prepared_owner_cohort(pass, ordinal).unwrap();
                let mut requests = probe_requests(&session);
                for request in &mut requests {
                    request.request.sampling_params.max_tokens = maximum;
                }
                session
                    .run_probe_cohort(requests, options, &mut budget)
                    .await
                    .unwrap();
                session.end_prepared_owner_cohort().unwrap();
            }
        }
        session.freeze_cost_model().await.unwrap();
        let observed = session.prepared_owner_capture.as_ref().unwrap().audit();
        assert!(!observed.owners.is_empty());
        assert!(
            observed
                .owners
                .iter()
                .all(|owner| owner.qualified && owner.failure.is_none()),
            "every original numerical population must actually qualify: {observed:#?}"
        );
        let prior_snapshot = runtime.snapshot();
        let prior_receipt = runtime.profile_receipt();
        let prior_feedback =
            serde_json::to_value(runtime.audit_snapshot()).unwrap()["structured_feedback"].clone();
        let activation = session.activate_prepared_owner_source().await;
        if source == 2 {
            assert!(activation
                .unwrap_err()
                .to_string()
                .contains("no effective catalog extension"));
            assert!(Arc::ptr_eq(
                prior_snapshot.as_ref().unwrap(),
                &runtime.snapshot().unwrap()
            ));
            assert_eq!(prior_receipt, runtime.profile_receipt());
            assert_eq!(
                serde_json::to_value(runtime.audit_snapshot()).unwrap()["structured_feedback"],
                prior_feedback
            );
        } else {
            let epoch = activation.unwrap();
            assert!(epoch > installed_epoch);
            installed_epoch = epoch;
        }
        let children = runtime.startup_series_children_for_test().unwrap();
        if source == 0 {
            assert_eq!(children.len(), 1);
            original_decode = Some(children[0].clone());
            let original_receipt = runtime.profile_receipt().unwrap();
            let mut unchanged = original_receipt.clone();
            runtime
                .snapshot()
                .unwrap()
                .check_startup_receipt_subset_for_test(&mut unchanged, &children, &[true])
                .unwrap();
            assert_eq!(
                unchanged, original_receipt,
                "keeping the original inventory preserves its receipt bytes"
            );
            let mut invalid = original_receipt;
            invalid.structured_whole_wave_v2.as_mut().unwrap().children[0].phases[0].members += 1;
            let invalid_before = invalid.clone();
            let error = runtime
                .snapshot()
                .unwrap()
                .check_startup_receipt_subset_for_test(&mut invalid, &children, &[false])
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("original source binding differs"),
                "even an omitted child must retain its complete original phase receipt: {error}"
            );
            assert_eq!(
                invalid, invalid_before,
                "failed validation cannot rewrite the original receipt"
            );
        } else {
            let original = original_decode.as_ref().unwrap();
            let retained = children
                .iter()
                .find(|child| child.same_population(original))
                .unwrap();
            assert_eq!(
                serde_json::to_value(retained.provenance()).unwrap(),
                serde_json::to_value(original.provenance()).unwrap(),
                "retained model, original phases, clocks, source and TTL must not be rebound"
            );
            assert!(children
                .iter()
                .any(|child| child.owner().role == StructuredWaveRoleV2::Prefill));
            let after_feedback = serde_json::to_value(runtime.audit_snapshot()).unwrap()
                ["structured_feedback"]
                .clone();
            assert!(after_feedback["revoked"].is_null());
            for old_scope in prior_feedback["scopes"].as_array().unwrap() {
                let new_scope = after_feedback["scopes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|scope| scope["signature"] == old_scope["signature"])
                    .unwrap();
                assert_eq!(
                    old_scope, new_scope,
                    "preserved source keeps its accumulated feedback"
                );
            }
            if source == 1 {
                assert!(observed
                    .owners
                    .iter()
                    .any(|owner| owner.owner.role == StructuredWaveRoleV2::OrdinaryDecode));
                let receipt = runtime.profile_receipt().unwrap();
                let selected = receipt.structured_whole_wave_v2.as_ref().unwrap();
                assert_eq!(selected.child_count, selected.children.len());
                assert!(selected.children.iter().all(|record| {
                    let child = children.iter().find(|child| child.domain_signature() == &record.domain_signature).unwrap();
                    child.owner().role == StructuredWaveRoleV2::Prefill
                        && record.capture_identity_sha256 == child.provenance().capture_identity
                        && record.parameters_sha256 == child.provenance().parameters_sha256
                }), "new receipt must describe accepted new children, never substitute the old decode's source");
                assert_eq!(receipt.offered_samples as u64, observed.offered);
            }
        }
        session.retire_startup_owner_source().await.unwrap();
    }
    assert_eq!(budget.requests_remaining(), 0);
    assert_eq!(budget.attempts_remaining(), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), waves);
    session.finish_startup_owner_series().unwrap();
    runtime.shutdown().await.unwrap();
}
