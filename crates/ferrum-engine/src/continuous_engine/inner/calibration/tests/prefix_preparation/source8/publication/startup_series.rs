//! Two original CPU source8 populations, one global execution budget, and an
//! unfinished successor. Every installed child comes from complete real waves.
use super::*;

mod coverage_append;
mod deadline;
mod inventory;

#[tokio::test]
async fn startup_series_merges_complete_sources_and_preserves_them_on_incomplete_successor() {
    let (mut session, executor) = automatic_session().await;
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(45);
    let mut declarations = [
        full_population(&session),
        full_population(&session),
        full_population(&session),
    ];
    // This fixture deliberately uses ExactOwnerV1: row count is an actual
    // statistical population dimension. Changing a transport contract alone
    // does not establish a different installed numerical host policy.
    for pass in 0..3 {
        for cohort in &mut declarations[1].cohort_plan.phases[pass] {
            cohort.requests.truncate(1);
        }
        for prefix in declarations[1].prefix_plan.phases[pass]
            .iter_mut()
            .flatten()
        {
            prefix.slots.truncate(1);
        }
    }
    assert_eq!(
        declarations[1].population.population_policy(),
        StructuredPopulationPolicyV1::ExactOwnerV1
    );
    declarations[1].cohort_manifest_payload = serde_json::value::to_raw_value(&serde_json::json!({
        "prompt":"test", "maximum_output":3,
        "prefix_cases":["empty","pending","pending_empty","empty_pending"],
        "outputs":["cli_text"], "width":1, "sampling":"installed_full_vocabulary_top_k"
    }))
    .unwrap();
    session
        .begin_startup_owner_series(declarations.len(), deadline)
        .unwrap();
    assert!(runtime.begin_automatic_calibration().is_err());
    assert!(session.begin_startup_owner_series(1, deadline).is_err());
    let mut budget = ProbeExecutionBudget::new(
        deadline,
        NonZeroUsize::new(96).unwrap(),
        NonZeroUsize::new(192).unwrap(),
    );
    let options = ProbeCohortSettings {
        prefill_plan:
            crate::continuous_engine::inner::calibration::cohort_driver::ProbePrefillPlan::Joint,
        prefill_chunk: NonZeroU32::MIN,
        decode_route: CalibrationDecodeRoute::FullLogits,
        reset_token_policy: false,
    };
    let mut original = Vec::new();
    let mut previous_epoch = 0;
    for (source, declared) in declarations.into_iter().enumerate() {
        session
            .begin_prepared_owner_source(declared, CostProfileLoadLimits::default())
            .await
            .unwrap();
        if source == 2 {
            let previous = runtime.snapshot().unwrap();
            let receipt = runtime.profile_receipt().unwrap();
            assert!(session.activate_prepared_owner_source().await.is_err());
            assert!(session.retire_startup_owner_source().await.is_err());
            assert!(Arc::ptr_eq(&previous, &runtime.snapshot().unwrap()));
            assert_eq!(receipt, runtime.profile_receipt().unwrap());
            break;
        }
        assert!(session.retire_startup_owner_source().await.is_err());
        for pass in 0..3 {
            for ordinal in 0..[16, 8, 8][pass] {
                session.begin_prepared_owner_cohort(pass, ordinal).unwrap();
                let mut requests = probe_requests(&session);
                if source == 1 {
                    // Execute the declared B1 cohort, including its original
                    // token-prefix slot and real CLI output contract.
                    requests.truncate(1);
                }
                session
                    .run_probe_cohort(requests, options, &mut budget)
                    .await
                    .unwrap();
                session.end_prepared_owner_cohort().unwrap();
                if source == 1 {
                    // The exact worker barrier includes this preparation and
                    // both ordinary waves. Detect revocation at its source,
                    // before the next publication could hide it by replacing.
                    session.freeze_cost_model().await.unwrap();
                    let audit = serde_json::to_value(runtime.audit_snapshot()).unwrap();
                    let feedback = &audit["structured_feedback"];
                    assert!(
                        feedback["revoked"].is_null(),
                        "pass={pass} cohort={ordinal}: {feedback:#}"
                    );
                    assert_eq!(feedback["uncomparable_observations"], 0);
                    assert!(
                        feedback["outside_preparation_observations"]
                            .as_u64()
                            .unwrap()
                            > 0
                    );
                    let retained = runtime.startup_series_children_for_test().unwrap();
                    for (domain, _) in &original {
                        assert!(retained.iter().any(|child| child.domain_signature() == domain),
                            "old source lost before activation at pass={pass} cohort={ordinal}: {feedback:#}");
                    }
                }
            }
        }
        let observed = session.prepared_owner_capture.as_ref().unwrap().audit();
        let qualified: Vec<_> = observed
            .owners
            .iter()
            .filter(|owner| owner.qualified)
            .collect();
        assert!(
            !qualified.is_empty(),
            "complete source must qualify: {observed:#?}"
        );
        assert!(qualified
            .iter()
            .all(|owner| owner.owner.rows == if source == 0 { 2 } else { 1 }));
        let epoch = session.activate_prepared_owner_source().await.unwrap();
        assert!(epoch > previous_epoch);
        previous_epoch = epoch;
        let children = runtime.startup_series_children_for_test().unwrap();
        if source == 0 {
            assert!(!children.is_empty());
            assert!(children
                .iter()
                .all(|child| child.owner().rows == 2 && child.numerical_family_key().is_none()));
            original = children
                .iter()
                .map(|child| {
                    (
                        *child.domain_signature(),
                        serde_json::to_value(child.provenance()).unwrap(),
                    )
                })
                .collect();
        } else {
            assert!(
                children.len() > original.len(),
                "real B1 exact population must merge with B2: {:?}",
                children
                    .iter()
                    .map(|child| (
                        child.owner(),
                        child.domain_signature(),
                        child.numerical_family_key()
                    ))
                    .collect::<Vec<_>>()
            );
            assert!(children
                .iter()
                .any(|child| child.owner().rows == 1 && child.numerical_family_key().is_none()));
            for (domain, provenance) in &original {
                let retained = children
                    .iter()
                    .find(|child| child.domain_signature() == domain)
                    .unwrap();
                assert_eq!(
                    serde_json::to_value(retained.provenance()).unwrap(),
                    *provenance,
                    "retention must not renew original source, phase clocks or TTL"
                );
            }
        }
        session.retire_startup_owner_source().await.unwrap();
    }
    assert_eq!(budget.requests_remaining(), 0);
    assert_eq!(budget.attempts_remaining(), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 192);
    assert_eq!(
        serde_json::to_value(runtime.audit_snapshot()).unwrap()["structured_feedback"]
            ["outside_preparation_observations"],
        32
    );
    session.finish_startup_owner_series().unwrap();
    assert!(session.prepared_owner_capture.is_none());
    assert!(session
        .begin_prepared_owner_source(full_population(&session), CostProfileLoadLimits::default())
        .await
        .is_err());
    runtime.begin_automatic_calibration().unwrap();
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn startup_series_complete_unqualified_source_can_retire_without_resetting_budget() {
    let (mut session, _) = automatic_session().await;
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(45);
    session.begin_startup_owner_series(2, deadline).unwrap();
    session
        .begin_prepared_owner_source(
            declaration(&session, 3, false),
            CostProfileLoadLimits::default(),
        )
        .await
        .unwrap();
    let mut budget = ProbeExecutionBudget::new(
        deadline,
        NonZeroUsize::new(10).unwrap(),
        NonZeroUsize::new(20).unwrap(),
    );
    let options = ProbeCohortSettings {
        prefill_plan:
            crate::continuous_engine::inner::calibration::cohort_driver::ProbePrefillPlan::Joint,
        prefill_chunk: NonZeroU32::MIN,
        decode_route: CalibrationDecodeRoute::FullLogits,
        reset_token_policy: false,
    };
    for pass in 0..3 {
        session.begin_prepared_owner_cohort(pass, 0).unwrap();
        let requests = probe_requests(&session);
        session
            .run_probe_cohort(requests, options, &mut budget)
            .await
            .unwrap();
        session.end_prepared_owner_cohort().unwrap();
    }
    assert!(session.activate_prepared_owner_source().await.is_err());
    assert!(runtime.snapshot().is_none());
    session.retire_startup_owner_source().await.unwrap();
    assert_eq!(budget.requests_remaining(), 4);
    assert_eq!(budget.attempts_remaining(), 11);
    session
        .begin_prepared_owner_source(full_population(&session), CostProfileLoadLimits::default())
        .await
        .unwrap();
    session.finish_startup_owner_series().unwrap();
    assert!(runtime.snapshot().is_none());
    runtime.shutdown().await.unwrap();
}
