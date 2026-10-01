//! A real native CPU source8 completes all three populations, then serves
//! original qualified inputs beyond its collection window without renewing age.
use super::*;

#[tokio::test]
async fn source8_prediction_validity_retains_original_sample_age_after_collection() {
    let clock = Arc::new(AdvancingClock(AtomicU64::new(100)));
    let (mut session, executor) = automatic_session_with_clock(2, clock.clone()).await;
    executor
        .native_structured_submission
        .store(true, Ordering::Release);
    executor
        .project_structured_cpu_fill
        .store(true, Ordering::Release);
    executor.enable_structured_query_route();
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    let template = numerical_family::homogeneous_template(&session);
    let mut declaration = numerical_family::family_population(&session);
    declaration.population.schedule.prediction_validity =
        Some(OwnerPredictionValidityPolicyV1::OriginalSampleAgeV1);
    declaration.population.maximum_window_ns = 1_000_000;
    session
        .begin_prepared_owner_source(declaration, CostProfileLoadLimits::default())
        .await
        .unwrap();
    let collection_deadline_upper = clock.0.load(Ordering::Acquire) + 1_000_000;
    let mut budget = ProbeExecutionBudget::new(
        tokio::time::Instant::now() + Duration::from_secs(45),
        NonZeroUsize::new(64).unwrap(),
        NonZeroUsize::new(96).unwrap(),
    );
    for pass in 0..3 {
        for ordinal in 0..[16, 8, 8][pass] {
            let width = 1 + ordinal % 2;
            session.begin_prepared_owner_cohort(pass, ordinal).unwrap();
            let completed = session
                .run_probe_cohort(
                    numerical_family::homogeneous_requests(&template, width),
                    ProbeCohortSettings {
                        prefill_plan: crate::continuous_engine::inner::calibration::cohort_driver::ProbePrefillPlan::Joint,
                        prefill_chunk: NonZeroU32::MIN,
                        decode_route: CalibrationDecodeRoute::FullLogits,
                        reset_token_policy: false,
                    },
                    &mut budget,
                )
                .await
                .unwrap();
            assert_eq!(completed.completed_requests, width);
            session.end_prepared_owner_cohort().unwrap();
        }
    }
    let audit = session
        .prepared_owner_capture
        .as_ref()
        .unwrap()
        .prepared_audit();
    assert!(!audit.population.poisoned, "{audit:#?}");
    assert_eq!(audit.population.offered, 96);
    let epoch = session.activate_prepared_owner_source().await.unwrap();
    assert!(epoch > 0);
    assert!(clock.0.load(Ordering::Acquire) < collection_deadline_upper);
    let trained = executor.native_structured_counts();
    assert!(
        trained.0 > 0 && trained.1 > 0,
        "original source must submit real CPU commands"
    );
    let children = runtime.startup_series_children_for_test().unwrap();
    let child = children
        .iter()
        .find(|child| child.numerical_family_key().is_some())
        .unwrap();
    let provenance = child.provenance();
    let original_expiry = provenance.clock.model_anchor_ns - provenance.oldest_imported_age_ns
        + child.runtime_limits().1;
    assert!(original_expiry > collection_deadline_upper);
    let receipt = runtime.profile_receipt().unwrap();

    // New real requests have no private preparation authority. Their original
    // query and full CPU host settlement both occur after collection ended.
    session = CalibrationSession::new_driver_session(
        session.engine,
        CalibrationLimits::new(NonZeroUsize::new(2).unwrap()).unwrap(),
    );
    clock
        .0
        .store(collection_deadline_upper + 1, Ordering::Release);
    numerical_family::query_and_execute(&mut session, &executor, &template, 2).await;
    session.freeze_cost_model().await.unwrap();
    let after = executor.native_structured_counts();
    assert!(after.0 > trained.0 && after.1 > trained.1);
    let feedback = serde_json::to_value(runtime.audit_snapshot()).unwrap();
    assert!(
        feedback["structured_feedback"]["revoked"].is_null(),
        "{feedback:#}"
    );
    assert_eq!(
        feedback["structured_feedback"]["uncomparable_observations"],
        0
    );
    assert_eq!(runtime.profile_receipt().unwrap(), receipt);
    child.is_current_local(original_expiry).unwrap();
    assert!(matches!(
        child.is_current_local(original_expiry + 1),
        Err(StructuredUnknownV2::Stale)
    ));
    session.shutdown().await.unwrap();
}
