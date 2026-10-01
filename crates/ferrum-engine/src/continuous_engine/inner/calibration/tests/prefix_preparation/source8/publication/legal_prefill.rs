//! Original CPU producer and source8 records for the declared preparation path.
use super::*;
use crate::continuous_engine::inner::calibration::cohort_driver::ProbePrefillPlan;

#[tokio::test]
async fn source8_legal_prefill_records_each_setup_wave_before_joint_suffix() {
    let (mut session, executor) = automatic_session().await;
    // Native submission needs the same declared core readback capability as
    // the future route; unknown runtime evidence must still fail its guard.
    executor.enable_structured_query_route();
    executor
        .native_structured_submission
        .store(true, Ordering::Release);
    executor
        .single_row_prefill_only
        .store(true, Ordering::Release);
    let mut declared = declaration(&session, 3, false);
    let cohorts: usize = declared.cohort_plan.phases.iter().map(Vec::len).sum();
    let requests = cohorts * 2;
    let waves = cohorts * (2 + 2); // two original prefills, two joint suffixes
    declared.maximum_offered_waves = waves;
    declared.cohort_manifest_payload = serde_json::value::to_raw_value(&serde_json::json!({
        "prompt":"test", "maximum_output":3,
        "prepared_prefix_prefill":"prepared_sequential_v1", "rows":2
    }))
    .unwrap();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    let mut insufficient = ProbeExecutionBudget::new(
        deadline,
        NonZeroUsize::new(requests).unwrap(),
        NonZeroUsize::new(waves - 1).unwrap(),
    );
    assert!(insufficient
        .reserve_selected_source(requests, waves)
        .is_err());
    assert_eq!(insufficient.requests_remaining(), requests);
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert!(session.frontiers().unwrap().is_empty());
    let mut budget = ProbeExecutionBudget::new(
        deadline,
        NonZeroUsize::new(requests).unwrap(),
        NonZeroUsize::new(waves).unwrap(),
    );
    budget.reserve_selected_source(requests, waves).unwrap();
    session
        .begin_prepared_owner_source(declared, CostProfileLoadLimits::default())
        .await
        .unwrap();
    let settings = ProbeCohortSettings {
        prefill_plan: ProbePrefillPlan::PreparedSequentialV1,
        prefill_chunk: NonZeroU32::MIN,
        decode_route: CalibrationDecodeRoute::FullLogits,
        reset_token_policy: false,
    };
    for phase in 0..3 {
        session.begin_prepared_owner_cohort(phase, 0).unwrap();
        let probes = probe_requests(&session);
        let summary = session
            .run_probe_cohort(probes, settings, &mut budget)
            .await
            .unwrap();
        assert_eq!(summary.completed_requests, 2);
        assert_eq!(summary.completed_output_tokens, 6);
        assert_eq!(summary.wave_attempts, 4);
        assert_eq!(summary.reconciled_waves, 4);
        assert_eq!(summary.released_prefix_rows, 2);
        session.end_prepared_owner_cohort().unwrap();
    }
    let audit = session
        .prepared_owner_capture
        .as_ref()
        .unwrap()
        .prepared_audit();
    assert!(!audit.population.poisoned, "{audit:#?}");
    assert_eq!(audit.preparation_attempts, (cohorts * 2) as u64);
    assert_eq!(audit.population.offered, waves as u64);
    assert_eq!(executor.physical.load(Ordering::Acquire), waves);
    assert_eq!(budget.requests_remaining(), 0);
    assert_eq!(budget.attempts_remaining(), 0);
    assert!(session.frontiers().unwrap().is_empty());
    // These three complete cohorts are lifecycle evidence, insufficient for
    // numerical qualification. Every setup remains preparation, never a fit.
    assert!(session.activate_prepared_owner_source().await.is_err());
    assert!(session
        .engine
        .inner
        .cost_runtime
        .as_ref()
        .unwrap()
        .snapshot()
        .is_none());
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn source8_legal_prefill_fixture_rejects_joint_execution_before_submission() {
    let (mut session, executor) = automatic_session().await;
    // Native submission needs the same declared core readback capability as
    // the future route; unknown runtime evidence must still fail its guard.
    executor.enable_structured_query_route();
    executor
        .native_structured_submission
        .store(true, Ordering::Release);
    executor
        .single_row_prefill_only
        .store(true, Ordering::Release);
    let declared = declaration(&session, 3, false);
    session
        .begin_prepared_owner_source(declared, CostProfileLoadLimits::default())
        .await
        .unwrap();
    session.begin_prepared_owner_cohort(0, 0).unwrap();
    let mut budget = ProbeExecutionBudget::new(
        tokio::time::Instant::now() + Duration::from_secs(20),
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(4).unwrap(),
    );
    let settings = ProbeCohortSettings {
        prefill_plan: ProbePrefillPlan::Joint,
        prefill_chunk: NonZeroU32::MIN,
        decode_route: CalibrationDecodeRoute::FullLogits,
        reset_token_policy: false,
    };
    let probes = probe_requests(&session);
    let error = session
        .run_probe_cohort(probes, settings, &mut budget)
        .await
        .err()
        .expect("actual joint route must be unavailable");
    assert!(error.to_string().contains("single-row prefill"), "{error}");
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert_eq!(budget.attempts_remaining(), 3);
    assert!(session
        .engine
        .inner
        .cost_runtime
        .as_ref()
        .unwrap()
        .snapshot()
        .is_none());
    assert!(session.activate_prepared_owner_source().await.is_err());
    session.shutdown().await.unwrap();
}
