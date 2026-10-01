//! One actual producer retires before distinct F/R/Q cohort owners restore.
//! This small lifecycle gate does not claim numerical qualification or TPS.
use super::*;
use crate::continuous_engine::inner::calibration::cohort_driver::ProbePrefillPlan;
use crate::continuous_engine::inner::calibration::startup::{
    ProbePrefixAcquisition, ProbePrefixFallback,
};

fn probes(session: &CalibrationSession) -> Vec<ProbeRequest> {
    probe_requests(session)
        .into_iter()
        .map(|mut value| {
            value.request.prompt = "test test test".into();
            if let Some(ferrum_types::ApiRequest::Completion(api)) = &mut value.request.api_request
            {
                api.prompt = value.request.prompt.clone();
            }
            value
        })
        .collect()
}

fn options() -> ProbeCohortSettings {
    ProbeCohortSettings {
        prefill_plan: ProbePrefillPlan::PreparedSequentialV1,
        prefill_chunk: NonZeroU32::MIN,
        decode_route: CalibrationDecodeRoute::FullLogits,
        reset_token_policy: false,
    }
}

async fn acquisition_session() -> (CalibrationSession, Arc<ControlledExecutor>) {
    automatic_session_with_storage_and_worker_options(
        2,
        Arc::new(AdvancingClock(AtomicU64::new(100))),
        Default::default(),
        ferrum_types::SloAutomaticCalibrationReuseV1::Disabled {},
        FixtureCostWorker::Background,
        true,
        true,
    )
    .await
}

fn configure_actual_executor(executor: &ControlledExecutor) {
    executor.enable_structured_query_route();
    executor
        .native_structured_submission
        .store(true, Ordering::Release);
    executor
        .project_structured_cpu_fill
        .store(true, Ordering::Release);
}

#[tokio::test]
async fn source8_native_prefix_once_seed_close_fresh_restore_final_prefill_and_three_phases() {
    let (mut session, executor) = acquisition_session().await;
    configure_actual_executor(&executor);
    let source = probes(&session).remove(0);
    let source_id = source.request.id.clone();
    let expected_plan =
        crate::continuous_engine::inner::calibration::startup::ProbePrefixAcquisitionPlan::new(
            3,
            2,
            NonZeroU32::MIN,
        )
        .unwrap();
    let setup = expected_plan.setup_work();
    let cohort = expected_plan
        .restored_cohort_work(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(3).unwrap(),
            ProbePrefillPlan::PreparedSequentialV1,
        )
        .unwrap();
    let requests = setup.requests + 3 * cohort.requests;
    let actions = setup.actions().unwrap() + 3 * cohort.actions().unwrap();
    // Original source8 declaration, prefix release and independent phase cuts.
    // Native copy is not an offered inference or fabricated Prefill sample.
    let mut declared = declaration(&session, 3, false);
    declared
        .population
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .workload_domain = session
        .engine
        .inner
        .cost_runtime
        .as_ref()
        .unwrap()
        .workload_domain()
        .unwrap()
        .clone();
    declared.maximum_offered_waves = 3 * cohort.inference_waves;
    declared.cohort_manifest_payload = serde_json::value::to_raw_value(&serde_json::json!({
        "prompt":"test test test", "maximum_output":3, "prepared_prefix":"actual_native_p_minus_1", "acquisition_plan":expected_plan,
        "acquisition_setup_work":setup, "restored_cohort_work":cohort,
        "outputs":["cli_text","completions_sse"]
    }))
    .unwrap();

    let mut budget = ProbeExecutionBudget::new(
        tokio::time::Instant::now() + Duration::from_secs(30),
        NonZeroUsize::new(requests).unwrap(),
        NonZeroUsize::new(actions).unwrap(),
    );
    let acquired = match session
        .acquire_probe_prefix(source, expected_plan, &mut budget, 64 * 1024)
        .await
        .unwrap()
    {
        ProbePrefixAcquisition::Ready(value) => value,
        other => panic!("real native acquisition was not ready: {other:?}"),
    };
    assert_eq!(acquired.plan(), expected_plan);
    assert!(acquired.ready());
    assert!(acquired.retained_payload_bytes().unwrap() <= 64 * 1024);
    assert_eq!(
        executor.physical.load(Ordering::Acquire),
        setup.inference_waves
    );
    assert!(!session
        .engine
        .inner
        .sequences
        .read()
        .contains_key(&source_id));
    session.completed_owner_boundary().unwrap();
    assert!(session.frontiers().unwrap().is_empty());
    assert_eq!(
        (budget.requests_remaining(), budget.attempts_remaining()),
        (3 * cohort.requests, 3 * cohort.actions().unwrap())
    );

    budget
        .reserve_selected_source(3 * cohort.requests, 3 * cohort.inference_waves)
        .unwrap();
    // Ready captures supply the exact physical scope before the original
    // header is signed or any numerical cohort is collected.
    declared.native_prefix_acquisition = Some(
        ferrum_scheduler::implementations::continuous::cost_profile::StructuredNativePrefixAcquisitionPlanV1 {
            phases: std::array::from_fn(|_| vec![Some(acquired.declaration())]),
        },
    );
    session
        .begin_prepared_owner_source(declared, CostProfileLoadLimits::default())
        .await
        .unwrap();
    let mut all_ids = vec![source_id];
    for phase in 0..3 {
        session.begin_prepared_owner_cohort(phase, 0).unwrap();
        let requests = probes(&session);
        for request in &requests {
            assert!(!all_ids.contains(&request.request.id));
            all_ids.push(request.request.id.clone());
        }
        let summary = session
            .run_acquired_prefix_probe_cohort(requests, options(), &mut budget, &acquired)
            .await
            .unwrap();
        assert_eq!(
            (summary.completed_requests, summary.completed_output_tokens),
            (2, 6)
        );
        assert_eq!(
            (summary.wave_attempts, summary.reconciled_waves),
            (cohort.inference_waves, cohort.inference_waves)
        );
        assert_eq!(
            (
                summary.native_restore_attempts,
                summary.acknowledged_prefix_restores,
                summary.prefix_restore_fallbacks
            ),
            (2, 2, 0)
        );
        assert_eq!(summary.released_prefix_rows, 2);
        session.end_prepared_owner_cohort().unwrap();
        session.completed_owner_boundary().unwrap();
        assert!(
            acquired.ready(),
            "native lease must survive fresh owner retirement"
        );
    }
    let audit = session
        .prepared_owner_capture
        .as_ref()
        .unwrap()
        .prepared_audit();
    assert!(!audit.population.poisoned, "{audit:#?}");
    assert_eq!(audit.preparation_attempts, 6);
    assert_eq!(
        audit.population.offered,
        (3 * cohort.inference_waves) as u64
    );
    assert_eq!(
        executor.physical.load(Ordering::Acquire),
        setup.inference_waves + 3 * cohort.inference_waves
    );
    assert_eq!(executor.native_prefix_terminal_counts(), (1, 6));
    assert_eq!(
        (budget.requests_remaining(), budget.attempts_remaining()),
        (0, 0)
    );
    // One cohort per independent phase proves real protocol/lifecycle, not
    // the unchanged numerical qualification sample floor.
    assert!(session.activate_prepared_owner_source().await.is_err());
    assert!(session
        .engine
        .inner
        .cost_runtime
        .as_ref()
        .unwrap()
        .snapshot()
        .is_none());
    drop(acquired);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn source8_native_prefix_acquisition_capacity_miss_drains_and_keeps_cold_fallback() {
    let (mut session, executor) = acquisition_session().await;
    configure_actual_executor(&executor);
    let mut budget = ProbeExecutionBudget::new(
        tokio::time::Instant::now() + Duration::from_secs(20),
        NonZeroUsize::new(3).unwrap(),
        NonZeroUsize::new(9).unwrap(),
    );
    let source = probes(&session).remove(0);
    let result = session
        .acquire_probe_prefix(
            source,
            crate::continuous_engine::inner::calibration::startup::ProbePrefixAcquisitionPlan::new(
                3,
                2,
                NonZeroU32::MIN,
            )
            .unwrap(),
            &mut budget,
            1,
        )
        .await
        .unwrap();
    assert!(matches!(
        result,
        ProbePrefixAcquisition::ColdFallback(ProbePrefixFallback::HostCapacity)
    ));
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert_eq!(executor.native_prefix_terminal_counts(), (0, 0));
    assert_eq!(
        (budget.requests_remaining(), budget.attempts_remaining()),
        (2, 9)
    );
    session.completed_owner_boundary().unwrap();
    // Use the ordinary driver after the typed miss. All three real prompt
    // tokens execute; no partially restored owner or fake ACK survives.
    let mut cold = options();
    cold.prefill_plan = ProbePrefillPlan::Joint;
    let summary = session
        .run_probe_cohort(probes(&session), cold, &mut budget)
        .await
        .unwrap();
    assert_eq!((summary.wave_attempts, summary.reconciled_waves), (5, 5));
    assert_eq!(
        (summary.completed_requests, summary.completed_output_tokens),
        (2, 6)
    );
    assert_eq!(summary.acknowledged_prefix_restores, 0);
    session.shutdown().await.unwrap();
}
