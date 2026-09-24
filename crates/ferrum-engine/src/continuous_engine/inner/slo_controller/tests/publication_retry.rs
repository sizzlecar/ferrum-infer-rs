//! Expiry after planning must not repeatedly consume the completion budget.
use super::*;

#[tokio::test(start_paused = true)]
async fn expired_publication_idle_reserves_a_fresh_completion_turn() {
    let (mut engine, scheduler, executor) = fixture().await;
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .config
        .scheduler
        .slo
        .mode = ferrum_types::SloMode::Enforce;
    let (id, session, prepared) =
        installed(&engine, &scheduler, &executor, Duration::from_secs(1)).await;
    drop(prepared);
    engine.inner.drain_slo_execution().await.unwrap();
    ready(&engine, &id).await;
    let hint = ferrum_interfaces::BatchHint::simple(1);
    let budget = ControllerBudget::new(slo_clock_now(), Duration::from_millis(1)).unwrap();
    let captured = engine
        .inner
        .capture_slo_controller_snapshot(&hint, Arc::clone(&budget))
        .unwrap();
    let selected = SelectedWave {
        protection: None,
        candidate: WaveCandidate {
            work: vec![CandidateWork {
                key: captured.snapshot.requests[0].key.clone(),
                action: WaveAction::Decode,
            }],
            execution_shape: PlanningShapeDomain::Exact(
                canonical_cost_shape(&canonical(captured.fences[0].context as u32)).unwrap(),
            ),
            based_on_generation: captured.snapshot.generation,
            cost_model_version: captured.snapshot.cost_model_version,
        },
        predicted_wall_ns: 1,
        planning_observed_at_ns: captured.snapshot.observed_at_ns,
        snapshot_observed_at_ns: captured.snapshot.observed_at_ns,
        snapshot_generation: captured.snapshot.generation,
        cost_model_version: captured.snapshot.cost_model_version,
        witness_valid_for_ns: 1_000_000_000,
    };
    tokio::time::advance(Duration::from_millis(1)).await;
    let attempted = engine
        .inner
        .prepare_slo_controller_wave(captured, selected, &hint);
    assert!(matches!(&attempted, Ok(SloIterationPlan::Idle)));
    assert!(matches!(
        engine
            .inner
            .finish_slo_controller_preparation(&budget, attempted)
            .unwrap(),
        SloIterationPlan::Idle
    ));
    assert_eq!(scheduled(&engine), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert!(matches!(
        engine.inner.slo_controller.lock().completion_next,
        Some(CompletionOnlyReason::SearchInconclusive)
    ));
    assert!(engine.inner.controller_retry_pending());

    // Recovery starts only after the real retry timer and captures new
    // resource/output authority. It cannot dispatch the expired witness.
    let backoff = Duration::from_millis(
        engine
            .inner
            .config
            .scheduler
            .slo
            .planner
            .retry_backoff_ms
            .get(),
    );
    let mut wake = Box::pin(engine.inner.wait_for_slo_controller_retry());
    assert!(wake.as_mut().now_or_never().is_none());
    tokio::time::advance(backoff + Duration::from_millis(1)).await;
    assert!(wake.as_mut().now_or_never().is_some());
    drop(wake);
    let recovered = engine.inner.prepare_slo_controller(&hint).unwrap();
    let SloIterationPlan::Selected(recovered) = recovered else {
        panic!("fresh completion turn must select the retained runnable request")
    };
    assert!(engine.inner.slo_controller.lock().completion_next.is_none());
    assert_eq!(
        engine
            .inner
            .slo_controller
            .lock()
            .last_observation
            .unwrap()
            .disposition,
        "complete_requests"
    );
    assert!(matches!(
        bounded(engine.inner.execute_slo_controller_wave(recovered))
            .await
            .unwrap(),
        EngineIterationOutcome::Progressed
    ));
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 2);
    assert_eq!(scheduled(&engine), 0);
    cleanup(engine, session).await;
}

#[tokio::test(start_paused = true)]
async fn expired_observe_preparation_keeps_the_legacy_path() {
    let (engine, _, _) = fixture().await;
    let budget = ControllerBudget::new(slo_clock_now(), Duration::from_millis(1)).unwrap();
    tokio::time::advance(Duration::from_millis(1)).await;
    assert!(matches!(
        engine
            .inner
            .finish_slo_controller_preparation(&budget, Ok(SloIterationPlan::Legacy))
            .unwrap(),
        SloIterationPlan::Legacy
    ));
    assert!(engine.inner.slo_controller.lock().completion_next.is_none());
    assert!(!engine.inner.controller_retry_pending());
}

#[tokio::test(start_paused = true)]
async fn expired_failed_preparation_preserves_the_error() {
    let (engine, _, _) = completion_fixture(1).await;
    let budget = ControllerBudget::new(slo_clock_now(), Duration::from_millis(1)).unwrap();
    tokio::time::advance(Duration::from_millis(1)).await;
    let result = engine.inner.finish_slo_controller_preparation(
        &budget,
        Err(FerrumError::backend("publication backend failed")),
    );
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("publication backend failed"));
    assert!(engine.inner.slo_controller.lock().completion_next.is_none());
}

#[tokio::test(start_paused = true)]
async fn expired_require_slo_preparation_keeps_its_admission_policy() {
    let (mut engine, _, _) = fixture().await;
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.config.scheduler.slo.mode = ferrum_types::SloMode::Enforce;
    inner.config.scheduler.slo.admission.time_policy =
        ferrum_types::SloTimeAdmissionPolicy::RequireSlo;
    let budget = ControllerBudget::new(slo_clock_now(), Duration::from_millis(1)).unwrap();
    tokio::time::advance(Duration::from_millis(1)).await;
    assert!(matches!(
        engine
            .inner
            .finish_slo_controller_preparation(&budget, Ok(SloIterationPlan::Idle))
            .unwrap(),
        SloIterationPlan::Idle
    ));
    assert!(engine.inner.controller_retry_pending());
    assert!(engine.inner.slo_controller.lock().completion_next.is_none());
}
