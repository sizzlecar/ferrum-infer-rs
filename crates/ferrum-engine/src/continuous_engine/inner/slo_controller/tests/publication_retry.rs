//! Expiry after planning must not repeatedly consume the completion budget.
use super::*;
use ferrum_interfaces::engine::InferenceEngine;

#[tokio::test(start_paused = true)]
async fn expired_publication_idle_reserves_a_fresh_completion_turn() {
    expired_publication_recovers_accepted_owner(
        ferrum_types::SloTimeAdmissionPolicy::CompleteRequests,
    )
    .await;
}

#[tokio::test(start_paused = true)]
async fn expired_require_slo_preparation_preserves_accepted_completion() {
    expired_publication_recovers_accepted_owner(ferrum_types::SloTimeAdmissionPolicy::RequireSlo)
        .await;
}

async fn expired_publication_recovers_accepted_owner(policy: ferrum_types::SloTimeAdmissionPolicy) {
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
    // Acceptance is an owner lifecycle fact, independent of the policy used
    // by subsequent scheduling turns. Keep the actual accepted stream alive.
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .config
        .scheduler
        .slo
        .admission
        .time_policy = policy;
    assert!(!engine.inner.sequences.read()[&id]
        .time_admission
        .as_ref()
        .unwrap()
        .before_acceptance());
    ready(&engine, &id).await;
    let hint = ferrum_interfaces::BatchHint::simple(1);
    let budget = ControllerBudget::new(slo_clock_now(), Duration::from_millis(1)).unwrap();
    let captured = engine
        .inner
        .capture_slo_controller_snapshot(&hint, Arc::clone(&budget))
        .unwrap();
    let selected = SelectedWave {
        final_replay_first_wave: None,
        protection: None,
        candidate: WaveCandidate {
            cost_evidence: None,
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
async fn expired_require_slo_preparation_keeps_pending_owner_waiting_and_fenced() {
    let (mut engine, scheduler, executor) = fixture().await;
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.config.scheduler.slo.mode = ferrum_types::SloMode::Enforce;
    inner.config.scheduler.slo.admission.time_policy =
        ferrum_types::SloTimeAdmissionPolicy::RequireSlo;
    inner.config.scheduler.slo.admission.max_wait_ms = NonZeroU64::new(30_000).unwrap();
    let mut request =
        ferrum_types::InferenceRequest::new("test", engine.inner.config.model.model_id.clone());
    request.stream = true;
    request.sampling_params.max_tokens = 2;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.repetition_penalty = 1.0;
    let id = request.id.clone();
    let ingress = slo_clock_now();
    let mut submitted = Box::pin(engine.infer_credited_stream(
        request,
        InferenceRequestContext::from_ingress(ingress),
        Arc::new(OutputProjectionContract::cli_text()),
    ));
    assert!(submitted.as_mut().now_or_never().is_none());
    ready(&engine, &id).await;
    prefill::admit(&engine, 1).await;
    let captured = prefill::captured(&engine, &executor).await;
    assert_eq!(captured.before_acceptance_candidate, Some(id.clone()));
    assert_eq!(captured.fences.len(), 1);
    assert_eq!(captured.fences[0].key.request_id, id);
    let queue_phase = scheduler.trace_phase(&id);
    assert!(queue_phase.is_some());
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
    // An intent to try completion does not authorize this pending owner.
    assert!(matches!(
        engine.inner.slo_controller.lock().completion_next,
        Some(CompletionOnlyReason::SearchInconclusive)
    ));
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
    let mut retry = Box::pin(engine.inner.wait_for_slo_controller_retry());
    assert!(retry.as_mut().now_or_never().is_none());
    tokio::time::advance(backoff + Duration::from_millis(1)).await;
    assert!(retry.as_mut().now_or_never().is_some());
    drop(retry);
    assert!(matches!(
        engine
            .inner
            .prepare_slo_controller(&ferrum_interfaces::BatchHint::simple(1))
            .unwrap(),
        SloIterationPlan::Idle
    ));
    assert!(engine.inner.slo_controller.lock().completion_next.is_none());
    assert_eq!(
        engine
            .inner
            .slo_controller
            .lock()
            .last_observation
            .unwrap()
            .reason,
        "no_accepted_work"
    );
    assert!(engine.inner.controller_frontiers_match(&captured));
    assert_eq!(scheduler.trace_phase(&id), queue_phase);
    {
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[&id];
        assert!(sequence
            .time_admission
            .as_ref()
            .unwrap()
            .before_acceptance());
        assert_eq!(sequence.slo.as_ref().unwrap().ingress(), ingress);
        assert_eq!(sequence.prefill_tokens_processed, 0);
        assert!(sequence.generated_tokens.is_empty());
        assert!(sequence.credited_output.as_ref().unwrap().grant.is_none());
    }
    assert_eq!(scheduled(&engine), 0);
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert!(submitted.as_mut().now_or_never().is_none());
    drop(captured);

    // Neither the expired planning turn nor its completion retry can restart
    // the original admission wait or accept the transport before its decision.
    let mut admission_wait = Box::pin(engine.inner.wait_for_slo_time_admission());
    assert!(admission_wait.as_mut().now_or_never().is_none());
    let deadline = ingress + engine.inner.config.scheduler.slo.admission.max_wait();
    tokio::time::advance(deadline.duration_since(slo_clock_now())).await;
    assert!(admission_wait.as_mut().now_or_never().is_some());
    drop(admission_wait);
    engine.inner.run_iteration().await.unwrap();
    assert!(matches!(
        submitted.await,
        Err(FerrumError::SloTimeAdmissionRejected {
            reason: ferrum_types::errors::SloTimeAdmissionRejection::WaitExpired,
        })
    ));
    assert!(engine.inner.sequences.read().is_empty());
    assert_eq!(scheduler.trace_phase(&id), None);
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    engine.shutdown().await.unwrap();
}
