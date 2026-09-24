//! Real ingress/output/scheduler owners and actual controlled submissions.
//! The pure planner tests prove deadline conflicts; these prove the fallback
//! really serves the due owner and never repays debt on a publication attempt.
use super::*;

async fn completion_selected(
    engine: &ContinuousBatchEngine,
    executor: &ControlledExecutor,
) -> owner::PreparedControllerWave {
    let mut hint = ferrum_interfaces::BatchHint::simple(1);
    hint.max_tokens = 64;
    prefill::selected_after_retry(engine, executor, || {
        let budget = ControllerBudget::new(slo_clock_now(), Duration::from_secs(30)).unwrap();
        engine.inner.prepare_completion_controller(
            &hint,
            &budget,
            CompletionOnlyReason::ExistingSloMiss,
        )
    })
    .await
}

#[tokio::test(start_paused = true)]
async fn actual_bypasses_force_due_owner_and_unsubmitted_attempt_does_not_clear_debt() {
    let (mut engine, scheduler, executor) = fixture_with_width(2).await;
    {
        let inner = Arc::get_mut(&mut engine.inner).unwrap();
        inner.config.scheduler.slo.mode = ferrum_types::SloMode::Enforce;
        inner.config.scheduler.slo.admission.max_active_requests = NonZeroUsize::new(2).unwrap();
    }
    let (late, late_session) = prefill::request(&engine, 4, 8).await;
    prefill::admit(&engine, 1).await;
    tokio::time::advance(Duration::from_secs(11)).await;
    {
        let mut sequences = engine.inner.sequences.write();
        sequences
            .get_mut(&late)
            .unwrap()
            .slo
            .as_mut()
            .unwrap()
            .observe_wait(slo_clock_now())
            .unwrap();
    }
    let (healthy, healthy_session) = prefill::request(&engine, 4, 8).await;
    prefill::admit(&engine, 2).await;
    let original_ingress = engine.inner.sequences.read()[&late]
        .slo
        .as_ref()
        .unwrap()
        .ingress();
    // Service the healthy owner twice through the real guarded path. The late
    // owner is retained, executable, and gets no physical or output progress.
    for offset in 0..2 {
        let prepared = prefill::install_with_admission(
            &engine,
            &executor,
            &scheduler,
            &[(
                healthy.clone(),
                PlanningWorkAction::Prefill {
                    offset,
                    count: NonZeroUsize::new(1).unwrap(),
                },
            )],
            None,
        )
        .await;
        assert!(matches!(
            engine
                .inner
                .execute_slo_controller_wave(prepared)
                .await
                .unwrap(),
            EngineIterationOutcome::Progressed
        ));
        assert_eq!(
            engine.inner.sequences.read()[&late]
                .time_admission
                .as_ref()
                .unwrap()
                .recovery_service
                .eligible_bypasses(),
            offset + 1
        );
    }
    let captured = prefill::captured(&engine, &executor).await;
    assert!(captured.protection.new_time_promises_closed());
    assert_eq!(
        captured
            .protection
            .required_first_service()
            .unwrap()
            .request_id,
        late
    );
    assert!(
        engine.inner.propose_slo_time_admission(&captured).is_none(),
        "recovery must not create a PendingTimeWitness"
    );
    drop(captured);
    {
        let sequences = engine.inner.sequences.read();
        let healthy_incarnation = sequences[&healthy]
            .cost_frontier
            .unwrap()
            .owner_incarnation
            .get();
        let late_incarnation = sequences[&late]
            .cost_frontier
            .unwrap()
            .owner_incarnation
            .get();
        // Deliberately put the healthy request at the ordinary FIFO head.
        engine.inner.slo_controller.lock().completion_order = VecDeque::from([
            (healthy.clone(), healthy_incarnation),
            (late.clone(), late_incarnation),
        ]);
    }
    // Unknown capability cannot infer the one-token quantum, even for the
    // same guarded backend. This first attempt retains its original bound.
    executor.prefill_granularity.store(0, Ordering::Release);
    let prepared = completion_selected(&engine, &executor).await;
    let before = executor.physical.load(Ordering::Acquire);
    let flight = engine
        .inner
        .slo_controller
        .lock()
        .pending_execution
        .clone()
        .unwrap();
    assert_eq!(
        flight
            .work
            .rows()
            .map(|row| row.request_id.clone())
            .collect::<Vec<_>>(),
        vec![late.clone()]
    );
    assert!(matches!(flight.work.rows().next().unwrap().input,
        ExpectedWaveInput::Prefill {chunk} if chunk.tokens_to_process() == 4));
    drop(prepared);
    bounded(engine.inner.drain_slo_execution()).await.unwrap();
    assert_eq!(executor.physical.load(Ordering::Acquire), before);
    assert!(engine.inner.sequences.read()[&late]
        .time_admission
        .as_ref()
        .unwrap()
        .recovery_service
        .due());
    // Withdrawal queues the returned grant to the real output actor. Until
    // that actor republishes Ready, this owner is genuinely not eligible and
    // a healthy peer may advance. Wait on its actual readiness notification,
    // just as the existing controller withdrawal regression does.
    ready(&engine, &late).await;
    assert_eq!(executor.physical.load(Ordering::Acquire), before);
    executor.prefill_granularity.store(1, Ordering::Release);
    let prepared = completion_selected(&engine, &executor).await;
    let selected = engine
        .inner
        .slo_controller
        .lock()
        .pending_execution
        .clone()
        .unwrap();
    assert_eq!(selected.work.rows().next().unwrap().request_id, late);
    assert!(matches!(selected.work.rows().next().unwrap().input,
        ExpectedWaveInput::Prefill {chunk} if chunk.tokens_to_process() == 1));
    assert!(matches!(
        engine
            .inner
            .execute_slo_controller_wave(prepared)
            .await
            .unwrap(),
        EngineIterationOutcome::Progressed
    ));
    assert_eq!(
        executor.submitted_requests.lock().last(),
        Some(&vec![late.clone()])
    );
    {
        let sequences = engine.inner.sequences.read();
        let actual = &sequences[&late];
        assert_eq!(actual.prefill_tokens_processed, 1);
        assert_eq!(
            actual
                .time_admission
                .as_ref()
                .unwrap()
                .recovery_service
                .eligible_bypasses(),
            0
        );
        assert_eq!(actual.slo.as_ref().unwrap().ingress(), original_ingress);
        assert!(actual.slo.as_ref().unwrap().violations().ttft);
        assert!(
            actual
                .time_admission
                .as_ref()
                .unwrap()
                .has_started_witness()
                == false
        );
        assert!(
            sequences[&healthy]
                .time_admission
                .as_ref()
                .unwrap()
                .has_started_witness()
                == false
        );
    }
    drop(healthy_session);
    cleanup(engine, late_session).await;
}

#[tokio::test]
async fn recovery_final_guard_rejects_changed_debt_without_a_physical_submission() {
    let (mut engine, _, executor) = fixture_with_width(1).await;
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .config
        .scheduler
        .slo
        .mode = ferrum_types::SloMode::Enforce;
    let (id, session) = prefill::request(&engine, 4, 8).await;
    prefill::admit(&engine, 1).await;
    let prepared = completion_selected(&engine, &executor).await;
    engine
        .inner
        .sequences
        .write()
        .get_mut(&id)
        .unwrap()
        .time_admission
        .as_mut()
        .unwrap()
        .recovery_service
        .bypass();
    let before = executor.physical.load(Ordering::Acquire);
    let _ = engine
        .inner
        .execute_slo_controller_wave(prepared)
        .await
        .unwrap();
    assert_eq!(executor.physical.load(Ordering::Acquire), before);
    assert_eq!(
        engine.inner.sequences.read()[&id].prefill_tokens_processed,
        0
    );
    assert_eq!(
        engine.inner.sequences.read()[&id]
            .time_admission
            .as_ref()
            .unwrap()
            .recovery_service
            .eligible_bypasses(),
        1
    );
    cleanup(engine, session).await;
}

#[tokio::test]
async fn cancelled_owner_debt_is_not_inherited_by_same_id_replacement() {
    let (mut engine, _, executor) = fixture_with_width(1).await;
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .config
        .scheduler
        .slo
        .mode = ferrum_types::SloMode::Enforce;
    let (id, old) = prefill::request(&engine, 4, 8).await;
    prefill::admit(&engine, 1).await;
    engine
        .inner
        .sequences
        .write()
        .get_mut(&id)
        .unwrap()
        .time_admission
        .as_mut()
        .unwrap()
        .recovery_service
        .bypass();
    let old_owner = engine.inner.sequences.read()[&id]
        .stream_projection_identity
        .clone();
    let prepared = completion_selected(&engine, &executor).await;
    drop(old);
    bounded(async {
        while engine.inner.sequences.read().contains_key(&id) {
            engine.inner.cancel_abandoned_requests().await.unwrap();
            tokio::task::yield_now().await;
        }
    })
    .await;
    let mut request = ferrum_types::InferenceRequest::new(
        "test test test test",
        engine.inner.config.model.model_id.clone(),
    );
    request.id = id.clone();
    request.stream = true;
    request.sampling_params.max_tokens = 8;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.repetition_penalty = 1.0;
    let replacement = engine
        .infer_credited_stream(
            request,
            InferenceRequestContext::from_ingress(slo_clock_now()),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    ready(&engine, &id).await;
    {
        let sequences = engine.inner.sequences.read();
        assert!(!Arc::ptr_eq(
            &old_owner,
            &sequences[&id].stream_projection_identity
        ));
        assert_eq!(
            sequences[&id]
                .time_admission
                .as_ref()
                .unwrap()
                .recovery_service
                .eligible_bypasses(),
            0
        );
    }
    assert!(matches!(
        engine
            .inner
            .execute_slo_controller_wave(prepared)
            .await
            .unwrap(),
        EngineIterationOutcome::Idle
    ));
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert_eq!(
        engine.inner.sequences.read()[&id]
            .time_admission
            .as_ref()
            .unwrap()
            .recovery_service
            .eligible_bypasses(),
        0
    );
    cleanup(engine, replacement).await;
}
