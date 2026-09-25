//! Real controller capture/publication/undo, with a controlled backend's typed
//! pre-submit capacity gate. No GPU timing or native cleanup is claimed here.
use super::*;

async fn fixture_for_search() -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<ControlledExecutor>,
) {
    let (mut engine, scheduler, executor) = fixture_with_width(1).await;
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.config.scheduler.slo.mode = ferrum_types::SloMode::Enforce;
    inner.config.scheduler.slo.admission.time_policy =
        ferrum_types::SloTimeAdmissionPolicy::CompleteRequests;
    (engine, scheduler, executor)
}

async fn prepare(
    engine: &ContinuousBatchEngine,
    executor: &ControlledExecutor,
) -> owner::PreparedControllerWave {
    prefill::selected_after_retry(engine, executor, || {
        engine
            .inner
            .prepare_slo_controller(&ferrum_interfaces::BatchHint::simple(1))
    })
    .await
}

fn search_calls(engine: &ContinuousBatchEngine) -> u64 {
    engine
        .inner
        .slo_controller
        .lock()
        .last_audit
        .unwrap()
        .stages
        .iter()
        .find(|stage| matches!(stage.stage, ControllerStage::SearchReplay))
        .unwrap()
        .calls
}

async fn maintain(engine: &ContinuousBatchEngine) {
    assert!(matches!(
        engine.inner.prepare_slo_maintenance_turn().await.unwrap(),
        Some(EngineIterationOutcome::Progressed)
    ));
    assert!(engine
        .inner
        .slo_controller
        .lock()
        .pending_maintenance
        .is_none());
}

#[tokio::test]
async fn completion_deferrals_skip_repeated_search_until_real_progress_then_reopen_it() {
    let (engine, _, executor) = fixture_for_search().await;
    let (id, mut session) = prefill::request(&engine, 1, 4).await;
    prefill::admit(&engine, 1).await;
    let model = engine.inner.cost_runtime.as_ref().unwrap();
    let original_version = model.snapshot().unwrap().model_version();
    let original_frontier = engine.inner.sequences.read()[&id].cost_frontier;

    // The first attempt captures current route evidence. A real transient read
    // failure may select safe completion before search; an available snapshot
    // reaches the planner, whose unsupported future route also selects it.
    // Neither case may repeat that work while the backend keeps deferring the
    // same frontier without new tokens, model evidence, or capacity.
    for attempt in 0..3 {
        let before_capture = executor.deferrals.planning_captures.load(Ordering::Acquire);
        let prepared = prepare(&engine, &executor).await;
        executor
            .deferrals
            .capacity(std::slice::from_ref(&id), Some(false));
        assert!(matches!(
            bounded(engine.inner.execute_slo_controller_wave(prepared))
                .await
                .unwrap(),
            EngineIterationOutcome::Idle
        ));
        let after_capture = executor.deferrals.planning_captures.load(Ordering::Acquire);
        if attempt == 0 {
            assert!(after_capture > before_capture);
        } else {
            assert_eq!(after_capture, before_capture);
            assert_eq!(search_calls(&engine), 0);
        }
        assert_eq!(executor.physical.load(Ordering::Acquire), 0);
        assert_eq!(scheduled(&engine), 0);
        assert_eq!(
            engine.inner.sequences.read()[&id].cost_frontier,
            original_frontier
        );
        assert_eq!(model.snapshot().unwrap().model_version(), original_version);
        assert!(engine.inner.sequences.read()[&id]
            .generated_tokens
            .is_empty());
        assert!(
            !engine.inner.controller_retry_pending(),
            "maintenance owns the wake, not a new timer"
        );
        maintain(&engine).await;
    }
    assert_eq!(
        executor.deferrals.maintenance_calls.load(Ordering::Acquire),
        3
    );

    // Capacity and learned costs can change while completion is pending. They
    // do not grant permission from the old attempt: publication captures anew.
    executor
        .deferrals
        .capacity_epoch
        .fetch_add(1, Ordering::AcqRel);
    train_runtime(model);
    assert!(model.snapshot().unwrap().model_version() > original_version);
    let before_resume = executor.deferrals.planning_captures.load(Ordering::Acquire);
    let prepared = prepare(&engine, &executor).await;
    assert!(matches!(
        bounded(engine.inner.execute_slo_controller_wave(prepared))
            .await
            .unwrap(),
        EngineIterationOutcome::Progressed
    ));
    assert_eq!(search_calls(&engine), 0);
    assert_eq!(
        executor.deferrals.planning_captures.load(Ordering::Acquire),
        before_resume
    );
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 1);
    assert!(engine.inner.slo_controller.lock().completion_next.is_none());
    consume(&engine, &id, &mut session).await;

    // There is no sticky mode or cached Unknown result: after that successful
    // wave the next current frontier must recapture route evidence with the
    // new model. That real nonblocking read can still be temporarily unavailable;
    // its availability must not decide whether this cache-invalidation test passes.
    let before_reopen = executor.deferrals.planning_captures.load(Ordering::Acquire);
    let prepared = prepare(&engine, &executor).await;
    assert!(matches!(
        bounded(engine.inner.execute_slo_controller_wave(prepared))
            .await
            .unwrap(),
        EngineIterationOutcome::Progressed
    ));
    assert!(executor.deferrals.planning_captures.load(Ordering::Acquire) > before_reopen);
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 2);
    cleanup(engine, session).await;
}

#[tokio::test]
async fn completion_capacity_deferral_rechecks_publication_after_an_epoch_change() {
    let (engine, scheduler, executor) = fixture_for_search().await;
    let (id, session) = prefill::request(&engine, 1, 3).await;
    prefill::admit(&engine, 1).await;
    let prepared = prepare(&engine, &executor).await;
    executor.deferrals.capacity(std::slice::from_ref(&id), None);
    assert!(matches!(
        bounded(engine.inner.execute_slo_controller_wave(prepared))
            .await
            .unwrap(),
        EngineIterationOutcome::Idle
    ));
    assert_eq!(search_calls(&engine), 1);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);

    // The real scheduler receives the old wait condition and the new capacity
    // snapshot. It may re-admit current work, never reuse an old publication.
    executor
        .deferrals
        .capacity_epoch
        .fetch_add(1, Ordering::AcqRel);
    maintain(&engine).await;
    let changed = Arc::clone(&scheduler);
    *executor.after_resource_revalidation.lock() = Some(Box::new(move || {
        changed.record_external_capacity_release();
    }));
    assert!(matches!(
        engine
            .inner
            .prepare_slo_controller(&ferrum_interfaces::BatchHint::simple(1))
            .unwrap(),
        SloIterationPlan::Idle
    ));
    assert!(executor.after_resource_revalidation.lock().is_none());
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert_eq!(scheduled(&engine), 0);
    assert!(engine.inner.controller_retry_pending());
    engine.inner.wait_for_slo_controller_retry().await;

    let prepared = prepare(&engine, &executor).await;
    assert!(matches!(
        bounded(engine.inner.execute_slo_controller_wave(prepared))
            .await
            .unwrap(),
        EngineIterationOutcome::Progressed
    ));
    assert_eq!(search_calls(&engine), 0);
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 1);
    cleanup(engine, session).await;
}

#[tokio::test]
async fn witnessed_deferral_does_not_turn_new_evidence_into_a_completion_override() {
    let (engine, scheduler, executor) = fixture_for_search().await;
    let (id, session, prepared) =
        installed(&engine, &scheduler, &executor, Duration::from_secs(30)).await;
    executor
        .deferrals
        .capacity(std::slice::from_ref(&id), Some(true));
    assert!(matches!(
        bounded(engine.inner.execute_slo_controller_wave(prepared))
            .await
            .unwrap(),
        EngineIterationOutcome::Idle
    ));
    assert!(engine.inner.slo_controller.lock().completion_next.is_none());
    maintain(&engine).await;
    let before = executor.deferrals.planning_captures.load(Ordering::Acquire);
    let prepared = prepare(&engine, &executor).await;
    assert!(matches!(
        bounded(engine.inner.execute_slo_controller_wave(prepared))
            .await
            .unwrap(),
        EngineIterationOutcome::Progressed
    ));
    assert_eq!(search_calls(&engine), 1);
    assert!(executor.deferrals.planning_captures.load(Ordering::Acquire) > before);
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 2);
    cleanup(engine, session).await;
}

#[tokio::test]
async fn deferred_completion_recaptures_a_replacement_owner_and_abandons_old_maintenance() {
    let (engine, _, executor) = fixture_for_search().await;
    let (id, old) = prefill::request(&engine, 1, 3).await;
    prefill::admit(&engine, 1).await;
    let original_owner = engine.inner.sequences.read()[&id]
        .cost_frontier
        .unwrap()
        .owner_incarnation;
    let prepared = prepare(&engine, &executor).await;
    executor
        .deferrals
        .capacity(std::slice::from_ref(&id), Some(true));
    assert!(matches!(
        bounded(engine.inner.execute_slo_controller_wave(prepared))
            .await
            .unwrap(),
        EngineIterationOutcome::Idle
    ));
    drop(old);
    bounded(async {
        while engine.inner.sequences.read().contains_key(&id) {
            engine.inner.cancel_abandoned_requests().await.unwrap();
            tokio::task::yield_now().await;
        }
    })
    .await;
    let (replacement, output) = request(&engine, 1, 3, Some(id.clone())).await;
    admit(&engine, 1).await;
    assert_eq!(replacement, id);
    assert_ne!(
        engine.inner.sequences.read()[&id]
            .cost_frontier
            .unwrap()
            .owner_incarnation,
        original_owner
    );
    maintain(&engine).await;
    assert_eq!(
        executor.deferrals.maintenance_calls.load(Ordering::Acquire),
        0
    );
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    let prepared = prepare(&engine, &executor).await;
    assert!(matches!(
        bounded(engine.inner.execute_slo_controller_wave(prepared))
            .await
            .unwrap(),
        EngineIterationOutcome::Progressed
    ));
    assert_eq!(search_calls(&engine), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 1);
    cleanup(engine, output).await;
}
