//! Real worker epoch publication reaches the production controller gates.
//! The existing controlled-backend witness installer tests ownership/guards;
//! it is not a claim that its synthetic route was certified by the fit model.
use super::*;
use crate::continuous_engine::inner::cost_observation::tests::statistical_model::runtime::feedback::FeedbackFixture;

#[tokio::test]
async fn selected_feedback_global_epoch_blocks_old_publication_before_resource_revalidation() {
    let f = FeedbackFixture::new(true);
    let runtime = Arc::new(f.build());
    let (mut engine, scheduler, executor) = fixture().await;
    Arc::get_mut(&mut engine.inner).unwrap().cost_runtime = Some(runtime.clone());
    let (id, session, prepared) =
        installed(&engine, &scheduler, &executor, Duration::from_secs(30)).await;
    drop(prepared);
    engine.inner.drain_slo_execution().await.unwrap();
    ready(&engine, &id).await;
    let hint = ferrum_interfaces::BatchHint::simple(1);
    let captured = prefill::captured_with_hint(&engine, &executor, &hint).await;
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
        witness_valid_for_ns: 30_000_000_000,
    };
    let revalidated = Arc::new(std::sync::atomic::AtomicBool::new(false));
    *executor.after_resource_revalidation.lock() = Some(Box::new({
        let flag = revalidated.clone();
        move || {
            flag.store(true, Ordering::Release);
        }
    }));
    let before = prefill::UnsubmittedState::capture(&engine, &executor);
    f.train_margin(&runtime, true);
    // The changed family need not be the first wave's family: the entire old
    // common-plan epoch, including every pending tail, is invalidated.
    assert_ne!(
        runtime.snapshot().unwrap().model_version(),
        selected.cost_model_version
    );
    assert!(matches!(
        engine
            .inner
            .prepare_slo_controller_wave(captured, selected, &hint)
            .unwrap(),
        SloIterationPlan::Idle
    ));
    before.assert_unchanged(&engine, &executor);
    assert!(!revalidated.load(Ordering::Acquire));
    assert_eq!(scheduled(&engine), 0);
    cleanup(engine, session).await;
}

#[tokio::test]
async fn selected_feedback_stale_published_witness_releases_grant_then_completes_with_fresh_authority(
) {
    let f = FeedbackFixture::new(true);
    let runtime = Arc::new(f.build());
    let (mut engine, scheduler, executor) = fixture().await;
    {
        let inner = Arc::get_mut(&mut engine.inner).unwrap();
        inner.cost_runtime = Some(runtime.clone());
        inner.config.scheduler.slo.mode = ferrum_types::SloMode::Enforce;
    }
    let (id, session, prepared) =
        installed(&engine, &scheduler, &executor, Duration::from_secs(30)).await;
    f.train_margin(&runtime, true);
    assert!(matches!(
        bounded(engine.inner.execute_slo_controller_wave(prepared))
            .await
            .unwrap(),
        EngineIterationOutcome::Idle
    ));
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert_eq!(scheduled(&engine), 0);
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 1);
    assert!(matches!(
        engine.inner.slo_controller.lock().completion_next,
        Some(CompletionOnlyReason::WitnessExpired)
    ));
    ready(&engine, &id).await;
    let hint = ferrum_interfaces::BatchHint::simple(1);
    let fresh = prefill::selected_after_retry(&engine, &executor, || {
        engine.inner.prepare_slo_controller(&hint)
    })
    .await;
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
        bounded(engine.inner.execute_slo_controller_wave(fresh))
            .await
            .unwrap(),
        EngineIterationOutcome::Progressed
    ));
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 2);
    cleanup(engine, session).await;
}

#[tokio::test]
async fn selected_feedback_update_after_preparation_is_seen_by_last_backend_host_gate() {
    let f = FeedbackFixture::new(true);
    let runtime = Arc::new(f.build());
    let (mut engine, scheduler, executor) = fixture().await;
    Arc::get_mut(&mut engine.inner).unwrap().cost_runtime = Some(runtime.clone());
    let (id, session, prepared) =
        installed(&engine, &scheduler, &executor, Duration::from_secs(30)).await;
    executor.park.store(true, Ordering::Release);
    let mut pending = Box::pin(engine.inner.execute_slo_controller_wave(prepared));
    assert!(pending.as_mut().now_or_never().is_none());
    bounded(executor.entered.notified()).await;
    // Initial controller gate passed; the backend is parked before its final
    // invocation of the SAME HostGuard, before its physical submit counter.
    f.train_margin(&runtime, true);
    executor.resume.notify_one();
    assert!(matches!(
        bounded(pending).await.unwrap(),
        EngineIterationOutcome::Idle
    ));
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert_eq!(scheduled(&engine), 0);
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 1);
    // This fixture's pre-encode Replan outcome is not a fabricated native
    // post-encode rollback. Native cleanup/retained-storage tests stay separate.
    cleanup(engine, session).await;
}

#[tokio::test]
async fn selected_feedback_revocation_preserves_off_and_observe_selection_semantics() {
    for mode in [ferrum_types::SloMode::Off, ferrum_types::SloMode::Observe] {
        let f = FeedbackFixture::with_ceiling(true, 1_000);
        let runtime = Arc::new(f.build());
        let (mut engine, _, executor) = fixture().await;
        {
            let inner = Arc::get_mut(&mut engine.inner).unwrap();
            inner.cost_runtime = Some(runtime.clone());
            inner.config.scheduler.slo.mode = mode;
        }
        let (_, session) = prefill::request(&engine, 2, 4).await;
        f.train_margin(&runtime, true);
        assert!(runtime.snapshot().is_none());
        assert!(matches!(
            engine
                .inner
                .prepare_slo_controller(&ferrum_interfaces::BatchHint::simple(1))
                .unwrap(),
            SloIterationPlan::Legacy
        ));
        assert_eq!(scheduled(&engine), 0);
        assert_eq!(executor.physical.load(Ordering::Acquire), 0);
        cleanup(engine, session).await;
    }
}
