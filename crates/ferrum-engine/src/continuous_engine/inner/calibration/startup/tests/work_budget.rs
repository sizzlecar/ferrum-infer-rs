//! Work admission changes only the not-yet-frozen domain. These cases use
//! real controlled-executor receipts and the original reference assembler.
use super::*;
use ferrum_scheduler::implementations::continuous::prefill_reference::{
    ReferenceChunkLimits, ReferenceUnknown,
};
use std::sync::atomic::AtomicUsize;

#[tokio::test]
async fn automatic_reference_budget_freezes_prefix_and_completes_every_declared_trial() {
    // Three anchors plus decode need 20 requests. This budget can completely
    // measure two anchors plus decode (15), with insufficient room to add a
    // third. All three formal repetitions must remain present.
    let (engine, executor) = fixture_with_trials(SloMode::Enforce, 16, 3).await;
    let original = engine.inner.config.scheduler.slo.clone();
    let original_capacity = engine.inner.runtime_config.max_model_len;
    let runtime = engine.inner.cost_runtime.as_ref().unwrap().clone();
    let engine = drive(runtime.clone(), engine.finish_automatic_startup())
        .await
        .unwrap();
    let reference = engine
        .inner
        .prefill_reference_runtime
        .as_ref()
        .expect("a complete selected prefix must remain product-loadable");
    let loaded = reference.calibration();
    assert_eq!(loaded.protocol().repetitions.get(), 3);
    assert_eq!(
        loaded.piecewise_domain(),
        Some((NonZeroU32::MIN, NonZeroU32::new(2).unwrap()))
    );
    assert_eq!(
        loaded.legal_chunks(
            NonZeroU32::new(3).unwrap(),
            0,
            ReferenceChunkLimits {
                maximum_tokens: NonZeroU32::new(3).unwrap(),
                alignment: NonZeroU32::MIN,
                allow_final_short_chunk: true,
                maximum_candidates: NonZeroUsize::new(8).unwrap(),
            }
        ),
        Err(ReferenceUnknown::LengthNotCalibrated)
    );
    let (_, wire) = reference.startup_evidence().unwrap();
    let artifact: serde_json::Value = serde_json::from_slice(wire).unwrap();
    for curve in artifact["curves"]
        .as_array()
        .expect("actual reference curves")
    {
        assert_eq!(curve["trials"].as_array().unwrap().len(), 3);
    }
    assert_eq!(engine.inner.config.scheduler.slo, original);
    assert_eq!(engine.inner.runtime_config.max_model_len, original_capacity);
    assert!(engine.inner.sequences.read().is_empty());
    assert!(executor
        .produced_caches
        .lock()
        .iter()
        .all(|cache| cache.upgrade().is_none()));
    drop(runtime);
    engine.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn automatic_reference_nonlinear_formal_slowdown_hits_hard_deadline_without_partial_publication(
) {
    let (engine, executor) = fixture_with_trials(SloMode::Enforce, 16, 3).await;
    let runtime = engine.inner.cost_runtime.as_ref().unwrap().clone();
    let prefill_seen = Arc::new(AtomicUsize::new(0));
    let reached_formal = Arc::new(tokio::sync::Notify::new());
    let signal = reached_formal.clone();
    let formal_entry = Arc::new(AtomicUsize::new(0));
    let entry = formal_entry.clone();
    let weak_executor = Arc::downgrade(&executor);
    *executor.actual_observation_unknown.lock() = Some(Box::new(move |prefills, _| {
        let row = prefills.first()?;
        if row.chunk.total_prompt_tokens() == 2
            && row.chunk.tokens_processed() == 0
            && prefill_seen.fetch_add(1, Ordering::AcqRel) == 2
        {
            // Warmup and discovery completed quickly. The first formal
            // request now encounters a much slower next real executor wave.
            let executor = weak_executor.upgrade().unwrap();
            entry.store(executor.entries.load(Ordering::Acquire), Ordering::Release);
            executor.park.store(true, Ordering::Release);
            signal.notify_one();
        }
        None
    }));
    let mut startup = Box::pin(engine.finish_automatic_startup());
    drive(runtime.clone(), async {
        tokio::select! {
            result = startup.as_mut() => panic!("startup returned before formal slowdown: {}", result.err().map_or("success".into(), |e| e.to_string())),
            () = reached_formal.notified() => {}
        }
    }).await;
    let before = formal_entry.load(Ordering::Acquire);
    drive(runtime.clone(), async {
        loop {
            if executor.entries.load(Ordering::Acquire) > before {
                break;
            }
            tokio::select! {
                result = startup.as_mut() => panic!("startup returned before the slow wave entered: {}", result.err().map_or("success".into(), |e| e.to_string())),
                () = tokio::task::yield_now() => {}
            }
        }
    }).await;
    tokio::time::advance(Duration::from_secs(31)).await;
    // The timed-out collector cannot return a partial reference. The actual
    // unfinished wave, if already submitted, must be reconciled before ready.
    assert!(startup.as_mut().now_or_never().is_none());
    executor.park.store(false, Ordering::Release);
    executor.resume.notify_one();
    let engine = drive(runtime.clone(), startup).await.unwrap();
    assert!(engine.inner.prefill_reference_runtime.is_none());
    assert!(engine.inner.sequences.read().is_empty());
    assert_eq!(engine.inner.scheduler.active_count(), 0);
    assert_eq!(engine.inner.scheduler.waiting_count(), 0);
    assert!(executor
        .produced_caches
        .lock()
        .iter()
        .all(|cache| cache.upgrade().is_none()));
    drop(runtime);
    engine.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn automatic_reference_measured_time_budget_publishes_only_a_complete_prefix() {
    let (mut engine, executor) = fixture_with_trials(SloMode::Enforce, 128, 3).await;
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    let SloLiveStructuredCalibration::AutomaticV1 { settings } = &mut inner
        .config
        .scheduler
        .slo
        .cost_observation
        .live_structured_calibration
    else {
        unreachable!()
    };
    settings.reference_probe.maximum_duration_ms = NonZeroU64::new(5_000).unwrap();
    let runtime = engine.inner.cost_runtime.as_ref().unwrap().clone();
    let mut startup = Box::pin(engine.finish_automatic_startup());
    let mut observed = 0;
    let engine = loop {
        let entries = executor.entries.load(Ordering::Acquire);
        if entries > observed {
            // Drive only the Tokio probe-budget clock. Actual CPU receipts
            // keep their original clock and settlement protocol unchanged.
            tokio::time::advance(Duration::from_millis(50 * (entries - observed) as u64)).await;
            observed = entries;
        }
        if let Some(result) = startup.as_mut().now_or_never() {
            break result.unwrap();
        }
        runtime.consume_samples();
        tokio::task::yield_now().await;
    };
    let loaded = engine
        .inner
        .prefill_reference_runtime
        .as_ref()
        .expect("budgeted complete trials must install a reference")
        .calibration();
    assert_eq!(loaded.protocol().repetitions.get(), 3);
    assert!(
        loaded.piecewise_domain().unwrap().1.get() < 4,
        "measured work must stop expansion before the full candidate domain"
    );
    assert!(engine.inner.sequences.read().is_empty());
    drop(runtime);
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_reference_retry_cost_counts_before_next_anchor_admission() {
    use ferrum_interfaces::execution_cost::ActualWaveEvidenceUnknown;
    for (maximum_requests, expected_maximum) in [(15, 1), (16, 2)] {
        let (engine, executor) = fixture_with_trials(SloMode::Enforce, maximum_requests, 3).await;
        let runtime = engine.inner.cost_runtime.as_ref().unwrap().clone();
        let seen = AtomicUsize::new(0);
        *executor.actual_observation_unknown.lock() = Some(Box::new(move |prefills, _| {
            let row = prefills.first()?;
            // One real cold discovery after a complete warmup. Its output is
            // drained and none of its evidence enters the replacement owner.
            (row.chunk.total_prompt_tokens() == 1 && seen.fetch_add(1, Ordering::AcqRel) == 1)
                .then_some(ActualWaveEvidenceUnknown::GraphPath)
        }));
        let engine = drive(runtime.clone(), engine.finish_automatic_startup())
            .await
            .unwrap();
        let loaded = engine
            .inner
            .prefill_reference_runtime
            .as_ref()
            .expect("cold discovery still leaves a complete prefix protocol")
            .calibration();
        assert_eq!(loaded.protocol().repetitions.get(), 3);
        assert_eq!(loaded.piecewise_domain().unwrap().1.get(), expected_maximum);
        let submitted: std::collections::HashSet<_> = executor
            .submitted_requests
            .lock()
            .iter()
            .flatten()
            .cloned()
            .collect();
        assert!(submitted.len() <= maximum_requests);
        assert!(engine.inner.sequences.read().is_empty());
        drop(runtime);
        engine.shutdown().await.unwrap();
    }
}
