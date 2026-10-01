//! Exercise actual zero-submit receipts and durable task failures; no report
//! or reference observation is manufactured by these controlled-backend tests.
use super::*;

fn prompt() -> (
    plan::ProbePrompt,
    ferrum_scheduler::implementations::continuous::prefill_reference::PiecewiseReferenceSpec,
) {
    let tokens = NonZeroU32::new(2).unwrap();
    (
        plan::ProbePrompt {
            text: " a a".into(),
            tokens,
        },
        ferrum_scheduler::implementations::continuous::prefill_reference::PiecewiseReferenceSpec {
            minimum_prompt_tokens: tokens,
            maximum_prompt_tokens: tokens,
            body_endpoints: vec![NonZeroU32::MIN],
        },
    )
}

#[tokio::test]
async fn automatic_probe_retries_not_submitted_without_advancing_reference_work() {
    let (mut engine, executor) = fixture(SloMode::Enforce, 128).await;
    let runtime = engine.inner.cost_runtime.as_ref().unwrap().clone();
    ContinuousBatchEngine::check_startup_session(&mut engine).unwrap();
    let mut session = CalibrationSession::new_driver_session(
        engine,
        CalibrationLimits::new(NonZeroUsize::MIN).unwrap(),
    );
    let (prompt, partition) = prompt();
    executor.replan_before_encode.store(true, Ordering::Release);
    executor.park.store(true, Ordering::Release);
    let mut budget = plan::ProbeBudget::new(NonZeroUsize::MIN);
    let mut progress = progress::StartupProgress::default();
    let mut probe = Box::pin(session.startup_probe_tracked(
        &prompt,
        &partition,
        probes::Capture::PrefillDiscovery,
        &mut budget,
        &mut progress,
    ));
    // Hold two actual executor entries. The second is reachable only if the
    // first conclusive NotSubmitted receipt was retried, without any progress.
    for expected_entries in 1..=2 {
        tokio::time::timeout(Duration::from_secs(5), async {
            tokio::select! {
                result = probe.as_mut() => panic!("probe ended before retry: {:?}", result.err()),
                () = executor.entered.notified() => {}
            }
        })
        .await
        .unwrap();
        assert_eq!(executor.entries.load(Ordering::Acquire), expected_entries);
        assert_eq!(executor.physical.load(Ordering::Acquire), 0);
        if expected_entries == 1 {
            executor.resume.notify_one();
        }
    }
    executor
        .replan_before_encode
        .store(false, Ordering::Release);
    executor.park.store(false, Ordering::Release);
    executor.resume.notify_one();
    let samples = drive(runtime.clone(), probe).await.unwrap();
    assert_eq!(
        samples.len(),
        2,
        "each original prefill segment is observed once"
    );
    assert_eq!(samples[0].shape().exact.prefill_chunks[0].offset, 0);
    assert_eq!(samples[1].shape().exact.prefill_chunks[0].offset, 1);
    assert!(samples[0].accepted_ordinal() < samples[1].accepted_ordinal());
    let physical_waves = plan::OUTPUT_TOKENS + 1; // Two prefill chunks, then decode.
    assert_eq!(executor.entries.load(Ordering::Acquire), physical_waves + 1);
    assert_eq!(executor.physical.load(Ordering::Acquire), physical_waves);
    let diagnostic = progress.current.as_ref().unwrap();
    assert_eq!(progress.completed_requests, 1);
    assert_eq!(diagnostic.counts.wave_attempts, physical_waves + 1);
    assert_eq!(diagnostic.counts.zero_submit_retries, 1);
    assert_eq!(diagnostic.counts.host_reconciled, physical_waves);
    // The final Length call is legacy Composite and the original recorder's
    // unknown-only diagnostic is absent on its known route. This fixture also
    // declares terminal cleanup work Unknown. Keep that missing physical-count
    // evidence separate from the executor's independently observed real waves.
    assert_eq!(executor.completion_calls.load(Ordering::Acquire), 1);
    assert_eq!(diagnostic.counts.known_physical_waves, physical_waves - 1);
    assert_eq!(diagnostic.counts.reports_without_physical_count, 1);
    assert_eq!(diagnostic.counts.not_submitted, 1);
    assert!(session.frontiers().unwrap().is_empty());
    assert!(!session.indeterminate);
    session.drain_startup().await.unwrap();
    drop(runtime);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_probe_consumes_zero_submit_maintenance_before_retry() {
    let (mut engine, executor) = fixture(SloMode::Enforce, 128).await;
    let runtime = engine.inner.cost_runtime.as_ref().unwrap().clone();
    ContinuousBatchEngine::check_startup_session(&mut engine).unwrap();
    let inner = Arc::clone(&engine.inner);
    let weak_executor = Arc::downgrade(&executor);
    *executor.before_resource_revalidation.lock() = Some(Box::new(move || {
        let ids = inner.sequences.read().keys().cloned().collect::<Vec<_>>();
        assert_eq!(ids.len(), 1);
        weak_executor
            .upgrade()
            .unwrap()
            .deferrals
            .capacity(&ids, Some(true));
    }));
    let mut session = CalibrationSession::new_driver_session(
        engine,
        CalibrationLimits::new(NonZeroUsize::MIN).unwrap(),
    );
    let (prompt, partition) = prompt();
    let mut budget = plan::ProbeBudget::new(NonZeroUsize::MIN);
    let mut progress = progress::StartupProgress::default();
    let samples = drive(
        runtime.clone(),
        session.startup_probe_tracked(
            &prompt,
            &partition,
            probes::Capture::PrefillDiscovery,
            &mut budget,
            &mut progress,
        ),
    )
    .await
    .unwrap();
    assert_eq!(samples.len(), 2);
    assert_eq!(
        executor.deferrals.maintenance_calls.load(Ordering::Acquire),
        1
    );
    let physical_waves = plan::OUTPUT_TOKENS + 1;
    assert_eq!(executor.entries.load(Ordering::Acquire), physical_waves + 1);
    assert_eq!(executor.physical.load(Ordering::Acquire), physical_waves);
    let counts = &progress.current.as_ref().unwrap().counts;
    assert_eq!(
        counts.maintenance_reconciled,
        executor.deferrals.maintenance_calls.load(Ordering::Acquire)
    );
    assert_eq!(counts.zero_submit_retries, 1);
    // One actual terminal wave has no published singleton sample or original
    // unknown-recorder DTO; maintenance and NotSubmitted are not substitutes.
    assert_eq!(executor.completion_calls.load(Ordering::Acquire), 1);
    assert_eq!(counts.host_reconciled, physical_waves);
    assert_eq!(counts.known_physical_waves, physical_waves - 1);
    assert_eq!(counts.reports_without_physical_count, 1);
    assert_eq!(counts.not_submitted, 1);
    assert!(counts.maintenance_turns > counts.maintenance_reconciled);
    session.drain_startup().await.unwrap();
    drop(runtime);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_probe_never_replays_indeterminate_or_failed_submission() {
    for indeterminate in [true, false] {
        let (mut engine, executor) = fixture(SloMode::Enforce, 128).await;
        ContinuousBatchEngine::check_startup_session(&mut engine).unwrap();
        executor
            .panic_before_submit
            .store(indeterminate, Ordering::Release);
        executor
            .fail_after_submit
            .store(!indeterminate, Ordering::Release);
        let mut session = CalibrationSession::new_driver_session(
            engine,
            CalibrationLimits::new(NonZeroUsize::MIN).unwrap(),
        );
        let (prompt, partition) = prompt();
        let mut budget = plan::ProbeBudget::new(NonZeroUsize::MIN);
        assert!(tokio::time::timeout(
            Duration::from_secs(5),
            session.startup_probe(&prompt, &partition, probes::Capture::Warmup, &mut budget,),
        )
        .await
        .unwrap()
        .is_err());
        // Output failure can cancel the driver first; reap its same durable
        // task before examining the final submission classification.
        let drained = session.drain_startup().await;
        assert_eq!(drained.is_err(), indeterminate);
        assert_eq!(session.indeterminate, indeterminate);
        assert!(session.pending.is_none());
        assert_eq!(executor.entries.load(Ordering::Acquire), 1);
        assert_eq!(
            executor.physical.load(Ordering::Acquire),
            usize::from(!indeterminate)
        );
        assert!(session.engine.inner.sequences.read().is_empty());
        session.shutdown().await.unwrap();
    }
}
