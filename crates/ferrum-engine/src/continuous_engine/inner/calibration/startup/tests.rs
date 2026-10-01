use super::*;
use crate::continuous_engine::inner::{
    cost_observation::{EngineCostClock, EngineCostRuntime},
    slo_controller::tests::fixture::{fixture_with_width, ControlledExecutor},
};
use ferrum_types::{SloAutomaticCalibrationSettingsV1, SloCostObservationConfig, SloMode};
use futures::FutureExt;
use sha2::{Digest, Sha256};

mod output_settlement;
mod probe_retry;
mod route_readiness;
mod work_budget;

async fn fixture(
    mode: SloMode,
    maximum_requests: usize,
) -> (ContinuousBatchEngine, Arc<ControlledExecutor>) {
    fixture_with_trials(mode, maximum_requests, 1).await
}

async fn fixture_with_trials(
    mode: SloMode,
    maximum_requests: usize,
    repetitions: usize,
) -> (ContinuousBatchEngine, Arc<ControlledExecutor>) {
    let (mut engine, _, executor) = fixture_with_width(1).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.bg_loop_spawned.store(false, Ordering::Release);
    inner.prefill_reference_runtime = None;
    inner.runtime_config.max_model_len = Some(7);
    inner.config.scheduler.slo.mode = mode;
    inner.config.scheduler.slo.cost_profile = None;
    inner.config.scheduler.slo.prefill_reference = None;
    let mut settings = SloAutomaticCalibrationSettingsV1::default();
    settings.reference_probe.fresh_trials_per_anchor = NonZeroUsize::new(repetitions).unwrap();
    settings.reference_probe.maximum_probe_requests = NonZeroUsize::new(maximum_requests).unwrap();
    settings.reference_probe.maximum_duration_ms = NonZeroU64::new(30_000).unwrap();
    let mut config = SloCostObservationConfig::structured_whole_wave_v2();
    config.live_structured_calibration = SloLiveStructuredCalibration::AutomaticV1 { settings };
    let identity = inner.cost_runtime.as_ref().unwrap().identity.clone();
    inner.cost_runtime = Some(Arc::new(
        EngineCostRuntime::build(
            identity,
            Arc::new(EngineCostClock::default()),
            &config,
            false,
        )
        .unwrap(),
    ));
    inner.config.scheduler.slo.cost_observation = config;
    executor
        .emit_cost_observations
        .store(true, Ordering::Release);
    (engine, executor)
}

async fn drive<T>(
    runtime: Arc<EngineCostRuntime>,
    future: impl std::future::Future<Output = T>,
) -> T {
    tokio::pin!(future);
    tokio::time::timeout(Duration::from_secs(40), async {
        loop {
            if let Some(value) = future.as_mut().now_or_never() {
                return value;
            }
            runtime.consume_samples();
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("automatic startup did not complete its real FIFO checkpoint")
}

#[tokio::test]
async fn automatic_reference_probe_protocol_completes_frozen_real_trials() {
    let (mut engine, _) = fixture(SloMode::Enforce, 128).await;
    let runtime = engine.inner.cost_runtime.as_ref().unwrap().clone();
    let SloLiveStructuredCalibration::AutomaticV1 { settings } = &engine
        .inner
        .config
        .scheduler
        .slo
        .cost_observation
        .live_structured_calibration
    else {
        unreachable!()
    };
    let settings = settings.reference_probe.clone();
    ContinuousBatchEngine::check_startup_session(&mut engine).unwrap();
    let mut session = CalibrationSession::new_driver_session(
        engine,
        CalibrationLimits::new(NonZeroUsize::MIN).unwrap(),
    );
    let result = drive(
        runtime.clone(),
        session.collect_startup_reference(&settings),
    )
    .await;
    session.drain_startup().await.unwrap();
    drop(runtime);
    session.shutdown().await.unwrap();
    if let Err(error) = result {
        panic!("complete automatic reference protocol: {error}");
    }
}

#[tokio::test]
async fn automatic_reference_memory_bootstrap_installs_only_after_complete_probes_without_config_rewrite(
) {
    let (engine, executor) = fixture(SloMode::Enforce, 128).await;
    let original = engine.inner.config.scheduler.slo.clone();
    let runtime = engine.inner.cost_runtime.as_ref().unwrap().clone();
    let engine = drive(runtime.clone(), engine.finish_automatic_startup())
        .await
        .unwrap();
    let reference = engine
        .inner
        .prefill_reference_runtime
        .as_ref()
        .expect("real complete probes must install reference");
    assert!(executor.entries.load(Ordering::Acquire) > 0);
    assert!(executor.physical.load(Ordering::Acquire) > 0);
    assert!(
        executor
            .produced_caches
            .lock()
            .iter()
            .all(|cache| cache.upgrade().is_none()),
        "complete probes must release their actual per-request KV owners"
    );
    assert_eq!(engine.inner.config.scheduler.slo, original);
    assert!(!engine.inner.manual_calibration_driver);
    assert!(!engine.inner.automatic_reference_bootstrap);
    assert!(!engine.inner.bg_loop_spawned.load(Ordering::Acquire));
    assert!(engine.inner.sequences.read().is_empty());
    assert_eq!(engine.inner.scheduler.active_count(), 0);
    assert_eq!(engine.inner.scheduler.waiting_count(), 0);
    assert_eq!(
        reference.calibration().piecewise_domain(),
        Some((NonZeroU32::MIN, NonZeroU32::new(4).unwrap()))
    );
    assert!(reference.calibration().source_path().is_none());
    let (source, artifact) = reference.startup_evidence().unwrap();
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(artifact)),
        reference.calibration().identity().artifact_sha256
    );
    let first: serde_json::Value =
        serde_json::from_slice(source.split(|b| *b == b'\n').next().unwrap()).unwrap();
    assert_eq!(first["header"]["artifact_type"], "ferrum.reference-source");
    assert!(first["header"].get("declared_clock_max_error_ns").is_none());
    assert!(
        runtime.snapshot().is_none(),
        "reference cannot manufacture a cost predictor"
    );
    assert!(runtime.profile_receipt().is_none());
    // Bootstrap waves precede the explicit automatic discovery start barrier.
    let audit = serde_json::to_value(runtime.audit_snapshot()).unwrap();
    assert_eq!(audit["live_calibration"]["population"]["issued"], 0);
    drop(runtime);
    engine.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn automatic_reference_cancelled_probe_waiter_is_drained_before_service_can_resume() {
    let (mut engine, executor) = fixture(SloMode::Enforce, 128).await;
    ContinuousBatchEngine::check_startup_session(&mut engine).unwrap();
    let mut session = CalibrationSession::new_driver_session(
        engine,
        CalibrationLimits::new(NonZeroUsize::MIN).unwrap(),
    );
    let prompt = plan::ProbePrompt {
        text: " a a".into(),
        tokens: NonZeroU32::new(2).unwrap(),
    };
    let partition =
        ferrum_scheduler::implementations::continuous::prefill_reference::PiecewiseReferenceSpec {
            minimum_prompt_tokens: prompt.tokens,
            maximum_prompt_tokens: prompt.tokens,
            body_endpoints: vec![NonZeroU32::MIN],
        };
    executor.park.store(true, Ordering::Release);
    let mut budget = plan::ProbeBudget::new(NonZeroUsize::MIN);
    let mut progress = progress::StartupProgress::default();
    let mut probe = Box::pin(session.startup_probe_tracked(
        &prompt,
        &partition,
        probes::Capture::Warmup,
        &mut budget,
        &mut progress,
    ));
    tokio::time::timeout(Duration::from_secs(5), async {
        tokio::select! {
            result = probe.as_mut() => panic!("probe returned before controlled executor entry: {}", result.err().map_or("success".into(), |error| error.to_string())),
            () = executor.entered.notified() => {}
        }
    }).await.expect("probe did not reach its real guarded executor entry");
    tokio::time::advance(Duration::from_millis(31)).await;
    drop(probe); // Same waiter cancellation caused by the startup duration limit.
    let diagnostic = progress.current.as_ref().unwrap();
    assert_eq!(diagnostic.anchor_tokens, prompt.tokens.get());
    assert_eq!(diagnostic.capture, "warmup");
    assert_eq!(diagnostic.request_ordinal, budget.used());
    assert_eq!(diagnostic.stage, progress::Stage::Wave);
    assert_eq!(diagnostic.counts.wave_attempts, 1);
    assert_eq!(diagnostic.counts.host_reconciled, 0);
    assert_eq!(diagnostic.counts.zero_submit_retries, 0);
    assert!(
        diagnostic.slowest_wave.is_none(),
        "pending work has no finished timing"
    );
    assert_eq!(progress.completed_requests, 0);
    assert!(session.pending.is_some());
    assert!(session.engine.inner.automatic_reference_bootstrap);
    executor.park.store(false, Ordering::Release);
    executor.resume.notify_one();
    tokio::time::timeout(
        Duration::from_secs(5),
        session.drain_startup_tracked(Some(&mut progress)),
    )
    .await
    .unwrap()
    .unwrap();
    let diagnostic = progress.current.as_ref().unwrap();
    let finished = diagnostic.slowest_wave.as_ref().unwrap();
    let controller = finished.controller.as_ref().unwrap();
    assert_eq!(finished.position.attempt, 1);
    assert_eq!(finished.position.prefill_progress, Some((0, 2)));
    assert_eq!(finished.position.generated_tokens, 0);
    let physical = executor.physical.load(Ordering::Acquire);
    assert!(finished
        .physical_waves
        .is_none_or(|observed| observed == physical));
    let known_physical = finished.physical_waves.unwrap_or(0);
    assert_eq!(diagnostic.counts.known_physical_waves, known_physical);
    assert_eq!(
        diagnostic.counts.reports_without_physical_count,
        usize::from(finished.physical_waves.is_none())
    );
    assert_eq!(
        diagnostic.counts.zero_submit_retries, 0,
        "draining never retries"
    );
    match finished.submission {
        Some(CalibrationSubmissionState::HostReconciled) => {
            assert_eq!(physical, 1);
            assert_eq!(diagnostic.counts.host_reconciled, 1);
        }
        Some(CalibrationSubmissionState::NotSubmitted) => {
            // Dropping the output waiter can invalidate the original host
            // guard before physical submission; retain that actual result.
            assert_eq!(physical, 0);
            assert_eq!(diagnostic.counts.not_submitted, 1);
        }
        other => panic!("controlled conclusive cleanup: {other:?}"),
    }
    assert_eq!(
        progress.completed_requests, 0,
        "reaping cannot finish the cancelled three-output probe"
    );
    assert!(controller.executor_await.wall_ns_total >= 31_000_000);
    assert_eq!(controller.executor_await.calls, 1);
    assert_eq!(controller.finalized_transactions, 1);
    // A second drain cannot observe or count the same model wave twice.
    session
        .drain_startup_tracked(Some(&mut progress))
        .await
        .unwrap();
    assert_eq!(
        progress
            .current
            .as_ref()
            .unwrap()
            .counts
            .known_physical_waves,
        known_physical
    );
    assert!(session.pending.is_none());
    assert!(!session.indeterminate);
    assert!(session.engine.inner.sequences.read().is_empty());
    assert_eq!(session.engine.inner.scheduler.active_count(), 0);
    assert_eq!(session.engine.inner.scheduler.waiting_count(), 0);
    assert!(session.engine.inner.prefill_reference_runtime.is_none());
    assert!(executor
        .produced_caches
        .lock()
        .iter()
        .all(|cache| cache.upgrade().is_none()));
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_reference_budget_exhaustion_keeps_unknown_and_releases_private_driver() {
    let (engine, executor) = fixture(SloMode::Enforce, 1).await;
    let runtime = engine.inner.cost_runtime.as_ref().unwrap().clone();
    let engine = drive(runtime.clone(), engine.finish_automatic_startup())
        .await
        .unwrap();
    assert!(engine.inner.prefill_reference_runtime.is_none());
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert!(!engine.inner.manual_calibration_driver);
    assert!(!engine.inner.automatic_reference_bootstrap);
    assert_eq!(engine.inner.config.scheduler.slo.mode, SloMode::Enforce);
    assert!(engine.inner.sequences.read().is_empty());
    drop(runtime);
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_bootstrap_does_not_open_public_manual_enforce_session_or_reuse_an_engine() {
    let (engine, _) = fixture(SloMode::Enforce, 128).await;
    assert!(CalibrationSession::from_fresh_engine(
        engine,
        CalibrationLimits::new(NonZeroUsize::MIN).unwrap()
    )
    .is_err());
    let (mut engine, _) = fixture(SloMode::Enforce, 128).await;
    engine.inner.bg_loop_spawned.store(true, Ordering::Release);
    assert!(ContinuousBatchEngine::check_startup_session(&mut engine).is_err());
    assert!(!engine.inner.automatic_reference_bootstrap);
    engine.inner.bg_loop_spawned.store(false, Ordering::Release);
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_reference_directory_persists_exact_memory_evidence_and_reports_optional_quota_failure(
) {
    for source_limit in [1u64, 1 << 20] {
        let directory = std::env::temp_dir().join(format!(
            "ferrum-bootstrap-reference-{}",
            uuid::Uuid::new_v4()
        ));
        let (mut engine, executor) = fixture(SloMode::Enforce, 128).await;
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
        settings.diagnostics = ferrum_types::SloAutomaticCalibrationDiagnosticsV1::Directory {
            directory: directory.clone(),
            maximum_source_bytes: NonZeroU64::new(source_limit).unwrap(),
            maximum_total_bytes: NonZeroU64::new(4 << 20).unwrap(),
            maximum_retained_generations: NonZeroUsize::new(2).unwrap(),
        };
        inner.cost_runtime = Some(Arc::new(
            EngineCostRuntime::build(
                inner.cost_runtime.as_ref().unwrap().identity.clone(),
                Arc::new(EngineCostClock::default()),
                &inner.config.scheduler.slo.cost_observation,
                false,
            )
            .unwrap(),
        ));
        let runtime = engine.inner.cost_runtime.as_ref().unwrap().clone();
        let result = drive(runtime.clone(), engine.finish_automatic_startup()).await;
        assert!(executor.physical.load(Ordering::Acquire) > 0);
        assert!(executor
            .produced_caches
            .lock()
            .iter()
            .all(|cache| cache.upgrade().is_none()));
        let engine = result.expect("optional diagnostic storage cannot invalidate complete probes");
        let installed = engine
            .inner
            .prefill_reference_runtime
            .as_ref()
            .expect("complete original probes still install the memory reference");
        assert!(installed.calibration().source_path().is_none());
        let (source, reference) = installed.startup_evidence().unwrap();
        assert!(!source.is_empty());
        assert_eq!(
            <[u8; 32]>::from(Sha256::digest(reference)),
            installed.calibration().identity().artifact_sha256
        );
        assert!(!engine.inner.manual_calibration_driver);
        assert!(!engine.inner.automatic_reference_bootstrap);
        assert!(engine.inner.sequences.read().is_empty());
        assert_eq!(engine.inner.scheduler.active_count(), 0);
        assert_eq!(engine.inner.scheduler.waiting_count(), 0);
        assert!(
            runtime.snapshot().is_none(),
            "reference is not a whole-wave cost model"
        );
        assert!(runtime.profile_receipt().is_none());
        let audit = serde_json::to_value(runtime.audit_snapshot()).unwrap();
        let automatic = &audit["live_calibration"]["automatic"];
        assert_eq!(automatic["start_requested"], true);
        assert_eq!(audit["live_calibration"]["population"]["issued"], 0);
        let failures = &automatic["diagnostic_failures"];
        let mut pending = if directory.exists() {
            vec![directory.clone()]
        } else {
            Vec::new()
        };
        let mut persisted_source = None;
        let mut persisted_reference = None;
        while let Some(parent) = pending.pop() {
            for entry in std::fs::read_dir(parent).unwrap() {
                let entry = entry.unwrap();
                if entry.file_type().unwrap().is_dir() {
                    pending.push(entry.path());
                } else if entry.file_name() == "source.jsonl" {
                    persisted_source = Some(std::fs::read(entry.path()).unwrap());
                } else if entry.file_name() == "reference.json" {
                    persisted_reference = Some(std::fs::read(entry.path()).unwrap());
                }
            }
        }
        if source_limit == 1 {
            assert!(source.len() as u64 > source_limit);
            assert!(
                failures["count"].as_u64().unwrap() > 0,
                "storage failure must be reported: {audit}"
            );
            assert_eq!(failures["first"]["generation"], 0);
            assert_eq!(failures["first"]["stage"], "reference");
            assert!(!failures["first"]["reason"].as_str().unwrap().is_empty());
            assert!(
                persisted_source.is_none(),
                "no over-quota source is published"
            );
            assert!(
                persisted_reference.is_none(),
                "no incomplete pair is published"
            );
        } else {
            assert_eq!(failures["count"], 0);
            assert_eq!(persisted_source.as_deref(), Some(source));
            assert_eq!(persisted_reference.as_deref(), Some(reference));
        }
        engine.shutdown().await.unwrap();
        drop(runtime);
        if directory.exists() {
            std::fs::remove_dir_all(directory).unwrap();
        }
    }
}
