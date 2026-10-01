//! Controlled CPU submissions write the real bounded recorder and complete
//! real request owners. These tests make no hardware timing or CUDA claim.
use super::*;
use ferrum_interfaces::execution_cost::ActualWaveEvidenceUnknown;
use parking_lot::Mutex;
use std::collections::{BTreeMap, HashSet};

fn probe() -> (
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

fn requests(executor: &ControlledExecutor) -> HashSet<RequestId> {
    executor
        .submitted_requests
        .lock()
        .iter()
        .flatten()
        .cloned()
        .collect()
}

async fn session() -> (
    CalibrationSession,
    Arc<ControlledExecutor>,
    Arc<EngineCostRuntime>,
) {
    let (mut engine, executor) = fixture(SloMode::Enforce, 128).await;
    let runtime = engine.inner.cost_runtime.as_ref().unwrap().clone();
    ContinuousBatchEngine::check_startup_session(&mut engine).unwrap();
    (
        CalibrationSession::new_driver_session(
            engine,
            CalibrationLimits::new(NonZeroUsize::MIN).unwrap(),
        ),
        executor,
        runtime,
    )
}

#[tokio::test]
async fn route_readiness_discards_partial_owner_and_repeats_complete_fresh_discovery() {
    let (mut session, executor, runtime) = session().await;
    let first_owner = Arc::new(Mutex::new(None));
    let owner = first_owner.clone();
    *executor.actual_observation_unknown.lock() = Some(Box::new(move |prefills, _| {
        let row = prefills.first()?;
        let mut owner = owner.lock();
        let first = owner.get_or_insert_with(|| row.request_id.clone());
        // Reject only the late segment. The accepted early segment must not
        // be spliced into the next fresh owner's discovery.
        (first == &row.request_id && row.chunk.tokens_processed() != 0)
            .then_some(ActualWaveEvidenceUnknown::GraphPath)
    }));
    let (prompt, partition) = probe();
    let mut budget = plan::ProbeBudget::new(NonZeroUsize::new(2).unwrap());
    let samples = drive(
        runtime.clone(),
        session.startup_discovery(&prompt, &partition, probes::Discovery::Prefill, &mut budget),
    )
    .await
    .unwrap();
    assert_eq!(budget.used(), 2);
    assert_eq!(requests(&executor).len(), budget.used());
    assert_eq!(
        samples.len(),
        partition.segment_count(prompt.tokens.get()).unwrap()
    );
    assert!(
        samples[0].accepted_ordinal() > 1,
        "discard the first owner's accepted prefix too"
    );
    assert_eq!(samples[0].shape().exact.prefill_chunks[0].offset, 0);
    assert_eq!(samples[1].shape().exact.prefill_chunks[0].offset, 1);
    assert!(samples[0].accepted_ordinal() < samples[1].accepted_ordinal());
    let waves_per_request =
        partition.segment_count(prompt.tokens.get()).unwrap() + plan::OUTPUT_TOKENS - 1;
    for owner in requests(&executor) {
        assert_eq!(
            executor
                .submitted_requests
                .lock()
                .iter()
                .filter(|rows| rows.contains(&owner))
                .count(),
            waves_per_request,
            "a discarded discovery still completes all output before the fresh owner"
        );
    }
    assert!(session.frontiers().unwrap().is_empty());
    assert!(executor
        .produced_caches
        .lock()
        .iter()
        .all(|cache| cache.upgrade().is_none()));
    session.drain_startup().await.unwrap();
    drop(runtime);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn route_readiness_counts_warmup_and_all_retries_and_checks_decode_preparation() {
    for discovery in [probes::Discovery::Prefill, probes::Discovery::Decode] {
        let (mut session, executor, runtime) = session().await;
        *executor.actual_observation_unknown.lock() = Some(Box::new(|prefills, _| {
            (!prefills.is_empty()).then_some(ActualWaveEvidenceUnknown::GraphPath)
        }));
        let (prompt, partition) = probe();
        let mut budget = plan::ProbeBudget::new(NonZeroUsize::new(3).unwrap());
        drive(
            runtime.clone(),
            session.startup_probe(&prompt, &partition, probes::Capture::Warmup, &mut budget),
        )
        .await
        .unwrap();
        let result = drive(
            runtime.clone(),
            session.startup_discovery(&prompt, &partition, discovery, &mut budget),
        )
        .await;
        assert!(
            result.is_err(),
            "a known decode cannot hide an unknown preparation chain"
        );
        assert_eq!(budget.used(), budget.maximum());
        assert_eq!(requests(&executor).len(), budget.maximum());
        assert!(session.frontiers().unwrap().is_empty());
        assert!(session.engine.inner.prefill_reference_runtime.is_none());
        assert!(!session.indeterminate);
        session.drain_startup().await.unwrap();
        drop(runtime);
        session.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn route_readiness_never_retries_non_graph_unknown_evidence() {
    let (mut session, executor, runtime) = session().await;
    *executor.actual_observation_unknown.lock() = Some(Box::new(|_, _| {
        Some(ActualWaveEvidenceUnknown::ProviderPath)
    }));
    let (prompt, partition) = probe();
    let mut budget = plan::ProbeBudget::new(NonZeroUsize::new(3).unwrap());
    assert!(drive(
        runtime.clone(),
        session.startup_discovery(&prompt, &partition, probes::Discovery::Prefill, &mut budget),
    )
    .await
    .is_err());
    assert_eq!(budget.used(), 1);
    assert_eq!(requests(&executor).len(), 1);
    session.drain_startup().await.unwrap();
    assert!(session.frontiers().unwrap().is_empty());
    drop(runtime);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn route_readiness_bootstrap_installs_only_after_real_routes_become_observable() {
    let (engine, executor) = fixture(SloMode::Enforce, 128).await;
    let runtime = engine.inner.cost_runtime.as_ref().unwrap().clone();
    let routes = Arc::new(Mutex::new(BTreeMap::new()));
    let observed = routes.clone();
    *executor.actual_observation_unknown.lock() = Some(Box::new(move |prefills, _| {
        let row = prefills.first()?;
        let key = (
            row.chunk.total_prompt_tokens(),
            row.chunk.tokens_processed(),
            row.chunk.tokens_to_process(),
        );
        let mut observed = observed.lock();
        let count = observed.entry(key).or_insert(0usize);
        *count += 1;
        // A real warmup submission and a capture submission both lack an
        // observable route. Readiness is proved only by later actual receipts.
        (*count <= 2).then_some(ActualWaveEvidenceUnknown::GraphPath)
    }));
    let original = engine.inner.config.scheduler.slo.clone();
    let engine = drive(runtime.clone(), engine.finish_automatic_startup())
        .await
        .unwrap();
    let reference =
        engine.inner.prefill_reference_runtime.as_ref().expect(
            "complete fresh trials after route readiness must produce a measured reference",
        );
    assert!(reference.startup_evidence().is_some());
    assert!(!routes.lock().is_empty());
    assert!(routes.lock().values().all(|count| *count > 2));
    assert_eq!(engine.inner.config.scheduler.slo, original);
    assert!(engine.inner.sequences.read().is_empty());
    assert!(requests(&executor).len() <= 128);
    assert!(executor
        .produced_caches
        .lock()
        .iter()
        .all(|cache| cache.upgrade().is_none()));
    drop(runtime);
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn route_readiness_never_replaces_a_frozen_trial_with_a_fresh_retry() {
    let (mut session, executor, runtime) = session().await;
    let routes = Arc::new(Mutex::new(BTreeMap::new()));
    let rejected_owner = Arc::new(Mutex::new(None));
    let owner = rejected_owner.clone();
    *executor.actual_observation_unknown.lock() = Some(Box::new(move |prefills, _| {
        let row = prefills.first()?;
        let key = (
            row.chunk.total_prompt_tokens(),
            row.chunk.tokens_processed(),
            row.chunk.tokens_to_process(),
        );
        let mut routes = routes.lock();
        let count = routes.entry(key).or_insert(0usize);
        *count += 1;
        // Each prefill anchor has a warmup and discovery before any frozen
        // trial. The decode anchor reuses the shortest route afterwards, so
        // reject a longer anchor's first post-freeze occurrence only.
        if row.chunk.total_prompt_tokens() > 1 && *count == 3 {
            *owner.lock() = Some(row.request_id.clone());
            Some(ActualWaveEvidenceUnknown::GraphPath)
        } else {
            None
        }
    }));
    let SloLiveStructuredCalibration::AutomaticV1 { settings } = &session
        .configuration()
        .scheduler
        .slo
        .cost_observation
        .live_structured_calibration
    else {
        unreachable!()
    };
    let settings = settings.reference_probe.clone();
    let result = drive(
        runtime.clone(),
        session.collect_startup_reference(&settings),
    )
    .await;
    assert!(result.is_err());
    assert!(
        rejected_owner.lock().is_some(),
        "the rejection must occur in a real frozen trial"
    );
    session.drain_startup().await.unwrap();
    assert_eq!(
        executor.submitted_requests.lock().last().unwrap().first(),
        rejected_owner.lock().as_ref(),
        "a rejected frozen trial cannot start a replacement owner"
    );
    assert!(session.frontiers().unwrap().is_empty());
    assert!(session.engine.inner.prefill_reference_runtime.is_none());
    drop(runtime);
    session.shutdown().await.unwrap();
}
