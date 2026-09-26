//! Hardware-independent controlled executor; no Metal numerical/performance claim.
use super::*;
use crate::continuous_engine::inner::cost_observation::{EngineCostClock, EngineCostRuntime};
use ferrum_scheduler::implementations::continuous::prefill_reference::{
    load_prefill_reference, ReferenceEstimator, ReferenceGraphRoutes, ReferenceProtocolV1,
};
use std::{num::NonZeroU64, path::PathBuf};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("ferrum-reference-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn cut(&self) -> CalibrationProfilePaths {
        CalibrationProfilePaths {
            profile: self.0.join("cut.json"),
            source: self.0.join("cut-source.jsonl"),
        }
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

async fn observed_fixture(directory: &Directory) -> (CalibrationSession, Arc<ControlledExecutor>) {
    observed_fixture_with_owner_capacity(directory, 4).await
}

async fn observed_fixture_with_owner_capacity(
    directory: &Directory,
    owners: usize,
) -> (CalibrationSession, Arc<ControlledExecutor>) {
    let (mut session, executor) = fixture(owners).await;
    let inner = Arc::get_mut(&mut session.engine.inner).unwrap();
    let identity = inner.cost_runtime.as_ref().unwrap().identity.clone();
    let mut config = ferrum_types::SloCostObservationConfig::default();
    config.model.min_samples = NonZeroUsize::MIN;
    config.profile_import.declared_local_clock_max_error_ns = Some(1_000_000);
    config.profile_export = Some(ferrum_types::SloCostProfileExportConfig {
        path: directory.0.join("live.json"),
        observations_path: directory.0.join("live.jsonl"),
        declared_clock_max_error_ns: Some(1_000_000),
        ..Default::default()
    });
    let clock = Arc::new(EngineCostClock::default());
    inner.cost_runtime = Some(Arc::new(
        EngineCostRuntime::build(identity, clock, &config, false).unwrap(),
    ));
    inner.config.scheduler.slo.cost_observation = config;
    executor
        .emit_cost_observations
        .store(true, Ordering::Release);
    (session, executor)
}

// These tests check exact source/owner/reference joins. The production sink
// deliberately drops observations on worker contention, so use the existing
// manual trainer to keep that unrelated race out of the reference fixtures.
// We still consume the real FIFO, exporter, checkpoint and product importer.
async fn drive_training<T>(
    runtime: Arc<EngineCostRuntime>,
    future: impl std::future::Future<Output = T>,
) -> T {
    tokio::pin!(future);
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            if let Some(value) = future.as_mut().now_or_never() {
                return value;
            }
            runtime.consume_samples();
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("manual reference training must finish its checkpoint")
}

async fn freeze(
    session: &mut CalibrationSession,
    plan: CalibrationReferencePlan,
    discovery: Vec<CalibrationReferenceDiscoverySample>,
) -> Result<CalibrationReferenceCollector> {
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    drive_training(runtime, session.freeze_reference_plan(plan, discovery)).await
}

async fn export(
    session: &mut CalibrationSession,
    paths: CalibrationProfilePaths,
) -> Result<ImportedCalibrationModel> {
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    drive_training(runtime, session.export_and_load_cost_profile(paths)).await
}

async fn next_wave(
    session: &mut CalibrationSession,
    executor: &ControlledExecutor,
    id: &RequestId,
    output: &mut CreditedOutputSession,
) -> (CalibrationFrontier, CalibrationWaveReport) {
    let before = frontier(session, id);
    let row = if before.prefill_progress().is_some() {
        before.prefill_work(NonZeroU32::MIN).unwrap()
    } else {
        before.decode_work().unwrap()
    };
    let report = wave(session, executor, vec![row]).await;
    assert!(
        matches!(report.observation, CalibrationObservation::Observed { .. }),
        "{:?}",
        report.observation
    );
    if frontier(session, id).generated_tokens() > before.generated_tokens() {
        drop(bounded(output.frames.next()).await.unwrap());
        ready(session, id, false).await;
    }
    (before, report)
}

async fn discovery(
    session: &mut CalibrationSession,
    executor: &ControlledExecutor,
) -> (
    CalibrationReferencePlan,
    Vec<CalibrationReferenceDiscoverySample>,
    CalibrationFrontier,
) {
    let (id, mut output) = add(session, 2).await;
    admit(session).await;
    let initial = frontier(session, &id);
    let input = *initial.request_evidence();
    let mut rows = Vec::new();
    for _ in 0..3 {
        let (before, report) = next_wave(session, executor, &id, &mut output).await;
        rows.push(
            session
                .capture_reference_discovery(&before, &report)
                .unwrap(),
        );
    }
    let plan = CalibrationReferencePlan {
        piecewise: None,
        reference_revision: NonZeroU64::MIN,
        protocol: ReferenceProtocolV1 {
            graph_routes: Default::default(),
            granule_tokens: NonZeroU32::MIN,
            repetitions: NonZeroUsize::MIN,
            estimator: ReferenceEstimator::UpperMedianWallV1,
            input_preprocessing_sha256: [7; 32],
            measurement_conditions_sha256: [8; 32],
            prefill_host: rows[0].host_features(),
            decode_host: rows[2].host_features(),
            decode_shape: rows[2].shape(),
        },
        decode_input_tokens: NonZeroU32::new(input.original_input_tokens as u32).unwrap(),
        decode_input_tokens_sha256: input.original_input_tokens_sha256,
        curves: vec![CalibrationReferenceCurve {
            total_prompt_tokens: NonZeroU32::new(2).unwrap(),
            input_tokens_sha256: input.original_input_tokens_sha256,
            partition: vec![rows[0].shape(), rows[1].shape()],
        }],
        limits: Default::default(),
    };
    drop(output);
    (plan, rows, initial)
}

async fn trial(
    session: &mut CalibrationSession,
    executor: &ControlledExecutor,
    collector: &mut CalibrationReferenceCollector,
    key: CalibrationReferenceTrial,
) {
    let (id, mut output) = add(session, 2).await;
    admit(session).await;
    session
        .begin_reference_trial(collector, key, &frontier(session, &id))
        .unwrap();
    loop {
        let (_, report) = next_wave(session, executor, &id, &mut output).await;
        if collector.observe(key, &report).unwrap() {
            break;
        }
    }
    drop(output);
}

#[tokio::test]
async fn actual_discovery_freeze_fresh_trials_and_cut_produce_product_loadable_reference() {
    let directory = Directory::new();
    let (mut session, executor) = observed_fixture(&directory).await;
    let (plan, discovery, _) = discovery(&mut session, &executor).await;
    let fingerprint = match &session.engine.inner.cost_runtime.as_ref().unwrap().identity {
        ferrum_interfaces::execution_cost::ExecutorCostIdentityAvailability::Known(identity) => {
            identity.clone()
        }
        _ => panic!("known controlled executor"),
    };
    let mut collector = freeze(&mut session, plan, discovery).await.unwrap();
    assert_eq!(collector.frozen_accepted_ordinal(), 3);
    trial(
        &mut session,
        &executor,
        &mut collector,
        CalibrationReferenceTrial::Prefill {
            curve: 0,
            repetition: 0,
        },
    )
    .await;
    trial(
        &mut session,
        &executor,
        &mut collector,
        CalibrationReferenceTrial::Decode { repetition: 0 },
    )
    .await;
    let model = export(&mut session, directory.cut()).await.unwrap();
    let output = directory.0.join("reference.json");
    let receipt = collector.finish(model.artifact(), &output).unwrap();
    let expected =
        ferrum_scheduler::implementations::continuous::cost_model::ExecutionFingerprint {
            model_weights: fingerprint.model_weights,
            numerical_policy: fingerprint.numerical_policy,
            device_runtime: fingerprint.device_runtime,
            execution_config: fingerprint.execution_config,
        };
    let loaded = load_prefill_reference(
        &output,
        &expected,
        receipt.protocol_sha256,
        &Default::default(),
    )
    .unwrap();
    assert!(loaded.tau_ref_ns().get() > 0);
    assert_eq!(
        loaded
            .curve(NonZeroU32::new(2).unwrap())
            .unwrap()
            .points
            .len(),
        3
    );
    let wire: serde_json::Value = serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
    assert_eq!(
        wire["curves"][0]["trials"][0]["samples"][0]["record"]["ordinal"], 3,
        "accepted ordinal 4 must join actual source_record 3"
    );
    assert_eq!(receipt.discovery_records.len(), 3);
    assert_eq!(receipt.training_accepted_ordinal, 8);
    let id = RequestId::new();
    let mut binding = loaded
        .bind(
            id.clone(),
            NonZeroU64::MIN,
            NonZeroU32::new(2).unwrap(),
            10,
            1000,
            0,
        )
        .unwrap();
    let partial = binding
        .record_committed(&id, NonZeroU64::MIN, 0, 1, false)
        .unwrap();
    assert!(partial > 0);
    assert_eq!(
        binding
            .record_committed(&id, NonZeroU64::MIN, 0, 1, false)
            .unwrap(),
        0,
        "physical prefix replay cannot reset useful reference work"
    );
    assert!(
        binding
            .record_committed(&id, NonZeroU64::MIN, 1, 2, true)
            .unwrap()
            > 0
    );
    assert_eq!(binding.admitted_at_ns(), 10);
    assert_eq!(binding.logical_high_water(), 2);
    assert!(
        loaded.curve(NonZeroU32::new(3).unwrap()).is_err(),
        "no interpolated expanded recompute curve"
    );
    drop(model);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn reference_freeze_rejects_changed_discovery_partition() {
    let directory = Directory::new();
    let (mut session, executor) = observed_fixture(&directory).await;
    let (mut plan, discovered, _) = discovery(&mut session, &executor).await;
    plan.curves[0].partition.swap(0, 1);
    assert!(freeze(&mut session, plan, discovered).await.is_err());
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn exact_graph_policy_freezes_only_the_complete_actual_discovery_route() {
    for changed_route in [false, true] {
        let directory = Directory::new();
        let (mut session, executor) = observed_fixture(&directory).await;
        let (mut plan, discovery, _) = discovery(&mut session, &executor).await;
        plan.protocol.graph_routes = ReferenceGraphRoutes::ExactObserved;
        if changed_route {
            plan.protocol.decode_shape.exact.graph_state =
                ferrum_scheduler::implementations::continuous::cost_profile::ProfileGraphState::Warm;
            assert!(freeze(&mut session, plan, discovery).await.is_err());
        } else {
            let expected_hash = plan.protocol_sha256().unwrap();
            let mut collector = freeze(&mut session, plan, discovery).await.unwrap();
            trial(
                &mut session,
                &executor,
                &mut collector,
                CalibrationReferenceTrial::Prefill {
                    curve: 0,
                    repetition: 0,
                },
            )
            .await;
            trial(
                &mut session,
                &executor,
                &mut collector,
                CalibrationReferenceTrial::Decode { repetition: 0 },
            )
            .await;
            let model = export(&mut session, directory.cut()).await.unwrap();
            let output = directory.0.join("exact-graph-reference.json");
            let receipt = collector.finish(model.artifact(), &output).unwrap();
            assert_eq!(receipt.protocol_sha256, expected_hash);
        }
        session.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn reference_rejects_changed_original_cut_bytes() {
    let directory = Directory::new();
    let (mut session, executor) = observed_fixture(&directory).await;
    let (plan, discovery, _) = discovery(&mut session, &executor).await;
    let mut collector = freeze(&mut session, plan, discovery).await.unwrap();
    trial(
        &mut session,
        &executor,
        &mut collector,
        CalibrationReferenceTrial::Prefill {
            curve: 0,
            repetition: 0,
        },
    )
    .await;
    trial(
        &mut session,
        &executor,
        &mut collector,
        CalibrationReferenceTrial::Decode { repetition: 0 },
    )
    .await;
    let model = export(&mut session, directory.cut()).await.unwrap();
    use std::io::Write;
    std::fs::OpenOptions::new()
        .append(true)
        .open(&model.artifact().source)
        .unwrap()
        .write_all(b"\n")
        .unwrap();
    let output = directory.0.join("reference.json");
    assert!(collector.finish(model.artifact(), &output).is_err());
    assert!(!output.exists());
    drop(model);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn discovery_owner_cannot_be_relabelled_as_a_fresh_trial() {
    let directory = Directory::new();
    let (mut session, executor) = observed_fixture(&directory).await;
    let (plan, discovered, old_frontier) = discovery(&mut session, &executor).await;
    let mut collector = freeze(&mut session, plan, discovered).await.unwrap();
    assert!(session
        .begin_reference_trial(
            &mut collector,
            CalibrationReferenceTrial::Prefill {
                curve: 0,
                repetition: 0
            },
            &old_frontier
        )
        .is_err());
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_missing_decode_trial_cannot_publish_a_partial_reference() {
    let directory = Directory::new();
    let (mut session, executor) = observed_fixture(&directory).await;
    let (plan, discovered, _) = discovery(&mut session, &executor).await;
    let mut collector = freeze(&mut session, plan, discovered).await.unwrap();
    trial(
        &mut session,
        &executor,
        &mut collector,
        CalibrationReferenceTrial::Prefill {
            curve: 0,
            repetition: 0,
        },
    )
    .await;
    let model = export(&mut session, directory.cut()).await.unwrap();
    let output = directory.0.join("reference.json");
    assert!(collector.finish(model.artifact(), &output).is_err());
    assert!(!output.exists());
    drop(model);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn real_training_capacity_rejections_cannot_become_reference_measurements() {
    let directory = Directory::new();
    let (mut session, executor) = observed_fixture(&directory).await;
    let inner = Arc::get_mut(&mut session.engine.inner).unwrap();
    let identity = inner.cost_runtime.as_ref().unwrap().identity.clone();
    let mut config = inner.config.scheduler.slo.cost_observation.clone();
    config.model.max_buckets = NonZeroUsize::MIN;
    config.profile_export.as_mut().unwrap().path = directory.0.join("limited-live.json");
    config.profile_export.as_mut().unwrap().observations_path =
        directory.0.join("limited-live.jsonl");
    inner.cost_runtime = Some(Arc::new(
        EngineCostRuntime::build(
            identity,
            inner.cost_runtime.as_ref().unwrap().clock.clone(),
            &config,
            false,
        )
        .unwrap(),
    ));
    inner.config.scheduler.slo.cost_observation = config;
    let (plan, discovered, _) = discovery(&mut session, &executor).await;
    let mut collector = freeze(&mut session, plan, discovered).await.unwrap();
    trial(
        &mut session,
        &executor,
        &mut collector,
        CalibrationReferenceTrial::Prefill {
            curve: 0,
            repetition: 0,
        },
    )
    .await;
    trial(
        &mut session,
        &executor,
        &mut collector,
        CalibrationReferenceTrial::Decode { repetition: 0 },
    )
    .await;
    let model = export(&mut session, directory.cut()).await.unwrap();
    assert!(model.artifact().retained_samples < model.artifact().raw_retained_observations);
    let output = directory.0.join("reference.json");
    let error = collector.finish(model.artifact(), &output).unwrap_err();
    assert!(error.to_string().contains("Recorded"), "{error}");
    assert!(!output.exists());
    drop(model);
    session.shutdown().await.unwrap();
}

async fn recompute_partial(session: &mut CalibrationSession, id: &RequestId) {
    // The same sequence release and scheduler transition as real prefill
    // capacity deferral; neither generation nor token progress is fabricated.
    {
        let inner = &session.engine.inner;
        let _iteration = inner.iteration_lock.lock().await;
        let resources = inner
            .sequences
            .write()
            .get_mut(id)
            .unwrap()
            .take_physical_resources_for_recompute();
        inner
            .release_sequence_physical_resources(id, resources)
            .await;
        assert!(inner.scheduler.defer_prefill_to_waiting(id));
    }
    admit(session).await;
}

#[tokio::test]
async fn reference_registration_rejects_stale_handle_and_live_recomputed_zero_prefix() {
    let directory = Directory::new();
    let (mut session, executor) = observed_fixture(&directory).await;
    let (plan, discovered, _) = discovery(&mut session, &executor).await;
    let mut collector = freeze(&mut session, plan, discovered).await.unwrap();
    let (id, mut output) = add(&mut session, 2).await;
    admit(&mut session).await;
    let original = frontier(&session, &id);
    next_wave(&mut session, &executor, &id, &mut output).await;
    recompute_partial(&mut session, &id).await;
    let replay = frontier(&session, &id);
    assert_eq!(replay.prefill_progress(), original.prefill_progress());
    assert_eq!(replay.generated_tokens(), 0);
    assert_ne!(replay.work_generation(), original.work_generation());
    let key = CalibrationReferenceTrial::Prefill {
        curve: 0,
        repetition: 0,
    };
    let submissions = executor.physical.load(Ordering::Acquire);
    assert!(session
        .begin_reference_trial(&mut collector, key, &original)
        .is_err());
    assert!(session
        .begin_reference_trial(&mut collector, key, &replay)
        .is_err());
    assert_eq!(executor.physical.load(Ordering::Acquire), submissions);
    drop(output);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn unreported_recompute_and_replay_cannot_join_a_reference_continuation() {
    let directory = Directory::new();
    let (mut session, executor) = observed_fixture(&directory).await;
    let (plan, discovered, _) = discovery(&mut session, &executor).await;
    let mut collector = freeze(&mut session, plan, discovered).await.unwrap();
    let (id, mut output) = add(&mut session, 2).await;
    admit(&mut session).await;
    let key = CalibrationReferenceTrial::Prefill {
        curve: 0,
        repetition: 0,
    };
    let original = frontier(&session, &id);
    session
        .begin_reference_trial(&mut collector, key, &original)
        .unwrap();
    let (_, partial) = next_wave(&mut session, &executor, &id, &mut output).await;
    assert!(!collector.observe(key, &partial).unwrap());
    let committed = frontier(&session, &id);
    recompute_partial(&mut session, &id).await;
    // Intentionally omit the real replay wave from the reference collector.
    // Its apparent next prefix then matches, but its generation must not.
    next_wave(&mut session, &executor, &id, &mut output).await;
    let replayed = frontier(&session, &id);
    assert_eq!(replayed.prefill_progress(), committed.prefill_progress());
    assert!(replayed.work_generation() > committed.work_generation());
    let (_, final_wave) = next_wave(&mut session, &executor, &id, &mut output).await;
    assert!(collector.observe(key, &final_wave).is_err());
    // A rejected join cannot complete or advance the existing trial.
    assert!(collector.observe(key, &final_wave).is_err());
    drop(output);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn session_reference_origins_are_bounded_and_failed_add_does_not_replace_live_origin() {
    let (mut session, _) = fixture(1).await;
    let (id, output) = add(&mut session, 2).await;
    let original = session.reference_origins[&id];
    let rejected =
        ferrum_types::InferenceRequest::new("test", session.configuration().model.model_id.clone());
    assert!(session
        .add_request(
            rejected,
            InferenceRequestContext::capture(),
            Arc::new(OutputProjectionContract::cli_text())
        )
        .await
        .is_err());
    assert_eq!(session.reference_origins.len(), 1);
    assert_eq!(session.reference_origins[&id], original);
    drop(output);
    bounded(async {
        while session.engine.inner.sequences.read().contains_key(&id) {
            session.step(CalibrationAction::Reap).await.unwrap();
            tokio::task::yield_now().await;
        }
    })
    .await;
    let (next, output) = add(&mut session, 2).await;
    assert_eq!(session.reference_origins.len(), 1);
    assert!(!session.reference_origins.contains_key(&id));
    assert!(session.reference_origins.contains_key(&next));
    drop(output);
    session.shutdown().await.unwrap();
}

#[path = "tests/piecewise.rs"]
mod piecewise;
