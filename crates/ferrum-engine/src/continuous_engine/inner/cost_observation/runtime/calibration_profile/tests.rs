use super::*;
use crate::continuous_engine::inner::calibration::{
    CalibrationLimits, CalibrationProfilePaths, CalibrationSession,
};
use crate::continuous_engine::inner::slo_controller::tests::fixture::fixture_with_width;
use ferrum_scheduler::implementations::continuous::cost_model as model;
use std::{
    fs,
    num::NonZeroUsize,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

struct Clock(AtomicU64);
impl CostObservationClock for Clock {
    fn now_ns(&self) -> Option<u64> {
        Some(self.0.load(Ordering::Acquire))
    }
}

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "ferrum-calibration-artifact-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn paths(&self) -> CalibrationProfilePaths {
        CalibrationProfilePaths {
            profile: self.0.join("training.json"),
            source: self.0.join("training-source.jsonl"),
        }
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

async fn fixture(
    directory: &Directory,
    local_clock_known: bool,
) -> (
    CalibrationSession,
    Arc<EngineCostRuntime>,
    Arc<Clock>,
    SloCostObservationConfig,
) {
    let (mut engine, _, _) = fixture_with_width(1).await;
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    let identity = inner.cost_runtime.as_ref().unwrap().identity.clone();
    let mut config = SloCostObservationConfig::default();
    config.model.min_samples = NonZeroUsize::MIN;
    config.model.max_sample_age_ns = std::num::NonZeroU64::new(30_000_000_000).unwrap();
    config.profile_import.declared_local_clock_max_error_ns =
        local_clock_known.then_some(1_000_000);
    config.profile_export = Some(ferrum_types::SloCostProfileExportConfig {
        path: directory.0.join("live.json"),
        observations_path: directory.0.join("live.jsonl"),
        declared_clock_max_error_ns: Some(1_000_000),
        ..Default::default()
    });
    let clock = Arc::new(Clock(AtomicU64::new(100)));
    let runtime =
        Arc::new(EngineCostRuntime::build(identity, clock.clone(), &config, true).unwrap());
    inner.config.scheduler.slo.cost_observation = config.clone();
    inner.cost_runtime = Some(runtime.clone());
    inner.bg_loop_spawned.store(false, Ordering::Release);
    let session = CalibrationSession::from_fresh_engine(
        engine,
        CalibrationLimits::new(NonZeroUsize::MIN).unwrap(),
    )
    .unwrap();
    (session, runtime, clock, config)
}

fn observation(kv: u32) -> model::WaveCostObservation {
    model::WaveCostObservation {
        fingerprint: model::ExecutionFingerprint {
            model_weights: [1; 32],
            numerical_policy: [2; 32],
            device_runtime: [3; 32],
            execution_config: [4; 32],
        },
        actual_shape: model::WaveExecutionShape {
            host_content_features: None,
            row_multiset_features: None,
            kind: model::WaveKind::Decode,
            path: model::WaveExecutionPath::PlanRuntime,
            provider_signature: [5; 32],
            output_policy_signature: [6; 32],
            numeric_features: None,
            graph_state: model::WaveGraphState::Disabled,
            order: model::BatchOrderSemantics::Ordered,
            decode_kv_tokens: vec![kv],
            prefill_chunks: vec![],
            recurrent_state_bytes: 0,
            restore_bytes: 0,
            maintenance_bytes: 0,
            maintenance_units: 0,
        },
        boundary: model::CostBoundary::PreparationToCommit,
        outcome: model::WaveObservationOutcome::Completed,
        timing: model::WaveTiming {
            wall_total_ns: 10,
            device_elapsed_ns: Some(5),
            stages: Default::default(),
        },
        observed_at_ns: 100,
    }
}

#[tokio::test]
async fn calibration_export_import_validates_the_product_loaded_cut_and_keeps_it_immutable() {
    let directory = Directory::new();
    let (mut session, runtime, clock, config) = fixture(&directory, true).await;
    let training = observation(16);
    runtime.sink.offer(training.clone()).unwrap();
    let model = tokio::time::timeout(
        // The real exporter hashes the current executable and syncs new
        // files. This is a lifecycle watchdog, not an IO latency SLO; debug
        // binaries and parallel export tests can take longer than 3 seconds.
        Duration::from_secs(30),
        session.export_and_load_cost_profile(directory.paths()),
    )
    .await
    .expect("the real worker must finish the accepted export/import cut")
    .unwrap();
    assert_eq!(model.accepted_ordinal(), 1);
    assert_eq!(model.import_receipt().recorded_samples, 1);
    assert_eq!(
        model.import_receipt().file_sha256,
        format!("sha256:{}", model.artifact().profile_sha256)
    );
    assert_eq!(
        model.import_receipt().source_observation_artifact_sha256,
        model.artifact().source_digest
    );
    assert!(matches!(
        model.predict(&training.actual_shape).unwrap(),
        model::CostPrediction::Known(_)
    ));

    let validation = observation(256);
    runtime.sink.offer(validation.clone()).unwrap();
    let checkpoint = runtime.request_checkpoint().unwrap().wait().await.unwrap();
    assert_eq!(checkpoint.accepted_ordinal, 2);
    assert!(matches!(
        runtime.snapshot().unwrap().predict(
            &validation.fingerprint,
            &validation.actual_shape,
            validation.boundary,
            100
        ),
        model::CostPrediction::Known(_)
    ));
    assert!(matches!(
        model.predict(&validation.actual_shape).unwrap(),
        model::CostPrediction::Unknown(_)
    ));
    let file: serde_json::Value =
        serde_json::from_slice(&fs::read(&model.artifact().profile).unwrap()).unwrap();
    assert_eq!(file["samples"].as_array().unwrap().len(), 1);
    assert_eq!(model.audit()["accepted_ordinal"], 1);
    let artifact = model.artifact();
    let cut = CostProfileCutReceipt {
        accepted_ordinal: artifact.accepted_ordinal,
        profile: artifact.profile.clone(),
        profile_sha256: artifact.profile_sha256.clone(),
        profile_bytes: artifact.profile_bytes,
        source: artifact.source.clone(),
        source_sha256: artifact.source_sha256.clone(),
        source_digest: artifact.source_digest,
        source_bytes: artifact.source_bytes,
        retained_samples: artifact.retained_samples,
        raw_retained_observations: artifact.raw_retained_observations,
    };
    // Trailing whitespace is valid JSON but changes the actual imported bytes.
    let mut changed = fs::read(&cut.profile).unwrap();
    changed.push(b' ');
    fs::write(&cut.profile, changed).unwrap();
    assert!(runtime.load_calibration_profile(&config, &cut).is_err());
    session.shutdown().await.unwrap();
    // Clock advancement makes this immutable imported evidence stale; it
    // cannot inherit a later online publication or a caller-supplied old time.
    clock.0.store(60_000_000_100, Ordering::Release);
    assert!(matches!(
        model.predict(&training.actual_shape).unwrap(),
        model::CostPrediction::Unknown(model::CostUnknownReason::StaleSamples)
    ));
    drop(model);
}

#[tokio::test]
async fn calibration_export_import_failure_preserves_the_published_training_evidence() {
    let directory = Directory::new();
    let (mut session, runtime, _, _) = fixture(&directory, false).await;
    runtime.sink.offer(observation(16)).unwrap();
    assert!(session
        .export_and_load_cost_profile(directory.paths())
        .await
        .is_err());
    assert!(directory.paths().profile.is_file());
    assert!(directory.paths().source.is_file());
    let bytes = fs::read(directory.paths().profile).unwrap();
    // Retrying never overwrites a source/profile produced by the first cut.
    assert!(session
        .export_and_load_cost_profile(directory.paths())
        .await
        .is_err());
    assert_eq!(fs::read(directory.paths().profile).unwrap(), bytes);
    session.shutdown().await.unwrap();
}
