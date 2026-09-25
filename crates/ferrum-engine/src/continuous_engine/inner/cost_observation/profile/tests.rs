use super::*;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    num::NonZeroUsize,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

struct Clock(AtomicU64);
impl CostObservationClock for Clock {
    fn now_ns(&self) -> Option<u64> {
        Some(self.0.load(Ordering::Relaxed))
    }
}

struct Fixture {
    path: PathBuf,
    config: SloCostObservationConfig,
    profile: file::CostProfileFile,
}

fn fingerprint() -> model::ExecutionFingerprint {
    model::ExecutionFingerprint {
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }
}

fn identity() -> ExecutorCostIdentityAvailability {
    let f = fingerprint();
    ExecutorCostIdentityAvailability::Known(Arc::new(ExecutorCostIdentity {
        schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
        model_weights: f.model_weights,
        numerical_policy: f.numerical_policy,
        device_runtime: f.device_runtime,
        execution_config: f.execution_config,
    }))
}

fn load_clock() -> file::ProfileLoadClock {
    file::ProfileLoadClock {
        wall_unix_ns: Some(1000),
        wall_max_error_ns: Some(0),
        monotonic_now_ns: 0,
    }
}

#[test]
fn structured_observe_without_artifact_never_starts_legacy_training() {
    let config = SloCostObservationConfig::structured_whole_wave_v1();
    let seed = load_seed(&identity(), &config, None, None).unwrap();
    assert!(seed.trainer.is_none());
    assert!(seed.snapshot.is_none());
    assert!(seed.receipt.is_none());
}

#[test]
fn structured_import_rejects_legacy_artifact_and_undeclared_clock() {
    let fixture = Fixture::new();
    let mut config = SloCostObservationConfig::structured_whole_wave_v1();
    assert!(load_seed(
        &identity(),
        &config,
        Some(&fixture.path),
        Some(load_clock())
    )
    .is_err());
    config.profile_import.declared_local_clock_max_error_ns = Some(0);
    assert!(load_seed(
        &identity(),
        &config,
        Some(&fixture.path),
        Some(load_clock())
    )
    .is_err());
    let mut inconsistent = load_clock();
    inconsistent.wall_max_error_ns = Some(1);
    assert!(load_seed(
        &identity(),
        &config,
        Some(&fixture.path),
        Some(inconsistent)
    )
    .is_err());
}

impl Fixture {
    fn new() -> Self {
        let mut config = SloCostObservationConfig::default();
        config.model.min_samples = NonZeroUsize::new(2).unwrap();
        config.model.max_samples_per_bucket = NonZeroUsize::new(2).unwrap();
        config.model.max_sample_age_ns = NonZeroU64::new(100).unwrap();
        config.profile_import.declared_local_clock_max_error_ns = Some(0);
        let profile = file::CostProfileFile {
            schema_version: file::COST_PROFILE_SCHEMA_VERSION,
            fingerprint: file::ProfileFingerprint::from(&fingerprint()),
            settings: file::ProfileModelSettings::from(&model_settings(&config.model)),
            generated_unix_ns: 980,
            source_clock_max_error_ns: Some(0),
            source: file::ProfileSource {
                generator: "Rust observation fixture".into(),
                generator_revision: "profile-import-contract".into(),
                measurement_protocol: "isolated preparation to host commit".into(),
                observation_artifact_sha256: Sha256::digest(b"original observation fixture").into(),
            },
            samples: (0..2)
                .map(|index| file::ProfileSample {
                    source_record: index,
                    measured_unix_ns: 950 + index * 10,
                    shape: file::ProfileWaveShape {
                        kind: file::ProfileWaveKind::Decode,
                        path: file::ProfileExecutionPath::PlanRuntime,
                        provider_signature: [5; 32],
                        output_policy_signature: [6; 32],
                        graph_state: file::ProfileGraphState::Disabled,
                        order: file::ProfileBatchOrder::Ordered,
                        decode_kv_tokens: vec![128],
                        prefill_chunks: vec![],
                        recurrent_state_bytes: 0,
                        restore_bytes: 0,
                        maintenance_bytes: 0,
                        maintenance_units: 0,
                    },
                    boundary: file::ProfileCostBoundary::PreparationToCommit,
                    outcome: file::ProfileObservationOutcome::Completed {},
                    timing: file::ProfileWaveTiming {
                        wall_total_ns: 20 + index,
                        device_elapsed_ns: Some(10),
                        stages: Default::default(),
                    },
                })
                .collect(),
        };
        let path =
            std::env::temp_dir().join(format!("ferrum-cost-import-{}.json", uuid::Uuid::new_v4()));
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap()
            .write_all(&serde_json::to_vec(&profile).unwrap())
            .unwrap();
        Self {
            path,
            config,
            profile,
        }
    }
    fn save(&self) {
        fs::write(&self.path, serde_json::to_vec(&self.profile).unwrap()).unwrap();
    }
    fn build(&self, clock: Arc<Clock>) -> Result<EngineCostRuntime, FerrumError> {
        EngineCostRuntime::build_with_profile(
            identity(),
            clock,
            &self.config,
            false,
            Some(&self.path),
            Some(load_clock()),
        )
    }
    fn shape(&self) -> model::WaveExecutionShape {
        self.profile.samples[0].shape.clone().into()
    }
    fn live(&self, at: u64) -> model::WaveCostObservation {
        model::WaveCostObservation {
            fingerprint: fingerprint(),
            actual_shape: self.shape(),
            boundary: model::CostBoundary::PreparationToCommit,
            outcome: model::WaveObservationOutcome::Completed,
            timing: self.profile.samples[0].timing.clone().into(),
            observed_at_ns: at,
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[test]
fn imported_epoch_overflow_has_its_own_training_rejection_and_preserves_snapshot() {
    let fixture = Fixture::new();
    let runtime = fixture.build(Arc::new(Clock(AtomicU64::new(0)))).unwrap();
    let before = runtime.snapshot().unwrap();
    runtime.sink.offer(fixture.live(u64::MAX)).unwrap();
    runtime.consume_samples();
    let audit = runtime.audit_snapshot();
    let reason = super::super::audit::TrainingErrorReason::ImportedClock;
    assert_eq!(audit.training.outcomes.errors[reason as usize].count, 1);
    assert_eq!(audit.training.outcomes.recorded, 0);
    assert_eq!(audit.training.publish_attempts, 0);
    assert!(Arc::ptr_eq(&before, &runtime.snapshot().unwrap()));
}

fn prediction(
    snapshot: &EngineCostSnapshot,
    shape: &model::WaveExecutionShape,
    now: u64,
) -> model::CostPrediction {
    snapshot.predict(
        &fingerprint(),
        shape,
        model::CostBoundary::PreparationToCommit,
        now,
    )
}

#[test]
fn typed_settings_and_limits_preserve_native_defaults() {
    assert_eq!(
        model_settings(&Default::default()),
        model::CostModelSettings::default()
    );
    let typed = load_limits(&Default::default());
    let native = file::CostProfileLoadLimits::default();
    assert_eq!(typed.max_file_bytes, native.max_file_bytes);
    assert_eq!(typed.max_samples, native.max_samples);
    assert_eq!(typed.max_total_shape_rows, native.max_total_shape_rows);
    assert_eq!(typed.max_source_field_bytes, native.max_source_field_bytes);
    assert_eq!(typed.max_profile_age_ns, native.max_profile_age_ns);
    assert_eq!(typed.max_clock_error_ns, native.max_clock_error_ns);
}

#[test]
fn file_import_retains_verified_provenance_and_age_at_process_start() {
    let fixture = Fixture::new();
    let runtime = fixture.build(Arc::new(Clock(AtomicU64::new(0)))).unwrap();
    let receipt = runtime.profile_receipt().unwrap();
    let bytes = fs::read(&fixture.path).unwrap();
    assert_eq!(receipt.path, fixture.path.canonicalize().unwrap());
    assert_eq!(
        receipt.file_sha256,
        format!("sha256:{:x}", Sha256::digest(&bytes))
    );
    assert_eq!(receipt.file_bytes, bytes.len());
    assert_eq!(
        receipt.source_observation_artifact_sha256,
        fixture.profile.source.observation_artifact_sha256
    );
    assert_eq!((receipt.recorded_samples, receipt.bucket_count), (2, 1));
    assert_eq!(receipt.oldest_imported_age_ns, Some(50));
    assert_eq!(receipt.newest_imported_age_ns, Some(40));
    assert_eq!(
        runtime.trained_samples(),
        0,
        "imported and online sample counts stay distinct"
    );
    let snapshot = runtime.snapshot().unwrap();
    let model::CostPrediction::Known(at_start) = prediction(&snapshot, &fixture.shape(), 0) else {
        panic!("fresh complete import must be usable");
    };
    assert_eq!(at_start.valid_for_ns, 50);
    let model::CostPrediction::Known(at_expiry) = prediction(&snapshot, &fixture.shape(), 50)
    else {
        panic!("inclusive sample TTL");
    };
    assert_eq!(at_expiry.valid_for_ns, 0);
    assert!(matches!(
        prediction(&snapshot, &fixture.shape(), 51),
        model::CostPrediction::Unknown(model::CostUnknownReason::StaleSamples)
    ));
    let mut unsupported = fixture.shape();
    unsupported.decode_kv_tokens[0] += 1;
    assert!(matches!(
        prediction(&snapshot, &unsupported, 0),
        model::CostPrediction::Unknown(_)
    ));
    let mut unsupported = fixture.shape();
    unsupported.provider_signature = [90; 32];
    assert!(matches!(
        prediction(&snapshot, &unsupported, 0),
        model::CostPrediction::Unknown(_)
    ));
}

#[test]
fn delayed_live_updates_use_receipt_clock_and_leave_prior_snapshot_immutable() {
    let fixture = Fixture::new();
    let clock = Arc::new(Clock(AtomicU64::new(0)));
    let runtime = fixture.build(clock.clone()).unwrap();
    let initial = runtime.snapshot().unwrap();
    runtime.sink.offer(fixture.live(10)).unwrap();
    runtime.sink.offer(fixture.live(20)).unwrap();
    clock.0.store(1000, Ordering::Relaxed);
    runtime.consume_samples();
    let updated = runtime.snapshot().unwrap();
    assert_eq!(runtime.trained_samples(), 2);
    assert!(updated.model_version() > initial.model_version());
    assert_eq!(
        runtime.profile_receipt().unwrap().model_version,
        initial.model_version()
    );
    let model::CostPrediction::Known(value) = prediction(&updated, &fixture.shape(), 20) else {
        panic!("both delayed receipts must train without reversal");
    };
    assert_eq!(value.valid_for_ns, 90);
    assert!(matches!(
        prediction(&initial, &fixture.shape(), 51),
        model::CostPrediction::Unknown(model::CostUnknownReason::StaleSamples)
    ));
    assert!(matches!(
        prediction(&updated, &fixture.shape(), 111),
        model::CostPrediction::Unknown(model::CostUnknownReason::StaleSamples)
    ));
    assert!(
        matches!(
            prediction(&updated, &fixture.shape(), clock.now_ns().unwrap()),
            model::CostPrediction::Unknown(model::CostUnknownReason::StaleSamples)
        ),
        "consumption must not refresh queued evidence"
    );
    assert!(
        PlanningCostModel::predict(updated.as_ref(), &fingerprint(), &fixture.shape(), 20)
            .is_some()
    );
    assert!(
        PlanningCostModel::predict(updated.as_ref(), &fingerprint(), &fixture.shape(), 111)
            .is_none()
    );
}

#[test]
fn explicit_profile_rejects_unknown_or_mismatched_execution_and_settings() {
    let fixture = Fixture::new();
    let unknown = ExecutorCostIdentityAvailability::default();
    assert!(load_seed(
        &unknown,
        &fixture.config,
        Some(&fixture.path),
        Some(load_clock())
    )
    .is_err());
    assert!(load_seed(&unknown, &fixture.config, None, None)
        .unwrap()
        .trainer
        .is_none());
    for field in 0..5 {
        let ExecutorCostIdentityAvailability::Known(mut actual) = identity() else {
            unreachable!()
        };
        let actual = Arc::make_mut(&mut actual);
        match field {
            0 => actual.model_weights[0] ^= 1,
            1 => actual.numerical_policy[0] ^= 1,
            2 => actual.device_runtime[0] ^= 1,
            3 => actual.execution_config[0] ^= 1,
            _ => actual.schema_version += 1,
        }
        let actual = ExecutorCostIdentityAvailability::Known(Arc::new(actual.clone()));
        assert!(
            load_seed(
                &actual,
                &fixture.config,
                Some(&fixture.path),
                Some(load_clock())
            )
            .is_err(),
            "fingerprint/schema field {field}"
        );
    }
    let mut config = fixture.config.clone();
    config.model.drift_margin_ns += 1;
    let error = load_seed(
        &identity(),
        &config,
        Some(&fixture.path),
        Some(load_clock()),
    )
    .err()
    .expect("settings must match");
    assert!(error.to_string().contains("settings differ"), "{error}");
}

#[test]
fn clock_declaration_and_actual_wall_age_are_required_independently() {
    let mut fixture = Fixture::new();
    fixture
        .config
        .profile_import
        .declared_local_clock_max_error_ns = None;
    assert!(fixture.build(Arc::new(Clock(AtomicU64::new(0)))).is_err());
    fixture
        .config
        .profile_import
        .declared_local_clock_max_error_ns = Some(0);
    for wall in [None, Some(0), Some(970)] {
        let clock = file::ProfileLoadClock {
            wall_unix_ns: wall,
            ..load_clock()
        };
        assert!(load_seed(
            &identity(),
            &fixture.config,
            Some(&fixture.path),
            Some(clock)
        )
        .is_err());
    }
    fixture.profile.source_clock_max_error_ns = None;
    fixture.save();
    assert!(fixture.build(Arc::new(Clock(AtomicU64::new(0)))).is_err());
    fixture.profile.source_clock_max_error_ns = Some(7);
    fixture
        .config
        .profile_import
        .declared_local_clock_max_error_ns = Some(5);
    fixture.save();
    let clock = file::ProfileLoadClock {
        wall_max_error_ns: Some(5),
        ..load_clock()
    };
    let loaded = load_seed(
        &identity(),
        &fixture.config,
        Some(&fixture.path),
        Some(clock),
    )
    .unwrap();
    assert_eq!(
        loaded.receipt.as_ref().unwrap().conservative_clock_error_ns,
        12
    );
    assert_eq!(
        loaded.receipt.as_ref().unwrap().oldest_imported_age_ns,
        Some(62)
    );
}

#[test]
fn profile_limits_and_missing_files_do_not_fall_back_to_empty_live_training() {
    let fixture = Fixture::new();
    let bytes = fs::metadata(&fixture.path).unwrap().len() as usize;
    let mut config = fixture.config.clone();
    config.profile_import.max_file_bytes = NonZeroUsize::new(bytes - 1).unwrap();
    assert!(load_seed(
        &identity(),
        &config,
        Some(&fixture.path),
        Some(load_clock())
    )
    .is_err());
    config = fixture.config.clone();
    config.profile_import.max_samples = NonZeroUsize::new(1).unwrap();
    assert!(load_seed(
        &identity(),
        &config,
        Some(&fixture.path),
        Some(load_clock())
    )
    .is_err());
    config = fixture.config.clone();
    config.profile_import.max_profile_age_ns = NonZeroU64::new(19).unwrap();
    assert!(load_seed(
        &identity(),
        &config,
        Some(&fixture.path),
        Some(load_clock())
    )
    .is_err());
    fs::remove_file(&fixture.path).unwrap();
    assert!(load_seed(
        &identity(),
        &fixture.config,
        Some(&fixture.path),
        Some(load_clock())
    )
    .is_err());
}

#[test]
fn stale_and_failed_profile_samples_never_become_zero_cost_coverage() {
    let mut fixture = Fixture::new();
    fixture.profile.samples[0].measured_unix_ns = 800;
    fixture.profile.samples[1].outcome = file::ProfileObservationOutcome::FailedAfterSubmit {};
    fixture.profile.samples[1].timing.wall_total_ns = 0;
    let mut deferred = fixture.profile.samples[1].clone();
    deferred.source_record = 2;
    deferred.outcome = file::ProfileObservationOutcome::Deferred {};
    fixture.profile.samples.push(deferred);
    fixture.save();
    let runtime = fixture.build(Arc::new(Clock(AtomicU64::new(0)))).unwrap();
    let receipt = runtime.profile_receipt().unwrap();
    assert_eq!(
        (
            receipt.offered_samples,
            receipt.recorded_samples,
            receipt.stale_samples
        ),
        (3, 0, 1)
    );
    assert_eq!(receipt.skipped_samples.get("failed_after_submit"), Some(&1));
    assert_eq!(receipt.skipped_samples.get("deferred"), Some(&1));
    assert!(matches!(
        prediction(&runtime.snapshot().unwrap(), &fixture.shape(), 0),
        model::CostPrediction::Unknown(_)
    ));
}

#[tokio::test]
async fn production_constructor_loads_file_before_online_background_updates() {
    let mut fixture = Fixture::new();
    fixture.config.model.max_sample_age_ns = NonZeroU64::new(60_000_000_000).unwrap();
    fixture.profile.settings =
        file::ProfileModelSettings::from(&model_settings(&fixture.config.model));
    let now = u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    )
    .unwrap();
    fixture.profile.generated_unix_ns = now;
    for (index, sample) in fixture.profile.samples.iter_mut().enumerate() {
        sample.measured_unix_ns = now - (index as u64 + 1) * 1_000_000;
    }
    fixture.save();
    let runtime = EngineCostRuntime::new(identity(), &fixture.config, Some(&fixture.path)).unwrap();
    let initial = runtime.snapshot().unwrap();
    assert!(matches!(
        prediction(&initial, &fixture.shape(), runtime.clock.now_ns().unwrap()),
        model::CostPrediction::Known(_)
    ));
    runtime.with_training_paused(|| {
        for _ in 0..2 {
            runtime
                .sink
                .offer(fixture.live(runtime.clock.now_ns().unwrap()))
                .unwrap();
        }
    });
    runtime.shutdown().await.unwrap();
    assert_eq!(runtime.trained_samples(), 2);
    assert!(runtime.snapshot().unwrap().model_version() > initial.model_version());
    assert_eq!(runtime.profile_receipt().unwrap().recorded_samples, 2);
}

#[test]
fn structured_v2_observe_has_no_legacy_trainer_and_import_requires_its_protocol() {
    let mut config = SloCostObservationConfig::structured_whole_wave_v2();
    let observe = load_seed(&identity(), &config, None, None).unwrap();
    assert!(observe.trainer.is_none() && observe.snapshot.is_none() && observe.receipt.is_none());
    let legacy = Fixture::new();
    assert!(load_seed(&identity(), &config, Some(&legacy.path), Some(load_clock())).is_err());
    config.profile_import.declared_local_clock_max_error_ns = Some(0);
    assert!(load_seed(&identity(), &config, Some(&legacy.path), Some(load_clock())).is_err());
    let mut wrong = load_clock();
    wrong.wall_max_error_ns = Some(1);
    assert!(load_seed(&identity(), &config, Some(&legacy.path), Some(wrong)).is_err());
}
