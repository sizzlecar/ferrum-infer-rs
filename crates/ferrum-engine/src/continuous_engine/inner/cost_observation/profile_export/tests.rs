use super::*;
use ferrum_types::SloCostObservationConfig;
use std::{fs, num::NonZeroUsize};

mod cut;
mod file_capacity;
mod host_content;
mod host_stages;
mod row_multiset;

struct Fixture {
    dir: PathBuf,
    options: SloCostProfileExportConfig,
    settings: model::CostModelSettings,
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

/// Exercise the real worker and its shutdown/Drop ownership independently of
/// platform wall-clock quantization. The export tests below explicitly inject
/// advancing and contradictory wall/monotonic endpoints; a zero-error dual
/// system-clock calibration is not a premise of these lifecycle tests.
fn runtime_with_fixed_clock(
    config: &SloCostObservationConfig,
    background: bool,
) -> EngineCostRuntime {
    struct FixedClock;
    impl CostObservationClock for FixedClock {
        fn now_ns(&self) -> Option<u64> {
            Some(100)
        }
    }
    EngineCostRuntime::build(identity(), Arc::new(FixedClock), config, background).unwrap()
}

fn observation(at: u64) -> model::WaveCostObservation {
    model::WaveCostObservation {
        fingerprint: fingerprint(),
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
            decode_kv_tokens: vec![128],
            prefill_chunks: vec![],
            recurrent_state_bytes: 0,
            restore_bytes: 0,
            maintenance_bytes: 0,
            maintenance_units: 0,
        },
        boundary: model::CostBoundary::PreparationToCommit,
        outcome: model::WaveObservationOutcome::Completed,
        timing: model::WaveTiming {
            wall_total_ns: 20,
            device_elapsed_ns: Some(10),
            stages: Default::default(),
        },
        observed_at_ns: at,
    }
}

fn training_stats() -> TrainingAuditSnapshot {
    let mut audit = TrainingAuditSnapshot::default();
    audit.outcomes.unavailable = 7;
    audit.publish_errors[0].count = 9;
    audit
}

fn stats() -> CostSampleStats {
    CostSampleStats {
        published: 4,
        drained: 4,
        dropped_capacity: 2,
        dropped_contention: 3,
        rejected: [0; CostCallRejection::COUNT],
        ..Default::default()
    }
}

impl Fixture {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!("ferrum-cost-export-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&dir).unwrap();
        fs::write(dir.join("producer"), b"actual bounded producer bytes").unwrap();
        Self {
            options: SloCostProfileExportConfig {
                path: dir.join("profile.json"),
                observations_path: dir.join("observations.jsonl"),
                declared_clock_max_error_ns: Some(0),
                ..Default::default()
            },
            settings: model::CostModelSettings {
                min_samples: NonZeroUsize::new(1).unwrap(),
                max_sample_age_ns: NonZeroU64::new(1000).unwrap(),
                ..Default::default()
            },
            dir,
        }
    }
    fn exporter(&self) -> Result<ProfileExporter, ExportError> {
        let plan = ExportPlan::new(
            &self.options,
            &fingerprint(),
            &self.settings,
            ExportClockReading {
                wall_unix_ns: 1000,
                monotonic_ns: 100,
            },
        )?;
        ProfileExporter::with_producer(plan, ProducerIdentity::read(&self.dir.join("producer"))?)
    }
    fn finish(&self, exporter: ProfileExporter) -> Result<ExportReceipt, ExportError> {
        exporter.finish(
            ExportClockReading {
                wall_unix_ns: 1300,
                monotonic_ns: 400,
            },
            stats(),
            training_stats(),
        )
    }
    fn records(&self) -> Vec<serde_json::Value> {
        fs::read_to_string(&self.options.observations_path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

#[test]
fn original_receipt_and_actual_source_digest_roundtrip_without_refreshing_age() {
    let f = Fixture::new();
    let mut export = f.exporter().unwrap();
    export
        .record(
            &observation(120),
            TrainingDisposition::Recorded,
            PreUpdatePrediction::NoPublishedModel,
        )
        .unwrap();
    let receipt = f.finish(export).unwrap();
    let raw = fs::read(&f.options.observations_path).unwrap();
    assert_eq!(
        receipt.source.digest,
        <[u8; 32]>::from(Sha256::digest(&raw))
    );
    assert_eq!(receipt.source.bytes, raw.len() as u64);
    let file: profile_v2::CostProfileFileV2 =
        serde_json::from_slice(&fs::read(&f.options.path).unwrap()).unwrap();
    assert_eq!(
        file.source.observation_artifact_sha256,
        receipt.source.digest
    );
    assert_eq!(file.generated_unix_ns, 1300);
    assert_eq!(
        file.samples[0].measured_unix_ns, 1020,
        "receipt age must survive delayed writing"
    );
    assert_eq!(file.samples[0].timing.wall_total_ns, 20);
    let records = f.records();
    assert_eq!(records[0]["schema_version"], 4);
    assert_eq!(records[1]["training"]["status"], "recorded");
    assert_eq!(
        records[0]["producer"]["executable_sha256"],
        format!("{:x}", Sha256::digest(b"actual bounded producer bytes"))
    );
    assert!(records[0]["producer"]["source_revision"].is_null());
    assert_eq!(records[1]["observed_at_monotonic_ns"], 120);
    let loaded = profile::load_cost_profile(
        &f.options.path,
        &fingerprint(),
        &f.settings,
        &Default::default(),
        profile::ProfileLoadClock {
            wall_unix_ns: Some(1400),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: 0,
        },
    )
    .unwrap();
    assert_eq!(loaded.provenance.oldest_imported_age_ns, Some(380));
    let shape = observation(0).actual_shape;
    let model::CostPrediction::Known(prediction) = loaded.snapshot.predict(
        &fingerprint(),
        &shape,
        model::CostBoundary::PreparationToCommit,
        0,
    ) else {
        panic!("valid exported sample must calibrate")
    };
    assert_eq!(prediction.valid_for_ns, 620);
}

#[tokio::test]
async fn cost_checkpoint_keeps_live_export_open_and_preserves_both_sides_of_cut() {
    let fixture = Fixture::new();
    let mut config = SloCostObservationConfig::default();
    config.model.min_samples = NonZeroUsize::MIN;
    config.profile_export = Some(fixture.options.clone());
    let runtime = runtime_with_fixed_clock(&config, false);
    runtime.sink.offer(observation(100)).unwrap();
    let waiter = runtime.request_checkpoint().unwrap();
    runtime.sink.offer(observation(100)).unwrap();
    runtime.consume_samples();
    let checkpoint = waiter.wait().await.unwrap();
    assert_eq!(checkpoint.accepted_ordinal, 1);
    assert_eq!(checkpoint.export.status, ExportStatus::Active);
    assert_eq!(checkpoint.export.counts.received_observations, 1);
    assert!(!fixture.options.path.exists());
    assert!(!fixture.options.observations_path.exists());
    runtime.shutdown().await.unwrap();
    let records = fixture.records();
    assert_eq!(records.len(), 4); // header, both real samples, final summary
    assert_eq!(records[1]["observed_at_monotonic_ns"], 100);
    assert_eq!(records[2]["observed_at_monotonic_ns"], 100);
    assert_eq!(
        runtime.audit_snapshot().export.counts.received_observations,
        2
    );
}

fn numeric_observation(at: u64) -> model::WaveCostObservation {
    let mut observation = observation(at);
    let host = HostCostFeaturesV1 {
        policy: HostCostPolicyV2 {
            empirical_content_domain: None,
            categorical_signature: [7; 32],
            decoder_text_bytes_per_token: 8,
            decoder_scratch_bytes_per_token: 3,
            raw_token_bytes_bound: 3,
        },
        state: HostCostStateV1 {
            generated_tokens_before: 3,
            maximum_output_tokens: 20,
            sampling_history_tokens: 3,
            sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
            pending_decoded_utf8: false,
            completion_state_signature: satisfied_completion_cost_signature(),
        },
    };
    let row = project_host_cost_features(
        host,
        ActualRowWork::Decode { kv_tokens: 128 },
        CostRowOutput::Decode {
            requires_full_logits: false,
            repetition_tokens: 0,
            repetition_penalty_bits: 1f32.to_bits(),
        },
    )
    .unwrap();
    observation.actual_shape.numeric_features = Some(CanonicalWaveCostFeatures {
        schema_version: COST_NUMERIC_FEATURE_SCHEMA_V1,
        output_policy_signature: [8; 32],
        rows: vec![row],
    });
    observation
}

#[test]
fn raw_v3_and_profile_v2_preserve_numeric_evidence_and_original_age() {
    let mut f = Fixture::new();
    f.settings.feature_model = model::CostFeatureModel::BoundedNumericV1 {
        host_history_bucket_tokens: std::num::NonZeroU32::new(16).unwrap(),
    };
    let original = numeric_observation(120);
    let mut export = f.exporter().unwrap();
    export
        .record(
            &original,
            TrainingDisposition::Recorded,
            PreUpdatePrediction::NoPublishedModel,
        )
        .unwrap();
    f.finish(export).unwrap();
    let bytes = fs::read(&f.options.path).unwrap();
    assert!(
        serde_json::from_slice::<profile::CostProfileFile>(&bytes).is_err(),
        "new fields cannot masquerade as strict v1"
    );
    let file: profile_v2::CostProfileFileV2 = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(file.schema_version, 2);
    assert_eq!(
        file.samples[0].shape.numeric_features,
        original.actual_shape.numeric_features
    );
    assert_eq!(
        file.samples[0].shape.exact.output_policy_signature,
        original.actual_shape.output_policy_signature
    );
    let raw = f.records();
    assert_eq!(raw[0]["schema_version"], 4);
    assert_eq!(
        raw[1]["sample"]["shape"]["numeric_features"]["rows"][0]["decoded_prefix_tokens"],
        4
    );
    let loaded = profile::load_cost_profile(
        &f.options.path,
        &fingerprint(),
        &f.settings,
        &Default::default(),
        profile::ProfileLoadClock {
            wall_unix_ns: Some(1400),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: 0,
        },
    )
    .unwrap();
    assert_eq!(loaded.provenance.oldest_imported_age_ns, Some(380));
    let model::CostPrediction::Known(prediction) = loaded.snapshot.predict(
        &fingerprint(),
        &original.actual_shape,
        model::CostBoundary::PreparationToCommit,
        0,
    ) else {
        panic!("valid numeric profile must remain queryable");
    };
    assert_eq!(prediction.valid_for_ns, 620);
}

#[test]
fn export_row_limit_covers_numeric_rows_without_hiding_rejected_population() {
    let mut f = Fixture::new();
    f.options.max_total_shape_rows = NonZeroUsize::new(1).unwrap();
    let mut export = f.exporter().unwrap();
    export
        .record(
            &numeric_observation(120),
            TrainingDisposition::Recorded,
            PreUpdatePrediction::NoPublishedModel,
        )
        .unwrap();
    assert_eq!(export.counts.received_observations, 1);
    assert_eq!(export.counts.dropped_shape_row_limit, 1);
    assert_eq!(export.counts.retained_samples, 0);
    assert_eq!(export.counts.raw_retained_observations, 0);
    f.options.max_total_shape_rows = NonZeroUsize::new(2).unwrap();
    let mut export = f.exporter().unwrap();
    export
        .record(
            &numeric_observation(120),
            TrainingDisposition::Recorded,
            PreUpdatePrediction::NoPublishedModel,
        )
        .unwrap();
    assert_eq!(export.rows, 2);
    assert_eq!(export.counts.retained_samples, 1);
}

#[test]
fn losses_outcomes_and_capacity_are_explicit_and_do_not_create_zero_cost_samples() {
    let mut f = Fixture::new();
    f.options.max_samples = NonZeroUsize::new(1).unwrap();
    let mut export = f.exporter().unwrap();
    for outcome in [
        model::WaveObservationOutcome::FailedAfterSubmit,
        model::WaveObservationOutcome::Deferred,
    ] {
        let mut sample = observation(101);
        sample.outcome = outcome;
        export
            .record(
                &sample,
                TrainingDisposition::Skipped {
                    reason: match outcome {
                        model::WaveObservationOutcome::FailedAfterSubmit => {
                            audit::TrainingSkipReason::FailedAfterSubmit
                        }
                        _ => audit::TrainingSkipReason::Deferred,
                    },
                },
                PreUpdatePrediction::NotCompleted,
            )
            .unwrap();
    }
    export
        .record(
            &observation(110),
            TrainingDisposition::Recorded,
            PreUpdatePrediction::NoPublishedModel,
        )
        .unwrap();
    export
        .record(
            &observation(120),
            TrainingDisposition::Recorded,
            PreUpdatePrediction::NoPublishedModel,
        )
        .unwrap();
    let receipt = f.finish(export).unwrap();
    assert_eq!(receipt.counts.retained_samples, 1);
    assert_eq!(receipt.counts.trainer_not_recorded, 2);
    assert_eq!(receipt.counts.non_completed, 2);
    assert_eq!(receipt.counts.dropped_sample_limit, 1);
    let records = f.records();
    let summary = records.last().unwrap();
    assert_eq!(summary["sink"]["dropped_capacity"], 2);
    assert_eq!(summary["sink"]["dropped_contention"], 3);
    assert_eq!(summary["training_rejected"], 7);
    assert_eq!(summary["publish_rejected"], 9);
    assert_eq!(records[1]["sample"]["source_record"], 2);
}

#[test]
fn shape_and_file_limits_drop_new_samples_but_preserve_valid_prefix() {
    let mut f = Fixture::new();
    f.options.max_total_shape_rows = NonZeroUsize::new(1).unwrap();
    let mut export = f.exporter().unwrap();
    export
        .record(
            &observation(110),
            TrainingDisposition::Recorded,
            PreUpdatePrediction::NoPublishedModel,
        )
        .unwrap();
    export
        .record(
            &observation(120),
            TrainingDisposition::Recorded,
            PreUpdatePrediction::NoPublishedModel,
        )
        .unwrap();
    assert_eq!(export.counts.dropped_shape_row_limit, 1);
    f.finish(export).unwrap();

    let mut f = Fixture::new();
    let mut sizing = f.exporter().unwrap();
    sizing
        .record(
            &observation(110),
            TrainingDisposition::Recorded,
            PreUpdatePrediction::NoPublishedModel,
        )
        .unwrap();
    let overhead = f.options.max_file_bytes.get() as u64 - sizing.max_body_bytes;
    f.options.max_file_bytes = NonZeroUsize::new((overhead + sizing.body_bytes) as usize).unwrap();
    drop(sizing);
    let mut export = f.exporter().unwrap();
    export
        .record(
            &observation(110),
            TrainingDisposition::Recorded,
            PreUpdatePrediction::NoPublishedModel,
        )
        .unwrap();
    export
        .record(
            &observation(120),
            TrainingDisposition::Recorded,
            PreUpdatePrediction::NoPublishedModel,
        )
        .unwrap();
    assert_eq!(export.counts.dropped_file_byte_limit, 1);
    let receipt = f.finish(export).unwrap();
    assert_eq!(receipt.counts.retained_samples, 1);
    assert!(receipt.profile.bytes <= f.options.max_file_bytes.get() as u64);
    assert!(receipt.source.bytes <= f.options.max_file_bytes.get() as u64);
}

#[test]
fn inconsistent_clocks_and_identity_cannot_publish_a_profile() {
    let f = Fixture::new();
    let mut exporter = f.exporter().unwrap();
    assert!(exporter
        .record(
            &observation(99),
            TrainingDisposition::Recorded,
            PreUpdatePrediction::NoPublishedModel
        )
        .is_err());
    let mut sample = observation(120);
    sample.fingerprint.execution_config[0] ^= 1;
    assert!(exporter
        .record(
            &sample,
            TrainingDisposition::Recorded,
            PreUpdatePrediction::NoPublishedModel
        )
        .is_err());
    exporter
        .record(
            &observation(120),
            TrainingDisposition::Recorded,
            PreUpdatePrediction::NoPublishedModel,
        )
        .unwrap();
    assert!(exporter
        .finish(
            ExportClockReading {
                wall_unix_ns: 1100,
                monotonic_ns: 400
            },
            stats(),
            TrainingAuditSnapshot::default()
        )
        .is_err());
    assert!(!f.options.path.exists());
    assert!(!f.options.observations_path.exists());
}

#[test]
fn declared_endpoint_error_is_inclusive_but_one_more_nanosecond_is_rejected() {
    for (closing_wall, accepted) in [(1298, true), (1297, false)] {
        let mut f = Fixture::new();
        f.options.declared_clock_max_error_ns = Some(1);
        let mut exporter = f.exporter().unwrap();
        exporter
            .record(
                &observation(120),
                TrainingDisposition::Recorded,
                PreUpdatePrediction::NoPublishedModel,
            )
            .unwrap();
        let result = exporter.finish(
            ExportClockReading {
                wall_unix_ns: closing_wall,
                monotonic_ns: 400,
            },
            stats(),
            TrainingAuditSnapshot::default(),
        );
        assert_eq!(result.is_ok(), accepted, "{result:?}");
        assert_eq!(f.options.path.exists(), accepted);
        if !accepted {
            assert!(matches!(result, Err(ExportError::Clock(_))));
            assert!(!f.options.observations_path.exists());
        }
    }
}

#[test]
fn empty_capture_preserves_raw_diagnostics_and_never_creates_an_empty_profile() {
    let f = Fixture::new();
    assert!(matches!(
        f.finish(f.exporter().unwrap()),
        Err(ExportError::SourceOnly { .. })
    ));
    assert!(!f.options.path.exists());
    assert_eq!(f.records().len(), 2);
}

#[test]
fn atomic_publish_cannot_replace_a_racing_destination_and_cleans_only_own_stage() {
    let f = Fixture::new();
    let target = f.dir.join("race");
    let mut stage = StagedFile::create(&target, 10).unwrap();
    stage.write_all(b"new").unwrap();
    fs::write(&target, b"existing").unwrap();
    assert!(stage.publish().is_err());
    assert_eq!(fs::read(&target).unwrap(), b"existing");
    assert!(!fs::read_dir(&f.dir).unwrap().any(|entry| entry
        .unwrap()
        .file_name()
        .to_string_lossy()
        .ends_with(".tmp")));
    let mut stage = StagedFile::create(&f.dir.join("limited"), 2).unwrap();
    assert!(stage.write_all(b"long").is_err());
    drop(stage);
    assert!(!f.dir.join("limited").exists());
}

#[test]
fn profile_publish_failure_keeps_verified_raw_source_and_existing_file() {
    let f = Fixture::new();
    let mut exporter = f.exporter().unwrap();
    exporter
        .record(
            &observation(120),
            TrainingDisposition::Recorded,
            PreUpdatePrediction::NoPublishedModel,
        )
        .unwrap();
    fs::write(&f.options.path, b"existing").unwrap();
    assert!(matches!(
        f.finish(exporter),
        Err(ExportError::SourceOnly { .. })
    ));
    assert_eq!(fs::read(&f.options.path).unwrap(), b"existing");
    assert_eq!(f.records().len(), 3);
}

#[test]
fn alias_and_provenance_byte_limits_are_rejected_without_output() {
    let mut f = Fixture::new();
    f.options.observations_path = f.dir.join(".").join("profile.json");
    assert!(f.exporter().is_err());
    f.options.observations_path = f.dir.join("observations.jsonl");
    f.options.max_file_bytes = NonZeroUsize::new(1).unwrap();
    assert!(f.exporter().is_err());
    assert!(!f.options.path.exists());
}

#[tokio::test]
async fn real_worker_shutdown_exports_once_and_propagates_write_failure() {
    let f = Fixture::new();
    let mut config = SloCostObservationConfig::default();
    config.profile_export = Some(f.options.clone());
    config.model.min_samples = NonZeroUsize::new(1).unwrap();
    let runtime = runtime_with_fixed_clock(&config, true);
    runtime.with_training_paused(|| {
        runtime
            .sink
            .offer(observation(runtime.clock.now_ns().unwrap()))
            .unwrap();
    });
    runtime.shutdown().await.unwrap();
    let bytes = fs::read(&f.options.path).unwrap();
    runtime.shutdown().await.unwrap();
    assert_eq!(fs::read(&f.options.path).unwrap(), bytes);
    assert_eq!(
        f.records()[0]["producer"]["executable_sha256"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    let file: profile_v2::CostProfileFileV2 = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(file.samples.len(), 1);
    assert_eq!(runtime.trained_samples(), 1);

    // Existing destinations are a product error, even though model training
    // can continue and must not overwrite the previous capture.
    let runtime = runtime_with_fixed_clock(&config, true);
    runtime.shutdown().await.unwrap_err();
    runtime.shutdown().await.unwrap_err();
    assert_eq!(fs::read(&f.options.path).unwrap(), bytes);
}

#[tokio::test]
async fn imported_trainer_exports_only_new_local_receipts_not_reanchored_model_time() {
    use std::sync::atomic::{AtomicU64, Ordering};
    struct Clock(AtomicU64);
    impl CostObservationClock for Clock {
        fn now_ns(&self) -> Option<u64> {
            Some(self.0.load(Ordering::Relaxed))
        }
    }
    let old = Fixture::new();
    let mut exporter = old.exporter().unwrap();
    exporter
        .record(
            &observation(120),
            TrainingDisposition::Recorded,
            PreUpdatePrediction::NoPublishedModel,
        )
        .unwrap();
    old.finish(exporter).unwrap();
    let fresh = Fixture::new();
    let mut config = SloCostObservationConfig::default();
    config.profile_export = Some(fresh.options.clone());
    config.profile_import.declared_local_clock_max_error_ns = Some(0);
    config.model.min_samples = NonZeroUsize::new(1).unwrap();
    config.model.max_sample_age_ns = NonZeroU64::new(1000).unwrap();
    let clock = Arc::new(Clock(AtomicU64::new(0)));
    let runtime = EngineCostRuntime::build_with_profile(
        identity(),
        clock.clone(),
        &config,
        false,
        Some(&old.options.path),
        Some(profile::ProfileLoadClock {
            wall_unix_ns: Some(1400),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: 0,
        }),
    )
    .unwrap();
    runtime.sink.offer(observation(5)).unwrap();
    runtime.consume_samples();
    clock.0.store(1000, Ordering::Relaxed);
    runtime.shutdown().await.unwrap();
    let records = fresh.records();
    assert_eq!(records[1]["observed_at_monotonic_ns"], 5);
    assert_eq!(
        records[1]["sample"]["measured_unix_ns"].as_u64().unwrap(),
        records[0]["opening"]["wall_unix_ns"].as_u64().unwrap() + 5
    );
    assert_eq!(
        records.len(),
        3,
        "old imported samples are not new observations"
    );
}

#[test]
fn dropping_engine_joins_export_worker_and_keeps_failure_non_panicking() {
    let f = Fixture::new();
    let mut config = SloCostObservationConfig::default();
    config.profile_export = Some(f.options.clone());
    config.model.min_samples = NonZeroUsize::new(1).unwrap();
    let runtime = runtime_with_fixed_clock(&config, true);
    runtime.with_training_paused(|| {
        runtime
            .sink
            .offer(observation(runtime.clock.now_ns().unwrap()))
            .unwrap();
    });
    drop(runtime);
    assert!(f.options.path.exists());
    assert_eq!(f.records().len(), 3);
    // Reusing occupied paths is reported by the worker, not by panicking Drop.
    drop(runtime_with_fixed_clock(&config, true));
}

#[tokio::test]
async fn real_bucket_rejection_is_retained_raw_while_profile_and_export_limits_remain_separate() {
    let mut f = Fixture::new();
    f.options.max_samples = NonZeroUsize::new(2).unwrap();
    let mut config = SloCostObservationConfig::default();
    config.profile_export = Some(f.options.clone());
    config.model.min_samples = NonZeroUsize::MIN;
    config.model.max_buckets = NonZeroUsize::MIN;
    let runtime = runtime_with_fixed_clock(&config, true);
    runtime.with_training_paused(|| {
        runtime
            .sink
            .offer(observation(runtime.clock.now_ns().unwrap()))
            .unwrap();
        let mut unsupported_bucket = observation(runtime.clock.now_ns().unwrap());
        unsupported_bucket.actual_shape.provider_signature[0] ^= 1;
        runtime.sink.offer(unsupported_bucket).unwrap();
        runtime
            .sink
            .offer(observation(runtime.clock.now_ns().unwrap()))
            .unwrap();
    });
    runtime.shutdown().await.unwrap();
    let records = f.records();
    assert_eq!(records.len(), 4);
    assert_eq!(records[0]["schema_version"], 4);
    assert_eq!(records[1]["training"]["status"], "recorded");
    assert_eq!(records[2]["training"]["status"], "error");
    assert_eq!(records[2]["training"]["reason"], "capacity_exceeded");
    assert_eq!(
        records[2]["sample"]["shape"]["exact"]["provider_signature"][0],
        4
    );
    let summary = records.last().unwrap();
    assert_eq!(summary["sink"]["offered"], 3);
    assert_eq!(summary["sink"]["offered_completed"], 3);
    assert_eq!(summary["training"]["consumed"], 3);
    assert_eq!(summary["training"]["outcomes"]["recorded"], 2);
    assert_eq!(
        summary["training"]["pre_update_prediction"]["counts"]["completed"],
        3
    );
    assert_eq!(
        summary["training"]["pre_update_prediction"]["counts"]["no_published_model"],
        3
    );
    assert_eq!(
        records[2]["pre_update_prediction"]["status"],
        "no_published_model"
    );
    assert_eq!(summary["counts"]["received_observations"], 3);
    assert_eq!(summary["counts"]["raw_retained_observations"], 2);
    assert_eq!(summary["counts"]["retained_samples"], 1);
    assert_eq!(summary["counts"]["trainer_not_recorded"], 1);
    assert_eq!(summary["counts"]["dropped_sample_limit"], 1);
    let file: profile_v2::CostProfileFileV2 =
        serde_json::from_slice(&fs::read(&f.options.path).unwrap()).unwrap();
    assert_eq!(file.samples.len(), 1);
    assert_eq!(file.samples[0].shape.exact.provider_signature, [5; 32]);
    let audit = runtime.audit_snapshot();
    assert_eq!(audit.training.outcomes.recorded, 2);
    assert_eq!(audit.export.status, ExportStatus::Published);
    assert_eq!(audit.export.counts.raw_retained_observations, 2);
    assert_eq!(audit.export.counts.received_by_wave, [3, 0, 0, 0, 0]);
    assert_eq!(
        audit.export.counts.profile_retained_by_wave,
        [1, 0, 0, 0, 0]
    );
}

#[tokio::test]
async fn raw_prediction_audit_retains_rejected_costs_without_changing_profile_population() {
    let mut f = Fixture::new();
    f.options.max_samples = NonZeroUsize::new(4).unwrap();
    let mut config = SloCostObservationConfig::default();
    config.profile_export = Some(f.options.clone());
    config.model.min_samples = NonZeroUsize::new(2).unwrap();
    config.model.max_retained_samples = NonZeroUsize::new(2).unwrap();
    config.model.drift_margin_ns = 0;
    let runtime = runtime_with_fixed_clock(&config, false);
    for _ in 0..2 {
        runtime.sink.offer(observation(100)).unwrap();
    }
    runtime.consume_samples();
    let mut capacity_rejected = observation(100);
    capacity_rejected.timing.wall_total_ns = 70;
    let mut invalid = observation(100);
    invalid.timing.device_elapsed_ns = Some(21); // Beyond its 20 ns wall.
    runtime.sink.offer(capacity_rejected.clone()).unwrap();
    runtime.sink.offer(invalid).unwrap();
    // This sample is also queried but cannot fit the independent raw limit.
    runtime.sink.offer(capacity_rejected).unwrap();
    runtime.consume_samples();
    runtime.shutdown().await.unwrap();
    let records = f.records();
    assert_eq!(records.len(), 6);
    for record in &records[1..3] {
        assert_eq!(
            record["pre_update_prediction"]["status"],
            "no_published_model"
        );
    }
    let rejected = &records[3];
    assert_eq!(rejected["training"]["reason"], "capacity_exceeded");
    assert_eq!(rejected["pre_update_prediction"]["status"], "known");
    assert_eq!(rejected["pre_update_prediction"]["planning_ns"], 20);
    assert_eq!(rejected["pre_update_prediction"]["observed_cost_ns"], 70);
    assert_eq!(rejected["pre_update_prediction"]["underestimate_ns"], 50);
    let invalid = &records[4];
    assert_eq!(invalid["training"]["reason"], "invalid_timing");
    assert_eq!(invalid["pre_update_prediction"]["status"], "known");
    assert!(invalid["pre_update_prediction"]["observed_cost_ns"].is_null());
    assert!(invalid["pre_update_prediction"]["underestimate_ns"].is_null());
    let summary = records.last().unwrap();
    let predictions = &summary["training"]["pre_update_prediction"]["counts"];
    assert_eq!(summary["sink"]["offered_completed"], 5);
    assert_eq!(predictions["completed"], 5);
    assert_eq!(predictions["known"], 3);
    assert_eq!(predictions["compared"], 2);
    assert_eq!(predictions["underestimates"], 2);
    assert_eq!(summary["counts"]["dropped_sample_limit"], 1);
    let profile: profile_v2::CostProfileFileV2 =
        serde_json::from_slice(&fs::read(&f.options.path).unwrap()).unwrap();
    assert_eq!(profile.samples.len(), 2);
    assert!(profile.samples.iter().all(|s| s.timing.wall_total_ns == 20));
}

#[test]
fn bounded_rejected_prefix_is_not_replaced_with_later_recorded_winners() {
    let mut f = Fixture::new();
    f.options.max_samples = NonZeroUsize::MIN;
    let mut settings = f.settings.clone();
    settings.max_buckets = NonZeroUsize::MIN;
    let mut trainer = model::CostModelTrainer::new(fingerprint(), settings).unwrap();
    trainer.observe(observation(110)).unwrap();
    let mut rejected = observation(120);
    rejected.actual_shape.provider_signature[0] ^= 1;
    let disposition = TrainingDisposition::from_result(Some(
        trainer.observe(rejected.clone()).map_err(Into::into),
    ));
    assert_eq!(
        disposition,
        TrainingDisposition::Error {
            reason: audit::TrainingErrorReason::CapacityExceeded
        }
    );
    let mut exporter = f.exporter().unwrap();
    exporter
        .record(
            &rejected,
            disposition,
            PreUpdatePrediction::NoPublishedModel,
        )
        .unwrap();
    let sample = observation(130);
    let recorded =
        TrainingDisposition::from_result(Some(trainer.observe(sample.clone()).map_err(Into::into)));
    exporter
        .record(&sample, recorded, PreUpdatePrediction::NoPublishedModel)
        .unwrap();
    assert!(matches!(
        f.finish(exporter),
        Err(ExportError::SourceOnly { .. })
    ));
    assert!(!f.options.path.exists());
    let records = f.records();
    assert_eq!(records.len(), 3);
    assert_eq!(records[1]["training"]["reason"], "capacity_exceeded");
    assert_eq!(records[2]["counts"]["raw_retained_observations"], 1);
    assert_eq!(records[2]["counts"]["retained_samples"], 0);
    assert_eq!(records[2]["counts"]["dropped_sample_limit"], 1);
}

#[test]
fn raw_skipped_completed_observation_remains_diagnostic_and_does_not_calibrate() {
    let f = Fixture::new();
    let mut trainer = model::CostModelTrainer::new(fingerprint(), f.settings.clone()).unwrap();
    let mut sample = observation(120);
    sample.boundary = model::CostBoundary::DeviceOnly;
    sample.timing.device_elapsed_ns = None;
    let disposition =
        TrainingDisposition::from_result(Some(trainer.observe(sample.clone()).map_err(Into::into)));
    assert_eq!(
        disposition,
        TrainingDisposition::Skipped {
            reason: audit::TrainingSkipReason::MissingDeviceTiming
        }
    );
    let mut exporter = f.exporter().unwrap();
    exporter
        .record(&sample, disposition, PreUpdatePrediction::NoPublishedModel)
        .unwrap();
    assert!(matches!(
        f.finish(exporter),
        Err(ExportError::SourceOnly { .. })
    ));
    assert_eq!(
        f.records()[1]["training"]["reason"],
        "missing_device_timing"
    );
    assert!(!f.options.path.exists());
}

#[tokio::test]
async fn capture_failure_preserves_typed_failure_and_uncaptured_observation_counts() {
    let f = Fixture::new();
    fs::write(&f.options.path, b"owned by someone else").unwrap();
    let mut config = SloCostObservationConfig::default();
    config.profile_export = Some(f.options.clone());
    config.model.min_samples = NonZeroUsize::MIN;
    let runtime = EngineCostRuntime::new(identity(), &config, None).unwrap();
    runtime.with_training_paused(|| {
        for _ in 0..2 {
            runtime
                .sink
                .offer(observation(runtime.clock.now_ns().unwrap()))
                .unwrap();
        }
    });
    runtime.shutdown().await.unwrap_err();
    let audit = runtime.audit_snapshot();
    assert_eq!(audit.training.consumed, 2);
    assert_eq!(audit.training.outcomes.recorded, 2);
    assert_eq!(audit.export.status, ExportStatus::Failed);
    assert!(audit.export.failure.is_some());
    assert_eq!(audit.export.unavailable_observations, 2);
    assert_eq!(audit.export.counts.received_observations, 0);
    assert_eq!(fs::read(&f.options.path).unwrap(), b"owned by someone else");
}
