use super::*;
use crate::continuous_engine::inner::cost_observation::*;
use std::{
    num::{NonZeroU64, NonZeroUsize},
    sync::Arc,
};

struct Clock;
impl CostObservationClock for Clock {
    fn now_ns(&self) -> Option<u64> {
        Some(10_000)
    }
}
fn identity() -> ExecutorCostIdentityAvailability {
    ExecutorCostIdentityAvailability::Known(Arc::new(ExecutorCostIdentity {
        schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }))
}
fn sample(at: u64, wall: u64) -> model::WaveCostObservation {
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
            graph_state: model::WaveGraphState::Disabled,
            provider_signature: [5; 32],
            output_policy_signature: [6; 32],
            numeric_features: None,
            order: model::BatchOrderSemantics::Ordered,
            decode_kv_tokens: vec![12],
            prefill_chunks: Vec::new(),
            recurrent_state_bytes: 0,
            restore_bytes: 0,
            maintenance_bytes: 0,
            maintenance_units: 0,
        },
        boundary: model::CostBoundary::PreparationToCommit,
        outcome: model::WaveObservationOutcome::Completed,
        timing: model::WaveTiming {
            wall_total_ns: wall,
            device_elapsed_ns: None,
            stages: Default::default(),
        },
        observed_at_ns: at,
    }
}
fn config() -> ferrum_types::SloCostObservationConfig {
    let mut config = ferrum_types::SloCostObservationConfig::default();
    config.model.min_samples = NonZeroUsize::new(2).unwrap();
    config.model.drift_margin_ns = 0;
    config.model.max_sample_age_ns = NonZeroU64::new(100).unwrap();
    config
}
fn seed(runtime: &EngineCostRuntime) {
    runtime.sink.offer(sample(10, 20)).unwrap();
    runtime.sink.offer(sample(11, 20)).unwrap();
    runtime.consume_samples();
}

#[test]
fn same_batch_does_not_learn_before_query_and_next_batch_detects_underestimate() {
    let runtime = EngineCostRuntime::build(identity(), Arc::new(Clock), &config(), false).unwrap();
    seed(&runtime);
    let first = runtime.audit_snapshot().training.pre_update_prediction;
    assert_eq!(first.counts.completed, 2);
    assert_eq!(first.counts.no_published_model, 2);
    assert_eq!(first.counts.known, 0);
    runtime.sink.offer(sample(12, 70)).unwrap();
    runtime.sink.offer(sample(13, 80)).unwrap();
    runtime.consume_samples();
    let second = runtime.audit_snapshot().training.pre_update_prediction;
    assert_eq!(second.counts.known, 2);
    assert_eq!(second.counts.compared, 2);
    assert_eq!(second.counts.underestimates, 2);
    assert_eq!(second.counts.max_underestimate_ns, 60);
    assert_eq!(second.by_wave[0].counts, second.counts);
}

#[test]
fn known_capacity_rejects_are_audited_but_invalid_timing_is_not_an_error_measurement() {
    let mut config = config();
    config.model.max_retained_samples = NonZeroUsize::new(2).unwrap();
    let runtime = EngineCostRuntime::build(identity(), Arc::new(Clock), &config, false).unwrap();
    seed(&runtime);
    runtime.sink.offer(sample(12, 70)).unwrap(); // Known shape, full retention.
    let mut invalid = sample(13, 200);
    invalid.timing.device_elapsed_ns = Some(201); // Positive, but impossible.
    runtime.sink.offer(invalid).unwrap();
    runtime.consume_samples();
    let audit = runtime.audit_snapshot();
    assert_eq!(audit.sink.offered_completed, 4);
    assert_eq!(audit.training.outcomes.recorded, 2);
    assert_eq!(audit.training.pre_update_prediction.counts.completed, 4);
    assert_eq!(audit.training.pre_update_prediction.counts.known, 2);
    assert_eq!(audit.training.pre_update_prediction.counts.compared, 1);
    assert_eq!(
        audit.training.pre_update_prediction.counts.underestimates,
        1
    );
    assert_eq!(
        audit
            .training
            .pre_update_prediction
            .counts
            .max_underestimate_ns,
        50
    );
}

#[test]
fn stale_receipt_time_and_training_reject_reasons_do_not_refresh_old_model() {
    let runtime = EngineCostRuntime::build(identity(), Arc::new(Clock), &config(), false).unwrap();
    seed(&runtime);
    runtime.sink.offer(sample(111, 20)).unwrap();
    runtime.consume_samples();
    let counts = runtime
        .audit_snapshot()
        .training
        .pre_update_prediction
        .counts;
    assert_eq!(
        counts.unknown[PredictionUnknownReason::StaleSamples as usize].count,
        1
    );
    assert_eq!(
        counts.known, 0,
        "worker wall time must not become the sample query time"
    );
}

#[test]
fn known_model_preserves_distinct_unknown_reasons_for_rejected_queries() {
    let runtime = EngineCostRuntime::build(identity(), Arc::new(Clock), &config(), false).unwrap();
    seed(&runtime);
    let mut new_bucket = sample(12, 20);
    new_bucket.actual_shape.provider_signature[0] ^= 1;
    let mut foreign = sample(13, 20);
    foreign.fingerprint.model_weights[0] ^= 1;
    let backwards = sample(10, 20); // Before the old snapshot's publication.
    let mut malformed = sample(14, 20);
    malformed.actual_shape.decode_kv_tokens.clear();
    for observation in [new_bucket, foreign, backwards, malformed] {
        runtime.sink.offer(observation).unwrap();
    }
    runtime.consume_samples();
    let audit = runtime.audit_snapshot();
    let counts = &audit.training.pre_update_prediction.counts;
    for reason in [
        PredictionUnknownReason::UnobservedBucket,
        PredictionUnknownReason::FingerprintMismatch,
        PredictionUnknownReason::ClockMovedBackwards,
        PredictionUnknownReason::InvalidShape,
    ] {
        assert_eq!(
            counts.unknown[reason as usize].count, 1,
            "{reason:?}: {audit:?}"
        );
    }
    assert_eq!(counts.completed, 6);
    assert_eq!(counts.no_published_model, 2);
    assert_eq!(counts.known, 0);
    assert_eq!(audit.training.outcomes.recorded, 3);
}

#[test]
fn lost_queue_samples_remain_offered_without_fabricated_prediction_results() {
    let mut config = config();
    config.max_queued_samples = NonZeroUsize::MIN;
    config.max_samples_per_update = NonZeroUsize::MIN;
    let runtime = EngineCostRuntime::build(identity(), Arc::new(Clock), &config, false).unwrap();
    runtime.sink.offer(sample(10, 20)).unwrap();
    assert_eq!(
        runtime.sink.offer(sample(11, 20)),
        Err(CostSampleDrop::Capacity)
    );
    runtime.consume_samples();
    let audit = runtime.audit_snapshot();
    assert_eq!(audit.sink.offered_completed, 2);
    assert_eq!(audit.sink.dropped_capacity, 1);
    assert_eq!(audit.training.pre_update_prediction.counts.completed, 1);
    assert_eq!(
        audit
            .training
            .pre_update_prediction
            .counts
            .no_published_model,
        1
    );
}

#[test]
fn non_completed_and_unavailable_model_are_separate_and_counters_saturate() {
    let completed = sample(10, 20);
    assert_eq!(
        PreUpdatePrediction::query(None, &completed),
        PreUpdatePrediction::NoPublishedModel
    );
    let mut failed = completed;
    failed.outcome = model::WaveObservationOutcome::FailedAfterSubmit;
    assert_eq!(
        PreUpdatePrediction::query(None, &failed),
        PreUpdatePrediction::NotCompleted
    );
    let mut counts = PredictionAuditCounts::filled(u64::MAX);
    let mut exhausted = false;
    counts.record(PreUpdatePrediction::NoPublishedModel, &mut exhausted);
    assert!(exhausted);
    assert_eq!(counts.completed, u64::MAX);
    assert_eq!(counts.no_published_model, u64::MAX);
}

#[test]
fn imported_prediction_uses_original_age_for_every_sample_in_the_batch() {
    use ferrum_scheduler::implementations::continuous::cost_profile as file;
    use std::io::Write;
    struct Temporary(std::path::PathBuf);
    impl Drop for Temporary {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let mut config = config();
    config.profile_import.declared_local_clock_max_error_ns = Some(0);
    let original = sample(0, 20);
    let profile = file::CostProfileFile {
        schema_version: file::COST_PROFILE_SCHEMA_VERSION,
        fingerprint: file::ProfileFingerprint::from(&original.fingerprint),
        settings: file::ProfileModelSettings::from(&super::super::super::profile::model_settings(
            &config.model,
        )),
        generated_unix_ns: 980,
        source_clock_max_error_ns: Some(0),
        source: file::ProfileSource {
            generator: "typed prediction audit fixture".into(),
            generator_revision: "v1".into(),
            measurement_protocol: "isolated complete observation".into(),
            observation_artifact_sha256: [9; 32],
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
                    decode_kv_tokens: vec![12],
                    prefill_chunks: Vec::new(),
                    recurrent_state_bytes: 0,
                    restore_bytes: 0,
                    maintenance_bytes: 0,
                    maintenance_units: 0,
                },
                boundary: file::ProfileCostBoundary::PreparationToCommit,
                outcome: file::ProfileObservationOutcome::Completed {},
                timing: file::ProfileWaveTiming {
                    wall_total_ns: 20,
                    device_elapsed_ns: None,
                    stages: file::ProfileStageTimings {
                        prepare: None,
                        device_wait: None,
                        commit: None,
                        restore: None,
                        maintenance: None,
                    },
                },
            })
            .collect(),
    };
    let path = Temporary(std::env::temp_dir().join(format!(
        "ferrum-prediction-audit-{}.json",
        uuid::Uuid::new_v4()
    )));
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path.0)
        .unwrap()
        .write_all(&serde_json::to_vec(&profile).unwrap())
        .unwrap();
    let runtime = EngineCostRuntime::build_with_profile(
        identity(),
        Arc::new(Clock),
        &config,
        false,
        Some(&path.0),
        Some(file::ProfileLoadClock {
            wall_unix_ns: Some(1000),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: 0,
        }),
    )
    .unwrap();
    let old = runtime.snapshot().unwrap();
    let query = PreUpdatePrediction::query(Some(&old), &sample(25, 20));
    assert!(
        matches!(
            query,
            PreUpdatePrediction::Known {
                valid_for_ns: 25,
                ..
            }
        ),
        "{query:?}"
    );
    for at in [25, 50, 51] {
        runtime.sink.offer(sample(at, 20)).unwrap();
    }
    runtime.consume_samples();
    let audit = runtime
        .audit_snapshot()
        .training
        .pre_update_prediction
        .counts;
    assert_eq!(audit.known, 2);
    assert_eq!(
        audit.unknown[PredictionUnknownReason::StaleSamples as usize].count,
        1,
        "the first fresh live update must not refresh this batch's imported snapshot"
    );
    assert!(matches!(
        PreUpdatePrediction::query(Some(&old), &sample(51, 20)),
        PreUpdatePrediction::Unknown {
            reason: PredictionUnknownReason::StaleSamples
        }
    ));
}
