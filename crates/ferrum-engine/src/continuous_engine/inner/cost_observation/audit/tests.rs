use super::super::*;
use super::*;
use std::num::NonZeroUsize;

struct Clock;
impl CostObservationClock for Clock {
    fn now_ns(&self) -> Option<u64> {
        Some(1000)
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
fn sample(at: u64) -> model::WaveCostObservation {
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
            decode_kv_tokens: vec![12],
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

#[test]
fn real_training_errors_and_skips_keep_independent_observation_denominator() {
    let mut config = ferrum_types::SloCostObservationConfig::default();
    config.model.min_samples = NonZeroUsize::MIN;
    config.model.max_buckets = NonZeroUsize::MIN;
    let runtime = EngineCostRuntime::build(identity(), Arc::new(Clock), &config, false).unwrap();
    let mut samples = vec![sample(100)];
    let mut capacity = sample(101);
    capacity.actual_shape.provider_signature[0] ^= 1;
    samples.push(capacity);
    samples.push(sample(99));
    let mut timing = sample(102);
    timing.timing.wall_total_ns = 0;
    samples.push(timing);
    let mut shape = sample(103);
    shape.actual_shape.decode_kv_tokens.clear();
    samples.push(shape);
    let mut fingerprint = sample(104);
    fingerprint.fingerprint.model_weights[0] ^= 1;
    samples.push(fingerprint);
    let mut deferred = sample(105);
    deferred.outcome = model::WaveObservationOutcome::Deferred;
    samples.push(deferred);
    let mut device = sample(106);
    device.boundary = model::CostBoundary::DeviceOnly;
    device.timing.device_elapsed_ns = None;
    samples.push(device);
    for sample in samples {
        runtime.sink.offer(sample).unwrap();
    }
    runtime.consume_samples();
    let snapshot = runtime.audit_snapshot();
    assert_eq!(
        (
            snapshot.sink.offered,
            snapshot.sink.published,
            snapshot.sink.drained
        ),
        (8, 8, 8)
    );
    assert_eq!(snapshot.sink.offered_completed, 7);
    assert_eq!(snapshot.training.consumed, 8);
    assert_eq!(snapshot.training.outcomes.recorded, 1);
    assert_eq!(snapshot.training.training_rejected(), 7);
    for reason in [
        TrainingErrorReason::CapacityExceeded,
        TrainingErrorReason::ClockMovedBackwards,
        TrainingErrorReason::InvalidTiming,
        TrainingErrorReason::InvalidShape,
        TrainingErrorReason::FingerprintMismatch,
    ] {
        assert_eq!(
            snapshot.training.outcomes.errors[reason as usize].count, 1,
            "{reason:?}: {snapshot:?}"
        );
    }
    assert_eq!(
        snapshot.training.outcomes.skipped[TrainingSkipReason::Deferred as usize].count,
        1
    );
    assert_eq!(
        snapshot.training.outcomes.skipped[TrainingSkipReason::MissingDeviceTiming as usize].count,
        1
    );
    assert_eq!(snapshot.training.published_snapshots, 1);
    assert_eq!(
        snapshot.training.by_wave[0].outcomes,
        snapshot.training.outcomes
    );
    assert!(!snapshot.training.counter_exhausted);
}

#[test]
fn missing_calibration_is_unavailable_not_a_model_error_or_recorded_sample() {
    let runtime = EngineCostRuntime::build(
        Default::default(),
        Arc::new(Clock),
        &Default::default(),
        false,
    )
    .unwrap();
    runtime.sink.offer(sample(100)).unwrap();
    runtime.consume_samples();
    let audit = runtime.audit_snapshot();
    assert_eq!(audit.training.outcomes.unavailable, 1);
    assert_eq!(audit.training.outcomes.recorded, 0);
    assert_eq!(audit.training.publish_attempts, 0);
    assert!(runtime.snapshot().is_none());
}

#[test]
fn queue_contention_and_capacity_losses_remain_in_the_offered_population() {
    let sink = BoundedCostSampleSink::new(CostSampleSinkLimits {
        max_samples: 1,
        max_shape_rows: 1,
    })
    .unwrap();
    sink.offer(sample(100)).unwrap();
    assert_eq!(sink.offer(sample(101)), Err(CostSampleDrop::Capacity));
    sink.with_locked_queue(|| assert_eq!(sink.offer(sample(102)), Err(CostSampleDrop::Contended)));
    let stats = sink.stats();
    assert_eq!(stats.offered, 3);
    assert_eq!(stats.offered_completed, 3);
    assert_eq!(
        stats.offered,
        stats.published + stats.dropped_capacity + stats.dropped_contention
    );
    assert_eq!(stats.offered_by_wave, [3, 0, 0, 0, 0]);
    assert!(stats.has_lost_samples());
    sink.pop().unwrap();
    assert_eq!(sink.stats().drained, 1);
}

#[test]
fn queue_accounts_numeric_allocations_and_returns_both_row_budgets_on_pop() {
    let mut observation = sample(100);
    observation.actual_shape.numeric_features = Some(CanonicalWaveCostFeatures {
        schema_version: COST_NUMERIC_FEATURE_SCHEMA_V1,
        output_policy_signature: [7; 32],
        rows: vec![CostRowNumericFeatures {
            generated_tokens_before: 1,
            maximum_output_tokens: 8,
            sampling_history_tokens: 1,
            repetition_tokens: 0,
            decoded_prefix_tokens: 2,
            decoded_text_bytes_bound: 8,
            decode_scratch_bytes_bound: 4,
        }],
    });
    let sink = BoundedCostSampleSink::new(CostSampleSinkLimits {
        max_samples: 2,
        max_shape_rows: 2,
    })
    .unwrap();
    sink.offer(observation.clone()).unwrap();
    assert_eq!(sink.offer(sample(101)), Err(CostSampleDrop::Capacity));
    assert_eq!(
        sink.pop().unwrap().actual_shape.numeric_features,
        observation.actual_shape.numeric_features
    );
    sink.offer(observation.clone()).unwrap();
    sink.pop().unwrap();
    observation
        .actual_shape
        .numeric_features
        .as_mut()
        .unwrap()
        .rows
        .reserve(8);
    assert_eq!(sink.offer(observation), Err(CostSampleDrop::Capacity));
    assert_eq!(sink.stats().dropped_capacity, 2);
    assert_eq!(sink.stats().drained, 2);
}

#[test]
fn different_actual_wave_kinds_do_not_merge_training_support() {
    let runtime =
        EngineCostRuntime::build(identity(), Arc::new(Clock), &Default::default(), false).unwrap();
    runtime.sink.offer(sample(100)).unwrap();
    let mut prefill = sample(101);
    prefill.actual_shape.kind = model::WaveKind::Prefill;
    prefill.actual_shape.decode_kv_tokens.clear();
    prefill
        .actual_shape
        .prefill_chunks
        .push(model::PrefillShape {
            offset: 0,
            count: std::num::NonZeroU32::new(4).unwrap(),
            total_prompt_tokens: std::num::NonZeroU32::new(8).unwrap(),
        });
    runtime.sink.offer(prefill).unwrap();
    runtime.consume_samples();
    let audit = runtime.audit_snapshot();
    assert_eq!(audit.sink.offered_by_wave, [1, 1, 0, 0, 0]);
    assert_eq!(audit.training.outcomes.recorded, 2);
    assert_eq!(audit.training.by_wave[0].outcomes.recorded, 1);
    assert_eq!(audit.training.by_wave[1].outcomes.recorded, 1);
    assert_eq!(audit.training.by_wave[2].outcomes.recorded, 0);
}

#[test]
fn fixed_counters_saturate_and_expose_exhaustion_without_wrapping() {
    let counter = AtomicU64::new(u64::MAX);
    let exhausted = AtomicBool::new(false);
    increment(&counter, &exhausted);
    assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
    assert!(exhausted.load(Ordering::Relaxed));
    let mut audit = TrainingAuditSnapshot::maximum_serialized_counts();
    audit.observe(model::WaveKind::Decode, TrainingDisposition::Recorded);
    audit.publish(Err(TrainingErrorReason::VersionExhausted));
    assert!(audit.counter_exhausted);
    assert_eq!(audit.consumed, u64::MAX);
    assert_eq!(audit.outcomes.recorded, u64::MAX);
    assert_eq!(audit.publish_attempts, u64::MAX);
}
