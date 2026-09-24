use super::super::{
    profile_export::ExportStatus,
    trainer::{CostTrainingState, TrainingWorkerOwner},
    worker::CostTrainingWorker,
    *,
};
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model as model;
use ferrum_types::SloCostObservationConfig;
use std::{
    num::{NonZeroU64, NonZeroUsize},
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

struct Clock(AtomicU64);
impl CostObservationClock for Clock {
    fn now_ns(&self) -> Option<u64> {
        Some(self.0.load(Ordering::Relaxed))
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
fn config() -> SloCostObservationConfig {
    let mut config = SloCostObservationConfig::default();
    config.model.min_samples = NonZeroUsize::MIN;
    config.model.max_sample_age_ns = NonZeroU64::new(100).unwrap();
    config
}
fn runtime(background: bool) -> EngineCostRuntime {
    EngineCostRuntime::build(
        identity(),
        Arc::new(Clock(AtomicU64::new(20))),
        &config(),
        background,
    )
    .unwrap()
}
fn sample(at: u64, kv: u32) -> model::WaveCostObservation {
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
        observed_at_ns: at,
    }
}
fn prediction(
    snapshot: &EngineCostSnapshot,
    sample: &model::WaveCostObservation,
    now: u64,
) -> model::CostPrediction {
    snapshot.predict(
        &sample.fingerprint,
        &sample.actual_shape,
        sample.boundary,
        now,
    )
}

#[tokio::test]
async fn cost_checkpoint_public_calibration_queries_use_live_clock_and_frozen_model() {
    use crate::continuous_engine::inner::calibration::FrozenCalibrationModel;
    let clock = Arc::new(Clock(AtomicU64::new(20)));
    let runtime = EngineCostRuntime::build(identity(), clock.clone(), &config(), false).unwrap();
    let old = sample(10, 12);
    runtime.sink.offer(old.clone()).unwrap();
    let waiter = runtime.request_checkpoint().unwrap();
    runtime.consume_samples();
    let frozen = FrozenCalibrationModel::new(waiter.wait().await.unwrap(), clock.clone());
    assert_eq!(frozen.accepted_ordinal(), 1);
    let version = frozen.model_version();
    assert!(matches!(
        frozen.predict(&old.actual_shape).unwrap(),
        Some(model::CostPrediction::Known(_))
    ));

    let validation = sample(12, 256);
    runtime.sink.offer(validation.clone()).unwrap();
    runtime.consume_samples();
    assert!(matches!(
        prediction(&runtime.snapshot().unwrap(), &validation, 20),
        model::CostPrediction::Known(_)
    ));
    assert!(matches!(
        frozen.predict(&validation.actual_shape).unwrap(),
        Some(model::CostPrediction::Unknown(_))
    ));
    assert_eq!(frozen.model_version(), version);
    let audit = frozen.audit().unwrap();
    assert_eq!(audit["accepted_ordinal"], 1);
    assert_eq!(audit["training"]["consumed"], 1);

    // The caller cannot supply receipt time to make this stale model appear fresh.
    clock.0.store(111, Ordering::Relaxed);
    assert!(matches!(
        frozen.predict(&old.actual_shape).unwrap(),
        Some(model::CostPrediction::Unknown(
            model::CostUnknownReason::StaleSamples
        ))
    ));
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn cost_checkpoint_rejected_observation_advances_cut_and_post_cut_stays_out() {
    let runtime = runtime(false);
    let before = sample(10, 12);
    runtime.sink.offer(before.clone()).unwrap();
    let mut invalid = before.clone();
    invalid.timing.wall_total_ns = 0;
    runtime.sink.offer(invalid).unwrap();
    let waiter = runtime.request_checkpoint().unwrap();
    let later = sample(12, 256);
    runtime.sink.offer(later.clone()).unwrap();
    runtime.consume_samples();
    let frozen = waiter.wait().await.unwrap();
    assert_eq!(frozen.accepted_ordinal, 2);
    assert_eq!(frozen.training.consumed, 2);
    assert_eq!(frozen.training.outcomes.recorded, 1);
    assert_eq!(runtime.sink.stats().drained, 2);
    let snapshot = frozen.snapshot.unwrap();
    assert!(matches!(
        prediction(&snapshot, &before, 20),
        model::CostPrediction::Known(_)
    ));
    assert!(matches!(
        prediction(&snapshot, &later, 20),
        model::CostPrediction::Unknown(_)
    ));
    runtime.consume_samples();
    assert_eq!(runtime.sink.stats().drained, 3);
    assert!(matches!(
        prediction(&runtime.snapshot().unwrap(), &later, 20),
        model::CostPrediction::Known(_)
    ));
    assert!(matches!(
        prediction(&snapshot, &later, 20),
        model::CostPrediction::Unknown(_)
    ));
}

#[tokio::test]
async fn cost_checkpoint_full_queue_cancelled_waiter_and_second_request_keep_progress() {
    let mut config = config();
    config.max_queued_samples = NonZeroUsize::new(2).unwrap();
    config.max_samples_per_update = NonZeroUsize::new(2).unwrap();
    let runtime = EngineCostRuntime::build(
        identity(),
        Arc::new(Clock(AtomicU64::new(20))),
        &config,
        false,
    )
    .unwrap();
    runtime.sink.offer(sample(10, 12)).unwrap();
    runtime.sink.offer(sample(11, 12)).unwrap();
    let cancelled = runtime.request_checkpoint().unwrap(); // independent control slot
    assert!(matches!(
        runtime.request_checkpoint(),
        Err(CheckpointRequestError::Busy)
    ));
    drop(cancelled);
    runtime.consume_samples();
    assert_eq!(runtime.trained_samples(), 2);
    assert_eq!(runtime.sink.stats().dropped_capacity, 0);
    let next = runtime.request_checkpoint().unwrap();
    runtime.consume_samples();
    assert_eq!(next.wait().await.unwrap().accepted_ordinal, 2);
    runtime.sink.offer(sample(12, 12)).unwrap();
    runtime.consume_samples();
    assert_eq!(runtime.trained_samples(), 3);
}

#[tokio::test]
async fn cost_checkpoint_empty_cut_preserves_snapshot_and_original_age() {
    let runtime = runtime(false);
    let old = sample(10, 12);
    runtime.sink.offer(old.clone()).unwrap();
    runtime.consume_samples();
    let original = runtime.snapshot().unwrap();
    let version = original.model_version();
    let waiter = runtime.request_checkpoint().unwrap();
    runtime.consume_samples();
    let frozen = waiter.wait().await.unwrap().snapshot.unwrap();
    assert!(Arc::ptr_eq(&original, &frozen));
    assert_eq!(frozen.model_version(), version);
    let model::CostPrediction::Known(value) = prediction(&frozen, &old, 20) else {
        panic!("valid receipt must remain known");
    };
    assert_eq!(
        (value.oldest_sample_at_ns, value.newest_sample_at_ns),
        (10, 10)
    );
    assert!(matches!(
        prediction(&frozen, &old, 111),
        model::CostPrediction::Unknown(model::CostUnknownReason::StaleSamples)
    ));
}

#[tokio::test]
async fn cost_checkpoint_unknown_identity_completes_without_inventing_snapshot() {
    let runtime = EngineCostRuntime::build(
        Default::default(),
        Arc::new(Clock(AtomicU64::new(20))),
        &config(),
        false,
    )
    .unwrap();
    runtime.sink.offer(sample(10, 12)).unwrap();
    let waiter = runtime.request_checkpoint().unwrap();
    runtime.consume_samples();
    let frozen = waiter.wait().await.unwrap();
    assert_eq!(frozen.accepted_ordinal, 1);
    assert_eq!(frozen.training.outcomes.unavailable, 1);
    assert!(frozen.snapshot.is_none());
    assert_eq!(frozen.export.status, ExportStatus::Disabled);
}

#[tokio::test]
async fn cost_checkpoint_waits_for_popped_sample_to_finish_and_excludes_concurrent_offer() {
    let runtime = runtime(true);
    let (held, waiting) = std::sync::mpsc::channel();
    let (release, resume) = std::sync::mpsc::channel();
    runtime.sink.on_next_pop(move || {
        held.send(()).unwrap();
        resume.recv().unwrap();
    });
    runtime.sink.offer(sample(10, 12)).unwrap();
    waiting.recv_timeout(Duration::from_secs(3)).unwrap();
    assert_eq!(runtime.sink.stats().drained, 1); // popped, not yet trained
    let waiter = runtime.request_checkpoint().unwrap().wait();
    tokio::pin!(waiter);
    assert!(futures::poll!(waiter.as_mut()).is_pending());
    runtime.sink.offer(sample(12, 256)).unwrap();
    release.send(()).unwrap();
    let frozen = tokio::time::timeout(Duration::from_secs(3), waiter)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(frozen.accepted_ordinal, 1);
    assert_eq!(frozen.training.outcomes.recorded, 1);
    assert!(matches!(
        prediction(&frozen.snapshot.unwrap(), &sample(12, 256), 20),
        model::CostPrediction::Unknown(_)
    ));
    runtime.shutdown().await.unwrap();
    assert_eq!(runtime.trained_samples(), 2);
}

#[tokio::test]
async fn cost_checkpoint_shutdown_completes_accepted_barrier_and_closes_new_requests() {
    let runtime = runtime(false);
    runtime.sink.offer(sample(10, 12)).unwrap();
    let waiter = runtime.request_checkpoint().unwrap();
    runtime.shutdown().await.unwrap();
    assert_eq!(waiter.wait().await.unwrap().accepted_ordinal, 1);
    assert!(matches!(
        runtime.request_checkpoint(),
        Err(CheckpointRequestError::Closing)
    ));
    runtime.shutdown().await.unwrap();
    assert_eq!(runtime.trained_samples(), 1);
}

#[tokio::test]
async fn cost_checkpoint_worker_unwind_notifies_waiter_while_runtime_state_is_retained() {
    let config = config();
    let seed = profile::load_seed(&identity(), &config, None, None).unwrap();
    let state = Arc::new(
        CostTrainingState::new(&config, seed, None, Arc::new(Clock(AtomicU64::new(20)))).unwrap(),
    );
    let owner = TrainingWorkerOwner(state.clone());
    let worker = CostTrainingWorker::spawn(move || {
        let _owner = &owner;
        panic!("injected worker exit before consuming its accepted queue");
    })
    .unwrap();
    state
        .sink
        .attach_worker(worker.notification_thread())
        .unwrap();
    let waiter = state.sink.request_checkpoint().unwrap();
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(3), waiter.wait())
            .await
            .unwrap(),
        Err(CheckpointError::WorkerStopped)
    ));
    assert!(!worker.shutdown().await);
    assert!(matches!(
        state.sink.request_checkpoint(),
        Err(CheckpointRequestError::Closing)
    ));
}
