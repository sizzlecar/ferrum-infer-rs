//! Calibration/queue tests. Native success authority and future/actual fragment
//! equivalence are tested by the checkpoint interface fixtures, not fabricated here.
use super::*;
use ferrum_scheduler::implementations::continuous::slo_planner::{
    PlanningCostClockAnchor, PlanningPrefixCostModel, PrefixMaintenanceStage,
    PrefixRendezvousOffer, RequestWorkKey,
};
use std::num::{NonZeroU32, NonZeroUsize};

struct Clock(AtomicU64);
impl CostObservationClock for Clock {
    fn now_ns(&self) -> Option<u64> {
        Some(self.0.load(Ordering::Acquire))
    }
}
fn fingerprint() -> model::ExecutionFingerprint {
    model::ExecutionFingerprint {
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }
}
fn config() -> ferrum_types::SloCostObservationConfig {
    let mut config = ferrum_types::SloCostObservationConfig::default();
    config.model.feature_model = model::CostFeatureModel::ExactV1 {};
    config.model.min_samples = NonZeroUsize::new(2).unwrap();
    config.model.max_samples_per_bucket = NonZeroUsize::new(4).unwrap();
    config.model.max_sample_age_ns = NonZeroU64::new(100).unwrap();
    config.model.drift_margin_ns = 7;
    config
}
fn shape() -> model::WaveExecutionShape {
    model::WaveExecutionShape {
        row_multiset_features: None,
        host_content_features: None,
        numeric_features: None,
        kind: model::WaveKind::Maintenance,
        path: model::WaveExecutionPath::PlanRuntime,
        provider_signature: [8; 32],
        output_policy_signature: [9; 32],
        graph_state: model::WaveGraphState::Disabled,
        order: model::BatchOrderSemantics::Ordered,
        decode_kv_tokens: Vec::new(),
        prefill_chunks: Vec::new(),
        recurrent_state_bytes: 0,
        restore_bytes: 0,
        maintenance_bytes: 64,
        maintenance_units: 2,
    }
}
fn sample(at: u64, cost: u64) -> PrefixSample {
    PrefixSample {
        loss_epoch: None,
        observation: model::WaveCostObservation {
            fingerprint: fingerprint(),
            actual_shape: shape(),
            boundary: model::CostBoundary::PreparationToCommit,
            outcome: model::WaveObservationOutcome::Completed,
            timing: model::WaveTiming {
                wall_total_ns: cost,
                device_elapsed_ns: None,
                stages: Default::default(),
            },
            observed_at_ns: at,
        },
    }
}
fn offer() -> PrefixRendezvousOffer {
    let key = || RequestWorkKey {
        request_id: RequestId::new(),
        incarnation: 1,
        work_generation: Default::default(),
    };
    PrefixRendezvousOffer {
        identity: [7; 32],
        based_on_generation: 3,
        producer: key(),
        target: key(),
        boundary_tokens: NonZeroU32::new(16).unwrap(),
        expires_at_ns: 2000,
    }
}
fn fixture() -> (Arc<BoundedCostSampleSink>, PrefixCostTraining) {
    let queue = Arc::new(
        BoundedCostSampleSink::new(CostSampleSinkLimits {
            max_samples: 8,
            max_shape_rows: 8,
        })
        .unwrap(),
    );
    let training = PrefixCostTraining::new(
        queue.clone(),
        fingerprint(),
        Arc::new(Clock(AtomicU64::new(20))),
        &config().model,
    )
    .unwrap();
    (queue, training)
}
fn drain(queue: &BoundedCostSampleSink, training: &PrefixCostTraining) {
    while let Some(input) = queue.pop_training_input() {
        let CostTrainingInput::Prefix { sample, memory, .. } = input else {
            panic!("maintenance input")
        };
        let _memory = memory;
        training.consume(sample);
    }
}
fn train(training: &PrefixCostTraining) {
    training.consume(sample(10, 10));
    training.consume(sample(20, 20));
}

#[test]
fn prefix_cold_sampling_checks_capture_and_restore_qualification_separately() {
    let (_, training) = fixture();
    train(&training);
    let capture_known = training.snapshot().unwrap();
    assert!(!capture_known.cost_missing(&shape(), 20));
    let mut restore = shape();
    restore.kind = model::WaveKind::Restore;
    restore.restore_bytes = restore.maintenance_bytes;
    restore.maintenance_bytes = 0;
    assert!(capture_known.cost_missing(&restore, 20));

    let mut restore_sample = sample(20, 12);
    restore_sample.observation.actual_shape = restore.clone();
    training.consume(restore_sample);
    let restore_short = training.snapshot().unwrap();
    assert!(!capture_known.current());
    assert!(restore_short.cost_missing(&restore, 20));
    assert!(!restore_short.cost_missing(&shape(), 20));

    let mut restore_sample = sample(20, 14);
    restore_sample.observation.actual_shape = restore.clone();
    training.consume(restore_sample);
    let both_known = training.snapshot().unwrap();
    assert!(!both_known.cost_missing(&restore, 20));
    assert!(!both_known.cost_missing(&shape(), 20));
    // Previously unqualified retained snapshots cannot authorize new work.
    assert!(!restore_short.cost_missing(&restore, 20));
}

#[test]
fn prefix_cold_sampling_rejects_clock_protocol_and_revoked_snapshot_errors() {
    let (_, training) = fixture();
    train(&training);
    let model = training.snapshot().unwrap();
    let mut missing = shape();
    missing.provider_signature = [77; 32];
    assert!(model.cost_missing(&missing, 20));
    assert!(
        !model.cost_missing(&missing, 19),
        "clock failure is not absent evidence"
    );
    let mut invalid = shape();
    invalid.kind = model::WaveKind::Decode;
    assert!(
        !model.cost_missing(&invalid, 20),
        "invalid inference shape is not a maintenance sample"
    );
    assert!(
        model.cost_missing(&shape(), 121),
        "expiry uses the original sample ages"
    );
    training.consume(sample(20, 21));
    assert!(
        !model.cost_missing(&missing, 20),
        "revoked snapshots cannot permit sampling"
    );
}

#[test]
fn prefix_cold_sampling_distinguishes_publication_contention_from_absence() {
    let (_, training) = fixture();
    assert!(matches!(training.try_snapshot(), Some(None)));
    let publication = training.snapshot.write();
    assert!(training.try_snapshot().is_none());
    drop(publication);
    train(&training);
    assert!(matches!(training.try_snapshot(), Some(Some(_))));
    let publication = training.snapshot.write();
    assert!(training.try_snapshot().is_none());
    drop(publication);
    assert!(!training
        .try_snapshot()
        .unwrap()
        .unwrap()
        .cost_missing(&shape(), 20));
}

#[test]
fn prefix_cost_min_samples_exact_quantile_original_ttl_and_offer_identity() {
    let (_, training) = fixture();
    let offer = offer();
    training.consume(sample(10, 10));
    let first = training.snapshot().unwrap();
    let anchored = first.anchored(PlanningCostClockAnchor::exact(1000, 20), &offer);
    assert!(anchored
        .predict(
            &fingerprint(),
            &offer,
            PrefixMaintenanceStage::Capture,
            &shape(),
            1005
        )
        .is_none());
    training.consume(sample(20, 20));
    assert!(
        !first.current(),
        "new publication revokes already-held old snapshots"
    );
    let current = training.snapshot().unwrap();
    let anchored = current.anchored(PlanningCostClockAnchor::exact(1000, 20), &offer);
    let cost = anchored
        .predict(
            &fingerprint(),
            &offer,
            PrefixMaintenanceStage::Capture,
            &shape(),
            1005,
        )
        .unwrap();
    assert_eq!(
        (cost.typical_ns, cost.planning_ns, cost.valid_for_ns),
        (10, 27, 85)
    );
    assert_eq!(cost.model_version, current.model_version());
    let mut wrong = fingerprint();
    wrong.model_weights = [4; 32];
    assert!(anchored
        .predict(
            &wrong,
            &offer,
            PrefixMaintenanceStage::Capture,
            &shape(),
            1005
        )
        .is_none());
    let mut other = offer.clone();
    other.target.request_id = RequestId::new();
    assert!(anchored
        .predict(
            &fingerprint(),
            &other,
            PrefixMaintenanceStage::Capture,
            &shape(),
            1005
        )
        .is_none());
    assert!(anchored
        .predict(
            &fingerprint(),
            &offer,
            PrefixMaintenanceStage::Restore,
            &shape(),
            1005
        )
        .is_none());
    let mut geometry = shape();
    geometry.provider_signature = [10; 32];
    assert!(anchored
        .predict(
            &fingerprint(),
            &offer,
            PrefixMaintenanceStage::Capture,
            &geometry,
            1005
        )
        .is_none());
    assert!(anchored
        .predict(
            &fingerprint(),
            &offer,
            PrefixMaintenanceStage::Capture,
            &shape(),
            1091
        )
        .is_none());
    assert!(anchored
        .predict(
            &fingerprint(),
            &offer,
            PrefixMaintenanceStage::Capture,
            &shape(),
            999
        )
        .is_none());
}

#[test]
fn prefix_cost_delayed_fifo_never_retimes_maintenance_receipts() {
    let (queue, training) = fixture();
    assert_eq!(queue.offer_prefix(sample(10, 10)).unwrap(), 1);
    assert_eq!(queue.offer_prefix(sample(20, 20)).unwrap(), 2);
    drain(&queue, &training);
    let snapshot = training.snapshot().unwrap();
    let offer = offer();
    let anchored = snapshot.anchored(PlanningCostClockAnchor::exact(1000, 1000), &offer);
    assert!(anchored
        .predict(
            &fingerprint(),
            &offer,
            PrefixMaintenanceStage::Capture,
            &shape(),
            1000
        )
        .is_none());
    let stats = queue.stats();
    assert_eq!(stats.offered, 0);
    assert_eq!(
        stats.entries_offered, 0,
        "maintenance is not an inference feedback denominator"
    );
    assert_eq!(stats.memory.queued_raw_bytes, 0);
}

#[test]
fn prefix_cost_restore_requires_its_own_complete_initialization_domain() {
    let (_, training) = fixture();
    train(&training);
    let offer = offer();
    let mut restore = shape();
    restore.kind = model::WaveKind::Restore;
    restore.provider_signature = [17; 32];
    restore.restore_bytes = 64;
    restore.maintenance_bytes = 32;
    restore.maintenance_units = 3;
    let before = training.snapshot().unwrap();
    assert!(before
        .anchored(PlanningCostClockAnchor::exact(1000, 20), &offer)
        .predict(
            &fingerprint(),
            &offer,
            PrefixMaintenanceStage::Restore,
            &restore,
            1000
        )
        .is_none());
    for (at, cost) in [(30, 15), (40, 25)] {
        let mut receipt = sample(at, cost);
        receipt.observation.actual_shape = restore.clone();
        training.consume(receipt);
    }
    let snapshot = training.snapshot().unwrap();
    let anchored = snapshot.anchored(PlanningCostClockAnchor::exact(1000, 40), &offer);
    let cost = anchored
        .predict(
            &fingerprint(),
            &offer,
            PrefixMaintenanceStage::Restore,
            &restore,
            1000,
        )
        .unwrap();
    assert_eq!(cost.planning_ns, 32);
    let mut incomplete = restore;
    incomplete.maintenance_bytes = 0;
    incomplete.maintenance_units = 2;
    assert!(anchored
        .predict(
            &fingerprint(),
            &offer,
            PrefixMaintenanceStage::Restore,
            &incomplete,
            1000
        )
        .is_none());
}

#[test]
fn prefix_cost_old_snapshot_keeps_its_lease_after_replacement() {
    let (queue, training) = fixture();
    train(&training);
    let old = training.snapshot().unwrap();
    let before = queue.stats().memory.worker_and_result_bytes;
    training.consume(sample(30, 15));
    let held = queue.stats().memory.worker_and_result_bytes;
    assert!(!old.current());
    assert!(
        held > before,
        "old externally-held snapshot remains charged"
    );
    drop(old);
    assert!(queue.stats().memory.worker_and_result_bytes < held);
    drop(training);
    assert_eq!(queue.stats().memory.worker_and_result_bytes, 0);
}

#[test]
fn prefix_cost_shared_quota_refusal_revokes_without_training_or_unbounded_growth() {
    let (queue, training) = fixture();
    train(&training);
    let old = training.snapshot().unwrap();
    let before = queue.stats().memory.worker_and_result_bytes;
    let limit = CostRecorderByteLimits::default().maximum_working_bytes;
    let blocker = queue.reserve_prefix_working_bytes(limit - before).unwrap();
    let samples = training.state.lock().trainer.retained_sample_count();
    training.consume(sample(30, 40));
    assert!(training.snapshot().is_none());
    assert!(!old.current());
    assert_eq!(
        training.state.lock().trainer.retained_sample_count(),
        samples
    );
    assert_eq!(queue.stats().memory.worker_and_result_bytes, limit);
    drop(blocker);
    training.consume(sample(40, 40));
    assert!(training.snapshot().is_some());
}

#[test]
fn prefix_cost_queue_loss_does_not_revoke_inference_population() {
    let queue = BoundedCostSampleSink::new(CostSampleSinkLimits {
        max_samples: 1,
        max_shape_rows: 1,
    })
    .unwrap();
    queue.offer_prefix(sample(10, 10)).unwrap();
    assert_eq!(
        queue.offer_prefix(sample(20, 20)),
        Err(CostSampleDrop::Capacity)
    );
    let stats = queue.stats();
    assert_eq!(stats.entries_dropped_capacity, 0);
    assert!(!stats.has_lost_samples());
    let CostTrainingInput::Prefix { ordinal, .. } = queue.pop_training_input().unwrap() else {
        panic!("prefix")
    };
    assert_eq!(ordinal, 1);
}

#[test]
fn prefix_cost_unwound_publication_advances_fifo_without_training_or_inference_loss() {
    let (queue, training) = fixture();
    train(&training);
    let old = training.snapshot().unwrap();
    let mut receipt = sample(30, 30);
    receipt.loss_epoch = Some(training.epoch.clone());
    queue.on_next_send(|| panic!("controlled producer publication unwind"));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| queue.offer_prefix(receipt)))
            .is_err()
    );
    assert!(!old.current());
    assert_eq!(queue.stats().entries_abandoned, 0);
    assert!(!queue.stats().has_lost_samples());
    assert!(matches!(
        queue.pop_training_input(),
        Some(CostTrainingInput::PrefixDiscarded { ordinal: 1 })
    ));
    assert_eq!(queue.stats().memory.queued_raw_bytes, 0);
    assert_eq!(queue.offer_prefix(sample(40, 40)).unwrap(), 2);
    drain(&queue, &training);
    assert!(training.snapshot().is_some());
}

#[test]
fn prefix_cost_control_lock_does_not_drop_but_shutdown_revokes_queued_maintenance() {
    let (queue, training) = fixture();
    train(&training);
    let old = training.snapshot().unwrap();
    let mut receipt = sample(30, 30);
    receipt.loss_epoch = Some(training.epoch.clone());
    assert_eq!(
        queue
            .with_locked_consumer(|| queue.offer_prefix(receipt))
            .unwrap(),
        1
    );
    assert!(old.current());
    queue.worker_stopped();
    assert!(!old.current());
    assert_eq!(queue.stats().memory.queued_raw_bytes, 0);
    assert!(!queue.stats().has_lost_samples());
    assert_eq!(
        queue.offer_prefix(sample(40, 40)),
        Err(CostSampleDrop::WorkerStopped)
    );
}

#[test]
fn prefix_cost_worker_preserves_inference_model_while_training_maintenance() {
    let clock = Arc::new(Clock(AtomicU64::new(20)));
    let fp = fingerprint();
    let identity = ExecutorCostIdentityAvailability::Known(Arc::new(ExecutorCostIdentity {
        schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
        model_weights: fp.model_weights,
        numerical_policy: fp.numerical_policy,
        device_runtime: fp.device_runtime,
        execution_config: fp.execution_config,
    }));
    let runtime = EngineCostRuntime::build(identity, clock, &config(), false).unwrap();
    for at in [10, 20] {
        let mut inference = sample(at, at).observation;
        inference.actual_shape.kind = model::WaveKind::Decode;
        inference.actual_shape.decode_kv_tokens = vec![16];
        inference.actual_shape.maintenance_bytes = 0;
        inference.actual_shape.maintenance_units = 0;
        runtime.sink.offer(inference).unwrap();
    }
    runtime.training.consume_batch();
    let prior = runtime.training.snapshot().unwrap();
    runtime.sink.offer_prefix(sample(10, 10)).unwrap();
    runtime.sink.offer_prefix(sample(20, 20)).unwrap();
    runtime.training.consume_batch();
    assert!(Arc::ptr_eq(&prior, &runtime.training.snapshot().unwrap()));
    let prefix = runtime.prefix_cost_snapshot().unwrap();
    let offer = offer();
    assert!(prefix
        .anchored(PlanningCostClockAnchor::exact(1000, 20), &offer)
        .predict(&fp, &offer, PrefixMaintenanceStage::Capture, &shape(), 1000)
        .is_some());
    assert!(runtime.prefix_cost_sink().is_some());
}

#[test]
fn prefix_cost_lost_receipt_cannot_be_overwritten_by_older_publication_or_epoch_aba() {
    let (_, training) = fixture();
    let in_progress = training.epoch.next().unwrap();
    training.epoch.next(); // Producer loses a later receipt during this update.
    training.consume_inner(sample(10, 10), in_progress).unwrap();
    assert!(
        training.snapshot().is_none(),
        "older update cannot reactivate after loss"
    );
    train(&training);
    assert!(training.snapshot().is_some());
    training.epoch.revision.store(u64::MAX, Ordering::Release);
    assert!(training.epoch.next().is_none());
    assert!(training.epoch.next().is_none());
    assert!(!training.epoch.current(u64::MAX));
}

#[test]
fn prefix_cost_invalid_zero_row_inference_and_other_fingerprint_never_train() {
    let (_, training) = fixture();
    train(&training);
    let count = training.state.lock().trainer.retained_sample_count();
    let mut invalid = sample(30, 30);
    invalid.observation.actual_shape.kind = model::WaveKind::Prefill;
    training.consume(invalid);
    assert!(training.snapshot().is_none());
    assert_eq!(training.state.lock().trainer.retained_sample_count(), count);
    let mut invalid = sample(40, 40);
    invalid.observation.fingerprint.execution_config = [42; 32];
    training.consume(invalid);
    assert!(training.snapshot().is_none());
    assert_eq!(training.state.lock().trainer.retained_sample_count(), count);
}

#[test]
fn ready_prefix_cost_keeps_original_ttl_and_exact_target_offer() {
    use ferrum_scheduler::implementations::continuous::slo_planner::ReadyPrefixRestoreOffer;
    let (_, training) = fixture();
    let mut restore = shape();
    restore.kind = model::WaveKind::Restore;
    restore.restore_bytes = restore.maintenance_bytes;
    restore.maintenance_bytes = 0;
    for (at, cost) in [(10, 10), (20, 20)] {
        let mut observed = sample(at, cost);
        observed.observation.actual_shape = restore.clone();
        training.consume(observed);
    }
    let rendezvous = offer();
    let offer = ReadyPrefixRestoreOffer {
        identity: rendezvous.identity,
        based_on_generation: rendezvous.based_on_generation,
        target: rendezvous.target.clone(),
        boundary_tokens: rendezvous.boundary_tokens,
        expires_at_ns: 2000,
    };
    let snapshot = training.snapshot().unwrap();
    let anchored = snapshot.anchored_ready(PlanningCostClockAnchor::exact(1000, 20), &offer);
    let predicted = anchored
        .predict_ready_restore(&fingerprint(), &offer, &restore, 1005)
        .unwrap();
    assert_eq!(
        (
            predicted.typical_ns,
            predicted.planning_ns,
            predicted.valid_for_ns
        ),
        (10, 27, 85)
    );
    assert!(anchored
        .predict_ready_restore(&fingerprint(), &offer, &restore, 1091)
        .is_none());
    let mut other = offer.clone();
    other.target.incarnation += 1;
    assert!(anchored
        .predict_ready_restore(&fingerprint(), &other, &restore, 1005)
        .is_none());
    other = offer.clone();
    other.expires_at_ns += 1;
    assert!(anchored
        .predict_ready_restore(&fingerprint(), &other, &restore, 1005)
        .is_none());
    assert!(anchored
        .predict(
            &fingerprint(),
            &rendezvous,
            PrefixMaintenanceStage::Restore,
            &restore,
            1005
        )
        .is_none());
    assert!(anchored
        .predict_ready_restore(&fingerprint(), &offer, &shape(), 1005)
        .is_none());
}

#[test]
fn ready_prefix_cost_rejects_unqualified_revoked_and_changed_hardware() {
    use ferrum_scheduler::implementations::continuous::slo_planner::ReadyPrefixRestoreOffer;
    let (_, training) = fixture();
    let mut observed = sample(20, 10);
    observed.observation.actual_shape.kind = model::WaveKind::Restore;
    observed.observation.actual_shape.restore_bytes =
        observed.observation.actual_shape.maintenance_bytes;
    observed.observation.actual_shape.maintenance_bytes = 0;
    let restore = observed.observation.actual_shape.clone();
    let ordinary = offer();
    let offer = ReadyPrefixRestoreOffer {
        identity: ordinary.identity,
        based_on_generation: ordinary.based_on_generation,
        target: ordinary.target,
        boundary_tokens: ordinary.boundary_tokens,
        expires_at_ns: 1050,
    };
    let observed = || {
        let mut result = sample(20, 10);
        result.observation.actual_shape = restore.clone();
        result
    };
    training.consume(observed());
    let insufficient = training.snapshot().unwrap();
    assert!(insufficient
        .anchored_ready(PlanningCostClockAnchor::exact(1000, 20), &offer)
        .predict_ready_restore(&fingerprint(), &offer, &restore, 1000)
        .is_none());
    training.consume(observed());
    let qualified = training.snapshot().unwrap();
    let anchored = qualified.anchored_ready(PlanningCostClockAnchor::exact(1000, 20), &offer);
    assert_eq!(
        anchored
            .predict_ready_restore(&fingerprint(), &offer, &restore, 1000)
            .unwrap()
            .valid_for_ns,
        50
    );
    let mut foreign = fingerprint();
    foreign.device_runtime = [55; 32];
    assert!(anchored
        .predict_ready_restore(&foreign, &offer, &restore, 1000)
        .is_none());
    assert!(anchored
        .predict_ready_restore(&fingerprint(), &offer, &restore, 1051)
        .is_none());
    training.consume(observed());
    assert!(!qualified.current());
    assert!(anchored
        .predict_ready_restore(&fingerprint(), &offer, &restore, 1000)
        .is_none());
}

#[test]
fn cache_capture_cost_uses_original_ttl_and_exact_source_offer() {
    use ferrum_scheduler::implementations::continuous::slo_planner::PrefixCacheCaptureOffer;
    let (_, training) = fixture();
    train(&training);
    let snapshot = training.snapshot().unwrap();
    let rendezvous = offer();
    let offer = PrefixCacheCaptureOffer {
        identity: [61; 32],
        based_on_generation: rendezvous.based_on_generation,
        source: rendezvous.producer.clone(),
        capture_span_start: 0,
        boundary_tokens: rendezvous.boundary_tokens,
        expires_at_ns: 1050,
    };
    let anchored = snapshot.anchored_capture(PlanningCostClockAnchor::exact(1000, 20), &offer);
    let cost = anchored
        .predict_cache_capture(&fingerprint(), &offer, &shape(), 1000)
        .unwrap();
    assert_eq!(cost.valid_for_ns, 50);
    assert!(anchored
        .predict_cache_capture(&fingerprint(), &offer, &shape(), 1051)
        .is_none());
    let mut foreign = offer.clone();
    foreign.source.incarnation += 1;
    assert!(anchored
        .predict_cache_capture(&fingerprint(), &foreign, &shape(), 1000)
        .is_none());
    foreign = offer.clone();
    foreign.capture_span_start += 1;
    assert!(anchored
        .predict_cache_capture(&fingerprint(), &foreign, &shape(), 1000)
        .is_none());
    assert!(anchored
        .predict(
            &fingerprint(),
            &rendezvous,
            PrefixMaintenanceStage::Capture,
            &shape(),
            1000
        )
        .is_none());
    let mut wrong_shape = shape();
    wrong_shape.kind = model::WaveKind::Restore;
    assert!(anchored
        .predict_cache_capture(&fingerprint(), &offer, &wrong_shape, 1000)
        .is_none());
    let mut device = fingerprint();
    device.device_runtime = [91; 32];
    assert!(anchored
        .predict_cache_capture(&device, &offer, &shape(), 1000)
        .is_none());
    training.consume(sample(30, 15));
    assert!(anchored
        .predict_cache_capture(&fingerprint(), &offer, &shape(), 1000)
        .is_none());
}

#[test]
fn cache_capture_cost_does_not_promote_an_unqualified_domain() {
    use ferrum_scheduler::implementations::continuous::slo_planner::PrefixCacheCaptureOffer;
    let (_, training) = fixture();
    training.consume(sample(10, 10));
    let snapshot = training.snapshot().unwrap();
    let old = offer();
    let offer = PrefixCacheCaptureOffer {
        identity: [62; 32],
        based_on_generation: old.based_on_generation,
        source: old.producer,
        capture_span_start: 0,
        boundary_tokens: old.boundary_tokens,
        expires_at_ns: 1100,
    };
    let anchored = snapshot.anchored_capture(PlanningCostClockAnchor::exact(1000, 20), &offer);
    assert!(anchored
        .predict_cache_capture(&fingerprint(), &offer, &shape(), 1000)
        .is_none());
}
