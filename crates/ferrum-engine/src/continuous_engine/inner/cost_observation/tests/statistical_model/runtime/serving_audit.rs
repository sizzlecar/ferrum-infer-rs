//! Ordinary worker + real recorder + real schema6 file loader. The virtual
//! clock controls host settlement/queue delay, not the predictor's answer.
use super::*;
use crate::continuous_engine::inner::cost_observation::audit::{
    SelectedNotCompleted, SelectedUnknownReason,
};

fn entry(runtime: &EngineCostRuntime, terminal: bool, epoch: u64, delay: u64) -> CostEvidenceEntry {
    recorded_case(&runtime.ids, &sink(4, 32), terminal, epoch, delay).1
}

#[test]
fn selected_serving_worker_compares_real_training_and_terminal_receipts_without_republication() {
    for terminal in [false, true] {
        let fixture = Fixture::with_terminal(terminal);
        let file_before = fs::read(&fixture.path).unwrap();
        let clock = Arc::new(VirtualClock(AtomicU64::new(500)));
        let runtime = fixture.build(clock.clone()).unwrap();
        let before = runtime.snapshot().unwrap();
        let normal = entry(&runtime, terminal, 200, 0);
        assert_eq!(
            matches!(&normal, CostEvidenceEntry::StagesOnly { .. }),
            terminal
        );
        runtime.sink.offer_evidence_numbered(normal).unwrap();
        runtime.consume_samples();
        // Same selected work, genuinely later host settlement; no shape/model
        // or expected error is edited. All complete host time must be compared.
        let slow = entry(&runtime, terminal, 300, 2_000_000);
        clock.set(3_000_000);
        runtime.sink.offer_evidence_numbered(slow).unwrap();
        runtime.consume_samples();
        let audit = runtime.audit_snapshot();
        let selected = audit.training.selected_serving.as_ref().unwrap();
        assert_eq!(
            (
                selected.drained_entries,
                selected.constructable_actual,
                selected.known_compared
            ),
            (2, 2, 2)
        );
        assert_eq!(
            selected.constructable_terminal,
            if terminal { 2 } else { 0 }
        );
        assert_eq!(selected.terminal_compared, if terminal { 2 } else { 0 });
        assert_eq!(selected.underestimates, 1);
        assert!(selected.max_underestimate_ns > 0);
        assert_eq!(
            selected.total_underestimate_ns,
            selected.max_underestimate_ns
        );
        assert_eq!(audit.training.publish_attempts, 0);
        assert_eq!(runtime.trained_samples(), 0);
        assert!(Arc::ptr_eq(&before, &runtime.snapshot().unwrap()));
        assert_eq!(
            before.model_version(),
            runtime.snapshot().unwrap().model_version()
        );
        assert_eq!(fs::read(&fixture.path).unwrap(), file_before);
        let serialized = serde_json::to_value(audit).unwrap();
        assert_eq!(
            serialized["training"]["selected_serving"]["known_compared"],
            2
        );
    }
}

#[test]
fn selected_serving_missing_producer_and_failed_receipts_never_become_zero_error_samples() {
    let fixture = Fixture::new();
    let clock = Arc::new(VirtualClock(AtomicU64::new(500)));
    let runtime = fixture.build(clock).unwrap();
    let CostEvidenceEntry::StagesOnly {
        stages,
        legacy_rejection,
    } = entry(&runtime, true, 200, 0)
    else {
        panic!()
    };
    let mut missing = stages.as_ref().clone();
    missing.statistical_evidence = None;
    let mut failed = stages.as_ref().clone();
    failed.completeness = HostStageCompleteness::Failed;
    for stages in [missing, failed] {
        runtime
            .sink
            .offer_evidence_numbered(CostEvidenceEntry::StagesOnly {
                stages: Arc::new(stages),
                legacy_rejection,
            })
            .unwrap();
    }
    // Explicit non-completion is independently typed even if a caller carries
    // a formerly complete receipt; it cannot override the declared outcome.
    let CostEvidenceEntry::Training { mut sample, stages } = entry(&runtime, false, 200, 0) else {
        panic!()
    };
    sample.outcome = model::WaveObservationOutcome::NotSubmitted;
    runtime
        .sink
        .offer_evidence_numbered(CostEvidenceEntry::Training { sample, stages })
        .unwrap();
    runtime.consume_samples();
    let audit = runtime.audit_snapshot().training.selected_serving.unwrap();
    assert_eq!(audit.drained_entries, 3);
    assert_eq!(
        (
            audit.constructable_actual,
            audit.known_compared,
            audit.underestimates
        ),
        (0, 0, 0)
    );
    assert_eq!(
        audit
            .not_completed
            .iter()
            .find(|c| c.reason == SelectedNotCompleted::NotSubmitted)
            .unwrap()
            .count,
        1
    );
    assert_eq!(
        audit
            .actual_unavailable
            .iter()
            .find(|c| c.reason
                == SelectedUnknownReason::Evidence(StatisticalEvidenceUnknown::MissingProducer))
            .unwrap()
            .count,
        1
    );
    assert_eq!(
        audit
            .actual_unavailable
            .iter()
            .find(|c| c.reason == SelectedUnknownReason::InvalidSample)
            .unwrap()
            .count,
        1
    );
}

#[test]
fn selected_serving_queued_observation_uses_consumption_freshness_without_refreshing_ttl() {
    let fixture = Fixture::new();
    let clock = Arc::new(VirtualClock(AtomicU64::new(500)));
    let runtime = fixture.build(clock.clone()).unwrap();
    let before = runtime.snapshot().unwrap();
    let queued = entry(&runtime, true, 200, 0);
    let original =
        super::super::super::super::trainer::whole_wave_observation(&queued, 1, [7; 32]).unwrap();
    assert!(before
        .predict_selected_wave(&original.exact, &original.selected, original.observed_at_ns)
        .is_ok());
    runtime.sink.offer_evidence_numbered(queued).unwrap();
    clock.set(100 + fixture.config.model.max_sample_age_ns.get());
    runtime.consume_samples();
    let audit = runtime.audit_snapshot().training.selected_serving.unwrap();
    assert_eq!(
        (
            audit.constructable_actual,
            audit.constructable_terminal,
            audit.known_compared
        ),
        (1, 1, 0)
    );
    assert_eq!(
        audit
            .prediction_unknown
            .iter()
            .find(|c| c.reason == SelectedUnknownReason::Stale)
            .unwrap()
            .count,
        1
    );
    assert_eq!(audit.underestimates, 0);
    assert!(Arc::ptr_eq(&before, &runtime.snapshot().unwrap()));
    assert_eq!(runtime.audit_snapshot().training.publish_attempts, 0);
}

#[test]
fn selected_serving_future_receipt_clock_does_not_produce_a_comparison() {
    let fixture = Fixture::new();
    let clock = Arc::new(VirtualClock(AtomicU64::new(220)));
    let runtime = fixture.build(clock).unwrap();
    runtime
        .sink
        .offer_evidence_numbered(entry(&runtime, true, 200, 0))
        .unwrap();
    runtime.consume_samples();
    let audit = runtime.audit_snapshot().training.selected_serving.unwrap();
    assert_eq!(audit.constructable_actual, 1);
    assert_eq!(audit.known_compared, 0);
    assert_eq!(
        audit
            .prediction_unknown
            .iter()
            .find(|c| c.reason == SelectedUnknownReason::Clock)
            .unwrap()
            .count,
        1
    );
}

#[test]
fn selected_serving_queue_losses_and_unsubmitted_preparations_stay_outside_compared_denominator() {
    let mut fixture = Fixture::new();
    fixture.config.max_queued_samples = NonZeroUsize::MIN;
    fixture.config.max_samples_per_update = NonZeroUsize::MIN;
    let runtime = fixture
        .build(Arc::new(VirtualClock(AtomicU64::new(500))))
        .unwrap();
    runtime
        .sink
        .reject_preparation(CostCallRejection::NoPhysicalWave);
    runtime
        .sink
        .offer_evidence_numbered(entry(&runtime, true, 200, 0))
        .unwrap();
    assert_eq!(
        runtime
            .sink
            .offer_evidence_numbered(entry(&runtime, true, 200, 0)),
        Err(CostSampleDrop::Capacity)
    );
    runtime.consume_samples();
    let audit = runtime.audit_snapshot();
    assert_eq!(
        (
            audit.sink.entries_offered,
            audit.sink.entries_published,
            audit.sink.entries_dropped_capacity
        ),
        (2, 1, 1)
    );
    assert_eq!(
        audit.sink.preparation_rejected[CostCallRejection::NoPhysicalWave.index()],
        1
    );
    let selected = audit.training.selected_serving.unwrap();
    assert_eq!((selected.drained_entries, selected.known_compared), (1, 1));
}

#[test]
fn selected_serving_without_publication_and_legacy_mode_keep_separate_audit_population() {
    let clock = Arc::new(VirtualClock(AtomicU64::new(500)));
    let runtime = EngineCostRuntime::build(
        identity(),
        clock.clone(),
        &SloCostObservationConfig::selected_whole_wave_v1(),
        false,
    )
    .unwrap();
    runtime
        .sink
        .offer_evidence_numbered(entry(&runtime, true, 200, 0))
        .unwrap();
    runtime.consume_samples();
    let selected = runtime.audit_snapshot().training.selected_serving.unwrap();
    assert_eq!(
        (
            selected.constructable_actual,
            selected.no_published_model,
            selected.known_compared
        ),
        (1, 1, 0)
    );
    assert!(runtime.snapshot().is_none());
    let legacy = EngineCostRuntime::build(
        identity(),
        clock,
        &SloCostObservationConfig::default(),
        false,
    )
    .unwrap();
    assert!(
        serde_json::to_value(legacy.audit_snapshot()).unwrap()["training"]
            .get("selected_serving")
            .is_none()
    );
}
