use super::*;
use crate::learning::{rejection::Recorder, RejectionWork};

fn decode(context: usize) -> CostWork {
    CostWork {
        requested: ExecutionRange::new(context - 1, 1).unwrap(),
        context_tokens: context,
        ..work()
    }
}

fn query(work: &[CostWork]) -> CostQuery<'_> {
    CostQuery {
        work,
        work_is_exact: true,
        execution: CostExecution::SingleInvocation,
    }
}

fn trained(family: ModelFamily) -> (Learner, Shared, Instant) {
    let config = config();
    let shared = shared(&config);
    let mut learner = Learner::new(config, identity(), family);
    let start = Instant::now();
    for i in 0..learner.config.required_samples() {
        let at = start + Duration::from_millis(i as u64);
        learner.observe(
            &shared,
            TrainingSample {
                features: WorkFeatures::extract(&[decode(16); 4], 4).unwrap(),
                service_seconds: if i < 6 { 0.01 } else { 0.02 },
            },
            at,
            at,
        );
    }
    learner.config.retrain_every = usize::MAX;
    (learner, shared, start + Duration::from_millis(7))
}

fn outside() -> CostPrediction {
    CostPrediction::Unknown(CostUnknownReason::UnsupportedWithDetail(
        CostUnsupportedDetail::OutsideObservedSupport,
    ))
}

#[test]
fn both_models_record_same_call_bounds_and_preserve_old_snapshot_after_publication() {
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        let (mut learner, shared, end) = trained(family);
        let (source, _) = snapshot::Source::new(
            learner.config.query,
            identity(),
            learner.config.work_bucket_boundaries.clone(),
        );
        source.install(learner.snapshot(end).unwrap());
        let old = source.load_typed();
        let original_model = serde_json::to_value(&*old.domains[0].model).unwrap();
        let known = old.predict(&query(&[decode(16); 4]), end);
        assert!(matches!(known, CostPrediction::Known { .. }));
        assert_eq!(source.rejection_stats().outside_observed_support, 0);

        let work = [decode(80)];
        assert_eq!(old.predict(&query(&work), end), outside());
        let stats = source.rejection_stats();
        let witness = stats.last.unwrap();
        assert!(witness.unjoined_to_scheduler);
        assert_eq!(witness.snapshot_revision, old.revision);
        assert_eq!(witness.model_revision, old.domains[0].model_revision);
        assert_eq!(witness.family, family);
        assert_eq!(witness.feature_schema, crate::model::FEATURE_SCHEMA);
        let features = WorkFeatures::extract(&work, 4).unwrap();
        assert_eq!(witness.domain, features.domain());
        assert_eq!(witness.values, *features.values());
        let (minimum, maximum) = old.domains[0].model.support_bounds();
        assert_eq!(witness.minimum, *minimum);
        assert_eq!(witness.maximum, *maximum);
        assert_eq!(witness.below_mask, (1 << 0) | (1 << 1) | (1 << 6));
        assert_eq!(witness.above_mask, (1 << 2) | (1 << 8));
        assert_eq!(stats.below_axis_counts, [1, 1, 0, 0, 0, 0, 1, 0, 0]);
        assert_eq!(stats.above_axis_counts, [0, 0, 1, 0, 0, 0, 0, 0, 1]);
        assert_eq!(stats.outside_observed_support, 1);
        assert_eq!(witness.work, vec![RejectionWork::from(work[0])]);
        assert_eq!(witness.execution, "single_invocation");
        assert!(witness.work_is_exact);
        assert_eq!(
            witness.model_valid_for_at_query_ns,
            Some((old.domains[0].model_expires_at - end).as_nanos() as u64)
        );
        serde_json::to_string(&witness).unwrap();

        // A real causal observation extends the new published envelope. The
        // old immutable snapshot still rejects using its own original bounds.
        let later = end + Duration::from_millis(1);
        learner.observe(
            &shared,
            TrainingSample {
                features,
                service_seconds: 0.03,
            },
            later,
            later,
        );
        source.install(learner.snapshot(later).unwrap());
        let new = source.load_typed();
        assert!(new.revision > old.revision);
        assert!(Arc::ptr_eq(
            old.rejections.as_ref().unwrap(),
            new.rejections.as_ref().unwrap()
        ));
        assert!(matches!(
            new.predict(&query(&work), later),
            CostPrediction::Known { .. }
        ));
        assert_eq!(source.rejection_stats().outside_observed_support, 1);
        assert_eq!(old.predict(&query(&work), later), outside());
        let retained_old = source.rejection_stats().last.unwrap();
        assert_eq!(retained_old.snapshot_revision, old.revision);
        assert_eq!(retained_old.minimum, witness.minimum);
        assert_eq!(retained_old.maximum, witness.maximum);
        assert_eq!(old.predict(&query(&[decode(16); 4]), end), known);
        assert_eq!(
            serde_json::to_value(&*old.domains[0].model).unwrap(),
            original_model
        );

        // Scenario callers reach the same single extraction/rejection path.
        assert_eq!(
            new.cost_scenarios(&query(&[decode(81)]), later, 4),
            CostScenarioPrediction::Unknown(CostUnknownReason::UnsupportedWithDetail(
                CostUnsupportedDetail::OutsideObservedSupport
            ))
        );
        let stats = source.rejection_stats();
        let witness = stats.last.unwrap();
        assert_eq!(stats.outside_observed_support, 3);
        assert_eq!(stats.stored, 3);
        assert_eq!(stats.overwritten, 2);
        assert_eq!(witness.snapshot_revision, new.revision);
        assert_eq!(witness.maximum, *new.domains[0].model.support_bounds().1);
        assert_eq!(witness.below_mask, 0);
        assert_eq!(witness.above_mask, (1 << 2) | (1 << 8));
    }
}

#[test]
fn prediction_and_stats_never_wait_for_slot_and_contention_keeps_previous_witness() {
    let (mut learner, _, end) = trained(ModelFamily::Physical);
    let mut snapshot = learner.snapshot(end).unwrap();
    let recorder = Arc::new(Recorder::new(4, end));
    snapshot.rejections = Some(recorder.clone());
    let first = [decode(80)];
    assert_eq!(snapshot.predict(&query(&first), end), outside());
    // Holding the same non-reentrant mutex proves both calls below must use
    // try_lock. No sleeps, races or wall-time performance thresholds are used.
    recorder.with_slot_held(|| {
        assert_eq!(snapshot.predict(&query(&[decode(81)]), end), outside());
        let busy = recorder.stats();
        assert!(busy.last_witness_read_busy);
        assert!(busy.last.is_none());
        assert_eq!(busy.outside_observed_support, 2);
        assert_eq!(busy.stored, 1);
        assert_eq!(busy.contention_dropped, 1);
        assert_eq!(busy.above_axis_counts[2], 2);
    });
    let stats = recorder.stats();
    assert!(!stats.last_witness_read_busy);
    assert_eq!(
        stats.last.unwrap().work,
        vec![RejectionWork::from(first[0])]
    );
    assert_eq!(snapshot.predict(&query(&[decode(82)]), end), outside());
    let stats = recorder.stats();
    assert_eq!(stats.stored, 2);
    assert_eq!(stats.overwritten, 1);
    assert_eq!(
        stats.outside_observed_support,
        stats.stored + stats.contention_dropped
    );
    assert_eq!(stats.last.unwrap().work[0].context_tokens, 82);
}

#[test]
fn fixed_work_storage_preserves_maximum_query_and_rejects_oversize_without_truncation() {
    let (mut learner, _, end) = trained(ModelFamily::Physical);
    let mut snapshot = learner.snapshot(end).unwrap();
    let recorder = Arc::new(Recorder::new(256, end));
    snapshot.rejections = Some(recorder.clone());
    snapshot.query.max_participants = 256;
    snapshot.query.max_query_bytes = CostQuery::projected_bytes(256).unwrap();
    let full = vec![decode(80); 256];
    assert_eq!(snapshot.predict(&query(&full), end), outside());
    assert_eq!(recorder.stats().last.unwrap().work.len(), 256);
    let oversized = vec![decode(80); 257];
    let features = WorkFeatures::extract(&oversized, 257).unwrap();
    // Defense in depth: the public query guard normally rejects this before
    // feature extraction. A direct recorder call must not resize or truncate.
    recorder.record(
        snapshot.revision,
        &snapshot.domains[0],
        &features,
        &query(&oversized),
        end,
    );
    let stats = recorder.stats();
    assert_eq!(stats.outside_observed_support, 2);
    assert_eq!(stats.stored, 1);
    assert_eq!(stats.oversize_dropped, 1);
    assert_eq!(stats.last.unwrap().work.len(), 256);
    assert_eq!(
        stats.retained_bytes_bound,
        Recorder::retained_bytes_bound(256).unwrap()
    );
}

fn minimum_retained_budget(mut config: LearningConfig) -> usize {
    let mut low = 0;
    let mut high = config.max_retained_bytes;
    while low < high {
        let middle = low + (high - low) / 2;
        config.max_retained_bytes = middle;
        if config.validate().is_ok() {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    config.max_retained_bytes = low;
    config.validate().unwrap();
    config.max_retained_bytes = low - 1;
    assert!(config.validate().is_err());
    low
}

#[test]
fn resident_recorder_memory_is_checked_and_charged_at_exact_config_boundary() {
    assert!(Recorder::retained_bytes_bound(usize::MAX).is_none());
    assert_eq!(
        Recorder::retained_bytes_bound(256).unwrap() - Recorder::retained_bytes_bound(1).unwrap(),
        255 * std::mem::size_of::<CostWork>()
    );
    let mut small = config();
    small.query.max_participants = 1;
    small.query.max_query_bytes = CostQuery::projected_bytes(256).unwrap();
    let small_minimum = minimum_retained_budget(small.clone());
    let mut large = small;
    large.query.max_participants = 256;
    let large_minimum = minimum_retained_budget(large);
    assert_eq!(
        large_minimum - small_minimum,
        255 * std::mem::size_of::<CostWork>()
    );
}

#[test]
fn work_evidence_distinguishes_unobserved_restore_and_unknown_internal_reuse() {
    let unknown = RejectionWork::from(work());
    let observed_zero = RejectionWork::from(CostWork {
        prefix_restore: AttemptPrefixRestore::Restored { tokens: 0 },
        internally_reused_tokens: Some(0),
        ..work()
    });
    assert_eq!(unknown.prefix_restore_tokens, None);
    assert_eq!(unknown.internally_reused_tokens, None);
    assert_eq!(observed_zero.prefix_restore_tokens, Some(0));
    assert_eq!(observed_zero.internally_reused_tokens, Some(0));
    assert_ne!(unknown, observed_zero);
}
