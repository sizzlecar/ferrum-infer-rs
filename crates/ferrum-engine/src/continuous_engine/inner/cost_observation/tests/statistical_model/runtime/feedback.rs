//! Real recorder/FIFO, immutable schema6 import and worker feedback. Durations
//! are fixture clock values, not a performance experiment or a drift threshold.
use super::*;
use ferrum_types::{
    SloSelectedFeedbackPolicy as Policy, SloSelectedFeedbackSettingsV1 as Settings,
    SloSelectedFeedbackStorageV1 as Storage,
};

pub(in crate::continuous_engine::inner) struct FeedbackFixture {
    fixture: Fixture,
    directory: PathBuf,
    pub clock: Arc<VirtualClock>,
}
impl FeedbackFixture {
    pub fn new(terminal: bool) -> Self {
        Self::with_ceiling(terminal, 5_000_000)
    }
    pub fn with_ceiling(terminal: bool, maximum_family_margin_ns: u64) -> Self {
        let mut fixture = Fixture::with_terminal(terminal);
        let directory =
            std::env::temp_dir().join(format!("ferrum-feedback-worker-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&directory).unwrap();
        fixture.config.selected_feedback = Policy::RetrospectiveFamilyMarginV1 {
            policy: Settings {
                window_samples: NonZeroUsize::new(2).unwrap(),
                minimum_underestimates: NonZeroUsize::new(2).unwrap(),
                minimum_consecutive_underestimates: NonZeroUsize::new(2).unwrap(),
                trigger_excess_ns: NonZeroU64::new(1_000).unwrap(),
                correction_padding_ns: 100,
                maximum_family_margin_ns: NonZeroU64::new(maximum_family_margin_ns).unwrap(),
                maximum_consumption_lag_ns: NonZeroU64::new(10_000_000).unwrap(),
                maximum_uncomparable_observations: 8,
                maximum_failed_or_partial: 8,
                maximum_queue_drops: 0,
                maximum_state_bytes: NonZeroUsize::new(64 * 1024).unwrap(),
            },
            storage: Storage::CreateNew {
                path: directory.join("feedback.json"),
            },
        };
        Self {
            fixture,
            directory,
            clock: Arc::new(VirtualClock(AtomicU64::new(500))),
        }
    }
    pub fn build(&self) -> EngineCostRuntime {
        self.fixture.build(self.clock.clone()).unwrap()
    }
    pub fn train_margin(&self, runtime: &EngineCostRuntime, terminal: bool) {
        self.clock.set(3_000_000);
        for epoch in [300, 400] {
            let entry = recorded_case(&runtime.ids, &sink(4, 32), terminal, epoch, 2_000_000).1;
            runtime.sink.offer_evidence_numbered(entry).unwrap();
        }
        runtime.consume_samples();
    }
    fn resume(&mut self) {
        let Policy::RetrospectiveFamilyMarginV1 { storage, .. } =
            &mut self.fixture.config.selected_feedback
        else {
            unreachable!()
        };
        *storage = Storage::Resume {
            path: self.directory.join("feedback.json"),
        };
    }
}
impl Drop for FeedbackFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

#[tokio::test]
async fn selected_feedback_worker_publishes_base_relative_margin_and_persists_without_refit_or_ttl_reset(
) {
    for terminal in [false, true] {
        let mut f = FeedbackFixture::new(terminal);
        let profile_bytes = fs::read(&f.fixture.path).unwrap();
        let runtime = f.build();
        let old = runtime.snapshot().unwrap();
        let sample = f.fixture.samples[0].clone();
        let base = old
            .predict_selected_wave(&sample.exact, &sample.selected, 500)
            .unwrap();
        // A normal genuine receipt changes diagnostics only.
        runtime
            .sink
            .offer_evidence_numbered(recorded_case(&runtime.ids, &sink(4, 32), terminal, 200, 0).1)
            .unwrap();
        runtime.consume_samples();
        assert!(Arc::ptr_eq(&old, &runtime.snapshot().unwrap()));
        f.train_margin(&runtime, terminal);
        let current = runtime.snapshot().unwrap();
        assert_eq!(current.model_version(), old.model_version() + 1);
        assert_eq!(
            old.predict_selected_wave(&sample.exact, &sample.selected, 3_000_000),
            Err(ModelUnknown::RuntimeValidity)
        );
        let corrected = current
            .predict_selected_wave(&sample.exact, &sample.selected, 3_000_000)
            .unwrap();
        assert!(corrected.planning_ns > base.planning_ns);
        assert_eq!(corrected.fitted_ns, base.fitted_ns);
        assert_eq!(corrected.residual_ns, base.residual_ns);
        f.train_margin(&runtime, terminal);
        assert!(
            Arc::ptr_eq(&current, &runtime.snapshot().unwrap()),
            "the same error cannot compound the overlay"
        );
        let audit = runtime.audit_snapshot();
        assert_eq!(audit.training.publish_attempts, 0);
        assert_eq!(runtime.trained_samples(), 0);
        assert_eq!(audit.selected_feedback.as_ref().unwrap().corrections, 1);
        assert_eq!(
            audit
                .training
                .selected_serving
                .as_ref()
                .unwrap()
                .terminal_compared,
            if terminal { 5 } else { 0 }
        );
        runtime.shutdown().await.unwrap();
        assert!(
            !current.current(),
            "clean shutdown closes the old process gate"
        );
        drop(runtime);
        f.resume();
        let resumed = f.build();
        let loaded = resumed.snapshot().unwrap();
        assert_eq!(loaded.model_version(), current.model_version());
        assert_eq!(
            loaded
                .predict_selected_wave(&sample.exact, &sample.selected, 3_000_000)
                .unwrap()
                .planning_ns,
            corrected.planning_ns
        );
        f.clock
            .set(100 + f.fixture.config.model.max_sample_age_ns.get());
        assert_eq!(
            loaded.predict_selected_wave(
                &sample.exact,
                &sample.selected,
                f.clock.now_ns().unwrap()
            ),
            Err(ModelUnknown::Stale)
        );
        assert_eq!(fs::read(&f.fixture.path).unwrap(), profile_bytes);
        resumed.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn selected_feedback_missing_expired_and_queue_loss_never_train_or_forge_zero_error() {
    let mut f = FeedbackFixture::new(true);
    f.fixture.config.max_queued_samples = NonZeroUsize::MIN;
    f.fixture.config.max_samples_per_update = NonZeroUsize::MIN;
    let runtime = f.build();
    let old = runtime.snapshot().unwrap();
    let entry = recorded_case(&runtime.ids, &sink(4, 32), true, 200, 0).1;
    let CostEvidenceEntry::StagesOnly {
        stages,
        legacy_rejection,
    } = entry
    else {
        panic!()
    };
    let mut missing = stages.as_ref().clone();
    missing.statistical_evidence = None;
    runtime
        .sink
        .offer_evidence_numbered(CostEvidenceEntry::StagesOnly {
            stages: Arc::new(missing),
            legacy_rejection,
        })
        .unwrap();
    runtime.consume_samples();
    assert!(Arc::ptr_eq(&old, &runtime.snapshot().unwrap()));
    let audit = runtime.audit_snapshot().selected_feedback.unwrap();
    assert_eq!((audit.compared, audit.uncomparable_observations), (0, 1));
    f.clock
        .set(100 + f.fixture.config.model.max_sample_age_ns.get());
    runtime
        .sink
        .offer_evidence_numbered(recorded_case(&runtime.ids, &sink(4, 32), true, 200, 0).1)
        .unwrap();
    runtime.consume_samples();
    assert_eq!(
        runtime.audit_snapshot().selected_feedback.unwrap().compared,
        0
    );
    runtime
        .sink
        .offer_evidence_numbered(recorded_case(&runtime.ids, &sink(4, 32), true, 200, 0).1)
        .unwrap();
    assert_eq!(
        runtime
            .sink
            .offer_evidence_numbered(recorded_case(&runtime.ids, &sink(4, 32), true, 200, 0).1),
        Err(CostSampleDrop::Capacity)
    );
    runtime.consume_samples();
    assert!(runtime.snapshot().is_none());
    assert!(!old.current());
    assert_eq!(
        serde_json::to_value(runtime.audit_snapshot()).unwrap()["selected_feedback"]["revoked"],
        "queue_loss"
    );
    runtime.shutdown().await.unwrap();
    drop(runtime);
    f.resume();
    let resumed = f.build();
    assert!(
        resumed.snapshot().is_none(),
        "clean restart cannot erase a sticky artifact revocation"
    );
    resumed.shutdown().await.unwrap();
}

#[tokio::test]
async fn selected_feedback_persistence_failure_closes_old_epoch_and_blocks_restart() {
    let mut f = FeedbackFixture::new(true);
    let runtime = f.build();
    let old = runtime.snapshot().unwrap();
    fs::create_dir(f.directory.join("feedback.json.pending")).unwrap();
    f.train_margin(&runtime, true);
    assert!(runtime.snapshot().is_none());
    assert!(!old.current());
    assert!(
        runtime
            .audit_snapshot()
            .selected_feedback
            .unwrap()
            .persistence_failed
    );
    assert!(runtime.shutdown().await.is_err());
    drop(runtime);
    f.resume();
    assert!(
        f.fixture.build(f.clock.clone()).is_err(),
        "dirty session remains unavailable, not a reset model"
    );
}

#[tokio::test]
async fn selected_presubmit_comparison_requires_the_real_call_issued_bound_and_exact_receipt() {
    let mut fixture = FeedbackFixture::new(true);
    let Policy::RetrospectiveFamilyMarginV1 { policy, .. } =
        &mut fixture.fixture.config.selected_feedback
    else {
        unreachable!()
    };
    policy.trigger_excess_ns = NonZeroU64::new(3_000_000).unwrap();
    fixture.clock.set(3_000_000);
    let runtime = fixture.build();
    let snapshot = runtime.snapshot().unwrap();
    let sample = &fixture.fixture.samples[0];
    let bound = snapshot
        .predict_selected_wave(&sample.exact, &sample.selected, 500)
        .unwrap()
        .planning_ns;
    let (_, entry, _) = recorded_case_with_prediction(
        &runtime.ids,
        &sink(4, 32),
        true,
        200,
        2_000_000,
        Some((&runtime, snapshot.model_version(), bound)),
    );
    let CostEvidenceEntry::StagesOnly {
        stages,
        legacy_rejection,
    } = entry
    else {
        panic!()
    };
    let issued = stages.presubmit_prediction.as_ref().unwrap();
    assert!(issued.exact_actual_matches);
    assert_eq!(issued.planning_ns, bound);
    // The valid same-call item and explicitly corrupted/missing negatives have
    // separate denominator buckets, irrespective of the later model lookup.
    runtime
        .sink
        .offer_evidence_numbered(CostEvidenceEntry::StagesOnly {
            stages: stages.clone(),
            legacy_rejection,
        })
        .unwrap();
    let mut mismatch = stages.as_ref().clone();
    mismatch
        .presubmit_prediction
        .as_mut()
        .unwrap()
        .exact_actual_matches = false;
    runtime
        .sink
        .offer_evidence_numbered(CostEvidenceEntry::StagesOnly {
            stages: Arc::new(mismatch),
            legacy_rejection,
        })
        .unwrap();
    runtime
        .sink
        .offer_evidence_numbered(recorded_case(&runtime.ids, &sink(4, 32), true, 300, 2_000_000).1)
        .unwrap();
    runtime.consume_samples();
    let audit = runtime.audit_snapshot().training.selected_serving.unwrap();
    assert_eq!((audit.known_compared, audit.underestimates), (3, 3));
    assert_eq!(
        (
            audit.presubmit.compared,
            audit.presubmit.underestimates,
            audit.presubmit.terminal_compared
        ),
        (1, 1, 1)
    );
    assert_eq!(
        (
            audit.presubmit.exact_mismatch,
            audit.presubmit.no_issued_bound
        ),
        (1, 1)
    );
    assert_eq!(audit.presubmit.maximum_underestimate_ns, 2_000_019 - bound);
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn selected_feedback_real_worker_unwind_closes_epoch_and_leaves_restart_dirty() {
    use crate::continuous_engine::inner::cost_observation::{
        profile,
        trainer::{CostTrainingState, TrainingWorkerOwner},
        worker::CostTrainingWorker,
    };
    let mut f = FeedbackFixture::new(true);
    let seed = profile::load_seed(
        &identity(),
        &f.fixture.config,
        Some(&f.fixture.path),
        Some(Fixture::load_clock()),
    )
    .unwrap();
    let training =
        Arc::new(CostTrainingState::new(&f.fixture.config, seed, None, f.clock.clone()).unwrap());
    let old = training.snapshot().unwrap();
    let owner = TrainingWorkerOwner(training.clone());
    let worker = CostTrainingWorker::spawn(move || {
        let _retained = &owner;
        panic!("injected cost-worker unwind");
    })
    .unwrap();
    assert!(!worker.shutdown().await);
    assert!(training.snapshot().is_none());
    assert!(!old.current());
    assert!(
        training
            .audit_snapshot()
            .selected_feedback
            .unwrap()
            .worker_failed
    );
    assert!(training.export_result().is_err());
    f.resume();
    assert!(f.fixture.build(f.clock.clone()).is_err());
}
