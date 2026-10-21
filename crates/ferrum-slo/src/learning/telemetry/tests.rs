use super::*;
use crate::{
    calibration::{CalibrationConfig, OnlineCalibrator},
    learning::snapshot::Domain,
    model::{ExecutionIdentity, FittedModel, TrainingBudget, TrainingSample},
};
use ferrum_execution::token_execution::{ExecutedTokenWork, ExecutionRange};
use ferrum_scheduler::cost_prediction::{CostExecution, CostPrediction, CostQuery, CostSnapshot};
use ferrum_types::{CostQueryLimits, RequestId};

struct Sink(bool);
impl ExecutionAttemptObserver for Sink {
    fn limits(&self) -> ExecutionAttemptLimits {
        ExecutionAttemptLimits {
            max_participants: 8,
            max_summary_bytes: 16384,
        }
    }
    fn try_record(&self, _: &ExecutionAttemptSummary) -> ExecutionAttemptDelivery {
        if self.0 {
            ExecutionAttemptDelivery::Recorded
        } else {
            ExecutionAttemptDelivery::Dropped(ExecutionAttemptDrop::QueueFull)
        }
    }
}

fn participant() -> ExecutionAttemptParticipant {
    ExecutionAttemptParticipant {
        request_id: RequestId::new(),
        kind: AttemptWorkKind::Decode,
        requested: ExecutionRange::new(3, 1).unwrap(),
        context_tokens: 4,
        prefix_restore: AttemptPrefixRestore::NoneObserved,
        internally_reused_tokens: None,
        product_failed: false,
        outcome: AttemptParticipantOutcome::Completed {
            work: ExecutedTokenWork::Decode { position: 3 },
            frontier_committed: true,
        },
    }
}
fn features() -> WorkFeatures {
    WorkFeatures::extract(&[sample_work(&participant())], 8).unwrap()
}
fn snapshot(family: ModelFamily, revision: u64, center: f64, upper: f64, now: Instant) -> Snapshot {
    let identity = ExecutionIdentity::new("telemetry-test-executor".into()).unwrap();
    let samples = [TrainingSample {
        features: features(),
        service_seconds: center,
    }; 4];
    let model = FittedModel::train(
        family,
        identity.clone(),
        &samples[..2],
        &samples[2..],
        &TrainingBudget::default(),
    )
    .unwrap();
    let base = model.predict(&identity, &features()).unwrap();
    let mut calibration = OnlineCalibrator::new(
        CalibrationConfig {
            max_domains: 1,
            buckets_per_domain: 1,
            window_samples: 4,
            min_samples: 2,
            ttl: Duration::from_secs(10),
            quantile: 0.95,
        },
        revision,
    )
    .unwrap();
    for _ in 0..2 {
        assert!(calibration.observe(
            revision,
            features().domain(),
            0,
            upper,
            base.risk_base_seconds,
            now
        ));
    }
    Snapshot {
        revision,
        identity: Arc::new(identity),
        query: CostQueryLimits::default(),
        boundaries: Vec::new().into(),
        quantile: 0.95,
        domains: vec![Domain {
            model: Arc::new(model),
            model_revision: revision,
            model_expires_at: now + Duration::from_secs(30),
            calibration: calibration.snapshot(now),
        }],
    }
}
fn completion(
    snapshot: &Snapshot,
    telemetry: &Arc<Telemetry>,
    now: Instant,
    forwarded: bool,
) -> Completion {
    let (prediction, unknown) = freeze(snapshot, features(), now);
    Completion {
        target: Arc::new(Sink(forwarded)),
        telemetry: telemetry.clone(),
        features: features(),
        frozen_at: now,
        prediction,
        unknown,
    }
}
fn summary(now: Instant, millis: u64) -> ExecutionAttemptSummary {
    ExecutionAttemptSummary {
        started_at: now + Duration::from_millis(1),
        completed_at: now + Duration::from_millis(1 + millis),
        participants: vec![participant()],
        eligibility: ExecutionAttemptEligibility::Completed,
    }
}

#[test]
fn frozen_prediction_survives_rebase_and_queue_drop_with_honest_error_denominators() {
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        let now = Instant::now();
        let telemetry = Arc::new(Telemetry::new(family, 0.95));
        let first = snapshot(family, 1, 0.01, 0.03, now);
        let ticket = completion(&first, &telemetry, now, false);
        let old = ticket.prediction.unwrap();
        let (source, _) = Source::new(first.query, (*first.identity).clone(), vec![]);
        source.install(first);
        source.install(snapshot(family, 2, 0.1, 0.2, now));
        let sample = summary(now, 40);
        assert_eq!(
            ticket.record_at(&sample, sample.completed_at),
            ExecutionAttemptDelivery::Dropped(ExecutionAttemptDrop::QueueFull)
        );
        let stats = telemetry.stats();
        let d = &stats.domains[index(features().domain())];
        assert_eq!((d.eligible, d.base_known, d.calibrated_known), (1, 1, 1));
        assert_eq!(
            (d.calibrated_exceedances, d.forwarded, d.forwarding_dropped),
            (1, 0, 1)
        );
        assert_eq!(
            (d.min_model_revision, d.max_model_revision),
            (Some(1), Some(1))
        );
        assert!((d.central_signed_error_seconds - (0.04 - old.central)).abs() < 1e-12);
        assert!((d.central_squared_error_seconds - (0.04 - old.central).powi(2)).abs() < 1e-12);
        assert!((d.calibrated_excess_seconds - 0.01).abs() < 1e-9);
        assert_eq!(source.load_typed().revision, 2);
    }
}

#[test]
fn frozen_calibrated_value_matches_public_prediction_and_ttl_is_checked_at_prepare() {
    let now = Instant::now();
    let snapshot = snapshot(ModelFamily::Gbdt, 4, 0.01, 0.03, now);
    let (frozen, unknown) = freeze(&snapshot, features(), now);
    assert!(unknown.is_none());
    let CostPrediction::Known {
        upper_service_time, ..
    } = snapshot.predict(
        &CostQuery {
            work: &[sample_work(&participant())],
            work_is_exact: true,
            execution: CostExecution::SingleInvocation,
        },
        now,
    )
    else {
        panic!("qualified fixture");
    };
    assert_eq!(
        frozen.unwrap().upper,
        Some(upper_service_time.as_secs_f64())
    );
    let telemetry = Arc::new(Telemetry::new(ModelFamily::Gbdt, 0.95));
    let expired = now + Duration::from_secs(11);
    let ticket = completion(&snapshot, &telemetry, expired, true);
    let sample = summary(expired, 40);
    ticket.record_at(&sample, sample.completed_at);
    let stats = telemetry.stats();
    let d = &stats.domains[index(features().domain())];
    assert_eq!((d.base_known, d.calibrated_known, d.stale), (1, 0, 1));
    assert_eq!(d.calibrated_exceedances, 0);
    assert!(freeze(&snapshot, features(), now + Duration::from_secs(31))
        .0
        .is_none());
}

#[test]
fn zero_calibrated_upper_is_not_counted_as_a_scheduler_usable_prediction() {
    let now = Instant::now();
    let mut snapshot = snapshot(ModelFamily::Physical, 1, 0.01, 0.005, now);
    let mut calibration = OnlineCalibrator::new(
        CalibrationConfig {
            max_domains: 1,
            buckets_per_domain: 1,
            window_samples: 4,
            min_samples: 2,
            ttl: Duration::from_secs(10),
            quantile: 0.95,
        },
        1,
    )
    .unwrap();
    // A signed same-domain residual from larger work can exceed the current
    // query's smaller base magnitude. No zero-duration observation is needed.
    for _ in 0..2 {
        assert!(calibration.observe(1, features().domain(), 0, 0.005, 0.02, now));
    }
    snapshot.domains[0].calibration = calibration.snapshot(now);
    let (base, unknown) = freeze(&snapshot, features(), now);
    assert!(base.is_some(), "raw center error remains measurable");
    assert_eq!(base.unwrap().upper, None);
    assert_eq!(unknown, Some(CostUnknownReason::UnsupportedDomain));
}

#[test]
fn invalid_or_late_tickets_do_not_turn_completed_labels_into_prediction_evidence() {
    let now = Instant::now();
    let snapshot = snapshot(ModelFamily::Physical, 1, 0.01, 0.03, now);
    let telemetry = Arc::new(Telemetry::new(ModelFamily::Physical, 0.95));
    let mut sample = summary(now, 40);
    sample.started_at = now - Duration::from_millis(1);
    completion(&snapshot, &telemetry, now, true).record_at(&sample, sample.completed_at);
    sample = summary(now, 40);
    sample.participants[0].context_tokens = 5;
    completion(&snapshot, &telemetry, now, true).record_at(&sample, sample.completed_at);
    sample = summary(now, 40);
    sample.eligibility = ExecutionAttemptEligibility::Interrupted;
    completion(&snapshot, &telemetry, now, true).record_at(&sample, sample.completed_at);
    let stats = telemetry.stats();
    let d = &stats.domains[index(features().domain())];
    assert_eq!(
        (d.completed, d.mismatched_or_noncausal, d.excluded),
        (3, 2, 1)
    );
    assert_eq!((d.eligible, d.base_known, d.calibrated_known), (0, 0, 0));
}

#[test]
fn unknown_and_worker_delay_are_separate_from_error_samples_and_direct_producers() {
    let now = Instant::now();
    let (source, _) = Source::new(
        CostQueryLimits::default(),
        ExecutionIdentity::new("cold".into()).unwrap(),
        vec![],
    );
    let telemetry = Arc::new(Telemetry::new(ModelFamily::Physical, 0.95));
    let snapshot = source.load_typed();
    let sample = summary(now, 20);
    completion(&snapshot, &telemetry, now, true).record_at(&sample, sample.completed_at);
    telemetry.feedback(
        features(),
        sample.completed_at,
        sample.completed_at + Duration::from_millis(7),
    );
    let observer = Observer {
        source: Arc::new(source),
        target: Arc::new(Sink(true)),
        telemetry: telemetry.clone(),
    };
    observer.try_record(&sample);
    let stats = telemetry.stats();
    assert_eq!(stats.unprepared_completions, 1);
    let d = &stats.domains[index(features().domain())];
    assert_eq!(
        (d.eligible, d.no_coverage, d.base_known, d.calibrated_known),
        (1, 1, 0, 0)
    );
    assert_eq!(
        (
            d.feedback_delay_samples,
            d.feedback_delay_total_ns,
            d.feedback_delay_max_ns
        ),
        (1, 7_000_000, 7_000_000)
    );
}
