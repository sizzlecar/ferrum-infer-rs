use super::*;
use crate::learning::LearningEstimator;
use ferrum_execution::{
    execution_attempt::*,
    token_execution::{ExecutedTokenWork, ExecutionRange},
};
use ferrum_scheduler::cost_prediction::*;
use ferrum_types::RequestId;

#[path = "online_qualification_tests.rs"]
mod online_qualification;

#[path = "width_history_fit_tests.rs"]
mod width_history_fit;

#[path = "prediction_rejection_tests.rs"]
mod prediction_rejection;

#[path = "rejection_witness_tests.rs"]
mod rejection_witness;

fn config() -> LearningConfig {
    LearningConfig {
        training_samples: 4,
        validation_samples: 2,
        calibration_samples: 2,
        min_calibration_samples: 2,
        retrain_every: 8,
        window_samples: 16,
        training: crate::model::TrainingBudget {
            max_trees: 4,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn identity() -> ExecutionIdentity {
    ExecutionIdentity::new("cpu/f32/fixture/source-v1".into()).unwrap()
}

fn work() -> CostWork {
    CostWork {
        kind: AttemptWorkKind::Decode,
        requested: ExecutionRange::new(3, 1).unwrap(),
        context_tokens: 4,
        prefix_restore: AttemptPrefixRestore::NoneObserved,
        internally_reused_tokens: None,
    }
}

fn sample(seconds: f64) -> TrainingSample {
    TrainingSample {
        features: WorkFeatures::extract(&[work()], 1).unwrap(),
        service_seconds: seconds,
    }
}

fn shared(config: &LearningConfig) -> Shared {
    let (wake, _) = crossbeam_channel::bounded(1);
    Shared::new(config.collection, wake)
}

fn prediction(snapshot: &snapshot::Snapshot, now: Instant) -> (f64, f64) {
    match snapshot.predict(
        &CostQuery {
            work: &[work()],
            work_is_exact: true,
            execution: CostExecution::SingleInvocation,
        },
        now,
    ) {
        CostPrediction::Known {
            central_service_time,
            upper_service_time,
            ..
        } => (
            central_service_time.as_secs_f64(),
            upper_service_time.as_secs_f64(),
        ),
        unknown => panic!("expected qualified prediction: {unknown:?}"),
    }
}

fn fill(
    learner: &mut Learner,
    shared: &Shared,
    start: Instant,
    fit_seconds: f64,
    calibration_seconds: f64,
) -> Instant {
    let required = learner.config.required_samples();
    let fit_end = required - learner.config.calibration_samples;
    for i in 0..required {
        let at = start + Duration::from_millis(i as u64);
        learner.observe(
            shared,
            sample(if i < fit_end {
                fit_seconds
            } else {
                calibration_seconds
            }),
            at,
            at,
        );
    }
    start + Duration::from_millis((required - 1) as u64)
}

#[test]
fn both_families_publish_disjoint_calibration_and_coherent_model_changes() {
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        let config = config();
        let shared = shared(&config);
        let mut learner = Learner::new(config, identity(), family);
        let now = Instant::now();
        let end = fill(&mut learner, &shared, now, 0.002, 0.1);
        let old = learner.snapshot(end).unwrap();
        let (center, tail) = prediction(&old, end);
        assert!(
            (center - 0.002).abs() < 1e-9,
            "calibration must not contaminate fitting"
        );
        assert!((tail - 0.1).abs() < 1e-9);
        assert_eq!(
            learner.domains[0].active.as_ref().unwrap().fitting_through,
            6
        );
        let later = end + Duration::from_millis(1);
        let end = fill(&mut learner, &shared, later, 0.001, 0.05);
        let new = learner.snapshot(end).unwrap();
        assert!(new.domains[0].model_revision > old.domains[0].model_revision);
        let (center, tail) = prediction(&new, end);
        assert!((center - 0.001).abs() < 1e-9);
        assert!(
            (tail - 0.05).abs() < 1e-9,
            "new model must recompute residuals from its own risk base"
        );
        assert!(
            (prediction(&old, end).1 - 0.1).abs() < 1e-9,
            "prior round retains immutable version"
        );
        assert_eq!(
            learner.domains[0].active.as_ref().unwrap().fitting_through,
            14
        );
        assert_eq!(shared.stats().worker_base_prediction_unknown, 8);
        assert_eq!(shared.stats().worker_base_prediction_known, 8);
        assert_eq!(shared.stats().model_fits, 2);
    }
}

#[test]
fn signed_fast_updates_precede_fitting_and_failure_keeps_previous_model() {
    let config = config();
    let shared = shared(&config);
    let mut learner = Learner::new(config, identity(), ModelFamily::Physical);
    let end = fill(&mut learner, &shared, Instant::now(), 0.002, 0.1);
    let model = Arc::clone(&learner.domains[0].active.as_ref().unwrap().model);
    learner.config.training.max_work_units = 1;
    for i in 1..=8 {
        let at = end + Duration::from_millis(i);
        assert!(learner.observe(&shared, sample(0.001), at, at));
    }
    let now = end + Duration::from_millis(8);
    let active = learner.domains[0].active.as_ref().unwrap();
    assert!(Arc::ptr_eq(&model, &active.model));
    assert_eq!(active.model_revision, 1);
    assert_eq!(shared.stats().training_failures, 1);
    let prediction = prediction(&learner.snapshot(now).unwrap(), now);
    assert!((prediction.0 - 0.002).abs() < 1e-9);
    assert!(
        (prediction.1 - 0.001).abs() < 1e-9,
        "negative residual must remain signed"
    );
}

#[test]
fn expired_calibration_and_split_work_are_explicit_unknown() {
    let config = config();
    let shared = shared(&config);
    let mut learner = Learner::new(config, identity(), ModelFamily::Physical);
    let end = fill(&mut learner, &shared, Instant::now(), 0.002, 0.002);
    let snapshot = learner.snapshot(end).unwrap();
    let query_work = [work()];
    let query = CostQuery {
        work: &query_work,
        work_is_exact: true,
        execution: CostExecution::SingleInvocation,
    };
    assert_eq!(
        snapshot.predict(&query, end + Duration::from_secs(30)),
        CostPrediction::Unknown(CostUnknownReason::Stale)
    );
    let mixed = [
        work(),
        CostWork {
            kind: AttemptWorkKind::Prefill,
            requested: ExecutionRange::new(0, 4).unwrap(),
            ..work()
        },
    ];
    assert_eq!(
        snapshot.predict(
            &CostQuery {
                work: &mixed,
                execution: CostExecution::SequentialPrefillThenDecode,
                ..query
            },
            end
        ),
        CostPrediction::Unknown(CostUnknownReason::UnsupportedWithDetail(
            CostUnsupportedDetail::ExecutionComposition,
        ))
    );
    assert!(matches!(
        snapshot.predict(
            &CostQuery {
                execution: CostExecution::SequentialPrefillThenDecode,
                ..query
            },
            end
        ),
        CostPrediction::Known { .. }
    ));
    assert_eq!(
        snapshot.predict(
            &CostQuery {
                work_is_exact: false,
                ..query
            },
            end
        ),
        CostPrediction::Unknown(CostUnknownReason::InexactWork)
    );
}

#[test]
fn old_facts_do_not_reenter_history_or_extend_model_freshness() {
    let config = config();
    let shared = shared(&config);
    let mut learner = Learner::new(config, identity(), ModelFamily::Physical);
    let end = fill(&mut learner, &shared, Instant::now(), 0.002, 0.002);
    assert!(!learner.observe(&shared, sample(9.0), end - Duration::from_secs(1), end));
    assert_eq!(learner.domains[0].seen, 8);
    assert_eq!(shared.stats().excluded_attempts, 1);
    let expires = learner.domains[0].active.as_ref().unwrap().model_expires_at;
    learner.config.retrain_every = usize::MAX;
    assert!(!learner.observe(&shared, sample(0.002), expires, expires));
    let snapshot = learner.snapshot(expires).unwrap();
    assert_eq!(
        snapshot.predict(
            &CostQuery {
                work: &[work()],
                work_is_exact: true,
                execution: CostExecution::SingleInvocation
            },
            expires
        ),
        CostPrediction::Unknown(CostUnknownReason::Stale)
    );
}

fn summary() -> ExecutionAttemptSummary {
    let completed_at = Instant::now();
    ExecutionAttemptSummary {
        started_at: completed_at - Duration::from_millis(2),
        completed_at,
        eligibility: ExecutionAttemptEligibility::Completed,
        participants: vec![ExecutionAttemptParticipant {
            request_id: RequestId::new(),
            kind: work().kind,
            requested: work().requested,
            context_tokens: 4,
            prefix_restore: AttemptPrefixRestore::NoneObserved,
            internally_reused_tokens: None,
            product_failed: false,
            outcome: AttemptParticipantOutcome::Completed {
                work: ExecutedTokenWork::Decode { position: 3 },
                frontier_committed: true,
            },
        }],
    }
}

#[tokio::test]
async fn real_observer_worker_and_shutdown_produce_predictions_for_both_families() {
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        let owner = LearningEstimator::spawn(config(), identity(), family).unwrap();
        let observer = owner.observer();
        for _ in 0..8 {
            assert_eq!(
                observer.try_record(&summary()),
                ExecutionAttemptDelivery::Recorded
            );
        }
        let exit = owner.shutdown().await.unwrap();
        assert_eq!(exit.stats.trained_samples, 8);
        assert_eq!(exit.stats.model_fits, 1);
        assert_eq!(exit.stats.feedback_delay_samples, 8);
        assert_eq!(exit.stats.queued_records, 0);
        let prediction = owner.source().load().predict(
            &CostQuery {
                work: &[work()],
                work_is_exact: true,
                execution: CostExecution::SingleInvocation,
            },
            Instant::now(),
        );
        assert!(matches!(prediction, CostPrediction::Known { .. }));
        assert_eq!(
            observer.try_record(&summary()),
            ExecutionAttemptDelivery::Dropped(ExecutionAttemptDrop::Closed)
        );
    }
}

#[tokio::test]
async fn worker_panic_joins_and_releases_collection_credits() {
    let owner = LearningEstimator::spawn_inner(
        config(),
        identity(),
        ModelFamily::Physical,
        crate::worker::Hooks {
            panic_next: Some(Arc::new(std::sync::atomic::AtomicBool::new(true))),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        owner.observer().try_record(&summary()),
        ExecutionAttemptDelivery::Recorded
    );
    assert_eq!(
        owner.shutdown().await,
        Err(crate::EstimatorError::WorkerPanicked)
    );
    assert_eq!(owner.stats().queued_records, 0);
    assert_eq!(owner.stats().queued_bytes, 0);
    assert_eq!(owner.stats().worker_failures, 1);
}

#[test]
fn config_checks_partition_memory_and_observation_query_compatibility() {
    config().validate().unwrap();
    assert!(LearningConfig {
        window_samples: 7,
        ..config()
    }
    .validate()
    .is_err());
    assert!(LearningConfig {
        max_retained_bytes: 1,
        ..config()
    }
    .validate()
    .is_err());
    assert!(LearningConfig {
        work_bucket_boundaries: vec![64, 16],
        ..config()
    }
    .validate()
    .is_err());
    let mut invalid = config();
    invalid.query.max_participants = invalid.collection.max_participants + 1;
    assert!(invalid.validate().is_err());
}

#[test]
fn chronological_decode_growth_and_jitter_qualify_only_measured_feature_envelopes() {
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        for jitter in [false, true] {
            let config = LearningConfig {
                training: crate::model::TrainingBudget {
                    max_trees: 8,
                    ..Default::default()
                },
                ..Default::default()
            };
            let shared = shared(&config);
            let mut learner = Learner::new(config, identity(), family);
            let start = Instant::now();
            let mut maximum = 0;
            for i in 0..192 {
                let context = if jitter { 32 + (i * 17 % 65) } else { 32 + i };
                maximum = maximum.max(context);
                let work = CostWork {
                    requested: ExecutionRange::new(context - 1, 1).unwrap(),
                    context_tokens: context,
                    ..work()
                };
                let at = start + Duration::from_millis(i as u64);
                learner.observe(
                    &shared,
                    TrainingSample {
                        features: WorkFeatures::extract(&[work], 1).unwrap(),
                        service_seconds: 0.001 + context as f64 * 0.000001,
                    },
                    at,
                    at,
                );
            }
            let now = start + Duration::from_millis(191);
            let snapshot = learner.snapshot(now).unwrap();
            assert_eq!(
                snapshot.domains.len(),
                1,
                "family={family:?}, jitter={jitter}"
            );
            let observed = [CostWork {
                requested: ExecutionRange::new(maximum - 1, 1).unwrap(),
                context_tokens: maximum,
                ..work()
            }];
            let query = CostQuery {
                work: &observed,
                work_is_exact: true,
                execution: CostExecution::SingleInvocation,
            };
            assert!(matches!(
                snapshot.predict(&query, now),
                CostPrediction::Known { .. }
            ));
            let future = [CostWork {
                requested: ExecutionRange::new(maximum, 1).unwrap(),
                context_tokens: maximum + 1,
                ..work()
            }];
            assert_eq!(
                snapshot.predict(
                    &CostQuery {
                        work: &future,
                        ..query
                    },
                    now
                ),
                CostPrediction::Unknown(CostUnknownReason::UnsupportedWithDetail(
                    CostUnsupportedDetail::OutsideObservedSupport,
                ))
            );
        }
    }
}

#[test]
fn scenario_api_uses_one_risk_base_and_a_complete_bounded_rank_distribution() {
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        let config = config();
        let shared = shared(&config);
        let mut learner = Learner::new(config, identity(), family);
        let end = fill(&mut learner, &shared, Instant::now(), 0.01, 0.005);
        let snapshot = learner.snapshot(end).unwrap();
        let work = [work()];
        let query = CostQuery {
            work: &work,
            work_is_exact: true,
            execution: CostExecution::SingleInvocation,
        };
        let result = snapshot.cost_scenarios(&query, end, 32).validated(end, 32);
        let CostScenarioPrediction::Known {
            scenarios,
            dependence,
            ..
        } = result
        else {
            panic!("{result:?}");
        };
        assert_eq!(dependence, CostScenarioDependence::SharedResidualRank);
        assert_eq!(scenarios.len(), 32);
        assert!((scenarios.iter().map(|s| s.weight).sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(scenarios
            .iter()
            .all(|s| s.service_time == Duration::from_millis(5)));
        assert!(matches!(
            snapshot.cost_scenarios(&query, end, 0),
            CostScenarioPrediction::Unknown(CostUnknownReason::QueryLimit)
        ));
    }
}
