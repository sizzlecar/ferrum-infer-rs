use super::*;

fn completed(
    start: Instant,
    elapsed_ms: u64,
    width: usize,
    context: usize,
    service_ms: u64,
) -> ExecutionAttemptSummary {
    let at = start + Duration::from_millis(elapsed_ms);
    ExecutionAttemptSummary {
        started_at: at - Duration::from_millis(service_ms),
        completed_at: at,
        eligibility: ExecutionAttemptEligibility::Completed,
        participants: (0..width)
            .map(|_| ExecutionAttemptParticipant {
                request_id: RequestId::new(),
                kind: AttemptWorkKind::Decode,
                requested: ExecutionRange::new(context - 1, 1).unwrap(),
                context_tokens: context,
                prefix_restore: AttemptPrefixRestore::NoneObserved,
                internally_reused_tokens: None,
                product_failed: false,
                outcome: AttemptParticipantOutcome::Completed {
                    work: ExecutedTokenWork::Decode {
                        position: context - 1,
                    },
                    frontier_committed: true,
                },
            })
            .collect(),
    }
}

fn predict_context(snapshot: &snapshot::Snapshot, context: usize, now: Instant) -> CostPrediction {
    snapshot.predict(
        &CostQuery {
            work: &[CostWork {
                requested: ExecutionRange::new(context - 1, 1).unwrap(),
                context_tokens: context,
                ..work()
            }],
            work_is_exact: true,
            execution: CostExecution::SingleInvocation,
        },
        now,
    )
}

fn outside() -> CostPrediction {
    CostPrediction::Unknown(CostUnknownReason::UnsupportedWithDetail(
        CostUnsupportedDetail::OutsideObservedSupport,
    ))
}

#[test]
fn extrema_training_keeps_real_context_across_refits_without_replaying_calibration() {
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        let config = LearningConfig {
            training_history: super::super::super::TrainingHistoryPolicy::WidthAndExtrema {
                samples_per_width: 2,
            },
            window_samples: 16,
            model_ttl_ms: 1_000,
            ..config()
        };
        config.validate().unwrap();
        let shared = shared(&config);
        let mut learner = Learner::new(config, identity(), family);
        let start = Instant::now();
        for sequence in 1..=8 {
            let observation = completed(start, sequence, 1, 80, 10);
            learner.apply(&shared, &observation, observation.completed_at);
        }
        let old = learner.snapshot(start + Duration::from_millis(8)).unwrap();
        let old_serialized = serde_json::to_value(&*old.domains[0].model).unwrap();
        for sequence in 9..=16 {
            let observation = completed(start, sequence, 1, 16, if sequence < 15 { 2 } else { 50 });
            learner.apply(&shared, &observation, observation.completed_at);
        }
        assert_eq!(shared.stats().model_fits, 2);
        let now = start + Duration::from_millis(16);
        let new = learner.snapshot(now).unwrap();
        assert!(new.domains[0].model_revision > old.domains[0].model_revision);
        assert_eq!(
            serde_json::to_value(&*old.domains[0].model).unwrap(),
            old_serialized
        );
        assert!(matches!(
            predict_context(&old, 80, now),
            CostPrediction::Known { .. }
        ));
        assert!(matches!(
            predict_context(&new, 80, now),
            CostPrediction::Known { .. }
        ));
        assert_eq!(predict_context(&new, 81, now), outside());
        let domain = &learner.domains[0];
        let selected = super::super::super::width_history::fit_observations(
            &learner.config,
            &domain.window,
            &domain.width_history,
            now,
        )
        .unwrap();
        assert_eq!(
            selected
                .iter()
                .map(|record| record.sequence)
                .collect::<Vec<_>>(),
            vec![8, 10, 11, 12, 13, 14, 15, 16]
        );
        let active = domain.active.as_ref().unwrap();
        assert_eq!(active.fitting_through, 14);
        // Original context80 at sequence8, not this fit or later observations,
        // determines the model's evidence expiry.
        assert_eq!(
            active.model_expires_at,
            start + Duration::from_millis(1_008)
        );
        let features = WorkFeatures::extract(
            &[CostWork {
                requested: ExecutionRange::new(15, 1).unwrap(),
                context_tokens: 16,
                ..work()
            }],
            1,
        )
        .unwrap();
        let base = active.model.predict(&identity(), &features).unwrap();
        let correction = active
            .calibration
            .snapshot(now)
            .correction(active.model_revision, &domain.key, u32::MAX, now)
            .unwrap();
        assert_eq!(correction.samples, 2);
        assert!((base.risk_base_seconds + correction.signed_seconds - 0.050).abs() < 1e-9);

        // Continue past eviction from both the ordinary window and the newest
        // two same-width representatives. Only the causal reservoir retains80.
        for sequence in 17..=32 {
            let observation = completed(start, sequence, 1, 16, 2);
            learner.apply(&shared, &observation, observation.completed_at);
        }
        assert!(learner.domains[0]
            .window
            .iter()
            .all(|record| record.sample.features.values()[8] == 16.0));
        let retained_at = start + Duration::from_millis(32);
        let retained = learner.snapshot(retained_at).unwrap();
        assert!(matches!(
            predict_context(&retained, 80, retained_at),
            CostPrediction::Known { .. }
        ));
        assert_eq!(
            retained.domains[0].model_expires_at,
            start + Duration::from_millis(1_008)
        );

        // Once the original extreme expires, only fresh actual context16 facts
        // can qualify later models. No old box or residual is transplanted.
        for sequence in 33..=48 {
            let observation = completed(start, 1_008 + sequence - 32, 1, 16, 2);
            learner.apply(&shared, &observation, observation.completed_at);
        }
        let later = start + Duration::from_millis(1_024);
        let latest = learner.snapshot(later).unwrap();
        assert_eq!(predict_context(&latest, 80, later), outside());
        assert!(matches!(
            predict_context(&latest, 16, later),
            CostPrediction::Known { .. }
        ));
        assert_eq!(
            predict_context(&new, 80, later),
            CostPrediction::Unknown(CostUnknownReason::Stale)
        );
    }
}

#[test]
fn extrema_over_budget_keeps_the_complete_previous_model_revision() {
    let config = LearningConfig {
        training_history: super::super::super::TrainingHistoryPolicy::WidthAndExtrema {
            samples_per_width: 2,
        },
        training_samples: 2,
        validation_samples: 1,
        calibration_samples: 1,
        min_calibration_samples: 1,
        retrain_every: 4,
        ..config()
    };
    let shared = shared(&config);
    let mut learner = Learner::new(config, identity(), ModelFamily::Physical);
    let start = Instant::now();
    for sequence in 1..=4 {
        let observation = completed(start, sequence, 1, 16, 2);
        learner.apply(&shared, &observation, observation.completed_at);
    }
    assert_eq!(shared.stats().model_fits, 1);
    learner.config.retrain_every = 5;
    for (index, (width, context)) in [(1, 100), (4, 30), (2, 20), (3, 25)]
        .into_iter()
        .enumerate()
    {
        let observation = completed(start, index as u64 + 5, width, context, 2);
        learner.apply(&shared, &observation, observation.completed_at);
    }
    let old = learner.snapshot(start + Duration::from_millis(8)).unwrap();
    let final_observation = completed(start, 9, 3, 25, 2);
    learner.apply(&shared, &final_observation, final_observation.completed_at);
    let new = learner.snapshot(final_observation.completed_at).unwrap();
    assert_eq!(shared.stats().model_fits, 1);
    assert_eq!(shared.stats().training_failures, 1);
    assert_eq!(new.domains[0].model_revision, old.domains[0].model_revision);
    assert_eq!(
        new.domains[0].model_expires_at,
        old.domains[0].model_expires_at
    );
    assert!(Arc::ptr_eq(&new.domains[0].model, &old.domains[0].model));
    assert!(new.domains[0]
        .calibration
        .correction(
            new.domains[0].model_revision,
            &learner.domains[0].key,
            u32::MAX,
            final_observation.completed_at
        )
        .is_some());
}

#[test]
fn extrema_history_does_not_retain_ineligible_late_or_future_attempts() {
    let config = LearningConfig {
        training_history: super::super::super::TrainingHistoryPolicy::WidthAndExtrema {
            samples_per_width: 2,
        },
        ..config()
    };
    let shared = shared(&config);
    let mut learner = Learner::new(config, identity(), ModelFamily::Physical);
    let start = Instant::now();
    for sequence in 1..=8 {
        let observation = completed(start, sequence, 1, 80, 2);
        learner.apply(&shared, &observation, observation.completed_at);
    }
    let before = shared.stats().trained_samples;
    let mut interrupted = completed(start, 9, 1, 160, 2);
    interrupted.eligibility = ExecutionAttemptEligibility::Interrupted;
    assert!(!learner.apply(&shared, &interrupted, interrupted.completed_at));
    let mut uncommitted = completed(start, 9, 1, 160, 2);
    if let AttemptParticipantOutcome::Completed {
        frontier_committed, ..
    } = &mut uncommitted.participants[0].outcome
    {
        *frontier_committed = false;
    }
    assert!(!learner.apply(&shared, &uncommitted, uncommitted.completed_at));
    let late = completed(start, 7, 1, 160, 2);
    assert!(!learner.apply(&shared, &late, start + Duration::from_millis(9)));
    let future = completed(start, 10, 1, 160, 2);
    assert!(!learner.apply(&shared, &future, start + Duration::from_millis(9)));
    assert_eq!(shared.stats().trained_samples, before);
    for sequence in 9..=24 {
        let observation = completed(start, sequence, 1, 16, 2);
        learner.apply(&shared, &observation, observation.completed_at);
    }
    let now = start + Duration::from_millis(24);
    let snapshot = learner.snapshot(now).unwrap();
    assert!(matches!(
        predict_context(&snapshot, 80, now),
        CostPrediction::Known { .. }
    ));
    assert_eq!(predict_context(&snapshot, 160, now), outside());
}
