use super::*;

fn width_sample(width: usize, seconds: f64) -> TrainingSample {
    TrainingSample {
        features: WorkFeatures::extract(&vec![work(); width], 256).unwrap(),
        service_seconds: seconds,
    }
}

#[test]
fn width_history_latest_same_width_fit_can_forget_fresh_observed_context() {
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        let config = LearningConfig {
            training_history: super::super::super::TrainingHistoryPolicy::WidthStratified {
                samples_per_width: 2,
            },
            window_samples: 16,
            model_ttl_ms: 1_000,
            ..config()
        };
        let shared = shared(&config);
        let mut learner = Learner::new(config, identity(), family);
        let start = Instant::now();
        let completed = |context: usize, sequence: u64| {
            let at = start + Duration::from_millis(sequence);
            ExecutionAttemptSummary {
                started_at: at - Duration::from_millis(1),
                completed_at: at,
                eligibility: ExecutionAttemptEligibility::Completed,
                participants: vec![ExecutionAttemptParticipant {
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
                }],
            }
        };
        for sequence in 1..=8 {
            let observation = completed(80, sequence);
            learner.apply(&shared, &observation, observation.completed_at);
        }
        assert_eq!(shared.stats().model_fits, 1);
        for sequence in 9..=15 {
            let observation = completed(16, sequence);
            learner.apply(&shared, &observation, observation.completed_at);
        }
        let old = learner.snapshot(start + Duration::from_millis(15)).unwrap();
        let old_revision = old.domains[0].model_revision;
        let original = completed(80, 8);
        let original_work: Vec<_> = original.participants.iter().map(sample_work).collect();
        let original_query = CostQuery {
            work: &original_work,
            work_is_exact: true,
            execution: CostExecution::SingleInvocation,
        };

        let observation = completed(16, 16);
        let now = observation.completed_at;
        learner.apply(&shared, &observation, now);
        let new = learner.snapshot(now).unwrap();
        assert_eq!(shared.stats().model_fits, 2);
        assert!(new.domains[0].model_revision > old_revision);
        assert!(now < old.domains[0].model_expires_at);
        assert!(now < new.domains[0].model_expires_at);
        // This is selection loss, not age expiration or a missing old fact:
        // every original context-80 observation still occupies the main window.
        let domain = &learner.domains[0];
        assert_eq!(domain.window.len(), 16);
        assert_eq!(
            domain
                .window
                .iter()
                .filter(|item| item.sample.features.values()[8] == 80.0)
                .count(),
            8
        );
        assert_eq!(domain.active.as_ref().unwrap().fitting_through, 14);
        assert!(matches!(
            old.predict(&original_query, now),
            CostPrediction::Known { .. }
        ));
        assert_eq!(
            new.predict(&original_query, now),
            CostPrediction::Unknown(CostUnknownReason::UnsupportedWithDetail(
                CostUnsupportedDetail::OutsideObservedSupport
            ))
        );
        let recent_work: Vec<_> = observation.participants.iter().map(sample_work).collect();
        assert!(matches!(
            new.predict(
                &CostQuery {
                    work: &recent_work,
                    work_is_exact: true,
                    execution: CostExecution::SingleInvocation,
                },
                now
            ),
            CostPrediction::Known { .. }
        ));
    }
}

#[test]
fn width_history_refits_both_families_from_real_support_without_renewing_old_evidence() {
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        for policy in [
            super::super::super::TrainingHistoryPolicy::Recent,
            super::super::super::TrainingHistoryPolicy::WidthStratified {
                samples_per_width: 2,
            },
        ] {
            let config = LearningConfig {
                training_history: policy,
                window_samples: 8,
                retrain_every: 16,
                model_ttl_ms: 1_000,
                ..config()
            };
            let shared = shared(&config);
            let mut learner = Learner::new(config, identity(), family);
            let start = Instant::now();
            for sequence in 1..=16 {
                let at = start + Duration::from_millis(sequence);
                learner.observe(
                    &shared,
                    width_sample(
                        if sequence <= 2 { 1 } else { 4 },
                        if sequence > 14 { 0.1 } else { 0.002 },
                    ),
                    at,
                    at,
                );
            }
            assert_eq!(shared.stats().model_fits, 1);
            let active = learner.domains[0].active.as_ref().unwrap();
            let stratified = policy != super::super::super::TrainingHistoryPolicy::Recent;
            assert_eq!(
                active
                    .model
                    .predict(&identity(), &width_sample(1, 0.002).features)
                    .is_ok(),
                stratified
            );
            assert!(active
                .model
                .predict(&identity(), &width_sample(8, 0.002).features)
                .is_err());
            assert_eq!(
                active.model_expires_at,
                start + Duration::from_millis(if stratified { 1_001 } else { 1_012 })
            );
            let old = learner.snapshot(start + Duration::from_millis(16)).unwrap();
            let old_model = Arc::clone(&old.domains[0].model);
            // The second fit is later than the old singleton's original TTL.
            // It must rebuild support and residuals rather than unioning either.
            for sequence in 1..=16 {
                let at = start + Duration::from_millis(1_001 + sequence);
                learner.observe(&shared, width_sample(4, 0.003), at, at);
            }
            assert_eq!(shared.stats().model_fits, 2);
            let active = learner.domains[0].active.as_ref().unwrap();
            assert!(active
                .model
                .predict(&identity(), &width_sample(1, 0.003).features)
                .is_err());
            assert_eq!(active.fitting_through, 30);
            let now = start + Duration::from_millis(1_017);
            let base = active
                .model
                .predict(&identity(), &width_sample(4, 0.003).features)
                .unwrap();
            let correction = active
                .calibration
                .snapshot(now)
                .correction(
                    active.model_revision,
                    &learner.domains[0].key,
                    u32::MAX,
                    now,
                )
                .unwrap();
            assert!((base.risk_base_seconds + correction.signed_seconds - 0.003).abs() < 1e-9);
            assert!(Arc::ptr_eq(&old.domains[0].model, &old_model));
            assert_eq!(
                old_model
                    .predict(&identity(), &width_sample(1, 0.002).features)
                    .is_ok(),
                stratified
            );
        }
    }
}

#[test]
fn width_history_isolated_by_actual_work_domain_and_new_executor_owner() {
    let config = LearningConfig {
        training_history: super::super::super::TrainingHistoryPolicy::WidthStratified {
            samples_per_width: 2,
        },
        window_samples: 8,
        retrain_every: 16,
        ..config()
    };
    let shared = shared(&config);
    let mut learner = Learner::new(config.clone(), identity(), ModelFamily::Physical);
    let start = Instant::now();
    for sequence in 1..=16 {
        let at = start + Duration::from_millis(sequence);
        learner.observe(
            &shared,
            width_sample(if sequence <= 2 { 1 } else { 4 }, 0.002),
            at,
            at,
        );
        let work = CostWork {
            kind: AttemptWorkKind::Prefill,
            requested: ExecutionRange::new(0, 4).unwrap(),
            ..work()
        };
        learner.observe(
            &shared,
            TrainingSample {
                features: WorkFeatures::extract(&[work; 4], 256).unwrap(),
                service_seconds: 0.003,
            },
            at,
            at,
        );
    }
    assert_eq!(learner.domains.len(), 2);
    let prefill = CostWork {
        kind: AttemptWorkKind::Prefill,
        requested: ExecutionRange::new(0, 4).unwrap(),
        ..work()
    };
    let singleton_prefill = WorkFeatures::extract(&[prefill], 256).unwrap();
    assert!(learner.domains.iter().all(|domain| domain
        .active
        .as_ref()
        .unwrap()
        .model
        .predict(&identity(), &singleton_prefill)
        .is_err()));
    let other = Learner::new(
        config,
        ExecutionIdentity::new("different/actual/executor".into()).unwrap(),
        ModelFamily::Physical,
    );
    assert!(other.domains.is_empty());
}
