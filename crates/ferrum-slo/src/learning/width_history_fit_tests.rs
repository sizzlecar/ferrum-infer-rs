use super::*;

fn width_sample(width: usize, seconds: f64) -> TrainingSample {
    TrainingSample {
        features: WorkFeatures::extract(&vec![work(); width], 256).unwrap(),
        service_seconds: seconds,
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
