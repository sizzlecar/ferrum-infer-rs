use super::*;
use ferrum_execution::{
    execution_attempt::{AttemptPrefixRestore, AttemptWorkKind},
    token_execution::ExecutionRange,
};
use ferrum_scheduler::cost_prediction::CostWork;

fn identity() -> ExecutionIdentity {
    ExecutionIdentity::new(
        "fixture-source:revision-1/device:cpu/precision:f32/executor:single".into(),
    )
    .unwrap()
}

fn prefill(tokens: usize) -> WorkFeatures {
    WorkFeatures::extract(
        &[CostWork {
            kind: AttemptWorkKind::Prefill,
            requested: ExecutionRange::new(0, tokens).unwrap(),
            context_tokens: tokens,
            prefix_restore: AttemptPrefixRestore::NoneObserved,
            internally_reused_tokens: None,
        }],
        32,
    )
    .unwrap()
}

fn sample(tokens: usize, seconds: f64) -> TrainingSample {
    TrainingSample {
        features: prefill(tokens),
        service_seconds: seconds,
    }
}

fn compact_budget() -> TrainingBudget {
    TrainingBudget {
        max_trees: 16,
        max_depth: 2,
        split_candidates: 16,
        learning_rate: 0.5,
        ..TrainingBudget::default()
    }
}

#[test]
fn features_use_pre_execution_causal_work_and_preserve_unknown_reuse() {
    let work = CostWork {
        kind: AttemptWorkKind::Prefill,
        requested: ExecutionRange::new(8, 4).unwrap(),
        context_tokens: 20,
        prefix_restore: AttemptPrefixRestore::Restored { tokens: 8 },
        internally_reused_tokens: None,
    };
    let features = WorkFeatures::extract(&[work], 1).unwrap();
    assert_eq!(
        features.values(),
        &[1.0, 4.0, 0.0, 42.0, 8.0, 0.0, 1.0, 1.0, 20.0]
    );
    let known = WorkFeatures::extract(
        &[CostWork {
            internally_reused_tokens: Some(0),
            ..work
        }],
        1,
    )
    .unwrap();
    assert_ne!(features.domain(), known.domain());
    assert_ne!(features.values(), known.values());
    let decode = CostWork {
        kind: AttemptWorkKind::Decode,
        requested: ExecutionRange::new(19, 1).unwrap(),
        context_tokens: 20,
        prefix_restore: AttemptPrefixRestore::NoneObserved,
        internally_reused_tokens: None,
    };
    assert_eq!(
        WorkFeatures::extract(&[decode], 1).unwrap().values()[2],
        20.0
    );
    assert_eq!(
        WorkFeatures::extract(&[work, decode], 2)
            .unwrap()
            .domain()
            .phase,
        Phase::Mixed
    );
    assert!(matches!(
        WorkFeatures::extract(&[work, decode], 1),
        Err(ModelError::ResourceLimit(_))
    ));
    assert!(WorkFeatures::extract(
        &[CostWork {
            context_tokens: 19,
            ..decode
        }],
        1
    )
    .is_err());
    assert!(WorkFeatures::extract(
        &[CostWork {
            prefix_restore: AttemptPrefixRestore::Restored { tokens: 9 },
            ..work
        }],
        1
    )
    .is_err());
}

#[test]
fn physical_recovers_nonnegative_work_cost_and_rejects_extrapolation() {
    let cost = |n: usize| 0.002 + n as f64 * 0.0002 + (n * (n + 1)) as f64 * 0.0000005;
    let training: Vec<_> = (1..=128).map(|n| sample(n, cost(n))).collect();
    let validation: Vec<_> = (1..=128).step_by(3).map(|n| sample(n, cost(n))).collect();
    let fit = FittedModel::train(
        ModelFamily::Physical,
        identity(),
        &training,
        &validation,
        &compact_budget(),
    )
    .unwrap();
    for n in [1, 16, 64, 128] {
        let prediction = fit.predict(&identity(), &prefill(n)).unwrap();
        assert!(
            (prediction.central_seconds - cost(n)).abs() < 0.0005,
            "n={n}, {prediction:?}"
        );
        assert_eq!(prediction.central_seconds, prediction.risk_base_seconds);
    }
    assert_eq!(
        fit.predict(&identity(), &prefill(129)),
        Err(ModelError::UnsupportedDomain)
    );
    assert_eq!(
        fit.predict(
            &ExecutionIdentity::new("other-precision".into()).unwrap(),
            &prefill(16)
        ),
        Err(ModelError::UnsupportedDomain)
    );
    let mut previous = 0.0;
    for n in 1..=128 {
        let next = fit
            .predict(&identity(), &prefill(n))
            .unwrap()
            .central_seconds;
        assert!(next >= previous);
        previous = next;
    }
}

#[test]
fn squared_center_and_pinball_risk_are_separate_objectives() {
    let training: Vec<_> = (0..100)
        .map(|i| sample(8, if i < 90 { 1.0 } else { 9.0 }))
        .collect();
    let validation: Vec<_> = (0..100)
        .map(|i| sample(8, if i % 10 == 0 { 9.0 } else { 1.0 }))
        .collect();
    let fit = FittedModel::train(
        ModelFamily::Gbdt,
        identity(),
        &training,
        &validation,
        &compact_budget(),
    )
    .unwrap();
    let prediction = fit.predict(&identity(), &prefill(8)).unwrap();
    assert!((prediction.central_seconds - 1.8).abs() < 1e-12);
    assert_eq!(prediction.risk_base_seconds, 9.0);
    assert_eq!(fit.risk_quantile(), Some(0.95));
}

#[test]
fn boosted_trees_learn_nonlinear_cost_and_training_is_deterministic() {
    let budget = compact_budget();
    let cost = |n| if n <= 16 { 0.002 } else { 0.02 };
    let training: Vec<_> = (1..=32)
        .cycle()
        .take(128)
        .map(|n| sample(n, cost(n)))
        .collect();
    let validation: Vec<_> = (1..=32).map(|n| sample(n, cost(n))).collect();
    let fit = FittedModel::train(
        ModelFamily::Gbdt,
        identity(),
        &training,
        &validation,
        &budget,
    )
    .unwrap();
    let repeated = FittedModel::train(
        ModelFamily::Gbdt,
        identity(),
        &training,
        &validation,
        &budget,
    )
    .unwrap();
    assert_eq!(
        fit.to_json(&budget).unwrap(),
        repeated.to_json(&budget).unwrap()
    );
    for n in [1, 16, 17, 32] {
        let prediction = fit.predict(&identity(), &prefill(n)).unwrap();
        assert!(
            (prediction.central_seconds - cost(n)).abs() < 0.0001,
            "{n}: {prediction:?}"
        );
        assert!(
            (prediction.risk_base_seconds - cost(n)).abs() < 0.0001,
            "{n}: {prediction:?}"
        );
    }
}

#[test]
fn mixed_invocations_and_reuse_domains_require_their_own_evidence() {
    let train = [sample(1, 0.1), sample(8, 0.2)];
    let fit = FittedModel::train(
        ModelFamily::Physical,
        identity(),
        &train,
        &train,
        &compact_budget(),
    )
    .unwrap();
    let decode = CostWork {
        kind: AttemptWorkKind::Decode,
        requested: ExecutionRange::new(7, 1).unwrap(),
        context_tokens: 8,
        prefix_restore: AttemptPrefixRestore::NoneObserved,
        internally_reused_tokens: None,
    };
    let mixed = WorkFeatures::extract(
        &[
            decode,
            CostWork {
                kind: AttemptWorkKind::Prefill,
                requested: ExecutionRange::new(0, 8).unwrap(),
                ..decode
            },
        ],
        2,
    )
    .unwrap();
    assert_eq!(
        fit.predict(&identity(), &mixed),
        Err(ModelError::UnsupportedDomain)
    );
    let known_reuse = WorkFeatures::extract(
        &[CostWork {
            kind: AttemptWorkKind::Prefill,
            requested: ExecutionRange::new(0, 8).unwrap(),
            internally_reused_tokens: Some(0),
            ..decode
        }],
        1,
    )
    .unwrap();
    assert_eq!(
        fit.predict(&identity(), &known_reuse),
        Err(ModelError::UnsupportedDomain)
    );
}

#[test]
fn training_rejects_empty_nonfinite_and_resource_overruns() {
    let train = [sample(1, 0.1), sample(8, 0.2)];
    let fit = |samples: &[TrainingSample], budget: &TrainingBudget| {
        FittedModel::train(ModelFamily::Physical, identity(), samples, &train, budget)
    };
    assert!(fit(&[], &compact_budget()).is_err());
    assert!(fit(&[sample(1, f64::NAN), sample(8, 0.2)], &compact_budget()).is_err());
    assert!(fit(
        &[sample(1, f64::INFINITY), sample(8, 0.2)],
        &compact_budget()
    )
    .is_err());
    assert!(matches!(
        fit(
            &train,
            &TrainingBudget {
                max_samples: 3,
                ..compact_budget()
            }
        ),
        Err(ModelError::ResourceLimit("training samples"))
    ));
    assert!(matches!(
        fit(
            &train,
            &TrainingBudget {
                max_owned_bytes: 1,
                ..compact_budget()
            }
        ),
        Err(ModelError::ResourceLimit("training allocation"))
    ));
    assert!(matches!(
        fit(
            &train,
            &TrainingBudget {
                max_work_units: 1,
                ..compact_budget()
            }
        ),
        Err(ModelError::ResourceLimit("training work"))
    ));
    assert!(TrainingBudget {
        risk_quantile: f64::NAN,
        ..compact_budget()
    }
    .validate()
    .is_err());
    assert!(TrainingBudget {
        risk_quantile: 0.0,
        ..compact_budget()
    }
    .validate()
    .is_err());
    assert!(TrainingBudget {
        max_depth: usize::MAX,
        ..compact_budget()
    }
    .owned_bytes_bound(2)
    .is_err());
    assert!(TrainingBudget::default()
        .owned_bytes_bound(usize::MAX)
        .is_err());
}

#[test]
fn persistence_roundtrip_checks_identity_schema_scales_and_tree_graphs() {
    let budget = compact_budget();
    let train: Vec<_> = (1..=32)
        .map(|n| sample(n, if n <= 16 { 1.0 } else { 9.0 }))
        .collect();
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        let fit = FittedModel::train(family, identity(), &train, &train, &budget).unwrap();
        let bytes = fit.to_json(&budget).unwrap();
        let restored = FittedModel::from_json(&bytes, &identity(), &budget).unwrap();
        for n in [1, 16, 17, 32] {
            let expected = fit.predict(&identity(), &prefill(n)).unwrap();
            let actual = restored.predict(&identity(), &prefill(n)).unwrap();
            assert!((actual.central_seconds - expected.central_seconds).abs() < 1e-12);
            assert!((actual.risk_base_seconds - expected.risk_base_seconds).abs() < 1e-12);
        }
        assert!(matches!(
            FittedModel::from_json(
                &bytes,
                &ExecutionIdentity::new("wrong-model".into()).unwrap(),
                &budget
            ),
            Err(ModelError::UnsupportedDomain)
        ));
        let mut invalid: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        invalid["schema"] = 0.into();
        assert!(FittedModel::from_json(
            &serde_json::to_vec(&invalid).unwrap(),
            &identity(),
            &budget
        )
        .is_err());
        invalid = serde_json::from_slice(&bytes).unwrap();
        invalid["scales"][0] = 0.into();
        assert!(FittedModel::from_json(
            &serde_json::to_vec(&invalid).unwrap(),
            &identity(),
            &budget
        )
        .is_err());
        if family == ModelFamily::Gbdt {
            invalid = serde_json::from_slice(&bytes).unwrap();
            invalid["predictor"]["Gbdt"]["center"]["trees"][0]["nodes"][0]["split"]["left"] =
                0.into();
            assert!(FittedModel::from_json(
                &serde_json::to_vec(&invalid).unwrap(),
                &identity(),
                &budget
            )
            .is_err());
            invalid = serde_json::from_slice(&bytes).unwrap();
            let tree = invalid["predictor"]["Gbdt"]["center"]["trees"][0].clone();
            invalid["predictor"]["Gbdt"]["center"]["trees"] = vec![tree; 33].into();
            assert!(FittedModel::from_json(
                &serde_json::to_vec(&invalid).unwrap(),
                &identity(),
                &budget
            )
            .is_err());
        }
    }
}

#[test]
fn complete_bounded_execution_identity_survives_persistence() {
    let identity = ExecutionIdentity::new("x".repeat(64 * 1024)).unwrap();
    assert!(ExecutionIdentity::new("x".repeat(64 * 1024 + 1)).is_err());
    for invalid in [String::new(), "x".repeat(64 * 1024 + 1), "a\0b".into()] {
        assert!(serde_json::from_str::<ExecutionIdentity>(
            &serde_json::to_string(&invalid).unwrap()
        )
        .is_err());
    }
    let train = [sample(1, 0.001), sample(8, 0.002)];
    let budget = compact_budget();
    let model = FittedModel::train(
        ModelFamily::Physical,
        identity.clone(),
        &train,
        &train,
        &budget,
    )
    .unwrap();
    let bytes = model.to_json(&budget).unwrap();
    assert!(FittedModel::from_json(&bytes, &identity, &budget).is_ok());
}
