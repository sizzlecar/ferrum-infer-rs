use super::*;
use ferrum_interfaces::execution_cost::{CostRowNumericFeatures, COST_NUMERIC_FEATURE_SCHEMA_V1};

fn fingerprint() -> ExecutionFingerprint {
    ExecutionFingerprint {
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }
}

fn settings() -> CostModelSettings {
    CostModelSettings {
        feature_model: CostFeatureModel::BoundedNumericV1 {
            host_history_bucket_tokens: NonZeroU32::new(64).unwrap(),
        },
        min_samples: NonZeroUsize::new(2).unwrap(),
        context_bucket_tokens: NonZeroU32::new(1024).unwrap(),
        max_sample_age_ns: NonZeroU64::new(100).unwrap(),
        drift_margin_ns: 10,
        // The numeric planning envelope must not fall to this median setting.
        residual_quantile: 0.5,
        ..Default::default()
    }
}

fn shape(contexts: &[u32], histories: &[u64], maximum: u64) -> WaveExecutionShape {
    assert_eq!(contexts.len(), histories.len());
    WaveExecutionShape {
        row_multiset_features: None,
        host_content_features: None,
        kind: WaveKind::Decode,
        path: WaveExecutionPath::PlanRuntime,
        provider_signature: [5; 32],
        output_policy_signature: [histories[0] as u8; 32],
        graph_state: WaveGraphState::Disabled,
        order: BatchOrderSemantics::Ordered,
        decode_kv_tokens: contexts.to_vec(),
        prefill_chunks: vec![],
        recurrent_state_bytes: 0,
        restore_bytes: 0,
        maintenance_bytes: 0,
        maintenance_units: 0,
        numeric_features: Some(CanonicalWaveCostFeatures {
            schema_version: COST_NUMERIC_FEATURE_SCHEMA_V1,
            output_policy_signature: [6; 32],
            rows: histories
                .iter()
                .map(|&n| CostRowNumericFeatures {
                    generated_tokens_before: n,
                    maximum_output_tokens: maximum,
                    sampling_history_tokens: n,
                    repetition_tokens: n,
                    decoded_prefix_tokens: n + 1,
                    decoded_text_bytes_bound: (n + 1) * 4,
                    decode_scratch_bytes_bound: (n + 1) * 8,
                })
                .collect(),
        }),
    }
}

fn sample(shape: WaveExecutionShape, cost: u64, at: u64) -> WaveCostObservation {
    WaveCostObservation {
        fingerprint: fingerprint(),
        actual_shape: shape,
        boundary: CostBoundary::PreparationToCommit,
        outcome: WaveObservationOutcome::Completed,
        timing: WaveTiming {
            wall_total_ns: cost,
            device_elapsed_ns: None,
            stages: Default::default(),
        },
        observed_at_ns: at,
    }
}

fn predict(snapshot: &CostModelSnapshot, shape: &WaveExecutionShape, at: u64) -> CostPrediction {
    snapshot.predict(&fingerprint(), shape, CostBoundary::PreparationToCommit, at)
}

#[test]
fn numeric_work_reuses_joint_support_across_history_and_capacity_without_changing_exact_hash() {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    trainer
        .observe(sample(shape(&[100], &[10], 100), 100, 1))
        .unwrap();
    trainer
        .observe(sample(shape(&[120], &[20], 200), 500, 2))
        .unwrap();
    let snapshot = trainer.publish(2).unwrap();
    assert_eq!(snapshot.bucket_count(), 1);
    let query = shape(&[110], &[15], 777);
    let CostPrediction::Known(result) = predict(&snapshot, &query, 2) else {
        panic!("jointly supported query")
    };
    assert_eq!(result.sample_count, 2);
    assert_eq!(result.typical_ns, 100);
    assert_eq!(result.planning_ns, 510);
    assert_eq!(result.empirical_envelope_ns, Some(500));
    assert_eq!(query.output_policy_signature, [15; 32]);
    assert_eq!(trainer.retained_shape_rows(), 4);
}

#[test]
fn numeric_support_does_not_combine_independent_peer_extrema() {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    trainer
        .observe(sample(shape(&[100, 120], &[10, 20], 100), 100, 1))
        .unwrap();
    trainer
        .observe(sample(shape(&[120, 100], &[20, 10], 100), 200, 2))
        .unwrap();
    let query = shape(&[110, 110], &[15, 15], 100);
    let old = trainer.publish(2).unwrap();
    assert_eq!(
        predict(&old, &query, 2),
        CostPrediction::Unknown(CostUnknownReason::OutsideJointNumericSupport)
    );
    trainer
        .observe(sample(shape(&[120, 120], &[20, 20], 100), 300, 3))
        .unwrap();
    assert!(matches!(
        predict(&trainer.publish(3).unwrap(), &query, 3),
        CostPrediction::Known(_)
    ));
    // A later publication cannot add evidence to an already-held snapshot.
    assert_eq!(
        predict(&old, &query, 3),
        CostPrediction::Unknown(CostUnknownReason::OutsideJointNumericSupport)
    );
}

#[test]
fn provider_host_branch_and_segments_remain_separate_and_no_extrapolation_is_invented() {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    trainer
        .observe(sample(shape(&[100], &[10], 100), 100, 1))
        .unwrap();
    trainer
        .observe(sample(shape(&[120], &[20], 100), 200, 2))
        .unwrap();
    let snapshot = trainer.publish(2).unwrap();
    let mut query = shape(&[110], &[15], 100);
    query.provider_signature[0] ^= 1;
    assert_eq!(
        predict(&snapshot, &query, 2),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
    query = shape(&[110], &[15], 100);
    query
        .numeric_features
        .as_mut()
        .unwrap()
        .output_policy_signature[0] ^= 1;
    assert_eq!(
        predict(&snapshot, &query, 2),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
    for n in [9, 21] {
        assert_eq!(
            predict(&snapshot, &shape(&[110], &[n], 100), 2),
            CostPrediction::Unknown(CostUnknownReason::OutsideJointNumericSupport)
        );
    }
    assert_eq!(
        predict(&snapshot, &shape(&[110], &[65], 100), 2),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
}

#[test]
fn absent_or_unordered_numeric_evidence_is_explicitly_unknown() {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    let snapshot = trainer.publish(0).unwrap();
    let mut missing = shape(&[100], &[10], 100);
    missing.numeric_features = None;
    assert_eq!(
        predict(&snapshot, &missing, 0),
        CostPrediction::Unknown(CostUnknownReason::NumericFeaturesMissing)
    );
    assert!(matches!(
        trainer.observe(sample(missing, 100, 1)),
        Err(CostModelError::InvalidShape(_))
    ));
    let mut unordered = shape(&[100], &[10], 100);
    unordered.order = BatchOrderSemantics::IndependentRows;
    assert_eq!(
        predict(&snapshot, &unordered, 0),
        CostPrediction::Unknown(CostUnknownReason::NumericRowOrderUnsupported)
    );
    assert_eq!(trainer.retained_sample_count(), 0);
    assert_eq!(trainer.last_clock_ns, 0);
}

#[test]
fn numeric_support_preserves_ttl_and_published_floor_without_manufacturing_coverage() {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    trainer
        .observe(sample(shape(&[100], &[10], 100), 100, 1))
        .unwrap();
    trainer
        .observe(sample(shape(&[120], &[20], 100), 500, 2))
        .unwrap();
    let old = trainer.publish(2).unwrap();
    let query = shape(&[110], &[15], 100);
    assert!(matches!(
        predict(&old, &query, 101),
        CostPrediction::Known(_)
    ));
    assert_eq!(
        predict(&old, &query, 102),
        CostPrediction::Unknown(CostUnknownReason::StaleSamples)
    );
    trainer
        .observe(sample(shape(&[100], &[10], 100), 10, 103))
        .unwrap();
    trainer
        .observe(sample(shape(&[120], &[20], 100), 20, 104))
        .unwrap();
    let CostPrediction::Known(result) = predict(&trainer.publish(104).unwrap(), &query, 104) else {
        panic!("fresh support")
    };
    assert_eq!(result.planning_ns, 510);
    assert_eq!(result.sample_count, 2);
    assert_eq!(
        predict(&old, &query, 104),
        CostPrediction::Unknown(CostUnknownReason::StaleSamples)
    );
}

#[test]
fn numeric_rows_consume_real_retention_budget_and_rejection_does_not_mutate() {
    let mut config = settings();
    config.min_samples = NonZeroUsize::new(1).unwrap();
    config.max_retained_shape_rows = NonZeroUsize::new(3).unwrap();
    config.shape_limits.max_rows = NonZeroUsize::new(1).unwrap();
    let mut trainer = CostModelTrainer::new(fingerprint(), config).unwrap();
    trainer
        .observe(sample(shape(&[100], &[10], 100), 100, 1))
        .unwrap();
    let before = trainer.publish(1).unwrap();
    assert_eq!(
        trainer.observe(sample(shape(&[120], &[20], 100), 200, 2)),
        Err(CostModelError::CapacityExceeded("retained shape rows"))
    );
    assert_eq!(trainer.retained_sample_count(), 1);
    assert_eq!(trainer.retained_shape_rows(), 2);
    assert_eq!(trainer.last_clock_ns, 1);
    assert!(Arc::ptr_eq(&before, &trainer.publish(1).unwrap()));
}

#[test]
fn exact_mode_does_not_let_new_numeric_fields_change_old_bucket_identity() {
    let mut config = settings();
    config.feature_model = CostFeatureModel::ExactV1 {};
    let mut trainer = CostModelTrainer::new(fingerprint(), config).unwrap();
    let mut old = shape(&[100], &[10], 100);
    old.numeric_features = None;
    trainer.observe(sample(old.clone(), 100, 1)).unwrap();
    trainer.observe(sample(old.clone(), 200, 2)).unwrap();
    let snapshot = trainer.publish(2).unwrap();
    let mut extended = shape(&[100], &[15], 200);
    extended.output_policy_signature = old.output_policy_signature;
    extended
        .numeric_features
        .as_mut()
        .unwrap()
        .output_policy_signature = [99; 32];
    assert!(matches!(
        predict(&snapshot, &extended, 2),
        CostPrediction::Known(_)
    ));
    extended.output_policy_signature[0] ^= 1;
    assert_eq!(
        predict(&snapshot, &extended, 2),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
}

#[test]
fn drift_is_compared_to_the_previous_prediction_before_learning_the_new_wall_cost() {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    trainer
        .observe(sample(shape(&[100], &[10], 100), 100, 1))
        .unwrap();
    trainer
        .observe(sample(shape(&[120], &[20], 100), 200, 2))
        .unwrap();
    trainer.publish(2).unwrap();
    trainer
        .observe(sample(shape(&[110], &[15], 333), 400, 3))
        .unwrap();
    let query = shape(&[110], &[15], 333);
    let CostPrediction::Known(result) = predict(&trainer.publish(3).unwrap(), &query, 3) else {
        panic!("supported query")
    };
    assert_eq!(result.errors.compared_samples, 1);
    assert_eq!(result.errors.underestimates, 1);
    assert_eq!(result.errors.max_underestimate_ns, 190);
    assert_eq!(result.planning_ns, 410);
}
