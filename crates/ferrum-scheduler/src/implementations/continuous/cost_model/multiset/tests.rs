use super::*;

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
        feature_model: CostFeatureModel::EmpiricalRowMultisetV2 {
            host_history_bucket_tokens: NonZeroU32::new(64).unwrap(),
        },
        min_samples: NonZeroUsize::new(2).unwrap(),
        context_bucket_tokens: NonZeroU32::new(128).unwrap(),
        max_sample_age_ns: NonZeroU64::new(100).unwrap(),
        drift_margin_ns: 5,
        ..Default::default()
    }
}
fn shape() -> WaveExecutionShape {
    WaveExecutionShape {
        row_multiset_features: Some(HostRowMultisetCostFeaturesV2 {
            schema_version: 2,
            wave_policy_signature: [9; 32],
            rows: vec![
                HostRowStaticCostFeaturesV2 {
                    role: HostRowRoleV2::Decode,
                    categorical_signature: [10; 32],
                },
                HostRowStaticCostFeaturesV2 {
                    role: HostRowRoleV2::Decode,
                    categorical_signature: [11; 32],
                },
            ],
        }),
        host_content_features: Some(HostContentCostFeaturesV1 {
            schema_version: 1,
            output_policy_signature: [6; 32],
        }),
        numeric_features: Some(CanonicalWaveCostFeatures {
            schema_version: 1,
            output_policy_signature: [7; 32],
            rows: [4, 9]
                .map(|n| CostRowNumericFeatures {
                    generated_tokens_before: n,
                    maximum_output_tokens: 20,
                    sampling_history_tokens: n,
                    repetition_tokens: 0,
                    decoded_prefix_tokens: n + 1,
                    decoded_text_bytes_bound: 4 * (n + 1),
                    decode_scratch_bytes_bound: 2 * (n + 1),
                })
                .to_vec(),
        }),
        kind: WaveKind::Decode,
        path: WaveExecutionPath::PlanRuntime,
        provider_signature: [8; 32],
        output_policy_signature: [5; 32],
        graph_state: WaveGraphState::Disabled,
        order: BatchOrderSemantics::Ordered,
        decode_kv_tokens: vec![30, 90],
        prefill_chunks: vec![],
        recurrent_state_bytes: 10,
        restore_bytes: 0,
        maintenance_bytes: 0,
        maintenance_units: 0,
    }
}
fn swapped(mut shape: WaveExecutionShape) -> WaveExecutionShape {
    shape
        .row_multiset_features
        .as_mut()
        .unwrap()
        .rows
        .swap(0, 1);
    shape.numeric_features.as_mut().unwrap().rows.swap(0, 1);
    shape.decode_kv_tokens.swap(0, 1);
    // The old identity really changes; V2 must use its own complete evidence.
    shape
        .host_content_features
        .as_mut()
        .unwrap()
        .output_policy_signature = [33; 32];
    shape
        .numeric_features
        .as_mut()
        .unwrap()
        .output_policy_signature = [34; 32];
    shape.output_policy_signature = [35; 32];
    shape
}
fn sample(shape: WaveExecutionShape, at: u64, wall: u64) -> WaveCostObservation {
    WaveCostObservation {
        fingerprint: fingerprint(),
        actual_shape: shape,
        boundary: CostBoundary::PreparationToHostSettledV1,
        outcome: WaveObservationOutcome::Completed,
        timing: WaveTiming {
            wall_total_ns: wall,
            device_elapsed_ns: None,
            stages: Default::default(),
        },
        observed_at_ns: at,
    }
}
fn predict(snapshot: &CostModelSnapshot, shape: &WaveExecutionShape, at: u64) -> CostPrediction {
    snapshot.predict(
        &fingerprint(),
        shape,
        CostBoundary::PreparationToHostSettledV1,
        at,
    )
}
fn trained() -> Arc<CostModelSnapshot> {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    trainer.observe(sample(shape(), 1, 90)).unwrap();
    trainer.observe(sample(swapped(shape()), 2, 110)).unwrap();
    trainer.publish(2).unwrap()
}

#[test]
fn physical_permutations_train_one_statistical_bucket_without_mutating_observations() {
    let first = shape();
    let second = swapped(shape());
    let original = second.clone();
    let snapshot = trained();
    assert_eq!(snapshot.bucket_count(), 1);
    for query in [&first, &second] {
        let CostPrediction::Known(p) = predict(&snapshot, query, 2) else {
            panic!("complete row permutation lost its samples")
        };
        assert_eq!((p.sample_count, p.planning_ns), (2, 115));
    }
    assert_eq!(
        second, original,
        "query rewrote executable/physical evidence"
    );
    assert_eq!(second.order, BatchOrderSemantics::Ordered);
}

#[test]
fn independent_axis_sorting_cannot_create_a_supported_row_association() {
    let snapshot = trained();
    let mut query = shape();
    query.numeric_features.as_mut().unwrap().rows.swap(0, 1);
    assert!(
        matches!(predict(&snapshot, &query, 2), CostPrediction::Unknown(_)),
        "same KV/history marginals must not invent complete row support"
    );
    let mut terminal = shape();
    terminal.row_multiset_features.as_mut().unwrap().rows[0].categorical_signature = [12; 32];
    assert_eq!(
        predict(&snapshot, &terminal, 2),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
}

#[test]
fn provider_product_readback_and_policy_domains_are_not_interchangeable() {
    let snapshot = trained();
    for change in 0..3 {
        let mut query = shape();
        match change {
            0 => query.provider_signature = [44; 32],
            1 => {
                query
                    .row_multiset_features
                    .as_mut()
                    .unwrap()
                    .wave_policy_signature = [45; 32]
            }
            _ => query.path = WaveExecutionPath::LegacySplit,
        }
        assert_eq!(
            predict(&snapshot, &query, 2),
            CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
        );
    }
}

#[test]
fn mixed_roles_use_physical_cursors_and_cannot_cross_segments() {
    let mut mixed = shape();
    mixed.kind = WaveKind::Mixed;
    mixed.decode_kv_tokens = vec![90];
    mixed.prefill_chunks = vec![PrefillShape {
        offset: 0,
        count: NonZeroU32::new(4).unwrap(),
        total_prompt_tokens: NonZeroU32::new(8).unwrap(),
    }];
    mixed.row_multiset_features.as_mut().unwrap().rows[0].role = HostRowRoleV2::Prefill;
    assert!(validate(&mixed).is_ok());
    let original = mixed.clone();
    statistical_order(&mut mixed);
    assert_eq!(
        mixed, original,
        "prefill-first wave was treated as decode-first"
    );
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    trainer.observe(sample(mixed.clone(), 1, 80)).unwrap();
    trainer.observe(sample(mixed.clone(), 2, 90)).unwrap();
    let snapshot = trainer.publish(2).unwrap();
    mixed
        .row_multiset_features
        .as_mut()
        .unwrap()
        .rows
        .swap(0, 1);
    mixed.numeric_features.as_mut().unwrap().rows.swap(0, 1);
    // Even if a producer erroneously reused a provider digest, segment order
    // remains explicit in the retained row classes and keeps this Unknown.
    assert_eq!(
        predict(&snapshot, &mixed, 2),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
}

#[test]
fn absent_or_inconsistent_v2_evidence_never_upgrades_v1() {
    let snapshot = trained();
    let mut absent = shape();
    absent.row_multiset_features = None;
    assert_eq!(
        predict(&snapshot, &absent, 2),
        CostPrediction::Unknown(CostUnknownReason::HostContentFeaturesMissing)
    );
    let mut malformed = shape();
    malformed.row_multiset_features.as_mut().unwrap().rows[0].role = HostRowRoleV2::Prefill;
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    assert!(matches!(
        trainer.observe(sample(malformed, 1, 100)),
        Err(CostModelError::InvalidShape(_))
    ));
    assert_eq!(trainer.retained_sample_count(), 0);
    let mut legacy = settings();
    legacy.feature_model = CostFeatureModel::EmpiricalHostContentV1 {
        host_history_bucket_tokens: NonZeroU32::new(64).unwrap(),
    };
    let mut trainer = CostModelTrainer::new(fingerprint(), legacy).unwrap();
    trainer.observe(sample(shape(), 1, 100)).unwrap();
    trainer.observe(sample(swapped(shape()), 2, 100)).unwrap();
    let old = trainer.publish(2).unwrap();
    assert_eq!(
        old.bucket_count(),
        2,
        "new feature must not merge old model keys"
    );
    assert_eq!(
        predict(&old, &shape(), 2),
        CostPrediction::Unknown(CostUnknownReason::InsufficientSamples)
    );
}

#[test]
fn extra_rows_consume_capacity_and_snapshot_age_keeps_original_receipt_time() {
    let mut limited = settings();
    limited.max_retained_shape_rows = NonZeroUsize::new(10).unwrap();
    limited.shape_limits.max_rows = NonZeroUsize::new(2).unwrap();
    let mut trainer = CostModelTrainer::new(fingerprint(), limited).unwrap();
    trainer.observe(sample(shape(), 1, 100)).unwrap();
    assert_eq!(trainer.retained_shape_rows(), 6);
    assert!(matches!(
        trainer.observe(sample(swapped(shape()), 2, 100)),
        Err(CostModelError::CapacityExceeded("retained shape rows"))
    ));
    assert_eq!(trainer.retained_sample_count(), 1);
    let snapshot = trained();
    assert!(matches!(
        predict(&snapshot, &shape(), 101),
        CostPrediction::Known(_)
    ));
    assert_eq!(
        predict(&snapshot, &shape(), 102),
        CostPrediction::Unknown(CostUnknownReason::StaleSamples)
    );
}
