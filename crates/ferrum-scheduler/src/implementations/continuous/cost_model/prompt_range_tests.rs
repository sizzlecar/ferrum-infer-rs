//! Hardware-independent statistical contract; these fixtures are not GPU evidence.
use super::*;
use ferrum_interfaces::execution_cost::{
    CostRowNumericFeatures, HostRowRoleV2, HostRowStaticCostFeaturesV2,
};

pub(crate) fn fingerprint() -> ExecutionFingerprint {
    ExecutionFingerprint {
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }
}
pub(crate) fn settings() -> CostModelSettings {
    CostModelSettings {
        feature_model: CostFeatureModel::EmpiricalPromptRangeV3 {
            host_history_bucket_tokens: NonZeroU32::new(64).unwrap(),
        },
        min_samples: NonZeroUsize::new(2).unwrap(),
        prefill_offset_bucket_tokens: NonZeroU32::new(128).unwrap(),
        max_sample_age_ns: NonZeroU64::new(1000).unwrap(),
        drift_margin_ns: 5,
        ..Default::default()
    }
}
pub(crate) fn shape(offset: u32, total: u32) -> WaveExecutionShape {
    WaveExecutionShape {
        row_multiset_features: Some(HostRowMultisetCostFeaturesV2 {
            schema_version: 2,
            wave_policy_signature: [9; 32],
            rows: vec![HostRowStaticCostFeaturesV2 {
                role: HostRowRoleV2::Prefill,
                categorical_signature: [10; 32],
            }],
        }),
        host_content_features: None,
        numeric_features: Some(CanonicalWaveCostFeatures {
            schema_version: 1,
            output_policy_signature: [7; 32],
            rows: vec![CostRowNumericFeatures {
                generated_tokens_before: 0,
                maximum_output_tokens: 20,
                sampling_history_tokens: 0,
                repetition_tokens: 0,
                decoded_prefix_tokens: 0,
                decoded_text_bytes_bound: 0,
                decode_scratch_bytes_bound: 0,
            }],
        }),
        kind: WaveKind::Prefill,
        path: WaveExecutionPath::PlanRuntime,
        provider_signature: [8; 32],
        output_policy_signature: [5; 32],
        graph_state: WaveGraphState::Disabled,
        order: BatchOrderSemantics::Ordered,
        decode_kv_tokens: vec![],
        prefill_chunks: vec![PrefillShape {
            offset,
            count: NonZeroU32::new(32).unwrap(),
            total_prompt_tokens: NonZeroU32::new(total).unwrap(),
        }],
        recurrent_state_bytes: 10,
        restore_bytes: 0,
        maintenance_bytes: 0,
        maintenance_units: 0,
    }
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
fn train(shapes: [WaveExecutionShape; 2], settings: CostModelSettings) -> Arc<CostModelSnapshot> {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings).unwrap();
    for (index, shape) in shapes.into_iter().enumerate() {
        assert_eq!(
            trainer
                .observe(sample(shape, index as u64 + 1, 90 + index as u64 * 20))
                .unwrap(),
            ObservationDisposition::Recorded
        );
    }
    trainer.publish(2).unwrap()
}
fn predict(snapshot: &CostModelSnapshot, shape: &WaveExecutionShape, at: u64) -> CostPrediction {
    snapshot.predict(
        &fingerprint(),
        shape,
        CostBoundary::PreparationToHostSettledV1,
        at,
    )
}
#[test]
fn new_model_reuses_intermediate_prompt_totals_without_changing_actual_work_or_v2_keys() {
    let observations = [shape(0, 64), shape(0, 128)];
    let originals = observations.clone();
    let query = shape(0, 96);
    let original_query = query.clone();
    let snapshot = train(observations.clone(), settings());
    assert_eq!(snapshot.bucket_count(), 1);
    let CostPrediction::Known(p) = predict(&snapshot, &query, 2) else {
        panic!("interior original prompt total did not share actual support")
    };
    assert_eq!((p.sample_count, p.planning_ns), (2, 115));
    assert_eq!(observations, originals);
    assert_eq!(query, original_query);
    assert_eq!(query.prefill_chunks[0].total_prompt_tokens.get(), 96);
    let mut old = settings();
    old.feature_model = CostFeatureModel::EmpiricalRowMultisetV2 {
        host_history_bucket_tokens: NonZeroU32::new(64).unwrap(),
    };
    let old_snapshot = train(observations, old);
    assert_eq!(old_snapshot.bucket_count(), 2);
    assert_eq!(
        predict(&old_snapshot, &query, 2),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
}
#[test]
fn prompt_total_needs_one_actual_joint_point_not_independent_coordinate_extrema() {
    let snapshot = train([shape(0, 256), shape(64, 128)], settings());
    assert!(matches!(
        predict(&snapshot, &shape(32, 128), 2),
        CostPrediction::Known(_)
    ));
    assert_eq!(
        predict(&snapshot, &shape(64, 256), 2),
        CostPrediction::Unknown(CostUnknownReason::OutsideJointNumericSupport)
    );
    for total in [100, 257] {
        assert_eq!(
            predict(&snapshot, &shape(32, total), 2),
            CostPrediction::Unknown(CostUnknownReason::OutsideJointNumericSupport)
        );
    }
}
#[test]
fn final_chunk_count_provider_and_host_branch_stay_exact() {
    let snapshot = train([shape(0, 64), shape(0, 128)], settings());
    // Even if a malformed producer reused its old hash, checked final bits
    // forbid grouping this end==total wave with a partial wave.
    let mut changes = vec![shape(0, 32)];
    let mut count = shape(0, 96);
    count.prefill_chunks[0].count = NonZeroU32::new(16).unwrap();
    changes.push(count);
    let mut route = shape(0, 96);
    route.provider_signature = [44; 32];
    changes.push(route);
    let mut product = shape(0, 96);
    product
        .row_multiset_features
        .as_mut()
        .unwrap()
        .wave_policy_signature = [45; 32];
    changes.push(product);
    let mut host = shape(0, 96);
    host.row_multiset_features.as_mut().unwrap().rows[0].categorical_signature = [46; 32];
    changes.push(host);
    for query in changes {
        assert_eq!(
            predict(&snapshot, &query, 2),
            CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
        );
    }
}
#[test]
fn actual_prompt_validation_and_original_sample_age_precede_statistical_projection() {
    let snapshot = train([shape(0, 64), shape(0, 128)], settings());
    assert_eq!(
        predict(&snapshot, &shape(0, 31), 2),
        CostPrediction::Unknown(CostUnknownReason::InvalidShape)
    );
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    assert!(matches!(
        trainer.observe(sample(shape(0, 31), 1, 100)),
        Err(CostModelError::InvalidShape(_))
    ));
    let query = shape(0, 96);
    let CostPrediction::Known(p) = predict(&snapshot, &query, 1001) else {
        panic!("sample at the TTL boundary should retain its original age")
    };
    assert_eq!(p.valid_for_ns, 0);
    assert_eq!(
        predict(&snapshot, &query, 1002),
        CostPrediction::Unknown(CostUnknownReason::StaleSamples)
    );
}

#[test]
fn complete_row_tuples_move_together_but_mixed_role_segments_remain_exact() {
    let mut first = shape(0, 64);
    first.prefill_chunks.push(PrefillShape {
        offset: 0,
        count: NonZeroU32::new(16).unwrap(),
        total_prompt_tokens: NonZeroU32::new(80).unwrap(),
    });
    let numeric_row = first.numeric_features.as_ref().unwrap().rows[0];
    first
        .numeric_features
        .as_mut()
        .unwrap()
        .rows
        .push(numeric_row);
    first
        .row_multiset_features
        .as_mut()
        .unwrap()
        .rows
        .push(HostRowStaticCostFeaturesV2 {
            role: HostRowRoleV2::Prefill,
            categorical_signature: [11; 32],
        });
    let mut second = first.clone();
    second.prefill_chunks[0].total_prompt_tokens = NonZeroU32::new(128).unwrap();
    second.prefill_chunks[1].total_prompt_tokens = NonZeroU32::new(160).unwrap();
    second.prefill_chunks.reverse();
    second.numeric_features.as_mut().unwrap().rows.reverse();
    second
        .row_multiset_features
        .as_mut()
        .unwrap()
        .rows
        .reverse();
    let snapshot = train([first.clone(), second], settings());
    let mut query = first;
    query.prefill_chunks[0].total_prompt_tokens = NonZeroU32::new(96).unwrap();
    query.prefill_chunks[1].total_prompt_tokens = NonZeroU32::new(120).unwrap();
    assert!(matches!(
        predict(&snapshot, &query, 2),
        CostPrediction::Known(_)
    ));
    // Independently sorting/moving a static category breaks its work tuple.
    query.row_multiset_features.as_mut().unwrap().rows.reverse();
    assert_eq!(
        predict(&snapshot, &query, 2),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
    let mut mixed = shape(0, 64);
    mixed.kind = WaveKind::Mixed;
    mixed.decode_kv_tokens.push(96);
    mixed
        .numeric_features
        .as_mut()
        .unwrap()
        .rows
        .push(numeric_row);
    mixed
        .row_multiset_features
        .as_mut()
        .unwrap()
        .rows
        .push(HostRowStaticCostFeaturesV2 {
            role: HostRowRoleV2::Decode,
            categorical_signature: [12; 32],
        });
    let mut later = mixed.clone();
    later.prefill_chunks[0].total_prompt_tokens = NonZeroU32::new(128).unwrap();
    let snapshot = train([mixed.clone(), later], settings());
    mixed.prefill_chunks[0].total_prompt_tokens = NonZeroU32::new(96).unwrap();
    assert!(matches!(
        predict(&snapshot, &mixed, 2),
        CostPrediction::Known(_)
    ));
    mixed.numeric_features.as_mut().unwrap().rows.reverse();
    mixed.row_multiset_features.as_mut().unwrap().rows.reverse();
    assert_eq!(
        predict(&snapshot, &mixed, 2),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
}
