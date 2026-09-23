use super::*;
use ferrum_interfaces::execution_cost::CostRowNumericFeatures;

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
        feature_model: CostFeatureModel::EmpiricalHostContentV1 {
            host_history_bucket_tokens: NonZeroU32::new(64).unwrap(),
        },
        min_samples: NonZeroUsize::new(2).unwrap(),
        context_bucket_tokens: NonZeroU32::new(128).unwrap(),
        residual_quantile: 1.0,
        drift_margin_ns: 5,
        max_sample_age_ns: NonZeroU64::new(100).unwrap(),
        ..Default::default()
    }
}
fn shape(contexts: &[u32], actual_content: u8) -> WaveExecutionShape {
    WaveExecutionShape {
        row_multiset_features: None,
        host_content_features: Some(HostContentCostFeaturesV1 {
            schema_version: 1,
            output_policy_signature: [42; 32],
        }),
        numeric_features: Some(CanonicalWaveCostFeatures {
            schema_version: 1,
            output_policy_signature: [actual_content; 32],
            rows: contexts
                .iter()
                .map(|_| CostRowNumericFeatures {
                    generated_tokens_before: 4,
                    maximum_output_tokens: 40,
                    sampling_history_tokens: 4,
                    repetition_tokens: 0,
                    decoded_prefix_tokens: 5,
                    decoded_text_bytes_bound: 50,
                    decode_scratch_bytes_bound: 20,
                })
                .collect(),
        }),
        kind: WaveKind::Decode,
        path: WaveExecutionPath::PlanRuntime,
        provider_signature: [8; 32],
        output_policy_signature: [actual_content; 32],
        graph_state: WaveGraphState::Warm,
        order: BatchOrderSemantics::Ordered,
        decode_kv_tokens: contexts.to_vec(),
        prefill_chunks: vec![],
        recurrent_state_bytes: 0,
        restore_bytes: 0,
        maintenance_bytes: 0,
        maintenance_units: 0,
    }
}
fn observation(contexts: &[u32], content: u8, wall: u64, at: u64) -> WaveCostObservation {
    WaveCostObservation {
        fingerprint: fingerprint(),
        actual_shape: shape(contexts, content),
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
fn prediction(snapshot: &CostModelSnapshot, query: &WaveExecutionShape, at: u64) -> CostPrediction {
    snapshot.predict(
        &fingerprint(),
        query,
        CostBoundary::PreparationToHostSettledV1,
        at,
    )
}
fn known(value: CostPrediction) -> WaveCostPrediction {
    match value {
        CostPrediction::Known(value) => value,
        other => panic!("expected empirical prediction: {other:?}"),
    }
}

#[test]
fn content_is_an_empirical_disturbance_without_fabricated_retry_upper_samples() {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    trainer.observe(observation(&[20], 10, 90, 1)).unwrap();
    trainer.observe(observation(&[20], 11, 110, 2)).unwrap();
    let snapshot = trainer.publish(2).unwrap();
    let result = known(prediction(&snapshot, &shape(&[20], 12), 2));
    assert_eq!(result.planning_ns, 115);
    assert_eq!(result.boundary, CostBoundary::PreparationToHostSettledV1);
    assert_eq!(result.sample_count, 2);
    assert_eq!(result.empirical_envelope_ns, None);
    assert_eq!(
        snapshot.planning_boundary(),
        CostBoundary::PreparationToHostSettledV1
    );
    // No retries/content realization were inserted as numerical support.
    assert_eq!(result.coverage.decode_context_ranges, vec![(20, 20)]);
}

#[test]
fn new_model_cannot_upgrade_legacy_timing_or_missing_domain() {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    let mut legacy = observation(&[20], 1, 100, 1);
    legacy.boundary = CostBoundary::PreparationToCommit;
    assert!(matches!(
        trainer.observe(legacy),
        Err(CostModelError::InvalidTiming(_))
    ));
    trainer.observe(observation(&[20], 1, 100, 1)).unwrap();
    trainer.observe(observation(&[20], 2, 100, 2)).unwrap();
    let snapshot = trainer.publish(2).unwrap();
    let mut missing = shape(&[20], 3);
    missing.host_content_features = None;
    assert_eq!(
        prediction(&snapshot, &missing, 2),
        CostPrediction::Unknown(CostUnknownReason::HostContentFeaturesMissing)
    );
    assert_eq!(
        snapshot.predict(
            &fingerprint(),
            &shape(&[20], 1),
            CostBoundary::PreparationToCommit,
            2
        ),
        CostPrediction::Unknown(CostUnknownReason::BoundaryUnsupported)
    );
    let mut failed = observation(&[20], 1, 1, 3);
    failed.outcome = WaveObservationOutcome::FailedAfterSubmit;
    assert_eq!(
        trainer.observe(failed).unwrap(),
        ObservationDisposition::Skipped(ObservationSkipReason::FailedAfterSubmit)
    );
}

#[test]
fn legacy_numeric_keys_remain_content_exact_and_ignore_new_extension() {
    let mut settings = settings();
    settings.feature_model = CostFeatureModel::BoundedNumericV1 {
        host_history_bucket_tokens: NonZeroU32::new(64).unwrap(),
    };
    let mut trainer = CostModelTrainer::new(fingerprint(), settings).unwrap();
    for at in 1..=2 {
        let mut sample = observation(&[20], 1, 100, at);
        sample.boundary = CostBoundary::PreparationToCommit;
        trainer.observe(sample).unwrap();
    }
    let snapshot = trainer.publish(2).unwrap();
    let mut same = shape(&[20], 1);
    same.host_content_features = None;
    assert!(matches!(
        snapshot.predict(&fingerprint(), &same, CostBoundary::PreparationToCommit, 2),
        CostPrediction::Known(_)
    ));
    assert_eq!(
        snapshot.predict(
            &fingerprint(),
            &shape(&[20], 2),
            CostBoundary::PreparationToCommit,
            2
        ),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
}

#[test]
fn unanticipated_long_host_work_is_an_underestimate_and_changes_future_margin() {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    trainer.observe(observation(&[20], 1, 100, 1)).unwrap();
    trainer.observe(observation(&[20], 2, 100, 2)).unwrap();
    let old = trainer.publish(2).unwrap();
    trainer.observe(observation(&[20], 3, 900, 3)).unwrap();
    let new = trainer.publish(3).unwrap();
    assert_eq!(
        known(prediction(&old, &shape(&[20], 4), 3)).planning_ns,
        105
    );
    let updated = known(prediction(&new, &shape(&[20], 4), 3));
    assert_eq!(updated.planning_ns, 905);
    assert_eq!(updated.errors.underestimates, 1);
    assert_eq!(updated.errors.max_underestimate_ns, 795);
    assert_eq!(
        known(prediction(&old, &shape(&[20], 1), 101)).valid_for_ns,
        0
    );
    assert_eq!(
        prediction(&old, &shape(&[20], 1), 102),
        CostPrediction::Unknown(CostUnknownReason::StaleSamples)
    );
}

#[test]
fn known_context_support_and_physical_route_still_cannot_be_spliced() {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    trainer.observe(observation(&[10, 20], 1, 100, 1)).unwrap();
    trainer.observe(observation(&[20, 10], 2, 100, 2)).unwrap();
    let snapshot = trainer.publish(2).unwrap();
    assert_eq!(
        prediction(&snapshot, &shape(&[15, 15], 3), 2),
        CostPrediction::Unknown(CostUnknownReason::OutsideJointNumericSupport)
    );
    let mut route = shape(&[10, 20], 3);
    route.provider_signature = [9; 32];
    assert_eq!(
        prediction(&snapshot, &route, 2),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
    let mut terminal = shape(&[10, 20], 3);
    terminal
        .host_content_features
        .as_mut()
        .unwrap()
        .output_policy_signature = [43; 32];
    assert_eq!(
        prediction(&snapshot, &terminal, 2),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
}
