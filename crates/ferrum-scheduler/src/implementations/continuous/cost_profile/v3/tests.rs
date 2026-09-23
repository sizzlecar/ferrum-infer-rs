use super::*;
use ferrum_interfaces::execution_cost::{CanonicalWaveCostFeatures, CostRowNumericFeatures};

fn settings() -> CostModelSettings {
    CostModelSettings {
        feature_model: CostFeatureModel::EmpiricalHostContentV1 {
            host_history_bucket_tokens: NonZeroU32::new(64).unwrap(),
        },
        min_samples: NonZeroUsize::MIN,
        max_sample_age_ns: NonZeroU64::new(1000).unwrap(),
        ..Default::default()
    }
}
fn file() -> CostProfileFileV3 {
    CostProfileFileV3 {
        schema_version: 3,
        fingerprint: ProfileFingerprint {
            model_weights: [1; 32],
            numerical_policy: [2; 32],
            device_runtime: [3; 32],
            execution_config: [4; 32],
        },
        settings: (&settings()).into(),
        generated_unix_ns: 1100,
        source_clock_max_error_ns: Some(0),
        source: ProfileSource {
            generator: "actual producer".into(),
            generator_revision: "test revision".into(),
            measurement_protocol: "complete host-settled wall".into(),
            observation_artifact_sha256: [9; 32],
        },
        samples: vec![ProfileSampleV3 {
            source_record: 7,
            accepted_ordinal: 11,
            measured_unix_ns: 1000,
            shape: ProfileWaveShapeV3 {
                host_content_features: HostContentCostFeaturesV1 {
                    schema_version: 1,
                    output_policy_signature: [7; 32],
                },
                exact: ProfileWaveShapeV2 {
                    exact: ProfileWaveShape {
                        kind: ProfileWaveKind::Decode,
                        path: ProfileExecutionPath::PlanRuntime,
                        provider_signature: [5; 32],
                        output_policy_signature: [6; 32],
                        graph_state: ProfileGraphState::Warm,
                        order: ProfileBatchOrder::Ordered,
                        decode_kv_tokens: vec![20],
                        prefill_chunks: vec![],
                        recurrent_state_bytes: 0,
                        restore_bytes: 0,
                        maintenance_bytes: 0,
                        maintenance_units: 0,
                    },
                    numeric_features: Some(CanonicalWaveCostFeatures {
                        schema_version: 1,
                        output_policy_signature: [8; 32],
                        rows: vec![CostRowNumericFeatures {
                            generated_tokens_before: 3,
                            maximum_output_tokens: 4,
                            sampling_history_tokens: 3,
                            repetition_tokens: 0,
                            decoded_prefix_tokens: 4,
                            decoded_text_bytes_bound: 16,
                            decode_scratch_bytes_bound: 16,
                        }],
                    }),
                },
            },
            boundary: ProfileCostBoundaryV3::PreparationToHostSettledV1,
            outcome: ProfileObservationOutcome::Completed {},
            timing: ProfileWaveTiming {
                wall_total_ns: 100,
                device_elapsed_ns: None,
                stages: Default::default(),
            },
        }],
    }
}
fn load(value: &impl Serialize) -> Result<LoadedCostProfile, CostProfileError> {
    load_cost_profile_bytes(
        &serde_json::to_vec(value).unwrap(),
        &file().fingerprint.into(),
        &settings(),
        &Default::default(),
        ProfileLoadClock {
            wall_unix_ns: Some(1200),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: 0,
        },
    )
}

#[test]
fn only_explicit_v3_can_carry_the_new_boundary_and_mode() {
    let file = file();
    let loaded = load(&file).unwrap();
    assert_eq!(loaded.provenance.schema_version, 3);
    assert_eq!(loaded.provenance.oldest_imported_age_ns, Some(200));
    assert_eq!(
        loaded.snapshot.planning_boundary(),
        CostBoundary::PreparationToHostSettledV1
    );
    let mut old = serde_json::to_value(&file).unwrap();
    old["schema_version"] = 2.into();
    assert!(load(&old).is_err());
    let mut legacy_model = file;
    legacy_model.settings.feature_model = CostFeatureModel::ExactV1 {};
    assert!(matches!(
        load(&legacy_model),
        Err(CostProfileError::Metadata(_))
    ));
}

#[test]
fn strict_versioned_domain_and_unique_fifo_source_ids_cannot_be_filled_by_callers() {
    for pointer in [
        "/samples/0/shape/host_content_features",
        "/samples/0",
        "/settings",
    ] {
        let mut value = serde_json::to_value(file()).unwrap();
        value
            .pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("undeclared".into(), true.into());
        assert!(load(&value).is_err(), "strict parse accepted {pointer}");
    }
    let mut missing = file();
    missing.samples[0]
        .shape
        .host_content_features
        .schema_version = 2;
    assert!(load(&missing).is_err());
    let mut duplicate = file();
    let mut second = duplicate.samples[0].clone();
    second.source_record += 1;
    duplicate.samples.push(second);
    assert!(matches!(
        load(&duplicate),
        Err(CostProfileError::Metadata(_))
    ));
    let mut missing_ordinal = file();
    missing_ordinal.samples[0].accepted_ordinal = 0;
    assert!(load(&missing_ordinal).is_err());
}

#[test]
fn new_samples_keep_shape_capacity_failure_and_source_clock_boundaries() {
    let mut limits = CostProfileLoadLimits::default();
    limits.max_total_shape_rows = NonZeroUsize::MIN;
    assert!(matches!(
        load_cost_profile_bytes(
            &serde_json::to_vec(&file()).unwrap(),
            &file().fingerprint.into(),
            &settings(),
            &limits,
            ProfileLoadClock {
                wall_unix_ns: Some(1200),
                wall_max_error_ns: Some(0),
                monotonic_now_ns: 0
            }
        ),
        Err(CostProfileError::Limit(_))
    ));
    let mut failed = file();
    failed.samples[0].outcome = ProfileObservationOutcome::FailedAfterSubmit {};
    let loaded = load(&failed).unwrap();
    assert_eq!(loaded.provenance.counts.recorded_samples, 0);
    assert_eq!(
        loaded.provenance.counts.skipped_samples[&ProfileSkippedObservation::FailedAfterSubmit],
        1
    );
    let mut future = file();
    future.samples[0].measured_unix_ns = 1300;
    assert!(matches!(load(&future), Err(CostProfileError::Clock(_))));
}
