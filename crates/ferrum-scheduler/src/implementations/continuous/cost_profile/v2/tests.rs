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
        ..Default::default()
    }
}

fn shape(n: u64) -> WaveExecutionShape {
    WaveExecutionShape {
        row_multiset_features: None,
        host_content_features: None,
        kind: WaveKind::Decode,
        path: WaveExecutionPath::PlanRuntime,
        provider_signature: [5; 32],
        output_policy_signature: [n as u8; 32],
        graph_state: WaveGraphState::Disabled,
        order: BatchOrderSemantics::Ordered,
        decode_kv_tokens: vec![100 + n as u32],
        prefill_chunks: vec![],
        recurrent_state_bytes: 0,
        restore_bytes: 0,
        maintenance_bytes: 0,
        maintenance_units: 0,
        numeric_features: Some(CanonicalWaveCostFeatures {
            schema_version: COST_NUMERIC_FEATURE_SCHEMA_V1,
            output_policy_signature: [6; 32],
            rows: vec![CostRowNumericFeatures {
                generated_tokens_before: n,
                maximum_output_tokens: 100,
                sampling_history_tokens: n,
                repetition_tokens: n,
                decoded_prefix_tokens: n + 1,
                decoded_text_bytes_bound: (n + 1) * 4,
                decode_scratch_bytes_bound: (n + 1) * 8,
            }],
        }),
    }
}

fn file() -> CostProfileFileV2 {
    CostProfileFileV2 {
        schema_version: COST_PROFILE_SCHEMA_VERSION_V2,
        fingerprint: (&fingerprint()).into(),
        settings: (&settings()).into(),
        generated_unix_ns: 1005,
        source_clock_max_error_ns: Some(0),
        source: ProfileSource {
            generator: "ferrum".into(),
            generator_revision: "test-source".into(),
            measurement_protocol: "actual-preparation-through-commit".into(),
            observation_artifact_sha256: [7; 32],
        },
        samples: [10, 20]
            .into_iter()
            .enumerate()
            .map(|(index, n)| ProfileSampleV2 {
                source_record: index as u64,
                measured_unix_ns: 1000 + index as u64,
                shape: (&shape(n)).into(),
                boundary: ProfileCostBoundary::PreparationToCommit,
                outcome: ProfileObservationOutcome::Completed {},
                timing: ProfileWaveTiming {
                    wall_total_ns: 10 * n,
                    device_elapsed_ns: None,
                    stages: Default::default(),
                },
            })
            .collect(),
    }
}

fn clock() -> ProfileLoadClock {
    ProfileLoadClock {
        wall_unix_ns: Some(1010),
        wall_max_error_ns: Some(0),
        monotonic_now_ns: 0,
    }
}

fn load(
    input: &impl Serialize,
    config: &CostModelSettings,
    limits: &CostProfileLoadLimits,
) -> Result<LoadedCostProfile, CostProfileError> {
    load_cost_profile_bytes(
        &serde_json::to_vec(input).unwrap(),
        &fingerprint(),
        config,
        limits,
        clock(),
    )
}

#[test]
fn v2_numeric_import_preserves_features_source_records_and_original_ages() {
    let mut wire = file();
    wire.samples.reverse();
    let loaded = load(&wire, &settings(), &Default::default()).unwrap();
    assert_eq!(loaded.provenance.schema_version, 2);
    assert_eq!(loaded.provenance.counts.recorded_samples, 2);
    assert_eq!(loaded.provenance.oldest_imported_age_ns, Some(10));
    assert_eq!(loaded.snapshot.bucket_count(), 1);
    let query = shape(15);
    let CostPrediction::Known(prediction) =
        loaded
            .snapshot
            .predict(&fingerprint(), &query, CostBoundary::PreparationToCommit, 0)
    else {
        panic!("fresh numeric support")
    };
    assert_eq!(prediction.planning_ns, 210);
    assert_eq!(prediction.valid_for_ns, 90);
    assert_eq!(
        loaded.snapshot.predict(
            &fingerprint(),
            &query,
            CostBoundary::PreparationToCommit,
            91
        ),
        CostPrediction::Unknown(CostUnknownReason::StaleSamples)
    );
}

#[test]
fn numeric_rows_are_included_in_profile_total_retention_limits() {
    let limits = CostProfileLoadLimits {
        max_total_shape_rows: NonZeroUsize::new(3).unwrap(),
        ..Default::default()
    };
    assert!(matches!(
        load(&file(), &settings(), &limits),
        Err(CostProfileError::Limit("total shape row limit exceeded"))
    ));
    let limits = CostProfileLoadLimits {
        max_total_shape_rows: NonZeroUsize::new(4).unwrap(),
        ..limits
    };
    assert!(load(&file(), &settings(), &limits).is_ok());
}

#[test]
fn v2_never_invents_missing_features_or_accepts_invalid_numeric_evidence() {
    let mut missing = file();
    missing.samples[0].shape.numeric_features = None;
    assert!(matches!(
        load(&missing, &settings(), &Default::default()),
        Err(CostProfileError::Sample {
            reason: CostModelError::InvalidShape(_),
            ..
        })
    ));
    let mut invalid = file();
    invalid.samples[0]
        .shape
        .numeric_features
        .as_mut()
        .unwrap()
        .rows[0]
        .maximum_output_tokens = 5;
    assert!(matches!(
        load(&invalid, &settings(), &Default::default()),
        Err(CostProfileError::Metadata("invalid numeric shape features"))
    ));
    let mut wrong_schema = file();
    wrong_schema.samples[0]
        .shape
        .numeric_features
        .as_mut()
        .unwrap()
        .schema_version = 99;
    assert!(matches!(
        load(&wrong_schema, &settings(), &Default::default()),
        Err(CostProfileError::Metadata(_))
    ));
}

#[test]
fn both_profile_versions_are_strict_and_old_profiles_cannot_train_numeric_mode() {
    let v2 = file();
    let v1 = CostProfileFile {
        schema_version: COST_PROFILE_SCHEMA_VERSION,
        fingerprint: v2.fingerprint,
        settings: v2.settings.exact,
        generated_unix_ns: v2.generated_unix_ns,
        source_clock_max_error_ns: v2.source_clock_max_error_ns,
        source: v2.source,
        samples: v2
            .samples
            .into_iter()
            .map(|sample| ProfileSample {
                source_record: sample.source_record,
                measured_unix_ns: sample.measured_unix_ns,
                shape: sample.shape.exact,
                boundary: sample.boundary,
                outcome: sample.outcome,
                timing: sample.timing,
            })
            .collect(),
    };
    let mut exact = settings();
    exact.feature_model = CostFeatureModel::ExactV1 {};
    assert_eq!(
        load(&v1, &exact, &Default::default())
            .unwrap()
            .provenance
            .counts
            .recorded_samples,
        2
    );
    assert!(matches!(
        load(&v1, &settings(), &Default::default()),
        Err(CostProfileError::SettingsMismatch)
    ));
    let mut old_wire = serde_json::to_value(&v1).unwrap();
    old_wire["settings"]["feature_model"] = serde_json::json!({"kind":"exact_v1"});
    assert!(matches!(
        load(&old_wire, &exact, &Default::default()),
        Err(CostProfileError::Json(_))
    ));
    let mut new_wire = serde_json::to_value(file()).unwrap();
    new_wire["settings"]["feature_model"]["unexpected"] = serde_json::json!(true);
    assert!(matches!(
        load(&new_wire, &settings(), &Default::default()),
        Err(CostProfileError::Json(_))
    ));
}

#[test]
fn schema_and_mode_mismatches_are_not_silent_fallbacks() {
    let mut unsupported = file();
    unsupported.schema_version = 99;
    assert!(matches!(
        load(&unsupported, &settings(), &Default::default()),
        Err(CostProfileError::UnsupportedVersion(99))
    ));
    let mut exact = settings();
    exact.feature_model = CostFeatureModel::ExactV1 {};
    assert!(matches!(
        load(&file(), &exact, &Default::default()),
        Err(CostProfileError::SettingsMismatch)
    ));
    let mut duplicate = file();
    duplicate.samples[1].source_record = duplicate.samples[0].source_record;
    assert!(matches!(
        load(&duplicate, &settings(), &Default::default()),
        Err(CostProfileError::Metadata("duplicate source record"))
    ));
}
