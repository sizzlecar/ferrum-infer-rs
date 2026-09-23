use super::*;
use ferrum_interfaces::execution_cost::{
    CanonicalWaveCostFeatures, CostRowNumericFeatures, HostRowRoleV2, HostRowStaticCostFeaturesV2,
};

fn settings() -> CostModelSettings {
    CostModelSettings {
        feature_model: CostFeatureModel::EmpiricalRowMultisetV2 {
            host_history_bucket_tokens: NonZeroU32::new(64).unwrap(),
        },
        min_samples: NonZeroUsize::MIN,
        max_sample_age_ns: NonZeroU64::new(1000).unwrap(),
        ..Default::default()
    }
}
fn shape() -> WaveExecutionShape {
    WaveExecutionShape {
        row_multiset_features: Some(HostRowMultisetCostFeaturesV2 {
            schema_version: 2,
            wave_policy_signature: [9; 32],
            rows: vec![HostRowStaticCostFeaturesV2 {
                role: HostRowRoleV2::Decode,
                categorical_signature: [10; 32],
            }],
        }),
        host_content_features: Some(HostContentCostFeaturesV1 {
            schema_version: 1,
            output_policy_signature: [8; 32],
        }),
        numeric_features: Some(CanonicalWaveCostFeatures {
            schema_version: 1,
            output_policy_signature: [7; 32],
            rows: vec![CostRowNumericFeatures {
                generated_tokens_before: 3,
                maximum_output_tokens: 4,
                sampling_history_tokens: 3,
                repetition_tokens: 0,
                decoded_prefix_tokens: 4,
                decoded_text_bytes_bound: 16,
                decode_scratch_bytes_bound: 8,
            }],
        }),
        kind: WaveKind::Decode,
        path: WaveExecutionPath::PlanRuntime,
        provider_signature: [5; 32],
        output_policy_signature: [6; 32],
        graph_state: WaveGraphState::Disabled,
        order: BatchOrderSemantics::Ordered,
        decode_kv_tokens: vec![20],
        prefill_chunks: vec![],
        recurrent_state_bytes: 0,
        restore_bytes: 0,
        maintenance_bytes: 0,
        maintenance_units: 0,
    }
}
fn file() -> CostProfileFileV4 {
    CostProfileFileV4 {
        schema_version: 4,
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
            generator: "typed actual test producer".into(),
            generator_revision: "fixture revision".into(),
            measurement_protocol: "complete host-settled".into(),
            observation_artifact_sha256: [11; 32],
        },
        samples: vec![ProfileSampleV4 {
            source_record: 7,
            accepted_ordinal: 11,
            measured_unix_ns: 1000,
            shape: ProfileWaveShapeV4::try_from(&shape()).unwrap(),
            boundary: ProfileCostBoundaryV4::PreparationToHostSettledV1,
            outcome: ProfileObservationOutcome::Completed {},
            timing: ProfileWaveTiming {
                wall_total_ns: 100,
                device_elapsed_ns: None,
                stages: Default::default(),
            },
        }],
    }
}
fn load(
    value: &impl Serialize,
    limits: &CostProfileLoadLimits,
) -> Result<LoadedCostProfile, CostProfileError> {
    load_cost_profile_bytes(
        &serde_json::to_vec(value).unwrap(),
        &file().fingerprint.into(),
        &settings(),
        limits,
        ProfileLoadClock {
            wall_unix_ns: Some(1200),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: 0,
        },
    )
}

#[test]
fn real_v4_loader_trains_original_age_and_never_refreshes_on_import() {
    let loaded = load(&file(), &Default::default()).unwrap();
    assert_eq!(loaded.provenance.schema_version, 4);
    assert_eq!(loaded.provenance.oldest_imported_age_ns, Some(200));
    assert_eq!(loaded.provenance.counts.recorded_samples, 1);
    let fingerprint = file().fingerprint.into();
    let CostPrediction::Known(p) = loaded.snapshot.predict(
        &fingerprint,
        &shape(),
        CostBoundary::PreparationToHostSettledV1,
        0,
    ) else {
        panic!("actual V4 sample not imported")
    };
    assert_eq!(p.valid_for_ns, 800);
    assert_eq!(
        loaded.snapshot.predict(
            &fingerprint,
            &shape(),
            CostBoundary::PreparationToHostSettledV1,
            801
        ),
        CostPrediction::Unknown(CostUnknownReason::StaleSamples)
    );
}

#[test]
fn old_schema_relabel_missing_rows_and_extra_fields_cannot_invent_v2_evidence() {
    for version in [2, 3] {
        let mut value = file();
        value.schema_version = version;
        assert!(load(&value, &Default::default()).is_err());
    }
    let mut value = file();
    value.settings.feature_model = CostFeatureModel::EmpiricalHostContentV1 {
        host_history_bucket_tokens: NonZeroU32::new(64).unwrap(),
    };
    assert!(load(&value, &Default::default()).is_err());
    for pointer in [
        "/samples/0/shape/row_multiset_features",
        "/samples/0/shape/row_multiset_features/rows/0",
    ] {
        let mut value = serde_json::to_value(file()).unwrap();
        value
            .pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("invented".into(), true.into());
        assert!(
            load(&value, &Default::default()).is_err(),
            "accepted undeclared {pointer}"
        );
    }
    let mut value = file();
    value.samples[0].shape.row_multiset_features.rows[0].role = HostRowRoleV2::Prefill;
    assert!(load(&value, &Default::default()).is_err());
    let mut value = serde_json::to_value(file()).unwrap();
    value["samples"][0]["shape"]
        .as_object_mut()
        .unwrap()
        .remove("row_multiset_features");
    assert!(load(&value, &Default::default()).is_err());
}

#[test]
fn static_rows_are_in_import_budget_and_failed_observations_are_not_cost_samples() {
    let mut limits = CostProfileLoadLimits::default();
    limits.max_total_shape_rows = NonZeroUsize::new(2).unwrap();
    assert!(matches!(
        load(&file(), &limits),
        Err(CostProfileError::Limit(_))
    ));
    limits.max_total_shape_rows = NonZeroUsize::new(3).unwrap();
    assert!(load(&file(), &limits).is_ok());
    let mut value = file();
    value.samples[0].outcome = ProfileObservationOutcome::FailedAfterSubmit {};
    let loaded = load(&value, &limits).unwrap();
    assert_eq!(loaded.provenance.counts.recorded_samples, 0);
    assert_eq!(
        loaded.provenance.counts.skipped_samples[&ProfileSkippedObservation::FailedAfterSubmit],
        1
    );
    let mut value = file();
    let mut duplicate = value.samples[0].clone();
    duplicate.source_record += 1;
    value.samples.push(duplicate);
    assert!(load(&value, &Default::default()).is_err());
}
