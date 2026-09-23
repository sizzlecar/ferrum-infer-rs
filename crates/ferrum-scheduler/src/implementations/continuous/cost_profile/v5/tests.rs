use super::*;
use crate::implementations::continuous::cost_model::prompt_range_tests::{
    fingerprint, settings, shape,
};

fn file() -> CostProfileFileV5 {
    CostProfileFileV5 {
        schema_version: COST_PROFILE_SCHEMA_VERSION_V5,
        fingerprint: (&fingerprint()).into(),
        settings: (&settings()).into(),
        generated_unix_ns: 1100,
        source_clock_max_error_ns: Some(0),
        source: ProfileSource {
            generator: "typed prompt-range fixture".into(),
            generator_revision: "test revision".into(),
            measurement_protocol: "complete actual host-settled samples".into(),
            observation_artifact_sha256: [11; 32],
        },
        samples: [64, 128]
            .into_iter()
            .enumerate()
            .map(|(index, total)| ProfileSampleV5 {
                source_record: 7 + index as u64,
                accepted_ordinal: 11 + index as u64,
                measured_unix_ns: 1000 + index as u64,
                shape: ProfileWaveShapeV5::try_from(&shape(0, total)).unwrap(),
                boundary: ProfileCostBoundaryV5::PreparationToHostSettledV1,
                outcome: ProfileObservationOutcome::Completed {},
                timing: ProfileWaveTiming {
                    wall_total_ns: 90 + index as u64 * 20,
                    device_elapsed_ns: None,
                    stages: Default::default(),
                },
            })
            .collect(),
    }
}
fn load(
    value: &impl Serialize,
    limits: &CostProfileLoadLimits,
) -> Result<LoadedCostProfile, CostProfileError> {
    load_cost_profile_bytes(
        &serde_json::to_vec(value).unwrap(),
        &fingerprint(),
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
fn real_loader_builds_prompt_support_from_original_totals_and_original_age() {
    let original = file();
    let loaded = load(&original, &Default::default()).unwrap();
    assert_eq!(loaded.provenance.schema_version, 5);
    assert_eq!(loaded.provenance.counts.recorded_samples, 2);
    assert_eq!(loaded.provenance.oldest_imported_age_ns, Some(200));
    let query = shape(0, 96);
    let predict = |now| {
        loaded.snapshot.predict(
            &fingerprint(),
            &query,
            CostBoundary::PreparationToHostSettledV1,
            now,
        )
    };
    let CostPrediction::Known(p) = predict(0) else {
        panic!("V5 support not imported")
    };
    assert_eq!(
        (p.sample_count, p.planning_ns, p.valid_for_ns),
        (2, 115, 800)
    );
    assert_eq!(
        predict(801),
        CostPrediction::Unknown(CostUnknownReason::StaleSamples)
    );
    assert_eq!(
        original.samples[0].shape.exact.exact.prefill_chunks[0]
            .total_prompt_tokens
            .get(),
        64
    );
    assert_eq!(
        original.samples[1].shape.exact.exact.prefill_chunks[0]
            .total_prompt_tokens
            .get(),
        128
    );
}
#[test]
fn schema_and_model_are_explicit_and_v4_cannot_silently_enable_prompt_pooling() {
    for version in [2, 3, 4] {
        let mut value = file();
        value.schema_version = version;
        assert!(
            load(&value, &Default::default()).is_err(),
            "relabel accepted schema {version}"
        );
    }
    for mode in [
        CostFeatureModel::ExactV1 {},
        CostFeatureModel::EmpiricalRowMultisetV2 {
            host_history_bucket_tokens: NonZeroU32::new(64).unwrap(),
        },
    ] {
        let mut value = file();
        value.settings.feature_model = mode;
        assert!(load(&value, &Default::default()).is_err());
    }
    let bytes = serde_json::to_vec(&file()).unwrap();
    let mut old_settings = settings();
    old_settings.feature_model = CostFeatureModel::EmpiricalRowMultisetV2 {
        host_history_bucket_tokens: NonZeroU32::new(64).unwrap(),
    };
    assert!(load_cost_profile_bytes(
        &bytes,
        &fingerprint(),
        &old_settings,
        &Default::default(),
        ProfileLoadClock {
            wall_unix_ns: Some(1200),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: 0
        }
    )
    .is_err());
}
#[test]
fn strict_import_keeps_limits_original_work_and_unique_source_join() {
    let mut limits = CostProfileLoadLimits::default();
    limits.max_total_shape_rows = NonZeroUsize::new(5).unwrap();
    assert!(matches!(
        load(&file(), &limits),
        Err(CostProfileError::Limit(_))
    ));
    limits.max_total_shape_rows = NonZeroUsize::new(6).unwrap();
    assert!(load(&file(), &limits).is_ok());
    let mut duplicate = file();
    duplicate.samples[1].accepted_ordinal = duplicate.samples[0].accepted_ordinal;
    assert!(load(&duplicate, &limits).is_err());
    let mut invalid = file();
    invalid.samples[0].shape.exact.exact.prefill_chunks[0].total_prompt_tokens =
        NonZeroU32::new(31).unwrap();
    assert!(load(&invalid, &limits).is_err());
    let mut extra = serde_json::to_value(file()).unwrap();
    extra["samples"][0]["shape"]["row_multiset_features"]["rows"][0]
        .as_object_mut()
        .unwrap()
        .insert("invented".into(), true.into());
    assert!(load(&extra, &limits).is_err());
}
