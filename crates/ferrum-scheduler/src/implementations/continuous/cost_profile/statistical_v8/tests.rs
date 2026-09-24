use super::*;
use crate::implementations::continuous::cost_model::statistical::model::{
    tests::{
        fingerprint, partition, settings,
        work_support::{populations, work_sample},
    },
    FittedWholeWaveModelV1, ModelUnknown, WORK_SUPPORT_MODEL_REVISION,
};
fn file() -> CostProfileFileV8 {
    let (fit, residual) = populations();
    let frozen = FittedWholeWaveModelV1::fit_work_support_v1(
        fingerprint(),
        settings(),
        partition(),
        &fit,
        108,
    )
    .unwrap();
    CostProfileFileV8::from_capture_observations(
        &fingerprint(),
        &settings(),
        partition(),
        ProfileSource {
            generator: "real-rust-contract-fixture".into(),
            generator_revision: "work-support.v1".into(),
            measurement_protocol: "fit-then-independent-residual".into(),
            observation_artifact_sha256: [99; 32],
        },
        120,
        1_000_000,
        2,
        frozen.parameter_signature(),
        &fit,
        &residual,
    )
    .unwrap()
}
fn clock() -> ProfileLoadClock {
    ProfileLoadClock {
        wall_unix_ns: Some(1_000_010),
        wall_max_error_ns: Some(3),
        monotonic_now_ns: 100,
    }
}
fn load(bytes: &[u8]) -> Result<ImportedWholeWaveModelV1, CostProfileError> {
    load_whole_wave_profile_v8_bytes(
        bytes,
        &fingerprint(),
        &settings(),
        &CostProfileLoadLimits::default(),
        clock(),
    )
}
#[test]
fn profile8_roundtrip_keeps_budget_metadata_v2_family_and_imported_age() {
    let file = file();
    let bytes = file.to_bounded_bytes(1_048_576).unwrap();
    assert_eq!(file.to_bounded_bytes(bytes.len()).unwrap(), bytes);
    assert!(file.to_bounded_bytes(bytes.len() - 1).is_err());
    let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(wire["schema_version"], 8);
    assert_eq!(wire["model_revision"], WORK_SUPPORT_MODEL_REVISION);
    assert_eq!(
        wire["samples"][0]["shape"]["numeric_features"]["rows"][0]["maximum_output_tokens"],
        16
    );
    let imported = load(&bytes).unwrap();
    assert_eq!(
        imported.model_revision(),
        WholeWaveModelRevision::IndependentAttentionWorkSupportV1
    );
    let query = work_sample(17, 32);
    let prediction =
        imported.predict_identified(&fingerprint(), &query.exact, &query.selected, 100);
    let id = prediction.query_identity.unwrap();
    assert_eq!(id.model_revision, WORK_SUPPORT_MODEL_REVISION);
    assert_eq!(id.family_schema_version, 2);
    let value = prediction.prediction.unwrap();
    assert_eq!((value.fit_samples, value.residual_samples), (8, 8));
    assert_eq!(
        value.valid_until_ns - imported.clock.model_anchor_ns,
        10_000 - 34
    );
    assert_eq!(imported.source_sha256, [99; 32]);
    assert_eq!(
        imported.predict(
            &fingerprint(),
            &query.exact,
            &query.selected,
            100 + 10_000 - 33
        ),
        Err(ModelUnknown::Stale)
    );
    assert_eq!(
        imported
            .predict_identified(&fingerprint(), &query.exact, &query.selected, 99)
            .query_identity,
        None
    );
}
#[test]
fn profile8_rejects_schema7_relabel_changed_fit_or_missing_real_v2_evidence() {
    let original = file();
    let bytes = original.to_bounded_bytes(1_048_576).unwrap();
    assert!(matches!(
        super::super::statistical_v7::load_whole_wave_profile_v7_bytes(
            &bytes,
            &fingerprint(),
            &settings(),
            &CostProfileLoadLimits::default(),
            clock()
        ),
        Err(CostProfileError::UnsupportedVersion(8))
    ));
    for fault in ["schema", "revision", "fit", "missing", "binding"] {
        let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        match fault {
            "schema" => value["schema_version"]=7.into(),
            "revision" => value["model_revision"]=serde_json::Value::String(crate::implementations::continuous::cost_model::statistical::model::INDEPENDENT_ATTENTION_MODEL_REVISION.into()),
            "fit" => { let n=value["fit_parameters_sha256"][0].as_u64().unwrap(); value["fit_parameters_sha256"][0]=(n^1).into(); },
            "missing" => { value["samples"][0].as_object_mut().unwrap().remove("independent_attention"); },
            _ => { let n=value["samples"][0]["independent_attention"]["exact_binding"][0].as_u64().unwrap(); value["samples"][0]["independent_attention"]["exact_binding"][0]=(n^1).into(); },
        }
        assert!(
            load(&serde_json::to_vec(&value).unwrap()).is_err(),
            "{fault}"
        );
    }
    // Even changing both labels cannot import an old frozen fit under new semantics.
    let (fit, residual) = populations();
    let old = FittedWholeWaveModelV1::fit_independent_attention_v2(
        fingerprint(),
        settings(),
        partition(),
        &fit,
        108,
    )
    .unwrap();
    assert!(CostProfileFileV8::from_capture_observations(
        &fingerprint(),
        &settings(),
        partition(),
        ProfileSource {
            generator: "fixture".into(),
            generator_revision: "old".into(),
            measurement_protocol: "old".into(),
            observation_artifact_sha256: [99; 32]
        },
        120,
        1_000_000,
        2,
        old.parameter_signature(),
        &fit,
        &residual
    )
    .is_err());
}
