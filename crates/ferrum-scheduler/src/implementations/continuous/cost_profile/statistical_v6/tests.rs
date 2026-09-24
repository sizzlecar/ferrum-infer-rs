use super::*;
use crate::implementations::continuous::cost_model::statistical::model::tests::{
    fingerprint, partition, populations, sample, settings,
};

fn file() -> CostProfileFileV6 {
    let (fit, residual) = populations();
    CostProfileFileV6::from_observations(
        &fingerprint(),
        &settings(),
        partition(),
        ProfileSource {
            generator: "rust-test".into(),
            generator_revision: "fixture-v1".into(),
            measurement_protocol: "fit-residual-heldout".into(),
            observation_artifact_sha256: partition().source_sha256,
        },
        120,
        1_000_000,
        2,
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
fn load(file: &CostProfileFileV6) -> Result<ImportedWholeWaveModelV1, CostProfileError> {
    load_whole_wave_profile_v6_bytes(
        &file.to_bounded_bytes(1_048_576).unwrap(),
        &fingerprint(),
        &settings(),
        &CostProfileLoadLimits::default(),
        clock(),
    )
}
#[test]
fn roundtrip_refits_then_calibrates_and_import_clock_preserves_age() {
    let file = file();
    let imported = load(&file).unwrap();
    let q = sample(17, 12, 130);
    let p = imported
        .predict(&fingerprint(), &q.exact, &q.selected, 100)
        .unwrap();
    assert_eq!(p.planning_ns, 125);
    assert_eq!(imported.fit_records, 8);
    assert_eq!(imported.residual_records, 8);
    assert_eq!(imported.conservative_clock_error_ns, 5);
    // Oldest sample is generated time - 19; import adds elapsed 10 plus error 5.
    assert_eq!(
        p.valid_until_ns - imported.clock.model_anchor_ns,
        10_000 - 34
    );
    let local_expiry = 100 + 10_000 - 34;
    assert!(imported
        .predict(&fingerprint(), &q.exact, &q.selected, local_expiry)
        .is_ok());
    assert_eq!(
        imported.predict(&fingerprint(), &q.exact, &q.selected, local_expiry + 1),
        Err(ModelUnknown::Stale)
    );
    assert_eq!(
        imported.predict(&fingerprint(), &q.exact, &q.selected, 99),
        Err(ModelUnknown::Clock)
    );
}
#[test]
fn legacy_versions_settings_and_exact_join_cannot_be_relabelled() {
    let original = file();
    assert!(matches!(
        super::super::load_cost_profile_bytes(
            &original.to_bounded_bytes(1_048_576).unwrap(),
            &fingerprint(),
            &CostModelSettings::default(),
            &CostProfileLoadLimits::default(),
            clock()
        ),
        Err(CostProfileError::UnsupportedVersion(6))
    ));
    for version in 1..=5 {
        let mut old = original.clone();
        old.schema_version = version;
        assert!(matches!(load(&old),Err(CostProfileError::UnsupportedVersion(v)) if v==version));
    }
    let mut wrong = original.clone();
    wrong.fingerprint.device_runtime = [99; 32];
    assert!(matches!(
        load(&wrong),
        Err(CostProfileError::FingerprintMismatch)
    ));
    let mut wrong = original.clone();
    wrong.settings.min_samples = NonZeroUsize::new(7).unwrap();
    assert!(matches!(
        load(&wrong),
        Err(CostProfileError::SettingsMismatch)
    ));
    let mut wrong = original.clone();
    wrong.samples[0].shape.provider_signature = [98; 32];
    assert!(matches!(load(&wrong), Err(CostProfileError::Metadata(_))));
    let mut wrong = original.clone();
    wrong.samples[8].call_id = wrong.samples[0].call_id;
    assert!(matches!(load(&wrong), Err(CostProfileError::Metadata(_))));
    let mut wrong = original;
    wrong.samples[8].phase = WholeWaveProfilePhaseV6::Fit;
    assert!(matches!(load(&wrong), Err(CostProfileError::Metadata(_))));
}
#[test]
fn expired_failed_missing_selected_and_unknown_nested_fields_are_rejected() {
    let mut wrong = file();
    wrong.samples[0].measured_unix_ns -= 10_000;
    assert!(matches!(load(&wrong), Err(CostProfileError::Clock(_))));
    let mut wrong = file();
    wrong.samples[0].outcome = ProfileObservationOutcome::FailedAfterSubmit {};
    assert!(matches!(load(&wrong), Err(CostProfileError::Metadata(_))));
    let original = serde_json::to_value(file()).unwrap();
    for path in ["missing", "unknown"] {
        let mut value = original.clone();
        if path == "missing" {
            value["samples"][0]
                .as_object_mut()
                .unwrap()
                .remove("selected");
        } else {
            value["samples"][0]["selected"]["work"]["invented_counter"] = 1.into();
        }
        assert!(load_whole_wave_profile_v6_bytes(
            &serde_json::to_vec(&value).unwrap(),
            &fingerprint(),
            &settings(),
            &CostProfileLoadLimits::default(),
            clock()
        )
        .is_err());
    }
}
#[test]
fn file_byte_and_shape_limits_are_enforced_on_real_serialized_records() {
    let file = file();
    let bytes = file.to_bounded_bytes(1_048_576).unwrap();
    assert_eq!(file.to_bounded_bytes(bytes.len()).unwrap(), bytes);
    assert!(file.to_bounded_bytes(bytes.len() - 1).is_err());
    let mut limits = CostProfileLoadLimits::default();
    limits.max_file_bytes = NonZeroUsize::new(bytes.len() - 1).unwrap();
    assert!(matches!(
        load_whole_wave_profile_v6_bytes(&bytes, &fingerprint(), &settings(), &limits, clock()),
        Err(CostProfileError::Limit(_))
    ));
    limits.max_file_bytes = NonZeroUsize::new(bytes.len()).unwrap();
    limits.max_total_shape_rows = NonZeroUsize::new(15).unwrap();
    assert!(matches!(
        load_whole_wave_profile_v6_bytes(&bytes, &fingerprint(), &settings(), &limits, clock()),
        Err(CostProfileError::Limit(_))
    ));
}
