use super::*;
use crate::implementations::continuous::cost_model::statistical::model::{
    tests::{fingerprint, independent_rows::independent_sample, partition, settings},
    ModelUnknown,
};
fn file() -> CostProfileFileV7 {
    let fit = (1..=8)
        .map(|n| independent_sample(n, [false, true, false]))
        .collect::<Vec<_>>();
    let residual = (9..=16)
        .map(|n| independent_sample(n, [true, false, false]))
        .collect::<Vec<_>>();
    let frozen = FittedWholeWaveModelV1::fit_independent_attention_v2(
        fingerprint(),
        settings(),
        partition(),
        &fit,
        108,
    )
    .unwrap();
    CostProfileFileV7::from_capture_observations(
        &fingerprint(),
        &settings(),
        partition(),
        ProfileSource {
            generator: "rust-fixture".into(),
            generator_revision: "v2".into(),
            measurement_protocol: "independent-attention-fit-residual-v2".into(),
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
fn load(value: &CostProfileFileV7) -> Result<ImportedWholeWaveModelV1, CostProfileError> {
    load_whole_wave_profile_v7_bytes(
        &value.to_bounded_bytes(1_048_576).unwrap(),
        &fingerprint(),
        &settings(),
        &CostProfileLoadLimits::default(),
        clock(),
    )
}
#[test]
fn profile7_refits_permuted_independent_populations_and_preserves_import_age() {
    let file = file();
    let model = load(&file).unwrap();
    assert_eq!(
        model.selected_family(),
        SelectedStatisticalFamily::IndependentAttentionV2
    );
    let query = independent_sample(17, [false, false, true]);
    let p = model
        .predict(&fingerprint(), &query.exact, &query.selected, 100)
        .unwrap();
    assert_eq!((p.fit_samples, p.residual_samples), (8, 8));
    assert_eq!(p.valid_until_ns - model.clock.model_anchor_ns, 10_000 - 34);
    assert_eq!(model.source_sha256, [99; 32]);
    assert_eq!(model.capture_identity_sha256, [7; 32]);
    assert!(matches!(
        model.predict(
            &fingerprint(),
            &query.exact,
            &query.selected,
            100 + 10_000 - 33
        ),
        Err(ModelUnknown::Stale)
    ));
    let local_expiry = 100 + p.valid_until_ns - model.clock.model_anchor_ns;
    for now in [99, 100, local_expiry, local_expiry + 1] {
        let identified =
            model.predict_identified(&fingerprint(), &query.exact, &query.selected, now);
        assert_eq!(
            identified.prediction,
            model.predict(&fingerprint(), &query.exact, &query.selected, now)
        );
        if now == 99 {
            assert_eq!(identified.query_identity, None);
        } else {
            let identity = identified.query_identity.unwrap();
            assert_eq!(identity.family_schema_version, 2);
            assert_eq!(
                identity.family_signature,
                *query
                    .selected
                    .independent_attention_v2()
                    .unwrap()
                    .family_signature()
            );
        }
    }
}
#[test]
fn profile7_rejects_legacy_relabel_missing_binding_work_or_changed_freeze() {
    let original = file();
    let bytes = original.to_bounded_bytes(1_048_576).unwrap();
    assert!(matches!(
        super::super::statistical_v6::load_whole_wave_profile_v6_bytes(
            &bytes,
            &fingerprint(),
            &settings(),
            &CostProfileLoadLimits::default(),
            clock()
        ),
        Err(CostProfileError::UnsupportedVersion(7))
    ));
    let mut wrong = original.clone();
    wrong.schema_version = 6;
    assert!(matches!(
        load(&wrong),
        Err(CostProfileError::UnsupportedVersion(6))
    ));
    let mut wrong = original.clone();
    wrong.model_revision =
        super::super::super::cost_model::statistical::model::MODEL_REVISION.into();
    assert!(load(&wrong).is_err());
    let mut wrong = original.clone();
    wrong.fit_parameters_sha256[0] ^= 1;
    assert!(load(&wrong).is_err());
    for kind in ["missing", "join", "work", "extra"] {
        let mut value = serde_json::to_value(&original).unwrap();
        match kind {
            "missing" => {
                value["samples"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("independent_attention");
            }
            "join" => {
                let byte = value["samples"][0]["independent_attention"]["exact_binding"][0]
                    .as_u64()
                    .unwrap();
                value["samples"][0]["independent_attention"]["exact_binding"][0] =
                    (byte ^ 1).into();
            }
            "work" => {
                value["samples"][0]["independent_attention"]["work"]["grid_blocks"] = 999.into();
            }
            _ => {
                value["samples"][0]["independent_attention"]["guess_from_v1"] = true.into();
            }
        }
        assert!(
            load_whole_wave_profile_v7_bytes(
                &serde_json::to_vec(&value).unwrap(),
                &fingerprint(),
                &settings(),
                &CostProfileLoadLimits::default(),
                clock()
            )
            .is_err(),
            "{kind}"
        );
    }
}
