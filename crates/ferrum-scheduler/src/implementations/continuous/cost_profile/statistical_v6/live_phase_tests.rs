use super::*;
use crate::implementations::continuous::cost_model::statistical::model::tests as fixture;

#[test]
fn fit_is_frozen_before_residual_cut_without_predicting_the_future_file_hash() {
    let (fit, residual) = fixture::populations();
    let mut open = fixture::partition();
    open.residual_through_ordinal = u64::MAX;
    let frozen =
        FittedWholeWaveModelV1::fit(fixture::fingerprint(), fixture::settings(), open, &fit, 108)
            .unwrap();
    let signature = frozen.parameter_signature();
    let sealed = frozen.seal_residual_cut(16).unwrap();
    assert_eq!(sealed.parameter_signature(), signature);
    let model = sealed.calibrate(&residual, 120).unwrap();
    let sample = fixture::sample(17, 12, 130);
    assert_eq!(
        model
            .predict(&sample.fingerprint, &sample.exact, &sample.selected, 120)
            .unwrap()
            .planning_ns,
        125
    );

    let source = ProfileSource {
        generator: "rust-live-capture-fixture".into(),
        generator_revision: "v1".into(),
        measurement_protocol: "fit then residual then heldout".into(),
        observation_artifact_sha256: [91; 32],
    };
    let file = CostProfileFileV6::from_capture_observations(
        &fixture::fingerprint(),
        &fixture::settings(),
        fixture::partition(),
        source,
        120,
        1_000_000,
        0,
        signature,
        &fit,
        &residual,
    )
    .unwrap();
    assert_ne!(
        file.capture_identity_sha256,
        file.source.observation_artifact_sha256
    );
    let loaded = load_whole_wave_profile_v6_bytes(
        &file.to_bounded_bytes(1_048_576).unwrap(),
        &fixture::fingerprint(),
        &fixture::settings(),
        &CostProfileLoadLimits::default(),
        ProfileLoadClock {
            monotonic_now_ns: 100,
            wall_unix_ns: Some(1_000_010),
            wall_max_error_ns: Some(0),
        },
    )
    .unwrap();
    assert_eq!(loaded.source_sha256, [91; 32]);
    assert_eq!(
        loaded.capture_identity_sha256,
        fixture::partition().source_sha256
    );
    assert_eq!(loaded.fit_parameters_sha256, signature);
    assert_eq!(
        loaded
            .predict(&sample.fingerprint, &sample.exact, &sample.selected, 100)
            .unwrap()
            .planning_ns,
        125
    );
}

#[test]
fn residual_cannot_expand_its_cut_or_rewrite_frozen_fit() {
    let (fit, mut residual) = fixture::populations();
    let frozen = fixture::fitted();
    assert!(matches!(
        frozen.clone().seal_residual_cut(17),
        Err(ModelUnknown::PhaseLeakage)
    ));
    assert!(matches!(
        frozen.clone().seal_residual_cut(8),
        Err(ModelUnknown::PhaseLeakage)
    ));
    assert!(matches!(
        frozen
            .clone()
            .seal_residual_cut(15)
            .unwrap()
            .calibrate(&residual, 120),
        Err(ModelUnknown::PhaseLeakage)
    ));
    let signature = frozen.parameter_signature();
    residual[0].wall_ns = 900;
    assert_eq!(fixture::fitted().parameter_signature(), signature);
    let mut altered_fit = fit.clone();
    altered_fit[0].wall_ns = 999;
    assert!(matches!(
        CostProfileFileV6::from_capture_observations(
            &fixture::fingerprint(),
            &fixture::settings(),
            fixture::partition(),
            ProfileSource {
                generator: "fixture".into(),
                generator_revision: "v1".into(),
                measurement_protocol: "independent".into(),
                observation_artifact_sha256: [91; 32]
            },
            120,
            1_000_000,
            0,
            signature,
            &altered_fit,
            &residual,
        ),
        Err(CostProfileError::Metadata(
            "fit changed after its pre-residual freeze"
        ))
    ));
}
