use super::independent_rows::independent_sample_with_state;
use super::*;

pub(crate) fn work_sample(n: u64, maximum: u64) -> WholeWaveObservationV1 {
    independent_sample_with_state(n, [false, true, false], 1, maximum, 0)
}
pub(crate) fn populations() -> (Vec<WholeWaveObservationV1>, Vec<WholeWaveObservationV1>) {
    (
        (1..=8).map(|n| work_sample(n, 16)).collect(),
        (9..=16).map(|n| work_sample(n, 16)).collect(),
    )
}
#[test]
fn work_support_changes_only_nonterminal_budget_membership_and_model_identity() {
    let (fit, residual) = populations();
    let old = FittedWholeWaveModelV1::fit_independent_attention_v2(
        fingerprint(),
        settings(),
        partition(),
        &fit,
        108,
    )
    .unwrap();
    let new = FittedWholeWaveModelV1::fit_work_support_v1(
        fingerprint(),
        settings(),
        partition(),
        &fit,
        108,
    )
    .unwrap();
    assert_ne!(old.parameter_signature(), new.parameter_signature());
    let old = old.calibrate(&residual, 116).unwrap();
    let new = new.calibrate(&residual, 116).unwrap();
    let reference = work_sample(17, 16);
    let baseline = old
        .predict(&fingerprint(), &reference.exact, &reference.selected, 117)
        .unwrap();
    for maximum in [8, 32] {
        let query = work_sample(17, maximum);
        assert_ne!(query.exact, reference.exact);
        assert_eq!(query.selected.work(), reference.selected.work());
        assert_eq!(
            query
                .selected
                .independent_attention_v2()
                .unwrap()
                .family_signature(),
            reference
                .selected
                .independent_attention_v2()
                .unwrap()
                .family_signature()
        );
        assert_eq!(
            query.exact.numeric_features.as_ref().unwrap().rows[0].maximum_output_tokens,
            maximum
        );
        assert_eq!(
            old.predict(&fingerprint(), &query.exact, &query.selected, 117),
            Err(ModelUnknown::JointSupport)
        );
        let identified = new.predict_identified(&fingerprint(), &query.exact, &query.selected, 117);
        assert_eq!(identified.prediction.unwrap(), baseline);
        let id = identified.query_identity.unwrap();
        assert_eq!(id.model_revision, WORK_SUPPORT_MODEL_REVISION);
        assert_eq!(id.family_schema_version, 2);
    }
}
#[test]
fn work_support_keeps_terminal_real_work_fingerprint_and_expiry_gates() {
    let (fit, residual) = populations();
    let model = FittedWholeWaveModelV1::fit_work_support_v1(
        fingerprint(),
        settings(),
        partition(),
        &fit,
        108,
    )
    .unwrap()
    .calibrate(&residual, 116)
    .unwrap();
    let terminal = work_sample(17, 2);
    assert_eq!(
        model.predict(&fingerprint(), &terminal.exact, &terminal.selected, 117),
        Err(ModelUnknown::FamilyMissing)
    );
    let larger = independent_sample_with_state(17, [false, true, false], 1, 32, 1);
    assert_eq!(
        model.predict(&fingerprint(), &larger.exact, &larger.selected, 117),
        Err(ModelUnknown::JointSupport)
    );
    let query = work_sample(17, 32);
    let mut wrong = fingerprint();
    wrong.execution_config[0] ^= 1;
    assert_eq!(
        model.predict(&wrong, &query.exact, &query.selected, 117),
        Err(ModelUnknown::WrongFingerprint)
    );
    assert_eq!(
        model.predict(&fingerprint(), &query.exact, &query.selected, 10_102),
        Err(ModelUnknown::Stale)
    );
}
#[test]
fn work_support_residual_is_independent_and_keeps_the_original_sample_counts() {
    let (fit, _) = populations();
    let residual: Vec<_> = (9..=16).map(|n| work_sample(n, 32)).collect();
    let old = FittedWholeWaveModelV1::fit_independent_attention_v2(
        fingerprint(),
        settings(),
        partition(),
        &fit,
        108,
    )
    .unwrap();
    assert!(matches!(
        old.calibrate(&residual, 116),
        Err(ModelUnknown::InsufficientResidual)
    ));
    let new = FittedWholeWaveModelV1::fit_work_support_v1(
        fingerprint(),
        settings(),
        partition(),
        &fit,
        108,
    )
    .unwrap();
    let sealed = new.parameter_signature();
    assert!(matches!(
        new.clone().calibrate(&residual[..7], 116),
        Err(ModelUnknown::InsufficientResidual)
    ));
    assert_eq!(new.parameter_signature(), sealed);
    let model = new.calibrate(&residual, 116).unwrap();
    let query = work_sample(17, 24);
    let prediction = model
        .predict(&fingerprint(), &query.exact, &query.selected, 117)
        .unwrap();
    assert_eq!(
        (prediction.fit_samples, prediction.residual_samples),
        (8, 8)
    );
    assert_eq!(prediction.valid_until_ns, 10_101);
}
