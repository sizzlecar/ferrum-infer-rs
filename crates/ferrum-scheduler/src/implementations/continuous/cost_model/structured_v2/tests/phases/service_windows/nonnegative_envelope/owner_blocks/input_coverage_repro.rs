//! Tiny numerical diagnostic for a legitimate pre-execution mask flag.
//! This canonical-producer fixture is not a GPU upload or private live receipt.
use super::*;

pub(super) fn mask_input(required: bool) -> StructuredInputV2 {
    let wave = wave_with_history_domain_and_mask(
        9,
        true,
        8,
        "fixture.first",
        [false; 2],
        [3; 32],
        [4; 32],
        64,
        2,
        HostContentDomainV1::PlainTextGreedyV1,
        [required, false],
    );
    let selected = wave.statistical.as_ref().unwrap();
    let input = StructuredInputV2::from_actual_with_domain(
        &wave.exact,
        selected,
        selected.structured_capture().unwrap().unwrap(),
        &domain(),
    )
    .unwrap()
    .with_cost_template_policy(StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1)
    .unwrap()
    .with_settled_terminal_causes(&[])
    .unwrap();
    assert_eq!(input.physical_host_rows()[0].mask_upload_required, required);
    assert!(!input.physical_host_rows()[1].mask_upload_required);
    input
}

fn mask_population(
    phase: StructuredPhaseV2,
    first_requires_mask: bool,
) -> Vec<StructuredNumericObservationV2> {
    let mut samples = original_block_population(phase);
    for (index, sample) in samples.iter_mut().enumerate() {
        sample.input = mask_input(first_requires_mask && index == 0);
        sample.wall_ns = 1000;
    }
    samples
}

fn fit_mask_population(samples: &[StructuredNumericObservationV2]) -> FittedStructuredModelV2 {
    let mut contract = physical_owner_contract();
    contract
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .template_policy = StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1;
    FittedStructuredModelV2::fit_owner_blocks(
        fp(),
        settings(),
        scope(samples),
        contract,
        block_close(StructuredPhaseV2::Fit, 2, 2, None, samples),
        samples,
    )
    .unwrap()
}

#[test]
fn owner_blocks_current_count_readiness_late_mask_direction_cannot_qualify() {
    let clean = mask_input(false);
    let upload = mask_input(true);
    assert_eq!(clean.owner(), upload.owner());
    assert_eq!(clean.domain_signature(), upload.domain_signature());
    assert!(
        clean
            .basis
            .iter()
            .zip(&upload.basis)
            .any(|(a, b)| *a == 0. && *b > 0.),
        "the original producer projects the mask input into a new positive work direction"
    );

    let fit = mask_population(StructuredPhaseV2::Fit, false);
    let residual = mask_population(StructuredPhaseV2::Residual, true);
    let fitted = fit_mask_population(&fit);
    assert_eq!(
        fitted.service_input_membership(&residual[0].input).unwrap(),
        StructuredServiceInputMembershipV1::OutsideFitSupport
    );
    assert!(residual.iter().skip(1).all(|sample| fitted
        .service_input_membership(&sample.input)
        .unwrap()
        == StructuredServiceInputMembershipV1::Eligible));
    let before: Vec<_> = residual
        .iter()
        .map(|sample| {
            (
                sample.membership.offered_ordinal,
                sample.ordinal,
                sample.call_id,
                sample.wall_ns,
            )
        })
        .collect();
    let close = block_close(
        StructuredPhaseV2::Residual,
        3,
        3,
        Some(fitted.parameters_signature()),
        &residual,
    );
    assert!(
        matches!(
            fitted.calibrate_owner_blocks(close, &residual),
            Err(StructuredUnknown::UnidentifiedDirection)
        ),
        "the full original Residual population is evaluated, not filtered by membership"
    );
    let after: Vec<_> = residual
        .iter()
        .map(|sample| {
            (
                sample.membership.offered_ordinal,
                sample.ordinal,
                sample.call_id,
                sample.wall_ns,
            )
        })
        .collect();
    assert_eq!(after, before);
}

#[test]
fn owner_blocks_mask_repro_control_without_new_direction_completes_all_phases() {
    let fit = mask_population(StructuredPhaseV2::Fit, false);
    let residual = mask_population(StructuredPhaseV2::Residual, false);
    let heldout = mask_population(StructuredPhaseV2::Qualification, false);
    let fitted = fit_mask_population(&fit);
    let close = block_close(
        StructuredPhaseV2::Residual,
        3,
        3,
        Some(fitted.parameters_signature()),
        &residual,
    );
    let calibrated = fitted.calibrate_owner_blocks(close, &residual).unwrap();
    let close = block_close(
        StructuredPhaseV2::Qualification,
        4,
        4,
        Some(calibrated.parameters_signature()),
        &heldout,
    );
    let qualified = calibrated.qualify_owner_blocks(close, &heldout).unwrap();
    assert_eq!(
        qualified.source_contract().phase_members,
        [fit.len(), residual.len(), heldout.len()]
    );
    assert!(qualified
        .predict_query(&fp(), &StructuredQueryV2::exact(mask_input(false)), 1600)
        .is_ok());
}

#[test]
fn owner_blocks_mask_work_present_in_each_fresh_phase_supports_future_uploads() {
    let fit = mask_population(StructuredPhaseV2::Fit, true);
    let residual = mask_population(StructuredPhaseV2::Residual, true);
    let heldout = mask_population(StructuredPhaseV2::Qualification, true);
    let fitted = fit_mask_population(&fit);
    let close = block_close(
        StructuredPhaseV2::Residual,
        3,
        3,
        Some(fitted.parameters_signature()),
        &residual,
    );
    let calibrated = fitted.calibrate_owner_blocks(close, &residual).unwrap();
    let close = block_close(
        StructuredPhaseV2::Qualification,
        4,
        4,
        Some(calibrated.parameters_signature()),
        &heldout,
    );
    let qualified = calibrated.qualify_owner_blocks(close, &heldout).unwrap();
    assert_eq!(
        qualified.source_contract().phase_members,
        [fit.len(), residual.len(), heldout.len()]
    );
    for required in [false, true] {
        assert!(qualified
            .predict_query(&fp(), &StructuredQueryV2::exact(mask_input(required)), 1600)
            .is_ok());
    }
}

#[test]
fn owner_blocks_missing_residual_mask_work_cannot_publish_fit_only_coverage() {
    let fit = mask_population(StructuredPhaseV2::Fit, true);
    let residual = mask_population(StructuredPhaseV2::Residual, false);
    let fitted = fit_mask_population(&fit);
    let close = block_close(
        StructuredPhaseV2::Residual,
        3,
        3,
        Some(fitted.parameters_signature()),
        &residual,
    );
    assert!(matches!(
        fitted.calibrate_owner_blocks(close, &residual),
        Err(StructuredUnknown::QualificationCoverage)
    ));
}
