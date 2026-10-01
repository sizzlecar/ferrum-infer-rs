use super::super::super::owner_blocks::{block_close, block_contract};
use super::*;

mod input_coverage_repro;
mod input_readiness;
mod phase_support;

fn original_block_population(p: StructuredPhaseV2) -> Vec<StructuredNumericObservationV2> {
    let mut rows = population(p, true);
    for (i, row) in rows.iter_mut().enumerate() {
        let ticket = (p.index() as u64 + 1) * 32 + i as u64 + 1;
        row.membership.offered_ordinal = ticket;
        row.ordinal = ticket * 3;
        row.call_id = ticket + 100;
        row.observed_at_ns = ticket * 10;
    }
    rows
}

fn physical_owner_contract() -> StructuredOwnerPhaseContractV1 {
    let mut c = block_contract();
    c.domain_policy = StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1;
    c.nonnegative_envelope = physical_contract().nonnegative_envelope;
    c
}

#[test]
fn owner_blocks_nonnegative_certificate_replay_matches_live_numerical_transitions() {
    let fit = original_block_population(StructuredPhaseV2::Fit);
    let fit_close = block_close(StructuredPhaseV2::Fit, 2, 2, None, &fit);
    let fitted = FittedStructuredModelV2::fit_owner_blocks(
        fp(),
        settings(),
        scope(&fit),
        physical_owner_contract(),
        fit_close.clone(),
        &fit,
    )
    .unwrap();
    let signature = fitted.parameters_signature();
    let certificate = fitted.nonnegative_fit_certificate().unwrap().clone();
    let replay = FittedStructuredModelV2::fit_owner_blocks_from_certificate(
        fp(),
        settings(),
        scope(&fit),
        physical_owner_contract(),
        fit_close.clone(),
        &fit,
        certificate.clone(),
    )
    .unwrap();
    assert_eq!(replay.parameters_signature(), signature);

    let mut changed = fit.clone();
    changed[0].wall_ns += 1;
    assert!(FittedStructuredModelV2::fit_owner_blocks_from_certificate(
        fp(),
        settings(),
        scope(&changed),
        physical_owner_contract(),
        fit_close,
        &changed,
        certificate,
    )
    .is_err());

    let residual = original_block_population(StructuredPhaseV2::Residual);
    assert_eq!(
        fitted.service_input_membership(&residual[0].input).unwrap(),
        StructuredServiceInputMembershipV1::Eligible,
    );
    let finish = |f: FittedStructuredModelV2| {
        let close = block_close(
            StructuredPhaseV2::Residual,
            3,
            3,
            Some(f.parameters_signature()),
            &residual,
        );
        let c = f.calibrate_owner_blocks(close, &residual).unwrap();
        let heldout = original_block_population(StructuredPhaseV2::Qualification);
        assert_eq!(
            c.service_input_membership(&heldout[0].input).unwrap(),
            StructuredServiceInputMembershipV1::Eligible
        );
        let close = block_close(
            StructuredPhaseV2::Qualification,
            4,
            4,
            Some(c.parameters_signature()),
            &heldout,
        );
        c.qualify_owner_blocks(close, &heldout).unwrap()
    };
    let original = finish(fitted);
    let replayed = finish(replay);
    assert_eq!(
        original.parameters_signature(),
        replayed.parameters_signature()
    );
    assert_eq!(
        original.owner_block_contract(),
        Some(&physical_owner_contract())
    );
    let first = original.predict_query(&fp(), &future(), 1600).unwrap();
    let second = replayed.predict_query(&fp(), &future(), 1600).unwrap();
    assert_eq!(first.planning_ns, second.planning_ns);
    assert_eq!(first.valid_until_ns, 10_000);
}
