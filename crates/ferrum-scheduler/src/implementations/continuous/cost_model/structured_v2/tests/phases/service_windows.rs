use super::*;
mod frozen_domain;
mod nonnegative_envelope;

fn window_contract() -> StructuredServiceWindowContractV2 {
    let old = contract();
    StructuredServiceWindowContractV2 {
        capture_identity: old.capture_identity,
        protocol: old.protocol,
        membership_rule: old.membership_rule,
        window_declaration: [73; 32],
        phase_offered: [64; 3],
        domain_policy: StructuredServiceDomainPolicyV1::AllOffered,
        nonnegative_envelope: None,
    }
}
fn samples(
    phase_kind: StructuredPhaseV2,
    count: usize,
    member_before: usize,
) -> Vec<StructuredNumericObservationV2> {
    let original = phase(phase_kind);
    let first = phase_kind.index() as u64 * 64;
    (0..count)
        .map(|i| {
            let mut sample = original[i % original.len()].clone();
            let ticket = first + i as u64 + 1;
            sample.ordinal = ticket * 3;
            sample.membership.offered_ordinal = ticket;
            sample.membership.member_ordinal = (member_before + i + 1) as u64;
            sample.call_id = ticket + 100;
            sample.observed_at_ns = ticket * 10;
            sample
        })
        .collect()
}
fn close(
    phase: StructuredPhaseV2,
    samples: &[StructuredNumericObservationV2],
) -> StructuredServiceWindowCloseV2 {
    StructuredServiceWindowCloseV2::new(phase, [phase.index() as u8 + 21; 32], samples)
}
fn window_fitted() -> FittedStructuredModelV2 {
    let fit = samples(StructuredPhaseV2::Fit, 16, 0);
    FittedStructuredModelV2::fit_service_window(
        fp(),
        settings(),
        scope(&fit),
        window_contract(),
        close(StructuredPhaseV2::Fit, &fit),
        &fit,
        170,
    )
    .unwrap()
}
fn window_calibrated() -> CalibratedStructuredModelV2 {
    let residual = samples(StructuredPhaseV2::Residual, 20, 16);
    window_fitted()
        .calibrate_service_window(
            close(StructuredPhaseV2::Residual, &residual),
            &residual,
            850,
        )
        .unwrap()
}

#[test]
fn service_window_independent_counts_preserve_each_freeze_and_original_ttl() {
    let fitted = window_fitted();
    let fit_signature = fitted.parameters_signature();
    let residual = samples(StructuredPhaseV2::Residual, 20, 16);
    let calibrated = fitted
        .calibrate_service_window(
            close(StructuredPhaseV2::Residual, &residual),
            &residual,
            850,
        )
        .unwrap();
    let residual_signature = calibrated.parameters_signature();
    let qualification = samples(StructuredPhaseV2::Qualification, 24, 36);
    let qualified = calibrated
        .qualify_service_window(
            close(StructuredPhaseV2::Qualification, &qualification),
            &qualification,
            1530,
        )
        .unwrap();
    assert_eq!(
        qualified.service_window_phase_signatures(),
        Some([
            fit_signature,
            residual_signature,
            qualified.parameters_signature()
        ])
    );
    assert_eq!(qualified.source_contract().phase_members, [16, 20, 24]);
    assert_eq!(
        qualified.service_window_contract(),
        Some(&window_contract())
    );
    let query = StructuredQueryV2::exact(qualification[0].input.clone());
    let p = qualified.predict_query(&fp(), &query, 1600).unwrap();
    assert_eq!(p.valid_until_ns, 10_010);
    assert_eq!(
        (
            p.fit_samples,
            p.residual_samples,
            qualified.qualification_samples
        ),
        (16, 20, 24)
    );
    assert!(matches!(
        qualified.predict_query(&fp(), &query, 10_011),
        Err(StructuredUnknown::Stale)
    ));
    assert_ne!(fit_signature, super::fitted().parameters_signature());
}

#[test]
fn service_window_frozen_population_cannot_drop_or_replace_ticket_or_use_old_transition() {
    let residual = samples(StructuredPhaseV2::Residual, 20, 16);
    let frozen_close = close(StructuredPhaseV2::Residual, &residual);
    let mut changed = residual.clone();
    changed.remove(3);
    assert!(matches!(
        window_fitted().calibrate_service_window(frozen_close.clone(), &changed, 850),
        Err(StructuredUnknown::IncompletePhasePopulation)
    ));
    let mut changed = residual.clone();
    changed.last_mut().unwrap().membership.offered_ordinal += 1;
    assert!(matches!(
        window_fitted().calibrate_service_window(frozen_close, &changed, 850),
        Err(StructuredUnknown::IncompletePhasePopulation)
    ));
    assert!(matches!(
        window_fitted().calibrate(&residual, 850),
        Err(StructuredUnknown::WrongProtocol)
    ));
    assert!(matches!(
        super::fitted().calibrate_service_window(
            close(StructuredPhaseV2::Residual, &residual),
            &residual,
            850
        ),
        Err(StructuredUnknown::WrongProtocol)
    ));
    let mut changed = residual;
    changed[0].membership.offered_ordinal = 64;
    assert!(matches!(
        window_fitted().calibrate_service_window(
            close(StructuredPhaseV2::Residual, &changed),
            &changed,
            850
        ),
        Err(StructuredUnknown::PhaseLeakage)
    ));
}

#[test]
fn service_window_qualification_does_not_learn_and_limits_remain_closed() {
    let qual = samples(StructuredPhaseV2::Qualification, 24, 36);
    let original = window_calibrated().parameters_signature();
    let qualified = window_calibrated()
        .qualify_service_window(close(StructuredPhaseV2::Qualification, &qual), &qual, 1530)
        .unwrap();
    let query = StructuredQueryV2::exact(qual[0].input.clone());
    let bound = qualified
        .predict_query(&fp(), &query, 1600)
        .unwrap()
        .planning_ns;
    let mut slow = qual.clone();
    slow[0].wall_ns = bound + 1;
    assert!(matches!(
        window_calibrated().qualify_service_window(
            close(StructuredPhaseV2::Qualification, &slow),
            &slow,
            1530
        ),
        Err(StructuredUnknown::QualificationUnderestimate)
    ));
    assert_eq!(window_calibrated().parameters_signature(), original);
    assert!(matches!(
        window_calibrated().qualify_service_window(
            close(StructuredPhaseV2::Qualification, &qual),
            &qual,
            10_011
        ),
        Err(StructuredUnknown::Stale)
    ));
    let mut declaration = window_contract();
    declaration.phase_offered[2] += 1;
    let fit = samples(StructuredPhaseV2::Fit, 16, 0);
    let changed = FittedStructuredModelV2::fit_service_window(
        fp(),
        settings(),
        scope(&fit),
        declaration,
        close(StructuredPhaseV2::Fit, &fit),
        &fit,
        170,
    )
    .unwrap();
    assert_ne!(
        changed.parameters_signature(),
        window_fitted().parameters_signature()
    );
}

#[test]
fn service_window_monotonic_decode_stream_cannot_extrapolate_later_kv_history() {
    let mut fit = samples(StructuredPhaseV2::Fit, 16, 0);
    for (i, sample) in fit.iter_mut().enumerate() {
        let w = wave_with_history(
            9,
            true,
            8,
            "fixture.first",
            [false, false],
            [3; 32],
            [4; 32],
            64 + i as u32,
            2 + i as u64,
        );
        sample.input = project(&w).unwrap();
        sample.wall_ns = 1000 + i as u64 * 10;
    }
    let fit_owner = fit[0].input.owner().clone();
    let fitted = FittedStructuredModelV2::fit_service_window(
        fp(),
        settings(),
        scope(&fit),
        window_contract(),
        close(StructuredPhaseV2::Fit, &fit),
        &fit,
        170,
    )
    .unwrap();
    let mut residual = samples(StructuredPhaseV2::Residual, 20, 16);
    for (i, sample) in residual.iter_mut().enumerate() {
        let w = wave_with_history(
            9,
            true,
            8,
            "fixture.first",
            [false, false],
            [3; 32],
            [4; 32],
            80 + i as u32,
            18 + i as u64,
        );
        sample.input = project(&w).unwrap();
        sample.wall_ns = 1160 + i as u64 * 10;
        assert_eq!(sample.input.owner(), &fit_owner);
    }
    assert!(matches!(
        fitted.calibrate_service_window(
            close(StructuredPhaseV2::Residual, &residual),
            &residual,
            850
        ),
        Err(StructuredUnknown::JointSupport)
    ));
}
