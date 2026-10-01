use super::*;

fn domain_contract() -> StructuredServiceWindowContractV2 {
    StructuredServiceWindowContractV2 {
        domain_policy: StructuredServiceDomainPolicyV1::FrozenFitSupportV1,
        ..window_contract()
    }
}

fn history_samples(
    phase: StructuredPhaseV2,
    histories: &[u64],
    preceding_members: usize,
) -> Vec<StructuredNumericObservationV2> {
    let mut rows = samples(phase, 16 * histories.len(), preceding_members);
    for (index, row) in rows.iter_mut().enumerate() {
        let pattern = index % 16;
        let generated = histories[index / 16];
        row.input = project(&wave_with_history(
            [9, 0, 1, 2][pattern / 4],
            true,
            8,
            "fixture.first",
            [pattern & 1 != 0, pattern & 2 != 0],
            [3; 32],
            [4; 32],
            64 + generated as u32,
            generated,
        ))
        .unwrap();
        row.wall_ns = 1000
            + generated * 10
            + row.input.pending_positions.len() as u64 * 13
            + row.input.length_positions.len() as u64 * 7;
    }
    rows
}

fn domain_fitted() -> FittedStructuredModelV2 {
    let fit = history_samples(StructuredPhaseV2::Fit, &[2, 8], 0);
    FittedStructuredModelV2::fit_service_window(
        fp(),
        settings(),
        scope(&fit),
        domain_contract(),
        close(StructuredPhaseV2::Fit, &fit),
        &fit,
        330,
    )
    .unwrap()
}

fn domain_calibrated() -> CalibratedStructuredModelV2 {
    let residual = history_samples(StructuredPhaseV2::Residual, &[2, 6], 32);
    domain_fitted()
        .calibrate_service_window(
            close(StructuredPhaseV2::Residual, &residual),
            &residual,
            970,
        )
        .unwrap()
}

#[test]
fn frozen_input_membership_distinguishes_fit_and_residual_support_without_wall_time() {
    let fitted = domain_fitted();
    let within = history_samples(StructuredPhaseV2::Residual, &[4], 32);
    let outside = history_samples(StructuredPhaseV2::Residual, &[9], 32);
    assert_eq!(
        fitted.service_input_membership(&within[0].input).unwrap(),
        StructuredServiceInputMembershipV1::Eligible
    );
    assert_eq!(
        fitted.service_input_membership(&outside[0].input).unwrap(),
        StructuredServiceInputMembershipV1::OutsideFitSupport
    );
    assert_eq!(
        fitted
            .diagnose_service_fit_support(&within[0].input)
            .unwrap(),
        None
    );
    assert!(matches!(
        fitted
            .diagnose_service_fit_support(&outside[0].input)
            .unwrap()
            .unwrap()
            .reason,
        StructuredFitSupportReasonV1::AboveAllFitMax { .. }
    ));
    // Real canonical work is coordinatewise inside the observed support, but
    // a new KV/history ratio is outside the frozen identified row space.
    let unidentifiable = project(&wave_with_history(
        9,
        true,
        8,
        "fixture.first",
        [false, false],
        [3; 32],
        [4; 32],
        67,
        4,
    ))
    .unwrap();
    let mut unidentified_samples = within.clone();
    for sample in &mut unidentified_samples {
        sample.input = unidentifiable.clone();
    }
    assert!(matches!(
        domain_fitted().calibrate_service_window(
            close(StructuredPhaseV2::Residual, &unidentified_samples),
            &unidentified_samples,
            970
        ),
        Err(StructuredUnknown::UnidentifiedDirection)
    ));
    assert_eq!(
        fitted.service_input_membership(&unidentifiable).unwrap(),
        StructuredServiceInputMembershipV1::OutsideFitSupport
    );
    assert!(matches!(
        fitted
            .diagnose_service_fit_support(&unidentifiable)
            .unwrap()
            .unwrap()
            .reason,
        StructuredFitSupportReasonV1::UnidentifiedDirection { .. }
    ));
    let mut changed_cost = within[0].clone();
    changed_cost.wall_ns = u64::MAX;
    assert_eq!(
        fitted
            .service_input_membership(&changed_cost.input)
            .unwrap(),
        StructuredServiceInputMembershipV1::Eligible
    );
    let calibrated = domain_calibrated();
    let residual_outside = history_samples(StructuredPhaseV2::Qualification, &[7], 64);
    assert_eq!(
        calibrated
            .service_input_membership(&residual_outside[0].input)
            .unwrap(),
        StructuredServiceInputMembershipV1::OutsideResidualSupport
    );
    assert_eq!(
        calibrated
            .diagnose_service_fit_support(&residual_outside[0].input)
            .unwrap(),
        None
    );
    let mut foreign = within[0].input.clone();
    foreign.owner.installed_policy = [91; 32];
    assert!(matches!(
        fitted.service_input_membership(&foreign),
        Err(StructuredUnknown::WrongDomain)
    ));
    assert!(matches!(
        fitted.diagnose_service_fit_support(&foreign),
        Err(StructuredUnknown::WrongDomain)
    ));
    let mut invalid = within[0].input.clone();
    invalid.basis[0] = f64::NAN;
    assert!(matches!(
        fitted.service_input_membership(&invalid),
        Err(StructuredUnknown::InvalidInput)
    ));
    assert!(matches!(
        fitted.diagnose_service_fit_support(&invalid),
        Err(StructuredUnknown::InvalidInput)
    ));
}

#[test]
fn frozen_domain_publication_is_also_bounded_by_heldout_inputs_and_original_expiry() {
    let qual = history_samples(StructuredPhaseV2::Qualification, &[2, 4], 64);
    let qualified = domain_calibrated()
        .qualify_service_window(close(StructuredPhaseV2::Qualification, &qual), &qual, 1610)
        .unwrap();
    let query = StructuredQueryV2::exact(qual[0].input.clone());
    let result = qualified.predict_query(&fp(), &query, 1620).unwrap();
    assert_eq!(result.valid_until_ns, 10_010);
    assert_eq!(
        (
            result.fit_samples,
            result.residual_samples,
            qualified.qualification_samples
        ),
        (32, 32, 32)
    );
    let outside_heldout = history_samples(StructuredPhaseV2::Qualification, &[5], 64);
    assert!(matches!(
        qualified.predict_query(
            &fp(),
            &StructuredQueryV2::exact(outside_heldout[0].input.clone()),
            1620
        ),
        Err(StructuredUnknown::JointSupport)
    ));
    assert!(matches!(
        qualified.predict_query(&fp(), &query, 10_011),
        Err(StructuredUnknown::Stale)
    ));
    assert!(qualified.retained_payload_bytes().unwrap() > std::mem::size_of_val(&qualified));
}

#[test]
fn eligible_heldout_underestimate_cannot_be_hidden_by_narrowing_publication() {
    let mut qual = history_samples(StructuredPhaseV2::Qualification, &[2, 4], 64);
    qual.last_mut().unwrap().wall_ns = 50_000;
    assert!(matches!(
        domain_calibrated().qualify_service_window(
            close(StructuredPhaseV2::Qualification, &qual),
            &qual,
            1610
        ),
        Err(StructuredUnknown::QualificationUnderestimate)
    ));
    let mut too_few = history_samples(StructuredPhaseV2::Residual, &[2], 32);
    too_few.truncate(1);
    assert!(matches!(
        domain_fitted().calibrate_service_window(
            close(StructuredPhaseV2::Residual, &too_few),
            &too_few,
            970
        ),
        Err(StructuredUnknown::IncompletePhasePopulation)
    ));
}

#[test]
fn frozen_domain_contract_is_explicit_and_cannot_reinterpret_default_membership() {
    let old = window_contract();
    let wire = serde_json::to_value(&old).unwrap();
    assert!(wire.get("domain_policy").is_none());
    assert_eq!(
        serde_json::from_value::<StructuredServiceWindowContractV2>(wire).unwrap(),
        old
    );
    let explicit = serde_json::to_value(domain_contract()).unwrap();
    assert_eq!(explicit["domain_policy"], "frozen_fit_support_v1");
    let mut unknown = explicit;
    unknown["domain_policy"] = serde_json::json!("future_unreviewed_policy");
    assert!(serde_json::from_value::<StructuredServiceWindowContractV2>(unknown).is_err());
    let fit = history_samples(StructuredPhaseV2::Fit, &[2, 8], 0);
    let legacy = FittedStructuredModelV2::fit_service_window(
        fp(),
        settings(),
        scope(&fit),
        old,
        close(StructuredPhaseV2::Fit, &fit),
        &fit,
        330,
    )
    .unwrap();
    assert_ne!(
        legacy.parameters_signature(),
        domain_fitted().parameters_signature()
    );
    assert!(matches!(
        legacy.service_input_membership(&fit[0].input),
        Err(StructuredUnknown::WrongProtocol)
    ));
}
