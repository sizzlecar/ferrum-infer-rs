use super::*;

fn contract() -> StructuredOwnerPhaseContractV1 {
    let mut c = declared(true);
    c.nonnegative_envelope.as_mut().unwrap().planning_estimator =
        NonNegativePlanningEstimatorV1::IdentifiedFitGlobalResidualV1;
    c.schedule.input_readiness =
        Some(OwnerInputReadinessV1::new_fit_target_v4([3; 3], 32_000_000).unwrap());
    c
}

fn fit(c: StructuredOwnerPhaseContractV1) -> FittedStructuredModelV2 {
    let f = rows(StructuredPhaseV2::Fit, 2, 2, 0, true);
    FittedStructuredModelV2::fit_owner_blocks(
        fp(),
        settings(),
        scope(&f),
        c,
        block_close(StructuredPhaseV2::Fit, 2, 3, None, &f),
        &f,
    )
    .unwrap()
}

#[test]
fn global_residual_keeps_identified_certificate_and_has_independent_binding() {
    let new = contract();
    let mut old = new.clone();
    old.nonnegative_envelope
        .as_mut()
        .unwrap()
        .planning_estimator = NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2;
    let old_model = fit(old);
    let model = fit(new.clone());
    assert_eq!(
        serde_json::to_value(model.nonnegative_fit_certificate()).unwrap(),
        serde_json::to_value(old_model.nonnegative_fit_certificate()).unwrap()
    );
    assert_ne!(
        model.parameters_signature(),
        old_model.parameters_signature()
    );
    let f = rows(StructuredPhaseV2::Fit, 2, 2, 0, true);
    let replay = FittedStructuredModelV2::fit_owner_blocks_from_certificate(
        fp(),
        settings(),
        scope(&f),
        new,
        block_close(StructuredPhaseV2::Fit, 2, 3, None, &f),
        &f,
        model.nonnegative_fit_certificate().unwrap().clone(),
    )
    .unwrap();
    assert_eq!(model.parameters_signature(), replay.parameters_signature());
    let mut predictions = Vec::new();
    for fitted in [model, replay] {
        let r = rows(StructuredPhaseV2::Residual, 4, 2, 32, true);
        let close = block_close(
            StructuredPhaseV2::Residual,
            4,
            5,
            Some(fitted.parameters_signature()),
            &r,
        );
        let calibrated = fitted.calibrate_owner_blocks(close, &r).unwrap();
        let q = rows(StructuredPhaseV2::Qualification, 6, 2, 64, true);
        let close = block_close(
            StructuredPhaseV2::Qualification,
            6,
            7,
            Some(calibrated.parameters_signature()),
            &q,
        );
        let model = calibrated.qualify_owner_blocks(close, &q).unwrap();
        let query = StructuredQueryV2::exact(mask_input(true));
        let p = model.predict_query(&fp(), &query, 3000).unwrap();
        assert_eq!(model.source_contract().phase_members, [32; 3]);
        assert_eq!(p.fitted_upper_ns, 1000);
        assert_eq!(p.residual_ns, 0);
        assert_eq!(p.fit_error_floor_ns, 0);
        assert_eq!(p.planning_ns, 1000 + settings().static_margin_ns);
        assert!(matches!(
            model.predict_query(&fp(), &query, p.valid_until_ns + 1),
            Err(StructuredUnknown::Stale)
        ));
        predictions.push(p.planning_ns);
    }
    assert_eq!(predictions[0], predictions[1]);
}

#[test]
fn global_residual_v4_requires_fit_directions_before_r_and_q_freeze() {
    let c = contract();
    let fitted = fit(c.clone());
    let target =
        OwnerInputTargetV1::from_samples(&rows(StructuredPhaseV2::Fit, 2, 2, 0, true)).unwrap();
    for phase in [
        StructuredPhaseV2::Residual,
        StructuredPhaseV2::Qualification,
    ] {
        let narrow = rows(phase, 4, 1, 32, false);
        let ready = rows(phase, 4, 2, 32, true);
        let assess = |samples: &[StructuredNumericObservationV2], offered| {
            c.schedule
                .assess_inputs(phase, offered, samples, Some(&target), &settings(), &mut 0)
                .unwrap()
        };
        assert_eq!(assess(&narrow, 32), OwnerInputReadinessDecisionV1::Wait);
        assert_eq!(assess(&ready, 64), OwnerInputReadinessDecisionV1::Freeze);
        let mut changed_walls = ready.clone();
        for sample in &mut changed_walls {
            sample.wall_ns = u64::MAX;
        }
        assert_eq!(assess(&changed_walls, 64), assess(&ready, 64));
        let missing_forever = rows(phase, 4, 3, 32, false);
        assert_eq!(
            assess(&missing_forever, 96),
            OwnerInputReadinessDecisionV1::Exhausted(
                OwnerInputReadinessGapV1::MissingInputCoverage
            )
        );
    }
    let narrow = rows(StructuredPhaseV2::Residual, 4, 1, 32, false);
    // Independent close validation cannot accept an early count-only close.
    let close = block_close(
        StructuredPhaseV2::Residual,
        4,
        4,
        Some(fitted.parameters_signature()),
        &narrow,
    );
    assert!(matches!(
        fitted.calibrate_owner_blocks(close, &narrow),
        Err(StructuredUnknown::IncompletePhasePopulation)
    ));
}

#[test]
fn global_residual_full_q_underestimate_fails_without_refitting_or_retry() {
    let fitted = fit(contract());
    let r = rows(StructuredPhaseV2::Residual, 4, 2, 32, true);
    let close = block_close(
        StructuredPhaseV2::Residual,
        4,
        5,
        Some(fitted.parameters_signature()),
        &r,
    );
    let calibrated = fitted.calibrate_owner_blocks(close, &r).unwrap();
    let mut q = rows(StructuredPhaseV2::Qualification, 6, 2, 64, true);
    q[0].wall_ns = 1_001_000;
    assert!(q
        .iter()
        .all(|s| calibrated.service_input_membership(&s.input).unwrap()
            == StructuredServiceInputMembershipV1::Eligible));
    let close = block_close(
        StructuredPhaseV2::Qualification,
        6,
        7,
        Some(calibrated.parameters_signature()),
        &q,
    );
    assert!(matches!(
        calibrated.qualify_owner_blocks(close, &q),
        Err(StructuredUnknown::QualificationUnderestimate)
    ));
}

#[test]
fn global_residual_v4_preserves_v3_fit_semantics_and_capacity() {
    let mut v3 = declared_v2();
    v3.schedule.input_readiness =
        Some(OwnerInputReadinessV1::new_zero_column_v3([3; 3], 32_000_000).unwrap());
    let mut v4 = v3.clone();
    v4.schedule.input_readiness =
        Some(OwnerInputReadinessV1::new_fit_target_v4([3; 3], 32_000_000).unwrap());
    let f = rows(StructuredPhaseV2::Fit, 2, 1, 0, false);
    let assess = |c: &StructuredOwnerPhaseContractV1| {
        let mut visits = 0;
        let decision = c
            .schedule
            .assess_inputs(
                StructuredPhaseV2::Fit,
                32,
                &f,
                c.input_target.as_ref(),
                &settings(),
                &mut visits,
            )
            .unwrap();
        (
            decision,
            visits,
            c.schedule.readiness_scratch_bytes(&f).unwrap(),
        )
    };
    assert_eq!(assess(&v3), assess(&v4));
    assert_eq!(assess(&v4).0, OwnerInputReadinessDecisionV1::Freeze);
    assert_eq!(
        v3.schedule.maximum_phase_members,
        v4.schedule.maximum_phase_members
    );
    assert_eq!(
        serde_json::to_value(v3.schedule.input_readiness).unwrap()["revision"],
        "work_axes_and_branches_v3"
    );
    assert_eq!(
        serde_json::to_value(v4.schedule.input_readiness).unwrap()["revision"],
        "work_axes_and_branches_v4"
    );
}
