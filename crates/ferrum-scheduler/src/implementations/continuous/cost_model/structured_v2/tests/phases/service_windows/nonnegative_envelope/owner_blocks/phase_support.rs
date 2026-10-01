//! The new source7 declaration narrows input support, never measured costs.
//! All inputs come from the canonical physical producer; walls are synthetic.
use super::input_coverage_repro::mask_input;
use super::*;
mod global_residual;

fn declared_v2() -> StructuredOwnerPhaseContractV1 {
    let mut c = declared(true);
    c.schedule.phase_support = Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV2);
    c
}

#[test]
fn physical_phase_support_v2_fit_close_stays_input_only_earliest_and_bounded() {
    let c = declared_v2();
    let f = rows(StructuredPhaseV2::Fit, 2, 1, 0, false);
    let assess = |samples: &[StructuredNumericObservationV2], cfg: &StructuredSettingsV2| {
        let mut visits = 0;
        let decision = c
            .schedule
            .assess_inputs(
                StructuredPhaseV2::Fit,
                32,
                samples,
                c.input_target.as_ref(),
                cfg,
                &mut visits,
            )
            .unwrap();
        (decision, visits)
    };
    let original = assess(&f, &settings());
    assert_eq!(original.0, OwnerInputReadinessDecisionV1::Freeze);
    let mut changed_walls = f.clone();
    for sample in &mut changed_walls {
        sample.wall_ns = u64::MAX;
    }
    assert_eq!(assess(&changed_walls, &settings()), original);
    assert_eq!(
        assess(&f[..7], &settings()).0,
        OwnerInputReadinessDecisionV1::Wait
    );
    let mut insufficient = settings();
    insufficient.min_fit_redundancy = f.len();
    assert_eq!(
        assess(&f, &insufficient).0,
        OwnerInputReadinessDecisionV1::Wait
    );
    let mut exhausted = c.schedule.clone();
    exhausted.input_readiness = Some(OwnerInputReadinessV1::new([3; 3], 1).unwrap());
    assert_eq!(
        exhausted
            .assess_inputs(
                StructuredPhaseV2::Fit,
                32,
                &f,
                c.input_target.as_ref(),
                &settings(),
                &mut 0
            )
            .unwrap(),
        OwnerInputReadinessDecisionV1::Exhausted(OwnerInputReadinessGapV1::GeometryWorkBudget)
    );
    let mut undeclared_geometry = c.schedule.clone();
    undeclared_geometry.input_readiness = None;
    assert_eq!(
        undeclared_geometry.validate(&settings()),
        Err(StructuredUnknown::WrongProtocol)
    );
    let late = rows(StructuredPhaseV2::Fit, 2, 2, 0, false);
    assert!(matches!(
        FittedStructuredModelV2::fit_owner_blocks(
            fp(),
            settings(),
            scope(&late),
            c,
            block_close(StructuredPhaseV2::Fit, 2, 3, None, &late),
            &late
        ),
        Err(StructuredUnknown::PhaseLeakage)
    ));
}

#[test]
fn physical_phase_support_v2_replays_fit_and_keeps_qualification_failures() {
    let f = rows(StructuredPhaseV2::Fit, 2, 1, 0, false);
    let close = block_close(StructuredPhaseV2::Fit, 2, 2, None, &f);
    let fitted = FittedStructuredModelV2::fit_owner_blocks(
        fp(),
        settings(),
        scope(&f),
        declared_v2(),
        close.clone(),
        &f,
    )
    .unwrap();
    let replayed = FittedStructuredModelV2::fit_owner_blocks_from_certificate(
        fp(),
        settings(),
        scope(&f),
        declared_v2(),
        close,
        &f,
        fitted.nonnegative_fit_certificate().unwrap().clone(),
    )
    .unwrap();
    assert_eq!(
        fitted.parameters_signature(),
        replayed.parameters_signature()
    );
    assert_eq!(
        fitted.service_input_membership(&mask_input(true)).unwrap(),
        StructuredServiceInputMembershipV1::OutsideFitSupport
    );
    for (fit, bad_wall) in [(fitted, false), (replayed, true)] {
        let r = rows(StructuredPhaseV2::Residual, 3, 1, 16, false);
        let close = block_close(
            StructuredPhaseV2::Residual,
            3,
            3,
            Some(fit.parameters_signature()),
            &r,
        );
        let calibrated = fit.calibrate_owner_blocks(close, &r).unwrap();
        let mut q = rows(StructuredPhaseV2::Qualification, 4, 1, 32, false);
        if bad_wall {
            q[0].wall_ns = 1_001_000;
        }
        assert!(q
            .iter()
            .all(|s| calibrated.service_input_membership(&s.input).unwrap()
                == StructuredServiceInputMembershipV1::Eligible));
        let close = block_close(
            StructuredPhaseV2::Qualification,
            4,
            4,
            Some(calibrated.parameters_signature()),
            &q,
        );
        let result = calibrated.qualify_owner_blocks(close, &q);
        if bad_wall {
            assert!(matches!(
                result,
                Err(StructuredUnknown::QualificationUnderestimate)
            ));
        } else {
            let model = result.unwrap();
            assert_eq!(model.source_contract().phase_members, [16; 3]);
            let clean = StructuredQueryV2::exact(mask_input(false));
            let absent = StructuredQueryV2::exact(mask_input(true));
            assert_eq!(model.catalog_input_membership(&clean), Ok(Some(true)));
            assert_eq!(model.catalog_input_membership(&absent), Ok(Some(false)));
            assert_eq!(
                model
                    .predict_query(&fp(), &clean, 1500)
                    .unwrap()
                    .valid_until_ns,
                10_000
            );
            assert!(matches!(
                model.predict_query(&fp(), &absent, 1500),
                Err(StructuredUnknown::QualificationCoverage)
            ));
        }
    }
}

fn rows(
    phase: StructuredPhaseV2,
    first: u64,
    blocks: usize,
    prior: usize,
    mask: bool,
) -> Vec<StructuredNumericObservationV2> {
    let template = original_block_population(phase).remove(0);
    (0..blocks * 16)
        .map(|i| {
            let mut s = template.clone();
            let ticket = (first - 1 + (i / 16) as u64) * 32 + (i % 16) as u64 + 1;
            s.membership.offered_ordinal = ticket;
            s.membership.member_ordinal = (prior + i + 1) as u64;
            s.ordinal = ticket * 3;
            s.call_id = ticket + 100;
            s.observed_at_ns = ticket * 10;
            s.input = mask_input(mask && i == 16);
            s.wall_ns = 1000;
            s
        })
        .collect()
}

fn declared(enabled: bool) -> StructuredOwnerPhaseContractV1 {
    let mut c = physical_owner_contract();
    c.nonnegative_envelope.as_mut().unwrap().template_policy =
        StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1;
    c.schedule = OwnerBlockScheduleV1::new_with_input_readiness(
        32,
        [32; 3],
        [8; 3],
        OwnerInputReadinessV1::new([3; 3], 32_000_000).unwrap(),
    )
    .unwrap();
    c.schedule.phase_support =
        enabled.then_some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV1);
    let mut target = OwnerInputTargetV1::from_input(&mask_input(false)).unwrap();
    target.observe(&mask_input(true)).unwrap();
    c.input_target = Some(target);
    c
}

fn fitted(enabled: bool) -> FittedStructuredModelV2 {
    let f = rows(StructuredPhaseV2::Fit, 2, 2, 0, true);
    FittedStructuredModelV2::fit_owner_blocks(
        fp(),
        settings(),
        scope(&f),
        declared(enabled),
        block_close(StructuredPhaseV2::Fit, 2, 3, None, &f),
        &f,
    )
    .unwrap()
}

fn calibrated() -> CalibratedStructuredModelV2 {
    let f = fitted(true);
    let r = rows(StructuredPhaseV2::Residual, 4, 1, 32, false);
    let close = block_close(
        StructuredPhaseV2::Residual,
        4,
        4,
        Some(f.parameters_signature()),
        &r,
    );
    f.calibrate_owner_blocks(close, &r).unwrap()
}

#[test]
fn physical_phase_support_intersection_keeps_three_phases_and_replays_original_fit() {
    let frows = rows(StructuredPhaseV2::Fit, 2, 2, 0, true);
    let f = fitted(true);
    let replay = FittedStructuredModelV2::fit_owner_blocks_from_certificate(
        fp(),
        settings(),
        scope(&frows),
        declared(true),
        block_close(StructuredPhaseV2::Fit, 2, 3, None, &frows),
        &frows,
        f.nonnegative_fit_certificate().unwrap().clone(),
    )
    .unwrap();
    assert_eq!(f.parameters_signature(), replay.parameters_signature());
    let finish = |f: FittedStructuredModelV2| {
        let r = rows(StructuredPhaseV2::Residual, 4, 1, 32, false);
        assert_eq!(
            f.service_input_membership(&mask_input(true)).unwrap(),
            StructuredServiceInputMembershipV1::Eligible
        );
        let rc = block_close(
            StructuredPhaseV2::Residual,
            4,
            4,
            Some(f.parameters_signature()),
            &r,
        );
        let c = f.calibrate_owner_blocks(rc, &r).unwrap();
        assert_eq!(
            c.service_input_membership(&mask_input(true)).unwrap(),
            StructuredServiceInputMembershipV1::OutsideResidualSupport
        );
        let q = rows(StructuredPhaseV2::Qualification, 5, 1, 48, false);
        let qc = block_close(
            StructuredPhaseV2::Qualification,
            5,
            5,
            Some(c.parameters_signature()),
            &q,
        );
        c.qualify_owner_blocks(qc, &q).unwrap()
    };
    let model = finish(f);
    let replayed = finish(replay);
    assert_eq!(
        model.parameters_signature(),
        replayed.parameters_signature()
    );
    assert_eq!(model.source_contract().phase_members, [32, 16, 16]);
    let clean = StructuredQueryV2::exact(mask_input(false));
    assert_eq!(model.catalog_input_membership(&clean), Ok(Some(true)));
    assert_eq!(replayed.catalog_input_membership(&clean), Ok(Some(true)));
    let masked = StructuredQueryV2::exact(mask_input(true));
    assert_eq!(model.catalog_input_membership(&masked), Ok(Some(false)));
    let mut wrong = clean.clone();
    wrong.input.physical_domain = Some([91; 32]);
    assert_eq!(
        model.catalog_input_membership(&wrong),
        Err(StructuredUnknown::WrongDomain)
    );
    let mut missing = clean.clone();
    missing.input.physical_domain = None;
    assert!(model.catalog_input_membership(&missing).is_err());
    let mut malformed = masked.clone();
    malformed.input.basis[0] = f64::NAN;
    assert_eq!(
        model.catalog_input_membership(&malformed),
        Err(StructuredUnknown::InvalidInput)
    );
    let a = model.predict_query(&fp(), &clean, 1700).unwrap();
    let b = replayed.predict_query(&fp(), &clean, 1700).unwrap();
    assert_eq!(
        (a.planning_ns, a.valid_until_ns),
        (b.planning_ns, b.valid_until_ns)
    );
    assert_eq!(
        a.valid_until_ns, 10_000,
        "publication cannot extend the old deadline"
    );
    assert!(matches!(
        model.predict_query(&fp(), &StructuredQueryV2::exact(mask_input(true)), 1700),
        Err(StructuredUnknown::QualificationCoverage)
    ));
    let mut unresolved = clean;
    unresolved.pending = Some(PendingQuery {
        eligible: vec![0, 1],
        constraint: HostPendingConstraintV2::AnySubset,
    });
    assert_eq!(model.catalog_input_membership(&unresolved), Ok(Some(false)));
    assert!(
        matches!(
            model.predict_query_detailed(&fp(), &unresolved, 1700),
            Err(StructuredQueryFailureV2::OutsideSupport(
                StructuredUnknown::QualificationCoverage
            ))
        ),
        "an unresolved 0-or-positive pending range needs both original branch challenges"
    );
}

#[test]
fn physical_phase_support_catalog_legacy_none_does_not_dispatch_on_coverage() {
    let f = fitted(false);
    let r = rows(StructuredPhaseV2::Residual, 4, 2, 32, true);
    let rc = block_close(
        StructuredPhaseV2::Residual,
        4,
        5,
        Some(f.parameters_signature()),
        &r,
    );
    let c = f.calibrate_owner_blocks(rc, &r).unwrap();
    let q = rows(StructuredPhaseV2::Qualification, 6, 2, 64, true);
    let qc = block_close(
        StructuredPhaseV2::Qualification,
        6,
        7,
        Some(c.parameters_signature()),
        &q,
    );
    let model = c.qualify_owner_blocks(qc, &q).unwrap();
    let mut query = StructuredQueryV2::exact(mask_input(false));
    assert_eq!(model.catalog_input_membership(&query), Ok(None));
    model.predict_query(&fp(), &query, 2400).unwrap();
    query.pending = Some(PendingQuery {
        eligible: vec![0, 1],
        constraint: HostPendingConstraintV2::AnySubset,
    });
    assert_eq!(model.catalog_input_membership(&query), Ok(None));
    assert!(matches!(
        model.predict_query(&fp(), &query, 2400),
        Err(StructuredUnknown::QualificationCoverage)
    ));
}

#[test]
fn physical_phase_support_preserves_legacy_binding_fit_readiness_and_counts() {
    let old = declared(false);
    let bytes = serde_json::to_vec(&old.schedule).unwrap();
    assert!(!String::from_utf8(bytes.clone())
        .unwrap()
        .contains("phase_support"));
    let decoded: OwnerBlockScheduleV1 = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(serde_json::to_vec(&decoded).unwrap(), bytes);
    assert_ne!(
        fitted(false).parameters_signature(),
        fitted(true).parameters_signature()
    );
    // The new declaration does not let the first clean-only Fit block pass.
    let f = rows(StructuredPhaseV2::Fit, 2, 1, 0, false);
    assert!(matches!(
        FittedStructuredModelV2::fit_owner_blocks(
            fp(),
            settings(),
            scope(&f),
            declared(true),
            block_close(StructuredPhaseV2::Fit, 2, 2, None, &f),
            &f
        ),
        Err(StructuredUnknown::IncompletePhasePopulation)
    ));
    let r = rows(StructuredPhaseV2::Residual, 4, 1, 32, false);
    let legacy = fitted(false);
    let close = block_close(
        StructuredPhaseV2::Residual,
        4,
        4,
        Some(legacy.parameters_signature()),
        &r,
    );
    assert!(matches!(
        legacy.calibrate_owner_blocks(close, &r),
        Err(StructuredUnknown::IncompletePhasePopulation)
    ));
    // Plenty of offered work cannot substitute for eight eligible members.
    let c = calibrated();
    let mut q = rows(StructuredPhaseV2::Qualification, 5, 1, 48, false);
    q.truncate(7);
    let close = block_close(
        StructuredPhaseV2::Qualification,
        5,
        5,
        Some(c.parameters_signature()),
        &q,
    );
    assert!(matches!(
        c.qualify_owner_blocks(close, &q),
        Err(StructuredUnknown::IncompletePhasePopulation)
    ));
}

#[test]
fn physical_phase_support_never_hides_eligible_underestimate_or_invalid_input() {
    let c = calibrated();
    let clean = mask_input(false);
    let mut wrong = clean.clone();
    wrong.physical_domain = Some([91; 32]);
    assert!(c.service_input_membership(&wrong).is_err());
    let mut corrupt = clean.clone();
    corrupt.basis[0] = f64::NAN;
    assert_eq!(
        c.service_input_membership(&corrupt),
        Err(StructuredUnknown::InvalidInput)
    );
    let mut q = rows(StructuredPhaseV2::Qualification, 5, 1, 48, false);
    q[0].wall_ns = 1_001_000;
    assert!(q
        .iter()
        .all(|s| c.service_input_membership(&s.input).unwrap()
            == StructuredServiceInputMembershipV1::Eligible));
    let qc = block_close(
        StructuredPhaseV2::Qualification,
        5,
        5,
        Some(c.parameters_signature()),
        &q,
    );
    assert!(matches!(
        c.qualify_owner_blocks(qc, &q),
        Err(StructuredUnknown::QualificationUnderestimate)
    ));
}

#[test]
fn physical_phase_support_eos_stop_continue_share_prospective_membership() {
    let mut contract = physical_owner_contract();
    contract.schedule.phase_support = Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV1);
    let f = original_block_population(StructuredPhaseV2::Fit);
    let make_fit = || {
        FittedStructuredModelV2::fit_owner_blocks(
            fp(),
            settings(),
            scope(&f),
            contract.clone(),
            block_close(StructuredPhaseV2::Fit, 2, 2, None, &f),
            &f,
        )
        .unwrap()
    };
    let model = make_fit();
    let r = original_block_population(StructuredPhaseV2::Residual);
    let f = make_fit();
    let rc = block_close(
        StructuredPhaseV2::Residual,
        3,
        3,
        Some(f.parameters_signature()),
        &r,
    );
    let c = f.calibrate_owner_blocks(rc, &r).unwrap();
    let q = original_block_population(StructuredPhaseV2::Qualification);
    let qc = block_close(
        StructuredPhaseV2::Qualification,
        4,
        4,
        Some(c.parameters_signature()),
        &q,
    );
    let qualified = c.qualify_owner_blocks(qc, &q).unwrap();
    let before = input(9, [false; 2], 64, 2);
    assert_eq!(
        model.service_input_membership(&before),
        Err(StructuredUnknown::MissingEvidence),
        "unsettled evidence is invalid, not an outside-support classification"
    );
    for causes in [
        vec![],
        vec![(0, FinishReason::EOS)],
        vec![(0, FinishReason::Stop)],
    ] {
        let original = before
            .clone()
            .with_settled_terminal_causes(&causes)
            .unwrap();
        assert_eq!(
            model.service_input_membership(&original).unwrap(),
            StructuredServiceInputMembershipV1::Eligible
        );
        assert_eq!(
            qualified.catalog_input_membership(&StructuredQueryV2::exact(original)),
            Ok(Some(true))
        );
    }
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
        HostContentDomainV1::PlainTextInstalledV2(PlainTextPolicyCapabilityV2 {
            sampling: PlainTextSamplingRouteV2::FullLogits,
            model_eos: true,
            user_stop: true,
        }),
        [true, false],
    );
    let selected = wave.statistical.as_ref().unwrap();
    let masked = StructuredInputV2::from_actual_with_domain(
        &wave.exact,
        selected,
        selected.structured_capture().unwrap().unwrap(),
        &domain(),
    )
    .unwrap();
    for causes in [
        vec![],
        vec![(0, FinishReason::EOS)],
        vec![(0, FinishReason::Stop)],
    ] {
        let original = masked
            .clone()
            .with_settled_terminal_causes(&causes)
            .unwrap();
        assert_eq!(
            model.service_input_membership(&original).unwrap(),
            StructuredServiceInputMembershipV1::OutsideFitSupport,
            "a real new input axis excludes the whole reachable outcome set"
        );
        assert_eq!(
            qualified.catalog_input_membership(&StructuredQueryV2::exact(original)),
            Ok(Some(false))
        );
    }
    // Independent residual without any early outcome still fails the original
    // branch challenge; intersection is not an EOS qualification exemption.
    let mut r = original_block_population(StructuredPhaseV2::Residual);
    let no_early = population(StructuredPhaseV2::Residual, false);
    for (row, plain) in r.iter_mut().zip(no_early) {
        row.input = plain.input;
    }
    let rc = block_close(
        StructuredPhaseV2::Residual,
        3,
        3,
        Some(model.parameters_signature()),
        &r,
    );
    assert!(matches!(
        model.calibrate_owner_blocks(rc, &r),
        Err(StructuredUnknown::QualificationCoverage)
    ));
}
