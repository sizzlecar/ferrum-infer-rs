use super::input_coverage_repro::mask_input;
use super::*;

fn schedule() -> OwnerBlockScheduleV1 {
    OwnerBlockScheduleV1::new_with_input_readiness(
        32,
        [32; 3],
        [8; 3],
        OwnerInputReadinessV1::new([3; 3], 32_000_000).unwrap(),
    )
    .unwrap()
}
fn mask_rows(
    phase: StructuredPhaseV2,
    first_block: u64,
    blocks: usize,
    prior: usize,
    late_mask: bool,
) -> Vec<StructuredNumericObservationV2> {
    let template = original_block_population(phase).remove(0);
    (0..blocks * 16)
        .map(|i| {
            let mut s = template.clone();
            let ticket = (first_block - 1 + (i / 16) as u64) * 32 + (i % 16) as u64 + 1;
            s.membership.offered_ordinal = ticket;
            s.membership.member_ordinal = (prior + i + 1) as u64;
            s.ordinal = ticket * 3;
            s.call_id = ticket + 100;
            s.observed_at_ns = ticket * 10;
            s.input = mask_input(late_mask && i >= 16 && i % 16 == 0);
            s.wall_ns = 1000;
            s
        })
        .collect()
}
fn contract_for(target: OwnerInputTargetV1) -> StructuredOwnerPhaseContractV1 {
    let mut c = physical_owner_contract();
    c.nonnegative_envelope.as_mut().unwrap().template_policy =
        StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1;
    c.schedule = schedule();
    c.input_target = Some(target);
    c
}
fn mask_target() -> OwnerInputTargetV1 {
    let mut target = OwnerInputTargetV1::from_input(&mask_input(false)).unwrap();
    target.observe(&mask_input(true)).unwrap();
    target
}

#[test]
fn owner_input_readiness_late_mask_keeps_whole_independent_phases_and_rejects_late_close() {
    let fit = mask_rows(StructuredPhaseV2::Fit, 2, 2, 0, true);
    let c = contract_for(mask_target());
    let first = block_close(StructuredPhaseV2::Fit, 2, 2, None, &fit[..16]);
    assert!(matches!(
        FittedStructuredModelV2::fit_owner_blocks(
            fp(),
            settings(),
            scope(&fit),
            c.clone(),
            first,
            &fit[..16]
        ),
        Err(StructuredUnknown::IncompletePhasePopulation)
    ));
    let fitted = FittedStructuredModelV2::fit_owner_blocks(
        fp(),
        settings(),
        scope(&fit),
        c.clone(),
        block_close(StructuredPhaseV2::Fit, 2, 3, None, &fit),
        &fit,
    )
    .unwrap();
    let later = mask_rows(StructuredPhaseV2::Fit, 2, 3, 0, true);
    assert!(
        matches!(
            FittedStructuredModelV2::fit_owner_blocks(
                fp(),
                settings(),
                scope(&later),
                c,
                block_close(StructuredPhaseV2::Fit, 2, 4, None, &later),
                &later
            ),
            Err(StructuredUnknown::PhaseLeakage)
        ),
        "a later valid population cannot erase the earlier ready boundary"
    );
    let residual = mask_rows(StructuredPhaseV2::Residual, 4, 2, 32, true);
    let residual_close = block_close(
        StructuredPhaseV2::Residual,
        4,
        5,
        Some(fitted.parameters_signature()),
        &residual,
    );
    let calibrated = fitted
        .calibrate_owner_blocks(residual_close, &residual)
        .unwrap();
    let qualification = mask_rows(StructuredPhaseV2::Qualification, 6, 2, 64, true);
    let qualification_close = block_close(
        StructuredPhaseV2::Qualification,
        6,
        7,
        Some(calibrated.parameters_signature()),
        &qualification,
    );
    let qualified = calibrated
        .qualify_owner_blocks(qualification_close, &qualification)
        .unwrap();
    assert_eq!(qualified.source_contract().phase_members, [32; 3]);
    assert!(qualified
        .predict_query(&fp(), &StructuredQueryV2::exact(mask_input(true)), 2300)
        .is_ok());
}

#[test]
fn owner_input_readiness_rank_redundancy_wait_is_bounded_and_wall_blind() {
    // Two actual canonical work directions, one extra redundant member needed.
    let mut cfg = settings();
    cfg.min_fit_redundancy = 7;
    let mut rows = mask_rows(StructuredPhaseV2::Fit, 2, 1, 0, false);
    rows.truncate(8);
    rows[0].input = mask_input(true);
    let target = mask_target();
    let mut visits = 0;
    assert_eq!(
        schedule()
            .assess_inputs(
                StructuredPhaseV2::Fit,
                32,
                &rows,
                Some(&target),
                &cfg,
                &mut visits
            )
            .unwrap(),
        OwnerInputReadinessDecisionV1::Wait
    );
    assert!(visits > 0);
    let original_visits = visits;
    for s in &mut rows {
        s.wall_ns = u64::MAX;
    }
    visits = 0;
    assert_eq!(
        schedule()
            .assess_inputs(
                StructuredPhaseV2::Fit,
                32,
                &rows,
                Some(&target),
                &cfg,
                &mut visits
            )
            .unwrap(),
        OwnerInputReadinessDecisionV1::Wait
    );
    assert_eq!(visits, original_visits);
    rows.push(rows[1].clone());
    assert_eq!(
        schedule()
            .assess_inputs(
                StructuredPhaseV2::Fit,
                64,
                &rows,
                Some(&target),
                &cfg,
                &mut visits
            )
            .unwrap(),
        OwnerInputReadinessDecisionV1::Freeze
    );
    let mut clean = mask_rows(StructuredPhaseV2::Fit, 2, 3, 0, false);
    for s in &mut clean {
        s.wall_ns = 1000;
    }
    assert_eq!(
        schedule()
            .assess_inputs(
                StructuredPhaseV2::Fit,
                96,
                &clean,
                Some(&target),
                &settings(),
                &mut 0
            )
            .unwrap(),
        OwnerInputReadinessDecisionV1::Exhausted(OwnerInputReadinessGapV1::MissingInputCoverage)
    );
}

#[test]
fn owner_input_readiness_settled_eos_changes_neither_target_nor_decision() {
    let early = original_block_population(StructuredPhaseV2::Fit);
    let mut continuation = early.clone();
    // Keep all original pre-execution row facts; change only optional observed
    // EOS/Stop outcomes and corresponding completion columns.
    for (i, sample) in continuation.iter_mut().enumerate().take(2) {
        sample.input = input(9, [i & 1 != 0, i & 2 != 0], 64, 2 + (i % 2) as u64)
            .with_settled_terminal_causes(&[])
            .unwrap();
    }
    let a = OwnerInputTargetV1::from_samples(&early).unwrap();
    let b = OwnerInputTargetV1::from_samples(&continuation).unwrap();
    assert_eq!(a, b);
    let mut av = 0;
    let mut bv = 0;
    let ad = schedule()
        .assess_inputs(
            StructuredPhaseV2::Fit,
            32,
            &early,
            Some(&a),
            &settings(),
            &mut av,
        )
        .unwrap();
    let bd = schedule()
        .assess_inputs(
            StructuredPhaseV2::Fit,
            32,
            &continuation,
            Some(&b),
            &settings(),
            &mut bv,
        )
        .unwrap();
    assert_eq!((ad, av), (bd, bv));
    // The original coverage gate still rejects absent observed early outcomes.
    let no_early = population(StructuredPhaseV2::Fit, false);
    assert!(matches!(
        FittedStructuredModelV2::fit_service_window(
            fp(),
            settings(),
            scope(&no_early),
            physical_contract(),
            close(StructuredPhaseV2::Fit, &no_early),
            &no_early,
            170
        ),
        Err(StructuredUnknown::QualificationCoverage)
    ));
}

#[test]
fn owner_input_readiness_checked_complete_block_bounds_and_legacy_contract_bytes() {
    let mut cfg = settings();
    cfg.max_phase_samples = 64;
    assert!(matches!(
        schedule().validate(&cfg),
        Err(StructuredUnknown::Capacity)
    ));
    assert!(OwnerBlockScheduleV1::new_with_input_readiness(
        usize::MAX,
        [8; 3],
        [8; 3],
        OwnerInputReadinessV1::new([2; 3], 1).unwrap()
    )
    .is_err());
    let c = physical_owner_contract();
    let wire = serde_json::to_value(&c).unwrap();
    assert!(!wire.as_object().unwrap().contains_key("input_target"));
    let decoded: StructuredOwnerPhaseContractV1 = serde_json::from_value(wire).unwrap();
    assert_eq!(decoded, c);
}

#[test]
fn owner_input_readiness_two_pass_total_and_cumulative_prefix_budget_are_enforced() {
    cumulative_prefix_budget(false);
}

#[test]
fn owner_input_readiness_v2_two_pass_total_and_cumulative_prefix_budget_are_enforced() {
    cumulative_prefix_budget(true);
}

fn cumulative_prefix_budget(cached: bool) {
    let bounded_schedule = |maximum| {
        OwnerBlockScheduleV1::new_with_input_readiness(
            32,
            [32; 3],
            [8; 3],
            if cached {
                OwnerInputReadinessV1::new_cached_residual_v2([3; 3], maximum).unwrap()
            } else {
                OwnerInputReadinessV1::new([3; 3], maximum).unwrap()
            },
        )
        .unwrap()
    };

    let rows = mask_rows(StructuredPhaseV2::Fit, 2, 2, 0, true);
    let target = mask_target();
    let mut first_visits = 0;
    assert_eq!(
        bounded_schedule(32_000_000)
            .assess_inputs(
                StructuredPhaseV2::Fit,
                32,
                &rows[..16],
                Some(&target),
                &settings(),
                &mut first_visits
            )
            .unwrap(),
        OwnerInputReadinessDecisionV1::Wait
    );
    let mut second_visits = 0;
    assert_eq!(
        bounded_schedule(32_000_000)
            .assess_inputs(
                StructuredPhaseV2::Fit,
                64,
                &rows,
                Some(&target),
                &settings(),
                &mut second_visits
            )
            .unwrap(),
        OwnerInputReadinessDecisionV1::Freeze
    );
    assert!(first_visits > 0 && second_visits > first_visits);
    let pass_total = first_visits + second_visits;
    let configured_total = 2 * pass_total;
    for maximum in [configured_total, configured_total + 1] {
        let bounded = bounded_schedule(maximum);
        let mut live_visits = 0;
        assert_eq!(
            bounded
                .assess_inputs(
                    StructuredPhaseV2::Fit,
                    32,
                    &rows[..16],
                    Some(&target),
                    &settings(),
                    &mut live_visits
                )
                .unwrap(),
            OwnerInputReadinessDecisionV1::Wait
        );
        assert_eq!(
            bounded
                .assess_inputs(
                    StructuredPhaseV2::Fit,
                    64,
                    &rows,
                    Some(&target),
                    &settings(),
                    &mut live_visits
                )
                .unwrap(),
            OwnerInputReadinessDecisionV1::Freeze
        );
        assert_eq!(live_visits, pass_total);
        let mut contract = contract_for(target.clone());
        contract.schedule = bounded;
        // The numerical entry independently verifies every original prefix;
        // it does not trust the caller's earlier readiness result.
        FittedStructuredModelV2::fit_owner_blocks(
            fp(),
            settings(),
            scope(&rows),
            contract,
            block_close(StructuredPhaseV2::Fit, 2, 3, None, &rows),
            &rows,
        )
        .unwrap();
        assert!(live_visits * 2 <= maximum);
    }
    for maximum in [configured_total - 2, 2 * second_visits] {
        let bounded = bounded_schedule(maximum);
        let mut alone = 0;
        assert_eq!(
            bounded
                .assess_inputs(
                    StructuredPhaseV2::Fit,
                    64,
                    &rows,
                    Some(&target),
                    &settings(),
                    &mut alone
                )
                .unwrap(),
            OwnerInputReadinessDecisionV1::Freeze
        );
        let mut cumulative = 0;
        assert_eq!(
            bounded
                .assess_inputs(
                    StructuredPhaseV2::Fit,
                    32,
                    &rows[..16],
                    Some(&target),
                    &settings(),
                    &mut cumulative
                )
                .unwrap(),
            OwnerInputReadinessDecisionV1::Wait
        );
        assert_eq!(
            bounded
                .assess_inputs(
                    StructuredPhaseV2::Fit,
                    64,
                    &rows,
                    Some(&target),
                    &settings(),
                    &mut cumulative
                )
                .unwrap(),
            OwnerInputReadinessDecisionV1::Exhausted(OwnerInputReadinessGapV1::GeometryWorkBudget)
        );
        assert!(cumulative <= maximum / 2);
        let mut contract = contract_for(target.clone());
        contract.schedule = bounded;
        assert!(matches!(
            FittedStructuredModelV2::fit_owner_blocks(
                fp(),
                settings(),
                scope(&rows),
                contract,
                block_close(StructuredPhaseV2::Fit, 2, 3, None, &rows),
                &rows
            ),
            Err(StructuredUnknown::IncompletePhasePopulation)
        ));
    }
}
