use super::*;

#[test]
fn checked_row_ceiling_preserves_width_and_charges_complete_source_schedule() {
    let (cases, opportunities, facts, population) = inventory();
    let select = |remaining_waves| {
        select_changed_with_geometry_and_source_limit(
            &cases,
            &opportunities,
            &facts,
            &[61],
            8,
            NonZeroU32::new(1),
            &population,
            100_000,
            remaining_waves,
            usize::MAX,
            None,
            None,
            None,
            None,
        )
        .unwrap()
    };
    let complete = select(10_000_000);
    assert!(complete.batches.iter().all(|batch| batch.scheduled));
    assert!(complete
        .execution_case_indices
        .iter()
        .any(|&index| cases[index].width > 1));
    let mut expected_serial = 0;
    for &index in &complete.execution_case_indices {
        let case = &cases[index];
        // With the declared per-row ceiling one, the actual driver consumes
        // each input token in its own row step, then each output continuation.
        let prompt = 61;
        expected_serial += case.width * (prompt + case.maximum_output.get() - 1);
    }
    assert_eq!(complete.serial_wave_upper_bound, expected_serial);
    for batch in &complete.batches {
        let cycle: usize = batch
            .representative_case_indices
            .iter()
            .map(|&index| {
                let case = &cases[index];
                // The original prepared Decode member has 61 prompt tokens
                // plus its three-token prefix: context 64, as in the facts.
                61 * case.width + case.maximum_output.get() - 1
            })
            .sum();
        assert_eq!(
            batch.input_opportunities.successful_cycle_wave_upper_bound,
            cycle
        );
        assert_eq!(batch.schedule.block_offered, cycle);
        assert!(batch
            .schedule
            .min_members
            .iter()
            .all(|members| *members > 0));
    }
    let exhausted = select(0);
    assert!(exhausted.execution_case_indices.is_empty());
    assert_eq!(exhausted.serial_wave_upper_bound, 0);
    assert_eq!(exhausted.populations.len(), complete.populations.len());
    assert!(exhausted.gaps.iter().any(|gap| matches!(gap.reason,
        SelectionGapReason::RemainingWaves { remaining: 0, required } if required > 0
    )));
}
use std::num::NonZeroU64;
use SloAutomaticCostProbeSamplingPresetV1::{Configured, GreedyLength};

fn unequal_prefill_work_inventory() -> (
    Vec<Case>,
    Vec<CaseOpportunity>,
    Vec<Vec<CheckedInputFacts>>,
    StructuredServiceDeclarationV7,
) {
    let population = population::declaration(&Default::default(), fixture::domain()).unwrap();
    let mut cases = Vec::new();
    let mut opportunities = Vec::new();
    let mut facts = Vec::new();
    // Both are original Configured/Actual ordinary inputs. The first has no
    // token/EOS opportunity during its checked intermediate prefill. Its full
    // independent source still has to execute all original prompt work.
    for (template, prompt) in [113_u64, 3].into_iter().enumerate() {
        // One exact owner per template keeps both sources in coverage round
        // zero. Later width owners belong to later rounds and are deliberately
        // not allowed to bypass another template's first coverage turn.
        for width in [1_u32] {
            for maximum in [1_u64, 2] {
                let input = related::prefill_input_phase(
                    width,
                    prompt.min(8 / u64::from(width)),
                    prompt,
                    maximum,
                    Configured,
                    true,
                );
                opportunities.push(CaseOpportunity {
                    population: classify_alternatives(
                        std::slice::from_ref(&input),
                        population.population_policy(),
                        true,
                    )
                    .unwrap(),
                    minimum_fresh_members: 1,
                });
                facts.push(vec![input_facts(&StructuredQueryV2::exact(input)).unwrap()]);
                cases.push(Case {
                    product: OpportunityProduct::Prefill,
                    template,
                    width: width as usize,
                    maximum_output: NonZeroUsize::new(maximum as usize).unwrap(),
                    release_generated: 0,
                    suffix_tokens: maximum as usize,
                    preset: Configured,
                    prefix: PrefixKind::Ordinary,
                    route: CalibrationDecodeRoute::Actual,
                    reset: true,
                    acquisition: None,
                });
            }
        }
    }
    assert!(facts[..2]
        .iter()
        .all(|row| row[0].branches[2] && !row[0].branches[4]));
    assert!(facts[2..]
        .iter()
        .all(|row| !row[0].branches[2] && row[0].branches[4]));
    let groups = member_groups(&opportunities).unwrap();
    assert_eq!(
        groups.len(),
        2,
        "one independent physical owner for each template"
    );
    assert!(groups.iter().all(|group| {
        let template = cases[group.guaranteed_case_indices[0]].template;
        group.guaranteed_case_indices.iter().all(|&i| cases[i].template == template)
    }), "counterexample requires independently complete physical populations, not mixed endpoints of the same owner");
    (cases, opportunities, facts, population)
}

#[test]
fn checked_policy_complete_work_precedes_no_token_risk_for_retention_and_execution() {
    let (cases, opportunities, facts, population) = unequal_prefill_work_inventory();
    let run = |maximum_sources| {
        select_changed_with_geometry_and_source_limit(
            &cases,
            &opportunities,
            &facts,
            &[113, 3],
            8,
            None,
            &population,
            100_000,
            10_000_000,
            usize::MAX,
            None,
            None,
            None,
            maximum_sources,
        )
        .unwrap()
    };
    let complete = run(None);
    eprintln!(
        "complete work source order: {:?}",
        complete
            .batches
            .iter()
            .map(|batch| (
                CoveragePriority::from_cases(&batch.representative_case_indices, &cases).unwrap(),
                batch.serial_token_work,
                batch.serial_wave_upper_bound,
                batch.requests,
                &batch.representative_case_indices,
            ))
            .collect::<Vec<_>>()
    );
    assert!(complete.batches.iter().all(|batch| batch.scheduled));
    let first = &complete.batches[0];
    assert!(first
        .representative_case_indices
        .iter()
        .all(|&i| cases[i].template == 1));
    assert!(complete
        .batches
        .iter()
        .filter(|batch| batch
            .representative_case_indices
            .iter()
            .any(|&i| cases[i].template == 0))
        .all(|batch| first.serial_token_work < batch.serial_token_work));
    let retained = run(NonZeroUsize::new(1));
    assert_eq!(retained.batches.len(), complete.batches.len());
    assert_eq!(
        retained.execution_case_indices,
        first
            .representative_case_indices
            .repeat(first.planned_cycles)
    );
    assert_eq!(retained.requests, first.requests);
    assert_eq!(
        retained.serial_wave_upper_bound,
        first.serial_wave_upper_bound
    );
    for (index, (actual, original)) in retained.batches.iter().zip(&complete.batches).enumerate() {
        assert_eq!(
            actual.representative_case_indices,
            original.representative_case_indices
        );
        assert_eq!(actual.planned_cycles, original.planned_cycles);
        assert_eq!(
            serde_json::to_value(&actual.schedule).unwrap(),
            serde_json::to_value(&original.schedule).unwrap()
        );
        assert_eq!(actual.scheduled, index == 0);
    }
    assert!(retained.gaps.iter().any(|gap| matches!(
        gap.reason,
        SelectionGapReason::OutcomeDependentEarlyTermination
    )));
    assert!(retained.gaps.iter().any(|gap| matches!(
        gap.reason,
        SelectionGapReason::RetainedSourceCapacity { maximum_sources: 1 }
    )));
    assert!(first
        .input_opportunities
        .phase_original_offer_bounds
        .iter()
        .zip(first.schedule.phase_min_offered)
        .all(|(actual, minimum)| *actual >= minimum));
}

#[test]
fn checked_policy_complete_work_orders_shared_geometry_before_no_token_risk() {
    let (cases, opportunities, facts, population) = unequal_prefill_work_inventory();
    let run = |cases: &[Case],
               opportunities: &[CaseOpportunity],
               facts: &[Vec<CheckedInputFacts>],
               work: &mut StructuredInputGeometryWorkV1| {
        select_changed_with_geometry(
            cases,
            opportunities,
            facts,
            &[113, 3],
            8,
            &population,
            100_000,
            10_000_000,
            usize::MAX,
            None,
            None,
            Some(work),
        )
        .unwrap()
    };
    let mut standalone_work =
        StructuredInputGeometryWorkV1::new(NonZeroU64::new(32_000_000).unwrap());
    let standalone = run(
        &cases[2..],
        &opportunities[2..],
        &facts[2..],
        &mut standalone_work,
    );
    assert!(standalone
        .populations
        .iter()
        .all(|p| p.input_geometry.as_ref().unwrap().complete));
    let mut shared =
        StructuredInputGeometryWorkV1::new(NonZeroU64::new(standalone_work.visits()).unwrap());
    let together = run(&cases, &opportunities, &facts, &mut shared);
    for expected in &standalone.populations {
        let actual = together
            .populations
            .iter()
            .find(|p| p.key == expected.key)
            .unwrap();
        assert_eq!(
            serde_json::to_value(&actual.input_geometry).unwrap(),
            serde_json::to_value(&expected.input_geometry).unwrap()
        );
    }
    assert_eq!(shared.visits(), standalone_work.visits());
    assert!(together
        .populations
        .iter()
        .filter(|p| p
            .representative_case_indices
            .iter()
            .any(|&i| cases[i].template == 0))
        .all(|p| !p.input_geometry.as_ref().unwrap().complete));
}

#[test]
fn checked_policy_priority_retained_source_limit_keeps_complete_required_origins() {
    let (cases, opportunities, facts, population) =
        related::prefill_inventory(&[GreedyLength, Configured]);
    let original = select_all(
        &cases,
        &opportunities,
        &facts,
        &population,
        100_000,
        10_000_000,
    );
    assert!(original.batches.len() > 1);
    assert!(original.batches.iter().all(|batch| batch.scheduled));
    for maximum_sources in 1..=original.batches.len() {
        let result = select_changed_with_geometry_and_source_limit(
            &cases,
            &opportunities,
            &facts,
            &[1, 2],
            8,
            None,
            &population,
            100_000,
            10_000_000,
            usize::MAX,
            None,
            None,
            None,
            NonZeroUsize::new(maximum_sources),
        )
        .unwrap();
        assert_eq!(
            result.batches.len(),
            original.batches.len(),
            "unreserved complete sources stay declared"
        );
        let mut requests = 0;
        let mut waves = 0;
        let mut expected_execution = Vec::new();
        for (index, (actual, expected)) in result.batches.iter().zip(&original.batches).enumerate()
        {
            assert_eq!(
                actual.representative_case_indices,
                expected.representative_case_indices
            );
            assert_eq!(actual.planned_cycles, expected.planned_cycles);
            assert_eq!(
                serde_json::to_value(&actual.schedule).unwrap(),
                serde_json::to_value(&expected.schedule).unwrap()
            );
            assert_eq!(actual.scheduled, index < maximum_sources);
            if actual.scheduled {
                requests += expected.requests;
                waves += expected.serial_wave_upper_bound;
                expected_execution.extend(
                    expected
                        .representative_case_indices
                        .repeat(expected.planned_cycles),
                );
            } else {
                for &member in &actual.population_indices {
                    assert!(result.gaps.iter().any(|gap| gap.population.as_ref() == Some(&result.populations[member].key)
                        && matches!(gap.reason, SelectionGapReason::RetainedSourceCapacity { maximum_sources: maximum } if maximum == maximum_sources)));
                }
            }
        }
        assert_eq!(result.execution_case_indices, expected_execution);
        assert_eq!(result.requests, requests);
        assert_eq!(result.serial_wave_upper_bound, waves);
        assert!(result.batches[0]
            .representative_case_indices
            .iter()
            .all(|&index| cases[index].preset == Configured));
    }
}

fn select_all(
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    facts: &[Vec<CheckedInputFacts>],
    population: &StructuredServiceDeclarationV7,
    requests: usize,
    waves: usize,
) -> CheckedSelection {
    select(
        cases,
        opportunities,
        facts,
        &[1, 2],
        8,
        population,
        requests,
        waves,
        usize::MAX,
    )
    .unwrap()
}

#[test]
fn checked_policy_priority_reserves_original_budget_for_configured_complete_sources() {
    // Auxiliary inputs are deliberately declared first. These are distinct
    // original canonical host policies, not manually rewritten owner keys.
    let (cases, opportunities, facts, population) =
        related::prefill_inventory(&[GreedyLength, Configured]);
    let original = serde_json::to_value(&cases).unwrap();
    let required = select_all(
        &cases[8..],
        &opportunities[8..],
        &facts[8..],
        &population,
        100_000,
        10_000_000,
    );
    assert!(required.populations.iter().all(|p| p.scheduled));
    let expected: Vec<_> = required
        .execution_case_indices
        .iter()
        .map(|i| i + 8)
        .collect();
    for (requests, waves, by_requests) in [
        (required.requests, 10_000_000, true),
        (100_000, required.serial_wave_upper_bound, false),
    ] {
        let result = select_all(&cases, &opportunities, &facts, &population, requests, waves);
        assert_eq!(result.populations.len(), 4);
        assert_eq!(result.execution_case_indices, expected);
        assert!(result.requests <= requests && result.serial_wave_upper_bound <= waves);
        for group in &result.populations {
            let configured = group
                .representative_case_indices
                .iter()
                .all(|&i| cases[i].preset == Configured);
            assert_eq!(group.scheduled, configured);
            if !configured {
                assert!(result
                    .gaps
                    .iter()
                    .any(|gap| gap.population.as_ref() == Some(&group.key)
                        && if by_requests {
                            matches!(gap.reason, SelectionGapReason::RemainingRequests { .. })
                        } else {
                            matches!(gap.reason, SelectionGapReason::RemainingWaves { .. })
                        }));
            }
        }
    }
    assert_eq!(serde_json::to_value(&cases).unwrap(), original);
}

#[test]
fn checked_policy_priority_does_not_promote_missing_original_members() {
    let (cases, mut opportunities, facts, population) =
        related::prefill_inventory(&[GreedyLength, Configured]);
    let original_groups = member_groups(&opportunities).unwrap();
    for opportunity in &mut opportunities[8..] {
        opportunity.minimum_fresh_members = 0;
    }
    let result = select_all(
        &cases,
        &opportunities,
        &facts,
        &population,
        100_000,
        10_000_000,
    );
    assert_eq!(result.populations.len(), 2);
    assert!(!result.execution_case_indices.is_empty());
    assert!(result
        .execution_case_indices
        .iter()
        .all(|&i| cases[i].preset == GreedyLength));
    for group in &original_groups[2..] {
        assert!(!result.populations.iter().any(|p| p.key == group.key));
        assert!(result
            .gaps
            .iter()
            .any(|gap| gap.population.as_ref() == Some(&group.key)
                && matches!(gap.reason, SelectionGapReason::NoGuaranteedMember)));
    }
}

#[test]
fn checked_policy_priority_geometry_uses_one_original_ledger_in_required_order() {
    let (cases, opportunities, facts, population) =
        related::prefill_inventory(&[GreedyLength, Configured]);
    let run = |cases: &[Case],
               opportunities: &[CaseOpportunity],
               facts: &[Vec<CheckedInputFacts>],
               work: &mut StructuredInputGeometryWorkV1| {
        select_changed_with_geometry(
            cases,
            opportunities,
            facts,
            &[1, 2],
            8,
            &population,
            100_000,
            10_000_000,
            usize::MAX,
            None,
            None,
            Some(work),
        )
        .unwrap()
    };
    let mut complete = StructuredInputGeometryWorkV1::new(NonZeroU64::new(32_000_000).unwrap());
    let standalone = run(&cases[8..], &opportunities[8..], &facts[8..], &mut complete);
    assert!(standalone
        .populations
        .iter()
        .all(|p| p.input_geometry.as_ref().unwrap().complete));
    let mut shared =
        StructuredInputGeometryWorkV1::new(NonZeroU64::new(complete.visits()).unwrap());
    let combined = run(&cases, &opportunities, &facts, &mut shared);
    for (expected, actual) in standalone
        .populations
        .iter()
        .zip(&combined.populations[2..])
    {
        assert_eq!(expected.key, actual.key);
        assert_eq!(
            serde_json::to_value(&expected.input_geometry).unwrap(),
            serde_json::to_value(&actual.input_geometry).unwrap()
        );
    }
    assert!(combined.populations[..2]
        .iter()
        .all(|p| !p.input_geometry.as_ref().unwrap().complete));
    assert!(combined.gaps.iter().any(|gap| matches!(
        gap.reason,
        SelectionGapReason::InputGeometryUnavailable {
            work_exhausted: true,
            ..
        }
    )));
    let visits: u64 = combined
        .populations
        .iter()
        .map(|p| p.input_geometry.as_ref().unwrap().visits)
        .sum();
    assert_eq!(visits, shared.visits());
    assert_eq!(visits, complete.visits());
    assert_eq!(
        combined.input_geometry.as_ref().unwrap().maximum_visits,
        complete.visits()
    );
}

#[test]
fn checked_policy_priority_preserves_original_early_terminal_qualification_gap() {
    let (mut cases, mut opportunities, mut facts, population) = two_families();
    for case in &mut cases[..4] {
        case.preset = GreedyLength;
    }
    install_natural_termination_facts(
        &mut cases[4..],
        &mut opportunities[4..],
        &mut facts[4..],
        false,
    );
    let result = select(
        &cases,
        &opportunities,
        &facts,
        &[61],
        8,
        &population,
        100_000,
        10_000_000,
        usize::MAX,
    )
    .unwrap();
    assert_eq!(result.batches[0].population_indices, [1]);
    assert!(result.populations[1].scheduled);
    assert!(result.gaps.iter().any(|gap| gap.population.as_ref()
        == Some(&result.populations[1].key)
        && matches!(
            gap.reason,
            SelectionGapReason::OutcomeDependentEarlyTermination
        )));
    assert!(facts[4..]
        .iter()
        .all(|rows| rows[0].branches[4] && rows[0].branches[5]));
}

#[test]
fn checked_policy_priority_rounds_preserve_complete_sources_and_all_original_members() {
    let (mut cases, opportunities, facts, population) = related::prefill_inventory(&[Configured]);
    let selected = select_all(
        &cases,
        &opportunities,
        &facts,
        &population,
        100_000,
        10_000_000,
    );
    // Rank already checked complete sources; no partial source or extra phase
    // is invented by the coverage policy. A different rendered template
    // does not manufacture a different checked execution role or algorithm.
    let batch = selected.batches[0].clone();
    let configured = CoveragePriority::from_cases(&[0], &cases).unwrap();
    cases[0].template = 1;
    let other_template = CoveragePriority::from_cases(&[0], &cases).unwrap();
    cases[0].route = CalibrationDecodeRoute::FullLogits;
    cases[0].preset = GreedyLength;
    let auxiliary_full = CoveragePriority::from_cases(&[0], &cases).unwrap();
    cases[0].route = CalibrationDecodeRoute::Actual;
    let auxiliary_actual = CoveragePriority::from_cases(&[0], &cases).unwrap();
    let priorities = [
        configured,
        configured,
        other_template,
        auxiliary_full,
        auxiliary_full,
        auxiliary_actual,
    ];
    let mut candidates: Vec<_> = priorities
        .into_iter()
        .enumerate()
        .map(|(index, coverage)| BatchCandidate {
            input_priority: 0,
            coverage,
            coverage_round: 0,
            decode_width_tier: 0,
            original_population_index: index,
            batch: batch.clone(),
        })
        .collect();
    priority::order_batches(&mut candidates, &facts);
    assert!(candidates
        .windows(2)
        .all(|pair| { pair[0].coverage.policy <= pair[1].coverage.policy }));
    assert_eq!(
        candidates
            .iter()
            .filter(|candidate| candidate.coverage_round == 0)
            .count(),
        2,
        "one checked role in each of the two original policy tiers"
    );
    for candidate in candidates {
        assert_eq!(
            serde_json::to_value(candidate.batch).unwrap(),
            serde_json::to_value(&batch).unwrap()
        );
    }
    cases[0].preset = Configured;
    cases[0].route = CalibrationDecodeRoute::Actual;
    cases[0].prefix = PrefixKind::Clean;
    assert_eq!(
        configured.policy,
        CoveragePriority::from_cases(&[0], &cases).unwrap().policy,
        "Configured/Actual native preparation retains the required product policy priority"
    );
}
