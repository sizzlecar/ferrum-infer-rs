//! Input-plan regression only: no input inventory fact is a measured sample.
use super::*;
use SloAutomaticCostProbeSamplingPresetV1::Configured;

fn inventory() -> (
    Vec<Case>,
    Vec<CaseOpportunity>,
    Vec<Vec<CheckedInputFacts>>,
    StructuredServiceDeclarationV7,
) {
    let population = population::declaration(&Default::default(), fixture::domain()).unwrap();
    let mut cases = Vec::new();
    let mut inputs = Vec::new();
    // Same installed natural-EOS FullLogits host policy. The real canonical
    // row roles and physical commands, rather than edited identity hashes,
    // distinguish initial prefill, ordinary decode and the larger prefill.
    for maximum in [1_u64, 2] {
        cases.push(Case {
            product: OpportunityProduct::Prefill,
            template: 0,
            width: 1,
            maximum_output: NonZeroUsize::new(maximum as usize).unwrap(),
            release_generated: 0,
            suffix_tokens: maximum as usize,
            preset: Configured,
            prefix: PrefixKind::Ordinary,
            route: CalibrationDecodeRoute::Actual,
            reset: true,
            acquisition: None,
        });
        inputs.push(related::prefill_input_phase(
            1, 3, 3, maximum, Configured, true,
        ));
    }
    for at_length_boundary in [true, false] {
        let maximum = if at_length_boundary { 4 } else { 21 };
        cases.push(Case {
            product: OpportunityProduct::Full,
            template: 0,
            width: 1,
            maximum_output: NonZeroUsize::new(maximum).unwrap(),
            release_generated: 3,
            suffix_tokens: maximum - 3,
            preset: Configured,
            prefix: PrefixKind::Clean,
            route: CalibrationDecodeRoute::Actual,
            reset: at_length_boundary,
            acquisition: None,
        });
        // The original short prompt has three tokens and the original
        // validated prefix releases after three generated tokens: frontier 6.
        // Prefix preparation gives an opportunity conditional on completion;
        // it never certifies that a natural-EOS cohort actually completed.
        inputs.push(natural_termination_input_at_frontier(
            1,
            CostProductOutput::FullLogits,
            at_length_boundary,
            6,
        ));
    }
    for maximum in [1_u64, 2] {
        cases.push(Case {
            product: OpportunityProduct::Prefill,
            template: 1,
            width: 1,
            maximum_output: NonZeroUsize::new(maximum as usize).unwrap(),
            release_generated: 0,
            suffix_tokens: maximum as usize,
            preset: Configured,
            prefix: PrefixKind::Ordinary,
            route: CalibrationDecodeRoute::Actual,
            reset: true,
            acquisition: None,
        });
        inputs.push(related::prefill_input_phase(
            1, 8, 1017, maximum, Configured, true,
        ));
    }
    let opportunities: Vec<_> = inputs
        .iter()
        .map(|input| CaseOpportunity {
            population: classify_alternatives(
                std::slice::from_ref(input),
                population.population_policy(),
                true,
            )
            .unwrap(),
            minimum_fresh_members: 1,
        })
        .collect();
    let facts: Vec<_> = inputs
        .into_iter()
        .map(|input| vec![input_facts(&StructuredQueryV2::exact(input)).unwrap()])
        .collect();
    assert_eq!(member_groups(&opportunities).unwrap().len(), 3);
    assert!(facts[..4].iter().all(|row| row[0].branches[4]));
    assert!(facts[..4].iter().any(|row| row[0].branches[5]));
    assert!(facts[4..].iter().all(|row| row[0].branches[2]));
    assert_ne!(facts[0][0].owner, facts[2][0].owner);
    assert_ne!(facts[0][0].owner, facts[4][0].owner);
    (cases, opportunities, facts, population)
}

fn select_inventory(
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    facts: &[Vec<CheckedInputFacts>],
    population: &StructuredServiceDeclarationV7,
    maximum_sources: Option<NonZeroUsize>,
) -> CheckedSelection {
    select_changed_with_geometry_and_source_limit(
        cases,
        opportunities,
        facts,
        &[3, 1017],
        8,
        None,
        population,
        100_000,
        10_000_000,
        usize::MAX,
        None,
        None,
        None,
        maximum_sources,
    )
    .unwrap()
}

#[test]
fn checked_bootstrap_roles_retain_complete_prefill_and_prepared_decode_sources() {
    let (cases, opportunities, facts, population) = inventory();
    let originals = serde_json::to_value((&cases, &opportunities, &facts)).unwrap();
    let required = select_inventory(
        &cases[..4],
        &opportunities[..4],
        &facts[..4],
        &population,
        None,
    );
    assert!(required.populations.iter().all(|member| member.scheduled));
    assert!(required.populations.len() > 1);
    let maximum_sources = NonZeroUsize::new(required.batches.len()).unwrap();
    let all = select_inventory(&cases, &opportunities, &facts, &population, None);
    assert!(all.batches.iter().all(|batch| batch.scheduled));
    let large = all
        .batches
        .iter()
        .find(|batch| batch.representative_case_indices.iter().any(|&i| i >= 4))
        .unwrap();
    assert!(required
        .batches
        .iter()
        .all(|batch| batch.serial_token_work < large.serial_token_work));

    let limited = select_inventory(
        &cases,
        &opportunities,
        &facts,
        &population,
        Some(maximum_sources),
    );
    eprintln!(
        "bootstrap role source reservation: cap={} batches={:?}",
        maximum_sources,
        limited
            .batches
            .iter()
            .map(|batch| (
                batch.scheduled,
                CoveragePriority::from_cases(&batch.representative_case_indices, &cases).unwrap(),
                batch.serial_token_work,
                batch.requests,
                batch.serial_wave_upper_bound,
                batch.schedule.min_members,
                &batch.representative_case_indices,
            ))
            .collect::<Vec<_>>()
    );
    // Retention and execution must preserve the whole original source, not
    // manufacture members, shorten a phase, or trim expensive endpoints.
    for original in &all.batches {
        let actual = limited
            .batches
            .iter()
            .find(|batch| batch.representative_case_indices == original.representative_case_indices)
            .unwrap();
        assert_eq!(
            serde_json::to_value(&actual.schedule).unwrap(),
            serde_json::to_value(&original.schedule).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&actual.input_plan).unwrap(),
            serde_json::to_value(&original.input_plan).unwrap()
        );
        let numerical = &population.settings;
        assert_eq!(
            actual.schedule.min_members,
            [
                numerical
                    .min_phase_samples
                    .max(numerical.max_rank + numerical.min_fit_redundancy),
                numerical.min_phase_samples,
                numerical.min_phase_samples,
            ]
        );
    }
    assert_eq!(
        serde_json::to_value((&cases, &opportunities, &facts)).unwrap(),
        originals
    );
    assert!(limited.gaps.iter().any(|gap| matches!(
        gap.reason,
        SelectionGapReason::OutcomeDependentEarlyTermination
    )));
    for required_member in &required.populations {
        let member = limited
            .populations
            .iter()
            .find(|member| member.key == required_member.key)
            .unwrap();
        assert!(
            member.scheduled,
            "a complete short Configured/Actual role was excluded while a larger ordinary prefill retained its slot: representatives={:?}, gaps={:?}",
            member.representative_case_indices,
            limited.gaps
        );
    }
    assert!(limited
        .execution_case_indices
        .iter()
        .all(|&index| index < 4));
}

#[test]
fn checked_bootstrap_roles_do_not_promote_unreachable_ordinary_decode() {
    let (mut cases, mut opportunities, facts, population) = inventory();
    let decode = facts[2][0].key(population.population_policy());
    // This is only a possible ordinary trajectory. EOS may prevent its first
    // decode input, so removing the real prepared release also removes its
    // member floor. Coverage priority must not recreate that missing proof.
    for index in 2..4 {
        cases[index].prefix = PrefixKind::Ordinary;
        cases[index].release_generated = 0;
        opportunities[index].minimum_fresh_members = 0;
    }
    let result = select_inventory(&cases, &opportunities, &facts, &population, None);
    assert!(!result.populations.iter().any(|member| member.key == decode));
    assert!(result.gaps.iter().any(|gap| {
        gap.population.as_ref() == Some(&decode)
            && matches!(gap.reason, SelectionGapReason::NoGuaranteedMember)
    }));
    assert!(result
        .execution_case_indices
        .iter()
        .all(|&index| !(2..4).contains(&index)));
}

#[test]
fn checked_bootstrap_roles_spend_geometry_on_prefill_and_decode_before_expansion() {
    let (cases, opportunities, facts, population) = inventory();
    let select_with_work = |cases: &[Case],
                            opportunities: &[CaseOpportunity],
                            facts: &[Vec<CheckedInputFacts>],
                            work: &mut StructuredInputGeometryWorkV1| {
        select_changed_with_geometry(
            cases,
            opportunities,
            facts,
            &[3, 1017],
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
    let mut original_work =
        StructuredInputGeometryWorkV1::new(std::num::NonZeroU64::new(32_000_000).unwrap());
    let required = select_with_work(
        &cases[..4],
        &opportunities[..4],
        &facts[..4],
        &mut original_work,
    );
    assert!(required.populations.iter().all(|member| member
        .input_geometry
        .as_ref()
        .unwrap()
        .complete));
    let mut bounded_work = StructuredInputGeometryWorkV1::new(
        std::num::NonZeroU64::new(original_work.visits()).unwrap(),
    );
    let together = select_with_work(&cases, &opportunities, &facts, &mut bounded_work);
    for original in &required.populations {
        let member = together
            .populations
            .iter()
            .find(|member| member.key == original.key)
            .unwrap();
        assert!(
            member.input_geometry.as_ref().unwrap().complete,
            "same-role expansion consumed a different required role's complete geometry allowance: {:?}",
            member.input_geometry
        );
    }
    assert!(bounded_work.visits() <= original_work.visits());
    assert!(together.gaps.iter().any(|gap| matches!(
        gap.reason,
        SelectionGapReason::InputGeometryUnavailable {
            reason: StructuredUnknownV2::Capacity,
            work_exhausted: true,
        }
    )));
}

#[test]
fn checked_bootstrap_roles_balance_execution_roles_before_policy_and_width_expansion() {
    let (mut cases, mut opportunities, mut facts, population) = inventory();
    cases.truncate(4);
    opportunities.truncate(4);
    facts.truncate(4);
    for (width, model_eos) in [(2_u32, true), (1_u32, false)] {
        for maximum in [1_u64, 2] {
            let input = related::prefill_input_phase(width, 3, 3, maximum, Configured, model_eos);
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
            let mut case = cases[0].clone();
            case.width = usize::try_from(width).unwrap();
            case.maximum_output = NonZeroUsize::new(usize::try_from(maximum).unwrap()).unwrap();
            case.suffix_tokens = case.maximum_output.get();
            cases.push(case);
        }
    }
    assert_ne!(
        facts[0][0].owner.installed_policy,
        facts[4][0].owner.installed_policy
    );
    assert_eq!(
        facts[0][0].homogeneous_host_policy,
        facts[4][0].homogeneous_host_policy
    );
    assert_ne!(
        facts[0][0].homogeneous_host_policy,
        facts[6][0].homogeneous_host_policy
    );
    // Releasing cold canonical metadata must not lose the bounded role key.
    for fact in facts.iter_mut().flatten() {
        fact.original = None;
    }
    let mut candidates = Vec::new();
    for (index, range) in [0..2, 2..4, 4..6, 6..8].into_iter().enumerate() {
        let original = select_inventory(
            &cases[range.clone()],
            &opportunities[range.clone()],
            &facts[range.clone()],
            &population,
            None,
        );
        assert_eq!(original.batches.len(), 1);
        let batch = &original.batches[0];
        assert!(batch.scheduled && batch.schedule_within_capacity);
        let indices: Vec<_> = range.collect();
        candidates.push(PopulationCandidate {
            coverage: CoveragePriority::from_cases(&indices, &cases).unwrap(),
            candidates: indices.clone(),
            population_index: index,
            input_priority: input_priority(
                indices.iter().any(|&i| facts[i][0].branches[4]),
                indices.iter().any(|&i| facts[i][0].branches[5]),
            ),
            declared_work: (
                batch.serial_token_work,
                batch.serial_wave_upper_bound,
                batch.requests,
            ),
            coverage_round: 0,
            decode_width_tier: 0,
        });
    }
    priority::order_populations(&mut candidates, &facts);
    let round = |index| {
        candidates
            .iter()
            .find(|c| c.population_index == index)
            .unwrap()
            .coverage_round
    };
    assert_eq!(
        [0, 2, 3]
            .into_iter()
            .filter(|&index| round(index) == 0)
            .count(),
        1,
        "only the lighter complete source of this execution role receives first-round coverage"
    );
    assert_eq!(
        round(1),
        0,
        "independent checked Decode is a required execution role"
    );
    assert!(
        round(0).max(round(2)) > 0,
        "a wider batch cannot manufacture a second first-round role"
    );
    assert!(
        [0, 2, 3].into_iter().any(|index| round(index) > round(1)),
        "additional original Prefill policies and widths cannot preempt the first complete Decode source"
    );
    assert_ne!(
        facts[0][0].key(population.population_policy()),
        facts[4][0].key(population.population_policy()),
        "ordering never merges the original physical populations"
    );
}
