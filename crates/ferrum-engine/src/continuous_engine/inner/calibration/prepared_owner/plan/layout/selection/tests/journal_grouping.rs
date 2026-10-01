use super::*;

pub(super) fn same_product_families() -> (
    Vec<Case>,
    Vec<CaseOpportunity>,
    Vec<Vec<CheckedInputFacts>>,
    StructuredServiceDeclarationV7,
) {
    let (original, _, _, population) = inventory();
    let mut cases = Vec::new();
    let mut opportunities = Vec::new();
    let mut facts = Vec::new();
    for (name, implementation) in [
        ("fixture.journal.short", [31; 32]),
        ("fixture.journal.long", [32; 32]),
    ] {
        for case in &original {
            let input = natural_termination_input_with_algorithm_and_eos(
                case.width as u32,
                CostProductOutput::GreedyToken,
                false,
                64,
                name,
                implementation,
                false,
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
            cases.push(case.clone());
        }
    }
    (cases, opportunities, facts, population)
}

#[derive(Default)]
struct Inventory {
    cases: Vec<Case>,
    opportunities: Vec<CaseOpportunity>,
    facts: Vec<Vec<CheckedInputFacts>>,
    prompts: Vec<usize>,
}

impl Inventory {
    fn family(
        &mut self,
        name: &'static str,
        implementation: [u8; 32],
        widths: &[usize],
        prompts: &[usize],
        product: CostProductOutput,
    ) {
        for &prompt in prompts {
            let template = self.prompts.len();
            self.prompts.push(prompt);
            for &width in widths {
                let input = natural_termination_input_with_algorithm_and_eos(
                    width as u32,
                    product,
                    true,
                    (prompt + 3) as u32,
                    name,
                    implementation,
                    false,
                );
                self.opportunities.push(CaseOpportunity {
                    population: classify_alternatives(
                        std::slice::from_ref(&input),
                        StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
                        true,
                    )
                    .unwrap(),
                    minimum_fresh_members: 1,
                });
                self.facts
                    .push(vec![input_facts(&StructuredQueryV2::exact(input)).unwrap()]);
                self.cases.push(Case {
                    product: if product == CostProductOutput::FullLogits {
                        OpportunityProduct::Full
                    } else {
                        OpportunityProduct::Greedy
                    },
                    template,
                    width,
                    maximum_output: NonZeroUsize::new(4).unwrap(),
                    release_generated: 3,
                    suffix_tokens: 1,
                    preset: SloAutomaticCostProbeSamplingPresetV1::Configured,
                    prefix: PrefixKind::Clean,
                    route: if product == CostProductOutput::FullLogits {
                        CalibrationDecodeRoute::FullLogits
                    } else {
                        CalibrationDecodeRoute::Actual
                    },
                    reset: false,
                });
            }
        }
    }

    fn select(
        &self,
        requests: usize,
        waves: usize,
        sources: Option<NonZeroUsize>,
    ) -> CheckedSelection {
        select_changed_with_geometry_and_source_limit(
            &self.cases,
            &self.opportunities,
            &self.facts,
            &self.prompts,
            8,
            None,
            &population::declaration(&Default::default(), fixture::domain()).unwrap(),
            requests,
            waves,
            usize::MAX,
            None,
            None,
            None,
            sources,
        )
        .unwrap()
    }
}

#[test]
fn checked_decode_journal_covers_context_families_before_wider_auxiliary_sources() {
    let mut input = Inventory::default();
    input.family(
        "fixture.journal.short",
        [31; 32],
        &[1],
        &[3, 5],
        CostProductOutput::GreedyToken,
    );
    input.family(
        "fixture.journal.long",
        [32; 32],
        &[1],
        &[127, 129],
        CostProductOutput::GreedyToken,
    );
    input.family(
        "fixture.journal.wide",
        [33; 32],
        &[3],
        &[3, 5],
        CostProductOutput::GreedyToken,
    );
    input.family(
        "fixture.journal.full",
        [34; 32],
        &[1],
        &[3, 5],
        CostProductOutput::FullLogits,
    );
    let original =
        serde_json::to_value((&input.cases, &input.opportunities, &input.facts)).unwrap();
    let selected = input.select(100_000, 10_000_000, NonZeroUsize::new(1));
    assert_eq!(selected.populations.len(), 4);
    let batch = selected
        .batches
        .iter()
        .find(|batch| batch.scheduled)
        .unwrap();
    assert_eq!(batch.population_indices, [0, 1]);
    assert_eq!(batch.representative_case_indices, [0, 1, 2, 3]);
    assert!(batch.algorithm_universe.is_none());
    assert!(selected
        .batches
        .iter()
        .skip(1)
        .all(|batch| !batch.scheduled));
    assert!(selected.gaps.iter().any(|gap| matches!(
        gap.reason,
        SelectionGapReason::RetainedSourceCapacity { maximum_sources: 1 }
    )));
    for member in &selected.populations[..2] {
        assert!(member.scheduled);
        for &index in &member.representative_case_indices {
            assert_eq!(
                input.facts[index][0]
                    .key(StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1),
                member.key
            );
            assert!(batch.representative_case_indices.contains(&index));
        }
    }
    assert_ne!(selected.populations[0].key, selected.populations[1].key);
    assert!(batch
        .schedule
        .phase_min_offered
        .iter()
        .all(|minimum| *minimum >= batch.maximum_anchor_span));
    assert_eq!(
        selected.execution_case_indices,
        batch
            .representative_case_indices
            .repeat(batch.planned_cycles)
    );
    assert_eq!(
        selected.requests,
        selected
            .execution_case_indices
            .iter()
            .map(|&i| input.cases[i].width)
            .sum::<usize>()
    );
    assert_eq!(
        selected.serial_wave_upper_bound,
        selected
            .execution_case_indices
            .iter()
            .map(|&i| {
                input.cases[i]
                    .waves(input.prompts[input.cases[i].template], 8)
                    .unwrap()
                    .1
            })
            .sum::<usize>()
    );
    assert_eq!(
        original,
        serde_json::to_value((&input.cases, &input.opportunities, &input.facts)).unwrap()
    );
}

#[test]
fn checked_decode_journal_never_trims_a_broad_population_to_the_first_width() {
    let mut input = Inventory::default();
    input.family(
        "fixture.journal.narrow",
        [31; 32],
        &[1],
        &[31, 33],
        CostProductOutput::GreedyToken,
    );
    input.family(
        "fixture.journal.broad",
        [32; 32],
        &[1, 3],
        &[3, 5],
        CostProductOutput::GreedyToken,
    );
    let selected = input.select(100_000, 10_000_000, NonZeroUsize::new(1));
    assert_eq!(selected.populations.len(), 2);
    assert!(selected.populations[0].scheduled);
    assert!(!selected.populations[1].scheduled);
    let broad = &selected.populations[1];
    assert!(broad
        .representative_case_indices
        .iter()
        .any(|&i| input.cases[i].width == 1));
    assert!(broad
        .representative_case_indices
        .iter()
        .any(|&i| input.cases[i].width == 3));
    assert!(selected
        .gaps
        .iter()
        .any(|gap| gap.population.as_ref() == Some(&broad.key)
            && matches!(
                gap.reason,
                SelectionGapReason::RetainedSourceCapacity { .. }
            )));
}

#[test]
fn checked_decode_journal_combination_cannot_borrow_requests_or_waves() {
    let mut input = Inventory::default();
    input.family(
        "fixture.journal.short",
        [31; 32],
        &[1],
        &[3, 5],
        CostProductOutput::GreedyToken,
    );
    input.family(
        "fixture.journal.long",
        [32; 32],
        &[1],
        &[127, 129],
        CostProductOutput::GreedyToken,
    );
    let full = input.select(100_000, 10_000_000, None);
    assert_eq!(full.batches.len(), 1);
    for (requests, waves, request_limited) in [
        (full.requests - 1, 10_000_000, true),
        (100_000, full.serial_wave_upper_bound - 1, false),
    ] {
        let selected = input.select(requests, waves, None);
        assert_eq!(selected.batches.len(), 2);
        assert!(selected
            .batches
            .iter()
            .all(|batch| batch.population_indices.len() == 1));
        assert!(selected.populations.iter().any(|member| !member.scheduled));
        assert!(selected.requests <= requests && selected.serial_wave_upper_bound <= waves);
        assert!(selected.gaps.iter().any(|gap| if request_limited {
            matches!(gap.reason, SelectionGapReason::RemainingRequests { .. })
        } else {
            matches!(gap.reason, SelectionGapReason::RemainingWaves { .. })
        }));
    }
}

#[test]
fn checked_decode_journal_preserves_prior_and_later_complete_source_reservations() {
    let mut input = Inventory::default();
    input.family(
        "fixture.journal.short",
        [31; 32],
        &[1],
        &[3, 5],
        CostProductOutput::GreedyToken,
    );
    let left = input.select(100_000, 10_000_000, None).batches.remove(0);
    let mut other = Inventory::default();
    other.family(
        "fixture.journal.long",
        [32; 32],
        &[1],
        &[127, 129],
        CostProductOutput::GreedyToken,
    );
    let right = other.select(100_000, 10_000_000, None).batches.remove(0);
    input.family(
        "fixture.journal.long",
        [32; 32],
        &[1],
        &[127, 129],
        CostProductOutput::GreedyToken,
    );
    let merged = input.select(100_000, 10_000_000, None);
    assert_eq!(merged.batches.len(), 1);
    let combined = &merged.batches[0];
    assert!(combined.requests > left.requests);
    assert!(combined.serial_wave_upper_bound > left.serial_wave_upper_bound);
    let candidate = |batch: SelectedBatch, index| BatchCandidate {
        input_priority: 0,
        coverage: CoveragePriority::from_cases(&[0], &input.cases).unwrap(),
        coverage_round: 0,
        decode_width_tier: 1,
        original_population_index: index,
        batch,
    };
    // These are complete schedules derived from real canonical inputs, not
    // hand-edited request/wave costs. Another complete journal consumes the
    // same work as the checked combined schedule at the preceding/later slot.
    let preceding = candidate(combined.clone(), 0);
    let requests = preceding.batch.requests + left.requests;
    let waves = preceding.batch.serial_wave_upper_bound + left.serial_wave_upper_bound;
    assert!(combined.requests <= requests);
    assert!(combined.serial_wave_upper_bound <= waves);
    let candidates = [
        preceding,
        candidate(left.clone(), 1),
        candidate(right.clone(), 2),
    ];
    assert!(!grouping::preserves_scheduled(
        &candidates,
        1,
        2,
        combined,
        requests,
        usize::MAX,
        None,
        None,
    )
    .unwrap());
    assert!(!grouping::preserves_scheduled(
        &candidates,
        1,
        2,
        combined,
        usize::MAX,
        waves,
        None,
        None,
    )
    .unwrap());
    let later = [
        candidate(left, 0),
        candidate(combined.clone(), 1),
        candidate(right, 2),
    ];
    assert!(!grouping::preserves_scheduled(
        &later,
        0,
        2,
        combined,
        requests,
        usize::MAX,
        None,
        None,
    )
    .unwrap());
    assert!(grouping::preserves_scheduled(
        &later,
        0,
        2,
        combined,
        usize::MAX,
        usize::MAX,
        None,
        None,
    )
    .unwrap());
    assert!(grouping::preserves_scheduled(
        &later,
        0,
        2,
        combined,
        usize::MAX,
        usize::MAX,
        None,
        NonZeroUsize::new(1),
    )
    .unwrap());
}
