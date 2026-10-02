use super::*;

fn native_source_inventory() -> (
    Vec<Case>,
    Vec<CaseOpportunity>,
    Vec<Vec<CheckedInputFacts>>,
    StructuredServiceDeclarationV7,
) {
    use ferrum_interfaces::vnext::CheckpointTokenSpanConstraint;
    use std::num::NonZeroU64;

    let (mut cases, opportunities, inputs, population) = journal_grouping::same_product_families();
    let blueprint = work::PrefixBlueprint {
        prompt_tokens: 61,
        boundary: 60,
        span: CheckpointTokenSpanConstraint::new(
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(2).unwrap(),
        )
        .unwrap(),
        input_tokens_sha256: [7; 32],
    };
    for case in &mut cases {
        case.acquisition = work::declared_plan(
            case,
            blueprint,
            NonZeroU32::new(8).unwrap(),
            NonZeroU32::new(2),
        )
        .unwrap();
        assert!(case.acquisition.is_some());
    }
    (cases, opportunities, inputs, population)
}

fn select_native_source(
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    population: &StructuredServiceDeclarationV7,
    capacity: SelectionCapacity,
) -> CheckedSelection {
    select_with_capacity(
        cases,
        opportunities,
        inputs,
        &[61],
        8,
        NonZeroU32::new(2),
        population,
        capacity,
        usize::MAX,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

#[test]
fn native_source_coalescing_charges_one_setup_without_advancing_offered_clock() {
    let (cases, opportunities, inputs, population) = native_source_inventory();
    let selected = select_native_source(
        &cases,
        &opportunities,
        &inputs,
        &population,
        SelectionCapacity::legacy(100_000, 10_000_000),
    );
    assert!(selected.gaps.is_empty());
    assert_eq!(selected.batches.len(), 1);
    let combined = &selected.batches[0];
    assert!(combined.scheduled);
    assert_eq!(combined.population_indices.len(), 2);
    assert_eq!(combined.representative_case_indices, [0, 2, 4, 6]);

    // Each family executes width-one and width-four cohorts: one suffix
    // token per row, twenty decode waves, and one restore per fresh request.
    // The sixty-token seed has thirty two-token prefills and one capture.
    assert_eq!(combined.schedule.block_offered, 2 * (21 + 24));
    assert_eq!(
        combined
            .input_opportunities
            .successful_cycle_wave_upper_bound,
        90
    );
    assert_eq!(
        combined
            .input_opportunities
            .minimum_original_offers_per_completed_cycle,
        2 * (4 + 7)
    );
    let cycles = combined.planned_cycles;
    assert_eq!(combined.requests, 10 * cycles + 1);
    assert_eq!(combined.serial_wave_upper_bound, 220 * cycles + 31);
    assert_eq!(combined.declared_offer_row_bound, 210 * cycles);
    assert_eq!(combined.serial_token_work, 210 * cycles + 60);
    assert_eq!(selected.requests, combined.requests);
    assert_eq!(
        selected.declared_offer_row_bound,
        combined.declared_offer_row_bound
    );
    assert_eq!(
        selected.serial_wave_upper_bound,
        combined.serial_wave_upper_bound
    );

    // Before coalescing, each complete source owns its own seed. The merged
    // plan must recalculate the source-local keys, not sum these setup charges.
    for &index in &combined.population_indices {
        let separate = batch_plan(
            &[index],
            &selected.populations,
            &cases,
            &opportunities,
            &[61],
            8,
            NonZeroU32::new(2),
            &population,
        )
        .unwrap();
        assert_eq!(separate.requests, 5 * separate.planned_cycles + 1);
        assert_eq!(
            separate.serial_wave_upper_bound,
            110 * separate.planned_cycles + 31
        );
        assert_eq!(
            separate.serial_token_work,
            105 * separate.planned_cycles + 60
        );
    }
}

#[test]
fn native_source_requires_complete_setup_and_restore_budget_before_selection() {
    let (cases, opportunities, inputs, population) = native_source_inventory();
    let selected = select_native_source(
        &cases,
        &opportunities,
        &inputs,
        &population,
        SelectionCapacity::legacy(100_000, 10_000_000),
    );
    let batch = &selected.batches[0];
    assert!(batch.scheduled);
    let offered_only = batch.schedule.block_offered * batch.planned_cycles;
    assert!(offered_only < batch.serial_wave_upper_bound);
    for (requests, actions, fits) in [
        (batch.requests, batch.serial_wave_upper_bound, true),
        (batch.requests - 1, batch.serial_wave_upper_bound, false),
        (batch.requests, batch.serial_wave_upper_bound - 1, false),
        (batch.requests, offered_only, false),
    ] {
        let mut out = CheckedSelection {
            populations: selected.populations.clone(),
            ..CheckedSelection::default()
        };
        for member in &mut out.populations {
            member.scheduled = false;
            member.batch_index = None;
        }
        append_batch(
            &mut out,
            batch.clone(),
            SelectionCapacity::legacy(requests, actions),
            None,
        )
        .unwrap();
        assert_eq!(out.batches[0].scheduled, fits);
        if fits {
            assert!(out.gaps.is_empty());
            assert_eq!(out.requests, requests);
            assert_eq!(out.serial_wave_upper_bound, actions);
            assert_eq!(
                out.execution_case_indices.len(),
                batch.representative_case_indices.len() * batch.planned_cycles
            );
        } else {
            assert_eq!((out.requests, out.serial_wave_upper_bound), (0, 0));
            assert!(out.execution_case_indices.is_empty());
            assert!(out.gaps.iter().any(|gap| match gap.reason {
                SelectionGapReason::RemainingRequests {
                    required,
                    remaining,
                } => required == batch.requests && remaining == requests,
                SelectionGapReason::RemainingWaves {
                    required,
                    remaining,
                } => required == batch.serial_wave_upper_bound && remaining == actions,
                _ => false,
            }));
        }
    }
}

#[test]
fn native_source_selection_separates_inference_row_capacity_from_execution_actions() {
    let (cases, opportunities, inputs, population) = native_source_inventory();
    // Check both an independent source and the source coalesced from two
    // numerical families. Capture/restore must not consume the row allowance.
    for count in [4, 8] {
        let run = |capacity| {
            select_native_source(
                &cases[..count],
                &opportunities[..count],
                &inputs[..count],
                &population,
                capacity,
            )
        };
        let full = run(SelectionCapacity::legacy(100_000, 10_000_000));
        assert_eq!(full.batches.len(), 1);
        let row_bound = full.declared_offer_row_bound;
        assert!(full.serial_wave_upper_bound > row_bound);
        assert!(row_bound > 0);
        for rows in [row_bound, row_bound - 1] {
            let capacity = SelectionCapacity {
                requests: 100_000,
                execution_actions: 10_000_000,
                declared_offer_rows: rows,
            };
            let selected = run(capacity);
            assert!(selected.declared_offer_row_bound <= rows);
            assert!(selected.serial_wave_upper_bound <= capacity.execution_actions);
            assert!(selected.requests <= capacity.requests);
            if rows == row_bound {
                assert_eq!(selected.execution_case_indices, full.execution_case_indices);
                assert_eq!(selected.declared_offer_row_bound, row_bound);
                assert_eq!(
                    selected.serial_wave_upper_bound,
                    full.serial_wave_upper_bound
                );
                assert!(selected.populations.iter().all(|member| member.scheduled));
                assert!(selected.gaps.is_empty());
            } else {
                assert!(selected.populations.iter().any(|member| !member.scheduled));
                assert!(selected.gaps.iter().any(|gap| matches!(
                    gap.reason, SelectionGapReason::RemainingOfferRows { required, remaining }
                        if required > remaining
                )));
                assert!(!selected.gaps.iter().any(|gap| matches!(
                    gap.reason,
                    SelectionGapReason::RemainingRequests { .. }
                        | SelectionGapReason::RemainingWaves { .. }
                )));
            }
        }
    }
}

fn original_batches() -> (Vec<BatchCandidate>, Vec<Case>, Vec<Vec<CheckedInputFacts>>) {
    let (cases, opportunities, inputs, population) = inventory();
    let mut selected = select(
        &cases,
        &opportunities,
        &inputs,
        &[61],
        8,
        &population,
        100_000,
        10_000_000,
        usize::MAX,
    )
    .unwrap();
    let batch = selected.batches.remove(0);
    let candidate = |batch| BatchCandidate {
        input_priority: 0,
        coverage: CoveragePriority::from_cases(&[0], &cases).unwrap(),
        coverage_round: 0,
        decode_width_tier: 0,
        original_population_index: 0,
        batch,
    };
    let mut batches = vec![
        candidate(batch.clone()),
        candidate(batch.clone()),
        candidate(batch),
    ];
    for (index, value) in batches.iter_mut().enumerate() {
        value.original_population_index = index;
    }
    (batches, cases, inputs)
}

#[test]
fn local_combination_reserves_later_original_sources_and_counts_complete_work() {
    let (batches, _, _) = original_batches();
    let mut single = SelectionCapacity::default();
    single.charge(&batches[0].batch).unwrap();
    let planned = super::super::composition::scheduled_prefix_budget(
        &batches,
        SelectionCapacity::legacy(100_000, 10_000_000),
        None,
        NonZeroUsize::new(2),
    )
    .unwrap();
    assert_eq!(
        planned,
        SelectionCapacity {
            requests: 2 * single.requests,
            execution_actions: 2 * single.execution_actions,
            declared_offer_rows: 2 * single.declared_offer_rows,
        }
    );
    let work_limited = super::super::composition::scheduled_prefix_budget(
        &batches,
        single,
        None,
        NonZeroUsize::new(3),
    )
    .unwrap();
    assert_eq!(work_limited, single);
    let row_limited = super::super::composition::scheduled_prefix_budget(
        &batches,
        SelectionCapacity {
            requests: 100_000,
            execution_actions: 10_000_000,
            declared_offer_rows: 2 * single.declared_offer_rows - 1,
        },
        None,
        NonZeroUsize::new(3),
    )
    .unwrap();
    assert_eq!(row_limited, single);
    let deferred = super::super::composition::scheduled_prefix_budget(
        &batches,
        SelectionCapacity::legacy(100_000, 10_000_000),
        Some(1),
        NonZeroUsize::new(3),
    )
    .unwrap();
    assert_eq!(deferred, SelectionCapacity::default());
}

#[test]
fn local_combination_cannot_promote_an_invalid_raw_source_into_a_reserved_slot() {
    let (mut batches, _, _) = original_batches();
    batches[0].batch.schedule_within_capacity = false;
    assert!(!super::super::composition::can_schedule(
        &batches[0].batch,
        SelectionCapacity::legacy(usize::MAX, usize::MAX)
    ));
    batches[1].batch.maximum_anchor_span = batches[1]
        .batch
        .schedule
        .phase_min_offered
        .iter()
        .min()
        .unwrap()
        .checked_add(1)
        .unwrap();
    assert!(!super::super::composition::can_schedule(
        &batches[1].batch,
        SelectionCapacity::legacy(usize::MAX, usize::MAX)
    ));
    let mut expected = SelectionCapacity::default();
    expected.charge(&batches[2].batch).unwrap();
    assert_eq!(
        super::super::composition::scheduled_prefix_budget(
            &batches,
            SelectionCapacity::legacy(usize::MAX, usize::MAX),
            None,
            NonZeroUsize::new(1)
        )
        .unwrap(),
        expected
    );
}

#[test]
fn local_combination_peak_authorization_preserves_raw_path_on_capacity_denial() {
    let (cases, opportunities, inputs, population) = inventory();
    let originals = inputs
        .iter()
        .flatten()
        .map(|f| f.original.as_deref().unwrap());
    let seed = ferrum_scheduler::implementations::continuous::cost_model::structured_v2::
        DeclaredAlgorithmUniverseV1::from_inputs(originals, population.settings.max_axes).unwrap();
    assert!(!composition_authorized(
        &opportunities,
        &inputs,
        100_000,
        &population,
        false,
        0,
        &seed
    )
    .unwrap());
    let raw = select(
        &cases,
        &opportunities,
        &inputs,
        &[61],
        8,
        &population,
        100_000,
        10_000_000,
        usize::MAX,
    )
    .unwrap();
    assert!(raw.batches.iter().any(|b| b.scheduled));
    assert!(raw.batches.iter().all(|b| b.algorithm_universe.is_none()));
    // A declaration consumes additional retained bytes even when this one-
    // algorithm fixture cannot benefit from a wider local source.
    assert!(
        super::super::composition::extra_peak(&opportunities, &seed).unwrap()
            > seed.retained_payload_bytes().unwrap()
    );
}

fn algorithm_pair_inventory(
    width_scales: [usize; 2],
) -> (
    Vec<Case>,
    Vec<CaseOpportunity>,
    Vec<Vec<CheckedInputFacts>>,
    StructuredServiceDeclarationV7,
) {
    let (original_cases, _, _, population) = inventory();
    let mut cases = Vec::new();
    let mut opportunities = Vec::new();
    let mut inputs = Vec::new();
    for (template, native_op_id, implementation) in [
        (0, "fixture.selection.a", [7; 32]),
        (1, "fixture.selection.b", [8; 32]),
    ] {
        for old in &original_cases[..2] {
            let mut case = old.clone();
            case.width *= width_scales[template];
            case.template = template;
            case.release_generated = 0;
            case.suffix_tokens = case.maximum_output.get();
            let input = natural_termination_input_with_algorithm(
                case.width as u32,
                CostProductOutput::GreedyToken,
                false,
                64,
                native_op_id,
                implementation,
            );
            opportunities.push(CaseOpportunity {
                population: classify_alternatives(
                    std::slice::from_ref(&input),
                    StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
                    true,
                )
                .unwrap(),
                minimum_fresh_members: 1,
            });
            inputs.push(vec![input_facts(&StructuredQueryV2::exact(input)).unwrap()]);
            cases.push(case);
        }
    }
    (cases, opportunities, inputs, population)
}

#[test]
fn local_combination_declares_only_checked_compatible_sources_before_collection() {
    // Different complete width sets deliberately remain separate raw
    // journals. Same-width context/algorithm families now share a journal.
    let (cases, opportunities, inputs, population) = algorithm_pair_inventory([1, 2]);
    let seed = ferrum_scheduler::implementations::continuous::cost_model::structured_v2::
        DeclaredAlgorithmUniverseV1::from_inputs(inputs.iter().flatten()
            .map(|f| f.original.as_deref().unwrap()), population.settings.max_axes).unwrap();
    let selected = select_with_local_composition(
        &cases,
        &opportunities,
        &inputs,
        &[61, 61],
        8,
        None,
        &population,
        100_000,
        10_000_000,
        usize::MAX,
        None,
        None,
        None,
        NonZeroUsize::new(2),
        Some(&seed),
    )
    .unwrap();
    let scheduled: Vec<_> = selected.batches.iter().filter(|b| b.scheduled).collect();
    assert_eq!(scheduled.len(), 2);
    assert!(scheduled[0].algorithm_universe.is_none());
    let local = scheduled[1].algorithm_universe.as_ref().unwrap();
    assert!(seed.contains_universe(local));
    assert_eq!(local.algorithm_count(), 2);
    assert_eq!(scheduled[1].population_indices.len(), 2);
    for member in &selected.populations {
        assert!(member.scheduled);
        assert!(member
            .representative_case_indices
            .iter()
            .all(|index| scheduled[1].representative_case_indices.contains(index)));
    }
    // The source is a declared acquisition plan. It carries no fitted model,
    // observed timing, or imported qualification and must execute both families.
    let actual_requests: usize = selected
        .execution_case_indices
        .iter()
        .map(|&index| cases[index].width)
        .sum();
    let actual_waves: usize = selected
        .execution_case_indices
        .iter()
        .map(|&index| cases[index].waves(61, 8).unwrap().1)
        .sum();
    assert_eq!(selected.requests, actual_requests);
    assert_eq!(selected.serial_wave_upper_bound, actual_waves);
    assert_eq!(selected.declared_offer_row_bound, actual_waves);
    let raw = select_with_local_composition(
        &cases,
        &opportunities,
        &inputs,
        &[61, 61],
        8,
        None,
        &population,
        100_000,
        10_000_000,
        usize::MAX,
        None,
        None,
        None,
        NonZeroUsize::new(2),
        None,
    )
    .unwrap();
    assert!(selected.declared_offer_row_bound > raw.declared_offer_row_bound);
    // Only the inference-row allowance is tight. Optional extra scope cannot
    // displace either raw source even when requests and actions still fit.
    let row_limited = select_with_capacity(
        &cases,
        &opportunities,
        &inputs,
        &[61, 61],
        8,
        None,
        &population,
        SelectionCapacity {
            requests: 100_000,
            execution_actions: 10_000_000,
            declared_offer_rows: raw.declared_offer_row_bound,
        },
        usize::MAX,
        None,
        None,
        None,
        NonZeroUsize::new(2),
        Some(&seed),
    )
    .unwrap();
    assert_eq!(
        row_limited.execution_case_indices,
        raw.execution_case_indices
    );
    assert_eq!(
        row_limited.declared_offer_row_bound,
        raw.declared_offer_row_bound
    );
    assert!(row_limited
        .batches
        .iter()
        .all(|batch| batch.algorithm_universe.is_none()));
    assert!(row_limited
        .gaps
        .iter()
        .any(|gap| matches!(gap.reason, SelectionGapReason::CombinationWorkCapacity)));
    let incomplete = ferrum_scheduler::implementations::continuous::cost_model::structured_v2::
        DeclaredAlgorithmUniverseV1::from_inputs(inputs[..2].iter().flatten()
            .map(|f| f.original.as_deref().unwrap()), population.settings.max_axes).unwrap();
    let excluded = select_with_local_composition(
        &cases,
        &opportunities,
        &inputs,
        &[61, 61],
        8,
        None,
        &population,
        100_000,
        10_000_000,
        usize::MAX,
        None,
        None,
        None,
        NonZeroUsize::new(2),
        Some(&incomplete),
    )
    .unwrap();
    assert!(excluded
        .batches
        .iter()
        .all(|b| b.algorithm_universe.is_none()));
}

#[test]
fn same_width_algorithm_families_keep_raw_sources_and_complete_combination_plan() {
    let (cases, opportunities, inputs, population) = algorithm_pair_inventory([1, 1]);
    assert!(cases.iter().all(|case| case.acquisition.is_none()));
    assert!(inputs[0][0].family.is_some());
    assert!(inputs[2][0].family.is_some());
    assert_ne!(inputs[0][0].family, inputs[2][0].family);
    let seed = ferrum_scheduler::implementations::continuous::cost_model::structured_v2::
        DeclaredAlgorithmUniverseV1::from_inputs(inputs.iter().flatten()
            .map(|fact| fact.original.as_deref().unwrap()), population.settings.max_axes).unwrap();
    assert_eq!(seed.algorithm_count(), 2);
    let raw = select_with_local_composition(
        &cases,
        &opportunities,
        &inputs,
        &[61, 61],
        8,
        None,
        &population,
        100_000,
        10_000_000,
        usize::MAX,
        None,
        None,
        None,
        NonZeroUsize::new(2),
        None,
    )
    .unwrap();
    assert_eq!(raw.populations.len(), 2);
    assert_ne!(raw.populations[0].key, raw.populations[1].key);
    assert!(raw.populations.iter().all(|member| member.scheduled));
    assert_eq!(raw.batches.len(), 1);
    assert!(raw.batches[0].scheduled);
    assert!(raw.batches[0].algorithm_universe.is_none());
    assert_eq!(raw.batches[0].population_indices.len(), 2);

    // Two complete executions fit the same two-source cap: the original raw
    // families and a separate A+B scope. Neither scope borrows fitted samples.
    let capacity = SelectionCapacity {
        requests: raw.requests.checked_mul(2).unwrap(),
        execution_actions: raw.serial_wave_upper_bound.checked_mul(2).unwrap(),
        declared_offer_rows: raw.declared_offer_row_bound.checked_mul(2).unwrap(),
    };
    let selected = select_with_capacity(
        &cases,
        &opportunities,
        &inputs,
        &[61, 61],
        8,
        None,
        &population,
        capacity,
        usize::MAX,
        None,
        None,
        None,
        NonZeroUsize::new(2),
        Some(&seed),
    )
    .unwrap();
    let scheduled: Vec<_> = selected
        .batches
        .iter()
        .filter(|batch| batch.scheduled)
        .collect();
    for member in &selected.populations {
        assert!(member.scheduled);
        assert!(
            scheduled.iter().any(|batch| {
                batch.algorithm_universe.is_none()
                    && member
                        .representative_case_indices
                        .iter()
                        .all(|index| batch.representative_case_indices.contains(index))
            }),
            "each original family retains its own raw numerical scope"
        );
    }
    let combination = scheduled
        .iter()
        .find(|batch| batch.algorithm_universe.is_some())
        .expect("same-width journal coalescing must retain a separate A+B collection plan");
    let local = combination.algorithm_universe.as_ref().unwrap();
    assert!(seed.contains_universe(local));
    assert_eq!(local.algorithm_count(), 2);
    assert_eq!(combination.population_indices.len(), 2);
    assert!(selected.populations.iter().all(|member| {
        member
            .representative_case_indices
            .iter()
            .all(|index| combination.representative_case_indices.contains(index))
    }));
    assert_eq!(scheduled.len(), 2);
    assert!(selected.requests <= capacity.requests);
    assert!(selected.serial_wave_upper_bound <= capacity.execution_actions);
    assert!(selected.declared_offer_row_bound <= capacity.declared_offer_rows);

    // These are input opportunities for all three numerical phases, not
    // qualification evidence. Each planned source still executes fresh cohorts.
    let fit_members = population
        .settings
        .min_phase_samples
        .max(population.settings.max_rank + population.settings.min_fit_redundancy);
    let mut cursor = 0;
    for batch in scheduled {
        assert!(batch.schedule_within_capacity);
        assert!(batch.schedule.min_members[0] >= fit_members);
        for phase in 0..3 {
            assert!(batch.schedule.min_members[phase] >= population.settings.min_phase_samples);
            assert!(batch.input_opportunities.phase_cycles[phase] > 0);
            assert!(
                batch.input_opportunities.phase_original_offer_bounds[phase]
                    >= batch.input_opportunities.maximum_fresh_member_span[phase]
                        .max(batch.schedule.phase_min_offered[phase])
            );
        }
        assert!(
            batch.maximum_anchor_span <= *batch.schedule.phase_min_offered.iter().min().unwrap()
        );
        assert!(batch.planned_cycles >= batch.input_opportunities.planned_cycles);
        let count = batch.representative_case_indices.len() * batch.planned_cycles;
        let execution = &selected.execution_case_indices[cursor..cursor + count];
        for cycle in execution.chunks_exact(batch.representative_case_indices.len()) {
            assert_eq!(cycle, batch.representative_case_indices);
        }
        cursor += count;
    }
    assert_eq!(cursor, selected.execution_case_indices.len());
    assert!(selected.requests > raw.requests);
    assert!(selected.serial_wave_upper_bound > raw.serial_wave_upper_bound);
    assert!(selected.declared_offer_row_bound > raw.declared_offer_row_bound);

    let complete = SelectionCapacity {
        requests: selected.requests,
        execution_actions: selected.serial_wave_upper_bound,
        declared_offer_rows: selected.declared_offer_row_bound,
    };
    let incomplete = ferrum_scheduler::implementations::continuous::cost_model::structured_v2::
        DeclaredAlgorithmUniverseV1::from_inputs(inputs[..2].iter().flatten()
            .map(|fact| fact.original.as_deref().unwrap()), population.settings.max_axes).unwrap();
    for (reason, capacity, sources, seed) in [
        (
            "request",
            SelectionCapacity {
                requests: complete.requests - 1,
                ..complete
            },
            2,
            &seed,
        ),
        (
            "action",
            SelectionCapacity {
                execution_actions: complete.execution_actions - 1,
                ..complete
            },
            2,
            &seed,
        ),
        (
            "row",
            SelectionCapacity {
                declared_offer_rows: complete.declared_offer_rows - 1,
                ..complete
            },
            2,
            &seed,
        ),
        ("source", complete, 1, &seed),
        ("seed", complete, 2, &incomplete),
    ] {
        let denied = select_with_capacity(
            &cases,
            &opportunities,
            &inputs,
            &[61, 61],
            8,
            None,
            &population,
            capacity,
            usize::MAX,
            None,
            None,
            None,
            NonZeroUsize::new(sources),
            Some(seed),
        )
        .unwrap();
        assert!(
            denied
                .batches
                .iter()
                .all(|batch| batch.algorithm_universe.is_none()),
            "{reason}"
        );
        assert!(
            denied.populations.iter().all(|member| member.scheduled),
            "{reason}"
        );
        assert_eq!(
            denied.execution_case_indices, raw.execution_case_indices,
            "{reason}"
        );
        assert_eq!(denied.requests, raw.requests, "{reason}");
        assert_eq!(
            denied.serial_wave_upper_bound, raw.serial_wave_upper_bound,
            "{reason}"
        );
        assert_eq!(
            denied.declared_offer_row_bound, raw.declared_offer_row_bound,
            "{reason}"
        );
    }
}

#[test]
fn same_width_combination_preserves_later_original_source_at_same_or_lower_policy() {
    use SloAutomaticCostProbeSamplingPresetV1::{Configured, GreedyLength};

    for preset in [Configured, GreedyLength] {
        let (mut cases, mut opportunities, mut inputs, population) =
            algorithm_pair_inventory([1, 1]);
        let seed = ferrum_scheduler::implementations::continuous::cost_model::structured_v2::
            DeclaredAlgorithmUniverseV1::from_inputs(inputs.iter().flatten()
                .map(|fact| fact.original.as_deref().unwrap()), population.settings.max_axes).unwrap();
        // The original A/B/C opportunities fit the three-source cap. A/B
        // coalescing saves a slot for a separate union; C keeps its complete
        // raw source at either the same policy or GreedyLength's later policy.
        for mut case in cases[..2].to_vec() {
            case.template = 2;
            case.width *= 4;
            case.preset = preset;
            let input = natural_termination_input_with_algorithm_and_eos(
                case.width as u32,
                CostProductOutput::GreedyToken,
                false,
                64,
                "fixture.selection.c",
                [9; 32],
                preset == Configured,
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
            inputs.push(vec![input_facts(&StructuredQueryV2::exact(input)).unwrap()]);
            cases.push(case);
        }
        let run = |capacity, seed| {
            select_with_capacity(
                &cases,
                &opportunities,
                &inputs,
                &[61, 61, 61],
                8,
                None,
                &population,
                capacity,
                usize::MAX,
                None,
                None,
                None,
                NonZeroUsize::new(3),
                seed,
            )
            .unwrap()
        };
        let capacity = SelectionCapacity::legacy(100_000, 10_000_000);
        let raw = run(capacity, None);
        assert_eq!(raw.populations.len(), 3);
        assert!(raw.populations.iter().all(|member| member.scheduled));
        let scheduled: Vec<_> = raw.batches.iter().filter(|batch| batch.scheduled).collect();
        assert_eq!(scheduled.len(), 2);
        assert!(scheduled
            .last()
            .unwrap()
            .representative_case_indices
            .iter()
            .all(|&index| cases[index].template == 2));

        let selected = run(capacity, Some(&seed));
        assert!(
            selected.populations.iter().all(|member| member.scheduled),
            "{preset:?}"
        );
        let selected_sources: Vec<_> = selected
            .batches
            .iter()
            .filter(|batch| batch.scheduled)
            .collect();
        assert_eq!(selected_sources.len(), 3);
        let combination_index = selected_sources
            .iter()
            .position(|batch| batch.algorithm_universe.is_some())
            .expect("the saved source slot must collect a separate A+B scope");
        let combination = selected_sources[combination_index];
        let local = combination.algorithm_universe.as_ref().unwrap();
        assert_eq!(local.algorithm_count(), 2);
        assert!(seed.contains_universe(local));
        assert!(combination
            .representative_case_indices
            .iter()
            .all(|&index| cases[index].template < 2));
        for original in &scheduled {
            let retained = selected_sources
                .iter()
                .find(|batch| {
                    batch.algorithm_universe.is_none()
                        && batch.representative_case_indices == original.representative_case_indices
                })
                .expect("composition must retain each original raw source");
            assert_eq!(retained.planned_cycles, original.planned_cycles);
            assert_eq!(retained.requests, original.requests);
            assert_eq!(
                retained.serial_wave_upper_bound,
                original.serial_wave_upper_bound
            );
            assert_eq!(
                retained.declared_offer_row_bound,
                original.declared_offer_row_bound
            );
        }
        let later_index = selected_sources
            .iter()
            .position(|batch| {
                batch.algorithm_universe.is_none()
                    && batch
                        .representative_case_indices
                        .iter()
                        .all(|&index| cases[index].template == 2)
            })
            .unwrap();
        assert_eq!(later_index < combination_index, preset == Configured);
        assert_eq!(selected.requests, raw.requests + combination.requests);
        assert_eq!(
            selected.serial_wave_upper_bound,
            raw.serial_wave_upper_bound + combination.serial_wave_upper_bound
        );
        assert_eq!(
            selected.declared_offer_row_bound,
            raw.declared_offer_row_bound + combination.declared_offer_row_bound
        );
        assert!(selected.execution_case_indices.len() > raw.execution_case_indices.len());

        // The extra union's work cannot borrow C's originally affordable
        // request allowance, even when its source slot is available.
        let denied = run(
            SelectionCapacity {
                requests: selected.requests - 1,
                execution_actions: selected.serial_wave_upper_bound,
                declared_offer_rows: selected.declared_offer_row_bound,
            },
            Some(&seed),
        );
        assert!(
            denied
                .batches
                .iter()
                .all(|batch| batch.algorithm_universe.is_none()),
            "{preset:?}"
        );
        assert!(
            denied.populations.iter().all(|member| member.scheduled),
            "{preset:?}"
        );
        assert_eq!(
            denied.execution_case_indices, raw.execution_case_indices,
            "{preset:?}"
        );
        assert_eq!(denied.requests, raw.requests, "{preset:?}");
        assert_eq!(
            denied.serial_wave_upper_bound, raw.serial_wave_upper_bound,
            "{preset:?}"
        );
        assert_eq!(
            denied.declared_offer_row_bound, raw.declared_offer_row_bound,
            "{preset:?}"
        );
    }
}
