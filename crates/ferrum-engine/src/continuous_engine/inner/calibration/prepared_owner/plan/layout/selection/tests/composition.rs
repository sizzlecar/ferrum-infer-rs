use super::*;

#[test]
fn checked_decode_journal_keeps_distinct_host_families_within_complete_source_capacity() {
    use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1;
    use std::num::NonZeroU64;
    use std::sync::Arc;
    use SloAutomaticCostProbeSamplingPresetV1::{Configured, GreedyLength};

    for product in [
        CostProductOutput::GreedyToken,
        CostProductOutput::FullLogits,
    ] {
        let population = population::declaration(&Default::default(), fixture::domain()).unwrap();
        let mut cases = vec![Case {
            product: OpportunityProduct::Prefill,
            template: 0,
            width: 1,
            maximum_output: NonZeroUsize::MIN,
            release_generated: 0,
            suffix_tokens: 1,
            preset: Configured,
            prefix: PrefixKind::Ordinary,
            route: CalibrationDecodeRoute::Actual,
            reset: true,
            acquisition: None,
        }];
        let originals = vec![
            // The first chunk obeys the original eight-token wave limit.
            // This planning fixture makes no claim about final-Prefill coverage.
            related::prefill_input_phase(1, 8, 61, 1, Configured, true),
            natural_termination_input_with_algorithm_and_eos(
                1,
                product,
                false,
                64,
                "fixture.selection.configured",
                [7; 32],
                true,
            ),
            fixture::input_with_context(1, 3, 64, product, false, true),
            // A different original template's host categorical identity is
            // independent even when both templates use GreedyLength. With
            // one row this is a homogeneous, checked family, not mixed rows.
            fixture::input_with_context(1, 3, 64, product, true, true),
        ];
        for (template, preset) in [(0, Configured), (0, GreedyLength), (1, GreedyLength)] {
            cases.push(Case {
                product: if product == CostProductOutput::FullLogits {
                    OpportunityProduct::Full
                } else {
                    OpportunityProduct::Greedy
                },
                template,
                width: 1,
                maximum_output: NonZeroUsize::new(21).unwrap(),
                release_generated: 3,
                suffix_tokens: 18,
                preset,
                prefix: PrefixKind::Clean,
                route: if product == CostProductOutput::FullLogits {
                    CalibrationDecodeRoute::FullLogits
                } else {
                    CalibrationDecodeRoute::Actual
                },
                reset: false,
                acquisition: None,
            });
        }
        let opportunities: Vec<_> = originals
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
        let inputs: Vec<_> = originals
            .iter()
            .cloned()
            .map(|input| vec![input_facts(&StructuredQueryV2::exact(input)).unwrap()])
            .collect();
        let inventory = inventory::CheckedCaseInventory {
            opportunities: opportunities.clone(),
            inputs: inputs.clone(),
            original_inputs: inputs
                .iter()
                .flatten()
                .filter_map(|fact| fact.original.clone())
                .collect(),
            algorithm_inputs: originals[1..].iter().cloned().map(Arc::new).collect(),
            algorithm_case_inputs: std::iter::once(Vec::new())
                .chain((0..originals.len() - 1).map(|index| vec![index]))
                .collect(),
            gaps: Vec::new(),
            charge: ProbePreflightCharge::default(),
        };
        // Match inventory::retain_algorithm_input: Prefill remains an exact-owner
        // opportunity and is outside the ordinary-decode algorithm universe.
        assert_eq!(
            originals[0].numerical_family_key(),
            Err(StructuredUnknownV2::UnsupportedScope)
        );
        for input in &originals[1..] {
            input.numerical_family_key().unwrap();
        }
        let unrelated = natural_termination_input_with_algorithm(
            1,
            product,
            false,
            257,
            "fixture.selection.unlinked",
            [9; 32],
        );
        let seed = ferrum_scheduler::implementations::continuous::cost_model::structured_v2::
            DeclaredAlgorithmUniverseV1::from_inputs(
                inventory.algorithm_inputs.iter().map(Arc::as_ref).chain(std::iter::once(&unrelated)),
                population.settings.max_axes,
            ).unwrap();
        let groups = member_groups(&opportunities).unwrap();
        assert_eq!(groups.len(), cases.len());
        let populations: Vec<_> = groups
            .iter()
            .map(|group| SelectedPopulation {
                key: group.key.clone(),
                representative_case_indices: group.guaranteed_case_indices.clone(),
                maximum_anchor_span: 0,
                scheduled: false,
                batch_index: None,
                input_geometry: None,
            })
            .collect();
        let decode_indices: Vec<_> = (1..populations.len()).collect();
        let prefill = batch_plan(
            &[0],
            &populations,
            &cases,
            &opportunities,
            &[61, 61],
            8,
            None,
            &population,
        )
        .unwrap();
        let packed = batch_plan(
            &decode_indices,
            &populations,
            &cases,
            &opportunities,
            &[61, 61],
            8,
            None,
            &population,
        )
        .unwrap();
        let independent = member_groups(&opportunities[1..]).unwrap();
        assert_eq!(independent.len(), decode_indices.len());
        for group in &independent {
            assert_eq!(group.guaranteed_case_indices.len(), 1);
        }
        assert_ne!(
            inputs[2][0].homogeneous_host_policy,
            inputs[3][0].homogeneous_host_policy
        );
        assert!(
            packed.planned_cycles
                * packed
                    .input_opportunities
                    .minimum_original_offers_per_completed_cycle
                >= packed.input_opportunities.required_original_offers
                    + packed.schedule.block_offered
        );
        let capacity = SelectionCapacity {
            requests: prefill.requests + packed.requests,
            execution_actions: prefill.serial_wave_upper_bound + packed.serial_wave_upper_bound,
            declared_offer_rows: prefill.declared_offer_row_bound + packed.declared_offer_row_bound,
        };
        assert!(super::super::composition::can_schedule(&prefill, capacity));
        assert!(super::super::composition::can_schedule(
            &packed,
            capacity.remaining(SelectionCapacity {
                requests: prefill.requests,
                execution_actions: prefill.serial_wave_upper_bound,
                declared_offer_rows: prefill.declared_offer_row_bound,
            })
        ));
        let select = |capacity,
                      selected_priority,
                      population: &StructuredServiceDeclarationV7,
                      seed: Option<&DeclaredAlgorithmUniverseV1>,
                      geometry: Option<&mut StructuredInputGeometryWorkV1>,
                      retained_bytes| {
            select_with_capacity_and_trajectories(
                &cases,
                &opportunities,
                &inputs,
                &[61, 61],
                8,
                None,
                population,
                capacity,
                retained_bytes,
                None,
                selected_priority,
                geometry,
                NonZeroUsize::new(2),
                seed,
                Some(&inventory),
            )
        };
        let selected = select(capacity, None, &population, Some(&seed), None, usize::MAX).unwrap();
        assert!(selected.populations.iter().all(|population| population.scheduled),
            "complete Prefill and distinct original host families fit two journals without sharing F/R/Q members: {:?}", selected.gaps);
        assert_eq!(
            selected
                .batches
                .iter()
                .filter(|batch| batch.scheduled)
                .count(),
            2
        );
        let actual = selected
            .batches
            .iter()
            .find(|batch| batch.scheduled && batch.population_indices.contains(&1))
            .unwrap();
        assert!(actual.algorithm_universe.is_none());
        assert!(actual.scoped_opportunities.is_none());
        for &index in &actual.representative_case_indices {
            assert_eq!(selected.populations[index].key, groups[index].key);
            assert_eq!(
                inputs[index][0]
                    .original
                    .as_ref()
                    .unwrap()
                    .numerical_family_key(),
                originals[index].numerical_family_key(),
                "raw numerical identities must not be projected into a new union"
            );
        }
        assert_eq!(
            actual.representative_case_indices,
            packed.representative_case_indices
        );
        assert_eq!(actual.population_indices.len(), independent.len());
        assert_eq!(selected.requests, capacity.requests);
        assert_eq!(selected.serial_wave_upper_bound, capacity.execution_actions);
        assert_eq!(
            selected.declared_offer_row_bound,
            capacity.declared_offer_rows
        );
        assert_eq!(
            selected
                .execution_case_indices
                .iter()
                .filter(|&&index| index == 0)
                .count(),
            prefill.planned_cycles
        );
        let frozen = source_inputs::freeze_sources(
            &cases,
            &opportunities,
            &[61, 61],
            &selected,
            NonZeroU32::new(8).unwrap(),
            None,
            usize::MAX,
        )
        .unwrap();
        let decode_source = selected
            .batches
            .iter()
            .filter(|batch| batch.scheduled)
            .position(|batch| batch.population_indices.contains(&1))
            .unwrap();
        let declared = population.clone();
        let source_inputs::ColdSourceRebuild::Ready(cold) = frozen[decode_source]
            .cold_plan(&declared, NonZeroU32::new(8).unwrap(), None, usize::MAX)
            .unwrap()
        else {
            panic!("unchanged cold work must retain all independent family horizons");
        };
        assert_eq!(cold.cycles, actual.planned_cycles);
        assert_eq!(cold.requests, actual.requests);
        assert_eq!(cold.execution_actions, actual.serial_wave_upper_bound);
        assert_eq!(
            cold.input_opportunities
                .minimum_input_family_opportunities_per_cycle,
            actual
                .input_opportunities
                .minimum_input_family_opportunities_per_cycle
        );
        assert_eq!(
            cold.input_opportunities.maximum_fresh_member_span,
            actual.input_opportunities.maximum_fresh_member_span
        );

        // Every original constituent must pass a partial priority filter,
        // including a later host family after an earlier merge succeeded.
        for wanted in [input_priority(true, true), input_priority(false, false)] {
            let partial = select(
                capacity,
                Some(wanted),
                &population,
                Some(&seed),
                None,
                usize::MAX,
            )
            .unwrap();
            for (index, member) in partial.populations.iter().enumerate() {
                let fact = &inputs[index][0];
                assert_eq!(
                    member.scheduled,
                    input_priority(fact.branches[4], fact.branches[5]) == wanted
                );
            }
        }
        // The successful plan above consumes these exact independent ledgers.
        // A missing unit cannot be borrowed from another ledger or source.
        for short in [
            SelectionCapacity {
                requests: capacity.requests - 1,
                ..capacity
            },
            SelectionCapacity {
                execution_actions: capacity.execution_actions - 1,
                ..capacity
            },
            SelectionCapacity {
                declared_offer_rows: capacity.declared_offer_rows - 1,
                ..capacity
            },
        ] {
            let denied = select(short, None, &population, Some(&seed), None, usize::MAX).unwrap();
            assert!(!denied.populations.iter().all(|member| member.scheduled));
            assert!(
                denied.populations[0].scheduled && denied.populations[1].scheduled,
                "packing must preserve the original complete Configured sources"
            );
            assert!(denied.requests <= short.requests);
            assert!(denied.serial_wave_upper_bound <= short.execution_actions);
            assert!(denied.declared_offer_row_bound <= short.declared_offer_rows);
        }
        let mut owner_limited = population.clone();
        owner_limited.maximum_owners = independent.len() - 1;
        let denied = select(
            capacity,
            None,
            &owner_limited,
            Some(&seed),
            None,
            usize::MAX,
        )
        .unwrap();
        assert!(!denied.populations.iter().all(|member| member.scheduled));
        assert!(denied.populations[0].scheduled && denied.populations[1].scheduled);
        for batch in denied.batches.iter().filter(|batch| batch.scheduled) {
            let original: Vec<_> = batch
                .representative_case_indices
                .iter()
                .map(|&index| opportunities[index].clone())
                .collect();
            assert!(member_groups(&original).unwrap().len() <= owner_limited.maximum_owners);
        }
        let mut geometry = StructuredInputGeometryWorkV1::new(NonZeroU64::MIN);
        let incomplete = select(
            capacity,
            None,
            &population,
            Some(&seed),
            Some(&mut geometry),
            usize::MAX,
        )
        .unwrap();
        assert!(geometry.exhausted());
        assert!(!incomplete.populations.iter().all(|member| member.scheduled));
        for member in &incomplete.populations[2..] {
            assert!(!member.input_geometry.as_ref().unwrap().complete);
            assert!(!member.scheduled);
            assert!(incomplete
                .gaps
                .iter()
                .any(|gap| gap.population.as_ref() == Some(&member.key)
                    && matches!(
                        gap.reason,
                        SelectionGapReason::InputGeometryUnavailable { .. }
                    )));
        }
        let missing_algorithm = DeclaredAlgorithmUniverseV1::from_inputs(
            std::slice::from_ref(&originals[1]),
            population.settings.max_axes,
        )
        .unwrap();
        for unavailable in [None, Some(&missing_algorithm)] {
            let raw = select(capacity, None, &population, unavailable, None, usize::MAX).unwrap();
            assert!(!raw.populations.iter().all(|member| member.scheduled));
            assert!(raw.populations[0].scheduled && raw.populations[1].scheduled);
        }
        assert!(select(capacity, None, &population, Some(&seed), None, 0).is_err());
    }
}

#[test]
fn independent_family_packing_preserves_equal_scopes_and_declines_mixed_interpretations() {
    use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1;

    // Each original source obtains its scope through the production same-host
    // composition pass. Only then may independent host families share a journal.
    for (scenario, algorithms) in [
        [
            ("fixture.selection.a", [7; 32]),
            ("fixture.selection.b", [8; 32]),
        ],
        [
            ("fixture.selection.c", [9; 32]),
            ("fixture.selection.d", [10; 32]),
        ],
        [
            ("fixture.selection.c", [9; 32]),
            ("fixture.selection.c", [9; 32]),
        ],
    ]
    .into_iter()
    .enumerate()
    {
        let (mut cases, mut opportunities, mut inputs, population) =
            algorithm_pair_inventory([1, 1]);
        let original_cases = cases.clone();
        for mut case in original_cases {
            let (name, implementation) = algorithms[case.template];
            case.template += 2;
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
            inputs.push(vec![input_facts(&StructuredQueryV2::exact(input)).unwrap()]);
            cases.push(case);
        }
        let seed = DeclaredAlgorithmUniverseV1::from_inputs(
            inputs
                .iter()
                .flatten()
                .map(|fact| fact.original.as_deref().unwrap()),
            population.settings.max_axes,
        )
        .unwrap();
        let run = |maximum_sources| {
            select_with_local_composition(
                &cases,
                &opportunities,
                &inputs,
                &[61; 4],
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
                Some(&seed),
            )
            .unwrap()
        };
        let separate = run(2);
        assert!(separate.populations.iter().all(|member| member.scheduled));
        let original: Vec<_> = separate
            .batches
            .iter()
            .filter(|batch| batch.scheduled)
            .collect();
        assert_eq!(
            original.len(),
            2,
            "no coverage gain leaves both original journals alone"
        );
        let equal = original[0].algorithm_universe == original[1].algorithm_universe;
        assert_eq!(
            equal,
            scenario == 0,
            "the original scopes establish the tested boundary"
        );
        assert_eq!(
            original
                .iter()
                .filter(|batch| batch.algorithm_universe.is_some())
                .count(),
            if scenario == 2 { 1 } else { 2 }
        );
        let packed = run(1);
        if equal {
            assert!(packed.populations.iter().all(|member| member.scheduled));
            let combined: Vec<_> = packed
                .batches
                .iter()
                .filter(|batch| batch.scheduled)
                .collect();
            assert_eq!(combined.len(), 1);
            assert_eq!(
                combined[0].algorithm_universe,
                original[0].algorithm_universe
            );
            let scope = combined[0].algorithm_universe.as_ref().unwrap();
            for before in &original {
                for &index in &before.representative_case_indices {
                    assert!(combined[0].representative_case_indices.contains(&index));
                    let fact = inputs[index][0].original.as_ref().unwrap();
                    assert_eq!(
                        fact.numerical_family_key_for_universe(scope),
                        fact.numerical_family_key_for_universe(
                            before.algorithm_universe.as_ref().unwrap()
                        ),
                    );
                }
            }
            assert_eq!(
                member_groups(combined[0].scoped_opportunities.as_ref().unwrap())
                    .unwrap()
                    .len(),
                original
                    .iter()
                    .map(
                        |batch| member_groups(batch.scoped_opportunities.as_ref().unwrap())
                            .unwrap()
                            .len()
                    )
                    .sum::<usize>(),
                "each original host family keeps independent phase membership"
            );
        } else {
            assert!(!packed.populations.iter().all(|member| member.scheduled));
            assert_eq!(packed.batches.len(), original.len());
            for before in original {
                let after = packed
                    .batches
                    .iter()
                    .find(|batch| batch.population_indices == before.population_indices)
                    .unwrap();
                assert_eq!(after.algorithm_universe, before.algorithm_universe);
                assert_eq!(
                    after.representative_case_indices,
                    before.representative_case_indices
                );
                assert_eq!(after.planned_cycles, before.planned_cycles);
                assert_eq!(after.requests, before.requests);
                assert_eq!(
                    after.serial_wave_upper_bound,
                    before.serial_wave_upper_bound
                );
                assert_eq!(
                    after.declared_offer_row_bound,
                    before.declared_offer_row_bound
                );
            }
        }
    }
}

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
        super::super::composition::extra_peak(&opportunities, &seed, raw.populations.len())
            .unwrap()
            > seed.retained_payload_bytes().unwrap()
    );
}

pub(super) fn algorithm_pair_inventory(
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
fn local_combination_repeated_groups_charge_one_retained_scope_per_candidate() {
    use std::num::NonZeroU64;

    let (mut cases, mut opportunities, mut inputs, population) = algorithm_pair_inventory([1, 1]);
    let seed = ferrum_scheduler::implementations::continuous::cost_model::structured_v2::
        DeclaredAlgorithmUniverseV1::from_inputs(inputs.iter().flatten()
            .map(|fact| fact.original.as_deref().unwrap()), population.settings.max_axes).unwrap();
    let requests = 2048;
    let before_groups = member_groups(&opportunities).unwrap();
    let before = super::super::memory::plan(
        &before_groups,
        &opportunities,
        &inputs,
        requests,
        Some(&population.settings),
    )
    .unwrap();
    let originals = (cases.clone(), opportunities.clone(), inputs.clone());
    // More original requests in the same checked families grow case vectors,
    // but cannot create another simultaneously retained candidate universe.
    for _ in 0..population.settings.max_rank {
        cases.extend(originals.0.iter().cloned());
        opportunities.extend(originals.1.iter().cloned());
        inputs.extend(originals.2.iter().cloned());
    }
    let groups = member_groups(&opportunities).unwrap();
    let memory = super::super::memory::plan(
        &groups,
        &opportunities,
        &inputs,
        requests,
        Some(&population.settings),
    )
    .unwrap();
    assert_eq!(memory.guaranteed_groups, before.guaranteed_groups);
    assert!(memory.guaranteed_cases > before.guaranteed_cases);
    assert!(memory.batch_scratch_bytes > before.batch_scratch_bytes);
    // The original inventory-sized bound remains a conservative reference;
    // only its nonexistent per-mention scopes are removed from the tight cap.
    let full_inventory_extra =
        super::super::composition::extra_peak(&opportunities, &seed, memory.key_mentions).unwrap();
    let duplicate_scope_charge = super::super::composition::builder_limit(&seed).unwrap()
        * (memory.key_mentions - memory.guaranteed_groups);
    let full_inventory_peak = memory.required_peak_bytes + full_inventory_extra;
    let candidate_peak = full_inventory_peak - duplicate_scope_charge;
    assert!(candidate_peak < full_inventory_peak);
    assert!(candidate_peak > memory.required_peak_bytes);
    eprintln!(
        "local scope memory: mentions={} groups={} full_inventory_peak={} candidate_peak={}",
        memory.key_mentions, memory.guaranteed_groups, full_inventory_peak, candidate_peak
    );
    assert!(composition_authorized(
        &opportunities,
        &inputs,
        requests,
        &population,
        true,
        candidate_peak,
        &seed,
    )
    .unwrap());
    assert!(!composition_authorized(
        &opportunities,
        &inputs,
        requests,
        &population,
        true,
        candidate_peak - 1,
        &seed,
    )
    .unwrap());

    let run = |maximum, work: &mut StructuredInputGeometryWorkV1| {
        select_with_capacity(
            &cases,
            &opportunities,
            &inputs,
            &[61, 61],
            8,
            None,
            &population,
            SelectionCapacity::legacy(requests, 10_000_000),
            maximum,
            None,
            None,
            Some(work),
            NonZeroUsize::new(1),
            Some(&seed),
        )
    };
    let work_limit = NonZeroU64::new(32_000_000).unwrap();
    let mut rejected_work = StructuredInputGeometryWorkV1::new(work_limit);
    assert!(run(candidate_peak - 1, &mut rejected_work).is_err());
    assert_eq!(rejected_work.visits(), 0);
    let mut expected_work = StructuredInputGeometryWorkV1::new(work_limit);
    let expected = run(full_inventory_peak, &mut expected_work).unwrap();
    let mut actual_work = StructuredInputGeometryWorkV1::new(work_limit);
    let actual = run(candidate_peak, &mut actual_work).unwrap();
    assert_eq!(actual_work.visits(), expected_work.visits());
    assert_eq!(
        serde_json::to_value(&actual).unwrap(),
        serde_json::to_value(&expected).unwrap(),
        "the complete plan, member floors, work ledgers and typed gaps are unchanged"
    );
    let scheduled: Vec<_> = actual
        .batches
        .iter()
        .filter(|batch| batch.scheduled)
        .collect();
    assert_eq!(scheduled.len(), 1);
    assert!(scheduled[0].algorithm_universe.is_some());
    assert!(actual.populations.iter().all(|member| member.scheduled));
    assert!(
        memory.retained_groups_bytes + actual.retained_payload_bytes().unwrap() <= candidate_peak
    );
}

#[test]
fn local_combination_preserves_different_complete_width_endpoints() {
    let (cases, opportunities, inputs, population) = algorithm_pair_inventory([1, 2]);
    let seed = ferrum_scheduler::implementations::continuous::cost_model::structured_v2::
        DeclaredAlgorithmUniverseV1::from_inputs(inputs.iter().flatten()
            .map(|f| f.original.as_deref().unwrap()), population.settings.max_axes).unwrap();
    let run = |seed| {
        select_with_local_composition(
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
            seed,
        )
        .unwrap()
    };
    let raw = run(None);
    let selected = run(Some(&seed));
    let scheduled: Vec<_> = selected
        .batches
        .iter()
        .filter(|batch| batch.scheduled)
        .collect();
    assert_eq!(scheduled.len(), 1);
    assert!(scheduled[0].algorithm_universe.is_some());
    for member in &raw.populations {
        assert!(member
            .representative_case_indices
            .iter()
            .all(|index| scheduled[0].representative_case_indices.contains(index)));
    }
    for facts in inputs.iter().flatten() {
        assert!(facts
            .original
            .as_ref()
            .unwrap()
            .numerical_family_key_for_universe(scheduled[0].algorithm_universe.as_ref().unwrap())
            .is_ok());
    }
    assert!(selected.populations.iter().all(|member| member.scheduled));
    let limited = select_with_local_composition(
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
        NonZeroUsize::new(1),
        Some(&seed),
    )
    .unwrap();
    let scheduled: Vec<_> = limited
        .batches
        .iter()
        .filter(|batch| batch.scheduled)
        .collect();
    assert_eq!(scheduled.len(), 1);
    assert!(
        scheduled[0].algorithm_universe.is_none(),
        "a deferred complete width set must not be pulled into the first source"
    );
    assert!(limited.populations.iter().any(|member| !member.scheduled));
}

#[test]
fn same_width_algorithm_families_share_one_complete_union_source() {
    let (cases, opportunities, inputs, population) = algorithm_pair_inventory([1, 1]);
    let original = serde_json::to_value((&cases, &opportunities, &inputs)).unwrap();
    let seed = ferrum_scheduler::implementations::continuous::cost_model::structured_v2::
        DeclaredAlgorithmUniverseV1::from_inputs(inputs.iter().flatten()
            .map(|fact| fact.original.as_deref().unwrap()), population.settings.max_axes).unwrap();
    let run = |capacity, seed| {
        select_with_capacity(
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
            NonZeroUsize::new(1),
            seed,
        )
        .unwrap()
    };
    let raw = run(SelectionCapacity::legacy(100_000, 10_000_000), None);
    let capacity = SelectionCapacity {
        requests: raw.requests,
        execution_actions: raw.serial_wave_upper_bound,
        declared_offer_rows: raw.declared_offer_row_bound,
    };
    let selected = run(capacity, Some(&seed));
    let scheduled: Vec<_> = selected
        .batches
        .iter()
        .filter(|batch| batch.scheduled)
        .collect();
    assert_eq!(
        scheduled.len(),
        1,
        "one source slot never requires a prior raw qualification"
    );
    let combined = scheduled[0];
    let local = combined.algorithm_universe.as_ref().unwrap();
    assert!(seed.contains_universe(local));
    assert_eq!(local.algorithm_count(), 2);
    assert_eq!(
        combined.representative_case_indices,
        raw.batches[0].representative_case_indices
    );
    let projected: Vec<_> = inputs
        .iter()
        .map(|facts| {
            facts[0]
                .original
                .as_ref()
                .unwrap()
                .as_ref()
                .clone()
                .with_algorithm_universe(local)
                .unwrap()
        })
        .collect();
    assert!(
        (0..projected[0].regression_axes().len()).any(|axis| {
            let value = |index: usize| projected[index].regression_axes()[axis];
            value(0) > 0. && value(0) < value(1) && value(2) == 0. && value(3) == 0.
        }),
        "the other algorithm's zeros must not erase the first algorithm's positive lower endpoint"
    );
    assert!(selected.populations.iter().all(|member| member.scheduled));
    assert_eq!(
        combined.population_indices,
        raw.batches[0].population_indices
    );
    let scoped = combined.scoped_opportunities.as_ref().unwrap();
    assert_eq!(member_groups(scoped).unwrap().len(), 1);
    for (&index, opportunity) in combined.representative_case_indices.iter().zip(scoped) {
        assert_eq!(
            opportunity.minimum_fresh_members,
            opportunities[index].minimum_fresh_members
        );
    }
    assert_eq!(
        combined.schedule.min_members,
        raw.batches[0].schedule.min_members
    );
    for phase in 0..3 {
        assert!(combined.input_opportunities.phase_cycles[phase] > 0);
        assert!(
            combined.input_opportunities.phase_original_offer_bounds[phase]
                >= combined.input_opportunities.maximum_fresh_member_span[phase]
                    .max(combined.schedule.phase_min_offered[phase])
        );
    }
    assert!(combined.planned_cycles < raw.batches[0].planned_cycles);
    assert!(selected.requests < raw.requests);
    assert!(selected.serial_wave_upper_bound < raw.serial_wave_upper_bound);
    assert!(selected.declared_offer_row_bound < raw.declared_offer_row_bound);
    assert_eq!(
        selected.execution_case_indices,
        combined
            .representative_case_indices
            .repeat(combined.planned_cycles)
    );
    assert_eq!(
        selected.requests,
        selected
            .execution_case_indices
            .iter()
            .map(|&i| cases[i].width)
            .sum::<usize>()
    );
    assert_eq!(
        selected.serial_wave_upper_bound,
        selected
            .execution_case_indices
            .iter()
            .map(|&i| cases[i].waves(61, 8).unwrap().1)
            .sum::<usize>()
    );
    assert_eq!(
        original,
        serde_json::to_value((&cases, &opportunities, &inputs)).unwrap()
    );

    // Native preparation may fall back to cold work. Its fixed membership
    // must retain the same union phase floor, without restoring raw groups.
    let frozen = source_inputs::freeze_sources(
        &cases,
        &opportunities,
        &[61, 61],
        &selected,
        NonZeroU32::new(8).unwrap(),
        None,
        usize::MAX,
    )
    .unwrap();
    let source_inputs::ColdSourceRebuild::Ready(cold) = frozen[0]
        .cold_plan(&population, NonZeroU32::new(8).unwrap(), None, usize::MAX)
        .unwrap()
    else {
        panic!("unchanged cold work must fit the original union horizon");
    };
    assert_eq!(cold.cycles, combined.planned_cycles);
    assert_eq!(
        cold.input_opportunities
            .minimum_input_family_opportunities_per_cycle,
        combined
            .input_opportunities
            .minimum_input_family_opportunities_per_cycle
    );

    let complete = SelectionCapacity {
        requests: selected.requests,
        execution_actions: selected.serial_wave_upper_bound,
        declared_offer_rows: selected.declared_offer_row_bound,
    };
    let exact = run(complete, Some(&seed));
    assert_eq!(
        exact.execution_case_indices,
        selected.execution_case_indices
    );
    assert!(exact.populations.iter().all(|member| member.scheduled));
    assert_eq!(exact.requests, complete.requests);
    assert_eq!(exact.serial_wave_upper_bound, complete.execution_actions);
    assert_eq!(exact.declared_offer_row_bound, complete.declared_offer_rows);
    for capacity in [
        SelectionCapacity {
            requests: complete.requests - 1,
            ..complete
        },
        SelectionCapacity {
            execution_actions: complete.execution_actions - 1,
            ..complete
        },
        SelectionCapacity {
            declared_offer_rows: complete.declared_offer_rows - 1,
            ..complete
        },
    ] {
        let denied = run(capacity, Some(&seed));
        assert!(denied.requests <= capacity.requests);
        assert!(denied.serial_wave_upper_bound <= capacity.execution_actions);
        assert!(denied.declared_offer_row_bound <= capacity.declared_offer_rows);
        assert!(denied.populations.iter().any(|member| !member.scheduled));
    }
    let incomplete = ferrum_scheduler::implementations::continuous::cost_model::structured_v2::
        DeclaredAlgorithmUniverseV1::from_inputs(inputs[..2].iter().flatten()
            .map(|fact| fact.original.as_deref().unwrap()), population.settings.max_axes).unwrap();
    let unchanged = run(capacity, Some(&incomplete));
    assert!(unchanged
        .batches
        .iter()
        .all(|batch| batch.algorithm_universe.is_none()));
    assert_eq!(unchanged.execution_case_indices, raw.execution_case_indices);
}

#[test]
fn local_scope_rejects_different_real_installed_host_policies() {
    let (cases, mut opportunities, mut inputs, population) = algorithm_pair_inventory([1, 1]);
    for index in 2..cases.len() {
        let input = natural_termination_input_with_algorithm_and_eos(
            cases[index].width as u32,
            CostProductOutput::GreedyToken,
            false,
            64,
            "fixture.selection.b",
            [8; 32],
            false,
        );
        opportunities[index].population = classify_alternatives(
            std::slice::from_ref(&input),
            population.population_policy(),
            true,
        )
        .unwrap();
        inputs[index] = vec![input_facts(&StructuredQueryV2::exact(input)).unwrap()];
    }
    assert_ne!(
        inputs[0][0].homogeneous_host_policy,
        inputs[2][0].homogeneous_host_policy
    );
    let seed = ferrum_scheduler::implementations::continuous::cost_model::structured_v2::
        DeclaredAlgorithmUniverseV1::from_inputs(inputs.iter().flatten()
            .map(|fact| fact.original.as_deref().unwrap()), population.settings.max_axes).unwrap();
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
    assert!(selected.populations.iter().all(|member| member.scheduled));
    assert_eq!(
        selected
            .batches
            .iter()
            .filter(|batch| batch.scheduled)
            .count(),
        2
    );
    assert!(selected
        .batches
        .iter()
        .all(|batch| batch.algorithm_universe.is_none()));
    // Exercise the scope gate directly as well: policy-priority ordering must
    // not be the only reason these independently valid sources stay separate.
    let members: Vec<_> = (0..selected.populations.len()).collect();
    let candidate = batch_plan(
        &members,
        &selected.populations,
        &cases,
        &opportunities,
        &[61, 61],
        8,
        None,
        &population,
    )
    .unwrap();
    assert!(super::super::composition::scoped_candidate(
        &candidate,
        &selected.populations,
        &cases,
        &opportunities,
        &inputs,
        None,
        &[61, 61],
        8,
        None,
        &population,
        &seed
    )
    .unwrap()
    .is_none());
}

#[test]
fn local_scope_uses_only_original_cases_checked_trajectory_algorithms() {
    use std::sync::Arc;
    let (mut cases, mut opportunities, mut inputs, population) = algorithm_pair_inventory([1, 1]);
    cases.truncate(2);
    opportunities.truncate(2);
    inputs.truncate(2);
    let later = natural_termination_input_with_algorithm(
        1,
        CostProductOutput::GreedyToken,
        false,
        65,
        "fixture.selection.later",
        [10; 32],
    );
    let unrelated = natural_termination_input_with_algorithm(
        1,
        CostProductOutput::GreedyToken,
        false,
        257,
        "fixture.selection.unselected",
        [11; 32],
    );
    let inventory = inventory::CheckedCaseInventory {
        opportunities: opportunities.clone(),
        inputs: inputs.clone(),
        original_inputs: inputs
            .iter()
            .flatten()
            .filter_map(|facts| facts.original.clone())
            .collect(),
        algorithm_inputs: vec![Arc::new(later.clone()), Arc::new(unrelated.clone())],
        algorithm_case_inputs: vec![vec![0], vec![0]],
        gaps: Vec::new(),
        charge: ProbePreflightCharge::default(),
    };
    let seed = ferrum_scheduler::implementations::continuous::cost_model::structured_v2::
        DeclaredAlgorithmUniverseV1::from_inputs(inputs.iter().flatten()
            .map(|fact| fact.original.as_deref().unwrap())
            .chain(inventory.algorithm_inputs.iter().map(Arc::as_ref)), population.settings.max_axes).unwrap();
    let selected = select_with_capacity_and_trajectories(
        &cases,
        &opportunities,
        &inputs,
        &[61],
        8,
        None,
        &population,
        SelectionCapacity::legacy(100_000, 10_000_000),
        usize::MAX,
        None,
        None,
        None,
        NonZeroUsize::new(1),
        Some(&seed),
        Some(&inventory),
    )
    .unwrap();
    let batch = selected
        .batches
        .iter()
        .find(|batch| batch.scheduled)
        .unwrap();
    let local = batch.algorithm_universe.as_ref().unwrap();
    assert!(later.numerical_family_key_for_universe(local).is_ok());
    assert!(
        unrelated.numerical_family_key_for_universe(local).is_err(),
        "the discovery seed cannot grant an unselected case's algorithm scope"
    );
    assert!(seed.contains_universe(local));
    assert!(local.algorithm_count() < seed.algorithm_count());
    assert_eq!(
        member_groups(batch.scoped_opportunities.as_ref().unwrap())
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn local_scope_keeps_linked_other_product_trajectory_without_merging_members() {
    use std::sync::Arc;

    let (cases, opportunities, inputs, population) = algorithm_pair_inventory([1, 1]);
    // These use the existing typed canonical-command fixture. C is a lawful
    // FullLogits host branch linked to an original case, not another formal
    // Greedy member. D exists only in the discovery seed.
    let later = natural_termination_input_with_algorithm(
        1,
        CostProductOutput::FullLogits,
        false,
        65,
        "fixture.selection.linked-full",
        [10; 32],
    );
    let unrelated = natural_termination_input_with_algorithm(
        1,
        CostProductOutput::GreedyToken,
        false,
        257,
        "fixture.selection.unselected",
        [11; 32],
    );
    let mut links = vec![Vec::new(); cases.len()];
    links[0].push(0);
    let inventory = inventory::CheckedCaseInventory {
        opportunities: opportunities.clone(),
        inputs: inputs.clone(),
        original_inputs: inputs
            .iter()
            .flatten()
            .filter_map(|facts| facts.original.clone())
            .collect(),
        algorithm_inputs: vec![Arc::new(later.clone()), Arc::new(unrelated.clone())],
        algorithm_case_inputs: links,
        gaps: Vec::new(),
        charge: ProbePreflightCharge::default(),
    };
    let seed = ferrum_scheduler::implementations::continuous::cost_model::structured_v2::
        DeclaredAlgorithmUniverseV1::from_inputs(inputs.iter().flatten()
            .map(|fact| fact.original.as_deref().unwrap())
            .chain(inventory.algorithm_inputs.iter().map(Arc::as_ref)), population.settings.max_axes).unwrap();
    let formal = inputs[0][0].original.as_deref().unwrap();
    assert_eq!(seed.contains_checked_algorithms(&later), Ok(true));
    assert_ne!(
        later.numerical_family_key_for_universe(&seed).unwrap(),
        formal.numerical_family_key_for_universe(&seed).unwrap(),
        "algorithm declaration must not merge distinct product/host families"
    );
    let run = |seed| {
        select_with_capacity_and_trajectories(
            &cases,
            &opportunities,
            &inputs,
            &[61, 61],
            8,
            None,
            &population,
            SelectionCapacity::legacy(100_000, 10_000_000),
            usize::MAX,
            None,
            None,
            None,
            NonZeroUsize::new(1),
            seed,
            Some(&inventory),
        )
        .unwrap()
    };
    let raw = run(None);
    let selected = run(Some(&seed));
    let batch = selected
        .batches
        .iter()
        .find(|batch| batch.scheduled)
        .unwrap();
    let raw = raw.batches.iter().find(|batch| batch.scheduled).unwrap();
    assert_eq!(
        batch.representative_case_indices,
        raw.representative_case_indices
    );
    assert!(batch.representative_case_indices.contains(&0));
    assert_eq!(batch.population_indices, raw.population_indices);
    assert_eq!(batch.schedule.min_members, raw.schedule.min_members);
    let scoped = batch.scoped_opportunities.as_ref().unwrap();
    assert_eq!(member_groups(scoped).unwrap().len(), 1);
    for (&index, opportunity) in batch.representative_case_indices.iter().zip(scoped) {
        assert_eq!(
            opportunity.minimum_fresh_members,
            opportunities[index].minimum_fresh_members
        );
    }
    let local = batch.algorithm_universe.as_ref().unwrap();
    assert!(seed.contains_universe(local));
    assert_eq!(local.contains_checked_algorithms(&unrelated), Ok(false));
    assert_eq!(
        local.contains_checked_algorithms(&later),
        Ok(true),
        "a linked physical trajectory must remain projectable before formal-family membership"
    );
    let mut contract = population.nonnegative_envelope.clone().unwrap();
    contract.algorithm_universe = Some(local.clone());
    let projected = contract.project_input(later).unwrap();
    assert_ne!(
        projected.numerical_family_key().unwrap(),
        formal.numerical_family_key_for_universe(local).unwrap(),
        "including C's algorithms grants no formal family membership or qualification"
    );
}

#[test]
fn local_scope_missing_linked_trajectory_seed_preserves_raw_budget_and_members() {
    use std::sync::Arc;

    let (cases, opportunities, inputs, population) = algorithm_pair_inventory([1, 1]);
    let later = natural_termination_input_with_algorithm(
        1,
        CostProductOutput::FullLogits,
        false,
        65,
        "fixture.selection.linked-full",
        [10; 32],
    );
    let mut links = vec![Vec::new(); cases.len()];
    links[0].push(0);
    let inventory = inventory::CheckedCaseInventory {
        opportunities: opportunities.clone(),
        inputs: inputs.clone(),
        original_inputs: inputs
            .iter()
            .flatten()
            .filter_map(|facts| facts.original.clone())
            .collect(),
        algorithm_inputs: vec![Arc::new(later.clone())],
        algorithm_case_inputs: links,
        gaps: Vec::new(),
        charge: ProbePreflightCharge::default(),
    };
    let seed = ferrum_scheduler::implementations::continuous::cost_model::structured_v2::
        DeclaredAlgorithmUniverseV1::from_inputs(inputs.iter().flatten()
            .map(|fact| fact.original.as_deref().unwrap()), population.settings.max_axes).unwrap();
    assert_eq!(seed.contains_checked_algorithms(&later), Ok(false));
    let run = |capacity, seed| {
        select_with_capacity_and_trajectories(
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
            NonZeroUsize::new(1),
            seed,
            Some(&inventory),
        )
        .unwrap()
    };
    let generous = run(SelectionCapacity::legacy(100_000, 10_000_000), None);
    let capacity = SelectionCapacity {
        requests: generous.requests,
        execution_actions: generous.serial_wave_upper_bound,
        declared_offer_rows: generous.declared_offer_row_bound,
    };
    let raw = run(capacity, None);
    let selected = run(capacity, Some(&seed));
    assert!(!selected.execution_case_indices.is_empty());
    assert_eq!(selected.execution_case_indices, raw.execution_case_indices);
    assert_eq!(selected.requests, raw.requests);
    assert_eq!(
        selected.serial_wave_upper_bound,
        raw.serial_wave_upper_bound
    );
    assert_eq!(
        selected.declared_offer_row_bound,
        raw.declared_offer_row_bound
    );
    assert_eq!(selected.batches.len(), raw.batches.len());
    for (selected, raw) in selected.batches.iter().zip(&raw.batches) {
        assert!(selected.algorithm_universe.is_none());
        assert!(selected.scoped_opportunities.is_none());
        assert_eq!(selected.scheduled, raw.scheduled);
        assert_eq!(selected.population_indices, raw.population_indices);
        assert_eq!(
            selected.representative_case_indices,
            raw.representative_case_indices
        );
        assert_eq!(selected.planned_cycles, raw.planned_cycles);
        assert_eq!(selected.schedule.min_members, raw.schedule.min_members);
    }
}

#[test]
fn local_scope_keeps_unknown_and_unreachable_member_floors() {
    let (_, opportunities, inputs, population) = algorithm_pair_inventory([1, 1]);
    let universe = ferrum_scheduler::implementations::continuous::cost_model::structured_v2::
        DeclaredAlgorithmUniverseV1::from_inputs(inputs.iter().flatten()
            .map(|fact| fact.original.as_deref().unwrap()), population.settings.max_axes).unwrap();
    for original in [
        CaseOpportunity {
            population: opportunities[0].population.clone(),
            minimum_fresh_members: 0,
        },
        CaseOpportunity {
            population: CasePopulation::Unknown {
                known_alternatives: vec![inputs[0][0].key(population.population_policy())],
            },
            minimum_fresh_members: 0,
        },
        CaseOpportunity {
            population: CasePopulation::Alternatives(vec![
                inputs[0][0].key(population.population_policy())
            ]),
            minimum_fresh_members: 0,
        },
    ] {
        let projected =
            super::super::composition::project_opportunity(&original, &inputs[0], &universe)
                .unwrap();
        assert_eq!(projected.minimum_fresh_members, 0);
        assert!(member_groups(std::slice::from_ref(&projected))
            .unwrap()
            .iter()
            .all(|group| group.guaranteed_case_indices.is_empty()));
        assert_eq!(
            std::mem::discriminant(&projected.population),
            std::mem::discriminant(&original.population)
        );
    }
}

#[test]
fn same_width_union_preserves_later_original_source_at_same_or_lower_policy() {
    use SloAutomaticCostProbeSamplingPresetV1::{Configured, GreedyLength};
    for preset in [Configured, GreedyLength] {
        let (mut cases, mut opportunities, mut inputs, population) =
            algorithm_pair_inventory([1, 1]);
        let seed = ferrum_scheduler::implementations::continuous::cost_model::structured_v2::
            DeclaredAlgorithmUniverseV1::from_inputs(inputs.iter().flatten()
                .map(|fact| fact.original.as_deref().unwrap()), population.settings.max_axes).unwrap();
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
        let run = |seed| {
            select_with_capacity(
                &cases,
                &opportunities,
                &inputs,
                &[61, 61, 61],
                8,
                None,
                &population,
                SelectionCapacity::legacy(100_000, 10_000_000),
                usize::MAX,
                None,
                None,
                None,
                NonZeroUsize::new(2),
                seed,
            )
            .unwrap()
        };
        let raw = run(None);
        let selected = run(Some(&seed));
        assert!(selected.populations.iter().all(|member| member.scheduled));
        assert_eq!(
            selected
                .batches
                .iter()
                .filter(|batch| batch.scheduled)
                .count(),
            2
        );
        let union = selected
            .batches
            .iter()
            .find(|batch| batch.algorithm_universe.is_some())
            .unwrap();
        assert!(union
            .representative_case_indices
            .iter()
            .all(|&i| cases[i].template < 2));
        let later = |selection: &CheckedSelection| {
            selection
                .batches
                .iter()
                .find(|batch| {
                    batch.scheduled
                        && batch
                            .representative_case_indices
                            .iter()
                            .all(|&i| cases[i].template == 2)
                })
                .unwrap()
                .clone()
        };
        let original = later(&raw);
        let retained = later(&selected);
        assert!(retained.algorithm_universe.is_none());
        assert_eq!(
            retained.representative_case_indices,
            original.representative_case_indices
        );
        assert_eq!(retained.planned_cycles, original.planned_cycles);
        assert_eq!(retained.requests, original.requests);
        assert_eq!(
            retained.serial_wave_upper_bound,
            original.serial_wave_upper_bound
        );
        assert!(selected.requests < raw.requests);
    }
}
