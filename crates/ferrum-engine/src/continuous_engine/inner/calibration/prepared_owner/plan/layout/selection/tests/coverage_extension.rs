//! Explicit extension of already assessed raw populations. This exercises the
//! existing composition contract, not a new default selector or fitted model.
use super::*;
use ferrum_interfaces::vnext::CheckpointTokenSpanConstraint;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    input_geometry_pivot_scratch_bytes_v1, input_geometry_pivots_v1, DeclaredAlgorithmUniverseV1,
    StructuredSettingsV2,
};
use std::num::NonZeroU64;

fn fixture(
    product: CostProductOutput,
) -> (
    Vec<Case>,
    Vec<CaseOpportunity>,
    Vec<Vec<CheckedInputFacts>>,
    StructuredServiceDeclarationV7,
) {
    let (original, _, _, population) = composition::algorithm_pair_inventory([1, 1]);
    let mut cases = Vec::new();
    let mut opportunities = Vec::new();
    let mut inputs = Vec::new();
    for (template, name, implementation, scale) in [
        (0, "fixture.extension.a", [71; 32], 1),
        (1, "fixture.extension.b", [72; 32], 1),
        (2, "fixture.extension.c", [73; 32], 2),
        (3, "fixture.extension.d", [74; 32], 2),
    ] {
        for source in &original[..2] {
            let mut case = source.clone();
            case.template = template;
            case.width *= scale;
            case.product = if product == CostProductOutput::FullLogits {
                OpportunityProduct::Full
            } else {
                OpportunityProduct::Greedy
            };
            case.route = if product == CostProductOutput::FullLogits {
                CalibrationDecodeRoute::FullLogits
            } else {
                CalibrationDecodeRoute::Actual
            };
            case.maximum_output = NonZeroUsize::new(4).unwrap();
            case.release_generated = 3;
            case.suffix_tokens = 1;
            case.prefix = PrefixKind::Clean;
            case.acquisition = work::declared_plan(
                &case,
                work::PrefixBlueprint {
                    prompt_tokens: 61,
                    boundary: 60,
                    // These are minimum/alignment, not lower/upper bounds.
                    // The native fixture's two-token constraint admits the
                    // final four- or two-token segment of this checkpoint.
                    span: CheckpointTokenSpanConstraint::new(
                        NonZeroU64::new(2).unwrap(),
                        NonZeroU64::new(2).unwrap(),
                    )
                    .unwrap(),
                    input_tokens_sha256: implementation,
                },
                NonZeroU32::new(8).unwrap(),
                None,
            )
            .unwrap();
            assert!(case.acquisition.is_some());
            // Freeze and every F/R/Q occurrence use the same prompt, chunk,
            // release boundary and acquisition; retain native restore charges.
            let native_work = work::case_work(&case, 61, 8, None).unwrap();
            assert_eq!(
                native_work.execution_actions,
                native_work.serial_declared_offer_rows + case.width
            );
            let input = natural_termination_input_with_algorithm_and_eos(
                case.width as u32,
                product,
                true,
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
    }
    (cases, opportunities, inputs, population)
}

#[test]
fn coverage_extension_retains_certified_raw_representatives_with_one_finite_source() {
    for product in [
        CostProductOutput::GreedyToken,
        CostProductOutput::FullLogits,
    ] {
        let (cases, opportunities, inputs, population) = fixture(product);
        let seed = DeclaredAlgorithmUniverseV1::from_inputs(
            inputs
                .iter()
                .flatten()
                .map(|f| f.original.as_deref().unwrap()),
            population.settings.max_axes,
        )
        .unwrap();
        let capacity = SelectionCapacity {
            requests: 2048,
            execution_actions: 16384,
            declared_offer_rows: 16384,
        };
        let sources = Some(NonZeroUsize::MIN);
        let groups = member_groups(&opportunities).unwrap();
        let memory = super::super::memory::plan(
            &groups,
            &opportunities,
            &inputs,
            capacity.requests,
            Some(&population.settings),
        )
        .unwrap();
        // Preserve the existing allowance for all original output/candidates,
        // plus the two replacement plans and the complete local U builder.
        let periodic_limit = memory.required_peak_bytes
            + super::super::composition::extra_peak(
                &opportunities,
                &seed,
                memory.guaranteed_groups,
            )
            .unwrap();
        let retained_limit = periodic_limit
            + super::super::memory::finite_extra_peak(&groups, capacity.requests).unwrap();
        let mut geometry = StructuredInputGeometryWorkV1::new(NonZeroU64::new(32_000_000).unwrap());
        let baseline = select_with_capacity(
            &cases,
            &opportunities,
            &inputs,
            &[61; 4],
            8,
            None,
            &population,
            capacity,
            periodic_limit,
            None,
            None,
            Some(&mut geometry),
            sources,
            Some(&seed),
        )
        .unwrap();
        let admitted: Vec<_> = baseline.batches.iter().filter(|b| b.scheduled).collect();
        assert_eq!(admitted.len(), sources.unwrap().get());
        assert!(baseline.batches.iter().any(|b| !b.scheduled));
        assert!(baseline
            .populations
            .iter()
            .all(|p| p.input_geometry.as_ref().unwrap().complete));
        assert!(baseline
            .batches
            .iter()
            .all(|b| b.algorithm_universe.is_some()));
        let old_scope = admitted[0].algorithm_universe.as_ref().unwrap();
        assert!(baseline
            .batches
            .iter()
            .any(|b| { !b.scheduled && b.algorithm_universe.as_ref().unwrap() != old_scope }));
        let original_geometry = serde_json::to_value(&baseline.populations).unwrap();
        let spent = geometry.visits();

        // All original populations, including the genuinely unselected scope,
        // retain their own assessed representatives. No new matrix is made.
        let members: Vec<_> = (0..baseline.populations.len()).collect();
        let raw = batch_plan(
            &members,
            &baseline.populations,
            &cases,
            &opportunities,
            &[61; 4],
            8,
            None,
            &population,
        )
        .unwrap();
        let extension = super::super::composition::scoped_candidate(
            &raw,
            &baseline.populations,
            &cases,
            &opportunities,
            &inputs,
            None,
            &[61; 4],
            8,
            None,
            &population,
            &seed,
            InputAllocationPolicy::FinitePreferred {
                maximum_requests: capacity.requests,
            },
        )
        .unwrap()
        .expect("complete raw families must declare their linked original algorithms");
        let scope = extension.algorithm_universe.as_ref().unwrap();
        assert!(scope.contains_universe(old_scope));
        assert_ne!(scope, old_scope);
        assert_eq!(scope, &seed);
        assert!(
            extension.representative_case_indices.iter().any(|&i| {
                let input = inputs[i][0].original.as_ref().unwrap();
                !old_scope.contains_checked_algorithms(input).unwrap()
                    && scope.contains_checked_algorithms(input).unwrap()
            }),
            "an actual new algorithm must enter the proposed source"
        );
        assert!(extension.finite_plan().is_some());
        assert_eq!(extension.population_indices, members);
        assert_eq!(
            extension.representative_case_indices,
            raw.representative_case_indices
        );
        assert_eq!(extension.schedule.min_members, raw.schedule.min_members);
        related::assert_complete_input_plan(&extension);
        for member in &baseline.populations {
            assert!(member
                .representative_case_indices
                .iter()
                .all(|i| { extension.representative_case_indices.contains(i) }));
        }
        for (&i, scoped) in extension
            .representative_case_indices
            .iter()
            .zip(extension.scoped_opportunities.as_ref().unwrap())
        {
            assert_eq!(
                scoped.minimum_fresh_members,
                opportunities[i].minimum_fresh_members
            );
            let projected = inputs[i][0]
                .original
                .as_ref()
                .unwrap()
                .as_ref()
                .clone()
                .with_algorithm_universe(scope)
                .unwrap();
            let facts =
                facts_from_input(&projected, inputs[i][0].original.clone().unwrap()).unwrap();
            assert_eq!(facts.branches, inputs[i][0].branches);
            assert_eq!(
                facts.homogeneous_host_policy,
                inputs[i][0].homogeneous_host_policy
            );
            assert_eq!(
                scoped.population,
                CasePopulation::Unique(facts.key(population.population_policy()))
            );
        }
        // Existing U cannot be relabelled: rebuilding above used the original
        // raw recipe. It does not reuse the old fitted model or its coefficients.
        let i = admitted[0].representative_case_indices[0];
        let old_input = inputs[i][0]
            .original
            .as_ref()
            .unwrap()
            .as_ref()
            .clone()
            .with_algorithm_universe(old_scope)
            .unwrap();
        assert_eq!(
            old_input.clone().with_algorithm_universe(scope),
            Err(StructuredUnknownV2::WrongDomain)
        );
        assert_eq!(
            old_input.algorithm_universe_signature(),
            Some(old_scope.signature())
        );

        assert!(super::super::composition::can_schedule(
            &extension, capacity
        ));
        let exact = SelectionCapacity {
            requests: extension.requests,
            execution_actions: extension.serial_wave_upper_bound,
            declared_offer_rows: extension.declared_offer_row_bound,
        };
        for available in [
            exact,
            SelectionCapacity {
                requests: exact.requests - 1,
                ..exact
            },
            SelectionCapacity {
                execution_actions: exact.execution_actions - 1,
                ..exact
            },
            SelectionCapacity {
                declared_offer_rows: exact.declared_offer_rows - 1,
                ..exact
            },
        ] {
            let mut used = SelectionCapacity::default();
            let mut count = 0;
            assert_eq!(
                grouping::reserve(&extension, 0, available, None, sources, &mut used, &mut count)
                    .unwrap(),
                available == exact
            );
            assert_eq!(count, usize::from(available == exact));
            assert_eq!(
                used,
                if available == exact {
                    exact
                } else {
                    SelectionCapacity::default()
                }
            );
        }
        let mut proposed = CheckedSelection::default();
        // Retain the original raw geometry evidence; the union does not gain
        // a fabricated common-matrix rank or a qualification certificate.
        proposed.populations = baseline.populations.clone();
        append_batch(&mut proposed, extension, capacity, None).unwrap();
        assert!(proposed.populations.iter().all(|p| p.scheduled));
        finite_policy::assert_work(&proposed, &cases, &[61; 4], 8);
        let raw_storage = CheckedSelection {
            batches: vec![raw],
            ..Default::default()
        };
        assert!(
            baseline.retained_payload_bytes().unwrap()
                + raw_storage.retained_payload_bytes().unwrap()
                + proposed.retained_payload_bytes().unwrap()
                <= retained_limit
        );
        assert_eq!(
            serde_json::to_value(&baseline.populations).unwrap(),
            original_geometry
        );
        assert_eq!(geometry.visits(), spent);
        assert!(!geometry.exhausted());

        // With the finite working set authorized, the normal selector must
        // reach this extension itself after preserving its original packing.
        // This independent control run uses one original geometry ledger;
        // extension must not spend a second matrix scan or retain extra plans.
        let mut actual_geometry =
            StructuredInputGeometryWorkV1::new(NonZeroU64::new(32_000_000).unwrap());
        let actual = select_with_capacity(
            &cases,
            &opportunities,
            &inputs,
            &[61; 4],
            8,
            None,
            &population,
            capacity,
            retained_limit,
            None,
            None,
            Some(&mut actual_geometry),
            sources,
            Some(&seed),
        )
        .unwrap();
        assert_eq!(actual.batches.len(), 1);
        assert!(actual.batches[0].scheduled);
        assert!(actual.batches[0].finite_plan().is_some());
        assert_eq!(actual.batches[0].algorithm_universe.as_ref(), Some(&seed));
        assert_eq!(
            actual.batches[0].representative_case_indices,
            proposed.batches[0].representative_case_indices
        );
        assert!(actual.populations.iter().all(|p| p.scheduled));
        for previous in &baseline.populations {
            let new = actual
                .populations
                .iter()
                .find(|p| p.key == previous.key)
                .unwrap();
            assert_eq!(
                new.representative_case_indices,
                previous.representative_case_indices
            );
            assert_eq!(
                serde_json::to_value(&new.input_geometry).unwrap(),
                serde_json::to_value(&previous.input_geometry).unwrap()
            );
        }
        finite_policy::assert_work(&actual, &cases, &[61; 4], 8);
        assert_eq!(actual_geometry.visits(), spent);
        assert!(!actual_geometry.exhausted());
        assert!(actual.retained_payload_bytes().unwrap() <= retained_limit);
    }
}

#[test]
fn coverage_extension_near_dependent_raw_geometry_is_not_exact_union_span_authority() {
    // Reuse the original cold-kernel near-dependent integer example. A
    // successful tolerance-based rank is deliberately not an exact span proof.
    let magnitude = 10_000_000_000_000u64;
    let values = [
        [magnitude as f64, magnitude as f64],
        [magnitude as f64, (magnitude + 1) as f64],
        [magnitude as f64, magnitude as f64],
    ];
    let rows: Vec<_> = values.iter().map(|v| v.as_slice()).collect();
    let settings = StructuredSettingsV2::default();
    let scratch = input_geometry_pivot_scratch_bytes_v1(rows.len(), 2, settings.max_rank).unwrap();
    let mut work = StructuredInputGeometryWorkV1::new(NonZeroU64::new(32_000_000).unwrap());
    let geometry = input_geometry_pivots_v1(&rows, &[0], &settings, &mut work, scratch).unwrap();
    assert_eq!(geometry.rank, 1);
    assert_eq!(geometry.pivot_indices, [0]);
    let m = u128::from(magnitude);
    assert_ne!(
        m * (m + 1) - m * m,
        0,
        "the original rows are exactly independent"
    );
    assert!(!work.exhausted());
    // This test grants no common-scope geometry or numerical model. Fresh
    // fit, residual, qualification and ordinary adoption remain separate gates.
}

#[test]
fn joint_extension_preserves_expanded_product_and_adds_only_uncovered_class() {
    joint_extension_fixture(false);
}

#[test]
fn joint_extension_freezes_each_class_frontier_before_any_admission() {
    joint_extension_fixture(true);
}

fn joint_extension_fixture(multiple_classes: bool) {
    let (mut cases, mut opportunities, mut inputs, population) =
        fixture(CostProductOutput::GreedyToken);
    let (mut full_cases, full_opportunities, full_inputs, _) =
        fixture(CostProductOutput::FullLogits);
    // Both products are actual declared outputs of the same execution route.
    // The original native checkpoint key does not include the output route.
    for case in &mut full_cases {
        case.route = CalibrationDecodeRoute::Actual;
    }
    cases.extend(full_cases);
    opportunities.extend(full_opportunities);
    inputs.extend(full_inputs);
    if multiple_classes {
        // Retain a genuinely expanded Full baseline, then give the two unmet
        // classes wider real rows. The combined obligation costs more than
        // the baseline, so an exact work bound can reject it atomically while
        // both original sources still fit.
        for (index, case) in cases.iter_mut().enumerate() {
            if case.template < 2 || case.product == OpportunityProduct::Full && case.template == 2 {
                continue;
            }
            case.width = 8;
            let (name, implementation) = match case.template {
                2 => ("fixture.extension.c", [73; 32]),
                3 => ("fixture.extension.d", [74; 32]),
                _ => unreachable!(),
            };
            case.acquisition = work::declared_plan(
                case,
                work::PrefixBlueprint {
                    prompt_tokens: 61,
                    boundary: 60,
                    span: CheckpointTokenSpanConstraint::new(
                        NonZeroU64::new(2).unwrap(),
                        NonZeroU64::new(2).unwrap(),
                    )
                    .unwrap(),
                    input_tokens_sha256: implementation,
                },
                NonZeroU32::new(8).unwrap(),
                None,
            )
            .unwrap();
            // At width eight the original two-token native span cannot fit
            // the one-token row chunk. Keep that cold fallback and its full
            // original work charge; do not retain an incompatible lease.
            let input = natural_termination_input_with_algorithm_and_eos(
                case.width as u32,
                if case.product == OpportunityProduct::Full {
                    CostProductOutput::FullLogits
                } else {
                    CostProductOutput::GreedyToken
                },
                true,
                64,
                name,
                implementation,
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
        // These real typed inputs precede the unmet frontier after the two
        // admitted sources. Neither an unrelated host nor a raw Prefill may
        // choose the decode component or consume its frontier obligations.
        let mut prefill = cases[0].clone();
        prefill.product = OpportunityProduct::Prefill;
        prefill.release_generated = 0;
        prefill.maximum_output = NonZeroUsize::MIN;
        prefill.prefix = PrefixKind::Ordinary;
        prefill.acquisition = None;
        let prefill_input = related::prefill_input_phase(
            1,
            8,
            61,
            1,
            SloAutomaticCostProbeSamplingPresetV1::Configured,
            false,
        );
        let mut unrelated = cases[0].clone();
        unrelated.template = 4;
        unrelated.acquisition = work::declared_plan(
            &unrelated,
            work::PrefixBlueprint {
                prompt_tokens: 61,
                boundary: 60,
                span: CheckpointTokenSpanConstraint::new(
                    NonZeroU64::new(2).unwrap(),
                    NonZeroU64::new(2).unwrap(),
                )
                .unwrap(),
                input_tokens_sha256: [71; 32],
            },
            NonZeroU32::new(8).unwrap(),
            None,
        )
        .unwrap();
        let unrelated_input = natural_termination_input_with_algorithm_and_eos(
            1,
            CostProductOutput::GreedyToken,
            true,
            64,
            "fixture.extension.a",
            [71; 32],
            true,
        );
        for (case, input) in [(prefill, prefill_input), (unrelated, unrelated_input)] {
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
    }
    // Check all mutated case bindings before selection, including native
    // setup keys. The ordinary Prefill and unsupported wide native spans use
    // the same cold-work path as their actual declarations.
    for case in &cases {
        assert_eq!(
            case.maximum_output.get(),
            case.release_generated + case.suffix_tokens
        );
        work::case_work(case, 61, 8, None).unwrap_or_else(|error| {
            panic!(
                "invalid fixture work for template {} width {}: {error}",
                case.template, case.width
            )
        });
    }
    work::setup_for_cases(&cases).unwrap();
    let seed = DeclaredAlgorithmUniverseV1::from_inputs(
        inputs
            .iter()
            .flatten()
            .filter(|f| f.family.is_some())
            .map(|f| f.original.as_deref().unwrap()),
        population.settings.max_axes,
    )
    .unwrap();
    let capacity = SelectionCapacity {
        requests: 2048,
        execution_actions: 16384,
        declared_offer_rows: 16384,
    };
    let sources = NonZeroUsize::new(2).unwrap();
    let allocation = InputAllocationPolicy::FinitePreferred {
        maximum_requests: capacity.requests,
    };
    let mut geometry = StructuredInputGeometryWorkV1::new(NonZeroU64::new(32_000_000).unwrap());
    let checked = select_with_capacity(
        &cases,
        &opportunities,
        &inputs,
        &[61; 5],
        8,
        None,
        &population,
        capacity,
        usize::MAX,
        None,
        None,
        Some(&mut geometry),
        None,
        None,
    )
    .unwrap();
    assert!(checked
        .populations
        .iter()
        .all(|p| p.input_geometry.as_ref().unwrap().complete));
    let geometry_before = serde_json::to_value(&checked.populations).unwrap();
    let spent = geometry.visits();
    // A mixed narrow journal and an already expanded Full journal are the
    // protected baseline. The new Greedy obligation uses algorithms already
    // present globally in Full, but absent from the Greedy class's own U.
    let frontier_start = if multiple_classes { 4 } else { 2 };
    let mut members = vec![Vec::new(); if multiple_classes { 6 } else { 3 }];
    for (index, member) in checked.populations.iter().enumerate() {
        let case = &cases[member.representative_case_indices[0]];
        let part = if case.product == OpportunityProduct::Prefill {
            2
        } else if case.template == 4 {
            3
        } else if case.width == 1 || multiple_classes && case.template < 2 {
            0
        } else if case.product == OpportunityProduct::Full {
            if multiple_classes && case.template == 3 {
                5
            } else {
                1
            }
        } else {
            frontier_start
        };
        members[part].push(index);
    }
    let plans: Vec<_> = members
        .iter()
        .enumerate()
        .map(|(part, members)| {
            assert!(!members.is_empty());
            let raw = batch_plan(
                members,
                &checked.populations,
                &cases,
                &opportunities,
                &[61; 5],
                8,
                None,
                &population,
            )
            .unwrap();
            if cases[raw.representative_case_indices[0]].product == OpportunityProduct::Prefill {
                return raw;
            }
            let candidate = super::super::composition::extension_candidate(
                &raw,
                None,
                std::iter::empty(),
                &checked.populations,
                &cases,
                &opportunities,
                &inputs,
                None,
                &[61; 5],
                8,
                None,
                &population,
                &seed,
                allocation,
            )
            .unwrap();
            candidate.unwrap_or_else(|| {
                // A previously unadmitted target need not be finite by
                // itself. Its cold wide-only schedule cannot use the cheap
                // original representatives until the whole joint is built.
                // Preserve that real scoped periodic candidate, as the
                // production frontier does, without promoting its schedule.
                assert!(
                    part >= frontier_start,
                    "baseline part {part} lost its finite plan"
                );
                let local = DeclaredAlgorithmUniverseV1::from_inputs(
                    raw.representative_case_indices
                        .iter()
                        .flat_map(|&i| inputs[i].iter().map(|f| f.original.as_deref().unwrap())),
                    population.settings.max_axes,
                )
                .unwrap();
                let scoped = batch_plan_with_scope(
                    members,
                    &checked.populations,
                    &cases,
                    &opportunities,
                    &[61; 5],
                    8,
                    None,
                    &population,
                    Some((&inputs, &local)),
                    allocation,
                )
                .unwrap();
                let packing = super::super::composition::packing_valid(
                    &scoped,
                    &inputs,
                    None,
                    &population,
                    &seed,
                )
                .unwrap();
                assert!(packing, "target part {part} lost its checked scope");
                assert!(scoped.schedule_within_capacity);
                assert!(scoped.finite_plan().is_none());
                scoped
            })
        })
        .collect();
    let make_candidates = || {
        plans
            .iter()
            .cloned()
            .map(|batch| BatchCandidate {
                input_priority: 0,
                coverage: CoveragePriority::from_cases(&batch.representative_case_indices, &cases)
                    .unwrap(),
                coverage_round: 0,
                decode_width_tier: 0,
                original_population_index: batch.population_indices[0],
                batch,
            })
            .collect::<Vec<_>>()
    };
    let (mut used, mut count) = (SelectionCapacity::default(), 0);
    for (index, plan) in plans.iter().enumerate() {
        assert_eq!(
            grouping::reserve(
                plan,
                0,
                capacity,
                None,
                Some(sources),
                &mut used,
                &mut count
            )
            .unwrap(),
            index < 2
        );
    }
    let target_input = inputs[plans[frontier_start].representative_case_indices[0]][0]
        .original
        .as_deref()
        .unwrap();
    assert_eq!(
        plans[0]
            .algorithm_universe
            .as_ref()
            .unwrap()
            .contains_checked_algorithms(target_input),
        Ok(false)
    );
    assert_eq!(
        plans[1]
            .algorithm_universe
            .as_ref()
            .unwrap()
            .contains_checked_algorithms(target_input),
        Ok(true)
    );
    let run = |candidates: &mut Vec<BatchCandidate>,
               populations: &[SelectedPopulation],
               available: SelectionCapacity| {
        super::super::coverage_extension::joint::extend(
            candidates,
            populations,
            &cases,
            &opportunities,
            &inputs,
            None,
            &[61; 5],
            8,
            None,
            &population,
            available,
            None,
            sources,
            &seed,
            allocation,
        )
        .unwrap();
    };
    let mut candidates = make_candidates();
    run(&mut candidates, &checked.populations, capacity);
    let combined = &candidates[0].batch;
    assert!(combined.finite_plan().is_some());
    assert_eq!(combined.algorithm_universe.as_ref(), Some(&seed));
    let required_plans = plans[..2].iter().chain(&plans[frontier_start..]);
    for old in required_plans {
        assert!(combined
            .algorithm_universe
            .as_ref()
            .unwrap()
            .contains_universe(old.algorithm_universe.as_ref().unwrap()));
        assert!(old
            .population_indices
            .iter()
            .all(|i| combined.population_indices.contains(i)));
        for &index in &old.representative_case_indices {
            let position = combined
                .representative_case_indices
                .iter()
                .position(|&i| i == index)
                .unwrap();
            assert_eq!(
                combined.scoped_opportunities.as_ref().unwrap()[position].minimum_fresh_members,
                opportunities[index].minimum_fresh_members
            );
            let expected = super::super::composition::project_opportunity(
                &opportunities[index],
                &inputs[index],
                &seed,
            )
            .unwrap();
            assert_eq!(
                combined.scoped_opportunities.as_ref().unwrap()[position].population,
                expected.population
            );
        }
    }
    if multiple_classes {
        for unrelated in &plans[2..frontier_start] {
            assert!(unrelated
                .population_indices
                .iter()
                .all(|i| !combined.population_indices.contains(i)));
        }
        let full_target = inputs[plans[5].representative_case_indices[0]][0]
            .original
            .as_deref()
            .unwrap();
        assert_eq!(
            plans[1]
                .algorithm_universe
                .as_ref()
                .unwrap()
                .contains_checked_algorithms(full_target),
            Ok(false)
        );
        // Greedy's target already declares this algorithm globally, but that
        // cannot discharge the Full class's independent input obligation.
        assert_eq!(
            plans[frontier_start]
                .algorithm_universe
                .as_ref()
                .unwrap()
                .contains_checked_algorithms(full_target),
            Ok(true)
        );
    }
    related::assert_complete_input_plan(combined);
    // Exact real work bounds are checked independently of baseline admission;
    // a cheaper standalone alternative is not falsely required to disappear.
    let exact = SelectionCapacity {
        requests: combined.requests,
        execution_actions: combined.serial_wave_upper_bound,
        declared_offer_rows: combined.declared_offer_row_bound,
    };
    let replaced: Vec<_> = (0..2).chain(frontier_start..plans.len()).collect();
    let targets: Vec<_> = (frontier_start..plans.len()).collect();
    for available in [
        SelectionCapacity {
            requests: exact.requests - 1,
            ..exact
        },
        SelectionCapacity {
            execution_actions: exact.execution_actions - 1,
            ..exact
        },
        SelectionCapacity {
            declared_offer_rows: exact.declared_offer_rows - 1,
            ..exact
        },
    ] {
        assert!(!grouping::preserves_joint_selection(
            &make_candidates(),
            &replaced,
            &targets,
            combined,
            available,
            None,
            sources
        )
        .unwrap());
    }
    if multiple_classes {
        // The joint can save requests by sharing the original cheap filler.
        // Isolate the actual added execution work: leave the other original
        // capacities intact, and first prove both baseline sources still fit.
        let available = SelectionCapacity {
            execution_actions: exact.execution_actions - 1,
            ..capacity
        };
        let (mut used, mut count) = (SelectionCapacity::default(), 0);
        for original in &plans[..2] {
            assert!(grouping::reserve(
                original,
                0,
                available,
                None,
                Some(sources),
                &mut used,
                &mut count,
            )
            .unwrap());
        }
        let mut rejected = make_candidates();
        run(&mut rejected, &checked.populations, available);
        assert_eq!(
            serde_json::to_value(rejected.iter().map(|c| &c.batch).collect::<Vec<_>>()).unwrap(),
            serde_json::to_value(&plans).unwrap()
        );
    }
    let mut invalid = checked.populations.clone();
    for family in &members[frontier_start..] {
        invalid[family[0]].input_geometry.as_mut().unwrap().complete = false;
    }
    let mut rejected = make_candidates();
    run(&mut rejected, &invalid, capacity);
    assert_eq!(
        serde_json::to_value(rejected.iter().map(|c| &c.batch).collect::<Vec<_>>()).unwrap(),
        serde_json::to_value(&plans).unwrap()
    );
    let mut proposed = CheckedSelection {
        populations: checked.populations.clone(),
        ..Default::default()
    };
    append_batch(&mut proposed, candidates.remove(0).batch, capacity, None).unwrap();
    for plan in plans[..2].iter().chain(&plans[frontier_start..]) {
        assert!(plan
            .population_indices
            .iter()
            .all(|&i| proposed.populations[i].scheduled));
    }
    finite_policy::assert_work(&proposed, &cases, &[61; 5], 8);
    assert_eq!(
        serde_json::to_value(&checked.populations).unwrap(),
        geometry_before
    );
    assert_eq!(geometry.visits(), spent);
    // These are input obligations and a freshly planned schedule, not samples,
    // parameters or a qualification certificate borrowed from either old U.
}
