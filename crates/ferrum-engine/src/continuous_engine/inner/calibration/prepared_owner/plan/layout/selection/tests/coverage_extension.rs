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
fn coverage_extension_matches_all_added_classes_to_the_admitted_mixed_source() {
    let (mut cases, mut opportunities, mut inputs, population) =
        fixture(CostProductOutput::GreedyToken);
    let (full_cases, full_opportunities, full_inputs, _) = fixture(CostProductOutput::FullLogits);
    // A model's actual route may provide FullLogits as well as GreedyToken.
    // Retain the original checked product/readback and native acquisition;
    // this is not the auxiliary forced-FullLogits route.
    for mut case in full_cases.into_iter().take(4) {
        case.route = CalibrationDecodeRoute::Actual;
        cases.push(case);
    }
    opportunities.extend(full_opportunities.into_iter().take(4));
    inputs.extend(full_inputs.into_iter().take(4));
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
    let allocation = InputAllocationPolicy::FinitePreferred {
        maximum_requests: capacity.requests,
    };
    let mut geometry = StructuredInputGeometryWorkV1::new(NonZeroU64::new(32_000_000).unwrap());
    let original = select_with_capacity(
        &cases,
        &opportunities,
        &inputs,
        &[61; 4],
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
    assert!(original
        .populations
        .iter()
        .all(|p| p.input_geometry.as_ref().unwrap().complete));
    let assessed = serde_json::to_value(&original.populations).unwrap();
    let spent = geometry.visits();
    let plan = |members: &[usize]| {
        let recipes = members.iter().flat_map(|&p| {
            original.populations[p]
                .representative_case_indices
                .iter()
                .flat_map(|&i| inputs[i].iter().map(|f| f.original.as_deref().unwrap()))
        });
        let scope = DeclaredAlgorithmUniverseV1::from_inputs(recipes, population.settings.max_axes)
            .unwrap();
        batch_plan_with_scope(
            members,
            &original.populations,
            &cases,
            &opportunities,
            &[61; 4],
            8,
            None,
            &population,
            Some((&inputs, &scope)),
            allocation,
        )
        .unwrap()
    };
    for already_mixed in [true, false] {
        let (existing, additional): (Vec<_>, Vec<_>) =
            (0..original.populations.len()).partition(|&p| {
                let belongs = |i: usize| i < 4 || (already_mixed && i >= 8);
                let indices = &original.populations[p].representative_case_indices;
                assert!(
                    indices.iter().all(|&i| belongs(i)) || indices.iter().all(|&i| !belongs(i))
                );
                belongs(indices[0])
            });
        let existing = plan(&existing);
        let additional = plan(&additional);
        assert!(existing.finite_plan().is_some());
        assert!(additional.finite_plan().is_some());
        let old_scope = existing.algorithm_universe.as_ref().unwrap();
        assert_ne!(old_scope, &seed);
        assert!(seed.contains_universe(old_scope));
        for reverse in [false, true] {
            let candidate = |batch: SelectedBatch, index| BatchCandidate {
                input_priority: 0,
                coverage: CoveragePriority::from_cases(&batch.representative_case_indices, &cases)
                    .unwrap(),
                coverage_round: 0,
                decode_width_tier: 0,
                original_population_index: index,
                batch,
            };
            // Exactly one source is admitted. Reversing the two complete
            // sources changes the admitted class set, not their input facts.
            let mut candidates = if reverse {
                vec![
                    candidate(additional.clone(), 0),
                    candidate(existing.clone(), 1),
                ]
            } else {
                vec![
                    candidate(existing.clone(), 0),
                    candidate(additional.clone(), 1),
                ]
            };
            let before: Vec<_> = candidates
                .iter()
                .map(|c| serde_json::to_value(&c.batch).unwrap())
                .collect();
            super::super::coverage_extension::extend(
                &mut candidates,
                &original.populations,
                &cases,
                &opportunities,
                &inputs,
                None,
                &[61; 4],
                8,
                None,
                &population,
                capacity,
                None,
                NonZeroUsize::MIN,
                &seed,
                allocation,
            )
            .unwrap();
            // Only the admitted mixed AB source may add Greedy CD. A Greedy
            // source cannot acquire Full via one matching Greedy constituent;
            // reversing the latter pair already admits the complete seed U.
            let accepted = already_mixed && !reverse;
            if !accepted {
                assert_eq!(
                    candidates
                        .iter()
                        .map(|c| serde_json::to_value(&c.batch).unwrap())
                        .collect::<Vec<_>>(),
                    before
                );
                continue;
            }
            assert_eq!(candidates.len(), 1);
            let combined = &candidates[0].batch;
            assert_eq!(combined.algorithm_universe.as_ref(), Some(&seed));
            assert_eq!(
                combined.population_indices,
                (0..original.populations.len()).collect::<Vec<_>>()
            );
            for p in &original.populations {
                assert!(p
                    .representative_case_indices
                    .iter()
                    .all(|i| combined.representative_case_indices.contains(i)));
            }
            let scoped = combined.scoped_opportunities.as_ref().unwrap();
            assert_eq!(member_groups(scoped).unwrap().len(), 2);
            for (&i, opportunity) in combined.representative_case_indices.iter().zip(scoped) {
                assert_eq!(
                    opportunity.minimum_fresh_members,
                    opportunities[i].minimum_fresh_members
                );
                assert_eq!(
                    opportunity.population,
                    super::super::composition::project_opportunity(
                        &opportunities[i],
                        &inputs[i],
                        &seed,
                    )
                    .unwrap()
                    .population
                );
            }
            let mut selected = CheckedSelection {
                populations: original.populations.clone(),
                ..Default::default()
            };
            append_batch(&mut selected, combined.clone(), capacity, None).unwrap();
            assert!(selected.populations.iter().all(|p| p.scheduled));
            finite_policy::assert_work(&selected, &cases, &[61; 4], 8);
        }
    }
    assert_eq!(
        serde_json::to_value(&original.populations).unwrap(),
        assessed
    );
    assert_eq!(geometry.visits(), spent);
    assert!(!geometry.exhausted());
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
