use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1;
use std::{num::NonZeroU64, sync::Arc};

fn seed(
    inputs: &[Vec<CheckedInputFacts>],
    population: &StructuredServiceDeclarationV7,
) -> DeclaredAlgorithmUniverseV1 {
    DeclaredAlgorithmUniverseV1::from_inputs(
        inputs
            .iter()
            .flatten()
            .map(|fact| fact.original.as_deref().unwrap()),
        population.settings.max_axes,
    )
    .unwrap()
}

#[test]
fn scope_first_experiment_preserves_positive_endpoints_linked_scope_and_all_three_budgets() {
    let (cases, opportunities, inputs, population) = composition::algorithm_pair_inventory([1, 2]);
    let linked = natural_termination_input_with_algorithm(
        1,
        CostProductOutput::FullLogits,
        false,
        91,
        "fixture.linked.full",
        [18; 32],
    );
    let unrelated = natural_termination_input_with_algorithm(
        1,
        CostProductOutput::GreedyToken,
        false,
        127,
        "fixture.unlinked",
        [19; 32],
    );
    let expected = DeclaredAlgorithmUniverseV1::from_inputs(
        inputs
            .iter()
            .flatten()
            .map(|fact| fact.original.as_deref().unwrap())
            .chain(std::iter::once(&linked)),
        population.settings.max_axes,
    )
    .unwrap();
    let seed = DeclaredAlgorithmUniverseV1::from_inputs(
        inputs
            .iter()
            .flatten()
            .map(|fact| fact.original.as_deref().unwrap())
            .chain([&linked, &unrelated]),
        population.settings.max_axes,
    )
    .unwrap();
    let inventory = inventory::CheckedCaseInventory {
        opportunities: opportunities.clone(),
        inputs: inputs.clone(),
        original_inputs: inputs
            .iter()
            .flatten()
            .filter_map(|fact| fact.original.clone())
            .collect(),
        algorithm_inputs: vec![Arc::new(linked)],
        algorithm_case_inputs: vec![vec![0], vec![], vec![], vec![]],
        gaps: Vec::new(),
        charge: ProbePreflightCharge::default(),
    };
    let run = |capacity, work: &mut StructuredInputGeometryWorkV1| {
        super::super::scoped_inputs::select_for_test(
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
            Some(work),
            NonZeroUsize::new(1),
            Some(&seed),
            Some(&inventory),
        )
        .unwrap()
    };
    let mut work = StructuredInputGeometryWorkV1::new(NonZeroU64::new(32_000_000).unwrap());
    let selected = run(
        SelectionCapacity {
            requests: 2048,
            execution_actions: 16384,
            declared_offer_rows: 16384,
        },
        &mut work,
    );
    assert_eq!(
        selected.populations.len(),
        1,
        "the complete common scope must precede grouping and geometry"
    );
    let member = &selected.populations[0];
    assert!(member.input_geometry.as_ref().unwrap().complete);
    let batch = &selected.batches[0];
    assert!(batch.scheduled);
    assert_eq!(batch.algorithm_universe.as_ref(), Some(&expected));
    assert!(!expected.contains_checked_algorithms(&unrelated).unwrap());
    assert_eq!(batch.representative_case_indices, [0, 1, 2, 3], "zero extension must retain each algorithm's positive minimum and maximum at every original width endpoint");
    let scoped = batch.scoped_opportunities.as_ref().unwrap();
    assert_eq!(member_groups(scoped).unwrap().len(), 1);
    for (&index, actual) in batch.representative_case_indices.iter().zip(scoped) {
        assert_eq!(
            actual.minimum_fresh_members,
            opportunities[index].minimum_fresh_members
        );
        assert_eq!(
            actual.population,
            CasePopulation::Unique(member.key.clone())
        );
    }
    let raw = batch_plan(
        &[0],
        &selected.populations,
        &cases,
        &opportunities,
        &[61, 61],
        8,
        None,
        &population,
    )
    .unwrap();
    assert_eq!(batch.schedule.min_members, raw.schedule.min_members);
    for phase in 0..3 {
        assert!(batch.input_opportunities.phase_cycles[phase] > 0);
        assert!(
            batch.input_opportunities.phase_original_offer_bounds[phase]
                >= batch.input_opportunities.maximum_fresh_member_span[phase]
        );
    }
    let mut actual = SelectionCapacity::default();
    for &index in &selected.execution_case_indices {
        let one = work::case_work(&cases[index], 61, 8, None).unwrap();
        actual.requests += one.requests;
        actual.execution_actions += one.execution_actions;
        actual.declared_offer_rows += one.serial_declared_offer_rows;
    }
    assert_eq!(actual.requests, selected.requests);
    assert_eq!(actual.execution_actions, selected.serial_wave_upper_bound);
    assert_eq!(
        actual.declared_offer_rows,
        selected.declared_offer_row_bound
    );
    for capacity in [
        actual,
        SelectionCapacity {
            requests: actual.requests - 1,
            ..actual
        },
        SelectionCapacity {
            execution_actions: actual.execution_actions - 1,
            ..actual
        },
        SelectionCapacity {
            declared_offer_rows: actual.declared_offer_rows - 1,
            ..actual
        },
    ] {
        let mut ledger = StructuredInputGeometryWorkV1::new(NonZeroU64::new(32_000_000).unwrap());
        let result = run(capacity, &mut ledger);
        assert_eq!(result.batches[0].scheduled, capacity == actual);
        assert_eq!(
            result.batches[0].algorithm_universe.as_ref(),
            Some(&expected)
        );
    }
    // A failed geometry pass spends this same ledger and is not retried with a
    // new allowance or a raw scope. Its original unavailable gap is retained.
    let mut short = StructuredInputGeometryWorkV1::new(NonZeroU64::new(work.visits() - 1).unwrap());
    let short_result = run(actual, &mut short);
    assert!(short.exhausted());
    assert_eq!(
        short_result
            .input_geometry
            .as_ref()
            .unwrap()
            .cumulative_visits,
        short.visits()
    );
    assert!(
        !short_result.populations[0]
            .input_geometry
            .as_ref()
            .unwrap()
            .complete
    );
    assert!(short_result.gaps.iter().any(|gap| matches!(
        gap.reason,
        SelectionGapReason::InputGeometryUnavailable { .. }
    )));
}

#[test]
fn scope_first_experiment_preserves_original_recipes_and_independent_host_floors() {
    let (mut cases, mut opportunities, mut inputs, population) =
        composition::algorithm_pair_inventory([1, 1]);
    // Both policies retain their real narrow and wider endpoint. A single
    // sparse GL case is a separate action-capacity negative below, not a
    // promise that a long shared F/R/Q horizon fits the original wave budget.
    for width in [1, 2] {
        let other_host =
            fixture::input_with_context(width, 3, 64, CostProductOutput::GreedyToken, false, true);
        let mut other_case = cases[0].clone();
        other_case.width = width as usize;
        other_case.preset = SloAutomaticCostProbeSamplingPresetV1::GreedyLength;
        other_case.release_generated = 3;
        other_case.suffix_tokens = other_case.maximum_output.get() - 3;
        cases.push(other_case);
        opportunities.push(CaseOpportunity {
            population: classify_alternatives(
                std::slice::from_ref(&other_host),
                population.population_policy(),
                true,
            )
            .unwrap(),
            minimum_fresh_members: 1,
        });
        inputs.push(vec![
            input_facts(&StructuredQueryV2::exact(other_host)).unwrap()
        ]);
    }
    let unknown = inputs.len();
    cases.push(cases[0].clone());
    inputs.push(inputs[0].clone());
    opportunities.push(CaseOpportunity {
        population: CasePopulation::Unknown {
            known_alternatives: vec![inputs[0][0].key(population.population_policy())],
        },
        minimum_fresh_members: 0,
    });
    let universe = seed(&inputs, &population);
    let view = super::super::scoped_inputs::prepare(
        &cases,
        &opportunities,
        &inputs,
        None,
        &population,
        &universe,
        2048,
        usize::MAX,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        view.opportunities[unknown].population,
        opportunities[unknown].population
    );
    assert!(view.scopes[unknown].is_none());
    assert_eq!(view.inputs[unknown], inputs[unknown]);
    assert_ne!(view.inputs[0][0].family, view.inputs[4][0].family);
    assert_eq!(view.scopes[0], view.scopes[4]);
    let groups = member_groups(&view.opportunities).unwrap();
    assert_eq!(
        groups
            .iter()
            .filter(|group| !group.guaranteed_case_indices.is_empty())
            .count(),
        2
    );
    for (index, original) in inputs.iter().enumerate() {
        assert_eq!(
            view.opportunities[index].minimum_fresh_members,
            opportunities[index].minimum_fresh_members
        );
        assert!(Arc::ptr_eq(
            view.inputs[index][0].original.as_ref().unwrap(),
            original[0].original.as_ref().unwrap()
        ));
        assert!(view.inputs[index][0]
            .original
            .as_ref()
            .unwrap()
            .algorithm_universe_signature()
            .is_none());
    }
    // Rebuild the failed r1 combination through the production schedule
    // planner. Its GL family has only one fresh case per cycle; the complete
    // horizon exceeds actions even though requests and source identity fit.
    let mut sparse_populations: Vec<_> = groups
        .iter()
        .filter(|group| !group.guaranteed_case_indices.is_empty())
        .map(|group| SelectedPopulation {
            key: group.key.clone(),
            representative_case_indices: group
                .guaranteed_case_indices
                .iter()
                .copied()
                .filter(|&index| index != 5)
                .collect(),
            maximum_anchor_span: 0,
            scheduled: false,
            batch_index: None,
            input_geometry: None,
        })
        .collect();
    let sparse = batch_plan_with_scope(
        &[0, 1],
        &sparse_populations,
        &cases,
        &opportunities,
        &[61, 61],
        8,
        None,
        &population,
        Some((&view.inputs, view.scopes[0].as_ref().unwrap())),
    )
    .unwrap();
    eprintln!("sparse independent-host source: requests={} actions={} rows={} cycles={} minimum_cycle={} full_cycle={}",
        sparse.requests, sparse.serial_wave_upper_bound, sparse.declared_offer_row_bound, sparse.planned_cycles,
        sparse.input_opportunities.minimum_original_offers_per_completed_cycle,
        sparse.input_opportunities.successful_cycle_wave_upper_bound);
    assert!(sparse.schedule_within_capacity);
    assert!(sparse.maximum_anchor_span <= *sparse.schedule.phase_min_offered.iter().min().unwrap());
    assert!(sparse.requests <= 2048);
    assert!(sparse.serial_wave_upper_bound > 16384);
    assert!(super::super::composition::packing_valid(
        &sparse,
        &view.inputs,
        None,
        &population,
        &universe
    )
    .unwrap());
    assert!(!super::super::composition::can_schedule(
        &sparse,
        SelectionCapacity {
            requests: 2048,
            execution_actions: 16384,
            declared_offer_rows: 16384
        }
    ));
    // The second real GL endpoint contributes its own independent request;
    // the original shared schedule must calculate whether this full plan fits.
    for member in &mut sparse_populations {
        if member.representative_case_indices.contains(&4) {
            member.representative_case_indices.push(5);
        }
    }
    let complete = batch_plan_with_scope(
        &[0, 1],
        &sparse_populations,
        &cases,
        &opportunities,
        &[61, 61],
        8,
        None,
        &population,
        Some((&view.inputs, view.scopes[0].as_ref().unwrap())),
    )
    .unwrap();
    eprintln!(
        "complete independent-host source: requests={} actions={} rows={} cycles={}",
        complete.requests,
        complete.serial_wave_upper_bound,
        complete.declared_offer_row_bound,
        complete.planned_cycles
    );
    assert!(super::super::composition::can_schedule(
        &complete,
        SelectionCapacity {
            requests: 2048,
            execution_actions: 16384,
            declared_offer_rows: 16384
        }
    ));
    let mut work = StructuredInputGeometryWorkV1::new(NonZeroU64::new(32_000_000).unwrap());
    let selection = super::super::scoped_inputs::select_for_test(
        &cases,
        &opportunities,
        &inputs,
        &[61, 61],
        8,
        None,
        &population,
        SelectionCapacity {
            requests: 2048,
            execution_actions: 16384,
            declared_offer_rows: 16384,
        },
        usize::MAX,
        None,
        None,
        Some(&mut work),
        NonZeroUsize::new(1),
        Some(&universe),
        None,
    )
    .unwrap();
    let selected: Vec<_> = selection
        .batches
        .iter()
        .filter(|batch| batch.scheduled)
        .collect();
    assert_eq!(selected.len(), 1);
    let batch = selected[0];
    assert_eq!(
        batch.population_indices.len(),
        2,
        "both independently planned host families must survive finite-source coalescing"
    );
    assert_eq!(batch.algorithm_universe.as_ref(), view.scopes[0].as_ref());
    assert_eq!(
        member_groups(batch.scoped_opportunities.as_ref().unwrap())
            .unwrap()
            .len(),
        2
    );
    assert!(!batch.representative_case_indices.contains(&unknown));
    assert!(selection
        .gaps
        .iter()
        .any(|gap| matches!(gap.reason, SelectionGapReason::UnknownPopulation)));
    for &member in &batch.population_indices {
        assert!(
            selection.populations[member]
                .input_geometry
                .as_ref()
                .unwrap()
                .complete
        );
    }
    assert!(super::super::scoped_inputs::prepare(
        &cases,
        &opportunities,
        &inputs,
        None,
        &population,
        &universe,
        2048,
        0
    )
    .unwrap()
    .is_none());
    let incomplete = DeclaredAlgorithmUniverseV1::from_inputs(
        std::iter::once(inputs[0][0].original.as_deref().unwrap()),
        population.settings.max_axes,
    )
    .unwrap();
    assert!(super::super::scoped_inputs::prepare(
        &cases,
        &opportunities,
        &inputs,
        None,
        &population,
        &incomplete,
        2048,
        usize::MAX
    )
    .unwrap()
    .is_none());
}

#[test]
fn scope_first_experiment_c8_keeps_complete_geometry_and_rejects_unaffordable_cold_horizon() {
    let (cases, opportunities, inputs, population) = composition::algorithm_pair_inventory([1, 4]);
    assert_eq!(
        cases.iter().map(|case| case.width).collect::<Vec<_>>(),
        [1, 2, 4, 8]
    );
    let universe = seed(&inputs, &population);
    let mut work = StructuredInputGeometryWorkV1::new(NonZeroU64::new(32_000_000).unwrap());
    // Keep this rejected broad plan as an explicit experiment. Product
    // selection must not replace its schedulable narrow sources with it.
    let selection = super::super::scoped_inputs::select_for_test(
        &cases,
        &opportunities,
        &inputs,
        &[61, 61],
        8,
        None,
        &population,
        SelectionCapacity {
            requests: 2048,
            execution_actions: 16384,
            declared_offer_rows: 16384,
        },
        usize::MAX,
        None,
        None,
        Some(&mut work),
        NonZeroUsize::new(1),
        Some(&universe),
        None,
    )
    .unwrap();
    assert_eq!(selection.populations.len(), 1);
    let member = &selection.populations[0];
    assert!(member.input_geometry.as_ref().unwrap().complete);
    assert_eq!(member.representative_case_indices, [0, 1, 2, 3]);
    let batch = &selection.batches[0];
    assert_eq!(batch.algorithm_universe.as_ref(), Some(&universe));
    assert!(batch.requests <= 2048);
    assert!(batch.serial_wave_upper_bound > 16384);
    assert!(
        !batch.scheduled,
        "retaining C8 geometry cannot bypass the original complete-horizon action cap"
    );
    assert!(selection.execution_case_indices.is_empty());
    assert!(selection.gaps.iter().any(|gap| matches!(gap.reason,
        SelectionGapReason::RemainingWaves { required, remaining }
        if required == batch.serial_wave_upper_bound && remaining == 16384)));
    eprintln!(
        "complete C8 cold source: requests={} actions={} rows={} cycles={} geometry_visits={}",
        batch.requests,
        batch.serial_wave_upper_bound,
        batch.declared_offer_row_bound,
        batch.planned_cycles,
        work.visits()
    );
}

#[test]
fn scope_first_preserves_schedulable_narrow_inputs_when_complete_c8_horizon_does_not_fit() {
    let (cases, opportunities, inputs, population) = composition::algorithm_pair_inventory([1, 4]);
    let universe = seed(&inputs, &population);
    let capacity = SelectionCapacity {
        requests: 2048,
        execution_actions: 16384,
        declared_offer_rows: 16384,
    };
    // Independent reference invocation of the original raw selector, with the
    // same inventory, source limit, geometry allowance and complete F/R/Q.
    // This is not a second production pass or a refunded geometry ledger.
    let mut original_work =
        StructuredInputGeometryWorkV1::new(NonZeroU64::new(32_000_000).unwrap());
    let original = select_prepared_inputs(
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
        Some(&mut original_work),
        NonZeroUsize::new(1),
        Some(&universe),
        None,
        None,
    )
    .unwrap();
    let original_batch = original
        .batches
        .iter()
        .find(|batch| batch.scheduled)
        .expect("the original narrow family must have a complete schedulable source");
    assert_eq!(
        original_batch
            .representative_case_indices
            .iter()
            .map(|&index| cases[index].width)
            .collect::<Vec<_>>(),
        [1, 2]
    );
    assert!(original_batch.algorithm_universe.is_none());
    assert!(original_batch.schedule_within_capacity);
    assert!(
        original_batch.maximum_anchor_span
            <= *original_batch
                .schedule
                .phase_min_offered
                .iter()
                .min()
                .unwrap()
    );
    for &index in &original_batch.population_indices {
        assert!(
            original.populations[index]
                .input_geometry
                .as_ref()
                .unwrap()
                .complete
        );
    }
    for phase in 0..3 {
        assert!(original_batch.input_opportunities.phase_cycles[phase] > 0);
        assert!(
            original_batch
                .input_opportunities
                .phase_original_offer_bounds[phase]
                >= original_batch.input_opportunities.maximum_fresh_member_span[phase]
        );
    }
    let mut actual = SelectionCapacity::default();
    for &index in &original.execution_case_indices {
        let work = work::case_work(&cases[index], 61, 8, None).unwrap();
        actual.requests += work.requests;
        actual.execution_actions += work.execution_actions;
        actual.declared_offer_rows += work.serial_declared_offer_rows;
    }
    assert_eq!(actual.requests, original.requests);
    assert_eq!(actual.execution_actions, original.serial_wave_upper_bound);
    assert_eq!(
        actual.declared_offer_rows,
        original.declared_offer_row_bound
    );
    assert!(actual.requests <= capacity.requests);
    assert!(actual.execution_actions <= capacity.execution_actions);
    assert!(actual.declared_offer_rows <= capacity.declared_offer_rows);

    let mut candidate_work =
        StructuredInputGeometryWorkV1::new(NonZeroU64::new(32_000_000).unwrap());
    let candidate = select_with_capacity(
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
        Some(&mut candidate_work),
        NonZeroUsize::new(1),
        Some(&universe),
    )
    .unwrap();
    eprintln!("original narrow reservation: requests={} actions={} rows={} cycles={}; candidate reservations={:?}",
        original.requests, original.serial_wave_upper_bound, original.declared_offer_row_bound,
        original_batch.planned_cycles,
        candidate.batches.iter().map(|batch| (batch.scheduled, batch.requests, batch.serial_wave_upper_bound, batch.declared_offer_row_bound)).collect::<Vec<_>>());
    // These are the two real positive endpoints of algorithm A. A wider
    // declaration may retain them in a different numerical scope, but merely
    // listing them in an unscheduled all-width candidate loses prior support.
    assert!(candidate.batches.iter().filter(|batch| batch.scheduled).any(|batch| {
        original_batch.representative_case_indices.iter().all(|index| batch.representative_case_indices.contains(index))
            && batch.schedule.min_members == original_batch.schedule.min_members
            && batch.population_indices.iter().all(|&index| candidate.populations[index].input_geometry.as_ref().is_some_and(|geometry| geometry.complete))
    }), "scope-first selection must retain the original schedulable numerical endpoints and complete independent F/R/Q within the unchanged budgets");
    assert_eq!(
        serde_json::to_value(&candidate).unwrap(),
        serde_json::to_value(&original).unwrap(),
        "product selection retains the original scopes, complete phase horizons, capacity gaps and executed case occurrences"
    );
    assert_eq!(
        candidate_work.visits(),
        original_work.visits(),
        "restoring the raw plan performs exactly one geometry pass; no broad trial is refunded or hidden"
    );
    assert_eq!(candidate_work.exhausted(), original_work.exhausted());
}
