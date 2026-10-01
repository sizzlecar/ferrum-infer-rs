use super::*;

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
    let single = (
        batches[0].batch.requests,
        batches[0].batch.serial_wave_upper_bound,
    );
    let planned = super::super::composition::scheduled_prefix_budget(
        &batches,
        100_000,
        10_000_000,
        None,
        NonZeroUsize::new(2),
    )
    .unwrap();
    assert_eq!(planned, (2 * single.0, 2 * single.1));
    let work_limited = super::super::composition::scheduled_prefix_budget(
        &batches,
        single.0,
        single.1,
        None,
        NonZeroUsize::new(3),
    )
    .unwrap();
    assert_eq!(work_limited, single);
    let deferred = super::super::composition::scheduled_prefix_budget(
        &batches,
        100_000,
        10_000_000,
        Some(1),
        NonZeroUsize::new(3),
    )
    .unwrap();
    assert_eq!(deferred, (0, 0));
}

#[test]
fn local_combination_cannot_promote_an_invalid_raw_source_into_a_reserved_slot() {
    let (mut batches, _, _) = original_batches();
    batches[0].batch.schedule_within_capacity = false;
    assert!(!super::super::composition::can_schedule(
        &batches[0].batch,
        usize::MAX,
        usize::MAX
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
        usize::MAX,
        usize::MAX
    ));
    let expected = (
        batches[2].batch.requests,
        batches[2].batch.serial_wave_upper_bound,
    );
    assert_eq!(
        super::super::composition::scheduled_prefix_budget(
            &batches,
            usize::MAX,
            usize::MAX,
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

#[test]
fn local_combination_declares_only_checked_compatible_sources_before_collection() {
    let (original_cases, _, _, population) = inventory();
    let mut cases = Vec::new();
    let mut opportunities = Vec::new();
    let mut inputs = Vec::new();
    for (template, native_op_id, implementation) in [
        (0, "fixture.selection.a", [7; 32]),
        (1, "fixture.selection.b", [8; 32]),
    ] {
        // Different complete width sets deliberately remain separate raw
        // journals. Same-width context/algorithm families now share a journal.
        for old in &original_cases[..2] {
            let mut case = old.clone();
            case.width *= template + 1;
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
