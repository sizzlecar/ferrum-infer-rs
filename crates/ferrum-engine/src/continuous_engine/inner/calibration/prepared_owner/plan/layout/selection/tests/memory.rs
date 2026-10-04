use super::*;

#[test]
fn checked_selection_distinct_family_stage_bound_preserves_the_full_plan() {
    use std::num::NonZeroU64;
    let (mut cases, mut opportunities, mut facts, population) = two_families();
    install_natural_termination_facts(
        &mut cases[4..],
        &mut opportunities[4..],
        &mut facts[4..],
        false,
    );
    let original_cases = cases.clone();
    let original_opportunities = opportunities.clone();
    let original_facts = facts.clone();
    // Grow the input frontier beyond a rank-sized set without inventing more
    // family identities. Each case still denotes its own original request.
    for _ in 0..population.settings.max_rank {
        cases.extend(original_cases.iter().cloned());
        opportunities.extend(original_opportunities.iter().cloned());
        facts.extend(original_facts.iter().cloned());
    }
    append_uncaptured(
        &mut cases,
        &mut opportunities,
        &mut facts,
        original_cases.len() * (population.settings.max_rank + 1) + 1,
    );
    let groups = member_groups(&opportunities).unwrap();
    let audit = super::super::memory::plan(
        &groups,
        &opportunities,
        &facts,
        2048,
        Some(&population.settings),
    )
    .unwrap();
    assert_eq!(audit.distinct_groups, groups.len());
    assert!(audit.guaranteed_groups < audit.guaranteed_cases);
    assert_eq!(
        audit.retained_groups_bytes
            + audit.output_bound_bytes
            + audit.candidates_scratch_bytes
            + audit
                .representative_scratch_bytes
                .max(audit.geometry_scratch_bytes)
                .max(audit.batch_scratch_bytes),
        audit.grouped_selection_peak_bytes
    );
    let old_peak = legacy_selection_peak_payload_bound(&cases, &opportunities, &facts, 2048)
        .unwrap()
        + super::super::input_geometry::scratch_bound(&opportunities, &facts, &population.settings)
            .unwrap();
    assert!(audit.required_peak_bytes < old_peak);
    let limit = NonZeroU64::new(32_000_000).unwrap();
    let mut expected_work = StructuredInputGeometryWorkV1::new(limit);
    let expected = select_changed_with_geometry(
        &cases,
        &opportunities,
        &facts,
        &[61],
        8,
        &population,
        2048,
        10_000_000,
        old_peak,
        None,
        Some(0),
        Some(&mut expected_work),
    )
    .unwrap();
    let mut rejected_work = StructuredInputGeometryWorkV1::new(limit);
    assert!(select_changed_with_geometry(
        &cases,
        &opportunities,
        &facts,
        &[61],
        8,
        &population,
        2048,
        10_000_000,
        audit.required_peak_bytes - 1,
        None,
        Some(0),
        Some(&mut rejected_work),
    )
    .is_err());
    assert_eq!(
        rejected_work.visits(),
        0,
        "capacity rejection precedes geometry"
    );
    let mut actual_work = StructuredInputGeometryWorkV1::new(limit);
    let actual = select_changed_with_geometry(
        &cases,
        &opportunities,
        &facts,
        &[61],
        8,
        &population,
        2048,
        10_000_000,
        audit.required_peak_bytes,
        None,
        Some(0),
        Some(&mut actual_work),
    )
    .unwrap();
    // Extra finite storage may shorten the roomy plan's occurrence stream.
    // It must retain every original admitted key, endpoint and scoped floor.
    finite_policy::assert_preserves_coverage(&actual, &expected);
    assert_eq!(
        serde_json::to_value(&actual.gaps).unwrap(),
        serde_json::to_value(&expected.gaps).unwrap()
    );
    finite_policy::assert_work(&actual, &cases, &[61], 8);
    finite_policy::assert_work(&expected, &cases, &[61], 8);
    assert_eq!(actual_work.visits(), expected_work.visits());
    assert!(actual
        .gaps
        .iter()
        .any(|gap| matches!(gap.reason, SelectionGapReason::UnknownPopulation)));
    assert!(actual
        .gaps
        .iter()
        .any(|gap| matches!(gap.reason, SelectionGapReason::DeferredInputPriority { .. })));
    assert!(
        audit.retained_groups_bytes + actual.retained_payload_bytes().unwrap()
            <= audit.required_peak_bytes
    );
}

#[test]
fn checked_selection_frozen_facts_release_raw_inputs_without_changing_selection() {
    use std::sync::Arc;
    let (cases, opportunities, facts, population) = two_families();
    let expected = select(
        &cases,
        &opportunities,
        &facts,
        &[61],
        8,
        &population,
        2048,
        10_000_000,
        usize::MAX,
    )
    .unwrap();
    let original = facts[0][0].original.as_ref().unwrap().as_ref().clone();
    let mut inventory = super::super::super::inventory::CheckedCaseInventory {
        opportunities,
        original_inputs: facts
            .iter()
            .flatten()
            .filter_map(|facts| facts.original.clone())
            .collect(),
        inputs: facts,
        algorithm_inputs: vec![Arc::new(original)],
        algorithm_case_inputs: Vec::new(),
        gaps: Vec::new(),
        charge: ProbePreflightCharge::default(),
    };
    let mut raw = inventory
        .inputs
        .iter()
        .flatten()
        .map(|facts| Arc::downgrade(facts.original.as_ref().unwrap()))
        .collect::<Vec<_>>();
    raw.extend(inventory.algorithm_inputs.iter().map(Arc::downgrade));
    let before = inventory.retained_payload_bytes().unwrap();
    inventory.release_original_inputs();
    let after = inventory.retained_payload_bytes().unwrap();
    assert!(after < before);
    assert!(
        raw.iter().all(|input| input.upgrade().is_none()),
        "release actual ownership, not only its charge"
    );
    assert_eq!(inventory.algorithm_inputs.capacity(), 0);
    assert!(inventory
        .inputs
        .iter()
        .flatten()
        .all(|facts| facts.original.is_none()));
    let actual = select(
        &cases,
        &inventory.opportunities,
        &inventory.inputs,
        &[61],
        8,
        &population,
        2048,
        10_000_000,
        usize::MAX,
    )
    .unwrap();
    assert_eq!(
        serde_json::to_value(actual).unwrap(),
        serde_json::to_value(expected).unwrap()
    );
    inventory.release_original_inputs();
    assert_eq!(inventory.retained_payload_bytes().unwrap(), after);
}

#[test]
fn checked_selection_grouping_and_unique_family_capacity_remain_bounded() {
    use std::num::NonZeroU64;
    let (mut cases, mut opportunities, mut facts, population) = inventory();
    cases.clear();
    opportunities.clear();
    facts.clear();
    let declared = two_families().0.remove(0);
    for width in 2..=fixture::domain().limits().maximum_rows.get() {
        let input = fixture::input(width, 3, CostProductOutput::GreedyToken, true, true);
        let mut case = declared.clone();
        case.width = width as usize;
        cases.push(case);
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
    }
    let groups = member_groups(&opportunities).unwrap();
    let audit = super::super::memory::plan(
        &groups,
        &opportunities,
        &facts,
        2048,
        Some(&population.settings),
    )
    .unwrap();
    assert_eq!(audit.distinct_groups, cases.len());
    assert_eq!(audit.guaranteed_groups, cases.len());
    let mut work = StructuredInputGeometryWorkV1::new(NonZeroU64::new(32_000_000).unwrap());
    let reason = select_changed_with_geometry(
        &cases,
        &opportunities,
        &facts,
        &[61],
        8,
        &population,
        2048,
        10_000_000,
        audit.grouping_peak_bytes - 1,
        None,
        None,
        Some(&mut work),
    )
    .unwrap_err()
    .to_string();
    assert!(reason.contains("grouping retained payload"));
    assert_eq!(work.visits(), 0);
    let actual = select_changed_with_geometry(
        &cases,
        &opportunities,
        &facts,
        &[61],
        8,
        &population,
        2048,
        10_000_000,
        audit.required_peak_bytes,
        None,
        None,
        Some(&mut work),
    )
    .unwrap();
    assert_eq!(actual.populations.len(), groups.len());
    assert!(
        audit.retained_groups_bytes + actual.retained_payload_bytes().unwrap()
            <= audit.required_peak_bytes
    );
}

fn append_uncaptured(
    cases: &mut Vec<Case>,
    opportunities: &mut Vec<CaseOpportunity>,
    facts: &mut Vec<Vec<CheckedInputFacts>>,
    total: usize,
) {
    let declared = cases[0].clone();
    while cases.len() < total {
        cases.push(declared.clone());
        opportunities.push(CaseOpportunity {
            population: CasePopulation::Unknown {
                known_alternatives: Vec::new(),
            },
            minimum_fresh_members: 0,
        });
        facts.push(Vec::new());
    }
}

#[test]
fn local_composition_unknown_declarations_share_the_original_gap_backing() {
    use std::num::NonZeroU64;

    let (mut cases, mut opportunities, mut facts, population) =
        composition::algorithm_pair_inventory([1, 1]);
    let pair_cases = cases.len();
    let seed = ferrum_scheduler::implementations::continuous::cost_model::structured_v2::
        DeclaredAlgorithmUniverseV1::from_inputs(facts.iter().flatten()
            .map(|f| f.original.as_deref().unwrap()), population.settings.max_axes).unwrap();
    // A real FullLogits length-boundary input has a distinct population from
    // the greedy pair. It retains its original early-stop obligation and is
    // deferred by input priority, while the ordinary pair can form one union.
    let mut terminal = cases[0].clone();
    terminal.product = OpportunityProduct::Full;
    terminal.maximum_output = NonZeroUsize::new(4).unwrap();
    terminal.release_generated = 3;
    terminal.suffix_tokens = 1;
    let input = natural_termination_input_with_algorithm(
        terminal.width as u32,
        CostProductOutput::FullLogits,
        true,
        64,
        "fixture.selection.a",
        [7; 32],
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
    let terminal_facts = input_facts(&StructuredQueryV2::exact(input)).unwrap();
    assert!(terminal_facts.branches[4]);
    assert!(!terminal_facts.branches[5]);
    assert!(facts
        .iter()
        .all(|row| row[0].key(population.population_policy())
            != terminal_facts.key(population.population_policy())));
    facts.push(vec![terminal_facts]);
    cases.push(terminal);
    let first_unknown = cases.len();
    append_uncaptured(
        &mut cases,
        &mut opportunities,
        &mut facts,
        first_unknown + 1,
    );
    let original_len = cases.len();
    let original_groups = member_groups(&opportunities).unwrap();
    // Uncaptured declarations create no candidate, scope, member, or per-case
    // gap: the same single aggregate Unknown gap was already present above.
    append_uncaptured(
        &mut cases,
        &mut opportunities,
        &mut facts,
        original_len + population.settings.max_rank,
    );
    let groups = member_groups(&opportunities).unwrap();
    let work_limit = NonZeroU64::new(32_000_000).unwrap();
    for capacity in [
        SelectionCapacity::legacy(2048, 10_000_000),
        SelectionCapacity {
            requests: 0,
            execution_actions: 0,
            declared_offer_rows: 0,
        },
    ] {
        let original_memory = super::super::memory::plan(
            &original_groups,
            &opportunities[..original_len],
            &facts[..original_len],
            capacity.requests,
            Some(&population.settings),
        )
        .unwrap();
        let memory = super::super::memory::plan(
            &groups,
            &opportunities,
            &facts,
            capacity.requests,
            Some(&population.settings),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(&memory).unwrap(),
            serde_json::to_value(&original_memory).unwrap(),
            "the same live groups and gap backing retain the same stage bound"
        );
        // Derive the cap from the original owned stages, not a copied byte
        // formula or a machine-sized allowance. Extra empty declarations must
        // fit the identical cap: they add no selection-owned allocation.
        let tight = original_memory.required_peak_bytes
            + super::super::composition::extra_peak(
                &opportunities[..original_len],
                &seed,
                original_memory.guaranteed_groups,
            )
            .unwrap();
        let declared_peak = memory.required_peak_bytes
            + super::super::composition::extra_peak(
                &opportunities,
                &seed,
                memory.guaranteed_groups,
            )
            .unwrap();
        // This bound describes the original prepared raw view. The public
        // entrypoint can now admit a different, smaller projected view first;
        // its bound is checked separately below. Keep the original core's
        // gap-backing and one-byte rejection contract exact.
        let run = |maximum, work: &mut StructuredInputGeometryWorkV1| {
            select_prepared_inputs(
                &cases,
                &opportunities,
                &facts,
                &[61, 61],
                8,
                None,
                &population,
                capacity,
                maximum,
                None,
                Some(1),
                Some(work),
                None,
                Some(&seed),
                None,
                None,
            )
        };
        let mut expected_work = StructuredInputGeometryWorkV1::new(work_limit);
        let expected = run(
            declared_peak + memory.required_peak_bytes,
            &mut expected_work,
        )
        .unwrap();
        assert!(expected
            .gaps
            .iter()
            .any(|g| matches!(g.reason, SelectionGapReason::UnknownPopulation)));
        assert!(expected.gaps.iter().any(|g| matches!(
            g.reason,
            SelectionGapReason::OutcomeDependentEarlyTermination
        )));
        assert!(expected.gaps.iter().any(|g| matches!(
            g.reason,
            SelectionGapReason::EarlyTerminalOpportunityMissing
        )));
        assert!(expected
            .gaps
            .iter()
            .any(|g| matches!(g.reason, SelectionGapReason::DeferredInputPriority { .. })));
        if capacity.requests != 0 {
            let union = expected
                .batches
                .iter()
                .find(|b| b.scheduled && b.algorithm_universe.is_some())
                .unwrap();
            for index in 0..pair_cases {
                assert!(
                    union.representative_case_indices.contains(&index),
                    "original width/algorithm endpoint lost"
                );
            }
            assert!(union.scoped_opportunities.is_some());
        } else {
            assert!(expected.execution_case_indices.is_empty());
            assert!(expected
                .gaps
                .iter()
                .any(|g| matches!(g.reason, SelectionGapReason::RemainingRequests { .. })));
            assert!(expected
                .gaps
                .iter()
                .any(|g| matches!(g.reason, SelectionGapReason::RemainingWaves { .. })));
            assert!(expected
                .gaps
                .iter()
                .any(|g| matches!(g.reason, SelectionGapReason::RemainingOfferRows { .. })));
        }
        eprintln!("composition gap backing: original_declarations={original_len} extended_declarations={} guaranteed_groups={} original_peak={tight} extended_peak={declared_peak} actual_gaps={} gap_capacity={}",
            cases.len(), memory.guaranteed_groups, expected.gaps.len(), expected.gaps.capacity());
        let mut actual_work = StructuredInputGeometryWorkV1::new(work_limit);
        let actual = run(tight, &mut actual_work)
            .expect("unchanged live gap backing must fit its original authorization");
        assert_eq!(serde_json::to_value(&actual).unwrap(), serde_json::to_value(&expected).unwrap(),
            "complete representatives, scopes, schedules, all gaps and all work ledgers survive the tight cap");
        assert_eq!(actual_work.visits(), expected_work.visits());
        assert!(memory.retained_groups_bytes + actual.retained_payload_bytes().unwrap() <= tight);
        let mut totals = [0usize; 3];
        for &index in &actual.execution_case_indices {
            let work = work::case_work(&cases[index], 61, 8, None).unwrap();
            totals[0] += work.requests;
            totals[1] += work.execution_actions;
            totals[2] += work.serial_declared_offer_rows;
        }
        assert_eq!(
            totals,
            [
                actual.requests,
                actual.serial_wave_upper_bound,
                actual.declared_offer_row_bound
            ]
        );
        let mut rejected_work = StructuredInputGeometryWorkV1::new(work_limit);
        assert!(run(tight - 1, &mut rejected_work).is_err());
        assert_eq!(
            rejected_work.visits(),
            0,
            "capacity rejection precedes geometry"
        );
        assert!(composition_authorized(
            &opportunities,
            &facts,
            capacity.requests,
            &population,
            true,
            tight,
            &seed
        )
        .unwrap());
        assert!(!composition_authorized(
            &opportunities,
            &facts,
            capacity.requests,
            &population,
            true,
            tight - 1,
            &seed
        )
        .unwrap());
        if capacity.requests != 0 {
            // Account the real new view alongside the same selector stages.
            // This fixture actually collapses two raw numerical families, so
            // both paths have distinct, independently derived admission caps.
            let view = super::super::scoped_inputs::prepare(
                &cases,
                &opportunities,
                &facts,
                None,
                &population,
                &seed,
                capacity.requests,
                usize::MAX,
            )
            .unwrap()
            .unwrap();
            let projected_groups = member_groups(&view.opportunities).unwrap();
            let projected = super::super::memory::plan(
                &projected_groups,
                &view.opportunities,
                &view.inputs,
                capacity.requests,
                Some(&population.settings),
            )
            .unwrap();
            assert!(projected.guaranteed_groups < memory.guaranteed_groups);
            let projected_peak = view.reserved_bytes
                + projected.required_peak_bytes
                + super::super::composition::extra_peak(
                    &view.opportunities,
                    &seed,
                    projected.guaranteed_groups,
                )
                .unwrap();
            assert!(projected_peak < tight, "this fixture must exercise genuinely smaller admission, not a bypass of the old bound");
            // The broad projected path is now an explicit experiment. Its
            // distinct memory gate must still reject before geometry; it is
            // not the product entrypoint's raw-plan preservation contract.
            let experimental_run = |maximum, work: &mut StructuredInputGeometryWorkV1| {
                super::super::scoped_inputs::select_for_test(
                    &cases,
                    &opportunities,
                    &facts,
                    &[61, 61],
                    8,
                    None,
                    &population,
                    capacity,
                    maximum,
                    None,
                    Some(1),
                    Some(work),
                    None,
                    Some(&seed),
                    None,
                )
            };
            let mut wide_work = StructuredInputGeometryWorkV1::new(work_limit);
            let wide = experimental_run(usize::MAX, &mut wide_work).unwrap();
            let mut tight_work = StructuredInputGeometryWorkV1::new(work_limit);
            let scoped = experimental_run(projected_peak, &mut tight_work).unwrap();
            finite_policy::assert_preserves_coverage(&scoped, &wide);
            assert_eq!(
                serde_json::to_value(&scoped.gaps).unwrap(),
                serde_json::to_value(&wide.gaps).unwrap()
            );
            finite_policy::assert_work(&scoped, &cases, &[61, 61], 8);
            finite_policy::assert_work(&wide, &cases, &[61, 61], 8);
            assert_eq!(tight_work.visits(), wide_work.visits());
            let union = scoped.batches.iter().find(|batch| batch.scheduled).unwrap();
            assert!(union.algorithm_universe.is_some());
            assert!((0..pair_cases).all(|index| union.representative_case_indices.contains(&index)));
            assert!(scoped
                .gaps
                .iter()
                .any(|gap| matches!(gap.reason, SelectionGapReason::UnknownPopulation)));
            assert!(
                view.reserved_bytes
                    + projected.retained_groups_bytes
                    + scoped.retained_payload_bytes().unwrap()
                    <= projected_peak
            );
            // Here the projected cap is strictly below the raw cap. One byte
            // less admits neither path: fallback must not start geometry.
            let mut denied = StructuredInputGeometryWorkV1::new(work_limit);
            assert!(experimental_run(projected_peak - 1, &mut denied).is_err());
            assert_eq!(denied.visits(), 0);
            eprintln!("distinct selection admission: raw_peak={tight} projected_peak={projected_peak} raw_groups={} projected_groups={}",
                memory.guaranteed_groups, projected.guaranteed_groups);
        }
    }
}

#[test]
fn checked_selection_sparse_declarations_keep_exact_capacity_and_typed_gaps() {
    let (mut cases, mut opportunities, mut facts, population) = two_families();
    install_natural_termination_facts(
        &mut cases[4..],
        &mut opportunities[4..],
        &mut facts[4..],
        false,
    );
    let guaranteed = cases.len();
    append_uncaptured(&mut cases, &mut opportunities, &mut facts, guaranteed + 1);
    let peak = selection_peak_payload_bound(&cases, &opportunities, &facts, 2048).unwrap();
    let expected = select_changed(
        &cases,
        &opportunities,
        &facts,
        &[61],
        8,
        &population,
        2048,
        10_000_000,
        peak,
        None,
        Some(0),
    )
    .unwrap();
    assert!(!expected.execution_case_indices.is_empty());
    assert!(expected
        .gaps
        .iter()
        .any(|gap| matches!(gap.reason, SelectionGapReason::UnknownPopulation)));
    assert!(expected.gaps.iter().any(|gap| matches!(
        gap.reason,
        SelectionGapReason::DeferredInputPriority {
            priority: 1,
            selected: 0
        }
    )));

    // The large declared space is already owned/charged by the inventory.
    // These untouched entries cannot allocate selector populations or batches.
    for total in [32, 1024, 8192] {
        append_uncaptured(&mut cases, &mut opportunities, &mut facts, total);
        assert_eq!(
            selection_inventory_cardinality(&opportunities).unwrap(),
            (guaranteed, guaranteed)
        );
        assert_eq!(
            selection_peak_payload_bound(&cases, &opportunities, &facts, 2048).unwrap(),
            peak
        );
        let error = select_changed(
            &cases,
            &opportunities,
            &facts,
            &[61],
            8,
            &population,
            2048,
            10_000_000,
            peak - 1,
            None,
            Some(0),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains(&format!("required_peak_bytes={peak}")));
        assert!(error.contains(&format!("remaining_bytes={}", peak - 1)));
        assert!(error.contains(&format!("key_mentions={guaranteed}")));
        assert!(error.contains(&format!("guaranteed_cases={guaranteed}")));
        let actual = select_changed(
            &cases,
            &opportunities,
            &facts,
            &[61],
            8,
            &population,
            2048,
            10_000_000,
            peak,
            None,
            Some(0),
        )
        .unwrap();
        // Full comparison covers execution order, widths, schedules, budgets,
        // every population and its deferred/Unknown gap, not just acceptance.
        assert_eq!(
            serde_json::to_value(&actual).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
        assert!(actual.retained_payload_bytes().unwrap() <= peak);
        assert_eq!(
            opportunities
                .iter()
                .filter(|o| matches!(o.population, CasePopulation::Unknown { .. }))
                .count(),
            total - guaranteed
        );
    }
}

#[test]
fn checked_selection_possible_only_groups_and_vec_growth_remain_charged() {
    let (mut cases, mut opportunities, mut facts, population) = two_families();
    let guaranteed = cases.len();
    let mut possible = Vec::new();
    // Real heterogeneous checked inputs take the original ExactOwner fallback;
    // their distinct widths form possible-only populations with no member floor.
    for width in 2..=8 {
        let input = fixture::input(width, 3, CostProductOutput::GreedyToken, true, true);
        let CasePopulation::Unique(key) = classify_alternatives(
            std::slice::from_ref(&input),
            StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
            true,
        )
        .unwrap() else {
            panic!("checked heterogeneous input must retain its exact owner")
        };
        let mut case = cases[0].clone();
        case.width = width as usize;
        possible.push((
            key,
            case,
            input_facts(&StructuredQueryV2::exact(input)).unwrap(),
        ));
    }
    let mut previous_peak =
        selection_peak_payload_bound(&cases, &opportunities, &facts, 2048).unwrap();
    for repeat in 1..=17 {
        for (key, case, input) in &possible {
            cases.push(case.clone());
            opportunities.push(CaseOpportunity {
                population: CasePopulation::Unique(key.clone()),
                minimum_fresh_members: 0,
            });
            facts.push(vec![input.clone()]);
        }
        let key_mentions = guaranteed + repeat * possible.len();
        assert_eq!(
            selection_inventory_cardinality(&opportunities).unwrap(),
            (key_mentions, guaranteed)
        );
        let peak = selection_peak_payload_bound(&cases, &opportunities, &facts, 2048).unwrap();
        // Repeated mentions need not allocate another population header.
        // Between Vec growth boundaries the actual retained peak can stay flat.
        assert!(peak >= previous_peak);
        previous_peak = peak;

        // Exercise the production grouping allocator, including capacity growth
        // in its per-key possible and guaranteed index vectors. Its real owned
        // backing plus the complete result must fit the simultaneous bound.
        let groups = member_groups(&opportunities).unwrap();
        assert_eq!(
            groups
                .iter()
                .map(|g| g.possible_case_indices.len())
                .sum::<usize>(),
            key_mentions
        );
        assert_eq!(
            groups
                .iter()
                .map(|g| g.guaranteed_case_indices.len())
                .sum::<usize>(),
            guaranteed
        );
        let group_bytes = groups.capacity()
            * std::mem::size_of::<populations::PopulationMemberGroup>()
            + groups
                .iter()
                .map(|g| {
                    (g.possible_case_indices.capacity() + g.guaranteed_case_indices.capacity())
                        * std::mem::size_of::<usize>()
                })
                .sum::<usize>();
        let selected = select(
            &cases,
            &opportunities,
            &facts,
            &[61],
            8,
            &population,
            2048,
            10_000_000,
            peak,
        )
        .unwrap();
        assert!(group_bytes + selected.retained_payload_bytes().unwrap() <= peak);
        assert!(selected.populations.len() <= guaranteed);
        for (key, _, _) in &possible {
            assert!(selected
                .gaps
                .iter()
                .any(|gap| gap.population.as_ref() == Some(key)
                    && matches!(gap.reason, SelectionGapReason::NoGuaranteedMember)));
            assert!(selected.populations.iter().all(|p| &p.key != key));
        }
    }
}
