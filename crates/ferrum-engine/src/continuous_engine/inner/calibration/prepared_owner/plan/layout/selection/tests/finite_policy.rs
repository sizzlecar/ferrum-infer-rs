use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    NonNegativePlanningEstimatorV1 as Estimator, OwnerAlgorithmUniversePolicyV1,
    OwnerInputReadinessV1, OwnerOpeningFrontierPolicyV1, OwnerPhaseSupportPolicyV1,
    StructuredServiceDomainPolicyV1 as Domain,
};
use std::num::NonZeroU64;

fn short_inventory() -> (
    Vec<Case>,
    Vec<CaseOpportunity>,
    Vec<Vec<CheckedInputFacts>>,
    StructuredServiceDeclarationV7,
) {
    let (mut cases, _, _, population) = inventory();
    cases.truncate(3);
    let mut opportunities = Vec::new();
    let mut facts = Vec::new();
    for case in &mut cases {
        case.maximum_output = NonZeroUsize::new(4).unwrap();
        case.suffix_tokens = 1;
        let input = natural_termination_input_with_algorithm_and_eos(
            case.width as u32,
            CostProductOutput::GreedyToken,
            true,
            64,
            "fixture.finite.policy",
            [67; 32],
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
    }
    (cases, opportunities, facts, population)
}

fn product_capacity() -> SelectionCapacity {
    SelectionCapacity {
        requests: 2048,
        execution_actions: 16384,
        declared_offer_rows: 16384,
    }
}

/// Allocation may change occurrence order and phase barriers, but never the
/// original checked geometry, representative obligations or numerical scope.
pub(super) fn assert_preserves_coverage(original: &CheckedSelection, proposed: &CheckedSelection) {
    assert_eq!(original.populations.len(), proposed.populations.len());
    assert_eq!(
        serde_json::to_value(&original.input_geometry).unwrap(),
        serde_json::to_value(&proposed.input_geometry).unwrap()
    );
    for old in &original.populations {
        let new = proposed
            .populations
            .iter()
            .find(|new| new.key == old.key)
            .unwrap();
        assert_eq!(
            new.representative_case_indices,
            old.representative_case_indices
        );
        assert_eq!(
            serde_json::to_value(&new.input_geometry).unwrap(),
            serde_json::to_value(&old.input_geometry).unwrap()
        );
        if old.scheduled {
            assert!(new.scheduled, "old admitted population lost: {:?}", old.key);
            let old_batch = &original.batches[old.batch_index.unwrap()];
            let new_batch = &proposed.batches[new.batch_index.unwrap()];
            assert_eq!(new_batch.algorithm_universe, old_batch.algorithm_universe);
            assert_eq!(
                new_batch.schedule.min_members,
                old_batch.schedule.min_members
            );
            for index in &old.representative_case_indices {
                let before = old_batch
                    .representative_case_indices
                    .iter()
                    .position(|i| i == index)
                    .unwrap();
                let after = new_batch
                    .representative_case_indices
                    .iter()
                    .position(|i| i == index)
                    .unwrap();
                assert_eq!(
                    serde_json::to_value(
                        old_batch.scoped_opportunities.as_ref().map(|v| &v[before])
                    )
                    .unwrap(),
                    serde_json::to_value(
                        new_batch.scoped_opportunities.as_ref().map(|v| &v[after])
                    )
                    .unwrap()
                );
            }
        }
    }
}

pub(super) fn assert_work(
    selection: &CheckedSelection,
    cases: &[Case],
    prompts: &[usize],
    chunk: usize,
) {
    let mut execution = Vec::new();
    let mut total = SelectionCapacity::default();
    for batch in &selection.batches {
        let setup = work::setup_for_indices(cases, &batch.representative_case_indices).unwrap();
        let mut actual = [
            setup.requests,
            setup.execution_actions,
            setup.serial_declared_offer_rows,
            setup.serial_token_work,
        ];
        for index in batch.execution_case_indices().unwrap() {
            let item = work::case_work(&cases[index], prompts[cases[index].template], chunk, None)
                .unwrap();
            for (sum, value) in actual.iter_mut().zip([
                item.requests,
                item.execution_actions,
                item.serial_declared_offer_rows,
                item.serial_token_work,
            ]) {
                *sum += value;
            }
            if batch.scheduled {
                execution.push(index);
            }
        }
        assert_eq!(
            actual,
            [
                batch.requests,
                batch.serial_wave_upper_bound,
                batch.declared_offer_row_bound,
                batch.serial_token_work
            ]
        );
        if batch.scheduled {
            related::assert_complete_input_plan(batch);
            total.charge(batch).unwrap();
        }
    }
    assert_eq!(execution, selection.execution_case_indices);
    assert_eq!(
        total,
        SelectionCapacity {
            requests: selection.requests,
            execution_actions: selection.serial_wave_upper_bound,
            declared_offer_rows: selection.declared_offer_row_bound
        }
    );
}

#[test]
fn finite_policy_extra_heap_keeps_original_admitted_geometry_and_work() {
    let (cases, opportunities, facts, population) = short_inventory();
    let groups = member_groups(&opportunities).unwrap();
    let capacity = product_capacity();
    let memory = super::super::memory::plan(
        &groups,
        &opportunities,
        &facts,
        capacity.requests,
        Some(&population.settings),
    )
    .unwrap();
    let run = |bytes, geometry: &mut StructuredInputGeometryWorkV1| {
        select_with_capacity(
            &cases,
            &opportunities,
            &facts,
            &[61],
            8,
            None,
            &population,
            capacity,
            bytes,
            None,
            None,
            Some(geometry),
            NonZeroUsize::new(4),
            None,
        )
    };
    let limit = NonZeroU64::new(32_000_000).unwrap();
    let mut tight_work = StructuredInputGeometryWorkV1::new(limit);
    let tight = run(memory.required_peak_bytes, &mut tight_work).unwrap();
    assert!(tight.batches.iter().any(|b| b.scheduled));
    assert!(tight.batches.iter().all(|b| b.finite_plan().is_none()));
    assert!(tight
        .populations
        .iter()
        .all(|p| p.scheduled && p.input_geometry.as_ref().unwrap().complete));
    assert!(
        memory.retained_groups_bytes + tight.retained_payload_bytes().unwrap()
            <= memory.required_peak_bytes
    );
    let finite_peak = memory.required_peak_bytes
        + super::super::memory::finite_extra_peak(&groups, capacity.requests).unwrap();
    let mut generous_work = StructuredInputGeometryWorkV1::new(limit);
    let generous = run(finite_peak, &mut generous_work).unwrap();
    assert!(
        generous
            .batches
            .iter()
            .any(|b| b.scheduled && b.finite_plan().is_some()),
        "supported control must actually exercise finite allocation"
    );
    assert_preserves_coverage(&tight, &generous);
    assert_eq!(
        serde_json::to_value(&tight.gaps).unwrap(),
        serde_json::to_value(&generous.gaps).unwrap()
    );
    assert_eq!(tight_work.visits(), generous_work.visits());
    assert_work(&tight, &cases, &[61], 8);
    assert_work(&generous, &cases, &[61], 8);
    assert!(generous.requests <= tight.requests);
    assert!(generous.serial_wave_upper_bound <= tight.serial_wave_upper_bound);
    assert!(generous.declared_offer_row_bound <= tight.declared_offer_row_bound);
    // Refusing the extra finite storage must retain the original admitted
    // geometry and execution work, rather than fail the complete selection.
    let mut below_work = StructuredInputGeometryWorkV1::new(limit);
    let below = run(finite_peak - 1, &mut below_work).unwrap();
    assert!(below.batches.iter().all(|b| b.finite_plan().is_none()));
    assert_preserves_coverage(&tight, &below);
    assert_work(&below, &cases, &[61], 8);
    assert_eq!(below.requests, tight.requests);
    assert_eq!(below.serial_wave_upper_bound, tight.serial_wave_upper_bound);
    assert_eq!(
        below.declared_offer_row_bound,
        tight.declared_offer_row_bound
    );
    assert_eq!(below_work.visits(), tight_work.visits());
    let mut denied_work = StructuredInputGeometryWorkV1::new(limit);
    assert!(run(memory.required_peak_bytes - 1, &mut denied_work).is_err());
    assert_eq!(denied_work.visits(), 0);
}

#[test]
fn finite_policy_preserves_periodic_for_filtered_membership_contracts() {
    let (cases, _, facts, original) = short_inventory();
    for policy in 0..6 {
        let mut population = original.clone();
        match policy {
            0 => {
                population.domain_policy = Domain::FrozenFitSupportV1;
                population.nonnegative_envelope = None;
            }
            1 => {
                population
                    .nonnegative_envelope
                    .as_mut()
                    .unwrap()
                    .planning_estimator = Estimator::IdentifiedFitJointCellsV1
            }
            2 => {
                population.schedule.phase_support =
                    Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV2)
            }
            3 => {
                population.schedule.input_readiness =
                    Some(OwnerInputReadinessV1::new([1; 3], 32_000_000).unwrap())
            }
            4 => {
                population.schedule.opening_frontier =
                    Some(OwnerOpeningFrontierPolicyV1::FirstOfferFifoV1)
            }
            5 => {
                population.schedule.algorithm_universe =
                    Some(OwnerAlgorithmUniversePolicyV1::FirstOrdinaryDiscoveryBlockSubsetV1)
            }
            _ => unreachable!(),
        }
        // Reclassify the same actual inputs if this domain requires exact
        // owners. These are selector compatibility controls, not permission
        // to serialize Source7-only filters into a Source8 header.
        let opportunities = facts
            .iter()
            .map(|row| CaseOpportunity {
                population: classify_alternatives(
                    std::slice::from_ref(row[0].original.as_ref().unwrap().as_ref()),
                    population.population_policy(),
                    true,
                )
                .unwrap(),
                minimum_fresh_members: 1,
            })
            .collect::<Vec<_>>();
        let selected = select_with_capacity(
            &cases,
            &opportunities,
            &facts,
            &[61],
            8,
            None,
            &population,
            product_capacity(),
            usize::MAX,
            None,
            None,
            None,
            NonZeroUsize::new(4),
            None,
        )
        .unwrap();
        assert!(
            selected.batches.iter().any(|b| b.scheduled),
            "policy {policy}"
        );
        assert!(
            selected
                .batches
                .iter()
                .all(|b| b.periodic_budget().is_some()),
            "policy {policy}"
        );
        assert_work(&selected, &cases, &[61], 8);
    }
}

#[test]
fn finite_policy_newly_affordable_earlier_source_cannot_evict_original_reservation() {
    let (mut cases, mut opportunities, mut facts, population) = short_inventory();
    // Configured Actual Greedy precedes the auxiliary FullLogits source.
    // Their distinct products prevent coalescing and retain exact identities.
    let mut full = cases[1].clone();
    full.template = 1;
    full.product = OpportunityProduct::Full;
    full.route = CalibrationDecodeRoute::FullLogits;
    let input = natural_termination_input_with_algorithm_and_eos(
        2,
        CostProductOutput::FullLogits,
        true,
        64,
        "fixture.finite.later",
        [68; 32],
        false,
    );
    cases.push(full);
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
    let groups = member_groups(&opportunities).unwrap();
    let original_memory =
        super::super::memory::plan(&groups, &opportunities, &facts, 2048, None).unwrap();
    let roomy_periodic = select_with_capacity(
        &cases,
        &opportunities,
        &facts,
        &[61, 61],
        8,
        None,
        &population,
        product_capacity(),
        original_memory.required_peak_bytes,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(roomy_periodic.batches.len(), 2);
    let first = &roomy_periodic.batches[0];
    let later = &roomy_periodic.batches[1];
    assert!(first
        .representative_case_indices
        .iter()
        .all(|&i| cases[i].product == OpportunityProduct::Greedy));
    assert_eq!(later.representative_case_indices, [3]);
    assert!(
        first.requests > later.requests,
        "fixture must skip the original earlier source"
    );
    let capacity = SelectionCapacity {
        requests: later.requests,
        ..product_capacity()
    };
    let peak = super::super::memory::plan(&groups, &opportunities, &facts, capacity.requests, None)
        .unwrap()
        .required_peak_bytes;
    let run = |bytes| {
        select_with_capacity(
            &cases,
            &opportunities,
            &facts,
            &[61, 61],
            8,
            None,
            &population,
            capacity,
            bytes,
            None,
            None,
            None,
            NonZeroUsize::new(1),
            None,
        )
        .unwrap()
    };
    let original = run(peak);
    assert!(!original.batches[0].scheduled && original.batches[1].scheduled);
    let finite = run(usize::MAX);
    let new_earlier = finite
        .batches
        .iter()
        .find(|b| b.representative_case_indices == first.representative_case_indices)
        .unwrap();
    assert!(
        new_earlier.finite_plan().is_some() && new_earlier.requests <= capacity.requests,
        "counterexample must make the earlier source individually affordable"
    );
    assert_preserves_coverage(&original, &finite);
    let selected = finite
        .batches
        .iter()
        .filter(|b| b.scheduled)
        .collect::<Vec<_>>();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].representative_case_indices, [3]);
    assert_work(&original, &cases, &[61, 61], 8);
    assert_work(&finite, &cases, &[61, 61], 8);
}
