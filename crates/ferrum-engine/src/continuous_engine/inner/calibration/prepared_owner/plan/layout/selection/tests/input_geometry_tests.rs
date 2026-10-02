use super::*;
use std::num::NonZeroU64;

fn cross_inventory() -> (
    Vec<Case>,
    Vec<CaseOpportunity>,
    Vec<Vec<CheckedInputFacts>>,
    StructuredServiceDeclarationV7,
) {
    let mut cases = Vec::new();
    let mut opportunities = Vec::new();
    let mut facts = Vec::new();
    let population = population::declaration(&Default::default(), fixture::domain()).unwrap();
    // Actual canonical row/command evidence supplies the axes. The third
    // original request varies KV independently of B; no axis is overwritten.
    for (width, kv, template) in [(1, 16, 0), (8, 64, 1), (1, 64, 1)] {
        let input =
            fixture::input_with_context(width, 3, kv, CostProductOutput::GreedyToken, false, true);
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
        cases.push(Case {
            product: OpportunityProduct::Greedy,
            template,
            width: width as usize,
            maximum_output: NonZeroUsize::new(21).unwrap(),
            release_generated: 3,
            suffix_tokens: 18,
            preset: SloAutomaticCostProbeSamplingPresetV1::GreedyLength,
            prefix: PrefixKind::Clean,
            route: CalibrationDecodeRoute::Actual,
            reset: false,
            acquisition: None,
        });
    }
    (cases, opportunities, facts, population)
}

fn select_geometry(
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    facts: &[Vec<CheckedInputFacts>],
    population: &StructuredServiceDeclarationV7,
    requests: usize,
    waves: usize,
    work: &mut StructuredInputGeometryWorkV1,
) -> CheckedSelection {
    select_changed_with_geometry(
        cases,
        opportunities,
        facts,
        &[13, 61],
        8,
        population,
        requests,
        waves,
        usize::MAX,
        None,
        None,
        Some(work),
    )
    .unwrap()
}

#[test]
fn checked_input_pivots_add_original_cross_case_and_charge_complete_source() {
    let (cases, opportunities, facts, population) = cross_inventory();
    let original_cases = serde_json::to_value(&cases).unwrap();
    let endpoints = representatives(&[0, 1, 2], &facts, &cases, &[13, 61], 8).unwrap();
    assert_eq!(
        endpoints,
        [0, 1],
        "both ends of every axis still miss the cross direction"
    );
    let mut work = StructuredInputGeometryWorkV1::new(NonZeroU64::new(32_000_000).unwrap());
    let result = select_geometry(
        &cases,
        &opportunities,
        &facts,
        &population,
        2048,
        65_536,
        &mut work,
    );
    assert_eq!(result.populations.len(), 1);
    let group = &result.populations[0];
    assert!(group.scheduled);
    assert_eq!(group.representative_case_indices, [0, 1, 2]);
    let geometry = group.input_geometry.as_ref().unwrap();
    assert!(geometry.complete);
    assert_eq!(geometry.candidate_rank, Some(3));
    assert_eq!(geometry.selected_rank, geometry.candidate_rank);
    assert_eq!(geometry.added_original_cases, 1);
    assert!(geometry.candidate_rank.unwrap() < facts[0][0].axes.len());
    let batch = &result.batches[group.batch_index.unwrap()];
    assert!(batch.schedule.input_readiness.is_none());
    assert_eq!(
        result.execution_case_indices,
        group
            .representative_case_indices
            .repeat(batch.planned_cycles)
    );
    assert_eq!(
        batch.requests,
        batch.planned_cycles * cases.iter().map(|case| case.width).sum::<usize>()
    );
    let cycle_setup_and_decode: usize = cases
        .iter()
        .map(|case| {
            let prompt = [13usize, 61][case.template];
            case.width * prompt.div_ceil(8 / case.width) + case.maximum_output.get() - 1
        })
        .sum();
    assert_eq!(
        batch.input_opportunities.successful_cycle_wave_upper_bound,
        cycle_setup_and_decode
    );
    assert!(
        batch.planned_cycles
            * batch
                .input_opportunities
                .minimum_original_offers_per_completed_cycle
            >= batch.input_opportunities.required_original_offers + batch.schedule.block_offered
    );
    assert_eq!(serde_json::to_value(&cases).unwrap(), original_cases);
    let groups = member_groups(&opportunities).unwrap();
    let peak = super::super::memory::plan(
        &groups,
        &opportunities,
        &facts,
        2048,
        Some(&population.settings),
    )
    .unwrap()
    .required_peak_bytes;
    let mut memory_work = StructuredInputGeometryWorkV1::new(NonZeroU64::new(32_000_000).unwrap());
    assert!(select_changed_with_geometry(
        &cases,
        &opportunities,
        &facts,
        &[13, 61],
        8,
        &population,
        2048,
        65_536,
        peak - 1,
        None,
        None,
        Some(&mut memory_work)
    )
    .is_err());
    assert_eq!(memory_work.visits(), 0);
    let exact_memory = select_changed_with_geometry(
        &cases,
        &opportunities,
        &facts,
        &[13, 61],
        8,
        &population,
        2048,
        65_536,
        peak,
        None,
        None,
        Some(&mut memory_work),
    )
    .unwrap();
    assert_eq!(
        exact_memory.execution_case_indices,
        result.execution_case_indices
    );

    for (requests, waves) in [
        (batch.requests - 1, 65_536),
        (2048, batch.serial_wave_upper_bound - 1),
    ] {
        let mut work = StructuredInputGeometryWorkV1::new(NonZeroU64::new(32_000_000).unwrap());
        let tight = select_geometry(
            &cases,
            &opportunities,
            &facts,
            &population,
            requests,
            waves,
            &mut work,
        );
        assert_eq!(
            tight.populations[0].representative_case_indices,
            group.representative_case_indices
        );
        assert!(!tight.populations[0].scheduled);
        assert!(tight.execution_case_indices.is_empty());
        assert!(tight.gaps.iter().any(|gap| matches!(
            gap.reason,
            SelectionGapReason::RemainingRequests { .. }
                | SelectionGapReason::RemainingWaves { .. }
        )));
    }
}

#[test]
fn checked_input_pivots_share_work_across_selection_calls_and_preserve_bounded_gaps() {
    use crate::continuous_engine::inner::calibration::cohort_driver::ProbeExecutionBudget;
    let (cases, opportunities, facts, population) = cross_inventory();
    let mut deferred_work =
        StructuredInputGeometryWorkV1::new(NonZeroU64::new(32_000_000).unwrap());
    let deferred = select_changed_with_geometry(
        &cases,
        &opportunities,
        &facts,
        &[13, 61],
        8,
        &population,
        2048,
        65_536,
        usize::MAX,
        None,
        Some(1),
        Some(&mut deferred_work),
    )
    .unwrap();
    assert!(deferred.execution_case_indices.is_empty());
    assert!(deferred.populations[0].input_geometry.is_none());
    assert_eq!(
        deferred_work.visits(),
        0,
        "deferred priorities cannot spend this round's geometry allowance"
    );
    let mut measured = StructuredInputGeometryWorkV1::new(NonZeroU64::new(32_000_000).unwrap());
    let full = select_geometry(
        &cases,
        &opportunities,
        &facts,
        &population,
        2048,
        65_536,
        &mut measured,
    );
    assert!(
        full.populations[0]
            .input_geometry
            .as_ref()
            .unwrap()
            .complete
    );
    let limit = NonZeroU64::new(measured.visits()).unwrap();
    let mut budget = ProbeExecutionBudget::new(
        tokio::time::Instant::now() + std::time::Duration::from_secs(60),
        NonZeroUsize::new(2048).unwrap(),
        NonZeroUsize::new(65_536).unwrap(),
    );
    let first = select_geometry(
        &cases,
        &opportunities,
        &facts,
        &population,
        2048,
        65_536,
        budget.input_geometry_work(Some(limit)).unwrap().unwrap(),
    );
    assert!(
        first.populations[0]
            .input_geometry
            .as_ref()
            .unwrap()
            .complete
    );
    let second = select_geometry(
        &cases,
        &opportunities,
        &facts,
        &population,
        2048,
        65_536,
        budget.input_geometry_work(Some(limit)).unwrap().unwrap(),
    );
    assert!(second.input_geometry.as_ref().unwrap().exhausted);
    assert_eq!(
        second.input_geometry.as_ref().unwrap().cumulative_visits,
        limit.get()
    );
    assert_eq!(second.populations[0].representative_case_indices, [0, 1]);
    assert!(
        !second.populations[0]
            .input_geometry
            .as_ref()
            .unwrap()
            .complete
    );
    assert!(second.gaps.iter().any(|gap| matches!(
        gap.reason,
        SelectionGapReason::InputGeometryUnavailable {
            work_exhausted: true,
            ..
        }
    )));
    assert_eq!(budget.requests_remaining(), 2048);
    assert_eq!(budget.selection_requests_remaining(), 2048);
    assert!(budget
        .input_geometry_work(NonZeroU64::new(limit.get() + 1))
        .is_err());
}
