//! Input-only opportunity budget. This cannot manufacture measured samples,
//! qualify an unseen provider, or make retries equivalent to successful work.
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    OwnerBlockScheduleV1, StructuredSettingsV2,
};
use populations::{member_groups, CaseOpportunity};

/// A prepared source owns a complete input cycle, unlike an open runtime
/// traffic stream. Freeze its barriers from that cycle before any execution.
/// The longest cut-to-anchor span prevents an early phase from omitting an
/// original representative; the fit member floor covers every accepted rank.
pub(super) fn startup_schedule(
    starts: &[usize],
    ends: &[usize],
    settings: &StructuredSettingsV2,
) -> Result<(OwnerBlockScheduleV1, usize)> {
    let cycle = *ends
        .last()
        .ok_or_else(|| error("empty startup input cycle"))?;
    if starts.len() != ends.len() || cycle == 0 {
        return Err(error("startup input cycle boundaries differ"));
    }
    let mut anchor = 0;
    for index in 0..ends.len() {
        anchor = anchor.max(fresh_span(starts, ends, &[index], 1)?);
    }
    let fit_members = settings
        .max_rank
        .checked_add(settings.min_fit_redundancy)
        .ok_or_else(|| error("startup fit member bound overflow"))?
        .max(settings.min_phase_samples);
    let schedule = OwnerBlockScheduleV1::new(
        cycle,
        [anchor.max(settings.min_phase_samples); 3],
        [
            fit_members,
            settings.min_phase_samples,
            settings.min_phase_samples,
        ],
    )
    .map_err(|reason| error(format!("startup input schedule: {reason:?}")))?;
    // `new` deliberately leaves runtime input-readiness unset. Prepared
    // cohorts retain their original source8 membership and replay contract.
    Ok((schedule, anchor))
}

/// Bound fresh original members of each checked population, preserving the
/// source8 fence across numerical phases. Ambiguous cases never gain a member
/// floor, even when their possible identity also has guaranteed cases.
///
/// This bounds member counts only. The layout must separately put every
/// required work-axis/branch challenge before the earliest phase freeze;
/// extending this total horizon cannot repair a phase frozen without it.
#[cfg(test)]
pub(super) fn plan_checked(
    cases: &[Case],
    prompts: &[usize],
    chunk: usize,
    population: &StructuredServiceDeclarationV7,
    minimum_cycle: usize,
    opportunities: &[CaseOpportunity],
) -> Result<ProbeInputOpportunityBudget> {
    plan_checked_with_schedule(
        cases,
        prompts,
        chunk,
        &population.schedule,
        minimum_cycle,
        opportunities,
    )
}

pub(super) fn plan_checked_with_schedule(
    cases: &[Case],
    prompts: &[usize],
    chunk: usize,
    schedule: &OwnerBlockScheduleV1,
    minimum_cycle: usize,
    opportunities: &[CaseOpportunity],
) -> Result<ProbeInputOpportunityBudget> {
    plan_checked_with_schedule_and_row_ceiling(
        cases,
        prompts,
        chunk,
        None,
        schedule,
        minimum_cycle,
        opportunities,
    )
}

pub(super) fn plan_checked_with_schedule_and_row_ceiling(
    cases: &[Case],
    prompts: &[usize],
    chunk: usize,
    prefill_row_ceiling: Option<NonZeroU32>,
    schedule: &OwnerBlockScheduleV1,
    minimum_cycle: usize,
    opportunities: &[CaseOpportunity],
) -> Result<ProbeInputOpportunityBudget> {
    if opportunities.len() != cases.len() {
        return Err(error(
            "checked probe opportunities differ from original cases",
        ));
    }
    let groups = member_groups(opportunities)
        .map_err(|reason| error(format!("checked probe population inventory: {reason:?}")))?;
    if groups.is_empty() {
        return Err(error("checked probe inventory has no known population"));
    }
    for (index, group) in groups.iter().enumerate() {
        if group.guaranteed_case_indices.is_empty() {
            return Err(error(format!(
                "checked probe population {index} has only possible cases; no fresh member floor"
            )));
        }
    }
    plan_groups(
        cases,
        prompts,
        chunk,
        prefill_row_ceiling,
        schedule,
        minimum_cycle,
        groups
            .iter()
            .map(|group| group.guaranteed_case_indices.as_slice()),
    )
}

pub(super) fn plan(
    cases: &[Case],
    prompts: &[usize],
    chunk: usize,
    population: &StructuredServiceDeclarationV7,
    minimum_cycle: usize,
) -> Result<ProbeInputOpportunityBudget> {
    let mut groups = Vec::new();
    for (index, key) in cases.iter().enumerate() {
        let same = |c: &Case| {
            c.template == key.template
                && c.width == key.width
                && c.preset == key.preset
                && c.product == key.product
        };
        if !cases[..index].iter().any(same) {
            groups.push(
                cases
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| same(c))
                    .map(|(i, _)| i)
                    .collect::<Vec<_>>(),
            );
        }
    }
    plan_groups(
        cases,
        prompts,
        chunk,
        None,
        &population.schedule,
        minimum_cycle,
        groups.iter().map(Vec::as_slice),
    )
}

fn plan_groups<'a>(
    cases: &[Case],
    prompts: &[usize],
    chunk: usize,
    prefill_row_ceiling: Option<NonZeroU32>,
    schedule: &OwnerBlockScheduleV1,
    minimum_cycle: usize,
    groups: impl Iterator<Item = &'a [usize]>,
) -> Result<ProbeInputOpportunityBudget> {
    let mut starts = Vec::with_capacity(cases.len());
    let mut ends = Vec::with_capacity(cases.len());
    let mut cycle_upper = 0usize;
    for case in cases {
        starts.push(cycle_upper);
        let prompt = *prompts
            .get(case.template)
            .ok_or_else(|| error("probe case template is outside prompt inventory"))?;
        cycle_upper = cycle_upper
            .checked_add(
                case.waves_with_row_ceiling(prompt, chunk, prefill_row_ceiling)?
                    .0,
            )
            .ok_or_else(|| error("probe opportunity cycle overflow"))?;
        ends.push(cycle_upper);
    }
    if minimum_cycle == 0 || cycle_upper == 0 {
        return Err(error("empty probe opportunity cycle"));
    }
    let block = schedule.block_offered;
    let mut required = 0usize;
    let mut minimum_opportunities = usize::MAX;
    let mut maximum_discovery = 0usize;
    let mut phase_bounds = [0usize; 3];
    let mut fresh_spans = [0usize; 3];
    for marked in groups {
        minimum_opportunities = minimum_opportunities.min(marked.len());
        let discovery = round_block(ends[marked[0]], block)?;
        maximum_discovery = maximum_discovery.max(discovery);
        let mut total = discovery;
        for phase in 0..3 {
            let span = fresh_span(&starts, &ends, marked, schedule.min_members[phase])?;
            let bound = round_block(span.max(schedule.phase_min_offered[phase]), block)?;
            fresh_spans[phase] = fresh_spans[phase].max(span);
            phase_bounds[phase] = phase_bounds[phase].max(bound);
            total = total
                .checked_add(bound)
                .ok_or_else(|| error("probe phase horizon overflow"))?;
        }
        required = required.max(total);
    }
    let rounds = required.div_ceil(minimum_cycle);
    Ok(ProbeInputOpportunityBudget {
        minimum_input_family_opportunities_per_cycle: minimum_opportunities,
        minimum_original_offers_per_completed_cycle: minimum_cycle,
        // Diagnostic conversions only. They are not separately rounded and
        // added to form the actual plan.
        discovery_cycles: maximum_discovery.div_ceil(minimum_cycle),
        phase_cycles: phase_bounds.map(|n| n.div_ceil(minimum_cycle)),
        planned_cycles: rounds,
        required_original_offers: required,
        successful_cycle_wave_upper_bound: cycle_upper,
        phase_original_offer_bounds: phase_bounds,
        maximum_fresh_member_span: fresh_spans,
    })
}

fn round_block(value: usize, block: usize) -> Result<usize> {
    if block == 0 {
        return Err(error("empty probe block"));
    }
    value
        .div_ceil(block)
        .checked_mul(block)
        .ok_or_else(|| error("probe block horizon overflow"))
}

/// Cut can fall anywhere inside a cohort; skip that entire original cohort.
/// Each marked successor contributes only its first fresh post-release input,
/// conservatively timed at its end. Do not reuse an old cohort across phases.
pub(super) fn fresh_span(
    starts: &[usize],
    ends: &[usize],
    marked: &[usize],
    members: usize,
) -> Result<usize> {
    let cycle = *ends.last().ok_or_else(|| error("empty probe cycle"))?;
    if marked.is_empty() || members == 0 {
        return Err(error("empty probe member population"));
    }
    let mut maximum = 0;
    for (cut, &start) in starts.iter().enumerate() {
        let next = marked.partition_point(|index| *index <= cut);
        let target = next
            .checked_add(members - 1)
            .ok_or_else(|| error("probe member horizon overflow"))?;
        let end = (target / marked.len())
            .checked_mul(cycle)
            .and_then(|n| n.checked_add(ends[marked[target % marked.len()]]))
            .and_then(|n| n.checked_sub(start))
            .ok_or_else(|| error("probe member horizon overflow"))?;
        maximum = maximum.max(end);
    }
    Ok(maximum)
}

#[cfg(test)]
mod tests {
    use super::*;
    mod schedule;
    use ferrum_interfaces::execution_cost::CostProductOutput;
    use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
        OwnerBlockScheduleV1, StructuredPopulationPolicyV1,
    };
    use populations::{classify_alternatives, tests as fixture, CasePopulation};

    fn cases_and_population() -> (Vec<Case>, StructuredServiceDeclarationV7) {
        let cases = [1, 2, 1, 2]
            .into_iter()
            .map(|width| Case {
                product: OpportunityProduct::Greedy,
                template: 0,
                width,
                maximum_output: NonZeroUsize::new(3).unwrap(),
                release_generated: 2,
                suffix_tokens: 1,
                preset: SloAutomaticCostProbeSamplingPresetV1::Configured,
                prefix: PrefixKind::Clean,
                route: CalibrationDecodeRoute::Actual,
                reset: false,
            })
            .collect();
        let mut population =
            population::declaration(&Default::default(), fixture::domain()).unwrap();
        population.schedule = OwnerBlockScheduleV1::new(8, [8; 3], [8; 3]).unwrap();
        (cases, population)
    }

    fn opportunity(
        rows: u32,
        product: CostProductOutput,
        policy: StructuredPopulationPolicyV1,
        floor: usize,
    ) -> CaseOpportunity {
        CaseOpportunity {
            population: classify_alternatives(
                &[fixture::input(rows, 3, product, false, true)],
                policy,
                true,
            )
            .unwrap(),
            minimum_fresh_members: floor,
        }
    }

    #[test]
    fn checked_opportunity_bound_merges_real_widths_and_preserves_phase_fence() {
        let (cases, population) = cases_and_population();
        let opportunities = cases
            .iter()
            .map(|case| {
                opportunity(
                    case.width as u32,
                    CostProductOutput::GreedyToken,
                    StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
                    1,
                )
            })
            .collect::<Vec<_>>();
        // Prompt one fits in each row's frozen share of the eight-token
        // capacity. Every prepared cohort performs one prefill per row and
        // then all original decode waves; all four requests reach Length.
        let waves: Vec<_> = cases
            .iter()
            .map(|case| case.width + case.maximum_output.get() - 1)
            .collect();
        let minimum_cycle: usize = waves.iter().sum();
        for (case, &expected) in cases.iter().zip(&waves) {
            assert_eq!(case.waves(1, 8).unwrap().0, expected);
        }
        let bound =
            plan_checked(&cases, &[1], 8, &population, minimum_cycle, &opportunities).unwrap();
        assert_eq!(
            bound.minimum_input_family_opportunities_per_cycle,
            cases.len()
        );

        // Independently walk the actual repeated cohort timeline at every
        // original offer cut. A cohort already started at that cut cannot
        // contribute a fresh member to the next numerical phase.
        let assert_fresh_fence = |bound: &ProbeInputOpportunityBudget, marked: &[usize]| {
            for phase in 0..3 {
                let members = population.schedule.min_members[phase];
                let expected_span = (0..minimum_cycle)
                    .map(|cut| {
                        let mut offered = 0;
                        let mut fresh = 0;
                        for index in (0..cases.len()).cycle() {
                            let start = offered;
                            offered += waves[index];
                            if start > cut && marked.contains(&index) {
                                fresh += 1;
                                if fresh == members {
                                    return offered - cut;
                                }
                            }
                        }
                        unreachable!()
                    })
                    .max()
                    .unwrap();
                assert_eq!(bound.maximum_fresh_member_span[phase], expected_span);
                let block = population.schedule.block_offered;
                let offers = bound.phase_original_offer_bounds[phase];
                assert_eq!(offers % block, 0);
                assert!(offers >= expected_span.max(population.schedule.phase_min_offered[phase]));
                assert!(
                    offers - block
                        < expected_span.max(population.schedule.phase_min_offered[phase])
                );
            }
            let discovery = waves[marked[0]].div_ceil(population.schedule.block_offered)
                * population.schedule.block_offered;
            assert_eq!(
                bound.required_original_offers,
                discovery + bound.phase_original_offer_bounds.iter().sum::<usize>()
            );
            assert_eq!(bound.successful_cycle_wave_upper_bound, minimum_cycle);
            assert!(bound.planned_cycles * minimum_cycle >= bound.required_original_offers);
            assert!((bound.planned_cycles - 1) * minimum_cycle < bound.required_original_offers);
        };
        assert_fresh_fence(&bound, &[0, 1, 2, 3]);

        // Reachability, not identity alone, supplies a member. Both unmarked
        // cohorts still consume the complete original offer timeline.
        let mut sparse = opportunities.clone();
        sparse[1].minimum_fresh_members = 0;
        sparse[3].minimum_fresh_members = 0;
        let sparse_bound =
            plan_checked(&cases, &[1], 8, &population, minimum_cycle, &sparse).unwrap();
        assert_eq!(sparse_bound.minimum_input_family_opportunities_per_cycle, 2);
        assert_fresh_fence(&sparse_bound, &[0, 2]);
        assert!(sparse_bound.required_original_offers > bound.required_original_offers);
    }

    #[test]
    fn checked_opportunity_bound_preserves_exact_population_rounding() {
        let (cases, population) = cases_and_population();
        let opportunities = cases
            .iter()
            .map(|case| {
                opportunity(
                    case.width as u32,
                    CostProductOutput::GreedyToken,
                    StructuredPopulationPolicyV1::ExactOwnerV1,
                    1,
                )
            })
            .collect::<Vec<_>>();
        let original = plan(&cases, &[1], 8, &population, 12).unwrap();
        let checked = plan_checked(&cases, &[1], 8, &population, 12, &opportunities).unwrap();
        assert_eq!(
            serde_json::to_value(original).unwrap(),
            serde_json::to_value(checked).unwrap()
        );
    }

    #[test]
    fn checked_opportunity_bound_rejects_possible_only_population_and_invalid_inventory() {
        let (cases, population) = cases_and_population();
        let known = opportunity(
            1,
            CostProductOutput::GreedyToken,
            StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
            1,
        );
        let mut opportunities = vec![known; cases.len()];
        opportunities[1] = opportunity(
            2,
            CostProductOutput::FullLogits,
            StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
            0,
        );
        let failure = plan_checked(&cases, &[1], 8, &population, 12, &opportunities)
            .err()
            .unwrap()
            .to_string();
        assert!(failure.contains("no fresh member floor"));
        assert!(plan_checked(&cases, &[1], 8, &population, 12, &opportunities[..3]).is_err());
        opportunities[1] = opportunities[0].clone();
        opportunities[1].minimum_fresh_members = 2;
        assert!(plan_checked(&cases, &[1], 8, &population, 12, &opportunities).is_err());
        for opportunity in &mut opportunities {
            opportunity.population = CasePopulation::Unknown {
                known_alternatives: Vec::new(),
            };
            opportunity.minimum_fresh_members = 0;
        }
        assert!(plan_checked(&cases, &[1], 8, &population, 12, &opportunities).is_err());
    }

    #[test]
    fn opportunity_bound_skips_cut_cohort_and_wraps_original_cycle() {
        let starts = [0, 2, 5, 7];
        let ends = [2, 5, 7, 10];
        assert_eq!(fresh_span(&starts, &ends, &[0, 2], 1).unwrap(), 7);
        assert_eq!(fresh_span(&starts, &ends, &[0, 2], 2).unwrap(), 12);
        assert_eq!(fresh_span(&starts, &ends, &[0, 2], 8).unwrap(), 42);
        assert!(fresh_span(&starts, &ends, &[], 8).is_err());
    }
    #[test]
    fn opportunity_bound_retains_original_complete_block_rounding() {
        assert_eq!(round_block(255, 256).unwrap(), 256);
        assert_eq!(round_block(256, 256).unwrap(), 256);
        assert_eq!(round_block(257, 256).unwrap(), 512);
        assert!(round_block(1, 0).is_err());
        assert!(round_block(usize::MAX, 256).is_err());
    }
}
