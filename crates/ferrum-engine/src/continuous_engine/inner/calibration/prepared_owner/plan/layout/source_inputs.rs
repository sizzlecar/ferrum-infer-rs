//! The original representatives and member floors survive source preparation.
//! A cold fallback changes work and barriers, never its selected cohort members.
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::OwnerBlockScheduleV1;
use populations::{CaseOpportunity, CasePopulation, CheckedPopulationKey, PopulationMemberGroup};

pub(in crate::continuous_engine::inner::calibration) struct PreparedProbeSourceInputs {
    cases: Vec<Case>,
    opportunities: Vec<CaseOpportunity>,
    prompts: Vec<usize>,
    original_cycles: usize,
    collection_actions: usize,
    declared_offer_rows: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct ColdSourcePlan {
    pub schedule: OwnerBlockScheduleV1,
    pub input_opportunities: ProbeInputOpportunityBudget,
    pub maximum_anchor_span: usize,
    pub cycles: usize,
    pub requests: usize,
    pub execution_actions: usize,
    pub declared_offer_row_bound: usize,
    pub serial_token_work: usize,
    pub original_collection_actions: usize,
    pub original_declared_offer_row_bound: usize,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration) enum ColdSourceSkip {
    ScheduleCapacity,
    OriginalCyclesInsufficient {
        available_offers: usize,
        required_offers: usize,
    },
    RetainedCapacity,
}

#[derive(Debug)]
pub(in crate::continuous_engine::inner::calibration) enum ColdSourceRebuild {
    Ready(ColdSourcePlan),
    Skip(ColdSourceSkip),
}

fn opportunity_heap_bytes(opportunity: &CaseOpportunity) -> Option<usize> {
    let capacity = match &opportunity.population {
        CasePopulation::Unique(_) => 0,
        CasePopulation::Alternatives(keys) => keys.capacity(),
        CasePopulation::Unknown { known_alternatives } => known_alternatives.capacity(),
    };
    capacity.checked_mul(std::mem::size_of::<CheckedPopulationKey>())
}

pub(in crate::continuous_engine::inner::calibration::prepared_owner::plan) fn retained_sources_bytes(
    sources: &Vec<PreparedProbeSourceInputs>,
) -> Option<usize> {
    sources.iter().try_fold(
        sources
            .capacity()
            .checked_mul(std::mem::size_of::<PreparedProbeSourceInputs>())?,
        |bytes, source| bytes.checked_add(source.retained_heap_bytes()?),
    )
}

/// The caller subtracts the still-live inventory, cases, selection and input
/// payload first. Authorize the entire new representation before cloning it.
pub(super) fn freeze_sources(
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    prompts: &[usize],
    selection: &selection::CheckedSelection,
    chunk: NonZeroU32,
    row_ceiling: Option<NonZeroU32>,
    maximum_bytes: usize,
) -> Result<Vec<PreparedProbeSourceInputs>> {
    let source_count = selection.batches.iter().filter(|b| b.scheduled).count();
    let mut bound = mul(
        source_count,
        std::mem::size_of::<PreparedProbeSourceInputs>(),
    )?;
    for batch in selection.batches.iter().filter(|b| b.scheduled) {
        if batch.planned_cycles == 0 || batch.representative_case_indices.is_empty() {
            return Err(error("source input reservation has no original cycle"));
        }
        bound = add(
            bound,
            mul(
                batch.representative_case_indices.len(),
                std::mem::size_of::<Case>()
                    + std::mem::size_of::<CaseOpportunity>()
                    + std::mem::size_of::<usize>(),
            )?,
        )?;
        for &index in &batch.representative_case_indices {
            let opportunity = opportunities
                .get(index)
                .ok_or_else(|| error("source opportunity outside original inventory"))?;
            bound = add(
                bound,
                opportunity_heap_bytes(opportunity)
                    .ok_or_else(|| error("source opportunity capacity overflow"))?,
            )?;
            let case = cases
                .get(index)
                .ok_or_else(|| error("source case outside original inventory"))?;
            prompts
                .get(case.template)
                .ok_or_else(|| error("source prompt outside original inputs"))?;
        }
    }
    if bound > maximum_bytes {
        return Err(error(
            "source preparation inputs exceed shared retained capacity",
        ));
    }
    let mut sources = Vec::new();
    sources
        .try_reserve_exact(source_count)
        .map_err(|_| error("source preparation inputs allocation failed"))?;
    for batch in selection.batches.iter().filter(|b| b.scheduled) {
        let mut source = PreparedProbeSourceInputs {
            cases: Vec::new(),
            opportunities: Vec::new(),
            prompts: Vec::new(),
            original_cycles: batch.planned_cycles,
            collection_actions: 0,
            declared_offer_rows: 0,
        };
        let count = batch.representative_case_indices.len();
        source
            .cases
            .try_reserve_exact(count)
            .map_err(|_| error("source case allocation failed"))?;
        source
            .opportunities
            .try_reserve_exact(count)
            .map_err(|_| error("source opportunity allocation failed"))?;
        source
            .prompts
            .try_reserve_exact(count)
            .map_err(|_| error("source prompt allocation failed"))?;
        for &index in &batch.representative_case_indices {
            let case = &cases[index];
            let prompt = prompts[case.template];
            let work = work::case_work(case, prompt, chunk.get() as usize, row_ceiling)?;
            source.collection_actions = add(source.collection_actions, work.execution_actions)?;
            source.declared_offer_rows =
                add(source.declared_offer_rows, work.serial_declared_offer_rows)?;
            source.cases.push(case.clone());
            source.opportunities.push(opportunities[index].clone());
            source.prompts.push(prompt);
        }
        source.collection_actions = mul(source.collection_actions, source.original_cycles)?;
        source.declared_offer_rows = mul(source.declared_offer_rows, source.original_cycles)?;
        sources.push(source);
    }
    if retained_sources_bytes(&sources).is_none_or(|n| n > maximum_bytes) {
        return Err(error(
            "source preparation inputs retained capacity exhausted",
        ));
    }
    Ok(sources)
}

impl PreparedProbeSourceInputs {
    fn retained_heap_bytes(&self) -> Option<usize> {
        self.opportunities.iter().try_fold(
            self.cases
                .capacity()
                .checked_mul(std::mem::size_of::<Case>())?
                .checked_add(
                    self.opportunities
                        .capacity()
                        .checked_mul(std::mem::size_of::<CaseOpportunity>())?,
                )?
                .checked_add(
                    self.prompts
                        .capacity()
                        .checked_mul(std::mem::size_of::<usize>())?,
                )?,
            |bytes, opportunity| bytes.checked_add(opportunity_heap_bytes(opportunity)?),
        )
    }

    pub fn cold_plan(
        &self,
        population: &StructuredServiceDeclarationV7,
        chunk: NonZeroU32,
        row_ceiling: Option<NonZeroU32>,
        maximum_temporary_bytes: usize,
    ) -> Result<ColdSourceRebuild> {
        // budget::member_groups uses growable groups and two index vectors per
        // group. Bound all possible key mentions, including ambiguous cases;
        // Vec growth and replacement buffers use the existing 4x/8-slot bound.
        let (mentions, guaranteed) = self.opportunities.iter().try_fold(
            (0usize, 0usize),
            |(count, guaranteed), opportunity| {
                let keys = match &opportunity.population {
                    CasePopulation::Unique(_) => 1,
                    CasePopulation::Alternatives(keys) => keys.len(),
                    CasePopulation::Unknown { known_alternatives } => known_alternatives.len(),
                };
                Ok::<_, FerrumError>((
                    add(count, keys)?,
                    add(
                        guaranteed,
                        usize::from(
                            opportunity.minimum_fresh_members == 1
                                && matches!(opportunity.population, CasePopulation::Unique(_)),
                        ),
                    )?,
                ))
            },
        )?;
        let workspace = add(
            add(
                mul(self.cases.len(), std::mem::size_of::<Case>())?,
                mul(mul(self.cases.len(), 2)?, std::mem::size_of::<usize>())?,
            )?,
            add(
                mul(
                    mul(add(mentions, 8)?, 4)?,
                    std::mem::size_of::<PopulationMemberGroup>(),
                )?,
                add(
                    std::mem::size_of::<Vec<PopulationMemberGroup>>(),
                    mul(
                        mul(add(add(mul(mentions, 18)?, mul(guaranteed, 2)?)?, 8)?, 4)?,
                        std::mem::size_of::<usize>(),
                    )?,
                )?,
            )?,
        )?;
        if workspace > maximum_temporary_bytes {
            return Ok(ColdSourceRebuild::Skip(ColdSourceSkip::RetainedCapacity));
        }
        let mut cases = self.cases.clone();
        let mut starts = Vec::with_capacity(cases.len());
        let mut ends = Vec::with_capacity(cases.len());
        let (mut upper, mut minimum, mut requests, mut actions, mut rows, mut tokens) =
            (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        for (index, case) in cases.iter_mut().enumerate() {
            case.acquisition = None;
            // Only this temporary work view uses local prompt indices. The
            // original case, identity and request factory remain unchanged.
            case.template = index;
            let work =
                work::case_work(case, self.prompts[index], chunk.get() as usize, row_ceiling)?;
            starts.push(upper);
            upper = add(upper, work.declared_offers_upper)?;
            ends.push(upper);
            minimum = add(minimum, work.declared_offers_minimum)?;
            requests = add(requests, work.requests)?;
            actions = add(actions, work.execution_actions)?;
            rows = add(rows, work.serial_declared_offer_rows)?;
            tokens = add(tokens, work.serial_token_work)?;
        }
        let (mut schedule, maximum_anchor_span) =
            budget::startup_schedule(&starts, &ends, &population.settings)?;
        drop(starts);
        drop(ends);
        schedule.prediction_validity = population.schedule.prediction_validity;
        let mut numerical = population.settings.clone();
        numerical.max_phase_samples = *schedule.maximum_phase_members.iter().max().unwrap();
        if schedule.validate(&numerical).is_err() {
            return Ok(ColdSourceRebuild::Skip(ColdSourceSkip::ScheduleCapacity));
        }
        let input_opportunities = budget::plan_checked_with_schedule_and_row_ceiling(
            &cases,
            &self.prompts,
            chunk.get() as usize,
            row_ceiling,
            &schedule,
            minimum,
            &self.opportunities,
        )?;
        let available_offers = mul(self.original_cycles, minimum)?;
        let required_offers = add(
            input_opportunities.required_original_offers,
            schedule.block_offered,
        )?;
        if available_offers < required_offers {
            return Ok(ColdSourceRebuild::Skip(
                ColdSourceSkip::OriginalCyclesInsufficient {
                    available_offers,
                    required_offers,
                },
            ));
        }
        Ok(ColdSourceRebuild::Ready(ColdSourcePlan {
            schedule,
            input_opportunities,
            maximum_anchor_span,
            cycles: self.original_cycles,
            requests: mul(requests, self.original_cycles)?,
            execution_actions: mul(actions, self.original_cycles)?,
            declared_offer_row_bound: mul(rows, self.original_cycles)?,
            serial_token_work: mul(tokens, self.original_cycles)?,
            original_collection_actions: self.collection_actions,
            original_declared_offer_row_bound: self.declared_offer_rows,
        }))
    }
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| error("source input work overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| error("source input work overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::execution_cost::CostProductOutput;
    use ferrum_interfaces::vnext::CheckpointTokenSpanConstraint;
    use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredPopulationPolicyV1;

    fn inputs(cycles: usize) -> (PreparedProbeSourceInputs, StructuredServiceDeclarationV7) {
        let population =
            population::declaration(&Default::default(), populations::tests::domain()).unwrap();
        let mut case = Case {
            product: OpportunityProduct::Full,
            template: 5,
            width: 1,
            maximum_output: NonZeroUsize::new(4).unwrap(),
            release_generated: 2,
            suffix_tokens: 2,
            preset: SloAutomaticCostProbeSamplingPresetV1::Configured,
            prefix: PrefixKind::Pending,
            route: CalibrationDecodeRoute::Actual,
            reset: false,
            acquisition: None,
        };
        case.acquisition = work::declared_plan(
            &case,
            work::PrefixBlueprint {
                prompt_tokens: 7,
                boundary: 6,
                span: CheckpointTokenSpanConstraint::any_positive(),
                input_tokens_sha256: [7; 32],
            },
            NonZeroU32::new(2).unwrap(),
            None,
        )
        .unwrap();
        let classified = populations::classify_alternatives(
            &[populations::tests::input(
                1,
                3,
                CostProductOutput::FullLogits,
                false,
                true,
            )],
            StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
            true,
        )
        .unwrap();
        let original = work::case_work(&case, 7, 2, None).unwrap();
        (
            PreparedProbeSourceInputs {
                cases: vec![case.clone(), case],
                opportunities: vec![
                    CaseOpportunity {
                        population: classified.clone(),
                        minimum_fresh_members: 1,
                    },
                    CaseOpportunity {
                        population: classified,
                        minimum_fresh_members: 0,
                    },
                ],
                prompts: vec![7, 7],
                original_cycles: cycles,
                collection_actions: original.execution_actions * 2 * cycles,
                declared_offer_rows: original.serial_declared_offer_rows * 2 * cycles,
            },
            population,
        )
    }

    #[test]
    fn cold_source_rebuild_preserves_members_and_requires_original_complete_horizon() {
        let (input, population) = inputs(1);
        let before = input.retained_heap_bytes().unwrap();
        let ColdSourceRebuild::Skip(ColdSourceSkip::OriginalCyclesInsufficient {
            available_offers,
            required_offers,
        }) = input
            .cold_plan(&population, NonZeroU32::new(2).unwrap(), None, usize::MAX)
            .unwrap()
        else {
            panic!("one original cycle must not claim three complete phases");
        };
        let cycles = required_offers.div_ceil(available_offers);
        let (sufficient, population) = inputs(cycles);
        let ColdSourceRebuild::Ready(cold) = sufficient
            .cold_plan(&population, NonZeroU32::new(2).unwrap(), None, usize::MAX)
            .unwrap()
        else {
            panic!("the same complete cohorts satisfy the recomputed cold horizon");
        };
        assert_eq!(cold.cycles, cycles);
        assert_eq!(cold.requests, sufficient.cases.len() * cycles);
        assert_eq!(
            cold.input_opportunities
                .minimum_input_family_opportunities_per_cycle,
            1
        );
        assert!(
            cycles
                * cold
                    .input_opportunities
                    .minimum_original_offers_per_completed_cycle
                >= cold.input_opportunities.required_original_offers + cold.schedule.block_offered
        );
        assert_eq!(
            cold.original_collection_actions,
            sufficient.collection_actions
        );
        assert_eq!(
            cold.original_declared_offer_row_bound,
            sufficient.declared_offer_rows
        );
        assert!(cold.execution_actions > cold.original_collection_actions);
        assert_eq!(cold.execution_actions, cold.declared_offer_row_bound);
        assert_eq!(input.retained_heap_bytes().unwrap(), before);
        assert!(sufficient
            .cases
            .iter()
            .all(|c| c.template == 5 && c.acquisition.is_some()));
        assert_eq!(sufficient.opportunities[1].minimum_fresh_members, 0);
    }

    #[test]
    fn cold_source_rebuild_rejects_capacity_and_never_synthesizes_member_floors() {
        let (mut input, population) = inputs(1);
        assert!(matches!(
            input
                .cold_plan(&population, NonZeroU32::new(2).unwrap(), None, 0)
                .unwrap(),
            ColdSourceRebuild::Skip(ColdSourceSkip::RetainedCapacity)
        ));
        for opportunity in &mut input.opportunities {
            opportunity.minimum_fresh_members = 0;
        }
        assert!(input
            .cold_plan(&population, NonZeroU32::new(2).unwrap(), None, usize::MAX)
            .is_err());
        assert!(input
            .opportunities
            .iter()
            .all(|o| o.minimum_fresh_members == 0));
    }
}
