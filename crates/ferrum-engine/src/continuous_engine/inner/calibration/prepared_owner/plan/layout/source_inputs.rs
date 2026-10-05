//! The original representatives and member floors survive source preparation.
//! A cold fallback changes work and barriers, never its selected cohort members.
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::OwnerBlockScheduleV1;
use populations::{CaseOpportunity, CasePopulation, CheckedPopulationKey, PopulationMemberGroup};

pub(in crate::continuous_engine::inner::calibration) struct PreparedProbeSourceInputs {
    cases: Vec<Case>,
    opportunities: Vec<CaseOpportunity>,
    prompts: Vec<usize>,
    original_occurrences: usize,
    collection_actions: usize,
    declared_offer_rows: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct ColdSourcePlan {
    pub schedule: OwnerBlockScheduleV1,
    pub input_plan: selection::SelectedInputPlan,
    pub planned_occurrences: usize,
    pub requests: usize,
    pub execution_actions: usize,
    pub declared_offer_row_bound: usize,
    pub serial_token_work: usize,
    pub original_collection_actions: usize,
    pub original_declared_offer_row_bound: usize,
}

impl ColdSourcePlan {
    pub fn retained_heap_bytes(&self) -> Option<usize> {
        match &self.input_plan {
            selection::SelectedInputPlan::Periodic { .. } => Some(0),
            selection::SelectedInputPlan::Finite { plan } => plan.retained_payload_bytes(),
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration) enum ColdSourceSkip {
    ScheduleCapacity,
    OriginalCyclesInsufficient {
        available_offers: usize,
        required_offers: usize,
    },
    OriginalFiniteSequenceInsufficient,
    RetainedCapacity,
}

#[derive(Debug)]
pub(in crate::continuous_engine::inner::calibration) enum ColdSourceRebuild {
    Ready(ColdSourcePlan),
    Skip(ColdSourceSkip),
}

pub(super) fn opportunity_heap_bytes(opportunity: &CaseOpportunity) -> Option<usize> {
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
        if batch.execution_case_count()? == 0 || batch.representative_case_indices.is_empty() {
            return Err(error(
                "source input reservation has no original occurrences",
            ));
        }
        for index in batch.execution_case_indices()? {
            if !batch.representative_case_indices.contains(&index) {
                return Err(error(
                    "source occurrence is outside its original representatives",
                ));
            }
        }
        if batch
            .scoped_opportunities
            .as_ref()
            .is_some_and(|items| items.len() != batch.representative_case_indices.len())
        {
            return Err(error(
                "source scope opportunities differ from original cases",
            ));
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
        for (position, &index) in batch.representative_case_indices.iter().enumerate() {
            let original = opportunities
                .get(index)
                .ok_or_else(|| error("source opportunity outside original inventory"))?;
            let opportunity = batch
                .scoped_opportunities
                .as_ref()
                .map_or(original, |items| &items[position]);
            if opportunity.minimum_fresh_members != original.minimum_fresh_members {
                return Err(error("source scope changed original fresh member floor"));
            }
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
            original_occurrences: batch.execution_case_count()?,
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
        for (position, &index) in batch.representative_case_indices.iter().enumerate() {
            let case = &cases[index];
            let prompt = prompts[case.template];
            source.cases.push(case.clone());
            source.opportunities.push(
                batch
                    .scoped_opportunities
                    .as_ref()
                    .map_or(&opportunities[index], |items| &items[position])
                    .clone(),
            );
            source.prompts.push(prompt);
        }
        for index in batch.execution_case_indices()? {
            let case = &cases[index];
            let work = work::case_work(
                case,
                prompts[case.template],
                chunk.get() as usize,
                row_ceiling,
            )?;
            source.collection_actions = add(source.collection_actions, work.execution_actions)?;
            source.declared_offer_rows =
                add(source.declared_offer_rows, work.serial_declared_offer_rows)?;
        }
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
        batch: &selection::SelectedBatch,
        population: &StructuredServiceDeclarationV7,
        chunk: NonZeroU32,
        row_ceiling: Option<NonZeroU32>,
        maximum_temporary_bytes: usize,
    ) -> Result<ColdSourceRebuild> {
        if batch.representative_case_indices.len() != self.cases.len()
            || self.cases.len() != self.opportunities.len()
            || self.cases.len() != self.prompts.len()
            || batch.execution_case_count()? != self.original_occurrences
        {
            return Err(error(
                "cold source differs from its original input allocation",
            ));
        }
        if let Some(plan) = batch.finite_plan() {
            return self.cold_finite_plan(
                batch,
                plan,
                population,
                chunk,
                row_ceiling,
                maximum_temporary_bytes,
            );
        }
        let original_cycles = batch
            .periodic_cycles()
            .ok_or_else(|| error("cold periodic source has no original cycle count"))?;
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
        let available_offers = mul(original_cycles, minimum)?;
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
            input_plan: selection::SelectedInputPlan::Periodic {
                planned_cycles: original_cycles,
                input_opportunities,
                maximum_anchor_span,
            },
            planned_occurrences: self.original_occurrences,
            requests: mul(requests, original_cycles)?,
            execution_actions: mul(actions, original_cycles)?,
            declared_offer_row_bound: mul(rows, original_cycles)?,
            serial_token_work: mul(tokens, original_cycles)?,
            original_collection_actions: self.collection_actions,
            original_declared_offer_row_bound: self.declared_offer_rows,
        }))
    }

    fn cold_finite_plan(
        &self,
        batch: &selection::SelectedBatch,
        original: &budget::finite::FinitePlan,
        population: &StructuredServiceDeclarationV7,
        chunk: NonZeroU32,
        row_ceiling: Option<NonZeroU32>,
        maximum_temporary_bytes: usize,
    ) -> Result<ColdSourceRebuild> {
        if original
            .retained_payload_bytes()
            .is_none_or(|n| n > maximum_temporary_bytes)
        {
            return Ok(ColdSourceRebuild::Skip(ColdSourceSkip::RetainedCapacity));
        }
        let position = |index| {
            batch
                .representative_case_indices
                .iter()
                .position(|&case| case == index)
                .ok_or_else(|| error("cold occurrence is outside its original representatives"))
        };
        let opportunity_for =
            |index| -> Result<&CaseOpportunity> { Ok(&self.opportunities[position(index)?]) };
        let work_for = |index| -> Result<work::CaseWork> {
            let position = position(index)?;
            let mut case = self.cases[position].clone();
            case.acquisition = None;
            // Only preparation changes. The original template, scope, case
            // index and frozen occurrence order remain authoritative.
            work::case_work(
                &case,
                self.prompts[position],
                chunk.get() as usize,
                row_ceiling,
            )
        };
        let plan = match budget::finite::verify_frozen(
            original,
            opportunity_for,
            work_for,
            work::CaseWork::default(),
            &population.settings,
            maximum_temporary_bytes,
        )? {
            budget::finite::FiniteVerification::Ready(plan) => plan,
            budget::finite::FiniteVerification::Skip(reason) => {
                let reason = match reason {
                    budget::finite::FiniteRejection::RetainedCapacity => {
                        ColdSourceSkip::RetainedCapacity
                    }
                    budget::finite::FiniteRejection::ScheduleCapacity => {
                        ColdSourceSkip::ScheduleCapacity
                    }
                    budget::finite::FiniteRejection::FrozenHorizon => {
                        ColdSourceSkip::OriginalFiniteSequenceInsufficient
                    }
                };
                return Ok(ColdSourceRebuild::Skip(reason));
            }
        };
        if plan.occurrence_case_indices != original.occurrence_case_indices {
            return Err(error(
                "cold verification changed the original occurrence sequence",
            ));
        }
        Ok(ColdSourceRebuild::Ready(ColdSourcePlan {
            schedule: plan.schedule.clone(),
            planned_occurrences: self.original_occurrences,
            requests: plan.work.requests,
            execution_actions: plan.work.execution_actions,
            declared_offer_row_bound: plan.work.serial_declared_offer_rows,
            serial_token_work: plan.work.serial_token_work,
            // verify_frozen authorized the new boxed header and backing while
            // the original selected proof remains owned by its source.
            input_plan: selection::SelectedInputPlan::Finite {
                plan: Box::new(plan),
            },
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
                original_occurrences: 2 * cycles,
                collection_actions: original.execution_actions * 2 * cycles,
                declared_offer_rows: original.serial_declared_offer_rows * 2 * cycles,
            },
            population,
        )
    }

    fn batch_for(
        input: &PreparedProbeSourceInputs,
        cycles: usize,
        population: &StructuredServiceDeclarationV7,
    ) -> selection::SelectedBatch {
        let prompts = vec![7; 6];
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        let (mut upper, mut minimum, mut requests, mut actions, mut rows, mut tokens) =
            (0, 0, 0, 0, 0, 0);
        for case in &input.cases {
            let work = work::case_work(case, prompts[case.template], 2, None).unwrap();
            starts.push(upper);
            upper += work.declared_offers_upper;
            ends.push(upper);
            minimum += work.declared_offers_minimum;
            requests += work.requests;
            actions += work.execution_actions;
            rows += work.serial_declared_offer_rows;
            tokens += work.serial_token_work;
        }
        let (schedule, maximum_anchor_span) =
            budget::startup_schedule(&starts, &ends, &population.settings).unwrap();
        let input_opportunities = budget::plan_checked_with_schedule_and_row_ceiling(
            &input.cases,
            &prompts,
            2,
            None,
            &schedule,
            minimum,
            &input.opportunities,
        )
        .unwrap();
        let setup = work::setup_for_cases(&input.cases).unwrap();
        selection::SelectedBatch {
            population_indices: vec![0],
            // Global case identity need not equal its source-local position.
            representative_case_indices: vec![6, 9],
            input_plan: selection::SelectedInputPlan::Periodic {
                planned_cycles: cycles,
                input_opportunities,
                maximum_anchor_span,
            },
            schedule,
            schedule_within_capacity: true,
            requests: requests * cycles + setup.requests,
            serial_token_work: tokens * cycles + setup.serial_token_work,
            serial_wave_upper_bound: actions * cycles + setup.execution_actions,
            declared_offer_row_bound: rows * cycles,
            scheduled: true,
            algorithm_universe: None,
            scoped_opportunities: None,
        }
    }

    #[test]
    fn cold_source_rebuild_preserves_members_and_requires_original_complete_horizon() {
        let (input, population) = inputs(1);
        let batch = batch_for(&input, 1, &population);
        let before = input.retained_heap_bytes().unwrap();
        let ColdSourceRebuild::Skip(ColdSourceSkip::OriginalCyclesInsufficient {
            available_offers,
            required_offers,
        }) = input
            .cold_plan(
                &batch,
                &population,
                NonZeroU32::new(2).unwrap(),
                None,
                usize::MAX,
            )
            .unwrap()
        else {
            panic!("one original cycle must not claim three complete phases");
        };
        let cycles = required_offers.div_ceil(available_offers);
        let (sufficient, population) = inputs(cycles);
        let batch = batch_for(&sufficient, cycles, &population);
        let ColdSourceRebuild::Ready(cold) = sufficient
            .cold_plan(
                &batch,
                &population,
                NonZeroU32::new(2).unwrap(),
                None,
                usize::MAX,
            )
            .unwrap()
        else {
            panic!("the same complete cohorts satisfy the recomputed cold horizon");
        };
        let selection::SelectedInputPlan::Periodic {
            planned_cycles,
            input_opportunities,
            ..
        } = &cold.input_plan
        else {
            panic!("cold periodic source changed plan kind");
        };
        assert_eq!(*planned_cycles, cycles);
        assert_eq!(cold.planned_occurrences, sufficient.cases.len() * cycles);
        assert_eq!(cold.requests, sufficient.cases.len() * cycles);
        assert_eq!(
            input_opportunities.minimum_input_family_opportunities_per_cycle,
            1
        );
        assert!(
            cycles * input_opportunities.minimum_original_offers_per_completed_cycle
                >= input_opportunities.required_original_offers + cold.schedule.block_offered
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
        let batch = batch_for(&input, 1, &population);
        assert!(matches!(
            input
                .cold_plan(&batch, &population, NonZeroU32::new(2).unwrap(), None, 0)
                .unwrap(),
            ColdSourceRebuild::Skip(ColdSourceSkip::RetainedCapacity)
        ));
        for opportunity in &mut input.opportunities {
            opportunity.minimum_fresh_members = 0;
        }
        assert!(input
            .cold_plan(
                &batch,
                &population,
                NonZeroU32::new(2).unwrap(),
                None,
                usize::MAX
            )
            .is_err());
        assert!(input
            .opportunities
            .iter()
            .all(|o| o.minimum_fresh_members == 0));
    }

    #[test]
    fn cold_finite_rebuild_preserves_original_occurrences_and_charges_new_proof() {
        let (mut original, population) = inputs(1);
        for opportunity in &mut original.opportunities {
            opportunity.minimum_fresh_members = 1;
        }
        let prompts = vec![7; 6];
        let budget::finite::FiniteVerification::Ready(plan) = budget::finite::build(
            &[vec![0, 1]],
            &original.opportunities,
            &original.cases,
            &prompts,
            2,
            None,
            &population.settings,
            usize::MAX,
            usize::MAX,
        )
        .unwrap() else {
            panic!("original native finite fixture must be provable");
        };
        let occurrences = plan.occurrence_case_indices.clone();
        let work = plan.work;
        let batch = selection::SelectedBatch {
            population_indices: vec![0],
            representative_case_indices: vec![0, 1],
            schedule: plan.schedule.clone(),
            input_plan: selection::SelectedInputPlan::Finite {
                plan: Box::new(plan),
            },
            schedule_within_capacity: true,
            requests: work.requests,
            serial_token_work: work.serial_token_work,
            serial_wave_upper_bound: work.execution_actions,
            declared_offer_row_bound: work.serial_declared_offer_rows,
            scheduled: true,
            algorithm_universe: None,
            scoped_opportunities: Some(original.opportunities.clone()),
        };
        let selected = selection::CheckedSelection {
            execution_case_indices: occurrences.clone(),
            requests: batch.requests,
            serial_wave_upper_bound: batch.serial_wave_upper_bound,
            declared_offer_row_bound: batch.declared_offer_row_bound,
            batches: vec![batch],
            ..Default::default()
        };
        let mut frozen = freeze_sources(
            &original.cases,
            &original.opportunities,
            &prompts,
            &selected,
            NonZeroU32::new(2).unwrap(),
            None,
            usize::MAX,
        )
        .unwrap();
        let input = &mut frozen[0];
        let batch = &selected.batches[0];
        let before = input.retained_heap_bytes().unwrap();
        let proof_bound = batch
            .finite_plan()
            .unwrap()
            .retained_payload_bytes()
            .unwrap();
        assert!(matches!(
            input
                .cold_plan(
                    batch,
                    &population,
                    NonZeroU32::new(2).unwrap(),
                    None,
                    proof_bound - 1,
                )
                .unwrap(),
            ColdSourceRebuild::Skip(ColdSourceSkip::RetainedCapacity)
        ));
        let ColdSourceRebuild::Ready(cold) = input
            .cold_plan(
                batch,
                &population,
                NonZeroU32::new(2).unwrap(),
                None,
                proof_bound,
            )
            .unwrap()
        else {
            panic!("cold work must reprove this unchanged finite sequence");
        };
        let selection::SelectedInputPlan::Finite { plan } = &cold.input_plan else {
            panic!("finite allocation was relabelled as a cycle");
        };
        assert_eq!(plan.occurrence_case_indices, occurrences);
        assert_eq!(cold.planned_occurrences, occurrences.len());
        assert_eq!(cold.requests, occurrences.len());
        let setup = work::setup_for_cases(&original.cases).unwrap();
        assert!(setup.requests > 0 && setup.execution_actions > 0);
        assert_eq!(cold.requests + setup.requests, batch.requests);
        assert_eq!(
            cold.original_collection_actions + setup.execution_actions,
            batch.serial_wave_upper_bound
        );
        assert!(cold.execution_actions > cold.original_collection_actions);
        assert_eq!(cold.execution_actions, cold.declared_offer_row_bound);
        assert_eq!(
            cold.retained_heap_bytes().unwrap(),
            std::mem::size_of::<budget::finite::FinitePlan>() + plan.retained_heap_bytes().unwrap()
        );
        assert!(cold.retained_heap_bytes().unwrap() <= proof_bound);
        // The cold proof is separately owned while the original selected
        // proof remains live. A further clone must retain its own box and
        // backing after the temporary cold plan is retired.
        let cold_clone = cold.clone();
        let selection::SelectedInputPlan::Finite { plan: cloned_plan } = &cold_clone.input_plan
        else {
            unreachable!();
        };
        assert!(!std::ptr::eq(plan.as_ref(), cloned_plan.as_ref()));
        assert!(!std::ptr::eq(plan.as_ref(), batch.finite_plan().unwrap()));
        let cold_wire = serde_json::to_value(&cold).unwrap();
        assert_eq!(serde_json::to_value(&cold_clone).unwrap(), cold_wire);
        assert_eq!(
            cold_clone.retained_heap_bytes().unwrap(),
            cloned_plan.retained_payload_bytes().unwrap()
        );
        drop(cold);
        assert_eq!(serde_json::to_value(&cold_clone).unwrap(), cold_wire);
        assert_eq!(input.retained_heap_bytes().unwrap(), before);
        assert!(input
            .cases
            .iter()
            .all(|case| case.template == 5 && case.acquisition.is_some()));
        assert_eq!(
            batch.finite_plan().unwrap().occurrence_case_indices,
            occurrences
        );
        assert!(matches!(
            input
                .cold_plan(batch, &population, NonZeroU32::new(2).unwrap(), None, 0,)
                .unwrap(),
            ColdSourceRebuild::Skip(ColdSourceSkip::RetainedCapacity)
        ));
        input.opportunities[0].minimum_fresh_members = 0;
        assert!(
            input
                .cold_plan(
                    batch,
                    &population,
                    NonZeroU32::new(2).unwrap(),
                    None,
                    usize::MAX,
                )
                .is_err(),
            "cold fallback cannot revive a missing original member floor"
        );
    }
}
