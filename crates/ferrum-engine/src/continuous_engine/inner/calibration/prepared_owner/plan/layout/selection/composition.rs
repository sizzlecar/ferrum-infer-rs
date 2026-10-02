//! Bounded input-only local source declarations. No raw fitted parameter or
//! observation is merged. The original collector independently qualifies all
//! complete case horizons again under the declared local axes.
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    DeclaredAlgorithmUniverseBuilderV1, DeclaredAlgorithmUniverseV1, StructuredCostTemplatePolicyV1,
};

pub(super) fn builder_limit(seed: &DeclaredAlgorithmUniverseV1) -> Result<usize> {
    add(
        mul(
            seed.retained_payload_bytes()
                .ok_or_else(|| error("combination seed capacity overflow"))?,
            2,
        )?,
        std::mem::size_of::<DeclaredAlgorithmUniverseBuilderV1>(),
    )
}

/// One local source per selection. Charge simultaneously live builder/finished
/// declaration and duplicated representatives/population indices beside raw
/// output. The complete union batch's sequential scratch is already bounded by
/// the original selection's all-population batch stage.
pub(super) fn extra_peak(
    opportunities: &[CaseOpportunity],
    seed: &DeclaredAlgorithmUniverseV1,
) -> Result<usize> {
    let (mentions, guaranteed) = selection_inventory_cardinality(opportunities)?;
    add(
        add(
            mul(builder_limit(seed)?, 2)?,
            mul(vector_peak_bytes::<usize>(add(mentions, guaranteed)?)?, 6)?,
        )?,
        vector_peak_bytes::<SelectionGap>(add(opportunities.len(), 1)?)?,
    )
}

pub(super) fn authorized(
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    remaining_requests: usize,
    population: &StructuredServiceDeclarationV7,
    geometry_enabled: bool,
    maximum_bytes: usize,
    seed: &DeclaredAlgorithmUniverseV1,
) -> Result<bool> {
    if seed.algorithm_count() < 2 || memory::grouping_peak(opportunities)? > maximum_bytes {
        return Ok(false);
    }
    let groups = member_groups(opportunities)
        .map_err(|reason| error(format!("combination selection population: {reason:?}")))?;
    let memory = memory::plan(
        &groups,
        opportunities,
        inputs,
        remaining_requests,
        geometry_enabled.then_some(&population.settings),
    )?;
    Ok(add(memory.required_peak_bytes, extra_peak(opportunities, seed)?)? <= maximum_bytes)
}

/// Exactly the capacity predicate used by original append_batch, before any
/// optional scope is added. Invalid raw plans cannot acquire a new source slot.
pub(super) fn can_schedule(batch: &SelectedBatch, capacity: SelectionCapacity) -> bool {
    batch.schedule_within_capacity
        && batch.maximum_anchor_span <= *batch.schedule.phase_min_offered.iter().min().unwrap()
        && batch.requests <= capacity.requests
        && batch.serial_wave_upper_bound <= capacity.execution_actions
        && batch.declared_offer_row_bound <= capacity.declared_offer_rows
}

/// Reserve the entire original executable source prefix before increasing a
/// later source's work. A local combination cannot spend a later first role or
/// installed policy source's original request/action/inference-row allowance.
pub(super) fn scheduled_prefix_budget(
    candidates: &[BatchCandidate],
    capacity: SelectionCapacity,
    priority: Option<u8>,
    maximum_sources: Option<NonZeroUsize>,
) -> Result<SelectionCapacity> {
    let mut used = SelectionCapacity::default();
    let mut sources = 0usize;
    for candidate in candidates {
        let batch = &candidate.batch;
        if priority.is_some_and(|selected| selected != candidate.input_priority)
            || maximum_sources.is_some_and(|maximum| sources >= maximum.get())
            || !can_schedule(batch, capacity.remaining(used))
        {
            continue;
        }
        used.charge(batch)?;
        sources = add(sources, 1)?;
    }
    Ok(used)
}

fn facts<'a>(
    batch: &'a SelectedBatch,
    inputs: &'a [Vec<CheckedInputFacts>],
) -> impl Iterator<Item = &'a CheckedInputFacts> {
    batch
        .representative_case_indices
        .iter()
        .flat_map(|&index| &inputs[index])
}

// Some family exists only after numerical_family_key accepts homogeneous
// ordinary decode. Exact prefill and unsupported/mixed populations stay None.
fn same_requirement(left: &CheckedInputFacts, right: &CheckedInputFacts) -> bool {
    left.family.is_some()
        && right.family.is_some()
        && left.owner.role == right.owner.role
        && left.owner.product == right.owner.product
        && left.owner.readback == right.owner.readback
        && left.homogeneous_host_policy.is_some()
        && left.homogeneous_host_policy == right.homogeneous_host_policy
}

#[allow(clippy::too_many_arguments)]
pub(super) fn candidate(
    current: &SelectedBatch,
    earlier: &[SelectedBatch],
    populations: &[SelectedPopulation],
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    prompts: &[usize],
    chunk: usize,
    row_ceiling: Option<NonZeroU32>,
    population: &StructuredServiceDeclarationV7,
    seed: &DeclaredAlgorithmUniverseV1,
) -> Result<Option<SelectedBatch>> {
    if population
        .nonnegative_envelope
        .as_ref()
        .is_none_or(|contract| {
            contract.population_policy != StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1
                || contract.template_policy
                    != StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1
        })
    {
        return Ok(None);
    }
    let Some(first) = facts(current, inputs).next() else {
        return Ok(None);
    };
    if current.algorithm_universe.is_some()
        || facts(current, inputs)
            .any(|fact| !same_requirement(first, fact) || fact.original.is_none())
    {
        return Ok(None);
    }
    // Conservatively restrict to pure ordinary decode batches. Exact prefill
    // populations stay in their original raw sources and cannot supply numeric
    // support for a combination. An earlier completed raw requirement protects
    // the first role/product/readback/installed policy opportunity.
    for prior in earlier
        .iter()
        .filter(|batch| batch.scheduled && batch.algorithm_universe.is_none())
    {
        if prior
            .population_indices
            .iter()
            .any(|index| current.population_indices.contains(index))
            || facts(prior, inputs)
                .any(|fact| !same_requirement(first, fact) || fact.original.is_none())
            || !facts(prior, inputs).any(|fact| fact.family != first.family)
        {
            continue;
        }
        let mut builder = DeclaredAlgorithmUniverseBuilderV1::new(
            population.settings.max_axes,
            builder_limit(seed)?,
        )
        .map_err(|reason| error(format!("combination declaration builder: {reason:?}")))?;
        let mut valid = true;
        for fact in facts(prior, inputs).chain(facts(current, inputs)) {
            if builder.observe(fact.original.as_deref().unwrap()).is_err() {
                valid = false;
                break;
            }
        }
        if !valid {
            continue;
        }
        let local = match builder.finish() {
            Ok(local) => local,
            Err(_) => continue,
        };
        if !seed.contains_universe(&local) {
            continue;
        }
        let mut common = None;
        for fact in facts(prior, inputs).chain(facts(current, inputs)) {
            match fact
                .original
                .as_deref()
                .unwrap()
                .numerical_family_key_for_universe(&local)
            {
                Ok(key) if common.is_none_or(|old| old == key) => common = Some(key),
                _ => {
                    valid = false;
                    break;
                }
            }
        }
        if !valid || common.is_none() {
            continue;
        }
        let mut members = Vec::new();
        members
            .try_reserve_exact(add(
                prior.population_indices.len(),
                current.population_indices.len(),
            )?)
            .map_err(|_| error("combination population allocation capacity"))?;
        members.extend_from_slice(&prior.population_indices);
        members.extend_from_slice(&current.population_indices);
        members.sort_unstable();
        members.dedup();
        let mut combined = batch_plan(
            &members,
            populations,
            cases,
            opportunities,
            prompts,
            chunk,
            row_ceiling,
            population,
        )?;
        // batch_plan preserves every raw population's complete representatives,
        // each original fresh-member phase fence, and all execution work.
        combined.algorithm_universe = Some(local);
        return Ok(Some(combined));
    }
    Ok(None)
}
