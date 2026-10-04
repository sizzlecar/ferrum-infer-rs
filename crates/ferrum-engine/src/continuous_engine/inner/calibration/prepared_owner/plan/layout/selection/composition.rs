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

/// Each guaranteed group creates at most one candidate/local declaration.
/// Coalescing only removes candidates, and append moves them into output.
/// Repeated case mentions grow vectors, not the number of retained scopes.
/// Keep two additional builder/replacement slots beside every possible scope
/// and the complete source-local opportunity arrays. Raw and projected facts
/// never share a numerical sample or member floor.
pub(super) fn extra_peak(
    opportunities: &[CaseOpportunity],
    seed: &DeclaredAlgorithmUniverseV1,
    retained_group_count: usize,
) -> Result<usize> {
    let (mentions, guaranteed) = selection_inventory_cardinality(opportunities)?;
    // Composition adds no separate gap buffer. The sole CheckedSelection::gaps
    // Vec, including its growth peak, is already charged by memory::plan.
    add(
        add(
            mul(builder_limit(seed)?, add(retained_group_count, 2)?)?,
            add(
                mul(vector_peak_bytes::<usize>(add(mentions, guaranteed)?)?, 6)?,
                mul(vector_peak_bytes::<CaseOpportunity>(guaranteed)?, 2)?,
            )?,
        )?,
        add(
            vector_peak_bytes::<usize>(mentions)?,
            mul(std::mem::size_of::<BatchCandidate>(), 2)?,
        )?,
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
    let extra_peak = extra_peak(opportunities, seed, memory.guaranteed_groups)?;
    let required_peak = add(memory.required_peak_bytes, extra_peak)?;
    let authorized = required_peak <= maximum_bytes;
    let seed_bytes = seed
        .retained_payload_bytes()
        .ok_or_else(|| error("combination seed capacity overflow"))?;
    tracing::info!(
        seed_algorithms = seed.algorithm_count(),
        seed_bytes,
        retained_group_count = memory.guaranteed_groups,
        required_peak,
        extra_peak,
        remaining = maximum_bytes,
        authorized,
        "Automatic local composition retained memory authorization"
    );
    Ok(authorized)
}

/// Exactly the capacity predicate used by original append_batch, before any
/// optional scope is added. Invalid raw plans cannot acquire a new source slot.
pub(super) fn can_schedule(batch: &SelectedBatch, capacity: SelectionCapacity) -> bool {
    batch.schedule_within_capacity
        && batch.anchors_within_schedule()
        && batch.requests <= capacity.requests
        && batch.serial_wave_upper_bound <= capacity.execution_actions
        && batch.declared_offer_row_bound <= capacity.declared_offer_rows
}

fn facts<'a>(
    batch: &'a SelectedBatch,
    inputs: &'a [Vec<CheckedInputFacts>],
) -> impl Iterator<Item = &'a CheckedInputFacts> + Clone {
    batch
        .representative_case_indices
        .iter()
        .flat_map(|&index| &inputs[index])
}

fn recipes<'a>(
    batch: &'a SelectedBatch,
    inputs: &'a [Vec<CheckedInputFacts>],
    trajectories: Option<&'a inventory::CheckedCaseInventory>,
) -> impl Iterator<Item = &'a ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredInputV2> + Clone{
    batch
        .representative_case_indices
        .iter()
        .flat_map(move |&index| {
            let indices = trajectories.map_or(&[][..], |inventory| {
                inventory.algorithm_case_inputs[index].as_slice()
            });
            inputs[index]
                .iter()
                .filter_map(|fact| fact.original.as_deref())
                .chain(
                    indices
                        .iter()
                        .map(move |&input| trajectories.unwrap().algorithm_inputs[input].as_ref()),
                )
        })
}

fn same_requirement(left: &CheckedInputFacts, right: &CheckedInputFacts) -> bool {
    left.family.is_some()
        && right.family.is_some()
        && left.owner.role == right.owner.role
        && left.owner.product == right.owner.product
        && left.owner.readback == right.owner.readback
        && left.homogeneous_host_policy.is_some()
        && left.homogeneous_host_policy == right.homogeneous_host_policy
}

fn local_universe(
    batch: &SelectedBatch,
    inputs: &[Vec<CheckedInputFacts>],
    trajectories: Option<&inventory::CheckedCaseInventory>,
    population: &StructuredServiceDeclarationV7,
    seed: &DeclaredAlgorithmUniverseV1,
) -> Result<Option<DeclaredAlgorithmUniverseV1>> {
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
    let Some(first) = facts(batch, inputs).next() else {
        return Ok(None);
    };
    if facts(batch, inputs).any(|fact| !same_requirement(first, fact) || fact.original.is_none()) {
        return Ok(None);
    }
    let first_input = first.original.as_deref().unwrap();
    // The complete seed is only a nonallocating compatibility check. The local
    // declaration below observes solely recipes linked to these original cases.
    let Ok(common) = first_input.numerical_family_key_for_universe(seed) else {
        return Ok(None);
    };
    if facts(batch, inputs).any(|fact| {
        fact.original
            .as_deref()
            .unwrap()
            .numerical_family_key_for_universe(seed)
            != Ok(common)
    }) {
        return Ok(None);
    }
    let compatible = || {
        recipes(batch, inputs, trajectories)
            .filter(|input| input.numerical_family_key_for_universe(seed) == Ok(common))
    };
    if !compatible().any(|input| input.numerical_family_key().ok() != first.family) {
        return Ok(None);
    }
    let mut builder =
        DeclaredAlgorithmUniverseBuilderV1::new(population.settings.max_axes, builder_limit(seed)?)
            .map_err(|reason| error(format!("combination declaration builder: {reason:?}")))?;
    // The physical envelope is replayed before numerical-family membership.
    // A complete original cohort can therefore need an associated algorithm
    // from another host/product family without making that wave a member of
    // this family's fit. Declare all linked original work, then keep the
    // original compatible-family checks and independent qualification below.
    for input in recipes(batch, inputs, trajectories) {
        if builder.observe(input).is_err() {
            return Ok(None);
        }
    }
    let local = match builder.finish() {
        Ok(local) if seed.contains_universe(&local) => local,
        _ => return Ok(None),
    };
    let mut projected = None;
    for input in compatible() {
        match input.numerical_family_key_for_universe(&local) {
            Ok(key) if projected.is_none_or(|old| old == key) => projected = Some(key),
            _ => return Ok(None),
        }
    }
    Ok(Some(local))
}

/// Packing shares a source journal, not a numerical interpretation. Its
/// original raw keys or identical existing universe must remain unchanged.
/// This validates the known independent owner lower bound without allocating
/// another family vector; every actual owner still faces the runtime limit.
pub(super) fn packing_valid(
    batch: &SelectedBatch,
    inputs: &[Vec<CheckedInputFacts>],
    trajectories: Option<&inventory::CheckedCaseInventory>,
    population: &StructuredServiceDeclarationV7,
    seed: &DeclaredAlgorithmUniverseV1,
) -> Result<bool> {
    if population
        .nonnegative_envelope
        .as_ref()
        .is_none_or(|contract| {
            contract.population_policy != StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1
                || contract.template_policy
                    != StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1
        })
    {
        return Ok(false);
    }
    let scope = batch.algorithm_universe.as_ref();
    if scope.is_some_and(|universe| !seed.contains_universe(universe))
        || recipes(batch, inputs, trajectories).any(|input| {
            seed.contains_checked_algorithms(input) != Ok(true)
                || scope
                    .is_some_and(|universe| universe.contains_checked_algorithms(input) != Ok(true))
        })
    {
        return Ok(false);
    }
    let key_for = |input: &ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredInputV2| {
        match scope {
            Some(universe) => input.numerical_family_key_for_universe(universe),
            None => input.numerical_family_key(),
        }
    };
    let mut families = 0usize;
    for (position, &index) in batch.representative_case_indices.iter().enumerate() {
        let mut family = None;
        if inputs[index].is_empty() {
            return Ok(false);
        }
        for fact in &inputs[index] {
            let Some(input) = fact.original.as_deref() else {
                return Ok(false);
            };
            let Ok(key) = key_for(input) else {
                return Ok(false);
            };
            if family.is_some_and(|previous| previous != key) {
                return Ok(false);
            }
            family = Some(key);
        }
        let key = family.unwrap();
        let seen = batch.representative_case_indices[..position]
            .iter()
            .any(|&earlier| {
                inputs[earlier][0]
                    .original
                    .as_deref()
                    .is_some_and(|input| key_for(input) == Ok(key))
            });
        if !seen {
            families = add(families, 1)?;
            if families > population.maximum_owners {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Project one original opportunity without inventing reachability or a fresh
/// member. Raw facts and their per-algorithm endpoint representatives survive.
pub(super) fn project_opportunity(
    original: &CaseOpportunity,
    facts: &[CheckedInputFacts],
    universe: &DeclaredAlgorithmUniverseV1,
) -> Result<CaseOpportunity> {
    if facts.is_empty() {
        return Err(error("source scope has no checked input"));
    }
    let mut common = None;
    for fact in facts {
        let input = fact
            .original
            .as_deref()
            .ok_or_else(|| error("source scope lost its checked recipe"))?;
        let key = input
            .numerical_family_key_for_universe(universe)
            .map_err(|reason| error(format!("source scope projection: {reason:?}")))?;
        if common.is_some_and(|old| old != key) {
            return Err(error(
                "source scope alternatives have different numerical families",
            ));
        }
        common = Some(key);
    }
    let key = CheckedPopulationKey::NumericalFamily(common.unwrap());
    let population = match &original.population {
        CasePopulation::Unique(_) => CasePopulation::Unique(key),
        CasePopulation::Alternatives(_) => CasePopulation::Alternatives(vec![key]),
        CasePopulation::Unknown { .. } => CasePopulation::Unknown {
            known_alternatives: vec![key],
        },
    };
    Ok(CaseOpportunity {
        population,
        minimum_fresh_members: original.minimum_fresh_members,
    })
}

/// Choose one complete source before collection. The union replaces its raw
/// basis; it has its own frozen family and independent unchanged F/R/Q rules.
#[allow(clippy::too_many_arguments)]
pub(super) fn scoped_candidate(
    raw: &SelectedBatch,
    populations: &[SelectedPopulation],
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    trajectories: Option<&inventory::CheckedCaseInventory>,
    prompts: &[usize],
    chunk: usize,
    row_ceiling: Option<NonZeroU32>,
    population: &StructuredServiceDeclarationV7,
    seed: &DeclaredAlgorithmUniverseV1,
    allocation: InputAllocationPolicy,
) -> Result<Option<SelectedBatch>> {
    let Some(local) = local_universe(raw, inputs, trajectories, population, seed)? else {
        return Ok(None);
    };
    batch_plan_with_scope(
        &raw.population_indices,
        populations,
        cases,
        opportunities,
        prompts,
        chunk,
        row_ceiling,
        population,
        Some((inputs, &local)),
        allocation,
    )
    .map(Some)
}
