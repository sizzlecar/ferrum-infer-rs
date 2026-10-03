//! Fixed-scope lookup and test-only scope-first experiments. Product selection
//! keeps its original raw inventory; experimental projections retain raw
//! recipes and every host/product/route family's own member floor.
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1;
#[cfg(test)]
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    DeclaredAlgorithmUniverseBuilderV1, StructuredCostTemplatePolicyV1,
};

#[cfg(test)]
pub(super) struct PreparedInputs {
    pub inputs: Vec<Vec<CheckedInputFacts>>,
    pub opportunities: Vec<CaseOpportunity>,
    pub scopes: Vec<Option<DeclaredAlgorithmUniverseV1>>,
    /// Temporary projected metadata plus scoped-only representative scratch.
    pub reserved_bytes: usize,
}

#[cfg(test)]
fn eligible(
    index: usize,
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
) -> bool {
    matches!(opportunities[index].population, CasePopulation::Unique(_))
        && !inputs[index].is_empty()
        && inputs[index].iter().all(|fact| {
            fact.family.is_some()
                && fact.homogeneous_host_policy.is_some()
                && fact
                    .original
                    .as_ref()
                    .is_some_and(|input| input.algorithm_universe_signature().is_none())
        })
}

#[cfg(test)]
fn same_class(a: usize, b: usize, cases: &[Case], inputs: &[Vec<CheckedInputFacts>]) -> bool {
    let first = &inputs[a][0];
    cases[a].route == cases[b].route
        && inputs[b].iter().all(|fact| {
            fact.owner.role == first.owner.role
                && fact.owner.product == first.owner.product
                && fact.owner.readback == first.owner.readback
                && fact.family.is_some_and(|family| {
                    Some(family.workload_domain_signature())
                        == first
                            .family
                            .as_ref()
                            .map(|key| key.workload_domain_signature())
                })
        })
}

/// A per-case fixed scope must survive representative compression and source
/// merging; the selected representatives are not a new declaration input.
pub(super) fn for_indices<'a>(
    indices: &[usize],
    scopes: Option<&'a [Option<DeclaredAlgorithmUniverseV1>]>,
) -> Result<Option<&'a DeclaredAlgorithmUniverseV1>> {
    let Some(scopes) = scopes else {
        return Ok(None);
    };
    let Some(&first) = indices.first() else {
        return Ok(None);
    };
    let scope = scopes[first].as_ref();
    if indices.iter().any(|&index| scopes[index].as_ref() != scope) {
        return Err(error(
            "one selected population has different predeclared scopes",
        ));
    }
    Ok(scope)
}

/// Explicit experiment only. Product selection keeps its original single pass;
/// this path cannot establish preservation of the raw plan's scheduled support.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn select_for_test(
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    prompts: &[usize],
    chunk: usize,
    prefill_row_ceiling: Option<NonZeroU32>,
    population: &StructuredServiceDeclarationV7,
    capacity: SelectionCapacity,
    maximum_retained_bytes: usize,
    changed: Option<&[CheckedPopulationKey]>,
    selected_priority: Option<u8>,
    geometry_work: Option<&mut StructuredInputGeometryWorkV1>,
    maximum_sources: Option<NonZeroUsize>,
    combination_seed: Option<&DeclaredAlgorithmUniverseV1>,
    trajectories: Option<&inventory::CheckedCaseInventory>,
) -> Result<CheckedSelection> {
    let prepared = if geometry_work.is_some() && changed.is_none() {
        combination_seed
            .map(|seed| {
                prepare(
                    cases,
                    opportunities,
                    inputs,
                    trajectories,
                    population,
                    seed,
                    capacity.requests,
                    maximum_retained_bytes,
                )
            })
            .transpose()?
            .flatten()
    } else {
        None
    };
    let (opportunities, inputs, scopes, retained) = match &prepared {
        Some(view) => (
            view.opportunities.as_slice(),
            view.inputs.as_slice(),
            Some(view.scopes.as_slice()),
            view.reserved_bytes,
        ),
        None => (opportunities, inputs, None, 0),
    };
    select_prepared_inputs(
        cases,
        opportunities,
        inputs,
        prompts,
        chunk,
        prefill_row_ceiling,
        population,
        capacity,
        maximum_retained_bytes - retained,
        changed,
        selected_priority,
        geometry_work,
        maximum_sources,
        combination_seed,
        trajectories,
        scopes,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare(
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    trajectories: Option<&inventory::CheckedCaseInventory>,
    population: &StructuredServiceDeclarationV7,
    seed: &DeclaredAlgorithmUniverseV1,
    remaining_requests: usize,
    maximum_bytes: usize,
) -> Result<Option<PreparedInputs>> {
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
    // Build class heads once. Repeated cases do not re-run a quadratic
    // search over all preceding cases; they compare only distinct classes.
    let class_scratch = add(
        vector_peak_bytes::<usize>(cases.len())?,
        vector_peak_bytes::<Option<usize>>(cases.len())?,
    )?;
    if class_scratch > maximum_bytes {
        return Ok(None);
    }
    let mut heads = Vec::new();
    let mut classes = Vec::with_capacity(cases.len());
    for index in 0..cases.len() {
        let class = if eligible(index, opportunities, inputs) {
            Some(
                match heads
                    .iter()
                    .position(|&first| same_class(first, index, cases, inputs))
                {
                    Some(class) => class,
                    None => {
                        heads.push(index);
                        heads.len() - 1
                    }
                },
            )
        } else {
            None
        };
        classes.push(class);
    }
    if heads.is_empty() {
        return Ok(None);
    }
    let scope_headers = mul(
        cases.len(),
        std::mem::size_of::<Option<DeclaredAlgorithmUniverseV1>>(),
    )?;
    let declarations = mul(heads.len(), composition::builder_limit(seed)?)?;
    let build_peak = add(
        class_scratch,
        add(
            scope_headers,
            add(declarations, mul(composition::builder_limit(seed)?, 2)?)?,
        )?,
    )?;
    if build_peak > maximum_bytes {
        return Ok(None);
    }
    let mut scopes = vec![None; cases.len()];
    for (class, &first) in heads.iter().enumerate() {
        let belongs = |index: usize| classes[index] == Some(class);
        // A single unchanged algorithm family gains no representative sharing.
        if !(0..cases.len())
            .filter(|&index| belongs(index))
            .any(|index| {
                inputs[index].iter().any(|fact| {
                    fact.owner.algorithm_domain != inputs[first][0].owner.algorithm_domain
                })
            })
        {
            continue;
        }
        let mut builder = DeclaredAlgorithmUniverseBuilderV1::new(
            population.settings.max_axes,
            composition::builder_limit(seed)?,
        )
        .map_err(|reason| error(format!("preselection scope builder: {reason:?}")))?;
        let mut complete = true;
        for index in (0..cases.len()).filter(|&index| belongs(index)) {
            for fact in &inputs[index] {
                if builder.observe(fact.original.as_deref().unwrap()).is_err() {
                    complete = false;
                    break;
                }
            }
            if let Some(inventory) = trajectories {
                for &linked in &inventory.algorithm_case_inputs[index] {
                    if builder
                        .observe(inventory.algorithm_inputs[linked].as_ref())
                        .is_err()
                    {
                        complete = false;
                        break;
                    }
                }
            }
            if !complete {
                break;
            }
        }
        if !complete {
            continue;
        }
        let scope = match builder.finish() {
            Ok(scope) if seed.contains_universe(&scope) => scope,
            _ => continue,
        };
        // Every alternative of a complete case must still name the same exact
        // host family; linked trajectory observations never supply its floor.
        if (0..cases.len())
            .filter(|&index| belongs(index))
            .any(|index| {
                let first = inputs[index][0]
                    .original
                    .as_deref()
                    .unwrap()
                    .numerical_family_key_for_universe(&scope);
                first.is_err()
                    || inputs[index].iter().any(|fact| {
                        fact.original
                            .as_deref()
                            .unwrap()
                            .numerical_family_key_for_universe(&scope)
                            != first
                    })
            })
        {
            continue;
        }
        for index in (0..cases.len()).filter(|&index| belongs(index)) {
            scopes[index] = Some(scope.clone());
        }
    }
    drop(classes);
    drop(heads);
    if scopes.iter().all(Option::is_none) {
        return Ok(None);
    }

    // All arrays below reserve exact lengths. Arc handles share the original
    // already charged recipes; no second original recipe pool is allocated.
    let mut retained = add(scope_headers, declarations)?;
    retained = add(
        retained,
        mul(
            cases.len(),
            std::mem::size_of::<Vec<CheckedInputFacts>>() + std::mem::size_of::<CaseOpportunity>(),
        )?,
    )?;
    let mut projection_peak = 0;
    let mut maximum_scoped_axes = 0;
    for (index, facts) in inputs.iter().enumerate() {
        retained = add(
            retained,
            mul(facts.len(), std::mem::size_of::<CheckedInputFacts>())?,
        )?;
        retained = add(
            retained,
            source_inputs::opportunity_heap_bytes(&opportunities[index])
                .ok_or_else(|| error("preselection opportunity size overflow"))?,
        )?;
        for fact in facts {
            let axes = match &scopes[index] {
                Some(scope) => {
                    let raw = fact.original.as_deref().unwrap();
                    let original = raw
                        .retained_payload_bytes()
                        .ok_or_else(|| error("preselection recipe size overflow"))?;
                    let projected = scope
                        .projected_input_retained_bytes(raw)
                        .ok_or_else(|| error("preselection projected size overflow"))?;
                    // The raw clone and replacement projection buffers can
                    // coexist while the final metadata view is being built.
                    projection_peak = projection_peak.max(add(original, projected)?);
                    let axes = raw
                        .numerical_family_key_for_universe(scope)
                        .map_err(|reason| error(format!("preselection scope axes: {reason:?}")))?
                        .basis_axes();
                    maximum_scoped_axes = maximum_scoped_axes.max(axes);
                    axes
                }
                None => fact.axes.len(),
            };
            retained = add(retained, mul(axes, std::mem::size_of::<f64>())?)?;
        }
    }
    // Only scoped representative selection adds a positive-minimum array and
    // one extra obligation per axis. Reserve those alongside the old selector
    // bound without changing the raw fallback's memory admission.
    retained = add(
        retained,
        add(
            vector_peak_bytes::<f64>(maximum_scoped_axes)?,
            vector_peak_bytes::<bool>(maximum_scoped_axes)?,
        )?,
    )?;
    let grouping = memory::grouping_peak(opportunities)?;
    if add(retained, projection_peak.max(grouping))? > maximum_bytes {
        return Ok(None);
    }
    let mut projected_inputs = Vec::with_capacity(inputs.len());
    let mut projected_opportunities = Vec::with_capacity(opportunities.len());
    for (index, facts) in inputs.iter().enumerate() {
        let mut projected_facts = Vec::with_capacity(facts.len());
        for fact in facts {
            projected_facts.push(match &scopes[index] {
                Some(scope) => {
                    let original = fact.original.as_ref().unwrap();
                    let projected = original
                        .as_ref()
                        .clone()
                        .with_algorithm_universe(scope)
                        .map_err(|reason| {
                            error(format!("preselection checked projection: {reason:?}"))
                        })?;
                    facts_from_input(&projected, std::sync::Arc::clone(original))
                        .map_err(|reason| error(format!("preselection facts: {reason:?}")))?
                }
                None => fact.clone(),
            });
        }
        projected_opportunities.push(match &scopes[index] {
            Some(scope) => composition::project_opportunity(&opportunities[index], facts, scope)?,
            None => opportunities[index].clone(),
        });
        projected_inputs.push(projected_facts);
    }
    // Allocator capacities, not requested lengths, determine the retained view.
    let mut actual = add(
        mul(
            scopes.capacity(),
            std::mem::size_of::<Option<DeclaredAlgorithmUniverseV1>>(),
        )?,
        declarations,
    )?;
    actual = add(
        actual,
        mul(
            projected_inputs.capacity(),
            std::mem::size_of::<Vec<CheckedInputFacts>>(),
        )?,
    )?;
    actual = add(
        actual,
        mul(
            projected_opportunities.capacity(),
            std::mem::size_of::<CaseOpportunity>(),
        )?,
    )?;
    for (facts, opportunity) in projected_inputs.iter().zip(&projected_opportunities) {
        actual = add(
            actual,
            mul(facts.capacity(), std::mem::size_of::<CheckedInputFacts>())?,
        )?;
        actual = add(
            actual,
            source_inputs::opportunity_heap_bytes(opportunity)
                .ok_or_else(|| error("preselection opportunity capacity overflow"))?,
        )?;
        for fact in facts {
            actual = add(
                actual,
                mul(fact.axes.capacity(), std::mem::size_of::<f64>())?,
            )?;
        }
    }
    actual = add(
        actual,
        add(
            vector_peak_bytes::<f64>(maximum_scoped_axes)?,
            vector_peak_bytes::<bool>(maximum_scoped_axes)?,
        )?,
    )?;
    if actual > retained {
        return Ok(None);
    }
    let groups = member_groups(&projected_opportunities)
        .map_err(|reason| error(format!("preselection members: {reason:?}")))?;
    let plan = memory::plan(
        &groups,
        &projected_opportunities,
        &projected_inputs,
        remaining_requests,
        Some(&population.settings),
    )?;
    let selection_peak = add(
        plan.required_peak_bytes,
        composition::extra_peak(&projected_opportunities, seed, plan.guaranteed_groups)?,
    )?;
    if add(retained, selection_peak)? > maximum_bytes {
        return Ok(None);
    }
    Ok(Some(PreparedInputs {
        inputs: projected_inputs,
        opportunities: projected_opportunities,
        scopes,
        reserved_bytes: retained,
    }))
}
