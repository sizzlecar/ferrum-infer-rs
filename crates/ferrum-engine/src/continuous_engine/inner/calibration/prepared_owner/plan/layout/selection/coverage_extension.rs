//! Extend an existing source only after both original packing passes. Raw
//! population geometry stays attached to its original representatives; this
//! operation declares no joint span or fitted numerical qualification.
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1;

pub(super) mod joint;

fn compatible(
    a: &BatchCandidate,
    b: &BatchCandidate,
    populations: &[SelectedPopulation],
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    selected_priority: Option<u8>,
) -> bool {
    if selected_priority.is_some_and(|p| a.input_priority != p || b.input_priority != p)
        || a.batch
            .population_indices
            .iter()
            .chain(&b.batch.population_indices)
            .any(|&i| {
                populations[i]
                    .input_geometry
                    .as_ref()
                    .is_none_or(|audit| !audit.complete)
            })
    {
        return false;
    }
    let Some(&first_index) = a.batch.representative_case_indices.first() else {
        return false;
    };
    let Some(first) = inputs[first_index].first() else {
        return false;
    };
    let Some(family) = first.family else {
        return false;
    };
    a.batch
        .representative_case_indices
        .iter()
        .chain(&b.batch.representative_case_indices)
        .all(|&i| {
            cases[i].route == cases[first_index].route
                && !inputs[i].is_empty()
                && matches!(opportunities[i].population, CasePopulation::Unique(_))
                && opportunities[i].minimum_fresh_members > 0
                && inputs[i].iter().all(|fact| {
                    fact.owner.role == first.owner.role
                        && fact.owner.product == first.owner.product
                        && fact.owner.readback == first.owner.readback
                        && fact.homogeneous_host_policy.is_some()
                        && fact.family.is_some_and(|key| {
                            key.workload_domain_signature() == family.workload_domain_signature()
                        })
                        && fact
                            .original
                            .as_ref()
                            .is_some_and(|input| input.algorithm_universe_signature().is_none())
                })
        })
}

/// Borrowed reservation replay: extension must add one of these two original
/// populations, not merely free a slot for an unrelated later source.
fn admission_pair(
    candidates: &[BatchCandidate],
    first: usize,
    second: usize,
    capacity: SelectionCapacity,
    selected_priority: Option<u8>,
    maximum_sources: NonZeroUsize,
) -> Result<(bool, bool)> {
    let mut used = SelectionCapacity::default();
    let mut sources = 0;
    let mut result = (false, false);
    for (index, candidate) in candidates.iter().enumerate().take(second + 1) {
        let admitted = grouping::reserve(
            &candidate.batch,
            candidate.input_priority,
            capacity,
            selected_priority,
            Some(maximum_sources),
            &mut used,
            &mut sources,
        )?;
        if index == first {
            result.0 = admitted;
        }
        if index == second {
            result.1 = admitted;
        }
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn extend(
    candidates: &mut Vec<BatchCandidate>,
    populations: &[SelectedPopulation],
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    trajectories: Option<&inventory::CheckedCaseInventory>,
    prompts: &[usize],
    chunk: usize,
    row_ceiling: Option<NonZeroU32>,
    population: &StructuredServiceDeclarationV7,
    capacity: SelectionCapacity,
    selected_priority: Option<u8>,
    maximum_sources: NonZeroUsize,
    seed: &DeclaredAlgorithmUniverseV1,
    allocation: InputAllocationPolicy,
) -> Result<()> {
    if !matches!(allocation, InputAllocationPolicy::FinitePreferred { .. }) {
        return Ok(());
    }
    let mut first = 0;
    while first < candidates.len() {
        let mut second = first + 1;
        while second < candidates.len() {
            let a = &candidates[first];
            let b = &candidates[second];
            if !compatible(
                a,
                b,
                populations,
                cases,
                opportunities,
                inputs,
                selected_priority,
            ) {
                second += 1;
                continue;
            }
            let admitted = admission_pair(
                candidates,
                first,
                second,
                capacity,
                selected_priority,
                maximum_sources,
            )?;
            let (existing, additional) = match admitted {
                (true, false) => (&a.batch, &b.batch),
                (false, true) => (&b.batch, &a.batch),
                _ => {
                    second += 1;
                    continue;
                }
            };
            let Some(existing_scope) = existing.algorithm_universe.as_ref() else {
                second += 1;
                continue;
            };
            let mut members = a.batch.population_indices.clone();
            members.extend_from_slice(&b.batch.population_indices);
            members.sort_unstable();
            // The same two replacement slots authorized for composition cover
            // the raw trial and new scoped finite plan. No candidate is cloned.
            let raw = batch_plan(
                &members,
                populations,
                cases,
                opportunities,
                prompts,
                chunk,
                row_ceiling,
                population,
            )?;
            let Some(combined) = composition::extension_candidate(
                &raw,
                Some(existing_scope),
                std::iter::once(existing_scope).chain(additional.algorithm_universe.as_ref()),
                populations,
                cases,
                opportunities,
                inputs,
                trajectories,
                prompts,
                chunk,
                row_ceiling,
                population,
                seed,
                allocation,
            )?
            else {
                second += 1;
                continue;
            };
            if !composition::can_schedule(&combined, capacity)
                || !grouping::preserves_scheduled(
                    candidates,
                    first,
                    second,
                    &combined,
                    capacity,
                    selected_priority,
                    Some(maximum_sources),
                    false,
                    true,
                )?
            {
                second += 1;
                continue;
            }
            tracing::info!(
                previous_algorithms = existing_scope.algorithm_count(),
                extended_algorithms = combined
                    .algorithm_universe
                    .as_ref()
                    .unwrap()
                    .algorithm_count(),
                retained_populations = combined.population_indices.len(),
                requests = combined.requests,
                execution_actions = combined.serial_wave_upper_bound,
                declared_offer_rows = combined.declared_offer_row_bound,
                "Automatic finite source extended original algorithm coverage"
            );
            let original = a.original_population_index.min(b.original_population_index);
            let width = a.decode_width_tier.max(b.decode_width_tier);
            candidates[first].batch = combined;
            candidates[first].original_population_index = original;
            candidates[first].decode_width_tier = width;
            candidates.remove(second);
            second = first + 1;
        }
        first += 1;
    }
    Ok(())
}
