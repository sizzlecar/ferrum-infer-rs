//! One bounded, atomic alternative to the complete original source selection.
//! Original geometry remains per population; a common U grants no joint span
//! or fitted support. Every resulting family still needs independent F/R/Q.
use super::*;

fn same_domain(left: &CheckedInputFacts, right: &CheckedInputFacts) -> bool {
    left.owner.role == right.owner.role
        && left.family.zip(right.family).is_some_and(|(left, right)| {
            left.workload_domain_signature() == right.workload_domain_signature()
        })
}

fn same_class(left: &CheckedInputFacts, right: &CheckedInputFacts) -> bool {
    same_domain(left, right)
        && left.owner.product == right.owner.product
        && left.owner.readback == right.owner.readback
        && left.homogeneous_host_policy.is_some()
        && left.homogeneous_host_policy == right.homogeneous_host_policy
}

fn complete_raw(
    batch: &SelectedBatch,
    populations: &[SelectedPopulation],
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
) -> bool {
    !batch.representative_case_indices.is_empty()
        && batch.population_indices.iter().all(|&i| {
            populations[i]
                .input_geometry
                .as_ref()
                .is_some_and(|audit| audit.complete)
        })
        && batch.representative_case_indices.iter().all(|&i| {
            matches!(opportunities[i].population, CasePopulation::Unique(_))
                && opportunities[i].minimum_fresh_members > 0
                && !inputs[i].is_empty()
                && inputs[i].iter().all(|fact| {
                    fact.owner.role
                        == ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredWaveRoleV2::OrdinaryDecode
                        && fact.family.is_some()
                        && fact.homogeneous_host_policy.is_some()
                        && fact
                            .original
                            .as_ref()
                            .is_some_and(|input| input.algorithm_universe_signature().is_none())
                })
        })
}

/// Check all admitted interpretations of each exact constituent class. An
/// algorithm present only in another product's U does not cover this class.
fn extends_class(
    target: &SelectedBatch,
    admitted: &[usize],
    candidates: &[BatchCandidate],
    cases: &[Case],
    inputs: &[Vec<CheckedInputFacts>],
    trajectories: Option<&inventory::CheckedCaseInventory>,
) -> bool {
    let first_index = target.representative_case_indices[0];
    let first = &inputs[first_index][0];
    let mut extends = false;
    for &index in &target.representative_case_indices {
        if cases[index].route != cases[first_index].route {
            return false;
        }
        for fact in &inputs[index] {
            if !same_domain(first, fact) {
                return false;
            }
            for recipe in composition::case_recipes(index, inputs, trajectories) {
                let (mut has_class, mut covered) = (false, false);
                for &source in admitted {
                    let batch = &candidates[source].batch;
                    let Some(scope) = &batch.algorithm_universe else {
                        continue;
                    };
                    let matches = batch.representative_case_indices.iter().any(|&old| {
                        cases[old].route == cases[index].route
                            && inputs[old].iter().any(|old| same_class(fact, old))
                    });
                    if !matches {
                        continue;
                    }
                    has_class = true;
                    match scope.contains_checked_algorithms(recipe) {
                        Ok(true) => covered = true,
                        Ok(false) => {}
                        Err(_) => return false,
                    }
                }
                if !has_class {
                    return false;
                }
                extends |= !covered;
            }
        }
    }
    extends
}

#[allow(clippy::too_many_arguments)]
pub(in super::super) fn extend(
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
    // This single candidate-index allocation has separate optional authorization.
    // It first freezes actual admission, then becomes the replacement set.
    let mut replaced = Vec::with_capacity(candidates.len());
    let (mut used, mut sources) = (SelectionCapacity::default(), 0);
    for (index, candidate) in candidates.iter().enumerate() {
        if grouping::reserve(
            &candidate.batch,
            candidate.input_priority,
            capacity,
            selected_priority,
            Some(maximum_sources),
            &mut used,
            &mut sources,
        )? {
            replaced.push(index);
        }
    }
    // Freeze the first legal unmet constituent in the original ordering.
    // A later capacity failure must not silently choose a cheaper obligation.
    let target = candidates
        .iter()
        .enumerate()
        .find_map(|(index, candidate)| {
            (!replaced.contains(&index)
                && selected_priority.is_none_or(|p| p == candidate.input_priority)
                && complete_raw(&candidate.batch, populations, opportunities, inputs)
                && extends_class(
                    &candidate.batch,
                    &replaced,
                    candidates,
                    cases,
                    inputs,
                    trajectories,
                ))
            .then_some(index)
        });
    let Some(target) = target else {
        return Ok(());
    };
    let target_index = candidates[target].batch.representative_case_indices[0];
    let target_fact = &inputs[target_index][0];
    replaced.retain(|&source| {
        candidates[source]
            .batch
            .representative_case_indices
            .iter()
            .any(|&i| {
                cases[i].route == cases[target_index].route
                    && inputs[i].iter().any(|fact| same_domain(fact, target_fact))
            })
    });
    // A raw sibling cannot be relabelled into a new scoped interpretation.
    // Nor may a mixed source lose constituents outside this physical component.
    if replaced.is_empty()
        || replaced.iter().any(|&source| {
            let batch = &candidates[source].batch;
            batch.algorithm_universe.is_none()
                || !complete_raw(batch, populations, opportunities, inputs)
                || batch.representative_case_indices.iter().any(|&i| {
                    cases[i].route != cases[target_index].route
                        || inputs[i].iter().any(|fact| !same_domain(fact, target_fact))
                })
        })
    {
        return Ok(());
    }
    replaced.push(target);
    replaced.sort_unstable();
    let count = replaced.iter().try_fold(0usize, |n, &i| {
        add(n, candidates[i].batch.population_indices.len())
    })?;
    let mut members = Vec::with_capacity(count);
    for &i in &replaced {
        members.extend_from_slice(&candidates[i].batch.population_indices);
    }
    members.sort_unstable();
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
    // Growth was proved relative to the target's exact class above. The
    // complete baseline's union can already contain these algorithms in a
    // different product; comparing only global algorithm counts loses that fact.
    let Some(combined) = composition::extension_candidate(
        &raw,
        None,
        replaced
            .iter()
            .filter_map(|&i| candidates[i].batch.algorithm_universe.as_ref()),
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
        return Ok(());
    };
    if !grouping::preserves_joint_selection(
        candidates,
        &replaced,
        target,
        &combined,
        capacity,
        selected_priority,
        maximum_sources,
    )? {
        return Ok(());
    }
    let first = replaced[0];
    let original = replaced
        .iter()
        .map(|&i| candidates[i].original_population_index)
        .min()
        .unwrap();
    let width = replaced
        .iter()
        .map(|&i| candidates[i].decode_width_tier)
        .max()
        .unwrap();
    tracing::info!(
        replaced_sources = replaced.len() - 1,
        retained_populations = combined.population_indices.len(),
        algorithms = combined
            .algorithm_universe
            .as_ref()
            .unwrap()
            .algorithm_count(),
        requests = combined.requests,
        execution_actions = combined.serial_wave_upper_bound,
        declared_offer_rows = combined.declared_offer_row_bound,
        "Automatic finite sources jointly retained and extended algorithm coverage"
    );
    candidates[first].batch = combined;
    candidates[first].original_population_index = original;
    candidates[first].decode_width_tier = width;
    let mut index = 0;
    candidates.retain(|_| {
        let keep = index == first || !replaced.contains(&index);
        index += 1;
        keep
    });
    Ok(())
}
