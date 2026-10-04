//! Journal relatedness is not numerical-family relatedness. Every population
//! keeps its original key, representatives and independent phase obligations.
use super::*;

fn same_requirement(left: &CheckedInputFacts, right: &CheckedInputFacts) -> bool {
    left.family.is_some()
        && right.family.is_some()
        && left.owner.role == right.owner.role
        && left.owner.product == right.owner.product
        && left.owner.readback == right.owner.readback
        && left.homogeneous_host_policy.is_some()
        && left.homogeneous_host_policy == right.homogeneous_host_policy
}

fn homogeneous_decode<'a>(
    indices: &[usize],
    inputs: &'a [Vec<CheckedInputFacts>],
) -> Option<&'a CheckedInputFacts> {
    let first = inputs.get(*indices.first()?)?.first()?;
    // A family exists only after the original input validator accepts ordinary
    // homogeneous Decode. Missing/mixed input evidence cannot acquire this rule.
    indices
        .iter()
        .all(|&index| {
            !inputs[index].is_empty()
                && inputs[index]
                    .iter()
                    .all(|fact| same_requirement(first, fact))
        })
        .then_some(first)
}

pub(super) fn width_tier(
    indices: &[usize],
    cases: &[Case],
    inputs: &[Vec<CheckedInputFacts>],
) -> usize {
    homogeneous_decode(indices, inputs).map_or(0, |_| {
        // Keep a broad population whole: a real width endpoint is not removed
        // just to obtain a cheaper source tier.
        indices
            .iter()
            .map(|&index| cases[index].width)
            .max()
            .unwrap_or(0)
    })
}

fn same_widths(a: &[usize], b: &[usize], cases: &[Case]) -> bool {
    let subset = |left: &[usize], right: &[usize]| {
        left.iter()
            .all(|&i| right.iter().any(|&j| cases[i].width == cases[j].width))
    };
    subset(a, b) && subset(b, a)
}

pub(super) fn related(
    a: &BatchCandidate,
    b: &BatchCandidate,
    cases: &[Case],
    inputs: &[Vec<CheckedInputFacts>],
) -> bool {
    let a_indices = &a.batch.representative_case_indices;
    let b_indices = &b.batch.representative_case_indices;
    match (
        homogeneous_decode(a_indices, inputs),
        homogeneous_decode(b_indices, inputs),
    ) {
        (Some(left), Some(right)) => {
            // Context/template and algorithm IDs may differ inside a journal.
            // They remain separate numerical populations, not a union model.
            a.coverage.same_policy_route(b.coverage)
                && same_requirement(left, right)
                && same_widths(a_indices, b_indices, cases)
        }
        (None, None) => {
            a.coverage == b.coverage && same_representative_inputs(&a.batch, &b.batch, cases)
        }
        // A mixed or exact-prefill source cannot absorb a pure Decode source
        // on template equality and pull its larger widths into the first tier.
        _ => false,
    }
}

/// A predeclared union may retain different complete width sets. It keeps
/// every original representative; only checked common numerical scope can
/// authorize this broader grouping, never ordinary raw journal relatedness.
pub(super) fn related_scope(
    a: &BatchCandidate,
    b: &BatchCandidate,
    inputs: &[Vec<CheckedInputFacts>],
) -> bool {
    match (
        homogeneous_decode(&a.batch.representative_case_indices, inputs),
        homogeneous_decode(&b.batch.representative_case_indices, inputs),
    ) {
        (Some(left), Some(right)) => {
            a.coverage.same_policy_route(b.coverage) && same_requirement(left, right)
        }
        _ => false,
    }
}

/// Complete Decode families may share a journal while retaining distinct
/// host identities and phase members. This does not make them one model.
pub(super) fn independent_families(
    a: &BatchCandidate,
    b: &BatchCandidate,
    populations: &[SelectedPopulation],
    cases: &[Case],
    inputs: &[Vec<CheckedInputFacts>],
    selected_priority: Option<u8>,
) -> bool {
    // In a priority-limited selection every previous merge passed this same
    // test, so the anchor's priority represents all its constituents. With
    // no priority filter the original anchor order is retained, never reranked.
    if selected_priority
        .is_some_and(|selected| a.input_priority != selected || b.input_priority != selected)
        || a.batch
            .population_indices
            .iter()
            .chain(&b.batch.population_indices)
            .any(|&index| {
                populations[index]
                    .input_geometry
                    .as_ref()
                    .is_some_and(|audit| !audit.complete)
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
    let Some(first_family) = first.family else {
        return false;
    };
    let mut distinct_host = false;
    for &index in a
        .batch
        .representative_case_indices
        .iter()
        .chain(&b.batch.representative_case_indices)
    {
        if cases[index].route != cases[first_index].route || inputs[index].is_empty() {
            return false;
        }
        for fact in &inputs[index] {
            if fact.owner.role != first.owner.role
                || fact.owner.product != first.owner.product
                || fact.owner.readback != first.owner.readback
                || fact.homogeneous_host_policy.is_none()
                || fact.family.is_none_or(|family| {
                    family.workload_domain_signature() != first_family.workload_domain_signature()
                })
            {
                return false;
            }
            distinct_host |= fact.homogeneous_host_policy != first.homogeneous_host_policy;
        }
    }
    distinct_host
}

#[allow(clippy::too_many_arguments)]
pub(super) fn reserve(
    batch: &SelectedBatch,
    priority: u8,
    capacity: SelectionCapacity,
    selected_priority: Option<u8>,
    maximum_sources: Option<NonZeroUsize>,
    used: &mut SelectionCapacity,
    sources: &mut usize,
) -> Result<bool> {
    if selected_priority.is_some_and(|selected| selected != priority)
        || maximum_sources.is_some_and(|maximum| *sources >= maximum.get())
        || !composition::can_schedule(batch, capacity.remaining(*used))
    {
        return Ok(false);
    }
    used.charge(batch)?;
    *sources = add(*sources, 1)?;
    Ok(true)
}

/// Keep the exact original scheduled source population even when two source
/// journals become one. Comparing only with global capacity misses both prior
/// spending and later required sources. These borrowed traversals allocate no
/// catalogue and use the same complete-plan predicate as original selection.
#[allow(clippy::too_many_arguments)]
pub(super) fn preserves_scheduled(
    candidates: &[BatchCandidate],
    first: usize,
    second: usize,
    combined: &SelectedBatch,
    capacity: SelectionCapacity,
    selected_priority: Option<u8>,
    maximum_sources: Option<NonZeroUsize>,
    require_both: bool,
    require_combined: bool,
) -> Result<bool> {
    debug_assert!(first < second && second < candidates.len());
    let (mut original, mut proposed) = (SelectionCapacity::default(), SelectionCapacity::default());
    let (mut original_sources, mut proposed_sources) = (0, 0);
    let mut union_scheduled = false;
    let mut coverage_gained = false;
    for (index, candidate) in candidates.iter().enumerate() {
        let before = reserve(
            &candidate.batch,
            candidate.input_priority,
            capacity,
            selected_priority,
            maximum_sources,
            &mut original,
            &mut original_sources,
        )?;
        // Cross-width scope may replace two already affordable complete
        // sources. It must not pull an unselected wide frontier into the
        // first source merely because a larger global ledger still exists.
        if require_both && (index == first || index == second) && !before {
            return Ok(false);
        }
        let after = if index == second {
            union_scheduled
        } else {
            let accepted = reserve(
                if index == first {
                    combined
                } else {
                    &candidate.batch
                },
                candidate.input_priority,
                capacity,
                selected_priority,
                maximum_sources,
                &mut proposed,
                &mut proposed_sources,
            )?;
            if index == first {
                union_scheduled = accepted;
            }
            accepted
        };
        if before && !after {
            return Ok(false);
        }
        coverage_gained |= !before && after;
    }
    // Independent-family packing is useful only if this same traversal
    // admits an additional original population, including a later source
    // that can now use the released journal slot.
    Ok(!require_combined || union_scheduled && coverage_gained)
}
