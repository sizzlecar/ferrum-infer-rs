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

#[allow(clippy::too_many_arguments)]
fn reserve(
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
) -> Result<bool> {
    debug_assert!(first < second && second < candidates.len());
    let (mut original, mut proposed) = (SelectionCapacity::default(), SelectionCapacity::default());
    let (mut original_sources, mut proposed_sources) = (0, 0);
    let mut union_scheduled = false;
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
    }
    Ok(true)
}
