//! Cold span preservation over original checked case inputs.
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    input_geometry_pivot_scratch_bytes_v1, input_geometry_pivots_v1, StructuredSettingsV2,
};

#[derive(Debug, Clone, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct InputGeometryAudit {
    pub candidate_rank: Option<usize>,
    pub selected_rank: Option<usize>,
    pub added_original_cases: usize,
    pub complete: bool,
    pub visits: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct InputGeometryCharge {
    pub cumulative_visits: u64,
    pub maximum_visits: u64,
    pub exhausted: bool,
}

#[cfg(test)]
pub(super) fn scratch_bound(
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    settings: &StructuredSettingsV2,
) -> Result<usize> {
    let (_, n) = selection_inventory_cardinality(opportunities)?;
    let d = inputs
        .iter()
        .flatten()
        .map(|facts| facts.axes.len())
        .max()
        .unwrap_or(0);
    scratch_for_dimensions(n, d, settings.max_rank)
}

pub(super) fn scratch_for_dimensions(n: usize, d: usize, maximum_rank: usize) -> Result<usize> {
    // Borrowed row headers and mandatory original row indices coexist with
    // one geometry decomposition; no complete query or trial matrix is cloned.
    input_geometry_pivot_scratch_bytes_v1(n, d, maximum_rank)
        .and_then(|bytes| bytes.checked_add(n.checked_mul(std::mem::size_of::<&[f64]>())?))
        .and_then(|bytes| bytes.checked_add(n.checked_mul(std::mem::size_of::<usize>())?))
        .and_then(|bytes| bytes.checked_add(2 * std::mem::size_of::<Vec<usize>>()))
        .ok_or_else(|| error("cold input geometry scratch overflow"))
}

pub(super) fn extend(
    candidates: &[usize],
    inputs: &[Vec<CheckedInputFacts>],
    selected: &mut Vec<usize>,
    settings: &StructuredSettingsV2,
    work: &mut StructuredInputGeometryWorkV1,
    maximum_scratch_bytes: usize,
) -> (InputGeometryAudit, Option<SelectionGapReason>) {
    let before = work.visits();
    let original_count = selected.len();
    let mut audit = InputGeometryAudit {
        candidate_rank: None,
        selected_rank: None,
        added_original_cases: 0,
        complete: false,
        visits: 0,
    };
    let result = (|| -> std::result::Result<(), StructuredUnknownV2> {
        let rows: Vec<_> = candidates
            .iter()
            .map(|&i| inputs[i][0].axes.as_slice())
            .collect();
        let mut anchors = Vec::with_capacity(selected.len());
        for index in selected.iter() {
            anchors.push(
                candidates
                    .binary_search(index)
                    .map_err(|_| StructuredUnknownV2::InvalidInput)?,
            );
        }
        // The required endpoint/branch anchors seed one basis. Extend that
        // basis over remaining original inputs without repeated decompositions.
        let geometry =
            input_geometry_pivots_v1(&rows, &anchors, settings, work, maximum_scratch_bytes)?;
        audit.candidate_rank = Some(geometry.rank);
        for &pivot in &geometry.pivot_indices[geometry.anchor_rank..] {
            let index = candidates[pivot];
            if !selected.contains(&index) {
                selected.push(index);
            }
        }
        audit.selected_rank = Some(geometry.rank);
        audit.complete = true;
        Ok(())
    })();
    selected.sort_unstable();
    audit.added_original_cases = selected.len() - original_count;
    audit.visits = work.visits() - before;
    let gap = result
        .err()
        .map(|reason| SelectionGapReason::InputGeometryUnavailable {
            reason,
            work_exhausted: work.exhausted(),
        });
    (audit, gap)
}

pub(super) fn serialize_reason<S: serde::Serializer>(
    reason: &StructuredUnknownV2,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    serializer.collect_str(&format_args!("{reason:?}"))
}
