//! Selection ownership stages. Input inventory is charged by the caller.
use super::*;
use populations::PopulationMemberGroup;

/// Runtime-only accounting: no change to source8 manifests or receipts.
#[derive(Debug, Clone, serde::Serialize)]
pub(super) struct SelectionMemoryAudit {
    pub key_mentions: usize,
    pub guaranteed_cases: usize,
    pub distinct_groups: usize,
    pub guaranteed_groups: usize,
    pub maximum_group_cases: usize,
    pub maximum_group_axes: usize,
    pub grouping_peak_bytes: usize,
    pub retained_groups_bytes: usize,
    pub output_bound_bytes: usize,
    pub candidates_scratch_bytes: usize,
    pub representative_scratch_bytes: usize,
    pub geometry_scratch_bytes: usize,
    pub batch_scratch_bytes: usize,
    pub grouped_selection_peak_bytes: usize,
    pub append_peak_bytes: usize,
    pub required_peak_bytes: usize,
}

/// Before grouping exists, authorize its complete worst-case allocation,
/// including original and replacement Vec buffers. Do not authorize unrelated
/// population/batch headers by the number of repeated case mentions.
pub(super) fn grouping_peak(opportunities: &[CaseOpportunity]) -> Result<usize> {
    let (mentions, guaranteed) = selection_inventory_cardinality(opportunities)?;
    add(
        add(
            std::mem::size_of::<Vec<PopulationMemberGroup>>(),
            vector_peak_bytes::<PopulationMemberGroup>(mentions)?,
        )?,
        vector_peak_bytes::<usize>(add(mul(mentions, 18)?, mul(guaranteed, 2)?)?)?,
    )
}

fn groups_retained(groups: &Vec<PopulationMemberGroup>) -> Result<usize> {
    let mut bytes = add(
        std::mem::size_of::<Vec<PopulationMemberGroup>>(),
        mul(
            groups.capacity(),
            std::mem::size_of::<PopulationMemberGroup>(),
        )?,
    )?;
    for group in groups {
        bytes = add(
            bytes,
            mul(
                add(
                    group.guaranteed_case_indices.capacity(),
                    group.possible_case_indices.capacity(),
                )?,
                std::mem::size_of::<usize>(),
            )?,
        )?;
    }
    Ok(bytes)
}

pub(super) fn plan(
    groups: &Vec<PopulationMemberGroup>,
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    remaining_requests: usize,
    geometry_settings: Option<&ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredSettingsV2>,
) -> Result<SelectionMemoryAudit> {
    let (key_mentions, n) = selection_inventory_cardinality(opportunities)?;
    let g = groups.len();
    // Count before changed/priority/numerical-consistency filtering. A rejected
    // group still retains its original member evidence and typed gaps.
    let p = groups
        .iter()
        .filter(|group| !group.guaranteed_case_indices.is_empty())
        .count();
    let mut max_n = 0;
    let mut max_d = 0;
    let mut geometry_scratch = 0;
    for group in groups {
        let rows = group.guaranteed_case_indices.len();
        let mut axes = 0;
        for &index in &group.guaranteed_case_indices {
            let facts = inputs
                .get(index)
                .ok_or_else(|| error("selection group input index differs"))?;
            axes = axes.max(
                facts
                    .iter()
                    .map(|facts| facts.axes.len())
                    .max()
                    .unwrap_or(0),
            );
        }
        max_n = max_n.max(rows);
        max_d = max_d.max(axes);
        if let Some(settings) = geometry_settings.filter(|_| rows != 0) {
            geometry_scratch = geometry_scratch.max(input_geometry::scratch_for_dimensions(
                rows,
                axes,
                settings.max_rank,
            )?);
        }
    }

    let grouping_peak_bytes = grouping_peak(opportunities)?;
    let retained_groups_bytes = groups_retained(groups)?;
    // Each guaranteed Unique case belongs to one population: N bounds total
    // representative entries; P bounds independently allocated vector headers.
    // Keep per-vector minimum capacity, rather than a single aggregate spare.
    let mut output = std::mem::size_of::<CheckedSelection>();
    for extra in [
        vector_peak_bytes::<SelectedPopulation>(p)?,
        vector_peak_bytes::<BatchCandidate>(p)?,
        vector_peak_bytes::<PopulationCandidate>(p)?,
        // Filtered original indices remain beside every population until
        // geometry is processed in coverage order. Include per-Vec minima.
        vector_peak_bytes::<usize>(add(n, mul(p, 4)?)?)?,
        vector_peak_bytes::<SelectedBatch>(p)?,
        mul(vector_peak_bytes::<usize>(add(n, mul(p, 8)?)?)?, 3)?,
        vector_peak_bytes::<SelectionGap>(add(g, add(mul(p, 7)?, 2)?)?)?,
        vector_peak_bytes::<usize>(remaining_requests)?,
    ] {
        output = add(output, extra)?;
    }

    // The current group's candidates survive representatives, geometry and
    // its independent batch plan. Selected indices are also charged in output.
    let candidates_scratch = vector_peak_bytes::<usize>(max_n)?;
    let representative_scratch = add(
        add(
            mul(vector_peak_bytes::<f64>(max_d)?, 2)?,
            vector_peak_bytes::<bool>(add(mul(max_d, 2)?, 14)?)?,
        )?,
        vector_peak_bytes::<usize>(max_n)?,
    )?;

    // A coalescing trial may contain every population and representative.
    // Old candidate batches, trial members and the new combined batch coexist.
    // The nested budget planner sees at most P groups, but up to N cases.
    let mut batch_scratch = mul(std::mem::size_of::<SelectedBatch>(), 2)?;
    for extra in [
        mul(vector_peak_bytes::<usize>(n)?, 8)?,
        vector_peak_bytes::<Case>(n)?,
        vector_peak_bytes::<CaseOpportunity>(n)?,
        vector_peak_bytes::<PopulationMemberGroup>(p)?,
        vector_peak_bytes::<usize>(add(mul(n, 2)?, mul(p, 16)?)?)?,
        mul(vector_peak_bytes::<usize>(n)?, 2)?,
    ] {
        batch_scratch = add(batch_scratch, extra)?;
    }

    // These scratch stages are sequential. The original groups' IntoIter
    // backing remains live through the whole loop, so retain it conservatively.
    let grouped_selection_peak_bytes = add(
        add(retained_groups_bytes, output)?,
        add(
            candidates_scratch,
            representative_scratch
                .max(geometry_scratch)
                .max(batch_scratch),
        )?,
    )?;
    let append_peak_bytes = add(output, vector_peak_bytes::<SelectionGapReason>(5)?)?;
    let required_peak_bytes = grouping_peak_bytes
        .max(grouped_selection_peak_bytes)
        .max(append_peak_bytes);
    Ok(SelectionMemoryAudit {
        key_mentions,
        guaranteed_cases: n,
        distinct_groups: g,
        guaranteed_groups: p,
        maximum_group_cases: max_n,
        maximum_group_axes: max_d,
        grouping_peak_bytes,
        retained_groups_bytes,
        output_bound_bytes: output,
        candidates_scratch_bytes: candidates_scratch,
        representative_scratch_bytes: representative_scratch,
        geometry_scratch_bytes: geometry_scratch,
        batch_scratch_bytes: batch_scratch,
        grouped_selection_peak_bytes,
        append_peak_bytes,
        required_peak_bytes,
    })
}
