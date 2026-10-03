//! Bounded cold input geometry. No wall value, numerical fit, or query authority.
use super::*;
use std::num::NonZeroU64;

mod first_pass;

/// One nonrefundable allowance shared by all cold selection scans.
pub struct StructuredInputGeometryWorkV1 {
    work: GeometryWork,
}

impl StructuredInputGeometryWorkV1 {
    pub fn new(maximum_visits: NonZeroU64) -> Self {
        Self {
            work: GeometryWork {
                used: 0,
                limit: maximum_visits.get(),
                exhausted: false,
            },
        }
    }

    pub fn visits(&self) -> u64 {
        self.work.used
    }
    pub fn maximum_visits(&self) -> u64 {
        self.work.limit
    }
    pub fn exhausted(&self) -> bool {
        self.work.exhausted
    }
}

/// Original borrowed row indices, never synthetic numerical observations.
#[derive(Debug, PartialEq, Eq)]
pub struct StructuredInputPivotsV1 {
    pub pivot_indices: Vec<usize>,
    pub rank: usize,
    pub anchor_rank: usize,
    pub work_visits: u64,
}

/// Includes normalized rows, basis, scale, simultaneous pivot/residual vectors,
/// vector headers, original-row indices, and the temporary FitRow headers.
/// Caller-owned axes and its borrowed slice headers remain caller charges.
pub fn input_geometry_pivot_scratch_bytes_v1(
    rows: usize,
    axes: usize,
    maximum_rank: usize,
) -> Option<usize> {
    let rank = maximum_rank.min(rows).min(axes);
    rows.checked_add(rank)?
        .checked_add(3)?
        .checked_mul(axes)?
        .checked_mul(std::mem::size_of::<f64>())?
        .checked_add(
            rows.checked_add(rank)?
                .checked_mul(std::mem::size_of::<Vec<f64>>())?,
        )?
        .checked_add(rank.checked_mul(std::mem::size_of::<usize>())?)?
        .checked_add(rows.checked_mul(std::mem::size_of::<FitRow<'_>>())?)?
        .checked_add(std::mem::size_of::<Vec<FitRow<'_>>>())?
        .checked_add(rows.checked_mul(std::mem::size_of::<bool>())?)?
        .checked_add(std::mem::size_of::<Vec<bool>>())?
        .checked_add(std::mem::size_of::<FitGeometry>())?
        .checked_add(std::mem::size_of::<StructuredInputPivotsV1>())
}

/// Assess the span of declared inputs using the original complete-pivoting
/// normalization and thresholds. Rank may be much smaller than axis count.
/// Sorted unique anchor indices are decomposed first; the same basis is then
/// extended over all rows. No candidate triggers a second full decomposition.
/// This deliberately skips sample redundancy: callers are choosing requests,
/// not asserting that actual Fit/Residual/Qualification members already exist.
pub fn input_geometry_pivots_v1(
    rows: &[&[f64]],
    anchor_indices: &[usize],
    settings: &StructuredSettingsV2,
    work: &mut StructuredInputGeometryWorkV1,
    maximum_scratch_bytes: usize,
) -> Result<StructuredInputPivotsV1> {
    input_geometry_pivots_with_first_pass::<true>(
        rows,
        anchor_indices,
        settings,
        work,
        maximum_scratch_bytes,
    )
}

#[cfg(test)]
fn input_geometry_pivots_original_v1(
    rows: &[&[f64]],
    anchor_indices: &[usize],
    settings: &StructuredSettingsV2,
    work: &mut StructuredInputGeometryWorkV1,
    maximum_scratch_bytes: usize,
) -> Result<StructuredInputPivotsV1> {
    input_geometry_pivots_with_first_pass::<false>(
        rows,
        anchor_indices,
        settings,
        work,
        maximum_scratch_bytes,
    )
}

fn input_geometry_pivots_with_first_pass<const RETAIN_FIRST_PASS: bool>(
    rows: &[&[f64]],
    anchor_indices: &[usize],
    settings: &StructuredSettingsV2,
    work: &mut StructuredInputGeometryWorkV1,
    maximum_scratch_bytes: usize,
) -> Result<StructuredInputPivotsV1> {
    settings.validate()?;
    if work.exhausted() {
        return Err(StructuredUnknown::Capacity);
    }
    let first = rows.first().ok_or(StructuredUnknown::InsufficientSamples)?;
    let dims = first.len();
    if dims == 0 || dims > settings.max_axes {
        return Err(StructuredUnknown::Capacity);
    }
    let scratch = input_geometry_pivot_scratch_bytes_v1(rows.len(), dims, settings.max_rank)
        .ok_or(StructuredUnknown::Capacity)?;
    if scratch > maximum_scratch_bytes {
        return Err(StructuredUnknown::Capacity);
    }
    if anchor_indices.windows(2).any(|pair| pair[0] >= pair[1])
        || anchor_indices
            .last()
            .is_some_and(|&index| index >= rows.len())
    {
        return Err(StructuredUnknown::InvalidInput);
    }
    let before = work.visits();
    work.work.charge(
        rows.len()
            .checked_mul(dims)
            .ok_or(StructuredUnknown::Capacity)?,
    )?;
    if rows.iter().any(|row| {
        row.len() != dims
            || row
                .iter()
                .any(|&value| !value.is_finite() || value < 0. || value > (1u64 << 53) as f64)
    }) {
        return Err(StructuredUnknown::InvalidInput);
    }
    let borrowed: Vec<_> = rows
        .iter()
        .map(|basis| FitRow { basis, wall_ns: 0 })
        .collect();
    let pivot_indices = if RETAIN_FIRST_PASS {
        first_pass::input_geometry_first_pass(&borrowed, settings, &mut work.work, anchor_indices)?
    } else {
        input_geometry_core(
            &borrowed,
            settings,
            Some(&mut work.work),
            false,
            anchor_indices,
        )?
        .pivot_indices
    };
    let rank = pivot_indices.len();
    if pivot_indices
        .iter()
        .enumerate()
        .any(|(position, index)| pivot_indices[..position].contains(index))
    {
        return Err(StructuredUnknown::IllConditioned);
    }
    let anchor_rank = pivot_indices
        .iter()
        .take_while(|&&index| anchor_indices.binary_search(&index).is_ok())
        .count();
    Ok(StructuredInputPivotsV1 {
        pivot_indices,
        rank,
        anchor_rank,
        work_visits: work.visits() - before,
    })
}

#[cfg(test)]
#[path = "input_pivots_tests.rs"]
mod tests;
