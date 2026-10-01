//! Exact zero-coordinate compaction for the explicitly versioned readiness V3.
//! The original numerical fit, axes and certificates are never compacted.
//! Every prefix is scanned again: a previously zero column may become positive.
use super::*;

#[cfg(test)]
mod tests;

pub(in crate::implementations::continuous::cost_model::structured_v2) fn input_geometry_readiness_v3(
    samples: &[FitRow<'_>],
    settings: &StructuredSettingsV2,
    work: &mut GeometryWork,
) -> Result<usize> {
    let dims = samples
        .first()
        .ok_or(StructuredUnknown::InsufficientSamples)?
        .basis
        .len();
    let n = samples.len();
    if dims == 0 || dims > settings.max_axes || n > settings.max_phase_samples {
        return Err(StructuredUnknown::Capacity);
    }
    // Original row shapes and values, zero-mask initialization, index scan and
    // index writes. This work is paid even when no column can be removed.
    work.charge(
        n.checked_mul(dims)
            .and_then(|v| v.checked_add(n))
            .and_then(|v| v.checked_add(dims.checked_mul(3)?))
            .ok_or(StructuredUnknown::Capacity)?,
    )?;
    let mut nonzero = vec![false; dims];
    for sample in samples {
        if sample.basis.len() != dims {
            return Err(StructuredUnknown::InvalidInput);
        }
        for (present, &value) in nonzero.iter_mut().zip(sample.basis) {
            if !value.is_finite() || value < 0.0 || value > (1u64 << 53) as f64 {
                return Err(StructuredUnknown::InvalidInput);
            }
            *present |= value != 0.0;
        }
    }
    let mut columns = Vec::with_capacity(dims);
    for (axis, present) in nonzero.into_iter().enumerate() {
        if present {
            columns.push(axis);
        }
    }
    if columns.is_empty() {
        return Err(StructuredUnknown::Numerical);
    }
    if columns.len() == dims {
        // Reuse the original kernel without allocating another dense matrix.
        return input_geometry_readiness_v2(samples, settings, work);
    }
    // Exact conversion and the owned/borrowed row headers. The V2 kernel below
    // separately pays all validation, normalization, cache and refresh visits
    // over these retained columns. No old budget or visit counter is reset.
    work.charge(
        n.checked_mul(columns.len())
            .and_then(|v| v.checked_add(n.checked_mul(2)?))
            .ok_or(StructuredUnknown::Capacity)?,
    )?;
    let compact: Vec<Vec<f64>> = samples
        .iter()
        .map(|sample| columns.iter().map(|&axis| sample.basis[axis]).collect())
        .collect();
    let rows: Vec<_> = compact
        .iter()
        .map(|basis| FitRow { basis, wall_ns: 0 })
        .collect();
    // Removing identically zero coordinates preserves every nonzero coordinate
    // and its order/scale. Unlike deleting collinear or near-zero columns, it
    // leaves norms, dot products and the original tolerance tests unchanged.
    input_geometry_readiness_v2(&rows, settings, work)
}
