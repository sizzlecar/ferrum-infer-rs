//! Input-only rank readiness. Cached residuals choose candidates, never certify
//! a direction or a completed span. Reordering finite-precision operations may
//! change pivots, so this is used only by explicitly versioned readiness V2.
//! Final numerical fitting keeps the original input_geometry implementation.
use super::*;

#[cfg(test)]
mod tests;

/// The caller reserves readiness_scratch_bytes before allocating converted
/// inputs or this scratch. Peak ownership here is one n*d cache, at most n*d
/// basis coordinates, n norms, n+r vector headers, scale and two d-vectors.
pub(in crate::implementations::continuous::cost_model::structured_v2) fn input_geometry_readiness_v2(
    samples: &[FitRow<'_>],
    settings: &StructuredSettingsV2,
    work: &mut GeometryWork,
) -> Result<usize> {
    let first = samples
        .first()
        .ok_or(StructuredUnknown::InsufficientSamples)?;
    let dims = first.basis.len();
    if dims == 0 || dims > settings.max_axes || samples.len() > settings.max_phase_samples {
        return Err(StructuredUnknown::Capacity);
    }
    if samples.iter().any(|row| row.basis.len() != dims) {
        return Err(StructuredUnknown::InvalidInput);
    }
    let n = samples.len();
    // Validation/maxima, normalization, norms, and cache/header initialization.
    work.charge(
        n.checked_mul(dims)
            .and_then(|v| v.checked_mul(5))
            .and_then(|v| v.checked_add(dims))
            .and_then(|v| v.checked_add(n))
            .ok_or(StructuredUnknown::Capacity)?,
    )?;
    let mut scale = vec![1.0_f64; dims];
    for sample in samples {
        for (maximum, &value) in scale.iter_mut().zip(sample.basis) {
            if !value.is_finite() || value < 0.0 || value > (1u64 << 53) as f64 {
                return Err(StructuredUnknown::InvalidInput);
            }
            *maximum = maximum.max(value);
        }
    }
    let mut residuals = Vec::with_capacity(n);
    let mut norms = Vec::with_capacity(n);
    let mut largest_norm = 0.0_f64;
    for sample in samples {
        let row = normalized(sample.basis, &scale);
        let length = checked_norm(&row)?;
        largest_norm = largest_norm.max(length);
        residuals.push(row);
        norms.push(length);
    }
    if largest_norm == 0.0 {
        return Err(StructuredUnknown::Numerical);
    }
    let maximum_rank = settings.max_rank.min(n).min(dims);
    let mut basis = Vec::with_capacity(maximum_rank);
    loop {
        let candidate = largest_index(&norms, work)?;
        let (mut pivot, mut length) =
            original_residual(samples[candidate].basis, &scale, &basis, work)?;
        // A small or suspect cached candidate cannot certify dependency. Refresh
        // every original row, including already selected rows, before stopping.
        if norms[candidate] / largest_norm <= DEPENDENT || length / largest_norm < MIN_PIVOT {
            refresh_all(samples, &scale, &basis, &mut residuals, &mut norms, work)?;
            let candidate = largest_index(&norms, work)?;
            length = norms[candidate];
            let relative = length / largest_norm;
            if relative <= DEPENDENT {
                break;
            }
            if relative < MIN_PIVOT {
                return Err(StructuredUnknown::IllConditioned);
            }
            work.charge(dims)?;
            pivot = residuals[candidate].clone();
        }
        if basis.len() == maximum_rank {
            return Err(StructuredUnknown::Capacity);
        }
        // Conservatively reject a basis that lost orthogonality. Two passes
        // alone are not an unconditional stability proof for singular inputs.
        work.charge(
            dims.checked_mul(basis.len() + 2)
                .ok_or(StructuredUnknown::Capacity)?,
        )?;
        for value in &mut pivot {
            *value /= length;
        }
        if (dot(&pivot, &pivot) - 1.0).abs() > DEPENDENT
            || basis.iter().any(|q| dot(&pivot, q).abs() > DEPENDENT)
        {
            return Err(StructuredUnknown::IllConditioned);
        }
        // Cache actual vectors, not a subtraction of squared norms. Cancellation
        // in a norm downdate must never hide an unobserved direction.
        for (row, cached_norm) in residuals.iter_mut().zip(&mut norms) {
            work.charge(dims.checked_mul(5).ok_or(StructuredUnknown::Capacity)?)?;
            for _ in 0..2 {
                let component = dot(row, &pivot);
                subtract(row, &pivot, component);
            }
            *cached_norm = checked_norm(row)?;
        }
        basis.push(pivot);
    }
    let rank = basis.len();
    let required = rank
        .checked_add(settings.min_fit_redundancy)
        .ok_or(StructuredUnknown::InvalidSettings)?;
    if rank == 0 || n < required {
        return Err(StructuredUnknown::InsufficientRedundancy);
    }
    Ok(rank)
}

fn normalized(row: &[f64], scale: &[f64]) -> Vec<f64> {
    row.iter().zip(scale).map(|(x, s)| x / s).collect()
}

fn checked_norm(row: &[f64]) -> Result<f64> {
    let value = norm(row);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(StructuredUnknown::Numerical)
    }
}

fn largest_index(norms: &[f64], work: &mut GeometryWork) -> Result<usize> {
    work.charge(norms.len())?;
    let mut selected = 0;
    for i in 1..norms.len() {
        // Stable original order resolves exact ties.
        if norms[i] > norms[selected] {
            selected = i;
        }
    }
    Ok(selected)
}

fn original_residual(
    row: &[f64],
    scale: &[f64],
    basis: &[Vec<f64>],
    work: &mut GeometryWork,
) -> Result<(Vec<f64>, f64)> {
    work.charge(
        basis
            .len()
            .checked_mul(4)
            .and_then(|v| v.checked_add(2))
            .and_then(|v| v.checked_mul(row.len()))
            .ok_or(StructuredUnknown::Capacity)?,
    )?;
    let mut residual = normalized(row, scale);
    orthogonalize(&mut residual, basis);
    let length = checked_norm(&residual)?;
    Ok((residual, length))
}

fn refresh_all(
    samples: &[FitRow<'_>],
    scale: &[f64],
    basis: &[Vec<f64>],
    residuals: &mut [Vec<f64>],
    norms: &mut [f64],
    work: &mut GeometryWork,
) -> Result<()> {
    for ((sample, residual), length) in samples.iter().zip(residuals).zip(norms) {
        let (original, norm) = original_residual(sample.basis, scale, basis, work)?;
        *residual = original;
        *length = norm;
    }
    Ok(())
}
