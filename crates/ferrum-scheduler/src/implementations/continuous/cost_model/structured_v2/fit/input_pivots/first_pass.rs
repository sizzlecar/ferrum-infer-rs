//! Cold pivots with the first orthogonalization pass retained in the existing
//! normalized rows. Numerical Fit continues to use its original geometry core.
use super::*;

pub(super) fn input_geometry_first_pass(
    samples: &[FitRow<'_>],
    settings: &StructuredSettingsV2,
    work: &mut GeometryWork,
    anchor_indices: &[usize],
) -> Result<Vec<usize>> {
    if samples.is_empty() {
        return Err(StructuredUnknown::InsufficientSamples);
    }
    let dims = samples[0].basis.len();
    if dims == 0 || dims > settings.max_axes {
        return Err(StructuredUnknown::Capacity);
    }
    work.charge(
        samples
            .len()
            .checked_mul(dims)
            .and_then(|value| value.checked_mul(3))
            .ok_or(StructuredUnknown::Capacity)?,
    )?;
    let mut scale = vec![1.0_f64; dims];
    for sample in samples {
        if sample.basis.len() != dims {
            return Err(StructuredUnknown::InvalidInput);
        }
        for (s, x) in scale.iter_mut().zip(sample.basis) {
            *s = s.max(*x);
        }
    }
    let mut rows: Vec<Vec<f64>> = samples
        .iter()
        .map(|sample| {
            sample
                .basis
                .iter()
                .zip(&scale)
                .map(|(x, s)| x / s)
                .collect()
        })
        .collect();
    // This norm belongs to the original normalized rows, before any retained
    // first-pass residual replaces their coordinates.
    let largest_norm = rows.iter().map(|row| norm(row)).fold(0.0_f64, f64::max);
    if largest_norm == 0.0 {
        return Err(StructuredUnknown::Numerical);
    }
    let maximum_rank = settings.max_rank.min(samples.len()).min(dims);
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(maximum_rank);
    let mut pivot_indices = Vec::with_capacity(maximum_rank);
    let mut anchor_mask = (!anchor_indices.is_empty()).then(|| vec![false; rows.len()]);
    if let Some(mask) = &mut anchor_mask {
        for &index in anchor_indices {
            *mask.get_mut(index).ok_or(StructuredUnknown::InvalidInput)? = true;
        }
    }
    let mut anchor_pass = anchor_mask.is_some();
    let mut anchor_first_pass_rank = 0_usize;
    let mut other_first_pass_rank = 0_usize;
    loop {
        let rank = basis.len();
        let mut pivot = vec![0.0; dims];
        let mut largest = 0.0;
        let mut pivot_index = 0;
        for (row_index, row) in rows.iter_mut().enumerate() {
            let is_anchor = anchor_mask.as_ref().is_some_and(|mask| mask[row_index]);
            if anchor_pass && !is_anchor {
                continue;
            }
            let done = if is_anchor {
                anchor_first_pass_rank
            } else {
                other_first_pass_rank
            };
            // Charge the complete row operation before changing its retained
            // state. A partial failed round is discarded with this whole call.
            let remaining = rank
                .checked_sub(done)
                .ok_or(StructuredUnknown::InvalidInput)?;
            let operations = remaining
                .checked_mul(2)
                .and_then(|first| {
                    rank.checked_mul(2)
                        .and_then(|second| first.checked_add(second))
                })
                .and_then(|value| value.checked_add(2))
                .ok_or(StructuredUnknown::Capacity)?;
            work.charge(
                dims.checked_mul(operations)
                    .ok_or(StructuredUnknown::Capacity)?,
            )?;
            // Old basis directions never change. Continuing this first pass
            // executes the same dot/subtract sequence as restarting from the
            // original normalized row against the entire basis.
            for direction in &basis[done..] {
                let component = dot(row, direction);
                subtract(row, direction, component);
            }
            let mut residual = row.clone();
            // The second pass is always complete and may only mutate the
            // clone; it must never become the next round's retained state.
            for direction in &basis {
                let component = dot(&residual, direction);
                subtract(&mut residual, direction, component);
            }
            let size = norm(&residual);
            if size > largest {
                largest = size;
                pivot = residual;
                pivot_index = row_index;
            }
        }
        // Every visited row in a group now has exactly this first-pass prefix.
        // Non-anchor rows remain untouched throughout the anchor-only phase.
        anchor_first_pass_rank = rank;
        if !anchor_pass {
            other_first_pass_rank = rank;
        }
        let relative = largest / largest_norm;
        if relative <= DEPENDENT {
            if anchor_pass {
                anchor_pass = false;
                continue;
            }
            break;
        }
        if relative < MIN_PIVOT {
            return Err(StructuredUnknown::IllConditioned);
        }
        if basis.len() == settings.max_rank {
            return Err(StructuredUnknown::Capacity);
        }
        for value in &mut pivot {
            *value /= largest;
        }
        basis.push(pivot);
        pivot_indices.push(pivot_index);
    }
    let rank = basis.len();
    // Keep the original cold core's overflow and zero-rank rejection even
    // though cold request selection does not require sample redundancy.
    rank.checked_add(settings.min_fit_redundancy)
        .ok_or(StructuredUnknown::InvalidSettings)?;
    if rank == 0 {
        return Err(StructuredUnknown::InsufficientRedundancy);
    }
    Ok(pivot_indices)
}
