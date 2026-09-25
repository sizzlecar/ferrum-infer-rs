//! Frozen linear pending generators and reachable-set extrema.
use super::fit::RowSpaceFit;
use super::input::{position_moments, PendingQuery};
use super::*;

pub(super) struct PendingGenerators {
    values: Vec<f64>,
    identified: Vec<bool>,
}
pub(super) struct EnvelopeBounds {
    pub lower_ns: u64,
    pub upper_ns: u64,
    pub support_lower: Vec<u64>,
    pub support_upper: Vec<u64>,
}
impl PendingGenerators {
    pub fn new(fit: &RowSpaceFit, exemplar: &StructuredInputV2) -> Result<Self> {
        let mut values = Vec::with_capacity(exemplar.owner.rows as usize);
        let mut identified = Vec::with_capacity(exemplar.owner.rows as usize);
        let mut direction = vec![0.; exemplar.basis.len()];
        for p in 0..exemplar.owner.rows {
            direction[exemplar.pending_basis_offset..]
                .copy_from_slice(&position_moments(&[p])?.map(|x| x as f64));
            values.push(fit.linear_value(&direction)?);
            identified.push(match fit.identify(&direction) {
                Ok(()) => true,
                Err(StructuredUnknown::UnidentifiedDirection) => false,
                Err(error) => return Err(error),
            });
        }
        Ok(Self { values, identified })
    }
    pub fn bounds(
        &self,
        fit: &RowSpaceFit,
        input: &StructuredInputV2,
        pending: Option<&PendingQuery>,
    ) -> Result<EnvelopeBounds> {
        let Some(pending) = pending else {
            let value = fit.predict(&input.basis)?;
            return Ok(EnvelopeBounds {
                lower_ns: value,
                upper_ns: value,
                support_lower: input.support.clone(),
                support_upper: input.support.clone(),
            });
        };
        let eligible = &pending.eligible;
        if eligible.len() > self.values.len()
            || eligible.iter().any(|p| *p as usize >= self.values.len())
            || eligible.windows(2).any(|p| p[0] >= p[1])
            || (eligible.is_empty()
                && pending.constraint == HostPendingConstraintV2::NonEmptySubset)
        {
            return Err(StructuredUnknown::InvalidInput);
        }
        let fixed = input
            .pending_positions
            .iter()
            .copied()
            .filter(|p| eligible.binary_search(p).is_err())
            .collect::<Vec<_>>();
        let fixed_moments = position_moments(&fixed)?;
        // The base is only algebra. In a nonempty singleton domain it need not
        // be reachable or separately identified, and is never a query result.
        let mut anchor = input.basis.clone();
        anchor[input.pending_basis_offset..].copy_from_slice(&fixed_moments.map(|n| n as f64));
        let base = fit.linear_value(&anchor)?;
        if pending.constraint == HostPendingConstraintV2::NonEmptySubset {
            let first = position_moments(&[eligible[0]])?;
            for (out, n) in anchor[input.pending_basis_offset..].iter_mut().zip(first) {
                *out += n as f64;
            }
        }
        fit.identify(&anchor)?;
        // With >=2 nonempty eligible choices, their union and singleton
        // differences span every generator. A singleton has no free direction.
        if (pending.constraint == HostPendingConstraintV2::AnySubset || eligible.len() > 1)
            && eligible.iter().any(|p| !self.identified[*p as usize])
        {
            return Err(StructuredUnknown::UnidentifiedDirection);
        }
        let values = eligible.iter().map(|p| self.values[*p as usize]);
        let (lo_delta, hi_delta) = linear_subset_extrema(values, pending.constraint)?;
        let lower = base + lo_delta;
        let upper = base + hi_delta;
        if !lower.is_finite()
            || !upper.is_finite()
            || lower <= 0.
            || upper < lower
            || upper > (1u64 << 53) as f64
        {
            return Err(StructuredUnknown::Numerical);
        }
        let mut support_lower = input.support.clone();
        let mut support_upper = input.support.clone();
        let full = position_moments(eligible)?;
        // Every support coordinate is a nonnegative moment. Nonempty minima
        // are coordinate-wise minima over singleton choices (positions sorted).
        let minimum = if pending.constraint == HostPendingConstraintV2::NonEmptySubset {
            position_moments(&[eligible[0]])?
        } else {
            [0; 3]
        };
        for i in 0..3 {
            support_lower[input.pending_support_offset + i] = fixed_moments[i] + minimum[i];
            support_upper[input.pending_support_offset + i] = fixed_moments[i] + full[i];
        }
        Ok(EnvelopeBounds {
            lower_ns: lower.ceil() as u64,
            upper_ns: upper.ceil() as u64,
            support_lower,
            support_upper,
        })
    }
}
fn linear_subset_extrema(
    values: impl Iterator<Item = f64>,
    constraint: HostPendingConstraintV2,
) -> Result<(f64, f64)> {
    let mut lower = 0.;
    let mut upper = 0.;
    let mut smallest = f64::INFINITY;
    let mut largest = f64::NEG_INFINITY;
    let mut count = 0usize;
    for value in values {
        if !value.is_finite() {
            return Err(StructuredUnknown::Numerical);
        }
        count += 1;
        smallest = smallest.min(value);
        largest = largest.max(value);
        lower += value.min(0.);
        upper += value.max(0.);
    }
    if constraint == HostPendingConstraintV2::NonEmptySubset {
        if count == 0 {
            return Err(StructuredUnknown::InvalidInput);
        }
        if lower == 0. {
            lower = smallest;
        }
        if upper == 0. {
            upper = largest;
        }
    }
    if !lower.is_finite() || !upper.is_finite() {
        return Err(StructuredUnknown::Numerical);
    }
    Ok((lower, upper))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_subset_extrema_cover_every_reachable_combination() {
        for coefficients in [
            vec![3., -7., 2.],
            vec![-2., -3.],
            vec![2., 3.],
            vec![0.],
            vec![-9.],
        ] {
            for constraint in [
                HostPendingConstraintV2::AnySubset,
                HostPendingConstraintV2::NonEmptySubset,
            ] {
                let (low, high) =
                    linear_subset_extrema(coefficients.iter().copied(), constraint).unwrap();
                let start = usize::from(constraint == HostPendingConstraintV2::NonEmptySubset);
                let points = (start..(1 << coefficients.len()))
                    .map(|mask| {
                        coefficients
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| mask & (1 << i) != 0)
                            .map(|(_, v)| *v)
                            .sum::<f64>()
                    })
                    .collect::<Vec<_>>();
                assert_eq!(low, points.iter().copied().fold(f64::INFINITY, f64::min));
                assert_eq!(
                    high,
                    points.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                );
            }
        }
    }
}
