use super::*;

pub(super) fn ceil_div(n: u128, d: u128) -> Result<u128> {
    if d == 0 {
        return Err(StructuredUnknown::Numerical);
    }
    (n / d)
        .checked_add(u128::from(!n.is_multiple_of(d)))
        .ok_or(StructuredUnknown::Numerical)
}

/// Bounds D times the exact dot product without forming a common denominator.
pub(super) fn scaled_dot(
    axes: &[u64],
    maxima: &[u64],
    coefficients: &[u128],
) -> Result<(u128, u128)> {
    if axes.len() != maxima.len() || axes.len() != coefficients.len() {
        return Err(StructuredUnknown::InvalidInput);
    }
    let (mut low, mut high) = (0u128, 0u128);
    for ((&x, &m), &q) in axes.iter().zip(maxima).zip(coefficients) {
        if m == 0 {
            if x != 0 || q != 0 {
                return Err(StructuredUnknown::UnidentifiedDirection);
            }
            continue;
        }
        let term = u128::from(x)
            .checked_mul(q)
            .ok_or(StructuredUnknown::Numerical)?;
        low = low
            .checked_add(term / u128::from(m))
            .ok_or(StructuredUnknown::Numerical)?;
        high = high
            .checked_add(ceil_div(term, u128::from(m))?)
            .ok_or(StructuredUnknown::Numerical)?;
    }
    Ok((low, high))
}

pub(super) fn verify_feasible(
    samples: &[FitSample<'_>],
    maxima: &[u64],
    coefficients: &[u128],
    maximum_wave_ns: u64,
) -> Result<u64> {
    let cap = u128::from(maximum_wave_ns)
        .checked_mul(2 * DENOMINATOR)
        .ok_or(StructuredUnknown::Numerical)?;
    if samples.is_empty()
        || coefficients.len() != maxima.len()
        || coefficients.iter().any(|&q| q > cap)
    {
        return Err(StructuredUnknown::InvalidInput);
    }
    let mut deviation = 0;
    for sample in samples {
        let (low, high) = scaled_dot(sample.axes, maxima, coefficients)?;
        let actual = u128::from(sample.wall_ns)
            .checked_mul(DENOMINATOR)
            .ok_or(StructuredUnknown::Numerical)?;
        deviation = deviation
            .max(low.abs_diff(actual))
            .max(high.abs_diff(actual));
    }
    let epsilon = u64::try_from(ceil_div(deviation, DENOMINATOR)?)
        .map_err(|_| StructuredUnknown::Numerical)?;
    if epsilon > maximum_wave_ns {
        return Err(StructuredUnknown::Numerical);
    }
    Ok(epsilon)
}

pub(super) fn build(
    samples: &[FitSample<'_>],
    epsilon_ns: u64,
    settings: EnvelopeSettings,
) -> Result<Vec<UpperCertificate>> {
    let mut sum = UpperCertificate {
        axes: vec![0; samples[0].axes.len()],
        bound_ns: 0,
    };
    for sample in samples {
        for (total, &x) in sum.axes.iter_mut().zip(sample.axes) {
            *total = total
                .checked_add(u128::from(x))
                .ok_or(StructuredUnknown::Numerical)?;
        }
        sum.bound_ns = sum
            .bound_ns
            .checked_add(u128::from(sample.wall_ns) + u128::from(epsilon_ns))
            .ok_or(StructuredUnknown::Numerical)?;
    }
    // Mandatory aggregate permits axes observed on separate genuine waves.
    let mut out = vec![sum];
    let individual = (settings.maximum_query_certificates - 1).min(samples.len());
    for position in 0..individual {
        // Fixed FIFO positions; measured costs never select the population.
        let index = if individual == 1 {
            0
        } else {
            position * (samples.len() - 1) / (individual - 1)
        };
        let row = &samples[index];
        out.push(UpperCertificate {
            axes: row.axes.iter().map(|&x| u128::from(x)).collect(),
            bound_ns: u128::from(row.wall_ns) + u128::from(epsilon_ns),
        });
    }
    Ok(out)
}

pub(super) fn upper_bound(certificate: &UpperCertificate, input: &[u64]) -> Result<Option<u64>> {
    if input.len() != certificate.axes.len() {
        return Err(StructuredUnknown::InvalidInput);
    }
    let (mut p, mut q) = (0u128, 1u128);
    for (&x, &a) in input.iter().zip(&certificate.axes) {
        if a == 0 {
            if x != 0 {
                return Ok(None);
            }
            continue;
        }
        let left = u128::from(x)
            .checked_mul(q)
            .ok_or(StructuredUnknown::Numerical)?;
        let right = p.checked_mul(a).ok_or(StructuredUnknown::Numerical)?;
        if left > right {
            p = u128::from(x);
            q = a;
        }
    }
    let product = p
        .checked_mul(certificate.bound_ns)
        .ok_or(StructuredUnknown::Numerical)?;
    let upper = ceil_div(product, q)?;
    // A wide but valid certificate is unusable for the existing u64 planning
    // contract. Another certificate may still provide a finite useful bound.
    Ok(u64::try_from(upper).ok())
}
