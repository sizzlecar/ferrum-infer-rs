use super::*;
use sha2::{Digest, Sha256};

pub(super) fn validate_population(
    samples: &[FitSample<'_>],
    settings: &StructuredSettingsV2,
    envelope: EnvelopeSettings,
) -> Result<(Vec<u64>, super::super::fit::FitGeometry, [u8; 32])> {
    settings.validate()?;
    envelope.validate()?;
    if samples.len() < settings.min_phase_samples {
        return Err(StructuredUnknown::InsufficientSamples);
    }
    if samples.len() > settings.max_phase_samples {
        return Err(StructuredUnknown::Capacity);
    }
    let dims = samples[0].axes.len();
    if dims == 0 || dims > settings.max_axes {
        return Err(StructuredUnknown::Capacity);
    }
    let visits = samples
        .len()
        .checked_mul(dims)
        .and_then(|n| n.checked_mul(SWEEPS))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(StructuredUnknown::Capacity)?;
    if visits > envelope.maximum_coordinate_visits {
        return Err(StructuredUnknown::Capacity);
    }
    let mut maxima = vec![0; dims];
    let mut digest = Sha256::new();
    digest.update(REVISION);
    for v in [
        samples.len() as u64,
        dims as u64,
        SWEEPS as u64,
        FRACTION_BITS as u64,
        envelope.maximum_coordinate_visits,
        envelope.maximum_query_certificates as u64,
        settings.min_phase_samples as u64,
        settings.min_fit_redundancy as u64,
        settings.max_phase_samples as u64,
        settings.max_axes as u64,
        settings.max_rank as u64,
        settings.max_wave_ns,
        settings.max_sample_age_ns,
        settings.static_margin_ns,
    ] {
        digest.update(v.to_le_bytes());
    }
    digest.update(settings.learned_drift.signature());
    for sample in samples {
        if sample.axes.len() != dims
            || sample.axes.first() != Some(&1)
            || sample.wall_ns == 0
            || sample.wall_ns > settings.max_wave_ns
        {
            return Err(StructuredUnknown::InvalidInput);
        }
        digest.update(sample.wall_ns.to_le_bytes());
        for (&value, maximum) in sample.axes.iter().zip(&mut maxima) {
            if value > EXACT_INTEGER_LIMIT {
                return Err(StructuredUnknown::InvalidInput);
            }
            *maximum = (*maximum).max(value);
            digest.update(value.to_le_bytes());
        }
    }
    // Reuse only input geometry, never the legacy unconstrained regression or
    // its prediction-row-space requirement.
    let float_rows: Vec<Vec<f64>> = samples
        .iter()
        .map(|sample| sample.axes.iter().map(|&v| v as f64).collect())
        .collect();
    let rows: Vec<_> = float_rows
        .iter()
        .zip(samples)
        .map(|(basis, sample)| super::super::fit::FitRow {
            basis,
            wall_ns: sample.wall_ns,
        })
        .collect();
    let geometry = super::super::fit::input_geometry(&rows, settings)?;
    Ok((maxima, geometry, digest.finalize().into()))
}

pub(super) fn solve(
    samples: &[FitSample<'_>],
    maxima: &[u64],
    settings: &StructuredSettingsV2,
    _envelope: EnvelopeSettings,
) -> Result<Vec<u128>> {
    // Validation already bounded all rows, dimensions, and coordinate visits.
    let mut coefficients = vec![0.0_f64; maxima.len()];
    coefficients[0] = samples.iter().map(|s| s.wall_ns as f64).sum::<f64>() / samples.len() as f64;
    let mut residual: Vec<_> = samples
        .iter()
        .map(|s| s.wall_ns as f64 - coefficients[0])
        .collect();
    for _ in 0..SWEEPS {
        for (j, &scale) in maxima.iter().enumerate() {
            if scale == 0 {
                continue;
            }
            let (mut numerator, mut denominator) = (0.0_f64, 0.0_f64);
            for (sample, &error) in samples.iter().zip(&residual) {
                let x = sample.axes[j] as f64 / scale as f64;
                numerator += x * error;
                denominator += x * x;
            }
            let next = (coefficients[j] + numerator / denominator).max(0.0);
            if !next.is_finite() {
                return Err(StructuredUnknown::Numerical);
            }
            let delta = next - coefficients[j];
            for (sample, error) in samples.iter().zip(&mut residual) {
                *error -= delta * (sample.axes[j] as f64 / scale as f64);
                if !error.is_finite() {
                    return Err(StructuredUnknown::Numerical);
                }
            }
            coefficients[j] = next;
        }
    }
    let cap = u128::from(settings.max_wave_ns)
        .checked_mul(2 * DENOMINATOR)
        .ok_or(StructuredUnknown::Numerical)?;
    coefficients
        .into_iter()
        .map(|value| {
            let scaled = (value * DENOMINATOR as f64).round();
            if !scaled.is_finite() || scaled < 0.0 || scaled > cap as f64 {
                return Err(StructuredUnknown::Numerical);
            }
            let q = scaled as u128;
            if q > cap {
                return Err(StructuredUnknown::Numerical);
            }
            Ok(q)
        })
        .collect()
}
