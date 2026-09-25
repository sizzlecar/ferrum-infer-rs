//! Normalized row-space identification followed by two-pass QR least squares.
//! Collinear work counters are allowed. A new direction outside the observed
//! row space is never silently assigned a zero coefficient.
use super::*;

// These constants are part of MODEL_REVISION, not per-dataset tuning knobs.
const DEPENDENT: f64 = 1e-12;
const MIN_PIVOT: f64 = 1e-7;
const QUERY_TOLERANCE: f64 = 1e-9;

pub(super) struct RowSpaceFit {
    scale: Vec<f64>,
    basis: Vec<Vec<f64>>,
    coefficients: Vec<f64>,
}
impl RowSpaceFit {
    pub(super) fn rank(&self) -> usize {
        self.basis.len()
    }
    pub(super) fn fit(
        samples: &[StructuredNumericObservationV1],
        settings: &StructuredSettingsV1,
    ) -> Result<Self> {
        let dims = samples[0].input.basis.len();
        if dims == 0 || dims > settings.max_axes {
            return Err(StructuredUnknown::Capacity);
        }
        let mut scale = vec![1.0_f64; dims];
        for sample in samples {
            if sample.input.basis.len() != dims {
                return Err(StructuredUnknown::InvalidInput);
            }
            for (s, x) in scale.iter_mut().zip(&sample.input.basis) {
                *s = s.max(*x);
            }
        }
        let rows: Vec<Vec<f64>> = samples
            .iter()
            .map(|sample| {
                sample
                    .input
                    .basis
                    .iter()
                    .zip(&scale)
                    .map(|(x, s)| x / s)
                    .collect()
            })
            .collect();
        let largest_norm = rows.iter().map(|r| norm(r)).fold(0.0_f64, f64::max);
        if largest_norm == 0.0 {
            return Err(StructuredUnknown::Numerical);
        }
        let mut basis: Vec<Vec<f64>> = Vec::new();
        loop {
            // Complete pivoting over the remaining sample directions. Two
            // orthogonalization passes reduce loss from correlated counters.
            let mut pivot = vec![0.0; dims];
            let mut largest = 0.0;
            for row in &rows {
                let mut residual = row.clone();
                orthogonalize(&mut residual, &basis);
                let size = norm(&residual);
                if size > largest {
                    largest = size;
                    pivot = residual;
                }
            }
            let relative = largest / largest_norm;
            if relative <= DEPENDENT {
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
        }
        let rank = basis.len();
        let required = rank
            .checked_add(settings.min_fit_redundancy)
            .ok_or(StructuredUnknown::InvalidSettings)?;
        if rank == 0 || samples.len() < required {
            return Err(StructuredUnknown::InsufficientRedundancy);
        }
        let projected: Vec<Vec<f64>> = rows
            .iter()
            .map(|row| basis.iter().map(|direction| dot(row, direction)).collect())
            .collect();
        let mut q: Vec<Vec<f64>> = Vec::new();
        let mut r = vec![vec![0.0; rank]; rank];
        for j in 0..rank {
            let mut column: Vec<f64> = projected.iter().map(|row| row[j]).collect();
            let original_norm = norm(&column);
            for _ in 0..2 {
                for i in 0..j {
                    let component = dot(&column, &q[i]);
                    r[i][j] += component;
                    subtract(&mut column, &q[i], component);
                }
            }
            let length = norm(&column);
            if !length.is_finite() || length <= MIN_PIVOT * original_norm.max(1.0) {
                return Err(StructuredUnknown::IllConditioned);
            }
            r[j][j] = length;
            for value in &mut column {
                *value /= length;
            }
            q.push(column);
        }
        let targets: Vec<_> = samples.iter().map(|s| s.wall_ns as f64).collect();
        let mut coefficients: Vec<f64> = q.iter().map(|col| dot(col, &targets)).collect();
        for i in (0..rank).rev() {
            let rest: f64 = (i + 1..rank).map(|j| r[i][j] * coefficients[j]).sum();
            coefficients[i] = (coefficients[i] - rest) / r[i][i];
            if !coefficients[i].is_finite() {
                return Err(StructuredUnknown::Numerical);
            }
        }
        let model = Self {
            scale,
            basis,
            coefficients,
        };
        // Reject physically invalid fits; never clip a negative model to zero.
        for sample in samples {
            model.predict(&sample.input.basis)?;
        }
        Ok(model)
    }
    pub(super) fn predict(&self, input: &[f64]) -> Result<u64> {
        if input.len() != self.scale.len() || input.iter().any(|x| !x.is_finite() || *x < 0.) {
            return Err(StructuredUnknown::InvalidInput);
        }
        let scaled: Vec<f64> = input.iter().zip(&self.scale).map(|(x, s)| x / s).collect();
        let mut residual = scaled.clone();
        orthogonalize(&mut residual, &self.basis);
        if norm(&residual) > QUERY_TOLERANCE * norm(&scaled).max(1.0) {
            return Err(StructuredUnknown::UnidentifiedDirection);
        }
        let predicted: f64 = self
            .basis
            .iter()
            .zip(&self.coefficients)
            .map(|(b, c)| dot(&scaled, b) * c)
            .sum();
        if !predicted.is_finite() || predicted <= 0.0 || predicted > (1u64 << 53) as f64 {
            return Err(StructuredUnknown::Numerical);
        }
        Ok(predicted.ceil() as u64)
    }
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}
fn subtract(a: &mut [f64], b: &[f64], scale: f64) {
    for (a, b) in a.iter_mut().zip(b) {
        *a -= scale * b;
    }
}
fn orthogonalize(row: &mut [f64], basis: &[Vec<f64>]) {
    for _ in 0..2 {
        for direction in basis {
            let component = dot(row, direction);
            subtract(row, direction, component);
        }
    }
}
