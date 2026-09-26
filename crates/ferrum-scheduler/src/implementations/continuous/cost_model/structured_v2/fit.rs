//! Normalized row-space identification followed by two-pass QR least squares.
//! Collinear work counters are allowed. A new direction outside the observed
//! row space is never silently assigned a zero coefficient.
use super::*;

pub(super) struct FitRow<'a> {
    pub basis: &'a [f64],
    pub wall_ns: u64,
}

// These constants are part of MODEL_REVISION, not per-dataset tuning knobs.
const DEPENDENT: f64 = 1e-12;
const MIN_PIVOT: f64 = 1e-7;
const QUERY_TOLERANCE: f64 = 1e-9;

pub(super) struct RowSpaceFit {
    scale: Vec<f64>,
    basis: Vec<Vec<f64>>,
    coefficients: Vec<f64>,
    input_coefficients: Vec<f64>,
    fit_error_floor_ns: u64,
}
impl RowSpaceFit {
    pub(super) fn bind_parameters(&self, digest: &mut sha2::Sha256) {
        use sha2::Digest;
        digest.update(b"row-space-parameters-with-fit-floor-v1\0");
        digest.update(self.fit_error_floor_ns.to_le_bytes());
        digest.update((self.basis.len() as u64).to_le_bytes());
        for values in std::iter::once(&self.scale)
            .chain(self.basis.iter())
            .chain(std::iter::once(&self.coefficients))
        {
            digest.update((values.len() as u64).to_le_bytes());
            for value in values {
                digest.update(value.to_bits().to_le_bytes());
            }
        }
    }
    /// Finite observed positive errors, not a future timing guarantee.
    pub(super) fn fit_error_floor_ns(&self) -> u64 {
        self.fit_error_floor_ns
    }
    pub(super) fn rank(&self) -> usize {
        self.basis.len()
    }
    pub(super) fn fit(samples: &[FitRow<'_>], settings: &StructuredSettingsV2) -> Result<Self> {
        settings.validate()?;
        if samples.len() < settings.min_phase_samples {
            return Err(StructuredUnknown::InsufficientSamples);
        }
        if samples.len() > settings.max_phase_samples {
            return Err(StructuredUnknown::Capacity);
        }
        if samples.iter().any(|s| {
            s.wall_ns == 0
                || s.wall_ns > settings.max_wave_ns
                || s.basis
                    .iter()
                    .any(|v| !v.is_finite() || *v < 0. || *v > (1u64 << 53) as f64)
        }) {
            return Err(StructuredUnknown::InvalidInput);
        }
        let dims = samples[0].basis.len();
        if dims == 0 || dims > settings.max_axes {
            return Err(StructuredUnknown::Capacity);
        }
        let mut scale = vec![1.0_f64; dims];
        for sample in samples {
            if sample.basis.len() != dims {
                return Err(StructuredUnknown::InvalidInput);
            }
            for (s, x) in scale.iter_mut().zip(sample.basis) {
                *s = s.max(*x);
            }
        }
        let rows: Vec<Vec<f64>> = samples
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
        // Freeze the linear map once. Queries never refit or decompose a matrix.
        let input_coefficients = (0..dims)
            .map(|j| {
                basis
                    .iter()
                    .zip(&coefficients)
                    .map(|(b, c)| b[j] * c)
                    .sum::<f64>()
                    / scale[j]
            })
            .collect::<Vec<_>>();
        if input_coefficients.iter().any(|v| !v.is_finite()) {
            return Err(StructuredUnknown::Numerical);
        }
        let mut model = Self {
            scale,
            basis,
            coefficients,
            input_coefficients,
            fit_error_floor_ns: 0,
        };
        // Reject physically invalid fits; never clip a negative model to zero.
        for sample in samples {
            let fitted = model.predict(sample.basis)?;
            model.fit_error_floor_ns = model
                .fit_error_floor_ns
                .max(sample.wall_ns.saturating_sub(fitted));
        }
        Ok(model)
    }
    /// Signed directions are allowed here. Physical inputs are checked separately.
    pub(super) fn identify(&self, input: &[f64]) -> Result<()> {
        if input.len() != self.scale.len() || input.iter().any(|x| !x.is_finite()) {
            return Err(StructuredUnknown::InvalidInput);
        }
        let mut residual: Vec<f64> = input.iter().zip(&self.scale).map(|(x, s)| x / s).collect();
        let original_norm = norm(&residual);
        orthogonalize(&mut residual, &self.basis);
        if norm(&residual) > QUERY_TOLERANCE * original_norm.max(1.0) {
            return Err(StructuredUnknown::UnidentifiedDirection);
        }
        Ok(())
    }
    /// Linear evaluation alone gives no support or identification authority.
    pub(super) fn linear_value(&self, input: &[f64]) -> Result<f64> {
        if input.len() != self.input_coefficients.len() || input.iter().any(|x| !x.is_finite()) {
            return Err(StructuredUnknown::InvalidInput);
        }
        let predicted = dot(input, &self.input_coefficients);
        if !predicted.is_finite() {
            return Err(StructuredUnknown::Numerical);
        }
        Ok(predicted)
    }
    pub(super) fn predict(&self, input: &[f64]) -> Result<u64> {
        if input.iter().any(|x| *x < 0.) {
            return Err(StructuredUnknown::InvalidInput);
        }
        self.identify(input)?;
        let predicted = self.linear_value(input)?;
        if predicted <= 0.0 || predicted > (1u64 << 53) as f64 {
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
