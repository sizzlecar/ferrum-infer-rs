use super::{ModelError, ScaledSample, TrainingBudget, WorkCounter, FEATURE_COUNT};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Physical {
    /// Coefficients are seconds per scaled feature, including a nonnegative intercept.
    coefficients: [f64; FEATURE_COUNT + 1],
    penalty: f64,
}

impl Physical {
    pub(super) fn predict(&self, x: &[f64; FEATURE_COUNT]) -> f64 {
        self.coefficients[0]
            + x.iter()
                .zip(&self.coefficients[1..])
                .map(|(x, b)| x * b)
                .sum::<f64>()
    }

    pub(super) fn validate(&self) -> Result<(), ModelError> {
        if self.coefficients.iter().any(|b| !b.is_finite() || *b < 0.0)
            || !self.penalty.is_finite()
            || self.penalty < 0.0
        {
            return Err(ModelError::InvalidData("invalid nonnegative regression"));
        }
        Ok(())
    }
}

/// Cyclic exact coordinate minimization of mean squared error plus lambda *
/// squared slope norm, projected onto the nonnegative orthant. Intercept is
/// unpenalized. Finite iteration/resource limits apply even under collinearity.
pub(super) fn train(
    training: &[ScaledSample],
    validation: &[ScaledSample],
    budget: &TrainingBudget,
    work: &mut WorkCounter,
) -> Result<Physical, ModelError> {
    let n = training.len() as f64;
    let mut best = None;
    let mut best_loss = f64::INFINITY;
    let mut residuals = vec![0.0; training.len()];
    for &penalty in &budget.ridge_penalties {
        let mut model = Physical {
            coefficients: [0.0; FEATURE_COUNT + 1],
            penalty,
        };
        for (residual, sample) in residuals.iter_mut().zip(training) {
            *residual = sample.y;
        }
        for _ in 0..budget.ridge_sweeps {
            let mut largest_change: f64 = 0.0;
            for column in 0..=FEATURE_COUNT {
                work.charge(training.len() * 2)?;
                let old = model.coefficients[column];
                let mut numerator = 0.0;
                let mut denominator = if column == 0 { 0.0 } else { n * penalty };
                for (sample, residual) in training.iter().zip(&residuals) {
                    let x = if column == 0 {
                        1.0
                    } else {
                        sample.x[column - 1]
                    };
                    numerator += x * (residual + x * old);
                    denominator += x * x;
                }
                let updated = if denominator == 0.0 {
                    0.0
                } else {
                    (numerator / denominator).max(0.0)
                };
                if !updated.is_finite() {
                    return Err(ModelError::InvalidData("nonfinite regression step"));
                }
                let delta = updated - old;
                model.coefficients[column] = updated;
                for (sample, residual) in training.iter().zip(&mut residuals) {
                    let x = if column == 0 {
                        1.0
                    } else {
                        sample.x[column - 1]
                    };
                    *residual -= x * delta;
                }
                largest_change = largest_change.max(delta.abs());
            }
            if largest_change <= 1e-12 {
                break;
            }
        }
        work.charge(validation.len() * FEATURE_COUNT)?;
        let loss = validation
            .iter()
            .map(|sample| (sample.y - model.predict(&sample.x)).powi(2))
            .sum::<f64>()
            / validation.len() as f64;
        if loss < best_loss {
            best_loss = loss;
            best = Some(model);
        }
    }
    best.ok_or(ModelError::InvalidData("no finite regression candidate"))
}
