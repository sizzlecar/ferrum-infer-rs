use super::*;
const AXES: usize = 7;
/// Fixed optimizer protocol is part of MODEL_REVISION. This is a bounded
/// empirical fit, not a convergence or worst-case latency theorem.
const SWEEPS: usize = 128;
const RIDGE: f64 = 1e-6;
fn axes(input: &StatisticalModelInputV1) -> [f64; AXES] {
    let d = input.device();
    let h = input.host_and_sequence();
    [
        1.,
        d.inner_work_units as f64,
        d.padded_units as f64,
        h.attention_pairs as f64,
        h.rows as f64,
        h.sampling_history_sum as f64,
        h.decoded_text_bytes_sum as f64,
    ]
}
#[derive(Debug, Clone)]
pub(super) struct Affine {
    scale: [f64; AXES],
    beta: [f64; AXES],
}
impl Affine {
    pub(super) fn parameter_bits(&self) -> impl Iterator<Item = u64> + '_ {
        self.scale.iter().chain(&self.beta).map(|v| v.to_bits())
    }

    pub(super) fn fit(points: &[Point]) -> Result<Self, ModelUnknown> {
        if points.is_empty() {
            return Err(ModelUnknown::InsufficientFit);
        }
        let mut scale = [1f64; AXES];
        for (input, _, _) in points {
            for (i, x) in axes(input).into_iter().enumerate() {
                scale[i] = scale[i].max(x);
            }
        }
        let x: Vec<_> = points
            .iter()
            .map(|(input, _, _)| {
                let mut x = axes(input);
                for i in 0..AXES {
                    x[i] /= scale[i];
                }
                x
            })
            .collect();
        let mut beta = [0f64; AXES];
        beta[0] = points.iter().map(|(_, wall, _)| *wall as f64).sum::<f64>() / points.len() as f64;
        let mut predicted = vec![beta[0]; points.len()];
        for _ in 0..SWEEPS {
            for j in 0..AXES {
                let mut numerator = 0.;
                let mut denominator = if j == 0 { 0. } else { RIDGE };
                for i in 0..points.len() {
                    numerator += x[i][j] * (points[i].1 as f64 - predicted[i] + beta[j] * x[i][j]);
                    denominator += x[i][j] * x[i][j];
                }
                if denominator == 0. {
                    continue;
                }
                let next = (numerator / denominator).max(0.);
                if !next.is_finite() {
                    return Err(ModelUnknown::Numerical);
                }
                let delta = next - beta[j];
                beta[j] = next;
                for i in 0..points.len() {
                    predicted[i] += delta * x[i][j];
                }
            }
        }
        Ok(Self { scale, beta })
    }
    pub(super) fn predict(&self, input: &StatisticalModelInputV1) -> Result<u64, ModelUnknown> {
        let value = axes(input)
            .iter()
            .enumerate()
            .map(|(i, x)| x / self.scale[i] * self.beta[i])
            .sum::<f64>();
        if !value.is_finite() || value < 0. || value > (1u64 << 53) as f64 {
            return Err(ModelUnknown::Numerical);
        }
        Ok((value.ceil() as u64).max(1))
    }
}
