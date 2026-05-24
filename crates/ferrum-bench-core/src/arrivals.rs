//! Poisson arrival-time generation for open-loop benchmarking.
//!
//! See `docs/bench/PLAYBOOK.md` § 2 Scenario A: goodput is only
//! meaningful under open-loop arrivals. We sample inter-arrival times
//! from an `Exp(rate)` distribution via inverse-CDF transform, which
//! gives a Poisson process at the given rate (requests/second).
//!
//! Implemented directly on `rand::Rng::random` to avoid a dependency
//! on `rand_distr` (version mismatch with workspace `rand = 0.9`).

use rand::Rng;

/// Sample one exponential inter-arrival time. Inverse-CDF: `-ln(1 - U) / λ`
/// where `U ~ Uniform[0, 1)`. We use `1 - U ∈ (0, 1]` to avoid `ln(0) = -∞`.
#[inline]
fn sample_exp(rate: f64, rng: &mut impl Rng) -> f64 {
    let u: f64 = rng.random();
    -((1.0 - u).ln()) / rate
}

/// Generate `n` inter-arrival times (seconds) for a Poisson(`rate`)
/// process. Each value is the gap to the next arrival.
///
/// # Panics
/// Panics if `rate <= 0`.
pub fn poisson_inter_arrivals(rate: f64, n: usize, rng: &mut impl Rng) -> Vec<f64> {
    assert!(rate > 0.0, "rate must be positive");
    (0..n).map(|_| sample_exp(rate, rng)).collect()
}

/// Cumulative arrival times (seconds, starting from t=0) for `n`
/// requests in a Poisson(`rate`) process.
///
/// # Panics
/// Panics if `rate <= 0`.
pub fn poisson_arrival_times(rate: f64, n: usize, rng: &mut impl Rng) -> Vec<f64> {
    assert!(rate > 0.0, "rate must be positive");
    let mut t = 0.0_f64;
    (0..n)
        .map(|_| {
            t += sample_exp(rate, rng);
            t
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn inter_arrival_mean_approximates_inverse_rate() {
        // For Exp(λ), E[X] = 1/λ. With 5000 samples we expect ~1-2% relative error.
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let rate = 20.0;
        let xs = poisson_inter_arrivals(rate, 5000, &mut rng);
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let expected = 1.0 / rate;
        let rel_err = (mean - expected).abs() / expected;
        assert!(
            rel_err < 0.05,
            "rel err {rel_err} too large (mean={mean}, expected={expected})"
        );
    }

    #[test]
    fn inter_arrival_variance_approximates_inverse_rate_squared() {
        // For Exp(λ), Var[X] = 1/λ² (so stddev = 1/λ).
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let rate = 10.0;
        let xs = poisson_inter_arrivals(rate, 5000, &mut rng);
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
        let stddev = var.sqrt();
        let expected_sd = 1.0 / rate;
        let rel_err = (stddev - expected_sd).abs() / expected_sd;
        assert!(rel_err < 0.05, "stddev rel err {rel_err} too large");
    }

    #[test]
    fn arrival_times_monotonic() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let ts = poisson_arrival_times(5.0, 100, &mut rng);
        for w in ts.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[test]
    #[should_panic(expected = "rate must be positive")]
    fn rejects_zero_rate() {
        let mut rng = rand::rng();
        let _ = poisson_inter_arrivals(0.0, 10, &mut rng);
    }

    #[test]
    fn deterministic_with_seed() {
        let mut a = rand::rngs::StdRng::seed_from_u64(99);
        let mut b = rand::rngs::StdRng::seed_from_u64(99);
        let xa = poisson_inter_arrivals(5.0, 20, &mut a);
        let xb = poisson_inter_arrivals(5.0, 20, &mut b);
        assert_eq!(xa, xb);
    }
}
