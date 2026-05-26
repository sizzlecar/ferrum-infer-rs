//! Statistical aggregates used in `BenchReport`: percentile (linear
//! interpolation), `ScalarStats` (mean / stddev / CI95), and Student-t
//! critical values for small-sample CI.
//!
//! See `docs/bench/PLAYBOOK.md` § 0.4 for the contract: `n_repeats < 3`
//! must omit dispersion fields rather than emit zeros that look like
//! "perfectly consistent" runs.

use serde::{Deserialize, Serialize};

/// Aggregated statistics across N independent samples of one quantity.
///
/// For `n < 3` only `mean` is meaningful; `stddev` and `ci95_hw` are
/// reported as 0 and the serializer omits them via `skip_serializing_if`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScalarStats {
    pub mean: f64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub stddev: f64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub ci95_hw: f64,
}

fn is_zero(x: &f64) -> bool {
    *x == 0.0
}

/// Latency-percentile statistics use the same shape as `ScalarStats`.
pub type PercentileStats = ScalarStats;

impl ScalarStats {
    /// Aggregate per-run scalars into mean/stddev/CI95 across runs.
    ///
    /// `xs.len()` is the number of repeats. Uses sample variance
    /// (divide by `n − 1`). CI95 half-width via Student's t (two-sided
    /// 95%, df = n − 1).
    pub fn from_samples(xs: &[f64]) -> Self {
        let n = xs.len();
        if n == 0 {
            return Self {
                mean: 0.0,
                stddev: 0.0,
                ci95_hw: 0.0,
            };
        }
        let mean = xs.iter().sum::<f64>() / n as f64;
        if n < 3 {
            // Schema rule: CI with n < 3 is degenerate; emit mean only.
            return Self {
                mean,
                stddev: 0.0,
                ci95_hw: 0.0,
            };
        }
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let stddev = var.sqrt();
        let ci95_hw = ci95_half_width(stddev, n);
        Self {
            mean,
            stddev,
            ci95_hw,
        }
    }

    /// True if the absolute mean difference exceeds the sum of the two
    /// half-widths — i.e. the difference is statistically significant
    /// at the 95% level (CIs do not overlap).
    pub fn differs_from(&self, other: &Self) -> bool {
        (self.mean - other.mean).abs() > (self.ci95_hw + other.ci95_hw)
    }
}

/// 95% confidence interval half-width via Student's t (two-sided).
/// Returns 0 for n < 3.
pub fn ci95_half_width(stddev: f64, n: usize) -> f64 {
    if n < 3 {
        return 0.0;
    }
    let df = n - 1;
    let t = student_t_975(df);
    t * stddev / (n as f64).sqrt()
}

/// Critical value of Student's t at α = 0.025 (one-sided), i.e. the
/// multiplier for a two-sided 95% confidence interval at the given df.
///
/// Table for df ≤ 29 (covers n_repeats ≤ 30); for larger df we use the
/// N(0,1) approximation `1.96`, which is within 0.5% of the true value
/// for df ≥ 30.
pub fn student_t_975(df: usize) -> f64 {
    const T_TABLE: &[f64] = &[
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045,
    ];
    if df == 0 {
        return f64::INFINITY;
    }
    if df <= T_TABLE.len() {
        return T_TABLE[df - 1];
    }
    1.960
}

/// Continuous-position percentile using linear interpolation between
/// adjacent order statistics — equivalent to numpy.percentile's default
/// linear method, which is what vLLM / SGLang / lmdeploy all use.
///
/// `q ∈ [0, 1]`. Empty input returns 0.
pub fn percentile(xs: &[f64], q: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_known_set() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert!((percentile(&xs, 0.50) - 5.5).abs() < 1e-9);
        assert!((percentile(&xs, 0.00) - 1.0).abs() < 1e-9);
        assert!((percentile(&xs, 1.00) - 10.0).abs() < 1e-9);
        // numpy.percentile(xs, 99) = 9.91
        assert!((percentile(&xs, 0.99) - 9.91).abs() < 1e-9);
    }

    #[test]
    fn percentile_single_element() {
        assert_eq!(percentile(&[42.0], 0.5), 42.0);
        assert_eq!(percentile(&[42.0], 0.99), 42.0);
    }

    #[test]
    fn percentile_empty() {
        assert_eq!(percentile(&[], 0.5), 0.0);
    }

    #[test]
    fn scalar_stats_zero_or_one_sample() {
        assert_eq!(ScalarStats::from_samples(&[]).mean, 0.0);
        let s = ScalarStats::from_samples(&[10.0]);
        assert_eq!(s.mean, 10.0);
        assert_eq!(s.stddev, 0.0);
        assert_eq!(s.ci95_hw, 0.0);
    }

    #[test]
    fn scalar_stats_two_samples_no_ci() {
        // n=2 is below the threshold — emit mean only.
        let s = ScalarStats::from_samples(&[10.0, 12.0]);
        assert_eq!(s.mean, 11.0);
        assert_eq!(s.stddev, 0.0);
        assert_eq!(s.ci95_hw, 0.0);
    }

    #[test]
    fn scalar_stats_five_samples_with_ci() {
        // mean = 100, sample stddev = 5.0
        let s = ScalarStats::from_samples(&[95.0, 100.0, 100.0, 100.0, 105.0]);
        assert!((s.mean - 100.0).abs() < 1e-9);
        // Sample variance = ((25 + 0 + 0 + 0 + 25) / 4) = 12.5 → stddev ≈ 3.5355
        assert!((s.stddev - 12.5_f64.sqrt()).abs() < 1e-9);
        // CI95 = t_4 * stddev / sqrt(5) = 2.776 * 3.5355 / 2.2361
        let expected = 2.776 * 12.5_f64.sqrt() / 5.0_f64.sqrt();
        assert!((s.ci95_hw - expected).abs() < 1e-6);
    }

    #[test]
    fn student_t_table_anchor_points() {
        // Standard reference values.
        assert!((student_t_975(1) - 12.706).abs() < 1e-3);
        assert!((student_t_975(4) - 2.776).abs() < 1e-3);
        assert!((student_t_975(29) - 2.045).abs() < 1e-3);
        // Large df → ~1.96 (Z critical).
        assert!((student_t_975(100) - 1.96).abs() < 1e-2);
    }

    #[test]
    fn differs_from_overlap() {
        // Two non-overlapping CIs → differ.
        let a = ScalarStats {
            mean: 100.0,
            stddev: 5.0,
            ci95_hw: 4.0,
        };
        let b = ScalarStats {
            mean: 110.0,
            stddev: 5.0,
            ci95_hw: 4.0,
        };
        assert!(a.differs_from(&b));
        // Overlap by 1 → not significant.
        let c = ScalarStats {
            mean: 107.0,
            stddev: 5.0,
            ci95_hw: 4.0,
        };
        assert!(!a.differs_from(&c));
    }

    #[test]
    fn json_omits_dispersion_when_zero() {
        let s = ScalarStats {
            mean: 42.0,
            stddev: 0.0,
            ci95_hw: 0.0,
        };
        let j = serde_json::to_string(&s).unwrap();
        assert_eq!(j, r#"{"mean":42.0}"#);
    }

    #[test]
    fn json_includes_dispersion_when_set() {
        let s = ScalarStats {
            mean: 42.0,
            stddev: 1.5,
            ci95_hw: 0.8,
        };
        let j = serde_json::to_string(&s).unwrap();
        assert!(j.contains("\"stddev\":1.5"));
        assert!(j.contains("\"ci95_hw\":0.8"));
    }
}
