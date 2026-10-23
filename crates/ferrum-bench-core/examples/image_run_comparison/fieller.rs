//! Paired Fieller interval for E(candidate throughput)/E(baseline throughput).
//! Formula: https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/ratimean.htm
//! Assumptions/unbounded sets: https://arxiv.org/pdf/0711.0198 sections 1.1 and 2.
use serde::Serialize;

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    AtLeastParity,
    BelowTarget,
    Insufficient,
}

#[derive(Debug, Serialize)]
pub struct Analysis {
    pub status: Status,
    pub reason: Option<String>,
    pub statistics: Option<Statistics>,
    pub interval: Option<Interval>,
}
#[derive(Debug, Serialize)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
}
#[derive(Debug, Serialize)]
pub struct Statistics {
    pub independent_pairs: usize,
    pub degrees_of_freedom: usize,
    pub baseline_mean_images_per_second: f64,
    pub candidate_mean_images_per_second: f64,
    pub ratio_of_means: f64,
    pub common_rate_scale: f64,
    pub scaled_mean_variance_baseline: f64,
    pub scaled_mean_variance_candidate: f64,
    pub scaled_mean_covariance: f64,
    pub t_critical: f64,
    pub critical_value_policy: &'static str,
}
impl Analysis {
    pub fn insufficient(reason: &str) -> Self {
        Self {
            status: Status::Insufficient,
            reason: Some(reason.into()),
            statistics: None,
            interval: None,
        }
    }
}

pub fn analyze(pairs: &[(f64, f64)]) -> Analysis {
    let n = pairs.len();
    if !(3..=30).contains(&n) {
        return Analysis::insufficient(
            "requires 3..=30 independent pairs; no large-df normal approximation",
        );
    }
    if pairs
        .iter()
        .any(|&(x, y)| !x.is_finite() || !y.is_finite() || x <= 0.0 || y <= 0.0)
    {
        return Analysis::insufficient("whole-run throughputs must be finite and positive");
    }
    let scale = pairs.iter().fold(0.0_f64, |s, &(x, y)| s.max(x).max(y));
    let x = pairs.iter().map(|p| p.0 / scale).sum::<f64>() / n as f64;
    let y = pairs.iter().map(|p| p.1 / scale).sum::<f64>() / n as f64;
    let divisor = (n * (n - 1)) as f64;
    let vx = pairs.iter().map(|p| (p.0 / scale - x).powi(2)).sum::<f64>() / divisor;
    let vy = pairs.iter().map(|p| (p.1 / scale - y).powi(2)).sum::<f64>() / divisor;
    let cov = pairs
        .iter()
        .map(|p| (p.0 / scale - x) * (p.1 / scale - y))
        .sum::<f64>()
        / divisor;
    // Existing df<=29 table is rounded to .001. The upper rounding endpoint
    // is conservative, unlike a rounded-down critical value. No df>29 fallback.
    let t = ferrum_bench_core::student_t_975(n - 1) + 0.0005;
    let a = x * x - t * t * vx;
    let b = x * y - t * t * cov;
    let c = y * y - t * t * vy;
    let d = b * b - a * c;
    let ratio = y / x;
    if !ratio.is_finite() || ![a, b, c, d].iter().all(|v| v.is_finite()) {
        return Analysis::insufficient("nonfinite Fieller arithmetic");
    }
    let mut result = Analysis { status: Status::Insufficient, reason: None, interval: None,
        statistics: Some(Statistics { independent_pairs:n, degrees_of_freedom:n-1,
            baseline_mean_images_per_second:x*scale, candidate_mean_images_per_second:y*scale,
            ratio_of_means:ratio, common_rate_scale:scale, scaled_mean_variance_baseline:vx,
            scaled_mean_variance_candidate:vy, scaled_mean_covariance:cov, t_critical:t,
            critical_value_policy:"two-sided at-least-95-percent under model; df<=29 table plus 0.0005 upper rounding endpoint" }) };
    if a <= 0.0 {
        result.reason = Some("denominator uncertainty permits an unbounded Fieller set".into());
        return result;
    }
    if d < 0.0 {
        result.reason = Some("negative discriminant; no fabricated/clamped interval".into());
        return result;
    }
    let q = b + d.sqrt().copysign(b);
    let (first, second) = if q == 0.0 {
        (b / a, b / a)
    } else {
        (q / a, c / q)
    };
    let (lower, upper) = (first.min(second), first.max(second));
    if !lower.is_finite() || !upper.is_finite() || lower > ratio || upper < ratio {
        result.reason = Some("unstable interval arithmetic".into());
        return result;
    }
    result.interval = Some(Interval { lower, upper });
    result.status = if lower >= 1.0 && ratio >= 1.0 {
        Status::AtLeastParity
    } else if upper < 1.0 {
        Status::BelowTarget
    } else {
        Status::Insufficient
    };
    if result.status == Status::Insufficient {
        result.reason = Some("interval includes the parity threshold".into());
    }
    result
}
