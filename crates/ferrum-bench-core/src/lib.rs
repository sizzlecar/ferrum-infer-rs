//! ferrum-bench-core — canonical schema, metric aggregation, and
//! variance reporting for ferrum's `bench` and `bench-serve` commands.
//!
//! Locked by `docs/bench/PLAYBOOK.md` § 7. Do not invent variants;
//! producers and consumers (bench, bench-serve, compare-commits,
//! visualizer, dashboards) all build against the types here.
//!
//! # Quick map
//!
//! - [`BenchReport`] — top-level: one bench cell, aggregated across `n_repeats`
//! - [`Scenario`] — closed-loop / open-loop / shared-prefix / cli
//! - [`MetricSet`] — p50/p75/p95/p99 of one latency metric
//! - [`ScalarStats`] — `{mean, stddev, ci95_hw}` ([`stats`] module)
//! - [`Env`] + [`EnvHash`] — apples-to-apples cell identity ([`env`] module)
//! - [`ProfileEvent`] — locked structured profile JSONL envelope ([`profile`] module)
//! - [`compute_metrics`] — the one aggregator both bench CLIs call
//! - [`arrivals`] module — Poisson inter-arrival times for open-loop
//!
//! # Determinism notes
//!
//! - JSON keys are emitted in struct field-declaration order; field
//!   order is part of the locked schema and should not change.
//! - `BTreeMap` (not `HashMap`) for any dynamic key-value bag.
//! - CI95 fields are suppressed when `n_repeats < 3` (degenerate).

pub mod arrivals;
pub mod env;
pub mod profile;
pub mod report;
pub mod stats;
pub mod trace;

pub use env::{Env, EnvHash};
pub use profile::{
    configure_global_profile, flush_global_profile, global_profile, parse_profile_event_value,
    parse_profile_jsonl_str, profile_fields_from_json, ProfileEvent, ProfileJsonlWriter,
    ProfileMetadata, ProfileSinkConfig,
};
pub use stats::{ci95_half_width, percentile, student_t_975, PercentileStats, ScalarStats};

use serde::{Deserialize, Serialize};

/// Locked enum of bench scenarios — see `docs/bench/PLAYBOOK.md` § 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scenario {
    /// `--concurrency K` — K workers in tight send→wait loop. Headline: throughput.
    ClosedLoop,
    /// `--request-rate R` — Poisson arrivals. Headline: goodput.
    OpenLoop,
    /// 1024-token shared prefix, burst arrival. Headline: cache hit rate.
    SharedPrefix,
    /// `ferrum bench` single-user batch=1. Headline: TTFT + TPOT.
    Cli,
}

/// SLO thresholds applied when computing goodput. All in milliseconds.
///
/// A request is "good" iff `ttft ≤ ttft_p99_ms` AND `tpot ≤ tpot_p99_ms`
/// AND `e2e ≤ e2e_p99_ms`. The `_p99_` naming is convention only — the
/// comparison is per-request, not against the distribution.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Slo {
    pub ttft_p99_ms: f64,
    pub tpot_p99_ms: f64,
    pub e2e_p99_ms: f64,
}

impl Default for Slo {
    fn default() -> Self {
        // Production defaults from PLAYBOOK § 4.B.
        Self {
            ttft_p99_ms: 500.0,
            tpot_p99_ms: 50.0,
            e2e_p99_ms: 30_000.0,
        }
    }
}

/// Four percentile points for a single latency metric. Each point is a
/// `ScalarStats` aggregate across `n_repeats` runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSet {
    pub p50: PercentileStats,
    pub p75: PercentileStats,
    pub p95: PercentileStats,
    pub p99: PercentileStats,
}

/// One bench cell — `n_repeats` independent runs aggregated.
///
/// Field order matters for `env_hash` determinism — do not reorder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub model: String,
    pub backend: String,
    pub scenario: Scenario,

    /// Set iff `scenario` is `ClosedLoop` (or `SharedPrefix` closed variant).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concurrency: Option<u32>,
    /// Set iff `scenario` is `OpenLoop` (or `SharedPrefix` open variant).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_rate: Option<f64>,

    pub n_prompt: u32,
    pub n_gen: u32,
    pub n_repeats: u32,
    pub n_requests_per_run: u32,
    pub warmup_requests: u32,

    pub ttft_ms: MetricSet,
    pub tpot_ms: MetricSet,
    pub itl_ms: MetricSet,
    pub e2e_ms: MetricSet,

    pub output_throughput_tps: ScalarStats,
    pub total_throughput_tps: ScalarStats,
    pub request_throughput_rps: ScalarStats,
    pub goodput_rps: ScalarStats,

    pub slo: Slo,

    pub completed_per_run: Vec<u32>,
    pub errored_per_run: Vec<u32>,

    pub env: Env,
    pub env_hash: EnvHash,
}

/// One request's measurements (input to [`compute_metrics`]).
#[derive(Debug, Clone)]
pub struct RequestRecord {
    pub success: bool,
    pub ttft_ms: f64,
    pub e2e_ms: f64,
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Per-token inter-arrival times within this request (decode steps,
    /// `len = output_tokens - 1`). Empty if not measured.
    pub itl_ms: Vec<f64>,
}

impl RequestRecord {
    /// Per-request TPOT in ms, or `None` if `output_tokens < 2`.
    pub fn tpot_ms(&self) -> Option<f64> {
        if self.output_tokens < 2 {
            return None;
        }
        Some((self.e2e_ms - self.ttft_ms) / (self.output_tokens - 1) as f64)
    }

    /// True if all three SLO thresholds are met (TPOT is treated as met
    /// when undefined — single-token responses don't have meaningful TPOT).
    pub fn meets_slo(&self, slo: &Slo) -> bool {
        if !self.success {
            return false;
        }
        let ttft_ok = self.ttft_ms <= slo.ttft_p99_ms;
        let e2e_ok = self.e2e_ms <= slo.e2e_p99_ms;
        let tpot_ok = self.tpot_ms().map(|t| t <= slo.tpot_p99_ms).unwrap_or(true);
        ttft_ok && e2e_ok && tpot_ok
    }
}

/// One independent run of the bench workload.
#[derive(Debug, Clone)]
pub struct RunRecord {
    pub records: Vec<RequestRecord>,
    /// Wall-clock duration of the run, in seconds. Used as the denominator
    /// for throughput / goodput.
    pub duration_s: f64,
}

impl RunRecord {
    pub fn n_completed(&self) -> u32 {
        self.records.iter().filter(|r| r.success).count() as u32
    }
    pub fn n_errored(&self) -> u32 {
        self.records.iter().filter(|r| !r.success).count() as u32
    }
}

/// Aggregate `n_repeats` independent runs into one [`BenchReport`].
///
/// The aggregation is two-level: within each run we compute the
/// per-request percentile distribution (p50/p75/p95/p99); across runs
/// we compute the mean + sample stddev + Student-t 95% CI half-width
/// of those per-run percentile values.
///
/// # Panics
///
/// Panics if `runs.is_empty()`.
#[allow(clippy::too_many_arguments)]
pub fn compute_metrics(
    model: String,
    backend: String,
    scenario: Scenario,
    concurrency: Option<u32>,
    request_rate: Option<f64>,
    n_prompt: u32,
    n_gen: u32,
    warmup_requests: u32,
    slo: Slo,
    runs: Vec<RunRecord>,
    env: Env,
) -> BenchReport {
    assert!(!runs.is_empty(), "compute_metrics: n_repeats must be ≥ 1");
    let n_repeats = runs.len() as u32;
    let n_requests_per_run = runs[0].records.len() as u32;

    let mut ttft_p50 = Vec::with_capacity(runs.len());
    let mut ttft_p75 = Vec::with_capacity(runs.len());
    let mut ttft_p95 = Vec::with_capacity(runs.len());
    let mut ttft_p99 = Vec::with_capacity(runs.len());
    let mut tpot_p50 = Vec::with_capacity(runs.len());
    let mut tpot_p75 = Vec::with_capacity(runs.len());
    let mut tpot_p95 = Vec::with_capacity(runs.len());
    let mut tpot_p99 = Vec::with_capacity(runs.len());
    let mut itl_p50 = Vec::with_capacity(runs.len());
    let mut itl_p75 = Vec::with_capacity(runs.len());
    let mut itl_p95 = Vec::with_capacity(runs.len());
    let mut itl_p99 = Vec::with_capacity(runs.len());
    let mut e2e_p50 = Vec::with_capacity(runs.len());
    let mut e2e_p75 = Vec::with_capacity(runs.len());
    let mut e2e_p95 = Vec::with_capacity(runs.len());
    let mut e2e_p99 = Vec::with_capacity(runs.len());

    let mut output_thr = Vec::with_capacity(runs.len());
    let mut total_thr = Vec::with_capacity(runs.len());
    let mut req_thr = Vec::with_capacity(runs.len());
    let mut good_thr = Vec::with_capacity(runs.len());

    let mut completed_per_run = Vec::with_capacity(runs.len());
    let mut errored_per_run = Vec::with_capacity(runs.len());

    for run in &runs {
        let success: Vec<&RequestRecord> = run.records.iter().filter(|r| r.success).collect();
        completed_per_run.push(success.len() as u32);
        errored_per_run.push((run.records.len() - success.len()) as u32);

        let ttfts: Vec<f64> = success.iter().map(|r| r.ttft_ms).collect();
        let tpots: Vec<f64> = success.iter().filter_map(|r| r.tpot_ms()).collect();
        let e2es: Vec<f64> = success.iter().map(|r| r.e2e_ms).collect();
        let itls: Vec<f64> = success
            .iter()
            .flat_map(|r| r.itl_ms.iter().copied())
            .collect();

        ttft_p50.push(percentile(&ttfts, 0.50));
        ttft_p75.push(percentile(&ttfts, 0.75));
        ttft_p95.push(percentile(&ttfts, 0.95));
        ttft_p99.push(percentile(&ttfts, 0.99));
        tpot_p50.push(percentile(&tpots, 0.50));
        tpot_p75.push(percentile(&tpots, 0.75));
        tpot_p95.push(percentile(&tpots, 0.95));
        tpot_p99.push(percentile(&tpots, 0.99));
        itl_p50.push(percentile(&itls, 0.50));
        itl_p75.push(percentile(&itls, 0.75));
        itl_p95.push(percentile(&itls, 0.95));
        itl_p99.push(percentile(&itls, 0.99));
        e2e_p50.push(percentile(&e2es, 0.50));
        e2e_p75.push(percentile(&e2es, 0.75));
        e2e_p95.push(percentile(&e2es, 0.95));
        e2e_p99.push(percentile(&e2es, 0.99));

        let total_in: u64 = success.iter().map(|r| r.input_tokens as u64).sum();
        let total_out: u64 = success.iter().map(|r| r.output_tokens as u64).sum();
        let dur = run.duration_s.max(f64::EPSILON);
        output_thr.push(total_out as f64 / dur);
        total_thr.push((total_in + total_out) as f64 / dur);
        req_thr.push(success.len() as f64 / dur);

        let good = success.iter().filter(|r| r.meets_slo(&slo)).count();
        good_thr.push(good as f64 / dur);
    }

    let env_hash = env.hash();
    BenchReport {
        model,
        backend,
        scenario,
        concurrency,
        request_rate,
        n_prompt,
        n_gen,
        n_repeats,
        n_requests_per_run,
        warmup_requests,
        ttft_ms: MetricSet {
            p50: ScalarStats::from_samples(&ttft_p50),
            p75: ScalarStats::from_samples(&ttft_p75),
            p95: ScalarStats::from_samples(&ttft_p95),
            p99: ScalarStats::from_samples(&ttft_p99),
        },
        tpot_ms: MetricSet {
            p50: ScalarStats::from_samples(&tpot_p50),
            p75: ScalarStats::from_samples(&tpot_p75),
            p95: ScalarStats::from_samples(&tpot_p95),
            p99: ScalarStats::from_samples(&tpot_p99),
        },
        itl_ms: MetricSet {
            p50: ScalarStats::from_samples(&itl_p50),
            p75: ScalarStats::from_samples(&itl_p75),
            p95: ScalarStats::from_samples(&itl_p95),
            p99: ScalarStats::from_samples(&itl_p99),
        },
        e2e_ms: MetricSet {
            p50: ScalarStats::from_samples(&e2e_p50),
            p75: ScalarStats::from_samples(&e2e_p75),
            p95: ScalarStats::from_samples(&e2e_p95),
            p99: ScalarStats::from_samples(&e2e_p99),
        },
        output_throughput_tps: ScalarStats::from_samples(&output_thr),
        total_throughput_tps: ScalarStats::from_samples(&total_thr),
        request_throughput_rps: ScalarStats::from_samples(&req_thr),
        goodput_rps: ScalarStats::from_samples(&good_thr),
        slo,
        completed_per_run,
        errored_per_run,
        env,
        env_hash,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(success: bool, ttft: f64, e2e: f64, in_tok: u32, out_tok: u32) -> RequestRecord {
        RequestRecord {
            success,
            ttft_ms: ttft,
            e2e_ms: e2e,
            input_tokens: in_tok,
            output_tokens: out_tok,
            itl_ms: vec![],
        }
    }

    #[test]
    fn tpot_undefined_for_short_response() {
        let r = req(true, 100.0, 100.0, 5, 1);
        assert_eq!(r.tpot_ms(), None);
        let r = req(true, 100.0, 200.0, 5, 2);
        assert_eq!(r.tpot_ms(), Some(100.0));
    }

    #[test]
    fn slo_short_response_treated_as_tpot_ok() {
        let slo = Slo::default();
        // 1-token response: TPOT N/A, must not fail SLO on TPOT.
        let r = req(true, 100.0, 200.0, 5, 1);
        assert!(r.meets_slo(&slo));
    }

    #[test]
    fn slo_failure_modes() {
        let slo = Slo::default();
        // TTFT too high.
        assert!(!req(true, 1000.0, 1100.0, 5, 10).meets_slo(&slo));
        // E2E too high.
        assert!(!req(true, 100.0, 40_000.0, 5, 10).meets_slo(&slo));
        // Errored.
        assert!(!req(false, 100.0, 200.0, 5, 10).meets_slo(&slo));
        // Good.
        assert!(req(true, 100.0, 200.0, 5, 10).meets_slo(&slo));
    }

    fn make_run(records: Vec<RequestRecord>, duration_s: f64) -> RunRecord {
        RunRecord {
            records,
            duration_s,
        }
    }

    #[test]
    fn aggregate_three_repeats() {
        // Three identical runs of 4 requests each. All meet SLO.
        let mk_run = || {
            make_run(
                vec![
                    req(true, 100.0, 200.0, 10, 10),
                    req(true, 120.0, 240.0, 10, 10),
                    req(true, 140.0, 280.0, 10, 10),
                    req(true, 160.0, 320.0, 10, 10),
                ],
                10.0,
            )
        };
        let report = compute_metrics(
            "test".into(),
            "cpu".into(),
            Scenario::ClosedLoop,
            Some(4),
            None,
            10,
            10,
            0,
            Slo::default(),
            vec![mk_run(), mk_run(), mk_run()],
            Env::default(),
        );
        assert_eq!(report.n_repeats, 3);
        assert_eq!(report.n_requests_per_run, 4);
        // All three runs identical → stddev = 0, ci95 = 0.
        assert_eq!(report.ttft_ms.p50.stddev, 0.0);
        // Mean p50 of [100, 120, 140, 160] = 130 (linear interp at q=0.5 of 4 elems).
        assert!((report.ttft_ms.p50.mean - 130.0).abs() < 1e-9);
        // Output throughput: 40 tokens / 10s = 4 tps.
        assert!((report.output_throughput_tps.mean - 4.0).abs() < 1e-9);
        // Request throughput: 4 req / 10s = 0.4 rps.
        assert!((report.request_throughput_rps.mean - 0.4).abs() < 1e-9);
        // Goodput: all 4 meet SLO → 0.4 rps.
        assert!((report.goodput_rps.mean - 0.4).abs() < 1e-9);
        // env_hash format check.
        assert!(report.env_hash.as_str().starts_with("sha256:"));
    }

    #[test]
    fn goodput_excludes_slo_violators() {
        let run = make_run(
            vec![
                req(true, 100.0, 200.0, 10, 10),    // good
                req(true, 1000.0, 1100.0, 10, 10),  // TTFT violator
                req(true, 100.0, 40_000.0, 10, 10), // E2E violator
                req(false, 100.0, 200.0, 10, 10),   // errored
            ],
            10.0,
        );
        let report = compute_metrics(
            "test".into(),
            "cpu".into(),
            Scenario::OpenLoop,
            None,
            Some(10.0),
            10,
            10,
            0,
            Slo::default(),
            vec![run],
            Env::default(),
        );
        // Request throughput: 3 successful / 10s = 0.3
        assert!((report.request_throughput_rps.mean - 0.3).abs() < 1e-9);
        // Goodput: 1 of 4 = 0.1 (errored excluded; both SLO violators excluded)
        assert!((report.goodput_rps.mean - 0.1).abs() < 1e-9);
    }

    #[test]
    fn json_round_trip() {
        let run = make_run(
            vec![
                req(true, 100.0, 200.0, 10, 10),
                req(true, 120.0, 240.0, 10, 10),
            ],
            5.0,
        );
        let report = compute_metrics(
            "qwen3:0.6b".into(),
            "metal".into(),
            Scenario::ClosedLoop,
            Some(2),
            None,
            256,
            128,
            10,
            Slo::default(),
            vec![run.clone(), run.clone(), run],
            Env::default(),
        );
        let json = serde_json::to_string_pretty(&report).unwrap();
        let parsed: BenchReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model, "qwen3:0.6b");
        assert_eq!(parsed.backend, "metal");
        assert_eq!(parsed.n_repeats, 3);
        assert_eq!(parsed.concurrency, Some(2));
        assert_eq!(parsed.request_rate, None);
    }
}
