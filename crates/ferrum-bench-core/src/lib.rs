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

impl Slo {
    /// Finite sentinel used when goodput was not requested. Keeping the
    /// sentinel finite preserves JSON round trips while accepting every
    /// finite request latency.
    pub fn unbounded() -> Self {
        Self {
            ttft_p99_ms: f64::MAX,
            tpot_p99_ms: f64::MAX,
            e2e_p99_ms: f64::MAX,
        }
    }

    pub fn is_unbounded(&self) -> bool {
        self.ttft_p99_ms == f64::MAX && self.tpot_p99_ms == f64::MAX && self.e2e_p99_ms == f64::MAX
    }

    pub fn is_valid(&self) -> bool {
        [self.ttft_p99_ms, self.tpot_p99_ms, self.e2e_p99_ms]
            .into_iter()
            .all(|value| value.is_finite() && value > 0.0)
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_input_tokens: Option<TokenLengthStats>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_input_tokens_per_request: Option<Vec<Vec<u32>>>,
    /// Per measured request output token counts for each repeat, in
    /// completion order. Failed requests keep their recorded count, usually 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_per_request: Option<Vec<Vec<u32>>>,
    /// Typed per-request evidence describing whether the recorded ITL samples
    /// have a one-event-per-token timing basis. Older reports omit this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub itl_evidence_per_request: Option<Vec<Vec<RequestItlEvidence>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_count_source: Option<String>,
    pub n_repeats: u32,
    pub n_requests_per_run: u32,
    pub warmup_requests: u32,

    /// Canonical per-repeat evidence used to derive the aggregate statistics
    /// below. Older reports may omit it; release gates can require it by
    /// checking that it contains exactly `n_repeats` rows.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub repeat_metrics: Vec<BenchRepeatMetrics>,

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
    #[serde(default)]
    pub bad_output_per_run: Vec<u32>,
    #[serde(default)]
    pub malformed_stream_per_run: Vec<u32>,
    #[serde(default)]
    pub missing_done_per_run: Vec<u32>,
    #[serde(default)]
    pub duplicate_done_per_run: Vec<u32>,
    #[serde(default)]
    pub zero_output_tokens_per_run: Vec<u32>,
    #[serde(default)]
    pub stream_bulk_flush_per_run: Vec<u32>,
    #[serde(default)]
    pub http_500_per_run: Vec<u32>,
    #[serde(default)]
    pub panic_per_run: Vec<u32>,
    #[serde(default)]
    pub quality_issues_per_run: Vec<QualityIssueCounts>,

    pub env: Env,
    pub env_hash: EnvHash,
}

impl BenchReport {
    /// True only when every expected measured request in every repeat has
    /// eligible one-event-per-token ITL evidence.
    pub fn has_complete_itl_evidence(&self) -> bool {
        self.repeat_metrics.len() == self.n_repeats as usize
            && !self.repeat_metrics.is_empty()
            && self.repeat_metrics.iter().all(|repeat| {
                repeat.expected_requests > 0
                    && repeat.expected_requests == self.n_requests_per_run
                    && repeat.completed_requests == repeat.expected_requests
                    && repeat.itl_eligible_requests == repeat.expected_requests
                    && repeat.itl_ineligible_requests == 0
                    && repeat.itl_expected_intervals == repeat.itl_observed_intervals
                    && repeat.itl_eligibility_counts.eligible == repeat.itl_eligible_requests
            })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLengthStats {
    pub requested: u32,
    pub min: u32,
    pub max: u32,
    pub mean: f64,
}

/// One request's measurements (input to [`compute_metrics`]).
#[derive(Debug, Clone)]
pub struct RequestRecord {
    pub success: bool,
    pub ttft_ms: f64,
    pub e2e_ms: f64,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub output_token_count_source: OutputTokenCountSource,
    pub itl_evidence: RequestItlEvidence,
    pub quality_issues: QualityIssueCounts,
    /// Observed output-event inter-arrival times within this request. Only
    /// eligible typed ITL evidence guarantees `len = output_tokens - 1` and
    /// permits these samples to enter aggregate ITL metrics.
    pub itl_ms: Vec<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualityIssueCounts {
    pub bad_output: u32,
    pub malformed_stream: u32,
    pub missing_done: u32,
    pub duplicate_done: u32,
    pub zero_output_tokens: u32,
    pub stream_bulk_flush: u32,
    pub http_500: u32,
    pub panic: u32,
}

/// Four percentile points measured within one repeat.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RepeatPercentiles {
    pub p50: f64,
    pub p75: f64,
    pub p95: f64,
    pub p99: f64,
}

/// Lossless per-repeat summary retained by [`BenchReport`].
///
/// These values are computed from the same [`RunRecord`] instances as the
/// aggregate `ScalarStats`, so a validator can independently recompute every
/// aggregate without trusting a second benchmark collector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchRepeatMetrics {
    pub repeat: u32,
    pub duration_s: f64,
    /// Configured measured request count for this repeat. This remains
    /// distinct from completion state so task failures cannot disappear.
    #[serde(default)]
    pub expected_requests: u32,
    pub completed_requests: u32,
    pub errored_requests: u32,
    pub warmup_expected: u32,
    pub warmup_completed: u32,
    pub warmup_errored: u32,
    pub actual_input_tokens: u64,
    pub output_tokens: u64,
    pub output_token_count_source: String,
    #[serde(default)]
    pub itl_eligible_requests: u32,
    #[serde(default)]
    pub itl_ineligible_requests: u32,
    #[serde(default)]
    pub itl_expected_intervals: u64,
    #[serde(default)]
    pub itl_observed_intervals: u64,
    #[serde(default)]
    pub itl_eligibility_counts: ItlEligibilityCounts,
    pub ttft_ms: RepeatPercentiles,
    pub tpot_ms: RepeatPercentiles,
    pub itl_ms: RepeatPercentiles,
    pub e2e_ms: RepeatPercentiles,
    pub output_throughput_tps: f64,
    pub total_throughput_tps: f64,
    pub request_throughput_rps: f64,
    pub goodput_rps: f64,
    pub quality_issues: QualityIssueCounts,
    pub warmup_quality_issues: QualityIssueCounts,
}

/// Origin of the events used to derive inter-token latency samples.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItlEvidenceSource {
    #[default]
    None,
    SseDeltaEvents,
    EngineTokenEvents,
}

/// Why a request can or cannot contribute to the ITL distribution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItlEligibility {
    Eligible,
    #[default]
    MissingEvidence,
    RequestFailed,
    MissingUsage,
    TooShort,
    EventUsageMismatch,
    IntervalCountMismatch,
    TransportCoalesced,
}

/// Per-request audit row for stream/event timing evidence.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestItlEvidence {
    #[serde(default)]
    pub source: ItlEvidenceSource,
    #[serde(default)]
    pub output_events: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage_output_tokens: Option<u32>,
    #[serde(default)]
    pub observed_intervals: u32,
    /// Number of HTTP transport chunks that made more than one output event
    /// observable. This is a client-side timing diagnostic, not evidence that
    /// the server bulk-flushed tokens.
    #[serde(default)]
    pub transport_coalesced_output_chunks: u32,
    #[serde(default)]
    pub eligibility: ItlEligibility,
}

impl RequestItlEvidence {
    pub fn sse(
        request_success: bool,
        output_events: u32,
        usage_output_tokens: Option<u32>,
        observed_intervals: u32,
        transport_coalesced_output_chunks: u32,
    ) -> Self {
        let eligibility = if !request_success {
            ItlEligibility::RequestFailed
        } else if usage_output_tokens.is_none() {
            ItlEligibility::MissingUsage
        } else if usage_output_tokens.is_some_and(|tokens| tokens < 2) {
            ItlEligibility::TooShort
        } else if usage_output_tokens != Some(output_events) {
            ItlEligibility::EventUsageMismatch
        } else if usage_output_tokens
            .is_some_and(|tokens| observed_intervals != tokens.saturating_sub(1))
        {
            ItlEligibility::IntervalCountMismatch
        } else if transport_coalesced_output_chunks > 0 {
            ItlEligibility::TransportCoalesced
        } else {
            ItlEligibility::Eligible
        };
        Self {
            source: ItlEvidenceSource::SseDeltaEvents,
            output_events,
            usage_output_tokens,
            observed_intervals,
            transport_coalesced_output_chunks,
            eligibility,
        }
    }

    pub fn engine(request_success: bool, output_events: u32, observed_intervals: u32) -> Self {
        let eligibility = if !request_success {
            ItlEligibility::RequestFailed
        } else if output_events < 2 {
            ItlEligibility::TooShort
        } else if observed_intervals != output_events.saturating_sub(1) {
            ItlEligibility::IntervalCountMismatch
        } else {
            ItlEligibility::Eligible
        };
        Self {
            source: ItlEvidenceSource::EngineTokenEvents,
            output_events,
            usage_output_tokens: None,
            observed_intervals,
            transport_coalesced_output_chunks: 0,
            eligibility,
        }
    }

    pub fn failed(source: ItlEvidenceSource) -> Self {
        Self {
            source,
            eligibility: ItlEligibility::RequestFailed,
            ..Self::default()
        }
    }

    pub fn is_eligible(&self) -> bool {
        self.eligibility == ItlEligibility::Eligible
    }
}

/// Typed repeat-level eligibility totals. The release validator checks these
/// before treating `itl_ms` as performance evidence.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ItlEligibilityCounts {
    #[serde(default)]
    pub eligible: u32,
    #[serde(default)]
    pub missing_evidence: u32,
    #[serde(default)]
    pub request_failed: u32,
    #[serde(default)]
    pub missing_usage: u32,
    #[serde(default)]
    pub too_short: u32,
    #[serde(default)]
    pub event_usage_mismatch: u32,
    #[serde(default)]
    pub interval_count_mismatch: u32,
    #[serde(default)]
    pub transport_coalesced: u32,
}

impl ItlEligibilityCounts {
    fn add(&mut self, eligibility: ItlEligibility) {
        let value = match eligibility {
            ItlEligibility::Eligible => &mut self.eligible,
            ItlEligibility::MissingEvidence => &mut self.missing_evidence,
            ItlEligibility::RequestFailed => &mut self.request_failed,
            ItlEligibility::MissingUsage => &mut self.missing_usage,
            ItlEligibility::TooShort => &mut self.too_short,
            ItlEligibility::EventUsageMismatch => &mut self.event_usage_mismatch,
            ItlEligibility::IntervalCountMismatch => &mut self.interval_count_mismatch,
            ItlEligibility::TransportCoalesced => &mut self.transport_coalesced,
        };
        *value = value
            .checked_add(1)
            .expect("ITL eligibility count overflow");
    }
}

impl QualityIssueCounts {
    pub fn add_assign(&mut self, other: &Self) {
        self.bad_output = self
            .bad_output
            .checked_add(other.bad_output)
            .expect("quality bad_output count overflow");
        self.malformed_stream = self
            .malformed_stream
            .checked_add(other.malformed_stream)
            .expect("quality malformed_stream count overflow");
        self.missing_done = self
            .missing_done
            .checked_add(other.missing_done)
            .expect("quality missing_done count overflow");
        self.duplicate_done = self
            .duplicate_done
            .checked_add(other.duplicate_done)
            .expect("quality duplicate_done count overflow");
        self.zero_output_tokens = self
            .zero_output_tokens
            .checked_add(other.zero_output_tokens)
            .expect("quality zero_output_tokens count overflow");
        self.stream_bulk_flush = self
            .stream_bulk_flush
            .checked_add(other.stream_bulk_flush)
            .expect("quality stream_bulk_flush count overflow");
        self.http_500 = self
            .http_500
            .checked_add(other.http_500)
            .expect("quality http_500 count overflow");
        self.panic = self
            .panic
            .checked_add(other.panic)
            .expect("quality panic count overflow");
    }

    pub fn request_error_count(&self) -> u32 {
        [
            self.bad_output,
            self.malformed_stream,
            self.missing_done,
            self.duplicate_done,
            self.zero_output_tokens,
            self.http_500,
            self.panic,
        ]
        .into_iter()
        .fold(0_u32, u32::saturating_add)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputTokenCountSource {
    Usage,
    StreamChunks,
    None,
}

impl OutputTokenCountSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Usage => "usage",
            Self::StreamChunks => "stream_chunks",
            Self::None => "none",
        }
    }
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
    /// Configured number of measured requests, including requests whose task
    /// failed before it could return a normal record.
    pub expected_requests: u32,
    /// Wall-clock duration of the run, in seconds. Used as the denominator
    /// for throughput / goodput.
    pub duration_s: f64,
    pub warmup: WarmupSummary,
}

/// Outcome of the warmup window associated with one measured repeat.
#[derive(Debug, Clone, Default)]
pub struct WarmupSummary {
    pub expected: u32,
    pub completed: u32,
    pub errored: u32,
    pub quality_issues: QualityIssueCounts,
}

impl RunRecord {
    pub fn n_completed(&self) -> u32 {
        self.records.iter().filter(|r| r.success).count() as u32
    }
    pub fn n_errored(&self) -> u32 {
        self.records.iter().filter(|r| !r.success).count() as u32
    }
}

fn repeat_percentiles(values: &[f64]) -> RepeatPercentiles {
    RepeatPercentiles {
        p50: percentile(values, 0.50),
        p75: percentile(values, 0.75),
        p95: percentile(values, 0.95),
        p99: percentile(values, 0.99),
    }
}

fn aggregate_quality(records: &[RequestRecord]) -> QualityIssueCounts {
    let mut quality = QualityIssueCounts::default();
    for record in records {
        quality.add_assign(&record.quality_issues);
    }
    quality
}

fn output_token_count_source(records: &[RequestRecord]) -> String {
    let mut usage = false;
    let mut stream_chunks = false;
    let mut none = false;
    for record in records {
        match record.output_token_count_source {
            OutputTokenCountSource::Usage => usage = true,
            OutputTokenCountSource::StreamChunks => stream_chunks = true,
            OutputTokenCountSource::None => none = true,
        }
    }
    match (usage, stream_chunks, none) {
        (true, false, false) => "usage",
        (false, true, false) => "stream_chunks",
        (false, false, true) => "none",
        _ => "mixed",
    }
    .to_string()
}

fn checked_token_sum(
    records: &[RequestRecord],
    field: impl Fn(&RequestRecord) -> u32,
    label: &str,
) -> u64 {
    records.iter().fold(0_u64, |total, record| {
        total
            .checked_add(field(record) as u64)
            .unwrap_or_else(|| panic!("{label} token count overflow"))
    })
}

fn aggregate_itl_evidence(records: &[RequestRecord]) -> (ItlEligibilityCounts, u32, u32, u64, u64) {
    let mut counts = ItlEligibilityCounts::default();
    let mut expected_intervals = 0_u64;
    let mut observed_intervals = 0_u64;
    for record in records {
        counts.add(record.itl_evidence.eligibility);
        expected_intervals = expected_intervals
            .checked_add(record.output_tokens.saturating_sub(1) as u64)
            .expect("ITL expected interval count overflow");
        observed_intervals = observed_intervals
            .checked_add(record.itl_evidence.observed_intervals as u64)
            .expect("ITL observed interval count overflow");
    }
    let total = u32::try_from(records.len()).expect("ITL request count overflow");
    let eligible = counts.eligible;
    let ineligible = total
        .checked_sub(eligible)
        .expect("ITL eligible request count exceeds total");
    (
        counts,
        eligible,
        ineligible,
        expected_intervals,
        observed_intervals,
    )
}

fn build_repeat_metrics(run: &RunRecord, repeat: u32, slo: &Slo) -> BenchRepeatMetrics {
    let success: Vec<&RequestRecord> = run.records.iter().filter(|record| record.success).collect();
    let (
        itl_eligibility_counts,
        itl_eligible_requests,
        itl_ineligible_requests,
        itl_expected_intervals,
        itl_observed_intervals,
    ) = aggregate_itl_evidence(&run.records);
    let completed_requests =
        u32::try_from(success.len()).expect("completed request count overflow");
    let itl_fully_eligible = completed_requests == run.expected_requests
        && itl_eligible_requests == run.expected_requests
        && itl_ineligible_requests == 0
        && itl_expected_intervals == itl_observed_intervals;
    let ttft: Vec<f64> = success.iter().map(|record| record.ttft_ms).collect();
    let tpot: Vec<f64> = success
        .iter()
        .filter_map(|record| record.tpot_ms())
        .collect();
    let itl: Vec<f64> = if itl_fully_eligible {
        success
            .iter()
            .flat_map(|record| record.itl_ms.iter().copied())
            .collect()
    } else {
        Vec::new()
    };
    let e2e: Vec<f64> = success.iter().map(|record| record.e2e_ms).collect();
    // Token throughput covers every attempted measured request. In
    // particular, partial tokens produced by a request that later fails stay
    // in the evidence instead of disappearing from totals while remaining in
    // the per-request vectors.
    let actual_input_tokens =
        checked_token_sum(&run.records, |record| record.input_tokens, "input");
    let output_tokens = checked_token_sum(&run.records, |record| record.output_tokens, "output");
    let good = success
        .iter()
        .filter(|record| record.meets_slo(slo))
        .count();
    let output_throughput_tps = output_tokens as f64 / run.duration_s;
    let total_throughput_tps = (actual_input_tokens as f64 + output_tokens as f64) / run.duration_s;
    let request_throughput_rps = completed_requests as f64 / run.duration_s;
    let goodput_rps = good as f64 / run.duration_s;
    assert!(
        [
            output_throughput_tps,
            total_throughput_tps,
            request_throughput_rps,
            goodput_rps,
        ]
        .into_iter()
        .all(f64::is_finite),
        "compute_metrics: derived repeat throughput must be finite"
    );
    BenchRepeatMetrics {
        repeat,
        duration_s: run.duration_s,
        expected_requests: run.expected_requests,
        completed_requests,
        errored_requests: run
            .expected_requests
            .checked_sub(completed_requests)
            .expect("completed requests exceed expected requests"),
        warmup_expected: run.warmup.expected,
        warmup_completed: run.warmup.completed,
        warmup_errored: run.warmup.errored,
        actual_input_tokens,
        output_tokens,
        output_token_count_source: output_token_count_source(&run.records),
        itl_eligible_requests,
        itl_ineligible_requests,
        itl_expected_intervals,
        itl_observed_intervals,
        itl_eligibility_counts,
        ttft_ms: repeat_percentiles(&ttft),
        tpot_ms: repeat_percentiles(&tpot),
        itl_ms: repeat_percentiles(&itl),
        e2e_ms: repeat_percentiles(&e2e),
        output_throughput_tps,
        total_throughput_tps,
        request_throughput_rps,
        goodput_rps,
        quality_issues: aggregate_quality(&run.records),
        warmup_quality_issues: run.warmup.quality_issues.clone(),
    }
}

fn checked_scalar_stats(samples: &[f64], label: &str) -> ScalarStats {
    let stats = ScalarStats::from_samples(samples);
    assert!(
        [stats.mean, stats.stddev, stats.ci95_hw]
            .into_iter()
            .all(f64::is_finite),
        "compute_metrics: derived {label} aggregate must be finite"
    );
    stats
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
/// Panics if runs are empty or violate the canonical request-count, warmup,
/// latency, duration, or SLO invariants.
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
    assert!(
        slo.is_valid(),
        "compute_metrics: SLO values must be positive and finite"
    );
    let n_repeats = u32::try_from(runs.len()).expect("compute_metrics: repeat count overflow");
    let n_requests_per_run = runs[0].expected_requests;
    assert!(
        n_requests_per_run > 0,
        "compute_metrics: expected measured requests must be ≥ 1"
    );
    for (index, run) in runs.iter().enumerate() {
        assert_eq!(
            run.expected_requests,
            n_requests_per_run,
            "compute_metrics: repeat {} expected request count differs",
            index + 1
        );
        assert_eq!(
            u32::try_from(run.records.len()).expect("compute_metrics: request count overflow"),
            run.expected_requests,
            "compute_metrics: repeat {} did not retain one record per expected request",
            index + 1
        );
        assert!(
            run.duration_s.is_finite() && run.duration_s > 0.0,
            "compute_metrics: repeat {} duration must be positive and finite",
            index + 1
        );
        assert_eq!(
            run.warmup.expected,
            warmup_requests,
            "compute_metrics: repeat {} warmup expected count differs",
            index + 1
        );
        assert_eq!(
            run.warmup
                .completed
                .checked_add(run.warmup.errored)
                .expect("compute_metrics: warmup count overflow"),
            run.warmup.expected,
            "compute_metrics: repeat {} warmup outcomes do not sum to expected",
            index + 1
        );
        for record in &run.records {
            assert!(
                record.ttft_ms.is_finite() && record.ttft_ms >= 0.0,
                "compute_metrics: request TTFT must be finite and nonnegative"
            );
            assert!(
                record.e2e_ms.is_finite() && record.e2e_ms >= record.ttft_ms,
                "compute_metrics: request E2E must be finite and at least TTFT"
            );
            assert!(
                record
                    .itl_ms
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0),
                "compute_metrics: request ITL values must be finite and nonnegative"
            );
            assert_eq!(
                usize::try_from(record.itl_evidence.observed_intervals)
                    .expect("compute_metrics: ITL interval count exceeds platform capacity"),
                record.itl_ms.len(),
                "compute_metrics: typed ITL observed interval count differs from samples"
            );
            if record.itl_evidence.source == ItlEvidenceSource::EngineTokenEvents {
                assert_eq!(
                    record.itl_evidence.output_events, record.output_tokens,
                    "compute_metrics: engine ITL event count differs from output tokens"
                );
            }
            if let Some(usage_tokens) = record.itl_evidence.usage_output_tokens {
                assert_eq!(
                    usage_tokens, record.output_tokens,
                    "compute_metrics: ITL usage count differs from output tokens"
                );
            }
            if record.itl_evidence.is_eligible() {
                assert!(
                    record.success,
                    "compute_metrics: failed request cannot have eligible ITL evidence"
                );
                assert_eq!(
                    record.itl_evidence.observed_intervals,
                    record.output_tokens.saturating_sub(1),
                    "compute_metrics: eligible ITL interval count mismatch"
                );
                assert!(
                    record.output_tokens >= 2,
                    "compute_metrics: eligible ITL request must contain at least two tokens"
                );
            }
        }
    }
    let repeat_metrics: Vec<BenchRepeatMetrics> = runs
        .iter()
        .enumerate()
        .map(|(index, run)| {
            let repeat = u32::try_from(index + 1).expect("compute_metrics: repeat index overflow");
            build_repeat_metrics(run, repeat, &slo)
        })
        .collect();

    let values = |field: fn(&BenchRepeatMetrics) -> f64| {
        repeat_metrics.iter().map(field).collect::<Vec<_>>()
    };
    let ttft_p50 = values(|row| row.ttft_ms.p50);
    let ttft_p75 = values(|row| row.ttft_ms.p75);
    let ttft_p95 = values(|row| row.ttft_ms.p95);
    let ttft_p99 = values(|row| row.ttft_ms.p99);
    let tpot_p50 = values(|row| row.tpot_ms.p50);
    let tpot_p75 = values(|row| row.tpot_ms.p75);
    let tpot_p95 = values(|row| row.tpot_ms.p95);
    let tpot_p99 = values(|row| row.tpot_ms.p99);
    let all_repeats_itl_eligible = repeat_metrics.iter().all(|row| {
        row.expected_requests > 0
            && row.completed_requests == row.expected_requests
            && row.itl_eligible_requests == row.expected_requests
            && row.itl_ineligible_requests == 0
            && row.itl_expected_intervals == row.itl_observed_intervals
    });
    let zero_itl = || vec![0.0; repeat_metrics.len()];
    let itl_p50 = if all_repeats_itl_eligible {
        values(|row| row.itl_ms.p50)
    } else {
        zero_itl()
    };
    let itl_p75 = if all_repeats_itl_eligible {
        values(|row| row.itl_ms.p75)
    } else {
        zero_itl()
    };
    let itl_p95 = if all_repeats_itl_eligible {
        values(|row| row.itl_ms.p95)
    } else {
        zero_itl()
    };
    let itl_p99 = if all_repeats_itl_eligible {
        values(|row| row.itl_ms.p99)
    } else {
        zero_itl()
    };
    let e2e_p50 = values(|row| row.e2e_ms.p50);
    let e2e_p75 = values(|row| row.e2e_ms.p75);
    let e2e_p95 = values(|row| row.e2e_ms.p95);
    let e2e_p99 = values(|row| row.e2e_ms.p99);
    let output_thr = values(|row| row.output_throughput_tps);
    let total_thr = values(|row| row.total_throughput_tps);
    let req_thr = values(|row| row.request_throughput_rps);
    let good_thr = values(|row| row.goodput_rps);
    let completed_per_run = repeat_metrics
        .iter()
        .map(|row| row.completed_requests)
        .collect();
    let errored_per_run = repeat_metrics
        .iter()
        .map(|row| row.errored_requests)
        .collect();
    let quality_issues_per_run: Vec<_> = repeat_metrics
        .iter()
        .map(|row| row.quality_issues.clone())
        .collect();
    let bad_output_per_run = quality_issues_per_run
        .iter()
        .map(|row| row.bad_output)
        .collect();
    let malformed_stream_per_run = quality_issues_per_run
        .iter()
        .map(|row| row.malformed_stream)
        .collect();
    let missing_done_per_run = quality_issues_per_run
        .iter()
        .map(|row| row.missing_done)
        .collect();
    let duplicate_done_per_run = quality_issues_per_run
        .iter()
        .map(|row| row.duplicate_done)
        .collect();
    let zero_output_tokens_per_run = quality_issues_per_run
        .iter()
        .map(|row| row.zero_output_tokens)
        .collect();
    let stream_bulk_flush_per_run = quality_issues_per_run
        .iter()
        .map(|row| row.stream_bulk_flush)
        .collect();
    let http_500_per_run = quality_issues_per_run
        .iter()
        .map(|row| row.http_500)
        .collect();
    let panic_per_run = quality_issues_per_run.iter().map(|row| row.panic).collect();
    let output_tokens_per_request = runs
        .iter()
        .map(|run| {
            run.records
                .iter()
                .map(|record| record.output_tokens)
                .collect()
        })
        .collect();
    let itl_evidence_per_request = runs
        .iter()
        .map(|run| {
            run.records
                .iter()
                .map(|record| record.itl_evidence.clone())
                .collect()
        })
        .collect();

    let env_hash = env.hash();
    BenchReport {
        model,
        backend,
        scenario,
        concurrency,
        request_rate,
        n_prompt,
        n_gen,
        actual_input_tokens: None,
        actual_input_tokens_per_request: None,
        output_tokens_per_request: Some(output_tokens_per_request),
        itl_evidence_per_request: Some(itl_evidence_per_request),
        output_token_count_source: None,
        n_repeats,
        n_requests_per_run,
        warmup_requests,
        repeat_metrics,
        ttft_ms: MetricSet {
            p50: checked_scalar_stats(&ttft_p50, "TTFT p50"),
            p75: checked_scalar_stats(&ttft_p75, "TTFT p75"),
            p95: checked_scalar_stats(&ttft_p95, "TTFT p95"),
            p99: checked_scalar_stats(&ttft_p99, "TTFT p99"),
        },
        tpot_ms: MetricSet {
            p50: checked_scalar_stats(&tpot_p50, "TPOT p50"),
            p75: checked_scalar_stats(&tpot_p75, "TPOT p75"),
            p95: checked_scalar_stats(&tpot_p95, "TPOT p95"),
            p99: checked_scalar_stats(&tpot_p99, "TPOT p99"),
        },
        itl_ms: MetricSet {
            p50: checked_scalar_stats(&itl_p50, "ITL p50"),
            p75: checked_scalar_stats(&itl_p75, "ITL p75"),
            p95: checked_scalar_stats(&itl_p95, "ITL p95"),
            p99: checked_scalar_stats(&itl_p99, "ITL p99"),
        },
        e2e_ms: MetricSet {
            p50: checked_scalar_stats(&e2e_p50, "E2E p50"),
            p75: checked_scalar_stats(&e2e_p75, "E2E p75"),
            p95: checked_scalar_stats(&e2e_p95, "E2E p95"),
            p99: checked_scalar_stats(&e2e_p99, "E2E p99"),
        },
        output_throughput_tps: checked_scalar_stats(&output_thr, "output throughput"),
        total_throughput_tps: checked_scalar_stats(&total_thr, "total throughput"),
        request_throughput_rps: checked_scalar_stats(&req_thr, "request throughput"),
        goodput_rps: checked_scalar_stats(&good_thr, "goodput"),
        slo,
        completed_per_run,
        errored_per_run,
        bad_output_per_run,
        malformed_stream_per_run,
        missing_done_per_run,
        duplicate_done_per_run,
        zero_output_tokens_per_run,
        stream_bulk_flush_per_run,
        http_500_per_run,
        panic_per_run,
        quality_issues_per_run,
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
            output_token_count_source: if out_tok > 0 {
                OutputTokenCountSource::Usage
            } else {
                OutputTokenCountSource::None
            },
            itl_evidence: RequestItlEvidence::sse(success, 0, Some(out_tok), 0, 0),
            quality_issues: QualityIssueCounts::default(),
            itl_ms: vec![],
        }
    }

    fn eligible_sse_req(ttft: f64, e2e: f64, in_tok: u32, out_tok: u32) -> RequestRecord {
        assert!(out_tok >= 2);
        let observed_intervals = out_tok - 1;
        RequestRecord {
            success: true,
            ttft_ms: ttft,
            e2e_ms: e2e,
            input_tokens: in_tok,
            output_tokens: out_tok,
            output_token_count_source: OutputTokenCountSource::Usage,
            itl_evidence: RequestItlEvidence::sse(
                true,
                out_tok,
                Some(out_tok),
                observed_intervals,
                0,
            ),
            quality_issues: QualityIssueCounts::default(),
            itl_ms: vec![1.0; observed_intervals as usize],
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
    fn direct_engine_itl_evidence_is_typed_without_usage() {
        let eligible = RequestItlEvidence::engine(true, 3, 2);
        assert_eq!(eligible.source, ItlEvidenceSource::EngineTokenEvents);
        assert_eq!(eligible.usage_output_tokens, None);
        assert_eq!(eligible.eligibility, ItlEligibility::Eligible);
        let mismatch = RequestItlEvidence::engine(true, 3, 1);
        assert_eq!(mismatch.eligibility, ItlEligibility::IntervalCountMismatch);
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
        let expected_requests = u32::try_from(records.len()).unwrap();
        RunRecord {
            records,
            expected_requests,
            duration_s,
            warmup: WarmupSummary::default(),
        }
    }

    fn compute_one(run: RunRecord, warmup_requests: u32, slo: Slo) -> BenchReport {
        compute_metrics(
            "test".into(),
            "cpu".into(),
            Scenario::ClosedLoop,
            Some(run.expected_requests),
            None,
            8,
            4,
            warmup_requests,
            slo,
            vec![run],
            Env::default(),
        )
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
        assert_eq!(report.repeat_metrics.len(), 3);
        assert_eq!(report.repeat_metrics[0].completed_requests, 4);
        assert_eq!(report.repeat_metrics[0].errored_requests, 0);
        assert_eq!(report.repeat_metrics[0].actual_input_tokens, 40);
        assert_eq!(report.repeat_metrics[0].output_tokens, 40);
        assert_eq!(report.repeat_metrics[0].output_token_count_source, "usage");
        assert_eq!(report.repeat_metrics[0].ttft_ms.p50, 130.0);
        assert_eq!(report.repeat_metrics[0].e2e_ms.p95, 314.0);
        assert!((report.repeat_metrics[0].output_throughput_tps - 4.0).abs() < 1e-9);
        assert!((report.repeat_metrics[0].total_throughput_tps - 8.0).abs() < 1e-9);
        assert!((report.repeat_metrics[0].request_throughput_rps - 0.4).abs() < 1e-9);
        assert!((report.repeat_metrics[0].goodput_rps - 0.4).abs() < 1e-9);
        assert_eq!(report.bad_output_per_run, vec![0, 0, 0]);
        assert_eq!(report.malformed_stream_per_run, vec![0, 0, 0]);
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
            0,
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
        assert_eq!(parsed.quality_issues_per_run.len(), 3);
    }

    #[test]
    fn old_json_without_repeat_metrics_still_deserializes() {
        let report = compute_one(
            make_run(vec![req(true, 10.0, 20.0, 8, 4)], 1.0),
            0,
            Slo::default(),
        );
        let mut value = serde_json::to_value(report).unwrap();
        let mut old_repeat_rows = value.clone();
        for row in old_repeat_rows["repeat_metrics"].as_array_mut().unwrap() {
            row.as_object_mut().unwrap().remove("expected_requests");
            for field in [
                "itl_eligible_requests",
                "itl_ineligible_requests",
                "itl_expected_intervals",
                "itl_observed_intervals",
                "itl_eligibility_counts",
            ] {
                row.as_object_mut().unwrap().remove(field);
            }
        }
        old_repeat_rows
            .as_object_mut()
            .unwrap()
            .remove("itl_evidence_per_request");
        let parsed_rows: BenchReport = serde_json::from_value(old_repeat_rows).unwrap();
        assert_eq!(parsed_rows.repeat_metrics[0].expected_requests, 0);
        assert_eq!(parsed_rows.repeat_metrics[0].itl_eligible_requests, 0);
        assert!(parsed_rows.itl_evidence_per_request.is_none());

        value.as_object_mut().unwrap().remove("repeat_metrics");
        let parsed: BenchReport = serde_json::from_value(value).unwrap();
        assert!(parsed.repeat_metrics.is_empty());
    }

    #[test]
    fn itl_aggregation_requires_every_request_in_every_repeat() {
        let eligible = eligible_sse_req(10.0, 20.0, 8, 3);
        let complete = compute_one(
            make_run(vec![eligible.clone(), eligible.clone()], 1.0),
            0,
            Slo::default(),
        );
        assert!(complete.has_complete_itl_evidence());
        assert_eq!(complete.repeat_metrics[0].itl_eligible_requests, 2);
        assert_eq!(complete.repeat_metrics[0].itl_ineligible_requests, 0);
        assert_eq!(complete.repeat_metrics[0].itl_expected_intervals, 4);
        assert_eq!(complete.repeat_metrics[0].itl_observed_intervals, 4);
        assert_eq!(complete.repeat_metrics[0].itl_ms.p50, 1.0);
        assert_eq!(complete.itl_ms.p50.mean, 1.0);

        let mut ineligible = eligible;
        ineligible.itl_evidence = RequestItlEvidence::sse(true, 1, Some(3), 0, 0);
        ineligible.itl_ms.clear();
        let partial = compute_one(
            make_run(vec![eligible_sse_req(10.0, 20.0, 8, 3), ineligible], 1.0),
            0,
            Slo::default(),
        );
        assert!(!partial.has_complete_itl_evidence());
        assert_eq!(partial.repeat_metrics[0].itl_eligible_requests, 1);
        assert_eq!(partial.repeat_metrics[0].itl_ineligible_requests, 1);
        assert_eq!(
            partial.repeat_metrics[0]
                .itl_eligibility_counts
                .event_usage_mismatch,
            1
        );
        assert_eq!(partial.repeat_metrics[0].itl_ms.p50, 0.0);
        assert_eq!(partial.itl_ms.p50.mean, 0.0);
        assert_eq!(
            partial.itl_evidence_per_request.as_ref().unwrap()[0].len(),
            2
        );

        let mut ineligible = eligible_sse_req(10.0, 20.0, 8, 3);
        ineligible.itl_evidence = RequestItlEvidence::sse(true, 1, Some(3), 0, 0);
        ineligible.itl_ms.clear();
        let mixed_repeats = compute_metrics(
            "test".into(),
            "cpu".into(),
            Scenario::ClosedLoop,
            Some(1),
            None,
            8,
            3,
            0,
            Slo::default(),
            vec![
                make_run(vec![eligible_sse_req(10.0, 20.0, 8, 3)], 1.0),
                make_run(vec![ineligible], 1.0),
            ],
            Env::default(),
        );
        assert_eq!(mixed_repeats.repeat_metrics[0].itl_ms.p50, 1.0);
        assert_eq!(mixed_repeats.repeat_metrics[1].itl_ms.p50, 0.0);
        assert_eq!(mixed_repeats.itl_ms.p50.mean, 0.0);
    }

    #[test]
    fn unbounded_slo_report_round_trips_without_nulls() {
        let report = compute_one(
            make_run(vec![req(true, 10.0, 20.0, 8, 4)], 1.0),
            0,
            Slo::unbounded(),
        );
        let json = serde_json::to_string(&report).unwrap();
        assert!(!json.contains(":null"));
        let parsed: BenchReport = serde_json::from_str(&json).unwrap();
        assert!(parsed.slo.is_unbounded());
    }

    #[test]
    fn aggregates_quality_issues_per_run() {
        let mut bad = req(false, 100.0, 200.0, 10, 0);
        bad.quality_issues.bad_output = 1;
        bad.quality_issues.missing_done = 1;
        let mut malformed = req(false, 100.0, 200.0, 10, 0);
        malformed.quality_issues.malformed_stream = 1;
        malformed.quality_issues.http_500 = 1;
        let report = compute_metrics(
            "test".into(),
            "cpu".into(),
            Scenario::ClosedLoop,
            Some(2),
            None,
            10,
            10,
            0,
            Slo::default(),
            vec![make_run(vec![bad], 1.0), make_run(vec![malformed], 1.0)],
            Env::default(),
        );
        assert_eq!(report.bad_output_per_run, vec![1, 0]);
        assert_eq!(report.malformed_stream_per_run, vec![0, 1]);
        assert_eq!(report.missing_done_per_run, vec![1, 0]);
        assert_eq!(report.http_500_per_run, vec![0, 1]);
    }

    #[test]
    fn repeat_metrics_retain_warmup_outcomes_and_quality() {
        let mut measured_bad = req(false, 0.0, 25.0, 8, 0);
        measured_bad.quality_issues.missing_done = 1;
        let mut warmup_quality = QualityIssueCounts::default();
        warmup_quality.http_500 = 1;
        let run = RunRecord {
            records: vec![req(true, 10.0, 30.0, 8, 4), measured_bad],
            expected_requests: 2,
            duration_s: 2.0,
            warmup: WarmupSummary {
                expected: 10,
                completed: 9,
                errored: 1,
                quality_issues: warmup_quality.clone(),
            },
        };
        let report = compute_metrics(
            "test".into(),
            "cpu".into(),
            Scenario::ClosedLoop,
            Some(2),
            None,
            8,
            4,
            10,
            Slo::default(),
            vec![run],
            Env::default(),
        );

        let repeat = &report.repeat_metrics[0];
        assert_eq!(repeat.repeat, 1);
        assert_eq!(repeat.completed_requests, 1);
        assert_eq!(repeat.errored_requests, 1);
        assert_eq!(repeat.warmup_expected, 10);
        assert_eq!(repeat.warmup_completed, 9);
        assert_eq!(repeat.warmup_errored, 1);
        assert_eq!(repeat.quality_issues.missing_done, 1);
        assert_eq!(repeat.warmup_quality_issues, warmup_quality);
        assert_eq!(repeat.output_token_count_source, "mixed");
        assert!((repeat.output_throughput_tps - 2.0).abs() < 1e-9);
    }

    #[test]
    fn partial_failed_tokens_are_retained_and_aggregates_derive_from_repeats() {
        let mut partial = req(false, 5.0, 25.0, 8, 2);
        partial.output_token_count_source = OutputTokenCountSource::StreamChunks;
        partial.quality_issues.missing_done = 1;
        let first = make_run(vec![req(true, 10.0, 30.0, 8, 4), partial], 2.0);
        let second = make_run(
            vec![req(true, 20.0, 40.0, 10, 5), req(true, 30.0, 60.0, 10, 5)],
            5.0,
        );
        let report = compute_metrics(
            "test".into(),
            "cpu".into(),
            Scenario::ClosedLoop,
            Some(2),
            None,
            8,
            4,
            0,
            Slo::default(),
            vec![first, second],
            Env::default(),
        );

        let first = &report.repeat_metrics[0];
        assert_eq!(first.expected_requests, 2);
        assert_eq!(first.actual_input_tokens, 16);
        assert_eq!(first.output_tokens, 6);
        assert_eq!(first.output_token_count_source, "mixed");
        assert_eq!(
            report.output_tokens_per_request,
            Some(vec![vec![4, 2], vec![5, 5]])
        );
        assert!((first.output_throughput_tps - 3.0).abs() < 1e-9);
        assert!((first.total_throughput_tps - 11.0).abs() < 1e-9);
        let expected_output_mean = report
            .repeat_metrics
            .iter()
            .map(|row| row.output_throughput_tps)
            .sum::<f64>()
            / report.repeat_metrics.len() as f64;
        let expected_total_mean = report
            .repeat_metrics
            .iter()
            .map(|row| row.total_throughput_tps)
            .sum::<f64>()
            / report.repeat_metrics.len() as f64;
        assert!((report.output_throughput_tps.mean - expected_output_mean).abs() < 1e-9);
        assert!((report.total_throughput_tps.mean - expected_total_mean).abs() < 1e-9);
        assert_eq!(report.ttft_ms.p50.mean, 17.5);
    }

    #[test]
    fn all_failed_repeat_serializes_finite_zero_metrics() {
        let mut first = req(false, 0.0, 1.0, 0, 0);
        first.quality_issues.panic = 1;
        let mut second = req(false, 0.0, 1.0, 0, 0);
        second.quality_issues.missing_done = 1;
        let report = compute_one(make_run(vec![first, second], 1.0), 0, Slo::default());
        let row = &report.repeat_metrics[0];
        for value in [
            row.ttft_ms.p50,
            row.ttft_ms.p75,
            row.ttft_ms.p95,
            row.ttft_ms.p99,
            row.tpot_ms.p50,
            row.itl_ms.p50,
            row.e2e_ms.p50,
            row.output_throughput_tps,
            row.total_throughput_tps,
            row.request_throughput_rps,
            row.goodput_rps,
        ] {
            assert_eq!(value, 0.0);
            assert!(value.is_finite());
        }
        let json = serde_json::to_string(&report).unwrap();
        assert!(!json.contains(":null"));
        let parsed: BenchReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.repeat_metrics[0].completed_requests, 0);
        assert_eq!(parsed.repeat_metrics[0].errored_requests, 2);
    }

    #[test]
    fn rejects_heterogeneous_expected_request_counts() {
        let first = make_run(vec![req(true, 10.0, 20.0, 8, 4)], 1.0);
        let second = make_run(
            vec![req(true, 10.0, 20.0, 8, 4), req(true, 10.0, 20.0, 8, 4)],
            1.0,
        );
        let result = std::panic::catch_unwind(|| {
            compute_metrics(
                "test".into(),
                "cpu".into(),
                Scenario::ClosedLoop,
                Some(1),
                None,
                8,
                4,
                0,
                Slo::default(),
                vec![first, second],
                Env::default(),
            )
        });
        assert!(result.is_err());
    }

    #[test]
    fn rejects_missing_request_records_and_warmup_mismatch() {
        let mut missing = make_run(vec![req(true, 10.0, 20.0, 8, 4)], 1.0);
        missing.expected_requests = 2;
        assert!(std::panic::catch_unwind(|| compute_one(missing, 0, Slo::default())).is_err());

        let warmup = make_run(vec![req(true, 10.0, 20.0, 8, 4)], 1.0);
        assert!(std::panic::catch_unwind(|| compute_one(warmup, 1, Slo::default())).is_err());
    }

    #[test]
    fn rejects_nonfinite_or_nonpositive_numeric_inputs() {
        for duration in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let run = make_run(vec![req(true, 10.0, 20.0, 8, 4)], duration);
            assert!(
                std::panic::catch_unwind(|| compute_one(run, 0, Slo::default())).is_err(),
                "duration {duration:?} should be rejected"
            );
        }
        let invalid_slo = Slo {
            ttft_p99_ms: f64::NAN,
            ..Slo::default()
        };
        let run = make_run(vec![req(true, 10.0, 20.0, 8, 4)], 1.0);
        assert!(std::panic::catch_unwind(|| compute_one(run, 0, invalid_slo)).is_err());
    }

    #[test]
    fn rejects_nonfinite_derived_throughput() {
        let run = make_run(
            vec![req(true, 0.0, 0.0, u32::MAX, u32::MAX)],
            f64::MIN_POSITIVE,
        );
        let result = std::panic::catch_unwind(|| compute_one(run, 0, Slo::default()));
        assert!(result.is_err());
    }

    #[test]
    #[should_panic(expected = "quality panic count overflow")]
    fn quality_count_overflow_is_explicit() {
        let mut total = QualityIssueCounts {
            panic: u32::MAX,
            ..Default::default()
        };
        total.add_assign(&QualityIssueCounts {
            panic: 1,
            ..Default::default()
        });
    }

    #[test]
    fn request_error_count_uses_non_overflowing_width() {
        let issues = QualityIssueCounts {
            bad_output: u32::MAX,
            malformed_stream: u32::MAX,
            missing_done: u32::MAX,
            duplicate_done: u32::MAX,
            zero_output_tokens: u32::MAX,
            stream_bulk_flush: u32::MAX,
            http_500: u32::MAX,
            panic: u32::MAX,
        };
        assert_eq!(issues.request_error_count(), u32::MAX);
    }
}
