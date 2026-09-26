//! Versioned client-visible SLO evaluation, independent of the legacy [`crate::Slo`].
//!
//! Each invocation evaluates one service class and one measurement window. The
//! caller supplies every offered request, including rejections and pending work.
//! Pooled visible gaps retain every attempt; other latency distributions identify
//! their completed-request scope. Joint attainment retains the full denominator.
//! This evaluator does not establish that
//! an arrival process was delivered or that a server reached steady state.

use crate::{
    percentile, ItlEvidenceSource, OutputTokenCountSource, RepeatPercentiles, RequestItlEvidence,
    RequestRecord,
};
use ferrum_types::{SloAttainmentTargets, SloClientVisibleConfig, SloItlPercentileScope};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TpotBoundary {
    /// Historical `(terminal - first visible text) / (usage output tokens - 1)`.
    LegacyTerminal,
    /// Visible emission span divided by usage tokens, not strict token timing.
    LastVisibleOutput,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SloEvaluationConfig {
    pub ttft_ms: f64,
    pub tpot_ms: f64,
    pub visible_itl_ms: f64,
    /// Shares the service contract, including percentile units `(0, 100]`.
    pub attainment: SloAttainmentTargets,
    pub tpot_boundary: TpotBoundary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestOutcome {
    Completed,
    Failed,
    Rejected,
    Pending,
}

/// Observable service acceptance, distinct from GPU authority or a time promise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionEvidence {
    Accepted,
    Rejected,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RawOutputCountSource {
    Usage,
    EngineTokens,
    TextEvents,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawOutputCount {
    pub count: u32,
    pub source: RawOutputCountSource,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisibleTextEvidence {
    pub output_events: u32,
    /// Unfiltered client-observed gaps, including zero gaps and long stalls.
    pub gaps_ms: Vec<f64>,
    pub transport_coalesced_output_chunks: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestSloEvidence {
    pub outcome: RequestOutcome,
    pub admission: AdmissionEvidence,
    /// All three times are offsets from the same client submission instant.
    pub first_visible_ms: Option<f64>,
    pub last_visible_ms: Option<f64>,
    pub terminal_ms: Option<f64>,
    pub usage_output_tokens: Option<u32>,
    pub raw_output_count: RawOutputCount,
    /// Only actual SSE text observations belong here; engine events do not.
    pub visible_text: Option<VisibleTextEvidence>,
    /// Independent diagnostic; its eligibility never filters visible gaps.
    pub strict_token_evidence: RequestItlEvidence,
}

impl RequestSloEvidence {
    /// Explicitly adapt legacy records without inventing a last-visible timer.
    /// Failed legacy rows cannot distinguish rejection from other failures.
    pub fn from_legacy_record(record: &RequestRecord, last_visible_ms: Option<f64>) -> Self {
        let is_sse = record.itl_evidence.source == ItlEvidenceSource::SseDeltaEvents;
        let observed_first =
            is_sse && record.itl_evidence.output_events > 0 && record.quality_issues.panic == 0;
        Self {
            // A completed request with actual output proves service acceptance.
            // Failed legacy rows lack the HTTP/stream-start evidence needed to
            // distinguish an accepted stream failure from a pre-acceptance error.
            admission: if record.success
                && record.output_tokens > 0
                && record.itl_evidence.output_events > 0
            {
                AdmissionEvidence::Accepted
            } else {
                AdmissionEvidence::Unknown
            },
            outcome: if record.success {
                RequestOutcome::Completed
            } else {
                RequestOutcome::Failed
            },
            first_visible_ms: observed_first.then_some(record.ttft_ms),
            last_visible_ms,
            terminal_ms: (record.quality_issues.panic == 0).then_some(record.e2e_ms),
            usage_output_tokens: (record.output_token_count_source
                == OutputTokenCountSource::Usage)
                .then_some(record.output_tokens),
            raw_output_count: RawOutputCount {
                count: record.output_tokens,
                source: match record.output_token_count_source {
                    OutputTokenCountSource::Usage => RawOutputCountSource::Usage,
                    OutputTokenCountSource::StreamChunks => RawOutputCountSource::TextEvents,
                    OutputTokenCountSource::None => RawOutputCountSource::Unknown,
                },
            },
            visible_text: is_sse.then(|| VisibleTextEvidence {
                output_events: record.itl_evidence.output_events,
                gaps_ms: record.itl_ms.clone(),
                transport_coalesced_output_chunks: record
                    .itl_evidence
                    .transport_coalesced_output_chunks,
            }),
            strict_token_evidence: record.itl_evidence.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SloStatus {
    Pass,
    Fail,
    Unknown,
    NotApplicable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceIssue {
    MissingFirstVisibleOutput,
    MissingLastVisibleOutput,
    MissingTerminal,
    MissingUsage,
    MissingVisibleEvents,
    IncompleteVisibleIntervals,
    TooFewOutputTokens,
    TooFewVisibleEvents,
    ZeroOutputTokens,
    RequestFailed,
    RequestRejected,
    RequestPending,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MetricEvaluation {
    pub status: SloStatus,
    pub observed_ms: Option<f64>,
    pub issue: Option<EvidenceIssue>,
}

impl MetricEvaluation {
    fn measured(value: f64, threshold: f64) -> Self {
        Self {
            status: if value <= threshold {
                SloStatus::Pass
            } else {
                SloStatus::Fail
            },
            observed_ms: Some(value),
            issue: None,
        }
    }

    fn absent(status: SloStatus, issue: EvidenceIssue) -> Self {
        Self {
            status,
            observed_ms: None,
            issue: Some(issue),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestSloEvaluation {
    pub outcome: RequestOutcome,
    /// Completed protocol + nonempty output + observed first visible text.
    pub task_success: SloStatus,
    pub ttft: MetricEvaluation,
    pub tpot: MetricEvaluation,
    pub request_max_visible_gap: MetricEvaluation,
    pub joint: SloStatus,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatusCounts {
    pub pass: u64,
    pub fail: u64,
    pub unknown: u64,
    pub not_applicable: u64,
}

impl StatusCounts {
    fn add(&mut self, status: SloStatus) {
        match status {
            SloStatus::Pass => self.pass += 1,
            SloStatus::Fail => self.fail += 1,
            SloStatus::Unknown => self.unknown += 1,
            SloStatus::NotApplicable => self.not_applicable += 1,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutcomeCounts {
    pub offered: u64,
    pub completed: u64,
    pub failed: u64,
    pub rejected: u64,
    pub pending: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmissionCounts {
    pub accepted: u64,
    pub rejected: u64,
    pub unknown: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatencyPopulation {
    CompletedRequests,
    AllOfferedObservedGaps,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatencyDistribution {
    pub population: LatencyPopulation,
    /// Observed samples in the declared population. Never zero-filled.
    pub observed_percentiles_ms: Option<RepeatPercentiles>,
    pub evaluated_percentile: f64,
    pub observed_percentile_ms: Option<f64>,
    pub threshold_ms: f64,
    pub sample_count: u64,
    pub request_evidence: StatusCounts,
    pub status: SloStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputThroughput {
    /// Known token count, also a lower bound when evidence is incomplete.
    pub known_tokens: u64,
    pub included_requests: u64,
    pub unknown_count_requests: u64,
    /// Used for SLO output: eligibility itself can be unknown.
    pub unknown_eligibility_requests: u64,
    pub tokens_per_second: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisibleEvidenceCounts {
    pub requests_with_evidence: u64,
    pub events: u64,
    pub intervals: u64,
    pub failed_or_pending_intervals: u64,
    pub transport_coalesced_requests: u64,
    pub event_usage_mismatch_requests: u64,
    pub missing_usage_requests: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SloEvaluationReport {
    pub schema_version: u32,
    pub config: SloEvaluationConfig,
    pub duration_s: f64,
    pub outcomes: OutcomeCounts,
    pub admissions: AdmissionCounts,
    pub task_success: StatusCounts,
    pub offered_joint: StatusCounts,
    pub accepted_joint: StatusCounts,
    pub accepted_task_success: StatusCounts,
    /// Confirmed passes / all offered requests. Unknowns are never passes.
    pub offered_joint_attainment_lower_bound: f64,
    /// Confirmed passes / explicitly accepted requests; absent if none exist.
    /// Admission coverage must also be complete before this can establish Pass.
    pub accepted_joint_attainment_lower_bound: Option<f64>,
    pub success_rate_lower_bound: f64,
    pub accepted_error_rate_lower_bound: Option<f64>,
    pub reject_rate: f64,
    pub offered_joint_attainment_status: SloStatus,
    pub accepted_joint_attainment_status: SloStatus,
    pub accepted_error_rate_status: SloStatus,
    pub reject_rate_status: SloStatus,
    pub ttft: LatencyDistribution,
    pub tpot: LatencyDistribution,
    /// Primary serving ITL: each visible interval contributes one sample.
    pub pooled_visible_itl: LatencyDistribution,
    /// Each contributing completed request contributes its maximum gap once.
    pub request_max_visible_itl: LatencyDistribution,
    pub raw_output: OutputThroughput,
    pub raw_counts_by_source: BTreeMap<RawOutputCountSource, u64>,
    pub successful_output: OutputThroughput,
    pub slo_output: OutputThroughput,
    pub confirmed_request_goodput_rps: f64,
    pub visible_evidence: VisibleEvidenceCounts,
    /// Contains the original timestamps, diagnostics and gaps without filtering.
    pub request_evidence: Vec<RequestSloEvidence>,
    pub requests: Vec<RequestSloEvaluation>,
    /// Does not include arrival delivery or steady-state validation.
    pub latency_and_outcome_status: SloStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SloEvaluationError(pub String);

impl std::fmt::Display for SloEvaluationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SloEvaluationError {}

fn invalid(message: impl Into<String>) -> SloEvaluationError {
    SloEvaluationError(message.into())
}

impl SloEvaluationConfig {
    /// Only explicit client-visible targets can construct a client evaluator.
    /// Server-token budgets must never silently become external thresholds.
    pub fn from_client_visible(
        config: &SloClientVisibleConfig,
    ) -> Result<Self, SloEvaluationError> {
        config.latency.validate().map_err(invalid)?;
        let result = Self {
            ttft_ms: config.latency.ttft_ms.get() as f64,
            tpot_ms: config.latency.tpot_ms.get() as f64,
            visible_itl_ms: config.latency.itl_ms.get() as f64,
            attainment: config.attainment.clone(),
            tpot_boundary: TpotBoundary::LastVisibleOutput,
        };
        result.validate()?;
        Ok(result)
    }

    pub fn validate(&self) -> Result<(), SloEvaluationError> {
        for (name, value) in [
            ("ttft_ms", self.ttft_ms),
            ("tpot_ms", self.tpot_ms),
            ("visible_itl_ms", self.visible_itl_ms),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid(format!("{name} must be finite and positive")));
            }
        }
        self.attainment.validate().map_err(invalid)
    }
}

fn validate_request(record: &RequestSloEvidence, index: usize) -> Result<(), SloEvaluationError> {
    if (record.outcome == RequestOutcome::Rejected)
        != (record.admission == AdmissionEvidence::Rejected)
    {
        return Err(invalid(format!(
            "request {index}: rejected outcome and admission must agree"
        )));
    }
    for (name, value) in [
        ("first_visible_ms", record.first_visible_ms),
        ("last_visible_ms", record.last_visible_ms),
        ("terminal_ms", record.terminal_ms),
    ] {
        if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err(invalid(format!(
                "request {index}: {name} must be finite and nonnegative"
            )));
        }
    }
    for (start, end) in [
        (record.first_visible_ms, record.last_visible_ms),
        (record.first_visible_ms, record.terminal_ms),
        (record.last_visible_ms, record.terminal_ms),
    ] {
        if start.zip(end).is_some_and(|(start, end)| start > end) {
            return Err(invalid(format!(
                "request {index}: inconsistent timestamp order"
            )));
        }
    }
    if record.raw_output_count.source == RawOutputCountSource::Usage
        && record.usage_output_tokens != Some(record.raw_output_count.count)
    {
        return Err(invalid(format!(
            "request {index}: conflicting usage counts"
        )));
    }
    if record.visible_text.as_ref().is_some_and(|text| {
        text.gaps_ms
            .iter()
            .any(|gap| !gap.is_finite() || *gap < 0.0)
    }) {
        return Err(invalid(format!(
            "request {index}: gaps must be finite and nonnegative"
        )));
    }
    Ok(())
}

fn combine(statuses: impl IntoIterator<Item = SloStatus>) -> SloStatus {
    let mut unknown = false;
    let mut applicable = false;
    for status in statuses {
        match status {
            SloStatus::Fail => return SloStatus::Fail,
            SloStatus::Unknown => unknown = true,
            SloStatus::Pass => applicable = true,
            SloStatus::NotApplicable => {}
        }
    }
    if unknown {
        SloStatus::Unknown
    } else if applicable {
        SloStatus::Pass
    } else {
        SloStatus::NotApplicable
    }
}

fn evaluate_request(
    record: &RequestSloEvidence,
    config: &SloEvaluationConfig,
) -> RequestSloEvaluation {
    let ttft = record.first_visible_ms.map_or_else(
        || MetricEvaluation::absent(SloStatus::Unknown, EvidenceIssue::MissingFirstVisibleOutput),
        |value| MetricEvaluation::measured(value, config.ttft_ms),
    );
    let tpot = match record.usage_output_tokens {
        None => MetricEvaluation::absent(SloStatus::Unknown, EvidenceIssue::MissingUsage),
        Some(0) => MetricEvaluation::absent(SloStatus::Unknown, EvidenceIssue::ZeroOutputTokens),
        Some(1) => {
            MetricEvaluation::absent(SloStatus::NotApplicable, EvidenceIssue::TooFewOutputTokens)
        }
        Some(tokens) => {
            let (end, missing) = match config.tpot_boundary {
                TpotBoundary::LegacyTerminal => {
                    (record.terminal_ms, EvidenceIssue::MissingTerminal)
                }
                TpotBoundary::LastVisibleOutput => (
                    record.last_visible_ms,
                    EvidenceIssue::MissingLastVisibleOutput,
                ),
            };
            match (record.first_visible_ms, end) {
                (None, _) => MetricEvaluation::absent(
                    SloStatus::Unknown,
                    EvidenceIssue::MissingFirstVisibleOutput,
                ),
                (_, None) => MetricEvaluation::absent(SloStatus::Unknown, missing),
                (Some(first), Some(last)) => MetricEvaluation::measured(
                    (last - first) / f64::from(tokens - 1),
                    config.tpot_ms,
                ),
            }
        }
    };
    let request_max_visible_gap = match &record.visible_text {
        None => MetricEvaluation::absent(SloStatus::Unknown, EvidenceIssue::MissingVisibleEvents),
        Some(text) if text.gaps_ms.len() != text.output_events.saturating_sub(1) as usize => {
            MetricEvaluation::absent(
                SloStatus::Unknown,
                EvidenceIssue::IncompleteVisibleIntervals,
            )
        }
        Some(text) if text.output_events < 2 => {
            MetricEvaluation::absent(SloStatus::NotApplicable, EvidenceIssue::TooFewVisibleEvents)
        }
        Some(text) => MetricEvaluation::measured(
            text.gaps_ms.iter().copied().fold(0.0, f64::max),
            config.visible_itl_ms,
        ),
    };
    let task_success = match record.outcome {
        RequestOutcome::Failed | RequestOutcome::Rejected => SloStatus::Fail,
        RequestOutcome::Pending => SloStatus::Unknown,
        RequestOutcome::Completed if record.usage_output_tokens == Some(0) => SloStatus::Fail,
        RequestOutcome::Completed if record.first_visible_ms.is_none() => SloStatus::Unknown,
        RequestOutcome::Completed => match &record.visible_text {
            Some(text) if text.output_events > 0 => SloStatus::Pass,
            _ => SloStatus::Unknown,
        },
    };
    RequestSloEvaluation {
        outcome: record.outcome,
        task_success,
        ttft,
        tpot,
        request_max_visible_gap,
        joint: combine([
            task_success,
            ttft.status,
            tpot.status,
            request_max_visible_gap.status,
        ]),
    }
}

fn distribution(
    samples: &[f64],
    request_evidence: StatusCounts,
    threshold: f64,
    quantile: f64,
    population: LatencyPopulation,
) -> LatencyDistribution {
    let measured = (!samples.is_empty()).then(|| percentile(samples, quantile));
    let status = if request_evidence.unknown > 0 {
        SloStatus::Unknown
    } else if let Some(value) = measured {
        if value <= threshold {
            SloStatus::Pass
        } else {
            SloStatus::Fail
        }
    } else if request_evidence.not_applicable > 0 {
        SloStatus::NotApplicable
    } else {
        SloStatus::Unknown
    };
    LatencyDistribution {
        population,
        observed_percentiles_ms: (!samples.is_empty()).then(|| RepeatPercentiles {
            p50: percentile(samples, 0.50),
            p75: percentile(samples, 0.75),
            p95: percentile(samples, 0.95),
            p99: percentile(samples, 0.99),
        }),
        evaluated_percentile: quantile * 100.0,
        observed_percentile_ms: measured,
        threshold_ms: threshold,
        sample_count: samples.len() as u64,
        request_evidence,
        status,
    }
}

fn attainment_status(counts: &StatusCounts, offered: u64, target: f64) -> SloStatus {
    if offered == 0 {
        SloStatus::Unknown
    } else if (counts.pass + counts.unknown) as f64 / (offered as f64) < target {
        SloStatus::Fail
    } else if counts.unknown > 0 {
        SloStatus::Unknown
    } else if counts.pass as f64 / offered as f64 >= target {
        SloStatus::Pass
    } else {
        SloStatus::Unknown
    }
}

fn throughput<'a>(
    records: impl Iterator<Item = &'a RequestSloEvidence>,
    duration_s: f64,
    usage_only: bool,
    unknown_eligibility_requests: u64,
) -> OutputThroughput {
    let mut known_tokens = 0;
    let mut included_requests = 0;
    let mut unknown_count_requests = 0;
    for record in records {
        included_requests += 1;
        let tokens = if usage_only {
            record.usage_output_tokens
        } else {
            match record.raw_output_count.source {
                RawOutputCountSource::Usage | RawOutputCountSource::EngineTokens => {
                    Some(record.raw_output_count.count)
                }
                RawOutputCountSource::TextEvents | RawOutputCountSource::Unknown => None,
            }
        };
        if let Some(tokens) = tokens {
            known_tokens += u64::from(tokens);
        } else {
            unknown_count_requests += 1;
        }
    }
    OutputThroughput {
        known_tokens,
        included_requests,
        unknown_count_requests,
        unknown_eligibility_requests,
        tokens_per_second: (unknown_count_requests == 0 && unknown_eligibility_requests == 0)
            .then_some(known_tokens as f64 / duration_s),
    }
}

pub fn evaluate_slo(
    config: &SloEvaluationConfig,
    records: &[RequestSloEvidence],
    duration_s: f64,
) -> Result<SloEvaluationReport, SloEvaluationError> {
    config.validate()?;
    if !duration_s.is_finite() || duration_s <= 0.0 {
        return Err(invalid("measurement duration must be finite and positive"));
    }
    if records.is_empty() {
        return Err(invalid("offered cohort must not be empty"));
    }
    for (index, record) in records.iter().enumerate() {
        validate_request(record, index)?;
    }
    let requests: Vec<_> = records
        .iter()
        .map(|record| evaluate_request(record, config))
        .collect();
    let mut outcomes = OutcomeCounts {
        offered: records.len() as u64,
        ..Default::default()
    };
    let mut admissions = AdmissionCounts::default();
    let mut task_success = StatusCounts::default();
    let mut joint = StatusCounts::default();
    let mut accepted_joint = StatusCounts::default();
    let mut accepted_task_success = StatusCounts::default();
    let mut ttft_samples = Vec::new();
    let mut tpot_samples = Vec::new();
    let mut gap_samples = Vec::new();
    let mut max_gap_samples = Vec::new();
    let mut ttft_counts = StatusCounts::default();
    let mut tpot_counts = StatusCounts::default();
    let mut gap_counts = StatusCounts::default();
    let mut pooled_gap_counts = StatusCounts::default();
    let mut raw_counts_by_source = BTreeMap::new();
    let mut visible_evidence = VisibleEvidenceCounts::default();
    for (record, evaluated) in records.iter().zip(&requests) {
        match record.admission {
            AdmissionEvidence::Accepted => {
                admissions.accepted += 1;
                accepted_joint.add(evaluated.joint);
                accepted_task_success.add(evaluated.task_success);
            }
            AdmissionEvidence::Rejected => admissions.rejected += 1,
            AdmissionEvidence::Unknown => admissions.unknown += 1,
        }
        match record.outcome {
            RequestOutcome::Completed => outcomes.completed += 1,
            RequestOutcome::Failed => outcomes.failed += 1,
            RequestOutcome::Rejected => outcomes.rejected += 1,
            RequestOutcome::Pending => outcomes.pending += 1,
        }
        task_success.add(evaluated.task_success);
        joint.add(evaluated.joint);
        *raw_counts_by_source
            .entry(record.raw_output_count.source)
            .or_insert(0) += u64::from(record.raw_output_count.count);
        if let Some(text) = &record.visible_text {
            // A failed or unfinished stream can contain the largest visible
            // stall. Preserve every observed gap regardless of task outcome.
            gap_samples.extend_from_slice(&text.gaps_ms);
            visible_evidence.requests_with_evidence += 1;
            visible_evidence.events += u64::from(text.output_events);
            visible_evidence.intervals += text.gaps_ms.len() as u64;
            visible_evidence.transport_coalesced_requests +=
                u64::from(text.transport_coalesced_output_chunks > 0);
            visible_evidence.event_usage_mismatch_requests += u64::from(
                record
                    .usage_output_tokens
                    .is_some_and(|tokens| tokens != text.output_events),
            );
            visible_evidence.missing_usage_requests +=
                u64::from(record.usage_output_tokens.is_none());
            if record.outcome != RequestOutcome::Completed {
                visible_evidence.failed_or_pending_intervals += text.gaps_ms.len() as u64;
            }
        }
        pooled_gap_counts.add(if record.outcome == RequestOutcome::Pending {
            // The stream is still open; the observed prefix cannot establish
            // complete interval coverage even when every current gap is small.
            SloStatus::Unknown
        } else {
            evaluated.request_max_visible_gap.status
        });
        if record.outcome != RequestOutcome::Completed {
            continue;
        }
        ttft_counts.add(evaluated.ttft.status);
        tpot_counts.add(evaluated.tpot.status);
        gap_counts.add(evaluated.request_max_visible_gap.status);
        if let Some(value) = evaluated.ttft.observed_ms {
            ttft_samples.push(value);
        }
        if let Some(value) = evaluated.tpot.observed_ms {
            tpot_samples.push(value);
        }
        if let Some(value) = evaluated.request_max_visible_gap.observed_ms {
            max_gap_samples.push(value);
        }
    }
    let ttft = distribution(
        &ttft_samples,
        ttft_counts,
        config.ttft_ms,
        config.attainment.ttft_percentile / 100.0,
        LatencyPopulation::CompletedRequests,
    );
    let tpot = distribution(
        &tpot_samples,
        tpot_counts,
        config.tpot_ms,
        config.attainment.tpot_percentile / 100.0,
        LatencyPopulation::CompletedRequests,
    );
    let pooled_visible_itl = distribution(
        &gap_samples,
        pooled_gap_counts,
        config.visible_itl_ms,
        config.attainment.itl_percentile / 100.0,
        LatencyPopulation::AllOfferedObservedGaps,
    );
    let request_max_visible_itl = distribution(
        &max_gap_samples,
        gap_counts,
        config.visible_itl_ms,
        config.attainment.itl_percentile / 100.0,
        LatencyPopulation::CompletedRequests,
    );
    let offered_joint_attainment_lower_bound = joint.pass as f64 / outcomes.offered as f64;
    let accepted_joint_attainment_lower_bound = (admissions.accepted > 0)
        .then_some(accepted_joint.pass as f64 / admissions.accepted as f64);
    let success_rate_lower_bound = task_success.pass as f64 / outcomes.offered as f64;
    let accepted_error_rate_lower_bound = (admissions.accepted > 0)
        .then_some(accepted_task_success.fail as f64 / admissions.accepted as f64);
    let reject_rate = outcomes.rejected as f64 / outcomes.offered as f64;
    let offered_joint_attainment_status = config
        .attainment
        .min_offered_joint_attainment
        .map_or(SloStatus::NotApplicable, |target| {
            attainment_status(&joint, outcomes.offered, target)
        });
    let accepted_joint_attainment_status = if admissions.unknown > 0 {
        SloStatus::Unknown
    } else {
        attainment_status(
            &accepted_joint,
            admissions.accepted,
            config.attainment.min_accepted_joint_attainment,
        )
    };
    let accepted_error_rate_status = if admissions.unknown > 0 {
        SloStatus::Unknown
    } else if accepted_error_rate_lower_bound
        .is_some_and(|rate| rate > config.attainment.max_error_rate)
    {
        SloStatus::Fail
    } else if accepted_error_rate_lower_bound.is_none() || accepted_task_success.unknown > 0 {
        SloStatus::Unknown
    } else {
        SloStatus::Pass
    };
    let reject_rate_status = if reject_rate <= config.attainment.max_reject_rate {
        SloStatus::Pass
    } else {
        SloStatus::Fail
    };
    let latency_and_outcome_status = combine([
        ttft.status,
        tpot.status,
        match config.attainment.itl_percentile_scope {
            SloItlPercentileScope::PooledGaps => pooled_visible_itl.status,
            SloItlPercentileScope::RequestMaximumGap => request_max_visible_itl.status,
        },
        offered_joint_attainment_status,
        accepted_joint_attainment_status,
        accepted_error_rate_status,
        reject_rate_status,
    ]);
    let raw_output = throughput(records.iter(), duration_s, false, 0);
    let successful_output = throughput(
        records
            .iter()
            .zip(&requests)
            .filter(|(_, evaluated)| evaluated.task_success == SloStatus::Pass)
            .map(|(record, _)| record),
        duration_s,
        true,
        task_success.unknown,
    );
    let slo_output = throughput(
        records
            .iter()
            .zip(&requests)
            .filter(|(_, evaluated)| evaluated.joint == SloStatus::Pass)
            .map(|(record, _)| record),
        duration_s,
        true,
        joint.unknown,
    );
    if [&raw_output, &successful_output, &slo_output]
        .into_iter()
        .any(|rate| {
            rate.tokens_per_second
                .is_some_and(|value| !value.is_finite())
        })
    {
        return Err(invalid("output throughput is not finite"));
    }
    let confirmed_request_goodput_rps = joint.pass as f64 / duration_s;
    if !confirmed_request_goodput_rps.is_finite() {
        return Err(invalid("request goodput is not finite"));
    }
    Ok(SloEvaluationReport {
        schema_version: SCHEMA_VERSION,
        config: config.clone(),
        duration_s,
        outcomes,
        admissions,
        task_success,
        offered_joint: joint,
        accepted_joint,
        accepted_task_success,
        offered_joint_attainment_lower_bound,
        accepted_joint_attainment_lower_bound,
        success_rate_lower_bound,
        accepted_error_rate_lower_bound,
        reject_rate,
        offered_joint_attainment_status,
        accepted_joint_attainment_status,
        accepted_error_rate_status,
        reject_rate_status,
        ttft,
        tpot,
        pooled_visible_itl,
        request_max_visible_itl,
        raw_output,
        raw_counts_by_source,
        successful_output,
        slo_output,
        confirmed_request_goodput_rps,
        visible_evidence,
        request_evidence: records.to_vec(),
        requests,
        latency_and_outcome_status,
    })
}

#[cfg(test)]
mod tests;
