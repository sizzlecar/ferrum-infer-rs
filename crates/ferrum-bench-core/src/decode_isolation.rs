//! Typed analysis for the decode-isolation benchmark workload.
//!
//! The HTTP collector records output-event timestamps on a shared monotonic
//! clock. This module classifies inter-event gaps before and during a long
//! prefill request without embedding a pass/fail performance threshold.

use crate::{percentile, Env, EnvHash, ItlEligibility, QualityIssueCounts, ScalarStats, Scenario};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodeEventTimeline {
    /// Monotonic milliseconds relative to the beginning of the run.
    pub output_event_ms: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DecodeLatencySummary {
    pub samples: u64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub max_ms: f64,
}

impl DecodeLatencySummary {
    fn from_samples(samples: &[f64]) -> Self {
        Self {
            samples: samples.len().try_into().unwrap_or(u64::MAX),
            p50_ms: percentile(samples, 0.50),
            p95_ms: percentile(samples, 0.95),
            max_ms: samples.iter().copied().reduce(f64::max).unwrap_or(0.0),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodeIsolationWindowMetrics {
    pub baseline_itl: DecodeLatencySummary,
    pub interference_itl: DecodeLatencySummary,
    pub maximum_decode_gap_ms: f64,
    pub decode_progress_events: u64,
    pub incumbents_with_progress: u32,
}

/// End marker for the only interval in which prefill interference is measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeIsolationWindowEnd {
    AggressorFirstOutputToken,
}

/// Behavior when the wire contract or orchestration invariants are invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeIsolationErrorPolicy {
    /// Emit diagnostics with all performance fields absent, then return success.
    EmitDiagnostics,
    /// Emit diagnostics with all performance fields absent, then return an error.
    EmitDiagnosticsAndFail,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodeIsolationScenarioContract {
    pub baseline_output_events_per_incumbent: u32,
    pub fixed_output_budget: bool,
    pub injection_requires_all_incumbents_ready: bool,
    pub interference_window_end: DecodeIsolationWindowEnd,
    pub post_aggressor_progress_required_per_incumbent: bool,
    /// Minimum number of scheduler token chunks covered by the long prefill.
    pub minimum_aggressor_scheduled_chunks: u32,
    pub kv_capacity_headroom_numerator: u32,
    pub kv_capacity_headroom_denominator: u32,
    pub invalid_evidence_policy: DecodeIsolationErrorPolicy,
    pub warmup_failure_always_fatal: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measured_error_rate_limit: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeIsolationCapabilities {
    pub effective_max_concurrent: u32,
    pub maximum_scheduled_tokens: u64,
    pub max_model_length: u64,
    pub kv_capacity_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_kv_capacity_tokens: Option<u64>,
    pub kv_block_size_tokens: u64,
    pub kv_block_size_source: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodeIsolationConfig {
    pub incumbents: u32,
    pub incumbents_source: String,
    pub incumbent_input_tokens: u64,
    pub incumbent_output_tokens: u64,
    pub aggressor_input_tokens: u64,
    pub aggressor_input_source: String,
    pub aggressor_scheduled_chunks: u64,
    pub aggressor_output_tokens: u64,
    pub aggregate_kv_budget_tokens: u64,
    pub aggregate_kv_budget_blocks: u64,
    /// Sum of logical request token budgets before allocator rounding.
    pub estimated_unrounded_aggregate_kv_tokens: u64,
    /// Aggregate KV allocation after rounding each sequence to a KV block.
    pub estimated_aggregate_kv_tokens: u64,
    pub estimated_aggregate_kv_blocks: u64,
    pub capabilities: DecodeIsolationCapabilities,
    pub contract: DecodeIsolationScenarioContract,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeIsolationRequestEvidence {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_request_id: Option<String>,
    pub success: bool,
    pub contract_valid: bool,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub output_token_count_source: String,
    pub itl_eligibility: ItlEligibility,
    pub output_events: u32,
    pub quality_issues: QualityIssueCounts,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeIsolationOrchestrationEvidence {
    pub incumbents_expected: u32,
    pub incumbents_ready_before_injection: u32,
    pub all_incumbents_ready_before_injection: bool,
    pub aggressor_first_token_observed: bool,
    pub incumbents_progressed_after_aggressor_first_token: u32,
    pub every_incumbent_progressed_after_aggressor_first_token: bool,
    pub all_tasks_drained: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeIsolationEvidenceValidity {
    pub warmup_valid: bool,
    pub all_incumbents_valid: bool,
    pub aggressor_valid: bool,
    pub orchestration_valid: bool,
    pub all_valid: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub invalid_reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodeIsolationRunReport {
    pub repeat: u32,
    pub orchestration: DecodeIsolationOrchestrationEvidence,
    pub validity: DecodeIsolationEvidenceValidity,
    /// Absent whenever evidence is invalid. Consumers must never infer a
    /// zero latency from missing evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<DecodeIsolationWindowMetrics>,
    /// Absent under the same validity rule as `metrics`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aggressor_ttft_ms: Option<f64>,
    pub incumbents: Vec<DecodeIsolationRequestEvidence>,
    pub aggressor: DecodeIsolationRequestEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodeIsolationAggregate {
    pub baseline_itl_p50_ms: ScalarStats,
    pub baseline_itl_p95_ms: ScalarStats,
    pub interference_itl_p50_ms: ScalarStats,
    pub interference_itl_p95_ms: ScalarStats,
    pub maximum_decode_gap_ms: ScalarStats,
    pub decode_progress_events: ScalarStats,
    pub incumbents_with_progress: ScalarStats,
    pub incumbent_progress_fraction: ScalarStats,
    pub aggressor_ttft_ms: ScalarStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeIsolationReport {
    pub schema_version: u32,
    pub scenario: Scenario,
    pub model: String,
    pub backend: String,
    pub n_repeats: u32,
    pub config: DecodeIsolationConfig,
    /// Present only when every repeat carries valid numeric evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aggregate: Option<DecodeIsolationAggregate>,
    pub runs: Vec<DecodeIsolationRunReport>,
    /// True only when every run has valid wire, usage, and orchestration
    /// evidence. Random-token workloads do not evaluate semantic answer quality.
    pub all_evidence_valid: bool,
    pub semantic_correctness_evaluated: bool,
    pub env: Env,
    pub env_hash: EnvHash,
}

pub fn aggregate_decode_isolation_runs(
    runs: &[DecodeIsolationRunReport],
) -> Option<DecodeIsolationAggregate> {
    let valid: Vec<_> = runs
        .iter()
        .map(|run| {
            if !run.validity.all_valid {
                return None;
            }
            Some((
                run.metrics.as_ref()?,
                run.aggressor_ttft_ms?,
                run.orchestration.incumbents_expected,
            ))
        })
        .collect::<Option<_>>()?;
    if valid.is_empty() {
        return None;
    }
    let stats = |value: fn(&(&DecodeIsolationWindowMetrics, f64, u32)) -> f64| {
        ScalarStats::from_samples(&valid.iter().map(value).collect::<Vec<_>>())
    };
    Some(DecodeIsolationAggregate {
        baseline_itl_p50_ms: stats(|(metrics, _, _)| metrics.baseline_itl.p50_ms),
        baseline_itl_p95_ms: stats(|(metrics, _, _)| metrics.baseline_itl.p95_ms),
        interference_itl_p50_ms: stats(|(metrics, _, _)| metrics.interference_itl.p50_ms),
        interference_itl_p95_ms: stats(|(metrics, _, _)| metrics.interference_itl.p95_ms),
        maximum_decode_gap_ms: stats(|(metrics, _, _)| metrics.maximum_decode_gap_ms),
        decode_progress_events: stats(|(metrics, _, _)| metrics.decode_progress_events as f64),
        incumbents_with_progress: stats(|(metrics, _, _)| metrics.incumbents_with_progress as f64),
        incumbent_progress_fraction: stats(|(metrics, _, expected)| {
            if *expected == 0 {
                0.0
            } else {
                metrics.incumbents_with_progress as f64 / f64::from(*expected)
            }
        }),
        aggressor_ttft_ms: stats(|(_, ttft, _)| *ttft),
    })
}

/// Analyze incumbent decode timelines around one aggressor prefill.
///
/// Baseline samples end at or before `injection_ms`. Interference samples are
/// complete inter-event gaps that overlap `[injection_ms,
/// aggressor_first_token_ms]`; retaining boundary-crossing gaps is important,
/// because a fully starved decoder may emit no event inside the window. Decode
/// progress counts events observed strictly after injection through the
/// aggressor's first token, inclusive.
pub fn analyze_decode_isolation(
    timelines: &[DecodeEventTimeline],
    injection_ms: f64,
    aggressor_first_token_ms: f64,
) -> Result<DecodeIsolationWindowMetrics, String> {
    if !injection_ms.is_finite()
        || !aggressor_first_token_ms.is_finite()
        || injection_ms < 0.0
        || aggressor_first_token_ms < injection_ms
    {
        return Err("decode-isolation window must be finite and ordered".to_string());
    }

    let mut baseline = Vec::new();
    let mut interference = Vec::new();
    let mut progress = 0_u64;
    let mut incumbents_with_progress = 0_u32;

    for timeline in timelines {
        if timeline
            .output_event_ms
            .windows(2)
            .any(|pair| !pair[0].is_finite() || !pair[1].is_finite() || pair[1] < pair[0])
        {
            return Err("decode event timelines must be finite and monotonic".to_string());
        }
        let request_progress = timeline
            .output_event_ms
            .iter()
            .filter(|&&event| event > injection_ms && event <= aggressor_first_token_ms)
            .count();
        progress = progress
            .checked_add(request_progress.try_into().unwrap_or(u64::MAX))
            .ok_or_else(|| "decode progress count overflow".to_string())?;
        if request_progress > 0 {
            incumbents_with_progress = incumbents_with_progress
                .checked_add(1)
                .ok_or_else(|| "incumbent progress count overflow".to_string())?;
        }

        for pair in timeline.output_event_ms.windows(2) {
            let gap = pair[1] - pair[0];
            if pair[1] <= injection_ms {
                baseline.push(gap);
            }
            if pair[0] < aggressor_first_token_ms && pair[1] > injection_ms {
                interference.push(gap);
            }
        }
    }

    let baseline_itl = DecodeLatencySummary::from_samples(&baseline);
    let interference_itl = DecodeLatencySummary::from_samples(&interference);
    Ok(DecodeIsolationWindowMetrics {
        baseline_itl,
        maximum_decode_gap_ms: interference_itl.max_ms,
        interference_itl,
        decode_progress_events: progress,
        incumbents_with_progress,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_run() -> DecodeIsolationRunReport {
        let evidence = DecodeIsolationRequestEvidence {
            role: "incumbent".to_string(),
            server_request_id: Some("chatcmpl-test".to_string()),
            success: true,
            contract_valid: true,
            input_tokens: 8,
            output_tokens: 4,
            output_token_count_source: "usage".to_string(),
            itl_eligibility: ItlEligibility::Eligible,
            output_events: 4,
            quality_issues: QualityIssueCounts::default(),
        };
        DecodeIsolationRunReport {
            repeat: 0,
            orchestration: DecodeIsolationOrchestrationEvidence {
                incumbents_expected: 1,
                incumbents_ready_before_injection: 1,
                all_incumbents_ready_before_injection: true,
                aggressor_first_token_observed: true,
                incumbents_progressed_after_aggressor_first_token: 1,
                every_incumbent_progressed_after_aggressor_first_token: true,
                all_tasks_drained: true,
            },
            validity: DecodeIsolationEvidenceValidity {
                warmup_valid: true,
                all_incumbents_valid: true,
                aggressor_valid: true,
                orchestration_valid: true,
                all_valid: true,
                invalid_reasons: vec![],
            },
            metrics: Some(DecodeIsolationWindowMetrics {
                baseline_itl: DecodeLatencySummary::from_samples(&[2.0, 3.0]),
                interference_itl: DecodeLatencySummary::from_samples(&[6.0]),
                maximum_decode_gap_ms: 6.0,
                decode_progress_events: 1,
                incumbents_with_progress: 1,
            }),
            aggressor_ttft_ms: Some(5.0),
            incumbents: vec![evidence.clone()],
            aggressor: DecodeIsolationRequestEvidence {
                role: "aggressor".to_string(),
                output_tokens: 1,
                output_events: 1,
                itl_eligibility: ItlEligibility::TooShort,
                ..evidence
            },
        }
    }

    #[test]
    fn classifies_baseline_and_overlapping_interference_gaps() {
        let metrics = analyze_decode_isolation(
            &[
                DecodeEventTimeline {
                    output_event_ms: vec![0.0, 10.0, 20.0, 70.0, 80.0],
                },
                DecodeEventTimeline {
                    output_event_ms: vec![2.0, 12.0, 22.0, 42.0, 72.0],
                },
            ],
            25.0,
            75.0,
        )
        .unwrap();

        assert_eq!(metrics.baseline_itl.samples, 4);
        assert_eq!(metrics.baseline_itl.p50_ms, 10.0);
        assert_eq!(metrics.interference_itl.samples, 4);
        assert_eq!(metrics.maximum_decode_gap_ms, 50.0);
        assert_eq!(metrics.decode_progress_events, 3);
        assert_eq!(metrics.incumbents_with_progress, 2);
    }

    #[test]
    fn boundary_crossing_gap_exposes_zero_progress_starvation() {
        let metrics = analyze_decode_isolation(
            &[DecodeEventTimeline {
                output_event_ms: vec![0.0, 10.0, 110.0],
            }],
            20.0,
            100.0,
        )
        .unwrap();

        assert_eq!(metrics.decode_progress_events, 0);
        assert_eq!(metrics.incumbents_with_progress, 0);
        assert_eq!(metrics.interference_itl.samples, 1);
        assert_eq!(metrics.maximum_decode_gap_ms, 100.0);
    }

    #[test]
    fn rejects_non_monotonic_or_inverted_windows() {
        assert!(analyze_decode_isolation(&[], 2.0, 1.0).is_err());
        assert!(analyze_decode_isolation(
            &[DecodeEventTimeline {
                output_event_ms: vec![2.0, 1.0],
            }],
            2.0,
            3.0,
        )
        .is_err());
    }

    #[test]
    fn report_contract_round_trips_json() {
        let run = valid_run();
        let report = DecodeIsolationReport {
            schema_version: 1,
            scenario: Scenario::DecodeIsolation,
            model: "model#tag".to_string(),
            backend: "cuda".to_string(),
            n_repeats: 1,
            config: DecodeIsolationConfig {
                incumbents: 1,
                incumbents_source: "cli".to_string(),
                incumbent_input_tokens: 8,
                incumbent_output_tokens: 4,
                aggressor_input_tokens: 65,
                aggressor_input_source: "cli".to_string(),
                aggressor_scheduled_chunks: 2,
                aggressor_output_tokens: 1,
                aggregate_kv_budget_tokens: 128,
                aggregate_kv_budget_blocks: 128,
                estimated_unrounded_aggregate_kv_tokens: 78,
                estimated_aggregate_kv_tokens: 78,
                estimated_aggregate_kv_blocks: 78,
                capabilities: DecodeIsolationCapabilities {
                    effective_max_concurrent: 3,
                    maximum_scheduled_tokens: 64,
                    max_model_length: 128,
                    kv_capacity_tokens: 144,
                    selected_kv_capacity_tokens: Some(128),
                    kv_block_size_tokens: 1,
                    kv_block_size_source: "default_unit_block".to_string(),
                },
                contract: DecodeIsolationScenarioContract {
                    baseline_output_events_per_incumbent: 2,
                    fixed_output_budget: true,
                    injection_requires_all_incumbents_ready: true,
                    interference_window_end: DecodeIsolationWindowEnd::AggressorFirstOutputToken,
                    post_aggressor_progress_required_per_incumbent: true,
                    minimum_aggressor_scheduled_chunks: 2,
                    kv_capacity_headroom_numerator: 9,
                    kv_capacity_headroom_denominator: 10,
                    invalid_evidence_policy: DecodeIsolationErrorPolicy::EmitDiagnostics,
                    warmup_failure_always_fatal: true,
                    measured_error_rate_limit: None,
                },
            },
            aggregate: aggregate_decode_isolation_runs(std::slice::from_ref(&run)),
            runs: vec![run],
            all_evidence_valid: true,
            semantic_correctness_evaluated: false,
            env: Env::default(),
            env_hash: Env::default().hash(),
        };
        let encoded = serde_json::to_string(&report).unwrap();
        let decoded: DecodeIsolationReport = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.scenario, Scenario::DecodeIsolation);
        assert_eq!(decoded.model, "model#tag");
        assert!(decoded.aggregate.is_some());
        assert!(decoded.runs[0].metrics.is_some());
    }

    #[test]
    fn invalid_repeat_suppresses_all_aggregate_evidence() {
        let mut invalid = valid_run();
        invalid.validity.all_valid = false;
        assert!(aggregate_decode_isolation_runs(&[valid_run(), invalid]).is_none());
    }
}
