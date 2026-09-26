use super::*;
use serde::{Deserialize, Serialize};

/// Explicit scope of the approximate inference, not a claim that finite pilot
/// data prove independence, exchangeability, or population coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceAssumptions {
    IndependentExchangeablePairedBlocksFixedWorkload,
}

/// Freeze before the baseline pilot. The final method binds this document's
/// digest after planning and before observing candidate measurements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenEligibilityPlan {
    pub schema_version: u32,
    pub frozen_unix_ns: u64,
    pub assumptions: InferenceAssumptions,
    /// Strictly increasing candidate allocations; selection uses the first
    /// satisfying forecast. These are not arbitrary sample-qualification gates.
    pub planning_pair_counts: Vec<u32>,
    pub planning_resamples: u32,
    pub seed: u64,
    /// Empirical CDF rank resolution, not an IID quantile confidence interval.
    /// Must be <= 0.01 to resolve the upper one-percent tail at all.
    pub maximum_request_rank_step: f64,
    pub maximum_visible_gap_rank_step: f64,
    /// These are preregistered acquisition diagnostics, not tests proving IID.
    pub maximum_order_log_ratio_shift: f64,
    pub maximum_time_trend_log_ratio_shift: f64,
    pub maximum_absolute_lag_one_correlation: f64,
    pub maximum_within_pair_idle_ns: u64,
    /// The pilot must exercise the same restart/warmup/reset protocol declared
    /// for the main experiment. A nonempty string alone does not certify it.
    pub independent_block_protocol: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PilotMetricDiagnostic {
    pub concurrency: u32,
    pub metric: ComparisonMetric,
    pub mean_forward_ratio: f64,
    pub mean_reverse_ratio: f64,
    pub order_log_ratio_shift: f64,
    pub time_trend_log_ratio_shift: f64,
    pub lag_one_correlation: f64,
    pub within_declared_limits: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanningMetricForecast {
    pub concurrency: u32,
    pub metric: ComparisonMetric,
    pub mean_pilot_ratio: f64,
    pub one_sided_bound: Option<f64>,
    pub relative_bound_width: Option<f64>,
    pub precision_target_met: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairAllocationForecast {
    pub planned_pairs: u32,
    pub metrics: Vec<PlanningMetricForecast>,
    pub all_precision_targets_met: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceEligibilityReport {
    pub schema_version: u32,
    pub final_contract_sha256: String,
    pub plan_sha256: String,
    pub plan: FrozenEligibilityPlan,
    pub pilot_contract_sha256: String,
    pub pilot_source_sha256: String,
    pub pilot_measurements_sha256: String,
    pub assumptions: InferenceAssumptions,
    pub pilot_finished_unix_ns: u64,
    pub selected_pairs: u32,
    pub planning_resamples: u32,
    pub planning_seed: u64,
    pub diagnostics: Vec<PilotMetricDiagnostic>,
    pub forecasts: Vec<PairAllocationForecast>,
    pub scope: String,
}

/// Only the raw-evidence verifier can construct this capability. In particular,
/// deserializing an InferenceEligibilityReport does not construct eligibility.
#[derive(Debug)]
pub struct VerifiedInferenceEligibility {
    pub(super) report: InferenceEligibilityReport,
    pub(super) pilot_runs: BTreeSet<(String, String, u32)>,
}

impl VerifiedInferenceEligibility {
    pub fn report(&self) -> &InferenceEligibilityReport {
        &self.report
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EligibilityFailure {
    pub issues: Vec<String>,
}
impl std::fmt::Display for EligibilityFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.issues.join("; "))
    }
}
impl std::error::Error for EligibilityFailure {}
impl From<ComparisonError> for EligibilityFailure {
    fn from(error: ComparisonError) -> Self {
        Self {
            issues: vec![error.0],
        }
    }
}
