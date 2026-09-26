use super::super::{ComparisonMetric, ComparisonStatus};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BootstrapEstimand {
    /// E[candidate metric / baseline metric] across independent, exchangeable paired runs;
    /// not ratio of means, mean log ratio, or a median/win probability.
    ArithmeticMeanPairedRatio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairedArmOrder {
    BaselineFirst,
    CandidateFirst,
}

/// A preregistered design declaration, NOT a verified claim of independence,
/// P99 adequacy, or bootstrap coverage. The pilot's raw evidence is not loaded
/// by this version. Nothing in this structure certifies statistical success.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeclaredBootstrapDesign {
    pub pilot_source_sha256: String,
    pub pilot_finished_unix_ns: u64,
    pub planned_pairs: u32,
    pub minimum_measured_requests_per_arm: u32,
    pub minimum_gap_bearing_requests_per_arm: u32,
    pub minimum_visible_gaps_per_arm: u64,
    /// Alternate AB/BA between complete paired blocks, in frozen pair order.
    pub first_pair_order: PairedArmOrder,
    /// Acquisition protocol for independent, exchangeable blocks, including
    /// restart, warmup and thermal/reset policy. Timestamps cannot establish
    /// independence or absence of residual order/drift effects.
    pub independent_block_protocol: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPairedBootstrap {
    pub schema_version: u32,
    pub estimand: BootstrapEstimand,
    pub seed: u64,
    pub resamples: u32,
    /// Total declared error budget. Subtract the Monte Carlo part before
    /// Bonferroni-dividing across all seven metrics of all primary cells.
    pub family_alpha: f64,
    pub monte_carlo_error_budget: f64,
    /// Maximum relative distance from point estimate to its one-sided bound.
    /// All seven targets must be declared; no workload defaults are inferred.
    pub maximum_relative_bound_width: BTreeMap<ComparisonMetric, f64>,
    /// Optional pre-pilot planning document; required to issue inference
    /// eligibility. Missing preserves the computation-only, unverified path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eligibility_plan_sha256: Option<String>,
    pub declared_design: DeclaredBootstrapDesign,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BootstrapCalibrationStatus {
    /// Raw independent pilot eligibility has not been verified.
    Unverified,
    /// Original pilot evidence passed frozen planning and diagnostics. Coverage
    /// remains approximate under explicitly declared experimental assumptions.
    EligibleUnderDeclaredAssumptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BootstrapBoundDirection {
    Upper,
    Lower,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapMetricBound {
    pub point_estimate: f64,
    pub direction: BootstrapBoundDirection,
    pub one_sided_bound: Option<f64>,
    pub relative_bound_width: Option<f64>,
    pub declared_maximum_relative_width: f64,
    pub precision_target_met: bool,
    /// A numerical comparison only; not calibrated statistical proof.
    pub strict_limit_cleared: bool,
    pub degenerate_empirical_distribution: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapPrimaryCell {
    pub concurrency: u32,
    pub pairs: usize,
    pub metrics: BTreeMap<ComparisonMetric, BootstrapMetricBound>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputedBootstrapInference {
    pub method_id: String,
    pub analysis_unit: String,
    pub estimand: BootstrapEstimand,
    pub configuration_sha256: String,
    pub paired_measurements_sha256: String,
    pub random_generator: String,
    pub seed: u64,
    pub resamples: u32,
    pub family_size: usize,
    pub family_alpha: f64,
    pub monte_carlo_error_budget: f64,
    pub per_comparison_tail_probability: f64,
    /// One-based kth smallest/largest bootstrap replicate used for the bound.
    /// None means finite simulation cannot resolve the requested tail budget.
    pub conservative_tail_rank: Option<usize>,
    pub calibration: BootstrapCalibrationStatus,
    /// ProofPass is conditional on declared assumptions and verified raw-pilot
    /// eligibility, never an unconditional population-coverage guarantee.
    pub status: ComparisonStatus,
    pub cells: Vec<BootstrapPrimaryCell>,
    pub issues: Vec<String>,
    pub scope: String,
}
