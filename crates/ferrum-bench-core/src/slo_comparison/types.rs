use crate::{
    dataset::{ShareGptDatasetEvidence, ShareGptSelection},
    env::HttpRequestSampling,
    slo::{SloEvaluationConfig, SloEvaluationReport, SloStatus},
    BenchReport,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonMetric {
    TtftP50,
    TtftP99,
    TpotP50,
    TpotP99,
    VisibleItlP50,
    VisibleItlP99,
    SuccessfulUsageOutputTps,
}

impl ComparisonMetric {
    pub const ALL: [Self; 7] = [
        Self::TtftP50,
        Self::TtftP99,
        Self::TpotP50,
        Self::TpotP99,
        Self::VisibleItlP50,
        Self::VisibleItlP99,
        Self::SuccessfulUsageOutputTps,
    ];
    pub(super) fn is_throughput(self) -> bool {
        self == Self::SuccessfulUsageOutputTps
    }
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::TtftP50 => "TTFT P50 (ms)",
            Self::TtftP99 => "TTFT P99 (ms)",
            Self::TpotP50 => "TPOT P50 (ms/token)",
            Self::TpotP99 => "TPOT P99 (ms/token)",
            Self::VisibleItlP50 => "Visible ITL P50 (ms)",
            Self::VisibleItlP99 => "Visible ITL P99 (ms)",
            Self::SuccessfulUsageOutputTps => "Successful usage output (token/s)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CellScope {
    Primary,
    Diagnostic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenCell {
    pub concurrency: u32,
    pub scope: CellScope,
}

/// Same logical capacity across arms, fixed throughout a client C sweep.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedServerCapacity {
    pub slots: u32,
    pub context_tokens_per_request: u32,
    pub batch_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SharedExecutionIdentity {
    pub hardware_fingerprint_sha256: String,
    pub hardware_label: String,
    pub model_content_sha256: String,
    pub weight_precision: String,
    pub kv_precision: String,
    pub tokenizer_sha256: String,
    pub chat_template_sha256: String,
    pub client_binary_sha256: String,
    pub client_slo_config_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenServerIdentity {
    pub implementation: String,
    pub backend: String,
    pub request_model_alias: String,
    pub binary_sha256: String,
    pub effective_configuration_sha256: String,
    pub numerical_policy: String,
    /// Declared implementation differences, such as allocator budget or ubatch.
    pub intentional_differences: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPair {
    pub pair_id: String,
    pub selection: ShareGptSelection,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenComparisonContract {
    pub schema_version: u32,
    pub frozen_unix_ns: u64,
    pub cells: Vec<FrozenCell>,
    pub pairs: Vec<FrozenPair>,
    pub shared: SharedExecutionIdentity,
    pub baseline: FrozenServerIdentity,
    pub candidate: FrozenServerIdentity,
    pub capacity: FixedServerCapacity,
    /// Source/filter policy uses the existing schema. Selections are frozen by
    /// pair above; `source_path` and this field's `repeats` are not identities.
    pub dataset: ShareGptDatasetEvidence,
    pub http_connection_mode: String,
    pub sampling: HttpRequestSampling,
    pub memory: FrozenMemoryPolicy,
    pub slo: SloEvaluationConfig,
    /// Exactly six latency upper ratios and one successful-usage TPS lower ratio.
    /// Values come from the externally frozen contract, never implicit defaults.
    pub ratio_limits: BTreeMap<ComparisonMetric, f64>,
    pub uncertainty: Option<FrozenStatisticalMethod>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenStatisticalMethod {
    pub method_id: String,
    pub analysis_unit: String,
    pub configuration_sha256: String,
    /// None preserves older external-method declarations. Only a recognized,
    /// digest-bound typed configuration can drive built-in CI computation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paired_bootstrap: Option<super::FrozenPairedBootstrap>,
}

/// Evidence is data, never a caller-supplied success decision. This descriptive
/// comparator never trusts these externally supplied intervals. Its optional
/// built-in bootstrap is computed from the validated paired observations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatisticalEvidence {
    pub method: FrozenStatisticalMethod,
    pub paired_measurements_sha256: String,
    pub source_artifact_sha256: String,
    pub confidence_level: f64,
    pub ratio_intervals: BTreeMap<ComparisonMetric, (f64, f64)>,
}

/// Acquired and verified by the loader/orchestrator, not inferred from a client
/// CPU model name or matching endpoint URL. Capture bounds must include the run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArmExecutionEvidence {
    pub shared: SharedExecutionIdentity,
    pub server: FrozenServerIdentity,
    pub capacity: FixedServerCapacity,
    pub source_manifest_sha256: String,
    pub measurement_started_unix_ns: u64,
    pub measurement_ended_unix_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryMeasurement {
    SampledDeviceAllocation,
    SampledOsPhysicalFootprint,
    /// Native OS high water mark for the entire process lifetime, not samples.
    ProcessPeakPhysicalFootprint,
    ProcessMaximumRss,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryPeakEvidence {
    pub measurement: MemoryMeasurement,
    pub peak_bytes: Option<u64>,
    pub source_sha256: String,
    pub complete: bool,
    pub error_count: u64,
    /// For sampled metrics these are the first and last actual sample times,
    /// not merely the sampler process launch/exit timestamps.
    pub started_unix_ns: u64,
    pub ended_unix_ns: u64,
    /// Explicit measurement boundary; native process peaks use process_lifetime.
    pub window: String,
    pub sample_count: Option<u64>,
    pub interval_ms: Option<u64>,
    pub max_sample_gap_ns: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PeakMemoryEvidence {
    pub device_allocation: MemoryPeakEvidence,
    pub os_footprint: MemoryPeakEvidence,
    pub maximum_rss: MemoryPeakEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledMemoryPolicy {
    pub window: String,
    pub interval_ms: u64,
    /// Predeclared tolerated sampler stall; zero does not mean unlimited.
    pub max_sample_gap_ns: u64,
}

/// The untagged sampled branch preserves the original contract's JSON bytes.
/// New process-lifetime evidence must opt into an explicit versioned method.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OsFootprintPolicy {
    Sampled(SampledMemoryPolicy),
    ProcessLifetime(ProcessLifetimeMemoryPolicy),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessLifetimeMemoryKind {
    #[serde(rename = "macos_time_l_v1")]
    MacosTimeLV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessLifetimeMemoryPolicy {
    pub kind: ProcessLifetimeMemoryKind,
}

impl From<SampledMemoryPolicy> for OsFootprintPolicy {
    fn from(value: SampledMemoryPolicy) -> Self {
        Self::Sampled(value)
    }
}

impl OsFootprintPolicy {
    pub fn measurement(&self) -> MemoryMeasurement {
        match self {
            Self::Sampled(_) => MemoryMeasurement::SampledOsPhysicalFootprint,
            Self::ProcessLifetime(_) => MemoryMeasurement::ProcessPeakPhysicalFootprint,
        }
    }

    pub fn window(&self) -> &str {
        match self {
            Self::Sampled(policy) => &policy.window,
            Self::ProcessLifetime(_) => "process_lifetime",
        }
    }

    pub fn sampled(&self) -> Option<&SampledMemoryPolicy> {
        match self {
            Self::Sampled(policy) => Some(policy),
            Self::ProcessLifetime(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenMemoryPolicy {
    pub device_allocation: SampledMemoryPolicy,
    pub os_footprint: OsFootprintPolicy,
    pub maximum_rss_window: String,
}

/// A projection of one actual sidecar repeat. Different pairs may borrow
/// different BenchReports, allowing the server to restart between repetitions.
pub struct ArmRepeatInput<'a> {
    pub legacy_benchmark: &'a BenchReport,
    /// Zero-based array/correlation index. The legacy summary's `repeat` is a
    /// one-based display sequence and is adapted explicitly by the validator.
    pub report_repeat_index: u32,
    pub evaluation: &'a SloEvaluationReport,
    pub execution: &'a ArmExecutionEvidence,
    pub memory: Option<&'a PeakMemoryEvidence>,
}

pub struct PairedRepeatInput<'a> {
    pub pair_id: String,
    pub baseline: ArmRepeatInput<'a>,
    pub candidate: ArmRepeatInput<'a>,
}

pub struct ComparisonCellInput<'a> {
    pub concurrency: u32,
    pub pairs: Vec<PairedRepeatInput<'a>>,
    pub statistical_evidence: Option<&'a StatisticalEvidence>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonStatus {
    ProofPass,
    ObservedPass,
    Failed,
    Unknown,
    Inconclusive,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepeatRatio {
    pub baseline: f64,
    pub candidate: f64,
    pub candidate_over_baseline: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricComparison {
    pub limit: f64,
    /// Arithmetic mean of per-pair candidate/baseline ratios, not ratio of means.
    pub mean_paired_ratio: Option<f64>,
    pub observed_ratio_range: Option<(f64, f64)>,
    pub status: ComparisonStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedRepeatComparison {
    pub pair_id: String,
    pub ratios: BTreeMap<ComparisonMetric, RepeatRatio>,
    pub baseline_memory: Option<PeakMemoryEvidence>,
    pub candidate_memory: Option<PeakMemoryEvidence>,
    pub baseline_source: Option<ObservedArmSource>,
    pub candidate_source: Option<ObservedArmSource>,
    pub issues: Vec<String>,
    pub evidence_status: ComparisonStatus,
    pub status: ComparisonStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservedArmSource {
    pub benchmark_run_id: Option<String>,
    pub cell_id: Option<String>,
    pub report_repeat_index: u32,
    pub execution: ArmExecutionEvidence,
    /// Recomputed from this arm's original request evidence using the frozen
    /// absolute thresholds; independent of the paired improvement decision.
    pub absolute_slo_status: SloStatus,
    pub offered_requests: usize,
    pub failed_requests: u64,
    pub rejected_requests: u64,
    pub pending_requests: u64,
    pub observed_visible_gaps: usize,
    #[serde(default)]
    pub visible_gap_requests: usize,
    pub usage_output_tokens: Vec<Option<u32>>,
    pub server_input_tokens: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellComparison {
    pub concurrency: u32,
    pub scope: CellScope,
    pub pairs: Vec<PairedRepeatComparison>,
    pub metrics: BTreeMap<ComparisonMetric, MetricComparison>,
    pub issues: Vec<String>,
    pub status: ComparisonStatus,
    pub statistical_evidence: Option<StatisticalEvidence>,
    pub paired_measurements_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloComparisonReport {
    pub schema_version: u32,
    pub frozen_contract_sha256: String,
    pub contract: FrozenComparisonContract,
    pub cells: Vec<CellComparison>,
    pub issues: Vec<String>,
    pub status: ComparisonStatus,
    pub uncertainty_scope: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub computed_bootstrap: Option<super::ComputedBootstrapInference>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inference_eligibility: Option<super::InferenceEligibilityReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inference_eligibility_failure: Option<super::EligibilityFailure>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComparisonError(pub String);
impl std::fmt::Display for ComparisonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for ComparisonError {}
