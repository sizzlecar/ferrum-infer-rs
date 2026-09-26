use super::super::*;
use crate::{slo::SloEvaluationReport, BenchReport};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactFileRef {
    /// Relative to the comparison manifest directory; resolved inside it.
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SidecarRef {
    pub file: ArtifactFileRef,
    /// Zero-based nonempty JSONL record, not physical line number.
    pub record_index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "format", rename_all = "snake_case", deny_unknown_fields)]
pub enum DeviceMemoryRef {
    FerrumMetalV1 {
        file: ArtifactFileRef,
    },
    /// Versioned raw observations; acquisition support is not supplied here.
    SampledMemoryV1 {
        file: ArtifactFileRef,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryArtifactRefs {
    pub device: Option<DeviceMemoryRef>,
    pub footprint: Option<ArtifactFileRef>,
    pub rss: Option<ArtifactFileRef>,
    /// Mutually exclusive with footprint/rss. Both peaks are parsed from the
    /// original native report; the declaration contains no peak or pass flag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub process_lifetime: Option<ProcessLifetimeMemoryRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "format", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProcessLifetimeMemoryRef {
    #[serde(rename = "macos_time_l_v1")]
    MacosTimeLV1 { capture: ArtifactFileRef },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MacosTimeSubject {
    /// A declaration by acquisition, never attested by parsing a time report.
    DeclaredDirectServerChild,
}

/// Acquisition-time source binding. Native time output contains neither PID nor
/// birth/death timestamps. These declared bounds must enclose the benchmark;
/// the loader cannot establish an actual parent/child relationship or execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MacosTimeCaptureArtifact {
    pub schema_version: u32,
    pub identity: RunArtifactIdentity,
    pub subject: MacosTimeSubject,
    pub process_started_unix_ns: u64,
    pub process_ended_unix_ns: Option<u64>,
    /// Original LC_ALL=C /usr/bin/time -l output. Values are bytes on macOS.
    pub time_output: ArtifactFileRef,
    /// Original wait/exit result as a decimal shell exit status. Missing means
    /// incomplete evidence; a declared file that cannot be loaded is an error.
    pub exit_status: Option<ArtifactFileRef>,
    /// Optional original command/ps evidence for human source review. Hashing
    /// these bytes does not automatically establish their semantic claims.
    pub launch_evidence: Option<ArtifactFileRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArmArtifactRefs {
    pub sidecar: SidecarRef,
    pub repeat_index: u32,
    pub selection_sha256: String,
    pub execution: ArtifactFileRef,
    pub memory: MemoryArtifactRefs,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairArtifactRefs {
    pub pair_id: String,
    pub baseline: ArmArtifactRefs,
    pub candidate: ArmArtifactRefs,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CellArtifactRefs {
    pub concurrency: u32,
    pub pairs: Vec<PairArtifactRefs>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComparisonArtifactManifest {
    pub schema_version: u32,
    pub contract: ArtifactFileRef,
    pub cells: Vec<CellArtifactRefs>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eligibility: Option<EligibilityArtifactRefs>,
}

/// Optional pre-pilot planning and raw A/A evidence. Referenced pilot manifests
/// cannot themselves reference eligibility; paths use the outer manifest root.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EligibilityArtifactRefs {
    pub plan: ArtifactFileRef,
    pub pilot_manifest: ArtifactFileRef,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunArtifactIdentity {
    pub benchmark_run_id: String,
    pub cell_id: String,
    pub repeat_index: u32,
    pub server_pid: u32,
}

/// An acquisition-time declaration, not hardware attestation. The loader hashes
/// these original bytes and aligns them with the frozen contract and sidecar.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionArtifact {
    pub schema_version: u32,
    pub identity: RunArtifactIdentity,
    pub shared: SharedExecutionIdentity,
    pub server: FrozenServerIdentity,
    pub capacity: FixedServerCapacity,
    pub measurement_started_unix_ns: u64,
    pub measurement_ended_unix_ns: u64,
    pub device_registry_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryObservation {
    pub elapsed_ns: u64,
    pub bytes: Option<u64>,
    pub error: Option<String>,
}

/// Wire contract for a future external/process sampler. This loader does not
/// implement OS footprint, process RSS or llama.cpp device-memory acquisition.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledMemoryArtifact {
    pub schema_version: u32,
    pub identity: RunArtifactIdentity,
    pub measurement: MemoryMeasurement,
    pub collector: String,
    pub window: String,
    pub started_unix_ns: u64,
    pub interval_ms: u64,
    pub observations: Vec<MemoryObservation>,
    pub finished_elapsed_ns: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaximumRssArtifact {
    pub schema_version: u32,
    pub identity: RunArtifactIdentity,
    pub collector: String,
    pub window: String,
    pub process_started_unix_ns: u64,
    pub process_ended_unix_ns: Option<u64>,
    pub maximum_rss_bytes: Option<u64>,
    pub error: Option<String>,
}

/// Exact outer wire schema emitted by bench_serve/slo_report.rs, including
/// arrival metadata. Legacy benchmark and SLO evaluation schemas are reused.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SloSidecarCell {
    pub schema_version: u32,
    pub config_sha256: String,
    pub legacy_benchmark: BenchReport,
    pub repeats: Vec<SloSidecarRepeat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SloSidecarRepeat {
    pub repeat_index: u32,
    pub evaluation: SloEvaluationReport,
    pub arrivals: Vec<SidecarArrival>,
    pub request_start_window_s: Option<f64>,
    pub observed_request_start_rate_rps: Option<f64>,
    pub max_client_dispatch_backlog: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SidecarArrival {
    pub scheduled_arrival_ms: Option<f64>,
    pub dispatched_ms: Option<f64>,
    pub request_started_ms: Option<f64>,
    pub client_dispatch_backlog: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedArtifact {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactComparisonReport {
    pub schema_version: u32,
    pub manifest_sha256: String,
    pub verified_files: Vec<VerifiedArtifact>,
    pub evidence_boundary: String,
    pub comparison: SloComparisonReport,
}

impl ArtifactComparisonReport {
    pub fn to_markdown(&self, language: MarkdownLanguage) -> String {
        let note = match language {
            MarkdownLanguage::English => "Artifact hashes verify file integrity and declared identity alignment, not hardware execution. Native time lifetime peaks are parsed when supplied; process identity and parent/child relationships remain acquisition-time declarations. This loader does not acquire OS samples.",
            MarkdownLanguage::Chinese => "产物哈希验证文件完整性及声明身份的一致性，不证明硬件真实执行；提供原生 time 文件时解析进程寿命峰值，进程身份与父子关系仍是采集声明。本加载器不采集 OS 样本。",
        };
        format!(
            "{}\n{note}\n\nManifest SHA-256: `{}`\n",
            self.comparison.to_markdown(language),
            self.manifest_sha256
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactError(pub String);
impl std::fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ArtifactError {}

/// Resource limits, not workload eligibility or performance thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ArtifactLoadLimits {
    pub max_manifest_bytes: u64,
    pub max_file_bytes: u64,
    /// Original bytes of the outer manifest and all distinct referenced files,
    /// shared across the main experiment and any pilot evidence.
    pub max_total_bytes: u64,
    /// Distinct referenced file paths, shared across main and pilot evidence.
    /// The outer manifest is separate; it still counts toward max_total_bytes.
    pub max_files: usize,
    pub max_jsonl_records: usize,
    pub max_cells: usize,
    pub max_pairs_per_cell: usize,
    pub max_requests_per_repeat: usize,
    pub max_total_visible_gaps: usize,
    pub max_memory_samples: usize,
}

impl Default for ArtifactLoadLimits {
    fn default() -> Self {
        Self {
            max_manifest_bytes: 4 << 20,
            max_file_bytes: 64 << 20,
            max_total_bytes: 256 << 20,
            max_files: 256,
            max_jsonl_records: 256,
            max_cells: 64,
            max_pairs_per_cell: 64,
            max_requests_per_repeat: 65_536,
            max_total_visible_gaps: 4_000_000,
            max_memory_samples: 200_000,
        }
    }
}
