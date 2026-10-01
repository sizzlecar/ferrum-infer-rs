use crate::{
    dataset::ShareGptSelection,
    env::HttpRequestSampling,
    slo::{RequestSloEvidence, SloEvaluationConfig, SloEvaluationReport, SloStatus},
    slo_comparison::{
        artifact::SidecarArrival, FixedServerCapacity, FrozenServerIdentity,
        SharedExecutionIdentity,
    },
    BenchmarkRequestRecord, QualityIssueCounts,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapacityError(pub String);

impl std::fmt::Display for CapacityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for CapacityError {}

pub(super) fn error(message: impl Into<String>) -> CapacityError {
    CapacityError(message.into())
}

/// One implementation/configuration per search. A/A uses this exact identity,
/// rather than comparing two separately labelled implementations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityIdentity {
    pub shared: SharedExecutionIdentity,
    pub server: FrozenServerIdentity,
    pub capacity: FixedServerCapacity,
    pub ordered_workload_sha256: String,
    pub dataset_source_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityRepetitions {
    pub aa_pairs: u32,
    pub coarse: u32,
    pub neighborhood: u32,
    pub confirmation: u32,
}

/// All tolerances are declared before observation; none are inferred from a
/// candidate's results. Queue growth is an empirical least-squares slope over
/// the declared observation window, not a proof of perpetual stability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityWindow {
    pub send_seconds: f64,
    pub observe_from_seconds: f64,
    pub maximum_drain_seconds: f64,
    pub maximum_request_start_lag_ms: f64,
    pub maximum_client_dispatch_backlog: u64,
    pub maximum_queue_sample_gap_seconds: f64,
    pub maximum_unfinished_requests_slope_per_second: f64,
    pub maximum_oldest_age_slope_ms_per_second: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueObservationSource {
    /// Due scheduled requests minus observed client completions. Includes
    /// undispatched work and transport time; does not identify server queuing.
    ClientScheduledLifecycle,
    ServerQueue,
    /// Original /health admission protocol, bracketed by client HTTP times.
    /// Server monotonic deltas drive slopes; no synchronized clock is assumed.
    ServerAdmissionV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerQueueFailure {
    Unavailable,
    Transport,
    Timeout,
    HttpStatus(u16),
    ResponseTooLarge,
    Malformed,
    Runtime(String),
}

/// HTTP intervals use the benchmark's monotonic measurement origin. A
/// negative interval is an actual pre-window baseline, not a clamped zero.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerQueueAttempt {
    pub request_started_seconds: f64,
    pub response_completed_seconds: f64,
    pub observation: Result<ferrum_types::ExecutorQueueObservation, ServerQueueFailure>,
}

/// Finite search resolution and acquisition resources are explicit. All
/// coarse indices are measured even after a failure. All declared fine indices
/// are subsequently visited, including intervals with equal endpoint results:
/// those endpoints cannot rule out an interior feasible or infeasible island.
/// Independent confirmation repeats every rate measured after A/A.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityContract {
    pub schema_version: u32,
    pub frozen_unix_ns: u64,
    pub identity: CapacityIdentity,
    pub rates_rps: Vec<f64>,
    pub coarse_indices: Vec<usize>,
    pub aa_indices: Vec<usize>,
    pub repetitions: CapacityRepetitions,
    pub window: CapacityWindow,
    pub queue_observation_source: QueueObservationSource,
    pub slo: SloEvaluationConfig,
    pub sampling: HttpRequestSampling,
    pub enable_thinking: Option<bool>,
    pub http_connection_mode: String,
    pub seed: u64,
    /// Bounds generated arrivals; exceeding it is an error, never truncation.
    pub maximum_requests_per_run: usize,
    pub maximum_planned_runs: usize,
    pub maximum_queue_samples_per_run: usize,
    /// Pinned ordered workload repeated cyclically for the duration of a run.
    /// This first API deliberately supports the primary exact-budget workload.
    /// A future natural-EOS adapter needs independent completion evidence.
    pub workload: ShareGptSelection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapacityPhase {
    Aa,
    Coarse,
    Neighborhood,
    Confirmation,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityRunKey {
    pub phase: CapacityPhase,
    pub rate_index: usize,
    pub repetition: u32,
    /// A/A pair member 0 or 1; zero for other phases.
    pub replica: u8,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PlannedCapacityRun {
    pub contract_sha256: String,
    pub key: CapacityRunKey,
    pub cell_id: String,
    pub rate_rps: f64,
    pub seed: u64,
    pub send_seconds: f64,
    pub scheduled_arrival_ms: Vec<f64>,
    pub output_token_budgets: Vec<u32>,
    pub workload_sample_indices: Vec<usize>,
}

/// Existing collector correlation and timing, aligned with the actual prompt
/// dispatched for this request. The workload row alone is not timing evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityRequestRecord {
    pub workload_sample_index: usize,
    pub dispatched_prompt_sha256: String,
    pub input_tokens: u32,
    pub server_input_tokens: Option<u32>,
    pub record: BenchmarkRequestRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityWarmupEvidence {
    #[serde(
        default,
        skip_serializing_if = "super::session::CapacityWarmupAcquisition::is_executed"
    )]
    pub acquisition: super::session::CapacityWarmupAcquisition,
    pub expected: u32,
    pub completed: u32,
    pub errored: u32,
    pub quality: QualityIssueCounts,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceQueueSample {
    /// Offset from the same measurement origin as the arrival evidence.
    pub at_seconds: f64,
    pub waiting_requests: u64,
    /// For client lifecycle observations, dispatched but not yet finished.
    pub active_requests: u64,
    /// With client lifecycle source, age since the oldest unfinished planned
    /// arrival. With server source, server-observed unfinished request age.
    pub oldest_request_age_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityRunEvidence {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<super::session::CapacitySessionBlock>,
    pub contract_sha256: String,
    pub key: CapacityRunKey,
    pub identity: CapacityIdentity,
    pub run_started_unix_ns: u64,
    pub run_ended_unix_ns: u64,
    /// Unique acquisition block, generated by the client. Server restarts and
    /// reset policy belong to the external experiment controller/receipts.
    pub independent_run_id: String,
    pub benchmark_run_id: String,
    pub warmup: CapacityWarmupEvidence,
    pub send_window_seconds: f64,
    pub measured_duration_seconds: f64,
    pub arrivals: Vec<SidecarArrival>,
    pub requests: Vec<RequestSloEvidence>,
    pub request_records: Vec<CapacityRequestRecord>,
    /// One protocol/output quality row for every offered request.
    pub quality: Vec<QualityIssueCounts>,
    pub queue_observation_source: QueueObservationSource,
    pub queue: Vec<ServiceQueueSample>,
    pub queue_capture_complete: bool,
    /// Retained even on unavailable/failed samples; never inferred from SSE.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub server_queue_attempts: Vec<ServerQueueAttempt>,
}

/// Reports are not accepted as inputs to CapacitySearch::record. It recomputes
/// them from original request, arrival and service queue observations.
#[derive(Debug, Clone, Serialize)]
pub struct CapacityRunAssessment {
    pub acquisition_disposition: CapacityAcquisitionDisposition,
    pub key: CapacityRunKey,
    pub status: SloStatus,
    pub evidence_complete: bool,
    pub workload_completed: bool,
    pub arrival_schedule_delivered: bool,
    pub issues: Vec<String>,
    pub evaluation: Option<SloEvaluationReport>,
    pub requested_rate_rps: f64,
    pub planned_requests: usize,
    pub delivered_requests: usize,
    pub realized_scheduled_rate_rps: f64,
    pub realized_delivered_rate_rps: f64,
    pub maximum_start_lag_ms: Option<f64>,
    pub maximum_client_backlog: Option<u64>,
    pub unfinished_requests_slope_per_second: Option<f64>,
    pub oldest_age_slope_ms_per_second: Option<f64>,
    /// Server-only diagnostics, distinct from all unfinished work.
    pub waiting_requests_slope_per_second: Option<f64>,
    pub oldest_waiting_ingress_age_slope_ms_per_second: Option<f64>,
    pub drain_seconds: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AaPairObservation {
    pub rate_index: usize,
    pub pair: u32,
    /// Signed member-1 minus member-0 measurements. No implicit noise gate.
    pub successful_output_tps_delta: Option<f64>,
    pub ttft_p99_ms_delta: Option<f64>,
    pub tpot_p99_ms_delta: Option<f64>,
    pub visible_itl_p99_ms_delta: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CapacitySearchReport {
    pub contract_sha256: String,
    pub complete: bool,
    pub runs: Vec<CapacityRunAssessment>,
    pub aa_noise: Vec<AaPairObservation>,
    pub measured_rate_indices: Vec<usize>,
    pub observed_coarse_transition_intervals: Vec<[usize; 2]>,
    pub unmeasured_rate_indices: Vec<usize>,
    pub independently_confirmed_rate_indices: Vec<usize>,
    /// Maximum actually tested and independently confirmed grid point only.
    /// None means no confirmed point, never zero service capacity.
    pub maximum_confirmed_tested_rate_rps: Option<f64>,
    pub scope: String,
}

/// Complete means the acquisition can safely precede another block. It says
/// nothing about performance: the independently recomputed status may be Fail
/// or Unknown. Other values preserve evidence but prohibit session reuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapacityAcquisitionDisposition {
    Complete,
    ProtocolFailure,
    ExecutionFailure,
    IncompleteEvidence,
    Undrained,
}
