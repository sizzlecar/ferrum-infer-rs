//! Replay DTOs are untrusted numerical input, never live receipt constructors.
use super::*;
use ferrum_interfaces::execution_cost::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredProfilePhaseV9 {
    Fit,
    Residual,
    Qualification,
}
impl StructuredProfilePhaseV9 {
    pub(super) fn index(self) -> usize {
        match self {
            Self::Fit => 0,
            Self::Residual => 1,
            Self::Qualification => 2,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredPhaseProvenanceV9 {
    pub phase: StructuredProfilePhaseV9,
    pub members: usize,
    pub member_cutoff: u64,
    pub accepted_fifo_cutoff: u64,
    pub frozen_at_ns: u64,
    pub source_prefix_bytes: u64,
    pub source_prefix_sha256: [u8; 32],
    pub parameters_sha256: [u8; 32],
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Envelope {
    pub schema_version: u32,
    pub model_revision: String,
    pub population_revision: String,
    pub fingerprint: ProfileFingerprint,
    pub source_path: PathBuf,
    pub source_bytes: u64,
    pub source_sha256: [u8; 32],
    pub source_clock_max_error_ns: u64,
    pub capture_identity: [u8; 32],
    pub protocol: [u8; 32],
    pub rule_signature: [u8; 32],
    pub parameters_sha256: [u8; 32],
    pub phases: [StructuredPhaseProvenanceV9; 3],
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PairedClock {
    pub wall_unix_ns: u64,
    pub monotonic_ns: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Settings {
    pub min_samples: usize,
    pub redundancy: usize,
    pub max_phase_samples: usize,
    pub max_axes: usize,
    pub max_rank: usize,
    pub max_wave_ns: u64,
    pub max_age_ns: u64,
    pub margin_ns: u64,
}
impl Settings {
    pub fn native(&self) -> StructuredSettingsV1 {
        StructuredSettingsV1 {
            min_phase_samples: self.min_samples,
            min_fit_redundancy: self.redundancy,
            max_phase_samples: self.max_phase_samples,
            max_axes: self.max_axes,
            max_rank: self.max_rank,
            max_wave_ns: self.max_wave_ns,
            max_sample_age_ns: self.max_age_ns,
            static_margin_ns: self.margin_ns,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Scope {
    pub rows: usize,
    pub domain: [u8; 32],
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Header {
    pub artifact_type: String,
    pub schema_version: u32,
    pub model_revision: String,
    pub population_revision: String,
    pub capture_identity: [u8; 32],
    pub protocol: [u8; 32],
    pub declared_protocol: [u8; 32],
    pub rule_signature: [u8; 32],
    pub fingerprint: ProfileFingerprint,
    pub producer: serde_json::Value,
    pub opening: PairedClock,
    pub opened_at_ns: u64,
    pub initial_fifo_cutoff: u64,
    pub scope: Scope,
    pub phase_members: [usize; 3],
    pub maximum_offered_waves: usize,
    pub maximum_file_bytes: u64,
    pub settings: Settings,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Freeze {
    pub capture_identity: [u8; 32],
    pub protocol: [u8; 32],
    pub rule_signature: [u8; 32],
    pub phase: StructuredProfilePhaseV9,
    pub accepted_fifo_cutoff: u64,
    pub member_cutoff: u64,
    pub source_prefix_bytes: u64,
    pub source_prefix_sha256: [u8; 32],
    pub frozen_at_ns: u64,
    pub parameters_sha256: [u8; 32],
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OfferedRow {
    pub request_id: String,
    pub owner: u64,
    pub generation: u64,
    pub generated: u64,
    pub decode: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Queue {
    pub accepted_ordinal: Option<u64>,
    pub disposition: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Numeric {
    pub fifo: u64,
    pub call_id: u64,
    pub observed_at_ns: u64,
    pub wall_ns: u64,
    pub domain: [u8; 32],
    #[serde(deserialize_with = "bounded_axes")]
    pub basis: Vec<f64>,
    #[serde(deserialize_with = "bounded_axes")]
    pub support: Vec<u64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Record {
    Offered {
        offered: u64,
        member_candidate: bool,
        phase: StructuredProfilePhaseV9,
        #[serde(deserialize_with = "bounded_rows")]
        rows: Vec<OfferedRow>,
    },
    Reserved {
        offered: u64,
        member: Option<u64>,
        phase: StructuredProfilePhaseV9,
        boundary: String,
    },
    PreparationUnavailable {
        offered: u64,
        member_candidate: bool,
        phase: StructuredProfilePhaseV9,
        reason: String,
    },
    Unsubmitted {
        offered: u64,
        member: Option<u64>,
        phase: StructuredProfilePhaseV9,
        reason: String,
    },
    Completed {
        offered: u64,
        member: Option<u64>,
        phase: StructuredProfilePhaseV9,
        queue: Option<Queue>,
        reconciled: bool,
        host_stages: Option<Stages>,
        selected_structured_capture: Option<std::result::Result<Recipe, serde_json::Value>>,
        #[serde(default)]
        selected_independent_attention_v2: Option<IndependentAttentionWaveEvidenceWireV2>,
        numeric: Option<Numeric>,
        conversion_error: Option<String>,
    },
    PhaseFreeze {
        receipt: Freeze,
    },
    Footer {
        phase: String,
        failure: Option<String>,
        offered: u64,
        members: u64,
        failed_members: u64,
        accepted_fifo_cutoff: u64,
        last_captured_fifo: u64,
        fifo_audit_complete: bool,
        closing: Option<PairedClock>,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Shape {
    pub exact: ProfileWaveShape,
    pub numeric_features: Option<CanonicalWaveCostFeatures>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_content_features: Option<HostContentCostFeaturesV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row_multiset_features: Option<HostRowMultisetCostFeaturesV2>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum RowWork {
    Decode {
        kv_tokens: u32,
    },
    Prefill {
        offset: u32,
        count: u32,
        total_prompt_tokens: u32,
    },
    Restore,
    Maintenance,
}
impl RowWork {
    pub fn actual(&self) -> ActualRowWork {
        match *self {
            Self::Decode { kv_tokens } => ActualRowWork::Decode { kv_tokens },
            Self::Prefill {
                offset,
                count,
                total_prompt_tokens,
            } => ActualRowWork::Prefill {
                offset,
                count,
                total_prompt_tokens,
            },
            Self::Restore => ActualRowWork::Restore,
            Self::Maintenance => ActualRowWork::Maintenance,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Terminal {
    pub finish_reason: ferrum_types::FinishReason,
    pub generated_tokens: u64,
    pub through_output_ordinal: u64,
    pub output_failed: bool,
    pub physical_failed: bool,
    pub scheduler_failed: bool,
    pub terminal_handoff_succeeded: bool,
    pub pending_restore_removed: bool,
    pub admission_cancellation_work: serde_json::Value,
    pub cache_completion_work: serde_json::Value,
    pub other_physical_resources: bool,
    pub request_slot_closed: bool,
    pub owner_matched: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StageRow {
    pub request_id: String,
    pub owner_incarnation: u64,
    pub work_generation: u64,
    pub input_index: u32,
    pub actual_work: RowWork,
    pub host_processing_ordinal: Option<u32>,
    pub host_started_at_ns: Option<u64>,
    pub token_committed_at_ns: Option<u64>,
    pub output_published_at_ns: Option<u64>,
    pub completion_started_at_ns: Option<u64>,
    pub settled_at_ns: Option<u64>,
    pub terminal: Option<Terminal>,
    pub completeness: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Stages {
    pub schema_version: u32,
    pub call_id: u64,
    #[serde(default)]
    pub presubmit_prediction: Option<serde_json::Value>,
    pub fingerprint: Option<ProfileFingerprint>,
    pub actual_shape: Option<Shape>,
    #[serde(default)]
    pub statistical_evidence: Option<StatisticalWaveEvidenceWireV1>,
    #[serde(default)]
    pub structured_evidence: Option<std::result::Result<Settlement, serde_json::Value>>,
    pub prepare_started_at_ns: Option<u64>,
    pub executor_returned_at_ns: Option<u64>,
    #[serde(deserialize_with = "bounded_rows")]
    pub rows: Vec<StageRow>,
    pub finalized_at_ns: Option<u64>,
    pub full_wall_ns: Option<u64>,
    pub completeness: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Settlement {
    pub protocol: String,
    pub call_id: u64,
    pub recipe: Recipe,
    pub stage_binding: [u8; 32],
    pub executor_envelope_ns: u64,
    pub host_settled_after_executor_ns: u64,
    pub full_wall_ns: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Recipe {
    pub protocol: String,
    pub exact_binding: [u8; 32],
    pub device: Device,
    #[serde(deserialize_with = "bounded_rows")]
    pub physical_host_rows: Vec<HostRow>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Device {
    pub ordered_template: [u8; 32],
    pub provider_grouped_template: Option<[u8; 32]>,
    pub physical_commands: u32,
    pub product: String,
    pub readback: CoreReadbackRoute,
    pub retries: u32,
    pub aggregate_work: Work,
    pub algorithm_work: std::result::Result<Algorithms, serde_json::Value>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Algorithms {
    pub protocol: String,
    pub exact_binding: [u8; 32],
    pub ordered_command_binding: [u8; 32],
    pub physical_commands: u32,
    pub selected_commands: u64,
    #[serde(deserialize_with = "bounded_algorithms")]
    pub entries: Vec<Algorithm>,
    pub aggregate_work: Work,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Algorithm {
    pub algorithm: [u8; 32],
    pub kind: Kind,
    pub commands: u64,
    pub work: Work,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Kind {
    Kernel,
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    Fill,
}
impl Kind {
    pub fn native(self) -> AlgorithmWorkKindV1 {
        match self {
            Self::Kernel => AlgorithmWorkKindV1::Kernel,
            Self::HostToDevice => AlgorithmWorkKindV1::HostToDevice,
            Self::DeviceToHost => AlgorithmWorkKindV1::DeviceToHost,
            Self::DeviceToDevice => AlgorithmWorkKindV1::DeviceToDevice,
            Self::Fill => AlgorithmWorkKindV1::Fill,
        }
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Work {
    pub logical_units: u64,
    pub padded_units: u64,
    pub inner_work_units: u64,
    pub grid_blocks: u64,
    pub peak_scratch_bytes: u64,
    pub staged_weight_bytes: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub device_to_device_bytes: u64,
    pub fill_bytes: u64,
}
impl Work {
    pub fn native(self) -> DeviceNumericWorkV1 {
        DeviceNumericWorkV1 {
            logical_units: self.logical_units,
            padded_units: self.padded_units,
            inner_work_units: self.inner_work_units,
            grid_blocks: self.grid_blocks,
            peak_scratch_bytes: self.peak_scratch_bytes,
            staged_weight_bytes: self.staged_weight_bytes,
            host_to_device_bytes: self.host_to_device_bytes,
            device_to_host_bytes: self.device_to_host_bytes,
            device_to_device_bytes: self.device_to_device_bytes,
            fill_bytes: self.fill_bytes,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct HostRow {
    pub physical_position: u32,
    pub role: HostRowRoleV2,
    pub installed_policy: HostCostPolicyV2,
    pub no_generated_history: bool,
    pub pending_decoded_utf8: bool,
    pub initial_prefill: bool,
    pub final_prefill: bool,
    pub mask_upload_required: bool,
    pub decode_requires_full_logits: Option<bool>,
    pub repetition_penalty_bits: Option<u32>,
    pub terminal_expectation: Expectation,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Expectation {
    NoTokenProduced,
    TokenMayTerminate,
    LengthBoundary,
}
impl HostRow {
    pub fn native(&self) -> StructuredHostRowV1 {
        StructuredHostRowV1 {
            physical_position: self.physical_position,
            role: self.role,
            installed_policy: self.installed_policy,
            no_generated_history: self.no_generated_history,
            pending_decoded_utf8: self.pending_decoded_utf8,
            initial_prefill: self.initial_prefill,
            final_prefill: self.final_prefill,
            mask_upload_required: self.mask_upload_required,
            decode_requires_full_logits: self.decode_requires_full_logits,
            repetition_penalty_bits: self.repetition_penalty_bits,
            terminal_expectation: match self.terminal_expectation {
                Expectation::NoTokenProduced => HostTerminalExpectationV1::NoTokenProduced,
                Expectation::TokenMayTerminate => HostTerminalExpectationV1::TokenMayTerminate,
                Expectation::LengthBoundary => HostTerminalExpectationV1::LengthBoundary,
            },
        }
    }
}

fn bounded_axes<'de, D: Deserializer<'de>, T: Deserialize<'de>>(d: D) -> Result<Vec<T>, D::Error> {
    bounded_vec::<D, T, 4096>(d)
}
fn bounded_algorithms<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    d: D,
) -> Result<Vec<T>, D::Error> {
    bounded_vec::<D, T, { ferrum_interfaces::execution_cost::MAX_COST_COMMANDS }>(d)
}
