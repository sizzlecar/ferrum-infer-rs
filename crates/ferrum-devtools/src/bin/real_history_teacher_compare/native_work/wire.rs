//! Read-only projections of teacher-native-work schema 1; never create core
//! execution authority from a decoded sidecar.
use super::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Index {
    pub schema_version: u32,
    pub artifact_type: String,
    pub complete: bool,
    pub identity: VNextTeacherCaptureIdentity,
    pub file_limit_bytes: u64,
    pub waves: Vec<IndexWave>,
    pub errors: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct IndexWave {
    pub wave_index: usize,
    pub completion_fingerprint: String,
    pub receipt_fingerprint: String,
    pub artifact: VNextTeacherRawArtifact,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Wave {
    pub schema_version: u32,
    pub artifact_type: String,
    pub wave_index: usize,
    pub kind: String,
    pub participant_count: usize,
    pub coverage: String,
    pub completion_fingerprint: String,
    pub receipt_fingerprint: String,
    pub submission_fingerprint: String,
    pub batch_identity_fingerprint: String,
    pub device: Device,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Device {
    pub commands: Vec<Command>,
    pub replayed_segments: Vec<Value>,
    pub graph_evidence: Option<Value>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Command {
    pub command_index: u32,
    pub node_index: Option<u32>,
    pub command_phase: String,
    pub native_op_id: String,
    pub execution_path: String,
    pub batching_form: String,
    pub participant_start: u32,
    pub participant_count: u32,
    pub token_count: u64,
    pub compute_dispatch_count: u64,
    pub transfer_command_count: u64,
    pub reusable_graph_node_count: Option<u64>,
    pub statistical_evidence: Option<Statistics>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Statistics {
    pub schema_version: u32,
    pub family_signature: [u8; 32],
    pub token_count: u64,
    pub compute_dispatches: u64,
    pub transfer_commands: u64,
    pub work: Work,
}
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
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
