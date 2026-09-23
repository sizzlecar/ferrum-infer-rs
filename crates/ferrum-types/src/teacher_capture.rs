//! Backend-independent wire records for real-history numerical diagnostics.
//! Fingerprints below identify actual completion/readback receipts; they are
//! not replacement checksums for the raw full-vocabulary logit artifacts.
use serde::{Deserialize, Serialize};

pub const REAL_HISTORY_TEACHER_CAPTURE_SCHEMA: u32 = 1;
pub const REAL_HISTORY_TEACHER_CAPTURE_TYPE: &str = "ferrum.real_history_teacher_capture";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VNextTeacherOwnerRecord {
    pub owner_id: String,
    pub prompt_token_ids: Vec<u32>,
    pub teacher_token_ids: Vec<u32>,
    pub prompt_token_ids_sha256: String,
    pub teacher_token_ids_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VNextTeacherFileIdentity {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VNextTeacherCaptureIdentity {
    pub model_id: String,
    /// Product source evidence includes the leased semantic/tokenizer/weight
    /// artifacts; preserved verbatim from the shared product source resolver.
    pub model_source: serde_json::Value,
    pub numerical_profile: String,
    pub kv_storage: String,
    pub family_fingerprint: String,
    pub program_fingerprint: String,
    pub resolved_plan_fingerprint: String,
    pub binary: VNextTeacherFileIdentity,
    pub history_file: VNextTeacherFileIdentity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VNextTeacherCaptureManifest {
    pub schema_version: u32,
    pub artifact_type: String,
    /// "serial" means one real decode wave per owner; "batched" means all
    /// declared owners participate in each real decode wave.
    pub mode: String,
    pub output_policy: String,
    pub identity: Option<VNextTeacherCaptureIdentity>,
    pub vocabulary_size: usize,
    pub configuration: serde_json::Value,
    pub owners: Vec<VNextTeacherOwnerRecord>,
    pub waves: Vec<VNextTeacherWaveEvidence>,
    pub decisions: Vec<VNextTeacherDecisionRecord>,
    pub complete: bool,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VNextTeacherWaveParticipant {
    pub owner_id: String,
    pub request_id: String,
    pub participant_index: usize,
    pub cache_id: String,
    pub history_tokens: usize,
    pub history_sha256: String,
    pub immediate_start: usize,
    pub immediate_end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VNextTeacherWaveEvidence {
    pub wave_index: usize,
    pub kind: String,
    pub participant_count: usize,
    pub completion_fingerprint: String,
    pub receipt_fingerprint: String,
    pub completion_receipt: Option<VNextTeacherRawArtifact>,
    pub readbacks: Vec<VNextTeacherReadbackEvidence>,
    pub participants: Vec<VNextTeacherWaveParticipant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VNextTeacherReadbackEvidence {
    pub participant_index: usize,
    pub request: serde_json::Value,
    pub byte_count: usize,
    pub sha256: String,
    pub raw_artifact: Option<VNextTeacherRawArtifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VNextTeacherRawArtifact {
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VNextTeacherDecisionEvidence {
    pub owner_id: String,
    pub decision_index: usize,
    pub teacher_token_id: u32,
    pub history_tokens: usize,
    pub history_sha256: String,
    pub wave_index: usize,
    pub participant_index: usize,
    pub request_id: String,
    pub cache_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VNextTeacherLogitArtifact {
    pub file: String,
    pub encoding: String,
    pub elements: usize,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VNextTeacherDecisionRecord {
    #[serde(flatten)]
    pub evidence: VNextTeacherDecisionEvidence,
    pub logits: VNextTeacherLogitArtifact,
}
