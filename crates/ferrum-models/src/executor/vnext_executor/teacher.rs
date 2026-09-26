//! Real-history numerical capture through the production prefill/decode path.
//! The caller supplies token IDs, keeps every raw vocabulary logit, and feeds
//! the same canonical continuation to each arm. No synthetic state is restored.
use super::*;
pub use ferrum_types::teacher_capture::{
    VNextTeacherDecisionEvidence, VNextTeacherReadbackEvidence, VNextTeacherWaveEvidence,
    VNextTeacherWaveParticipant,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

mod admission;
mod capture;
mod execute;
#[cfg(test)]
mod tests;

pub const MAX_VNEXT_TEACHER_OWNERS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VNextTeacherMode {
    Serial,
    Batched,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VNextTeacherOwner {
    pub owner_id: String,
    pub prompt_token_ids: Vec<u32>,
    pub teacher_token_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VNextTeacherExecutionSpec {
    pub owners: Vec<VNextTeacherOwner>,
    pub mode: VNextTeacherMode,
    pub maximum_sequence_tokens: usize,
    pub prefill_chunk_tokens: usize,
}

impl VNextTeacherExecutionSpec {
    /// The model context ceiling is not the request's declared output budget.
    /// Match product admission: retain room for the complete supplied prompt
    /// and continuation, while keeping the global context/capacity unchanged.
    fn owner_sequence_ceiling(&self, owner: &VNextTeacherOwner) -> Result<usize> {
        owner
            .prompt_token_ids
            .len()
            .checked_add(owner.teacher_token_ids.len())
            .filter(|&tokens| tokens <= self.maximum_sequence_tokens)
            .ok_or_else(|| {
                FerrumError::request_validation(
                    "teacher token history exceeds the declared context",
                )
            })
    }

    pub fn validate(&self, vocabulary_size: usize, maximum_model_tokens: usize) -> Result<()> {
        if self.owners.is_empty() || self.owners.len() > MAX_VNEXT_TEACHER_OWNERS {
            return Err(FerrumError::request_validation(format!(
                "real-history teacher capture requires 1..={MAX_VNEXT_TEACHER_OWNERS} owners"
            )));
        }
        if self.prefill_chunk_tokens == 0
            || self.maximum_sequence_tokens == 0
            || self.maximum_sequence_tokens > maximum_model_tokens
        {
            return Err(FerrumError::request_validation(
                "teacher capture requires a positive prefill chunk and a sequence ceiling within the model",
            ));
        }
        let mut ids = BTreeSet::new();
        let decisions = self.owners[0].teacher_token_ids.len();
        for owner in &self.owners {
            if owner.owner_id.trim().is_empty() || !ids.insert(owner.owner_id.as_str()) {
                return Err(FerrumError::request_validation(
                    "teacher owner IDs must be nonempty and unique",
                ));
            }
            if owner.prompt_token_ids.is_empty()
                || owner.teacher_token_ids.len() < 2
                || owner.teacher_token_ids.len() > ferrum_types::MAX_VNEXT_TEACHER_FORCED_TOKENS
                || owner.teacher_token_ids.len() != decisions
            {
                return Err(FerrumError::request_validation(
                    "teacher owners require nonempty real prompts and equally long bounded continuations containing decode decisions",
                ));
            }
            self.owner_sequence_ceiling(owner)?;
            if owner
                .prompt_token_ids
                .iter()
                .chain(&owner.teacher_token_ids)
                .any(|&token| token as usize >= vocabulary_size)
            {
                return Err(FerrumError::request_validation(
                    "teacher token history exceeds the declared context or live vocabulary",
                ));
            }
        }
        Ok(())
    }
}

pub fn vnext_teacher_token_digest(tokens: &[u32]) -> String {
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

/// Synchronous persistence keeps raw logits bounded to one physical wave.
/// A sink error aborts the capture; it cannot turn missing rows into success.
pub trait VNextTeacherEvidenceSink: Send {
    fn wave(
        &mut self,
        evidence: &VNextTeacherWaveEvidence,
        raw_readbacks: &[Vec<u8>],
        completion_receipt: &serde_json::Value,
    ) -> Result<()>;
    fn decision(&mut self, evidence: &VNextTeacherDecisionEvidence, logits: &[f32]) -> Result<()>;
}

#[derive(Debug, Clone, Serialize)]
pub struct VNextTeacherCaptureSummary {
    pub mode: VNextTeacherMode,
    pub owner_count: usize,
    pub vocabulary_size: usize,
    pub decision_count: usize,
    pub physical_wave_count: usize,
    pub resolved_plan_fingerprint: String,
    pub family_fingerprint: String,
    pub program_fingerprint: String,
    pub output_policy: &'static str,
}

pub(super) struct TeacherExpectedParticipant {
    owner_id: String,
    request_id: RequestId,
    cache_id: Option<String>,
    history: Vec<u32>,
    immediate_start: usize,
}

pub(super) struct TeacherPendingWave {
    wave_index: usize,
    kind: VNextExecutionWaveKind,
    expected: Vec<TeacherExpectedParticipant>,
    observed: Option<TeacherObservedWave>,
}

pub(super) struct TeacherObservedWave {
    evidence: VNextTeacherWaveEvidence,
    raw_readbacks: Vec<Vec<u8>>,
    completion_receipt: serde_json::Value,
}

fn validate_logits(logits: &[f32], vocabulary: usize) -> Result<()> {
    if logits.len() != vocabulary || logits.iter().any(|value| !value.is_finite()) {
        return Err(FerrumError::model(
            "teacher capture requires one finite unmodified full-vocabulary logit row",
        ));
    }
    Ok(())
}

fn decision_evidence(
    owner: &VNextTeacherOwner,
    index: usize,
    history: &[u32],
    request_id: &RequestId,
    cache: &dyn KvCacheHandle,
    wave: &VNextTeacherWaveEvidence,
) -> Result<VNextTeacherDecisionEvidence> {
    let digest = validate_decision_history(owner, index, history, cache.num_tokens())?;
    let participant = wave
        .participants
        .iter()
        .find(|participant| participant.request_id == request_id.to_string())
        .ok_or_else(|| {
            FerrumError::internal("teacher decision has no completed physical participant")
        })?;
    if participant.owner_id != owner.owner_id
        || participant.cache_id != cache.cache_id()
        || participant.history_tokens != history.len()
        || participant.history_sha256 != digest
    {
        return Err(FerrumError::internal(
            "teacher decision differs from its completed physical wave identity",
        ));
    }
    Ok(VNextTeacherDecisionEvidence {
        owner_id: owner.owner_id.clone(),
        decision_index: index,
        teacher_token_id: owner.teacher_token_ids[index],
        history_tokens: history.len(),
        history_sha256: digest,
        wave_index: wave.wave_index,
        participant_index: participant.participant_index,
        request_id: participant.request_id.clone(),
        cache_id: participant.cache_id.clone(),
    })
}

fn validate_decision_history(
    owner: &VNextTeacherOwner,
    index: usize,
    history: &[u32],
    committed_tokens: usize,
) -> Result<String> {
    if index >= owner.teacher_token_ids.len() {
        return Err(FerrumError::internal(
            "teacher decision exceeds its canonical continuation",
        ));
    }
    let mut expected = owner.prompt_token_ids.clone();
    expected.extend_from_slice(&owner.teacher_token_ids[..index]);
    let digest = vnext_teacher_token_digest(history);
    if history != expected || committed_tokens != history.len() {
        return Err(FerrumError::internal(
            "teacher owner history or committed cache extent drifted before capture",
        ));
    }
    Ok(digest)
}
