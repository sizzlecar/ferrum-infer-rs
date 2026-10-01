//! Session identity is declared by the process-owning experiment controller.
//! Reuse authority additionally requires the original successful acquisition
//! in this search's chronological history; a receipt string alone is insufficient.
use super::types::*;
use crate::{BenchmarkPhase, QualityIssueCounts};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CapacityProcessBirth {
    LinuxBootTicks { boot_id: String, start_ticks: u64 },
    UnixStartTime { seconds: u64, microseconds: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityServerSession {
    pub session_id: String,
    pub process_id: u32,
    pub process_birth: CapacityProcessBirth,
    pub server_binary_sha256: String,
    pub effective_configuration_sha256: String,
    /// Original controller-observed argv/configuration bytes, held unchanged
    /// for this process. Distinct from a portable effective-config identity.
    pub process_configuration_sha256: String,
    pub endpoint: String,
    /// Required for the Ferrum admission protocol; absent for servers without it.
    pub engine_instance: Option<String>,
    pub started_unix_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacitySessionBlock {
    pub session: CapacityServerSession,
    pub key: CapacityRunKey,
    pub block_ordinal: u32,
    pub checked_live_unix_ns: u64,
    pub previous_evidence_sha256: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CapacityWarmupAcquisition {
    #[default]
    ExecutedInThisBlock,
    ReusedVerifiedSession {
        origin_evidence_sha256: String,
    },
}
impl CapacityWarmupAcquisition {
    pub fn is_executed(&self) -> bool {
        matches!(self, Self::ExecutedInThisBlock)
    }
}

/// Constructed only by validating the fixed contract and original history.
/// It cannot be deserialized into permission to skip warmup.
#[derive(Debug, Clone)]
pub struct AuthorizedCapacitySessionBlock {
    block: CapacitySessionBlock,
    acquisition: CapacityWarmupAcquisition,
}
impl AuthorizedCapacitySessionBlock {
    pub fn block(&self) -> &CapacitySessionBlock {
        &self.block
    }
    pub fn acquisition(&self) -> &CapacityWarmupAcquisition {
        &self.acquisition
    }
    pub fn executes_warmup(&self) -> bool {
        self.acquisition.is_executed()
    }
}

pub fn capacity_evidence_sha256(value: &CapacityRunEvidence) -> Result<String, CapacityError> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(value).map_err(|e| error(e.to_string()))?)
    ))
}
fn hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|v| v.is_ascii_hexdigit())
}
pub(super) fn successful_warmup(
    contract: &CapacityContract,
    value: &CapacityWarmupEvidence,
) -> bool {
    value.expected as usize
        == contract
            .workload
            .samples
            .iter()
            .filter(|s| s.phase == BenchmarkPhase::Warmup)
            .count()
        && value.completed == value.expected
        && value.errored == 0
        && value.quality.request_error_count() == 0
}

struct SessionState {
    identity: CapacityServerSession,
    ordinal: u32,
    last_digest: String,
    last_ended: u64,
    origin_digest: String,
    reusable: bool,
    confirmation: bool,
}
#[derive(Default)]
pub(super) struct SessionHistory {
    sessions: BTreeMap<String, SessionState>,
}
impl SessionHistory {
    pub(super) fn authorize(
        &self,
        contract: &CapacityContract,
        block: &CapacitySessionBlock,
        origin: Option<&str>,
    ) -> Result<AuthorizedCapacitySessionBlock, CapacityError> {
        let session = &block.session;
        let birth_valid = match &session.process_birth {
            CapacityProcessBirth::LinuxBootTicks {
                boot_id,
                start_ticks,
            } => !boot_id.is_empty() && boot_id.len() <= 128 && *start_ticks > 0,
            CapacityProcessBirth::UnixStartTime {
                seconds,
                microseconds,
            } => *seconds > 0 && *microseconds < 1_000_000,
        };
        if session.session_id.is_empty()
            || session.session_id.len() > 128
            || session.process_id == 0
            || !birth_valid
            || session
                .engine_instance
                .as_ref()
                .is_some_and(|v| v.is_empty() || v.len() > 128)
            || (contract.queue_observation_source == QueueObservationSource::ServerAdmissionV1
                && session.engine_instance.is_none())
            || session.endpoint.is_empty()
            || session.endpoint.len() > 2048
            || session.server_binary_sha256 != contract.identity.server.binary_sha256
            || session.effective_configuration_sha256
                != contract.identity.server.effective_configuration_sha256
            || !hash(&session.process_configuration_sha256)
            || session.started_unix_ns < contract.frozen_unix_ns
            || block.checked_live_unix_ns < session.started_unix_ns
            || block.block_ordinal == 0
        {
            return Err(error("invalid or changed capacity server session identity"));
        }
        let acquisition = match self.sessions.get(&session.session_id) {
            None => {
                if self.sessions.values().any(|prior| {
                    prior.identity.process_id == session.process_id
                        && prior.identity.process_birth == session.process_birth
                }) {
                    return Err(error(
                        "renaming a server session cannot establish a fresh process",
                    ));
                }
                if block.block_ordinal != 1
                    || block.previous_evidence_sha256.is_some()
                    || origin.is_some()
                {
                    return Err(error("new capacity session requires its own real warmup"));
                }
                CapacityWarmupAcquisition::ExecutedInThisBlock
            }
            Some(prior) => {
                if &prior.identity != session
                    || prior.ordinal.checked_add(1) != Some(block.block_ordinal)
                    || block.previous_evidence_sha256.as_deref() != Some(prior.last_digest.as_str())
                    || block.checked_live_unix_ns < prior.last_ended
                    || !prior.reusable
                    || prior.confirmation != (block.key.phase == CapacityPhase::Confirmation)
                    || origin != Some(prior.origin_digest.as_str())
                {
                    return Err(error("warmup reuse requires the same live session, successful origin and verified prior drain; confirmation requires a new session"));
                }
                CapacityWarmupAcquisition::ReusedVerifiedSession {
                    origin_evidence_sha256: prior.origin_digest.clone(),
                }
            }
        };
        Ok(AuthorizedCapacitySessionBlock {
            block: block.clone(),
            acquisition,
        })
    }
    pub(super) fn validate_run(
        &self,
        contract: &CapacityContract,
        evidence: &CapacityRunEvidence,
    ) -> Result<bool, CapacityError> {
        let Some(block) = &evidence.session else {
            return if evidence.warmup.acquisition.is_executed() {
                Ok(false)
            } else {
                Err(error("warmup reuse has no controller session"))
            };
        };
        if block.key != evidence.key || block.checked_live_unix_ns > evidence.run_started_unix_ns {
            return Err(error(
                "session checkpoint must precede this exact acquisition block",
            ));
        }
        if evidence
            .server_queue_attempts
            .iter()
            .filter_map(|a| a.observation.as_ref().ok())
            .any(|q| Some(&q.engine_instance) != block.session.engine_instance.as_ref())
        {
            return Err(error(
                "server admission epoch changed within the controller session",
            ));
        }
        let origin = match &evidence.warmup.acquisition {
            CapacityWarmupAcquisition::ExecutedInThisBlock => None,
            CapacityWarmupAcquisition::ReusedVerifiedSession {
                origin_evidence_sha256,
            } => Some(origin_evidence_sha256.as_str()),
        };
        let authorized = self.authorize(contract, block, origin)?;
        if authorized.acquisition != evidence.warmup.acquisition {
            return Err(error(
                "warmup acquisition differs from session authorization",
            ));
        }
        if origin.is_some()
            && (evidence.warmup.expected != 0
                || evidence.warmup.completed != 0
                || evidence.warmup.errored != 0
                || evidence.warmup.quality != QualityIssueCounts::default())
        {
            return Err(error(
                "reused warmup must not claim requests executed in this block",
            ));
        }
        Ok(origin.is_some())
    }
    pub(super) fn record(
        &mut self,
        contract: &CapacityContract,
        evidence: &CapacityRunEvidence,
        assessment: &CapacityRunAssessment,
    ) -> Result<(), CapacityError> {
        let Some(block) = &evidence.session else {
            return Ok(());
        };
        let digest = capacity_evidence_sha256(evidence)?;
        let origin = match &evidence.warmup.acquisition {
            CapacityWarmupAcquisition::ExecutedInThisBlock => digest.clone(),
            CapacityWarmupAcquisition::ReusedVerifiedSession {
                origin_evidence_sha256,
            } => origin_evidence_sha256.clone(),
        };
        self.sessions.insert(
            block.session.session_id.clone(),
            SessionState {
                identity: block.session.clone(),
                ordinal: block.block_ordinal,
                last_digest: digest,
                last_ended: evidence.run_ended_unix_ns,
                origin_digest: origin,
                reusable: assessment.acquisition_disposition
                    == CapacityAcquisitionDisposition::Complete
                    && (!evidence.warmup.acquisition.is_executed()
                        || successful_warmup(contract, &evidence.warmup)),
                confirmation: evidence.key.phase == CapacityPhase::Confirmation,
            },
        );
        Ok(())
    }
}
