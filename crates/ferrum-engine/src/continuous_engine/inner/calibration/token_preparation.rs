//! Manual-session preparation evidence. This is deliberately not a calibration
//! source wire, a qualified sample, or authority to predict any future wave.
use super::*;
use crate::continuous_engine::SequenceState;
use ferrum_interfaces::execution_cost::HostCostPolicyV2;
use ferrum_types::TokenId;
use std::collections::HashMap;

mod plan;
mod sequence;
mod session;
mod source5;
pub use plan::CalibrationPrefixTokensV1;
use plan::ValidatedCalibrationPrefixTokensV1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixCandidateRouteV1 {
    FullLogitsSampler,
    ModelGreedyArgmax,
}

/// Original candidate and actual committed token are distinct facts. These
/// diagnostic values cannot construct the private installed capability.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PrefixTokenCommitV1 {
    pub request_id: RequestId,
    pub owner_incarnation: u64,
    pub work_generation: u64,
    pub generated_before: usize,
    pub generated_after: usize,
    pub original_candidate: TokenId,
    pub committed_token: TokenId,
    pub route: PrefixCandidateRouteV1,
    pub pending_before: Vec<u8>,
    pub pending_after: Vec<u8>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PrefixFrontierV1 {
    pub request_id: RequestId,
    pub owner_incarnation: u64,
    pub work_generation: u64,
    pub generated_tokens: usize,
    pub kv_tokens: usize,
    pub model_cache_id: Option<String>,
    pub pending_utf8: Vec<u8>,
    pub output_accepted_ordinal: u64,
}
impl PrefixFrontierV1 {
    fn capture(sequence: &SequenceState) -> Result<Self> {
        let cost = sequence
            .cost_frontier
            .ok_or_else(|| invalid("prefix owner unavailable"))?;
        let output = sequence
            .credited_output
            .as_ref()
            .ok_or_else(|| invalid("prefix requires credited output"))?;
        Ok(Self {
            request_id: sequence.request_id.clone(),
            owner_incarnation: cost.owner_incarnation.get(),
            work_generation: cost.work_generation.get(),
            generated_tokens: sequence.generated_tokens.len(),
            kv_tokens: sequence
                .model_kv
                .as_ref()
                .map_or(0, |kv| kv.handle().num_tokens()),
            model_cache_id: sequence.model_cache_id().map(str::to_owned),
            pending_utf8: sequence.pending_decoded_utf8_bytes.clone(),
            output_accepted_ordinal: output.accepted_ordinal,
        })
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PrefixRowEvidenceV1 {
    pub before: PrefixFrontierV1,
    /// A terminal owner has been removed; use the real host terminal receipt.
    pub after: Option<PrefixFrontierV1>,
    pub preparation_commit: Option<PrefixTokenCommitV1>,
}

/// Only this module constructs these snapshots from the actual private
/// sequence and declared manual work. Public diagnostic DTOs are never input
/// authority for a source writer.
#[derive(Debug, Clone, serde::Serialize)]
pub(in crate::continuous_engine::inner) struct PrefixPreparedRowV5 {
    before: PrefixFrontierV1,
    work: ferrum_scheduler::implementations::continuous::cost_model::structured_v2::windows::PreparedWorkV2,
}
impl PrefixPreparedRowV5 {
    pub(in crate::continuous_engine::inner) fn before(&self) -> &PrefixFrontierV1 {
        &self.before
    }
    pub(in crate::continuous_engine::inner) fn work(&self) -> ferrum_scheduler::implementations::continuous::cost_model::structured_v2::windows::PreparedWorkV2{
        self.work
    }
}
pub(in crate::continuous_engine::inner) struct CapturedPrefixWaveV5(PrefixWaveEvidenceV1);
impl CapturedPrefixWaveV5 {
    pub(in crate::continuous_engine::inner) fn evidence(&self) -> &PrefixWaveEvidenceV1 {
        &self.0
    }
}
pub(in crate::continuous_engine::inner) struct CapturedPrefixReleaseV5(PrefixReleasedV1);
impl CapturedPrefixReleaseV5 {
    pub(in crate::continuous_engine::inner) fn receipt(&self) -> &PrefixReleasedV1 {
        &self.0
    }
}

#[derive(Debug, serde::Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum PrefixReleaseProgressV5 {
    Inactive,
    Preparing,
    AwaitingCredit,
    Released { receipts: Vec<PrefixReleasedV1> },
}

/// One actual wave, including ordinary suffix/terminal waves. The manual
/// driver requires this bounded slot to be taken before the next submission.
#[derive(Debug, Clone)]
pub struct PrefixWaveEvidenceV1 {
    pub rows: Vec<PrefixRowEvidenceV1>,
    pub submission: CalibrationSubmissionState,
    pub error: Option<String>,
    pub host_stages: Option<Arc<super::super::cost_observation::HostStageEvidenceV1>>,
    pub host_stage_queue: Option<super::super::cost_observation::HostStageQueueReceipt>,
    pub actual_evidence_diagnostic:
        Option<Arc<super::super::cost_observation::CalibrationActualEvidenceDiagnostic>>,
    pub chain_error: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PrefixReleasedV1 {
    pub frontier: PrefixFrontierV1,
    pub original_policy_signature: [u8; 32],
    pub original_numeric_policy: HostCostPolicyV2,
    pub generated_prefix_sha256: [u8; 32],
    pub through_call_id: u64,
    pub through_fifo_ordinal: u64,
    /// Actual actor application, not network delivery or client receipt.
    pub actor_applied_output_ordinal: u64,
}

/// Constructor and fields stay inside the calibration preparation module.
/// SequenceState can consume it, but ordinary request metadata cannot make it.
#[derive(Debug)]
pub(in crate::continuous_engine) struct InstalledPrefix {
    plan: ValidatedCalibrationPrefixTokensV1,
    owner: u64,
    original_policy: [u8; 32],
    original_numeric: HostCostPolicyV2,
    pending_commit: Option<PrefixTokenCommitV1>,
}

struct RequestRecord {
    owner: u64,
    maximum_output: usize,
    declaration: CalibrationPrefixTokensV1,
    released: Option<PrefixReleasedV1>,
    completed_length: bool,
    last_call: u64,
    last_fifo: u64,
}
#[derive(Default)]
pub(super) struct PrefixPreparationRun {
    records: HashMap<RequestId, RequestRecord>,
    pending_offer: Option<Vec<PrefixPreparedRowV5>>,
    pending_wave: Option<PrefixWaveEvidenceV1>,
    last_fifo: u64,
    last_call: u64,
    failure: Option<String>,
}

fn invalid(message: impl Into<String>) -> FerrumError {
    FerrumError::invalid_request(message.into())
}

impl CalibrationSession {
    /// Returns diagnostic data only. A future source5 writer must own the
    /// complete declared/settled/released/Length chain, not just this record.
    pub fn take_prefix_wave_evidence(&mut self) -> Option<PrefixWaveEvidenceV1> {
        self.prefix_preparation.as_mut()?.pending_wave.take()
    }
    pub fn prefix_preparation_failure(&self) -> Option<&str> {
        self.prefix_preparation.as_ref()?.failure.as_deref()
    }
}
