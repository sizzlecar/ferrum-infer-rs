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
mod source8;
pub use plan::CalibrationPrefixTokensV1;
mod probe_tokens;
use plan::ValidatedCalibrationPrefixTokensV1;
pub(in crate::continuous_engine::inner::calibration) use probe_tokens::{
    discover_aligned_prefix_tokens, discover_prefix_tokens, CalibrationPrefixTokenBudgetV1,
    CalibrationPrefixTokenDiscoveryAuditV1, CalibrationPrefixTokenDiscoveryV1,
    CalibrationPrefixTokenUnavailableReasonV1, DiscoveredCalibrationPrefixTokensV1,
};

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
    pub(in crate::continuous_engine::inner::calibration) fn capture(
        sequence: &SequenceState,
    ) -> Result<Self> {
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
pub(in crate::continuous_engine::inner) struct CapturedPrefixWaveV5(
    PrefixWaveEvidenceV1,
    Option<Arc<super::super::cost_observation::CompletePrivateCalibrationSettlement>>,
);
impl CapturedPrefixWaveV5 {
    pub(in crate::continuous_engine::inner) fn evidence(&self) -> &PrefixWaveEvidenceV1 {
        &self.0
    }
    pub(in crate::continuous_engine::inner) fn private_settlement(
        &self,
    ) -> Option<&super::super::cost_observation::CompletePrivateCalibrationSettlement> {
        self.1.as_deref()
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

/// Source8 installation is granted only by its frozen private collector.
#[derive(Debug)]
enum PrefixPreparationAuthority {
    LengthV5,
    InstalledPlainTextV8 {
        // Captured after the intervention marker is installed. Recompute from
        // the actual sampler/masks/output owner before each forced commit.
        prepared_policy: Option<([u8; 32], HostCostPolicyV2)>,
    },
}

/// Constructor and fields stay inside the calibration preparation module.
/// SequenceState can consume it, but ordinary request metadata cannot make it.
#[derive(Debug)]
pub(in crate::continuous_engine) struct InstalledPrefix {
    plan: ValidatedCalibrationPrefixTokensV1,
    authority: PrefixPreparationAuthority,
    owner: u64,
    original_policy: [u8; 32],
    original_numeric: HostCostPolicyV2,
    captured_prepared_policy: Option<([u8; 32], HostCostPolicyV2)>,
    pending_commit: Option<PrefixTokenCommitV1>,
}

/// Minted under the same sequence lock as the call participant. No public
/// diagnostic or participant flag can construct a private preparation binding.
#[derive(Debug)]
pub(in crate::continuous_engine) struct BoundPrefixPreparationRow {
    participant: ferrum_interfaces::execution_cost::CostObservationParticipant,
}
impl BoundPrefixPreparationRow {
    pub(in crate::continuous_engine) fn matches(
        &self,
        participant: &ferrum_interfaces::execution_cost::CostObservationParticipant,
    ) -> bool {
        self.participant.request_id == participant.request_id
            && self.participant.owner_incarnation == participant.owner_incarnation
            && self.participant.work_generation == participant.work_generation
            && self.participant.input_index == participant.input_index
            && self.participant.output_policy_signature == participant.output_policy_signature
            && self.participant.host_features == participant.host_features
    }
}

impl InstalledPrefix {
    pub(in crate::continuous_engine) fn bind_cost_row(
        &self,
        sequence: &SequenceState,
        participant: &ferrum_interfaces::execution_cost::CostObservationParticipant,
    ) -> Option<BoundPrefixPreparationRow> {
        use ferrum_interfaces::execution_cost::{
            satisfied_completion_cost_signature, CostSamplingHistoryScope, HostContentDomainV1,
        };
        let frontier = sequence.cost_frontier?;
        let host = participant.host_features?;
        let prepared = self.captured_prepared_policy?;
        if frontier.owner_incarnation.get() != self.owner
            || participant.request_id != sequence.request_id
            || participant.owner_incarnation != self.owner
            || participant.work_generation != frontier.work_generation.get()
            || self.pending_commit.is_some()
            || sequence.generated_tokens.len() >= self.plan.declaration().release_generated
            || sequence.cost_policy_signature != Some(prepared.0)
            || sequence.cost_numeric_policy != Some(prepared.1)
            || host.policy != prepared.1
            || prepared.0 == self.original_policy
            || host.policy.empirical_content_domain.is_some()
            || !matches!(
                self.original_numeric.empirical_content_domain,
                Some(
                    HostContentDomainV1::PlainTextGreedyV1
                        | HostContentDomainV1::PlainTextInstalledV2(_)
                )
            )
            || host.state.generated_tokens_before != sequence.generated_tokens.len() as u64
            || host.state.maximum_output_tokens != sequence.sampling_params.max_tokens as u64
            || host.state.generated_tokens_before >= host.state.maximum_output_tokens
            || host.state.sampling_history_scope != CostSamplingHistoryScope::FullGeneration
            || host.state.sampling_history_tokens != host.state.generated_tokens_before
            || host.state.completion_state_signature != satisfied_completion_cost_signature()
        {
            return None;
        }
        Some(BoundPrefixPreparationRow {
            participant: participant.clone(),
        })
    }
    pub(in crate::continuous_engine) fn policy_marker(&self) -> &'static str {
        match self.authority {
            PrefixPreparationAuthority::LengthV5 => "calibration-prefix-preparation.v1",
            PrefixPreparationAuthority::InstalledPlainTextV8 { .. } => {
                "calibration-prefix-preparation.installed-v8"
            }
        }
    }
}

struct RequestRecord {
    owner: u64,
    maximum_output: usize,
    declaration: CalibrationPrefixTokensV1,
    released: Option<PrefixReleasedV1>,
    completed_length: bool,
    // A source8 request retains the original terminal capability; no output
    // policy is rewritten to force a convenient calibration completion.
    installed_policy: Option<HostCostPolicyV2>,
    last_call: u64,
    last_fifo: u64,
}
impl RequestRecord {
    fn accepts_terminal(&self, reason: ferrum_types::FinishReason, generated: u64) -> bool {
        if generated == 0 || generated > self.maximum_output as u64 {
            return false;
        }
        match reason {
            ferrum_types::FinishReason::Length => generated == self.maximum_output as u64,
            ferrum_types::FinishReason::EOS => {
                matches!(self.installed_policy.and_then(|p| p.empirical_content_domain),
                Some(ferrum_interfaces::execution_cost::HostContentDomainV1::PlainTextInstalledV2(cap)) if cap.model_eos)
            }
            ferrum_types::FinishReason::Stop => {
                matches!(self.installed_policy.and_then(|p| p.empirical_content_domain),
                Some(ferrum_interfaces::execution_cost::HostContentDomainV1::PlainTextInstalledV2(cap)) if cap.user_stop)
            }
            _ => false,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PrefixSource {
    Standalone,
    Source5,
    Source8,
}
#[derive(Default)]
pub(super) struct PrefixPreparationRun {
    records: HashMap<RequestId, RequestRecord>,
    pending_offer: Option<Vec<PrefixPreparedRowV5>>,
    pending_wave: Option<CapturedPrefixWaveV5>,
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
        self.prefix_preparation
            .as_mut()?
            .pending_wave
            .take()
            .map(|wave| wave.0)
    }
    pub fn prefix_preparation_failure(&self) -> Option<&str> {
        self.prefix_preparation.as_ref()?.failure.as_deref()
    }
}
