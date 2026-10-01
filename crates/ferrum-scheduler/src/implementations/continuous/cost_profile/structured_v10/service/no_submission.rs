//! A declared attempt whose original producer proved no inference submission.
//! This replay DTO cannot manufacture the private producer capability.
use super::*;
use ferrum_interfaces::execution_cost::{
    no_submission_participant_signature, CallNoSubmissionParticipantV1, CallNoSubmissionReasonV1,
    CallNoSubmissionV1, CostObservationParticipant,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredServiceNoSubmissionV6 {
    pub ticket: u64,
    pub phase: StructuredPhaseV2,
    pub issued_at_ns: u64,
    pub fifo: u64,
    pub(super) protocol: String,
    pub(super) fingerprint: ProfileFingerprint,
    pub(super) call_id: u64,
    pub(super) prepare_started_at_ns: u64,
    pub(super) returned_at_ns: u64,
    pub(super) call_returned_at_ns: u64,
    pub(super) finalized_at_ns: u64,
    pub(super) participant_signature: [u8; 32],
    #[serde(deserialize_with = "bounded_rows")]
    pub(super) participants: Vec<CallNoSubmissionParticipantV1>,
    pub(super) reason: CallNoSubmissionReasonV1,
}

impl StructuredServiceNoSubmissionV6 {
    pub fn from_original(
        ticket: u64,
        phase: StructuredPhaseV2,
        fifo: u64,
        fingerprint: ProfileFingerprint,
        proof: &CallNoSubmissionV1,
        participants: &[CostObservationParticipant],
        call_returned_at_ns: u64,
        finalized_at_ns: u64,
    ) -> Result<Self, CostProfileError> {
        if !proof.matches_participants(participants) {
            return Err(invalid("source6 no-submission participant binding differs"));
        }
        let value = Self {
            ticket,
            phase,
            issued_at_ns: proof.prepare_started_at_ns(),
            fifo,
            protocol: proof.reason().protocol().into(),
            fingerprint,
            call_id: proof.call_id(),
            prepare_started_at_ns: proof.prepare_started_at_ns(),
            returned_at_ns: proof.returned_at_ns(),
            call_returned_at_ns,
            finalized_at_ns,
            participant_signature: *proof.participant_signature(),
            participants: participants.iter().map(Into::into).collect(),
            reason: proof.reason(),
        };
        value.validate_settlement(&value.fingerprint, value.issued_at_ns)?;
        Ok(value)
    }

    /// Bind a phase-local private attempt to its cumulative source ordinal.
    /// Live identity is checked before this replay-only representation exists.
    pub fn with_source_position(mut self, ticket: u64, phase: StructuredPhaseV2) -> Self {
        self.ticket = ticket;
        self.phase = phase;
        self
    }

    pub fn call_id(&self) -> u64 {
        self.call_id
    }
    pub fn returned_at_ns(&self) -> u64 {
        self.returned_at_ns
    }

    pub fn retained_payload_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>()
            .checked_add(self.protocol.capacity())?
            .checked_add(
                self.participants
                    .capacity()
                    .checked_mul(std::mem::size_of::<CallNoSubmissionParticipantV1>())?,
            )
    }

    /// No host/device timing observation, owner transition, or numerical member
    /// is synthesized by this validation. Its only timing is call closure.
    pub fn validate_settlement(
        &self,
        fingerprint: &ProfileFingerprint,
        opened_at_ns: u64,
    ) -> Result<u64, CostProfileError> {
        let reason_valid = self.reason.valid();
        if self.protocol != self.reason.protocol()
            || &self.fingerprint != fingerprint
            || self.ticket == 0
            || self.fifo == 0
            || self.call_id == 0
            || self.issued_at_ns != self.prepare_started_at_ns
            || self.issued_at_ns < opened_at_ns
            || self.returned_at_ns < self.issued_at_ns
            || self.call_returned_at_ns < self.returned_at_ns
            || self.finalized_at_ns < self.call_returned_at_ns
            || self.participants.is_empty()
            || self.participants.len() > 128
            || no_submission_participant_signature(&self.participants).as_ref()
                != Some(&self.participant_signature)
            || !reason_valid
        {
            return Err(invalid("source6 no-submission original call proof differs"));
        }
        Ok(self.finalized_at_ns)
    }
}
