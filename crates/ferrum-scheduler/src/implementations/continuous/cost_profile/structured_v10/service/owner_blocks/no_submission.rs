//! A declared attempt whose original producer proved no inference submission.
//! This replay DTO cannot manufacture the private producer capability.
use super::*;
use ferrum_interfaces::execution_cost::{
    no_submission_participant_signature, CallNoSubmissionParticipantV1, CallNoSubmissionReasonV1,
    CallNoSubmissionV1, CostObservationParticipant,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredServiceNoSubmissionV7 {
    pub ticket: u64,
    pub issued_at_ns: u64,
    pub fifo: u64,
    protocol: String,
    fingerprint: ProfileFingerprint,
    call_id: u64,
    prepare_started_at_ns: u64,
    returned_at_ns: u64,
    call_returned_at_ns: u64,
    finalized_at_ns: u64,
    participant_signature: [u8; 32],
    #[serde(deserialize_with = "bounded_rows")]
    participants: Vec<CallNoSubmissionParticipantV1>,
    reason: CallNoSubmissionReasonV1,
}

impl StructuredServiceNoSubmissionV7 {
    pub(super) fn original_participants(&self) -> &[CallNoSubmissionParticipantV1] {
        &self.participants
    }
    pub(super) fn finalized_at_ns(&self) -> u64 {
        self.finalized_at_ns
    }
    /// Reposition the original replay DTO after private engine settlement has
    /// already authenticated the live receipt. This grants no live capability.
    pub fn from_original_record(
        original: &StructuredServiceNoSubmissionV6,
        ticket: u64,
    ) -> Result<Self, CostProfileError> {
        original.validate_settlement(&original.fingerprint, original.issued_at_ns)?;
        let out = Self {
            ticket,
            issued_at_ns: original.issued_at_ns,
            fifo: original.fifo,
            protocol: original.protocol.clone(),
            fingerprint: original.fingerprint.clone(),
            call_id: original.call_id,
            prepare_started_at_ns: original.prepare_started_at_ns,
            returned_at_ns: original.returned_at_ns,
            call_returned_at_ns: original.call_returned_at_ns,
            finalized_at_ns: original.finalized_at_ns,
            participant_signature: original.participant_signature,
            participants: original.participants.clone(),
            reason: original.reason,
        };
        out.validate_settlement(&out.fingerprint, out.issued_at_ns)?;
        Ok(out)
    }

    pub fn from_original(
        ticket: u64,
        fifo: u64,
        fingerprint: ProfileFingerprint,
        proof: &CallNoSubmissionV1,
        participants: &[CostObservationParticipant],
        call_returned_at_ns: u64,
        finalized_at_ns: u64,
    ) -> Result<Self, CostProfileError> {
        if !proof.matches_participants(participants) {
            return Err(invalid("source7 no-submission participant binding differs"));
        }
        let value = Self {
            ticket,
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

    pub fn with_source_position(mut self, ticket: u64) -> Self {
        self.ticket = ticket;
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
            return Err(invalid("source7 no-submission original call proof differs"));
        }
        Ok(self.finalized_at_ns)
    }
}
