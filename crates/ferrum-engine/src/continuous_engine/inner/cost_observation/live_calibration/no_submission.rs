//! Private original call settlement for the explicitly declared V2 population.
use super::*;
use ferrum_interfaces::execution_cost::CallNoSubmissionV1;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredPhaseV2;
use ferrum_scheduler::implementations::continuous::cost_profile::{
    CostProfileError, ProfileFingerprint, StructuredServiceNoSubmissionV6,
};

#[derive(Debug)]
pub(in crate::continuous_engine::inner::cost_observation) struct NoSubmissionReceipt {
    pub(super) wire: StructuredServiceNoSubmissionV6,
    // Keep the original worker reservation until the last retained owner drops.
    _memory: Arc<super::super::memory::ObservationBytePermit>,
}
impl NoSubmissionReceipt {
    pub(in crate::continuous_engine::inner::cost_observation) fn maximum_retained_bytes(
        rows: usize,
    ) -> Option<usize> {
        std::mem::size_of::<Self>()
            .checked_add(rows.checked_mul(std::mem::size_of::<
                ferrum_interfaces::execution_cost::CallNoSubmissionParticipantV1,
            >())?)?
            .checked_add("ferrum.inference-guard-rollback.v1".len())?
            .checked_add(2 * std::mem::size_of::<usize>())
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn retained_bytes(
        &self,
    ) -> Option<usize> {
        self.wire
            .retained_payload_bytes()?
            .checked_add(
                std::mem::size_of::<Self>().checked_sub(std::mem::size_of_val(&self.wire))?,
            )?
            .checked_add(2 * std::mem::size_of::<usize>())
    }
}

/// Original worker-validated non-submission. Neither an exported DTO nor a
/// calibration label can create this private provenance. The source adapter
/// binds its predeclared offer only after comparing the original participants.
#[derive(Debug)]
pub(in crate::continuous_engine::inner) struct OriginalNoSubmissionReceipt {
    proof: CallNoSubmissionV1,
    fingerprint: ProfileFingerprint,
    participants: Box<[CostObservationParticipant]>,
    fifo: u64,
    call_returned_at_ns: u64,
    finalized_at_ns: u64,
    _memory: Arc<super::super::memory::ObservationBytePermit>,
}
impl OriginalNoSubmissionReceipt {
    pub(in crate::continuous_engine::inner::cost_observation) fn feedback_observed_at(
        &self,
        fifo: u64,
        fingerprint: &ExecutionFingerprint,
    ) -> Option<u64> {
        (self.fifo == fifo && self.fingerprint == ProfileFingerprint::from(fingerprint))
            .then_some(self.finalized_at_ns)
    }
    pub(in crate::continuous_engine::inner) fn call_id(&self) -> u64 {
        self.proof.call_id()
    }
    pub(in crate::continuous_engine::inner) fn fifo(&self) -> u64 {
        self.fifo
    }
    pub(in crate::continuous_engine::inner) fn issued_at_ns(&self) -> u64 {
        self.proof.prepare_started_at_ns()
    }
    pub(in crate::continuous_engine::inner) fn observed_at_ns(&self) -> u64 {
        self.finalized_at_ns
    }
    pub(in crate::continuous_engine::inner) fn participants(
        &self,
    ) -> &[CostObservationParticipant] {
        &self.participants
    }
    pub(in crate::continuous_engine::inner) fn bind_source_position(
        &self,
        ticket: u64,
        phase: StructuredPhaseV2,
    ) -> Result<StructuredServiceNoSubmissionV6, CostProfileError> {
        StructuredServiceNoSubmissionV6::from_original(
            ticket,
            phase,
            self.fifo,
            self.fingerprint.clone(),
            &self.proof,
            &self.participants,
            self.call_returned_at_ns,
            self.finalized_at_ns,
        )
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn maximum_retained_bytes(
        rows: usize,
    ) -> Option<usize> {
        std::mem::size_of::<Self>()
            .checked_add(rows.checked_mul(std::mem::size_of::<CostObservationParticipant>())?)?
            .checked_add(2 * std::mem::size_of::<usize>())
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn retained_bytes(
        &self,
    ) -> Option<usize> {
        Self::maximum_retained_bytes(self.participants.len())
    }
}

impl EngineCostCall {
    pub(in crate::continuous_engine::inner::cost_observation) fn feedback_allows_no_submission(
        &self,
    ) -> bool {
        self.source_generation != 0
            && self
                .feedback_population
                .is_some_and(|p| p.allows_no_submission())
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn original_no_submission_receipt(
        &self,
        fifo: u64,
        memory: Arc<super::super::memory::ObservationBytePermit>,
    ) -> Option<Arc<OriginalNoSubmissionReceipt>> {
        if self.calibration_capture.is_none()
            && !self.feedback_allows_no_submission()
            && !self
                .live_ticket
                .as_ref()
                .is_some_and(|ticket| ticket.route_population().allows_no_submission())
        {
            return None;
        }
        let proof = self.recorder.no_submission()?;
        // Preparation is intentionally excluded from numerical training, but
        // that label does not invalidate a recorder-minted proof that this
        // privately authorized source8 attempt never encoded or submitted.
        // Existing live-ticket policy and every real rejection stay unchanged.
        let private_preparation = self.live_ticket.is_none()
            && self
                .calibration_capture
                .as_ref()
                .is_some_and(|c| c.requests_original_route())
            && self.rejection == Some(CostCallRejection::CalibrationPreparation);
        if fifo == 0
            || self.participants.is_empty()
            || self.participants.len() > 128
            || proof.call_id() != self.call_id.get()
            || Some(proof.prepare_started_at_ns()) != self.prepare_started_at_ns
            || self
                .dispatch
                .returned_at_ns
                .is_none_or(|at| at < proof.returned_at_ns())
            || !proof.matches_participants(&self.participants)
            || !matches!(
                self.dispatch.outcome,
                Some(ObservedCallOutcome::Deferred | ObservedCallOutcome::NotSubmitted)
            )
            || self.dispatch.waves != 0
            || self.dispatch.unknown.is_some()
            || (self.rejection.is_some() && !private_preparation)
            || self.stage_rejection.is_some()
            || self.boundary != WaveObservationBoundary::IsolatedPreparationToCommit
            || self.host.iter().any(Option::is_some)
            || self.host_stages.iter().any(|row| !row.is_pristine())
            || self
                .observation_time()
                .zip(self.dispatch.returned_at_ns)
                .is_none_or(|(now, returned)| now < returned)
        {
            return None;
        }
        let ExecutorCostIdentityAvailability::Known(identity) = &self.identity else {
            return None;
        };
        if identity.schema_version != EXECUTOR_COST_IDENTITY_SCHEMA {
            return None;
        }
        let fingerprint = ExecutionFingerprint {
            model_weights: identity.model_weights,
            numerical_policy: identity.numerical_policy,
            device_runtime: identity.device_runtime,
            execution_config: identity.execution_config,
        };
        Some(Arc::new(OriginalNoSubmissionReceipt {
            proof: *proof,
            fingerprint: ProfileFingerprint::from(&fingerprint),
            participants: self.participants.clone().into_boxed_slice(),
            fifo,
            call_returned_at_ns: self.dispatch.returned_at_ns?,
            finalized_at_ns: self.observation_time()?,
            _memory: memory,
        }))
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn no_submission_receipt(
        &self,
        original: &OriginalNoSubmissionReceipt,
        memory: Arc<super::super::memory::ObservationBytePermit>,
    ) -> Option<Arc<NoSubmissionReceipt>> {
        let ticket = self.live_ticket.as_ref()?;
        if !ticket.route_population().allows_no_submission()
            || !ticket.matches(
                original.call_id(),
                original.fifo(),
                Some(original.issued_at_ns()),
            )
        {
            return None;
        }
        let wire = original
            .bind_source_position(ticket.ordinal(), publication::phase(ticket.phase().min(2)))
            .ok()?;
        Some(Arc::new(NoSubmissionReceipt {
            wire,
            _memory: memory,
        }))
    }
}

impl LiveCalibration {
    pub(in crate::continuous_engine::inner::cost_observation) fn consume_no_submission(
        &self,
        ticket: Ticket,
        fifo: u64,
    ) -> LiveConsumption {
        let result = (|| {
            let receipt = ticket
                .no_submission()
                .ok_or("missing_private_no_submission")?;
            let wire = &receipt.wire;
            if !ticket.route_population().allows_no_submission()
                || !ticket.matches(wire.call_id(), fifo, Some(wire.issued_at_ns))
                || wire.ticket != ticket.ordinal()
                || wire.fifo != fifo
                || wire.phase != publication::phase(ticket.phase().min(2))
            {
                return Err("no_submission_ticket_policy_call_fifo_or_clock");
            }
            let observed = wire
                .validate_settlement(
                    &ProfileFingerprint::from(&self.fingerprint),
                    wire.issued_at_ns,
                )
                .map_err(|_| "invalid_original_no_submission")?;
            self.automatic
                .as_ref()
                .ok_or("no_submission_automatic_population_missing")?
                .observe_outside(&ticket, fifo)?;
            let bytes = receipt
                .retained_bytes()
                .ok_or("retained_capacity_overflow")?;
            Ok((Arc::clone(receipt), observed, bytes))
        })();
        let mut collected = self.collected.lock();
        match result.and_then(|(receipt, observed, bytes)| {
            let total = collected
                .retained_bytes
                .checked_add(bytes)
                .filter(|n| *n <= self.maximum_retained_bytes)
                .ok_or("retained_capacity_exceeded")?;
            // No work/frontier transition is claimed. The original offer still
            // occupies exactly one phase position and remains in raw replay.
            collected.waves.push(Wave {
                ticket: ticket.ordinal(),
                fifo,
                evidence: WaveEvidence::NoSubmission(receipt),
            });
            collected.retained_bytes = total;
            Ok(observed)
        }) {
            Ok(observed_at_ns) => {
                ticket.complete_no_submission();
                LiveConsumption::NoSubmission { observed_at_ns }
            }
            Err(reason) => {
                collected.failure.get_or_insert(reason);
                drop(ticket);
                LiveConsumption::Failed
            }
        }
    }
}
