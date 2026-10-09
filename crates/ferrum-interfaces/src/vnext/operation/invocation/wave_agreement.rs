//! Immutable participant agreement, borrowed for one dispatch attempt only.
//! This is never physical backing or permission to submit a wave.
use std::cell::Cell;

use super::*;
use crate::vnext::ExecutionPlan;

struct ParticipantAgreement<'plan, 'binding> {
    plan: &'plan ExecutionPlan,
    active: &'binding TrustedActiveSequenceBinding,
}

pub(in crate::vnext::operation) struct WaveInvocationAgreements<'plan, 'binding, R: DeviceRuntime> {
    batch: &'plan BatchOperationIdentity,
    wave: &'plan PreparedStepSubmissionWave<R>,
    participants: Vec<Option<ParticipantAgreement<'plan, 'binding>>>,
    pending: Vec<(usize, ParticipantAgreement<'plan, 'binding>)>,
    stats: super::super::InvocationPreparationStats,
    observation: Option<&'plan Cell<super::super::InvocationPreparationStats>>,
    current_batch_matches: bool,
}

impl<'plan, 'binding, R: DeviceRuntime> WaveInvocationAgreements<'plan, 'binding, R> {
    pub(in crate::vnext::operation) fn new(
        batch: &'plan BatchOperationIdentity,
        wave: &'plan PreparedStepSubmissionWave<R>,
    ) -> Self {
        let count = wave
            .nodes()
            .first()
            .map_or(0, |node| node.participant_count() as usize);
        Self {
            batch,
            wave,
            participants: (0..count).map(|_| None).collect(),
            pending: Vec::with_capacity(count),
            stats: Default::default(),
            observation: None,
            current_batch_matches: false,
        }
    }

    pub(in crate::vnext::operation) fn observe(
        &mut self,
        observation: Option<&'plan Cell<super::super::InvocationPreparationStats>>,
    ) {
        self.observation = observation;
    }

    pub(in crate::vnext::operation) fn snapshot(&self) -> super::super::InvocationPreparationStats {
        self.stats
    }

    fn has_origin(
        &self,
        resources: OperationInvocationResources<'plan, R>,
        identity: &ExecutionIdentityEnvelope,
        participant_index: usize,
    ) -> bool {
        self.current_batch_matches
            && matches!(resources, OperationInvocationResources::Wave { wave, node_index }
            if std::ptr::eq(wave, self.wave)
                && self.batch.has_compiled_participant_origin(identity, node_index, participant_index))
    }

    pub(super) fn matches(
        &mut self,
        resources: OperationInvocationResources<'plan, R>,
        identity: &ExecutionIdentityEnvelope,
        participant_index: usize,
        plan: &'plan ExecutionPlan,
        active: &'binding TrustedActiveSequenceBinding,
    ) -> bool {
        let Some(Some(agreement)) = self.participants.get(participant_index) else {
            return false;
        };
        if self.has_origin(resources, identity, participant_index)
            && std::ptr::eq(agreement.plan, plan)
            && std::ptr::eq(agreement.active, active)
        {
            self.stats.agreement_reuses += 1;
            true
        } else {
            self.stats.agreement_fallbacks += 1;
            false
        }
    }

    pub(super) fn stage(
        &mut self,
        resources: OperationInvocationResources<'plan, R>,
        identity: &ExecutionIdentityEnvelope,
        participant_index: usize,
        plan: &'plan ExecutionPlan,
        active: &'binding TrustedActiveSequenceBinding,
    ) {
        if self.has_origin(resources, identity, participant_index)
            && participant_index < self.participants.len()
        {
            self.pending
                .push((participant_index, ParticipantAgreement { plan, active }));
        }
    }

    pub(super) fn begin_node(&mut self, batch: &BatchOperationIdentity) {
        // Also discards a partially constructed node if a caller caught a panic.
        self.pending.clear();
        self.current_batch_matches = std::ptr::eq(batch, self.batch);
    }

    pub(super) fn finish_node(&mut self, success: bool) {
        if success {
            for (index, agreement) in self.pending.drain(..) {
                self.participants[index] = Some(agreement);
                self.stats.agreement_builds += 1;
            }
        } else {
            self.pending.clear();
        }
    }
}

impl<R: DeviceRuntime> Drop for WaveInvocationAgreements<'_, '_, R> {
    fn drop(&mut self) {
        if let Some(observation) = self.observation {
            observation.set(self.stats);
        }
    }
}
