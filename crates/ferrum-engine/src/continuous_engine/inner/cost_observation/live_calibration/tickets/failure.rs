//! Fixed-size evidence for the first lost original ticket. This is diagnostic
//! only: it cannot retire, replace, classify, or qualify an offered sample.
use super::*;
use crate::continuous_engine::inner::cost_observation::{CostCallRejection, EngineCostCall};
use ferrum_interfaces::execution_cost::{
    ActualWaveEvidenceUnknown, CostObservationCoverage, ObservedCallOutcome,
};
use serde::{Serialize, Serializer};

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(in crate::continuous_engine::inner::cost_observation) enum TicketFailureCause {
    UnretiredTicket,
    CallRejected {
        reason: CostCallRejection,
        physical_waves: usize,
        retained_waves: usize,
        lost_observations: u64,
        participant_count: usize,
        #[serde(serialize_with = "serialize_optional_debug")]
        dispatch_outcome: Option<ObservedCallOutcome>,
        #[serde(serialize_with = "serialize_optional_debug")]
        dispatch_unknown: Option<ActualWaveEvidenceUnknown>,
        unknown_wave_count: usize,
        first_unknown_wave_ordinal: Option<u32>,
        #[serde(serialize_with = "serialize_optional_debug")]
        first_unknown_wave_reason: Option<ActualWaveEvidenceUnknown>,
    },
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct TicketFailureAudit {
    pub ticket: u64,
    pub issued_at_ns: u64,
    pub call_id: u64,
    pub accepted_fifo_ordinal: u64,
    pub cause: TicketFailureCause,
}

fn serialize_optional_debug<T: std::fmt::Debug, S: Serializer>(
    value: &Option<T>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    match value {
        Some(value) => serializer.serialize_some(&format!("{value:?}")),
        None => serializer.serialize_none(),
    }
}

impl Ticket {
    fn record_failure(&self, cause: TicketFailureCause) {
        let slot = &self.window.slots[self.index];
        let _ = self.window.first_ticket_failure.set(TicketFailureAudit {
            ticket: self.ordinal(),
            issued_at_ns: slot.issued_at_ns.load(Ordering::Relaxed),
            call_id: slot.call_id.load(Ordering::Relaxed),
            accepted_fifo_ordinal: slot.fifo.load(Ordering::Relaxed),
            cause,
        });
    }

    pub(super) fn record_unretired_failure(&self) {
        self.record_failure(TicketFailureCause::UnretiredTicket);
    }

    pub(in crate::continuous_engine::inner::cost_observation) fn record_call_failure(
        &self,
        call: &EngineCostCall,
        reason: CostCallRejection,
    ) {
        let observations = call.recorder.observations();
        let unknown_wave_count = observations
            .iter()
            .filter(|wave| wave.shape_unknown.is_some())
            .count();
        let first_unknown = observations
            .iter()
            .find(|wave| wave.shape_unknown.is_some());
        self.record_failure(TicketFailureCause::CallRejected {
            reason,
            physical_waves: call.dispatch.waves,
            retained_waves: observations.len(),
            lost_observations: match call.recorder.coverage() {
                CostObservationCoverage::Complete => 0,
                CostObservationCoverage::Unknown {
                    lost_observations, ..
                } => lost_observations,
            },
            participant_count: call.participants.len(),
            dispatch_outcome: call.dispatch.outcome,
            dispatch_unknown: call.dispatch.unknown,
            unknown_wave_count,
            first_unknown_wave_ordinal: first_unknown.map(|wave| wave.physical_wave_ordinal),
            first_unknown_wave_reason: first_unknown.and_then(|wave| wave.shape_unknown),
        });
    }
}
