//! Typed record faults at the real collector/archive boundary. No synthetic
//! publication, receipt, model or filesystem result is returned by these hooks.
use super::*;

impl LiveCalibration {
    pub(in crate::continuous_engine::inner::cost_observation) fn test_duplicate_phase_in_optional_source(
        &self,
    ) -> Result<(), FerrumError> {
        let mut state = self.automatic.as_ref().unwrap().state.lock();
        let opened_at_ns = state.opening_ns;
        let session = state
            .session
            .as_mut()
            .expect("actual open numerical session");
        let record = StructuredServiceRecordV6::PhaseOpen {
            phase: phase(session.phase),
            opened_at_ns,
            fifo_cutoff: 0,
        };
        // Mutate only the actual bounded StagedFile. Its eventual real receipt
        // must disagree with the unmodified collector's canonical stream.
        session
            .source
            .as_mut()
            .expect("actual open sidecar")
            .write(&record)
    }

    pub(in crate::continuous_engine::inner::cost_observation) fn test_duplicate_phase_in_collector(
        &self,
    ) -> Result<(), FerrumError> {
        let mut state = self.automatic.as_ref().unwrap().state.lock();
        let opened_at_ns = state.opening_ns;
        let session = state
            .session
            .as_mut()
            .expect("actual open numerical session");
        let record = StructuredServiceRecordV6::PhaseOpen {
            phase: phase(session.phase),
            opened_at_ns,
            fifo_cutoff: 0,
        };
        // The real push permanently poisons this candidate. Do not synthesize
        // an Err or manually change the ticket/generation failure state.
        session.record(self, &record)
    }
}
