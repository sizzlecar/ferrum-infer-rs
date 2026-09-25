//! Read-only discovery from original live settlement evidence. This creates no
//! collector membership, source identity, observation ordinal or training data.
use super::*;
use crate::continuous_engine::inner::cost_observation::structured_discovery_input;
use ferrum_scheduler::implementations::continuous::cost_model::structured::{
    StructuredInputV1, StructuredUnknown,
};

impl CalibrationWaveReport {
    /// Inspect an actual completed wave for a later, independently declared
    /// structured calibration. The private settlement receipt must still bind
    /// all diagnostic fields; deserialized JSON cannot supply that receipt.
    /// A successful input is neither a qualified model nor a source member.
    pub fn structured_cost_input(
        &self,
    ) -> std::result::Result<StructuredInputV1, StructuredUnknown> {
        if self.submission != CalibrationSubmissionState::HostReconciled || self.error.is_some() {
            return Err(StructuredUnknown::InvalidSample);
        }
        structured_discovery_input(
            self.host_stages
                .as_ref()
                .ok_or(StructuredUnknown::MissingEvidence)?,
        )
    }
}
