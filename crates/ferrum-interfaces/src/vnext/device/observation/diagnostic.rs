//! Fixed-size passive failure context. It grants no evidence or execution authority.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum DeviceObservationFailureStage {
    Recipe,
    Reserve,
    Retain,
    Packet,
    Segment,
    Projection,
    Binding,
    Statistics,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct DeviceObservationBudgetFailure {
    /// Includes the ledger's reservation overhead; None means checked overflow.
    pub required: Option<usize>,
    /// Actual failed CAS value, not a later estimate of available capacity.
    pub current: usize,
    pub maximum: usize,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct DeviceObservationDiagnostic {
    pub stage: DeviceObservationFailureStage,
    pub site: &'static str,
    /// None preserves a producer's original Option::None, not an invented error.
    pub error: Option<StatisticalEvidenceUnknown>,
    pub budget: Option<DeviceObservationBudgetFailure>,
    pub command_index: Option<u32>,
    pub logical_command_ordinal: Option<u32>,
    pub native_operation: Option<DeviceNativeOperationId>,
}

impl DeviceObservationDiagnostic {
    pub const fn new(
        stage: DeviceObservationFailureStage,
        site: &'static str,
        error: Option<StatisticalEvidenceUnknown>,
    ) -> Self {
        Self {
            stage,
            site,
            error,
            budget: None,
            command_index: None,
            logical_command_ordinal: None,
            native_operation: None,
        }
    }

    pub fn bind_command(mut self, index: u32, operation: DeviceNativeOperationId) -> Self {
        self.command_index = Some(index);
        self.native_operation.get_or_insert(operation);
        self
    }
}
