//! Byte capacities are independent of physical row and wave capacities.
use super::*;

pub const MAXIMUM_COST_OBSERVATION_PAYLOAD_BYTES: usize = 128 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CostRecorderByteLimits {
    pub maximum_retained_bytes: usize,
    pub maximum_working_bytes: usize,
}
impl Default for CostRecorderByteLimits {
    fn default() -> Self {
        Self {
            maximum_retained_bytes: MAXIMUM_COST_OBSERVATION_PAYLOAD_BYTES,
            maximum_working_bytes: MAXIMUM_COST_OBSERVATION_PAYLOAD_BYTES,
        }
    }
}
impl CostRecorderByteLimits {
    pub fn validate(self) -> Result<(), CostRecorderError> {
        if self.maximum_retained_bytes == 0
            || self.maximum_working_bytes == 0
            || self.maximum_retained_bytes > MAXIMUM_COST_OBSERVATION_PAYLOAD_BYTES
            || self.maximum_working_bytes > MAXIMUM_COST_OBSERVATION_PAYLOAD_BYTES
        {
            Err(CostRecorderError::InvalidLimits)
        } else {
            Ok(())
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum CostRecorderCapacityResource {
    Rows,
    RawBytes,
    WorkingBytes,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct CostRecorderCapacityDiagnostic {
    pub resource: CostRecorderCapacityResource,
    /// None means checked arithmetic overflow, never a zero request.
    pub requested: Option<usize>,
    pub limit: usize,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct CostRecorderMemoryAudit {
    pub requested_rows_peak: usize,
    pub requested_raw_bytes_peak: usize,
    pub requested_working_bytes_peak: usize,
    pub first_rejection: Option<CostRecorderCapacityDiagnostic>,
}
impl CostRecorderMemoryAudit {
    pub(super) fn observe(
        &mut self,
        rows: Option<usize>,
        raw: Option<usize>,
        working: Option<usize>,
    ) {
        self.requested_rows_peak = self.requested_rows_peak.max(rows.unwrap_or(usize::MAX));
        self.requested_raw_bytes_peak =
            self.requested_raw_bytes_peak.max(raw.unwrap_or(usize::MAX));
        self.requested_working_bytes_peak = self
            .requested_working_bytes_peak
            .max(working.unwrap_or(usize::MAX));
    }
}
impl BoundedWaveRecorder {
    pub fn memory_audit(&self) -> CostRecorderMemoryAudit {
        self.memory_audit
    }
    pub(super) fn note_memory_rejection(
        &mut self,
        resource: CostRecorderCapacityResource,
        requested: Option<usize>,
        limit: usize,
    ) {
        self.memory_audit
            .first_rejection
            .get_or_insert(CostRecorderCapacityDiagnostic {
                resource,
                requested,
                limit,
            });
    }
    /// Simultaneously retained raw facts and worker projection scratch. Ready
    /// observations have no deferred scratch; their ordinary storage stays counted.
    pub fn maximum_working_bytes_upper_bound(&self) -> Option<usize> {
        self.retained_payload_bytes_upper_bound()?
            .checked_sub(self.pending_retained_bytes)?
            .checked_add(self.pending_working_bytes)
    }
}
