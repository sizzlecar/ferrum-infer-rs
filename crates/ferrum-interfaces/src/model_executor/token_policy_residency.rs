//! An explicit calibration operation, not a device-cache reset or cold-start
//! claim. Clearing residency evidence forces the next prepare to establish its
//! own upload; only the subsequent actual route can prove that upload occurred.
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum TokenPolicyResidencyInvalidation {
    Cleared {
        cleared_entries: usize,
    },
    Unavailable {
        reason: TokenPolicyResidencyUnavailable,
    },
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenPolicyResidencyUnavailable {
    SessionClosed,
    SessionIndeterminate,
    PendingWave,
    IterationBusy,
    ActiveRequests,
    PendingMaintenanceOrPublication,
    PendingRestore,
    ExecutorStateBusy,
    CompletionWork,
    ExecutionLane,
    ResidencyBusy,
}
