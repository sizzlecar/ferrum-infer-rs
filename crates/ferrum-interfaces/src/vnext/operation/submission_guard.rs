//! Explicit per-call native commit gate. No resource authority is minted here.
use super::{ProfiledSubmissionHandle, SubmissionWaveDispatchError};
use crate::execution_cost::{CoreReadbackRoute, GuardedNotSubmitted, GuardedNotSubmittedReason};
use crate::vnext::{
    BatchStepId, DeviceRuntime, DeviceSubmissionAttribution, StepResourceLease, VNextError,
};
use std::sync::Arc;

pub trait PreparedWaveSubmissionGuard: Send + Sync {
    /// True only for a dispatch committed to a cost witness. This does not
    /// require legacy commands to provide selected cost evidence. Backends use
    /// it only for commands that explicitly declare a library cost contract.
    fn relies_on_cost_witness(&self) -> bool {
        false
    }

    fn submission_mode(&self) -> crate::vnext::GuardedSubmissionMode {
        crate::vnext::GuardedSubmissionMode::ExactRoute
    }
    fn check_adaptive_preparation(
        &self,
        _intent: &crate::vnext::DeviceAdaptiveSubmissionIntent<'_>,
        _readback: CoreReadbackRoute,
    ) -> Result<(), GuardedNotSubmittedReason> {
        Err(GuardedNotSubmittedReason::AttributionUnavailable)
    }
    fn check(
        &self,
        attribution: &DeviceSubmissionAttribution,
        readback: CoreReadbackRoute,
    ) -> Result<(), GuardedNotSubmittedReason>;
}

/// Encoding and staging are already dropped, and this exact wave was never
/// submitted. The parent Step must still be rolled back before refunding any
/// caller-owned logical permission.
#[derive(Debug)]
pub struct PendingGuardedWaveRejection {
    reason: GuardedNotSubmittedReason,
    step_id: BatchStepId,
}

impl PendingGuardedWaveRejection {
    pub(super) fn new(reason: GuardedNotSubmittedReason, step_id: BatchStepId) -> Self {
        Self { reason, step_id }
    }

    pub fn reconcile_step<R: DeviceRuntime>(
        self,
        step: Arc<StepResourceLease<R>>,
    ) -> Result<GuardedNotSubmitted, (VNextError, Arc<StepResourceLease<R>>)> {
        if step.batch_step_id() != self.step_id {
            return Err((
                VNextError::InvalidExecutionPlan {
                    reason: "guard rejection belongs to another Step".to_owned(),
                },
                step,
            ));
        }
        match step.try_rollback_unsubmitted() {
            Ok(_) => Ok(GuardedNotSubmitted::reconciled(self.reason)),
            Err(failure) => {
                let error = VNextError::InvalidExecutionPlan {
                    reason: failure.error().to_string(),
                };
                Err((error, failure.into_step()))
            }
        }
    }
}

pub enum GuardedWaveSubmissionOutcome<R: DeviceRuntime> {
    /// The ordinary result retains the normal physical completion/error owner.
    Dispatch(Result<ProfiledSubmissionHandle<R>, SubmissionWaveDispatchError<R>>),
    NotSubmitted(PendingGuardedWaveRejection),
}
