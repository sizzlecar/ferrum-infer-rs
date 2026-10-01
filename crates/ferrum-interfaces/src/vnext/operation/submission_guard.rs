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
    invocation_id: crate::vnext::BatchInvocationId,
    lane_id: crate::vnext::ExecutionLaneId,
}

impl PendingGuardedWaveRejection {
    pub(super) fn new(
        reason: GuardedNotSubmittedReason,
        identity: &super::BatchOperationIdentity,
    ) -> Self {
        Self {
            reason,
            step_id: identity.batch_step_id(),
            invocation_id: identity.batch_invocation_id(),
            lane_id: identity.lane_id(),
        }
    }

    /// The host names/frontiers must describe this exact real Step. Only the
    /// successful rollback below can turn that checked binding into a receipt.
    pub fn reconcile_step_with_observation<R: DeviceRuntime>(
        self,
        step: Arc<StepResourceLease<R>>,
        expected: &crate::execution_cost::ExpectedWaveWork,
    ) -> Result<GuardedNotSubmitted, (VNextError, Arc<StepResourceLease<R>>)> {
        let rows = expected.participants();
        let signature = crate::execution_cost::guarded_participant_signature(rows.len(), |i| {
            let row = rows[i].selection();
            (
                &row.request_id,
                row.owner_incarnation.get(),
                row.work_generation.get(),
            )
        });
        if expected.lane_id() != self.lane_id
            || step.execution_lane().id() != self.lane_id
            || expected.plan_hash() != step.plan_evidence().plan_hash()
            || !step.matches_planning_participants(rows.iter().map(|row| row.resource()))
            || signature.is_none()
        {
            // Missing observation correlation must not alter the executor's
            // rollback or request-preservation semantics. It cannot mint the
            // optional observation proof; the original rollback still applies.
            return self.reconcile_step(step);
        }
        let binding = crate::execution_cost::GuardRollbackObservationBinding {
            batch_step: self.step_id.get(),
            batch_invocation: self.invocation_id.get(),
            lane_id: self.lane_id.get(),
            participant_count: rows.len(),
            participant_signature: signature.expect("checked above"),
        };
        self.reconcile_step(step)
            .map(|receipt| receipt.with_observation(binding))
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
