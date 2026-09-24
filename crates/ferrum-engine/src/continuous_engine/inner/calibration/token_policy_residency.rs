use super::*;
use ferrum_interfaces::model_executor::{
    TokenPolicyResidencyInvalidation as Outcome, TokenPolicyResidencyUnavailable as Busy,
};

impl CalibrationSession {
    /// Forget only token-policy upload residency, between complete cohorts.
    /// This operation does not execute or train a wave, drain/abandon work,
    /// reset resource state, or establish a general device cold-start claim.
    /// The caller records this receipt separately from subsequent actual waves.
    pub async fn invalidate_token_policy_residency(&mut self) -> Outcome {
        if self.indeterminate {
            return unavailable(Busy::SessionIndeterminate);
        }
        if self.pending.is_some() {
            return unavailable(Busy::PendingWave);
        }
        let inner = &self.engine.inner;
        if inner.shutdown_started.load(Ordering::Acquire) {
            return unavailable(Busy::SessionClosed);
        }
        let Ok(_iteration) = inner.iteration_lock.try_lock() else {
            return unavailable(Busy::IterationBusy);
        };
        let Some(sequences) = inner.sequences.try_read() else {
            return unavailable(Busy::ActiveRequests);
        };
        if !sequences.is_empty()
            || inner.scheduler.active_count() != 0
            || inner.scheduler.waiting_count() != 0
        {
            return unavailable(Busy::ActiveRequests);
        }
        let Some(controller) = inner.slo_controller.try_lock() else {
            return unavailable(Busy::PendingMaintenanceOrPublication);
        };
        if controller.has_pending_calibration_work() {
            return unavailable(Busy::PendingMaintenanceOrPublication);
        }
        let Some(restores) = inner.prefix_restore_pending.try_lock() else {
            return unavailable(Busy::PendingRestore);
        };
        if !restores.is_empty() {
            return unavailable(Busy::PendingRestore);
        }
        // All engine gates remain held through the backend's own nonblocking
        // registry/worker/lane checks. No old cleanup ticket is silently dropped.
        inner
            .model_executor
            .calibration_invalidate_token_policy_residency()
    }
}

fn unavailable(reason: Busy) -> Outcome {
    Outcome::Unavailable { reason }
}
