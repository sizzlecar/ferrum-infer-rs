//! Transient failures get a bounded wake, independently of already missed SLOs.
use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) enum ControllerRetryReason {
    SnapshotBusy,
    ComputeBudget,
    ChangedEvidence,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ControllerRetryWake {
    at: Instant,
    reason: ControllerRetryReason,
}

impl EngineInner {
    pub(super) fn controller_retry_pending(&self) -> bool {
        if self.config.scheduler.slo.mode != ferrum_types::SloMode::Enforce {
            return false;
        }
        let mut state = self.slo_controller.lock();
        match state.retry {
            Some(wake) if slo_clock_now() < wake.at => true,
            Some(_) => {
                state.retry = None;
                false
            }
            None => false,
        }
    }

    pub(super) fn arm_controller_retry(&self, reason: ControllerRetryReason) {
        if self.config.scheduler.slo.mode != ferrum_types::SloMode::Enforce {
            return;
        }
        let Some(at) = slo_clock_now().checked_add(std::time::Duration::from_millis(
            self.config.scheduler.slo.planner.retry_backoff_ms.get(),
        )) else {
            return;
        };
        let mut state = self.slo_controller.lock();
        if state.retry.is_none_or(|pending| at < pending.at) {
            state.retry = Some(ControllerRetryWake { at, reason });
        }
    }

    pub(in crate::continuous_engine) async fn wait_for_slo_controller_retry(&self) {
        let wake = { self.slo_controller.lock().retry };
        let Some(wake) = wake else {
            return std::future::pending().await;
        };
        tokio::time::sleep_until(tokio::time::Instant::from_std(wake.at)).await;
        let mut state = self.slo_controller.lock();
        if state
            .retry
            .is_some_and(|current| current.at <= slo_clock_now())
        {
            state.retry = None;
            tracing::trace!(reason = ?wake.reason, "bounded controller retry wake");
        }
    }
}
