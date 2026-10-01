//! The native CPU fill retains no device token-policy uploads. Its empty
//! residency receipt still requires an independently idle executor.
use super::*;
use ferrum_interfaces::model_executor::{
    TokenPolicyResidencyInvalidation as Outcome, TokenPolicyResidencyUnavailable as Busy,
};

#[derive(Default)]
pub(super) struct Lifecycle {
    pub(super) admitted: std::collections::HashSet<RequestId>,
    pub(super) executing: usize,
}

pub(super) struct Work<'a>(&'a Mutex<Lifecycle>);
impl Drop for Work<'_> {
    fn drop(&mut self) {
        let mut state = self.0.lock();
        state.executing = state.executing.checked_sub(1).unwrap();
    }
}

impl ControlledExecutor {
    pub(super) fn token_policy_work(&self) -> Work<'_> {
        let mut state = self.token_policy_lifecycle.lock();
        state.executing = state.executing.checked_add(1).unwrap();
        Work(&self.token_policy_lifecycle)
    }
}

pub(super) fn invalidate(executor: &ControlledExecutor) -> Outcome {
    if !executor
        .native_structured_submission
        .load(Ordering::Acquire)
    {
        return Outcome::Unsupported;
    }
    let unavailable = |reason| Outcome::Unavailable { reason };
    // Hold the same gate used by admission and execution entry through every
    // check. A parked call counts as work even before a Step or cache exists.
    let Some(state) = executor.token_policy_lifecycle.try_lock() else {
        return unavailable(Busy::ExecutorStateBusy);
    };
    if !state.admitted.is_empty() {
        return unavailable(Busy::ActiveRequests);
    }
    if state.executing != 0 {
        return unavailable(Busy::CompletionWork);
    }
    let Some(histories) = executor.native_structured_history.try_lock() else {
        return unavailable(Busy::ExecutorStateBusy);
    };
    if !histories.is_empty() {
        return unavailable(Busy::ActiveRequests);
    }
    let Some(caches) = executor.produced_caches.try_lock() else {
        return unavailable(Busy::ExecutorStateBusy);
    };
    if caches.iter().any(|cache| cache.upgrade().is_some()) {
        return unavailable(Busy::ActiveRequests);
    }
    if executor
        .evidence
        .lane
        .as_ref()
        .and_then(|lane| lane.cost_readback_available_bytes())
        .is_none()
    {
        return unavailable(Busy::ExecutionLane);
    }
    // ControlledCpuFill constructs fresh CPU output per call; it has no mask
    // cache, upload cache or retained product policy to erase.
    Outcome::Cleared { cleared_entries: 0 }
}
