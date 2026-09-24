//! Only the explicit calibration entry uses this path. The ordinary prepare,
//! upload and successful-submission publication paths remain unchanged.
use super::*;
use ferrum_interfaces::model_executor::{
    TokenPolicyResidencyInvalidation as Outcome, TokenPolicyResidencyUnavailable as Busy,
};

pub(super) fn invalidate<R: DeviceRuntime>(
    registry: &Mutex<VNextSequenceRegistry<R>>,
    worker: &VNextCompletionWorker,
    lane: &ExecutionLane<R>,
    residency: &Mutex<VNextProductTokenMaskResidency>,
) -> Outcome {
    let Some(registry) = registry.try_lock() else {
        return unavailable(Busy::ExecutorStateBusy);
    };
    if registry.total_len() != 0 {
        return unavailable(Busy::ActiveRequests);
    }
    // Keep the empty registry locked: all new product admission must install
    // a registry owner before it can prepare/upload or publish residency.
    worker
        .try_with_idle(|| {
            // This existing nonblocking query verifies a ready, non-failed lane,
            // its real descriptor, and zero in-flight submissions. Its byte
            // result is deliberately not used as capacity or execution authority.
            if lane.cost_readback_available_bytes().is_none() {
                return unavailable(Busy::ExecutionLane);
            }
            let Some(mut residency) = residency.try_lock() else {
                return unavailable(Busy::ResidencyBusy);
            };
            let cleared_entries = residency.entries.len();
            residency.clear();
            Outcome::Cleared { cleared_entries }
        })
        .unwrap_or_else(|| unavailable(Busy::CompletionWork))
}

fn unavailable(reason: Busy) -> Outcome {
    Outcome::Unavailable { reason }
}

#[cfg(test)]
mod tests;
