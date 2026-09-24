//! Internal obligation timers never manufacture resource readiness. Observe
//! records sticky violations inside the existing wait, without another batch
//! selection or admission probe merely because a time budget expired.

use super::*;
use ferrum_interfaces::{SloTimingBoundary, SloViolationWake};
use ferrum_types::SloMode;

/// Shared by SLO wait observation, token commit and terminal observation. Tokio
/// delegates to the platform monotonic clock outside a paused test runtime.
pub(in crate::continuous_engine) fn slo_clock_now() -> Instant {
    tokio::time::Instant::now().into_std()
}

fn has_token_obligation(sequence: &SequenceState) -> bool {
    !matches!(
        sequence.phase,
        RequestPhase::Completed | RequestPhase::Cancelled | RequestPhase::AdmissionFailed
    ) && sequence.generated_tokens.len() < sequence.sampling_params.max_tokens
}

impl EngineInner {
    pub(in crate::continuous_engine) fn observe_slo_waits(
        &self,
        now: impl FnOnce() -> Instant,
    ) -> Option<SloTimingBoundary> {
        if self.config.scheduler.slo.mode == SloMode::Off {
            return None;
        }
        let mut sequences = self.sequences.write();
        // Read inside the owner lock: a concurrent commit while acquiring it
        // must not make an otherwise valid observation appear to go backwards.
        let at = now();
        let mut newly_due = None;
        for sequence in sequences.values_mut().filter(|s| has_token_obligation(s)) {
            let Some(timing) = sequence.slo.as_mut().filter(|state| state.is_trusted()) else {
                continue;
            };
            // Invalid time remains sticky in the timing owner, and will be
            // reported as Unknown. It must not disable unrelated requests.
            let before = timing.violations();
            if timing.observe_wait(at).is_ok() {
                let after = timing.violations();
                let changed = if !before.ttft && after.ttft {
                    Some(SloTimingBoundary::FirstToken)
                } else if !before.itl && after.itl {
                    Some(SloTimingBoundary::InterToken)
                } else if !before.tpot && after.tpot {
                    Some(SloTimingBoundary::TokenPrefix)
                } else {
                    None
                };
                newly_due = newly_due.or(changed);
            }
        }
        newly_due
    }

    pub(in crate::continuous_engine) fn next_slo_violation_wake(&self) -> Option<SloViolationWake> {
        if self.config.scheduler.slo.mode == SloMode::Off {
            return None;
        }
        self.sequences
            .write()
            .values_mut()
            .filter(|sequence| has_token_obligation(sequence))
            .filter_map(|sequence| sequence.slo.as_mut()?.arm_violation_wake().ok()?)
            .min_by_key(|wake| wake.at)
    }

    pub(in crate::continuous_engine) async fn wait_for_slo_deadline(&self) -> SloTimingBoundary {
        loop {
            let Some(wake) = self.next_slo_violation_wake() else {
                return std::future::pending().await;
            };
            tokio::time::sleep_until(tokio::time::Instant::from_std(wake.at)).await;
            // Tokio's clock is the same monotonic clock in production, and
            // allows this waiting path to be exercised with virtual time.
            let newly_due = self.observe_slo_waits(slo_clock_now);
            if self.config.scheduler.slo.mode == SloMode::Enforce {
                if let Some(boundary) = newly_due {
                    return boundary;
                }
            }
            // Observe does not invoke the scheduler or alter its fairness/
            // retry counters. A different, unrecorded boundary may still be
            // due later; recorded violations cannot cause a timer spin.
        }
    }
}
