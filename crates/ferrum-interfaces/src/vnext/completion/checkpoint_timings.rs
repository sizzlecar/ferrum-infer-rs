//! Host timing of the existing checkpoint path. No device query or wait is
//! introduced here; these counters do not measure GPU execution time.

use super::{CompletionReaper, StateTransferKind};
use crate::vnext::DeviceRuntime;
use serde::Serialize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CheckpointTimingMeasurement {
    pub samples: u64,
    pub total_ns: u64,
    pub max_ns: u64,
}

impl CheckpointTimingMeasurement {
    fn record(&mut self, duration: Duration) {
        let ns = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        self.samples = self.samples.saturating_add(1);
        self.total_ns = self.total_ns.saturating_add(ns);
        self.max_ns = self.max_ns.max(ns);
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CheckpointOperationTimings {
    /// Public facade validation and backing claim, including skipped and
    /// deferred attempts. Private native submission tests bypass this phase.
    pub prepare_claim: CheckpointTimingMeasurement,
    /// Native lease/slot preparation, command encoding and submission,
    /// including definitely-not-submitted and indeterminate outcomes.
    pub encode_submit: CheckpointTimingMeasurement,
    /// Existing fence queries, waits and recovery drains. Samples count calls,
    /// not transfers. Time between observations is not included.
    pub fence_recovery: CheckpointTimingMeasurement,
    /// Terminal resource transitions and result outbox publication. Restore
    /// still requires its consumer's separate conditional frontier commit.
    pub publication: CheckpointTimingMeasurement,
}

/// Samples count calls, including empty replacement/eviction lookups, rather
/// than the number of checkpoint owners released by those calls.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CheckpointCacheTimings {
    pub maintenance: CheckpointTimingMeasurement,
    pub replacement_drop: CheckpointTimingMeasurement,
    pub eviction_drop: CheckpointTimingMeasurement,
    pub index_publication: CheckpointTimingMeasurement,
    /// May also record nested native fence/recovery/publication measurements.
    pub abandoned_recovery: CheckpointTimingMeasurement,
}

/// Host phase totals may overlap (notably abandoned recovery and the native
/// phases it invokes). They must not be summed as device execution time.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CheckpointTimingSnapshot {
    pub capture: CheckpointOperationTimings,
    pub restore: CheckpointOperationTimings,
    pub cache: CheckpointCacheTimings,
}

#[derive(Debug, Clone, Copy)]
pub enum CheckpointCacheTimingPhase {
    Maintenance,
    ReplacementDrop,
    EvictionDrop,
    IndexPublication,
    AbandonedRecovery,
}

#[derive(Clone, Copy)]
pub(super) enum CheckpointTimingPhase {
    PrepareClaim,
    EncodeSubmit,
    FenceRecovery,
    Publication,
}

#[derive(Default)]
pub(super) struct CheckpointTimingCounters(Mutex<CheckpointTimingSnapshot>);

impl CheckpointTimingCounters {
    fn record(&self, kind: StateTransferKind, phase: CheckpointTimingPhase, elapsed: Duration) {
        let mut snapshot = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let operation = match kind {
            StateTransferKind::Capture => &mut snapshot.capture,
            StateTransferKind::Restore => &mut snapshot.restore,
        };
        match phase {
            CheckpointTimingPhase::PrepareClaim => &mut operation.prepare_claim,
            CheckpointTimingPhase::EncodeSubmit => &mut operation.encode_submit,
            CheckpointTimingPhase::FenceRecovery => &mut operation.fence_recovery,
            CheckpointTimingPhase::Publication => &mut operation.publication,
        }
        .record(elapsed);
    }

    pub(super) fn start(
        self: &Arc<Self>,
        kind: StateTransferKind,
        phase: CheckpointTimingPhase,
    ) -> CheckpointPhaseTimer {
        CheckpointPhaseTimer {
            counters: Arc::clone(self),
            kind,
            phase,
            started: Instant::now(),
        }
    }
}

/// Drop also accounts for early errors and unwinding, without retaining the
/// reaper or touching its slot/lane/resource locks.
pub(super) struct CheckpointPhaseTimer {
    counters: Arc<CheckpointTimingCounters>,
    kind: StateTransferKind,
    phase: CheckpointTimingPhase,
    started: Instant,
}

impl Drop for CheckpointPhaseTimer {
    fn drop(&mut self) {
        self.counters
            .record(self.kind, self.phase, self.started.elapsed());
    }
}

impl<R: DeviceRuntime> CompletionReaper<R> {
    pub fn checkpoint_timing_snapshot(&self) -> CheckpointTimingSnapshot {
        *self
            .checkpoint_timings
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Resets only observations, never transfer state. For a clean measurement
    /// interval callers should first quiesce workers: a phase spanning reset is
    /// recorded in full when it ends, just like other host timing counters.
    pub fn reset_checkpoint_timings(&self) {
        *self
            .checkpoint_timings
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            CheckpointTimingSnapshot::default();
    }

    pub fn record_checkpoint_cache_timing(
        &self,
        phase: CheckpointCacheTimingPhase,
        elapsed: Duration,
    ) {
        let mut snapshot = self
            .checkpoint_timings
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let cache = &mut snapshot.cache;
        match phase {
            CheckpointCacheTimingPhase::Maintenance => &mut cache.maintenance,
            CheckpointCacheTimingPhase::ReplacementDrop => &mut cache.replacement_drop,
            CheckpointCacheTimingPhase::EvictionDrop => &mut cache.eviction_drop,
            CheckpointCacheTimingPhase::IndexPublication => &mut cache.index_publication,
            CheckpointCacheTimingPhase::AbandonedRecovery => &mut cache.abandoned_recovery,
        }
        .record(elapsed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_timing_totals_saturate_without_wrapping() {
        let mut timing = CheckpointTimingMeasurement::default();
        timing.record(Duration::from_nanos(7));
        timing.record(Duration::from_nanos(3));
        assert_eq!(
            timing,
            CheckpointTimingMeasurement {
                samples: 2,
                total_ns: 10,
                max_ns: 7
            }
        );
        timing.record(Duration::MAX);
        timing.record(Duration::from_nanos(1));
        assert_eq!(timing.total_ns, u64::MAX);
        assert_eq!(timing.max_ns, u64::MAX);
    }
}
