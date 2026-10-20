//! Read-only counts that do not retain an executor or any execution resources.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Default)]
pub(crate) struct ExecutorCounters {
    pub(crate) prefill: AtomicU64,
    pub(crate) decode: AtomicU64,
}

/// May outlive the executor's checked retirement. Clones share only counters.
#[derive(Clone)]
pub struct ExecutorObservation {
    counters: Arc<ExecutorCounters>,
}

impl ExecutorObservation {
    pub(crate) fn new(counters: &Arc<ExecutorCounters>) -> Self {
        Self {
            counters: Arc::clone(counters),
        }
    }

    pub fn prefill_count(&self) -> u64 {
        self.counters.prefill.load(Ordering::Relaxed)
    }

    pub fn decode_count(&self) -> u64 {
        self.counters.decode.load(Ordering::Relaxed)
    }
}
