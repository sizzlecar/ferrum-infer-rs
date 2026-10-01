//! Queue occupancy leases; consumer release never takes producer admission.
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

pub(super) struct QueuedRowBudget {
    maximum: usize,
    retained: AtomicUsize,
}
impl QueuedRowBudget {
    pub(super) fn new(maximum: usize) -> Arc<Self> {
        Arc::new(Self {
            maximum,
            retained: AtomicUsize::new(0),
        })
    }
    pub(super) fn reserve(self: &Arc<Self>, rows: usize) -> Option<QueuedRows> {
        self.retained
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                n.checked_add(rows).filter(|n| *n <= self.maximum)
            })
            .ok()?;
        Some(QueuedRows {
            budget: Arc::clone(self),
            rows,
        })
    }
}
pub(super) struct QueuedRows {
    budget: Arc<QueuedRowBudget>,
    rows: usize,
}
impl Drop for QueuedRows {
    fn drop(&mut self) {
        let previous = self.budget.retained.fetch_sub(self.rows, Ordering::AcqRel);
        debug_assert!(previous >= self.rows);
    }
}

/// Covers the only interval where a channel slot exists without a committed
/// acceptance watermark. An unwound producer publishes a failed FIFO position;
/// it cannot strand the channel or let the next producer reuse its ordinal.
pub(super) struct QueuedPublication<'a> {
    sink: &'a super::BoundedCostSampleSink,
    ordinal: u64,
    entry: Option<Arc<super::QueueEntry>>,
}
impl<'a> QueuedPublication<'a> {
    pub(super) fn new(
        sink: &'a super::BoundedCostSampleSink,
        ordinal: u64,
        entry: Arc<super::QueueEntry>,
    ) -> Self {
        Self {
            sink,
            ordinal,
            entry: Some(entry),
        }
    }
    pub(super) fn entry(&self) -> &super::QueueEntry {
        self.entry.as_ref().expect("publication owns entry")
    }
    pub(super) fn complete(self) {
        self.entry()
            .publication_completed
            .store(true, Ordering::Release);
        // Drop performs the same unique-owner transfer and release publication.
    }
}
impl Drop for QueuedPublication<'_> {
    fn drop(&mut self) {
        let entry = self.entry.take().expect("publication owns entry");
        if !entry.publication_completed.load(Ordering::Acquire) {
            if let super::QueuedCostPayload::Prefix(sample) = &entry.payload {
                sample.abandon();
            } else {
                self.sink.increment(&self.sink.entries_abandoned);
            }
            if matches!(&entry.payload, super::QueuedCostPayload::Raw(_)) {
                self.sink.increment(&self.sink.raw_lost);
                self.sink.increment(&self.sink.raw_abandoned);
            }
        }
        drop(entry);
        self.sink.accepted.store(self.ordinal, Ordering::Release);
        if let Some(worker) = self.sink.worker.get() {
            worker.unpark();
        }
    }
}
