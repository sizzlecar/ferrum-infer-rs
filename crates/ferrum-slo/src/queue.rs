use ferrum_execution::execution_attempt::*;
use ferrum_types::ReferenceEstimatorConfig;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc,
};

macro_rules! counters {
    ($($field:ident),+ $(,)?) => {
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
        pub struct ReferenceEstimatorStats {
            $(pub $field: u64,)+
            /// Reservations cover producer copies, queued records and active work.
            pub queued_records: usize,
            pub queued_bytes: usize,
        }
        #[derive(Default)]
        pub(super) struct Counters { $(pub $field: AtomicU64,)+ }
        impl Counters {
            fn snapshot(&self, records: usize, bytes: usize) -> ReferenceEstimatorStats {
                ReferenceEstimatorStats { $($field: self.$field.load(Ordering::Relaxed),)+ queued_records: records, queued_bytes: bytes }
            }
        }
    };
}

counters!(
    accepted,
    dropped_full,
    dropped_bytes,
    dropped_oversize,
    dropped_closed,
    excluded_attempts,
    invalid_attempts,
    domain_limit,
    trained_samples,
    published_snapshots,
    training_failures,
    worker_failures
);

pub(super) struct Shared {
    pub config: ReferenceEstimatorConfig,
    pub counters: Counters,
    records: AtomicUsize,
    bytes: AtomicUsize,
    publishing: AtomicUsize,
    closed: AtomicBool,
    wake: crossbeam_channel::Sender<()>,
}

impl Shared {
    pub fn new(config: ReferenceEstimatorConfig, wake: crossbeam_channel::Sender<()>) -> Self {
        Self {
            config,
            counters: Counters::default(),
            records: AtomicUsize::new(0),
            bytes: AtomicUsize::new(0),
            publishing: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
            wake,
        }
    }
    pub fn stats(&self) -> ReferenceEstimatorStats {
        self.counters.snapshot(
            self.records.load(Ordering::Acquire),
            self.bytes.load(Ordering::Acquire),
        )
    }
    pub fn stop(&self) {
        self.closed.store(true, Ordering::SeqCst);
        let _ = self.wake.try_send(());
    }
    pub fn worker_failed(&self) {
        self.counters
            .worker_failures
            .fetch_add(1, Ordering::Relaxed);
        self.stop();
    }
    pub fn drained(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
            && self.publishing.load(Ordering::SeqCst) == 0
            && self.records.load(Ordering::Acquire) == 0
    }
    fn notify_if_closed(&self) {
        if self.closed.load(Ordering::SeqCst) {
            let _ = self.wake.try_send(());
        }
    }
}

struct Publication(Arc<Shared>);
impl Drop for Publication {
    fn drop(&mut self) {
        self.0.publishing.fetch_sub(1, Ordering::SeqCst);
        self.0.notify_if_closed();
    }
}

struct Credit {
    shared: Arc<Shared>,
    bytes: usize,
}
impl Drop for Credit {
    fn drop(&mut self) {
        self.shared.bytes.fetch_sub(self.bytes, Ordering::AcqRel);
        self.shared.records.fetch_sub(1, Ordering::AcqRel);
        self.shared.notify_if_closed();
    }
}

pub(super) struct Envelope {
    // Payload drops before its reservation, including failed sends and unwind.
    pub summary: ExecutionAttemptSummary,
    _credit: Credit,
}

pub(super) struct Observer {
    shared: Arc<Shared>,
    sender: crossbeam_channel::Sender<Envelope>,
}

impl Observer {
    pub fn new(shared: Arc<Shared>, sender: crossbeam_channel::Sender<Envelope>) -> Self {
        Self { shared, sender }
    }

    fn reserve(&self, bytes: usize) -> Result<Credit, ExecutionAttemptDelivery> {
        if self
            .shared
            .records
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < self.shared.config.queue_capacity).then(|| n + 1)
            })
            .is_err()
        {
            self.shared
                .counters
                .dropped_full
                .fetch_add(1, Ordering::Relaxed);
            return Err(ExecutionAttemptDelivery::Dropped(
                ExecutionAttemptDrop::QueueFull,
            ));
        }
        if self
            .shared
            .bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                n.checked_add(bytes)
                    .filter(|total| *total <= self.shared.config.max_queue_bytes)
            })
            .is_err()
        {
            self.shared.records.fetch_sub(1, Ordering::AcqRel);
            self.shared
                .counters
                .dropped_bytes
                .fetch_add(1, Ordering::Relaxed);
            self.shared.notify_if_closed();
            return Err(ExecutionAttemptDelivery::Dropped(
                ExecutionAttemptDrop::QueueFull,
            ));
        }
        Ok(Credit {
            shared: Arc::clone(&self.shared),
            bytes,
        })
    }
}

impl ExecutionAttemptObserver for Observer {
    fn limits(&self) -> ExecutionAttemptLimits {
        ExecutionAttemptLimits {
            max_participants: self.shared.config.max_participants,
            max_summary_bytes: self
                .shared
                .config
                .max_summary_bytes
                .min(self.shared.config.max_queue_bytes),
        }
    }

    fn try_record(&self, summary: &ExecutionAttemptSummary) -> ExecutionAttemptDelivery {
        // The publication guard lasts through successful-send accounting, so a
        // concurrent stop cannot finish between send and accepted increment.
        self.shared.publishing.fetch_add(1, Ordering::SeqCst);
        let _publication = Publication(Arc::clone(&self.shared));
        if self.shared.closed.load(Ordering::SeqCst) {
            self.shared
                .counters
                .dropped_closed
                .fetch_add(1, Ordering::Relaxed);
            return ExecutionAttemptDelivery::Dropped(ExecutionAttemptDrop::Closed);
        }
        if !self.limits().accepts(summary.participants.len()) {
            self.record_drop(ExecutionAttemptDrop::ProjectionLimit);
            return ExecutionAttemptDelivery::Dropped(ExecutionAttemptDrop::ProjectionLimit);
        }
        let Some(bytes) = ExecutionAttemptSummary::projected_bytes(summary.participants.len())
        else {
            self.record_drop(ExecutionAttemptDrop::ProjectionLimit);
            return ExecutionAttemptDelivery::Dropped(ExecutionAttemptDrop::ProjectionLimit);
        };
        let credit = match self.reserve(bytes) {
            Ok(credit) => credit,
            Err(delivery) => return delivery,
        };
        if self.shared.closed.load(Ordering::SeqCst) {
            self.shared
                .counters
                .dropped_closed
                .fetch_add(1, Ordering::Relaxed);
            return ExecutionAttemptDelivery::Dropped(ExecutionAttemptDrop::Closed);
        }
        // Every member is fixed-size pure data. No clone precedes its credit.
        let envelope = Envelope {
            summary: summary.clone(),
            _credit: credit,
        };
        match self.sender.try_send(envelope) {
            Ok(()) => {
                self.shared
                    .counters
                    .accepted
                    .fetch_add(1, Ordering::Relaxed);
                ExecutionAttemptDelivery::Recorded
            }
            Err(crossbeam_channel::TrySendError::Full(_envelope)) => {
                self.shared
                    .counters
                    .dropped_full
                    .fetch_add(1, Ordering::Relaxed);
                ExecutionAttemptDelivery::Dropped(ExecutionAttemptDrop::QueueFull)
            }
            Err(crossbeam_channel::TrySendError::Disconnected(_envelope)) => {
                self.shared
                    .counters
                    .dropped_closed
                    .fetch_add(1, Ordering::Relaxed);
                ExecutionAttemptDelivery::Dropped(ExecutionAttemptDrop::Closed)
            }
        }
    }

    fn record_drop(&self, reason: ExecutionAttemptDrop) {
        let counter = match reason {
            ExecutionAttemptDrop::ProjectionLimit => &self.shared.counters.dropped_oversize,
            ExecutionAttemptDrop::QueueFull => &self.shared.counters.dropped_full,
            ExecutionAttemptDrop::Closed => &self.shared.counters.dropped_closed,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }
}
