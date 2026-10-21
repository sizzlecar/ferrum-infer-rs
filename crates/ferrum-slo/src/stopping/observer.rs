//! Bounded nonblocking request collection and owned background publication.

use super::*;
use crate::EstimatorError;
use arc_swap::{ArcSwap, ArcSwapOption};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc,
};
use tokio::sync::watch;

#[derive(Debug, Clone, Copy)]
pub struct StoppingCollectorConfig {
    pub stopping: StoppingConfig,
    pub queue_capacity: usize,
    /// Conservative logical payload reservation, including full rule identity
    /// per notice even if several notices share an Arc allocation.
    pub max_queue_bytes: usize,
    pub publish_interval: Duration,
}

impl Default for StoppingCollectorConfig {
    fn default() -> Self {
        Self {
            stopping: StoppingConfig::default(),
            queue_capacity: 1024,
            max_queue_bytes: 4 * 1024 * 1024,
            publish_interval: Duration::from_millis(20),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct StoppingCollectorStats {
    pub accepted: u64,
    pub dropped_full: u64,
    pub dropped_bytes: u64,
    pub dropped_projection: u64,
    pub dropped_closed: u64,
    pub processed: u64,
    pub published: u64,
    pub worker_failures: u64,
    pub max_feedback_delay_ns: u64,
    pub queued_records: usize,
    pub queued_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct StoppingWorkerExit {
    pub collector: StoppingCollectorStats,
    pub window: StoppingStats,
}

#[derive(Default)]
struct Counters {
    accepted: AtomicU64,
    dropped_full: AtomicU64,
    dropped_bytes: AtomicU64,
    dropped_projection: AtomicU64,
    dropped_closed: AtomicU64,
    processed: AtomicU64,
    published: AtomicU64,
    worker_failures: AtomicU64,
    max_feedback_delay_ns: AtomicU64,
}

struct Shared {
    config: StoppingCollectorConfig,
    counters: Counters,
    closed: AtomicBool,
    publishing: AtomicUsize,
    records: AtomicUsize,
    bytes: AtomicUsize,
    producer: AtomicU64,
    highest_admission: AtomicU64,
    admissions: Box<[ArcSwapOption<AdmissionTime>]>,
    wake: crossbeam_channel::Sender<()>,
}

struct AdmissionTime {
    lifecycle: RequestLifecycleId,
    admitted_at: Instant,
}

impl Shared {
    fn stop(&self) {
        self.closed.store(true, Ordering::SeqCst);
        let _ = self.wake.try_send(());
    }
    fn drained(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
            && self.publishing.load(Ordering::SeqCst) == 0
            && self.records.load(Ordering::Acquire) == 0
    }
    fn stats(&self) -> StoppingCollectorStats {
        let c = &self.counters;
        StoppingCollectorStats {
            accepted: c.accepted.load(Ordering::Relaxed),
            dropped_full: c.dropped_full.load(Ordering::Relaxed),
            dropped_bytes: c.dropped_bytes.load(Ordering::Relaxed),
            dropped_projection: c.dropped_projection.load(Ordering::Relaxed),
            dropped_closed: c.dropped_closed.load(Ordering::Relaxed),
            processed: c.processed.load(Ordering::Relaxed),
            published: c.published.load(Ordering::Relaxed),
            worker_failures: c.worker_failures.load(Ordering::Relaxed),
            max_feedback_delay_ns: c.max_feedback_delay_ns.load(Ordering::Relaxed),
            queued_records: self.records.load(Ordering::Acquire),
            queued_bytes: self.bytes.load(Ordering::Acquire),
        }
    }
    fn advance_window(&self, window: &mut StoppingWindow) -> bool {
        let admission = self.highest_admission.load(Ordering::Acquire);
        if admission == 0 {
            return false;
        }
        let producer = self.producer.load(Ordering::Acquire);
        let mut changed = window.advance_admission_watermark(RequestLifecycleId {
            producer,
            admission,
        });
        for slot in &self.admissions {
            let record = slot.load();
            if let Some(record) = record.as_ref().filter(|record| {
                record.lifecycle.producer == producer && record.lifecycle.admission <= admission
            }) {
                changed |= window.record_admission(record.lifecycle, record.admitted_at);
            }
        }
        changed
    }
}

struct Publication(Arc<Shared>);
impl Drop for Publication {
    fn drop(&mut self) {
        self.0.publishing.fetch_sub(1, Ordering::SeqCst);
        if self.0.closed.load(Ordering::SeqCst) {
            let _ = self.0.wake.try_send(());
        }
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
        if self.shared.closed.load(Ordering::SeqCst) {
            let _ = self.shared.wake.try_send(());
        }
    }
}
struct Envelope {
    observation: RequestStoppingObservation,
    _credit: Credit,
}
struct Observer {
    shared: Arc<Shared>,
    sender: crossbeam_channel::Sender<Envelope>,
}

impl RequestStoppingObserver for Observer {
    fn limits(&self) -> RequestObservationLimits {
        RequestObservationLimits {
            max_rule_bytes: self.shared.config.stopping.max_rule_bytes,
        }
    }
    fn record_admission(&self, lifecycle: RequestLifecycleId, admitted_at: Instant) {
        self.shared.publishing.fetch_add(1, Ordering::SeqCst);
        let _publication = Publication(self.shared.clone());
        if self.shared.closed.load(Ordering::SeqCst) {
            return;
        }
        if lifecycle.producer == 0 || lifecycle.admission == 0 {
            return;
        }
        if let Err(existing) = self.shared.producer.compare_exchange(
            0,
            lifecycle.producer,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            if existing != lifecycle.producer {
                return;
            }
        }
        let index = (lifecycle.admission % self.shared.admissions.len() as u64) as usize;
        let slot = &self.shared.admissions[index];
        let previous = slot.load();
        if previous.as_ref().is_some_and(|previous| {
            previous.lifecycle.admission > lifecycle.admission
                || (previous.lifecycle == lifecycle && previous.admitted_at <= admitted_at)
        }) {
            return;
        }
        drop(previous);
        let entry = Arc::new(AdmissionTime {
            lifecycle,
            admitted_at,
        });
        // Monotone serial replacement keeps the latest admission window even
        // when concurrent producers publish in reverse order. A conflicting
        // duplicate may only move its ingress earlier, never renew its age.
        // The fixed ring owns at most window_requests entries; no queue or
        // blocking lock can lose the last admission's maturity evidence.
        slot.rcu(|previous| {
            if previous.as_ref().is_some_and(|previous| {
                previous.lifecycle.admission > lifecycle.admission
                    || (previous.lifecycle == lifecycle && previous.admitted_at <= admitted_at)
            }) {
                previous.clone()
            } else {
                Some(entry.clone())
            }
        });
        if self
            .shared
            .highest_admission
            .fetch_max(lifecycle.admission, Ordering::AcqRel)
            < lifecycle.admission
        {
            let _ = self.shared.wake.try_send(());
        }
    }
    fn try_record(&self, observation: &RequestStoppingObservation) -> RequestObservationDelivery {
        self.record_admission(observation.lifecycle, observation.admitted_at);
        self.shared.publishing.fetch_add(1, Ordering::SeqCst);
        let _publication = Publication(self.shared.clone());
        if self.shared.closed.load(Ordering::SeqCst) {
            self.record_drop(RequestObservationDrop::Closed);
            return RequestObservationDelivery::Dropped(RequestObservationDrop::Closed);
        }
        let Some(bytes) = std::mem::size_of::<Envelope>()
            .checked_add(observation.rules.identity().len())
            .filter(|_| observation.rules.identity().len() <= self.limits().max_rule_bytes)
        else {
            self.record_drop(RequestObservationDrop::ProjectionLimit);
            return RequestObservationDelivery::Dropped(RequestObservationDrop::ProjectionLimit);
        };
        if self
            .shared
            .records
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (current < self.shared.config.queue_capacity).then(|| current + 1)
            })
            .is_err()
        {
            self.record_drop(RequestObservationDrop::QueueFull);
            return RequestObservationDelivery::Dropped(RequestObservationDrop::QueueFull);
        }
        if self
            .shared
            .bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current
                    .checked_add(bytes)
                    .filter(|total| *total <= self.shared.config.max_queue_bytes)
            })
            .is_err()
        {
            self.shared.records.fetch_sub(1, Ordering::AcqRel);
            self.shared
                .counters
                .dropped_bytes
                .fetch_add(1, Ordering::Relaxed);
            return RequestObservationDelivery::Dropped(RequestObservationDrop::QueueFull);
        }
        let credit = Credit {
            shared: self.shared.clone(),
            bytes,
        };
        if self.shared.closed.load(Ordering::SeqCst) {
            self.record_drop(RequestObservationDrop::Closed);
            return RequestObservationDelivery::Dropped(RequestObservationDrop::Closed);
        }
        // No deep allocation/copy happens before credit. Rules remain immutable.
        let envelope = Envelope {
            observation: observation.clone(),
            _credit: credit,
        };
        match self.sender.try_send(envelope) {
            Ok(()) => {
                self.shared
                    .counters
                    .accepted
                    .fetch_add(1, Ordering::Relaxed);
                RequestObservationDelivery::Recorded
            }
            Err(crossbeam_channel::TrySendError::Full(_)) => {
                self.record_drop(RequestObservationDrop::QueueFull);
                RequestObservationDelivery::Dropped(RequestObservationDrop::QueueFull)
            }
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                self.record_drop(RequestObservationDrop::Closed);
                RequestObservationDelivery::Dropped(RequestObservationDrop::Closed)
            }
        }
    }
    fn record_drop(&self, reason: RequestObservationDrop) {
        let counter = match reason {
            RequestObservationDrop::ProjectionLimit => &self.shared.counters.dropped_projection,
            RequestObservationDrop::QueueFull => &self.shared.counters.dropped_full,
            RequestObservationDrop::Closed => &self.shared.counters.dropped_closed,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

struct Source {
    table: ArcSwap<StoppingTable>,
}
impl StoppingSnapshotSource for Source {
    fn snapshot(&self) -> Arc<dyn StoppingSnapshot> {
        self.table.load_full()
    }
}

/// Independent collector owner assembled alongside the cost-learning owner.
/// Both use bounded nonblocking observation and immutable publication; dropping
/// a public shutdown future never aborts the original background worker.
pub struct StoppingEstimatorOwner {
    shared: Arc<Shared>,
    observer: Arc<Observer>,
    source: Arc<Source>,
    completion: watch::Receiver<Option<Result<StoppingWorkerExit, EstimatorError>>>,
}

impl StoppingEstimatorOwner {
    pub fn spawn(config: StoppingCollectorConfig) -> Result<Self, EstimatorError> {
        config
            .stopping
            .validate()
            .map_err(|reason| EstimatorError::Configuration(reason.into()))?;
        if config.queue_capacity == 0
            || config.publish_interval.is_zero()
            || config.max_queue_bytes < std::mem::size_of::<Envelope>() + 1
            || config.publish_interval > Duration::from_secs(60)
        {
            return Err(EstimatorError::Configuration(
                "invalid stopping collector capacities or publication interval".into(),
            ));
        }
        config
            .queue_capacity
            .checked_add(1)
            .and_then(usize::checked_next_power_of_two)
            .and_then(|mark| mark.checked_mul(2))
            .ok_or_else(|| {
                EstimatorError::Configuration("stopping queue index range overflow".into())
            })?;
        std::alloc::Layout::array::<Envelope>(config.queue_capacity).map_err(|_| {
            EstimatorError::Configuration("stopping queue allocation extent overflow".into())
        })?;
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|_| EstimatorError::RuntimeUnavailable)?;
        let (sender, receiver) = crossbeam_channel::bounded(config.queue_capacity);
        let (wake, wake_receiver) = crossbeam_channel::bounded(1);
        let shared = Arc::new(Shared {
            config,
            counters: Counters::default(),
            closed: AtomicBool::new(false),
            publishing: AtomicUsize::new(0),
            records: AtomicUsize::new(0),
            bytes: AtomicUsize::new(0),
            producer: AtomicU64::new(0),
            highest_admission: AtomicU64::new(0),
            admissions: (0..config.stopping.window_requests)
                .map(|_| ArcSwapOption::empty())
                .collect(),
            wake,
        });
        let observer = Arc::new(Observer {
            shared: shared.clone(),
            sender,
        });
        let mut window = StoppingWindow::new(config.stopping)
            .map_err(|reason| EstimatorError::Configuration(reason.into()))?;
        let source = Arc::new(Source {
            table: ArcSwap::from_pointee(window.snapshot()),
        });
        let worker_shared = shared.clone();
        let worker_source = source.clone();
        let worker = runtime.spawn_blocking(move || {
            let mut dirty = false; let mut published_at = Instant::now();
            while !worker_shared.drained() {
                crossbeam_channel::select! {
                    recv(receiver) -> message => {
                        let Ok(envelope) = message else { break; };
                        let now = Instant::now();
                        if let Some(delay) = now.checked_duration_since(envelope.observation.observed_at) {
                            worker_shared.counters.max_feedback_delay_ns.fetch_max(delay.as_nanos().min(u64::MAX as u128) as u64, Ordering::Relaxed);
                        }
                        window.observe(&envelope.observation, now);
                        worker_shared.counters.processed.fetch_add(1, Ordering::Relaxed);
                        dirty = true;
                    }
                    recv(wake_receiver) -> _ => {},
                    default(config.publish_interval) => {},
                }
                dirty |= worker_shared.advance_window(&mut window);
                if dirty && published_at.elapsed() >= config.publish_interval {
                    worker_source.table.store(Arc::new(window.snapshot()));
                    worker_shared.counters.published.fetch_add(1, Ordering::Relaxed);
                    published_at = Instant::now(); dirty = false;
                }
            }
            dirty |= worker_shared.advance_window(&mut window);
            if dirty {
                worker_source.table.store(Arc::new(window.snapshot()));
                worker_shared.counters.published.fetch_add(1, Ordering::Relaxed);
            }
            window.stats()
        });
        let (completion_sender, completion) = watch::channel(None);
        let joined = shared.clone();
        runtime.spawn(async move {
            let result = match worker.await {
                Ok(window) => Ok(StoppingWorkerExit {
                    collector: joined.stats(),
                    window,
                }),
                Err(error) => {
                    joined
                        .counters
                        .worker_failures
                        .fetch_add(1, Ordering::Relaxed);
                    joined.stop();
                    Err(if error.is_panic() {
                        EstimatorError::WorkerPanicked
                    } else {
                        EstimatorError::WorkerCancelled
                    })
                }
            };
            completion_sender.send_replace(Some(result));
        });
        Ok(Self {
            shared,
            observer,
            source,
            completion,
        })
    }
    pub fn observer(&self) -> Arc<dyn RequestStoppingObserver> {
        self.observer.clone()
    }
    pub fn source(&self) -> Arc<dyn StoppingSnapshotSource> {
        self.source.clone()
    }
    pub fn stats(&self) -> StoppingCollectorStats {
        self.shared.stats()
    }
    pub async fn shutdown(&self) -> Result<StoppingWorkerExit, EstimatorError> {
        self.shared.stop();
        let mut completion = self.completion.clone();
        loop {
            if let Some(result) = completion.borrow_and_update().clone() {
                return result;
            }
            completion
                .changed()
                .await
                .map_err(|_| EstimatorError::CompletionDisconnected)?;
        }
    }
}
impl Drop for StoppingEstimatorOwner {
    fn drop(&mut self) {
        self.shared.stop();
    }
}

#[cfg(test)]
mod tests;
