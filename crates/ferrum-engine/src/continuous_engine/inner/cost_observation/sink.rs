use super::checkpoint::{
    CheckpointError, CheckpointRequestError, CostCheckpointWaiter, FrozenCostCheckpoint,
    PendingCheckpoint,
};
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::WaveCostObservation;
use parking_lot::Mutex;
use std::{
    collections::VecDeque,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    sync::OnceLock,
};

#[derive(Debug, Clone, Copy)]
pub(in crate::continuous_engine) struct CostSampleSinkLimits {
    pub max_samples: usize,
    pub max_shape_rows: usize,
}
impl Default for CostSampleSinkLimits {
    fn default() -> Self {
        Self {
            max_samples: 256,
            max_shape_rows: 8192,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::continuous_engine) enum CostSampleDrop {
    Capacity,
    Contended,
}

/// One acceptance position, with the original sample and its same-call host
/// evidence kept together. Auxiliary evidence never becomes a training sample.
#[derive(Debug)]
pub(in crate::continuous_engine) enum CostEvidenceEntry {
    Training {
        sample: WaveCostObservation,
        stages: Option<Arc<HostStageEvidenceV1>>,
    },
    StagesOnly {
        stages: Arc<HostStageEvidenceV1>,
        legacy_rejection: CostCallRejection,
    },
}
impl CostEvidenceEntry {
    fn sample(&self) -> Option<&WaveCostObservation> {
        match self {
            Self::Training { sample, .. } => Some(sample),
            Self::StagesOnly { .. } => None,
        }
    }
    fn stages(&self) -> Option<&HostStageEvidenceV1> {
        match self {
            Self::Training { stages, .. } => stages.as_deref(),
            Self::StagesOnly { stages, .. } => Some(stages),
        }
    }
    fn retained_rows(&self) -> Option<usize> {
        let mut rows = 0usize;
        if let Some(sample) = self.sample() {
            rows = sample
                .actual_shape
                .decode_kv_tokens
                .capacity()
                .checked_add(sample.actual_shape.prefill_chunks.capacity())?
                .checked_add(
                    sample
                        .actual_shape
                        .numeric_features
                        .as_ref()
                        .map_or(0, |features| features.rows.capacity()),
                )?
                .checked_add(
                    sample
                        .actual_shape
                        .row_multiset_features
                        .as_ref()
                        .map_or(0, |features| features.rows.capacity()),
                )?;
        }
        if let Some(stages) = self.stages() {
            rows = rows.checked_add(stages.retained_rows()?)?;
        }
        Some(rows)
    }
    fn retained_bytes(&self, rows: usize) -> Option<usize> {
        // Every variable allocation is a row Vec; charge its full capacity at
        // the largest row size. Shared Arcs are conservatively charged per entry.
        let row_bytes = std::mem::size_of::<HostRowStageV1>()
            .max(std::mem::size_of::<
                ferrum_interfaces::execution_cost::StructuredHostRowV1,
            >())
            .max(std::mem::size_of::<CostRowNumericFeatures>())
            .max(std::mem::size_of::<HostRowStaticCostFeaturesV2>())
            .max(std::mem::size_of::<
                ferrum_scheduler::implementations::continuous::cost_model::PrefillShape,
            >());
        rows.checked_mul(row_bytes)?
            .checked_add(
                self.stages()
                    .map_or(0, |stages| stages.structured_retained_overhead_bytes()),
            )?
            .checked_add(std::mem::size_of::<Self>())?
            .checked_add(if self.stages().is_some() {
                std::mem::size_of::<HostStageEvidenceV1>() + 2 * std::mem::size_of::<usize>()
            } else {
                0
            })
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub(in crate::continuous_engine) struct CostSampleStats {
    pub preparations_started: u64,
    pub preparations_abandoned: u64,
    pub preparation_rejected: [u64; CostCallRejection::COUNT],
    pub initialization_rejected: [u64; CostCallRejection::COUNT],
    pub calls_started: u64,
    pub calls_finished: u64,
    /// Complete observations offered to the queue, before either drop gate.
    pub offered: u64,
    pub offered_completed: u64,
    pub offered_by_wave: [u64; audit::WAVE_COUNT],
    pub published: u64,
    pub drained: u64,
    pub dropped_capacity: u64,
    pub dropped_contention: u64,
    /// All FIFO entries, including StagesOnly. Legacy fields above keep the
    /// original training-observation denominator.
    pub entries_offered: u64,
    pub entries_published: u64,
    pub entries_drained: u64,
    pub entries_dropped_capacity: u64,
    pub entries_dropped_contention: u64,
    pub host_stages_offered: u64,
    pub host_stages_published: u64,
    pub host_stages_drained: u64,
    /// Terminal call rejections only. Preparation/sequence initialization
    /// failures have separate populations and must not inflate physical waves.
    pub rejected: [u64; CostCallRejection::COUNT],
    pub counter_exhausted: bool,
}
impl CostSampleStats {
    #[cfg(test)]
    pub fn rejected(&self, reason: CostCallRejection) -> u64 {
        self.rejected[reason.index()]
    }
    pub fn has_lost_samples(&self) -> bool {
        self.dropped_capacity > 0
            || self.dropped_contention > 0
            || self.entries_dropped_capacity > 0
            || self.entries_dropped_contention > 0
            || self.counter_exhausted
    }
}
struct Queue {
    samples: VecDeque<CostEvidenceEntry>,
    rows: usize,
    bytes: usize,
    accepted: u64,
    popped: u64,
    checkpoint: Option<PendingCheckpoint>,
    checkpoints_closed: bool,
}

/// The execution side never waits for a slow trainer or file writer. Unknown
/// and dropped evidence are counted separately from successfully retained data.
pub(in crate::continuous_engine) struct BoundedCostSampleSink {
    limits: CostSampleSinkLimits,
    queue: Mutex<Queue>,
    preparations_started: AtomicU64,
    preparations_abandoned: AtomicU64,
    preparation_rejected: [AtomicU64; CostCallRejection::COUNT],
    initialization_rejected: [AtomicU64; CostCallRejection::COUNT],
    calls_started: AtomicU64,
    calls_finished: AtomicU64,
    offered: AtomicU64,
    offered_completed: AtomicU64,
    offered_by_wave: [AtomicU64; audit::WAVE_COUNT],
    published: AtomicU64,
    drained: AtomicU64,
    dropped_capacity: AtomicU64,
    dropped_contention: AtomicU64,
    entries_offered: AtomicU64,
    entries_published: AtomicU64,
    entries_drained: AtomicU64,
    entries_dropped_capacity: AtomicU64,
    entries_dropped_contention: AtomicU64,
    host_stages_offered: AtomicU64,
    host_stages_published: AtomicU64,
    host_stages_drained: AtomicU64,
    rejected: [AtomicU64; CostCallRejection::COUNT],
    counter_exhausted: AtomicBool,
    checkpoint_pending: AtomicBool,
    worker: OnceLock<std::thread::Thread>,
    #[cfg(test)]
    after_pop: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}
impl BoundedCostSampleSink {
    pub fn new(limits: CostSampleSinkLimits) -> Result<Self, CostCallRejection> {
        if limits.max_samples == 0
            || limits.max_samples > 4096
            || limits.max_shape_rows == 0
            || limits.max_shape_rows > 65_536
        {
            return Err(CostCallRejection::RecorderCapacity);
        }
        let mut samples = VecDeque::new();
        samples
            .try_reserve_exact(limits.max_samples)
            .map_err(|_| CostCallRejection::RecorderCapacity)?;
        Ok(Self {
            limits,
            queue: Mutex::new(Queue {
                samples,
                rows: 0,
                bytes: 0,
                accepted: 0,
                popped: 0,
                checkpoint: None,
                checkpoints_closed: false,
            }),
            preparations_started: AtomicU64::new(0),
            preparations_abandoned: AtomicU64::new(0),
            preparation_rejected: std::array::from_fn(|_| AtomicU64::new(0)),
            initialization_rejected: std::array::from_fn(|_| AtomicU64::new(0)),
            calls_started: AtomicU64::new(0),
            calls_finished: AtomicU64::new(0),
            offered: AtomicU64::new(0),
            offered_completed: AtomicU64::new(0),
            offered_by_wave: std::array::from_fn(|_| AtomicU64::new(0)),
            published: AtomicU64::new(0),
            drained: AtomicU64::new(0),
            dropped_capacity: AtomicU64::new(0),
            dropped_contention: AtomicU64::new(0),
            entries_offered: AtomicU64::new(0),
            entries_published: AtomicU64::new(0),
            entries_drained: AtomicU64::new(0),
            entries_dropped_capacity: AtomicU64::new(0),
            entries_dropped_contention: AtomicU64::new(0),
            host_stages_offered: AtomicU64::new(0),
            host_stages_published: AtomicU64::new(0),
            host_stages_drained: AtomicU64::new(0),
            rejected: std::array::from_fn(|_| AtomicU64::new(0)),
            counter_exhausted: AtomicBool::new(false),
            checkpoint_pending: AtomicBool::new(false),
            worker: OnceLock::new(),
            #[cfg(test)]
            after_pop: Mutex::new(None),
        })
    }
    pub(super) fn attach_worker(&self, worker: std::thread::Thread) -> ferrum_types::Result<()> {
        self.worker.set(worker).map_err(|_| {
            ferrum_types::FerrumError::internal("cost sample sink already has a training worker")
        })
    }
    pub(super) fn reject(&self, reason: CostCallRejection) {
        self.increment(&self.rejected[reason.index()]);
        self.call_finished();
    }
    fn increment(&self, counter: &AtomicU64) {
        audit::increment(counter, &self.counter_exhausted);
    }
    pub(super) fn preparation_started(&self) {
        self.increment(&self.preparations_started);
    }
    pub(super) fn preparation_abandoned(&self) {
        self.increment(&self.preparations_abandoned);
    }
    pub(super) fn reject_preparation(&self, reason: CostCallRejection) {
        self.increment(&self.preparation_rejected[reason.index()]);
    }
    pub(super) fn reject_initialization(&self, reason: CostCallRejection) {
        self.increment(&self.initialization_rejected[reason.index()]);
    }
    pub(super) fn call_started(&self) {
        self.increment(&self.calls_started);
    }
    pub(super) fn call_finished(&self) {
        self.increment(&self.calls_finished);
    }
    pub(super) fn maximum_stats() -> CostSampleStats {
        CostSampleStats {
            preparations_started: u64::MAX,
            preparations_abandoned: u64::MAX,
            preparation_rejected: [u64::MAX; CostCallRejection::COUNT],
            initialization_rejected: [u64::MAX; CostCallRejection::COUNT],
            calls_started: u64::MAX,
            calls_finished: u64::MAX,
            offered: u64::MAX,
            offered_completed: u64::MAX,
            offered_by_wave: [u64::MAX; audit::WAVE_COUNT],
            published: u64::MAX,
            drained: u64::MAX,
            dropped_capacity: u64::MAX,
            dropped_contention: u64::MAX,
            entries_offered: u64::MAX,
            entries_published: u64::MAX,
            entries_drained: u64::MAX,
            entries_dropped_capacity: u64::MAX,
            entries_dropped_contention: u64::MAX,
            host_stages_offered: u64::MAX,
            host_stages_published: u64::MAX,
            host_stages_drained: u64::MAX,
            rejected: [u64::MAX; CostCallRejection::COUNT],
            counter_exhausted: false, // Longer JSON spelling for footer sizing.
        }
    }
    #[cfg(test)]
    pub(super) fn offer(&self, sample: WaveCostObservation) -> Result<(), CostSampleDrop> {
        self.offer_numbered(sample).map(|_| ())
    }
    #[cfg(test)]
    pub(super) fn offer_numbered(
        &self,
        sample: WaveCostObservation,
    ) -> Result<u64, CostSampleDrop> {
        self.offer_evidence_numbered(CostEvidenceEntry::Training {
            sample,
            stages: None,
        })
    }
    pub(super) fn offer_evidence_numbered(
        &self,
        entry: CostEvidenceEntry,
    ) -> Result<u64, CostSampleDrop> {
        self.increment(&self.entries_offered);
        let is_training = entry.sample().is_some();
        let has_stages = entry.stages().is_some();
        if has_stages {
            self.increment(&self.host_stages_offered);
        }
        if let Some(sample) = entry.sample() {
            self.increment(&self.offered);
            self.increment(&self.offered_by_wave[audit::wave_index(sample.actual_shape.kind)]);
            if sample.outcome == ferrum_scheduler::implementations::continuous::cost_model::WaveObservationOutcome::Completed {
                self.increment(&self.offered_completed);
            }
        }
        // Capacity includes spare allocation, not just logical vector lengths.
        let rows = entry.retained_rows();
        let bytes = rows.and_then(|rows| entry.retained_bytes(rows));
        let capacity_drop = || {
            self.increment(&self.entries_dropped_capacity);
            if is_training {
                self.increment(&self.dropped_capacity);
            }
            CostSampleDrop::Capacity
        };
        let Some(mut queue) = self.queue.try_lock() else {
            self.increment(&self.entries_dropped_contention);
            if is_training {
                self.increment(&self.dropped_contention);
            }
            return Err(CostSampleDrop::Contended);
        };
        let Some(ordinal) = queue.accepted.checked_add(1) else {
            self.counter_exhausted.store(true, Ordering::Relaxed);
            return Err(capacity_drop());
        };
        let Some(total) = rows
            .and_then(|rows| queue.rows.checked_add(rows))
            .filter(|total| *total <= self.limits.max_shape_rows)
            .filter(|_| queue.samples.len() < self.limits.max_samples)
        else {
            return Err(capacity_drop());
        };
        let Some(total_bytes) = bytes
            .and_then(|bytes| queue.bytes.checked_add(bytes))
            .filter(|bytes| *bytes <= 128 * 1024 * 1024)
        else {
            return Err(capacity_drop());
        };
        queue.rows = total;
        queue.bytes = total_bytes;
        queue.samples.push_back(entry);
        queue.accepted = ordinal;
        self.increment(&self.entries_published);
        if is_training {
            self.increment(&self.published);
        }
        if has_stages {
            self.increment(&self.host_stages_published);
        }
        drop(queue);
        // Publication itself owns the wake, including when an enclosing
        // multi-leaf batch is cancelled before its final notification.
        if let Some(worker) = self.worker.get() {
            worker.unpark();
        }
        Ok(ordinal)
    }
    /// Consumer slow path. Preserve the original receipt timestamp; draining
    /// late must not refresh old measurements. Feed these before publishing a
    /// newer trainer clock, or explicitly account for ClockMovedBackwards.
    /// Imported trainers map this timestamp through observe_live's local clock.
    pub(super) fn pop_numbered(&self) -> Option<(u64, CostEvidenceEntry)> {
        let mut queue = self.queue.lock();
        if queue
            .checkpoint
            .as_ref()
            .is_some_and(|checkpoint| queue.popped >= checkpoint.cutoff)
        {
            return None;
        }
        let entry = queue.samples.pop_front()?;
        let rows = entry.retained_rows().expect("validated queue capacity");
        queue.rows -= rows;
        queue.bytes -= entry
            .retained_bytes(rows)
            .expect("validated queue byte capacity");
        self.increment(&self.entries_drained);
        if entry.sample().is_some() {
            self.increment(&self.drained);
        }
        if entry.stages().is_some() {
            self.increment(&self.host_stages_drained);
        }
        queue.popped += 1; // bounded by the checked accepted ordinal
        let ordinal = queue.popped;
        drop(queue);
        #[cfg(test)]
        if let Some(hook) = self.after_pop.lock().take() {
            hook();
        }
        Some((ordinal, entry))
    }
    #[cfg(test)]
    pub fn pop(&self) -> Option<WaveCostObservation> {
        self.pop_numbered().and_then(|(_, entry)| match entry {
            CostEvidenceEntry::Training { sample, .. } => Some(sample),
            CostEvidenceEntry::StagesOnly { .. } => None,
        })
    }
    #[cfg(test)]
    pub(super) fn on_next_pop(&self, hook: impl FnOnce() + Send + 'static) {
        *self.after_pop.lock() = Some(Box::new(hook));
    }

    pub(super) fn request_checkpoint(
        &self,
    ) -> Result<CostCheckpointWaiter, CheckpointRequestError> {
        self.request_checkpoint_with_export(None)
    }

    pub(super) fn request_checkpoint_with_export(
        &self,
        paths: Option<super::profile_export::CostProfileCutPaths>,
    ) -> Result<CostCheckpointWaiter, CheckpointRequestError> {
        let mut queue = self.queue.lock();
        if queue.checkpoints_closed {
            return Err(CheckpointRequestError::Closing);
        }
        if queue.checkpoint.is_some() {
            return Err(CheckpointRequestError::Busy);
        }
        if self.counter_exhausted.load(Ordering::Relaxed) {
            return Err(CheckpointRequestError::CounterExhausted);
        }
        let (pending, waiter) = PendingCheckpoint::new(queue.accepted, paths);
        queue.checkpoint = Some(pending);
        self.checkpoint_pending.store(true, Ordering::Release);
        drop(queue);
        if let Some(worker) = self.worker.get() {
            worker.unpark();
        }
        Ok(waiter)
    }

    /// `processed` advances after observe/export/audit, not when pop removes a
    /// sample. Thus an in-flight last sample cannot slip past an empty queue cut.
    pub(super) fn begin_checkpoint(
        &self,
        processed: u64,
    ) -> Option<(u64, Option<super::profile_export::CostProfileCutPaths>)> {
        if !self.checkpoint_pending.load(Ordering::Acquire) {
            return None;
        }
        let mut queue = self.queue.lock();
        let pending = queue.checkpoint.as_mut()?;
        if pending.freezing || pending.cutoff != processed {
            return None;
        }
        pending.freezing = true;
        Some((pending.cutoff, pending.export_paths.take()))
    }

    pub(super) fn complete_checkpoint(&self, result: FrozenCostCheckpoint) {
        let mut queue = self.queue.lock();
        let matches = queue
            .checkpoint
            .as_ref()
            .is_some_and(|pending| pending.freezing && pending.cutoff == result.accepted_ordinal);
        if matches {
            let pending = queue.checkpoint.take().expect("matched control slot");
            self.checkpoint_pending.store(false, Ordering::Release);
            drop(queue);
            pending.complete(Ok(result));
        }
    }

    pub(super) fn close_checkpoints(&self) {
        self.queue.lock().checkpoints_closed = true;
    }

    pub(super) fn needs_drain(&self, processed: u64) -> bool {
        let queue = self.queue.lock();
        queue.accepted > processed || queue.checkpoint.is_some()
    }

    pub(super) fn worker_stopped(&self) {
        let mut queue = self.queue.lock();
        queue.checkpoints_closed = true;
        let pending = queue.checkpoint.take();
        self.checkpoint_pending.store(false, Ordering::Release);
        drop(queue);
        if let Some(pending) = pending {
            pending.complete(Err(CheckpointError::WorkerStopped));
        }
    }
    #[cfg(test)]
    pub(super) fn with_locked_queue<T>(&self, action: impl FnOnce() -> T) -> T {
        let _guard = self.queue.lock();
        action()
    }

    pub fn stats(&self) -> CostSampleStats {
        CostSampleStats {
            preparations_started: self.preparations_started.load(Ordering::Relaxed),
            preparations_abandoned: self.preparations_abandoned.load(Ordering::Relaxed),
            preparation_rejected: std::array::from_fn(|i| {
                self.preparation_rejected[i].load(Ordering::Relaxed)
            }),
            initialization_rejected: std::array::from_fn(|i| {
                self.initialization_rejected[i].load(Ordering::Relaxed)
            }),
            calls_started: self.calls_started.load(Ordering::Relaxed),
            calls_finished: self.calls_finished.load(Ordering::Relaxed),
            offered: self.offered.load(Ordering::Relaxed),
            offered_completed: self.offered_completed.load(Ordering::Relaxed),
            offered_by_wave: std::array::from_fn(|i| {
                self.offered_by_wave[i].load(Ordering::Relaxed)
            }),
            published: self.published.load(Ordering::Relaxed),
            drained: self.drained.load(Ordering::Relaxed),
            dropped_capacity: self.dropped_capacity.load(Ordering::Relaxed),
            dropped_contention: self.dropped_contention.load(Ordering::Relaxed),
            entries_offered: self.entries_offered.load(Ordering::Relaxed),
            entries_published: self.entries_published.load(Ordering::Relaxed),
            entries_drained: self.entries_drained.load(Ordering::Relaxed),
            entries_dropped_capacity: self.entries_dropped_capacity.load(Ordering::Relaxed),
            entries_dropped_contention: self.entries_dropped_contention.load(Ordering::Relaxed),
            host_stages_offered: self.host_stages_offered.load(Ordering::Relaxed),
            host_stages_published: self.host_stages_published.load(Ordering::Relaxed),
            host_stages_drained: self.host_stages_drained.load(Ordering::Relaxed),
            rejected: std::array::from_fn(|index| self.rejected[index].load(Ordering::Relaxed)),
            counter_exhausted: self.counter_exhausted.load(Ordering::Relaxed),
        }
    }
}
