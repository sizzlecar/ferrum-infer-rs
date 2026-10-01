use super::checkpoint::{
    CheckpointError, CheckpointRequestError, CostCheckpointWaiter, FrozenCostCheckpoint,
    PendingCheckpoint,
};
use super::memory::{
    CaptureCapacityFailure, ObservationBytePermit, ObservationBytePool, ObservationMemoryStats,
};
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::WaveCostObservation;
use parking_lot::Mutex;
mod handoff;
use handoff::{QueuedPublication, QueuedRowBudget, QueuedRows};
use std::sync::atomic::AtomicUsize;
use std::{
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
    WorkerStopped,
}

struct RawResolutionGuard<'a> {
    sink: &'a BoundedCostSampleSink,
    completed: bool,
}
impl Drop for RawResolutionGuard<'_> {
    fn drop(&mut self) {
        if !self.completed {
            self.sink.increment(&self.sink.raw_lost);
            self.sink.increment(&self.sink.raw_abandoned);
        }
    }
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
    pub(super) fn retained_rows(&self) -> Option<usize> {
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
    pub(super) fn retained_bytes(&self, rows: usize) -> Option<usize> {
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
        let route_bytes = match self.stages().and_then(|s| s.route_evidence.as_ref()) {
            Some(route) => route.retained_bytes()?,
            None => 0,
        };
        rows.checked_mul(row_bytes)?
            .checked_add(route_bytes)?
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
    pub memory: ObservationMemoryStats,
    pub raw_lost: u64,
    pub raw_abandoned: u64,
    pub raw_resolve_ns_total: u64,
    pub raw_resolve_ns_max: u64,
    pub raw_queue_wait_ns_total: u64,
    pub raw_queue_wait_ns_max: u64,
    pub raw_pending: u64,
    pub raw_offered: u64,
    pub raw_accepted: u64,
    pub raw_resolved: u64,
    pub raw_resolution_failed: u64,
    pub raw_no_submission: u64,
    pub entries_offered: u64,
    pub entries_published: u64,
    pub entries_drained: u64,
    pub entries_abandoned: u64,
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
        self.raw_lost > 0
            || self.dropped_capacity > 0
            || self.dropped_contention > 0
            || self.entries_abandoned > 0
            || self.entries_dropped_capacity > 0
            || self.entries_dropped_contention > 0
            || self.counter_exhausted
    }
}
enum QueuedCostPayload {
    Ready(CostEvidenceEntry),
    Raw(super::sealed::SealedCostCall),
    Prefix(super::prefix::PrefixSample),
}

pub(super) type ResolvedInferenceInput = (
    u64,
    Option<super::resolved::ResolvedCostEntry>,
    Option<super::live_calibration::Ticket>,
    u64,
    Option<Arc<live_calibration::OriginalNoSubmissionReceipt>>,
);
pub(super) enum CostTrainingInput {
    Inference(ResolvedInferenceInput),
    Prefix {
        ordinal: u64,
        sample: super::prefix::PrefixSample,
        memory: ObservationBytePermit,
    },
    PrefixDiscarded {
        ordinal: u64,
    },
}
struct QueueEntry {
    payload: QueuedCostPayload,
    ticket: Option<super::live_calibration::Ticket>,
    source_generation: u64,
    rows: usize,
    bytes: usize,
    memory: ObservationBytePermit,
    queued_rows: QueuedRows,
    publication_completed: AtomicBool,
}
struct ConsumerControl {
    popped: u64,
    checkpoint: Option<PendingCheckpoint>,
    checkpoints_closed: bool,
}

/// The execution side never waits for a slow trainer or file writer. Unknown
/// and dropped evidence are counted separately from successfully retained data.
pub(in crate::continuous_engine) struct BoundedCostSampleSink {
    byte_limits: CostRecorderByteLimits,
    raw_memory: Arc<ObservationBytePool>,
    working_memory: Arc<ObservationBytePool>,
    requested_rows_peak: AtomicUsize,
    requested_raw_bytes_peak: AtomicUsize,
    requested_working_bytes_peak: AtomicUsize,
    accepted_raw_bytes_peak: AtomicUsize,
    accepted_working_bytes_peak: AtomicUsize,
    first_capture_rejection: OnceLock<CaptureCapacityFailure>,
    workload_domain: OnceLock<CostWorkloadDomainV1>,
    source_generation: AtomicU64,
    feedback_population: Option<ferrum_types::SloCalibrationRoutePopulationV1>,
    // Only producers serialize admission. Worker/control never take this lock
    // except the one terminal stop barrier after producers have been closed.
    admission: Mutex<()>,
    sender: crossbeam_channel::Sender<Arc<QueueEntry>>,
    receiver: crossbeam_channel::Receiver<Arc<QueueEntry>>,
    queued_rows: Arc<QueuedRowBudget>,
    accepted: AtomicU64,
    consumer: Mutex<ConsumerControl>,
    worker_stopped: AtomicBool,
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
    raw_lost: AtomicU64,
    raw_abandoned: AtomicU64,
    raw_resolve_ns_total: AtomicU64,
    raw_resolve_ns_max: AtomicU64,
    raw_queue_wait_ns_total: AtomicU64,
    raw_queue_wait_ns_max: AtomicU64,
    raw_offered: AtomicU64,
    raw_accepted: AtomicU64,
    raw_resolved: AtomicU64,
    raw_resolution_failed: AtomicU64,
    raw_no_submission: AtomicU64,
    entries_offered: AtomicU64,
    entries_published: AtomicU64,
    entries_drained: AtomicU64,
    entries_abandoned: AtomicU64,
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
    #[cfg(test)]
    after_send: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}
impl BoundedCostSampleSink {
    pub(super) fn reserve_prefix_working_bytes(
        &self,
        bytes: usize,
    ) -> Option<ObservationBytePermit> {
        self.working_memory.reserve(bytes)
    }

    pub(super) fn offer_prefix(
        &self,
        sample: super::prefix::PrefixSample,
    ) -> Result<u64, CostSampleDrop> {
        self.offer_payload(QueuedCostPayload::Prefix(sample), None, 0)
    }
    /// Optional cold archival shares the existing worker/result byte ceiling.
    /// A failed reservation does not enter the sample queue or reject a call.
    pub(super) fn reserve_diagnostic_bytes(&self, bytes: usize) -> Option<ObservationBytePermit> {
        self.working_memory.reserve(bytes)
    }

    pub(super) fn workload_domain(&self) -> Option<&CostWorkloadDomainV1> {
        self.workload_domain.get()
    }
    pub(super) fn install_workload_domain(
        &self,
        domain: CostWorkloadDomainV1,
    ) -> Result<(), ferrum_types::FerrumError> {
        self.workload_domain.set(domain).map_err(|_| {
            ferrum_types::FerrumError::config(
                "observation workload domain is immutable after construction",
            )
        })
    }
    pub(super) fn with_feedback_population(
        mut self,
        population: Option<ferrum_types::SloCalibrationRoutePopulationV1>,
    ) -> Self {
        self.feedback_population = population;
        self
    }
    pub(super) fn feedback_population(
        &self,
    ) -> Option<ferrum_types::SloCalibrationRoutePopulationV1> {
        self.feedback_population
    }
    pub(super) fn source_generation(&self) -> u64 {
        self.source_generation.load(Ordering::Acquire)
    }
    pub(super) fn set_source_generation(&self, value: u64) {
        self.source_generation.store(value, Ordering::Release);
    }
    pub fn new(limits: CostSampleSinkLimits) -> Result<Self, CostCallRejection> {
        Self::new_with_byte_limits(limits, CostRecorderByteLimits::default())
    }
    pub(super) fn byte_limits(&self) -> CostRecorderByteLimits {
        self.byte_limits
    }
    pub(super) fn capture_memory(&self, call_id: u64, audit: CostRecorderMemoryAudit) {
        self.requested_rows_peak
            .fetch_max(audit.requested_rows_peak, Ordering::Relaxed);
        self.requested_raw_bytes_peak
            .fetch_max(audit.requested_raw_bytes_peak, Ordering::Relaxed);
        self.requested_working_bytes_peak
            .fetch_max(audit.requested_working_bytes_peak, Ordering::Relaxed);
        if let Some(diagnostic) = audit.first_rejection {
            let _ = self.first_capture_rejection.set(CaptureCapacityFailure {
                call_id,
                gate: "call_capture",
                diagnostic,
            });
        }
    }
    fn memory_rejection(
        &self,
        call_id: u64,
        gate: &'static str,
        resource: CostRecorderCapacityResource,
        requested: Option<usize>,
        limit: usize,
    ) {
        let _ = self.first_capture_rejection.set(CaptureCapacityFailure {
            call_id,
            gate,
            diagnostic: CostRecorderCapacityDiagnostic {
                resource,
                requested,
                limit,
            },
        });
    }
    pub(super) fn new_with_byte_limits(
        limits: CostSampleSinkLimits,
        byte_limits: CostRecorderByteLimits,
    ) -> Result<Self, CostCallRejection> {
        byte_limits
            .validate()
            .map_err(|_| CostCallRejection::RecorderCapacity)?;
        if limits.max_samples == 0
            || limits.max_samples > 4096
            || limits.max_shape_rows == 0
            || limits.max_shape_rows > 65_536
        {
            return Err(CostCallRejection::RecorderCapacity);
        }
        let (sender, receiver) = crossbeam_channel::bounded(limits.max_samples);
        Ok(Self {
            byte_limits,
            raw_memory: ObservationBytePool::new(byte_limits.maximum_retained_bytes),
            working_memory: ObservationBytePool::new(byte_limits.maximum_working_bytes),
            requested_rows_peak: AtomicUsize::new(0),
            requested_raw_bytes_peak: AtomicUsize::new(0),
            requested_working_bytes_peak: AtomicUsize::new(0),
            accepted_raw_bytes_peak: AtomicUsize::new(0),
            accepted_working_bytes_peak: AtomicUsize::new(0),
            first_capture_rejection: OnceLock::new(),
            workload_domain: OnceLock::new(),
            source_generation: AtomicU64::new(0),
            feedback_population: None,
            admission: Mutex::new(()),
            sender,
            receiver,
            queued_rows: QueuedRowBudget::new(limits.max_shape_rows),
            accepted: AtomicU64::new(0),
            consumer: Mutex::new(ConsumerControl {
                popped: 0,
                checkpoint: None,
                checkpoints_closed: false,
            }),
            worker_stopped: AtomicBool::new(false),
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
            raw_lost: AtomicU64::new(0),
            raw_abandoned: AtomicU64::new(0),
            raw_resolve_ns_total: AtomicU64::new(0),
            raw_resolve_ns_max: AtomicU64::new(0),
            raw_queue_wait_ns_total: AtomicU64::new(0),
            raw_queue_wait_ns_max: AtomicU64::new(0),
            raw_offered: AtomicU64::new(0),
            raw_accepted: AtomicU64::new(0),
            raw_resolved: AtomicU64::new(0),
            raw_resolution_failed: AtomicU64::new(0),
            raw_no_submission: AtomicU64::new(0),
            entries_offered: AtomicU64::new(0),
            entries_published: AtomicU64::new(0),
            entries_drained: AtomicU64::new(0),
            entries_abandoned: AtomicU64::new(0),
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
            #[cfg(test)]
            after_send: Mutex::new(None),
        })
    }
    pub(super) fn attach_worker(&self, worker: std::thread::Thread) -> ferrum_types::Result<()> {
        self.worker.set(worker).map_err(|_| {
            ferrum_types::FerrumError::internal("cost sample sink already has a training worker")
        })
    }
    pub(super) fn reject_resolved(&self, reason: CostCallRejection) {
        self.increment(&self.rejected[reason.index()]);
    }
    pub(super) fn reject(&self, reason: CostCallRejection) {
        self.increment(&self.rejected[reason.index()]);
        self.call_finished();
    }
    fn add_duration(&self, total: &AtomicU64, maximum: &AtomicU64, ns: u64) {
        if total
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(ns))
            .is_err()
        {
            self.counter_exhausted.store(true, Ordering::Relaxed);
        }
        maximum.fetch_max(ns, Ordering::Relaxed);
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
            memory: ObservationMemoryStats::maximum(),
            raw_lost: u64::MAX,
            raw_abandoned: u64::MAX,
            raw_resolve_ns_total: u64::MAX,
            raw_resolve_ns_max: u64::MAX,
            raw_queue_wait_ns_total: u64::MAX,
            raw_queue_wait_ns_max: u64::MAX,
            raw_pending: u64::MAX,
            raw_offered: u64::MAX,
            raw_accepted: u64::MAX,
            raw_resolved: u64::MAX,
            raw_resolution_failed: u64::MAX,
            raw_no_submission: u64::MAX,
            entries_offered: u64::MAX,
            entries_published: u64::MAX,
            entries_drained: u64::MAX,
            entries_abandoned: u64::MAX,
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
    #[cfg(test)]
    pub(super) fn offer_evidence_numbered(
        &self,
        entry: CostEvidenceEntry,
    ) -> Result<u64, CostSampleDrop> {
        self.offer_evidence_with_ticket(entry, None)
    }
    #[cfg(test)]
    pub(super) fn offer_evidence_with_ticket(
        &self,
        entry: CostEvidenceEntry,
        ticket: Option<super::live_calibration::Ticket>,
    ) -> Result<u64, CostSampleDrop> {
        self.offer_evidence_bound(entry, ticket, 0)
    }
    pub(super) fn offer_evidence_bound(
        &self,
        entry: CostEvidenceEntry,
        ticket: Option<super::live_calibration::Ticket>,
        source_generation: u64,
    ) -> Result<u64, CostSampleDrop> {
        self.offer_payload(QueuedCostPayload::Ready(entry), ticket, source_generation)
    }
    pub(super) fn offer_raw(
        &self,
        raw: super::sealed::SealedCostCall,
    ) -> Result<u64, CostSampleDrop> {
        self.increment(&self.raw_offered);
        self.requested_raw_bytes_peak
            .fetch_max(raw.retained_bytes, Ordering::Relaxed);
        self.requested_working_bytes_peak
            .fetch_max(raw.working_bytes, Ordering::Relaxed);
        let generation = raw.source_generation();
        self.offer_payload(QueuedCostPayload::Raw(raw), None, generation)
    }
    fn offer_payload(
        &self,
        payload: QueuedCostPayload,
        ticket: Option<super::live_calibration::Ticket>,
        source_generation: u64,
    ) -> Result<u64, CostSampleDrop> {
        // Maintenance owns the same FIFO positions/capacity, with a separate
        // statistical population. Its losses never invalidate inference.
        let is_prefix = matches!(&payload, QueuedCostPayload::Prefix(_));
        if !is_prefix {
            self.increment(&self.entries_offered);
        }
        let entry = match &payload {
            QueuedCostPayload::Ready(entry) => Some(entry),
            _ => None,
        };
        let is_raw = matches!(&payload, QueuedCostPayload::Raw(_));
        let is_training = entry.and_then(|e| e.sample()).is_some();
        let has_stages = entry.and_then(|e| e.stages()).is_some();
        if has_stages {
            self.increment(&self.host_stages_offered);
        }
        if let Some(sample) = entry.and_then(|e| e.sample()) {
            self.increment(&self.offered);
            self.increment(&self.offered_by_wave[audit::wave_index(sample.actual_shape.kind)]);
            if sample.outcome == ferrum_scheduler::implementations::continuous::cost_model::WaveObservationOutcome::Completed {
                self.increment(&self.offered_completed);
            }
        }
        // Capacity includes spare allocation, not just logical vector lengths.
        let rows = match &payload {
            QueuedCostPayload::Ready(entry) => entry.retained_rows(),
            QueuedCostPayload::Raw(raw) => Some(raw.retained_rows),
            QueuedCostPayload::Prefix(_) => Some(1),
        };
        let bytes = rows
            .and_then(|rows| match &payload {
                QueuedCostPayload::Ready(entry) => entry.retained_bytes(rows),
                QueuedCostPayload::Raw(raw) => Some(raw.retained_bytes),
                QueuedCostPayload::Prefix(_) => {
                    Some(std::mem::size_of::<super::prefix::PrefixSample>())
                }
            })
            // Charge the complete heap envelope (including union slack,
            // ticket, permits and Arc counters), conservatively in addition
            // to the payload's independently measured retained allocation.
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<QueueEntry>()))
            .and_then(|bytes| bytes.checked_add(2 * std::mem::size_of::<usize>()));
        if self.worker_stopped.load(Ordering::Acquire) {
            return Err(self.drop_payload(&payload, CostSampleDrop::WorkerStopped));
        }
        let Some(admission) = self.admission.try_lock() else {
            return Err(self.drop_payload(&payload, CostSampleDrop::Contended));
        };
        if self.worker_stopped.load(Ordering::Acquire) {
            return Err(self.drop_payload(&payload, CostSampleDrop::WorkerStopped));
        }
        let Some(ordinal) = self.accepted.load(Ordering::Acquire).checked_add(1) else {
            self.counter_exhausted.store(true, Ordering::Relaxed);
            return Err(self.drop_payload(&payload, CostSampleDrop::Capacity));
        };
        let Some(queued_rows) = rows.and_then(|rows| self.queued_rows.reserve(rows)) else {
            return Err(self.drop_payload(&payload, CostSampleDrop::Capacity));
        };
        let Some(memory) = bytes.and_then(|bytes| self.raw_memory.reserve(bytes)) else {
            if let QueuedCostPayload::Raw(raw) = &payload {
                self.memory_rejection(
                    raw.call_id(),
                    "queue_raw_bytes",
                    CostRecorderCapacityResource::RawBytes,
                    bytes.and_then(|bytes| self.raw_memory.retained().checked_add(bytes)),
                    self.byte_limits.maximum_retained_bytes,
                );
            }
            return Err(self.drop_payload(&payload, CostSampleDrop::Capacity));
        };
        let queued = Arc::new(QueueEntry {
            payload,
            ticket,
            source_generation,
            rows: rows.expect("validated queue rows"),
            bytes: bytes.expect("validated queue bytes"),
            memory,
            queued_rows,
            publication_completed: AtomicBool::new(false),
        });
        // A bounded channel may reject capacity before any source ticket is
        // accepted. The receiver cannot access this entry until the committed
        // watermark advances, even though its channel slot is already filled.
        if let Err(error) = self.sender.try_send(Arc::clone(&queued)) {
            let reason = match error {
                crossbeam_channel::TrySendError::Full(_) => CostSampleDrop::Capacity,
                crossbeam_channel::TrySendError::Disconnected(_) => CostSampleDrop::WorkerStopped,
            };
            return Err(self.drop_payload(&queued.payload, reason));
        }
        let publication = QueuedPublication::new(self, ordinal, queued);
        if !is_prefix {
            self.increment(&self.entries_published);
        }
        if is_raw {
            self.increment(&self.raw_accepted);
        }
        #[cfg(test)]
        if let Some(hook) = self.after_send.lock().take() {
            hook();
        }
        let queued = publication.entry();
        let original_ticket = match &queued.payload {
            QueuedCostPayload::Raw(raw) => raw.ticket(),
            QueuedCostPayload::Ready(_) => queued.ticket.as_ref(),
            QueuedCostPayload::Prefix(_) => None,
        };
        if let Some(ticket) = original_ticket {
            ticket.accepted(ordinal);
        }
        if let QueuedCostPayload::Raw(raw) = &queued.payload {
            raw.accepted(ordinal);
        }
        self.accepted_raw_bytes_peak
            .fetch_max(queued.bytes, Ordering::Relaxed);
        if is_training {
            self.increment(&self.published);
        }
        if has_stages {
            self.increment(&self.host_stages_published);
        }
        publication.complete();
        drop(admission);
        Ok(ordinal)
    }
    fn drop_payload(&self, payload: &QueuedCostPayload, reason: CostSampleDrop) -> CostSampleDrop {
        if let QueuedCostPayload::Prefix(sample) = payload {
            sample.abandon();
            return reason;
        }
        if let QueuedCostPayload::Raw(raw) = payload {
            self.increment(&self.raw_lost);
            raw.dropped(reason);
        }
        let training = matches!(
            payload,
            QueuedCostPayload::Ready(CostEvidenceEntry::Training { .. })
        );
        match reason {
            CostSampleDrop::Capacity => {
                self.increment(&self.entries_dropped_capacity);
                if training {
                    self.increment(&self.dropped_capacity);
                }
            }
            CostSampleDrop::Contended => {
                self.increment(&self.entries_dropped_contention);
                if training {
                    self.increment(&self.dropped_contention);
                }
            }
            CostSampleDrop::WorkerStopped => {}
        }
        reason
    }

    /// Consumer slow path. Preserve the original receipt timestamp; draining
    /// late must not refresh old measurements. Feed these before publishing a
    /// newer trainer clock, or explicitly account for ClockMovedBackwards.
    /// Imported trainers map this timestamp through observe_live's local clock.
    pub(super) fn pop_numbered(&self) -> Option<(u64, CostEvidenceEntry)> {
        self.pop_numbered_with_ticket()
            .map(|(ordinal, entry, _ticket)| (ordinal, entry))
    }
    pub(super) fn pop_numbered_with_ticket(
        &self,
    ) -> Option<(
        u64,
        CostEvidenceEntry,
        Option<super::live_calibration::Ticket>,
    )> {
        self.pop_numbered_bound()
            .map(|(ordinal, entry, ticket, _generation)| (ordinal, entry, ticket))
    }
    pub(super) fn pop_numbered_bound(
        &self,
    ) -> Option<(
        u64,
        CostEvidenceEntry,
        Option<super::live_calibration::Ticket>,
        u64,
    )> {
        while let Some((ordinal, entry, ticket, generation, _proof)) = self.pop_resolved_bound() {
            if let Some(entry) = entry {
                return Some((ordinal, entry.into_entry(), ticket, generation));
            }
            drop(ticket);
        }
        None
    }
    /// Sole-worker operation: resolve outside the control lock, then retain the
    /// acceptance ordinal even if no eligible evidence can be produced.
    pub(super) fn pop_resolved_bound(&self) -> Option<ResolvedInferenceInput> {
        // Compatibility for existing inference-only diagnostic readers.
        loop {
            match self.pop_training_input()? {
                CostTrainingInput::Inference(value) => return Some(value),
                CostTrainingInput::Prefix { sample, .. } => sample.abandon(),
                CostTrainingInput::PrefixDiscarded { .. } => {}
            }
        }
    }

    pub(super) fn pop_training_input(&self) -> Option<CostTrainingInput> {
        let mut control = self.consumer.lock();
        if control
            .checkpoint
            .as_ref()
            .is_some_and(|c| control.popped >= c.cutoff)
            || control.popped >= self.accepted.load(Ordering::Acquire)
        {
            return None;
        }
        let queued = self.receiver.try_recv().ok()?;
        let queued = Arc::try_unwrap(queued)
            .unwrap_or_else(|_| unreachable!("committed queue envelope has no producer owner"));
        control.popped += 1;
        let ordinal = control.popped;
        drop(control);
        // Queue occupancy ends here; raw bytes remain held through resolution.
        drop(queued.queued_rows);
        if !matches!(&queued.payload, QueuedCostPayload::Prefix(_)) {
            self.increment(&self.entries_drained);
        }
        if !queued.publication_completed.load(Ordering::Acquire) {
            if let QueuedCostPayload::Prefix(sample) = &queued.payload {
                sample.abandon();
                return Some(CostTrainingInput::PrefixDiscarded { ordinal });
            }
            // The producer unwound after reserving the FIFO slot. Consume the
            // failed position without projecting or training its payload.
            return Some(CostTrainingInput::Inference((
                ordinal,
                None,
                queued.ticket,
                queued.source_generation,
                None,
            )));
        }
        let mut resolution =
            matches!(&queued.payload, QueuedCostPayload::Raw(_)).then(|| RawResolutionGuard {
                sink: self,
                completed: false,
            });
        #[cfg(test)]
        if let Some(hook) = self.after_pop.lock().take() {
            hook();
        }
        let raw_memory = queued.memory;
        let (entry, ticket, generation, proof, raw) = match queued.payload {
            QueuedCostPayload::Prefix(sample) => {
                return Some(CostTrainingInput::Prefix {
                    ordinal,
                    sample,
                    memory: raw_memory,
                });
            }
            QueuedCostPayload::Ready(entry) => {
                let requested = super::memory::maximum_resolution_overhead(queued.rows)
                    .and_then(|n| n.checked_add(queued.bytes));
                let resolved = match requested.and_then(|n| self.working_memory.reserve(n)) {
                    Some(memory) => {
                        let memory = Arc::new(memory);
                        let resolved =
                            super::resolved::ResolvedCostEntry::new_with_domain_and_memory(
                                entry,
                                self.workload_domain(),
                                Some(Arc::clone(&memory)),
                            );
                        drop(raw_memory);
                        match resolved
                            .retained_payload_bytes()
                            .filter(|n| *n <= memory.bytes())
                        {
                            Some(bytes) => {
                                let shrunk = memory.shrink_to(bytes);
                                debug_assert!(shrunk);
                                Some(resolved)
                            }
                            None => {
                                self.increment(&self.entries_dropped_capacity);
                                None
                            }
                        }
                    }
                    None => {
                        self.increment(&self.entries_dropped_capacity);
                        None
                    }
                };
                (
                    resolved,
                    queued.ticket,
                    queued.source_generation,
                    None,
                    false,
                )
            }
            QueuedCostPayload::Raw(raw) => {
                if let Some(wait) = raw.queue_wait_ns() {
                    self.add_duration(
                        &self.raw_queue_wait_ns_total,
                        &self.raw_queue_wait_ns_max,
                        wait,
                    );
                }
                let started = std::time::Instant::now();
                let working = self.working_memory.reserve(raw.working_bytes);
                let (entry, ticket, generation, proof) = match working {
                    Some(working) => {
                        self.accepted_working_bytes_peak
                            .fetch_max(raw.working_bytes, Ordering::Relaxed);
                        raw.resolve(self, ordinal, Arc::new(working))
                    }
                    None => {
                        self.memory_rejection(
                            raw.call_id(),
                            "worker_and_result_bytes",
                            CostRecorderCapacityResource::WorkingBytes,
                            self.working_memory
                                .retained()
                                .checked_add(raw.working_bytes),
                            self.byte_limits.maximum_working_bytes,
                        );
                        self.increment(&self.raw_lost);
                        let generation = raw.source_generation();
                        raw.dropped(CostSampleDrop::Capacity);
                        (None, None, generation, None)
                    }
                };
                drop(raw_memory);
                if let Ok(ns) = u64::try_from(started.elapsed().as_nanos()) {
                    self.add_duration(&self.raw_resolve_ns_total, &self.raw_resolve_ns_max, ns);
                } else {
                    self.counter_exhausted.store(true, Ordering::Relaxed);
                }
                self.increment(&self.raw_resolved);
                resolution.as_mut().expect("raw resolution guard").completed = true;
                if entry.is_none() {
                    if proof.is_some()
                        || ticket
                            .as_ref()
                            .is_some_and(|ticket| ticket.no_submission().is_some())
                    {
                        self.increment(&self.raw_no_submission);
                    } else {
                        self.increment(&self.raw_resolution_failed);
                    }
                }
                (entry, ticket, generation, proof, true)
            }
        };
        if let Some(entry) = &entry {
            let entry = entry.entry();
            if let Some(sample) = entry.sample() {
                if raw {
                    self.increment(&self.offered);
                    self.increment(&self.published);
                    self.increment(
                        &self.offered_by_wave[audit::wave_index(sample.actual_shape.kind)],
                    );
                    if sample.outcome == ferrum_scheduler::implementations::continuous::cost_model::WaveObservationOutcome::Completed {
                        self.increment(&self.offered_completed);
                    }
                }
                self.increment(&self.drained);
            }
            if entry.stages().is_some() {
                if raw {
                    self.increment(&self.host_stages_offered);
                    self.increment(&self.host_stages_published);
                }
                self.increment(&self.host_stages_drained);
            }
        }
        Some(CostTrainingInput::Inference((
            ordinal, entry, ticket, generation, proof,
        )))
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
        self.request_checkpoint_inner(paths, None, None)
    }

    /// Private qualified-source installation shares the existing bounded
    /// control slot. The source must end at this exact accepted FIFO cut.
    pub(super) fn request_catalog_activation(
        &self,
        cutoff: u64,
        publication: super::live_calibration::Publication,
    ) -> Result<CostCheckpointWaiter, CheckpointRequestError> {
        self.request_checkpoint_inner(
            None,
            Some(cutoff),
            Some(super::live_calibration::StartupCatalogActivation {
                publication,
                series: None,
            }),
        )
    }

    pub(super) fn request_catalog_activation_in_series(
        &self,
        cutoff: u64,
        publication: super::live_calibration::Publication,
        series: &mut super::live_calibration::StartupOwnerSeries,
    ) -> Result<CostCheckpointWaiter, ferrum_types::FerrumError> {
        let authorization = series.activation()?;
        let waiter = self
            .request_checkpoint_inner(
                None,
                Some(cutoff),
                Some(super::live_calibration::StartupCatalogActivation {
                    publication,
                    series: Some(authorization),
                }),
            )
            .map_err(|reason| {
                ferrum_types::FerrumError::config(format!("startup series activation: {reason}"))
            })?;
        // Failed queue reservations do not consume or skip a source ordinal.
        // Once queued, dropping its waiter never grants a second activation.
        series.accepted_activation();
        Ok(waiter)
    }

    fn request_checkpoint_inner(
        &self,
        paths: Option<super::profile_export::CostProfileCutPaths>,
        expected_cutoff: Option<u64>,
        publication: Option<super::live_calibration::StartupCatalogActivation>,
    ) -> Result<CostCheckpointWaiter, CheckpointRequestError> {
        let mut queue = self.consumer.lock();
        if queue.checkpoints_closed {
            return Err(CheckpointRequestError::Closing);
        }
        if queue.checkpoint.is_some() {
            return Err(CheckpointRequestError::Busy);
        }
        if self.counter_exhausted.load(Ordering::Relaxed) {
            return Err(CheckpointRequestError::CounterExhausted);
        }
        let accepted = self.accepted.load(Ordering::Acquire);
        if expected_cutoff.is_some_and(|expected| expected != accepted) {
            return Err(CheckpointRequestError::CutoffChanged);
        }
        let (mut pending, waiter) = PendingCheckpoint::new(accepted, paths);
        pending.catalog_activation = publication;
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
    ) -> Option<super::checkpoint::CheckpointWork> {
        if !self.checkpoint_pending.load(Ordering::Acquire) {
            return None;
        }
        let mut queue = self.consumer.lock();
        let pending = queue.checkpoint.as_mut()?;
        if pending.freezing || pending.cutoff != processed {
            return None;
        }
        pending.freezing = true;
        Some(super::checkpoint::CheckpointWork {
            cutoff: pending.cutoff,
            export_paths: pending.export_paths.take(),
            catalog_activation: pending.catalog_activation.take(),
        })
    }

    pub(super) fn complete_checkpoint(&self, result: FrozenCostCheckpoint) {
        let mut queue = self.consumer.lock();
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
        self.consumer.lock().checkpoints_closed = true;
    }

    pub(super) fn needs_drain(&self, processed: u64) -> bool {
        let queue = self.consumer.lock();
        self.accepted.load(Ordering::Acquire) > processed
            || queue.checkpoint.is_some()
            // Shutdown/export must not mistake a sent-but-uncommitted entry
            // for an empty source. This observes admission without acquiring it.
            || self.admission.is_locked()
    }

    pub(super) fn worker_stopped(&self) {
        // Closing is a worker/control operation. It waits only for an already
        // admitted producer, then drains everything that producer committed.
        self.worker_stopped.store(true, Ordering::Release);
        let admission = self.admission.lock();
        let mut control = self.consumer.lock();
        control.checkpoints_closed = true;
        let pending = control.checkpoint.take();
        self.checkpoint_pending.store(false, Ordering::Release);
        drop(control);
        while let Ok(entry) = self.receiver.try_recv() {
            if let QueuedCostPayload::Prefix(sample) = &entry.payload {
                sample.abandon();
            }
            if entry.publication_completed.load(Ordering::Acquire) {
                if let QueuedCostPayload::Raw(raw) = &entry.payload {
                    self.increment(&self.raw_lost);
                    self.increment(&self.raw_abandoned);
                    raw.dropped(CostSampleDrop::WorkerStopped);
                }
            }
            drop(entry); // Drops capture/row/byte owners outside control lock.
        }
        drop(admission);
        if let Some(pending) = pending {
            pending.complete(Err(CheckpointError::WorkerStopped));
        }
    }
    #[cfg(test)]
    pub(super) fn with_locked_consumer<T>(&self, action: impl FnOnce() -> T) -> T {
        let _guard = self.consumer.lock();
        action()
    }
    #[cfg(test)]
    pub(super) fn with_locked_producer<T>(&self, action: impl FnOnce() -> T) -> T {
        let _guard = self.admission.lock();
        action()
    }
    #[cfg(test)]
    pub(super) fn on_next_send(&self, hook: impl FnOnce() + Send + 'static) {
        *self.after_send.lock() = Some(Box::new(hook));
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
            memory: ObservationMemoryStats {
                requested_rows_peak: self.requested_rows_peak.load(Ordering::Relaxed),
                requested_raw_bytes_peak: self.requested_raw_bytes_peak.load(Ordering::Relaxed),
                requested_working_bytes_peak: self
                    .requested_working_bytes_peak
                    .load(Ordering::Relaxed),
                accepted_raw_bytes_peak: self.accepted_raw_bytes_peak.load(Ordering::Relaxed),
                accepted_working_bytes_peak: self
                    .accepted_working_bytes_peak
                    .load(Ordering::Relaxed),
                queued_raw_bytes: self.raw_memory.retained(),
                queued_raw_bytes_peak: self.raw_memory.peak(),
                worker_and_result_bytes: self.working_memory.retained(),
                worker_and_result_bytes_peak: self.working_memory.peak(),
                first_capture_rejection: self.first_capture_rejection.get().copied(),
            },
            raw_lost: self.raw_lost.load(Ordering::Relaxed),
            raw_abandoned: self.raw_abandoned.load(Ordering::Relaxed),
            raw_resolve_ns_total: self.raw_resolve_ns_total.load(Ordering::Relaxed),
            raw_resolve_ns_max: self.raw_resolve_ns_max.load(Ordering::Relaxed),
            raw_queue_wait_ns_total: self.raw_queue_wait_ns_total.load(Ordering::Relaxed),
            raw_queue_wait_ns_max: self.raw_queue_wait_ns_max.load(Ordering::Relaxed),
            raw_pending: self
                .raw_accepted
                .load(Ordering::Relaxed)
                .saturating_sub(self.raw_resolved.load(Ordering::Relaxed))
                .saturating_sub(self.raw_abandoned.load(Ordering::Relaxed)),
            raw_offered: self.raw_offered.load(Ordering::Relaxed),
            raw_accepted: self.raw_accepted.load(Ordering::Relaxed),
            raw_resolved: self.raw_resolved.load(Ordering::Relaxed),
            raw_resolution_failed: self.raw_resolution_failed.load(Ordering::Relaxed),
            raw_no_submission: self.raw_no_submission.load(Ordering::Relaxed),
            entries_offered: self.entries_offered.load(Ordering::Relaxed),
            entries_published: self.entries_published.load(Ordering::Relaxed),
            entries_drained: self.entries_drained.load(Ordering::Relaxed),
            entries_abandoned: self.entries_abandoned.load(Ordering::Relaxed),
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
