//! Continuous Batching Scheduler
//!
//! This scheduler implements iteration-level scheduling that allows dynamic
//! addition and removal of requests from running batches. Key features:
//!
//! - Iteration-level granularity: can add/remove requests between decode steps
//! - Separate prefill and decode queues for optimal scheduling
//! - Request state machine: Waiting -> Prefilling -> Decoding -> Completed
//! - Memory-aware scheduling based on KV cache usage
//! - Preemption support for long-running requests

mod pressure;

#[cfg(test)]
mod historical_replay_tests;

use pressure::{
    LogicalWorkFrontier, PressureCandidate, PressureCoordinator, PressureDecision,
    PressureHoldStatus, PressureReleaseFenceDisposition,
};
pub use pressure::{
    LogicalWorkGeneration, PressureEpisodeId, PressureEpisodeState, PressureHoldReleaseReason,
    PressureInvariantViolation, PressureInvariantViolationClass, PressureTransition,
    PressureTransitionKind, PressureTransitionOrdinal, PressureYieldKind, PressureYieldTransaction,
};

use crate::vnext::{
    AdmissionDeferral, AdmissionProbeOutcome, AdmissionQueueEligibility, AdmissionQueueEvent,
    AdmissionTickReceipt, AdmissionWakeEpochs, AdmissionWakeSnapshot, DynamicAdmissionQueue,
    DynamicAdmissionQueuePolicy, WaitingAdmissionTicket,
};
use crate::{
    BatchHint, BatchPlan, BatchResourceRequirements, PreemptionResult, PreemptionState,
    ScheduledRequest, Scheduler,
};
use async_trait::async_trait;
use ferrum_interfaces::model_executor::{
    ExecutorExecutionMaintenanceRetry, ExecutorPrefillAdmissionReceipt,
};
use ferrum_interfaces::scheduler::SchedulerMetrics;
use ferrum_interfaces::vnext::{
    AdmissionRejected, CapacityAvailabilityEpoch, CapacityAvailabilitySource, CapacityWaitCondition,
};
use ferrum_types::{
    BatchId, FerrumError, InferenceRequest, InferenceResponse, Priority, RequestId, RequestState,
    Result, SchedulerConfig, PROMPT_TOKENS_METADATA_KEY,
};
use indexmap::IndexMap;
use parking_lot::{Mutex, RwLock};
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    num::NonZeroU64,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};
use tracing::{debug, info, warn};

const NO_CAPACITY_BACKPRESSURE_LIMIT: usize = usize::MAX;
const CAPACITY_DECODE_FREE_BLOCK_HEADROOM: usize = 1;
const CAPACITY_MIXED_RECOMPUTE_FREE_BLOCK_HEADROOM: usize = 1;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ContinuousBatchRuntimeConfig {
    prompt_token_estimate: bool,
    prefill_first_until_active: Option<usize>,
    prefill_step_chunk: Option<usize>,
    active_decode_prefill_chunk: Option<usize>,
    scheduler_none_prof: bool,
}

impl ContinuousBatchRuntimeConfig {
    fn from_scheduler_config(config: &SchedulerConfig) -> Self {
        Self {
            prompt_token_estimate: config.prompt_token_estimate,
            prefill_first_until_active: config.prefill_first_until_active,
            prefill_step_chunk: config.prefill_step_chunk,
            active_decode_prefill_chunk: config.active_decode_prefill_chunk,
            scheduler_none_prof: config.scheduler_none_prof,
        }
    }
}

/// Request phase in continuous batching
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestPhase {
    /// Waiting in queue
    Waiting,
    /// Currently in prefill phase
    Prefilling,
    /// In decode phase (generating tokens)
    Decoding,
    /// Request completed
    Completed,
    /// Request was preempted
    Preempted,
    /// Request was cancelled
    Cancelled,
    /// Typed admission failed before prefill submission.
    AdmissionFailed,
}

/// Scheduler-owned response to an authoritative execution-capacity failure.
///
/// Prefill and decode use the same decision. `YieldPlanned` is not a logical
/// release: the engine must arm and complete the physical release fence before
/// the selected frontier becomes resumable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionCapacityAction {
    Deferred {
        count: usize,
    },
    YieldPlanned {
        transaction: PressureYieldTransaction,
    },
    InvariantViolation {
        violation: PressureInvariantViolation,
    },
}

const EXECUTION_READINESS_PENDING: u8 = 0;
const EXECUTION_READINESS_READY: u8 = 1;
const EXECUTION_READINESS_FAILED: u8 = 2;
const EXECUTION_READINESS_CANCELLED: u8 = 3;

#[derive(Debug)]
struct ExecutionReadinessState {
    status: AtomicU8,
}

/// Exact, generation-bearing wake authority for one scheduler readiness
/// deferral. A wake only makes the frontier eligible for an authoritative
/// executor reprobe; it never grants a resource permit.
#[derive(Debug, Clone)]
pub struct ExecutionReadinessWake {
    ticket_id: NonZeroU64,
    state: Arc<ExecutionReadinessState>,
}

impl ExecutionReadinessWake {
    pub const fn ticket_id(&self) -> NonZeroU64 {
        self.ticket_id
    }

    pub fn mark_ready(&self) -> bool {
        self.transition(EXECUTION_READINESS_READY)
    }

    pub fn mark_failed(&self) -> bool {
        self.transition(EXECUTION_READINESS_FAILED)
    }

    pub fn cancel(&self) -> bool {
        self.transition(EXECUTION_READINESS_CANCELLED)
    }

    fn transition(&self, next: u8) -> bool {
        self.state
            .status
            .compare_exchange(
                EXECUTION_READINESS_PENDING,
                next,
                Ordering::Release,
                Ordering::Acquire,
            )
            .is_ok()
    }
}

#[derive(Debug, Clone)]
struct ExecutionReadinessBlock {
    ticket_id: NonZeroU64,
    state: Arc<ExecutionReadinessState>,
}

impl ExecutionReadinessBlock {
    fn status(&self) -> u8 {
        self.state.status.load(Ordering::Acquire)
    }

    fn matches(&self, other: &Self) -> bool {
        self.ticket_id == other.ticket_id && Arc::ptr_eq(&self.state, &other.state)
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionReadinessDeferralReceipt {
    deferred_count: usize,
    wake: ExecutionReadinessWake,
}

impl ExecutionReadinessDeferralReceipt {
    pub const fn deferred_count(&self) -> usize {
        self.deferred_count
    }

    pub const fn wake(&self) -> &ExecutionReadinessWake {
        &self.wake
    }

    pub fn into_wake(self) -> ExecutionReadinessWake {
        self.wake
    }
}

/// Scheduler receipt for a voluntary fairness yield backed by real dynamic
/// pool mutation evidence. This path never opens a capacity-pressure episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ExecutionMaintenanceRetryReceipt {
    deferred_count: usize,
    not_before_iteration: u64,
    latest_capacity_epoch: u64,
}

impl ExecutionMaintenanceRetryReceipt {
    pub const fn deferred_count(&self) -> usize {
        self.deferred_count
    }

    pub const fn not_before_iteration(&self) -> u64 {
        self.not_before_iteration
    }

    pub const fn latest_capacity_epoch(&self) -> u64 {
        self.latest_capacity_epoch
    }
}

/// Typed terminal disposition of one physical execution-capacity yield.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionCapacityYieldDisposition {
    ProgressOwnerResumable,
    ProgressOwnerAdmissionPending,
    SelfRecomputeQueued,
    OwnerTerminal,
}

impl ExecutionCapacityYieldDisposition {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ProgressOwnerResumable => "progress_owner_resumable",
            Self::ProgressOwnerAdmissionPending => "progress_owner_admission_pending",
            Self::SelfRecomputeQueued => "self_recompute_queued",
            Self::OwnerTerminal => "owner_terminal",
        }
    }
}

/// Exact receipt for a peer victim installed behind a release fence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionCapacityPressureHoldReceipt {
    episode_id: PressureEpisodeId,
    transition_ordinal: PressureTransitionOrdinal,
    request_id: RequestId,
    progress_owner_id: RequestId,
    progress_baseline: LogicalWorkGeneration,
    progress_current: LogicalWorkGeneration,
    waiting_ticket: u64,
}

impl ExecutionCapacityPressureHoldReceipt {
    pub const fn episode_id(&self) -> PressureEpisodeId {
        self.episode_id
    }

    pub const fn transition_ordinal(&self) -> PressureTransitionOrdinal {
        self.transition_ordinal
    }

    pub const fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    pub const fn progress_owner_id(&self) -> &RequestId {
        &self.progress_owner_id
    }

    pub const fn progress_baseline(&self) -> LogicalWorkGeneration {
        self.progress_baseline
    }

    pub const fn progress_current(&self) -> LogicalWorkGeneration {
        self.progress_current
    }

    pub const fn waiting_ticket(&self) -> u64 {
        self.waiting_ticket
    }
}

/// Terminal result of one physical execution-capacity yield transaction.
///
/// A completed release can make a peer progress owner runnable, queue the same
/// logical frontier for recompute, or close because the owner became terminal.
/// The engine only resubmits a peer owner for
/// `ProgressOwnerResumable`; self recompute progresses through normal waiting
/// admission after the release fence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionCapacityYieldCompletion {
    victim_requeued: bool,
    installed_hold: Option<ExecutionCapacityPressureHoldReceipt>,
    release_transition_ordinal: PressureTransitionOrdinal,
    resumable_transition_ordinal: Option<PressureTransitionOrdinal>,
    owner_admission_pending_transition_ordinal: Option<PressureTransitionOrdinal>,
    closed_transition_ordinal: Option<PressureTransitionOrdinal>,
    disposition: ExecutionCapacityYieldDisposition,
}

impl ExecutionCapacityYieldCompletion {
    pub const fn victim_requeued(&self) -> bool {
        self.victim_requeued
    }

    pub const fn installed_hold(&self) -> Option<&ExecutionCapacityPressureHoldReceipt> {
        self.installed_hold.as_ref()
    }

    pub const fn progress_owner_resumable(&self) -> bool {
        matches!(
            self.disposition,
            ExecutionCapacityYieldDisposition::ProgressOwnerResumable
        )
    }

    pub const fn release_transition_ordinal(&self) -> PressureTransitionOrdinal {
        self.release_transition_ordinal
    }

    pub const fn resumable_transition_ordinal(&self) -> Option<PressureTransitionOrdinal> {
        self.resumable_transition_ordinal
    }

    pub const fn owner_admission_pending_transition_ordinal(
        &self,
    ) -> Option<PressureTransitionOrdinal> {
        self.owner_admission_pending_transition_ordinal
    }

    pub const fn closed_transition_ordinal(&self) -> Option<PressureTransitionOrdinal> {
        self.closed_transition_ordinal
    }

    pub const fn closed_reason(&self) -> Option<PressureHoldReleaseReason> {
        match self.disposition {
            ExecutionCapacityYieldDisposition::OwnerTerminal => {
                Some(PressureHoldReleaseReason::OwnerTerminal)
            }
            ExecutionCapacityYieldDisposition::ProgressOwnerResumable
            | ExecutionCapacityYieldDisposition::ProgressOwnerAdmissionPending
            | ExecutionCapacityYieldDisposition::SelfRecomputeQueued => None,
        }
    }

    pub const fn disposition(&self) -> ExecutionCapacityYieldDisposition {
        self.disposition
    }
}

/// Engine-owned physical release capabilities observed at the instant an
/// authoritative execution-capacity failure is routed to the scheduler.
///
/// Building this snapshot is a pressure-only operation. It keeps physical
/// ownership out of the scheduler while preventing a logical queue phase from
/// being mistaken for proof that a request can actually release capacity.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecutionCapacityReleaseSnapshot {
    release_sources_by_request: HashMap<RequestId, Vec<CapacityAvailabilitySource>>,
}

impl ExecutionCapacityReleaseSnapshot {
    pub fn new(
        capabilities: impl IntoIterator<Item = (RequestId, Vec<CapacityAvailabilitySource>)>,
    ) -> Self {
        let mut release_sources_by_request = HashMap::new();
        for (request_id, mut sources) in capabilities {
            sources.sort_unstable();
            sources.dedup();
            if !sources.is_empty() {
                release_sources_by_request.insert(request_id, sources);
            }
        }
        Self {
            release_sources_by_request,
        }
    }

    fn can_advance(&self, request_id: &RequestId, condition: &CapacityWaitCondition) -> bool {
        let Some(release_sources) = self.release_sources_by_request.get(request_id) else {
            return false;
        };
        condition
            .observed()
            .iter()
            .any(|observed| release_sources.binary_search(&observed.source()).is_ok())
    }

    pub fn has_external_releaser(
        &self,
        blocked_request_id: &RequestId,
        condition: &CapacityWaitCondition,
    ) -> bool {
        self.release_sources_by_request.keys().any(|request_id| {
            request_id != blocked_request_id && self.can_advance(request_id, condition)
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExecutionMaintenanceRetryTicket {
    not_before_iteration: u64,
    latest_capacity_epoch: u64,
}

/// Extended scheduled request with continuous batching metadata
#[derive(Debug, Clone)]
pub struct ContinuousBatchRequest {
    /// Base scheduled request
    pub inner: ScheduledRequest,
    /// Current phase
    pub phase: RequestPhase,
    /// Number of prefill tokens
    pub prefill_tokens: usize,
    /// Number of decode tokens generated
    pub decode_tokens: usize,
    /// Phase-independent logical progress and resident-work state.
    logical_work_frontier: LogicalWorkFrontier,
    /// KV cache blocks allocated
    pub kv_blocks: Vec<ferrum_types::BlockId>,
    /// Whether prefill is chunked
    pub chunked_prefill: bool,
    /// Current chunk offset for chunked prefill
    pub prefill_chunk_offset: usize,
    /// Request-local upper bound learned from definitely-not-submitted
    /// execution-capacity probes.
    pub prefill_execution_chunk_ceiling: Option<usize>,
    /// Last iteration this request was processed
    pub last_iteration: u64,
    /// Time spent in prefill (ms)
    pub prefill_time_ms: u64,
    /// Time spent in decode (ms)
    pub decode_time_ms: u64,
    /// Capacity-deferred requests wait for real capacity release before re-admission.
    pub capacity_deferred_until_release_epoch: u64,
    /// Capacity evidence epoch in which a mixed recompute attempt made no recorded progress.
    pub capacity_deferred_mixed_attempt_epoch: Option<u64>,
    /// Release epoch in which an otherwise idle scheduler already retried this request.
    pub capacity_deferred_empty_retry_epoch: Option<u64>,
    /// True when a decode request was evicted to waiting and must recompute KV.
    pub capacity_deferred_from_decode: bool,
    /// Stable identity retained across waiting -> active -> waiting cycles.
    pub waiting_admission_ticket: Option<WaitingAdmissionTicket>,
    /// Exact PlanRuntime capacity predicate suppressing blind execution retries.
    pub execution_capacity_deferral: Option<AdmissionDeferral>,
    /// Exact non-capacity readiness ticket, currently used for Request-state
    /// hazards. It is compare-exact so a stale waiter cannot unblock a newer
    /// frontier generation that reuses the same product request id.
    execution_readiness_block: Option<ExecutionReadinessBlock>,
    /// Scheduler-owned fairness yield after a proven physical backing mutation.
    execution_maintenance_retry: Option<ExecutionMaintenanceRetryTicket>,
    /// Rejects replay of a previously consumed maintenance mutation receipt.
    last_execution_maintenance_capacity_epoch: Option<u64>,
}

impl ContinuousBatchRequest {
    /// Create from inference request
    pub fn new(request: InferenceRequest) -> Self {
        Self {
            inner: ScheduledRequest::new(request),
            phase: RequestPhase::Waiting,
            prefill_tokens: 0,
            decode_tokens: 0,
            logical_work_frontier: LogicalWorkFrontier::default(),
            kv_blocks: Vec::new(),
            chunked_prefill: false,
            prefill_chunk_offset: 0,
            prefill_execution_chunk_ceiling: None,
            last_iteration: 0,
            prefill_time_ms: 0,
            decode_time_ms: 0,
            capacity_deferred_until_release_epoch: 0,
            capacity_deferred_mixed_attempt_epoch: None,
            capacity_deferred_empty_retry_epoch: None,
            capacity_deferred_from_decode: false,
            waiting_admission_ticket: None,
            execution_capacity_deferral: None,
            execution_readiness_block: None,
            execution_maintenance_retry: None,
            last_execution_maintenance_capacity_epoch: None,
        }
    }

    /// Get total tokens processed
    pub fn total_tokens(&self) -> usize {
        self.prefill_tokens + self.decode_tokens
    }

    /// Check if request is active (prefilling or decoding)
    pub fn is_active(&self) -> bool {
        matches!(
            self.phase,
            RequestPhase::Prefilling | RequestPhase::Decoding
        )
    }

    /// Check if request is finished
    pub fn is_finished(&self) -> bool {
        matches!(
            self.phase,
            RequestPhase::Completed | RequestPhase::Cancelled | RequestPhase::AdmissionFailed
        )
    }
}

pub type ExecutorAdmissionProbeOutcome =
    AdmissionProbeOutcome<ExecutorPrefillAdmissionReceipt, AdmissionRejected, FerrumError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutorAdmissionQueueObservation {
    PressureHoldReleased {
        episode_id: PressureEpisodeId,
        transition_ordinal: PressureTransitionOrdinal,
        request_id: RequestId,
        progress_owner_id: RequestId,
        progress_baseline: LogicalWorkGeneration,
        progress_current: LogicalWorkGeneration,
        reason: PressureHoldReleaseReason,
        previous_wait_condition: Option<CapacityWaitCondition>,
        current_wait_condition: Option<CapacityWaitCondition>,
        ticket: u64,
    },
    SkippedUnchanged {
        request_id: RequestId,
        ticket: u64,
        deferral: AdmissionDeferral,
        current: AdmissionWakeEpochs,
    },
    DecodeSkippedUnchanged {
        request_id: RequestId,
        deferral: AdmissionDeferral,
        current: AdmissionWakeEpochs,
        current_wait_sources: Vec<CapacityAvailabilityEpoch>,
    },
    DecodeResumed {
        request_id: RequestId,
        deferral: AdmissionDeferral,
        current: AdmissionWakeEpochs,
        current_wait_sources: Vec<CapacityAvailabilityEpoch>,
        exact_source_changed: bool,
        policy_epoch_changed: bool,
    },
    PrefillSkippedUnchanged {
        request_id: RequestId,
        deferral: AdmissionDeferral,
        current: AdmissionWakeEpochs,
        current_wait_sources: Vec<CapacityAvailabilityEpoch>,
    },
    PrefillResumed {
        request_id: RequestId,
        deferral: AdmissionDeferral,
        current: AdmissionWakeEpochs,
        current_wait_sources: Vec<CapacityAvailabilityEpoch>,
        exact_source_changed: bool,
        policy_epoch_changed: bool,
    },
}

#[derive(Debug, Clone, Copy)]
enum ExecutionCapacityQueuePhase {
    Prefill,
    Decode,
}

type ExecutorAdmissionQueueEvent = AdmissionQueueEvent<
    ContinuousBatchRequest,
    ExecutorPrefillAdmissionReceipt,
    AdmissionRejected,
    FerrumError,
>;

enum WaitingAdmissionMode<'a> {
    Legacy,
    Dynamic {
        wake: AdmissionWakeSnapshot<'a>,
        probe: &'a mut dyn FnMut(&InferenceRequest) -> ExecutorAdmissionProbeOutcome,
        observer: Option<&'a mut dyn FnMut(ExecutorAdmissionQueueObservation)>,
    },
}

impl<'a> WaitingAdmissionMode<'a> {
    fn wake(&self) -> Option<AdmissionWakeSnapshot<'a>> {
        match self {
            Self::Legacy => None,
            Self::Dynamic { wake, .. } => Some(*wake),
        }
    }

    fn observe(&mut self, observation: ExecutorAdmissionQueueObservation) {
        if let Self::Dynamic {
            observer: Some(observer),
            ..
        } = self
        {
            observer(observation);
        }
    }

    fn observes(&self) -> bool {
        matches!(
            self,
            Self::Dynamic {
                observer: Some(_),
                ..
            }
        )
    }
}

#[derive(Debug, Default)]
struct DecodeQueueState {
    requests: IndexMap<RequestId, ContinuousBatchRequest>,
    selection_cursor: Option<RequestId>,
}

impl DecodeQueueState {
    fn remove(&mut self, request_id: &RequestId) -> Option<ContinuousBatchRequest> {
        let removed_index = self.requests.get_index_of(request_id)?;
        let old_len = self.requests.len();
        let cursor_was_removed = self.selection_cursor.as_ref() == Some(request_id);
        let successor = if cursor_was_removed && old_len > 1 {
            Some(
                self.requests
                    .get_index((removed_index + 1) % old_len)
                    .expect("decode successor remains in bounds before removal")
                    .0
                    .clone(),
            )
        } else {
            None
        };

        let removed = self.requests.swap_remove(request_id);
        if self.requests.is_empty() {
            self.selection_cursor = None;
        } else if cursor_was_removed {
            self.selection_cursor = successor;
        } else if self
            .selection_cursor
            .as_ref()
            .is_some_and(|cursor_id| !self.requests.contains_key(cursor_id))
        {
            self.selection_cursor = self.requests.get_index(0).map(|(id, _)| id.clone());
        }
        removed
    }
}

/// Read-only scheduler counters for explicit engine diagnostics.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ContinuousSchedulerTraceSnapshot {
    pub current_iteration: u64,
    pub waiting_queue_len: usize,
    pub prefill_queue_len: usize,
    pub decode_queue_len: usize,
    pub decode_selection_cursor: Option<RequestId>,
    pub preempted_queue_len: usize,
    pub active_len: usize,
    pub completed_total: u64,
    pub failed_total: u64,
    pub cancelled_total: u64,
    pub preempted_total: u64,
    pub admitted_total: u64,
    pub capacity_deferred_total: u64,
    pub capacity_backpressure_admit_limit: Option<usize>,
    pub decode_capacity_backpressure_admit_limit: Option<usize>,
    pub decode_execution_pressure_enforced: bool,
    pub capacity_blocked_waiting_len: usize,
    pub execution_capacity_blocked_prefill_len: usize,
    pub execution_capacity_blocked_decode_len: usize,
    pub execution_readiness_deferred_total: u64,
    pub execution_readiness_blocked_prefill_len: usize,
    pub execution_readiness_blocked_decode_len: usize,
    pub capacity_release_epoch: u64,
    pub capacity_mixed_recompute_epoch: u64,
    pub capacity_mixed_recompute_blocked_until_epoch: u64,
    pub capacity_mixed_recompute_required_blocks_per_slot: Option<usize>,
    pub capacity_mixed_recompute_observed_free_blocks: Option<usize>,
    pub legacy_waiting_admission_ticks: u64,
    pub dynamic_admission_ticks: u64,
    pub dynamic_admission_probes: u64,
    pub dynamic_admission_skipped_unchanged: u64,
    pub dynamic_admission_deferred: u64,
    pub dynamic_backing_growth_requested: u64,
    pub dynamic_admission_failed: u64,
    pub pressure_episodes_created: u64,
    pub pressure_episodes_merged: u64,
    pub pressure_episode_bridges_deferred: u64,
    pub pressure_active_episodes: usize,
    pub pressure_pending_release_fences: usize,
    pub pressure_candidate_scans: u64,
    pub pressure_last_transition_ordinal: u64,
    pub pressure_dropped_journal_entries: u64,
}

/// Admission phases observed from one read of the scheduler request index.
///
/// Queue counters are intentionally excluded: separate queue locks cannot
/// produce a single-generation observation while requests change phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContinuousSchedulerAdmissionCounts {
    pub waiting_requests: usize,
    pub active_prefill_sequences: usize,
    pub active_decode_sequences: usize,
}

/// Continuous batching scheduler
///
/// This scheduler manages requests through their lifecycle in a continuous
/// batching system, allowing for iteration-level scheduling decisions.
pub struct ContinuousBatchScheduler {
    /// Configuration
    config: SchedulerConfig,

    /// Waiting queue (requests waiting to start)
    waiting_queue: RwLock<DynamicAdmissionQueue<ContinuousBatchRequest>>,

    /// Prefill queue (requests in prefill phase)
    prefill_queue: RwLock<VecDeque<ContinuousBatchRequest>>,

    /// Decode queue (requests in decode phase)
    decode_queue: RwLock<DecodeQueueState>,

    /// Preempted requests (can be resumed)
    preempted_requests: RwLock<HashMap<RequestId, ContinuousBatchRequest>>,

    /// Requests removed from waiting by a permanent/faulted typed admission.
    admission_failed_requests: RwLock<HashMap<RequestId, ContinuousBatchRequest>>,
    admission_failures: Mutex<VecDeque<(RequestId, FerrumError)>>,
    dynamic_admission_events: Mutex<Vec<ExecutorAdmissionQueueEvent>>,

    /// Request lookup table
    request_index: RwLock<HashMap<RequestId, RequestPhase>>,

    /// Current iteration number
    current_iteration: AtomicU64,

    /// Statistics
    completed_counter: AtomicU64,
    failed_counter: AtomicU64,
    cancelled_counter: AtomicU64,
    preempted_counter: AtomicU64,
    admitted_counter: AtomicU64,
    capacity_deferred_counter: AtomicU64,
    execution_readiness_deferred_counter: AtomicU64,
    next_execution_readiness_ticket: AtomicU64,
    capacity_backpressure_limit: AtomicUsize,
    decode_capacity_backpressure_limit: AtomicUsize,
    decode_execution_pressure_enforced: AtomicBool,
    decode_capacity_feedback_lock: Mutex<()>,
    capacity_backpressure_iteration: AtomicU64,
    capacity_release_epoch: AtomicU64,
    capacity_mixed_recompute_epoch: AtomicU64,
    capacity_mixed_recompute_blocked_until_epoch: AtomicU64,
    capacity_mixed_recompute_required_blocks_per_slot: AtomicUsize,
    capacity_mixed_recompute_observed_free_blocks: AtomicUsize,
    total_wait_time_us: AtomicU64,
    legacy_waiting_admission_ticks: AtomicU64,
    dynamic_admission_ticks: AtomicU64,
    dynamic_admission_probes: AtomicU64,
    dynamic_admission_skipped_unchanged: AtomicU64,
    dynamic_admission_deferred: AtomicU64,
    dynamic_backing_growth_requested: AtomicU64,
    dynamic_admission_failed: AtomicU64,

    /// Cold-path, phase-independent execution-capacity coordinator.
    pressure_coordinator: Mutex<PressureCoordinator>,
    /// A read-only hot-path guard. False avoids taking the coordinator lock.
    pressure_active: AtomicBool,

    /// Start time
    start_time: Instant,

    /// Metrics tracker
    metrics_tracker: Arc<ContinuousBatchMetrics>,

    /// Continuous batching specific config
    cb_config: ContinuousBatchConfig,

    /// Runtime env-derived switches parsed once at scheduler construction.
    runtime_config: ContinuousBatchRuntimeConfig,
}

/// Continuous batching specific configuration
#[derive(Debug, Clone)]
pub struct ContinuousBatchConfig {
    /// Maximum batch size for prefill
    pub max_prefill_batch: usize,
    /// Maximum batch size for decode
    pub max_decode_batch: usize,
    /// Enable chunked prefill
    pub enable_chunked_prefill: bool,
    /// Chunk size for chunked prefill (tokens)
    pub prefill_chunk_size: usize,
    /// Maximum KV cache blocks per request
    pub max_kv_blocks_per_request: usize,
    /// Enable request swapping (preemption)
    pub enable_swapping: bool,
    /// Swap priority threshold
    pub swap_priority_threshold: Priority,
    /// Target iteration time (ms)
    pub target_iteration_time_ms: u64,
}

impl Default for ContinuousBatchConfig {
    fn default() -> Self {
        Self {
            max_prefill_batch: 8,
            max_decode_batch: 256,
            enable_chunked_prefill: true,
            prefill_chunk_size: 512,
            max_kv_blocks_per_request: 1024,
            enable_swapping: true,
            swap_priority_threshold: Priority::Low,
            target_iteration_time_ms: 50,
        }
    }
}

/// Metrics tracker for continuous batching
struct ContinuousBatchMetrics {
    total_prefill_tokens: AtomicU64,
    total_decode_tokens: AtomicU64,
    total_prefill_time_ms: AtomicU64,
    total_decode_time_ms: AtomicU64,
    request_count: AtomicU64,
    iteration_count: AtomicU64,
}

impl ContinuousBatchMetrics {
    fn new() -> Self {
        Self {
            total_prefill_tokens: AtomicU64::new(0),
            total_decode_tokens: AtomicU64::new(0),
            total_prefill_time_ms: AtomicU64::new(0),
            total_decode_time_ms: AtomicU64::new(0),
            request_count: AtomicU64::new(0),
            iteration_count: AtomicU64::new(0),
        }
    }

    fn record_completion(&self, req: &ContinuousBatchRequest) {
        self.total_prefill_tokens
            .fetch_add(req.prefill_tokens as u64, Ordering::Relaxed);
        self.total_decode_tokens
            .fetch_add(req.decode_tokens as u64, Ordering::Relaxed);
        self.total_prefill_time_ms
            .fetch_add(req.prefill_time_ms, Ordering::Relaxed);
        self.total_decode_time_ms
            .fetch_add(req.decode_time_ms, Ordering::Relaxed);
        self.request_count.fetch_add(1, Ordering::Relaxed);
    }

    fn record_iteration(&self) {
        self.iteration_count.fetch_add(1, Ordering::Relaxed);
    }
}

impl ContinuousBatchScheduler {
    /// Create new continuous batch scheduler
    pub fn new(config: SchedulerConfig) -> Self {
        Self::with_cb_config(config, ContinuousBatchConfig::default())
    }

    /// Create with specific continuous batching configuration
    pub fn with_cb_config(config: SchedulerConfig, cb_config: ContinuousBatchConfig) -> Self {
        info!(
            "Creating continuous batch scheduler: max_prefill={}, max_decode={}",
            cb_config.max_prefill_batch, cb_config.max_decode_batch
        );
        let runtime_config = ContinuousBatchRuntimeConfig::from_scheduler_config(&config);

        Self {
            config,
            waiting_queue: RwLock::new(DynamicAdmissionQueue::new(
                DynamicAdmissionQueuePolicy::default(),
            )),
            prefill_queue: RwLock::new(VecDeque::new()),
            decode_queue: RwLock::new(DecodeQueueState::default()),
            preempted_requests: RwLock::new(HashMap::new()),
            admission_failed_requests: RwLock::new(HashMap::new()),
            admission_failures: Mutex::new(VecDeque::new()),
            dynamic_admission_events: Mutex::new(Vec::new()),
            request_index: RwLock::new(HashMap::new()),
            current_iteration: AtomicU64::new(0),
            completed_counter: AtomicU64::new(0),
            failed_counter: AtomicU64::new(0),
            cancelled_counter: AtomicU64::new(0),
            preempted_counter: AtomicU64::new(0),
            admitted_counter: AtomicU64::new(0),
            capacity_deferred_counter: AtomicU64::new(0),
            execution_readiness_deferred_counter: AtomicU64::new(0),
            next_execution_readiness_ticket: AtomicU64::new(1),
            capacity_backpressure_limit: AtomicUsize::new(NO_CAPACITY_BACKPRESSURE_LIMIT),
            decode_capacity_backpressure_limit: AtomicUsize::new(NO_CAPACITY_BACKPRESSURE_LIMIT),
            decode_execution_pressure_enforced: AtomicBool::new(false),
            decode_capacity_feedback_lock: Mutex::new(()),
            capacity_backpressure_iteration: AtomicU64::new(u64::MAX),
            capacity_release_epoch: AtomicU64::new(0),
            capacity_mixed_recompute_epoch: AtomicU64::new(0),
            capacity_mixed_recompute_blocked_until_epoch: AtomicU64::new(0),
            capacity_mixed_recompute_required_blocks_per_slot: AtomicUsize::new(0),
            capacity_mixed_recompute_observed_free_blocks: AtomicUsize::new(usize::MAX),
            total_wait_time_us: AtomicU64::new(0),
            legacy_waiting_admission_ticks: AtomicU64::new(0),
            dynamic_admission_ticks: AtomicU64::new(0),
            dynamic_admission_probes: AtomicU64::new(0),
            dynamic_admission_skipped_unchanged: AtomicU64::new(0),
            dynamic_admission_deferred: AtomicU64::new(0),
            dynamic_backing_growth_requested: AtomicU64::new(0),
            dynamic_admission_failed: AtomicU64::new(0),
            pressure_coordinator: Mutex::new(PressureCoordinator::default()),
            pressure_active: AtomicBool::new(false),
            start_time: Instant::now(),
            metrics_tracker: Arc::new(ContinuousBatchMetrics::new()),
            cb_config,
            runtime_config,
        }
    }

    /// Get number of active requests (prefilling + decoding)
    pub fn active_count(&self) -> usize {
        self.prefill_queue.read().len() + self.decode_queue.read().requests.len()
    }

    /// Get number of waiting requests
    pub fn waiting_count(&self) -> usize {
        self.waiting_queue.read().len()
    }

    /// Returns an aggregate exact wait predicate only when every queued item is
    /// passively blocked. Runnable prefill/decode work and first-probe waiting
    /// work deliberately return `None` so the engine keeps driving iterations.
    pub fn passive_capacity_wait_condition(
        &self,
    ) -> Result<Option<ferrum_interfaces::vnext::CapacityWaitCondition>> {
        let mut conditions = Vec::new();
        {
            let prefill = self.prefill_queue.read();
            for request in prefill.iter() {
                if request
                    .execution_readiness_block
                    .as_ref()
                    .is_some_and(|block| block.status() != EXECUTION_READINESS_CANCELLED)
                {
                    continue;
                }
                let Some(deferral) = request.execution_capacity_deferral.as_ref() else {
                    return Ok(None);
                };
                conditions.push(deferral.wait_condition().clone());
            }
        }
        {
            let decode = self.decode_queue.read();
            for request in decode.requests.values() {
                if request
                    .execution_readiness_block
                    .as_ref()
                    .is_some_and(|block| block.status() != EXECUTION_READINESS_CANCELLED)
                {
                    continue;
                }
                let Some(deferral) = request.execution_capacity_deferral.as_ref() else {
                    return Ok(None);
                };
                conditions.push(deferral.wait_condition().clone());
            }
        }
        let waiting_queue = self.waiting_queue.read();
        let pressure_hold_is_active = |request: &ContinuousBatchRequest| {
            self.pressure_active.load(Ordering::Acquire)
                && matches!(
                    self.pressure_coordinator
                        .lock()
                        .hold_status(&request.inner.request.id),
                    PressureHoldStatus::Held { .. }
                )
        };
        let waiting_count = waiting_queue
            .iter()
            .filter(|request| !pressure_hold_is_active(request))
            .count();
        let waiting = waiting_queue
            .passive_wait_condition_for(|request| !pressure_hold_is_active(request))
            .map_err(|error| FerrumError::scheduler(error.to_string()))?;
        drop(waiting_queue);
        if waiting_count > 0 && waiting.is_none() {
            return Ok(None);
        }
        if let Some(waiting) = waiting {
            conditions.push(waiting);
        }
        if conditions.is_empty() {
            return Ok(None);
        }

        let coordinator = conditions[0].coordinator_id();
        let mut observed_by_source = BTreeMap::new();
        for condition in conditions {
            if condition.coordinator_id() != coordinator {
                return Err(FerrumError::scheduler(
                    "passive capacity waits belong to different coordinators",
                ));
            }
            for observed in condition.observed() {
                observed_by_source
                    .entry(observed.source())
                    .and_modify(|epoch: &mut u64| *epoch = (*epoch).min(observed.epoch()))
                    .or_insert(observed.epoch());
            }
        }
        let observed = observed_by_source
            .into_iter()
            .map(|(source, epoch)| {
                ferrum_interfaces::vnext::CapacityAvailabilityEpoch::new(source, epoch)
                    .map_err(|error| FerrumError::scheduler(error.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        let condition = ferrum_interfaces::vnext::CapacityWaitCondition::new(coordinator, observed)
            .map_err(|error| FerrumError::scheduler(error.to_string()))?;
        let pressure = self.pressure_coordinator.lock();
        if pressure.has_pending_release_for(&condition) {
            return Ok(None);
        }
        if pressure.all_blocked_without_release_for(&condition) {
            return Err(FerrumError::scheduler(
                "capacity pressure contract reached all blocked frontiers without a pending release",
            ));
        }
        Ok(Some(condition))
    }

    /// True only when every active frontier is parked behind an exact
    /// non-capacity readiness ticket. The engine may sleep on `work_notify` in
    /// this state because each pending ticket has a separately owned waiter.
    pub fn all_active_execution_readiness_blocked(&self) -> bool {
        let prefill = self.prefill_queue.read();
        let decode = self.decode_queue.read();
        let active = prefill.len() + decode.requests.len();
        active != 0
            && prefill
                .iter()
                .chain(decode.requests.values())
                .all(|request| {
                    request
                        .execution_readiness_block
                        .as_ref()
                        .is_some_and(|block| {
                            matches!(
                                block.status(),
                                EXECUTION_READINESS_PENDING | EXECUTION_READINESS_FAILED
                            )
                        })
                })
    }

    /// Get number of decoding requests
    pub fn decoding_count(&self) -> usize {
        self.decode_queue.read().requests.len()
    }

    /// Get number of prefilling requests
    pub fn prefilling_count(&self) -> usize {
        self.prefill_queue.read().len()
    }

    /// Snapshot queue lengths and counters for explicit scheduler trace artifacts.
    pub fn trace_snapshot(&self) -> ContinuousSchedulerTraceSnapshot {
        self.trace_snapshot_with_prefill_read_observer(|| {})
    }

    fn trace_snapshot_with_prefill_read_observer(
        &self,
        prefill_read_observer: impl FnOnce(),
    ) -> ContinuousSchedulerTraceSnapshot {
        let waiting_queue_len = self.waiting_queue.read().len();
        // Keep exactly one fair read guard per queue. Reacquiring one of these
        // locks while its first guard is alive can self-deadlock when a writer
        // queues between the two reads: parking_lot then blocks the recursive
        // read behind the writer, while the writer waits for the first guard.
        let (
            prefill_queue_len,
            execution_capacity_blocked_prefill_len,
            execution_readiness_blocked_prefill_len,
        ) = {
            let prefill_queue = self.prefill_queue.read();
            let counts = (
                prefill_queue.len(),
                prefill_queue
                    .iter()
                    .filter(|request| request.execution_capacity_deferral.is_some())
                    .count(),
                prefill_queue
                    .iter()
                    .filter(|request| request.execution_readiness_block.is_some())
                    .count(),
            );
            prefill_read_observer();
            counts
        };
        let (
            decode_queue_len,
            decode_selection_cursor,
            execution_capacity_blocked_decode_len,
            execution_readiness_blocked_decode_len,
        ) = {
            let decode_queue = self.decode_queue.read();
            (
                decode_queue.requests.len(),
                decode_queue.selection_cursor.clone(),
                decode_queue
                    .requests
                    .values()
                    .filter(|request| request.execution_capacity_deferral.is_some())
                    .count(),
                decode_queue
                    .requests
                    .values()
                    .filter(|request| request.execution_readiness_block.is_some())
                    .count(),
            )
        };
        let preempted_queue_len = self.preempted_requests.read().len();
        let pressure = self.pressure_coordinator.lock().stats();
        let (decode_capacity_backpressure_admit_limit, decode_execution_pressure_enforced) = {
            let _feedback = self.decode_capacity_feedback_lock.lock();
            (
                self.decode_capacity_backpressure_limit(),
                self.decode_execution_pressure_enforced
                    .load(Ordering::Acquire),
            )
        };

        ContinuousSchedulerTraceSnapshot {
            current_iteration: self.current_iteration.load(Ordering::Relaxed),
            waiting_queue_len,
            prefill_queue_len,
            decode_queue_len,
            decode_selection_cursor,
            preempted_queue_len,
            active_len: prefill_queue_len + decode_queue_len,
            completed_total: self.completed_counter.load(Ordering::Relaxed),
            failed_total: self.failed_counter.load(Ordering::Relaxed),
            cancelled_total: self.cancelled_counter.load(Ordering::Relaxed),
            preempted_total: self.preempted_counter.load(Ordering::Relaxed),
            admitted_total: self.admitted_counter.load(Ordering::Relaxed),
            capacity_deferred_total: self.capacity_deferred_counter.load(Ordering::Relaxed),
            capacity_backpressure_admit_limit: self.capacity_backpressure_admit_limit(),
            decode_capacity_backpressure_admit_limit,
            decode_execution_pressure_enforced,
            capacity_blocked_waiting_len: self.capacity_blocked_waiting_len(),
            execution_capacity_blocked_prefill_len,
            execution_capacity_blocked_decode_len,
            execution_readiness_deferred_total: self
                .execution_readiness_deferred_counter
                .load(Ordering::Relaxed),
            execution_readiness_blocked_prefill_len,
            execution_readiness_blocked_decode_len,
            capacity_release_epoch: self.capacity_release_epoch.load(Ordering::Relaxed),
            capacity_mixed_recompute_epoch: self
                .capacity_mixed_recompute_epoch
                .load(Ordering::Relaxed),
            capacity_mixed_recompute_blocked_until_epoch: self
                .capacity_mixed_recompute_blocked_until_epoch
                .load(Ordering::Relaxed),
            capacity_mixed_recompute_required_blocks_per_slot: match self
                .capacity_mixed_recompute_required_blocks_per_slot
                .load(Ordering::Relaxed)
            {
                0 => None,
                value => Some(value),
            },
            capacity_mixed_recompute_observed_free_blocks: match self
                .capacity_mixed_recompute_observed_free_blocks
                .load(Ordering::Relaxed)
            {
                usize::MAX => None,
                value => Some(value),
            },
            legacy_waiting_admission_ticks: self
                .legacy_waiting_admission_ticks
                .load(Ordering::Relaxed),
            dynamic_admission_ticks: self.dynamic_admission_ticks.load(Ordering::Relaxed),
            dynamic_admission_probes: self.dynamic_admission_probes.load(Ordering::Relaxed),
            dynamic_admission_skipped_unchanged: self
                .dynamic_admission_skipped_unchanged
                .load(Ordering::Relaxed),
            dynamic_admission_deferred: self.dynamic_admission_deferred.load(Ordering::Relaxed),
            dynamic_backing_growth_requested: self
                .dynamic_backing_growth_requested
                .load(Ordering::Relaxed),
            dynamic_admission_failed: self.dynamic_admission_failed.load(Ordering::Relaxed),
            pressure_episodes_created: pressure.episodes_created,
            pressure_episodes_merged: pressure.episodes_merged,
            pressure_episode_bridges_deferred: pressure.episode_bridges_deferred,
            pressure_active_episodes: pressure.active_episodes,
            pressure_pending_release_fences: pressure.pending_release_fences,
            pressure_candidate_scans: pressure.candidate_scans,
            pressure_last_transition_ordinal: pressure.last_transition_ordinal,
            pressure_dropped_journal_entries: pressure.dropped_journal_entries,
        }
    }

    /// Returns mutually exclusive admission phases from one authoritative map.
    pub fn admission_phase_counts(&self) -> ContinuousSchedulerAdmissionCounts {
        let request_index = self.request_index.read();
        let mut counts = ContinuousSchedulerAdmissionCounts {
            waiting_requests: 0,
            active_prefill_sequences: 0,
            active_decode_sequences: 0,
        };
        for phase in request_index.values() {
            match phase {
                RequestPhase::Waiting => counts.waiting_requests += 1,
                RequestPhase::Prefilling => counts.active_prefill_sequences += 1,
                RequestPhase::Decoding => counts.active_decode_sequences += 1,
                RequestPhase::Completed
                | RequestPhase::Preempted
                | RequestPhase::Cancelled
                | RequestPhase::AdmissionFailed => {}
            }
        }
        counts
    }

    /// Return the scheduler phase for trace-only plan classification.
    pub fn trace_phase(&self, request_id: &RequestId) -> Option<RequestPhase> {
        self.request_index.read().get(request_id).copied()
    }

    /// Bounded, ordinal scheduler journal used by replay and release artifacts.
    pub fn pressure_transition_journal(&self) -> Vec<PressureTransition> {
        self.pressure_coordinator.lock().journal()
    }

    fn requeue_waiting_request(
        &self,
        waiting_queue: &mut DynamicAdmissionQueue<ContinuousBatchRequest>,
        request_index: &mut HashMap<RequestId, RequestPhase>,
        mut request: ContinuousBatchRequest,
    ) -> bool {
        let request_id = request.inner.request.id.clone();
        let Some(ticket) = request.waiting_admission_ticket else {
            let error = FerrumError::scheduler(format!(
                "request {request_id} lost its waiting admission identity"
            ));
            request.phase = RequestPhase::AdmissionFailed;
            request.inner.state = RequestState::Failed;
            request_index.insert(request_id.clone(), RequestPhase::AdmissionFailed);
            self.admission_failed_requests
                .write()
                .insert(request_id.clone(), request);
            self.admission_failures
                .lock()
                .push_back((request_id, error));
            self.dynamic_admission_failed
                .fetch_add(1, Ordering::Relaxed);
            return false;
        };
        let result = waiting_queue.requeue(ticket, request);
        match result {
            Ok(()) => {
                request_index.insert(request_id, RequestPhase::Waiting);
                true
            }
            Err((error, mut request)) => {
                let error = FerrumError::scheduler(error.to_string());
                request.phase = RequestPhase::AdmissionFailed;
                request.inner.state = RequestState::Failed;
                request_index.insert(request_id.clone(), RequestPhase::AdmissionFailed);
                self.admission_failed_requests
                    .write()
                    .insert(request_id.clone(), request);
                self.admission_failures
                    .lock()
                    .push_back((request_id, error));
                self.dynamic_admission_failed
                    .fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    fn promote_to_prefill_with_empty_retry(
        &self,
        request_id: &RequestId,
        empty_retry_epoch: Option<u64>,
    ) -> bool {
        let mut waiting_queue = self.waiting_queue.write();
        let mut prefill_queue = self.prefill_queue.write();
        let mut request_index = self.request_index.write();

        let waiting_position =
            waiting_queue.position(|request| request.inner.request.id == *request_id);
        if let Some(pos) = waiting_position {
            let mut req = waiting_queue.remove(pos).unwrap();
            if let Some(epoch) = empty_retry_epoch {
                req.capacity_deferred_empty_retry_epoch = Some(epoch);
            }
            req.logical_work_frontier
                .begin_prefill(req.capacity_deferred_from_decode);
            req.phase = RequestPhase::Prefilling;
            req.inner.state = RequestState::Running;
            let started_at = chrono::Utc::now();
            let wait_us = started_at
                .signed_duration_since(req.inner.submitted_at)
                .num_microseconds()
                .unwrap_or(0)
                .max(0) as u64;
            req.inner.started_at = Some(started_at);
            self.total_wait_time_us
                .fetch_add(wait_us, Ordering::Relaxed);
            self.admitted_counter.fetch_add(1, Ordering::Relaxed);

            request_index.insert(request_id.clone(), RequestPhase::Prefilling);
            prefill_queue.push_back(req);
            self.consume_pressure_hold(request_id);

            debug!("Promoted request {} to prefill queue", request_id);
            true
        } else {
            false
        }
    }

    fn promote_admitted_request(
        &self,
        mut request: ContinuousBatchRequest,
        receipt: &ExecutorPrefillAdmissionReceipt,
    ) {
        let request_id = request.inner.request.id.clone();
        if receipt.request_id != request_id {
            self.fail_typed_admission(
                request,
                FerrumError::scheduler(format!(
                    "executor admission receipt belongs to {}, expected {}",
                    receipt.request_id, request_id
                )),
            );
            return;
        }
        request
            .logical_work_frontier
            .begin_prefill(request.capacity_deferred_from_decode);
        request.phase = RequestPhase::Prefilling;
        request.inner.state = RequestState::Running;
        let started_at = chrono::Utc::now();
        let wait_us = started_at
            .signed_duration_since(request.inner.submitted_at)
            .num_microseconds()
            .unwrap_or(0)
            .max(0) as u64;
        request.inner.started_at = Some(started_at);
        self.total_wait_time_us
            .fetch_add(wait_us, Ordering::Relaxed);
        self.admitted_counter.fetch_add(1, Ordering::Relaxed);
        self.request_index
            .write()
            .insert(request_id.clone(), RequestPhase::Prefilling);
        self.prefill_queue.write().push_back(request);
        self.consume_pressure_hold(&request_id);
        debug!("Typed admission promoted request {} to prefill", request_id);
    }

    fn fail_typed_admission(&self, mut request: ContinuousBatchRequest, error: FerrumError) {
        let request_id = request.inner.request.id.clone();
        request.logical_work_frontier.finish();
        self.record_pressure_frontier_terminal(&request_id);
        request.phase = RequestPhase::AdmissionFailed;
        request.inner.state = RequestState::Failed;
        self.request_index
            .write()
            .insert(request_id.clone(), RequestPhase::AdmissionFailed);
        self.admission_failed_requests
            .write()
            .insert(request_id.clone(), request);
        self.admission_failures
            .lock()
            .push_back((request_id, error));
        self.dynamic_admission_failed
            .fetch_add(1, Ordering::Relaxed);
    }

    fn admit_waiting_dynamically(
        &self,
        maximum_probes: usize,
        maximum_admissions: usize,
        waiting_admission: &mut WaitingAdmissionMode<'_>,
    ) -> Result<AdmissionTickReceipt> {
        let WaitingAdmissionMode::Dynamic {
            wake,
            probe,
            observer,
        } = waiting_admission
        else {
            return Err(FerrumError::scheduler(
                "dynamic admission requires a typed wake/probe mode",
            ));
        };
        let wake = *wake;
        self.dynamic_admission_ticks.fetch_add(1, Ordering::Relaxed);
        let mut waiting = self.waiting_queue.write();
        let maximum_probes = waiting.len().min(maximum_probes);
        let mut events = self.dynamic_admission_events.lock();
        let observer = std::cell::RefCell::new(observer);
        let receipt = waiting
            .schedule_into_observed_with_eligibility(
                wake,
                maximum_probes,
                maximum_admissions,
                &mut events,
                |request, ticket| {
                    let request_id = &request.inner.request.id;
                    if !self.pressure_active.load(Ordering::Acquire) {
                        return AdmissionQueueEligibility::Eligible;
                    }
                    let hold_status = {
                        let coordinator = self.pressure_coordinator.lock();
                        coordinator.hold_status(request_id)
                    };
                    match hold_status {
                        PressureHoldStatus::Held { .. } => AdmissionQueueEligibility::Held,
                        PressureHoldStatus::OwnerAdmissionEligible { .. } => {
                            AdmissionQueueEligibility::Eligible
                        }
                        PressureHoldStatus::Released {
                            episode_id,
                            progress_owner_id,
                            progress_baseline,
                            progress_current,
                            reason,
                            ordinal,
                            previous_wait_condition,
                            current_wait_condition,
                        } => {
                            {
                                let mut coordinator = self.pressure_coordinator.lock();
                                if let Err(error) = coordinator.consume_released_hold(request_id) {
                                    warn!(
                                        request_id = %request_id,
                                        error = %error,
                                        "Pressure coordinator rejected terminal hold release"
                                    );
                                    return AdmissionQueueEligibility::Held;
                                }
                                self.pressure_active
                                    .store(coordinator.has_records(), Ordering::Release);
                            }
                            if let Some(observer) = observer.borrow_mut().as_deref_mut() {
                                observer(ExecutorAdmissionQueueObservation::PressureHoldReleased {
                                    episode_id,
                                    transition_ordinal: ordinal,
                                    request_id: request_id.clone(),
                                    progress_owner_id,
                                    progress_baseline,
                                    progress_current,
                                    reason,
                                    previous_wait_condition,
                                    current_wait_condition,
                                    ticket: ticket.get(),
                                });
                            }
                            AdmissionQueueEligibility::Eligible
                        }
                        PressureHoldStatus::None => AdmissionQueueEligibility::Eligible,
                    }
                },
                |_request, _ticket| {},
                |request, ticket, deferral| {
                    if let Some(observer) = observer.borrow_mut().as_deref_mut() {
                        observer(ExecutorAdmissionQueueObservation::SkippedUnchanged {
                            request_id: request.inner.request.id.clone(),
                            ticket: ticket.get(),
                            deferral: deferral.clone(),
                            current: wake.epochs(),
                        });
                    }
                },
                |request| probe(&request.inner.request),
            )
            .map_err(|error| FerrumError::scheduler(error.to_string()))?;
        drop(waiting);

        self.dynamic_admission_probes
            .fetch_add(receipt.probed() as u64, Ordering::Relaxed);
        self.dynamic_admission_skipped_unchanged
            .fetch_add(receipt.skipped_unchanged() as u64, Ordering::Relaxed);
        self.dynamic_admission_deferred
            .fetch_add(receipt.deferred() as u64, Ordering::Relaxed);
        self.dynamic_backing_growth_requested
            .fetch_add(receipt.backing_growth_requested() as u64, Ordering::Relaxed);

        for event in events.drain(..) {
            match event {
                AdmissionQueueEvent::Admitted {
                    request, admission, ..
                } => self.promote_admitted_request(request, &admission),
                AdmissionQueueEvent::PermanentRejected {
                    request, rejection, ..
                } => self.fail_typed_admission(
                    request,
                    FerrumError::request_validation(format!(
                        "request cannot fit the vNext runtime: {rejection:?}"
                    )),
                ),
                AdmissionQueueEvent::Faulted { request, error, .. } => {
                    self.fail_typed_admission(request, error)
                }
                AdmissionQueueEvent::ContractFaulted { request, error, .. } => {
                    self.fail_typed_admission(request, FerrumError::scheduler(error.to_string()))
                }
                AdmissionQueueEvent::PreemptionRequested { .. } => {}
                AdmissionQueueEvent::BackingGrowthRequested { .. } => {}
            }
        }
        Ok(receipt)
    }

    pub fn take_admission_failures(&self) -> Vec<(RequestId, FerrumError)> {
        self.admission_failures.lock().drain(..).collect()
    }

    /// Fail one still-waiting typed admission after executor maintenance
    /// returned a terminal error. The queue entry is removed before any
    /// completion or backend work can run.
    pub fn fail_waiting_admission(&self, request_id: &RequestId, error: FerrumError) -> bool {
        let request = {
            let mut waiting = self.waiting_queue.write();
            waiting
                .position(|request| request.inner.request.id == *request_id)
                .and_then(|position| waiting.remove(position))
        };
        let Some(request) = request else {
            return false;
        };
        self.fail_typed_admission(request, error);
        true
    }

    /// Preserve one waiting request after backing growth hit live device
    /// pressure. The original queue ticket and fairness age remain unchanged.
    pub fn wait_for_release_after_backing_pressure(
        &self,
        request_id: &RequestId,
        observed: AdmissionWakeEpochs,
        wait_condition: ferrum_interfaces::vnext::CapacityWaitCondition,
    ) -> Result<bool> {
        self.waiting_queue
            .write()
            .wait_for_release_after_backing_pressure(
                |request| request.inner.request.id == *request_id,
                observed,
                wait_condition,
            )
            .map_err(|error| FerrumError::scheduler(error.to_string()))
    }

    pub fn retry_after_backing_recheck(
        &self,
        request_id: &RequestId,
        observed: AdmissionWakeEpochs,
    ) -> Result<bool> {
        self.waiting_queue
            .write()
            .retry_after_backing_recheck(
                |request| request.inner.request.id == *request_id,
                observed,
            )
            .map_err(|error| FerrumError::scheduler(error.to_string()))
    }

    pub fn next_batch_with_dynamic_admission(
        &self,
        hint: BatchHint,
        wake: AdmissionWakeSnapshot<'_>,
        probe: &mut dyn FnMut(&InferenceRequest) -> ExecutorAdmissionProbeOutcome,
    ) -> Result<Option<BatchPlan>> {
        self.create_iteration_batch_with_admission(
            hint,
            WaitingAdmissionMode::Dynamic {
                wake,
                probe,
                observer: None,
            },
        )
    }

    pub fn next_batch_with_dynamic_admission_observed(
        &self,
        hint: BatchHint,
        wake: AdmissionWakeSnapshot<'_>,
        probe: &mut dyn FnMut(&InferenceRequest) -> ExecutorAdmissionProbeOutcome,
        observer: &mut dyn FnMut(ExecutorAdmissionQueueObservation),
    ) -> Result<Option<BatchPlan>> {
        self.create_iteration_batch_with_admission(
            hint,
            WaitingAdmissionMode::Dynamic {
                wake,
                probe,
                observer: Some(observer),
            },
        )
    }

    /// Retain dynamically admitted work without constructing an execution
    /// batch. The engine uses this bounded phase to converge backing growth
    /// for a fill-first cohort before any participant is submitted. Capacity
    /// and pressure limits remain scheduler-owned, so preparing a cohort can
    /// never reserve more request authorities than a normal admission tick.
    pub fn prepare_dynamic_admission_observed(
        &self,
        maximum_admissions: usize,
        wake: AdmissionWakeSnapshot<'_>,
        probe: &mut dyn FnMut(&InferenceRequest) -> ExecutorAdmissionProbeOutcome,
        observer: &mut dyn FnMut(ExecutorAdmissionQueueObservation),
    ) -> Result<AdmissionTickReceipt> {
        let active_capacity = self
            .config
            .max_running_requests
            .saturating_sub(self.active_count());
        let decode_capacity = self
            .cb_config
            .max_decode_batch
            .saturating_sub(self.decoding_count());
        let available_slots = active_capacity.min(decode_capacity);
        let available_slots = self
            .capacity_backpressure_admit_limit()
            .map(|limit| available_slots.min(limit))
            .unwrap_or(available_slots)
            .min(maximum_admissions);
        let mut mode = WaitingAdmissionMode::Dynamic {
            wake,
            probe,
            observer: Some(observer),
        };
        self.admit_waiting_dynamically(available_slots, available_slots, &mut mode)
    }

    /// Maximum waiting-prefix width that can join the next fill-first prefill
    /// batch without exceeding its typed request or token budget. Existing
    /// admitted prefills consume the budget first; a waiting request that does
    /// not fit seals the fair prefix instead of reserving unused authority.
    pub fn fill_first_dynamic_admission_limit(&self, hint: &BatchHint, target: usize) -> usize {
        let mut remaining_tokens = hint.max_tokens;
        let prefill_step_chunk = self.runtime_config.prefill_step_chunk;
        let prefill_queue = self.prefill_queue.read();
        for request in prefill_queue.iter().take(hint.max_batch_size) {
            let tokens =
                self.prefill_budget_tokens(request, None, prefill_step_chunk, remaining_tokens);
            if tokens == 0 || tokens > remaining_tokens {
                return 0;
            }
            remaining_tokens -= tokens;
        }
        drop(prefill_queue);

        let mut limit = target
            .saturating_sub(self.active_count())
            .min(hint.max_batch_size);
        if limit == 0 || remaining_tokens == 0 {
            return 0;
        }
        let waiting = self.waiting_queue.read();
        let mut admitted_tokens = 0usize;
        let mut admitted = 0usize;
        for request in waiting.iter() {
            if admitted >= limit {
                break;
            }
            let available = remaining_tokens.saturating_sub(admitted_tokens);
            let tokens = self.prefill_budget_tokens(request, None, prefill_step_chunk, available);
            if tokens == 0 || tokens > available {
                break;
            }
            admitted_tokens += tokens;
            admitted += 1;
        }
        limit = limit.min(admitted);
        limit
    }

    fn capacity_backpressure_admit_limit(&self) -> Option<usize> {
        Self::read_backpressure_limit(&self.capacity_backpressure_limit)
    }

    fn decode_capacity_backpressure_limit(&self) -> Option<usize> {
        Self::read_backpressure_limit(&self.decode_capacity_backpressure_limit)
    }

    fn read_backpressure_limit(limit: &AtomicUsize) -> Option<usize> {
        let limit = limit.load(Ordering::Relaxed);
        if limit == NO_CAPACITY_BACKPRESSURE_LIMIT {
            None
        } else {
            Some(limit.max(1))
        }
    }

    fn capacity_blocked_waiting_len(&self) -> usize {
        let has_active_requests = self.active_count() > 0;
        let release_epoch = self.capacity_release_epoch.load(Ordering::Relaxed);
        self.waiting_queue
            .read()
            .iter()
            .filter(|req| {
                req.capacity_deferred_until_release_epoch > release_epoch
                    && (has_active_requests
                        || req.capacity_deferred_empty_retry_epoch == Some(release_epoch))
            })
            .count()
    }

    fn decode_capacity_deferred_backlog_len(&self) -> usize {
        let waiting = self
            .waiting_queue
            .read()
            .iter()
            .filter(|req| req.capacity_deferred_from_decode)
            .count();
        let prefilling = self
            .prefill_queue
            .read()
            .iter()
            .filter(|req| req.capacity_deferred_from_decode)
            .count();
        waiting + prefilling
    }

    fn record_capacity_defer_feedback(&self, attempted_prefill_width: usize) {
        self.capacity_deferred_counter
            .fetch_add(1, Ordering::Relaxed);

        let iteration = self.current_iteration.load(Ordering::Relaxed);
        let previous_iteration = self
            .capacity_backpressure_iteration
            .swap(iteration, Ordering::Relaxed);
        if previous_iteration == iteration {
            return;
        }

        let max_running = self.config.max_running_requests.max(1);
        let proposed = attempted_prefill_width
            .max(1)
            .div_ceil(2)
            .max(1)
            .min(max_running);
        let _ = self.capacity_backpressure_limit.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |current| {
                let current = if current == NO_CAPACITY_BACKPRESSURE_LIMIT {
                    max_running
                } else {
                    current.max(1).min(max_running)
                };
                let next = proposed.min(current).max(1);
                if next >= max_running {
                    Some(NO_CAPACITY_BACKPRESSURE_LIMIT)
                } else {
                    Some(next)
                }
            },
        );
    }

    fn decode_capacity_pressure_limit(
        attempted_decode_width: usize,
        observed_free_blocks: Option<usize>,
        max_running: usize,
    ) -> usize {
        let attempted = attempted_decode_width.max(1).min(max_running);
        let half_width = attempted.div_ceil(2).max(1);
        let near_fit_width = observed_free_blocks
            .filter(|free_blocks| *free_blocks > 0)
            .map(|free_blocks| {
                let usable_free_blocks =
                    free_blocks.saturating_sub(CAPACITY_DECODE_FREE_BLOCK_HEADROOM);
                usable_free_blocks
                    .max(1)
                    .min(attempted.saturating_sub(1).max(1))
            });
        near_fit_width
            .unwrap_or(half_width)
            .max(half_width)
            .min(max_running)
    }

    pub fn record_decode_capacity_pressure(
        &self,
        attempted_decode_width: usize,
        observed_free_blocks: Option<usize>,
    ) {
        let _feedback = self.decode_capacity_feedback_lock.lock();
        self.record_decode_capacity_pressure_inner(attempted_decode_width, observed_free_blocks);
    }

    fn record_decode_capacity_pressure_inner(
        &self,
        attempted_decode_width: usize,
        observed_free_blocks: Option<usize>,
    ) {
        let max_running = self.config.max_running_requests.max(1);
        let proposed = Self::decode_capacity_pressure_limit(
            attempted_decode_width,
            observed_free_blocks,
            max_running,
        );
        let _ = self.decode_capacity_backpressure_limit.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |current| {
                let current = if current == NO_CAPACITY_BACKPRESSURE_LIMIT {
                    max_running
                } else {
                    current.max(1).min(max_running)
                };
                let next = proposed.min(current).max(1);
                if next >= max_running {
                    Some(NO_CAPACITY_BACKPRESSURE_LIMIT)
                } else {
                    Some(next)
                }
            },
        );
    }

    pub fn record_decode_execution_capacity_pressure(&self, attempted_decode_width: usize) {
        let _feedback = self.decode_capacity_feedback_lock.lock();
        self.record_decode_capacity_pressure_inner(attempted_decode_width, None);
        self.decode_execution_pressure_enforced
            .store(true, Ordering::Release);
    }

    /// Recover an execution-scoped decode limit only from an authoritative
    /// root-cohort success. Capacity failures use multiplicative decrease;
    /// successful saturated cohorts recover additively so one completion does
    /// not immediately recreate the oversized submission wave that failed.
    pub fn record_decode_execution_capacity_success(&self, successful_decode_width: usize) -> bool {
        if successful_decode_width == 0 {
            return false;
        }
        let _feedback = self.decode_capacity_feedback_lock.lock();
        if !self
            .decode_execution_pressure_enforced
            .load(Ordering::Acquire)
        {
            return false;
        }

        let max_running = self.config.max_running_requests.max(1);
        let successful_decode_width = successful_decode_width.max(1).min(max_running);
        let relaxed = self
            .decode_capacity_backpressure_limit
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                if current == NO_CAPACITY_BACKPRESSURE_LIMIT {
                    return Some(NO_CAPACITY_BACKPRESSURE_LIMIT);
                }
                let current = current.max(1).min(max_running);
                if successful_decode_width < current {
                    return None;
                }
                let next = current.saturating_add(1).min(max_running);
                Some(if next >= max_running {
                    NO_CAPACITY_BACKPRESSURE_LIMIT
                } else {
                    next
                })
            })
            .is_ok();

        if relaxed && self.decode_capacity_backpressure_limit().is_none() {
            self.decode_execution_pressure_enforced
                .store(false, Ordering::Release);
        }
        relaxed
    }

    /// Route an active prefill failure through the phase-independent pressure
    /// coordinator.
    pub fn defer_prefill_for_execution_capacity(
        &self,
        request_id: &RequestId,
        deferral: AdmissionDeferral,
        release_snapshot: &ExecutionCapacityReleaseSnapshot,
    ) -> Result<ExecutionCapacityAction> {
        self.plan_execution_capacity_pressure(
            std::slice::from_ref(request_id),
            deferral,
            release_snapshot,
        )
    }

    /// Route active decode failures through the same logical work frontier as
    /// prefill/recompute failures.
    pub fn defer_decode_for_execution_capacity(
        &self,
        request_ids: &[RequestId],
        deferral: AdmissionDeferral,
        release_snapshot: &ExecutionCapacityReleaseSnapshot,
    ) -> Result<ExecutionCapacityAction> {
        self.plan_execution_capacity_pressure(request_ids, deferral, release_snapshot)
    }

    /// Suspend only the exact active frontiers blocked by a non-capacity
    /// execution dependency. The caller must already own a live waiter before
    /// installing this ticket and must drive the returned wake to one terminal
    /// state. Installation is all-or-nothing across the cohort.
    pub fn defer_for_execution_readiness(
        &self,
        request_ids: &[RequestId],
    ) -> Result<ExecutionReadinessDeferralReceipt> {
        let requested = request_ids.iter().collect::<HashSet<_>>();
        if request_ids.is_empty() || requested.len() != request_ids.len() {
            return Err(FerrumError::scheduler(
                "execution readiness deferral requires a non-empty unique cohort",
            ));
        }
        let ticket_value = self
            .next_execution_readiness_ticket
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1)
            })
            .map_err(|_| FerrumError::scheduler("execution readiness ticket space exhausted"))?;
        let ticket_id = NonZeroU64::new(ticket_value)
            .ok_or_else(|| FerrumError::scheduler("execution readiness issued a zero ticket id"))?;
        let state = Arc::new(ExecutionReadinessState {
            status: AtomicU8::new(EXECUTION_READINESS_PENDING),
        });
        let block = ExecutionReadinessBlock {
            ticket_id,
            state: Arc::clone(&state),
        };

        let mut prefill = self.prefill_queue.write();
        let mut decode = self.decode_queue.write();
        let existing = prefill
            .iter()
            .chain(decode.requests.values())
            .find(|request| {
                requested.contains(&request.inner.request.id)
                    && request.execution_readiness_block.is_some()
            });
        if let Some(request) = existing {
            return Err(FerrumError::scheduler(format!(
                "request {} already owns an execution readiness ticket",
                request.inner.request.id
            )));
        }
        let mut installed = 0usize;
        for request in prefill.iter_mut().chain(decode.requests.values_mut()) {
            if requested.contains(&request.inner.request.id) {
                request.execution_readiness_block = Some(block.clone());
                installed += 1;
            }
        }
        if installed != request_ids.len() {
            Self::rollback_execution_readiness_install(&mut prefill, &mut decode, &block);
            return Err(FerrumError::scheduler(format!(
                "execution readiness retained {installed} of {} active frontiers",
                request_ids.len()
            )));
        }
        drop(decode);
        drop(prefill);
        self.execution_readiness_deferred_counter
            .fetch_add(installed as u64, Ordering::Relaxed);
        Ok(ExecutionReadinessDeferralReceipt {
            deferred_count: installed,
            wake: ExecutionReadinessWake { ticket_id, state },
        })
    }

    fn rollback_execution_readiness_install(
        prefill: &mut VecDeque<ContinuousBatchRequest>,
        decode: &mut DecodeQueueState,
        block: &ExecutionReadinessBlock,
    ) {
        for request in prefill.iter_mut() {
            if request
                .execution_readiness_block
                .as_ref()
                .is_some_and(|installed| installed.matches(block))
            {
                request.execution_readiness_block = None;
            }
        }
        for request in decode.requests.values_mut() {
            if request
                .execution_readiness_block
                .as_ref()
                .is_some_and(|installed| installed.matches(block))
            {
                request.execution_readiness_block = None;
            }
        }
    }

    /// Yield active frontiers after the executor has committed its bounded
    /// backing-maintenance budget. One complete scheduler iteration must pass
    /// before these frontiers are eligible again, so peers retain fairness and
    /// a lone request cannot turn maintenance into a tight retry loop.
    pub fn defer_retry_after_execution_maintenance(
        &self,
        retry: &ExecutorExecutionMaintenanceRetry,
    ) -> Result<ExecutionMaintenanceRetryReceipt> {
        let progress = retry.progress();
        if progress.mutations().is_empty() {
            return Err(FerrumError::scheduler(
                "execution maintenance retry requires physical mutations",
            ));
        }
        self.defer_retry_after_execution_maintenance_epoch(
            retry.affected_request_ids(),
            progress.latest_capacity_epoch(),
        )
    }

    fn defer_retry_after_execution_maintenance_epoch(
        &self,
        request_ids: &[RequestId],
        latest_capacity_epoch: u64,
    ) -> Result<ExecutionMaintenanceRetryReceipt> {
        if request_ids.is_empty() || latest_capacity_epoch == 0 {
            return Err(FerrumError::scheduler(
                "execution maintenance retry requires active requests and a physical mutation epoch",
            ));
        }
        let requested = request_ids.iter().cloned().collect::<HashSet<_>>();
        if requested.len() != request_ids.len() {
            return Err(FerrumError::scheduler(
                "execution maintenance retry contains duplicate request identities",
            ));
        }

        // Hold both active queues across validation and mutation. This makes
        // ticket installation all-or-nothing even if an index/queue invariant
        // has already been violated by another lifecycle transition.
        let mut prefill = self.prefill_queue.write();
        let mut decode = self.decode_queue.write();
        let request_index = self.request_index.read();
        let validate = |request: &ContinuousBatchRequest| -> Result<()> {
            if request.execution_capacity_deferral.is_some()
                || request.execution_maintenance_retry.is_some()
                || request
                    .last_execution_maintenance_capacity_epoch
                    .is_some_and(|epoch| epoch >= latest_capacity_epoch)
            {
                return Err(FerrumError::scheduler(
                    "execution maintenance retry reuses stale or concurrently blocked evidence",
                ));
            }
            Ok(())
        };
        for request_id in &requested {
            let prefill_matches = prefill
                .iter()
                .filter(|request| request.inner.request.id == *request_id)
                .count();
            let decode_request = decode.requests.get(request_id);
            let request = match request_index.get(request_id) {
                Some(RequestPhase::Prefilling)
                    if prefill_matches == 1 && decode_request.is_none() =>
                {
                    prefill
                        .iter()
                        .find(|request| request.inner.request.id == *request_id)
                        .expect("validated prefill frontier remains locked")
                }
                Some(RequestPhase::Decoding)
                    if prefill_matches == 0 && decode_request.is_some() =>
                {
                    decode_request.expect("validated decode frontier remains locked")
                }
                _ => {
                    return Err(FerrumError::scheduler(
                        "execution maintenance retry lost an exact active logical frontier",
                    ));
                }
            };
            validate(request)?;
        }

        // `current_iteration` points at the next scheduler iteration after the
        // batch that just yielded. Advancing once more skips exactly that next
        // iteration and makes the frontier eligible in the following one.
        let not_before_iteration = self
            .current_iteration
            .load(Ordering::Relaxed)
            .saturating_add(1);
        let ticket = ExecutionMaintenanceRetryTicket {
            not_before_iteration,
            latest_capacity_epoch,
        };
        let mut deferred_count = 0;
        for request in prefill.iter_mut() {
            if requested.contains(&request.inner.request.id) {
                request.execution_maintenance_retry = Some(ticket);
                request.last_execution_maintenance_capacity_epoch = Some(latest_capacity_epoch);
                deferred_count += 1;
            }
        }
        for request in decode.requests.values_mut() {
            if requested.contains(&request.inner.request.id) {
                request.execution_maintenance_retry = Some(ticket);
                request.last_execution_maintenance_capacity_epoch = Some(latest_capacity_epoch);
                deferred_count += 1;
            }
        }
        if deferred_count != request_ids.len() {
            return Err(FerrumError::scheduler(format!(
                "execution maintenance retry retained {deferred_count} of {} active frontiers",
                request_ids.len()
            )));
        }

        Ok(ExecutionMaintenanceRetryReceipt {
            deferred_count,
            not_before_iteration,
            latest_capacity_epoch,
        })
    }

    fn plan_execution_capacity_pressure(
        &self,
        request_ids: &[RequestId],
        deferral: AdmissionDeferral,
        release_snapshot: &ExecutionCapacityReleaseSnapshot,
    ) -> Result<ExecutionCapacityAction> {
        if deferral.action() != ferrum_interfaces::vnext::DeferredAction::WaitForRelease {
            return Err(FerrumError::scheduler(
                "active execution-capacity deferral must wait for release",
            ));
        }
        let active_ids = {
            let request_index = self.request_index.read();
            request_ids
                .iter()
                .filter(|request_id| {
                    matches!(
                        request_index.get(*request_id),
                        Some(RequestPhase::Prefilling | RequestPhase::Decoding)
                    )
                })
                .cloned()
                .collect::<Vec<_>>()
        };
        if active_ids.is_empty() {
            return Ok(ExecutionCapacityAction::Deferred { count: 0 });
        }

        let candidates =
            self.execution_capacity_candidates(release_snapshot, deferral.wait_condition());
        let decision = {
            let mut coordinator = self.pressure_coordinator.lock();
            let decision = coordinator
                .plan_failure(&active_ids, deferral.wait_condition(), &candidates)
                .map_err(|error| FerrumError::scheduler(error.to_string()))?;
            self.pressure_active
                .store(coordinator.has_records(), Ordering::Release);
            decision
        };

        match decision {
            PressureDecision::Deferred { count, .. } => {
                let installed =
                    self.install_execution_capacity_deferral(&active_ids, &deferral, None);
                self.capacity_deferred_counter
                    .fetch_add(installed as u64, Ordering::Relaxed);
                if installed != count {
                    return Err(FerrumError::scheduler(format!(
                        "execution-capacity deferral retained {installed} of {count} active frontiers"
                    )));
                }
                Ok(ExecutionCapacityAction::Deferred { count: installed })
            }
            PressureDecision::YieldPlanned(transaction) => {
                let installed = self.install_execution_capacity_deferral(
                    &active_ids,
                    &deferral,
                    Some(transaction.victim_request_id()),
                );
                self.capacity_deferred_counter
                    .fetch_add(installed as u64, Ordering::Relaxed);
                Ok(ExecutionCapacityAction::YieldPlanned { transaction })
            }
            PressureDecision::InvariantViolation(violation) => {
                Ok(ExecutionCapacityAction::InvariantViolation { violation })
            }
        }
    }

    fn execution_capacity_candidates(
        &self,
        release_snapshot: &ExecutionCapacityReleaseSnapshot,
        condition: &CapacityWaitCondition,
    ) -> Vec<PressureCandidate> {
        let mut candidates = Vec::new();
        {
            let prefill = self.prefill_queue.read();
            candidates.extend(prefill.iter().map(|request| {
                PressureCandidate {
                    request_id: request.inner.request.id.clone(),
                    work_kind: request.logical_work_frontier.work_kind(),
                    priority: request.inner.request.priority,
                    progress: request.logical_work_frontier.progress_generation(),
                    recompute_cost: request.logical_work_frontier.recompute_cost(),
                    advances_wait_source: release_snapshot
                        .can_advance(&request.inner.request.id, condition),
                    blocked_on: request
                        .execution_capacity_deferral
                        .as_ref()
                        .map(|deferral| deferral.wait_condition().clone()),
                }
            }));
        }
        {
            let decode = self.decode_queue.read();
            candidates.extend(decode.requests.values().map(|request| {
                PressureCandidate {
                    request_id: request.inner.request.id.clone(),
                    work_kind: request.logical_work_frontier.work_kind(),
                    priority: request.inner.request.priority,
                    progress: request.logical_work_frontier.progress_generation(),
                    recompute_cost: request.logical_work_frontier.recompute_cost(),
                    advances_wait_source: release_snapshot
                        .can_advance(&request.inner.request.id, condition),
                    blocked_on: request
                        .execution_capacity_deferral
                        .as_ref()
                        .map(|deferral| deferral.wait_condition().clone()),
                }
            }));
        }
        if self.pressure_active.load(Ordering::Acquire) {
            let waiting = self.waiting_queue.read();
            candidates.extend(waiting.iter().map(|request| {
                PressureCandidate {
                    request_id: request.inner.request.id.clone(),
                    work_kind: request.logical_work_frontier.work_kind(),
                    priority: request.inner.request.priority,
                    progress: request.logical_work_frontier.progress_generation(),
                    recompute_cost: request.logical_work_frontier.recompute_cost(),
                    advances_wait_source: false,
                    blocked_on: request
                        .execution_capacity_deferral
                        .as_ref()
                        .map(|deferral| deferral.wait_condition().clone()),
                }
            }));
        }
        candidates
    }

    fn install_execution_capacity_deferral(
        &self,
        request_ids: &[RequestId],
        deferral: &AdmissionDeferral,
        yielding: Option<&RequestId>,
    ) -> usize {
        let requested = request_ids.iter().collect::<HashSet<_>>();
        let mut installed = 0usize;
        {
            let mut prefill = self.prefill_queue.write();
            for request in prefill.iter_mut() {
                let request_id = &request.inner.request.id;
                if requested.contains(request_id) && yielding != Some(request_id) {
                    request.execution_capacity_deferral = Some(deferral.clone());
                    installed += 1;
                }
            }
        }
        {
            let mut decode = self.decode_queue.write();
            for request in decode.requests.values_mut() {
                let request_id = &request.inner.request.id;
                if requested.contains(request_id) && yielding != Some(request_id) {
                    request.execution_capacity_deferral = Some(deferral.clone());
                    installed += 1;
                }
            }
        }
        installed
    }

    fn relax_backpressure_limit(limit: &AtomicUsize, max_running: usize) {
        let current = limit.load(Ordering::Relaxed);
        if current == NO_CAPACITY_BACKPRESSURE_LIMIT {
            return;
        }

        let current = current.max(1).min(max_running);
        let grown = current.saturating_mul(2).min(max_running);
        let next = if grown >= max_running {
            NO_CAPACITY_BACKPRESSURE_LIMIT
        } else {
            grown.max(1)
        };
        limit.store(next, Ordering::Relaxed);
    }

    fn record_resource_progress(&self) {
        let max_running = self.config.max_running_requests.max(1);
        Self::relax_backpressure_limit(&self.capacity_backpressure_limit, max_running);
        let _feedback = self.decode_capacity_feedback_lock.lock();
        if !self
            .decode_execution_pressure_enforced
            .load(Ordering::Acquire)
        {
            Self::relax_backpressure_limit(&self.decode_capacity_backpressure_limit, max_running);
        }
    }

    fn record_capacity_release_progress(&self) {
        self.capacity_release_epoch.fetch_add(1, Ordering::Relaxed);
        self.capacity_mixed_recompute_epoch
            .fetch_add(1, Ordering::Relaxed);
        self.capacity_mixed_recompute_required_blocks_per_slot
            .store(0, Ordering::Relaxed);
        self.capacity_mixed_recompute_observed_free_blocks
            .store(usize::MAX, Ordering::Relaxed);
        self.record_resource_progress();
    }

    /// Record physical capacity released outside an active scheduler queue.
    pub fn record_external_capacity_release(&self) {
        self.record_capacity_release_progress();
    }

    fn record_capacity_recompute_progress(&self) {
        self.capacity_mixed_recompute_epoch
            .fetch_add(1, Ordering::Relaxed);
    }

    fn capacity_mixed_recompute_usable_free_blocks(
        observed_free_blocks: usize,
        required_blocks_per_slot: usize,
    ) -> usize {
        if required_blocks_per_slot == 0 || observed_free_blocks == usize::MAX {
            return observed_free_blocks;
        }
        observed_free_blocks.saturating_sub(CAPACITY_MIXED_RECOMPUTE_FREE_BLOCK_HEADROOM)
    }

    pub fn record_capacity_deferred_mixed_recompute_release_evidence(&self) {
        self.capacity_mixed_recompute_required_blocks_per_slot
            .store(0, Ordering::Relaxed);
        self.capacity_mixed_recompute_observed_free_blocks
            .store(usize::MAX, Ordering::Relaxed);
        self.record_capacity_recompute_progress();
    }

    pub fn record_capacity_deferred_mixed_recompute_kv_capacity_snapshot(
        &self,
        free_blocks: usize,
    ) {
        self.capacity_mixed_recompute_observed_free_blocks
            .store(free_blocks, Ordering::Relaxed);
        let required_blocks_per_slot = self
            .capacity_mixed_recompute_required_blocks_per_slot
            .load(Ordering::Relaxed);
        let usable_free_blocks = Self::capacity_mixed_recompute_usable_free_blocks(
            free_blocks,
            required_blocks_per_slot,
        );
        if required_blocks_per_slot > 0 && usable_free_blocks >= required_blocks_per_slot {
            self.record_capacity_recompute_progress();
        }
    }

    /// Suppress release-blocked mixed recompute until fresh capacity evidence.
    ///
    /// The engine calls this after a mixed decode+recompute KV admission
    /// failure. Trying a different blocked recompute candidate in the same
    /// capacity evidence epoch cannot create free KV blocks; it only repeats
    /// the failed unified admission overhead.
    pub fn defer_capacity_deferred_mixed_recompute_until_release(&self) {
        self.capacity_mixed_recompute_required_blocks_per_slot
            .store(0, Ordering::Relaxed);
        self.capacity_mixed_recompute_observed_free_blocks
            .store(usize::MAX, Ordering::Relaxed);
        self.defer_capacity_deferred_mixed_recompute_until_kv_capacity(None, None, None);
    }

    /// Suppress release-blocked mixed recompute until enough KV capacity exists.
    ///
    /// When paged-KV admission returns structured pressure, the engine passes
    /// the failed batch's admission blocks, attempted prefill width, and
    /// observed free-block count here. Decode recompute reopens once a later
    /// capacity snapshot can fit at least one bounded recompute, and its
    /// per-iteration width is paced by the same per-slot estimate. This avoids
    /// both blind same-pressure retries and waiting for enough free blocks to
    /// replay the entire failed mixed batch at once.
    pub fn defer_capacity_deferred_mixed_recompute_until_kv_capacity(
        &self,
        required_admission_blocks: Option<usize>,
        observed_free_blocks: Option<usize>,
        attempted_prefill_width: Option<usize>,
    ) {
        let mixed_epoch = self.capacity_mixed_recompute_epoch.load(Ordering::Relaxed);
        let blocked_until = mixed_epoch.saturating_add(1);
        let mut required_blocks_per_slot_for_feedback = None;
        if let Some(required) = required_admission_blocks.filter(|required| *required > 0) {
            let width = attempted_prefill_width.unwrap_or(1).max(1);
            let required_blocks_per_slot = required.div_ceil(width).max(1);
            required_blocks_per_slot_for_feedback = Some(required_blocks_per_slot);
            self.capacity_mixed_recompute_required_blocks_per_slot
                .store(required_blocks_per_slot, Ordering::Relaxed);
        }
        if let Some(observed) = observed_free_blocks {
            self.capacity_mixed_recompute_observed_free_blocks
                .store(observed, Ordering::Relaxed);
            if let Some(required_blocks_per_slot) = required_blocks_per_slot_for_feedback {
                let usable_free_blocks = Self::capacity_mixed_recompute_usable_free_blocks(
                    observed,
                    required_blocks_per_slot,
                );
                if usable_free_blocks >= required_blocks_per_slot {
                    self.record_capacity_recompute_progress();
                }
            }
        }
        let _ = self
            .capacity_mixed_recompute_blocked_until_epoch
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.max(blocked_until))
            });
    }

    /// Move a capacity-deferred prefill back to the waiting queue.
    ///
    /// The engine uses this when it could not allocate physical KV or
    /// recurrent state for a prefill. Leaving the request in `prefill_queue`
    /// would make `next_batch` schedule the same un-runnable work every
    /// iteration, which can starve decode and spin the scheduler.
    pub fn defer_prefill_to_waiting(&self, request_id: &RequestId) -> bool {
        let mut prefill_queue = self.prefill_queue.write();
        let mut waiting_queue = self.waiting_queue.write();
        let mut request_index = self.request_index.write();
        let attempted_prefill_width = prefill_queue.len();

        if let Some(pos) = prefill_queue
            .iter()
            .position(|r| r.inner.request.id == *request_id)
        {
            let mut req = prefill_queue.remove(pos).unwrap();
            req.phase = RequestPhase::Waiting;
            req.inner.state = RequestState::Waiting;
            req.inner.started_at = None;
            req.prefill_tokens = 0;
            req.kv_blocks.clear();
            req.chunked_prefill = false;
            req.prefill_chunk_offset = 0;
            req.prefill_execution_chunk_ceiling = None;
            req.logical_work_frontier.yield_for_recompute();
            req.capacity_deferred_until_release_epoch = self
                .capacity_release_epoch
                .load(Ordering::Relaxed)
                .saturating_add(1);
            if !self.decode_queue.read().requests.is_empty() {
                req.capacity_deferred_mixed_attempt_epoch =
                    Some(self.capacity_mixed_recompute_epoch.load(Ordering::Relaxed));
            }
            req.last_iteration = self.current_iteration.load(Ordering::Relaxed);
            if !self.requeue_waiting_request(&mut waiting_queue, &mut request_index, req) {
                return false;
            }
            self.record_capacity_defer_feedback(attempted_prefill_width);
            debug!("Deferred prefill request {} back to waiting", request_id);
            true
        } else {
            false
        }
    }

    /// Move a capacity-deferred decode request back to waiting for KV recompute.
    ///
    /// The engine calls this after releasing the request's physical KV/cache
    /// state. Logical output lives in the engine sequence state; scheduler
    /// token counters are reset so the next prefill rebuilds from that logical
    /// context instead of resuming the stale physical decode phase.
    pub fn defer_decode_to_waiting_for_capacity(
        &self,
        request_id: &RequestId,
        attempted_decode_width: usize,
    ) -> bool {
        self.defer_decode_to_waiting_for_capacity_with_pressure(
            request_id,
            attempted_decode_width,
            None,
        )
    }

    pub fn defer_decode_to_waiting_for_capacity_with_pressure(
        &self,
        request_id: &RequestId,
        attempted_decode_width: usize,
        observed_free_blocks: Option<usize>,
    ) -> bool {
        self.defer_decode_to_waiting_for_capacity_inner(
            request_id,
            attempted_decode_width,
            observed_free_blocks,
        )
    }

    /// Mark the planned yield as owning the physical release obligation.
    pub fn arm_execution_capacity_yield(
        &self,
        transaction: &PressureYieldTransaction,
    ) -> Result<PressureTransitionOrdinal> {
        let ordinal = self
            .pressure_coordinator
            .lock()
            .arm_release_fence(transaction)
            .map_err(|error| FerrumError::scheduler(error.to_string()))?;
        self.pressure_active.store(true, Ordering::Release);
        Ok(ordinal)
    }

    /// Complete a phase-independent yield after the engine has released all
    /// physical resources and its release fence reached terminal state.
    pub fn complete_execution_capacity_yield(
        &self,
        transaction: &PressureYieldTransaction,
        attempted_decode_width: usize,
        observed_free_blocks: Option<usize>,
    ) -> Result<ExecutionCapacityYieldCompletion> {
        let request_id = transaction.victim_request_id();
        let victim_waiting_ticket = self.requeue_execution_capacity_victim(
            request_id,
            attempted_decode_width,
            observed_free_blocks,
        );
        let requeued = victim_waiting_ticket.is_some();
        let progress_owner_wait_condition =
            self.execution_capacity_wait_condition(transaction.progress_owner_id());

        let (release_ordinal, disposition, installed_hold) = {
            let mut coordinator = self.pressure_coordinator.lock();
            if !requeued && transaction.kind() == PressureYieldKind::SelfRecompute {
                let _ = coordinator
                    .record_terminal(request_id)
                    .map_err(|error| FerrumError::scheduler(error.to_string()))?;
            }
            let completion = coordinator
                .complete_release_fence(transaction, progress_owner_wait_condition.as_ref())
                .map_err(|error| FerrumError::scheduler(error.to_string()))?;
            if !requeued && transaction.kind() == PressureYieldKind::PeerHandoff {
                let _ = coordinator
                    .record_terminal(request_id)
                    .map_err(|error| FerrumError::scheduler(error.to_string()))?;
            }
            self.pressure_active
                .store(coordinator.has_records(), Ordering::Release);
            let installed_hold = match (coordinator.hold_status(request_id), victim_waiting_ticket)
            {
                (
                    PressureHoldStatus::Held {
                        episode_id,
                        progress_owner_id,
                        progress_baseline,
                        progress_current,
                    },
                    Some(waiting_ticket),
                ) => Some(ExecutionCapacityPressureHoldReceipt {
                    episode_id,
                    transition_ordinal: completion.0,
                    request_id: request_id.clone(),
                    progress_owner_id,
                    progress_baseline,
                    progress_current,
                    waiting_ticket,
                }),
                _ => None,
            };
            (completion.0, completion.1, installed_hold)
        };
        let (
            resumable_transition_ordinal,
            owner_admission_pending_transition_ordinal,
            closed_transition_ordinal,
            disposition,
        ) = match disposition {
            PressureReleaseFenceDisposition::Resumable(ordinal) => (
                Some(ordinal),
                None,
                None,
                ExecutionCapacityYieldDisposition::ProgressOwnerResumable,
            ),
            PressureReleaseFenceDisposition::OwnerAdmissionPending(ordinal) => (
                None,
                Some(ordinal),
                None,
                ExecutionCapacityYieldDisposition::ProgressOwnerAdmissionPending,
            ),
            PressureReleaseFenceDisposition::SelfRecomputeQueued(ordinal) => (
                None,
                None,
                Some(ordinal),
                ExecutionCapacityYieldDisposition::SelfRecomputeQueued,
            ),
            PressureReleaseFenceDisposition::Closed { ordinal, reason } => {
                let disposition = match reason {
                    PressureHoldReleaseReason::OwnerTerminal => {
                        ExecutionCapacityYieldDisposition::OwnerTerminal
                    }
                };
                (None, None, Some(ordinal), disposition)
            }
        };
        Ok(ExecutionCapacityYieldCompletion {
            victim_requeued: requeued,
            installed_hold,
            release_transition_ordinal: release_ordinal,
            resumable_transition_ordinal,
            owner_admission_pending_transition_ordinal,
            closed_transition_ordinal,
            disposition,
        })
    }

    fn execution_capacity_wait_condition(
        &self,
        request_id: &RequestId,
    ) -> Option<CapacityWaitCondition> {
        if let Some(condition) = self
            .prefill_queue
            .read()
            .iter()
            .find(|request| request.inner.request.id == *request_id)
            .and_then(|request| request.execution_capacity_deferral.as_ref())
            .map(|deferral| deferral.wait_condition().clone())
        {
            return Some(condition);
        }
        if let Some(condition) = self
            .decode_queue
            .read()
            .requests
            .get(request_id)
            .and_then(|request| request.execution_capacity_deferral.as_ref())
            .map(|deferral| deferral.wait_condition().clone())
        {
            return Some(condition);
        }
        self.waiting_queue
            .read()
            .iter()
            .find(|request| request.inner.request.id == *request_id)
            .and_then(|request| request.execution_capacity_deferral.as_ref())
            .map(|deferral| deferral.wait_condition().clone())
    }

    /// Resolve every planned-yield error path so a failed physical release
    /// cannot leave the scheduler claiming a pending fence forever.
    pub fn abort_execution_capacity_yield(
        &self,
        transaction: &PressureYieldTransaction,
        victim_released: bool,
        attempted_decode_width: usize,
        observed_free_blocks: Option<usize>,
    ) -> Result<(bool, PressureTransitionOrdinal, PressureTransitionOrdinal)> {
        let (aborted_ordinal, closed_ordinal, participants) = {
            let mut coordinator = self.pressure_coordinator.lock();
            let (aborted, closed, participants) = coordinator
                .abort_yield(transaction)
                .map_err(|error| FerrumError::scheduler(error.to_string()))?;
            self.pressure_active
                .store(coordinator.has_records(), Ordering::Release);
            (aborted, closed, participants)
        };
        let participants = participants.into_iter().collect::<HashSet<_>>();
        for request in self.prefill_queue.write().iter_mut() {
            if participants.contains(&request.inner.request.id) {
                request.execution_capacity_deferral = None;
            }
        }
        for request in self.decode_queue.write().requests.values_mut() {
            if participants.contains(&request.inner.request.id) {
                request.execution_capacity_deferral = None;
            }
        }
        let requeued = victim_released
            && self
                .requeue_execution_capacity_victim(
                    transaction.victim_request_id(),
                    attempted_decode_width,
                    observed_free_blocks,
                )
                .is_some();
        Ok((requeued, aborted_ordinal, closed_ordinal))
    }

    fn requeue_execution_capacity_victim(
        &self,
        request_id: &RequestId,
        attempted_decode_width: usize,
        observed_free_blocks: Option<usize>,
    ) -> Option<u64> {
        let request = {
            let mut prefill = self.prefill_queue.write();
            prefill
                .iter()
                .position(|request| request.inner.request.id == *request_id)
                .and_then(|position| prefill.remove(position))
        }
        .or_else(|| {
            let mut decode = self.decode_queue.write();
            decode.remove(request_id)
        });

        let mut requeued_ticket = None;
        if let Some(mut request) = request {
            let waiting_ticket = request.waiting_admission_ticket;
            request.phase = RequestPhase::Waiting;
            request.inner.state = RequestState::Waiting;
            request.inner.started_at = None;
            request.prefill_tokens = 0;
            request.decode_tokens = 0;
            request.kv_blocks.clear();
            request.chunked_prefill = false;
            request.prefill_chunk_offset = 0;
            request.prefill_execution_chunk_ceiling = None;
            request.capacity_deferred_until_release_epoch = self
                .capacity_release_epoch
                .load(Ordering::Relaxed)
                .saturating_add(1);
            request.capacity_deferred_mixed_attempt_epoch = None;
            request.capacity_deferred_empty_retry_epoch = None;
            request.capacity_deferred_from_decode = true;
            request.execution_capacity_deferral = None;
            request.logical_work_frontier.yield_for_recompute();
            request.last_iteration = self.current_iteration.load(Ordering::Relaxed);

            let mut waiting = self.waiting_queue.write();
            let mut request_index = self.request_index.write();
            if self.requeue_waiting_request(&mut waiting, &mut request_index, request) {
                requeued_ticket = waiting_ticket.map(|ticket| ticket.get());
                self.record_capacity_defer_feedback(attempted_decode_width.max(1));
                self.record_decode_capacity_pressure(
                    attempted_decode_width.max(1),
                    observed_free_blocks,
                );
            }
        }

        requeued_ticket
    }

    fn defer_decode_to_waiting_for_capacity_inner(
        &self,
        request_id: &RequestId,
        attempted_decode_width: usize,
        observed_free_blocks: Option<usize>,
    ) -> bool {
        let mut decode_queue = self.decode_queue.write();
        let mut waiting_queue = self.waiting_queue.write();
        let mut request_index = self.request_index.write();

        if let Some(mut req) = decode_queue.remove(request_id) {
            req.phase = RequestPhase::Waiting;
            req.inner.state = RequestState::Waiting;
            req.inner.started_at = None;
            req.prefill_tokens = 0;
            req.decode_tokens = 0;
            req.kv_blocks.clear();
            req.chunked_prefill = false;
            req.prefill_chunk_offset = 0;
            req.prefill_execution_chunk_ceiling = None;
            req.capacity_deferred_until_release_epoch = self
                .capacity_release_epoch
                .load(Ordering::Relaxed)
                .saturating_add(1);
            req.capacity_deferred_mixed_attempt_epoch = None;
            req.capacity_deferred_empty_retry_epoch = None;
            req.capacity_deferred_from_decode = true;
            req.execution_capacity_deferral = None;
            req.logical_work_frontier.yield_for_recompute();
            req.last_iteration = self.current_iteration.load(Ordering::Relaxed);
            if !self.requeue_waiting_request(&mut waiting_queue, &mut request_index, req) {
                return false;
            }
            self.record_capacity_defer_feedback(attempted_decode_width.max(1));
            self.record_decode_capacity_pressure(
                attempted_decode_width.max(1),
                observed_free_blocks,
            );
            debug!("Deferred decode request {} back to waiting", request_id);
            true
        } else {
            false
        }
    }

    /// Move request from prefill to decode queue
    fn promote_to_decode(&self, request_id: &RequestId) -> bool {
        let mut prefill_queue = self.prefill_queue.write();
        let mut decode_queue = self.decode_queue.write();
        let mut request_index = self.request_index.write();

        if let Some(pos) = prefill_queue
            .iter()
            .position(|r| r.inner.request.id == *request_id)
        {
            let mut req = prefill_queue.remove(pos).unwrap();
            req.phase = RequestPhase::Decoding;
            req.capacity_deferred_until_release_epoch = 0;
            req.capacity_deferred_mixed_attempt_epoch = None;
            req.capacity_deferred_empty_retry_epoch = None;
            req.capacity_deferred_from_decode = false;
            req.execution_capacity_deferral = None;
            req.logical_work_frontier.begin_decode();

            request_index.insert(request_id.clone(), RequestPhase::Decoding);
            decode_queue.requests.insert(request_id.clone(), req);

            debug!("Promoted request {} to decode queue", request_id);
            true
        } else {
            false
        }
    }

    fn initial_prefill_token_estimate(&self, req: &ContinuousBatchRequest) -> usize {
        if !self.runtime_config.prompt_token_estimate {
            return self.cb_config.prefill_chunk_size;
        }

        self.prompt_token_estimate(req)
            .unwrap_or(self.cb_config.prefill_chunk_size)
    }

    fn prompt_token_estimate(&self, req: &ContinuousBatchRequest) -> Option<usize> {
        req.inner
            .request
            .metadata
            .get(PROMPT_TOKENS_METADATA_KEY)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .filter(|&v| v > 0)
    }

    fn default_active_decode_prefill_chunk(&self) -> usize {
        self.cb_config.prefill_chunk_size.div_ceil(8).max(1)
    }

    fn decode_pressure_prefill_cap_threshold(&self, hint: &BatchHint) -> usize {
        hint.max_batch_size
            .min(self.cb_config.max_decode_batch)
            .min(self.config.max_running_requests)
            .max(1)
            .div_ceil(2)
            .max(1)
    }

    fn active_decode_prefill_chunk_for_iteration(
        &self,
        hint: &BatchHint,
        scheduled_decode_count: usize,
    ) -> Option<usize> {
        if scheduled_decode_count == 0 {
            return None;
        }
        if let Some(chunk) = self.runtime_config.active_decode_prefill_chunk {
            return Some(chunk);
        }
        let capacity_deferred_decode_backpressure = self.decode_capacity_deferred_backlog_len() > 0;
        if scheduled_decode_count < self.decode_pressure_prefill_cap_threshold(hint)
            && !capacity_deferred_decode_backpressure
        {
            return None;
        }
        Some(self.default_active_decode_prefill_chunk())
    }

    fn effective_active_decode_prefill_chunk(
        &self,
        active_decode_prefill_chunk: Option<usize>,
        prefill_step_chunk: Option<usize>,
    ) -> Option<usize> {
        let chunk = active_decode_prefill_chunk?;
        Some(
            prefill_step_chunk
                .map(|step_chunk| step_chunk.min(chunk))
                .unwrap_or(chunk)
                .max(1),
        )
    }

    fn active_decode_prefill_target_chunks(
        &self,
        hint: &BatchHint,
        scheduled_decode_count: usize,
        prefill_backlog: usize,
    ) -> usize {
        if scheduled_decode_count == 0 || prefill_backlog == 0 {
            return 0;
        }

        let free_batch_slots = hint.max_batch_size.saturating_sub(scheduled_decode_count);
        if free_batch_slots == 0 {
            return 0;
        }

        // Keep the mixed-prefill lane bounded, but spend real batch headroom.
        // The previous proportional scaling often collapsed to a single tiny
        // chunk at c=32 even when 4-7 batch slots were idle, serializing
        // capacity-deferred recompute behind decode work.
        let max_mixed_prefill_chunks = self.cb_config.max_prefill_batch.div_ceil(2).max(1);
        free_batch_slots
            .min(max_mixed_prefill_chunks)
            .min(prefill_backlog)
    }

    fn maybe_active_decode_prefill_chunk(
        &self,
        req: &ContinuousBatchRequest,
        active_decode_prefill_chunk: Option<usize>,
    ) -> Option<usize> {
        let chunk = active_decode_prefill_chunk?;
        if !req.chunked_prefill && self.decoding_count() == 0 {
            return None;
        }
        Some(chunk)
    }

    fn remaining_prefill_tokens(&self, req: &ContinuousBatchRequest) -> usize {
        if req.prefill_tokens == 0 {
            self.initial_prefill_token_estimate(req)
        } else {
            req.prefill_tokens.saturating_sub(req.prefill_chunk_offset)
        }
    }

    fn chunked_prefill_budget_tokens(&self, req: &ContinuousBatchRequest, chunk: usize) -> usize {
        let remaining = if req.prefill_tokens == 0 {
            self.prompt_token_estimate(req)
                .unwrap_or(self.cb_config.prefill_chunk_size)
        } else {
            req.prefill_tokens.saturating_sub(req.prefill_chunk_offset)
        };
        chunk.min(remaining).max(1)
    }

    fn apply_prefill_execution_chunk_ceiling(req: &ContinuousBatchRequest, tokens: usize) -> usize {
        req.prefill_execution_chunk_ceiling
            .map(|ceiling| tokens.min(ceiling))
            .unwrap_or(tokens)
            .max(1)
    }

    fn prefill_budget_tokens(
        &self,
        req: &ContinuousBatchRequest,
        active_decode_prefill_chunk: Option<usize>,
        prefill_step_chunk: Option<usize>,
        step_tokens_remaining: usize,
    ) -> usize {
        if step_tokens_remaining == 0 {
            return 0;
        }
        if let Some(chunk) =
            self.maybe_active_decode_prefill_chunk(req, active_decode_prefill_chunk)
        {
            let chunk = self
                .effective_active_decode_prefill_chunk(Some(chunk), prefill_step_chunk)
                .unwrap_or(chunk.max(1));
            let tokens = self
                .chunked_prefill_budget_tokens(req, chunk)
                .min(step_tokens_remaining)
                .max(1);
            return Self::apply_prefill_execution_chunk_ceiling(req, tokens);
        }

        let remaining = self.remaining_prefill_tokens(req);
        if let Some(chunk) = prefill_step_chunk {
            let tokens = self
                .chunked_prefill_budget_tokens(req, chunk)
                .min(step_tokens_remaining)
                .max(1);
            return Self::apply_prefill_execution_chunk_ceiling(req, tokens);
        }
        let tokens = if self.cb_config.enable_chunked_prefill {
            remaining.min(step_tokens_remaining).max(1)
        } else {
            remaining.max(1)
        };
        Self::apply_prefill_execution_chunk_ceiling(req, tokens)
    }

    fn active_decode_prefill_budget_tokens(
        &self,
        hint: &BatchHint,
        scheduled_decode_count: usize,
        active_decode_prefill_chunk: Option<usize>,
        prefill_step_chunk: Option<usize>,
    ) -> Option<usize> {
        let chunk = self.effective_active_decode_prefill_chunk(
            active_decode_prefill_chunk,
            prefill_step_chunk,
        )?;
        if scheduled_decode_count == 0 {
            return None;
        }

        let remaining_step_tokens = hint.max_tokens.saturating_sub(scheduled_decode_count);
        let free_batch_slots = hint.max_batch_size.saturating_sub(scheduled_decode_count);
        if remaining_step_tokens == 0 || free_batch_slots == 0 {
            return Some(0);
        }

        let prefill_backlog = self.prefilling_count().saturating_add(self.waiting_count());
        if prefill_backlog == 0 {
            return Some(0);
        }

        let target_chunks =
            self.active_decode_prefill_target_chunks(hint, scheduled_decode_count, prefill_backlog);

        Some(
            chunk
                .saturating_mul(target_chunks)
                .min(remaining_step_tokens),
        )
    }

    fn active_decode_prefill_budget_chunks(
        &self,
        hint: &BatchHint,
        scheduled_decode_count: usize,
        active_decode_prefill_chunk: Option<usize>,
    ) -> Option<usize> {
        active_decode_prefill_chunk?;
        if scheduled_decode_count == 0 {
            return None;
        }

        let remaining_step_tokens = hint.max_tokens.saturating_sub(scheduled_decode_count);
        let free_batch_slots = hint.max_batch_size.saturating_sub(scheduled_decode_count);
        if remaining_step_tokens == 0 || free_batch_slots == 0 {
            return Some(0);
        }

        let prefill_backlog = self.prefilling_count().saturating_add(self.waiting_count());
        if prefill_backlog == 0 {
            return Some(0);
        }

        Some(
            self.active_decode_prefill_target_chunks(hint, scheduled_decode_count, prefill_backlog)
                .min(remaining_step_tokens),
        )
    }

    fn capacity_deferred_mixed_recompute_slot_budget(
        &self,
        active_decode_prefill_chunk: Option<usize>,
        prefill_step_chunk: Option<usize>,
        active_decode_prefill_tokens_remaining: Option<usize>,
        required_blocks_per_slot: usize,
        observed_free_blocks: usize,
    ) -> usize {
        let Some(tokens_remaining) = active_decode_prefill_tokens_remaining else {
            return 0;
        };
        if tokens_remaining == 0 {
            return 0;
        }
        let Some(chunk) = self
            .effective_active_decode_prefill_chunk(active_decode_prefill_chunk, prefill_step_chunk)
        else {
            return 0;
        };
        let token_budget_slots = tokens_remaining.div_ceil(chunk).max(1);
        if required_blocks_per_slot == 0 || observed_free_blocks == usize::MAX {
            return token_budget_slots;
        }
        let usable_free_blocks = Self::capacity_mixed_recompute_usable_free_blocks(
            observed_free_blocks,
            required_blocks_per_slot,
        );
        token_budget_slots.min(usable_free_blocks / required_blocks_per_slot)
    }

    fn should_budget_capacity_deferred_mixed_recompute(
        req: &ContinuousBatchRequest,
        active_decode_prefill_chunk: Option<usize>,
    ) -> bool {
        req.capacity_deferred_until_release_epoch > 0 && active_decode_prefill_chunk.is_some()
    }

    fn add_prefill_requests_to_batch(
        &self,
        iteration: u64,
        hint: &BatchHint,
        batch_requests: &mut Vec<ScheduledRequest>,
        total_tokens: &mut usize,
        scheduled_request_ids: &mut HashSet<RequestId>,
        active_decode_prefill_tokens_remaining: &mut Option<usize>,
        active_decode_prefill_chunks_remaining: &mut Option<usize>,
        capacity_deferred_mixed_recompute_slots_remaining: &mut Option<usize>,
        active_decode_prefill_chunk: Option<usize>,
        prefill_step_chunk: Option<usize>,
        waiting_admission: &mut WaitingAdmissionMode<'_>,
        _capacity_release_epoch: u64,
        capacity_mixed_recompute_epoch: u64,
    ) -> Result<()> {
        if batch_requests.len() >= hint.max_batch_size || *total_tokens >= hint.max_tokens {
            return Ok(());
        }

        let mut prefill_queue = self.prefill_queue.write();
        for req in prefill_queue.iter_mut() {
            if batch_requests.len() >= hint.max_batch_size {
                break;
            }
            if scheduled_request_ids.contains(&req.inner.request.id) {
                continue;
            }
            if Self::execution_readiness_is_blocked(req) {
                continue;
            }
            if Self::execution_maintenance_retry_is_blocked(req, iteration)? {
                continue;
            }
            if Self::execution_capacity_is_blocked(
                req,
                waiting_admission,
                ExecutionCapacityQueuePhase::Prefill,
            )? {
                continue;
            }
            let budgeted_capacity_deferred = Self::should_budget_capacity_deferred_mixed_recompute(
                req,
                active_decode_prefill_chunk,
            );
            if budgeted_capacity_deferred
                && req.capacity_deferred_mixed_attempt_epoch == Some(capacity_mixed_recompute_epoch)
            {
                continue;
            }
            if budgeted_capacity_deferred
                && capacity_deferred_mixed_recompute_slots_remaining
                    .as_ref()
                    .copied()
                    .unwrap_or(0)
                    == 0
            {
                continue;
            }
            if active_decode_prefill_chunks_remaining
                .as_ref()
                .is_some_and(|remaining| *remaining == 0)
            {
                break;
            }

            let mut step_tokens_remaining = hint.max_tokens.saturating_sub(*total_tokens);
            if let Some(remaining) = active_decode_prefill_tokens_remaining.as_ref() {
                step_tokens_remaining = step_tokens_remaining.min(*remaining);
            }
            let prefill_chunk_tokens = self.prefill_budget_tokens(
                req,
                active_decode_prefill_chunk,
                prefill_step_chunk,
                step_tokens_remaining,
            );
            // Skip fully-prefilled requests that are still in the queue
            // (they'll be promoted by mark_prefill_chunk_processed on the
            // next iteration boundary).
            if prefill_chunk_tokens == 0 {
                continue;
            }
            if let Some(remaining) = active_decode_prefill_tokens_remaining.as_mut() {
                if *remaining == 0 {
                    break;
                }
            }

            if *total_tokens + prefill_chunk_tokens <= hint.max_tokens {
                let mut scheduled = req.inner.clone();
                scheduled.tokens_processed = req.prefill_chunk_offset;
                scheduled.tokens_to_process = Some(prefill_chunk_tokens);
                req.logical_work_frontier
                    .mark_scheduled(prefill_chunk_tokens);
                scheduled_request_ids.insert(scheduled.request.id.clone());
                batch_requests.push(scheduled);
                *total_tokens += prefill_chunk_tokens;
                if let Some(remaining) = active_decode_prefill_tokens_remaining.as_mut() {
                    *remaining = remaining.saturating_sub(prefill_chunk_tokens);
                }
                if let Some(remaining) = active_decode_prefill_chunks_remaining.as_mut() {
                    *remaining = remaining.saturating_sub(1);
                }
                if budgeted_capacity_deferred {
                    req.capacity_deferred_mixed_attempt_epoch =
                        Some(capacity_mixed_recompute_epoch);
                    if let Some(remaining) =
                        capacity_deferred_mixed_recompute_slots_remaining.as_mut()
                    {
                        *remaining = remaining.saturating_sub(1);
                    }
                }
            }
        }
        Ok(())
    }

    /// Create batch plan for current iteration
    fn create_iteration_batch(&self, hint: BatchHint) -> Option<BatchPlan> {
        match self.create_iteration_batch_with_admission(hint, WaitingAdmissionMode::Legacy) {
            Ok(batch) => batch,
            Err(error) => {
                warn!("Legacy waiting admission failed: {}", error);
                None
            }
        }
    }

    fn execution_capacity_is_blocked(
        req: &mut ContinuousBatchRequest,
        waiting_admission: &mut WaitingAdmissionMode<'_>,
        phase: ExecutionCapacityQueuePhase,
    ) -> Result<bool> {
        let Some(deferral) = req.execution_capacity_deferral.clone() else {
            return Ok(false);
        };
        let wake = waiting_admission.wake().ok_or_else(|| {
            FerrumError::scheduler(
                "typed execution capacity deferral reached a legacy scheduler tick",
            )
        })?;
        if deferral.observed().coordinator_id() != wake.epochs().coordinator_id()
            || deferral.wait_condition().coordinator_id().get()
                != wake.epochs().coordinator_id().get()
        {
            return Err(FerrumError::scheduler(
                "typed execution capacity deferral belongs to another coordinator",
            ));
        }
        let observed = deferral.observed();
        let current = wake.epochs();
        if current.release_epoch() < observed.release_epoch()
            || current.capacity_epoch() < observed.capacity_epoch()
            || current.policy_epoch() < observed.policy_epoch()
        {
            return Err(FerrumError::scheduler(
                "typed execution capacity audit epoch regressed",
            ));
        }
        let exact_source_changed = deferral
            .wait_condition()
            .changed_since(wake.availability())
            .map_err(|error| FerrumError::scheduler(error.to_string()))?;
        let policy_epoch_changed = current.policy_epoch() != observed.policy_epoch();
        let current_wait_sources = waiting_admission.observes().then(|| {
            deferral
                .wait_condition()
                .observed()
                .iter()
                .map(|observed| {
                    let index = wake
                        .availability()
                        .binary_search_by_key(&observed.source(), |entry| entry.source())
                        .expect("validated wait source remains available");
                    wake.availability()[index]
                })
                .collect::<Vec<_>>()
        });
        if !exact_source_changed && !policy_epoch_changed {
            if let Some(current_wait_sources) = current_wait_sources {
                let observation = match phase {
                    ExecutionCapacityQueuePhase::Prefill => {
                        ExecutorAdmissionQueueObservation::PrefillSkippedUnchanged {
                            request_id: req.inner.request.id.clone(),
                            deferral,
                            current,
                            current_wait_sources,
                        }
                    }
                    ExecutionCapacityQueuePhase::Decode => {
                        ExecutorAdmissionQueueObservation::DecodeSkippedUnchanged {
                            request_id: req.inner.request.id.clone(),
                            deferral,
                            current,
                            current_wait_sources,
                        }
                    }
                };
                waiting_admission.observe(observation);
            }
            return Ok(true);
        }
        if let Some(current_wait_sources) = current_wait_sources {
            let observation = match phase {
                ExecutionCapacityQueuePhase::Prefill => {
                    ExecutorAdmissionQueueObservation::PrefillResumed {
                        request_id: req.inner.request.id.clone(),
                        deferral,
                        current,
                        current_wait_sources,
                        exact_source_changed,
                        policy_epoch_changed,
                    }
                }
                ExecutionCapacityQueuePhase::Decode => {
                    ExecutorAdmissionQueueObservation::DecodeResumed {
                        request_id: req.inner.request.id.clone(),
                        deferral,
                        current,
                        current_wait_sources,
                        exact_source_changed,
                        policy_epoch_changed,
                    }
                }
            };
            waiting_admission.observe(observation);
        }
        req.execution_capacity_deferral = None;
        Ok(false)
    }

    fn execution_readiness_is_blocked(req: &mut ContinuousBatchRequest) -> bool {
        let Some(block) = req.execution_readiness_block.as_ref() else {
            return false;
        };
        match block.status() {
            EXECUTION_READINESS_PENDING | EXECUTION_READINESS_FAILED => true,
            EXECUTION_READINESS_READY | EXECUTION_READINESS_CANCELLED => {
                req.execution_readiness_block = None;
                false
            }
            _ => true,
        }
    }

    fn execution_maintenance_retry_is_blocked(
        req: &mut ContinuousBatchRequest,
        iteration: u64,
    ) -> Result<bool> {
        let Some(ticket) = req.execution_maintenance_retry else {
            return Ok(false);
        };
        if req.last_execution_maintenance_capacity_epoch != Some(ticket.latest_capacity_epoch) {
            return Err(FerrumError::scheduler(
                "execution maintenance retry ticket lost its mutation generation",
            ));
        }
        if iteration < ticket.not_before_iteration {
            return Ok(true);
        }
        req.execution_maintenance_retry = None;
        Ok(false)
    }

    fn add_decode_requests_to_batch(
        &self,
        iteration: u64,
        hint: &BatchHint,
        batch_requests: &mut Vec<ScheduledRequest>,
        total_tokens: &mut usize,
        scheduled_request_ids: &mut HashSet<RequestId>,
        waiting_admission: &mut WaitingAdmissionMode<'_>,
    ) -> Result<()> {
        let has_deferred_recompute_backlog = self.decode_capacity_deferred_backlog_len() > 0;
        let (enforce_execution_backpressure, decode_capacity_backpressure_limit) = {
            let _feedback = self.decode_capacity_feedback_lock.lock();
            (
                self.decode_execution_pressure_enforced
                    .load(Ordering::Acquire),
                self.decode_capacity_backpressure_limit(),
            )
        };
        let decode_batch_limit =
            if has_deferred_recompute_backlog && !enforce_execution_backpressure {
                hint.max_batch_size
            } else {
                decode_capacity_backpressure_limit
                    .map(|limit| hint.max_batch_size.min(limit.max(1)))
                    .unwrap_or(hint.max_batch_size)
            };
        let mut decode_queue = self.decode_queue.write();
        let decode_len = decode_queue.requests.len();
        if decode_len == 0 {
            decode_queue.selection_cursor = None;
        } else if decode_queue
            .selection_cursor
            .as_ref()
            .is_none_or(|cursor_id| !decode_queue.requests.contains_key(cursor_id))
        {
            decode_queue.selection_cursor =
                decode_queue.requests.get_index(0).map(|(id, _)| id.clone());
        }
        let start = decode_queue
            .selection_cursor
            .as_ref()
            .and_then(|cursor_id| decode_queue.requests.get_index_of(cursor_id))
            .unwrap_or(0);
        let mut next_cursor_index = start;
        let mut scheduled_count = 0usize;
        for offset in 0..decode_len {
            if batch_requests.len() >= decode_batch_limit || *total_tokens >= hint.max_tokens {
                break;
            }
            let index = (start + offset) % decode_len;
            let (_, req) = decode_queue
                .requests
                .get_index_mut(index)
                .expect("decode round-robin index remains in bounds");
            if scheduled_request_ids.contains(&req.inner.request.id) {
                continue;
            }
            if Self::execution_readiness_is_blocked(req) {
                continue;
            }
            if Self::execution_maintenance_retry_is_blocked(req, iteration)? {
                continue;
            }
            if Self::execution_capacity_is_blocked(
                req,
                waiting_admission,
                ExecutionCapacityQueuePhase::Decode,
            )? {
                continue;
            }

            let mut scheduled = req.inner.clone();
            scheduled.tokens_processed = req.total_tokens();
            scheduled.tokens_to_process = Some(1);
            req.logical_work_frontier.mark_scheduled(1);
            scheduled_request_ids.insert(scheduled.request.id.clone());
            batch_requests.push(scheduled);
            *total_tokens += 1;
            scheduled_count += 1;
            next_cursor_index = (index + 1) % decode_len;
        }
        if scheduled_count > 0 {
            decode_queue.selection_cursor = decode_queue
                .requests
                .get_index(next_cursor_index)
                .map(|(id, _)| id.clone());
        }
        Ok(())
    }

    fn create_iteration_batch_with_admission(
        &self,
        hint: BatchHint,
        mut waiting_admission: WaitingAdmissionMode<'_>,
    ) -> Result<Option<BatchPlan>> {
        let iteration = self.current_iteration.fetch_add(1, Ordering::Relaxed);
        self.metrics_tracker.record_iteration();

        let mut batch_requests = Vec::new();
        let mut scheduled_request_ids = HashSet::new();
        let mut total_tokens = 0;
        let prefill_first_target = self
            .runtime_config
            .prefill_first_until_active
            .map(|target| {
                target
                    .min(hint.max_batch_size)
                    .min(self.cb_config.max_decode_batch)
            })
            .unwrap_or(0);
        let decoding_count = self.decoding_count();
        let active_count = self.active_count();
        let capacity_backpressure_active = self.capacity_backpressure_admit_limit().is_some();
        let skip_decode_for_prefill_first = prefill_first_target > 0
            && decoding_count < prefill_first_target
            && active_count < prefill_first_target
            && !(capacity_backpressure_active && decoding_count > 0)
            && (self.prefilling_count() > 0 || self.waiting_count() > 0);
        // First, collect decode requests (they have priority). The opt-in
        // fill-first experiment skips decodes until the active decode cohort
        // reaches the requested target, reducing early mixed prefill+decode
        // spikes in c=32 closed-loop runs.
        if !skip_decode_for_prefill_first {
            self.add_decode_requests_to_batch(
                iteration,
                &hint,
                &mut batch_requests,
                &mut total_tokens,
                &mut scheduled_request_ids,
                &mut waiting_admission,
            )?;
        }
        let scheduled_decode_count = batch_requests.len();
        let active_decode_prefill_chunk =
            self.active_decode_prefill_chunk_for_iteration(&hint, scheduled_decode_count);
        let prefill_step_chunk = self.runtime_config.prefill_step_chunk;
        let mut active_decode_prefill_tokens_remaining = self.active_decode_prefill_budget_tokens(
            &hint,
            scheduled_decode_count,
            active_decode_prefill_chunk,
            prefill_step_chunk,
        );
        let mut active_decode_prefill_chunks_remaining = self.active_decode_prefill_budget_chunks(
            &hint,
            scheduled_decode_count,
            active_decode_prefill_chunk,
        );
        let capacity_release_epoch = self.capacity_release_epoch.load(Ordering::Relaxed);
        let capacity_mixed_recompute_epoch =
            self.capacity_mixed_recompute_epoch.load(Ordering::Relaxed);
        let mixed_recompute_release_ready = self
            .capacity_mixed_recompute_blocked_until_epoch
            .load(Ordering::Relaxed)
            <= capacity_mixed_recompute_epoch;
        let mixed_recompute_required_blocks_per_slot = self
            .capacity_mixed_recompute_required_blocks_per_slot
            .load(Ordering::Relaxed);
        let mixed_recompute_observed_free_blocks = self
            .capacity_mixed_recompute_observed_free_blocks
            .load(Ordering::Relaxed);
        let mixed_recompute_kv_capacity_ready = mixed_recompute_required_blocks_per_slot == 0
            || Self::capacity_mixed_recompute_usable_free_blocks(
                mixed_recompute_observed_free_blocks,
                mixed_recompute_required_blocks_per_slot,
            ) >= mixed_recompute_required_blocks_per_slot;
        let allow_capacity_deferred_mixed_recompute = scheduled_decode_count > 0
            && active_decode_prefill_chunk.is_some()
            && active_decode_prefill_tokens_remaining.unwrap_or(0) > 0
            && mixed_recompute_release_ready
            && mixed_recompute_kv_capacity_ready;
        let mut capacity_deferred_mixed_recompute_slots_remaining =
            allow_capacity_deferred_mixed_recompute.then(|| {
                self.capacity_deferred_mixed_recompute_slot_budget(
                    active_decode_prefill_chunk,
                    prefill_step_chunk,
                    active_decode_prefill_tokens_remaining,
                    mixed_recompute_required_blocks_per_slot,
                    mixed_recompute_observed_free_blocks,
                )
            });

        // Then, add prefill requests up to the per-iter token budget.
        // Phase 3: `max_prefill_batch=8` no longer caps the count —
        // the only budget is `hint.max_tokens` (= EngineConfig's
        // `max_num_batched_tokens`, default 4096). Decodes contribute
        // 1 token each; prefill chunks contribute their chunk size.
        // This is what lets the Qwen3MoE `unified_forward` path
        // activate for cohort prefills (m_total must stay ≤ scratch
        // max_tokens, which is pre-allocated to the same budget).
        self.add_prefill_requests_to_batch(
            iteration,
            &hint,
            &mut batch_requests,
            &mut total_tokens,
            &mut scheduled_request_ids,
            &mut active_decode_prefill_tokens_remaining,
            &mut active_decode_prefill_chunks_remaining,
            &mut capacity_deferred_mixed_recompute_slots_remaining,
            active_decode_prefill_chunk,
            prefill_step_chunk,
            &mut waiting_admission,
            capacity_release_epoch,
            capacity_mixed_recompute_epoch,
        )?;

        // Check if we should admit new requests from waiting queue.
        let active_capacity = self
            .config
            .max_running_requests
            .saturating_sub(self.active_count());
        let decode_capacity = self
            .cb_config
            .max_decode_batch
            .saturating_sub(self.decoding_count());
        let available_slots = active_capacity.min(decode_capacity);
        let available_slots = self
            .capacity_backpressure_admit_limit()
            .map(|limit| available_slots.min(limit))
            .unwrap_or(available_slots);
        if matches!(&waiting_admission, WaitingAdmissionMode::Legacy) {
            self.legacy_waiting_admission_ticks
                .fetch_add(1, Ordering::Relaxed);
            let active_count_for_capacity_wait = self.active_count();
            let waiting_queue = self.waiting_queue.read();
            let mut requests_to_admit = Vec::new();
            let mut release_blocked_capacity_deferred_admissions = 0usize;
            for req in waiting_queue.iter() {
                if requests_to_admit.len() >= available_slots {
                    break;
                }
                if self.pressure_active.load(Ordering::Acquire)
                    && matches!(
                        self.pressure_coordinator
                            .lock()
                            .hold_status(&req.inner.request.id),
                        PressureHoldStatus::Held { .. }
                    )
                {
                    continue;
                }
                let budgeted_capacity_deferred =
                    Self::should_budget_capacity_deferred_mixed_recompute(
                        req,
                        active_decode_prefill_chunk,
                    );
                let release_ready =
                    req.capacity_deferred_until_release_epoch <= capacity_release_epoch;
                let empty_scheduler_retry = active_count_for_capacity_wait == 0
                    && !release_ready
                    && req.capacity_deferred_empty_retry_epoch != Some(capacity_release_epoch);
                if release_ready && !budgeted_capacity_deferred {
                    requests_to_admit.push((req.inner.request.id.clone(), None));
                } else if empty_scheduler_retry && !budgeted_capacity_deferred {
                    requests_to_admit
                        .push((req.inner.request.id.clone(), Some(capacity_release_epoch)));
                } else if req.capacity_deferred_mixed_attempt_epoch
                    == Some(capacity_mixed_recompute_epoch)
                {
                    continue;
                } else if allow_capacity_deferred_mixed_recompute
                    && release_blocked_capacity_deferred_admissions
                        < capacity_deferred_mixed_recompute_slots_remaining.unwrap_or(0)
                {
                    requests_to_admit.push((req.inner.request.id.clone(), None));
                    release_blocked_capacity_deferred_admissions += 1;
                }
            }
            drop(waiting_queue);
            for (req_id, empty_retry_epoch) in requests_to_admit {
                self.promote_to_prefill_with_empty_retry(&req_id, empty_retry_epoch);
            }
        } else {
            self.admit_waiting_dynamically(usize::MAX, available_slots, &mut waiting_admission)?;
        }

        // vLLM's scheduler spends the remaining per-step token budget on
        // waiting requests after running requests. Mirror that behavior so
        // newly admitted prefills do not wait an extra iteration just because
        // the current batch already contains decode work.
        self.add_prefill_requests_to_batch(
            iteration,
            &hint,
            &mut batch_requests,
            &mut total_tokens,
            &mut scheduled_request_ids,
            &mut active_decode_prefill_tokens_remaining,
            &mut active_decode_prefill_chunks_remaining,
            &mut capacity_deferred_mixed_recompute_slots_remaining,
            active_decode_prefill_chunk,
            prefill_step_chunk,
            &mut waiting_admission,
            capacity_release_epoch,
            capacity_mixed_recompute_epoch,
        )?;

        // Fill-first is a throughput policy, not permission to deadlock. A
        // typed WaitForRelease request may need active decodes to retire before
        // its epoch can advance. If admission produced no runnable prefill,
        // restore decode work so the release condition can become true.
        if skip_decode_for_prefill_first && batch_requests.is_empty() {
            self.add_decode_requests_to_batch(
                iteration,
                &hint,
                &mut batch_requests,
                &mut total_tokens,
                &mut scheduled_request_ids,
                &mut waiting_admission,
            )?;
        }

        // FERRUM_SCHED_NONE_PROF=1: log when next_batch is about to return SOME.
        if self.runtime_config.scheduler_none_prof && !batch_requests.is_empty() {
            use std::sync::atomic::AtomicU64;
            static SOME_PROF_N: AtomicU64 = AtomicU64::new(0);
            let n = SOME_PROF_N.fetch_add(1, Ordering::Relaxed);
            if n.is_multiple_of(64) {
                let d_len = self.decode_queue.read().requests.len();
                let p_len = self.prefill_queue.read().len();
                let w_len = self.waiting_queue.read().len();
                eprintln!(
                    "[sched-some] n={} returning_batch={} | decode_queue={} prefill_queue={} waiting_queue={}",
                    n,
                    batch_requests.len(),
                    d_len,
                    p_len,
                    w_len,
                );
            }
        }
        if batch_requests.is_empty() {
            // FERRUM_SCHED_NONE_PROF=1: log why we returned None. Rate-limited.
            if self.runtime_config.scheduler_none_prof {
                use std::sync::atomic::AtomicU64;
                static NONE_PROF_N: AtomicU64 = AtomicU64::new(0);
                let n = NONE_PROF_N.fetch_add(1, Ordering::Relaxed);
                if n.is_multiple_of(512) {
                    let d_len = self.decode_queue.read().requests.len();
                    let p_len = self.prefill_queue.read().len();
                    let w_len = self.waiting_queue.read().len();
                    let d_count = self.decoding_count();
                    eprintln!(
                        "[sched-none] n={} decode_queue={} prefill_queue={} waiting_queue={} decoding_count={} hint.max_batch={}",
                        n,
                        d_len,
                        p_len,
                        w_len,
                        d_count,
                        hint.max_batch_size,
                    );
                }
            }
            return Ok(None);
        }

        let batch_id = BatchId::new();
        let max_seq_len = batch_requests
            .iter()
            .map(|r| r.request.sampling_params.max_tokens)
            .max()
            .unwrap_or(2048);

        debug!(
            "Created iteration {} batch: {} requests, {} tokens",
            iteration,
            batch_requests.len(),
            total_tokens
        );

        Ok(Some(BatchPlan {
            batch_id,
            requests: batch_requests,
            max_sequence_length: max_seq_len,
            estimated_time_ms: Some(self.cb_config.target_iteration_time_ms),
            resource_requirements: BatchResourceRequirements {
                gpu_memory: (total_tokens * 16) as u64,
                cpu_memory: (total_tokens * 4) as u64,
                kv_cache_blocks: total_tokens / 16,
                recurrent_state_bytes: 0,
                recurrent_state_slots: 0,
                compute_units: 1,
            },
            created_at: chrono::Utc::now(),
        }))
    }

    /// Mark a request as having completed prefill
    pub fn mark_prefill_complete(&self, request_id: &RequestId, tokens: usize) {
        let mut prefill_queue = self.prefill_queue.write();
        let mut found = false;
        let mut progress = None;
        if let Some(pos) = prefill_queue
            .iter()
            .position(|r| r.inner.request.id == *request_id)
        {
            let req = &mut prefill_queue[pos];
            let delta = tokens.saturating_sub(req.prefill_chunk_offset);
            req.prefill_tokens = tokens;
            req.prefill_chunk_offset = tokens;
            req.chunked_prefill = false;
            req.logical_work_frontier.commit_prefill(tokens, delta);
            progress = Some(req.logical_work_frontier.progress_generation());
            found = true;
        }
        drop(prefill_queue);

        // Promote to decode
        self.promote_to_decode(request_id);
        if found {
            if let Some(progress) = progress {
                self.record_pressure_frontier_progress(request_id, progress);
            }
            self.record_resource_progress();
        }
    }

    /// Mark a chunk of prefill as processed. Used by engines that split a
    /// long prompt across multiple iterations to reduce TTFT under load.
    ///
    /// `total_prompt_tokens` should be the full prompt length — pass it
    /// every call (idempotent: the scheduler uses the last value it sees).
    /// `chunk_tokens` is how many tokens were processed *this iteration*.
    ///
    /// Returns `true` if the request is now fully prefilled and has been
    /// promoted to the decode queue.
    pub fn mark_prefill_chunk_processed(
        &self,
        request_id: &RequestId,
        total_prompt_tokens: usize,
        chunk_tokens: usize,
    ) -> bool {
        self.mark_prefill_chunk_processed_inner(request_id, total_prompt_tokens, chunk_tokens, None)
    }

    /// Commit an executor-selected prefix of the scheduler's maximum prefill
    /// frontier and retain the observed fit as a request-local scheduling cap.
    pub fn mark_prefill_chunk_processed_with_capacity_feedback(
        &self,
        request_id: &RequestId,
        total_prompt_tokens: usize,
        planned_chunk_tokens: usize,
        completed_chunk_tokens: usize,
    ) -> Result<bool> {
        if completed_chunk_tokens == 0 || completed_chunk_tokens > planned_chunk_tokens {
            return Err(FerrumError::scheduler(
                "completed prefill prefix must be non-empty and no wider than its planned frontier",
            ));
        }
        Ok(self.mark_prefill_chunk_processed_inner(
            request_id,
            total_prompt_tokens,
            completed_chunk_tokens,
            Some((planned_chunk_tokens, completed_chunk_tokens)),
        ))
    }

    fn mark_prefill_chunk_processed_inner(
        &self,
        request_id: &RequestId,
        total_prompt_tokens: usize,
        chunk_tokens: usize,
        execution_frontier_feedback: Option<(usize, usize)>,
    ) -> bool {
        let mut prefill_queue = self.prefill_queue.write();
        let mut fully_done = false;
        let mut made_progress = false;
        let mut progress = None;
        if let Some(pos) = prefill_queue
            .iter()
            .position(|r| r.inner.request.id == *request_id)
        {
            let req = &mut prefill_queue[pos];
            req.prefill_tokens = total_prompt_tokens;
            req.chunked_prefill = true;
            let previous_offset = req.prefill_chunk_offset;
            req.prefill_chunk_offset = req
                .prefill_chunk_offset
                .saturating_add(chunk_tokens)
                .min(total_prompt_tokens);
            let committed_tokens = req.prefill_chunk_offset.saturating_sub(previous_offset);
            req.logical_work_frontier
                .commit_prefill(req.prefill_chunk_offset, committed_tokens);
            if let Some((planned, completed)) = execution_frontier_feedback {
                if completed < planned {
                    req.prefill_execution_chunk_ceiling = Some(
                        req.prefill_execution_chunk_ceiling
                            .map(|current| current.min(completed))
                            .unwrap_or(completed),
                    );
                } else if let Some(current) = req.prefill_execution_chunk_ceiling {
                    req.prefill_execution_chunk_ceiling = Some(current.saturating_mul(2));
                }
            }
            progress = Some(req.logical_work_frontier.progress_generation());
            if committed_tokens > 0 {
                req.capacity_deferred_mixed_attempt_epoch = None;
                req.capacity_deferred_empty_retry_epoch = None;
            }
            fully_done = req.prefill_chunk_offset >= total_prompt_tokens;
            made_progress = committed_tokens > 0;
        }
        drop(prefill_queue);

        if fully_done {
            self.promote_to_decode(request_id);
        }
        if made_progress {
            if let Some(progress) = progress {
                self.record_pressure_frontier_progress(request_id, progress);
            }
        }
        if made_progress && fully_done {
            self.record_resource_progress();
        }
        fully_done
    }

    /// Update decode progress for a request
    pub fn update_decode_progress(&self, request_id: &RequestId, tokens_generated: usize) {
        let mut decode_queue = self.decode_queue.write();
        let mut progress = None;
        if let Some(req) = decode_queue.requests.get_mut(request_id) {
            req.decode_tokens = tokens_generated;
            req.logical_work_frontier.commit_decode(tokens_generated);
            progress = Some(req.logical_work_frontier.progress_generation());
            req.last_iteration = self.current_iteration.load(Ordering::Relaxed);
        }
        drop(decode_queue);
        if let Some(progress) = progress {
            self.record_pressure_frontier_progress(request_id, progress);
        }
        // Decode progress consumes KV capacity; only actual prefill progress or
        // completion should relax capacity backpressure.
    }

    fn record_pressure_frontier_progress(
        &self,
        request_id: &RequestId,
        progress: LogicalWorkGeneration,
    ) {
        if !self.pressure_active.load(Ordering::Acquire) {
            return;
        }
        let mut coordinator = self.pressure_coordinator.lock();
        if let Err(error) = coordinator.record_progress(request_id, progress) {
            warn!(
                request_id = %request_id,
                error = %error,
                "Pressure coordinator rejected logical frontier progress"
            );
        }
        self.pressure_active
            .store(coordinator.has_records(), Ordering::Release);
    }

    fn record_pressure_frontier_terminal(&self, request_id: &RequestId) {
        if !self.pressure_active.load(Ordering::Acquire) {
            return;
        }
        let mut coordinator = self.pressure_coordinator.lock();
        if let Err(error) = coordinator.record_terminal(request_id) {
            warn!(
                request_id = %request_id,
                error = %error,
                "Pressure coordinator rejected logical frontier terminal state"
            );
        }
        self.pressure_active
            .store(coordinator.has_records(), Ordering::Release);
    }

    fn consume_pressure_hold(&self, request_id: &RequestId) {
        if !self.pressure_active.load(Ordering::Acquire) {
            return;
        }
        let mut coordinator = self.pressure_coordinator.lock();
        if let Err(error) = coordinator.consume_released_hold(request_id) {
            warn!(
                request_id = %request_id,
                error = %error,
                "Pressure coordinator rejected admitted owner transition"
            );
        }
        self.pressure_active
            .store(coordinator.has_records(), Ordering::Release);
    }
}

#[async_trait]
impl Scheduler for ContinuousBatchScheduler {
    async fn submit(&self, request: InferenceRequest) -> Result<RequestId> {
        let request_id = request.id.clone();
        debug!(
            "Submitting request {} to continuous batch scheduler",
            request_id
        );

        // Check queue capacity
        let waiting_count = self.waiting_count();
        if waiting_count >= self.config.max_waiting_requests {
            warn!("Queue is full, rejecting request {}", request_id);
            return Err(FerrumError::scheduler(
                "Queue is full, cannot accept more requests",
            ));
        }

        // Create continuous batch request
        let cb_request = ContinuousBatchRequest::new(request);

        // Add to waiting queue
        let mut waiting_queue = self.waiting_queue.write();
        let queue_position = waiting_queue.len();

        let mut req = cb_request;
        req.inner.queue_position = Some(queue_position);

        let ticket = waiting_queue
            .enqueue(req)
            .map_err(|error| FerrumError::scheduler(error.to_string()))?;
        waiting_queue
            .request_mut(ticket)
            .expect("newly enqueued admission ticket remains present")
            .waiting_admission_ticket = Some(ticket);

        // Update index
        self.request_index
            .write()
            .insert(request_id.clone(), RequestPhase::Waiting);

        info!(
            "Request {} queued at position {}",
            request_id, queue_position
        );
        Ok(request_id)
    }

    async fn next_batch(&self, hint: BatchHint) -> Option<BatchPlan> {
        self.create_iteration_batch(hint)
    }

    async fn complete(&self, request_id: RequestId, response: &InferenceResponse) -> Result<()> {
        debug!("Completing request {}", request_id);

        // Remove from decode queue
        let mut decode_queue = self.decode_queue.write();
        if let Some(req) = decode_queue.remove(&request_id) {
            // Record metrics
            self.metrics_tracker.record_completion(&req);

            match response.finish_reason {
                ferrum_types::FinishReason::EOS
                | ferrum_types::FinishReason::Stop
                | ferrum_types::FinishReason::Length => {
                    self.completed_counter.fetch_add(1, Ordering::Relaxed);
                    debug!("Request {} completed successfully", request_id);
                }
                _ => {
                    self.failed_counter.fetch_add(1, Ordering::Relaxed);
                    warn!(
                        "Request {} completed with error: {:?}",
                        request_id, response.finish_reason
                    );
                }
            }

            // Remove from index
            self.request_index.write().remove(&request_id);
            self.record_pressure_frontier_terminal(&request_id);
            self.record_capacity_release_progress();

            Ok(())
        } else {
            // Try removing from prefill queue
            let mut prefill_queue = self.prefill_queue.write();
            if let Some(pos) = prefill_queue
                .iter()
                .position(|r| r.inner.request.id == request_id)
            {
                prefill_queue.remove(pos);
                self.request_index.write().remove(&request_id);
                self.record_pressure_frontier_terminal(&request_id);
                match response.finish_reason {
                    ferrum_types::FinishReason::EOS
                    | ferrum_types::FinishReason::Stop
                    | ferrum_types::FinishReason::Length => {
                        self.completed_counter.fetch_add(1, Ordering::Relaxed);
                    }
                    _ => {
                        self.failed_counter.fetch_add(1, Ordering::Relaxed);
                        warn!(
                            "Request {} completed with error during prefill: {:?}",
                            request_id, response.finish_reason
                        );
                    }
                }
                self.record_capacity_release_progress();
                return Ok(());
            }
            drop(prefill_queue);

            if self
                .admission_failed_requests
                .write()
                .remove(&request_id)
                .is_some()
            {
                self.request_index.write().remove(&request_id);
                self.failed_counter.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }

            warn!("Attempted to complete unknown request: {}", request_id);
            Err(FerrumError::scheduler(format!(
                "Request {} not found in active queues",
                request_id
            )))
        }
    }

    async fn cancel(&self, request_id: RequestId) -> Result<bool> {
        debug!("Cancelling request {}", request_id);

        // Check and remove from waiting queue
        {
            let mut waiting_queue = self.waiting_queue.write();
            let waiting_position =
                waiting_queue.position(|request| request.inner.request.id == request_id);
            if let Some(pos) = waiting_position {
                waiting_queue.remove(pos);
                self.request_index.write().remove(&request_id);
                self.record_pressure_frontier_terminal(&request_id);
                self.cancelled_counter.fetch_add(1, Ordering::Relaxed);
                info!("Request {} cancelled from waiting queue", request_id);
                return Ok(true);
            }
        }

        // Check and remove from prefill queue
        {
            let mut prefill_queue = self.prefill_queue.write();
            if let Some(pos) = prefill_queue
                .iter()
                .position(|r| r.inner.request.id == request_id)
            {
                prefill_queue.remove(pos);
                self.request_index.write().remove(&request_id);
                self.record_pressure_frontier_terminal(&request_id);
                self.cancelled_counter.fetch_add(1, Ordering::Relaxed);
                self.record_capacity_release_progress();
                warn!("Request {} cancelled during prefill", request_id);
                return Ok(true);
            }
        }

        // Check and remove from decode queue
        {
            let mut decode_queue = self.decode_queue.write();
            if decode_queue.remove(&request_id).is_some() {
                self.request_index.write().remove(&request_id);
                self.record_pressure_frontier_terminal(&request_id);
                self.cancelled_counter.fetch_add(1, Ordering::Relaxed);
                self.record_capacity_release_progress();
                warn!("Request {} cancelled during decode", request_id);
                return Ok(true);
            }
            drop(decode_queue);
        }

        if self
            .admission_failed_requests
            .write()
            .remove(&request_id)
            .is_some()
        {
            self.request_index.write().remove(&request_id);
            self.cancelled_counter.fetch_add(1, Ordering::Relaxed);
            return Ok(true);
        }

        warn!("Request {} not found for cancellation", request_id);
        Ok(false)
    }

    async fn update_priority(&self, request_id: RequestId, priority: Priority) -> Result<()> {
        debug!(
            "Updating priority for request {} to {:?}",
            request_id, priority
        );

        // Update in waiting queue
        {
            let mut waiting_queue = self.waiting_queue.write();
            if let Some(req) =
                waiting_queue.find_mut(|request| request.inner.request.id == request_id)
            {
                req.inner.request.priority = priority;
                return Ok(());
            }
        }

        // Update in prefill queue
        {
            let mut prefill_queue = self.prefill_queue.write();
            if let Some(req) = prefill_queue
                .iter_mut()
                .find(|r| r.inner.request.id == request_id)
            {
                req.inner.request.priority = priority;
                return Ok(());
            }
        }

        // Update in decode queue
        {
            let mut decode_queue = self.decode_queue.write();
            if let Some(req) = decode_queue.requests.get_mut(&request_id) {
                req.inner.request.priority = priority;
                return Ok(());
            }
        }

        Ok(())
    }

    fn metrics(&self) -> SchedulerMetrics {
        let waiting_count = self.waiting_count();
        let prefill_count = self.prefilling_count();
        let decode_count = self.decoding_count();
        let running_count = prefill_count + decode_count;

        let completed_count = self.completed_counter.load(Ordering::Relaxed);
        let failed_count = self.failed_counter.load(Ordering::Relaxed);
        let cancelled_count = self.cancelled_counter.load(Ordering::Relaxed);
        let preempted_count = self.preempted_counter.load(Ordering::Relaxed);
        let admitted_count = self.admitted_counter.load(Ordering::Relaxed);
        let total_wait_time_us = self.total_wait_time_us.load(Ordering::Relaxed);

        let uptime_secs = self.start_time.elapsed().as_secs_f64();
        let throughput = if uptime_secs > 0.0 {
            completed_count as f64 / uptime_secs
        } else {
            0.0
        };

        let queue_utilization = waiting_count as f32 / self.config.max_waiting_requests as f32;
        let avg_wait_time_ms = if admitted_count > 0 {
            total_wait_time_us as f64 / admitted_count as f64 / 1000.0
        } else {
            0.0
        };

        ferrum_types::SchedulerStats {
            waiting_requests: waiting_count,
            running_requests: running_count,
            preempted_requests: preempted_count as usize,
            completed_requests: completed_count,
            failed_requests: failed_count,
            cancelled_requests: cancelled_count,
            avg_wait_time_ms,
            avg_execution_time_ms: 0.0,
            throughput_rps: throughput,
            queue_utilization,
        }
    }

    fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    fn request_state(&self, request_id: &RequestId) -> Option<RequestState> {
        self.request_index
            .read()
            .get(request_id)
            .copied()
            .map(|phase| match phase {
                RequestPhase::Waiting => RequestState::Waiting,
                RequestPhase::Prefilling | RequestPhase::Decoding => RequestState::Running,
                RequestPhase::Completed => RequestState::Completed,
                RequestPhase::Preempted => RequestState::Preempted,
                RequestPhase::Cancelled => RequestState::Cancelled,
                RequestPhase::AdmissionFailed => RequestState::Failed,
            })
    }

    async fn preempt(&self, request_id: RequestId) -> Result<PreemptionResult> {
        if !self.cb_config.enable_swapping {
            return Err(FerrumError::unsupported("Swapping is not enabled"));
        }

        debug!("Preempting request {}", request_id);

        // Remove from decode queue
        let mut decode_queue = self.decode_queue.write();
        if let Some(mut req) = decode_queue.remove(&request_id) {
            req.phase = RequestPhase::Preempted;

            // Save preemption state
            let state = PreemptionState {
                kv_cache_checkpoint: Vec::new(), // TODO: implement actual checkpoint
                tokens_processed: req.total_tokens(),
                generation_state: HashMap::new(),
            };

            let freed_resources = req.inner.allocated_resources.clone();

            // Move to preempted queue
            self.preempted_requests
                .write()
                .insert(request_id.clone(), req);
            self.request_index
                .write()
                .insert(request_id, RequestPhase::Preempted);
            self.preempted_counter.fetch_add(1, Ordering::Relaxed);
            self.record_capacity_release_progress();

            Ok(PreemptionResult {
                success: true,
                saved_state: Some(state),
                freed_resources,
            })
        } else {
            Err(FerrumError::scheduler(format!(
                "Request {} not found in decode queue",
                request_id
            )))
        }
    }

    async fn resume(&self, request_id: RequestId) -> Result<()> {
        debug!("Resuming request {}", request_id);

        let mut preempted = self.preempted_requests.write();
        if let Some(mut req) = preempted.remove(&request_id) {
            req.phase = RequestPhase::Decoding;

            self.decode_queue
                .write()
                .requests
                .insert(request_id.clone(), req);
            self.request_index
                .write()
                .insert(request_id, RequestPhase::Decoding);

            Ok(())
        } else {
            Err(FerrumError::scheduler(format!(
                "Request {} not found in preempted queue",
                request_id
            )))
        }
    }
}

impl std::fmt::Debug for ContinuousBatchScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContinuousBatchScheduler")
            .field("waiting", &self.waiting_count())
            .field("prefilling", &self.prefilling_count())
            .field("decoding", &self.decoding_count())
            .field("iteration", &self.current_iteration.load(Ordering::Relaxed))
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_types::{ModelId, SamplingParams};

    fn create_test_request(priority: Priority) -> InferenceRequest {
        InferenceRequest {
            id: RequestId::new(),
            prompt: "test".to_string(),
            model_id: ModelId::new("test-model"),
            sampling_params: SamplingParams::default(),
            stream: false,
            priority,
            client_id: None,
            session_id: None,
            created_at: chrono::Utc::now(),
            api_request: None,
            evidence_request: Default::default(),
            metadata: std::collections::HashMap::new(),
        }
    }

    fn create_test_request_with_prompt_tokens(
        priority: Priority,
        prompt_tokens: usize,
    ) -> InferenceRequest {
        create_test_request(priority).with_metadata(
            PROMPT_TOKENS_METADATA_KEY,
            serde_json::Value::from(prompt_tokens as u64),
        )
    }

    fn enqueue_waiting(scheduler: &ContinuousBatchScheduler, request: InferenceRequest) {
        let request_id = request.id.clone();
        let mut waiting = scheduler.waiting_queue.write();
        let ticket = waiting
            .enqueue(ContinuousBatchRequest::new(request))
            .unwrap();
        waiting
            .request_mut(ticket)
            .unwrap()
            .waiting_admission_ticket = Some(ticket);
        drop(waiting);
        scheduler
            .request_index
            .write()
            .insert(request_id, RequestPhase::Waiting);
    }

    fn execution_capacity_release_snapshot<'a>(
        request_ids: impl IntoIterator<Item = &'a RequestId>,
        condition: &CapacityWaitCondition,
    ) -> ExecutionCapacityReleaseSnapshot {
        let sources = condition
            .observed()
            .iter()
            .map(|observed| observed.source())
            .collect::<Vec<_>>();
        ExecutionCapacityReleaseSnapshot::new(
            request_ids
                .into_iter()
                .map(|request_id| (request_id.clone(), sources.clone())),
        )
    }

    #[tokio::test]
    async fn test_scheduler_creation() {
        let config = SchedulerConfig::default();
        let scheduler = ContinuousBatchScheduler::new(config);
        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(scheduler.active_count(), 0);
    }

    #[test]
    fn admission_phase_counts_are_mutually_exclusive_per_request() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
        let phases = [
            RequestPhase::Waiting,
            RequestPhase::Prefilling,
            RequestPhase::Decoding,
            RequestPhase::Decoding,
            RequestPhase::Completed,
            RequestPhase::Cancelled,
        ];
        {
            let mut request_index = scheduler.request_index.write();
            for phase in phases {
                request_index.insert(RequestId::new(), phase);
            }
        }

        let counts = scheduler.admission_phase_counts();
        assert_eq!(counts.waiting_requests, 1);
        assert_eq!(counts.active_prefill_sequences, 1);
        assert_eq!(counts.active_decode_sequences, 2);
        assert_eq!(
            counts.waiting_requests
                + counts.active_prefill_sequences
                + counts.active_decode_sequences,
            4
        );
    }

    #[tokio::test]
    async fn test_submit_and_counts() {
        let config = SchedulerConfig::default();
        let scheduler = ContinuousBatchScheduler::new(config);

        scheduler
            .submit(create_test_request(Priority::Normal))
            .await
            .unwrap();
        scheduler
            .submit(create_test_request(Priority::High))
            .await
            .unwrap();

        assert_eq!(scheduler.waiting_count(), 2);
        assert_eq!(scheduler.active_count(), 0);
    }

    #[tokio::test]
    async fn trace_snapshot_reports_queue_counters_and_phase() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
        let request = create_test_request(Priority::Normal);
        let request_id = request.id.clone();
        scheduler.submit(request).await.unwrap();

        let before = scheduler.trace_snapshot();
        assert_eq!(before.waiting_queue_len, 1);
        assert_eq!(before.active_len, 0);
        assert_eq!(before.admitted_total, 0);
        assert_eq!(before.capacity_release_epoch, 0);
        assert_eq!(before.capacity_mixed_recompute_epoch, 0);
        assert_eq!(before.capacity_mixed_recompute_blocked_until_epoch, 0);
        assert_eq!(
            before.capacity_mixed_recompute_required_blocks_per_slot,
            None
        );
        assert_eq!(before.capacity_mixed_recompute_observed_free_blocks, None);
        assert_eq!(before.decode_capacity_backpressure_admit_limit, None);
        assert_eq!(
            scheduler.trace_phase(&request_id),
            Some(RequestPhase::Waiting)
        );

        let batch = scheduler.next_batch(BatchHint::simple(4)).await.unwrap();
        assert_eq!(batch.size(), 1);

        let after = scheduler.trace_snapshot();
        assert_eq!(after.waiting_queue_len, 0);
        assert_eq!(after.prefill_queue_len, 1);
        assert_eq!(after.active_len, 1);
        assert_eq!(after.admitted_total, 1);
        assert_eq!(
            scheduler.trace_phase(&request_id),
            Some(RequestPhase::Prefilling)
        );
    }

    #[test]
    fn trace_snapshot_releases_prefill_read_before_later_snapshot_work() {
        use std::sync::{mpsc::sync_channel, Arc, Barrier};
        use std::time::Duration;

        const WRITER_WAIT: Duration = Duration::from_millis(250);

        let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
        let writer_scheduler = Arc::clone(&scheduler);
        let writer_start = Arc::new(Barrier::new(2));
        let writer_barrier = Arc::clone(&writer_start);
        let (attempting_tx, attempting_rx) = sync_channel(1);
        let (acquired_tx, acquired_rx) = sync_channel(1);
        let writer = std::thread::spawn(move || {
            writer_barrier.wait();
            attempting_tx.send(()).unwrap();
            let acquired = writer_scheduler
                .prefill_queue
                .try_write_for(WRITER_WAIT)
                .is_some();
            acquired_tx.send(acquired).unwrap();
        });

        let snapshot = scheduler.trace_snapshot_with_prefill_read_observer(|| {
            writer_start.wait();
            attempting_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("bounded writer must begin its fair-lock attempt");
            std::thread::sleep(Duration::from_millis(20));
        });
        let writer_acquired = acquired_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("bounded writer must finish its fair-lock attempt");
        writer.join().expect("bounded snapshot writer must join");

        assert!(
            writer_acquired,
            "trace snapshot must release its first prefill read before later snapshot work"
        );
        assert_eq!(snapshot.prefill_queue_len, 0);
        assert_eq!(snapshot.execution_capacity_blocked_prefill_len, 0);
        assert_eq!(snapshot.execution_readiness_blocked_prefill_len, 0);
    }

    #[tokio::test]
    async fn execution_maintenance_retry_yields_one_iteration_without_opening_pressure() {
        let mut config = SchedulerConfig::default();
        config.max_running_requests = 2;
        let scheduler = ContinuousBatchScheduler::new(config);
        let maintained = create_test_request(Priority::Normal);
        let maintained_id = maintained.id.clone();
        let peer = create_test_request(Priority::Normal);
        let peer_id = peer.id.clone();
        scheduler.submit(maintained).await.unwrap();
        scheduler.submit(peer).await.unwrap();

        let initial = scheduler
            .create_iteration_batch(BatchHint::simple(2))
            .unwrap();
        assert_eq!(initial.requests.len(), 2);

        let receipt = scheduler
            .defer_retry_after_execution_maintenance_epoch(std::slice::from_ref(&maintained_id), 7)
            .unwrap();
        assert_eq!(receipt.deferred_count(), 1);
        assert_eq!(receipt.latest_capacity_epoch(), 7);

        let fairness_iteration = scheduler
            .create_iteration_batch(BatchHint::simple(1))
            .unwrap();
        assert_eq!(fairness_iteration.requests.len(), 1);
        assert_eq!(fairness_iteration.requests[0].request.id, peer_id);
        let pressure = scheduler.trace_snapshot();
        assert_eq!(pressure.pressure_active_episodes, 0);
        assert_eq!(pressure.pressure_pending_release_fences, 0);

        let retry_iteration = scheduler
            .create_iteration_batch(BatchHint::simple(1))
            .unwrap();
        assert_eq!(retry_iteration.requests.len(), 1);
        assert_eq!(retry_iteration.requests[0].request.id, maintained_id);

        let replay = scheduler
            .defer_retry_after_execution_maintenance_epoch(std::slice::from_ref(&maintained_id), 7)
            .unwrap_err();
        assert!(replay
            .to_string()
            .contains("stale or concurrently blocked evidence"));
    }

    #[tokio::test]
    async fn execution_maintenance_retry_is_atomic_when_queue_identity_is_inconsistent() {
        let mut config = SchedulerConfig::default();
        config.max_running_requests = 2;
        let scheduler = ContinuousBatchScheduler::new(config);
        let first = create_test_request(Priority::Normal);
        let first_id = first.id.clone();
        let second = create_test_request(Priority::Normal);
        let second_id = second.id.clone();
        scheduler.submit(first).await.unwrap();
        scheduler.submit(second).await.unwrap();
        scheduler
            .create_iteration_batch(BatchHint::simple(2))
            .unwrap();

        let removed = {
            let mut prefill = scheduler.prefill_queue.write();
            let position = prefill
                .iter()
                .position(|request| request.inner.request.id == second_id)
                .unwrap();
            prefill.remove(position).unwrap()
        };
        let error = scheduler
            .defer_retry_after_execution_maintenance_epoch(&[first_id.clone(), second_id], 11)
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("lost an exact active logical frontier"));
        let prefill = scheduler.prefill_queue.read();
        let first = prefill
            .iter()
            .find(|request| request.inner.request.id == first_id)
            .unwrap();
        assert!(first.execution_maintenance_retry.is_none());
        assert!(first.last_execution_maintenance_capacity_epoch.is_none());
        drop(prefill);
        drop(removed);
    }

    #[tokio::test]
    async fn typed_dynamic_admission_defers_without_blocking_decode_or_smaller_work() {
        use ferrum_interfaces::vnext::{
            CapacityAvailabilityEpoch, CapacityAvailabilitySource, CapacityWaitCondition,
            DeferredAction,
        };
        use std::num::NonZeroU64;

        let mut config = SchedulerConfig::default();
        config.max_running_requests = 2;
        let scheduler = ContinuousBatchScheduler::new(config);
        let large = create_test_request_with_prompt_tokens(Priority::Normal, 512);
        let large_id = large.id.clone();
        let small = create_test_request_with_prompt_tokens(Priority::Normal, 8);
        let small_id = small.id.clone();
        scheduler.submit(large).await.unwrap();
        scheduler.submit(small).await.unwrap();

        let wake0 = AdmissionWakeEpochs::new(NonZeroU64::new(19).unwrap(), 0, 0, 0);
        let availability0 =
            [
                CapacityAvailabilityEpoch::new(CapacityAvailabilitySource::ActiveSequenceSlots, 1)
                    .unwrap(),
            ];
        let condition0 =
            CapacityWaitCondition::from_observation(19, availability0.to_vec()).unwrap();
        let mut first_probes = Vec::new();
        let mut first_probe = |request: &InferenceRequest| {
            first_probes.push(request.id.clone());
            if request.id == large_id {
                AdmissionProbeOutcome::Deferred(crate::vnext::AdmissionDeferral::new(
                    DeferredAction::WaitForRelease,
                    wake0,
                    condition0.clone(),
                ))
            } else {
                AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                    request_id: request.id.clone(),
                })
            }
        };
        let first = scheduler
            .next_batch_with_dynamic_admission(
                BatchHint::simple(2),
                AdmissionWakeSnapshot::new(wake0, &availability0),
                &mut first_probe,
            )
            .unwrap()
            .unwrap();
        assert_eq!(first_probes, vec![large_id.clone(), small_id.clone()]);
        assert_eq!(first.requests.len(), 1);
        assert_eq!(first.requests[0].request.id, small_id);
        assert_eq!(
            scheduler.trace_phase(&large_id),
            Some(RequestPhase::Waiting)
        );

        scheduler.mark_prefill_complete(&small_id, 8);
        let mut unchanged_probe_count = 0;
        let mut unchanged_probe = |request: &InferenceRequest| {
            unchanged_probe_count += 1;
            AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                request_id: request.id.clone(),
            })
        };
        let mut observations = Vec::new();
        let unchanged = scheduler
            .next_batch_with_dynamic_admission_observed(
                BatchHint::simple(2),
                AdmissionWakeSnapshot::new(wake0, &availability0),
                &mut unchanged_probe,
                &mut |observation| observations.push(observation),
            )
            .unwrap()
            .unwrap();
        assert_eq!(unchanged_probe_count, 0);
        assert_eq!(unchanged.requests.len(), 1);
        assert_eq!(unchanged.requests[0].request.id, small_id);
        assert!(matches!(
            observations.as_slice(),
            [ExecutorAdmissionQueueObservation::SkippedUnchanged {
                request_id,
                deferral,
                current,
                ..
            }] if request_id == &large_id
                && deferral.observed() == wake0
                && *current == wake0
        ));

        let wake1 = AdmissionWakeEpochs::new(NonZeroU64::new(19).unwrap(), 1, 0, 0);
        let availability1 =
            [
                CapacityAvailabilityEpoch::new(CapacityAvailabilitySource::ActiveSequenceSlots, 2)
                    .unwrap(),
            ];
        let mut released_probe_count = 0;
        let mut released_probe = |request: &InferenceRequest| {
            released_probe_count += 1;
            AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                request_id: request.id.clone(),
            })
        };
        let released = scheduler
            .next_batch_with_dynamic_admission(
                BatchHint::simple(2),
                AdmissionWakeSnapshot::new(wake1, &availability1),
                &mut released_probe,
            )
            .unwrap()
            .unwrap();
        assert_eq!(released_probe_count, 1);
        assert_eq!(released.requests.len(), 2);
        assert!(released
            .requests
            .iter()
            .any(|request| request.request.id == large_id));
        let trace = scheduler.trace_snapshot();
        assert_eq!(trace.legacy_waiting_admission_ticks, 0);
        assert_eq!(trace.dynamic_admission_probes, 3);
        assert_eq!(trace.dynamic_admission_skipped_unchanged, 1);
        assert_eq!(trace.dynamic_admission_deferred, 1);
    }

    #[tokio::test]
    async fn open_pressure_episode_closes_after_resumed_decode_commits_progress() {
        use ferrum_interfaces::vnext::{
            CapacityAvailabilityEpoch, CapacityAvailabilitySource, CapacityDomainId,
            CapacityWaitCondition, DeferredAction,
        };
        use std::num::NonZeroU64;

        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
        let blocked = create_test_request(Priority::Normal);
        let blocked_id = blocked.id.clone();
        let peer = create_test_request(Priority::Normal);
        let peer_id = peer.id.clone();
        scheduler.submit(blocked).await.unwrap();
        scheduler.submit(peer).await.unwrap();

        let source = CapacityAvailabilitySource::Domain(CapacityDomainId::new(8).unwrap());
        let availability0 = [CapacityAvailabilityEpoch::new(source, 1).unwrap()];
        let wake0 = AdmissionWakeEpochs::new(NonZeroU64::new(29).unwrap(), 0, 0, 0);
        let admitted = scheduler
            .next_batch_with_dynamic_admission(
                BatchHint::simple(2),
                AdmissionWakeSnapshot::new(wake0, &availability0),
                &mut |request| {
                    AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                        request_id: request.id.clone(),
                    })
                },
            )
            .unwrap()
            .unwrap();
        assert_eq!(admitted.size(), 2);
        scheduler.mark_prefill_complete(&blocked_id, 1);
        scheduler.mark_prefill_complete(&peer_id, 1);

        let condition =
            CapacityWaitCondition::from_observation(29, availability0.to_vec()).unwrap();
        assert_eq!(
            scheduler
                .defer_decode_for_execution_capacity(
                    std::slice::from_ref(&blocked_id),
                    AdmissionDeferral::new(DeferredAction::WaitForRelease, wake0, condition),
                    &ExecutionCapacityReleaseSnapshot::default(),
                )
                .unwrap(),
            ExecutionCapacityAction::Deferred { count: 1 }
        );
        assert_eq!(scheduler.trace_snapshot().pressure_active_episodes, 1);

        let availability1 = [CapacityAvailabilityEpoch::new(source, 2).unwrap()];
        let wake1 = AdmissionWakeEpochs::new(NonZeroU64::new(29).unwrap(), 1, 0, 0);
        let resumed = scheduler
            .next_batch_with_dynamic_admission(
                BatchHint::simple(2),
                AdmissionWakeSnapshot::new(wake1, &availability1),
                &mut |_| panic!("decode resume must not probe waiting admission"),
            )
            .unwrap()
            .expect("the exact-source wake must make the blocked decode schedulable");
        assert!(resumed
            .requests
            .iter()
            .any(|request| request.request.id == blocked_id));
        assert_eq!(
            scheduler.trace_snapshot().pressure_active_episodes,
            1,
            "an epoch change permits retry but is not execution-success evidence"
        );

        scheduler.update_decode_progress(&blocked_id, 1);
        let snapshot = scheduler.trace_snapshot();
        assert_eq!(snapshot.pressure_active_episodes, 0);
        let journal = scheduler.pressure_transition_journal();
        let satisfied = journal
            .iter()
            .find(|transition| transition.kind() == PressureTransitionKind::WaitSatisfied)
            .unwrap();
        let closed = journal
            .iter()
            .find(|transition| transition.kind() == PressureTransitionKind::Closed)
            .unwrap();
        assert!(satisfied.ordinal() < closed.ordinal());
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum SchedulerReplayRole {
        Blocked,
        Runnable,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct SchedulerReplayBatchMember {
        role: SchedulerReplayRole,
        tokens_processed: usize,
        tokens_to_process: Option<usize>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct SchedulerReplayProjection {
        batches: Vec<Vec<SchedulerReplayBatchMember>>,
        traces: Vec<ContinuousSchedulerTraceSnapshot>,
        observations: Vec<Vec<ExecutorAdmissionQueueObservation>>,
        journal: Vec<PressureTransition>,
    }

    fn scheduler_replay_batch(
        batch: &BatchPlan,
        blocked_id: &RequestId,
        runnable_id: &RequestId,
    ) -> Vec<SchedulerReplayBatchMember> {
        batch
            .requests
            .iter()
            .map(|request| {
                let role = if request.request.id == *blocked_id {
                    SchedulerReplayRole::Blocked
                } else if request.request.id == *runnable_id {
                    SchedulerReplayRole::Runnable
                } else {
                    panic!("unexpected scheduler replay request {}", request.request.id);
                };
                SchedulerReplayBatchMember {
                    role,
                    tokens_processed: request.tokens_processed,
                    tokens_to_process: request.tokens_to_process,
                }
            })
            .collect()
    }

    async fn collect_cross_phase_pressure_replay() -> SchedulerReplayProjection {
        use ferrum_interfaces::vnext::{
            CapacityAvailabilityEpoch, CapacityAvailabilitySource, CapacityWaitCondition,
            DeferredAction,
        };
        use std::num::NonZeroU64;

        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
        let mut blocked = create_test_request(Priority::Normal);
        blocked.id = RequestId(uuid::Uuid::from_u128(1));
        let blocked_id = blocked.id.clone();
        let mut runnable = create_test_request(Priority::Normal);
        runnable.id = RequestId(uuid::Uuid::from_u128(2));
        let runnable_id = runnable.id.clone();
        let mut batches = Vec::new();
        let mut traces = Vec::new();
        let mut observation_batches = Vec::new();
        scheduler.submit(blocked).await.unwrap();
        scheduler.submit(runnable).await.unwrap();

        let source = CapacityAvailabilitySource::ActiveSequenceSlots;
        let availability0 = [CapacityAvailabilityEpoch::new(source, 1).unwrap()];
        let wake0 = AdmissionWakeEpochs::new(NonZeroU64::new(29).unwrap(), 0, 0, 0);
        let admitted = scheduler
            .next_batch_with_dynamic_admission(
                BatchHint::simple(2),
                AdmissionWakeSnapshot::new(wake0, &availability0),
                &mut |request| {
                    AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                        request_id: request.id.clone(),
                    })
                },
            )
            .unwrap()
            .unwrap();
        assert_eq!(admitted.size(), 2);
        batches.push(scheduler_replay_batch(&admitted, &blocked_id, &runnable_id));
        scheduler.mark_prefill_complete(&blocked_id, 1);
        scheduler.mark_prefill_complete(&runnable_id, 1);
        traces.push(scheduler.trace_snapshot());

        let condition =
            CapacityWaitCondition::from_observation(29, availability0.to_vec()).unwrap();
        let deferral =
            AdmissionDeferral::new(DeferredAction::WaitForRelease, wake0, condition.clone());
        assert_eq!(
            scheduler
                .defer_decode_for_execution_capacity(
                    std::slice::from_ref(&blocked_id),
                    deferral.clone(),
                    &execution_capacity_release_snapshot([&blocked_id, &runnable_id], &condition,),
                )
                .unwrap(),
            ExecutionCapacityAction::Deferred { count: 1 }
        );
        traces.push(scheduler.trace_snapshot());
        let mut observations = Vec::new();
        let bypass = scheduler
            .next_batch_with_dynamic_admission_observed(
                BatchHint::simple(2),
                AdmissionWakeSnapshot::new(wake0, &availability0),
                &mut |_| panic!("no waiting admission probe is allowed"),
                &mut |observation| observations.push(observation),
            )
            .unwrap()
            .expect("runnable decode must bypass an unchanged blocked decode");
        assert_eq!(bypass.requests.len(), 1);
        assert_eq!(bypass.requests[0].request.id, runnable_id);
        batches.push(scheduler_replay_batch(&bypass, &blocked_id, &runnable_id));
        assert!(matches!(
            observations.as_slice(),
            [ExecutorAdmissionQueueObservation::DecodeSkippedUnchanged {
                request_id,
                current_wait_sources,
                ..
            }] if request_id == &blocked_id && current_wait_sources == &availability0
        ));
        observation_batches.push(observations.clone());

        let unavailable_action = scheduler
            .defer_decode_for_execution_capacity(
                std::slice::from_ref(&runnable_id),
                deferral.clone(),
                &ExecutionCapacityReleaseSnapshot::default(),
            )
            .unwrap();
        assert!(matches!(
            unavailable_action,
            ExecutionCapacityAction::InvariantViolation {
                violation
            } if violation.class() == PressureInvariantViolationClass::NoReleasableFrontier
        ));

        let action = scheduler
            .defer_decode_for_execution_capacity(
                std::slice::from_ref(&runnable_id),
                deferral,
                &execution_capacity_release_snapshot([&blocked_id, &runnable_id], &condition),
            )
            .unwrap();
        let ExecutionCapacityAction::YieldPlanned { transaction } = action else {
            panic!("the last runnable frontier must produce a pressure yield");
        };
        assert_eq!(transaction.victim_request_id(), &runnable_id);
        assert_eq!(transaction.progress_owner_id(), &blocked_id);
        let armed = scheduler
            .arm_execution_capacity_yield(&transaction)
            .unwrap();
        let completion = scheduler
            .complete_execution_capacity_yield(&transaction, 1, Some(0))
            .unwrap();
        let released = completion.release_transition_ordinal();
        let resumable = completion
            .resumable_transition_ordinal()
            .expect("live progress owner must become resumable");
        assert!(completion.victim_requeued());
        assert!(completion.progress_owner_resumable());
        assert!(completion.closed_transition_ordinal().is_none());
        assert!(transaction.planned_ordinal() < armed);
        assert!(armed < released);
        assert!(released < resumable);
        assert_eq!(
            scheduler
                .trace_snapshot()
                .execution_capacity_blocked_decode_len,
            1
        );
        assert_eq!(
            scheduler.trace_phase(&runnable_id),
            Some(RequestPhase::Waiting)
        );
        assert!(scheduler
            .passive_capacity_wait_condition()
            .unwrap()
            .is_some());
        traces.push(scheduler.trace_snapshot());

        let availability1 = [CapacityAvailabilityEpoch::new(source, 2).unwrap()];
        let released = AdmissionWakeEpochs::new(NonZeroU64::new(29).unwrap(), 2, 0, 0);
        observations.clear();
        let resumed = scheduler
            .next_batch_with_dynamic_admission_observed(
                BatchHint::simple(2),
                AdmissionWakeSnapshot::new(released, &availability1),
                &mut |_| panic!("decode resume does not probe waiting admission"),
                &mut |observation| observations.push(observation),
            )
            .unwrap()
            .expect("relevant source change must resume the selected progress owner");
        assert_eq!(resumed.requests.len(), 1);
        assert_eq!(resumed.requests[0].request.id, blocked_id);
        batches.push(scheduler_replay_batch(&resumed, &blocked_id, &runnable_id));
        let resumed_ids = observations
            .iter()
            .filter_map(|observation| match observation {
                ExecutorAdmissionQueueObservation::DecodeResumed {
                    request_id,
                    current_wait_sources,
                    exact_source_changed: true,
                    policy_epoch_changed: false,
                    ..
                } if current_wait_sources == &availability1 => Some(request_id.clone()),
                _ => None,
            })
            .collect::<HashSet<_>>();
        assert_eq!(resumed_ids, HashSet::from([blocked_id.clone()]));
        observation_batches.push(observations.clone());
        assert_eq!(
            scheduler
                .trace_snapshot()
                .execution_capacity_blocked_decode_len,
            0
        );

        scheduler.update_decode_progress(&blocked_id, 1);
        observations.clear();
        let owner_only = scheduler
            .next_batch_with_dynamic_admission_observed(
                BatchHint::simple(2),
                AdmissionWakeSnapshot::new(released, &availability1),
                &mut |_| panic!("owner token progress must not probe a held victim"),
                &mut |observation| observations.push(observation),
            )
            .unwrap()
            .expect("progress owner must continue while the victim remains held");
        assert_eq!(owner_only.requests.len(), 1);
        assert_eq!(owner_only.requests[0].request.id, blocked_id);
        batches.push(scheduler_replay_batch(
            &owner_only,
            &blocked_id,
            &runnable_id,
        ));
        assert!(!observations.iter().any(|observation| matches!(
            observation,
            ExecutorAdmissionQueueObservation::PressureHoldReleased {
                request_id,
                ..
            } if request_id == &runnable_id
        )));
        observation_batches.push(observations.clone());
        assert_eq!(scheduler.trace_snapshot().pressure_active_episodes, 1);
        traces.push(scheduler.trace_snapshot());

        let availability2 = [CapacityAvailabilityEpoch::new(source, 3).unwrap()];
        let wake2 = AdmissionWakeEpochs::new(NonZeroU64::new(29).unwrap(), 3, 0, 0);
        let owner_pressure =
            CapacityWaitCondition::from_observation(29, availability2.to_vec()).unwrap();
        let owner_deferral = AdmissionDeferral::new(
            DeferredAction::WaitForRelease,
            wake2,
            owner_pressure.clone(),
        );
        let owner_action = scheduler
            .defer_decode_for_execution_capacity(
                std::slice::from_ref(&blocked_id),
                owner_deferral,
                &execution_capacity_release_snapshot([&blocked_id], &owner_pressure),
            )
            .unwrap();
        let ExecutionCapacityAction::YieldPlanned {
            transaction: owner_transaction,
        } = owner_action
        else {
            panic!("the stable progress owner must self recompute under renewed pressure");
        };
        assert_eq!(owner_transaction.kind(), PressureYieldKind::SelfRecompute);
        assert_eq!(owner_transaction.progress_owner_id(), &blocked_id);
        assert_eq!(owner_transaction.victim_request_id(), &blocked_id);
        scheduler
            .arm_execution_capacity_yield(&owner_transaction)
            .unwrap();
        let owner_completion = scheduler
            .complete_execution_capacity_yield(&owner_transaction, 1, Some(0))
            .unwrap();
        assert_eq!(
            owner_completion.disposition(),
            ExecutionCapacityYieldDisposition::ProgressOwnerAdmissionPending
        );
        assert!(owner_completion
            .owner_admission_pending_transition_ordinal()
            .is_some());
        assert!(owner_completion.closed_transition_ordinal().is_none());
        assert_eq!(scheduler.trace_snapshot().waiting_queue_len, 2);

        let mut owner_recompute_probes = Vec::new();
        let owner_recompute = scheduler
            .next_batch_with_dynamic_admission(
                BatchHint::simple(2),
                AdmissionWakeSnapshot::new(wake2, &availability2),
                &mut |request| {
                    owner_recompute_probes.push(request.id.clone());
                    AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                        request_id: request.id.clone(),
                    })
                },
            )
            .unwrap()
            .expect("the stable owner must be the only admission-eligible recompute");
        assert_eq!(owner_recompute_probes, vec![blocked_id.clone()]);
        assert_eq!(owner_recompute.requests.len(), 1);
        assert_eq!(owner_recompute.requests[0].request.id, blocked_id);
        batches.push(scheduler_replay_batch(
            &owner_recompute,
            &blocked_id,
            &runnable_id,
        ));
        assert!(matches!(
            scheduler
                .pressure_coordinator
                .lock()
                .hold_status(&runnable_id),
            PressureHoldStatus::Held { .. }
        ));
        let journal = scheduler.pressure_transition_journal();
        let admission_pending = journal
            .iter()
            .find(|event| event.kind() == PressureTransitionKind::OwnerAdmissionPending)
            .expect("owner self recompute must publish admission-pending state");
        let owner_admitted = journal
            .iter()
            .find(|event| event.kind() == PressureTransitionKind::OwnerAdmitted)
            .expect("typed admission receipt must commit owner admission");
        assert!(admission_pending.ordinal() < owner_admitted.ordinal());
        scheduler.mark_prefill_complete(&blocked_id, 1);

        let response = InferenceResponse {
            request_id: blocked_id.clone(),
            text: String::new(),
            tokens: Vec::new(),
            finish_reason: ferrum_types::FinishReason::Length,
            usage: ferrum_types::TokenUsage::new(0, 0),
            latency_ms: 0,
            created_at: chrono::Utc::now(),
            metadata: Default::default(),
            api_response: None,
            execution_evidence: None,
        };
        scheduler
            .complete(blocked_id.clone(), &response)
            .await
            .unwrap();

        observations.clear();
        let callback_order = std::cell::RefCell::new(Vec::new());
        let admitted = scheduler
            .next_batch_with_dynamic_admission_observed(
                BatchHint::simple(2),
                AdmissionWakeSnapshot::new(wake2, &availability2),
                &mut |request| {
                    callback_order.borrow_mut().push("admission_probe");
                    AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                        request_id: request.id.clone(),
                    })
                },
                &mut |observation| {
                    if matches!(
                        observation,
                        ExecutorAdmissionQueueObservation::PressureHoldReleased { .. }
                    ) {
                        callback_order.borrow_mut().push("pressure_hold_released");
                    }
                    observations.push(observation);
                },
            )
            .unwrap()
            .expect("owner terminal release must admit the yielded frontier");
        assert_eq!(
            callback_order.into_inner(),
            vec!["pressure_hold_released", "admission_probe"],
            "the trace capture boundary must observe causal hold release before re-admission"
        );
        assert_eq!(admitted.requests.len(), 1);
        assert_eq!(admitted.requests[0].request.id, runnable_id);
        batches.push(scheduler_replay_batch(&admitted, &blocked_id, &runnable_id));
        assert!(observations.iter().any(|observation| matches!(
            observation,
            ExecutorAdmissionQueueObservation::PressureHoldReleased {
                request_id,
                progress_owner_id,
                reason: PressureHoldReleaseReason::OwnerTerminal,
                ..
            } if request_id == &runnable_id && progress_owner_id == &blocked_id
        )));
        observation_batches.push(observations.clone());
        let journal = scheduler.pressure_transition_journal();
        assert!(journal
            .windows(2)
            .all(|pair| pair[0].ordinal() < pair[1].ordinal()));
        let final_trace = scheduler.trace_snapshot();
        assert_eq!(final_trace.pressure_active_episodes, 0);
        assert_eq!(final_trace.pressure_dropped_journal_entries, 0);
        traces.push(final_trace);
        SchedulerReplayProjection {
            batches,
            traces,
            observations: observation_batches,
            journal,
        }
    }

    #[tokio::test]
    async fn cross_phase_pressure_yield_holds_victim_until_owner_terminal() {
        let projection = collect_cross_phase_pressure_replay().await;
        assert_eq!(projection.batches.len(), 6);
        assert!(!projection.journal.is_empty());
    }

    #[tokio::test]
    async fn cross_phase_scheduler_is_deterministic_one_hundred_of_one_hundred() {
        const REPLAY_COUNT: usize = 100;
        let expected = collect_cross_phase_pressure_replay().await;
        for ordinal in 1..REPLAY_COUNT {
            assert_eq!(
                collect_cross_phase_pressure_replay().await,
                expected,
                "scheduler execution {ordinal} diverged from execution 0"
            );
        }
        println!(
            "FERRUM G04 SCHEDULER DETERMINISM KEEP: deterministic_executions={REPLAY_COUNT}/{REPLAY_COUNT} batch_ticks={} trace_snapshots={} observation_batches={} journal_transitions={}",
            expected.batches.len(),
            expected.traces.len(),
            expected.observations.len(),
            expected.journal.len(),
        );
    }

    async fn collect_constrained_decode_membership() -> Vec<Vec<RequestId>> {
        const REQUEST_COUNT: usize = 8;
        const BATCH_LIMIT: usize = 3;
        const ROUND_COUNT: usize = 4;

        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: REQUEST_COUNT,
            ..SchedulerConfig::default()
        });
        let request_ids = (0..REQUEST_COUNT)
            .map(|ordinal| RequestId(uuid::Uuid::from_u128(0x100 + ordinal as u128)))
            .collect::<Vec<_>>();
        for request_id in &request_ids {
            let mut request = create_test_request(Priority::Normal);
            request.id = request_id.clone();
            scheduler.submit(request).await.unwrap();
        }

        let initial = scheduler
            .create_iteration_batch(BatchHint::simple(REQUEST_COUNT))
            .expect("all constrained-decode fixtures must enter prefill");
        assert_eq!(
            initial
                .requests
                .iter()
                .map(|request| request.request.id.clone())
                .collect::<Vec<_>>(),
            request_ids,
            "initial admission must preserve the scheduler-owned queue order"
        );
        for request_id in &request_ids {
            scheduler.mark_prefill_complete(request_id, 1);
        }

        let mut progress = HashMap::<RequestId, usize>::new();
        let mut memberships = Vec::new();
        for _ in 0..ROUND_COUNT {
            let batch = scheduler
                .create_iteration_batch(BatchHint::simple(BATCH_LIMIT))
                .expect("eligible decode work must fill every constrained batch");
            let membership = batch
                .requests
                .iter()
                .map(|request| request.request.id.clone())
                .collect::<Vec<_>>();
            assert_eq!(membership.len(), BATCH_LIMIT);
            assert_eq!(membership.iter().collect::<HashSet<_>>().len(), BATCH_LIMIT);
            for request_id in &membership {
                let generated = progress.entry(request_id.clone()).or_default();
                *generated += 1;
                scheduler.update_decode_progress(request_id, *generated);
            }
            memberships.push(membership);
        }
        assert!(
            request_ids
                .iter()
                .all(|request_id| progress.contains_key(request_id)),
            "round-robin decode selection must schedule every eligible frontier"
        );
        memberships
    }

    #[tokio::test]
    async fn constrained_decode_membership_is_stable_and_fair_one_hundred_of_one_hundred() {
        const EXECUTION_COUNT: usize = 100;
        let expected = collect_constrained_decode_membership().await;
        let mut selection_counts = HashMap::<RequestId, usize>::new();
        for request_id in expected.iter().flatten() {
            *selection_counts.entry(request_id.clone()).or_default() += 1;
        }
        let min_count = selection_counts.values().copied().min().unwrap();
        let max_count = selection_counts.values().copied().max().unwrap();
        assert!(
            max_count - min_count <= 1,
            "constrained round-robin selection must remain balanced"
        );
        for ordinal in 1..EXECUTION_COUNT {
            assert_eq!(
                collect_constrained_decode_membership().await,
                expected,
                "constrained decode execution {ordinal} diverged from execution 0"
            );
        }
        println!(
            "FERRUM G04 CONSTRAINED DECODE DETERMINISM KEEP: deterministic_executions={EXECUTION_COUNT}/{EXECUTION_COUNT} requests=8 batch_limit=3 rounds=4"
        );
    }

    #[tokio::test]
    async fn decode_cursor_keeps_stable_successor_after_swap_remove() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            ..SchedulerConfig::default()
        });
        let request_ids = activate_decode_requests(&scheduler, 4);
        let first = scheduler
            .create_iteration_batch(BatchHint::simple(3))
            .expect("first constrained decode batch must be scheduled");
        assert_eq!(
            first
                .requests
                .iter()
                .map(|request| request.request.id.clone())
                .collect::<Vec<_>>(),
            request_ids[..3]
        );
        assert_eq!(
            scheduler.trace_snapshot().decode_selection_cursor,
            Some(request_ids[3].clone())
        );

        assert!(scheduler.cancel(request_ids[1].clone()).await.unwrap());
        assert_eq!(
            scheduler.trace_snapshot().decode_selection_cursor,
            Some(request_ids[3].clone()),
            "removing an earlier slot must not move the stable round-robin frontier"
        );
        let next = scheduler
            .create_iteration_batch(BatchHint::simple(1))
            .expect("cursor successor must remain runnable");
        assert_eq!(next.requests[0].request.id, request_ids[3]);
    }

    #[tokio::test]
    async fn empty_decode_queue_resets_cursor_before_new_cohort() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            ..SchedulerConfig::default()
        });
        let old_ids = activate_decode_requests(&scheduler, 4);
        let first = scheduler
            .create_iteration_batch(BatchHint::simple(1))
            .expect("old cohort must establish a nonzero cursor");
        assert_eq!(first.requests[0].request.id, old_ids[0]);
        assert_eq!(
            scheduler.trace_snapshot().decode_selection_cursor,
            Some(old_ids[1].clone())
        );
        for request_id in old_ids {
            assert!(scheduler.cancel(request_id).await.unwrap());
        }
        assert_eq!(scheduler.trace_snapshot().decode_selection_cursor, None);

        let new_ids = activate_decode_requests(&scheduler, 3);
        let new_first = scheduler
            .create_iteration_batch(BatchHint::simple(1))
            .expect("new cohort must start from its first admission");
        assert_eq!(new_first.requests[0].request.id, new_ids[0]);
    }

    #[test]
    fn readiness_blocked_decode_rejoins_within_one_round_robin_rotation() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            ..SchedulerConfig::default()
        });
        let request_ids = activate_decode_requests(&scheduler, 4);
        let first = scheduler
            .create_iteration_batch(BatchHint::simple(1))
            .expect("first decode frontier must run");
        assert_eq!(first.requests[0].request.id, request_ids[0]);
        scheduler.update_decode_progress(&request_ids[0], 1);

        let blocked = scheduler
            .defer_for_execution_readiness(std::slice::from_ref(&request_ids[1]))
            .unwrap();
        for expected_id in [&request_ids[2], &request_ids[3], &request_ids[0]] {
            let batch = scheduler
                .create_iteration_batch(BatchHint::simple(1))
                .expect("an eligible peer must keep decode progressing");
            assert_eq!(&batch.requests[0].request.id, expected_id);
            scheduler.update_decode_progress(expected_id, 2);
        }

        assert!(blocked.wake().mark_ready());
        let resumed = scheduler
            .create_iteration_batch(BatchHint::simple(1))
            .expect("ready frontier must rejoin at its preserved cursor");
        assert_eq!(resumed.requests[0].request.id, request_ids[1]);
    }

    #[tokio::test]
    async fn disjoint_release_footprints_self_recompute_without_stranded_episode() {
        use ferrum_interfaces::vnext::{
            CapacityAvailabilityEpoch, CapacityAvailabilitySource, CapacityDomainId,
            CapacityWaitCondition, DeferredAction,
        };
        use std::num::NonZeroU64;

        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
        let first = create_test_request(Priority::Normal);
        let first_id = first.id.clone();
        let second = create_test_request(Priority::Normal);
        let second_id = second.id.clone();
        scheduler.submit(first).await.unwrap();
        scheduler.submit(second).await.unwrap();

        let domain_two = CapacityAvailabilitySource::Domain(CapacityDomainId::new(2).unwrap());
        let domain_four = CapacityAvailabilitySource::Domain(CapacityDomainId::new(4).unwrap());
        let plan_budget = CapacityAvailabilitySource::PlanDeviceBudget;
        let availability = [
            CapacityAvailabilityEpoch::new(domain_two, 176).unwrap(),
            CapacityAvailabilityEpoch::new(domain_four, 136).unwrap(),
            CapacityAvailabilityEpoch::new(plan_budget, 1).unwrap(),
        ];
        let wake = AdmissionWakeEpochs::new(NonZeroU64::new(41).unwrap(), 0, 0, 0);
        scheduler
            .next_batch_with_dynamic_admission(
                BatchHint::simple(2),
                AdmissionWakeSnapshot::new(wake, &availability),
                &mut |request| {
                    AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                        request_id: request.id.clone(),
                    })
                },
            )
            .unwrap()
            .unwrap();
        scheduler.mark_prefill_complete(&first_id, 1);
        scheduler.mark_prefill_complete(&second_id, 1);

        let wait_for = |source, epoch| {
            CapacityWaitCondition::from_observation(
                41,
                vec![
                    CapacityAvailabilityEpoch::new(source, epoch).unwrap(),
                    CapacityAvailabilityEpoch::new(plan_budget, 1).unwrap(),
                ],
            )
            .unwrap()
        };
        let first_wait = wait_for(domain_four, 136);
        let second_wait = wait_for(domain_two, 176);
        let release_snapshot = ExecutionCapacityReleaseSnapshot::new([
            (first_id.clone(), vec![domain_four]),
            (second_id.clone(), vec![domain_two]),
        ]);

        let first_action = scheduler
            .defer_decode_for_execution_capacity(
                std::slice::from_ref(&first_id),
                AdmissionDeferral::new(DeferredAction::WaitForRelease, wake, first_wait),
                &release_snapshot,
            )
            .unwrap();
        let ExecutionCapacityAction::YieldPlanned {
            transaction: first_transaction,
        } = first_action
        else {
            panic!("a disjoint runnable footprint must not suppress exact-source self recompute");
        };
        assert_eq!(first_transaction.kind(), PressureYieldKind::SelfRecompute);
        assert_eq!(first_transaction.victim_request_id(), &first_id);
        assert_eq!(
            scheduler.trace_snapshot().pressure_pending_release_fences,
            1
        );
        scheduler
            .arm_execution_capacity_yield(&first_transaction)
            .unwrap();
        assert!(scheduler
            .complete_execution_capacity_yield(&first_transaction, 1, None)
            .unwrap()
            .victim_requeued());

        let second_action = scheduler
            .defer_decode_for_execution_capacity(
                std::slice::from_ref(&second_id),
                AdmissionDeferral::new(DeferredAction::WaitForRelease, wake, second_wait),
                &release_snapshot,
            )
            .unwrap();
        let ExecutionCapacityAction::YieldPlanned {
            transaction: second_transaction,
        } = second_action
        else {
            panic!("the remaining exact-source owner must self recompute");
        };
        assert_eq!(second_transaction.kind(), PressureYieldKind::SelfRecompute);
        assert_eq!(second_transaction.victim_request_id(), &second_id);
        scheduler
            .arm_execution_capacity_yield(&second_transaction)
            .unwrap();
        assert!(scheduler
            .complete_execution_capacity_yield(&second_transaction, 1, None)
            .unwrap()
            .victim_requeued());

        let snapshot = scheduler.trace_snapshot();
        assert_eq!(snapshot.pressure_active_episodes, 0);
        assert_eq!(snapshot.pressure_pending_release_fences, 0);
        assert_eq!(snapshot.waiting_queue_len, 2);
        assert_eq!(snapshot.execution_capacity_blocked_decode_len, 0);
    }

    #[tokio::test]
    async fn release_fence_preserves_stable_owner_across_retargeted_condition() {
        use ferrum_interfaces::vnext::{
            CapacityAvailabilityEpoch, CapacityAvailabilitySource, CapacityDomainId,
            CapacityWaitCondition, DeferredAction,
        };
        use std::num::NonZeroU64;

        let mut config = SchedulerConfig::default();
        config.max_running_requests = 2;
        let scheduler = ContinuousBatchScheduler::new(config);
        let owner = create_test_request(Priority::Normal);
        let owner_id = owner.id.clone();
        let victim = create_test_request(Priority::Normal);
        let victim_id = victim.id.clone();
        scheduler.submit(owner).await.unwrap();
        scheduler.submit(victim).await.unwrap();

        let wait_for = |domain: u32, epoch: u64| {
            CapacityWaitCondition::from_observation(
                41,
                vec![
                    CapacityAvailabilityEpoch::new(
                        CapacityAvailabilitySource::Domain(CapacityDomainId::new(domain).unwrap()),
                        epoch,
                    )
                    .unwrap(),
                    CapacityAvailabilityEpoch::new(CapacityAvailabilitySource::PlanDeviceBudget, 1)
                        .unwrap(),
                ],
            )
            .unwrap()
        };
        let original_wait = wait_for(4, 136);
        let retargeted_wait = wait_for(2, 178);
        let availability = [
            CapacityAvailabilityEpoch::new(
                CapacityAvailabilitySource::Domain(CapacityDomainId::new(2).unwrap()),
                178,
            )
            .unwrap(),
            CapacityAvailabilityEpoch::new(
                CapacityAvailabilitySource::Domain(CapacityDomainId::new(4).unwrap()),
                136,
            )
            .unwrap(),
            CapacityAvailabilityEpoch::new(CapacityAvailabilitySource::PlanDeviceBudget, 1)
                .unwrap(),
        ];
        let wake = AdmissionWakeEpochs::new(NonZeroU64::new(41).unwrap(), 0, 0, 0);
        scheduler
            .next_batch_with_dynamic_admission(
                BatchHint::simple(2),
                AdmissionWakeSnapshot::new(wake, &availability),
                &mut |request| {
                    AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                        request_id: request.id.clone(),
                    })
                },
            )
            .unwrap()
            .unwrap();
        scheduler.mark_prefill_complete(&owner_id, 1);
        scheduler.mark_prefill_complete(&victim_id, 1);

        let original_deferral =
            AdmissionDeferral::new(DeferredAction::WaitForRelease, wake, original_wait.clone());
        assert_eq!(
            scheduler
                .defer_decode_for_execution_capacity(
                    std::slice::from_ref(&owner_id),
                    original_deferral.clone(),
                    &execution_capacity_release_snapshot([&owner_id, &victim_id], &original_wait,),
                )
                .unwrap(),
            ExecutionCapacityAction::Deferred { count: 1 }
        );
        let action = scheduler
            .defer_decode_for_execution_capacity(
                std::slice::from_ref(&victim_id),
                original_deferral,
                &execution_capacity_release_snapshot([&owner_id, &victim_id], &original_wait),
            )
            .unwrap();
        let ExecutionCapacityAction::YieldPlanned { transaction } = action else {
            panic!("all-blocked original domain must plan a typed yield");
        };
        assert_eq!(transaction.progress_owner_id(), &owner_id);
        assert_eq!(transaction.victim_request_id(), &victim_id);
        scheduler
            .arm_execution_capacity_yield(&transaction)
            .unwrap();

        let retargeted_deferral = AdmissionDeferral::new(
            DeferredAction::WaitForRelease,
            wake,
            retargeted_wait.clone(),
        );
        assert_eq!(
            scheduler
                .defer_decode_for_execution_capacity(
                    std::slice::from_ref(&owner_id),
                    retargeted_deferral,
                    &execution_capacity_release_snapshot(
                        [&owner_id, &victim_id],
                        &retargeted_wait,
                    ),
                )
                .unwrap(),
            ExecutionCapacityAction::Deferred { count: 1 }
        );

        let completion = scheduler
            .complete_execution_capacity_yield(&transaction, 1, Some(0))
            .unwrap();
        assert!(completion.victim_requeued());
        assert!(completion.progress_owner_resumable());
        assert!(completion.resumable_transition_ordinal().is_some());
        assert!(completion.closed_transition_ordinal().is_none());
        assert_eq!(completion.closed_reason(), None);
        assert!(matches!(
            scheduler
                .pressure_coordinator
                .lock()
                .hold_status(&victim_id),
            PressureHoldStatus::Held { .. }
        ));
    }

    #[tokio::test]
    async fn pressure_yield_releases_after_phase_independent_owner_terminal() {
        use ferrum_interfaces::vnext::{
            CapacityAvailabilityEpoch, CapacityAvailabilitySource, CapacityWaitCondition,
            DeferredAction,
        };
        use std::num::NonZeroU64;

        let mut config = SchedulerConfig::default();
        config.max_running_requests = 2;
        let scheduler = ContinuousBatchScheduler::new(config);
        let victim = create_test_request(Priority::Normal);
        let victim_id = victim.id.clone();
        let owner = create_test_request(Priority::Normal);
        let owner_id = owner.id.clone();
        scheduler.submit(victim).await.unwrap();
        scheduler.submit(owner).await.unwrap();

        let source = CapacityAvailabilitySource::ActiveSequenceSlots;
        let availability = [CapacityAvailabilityEpoch::new(source, 1).unwrap()];
        let wake = AdmissionWakeEpochs::new(NonZeroU64::new(41).unwrap(), 0, 0, 0);
        scheduler
            .next_batch_with_dynamic_admission(
                BatchHint::simple(4),
                AdmissionWakeSnapshot::new(wake, &availability),
                &mut |request| {
                    AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                        request_id: request.id.clone(),
                    })
                },
            )
            .unwrap()
            .unwrap();
        scheduler.mark_prefill_complete(&victim_id, 1);
        scheduler.mark_prefill_complete(&owner_id, 1);
        let condition = CapacityWaitCondition::from_observation(41, availability.to_vec()).unwrap();
        let deferral =
            AdmissionDeferral::new(DeferredAction::WaitForRelease, wake, condition.clone());
        assert_eq!(
            scheduler
                .defer_decode_for_execution_capacity(
                    std::slice::from_ref(&victim_id),
                    deferral.clone(),
                    &execution_capacity_release_snapshot([&victim_id, &owner_id], &condition,),
                )
                .unwrap(),
            ExecutionCapacityAction::Deferred { count: 1 }
        );
        let action = scheduler
            .defer_decode_for_execution_capacity(
                std::slice::from_ref(&owner_id),
                deferral,
                &execution_capacity_release_snapshot([&victim_id, &owner_id], &condition),
            )
            .unwrap();
        let ExecutionCapacityAction::YieldPlanned { transaction } = action else {
            panic!("capacity pressure must plan a typed yield");
        };
        assert_eq!(transaction.progress_owner_id(), &victim_id);
        assert_eq!(transaction.victim_request_id(), &owner_id);
        scheduler
            .arm_execution_capacity_yield(&transaction)
            .unwrap();
        assert!(scheduler
            .complete_execution_capacity_yield(&transaction, 1, None)
            .unwrap()
            .victim_requeued());
        assert!(scheduler.cancel(victim_id.clone()).await.unwrap());

        let mut probes = Vec::new();
        let mut observations = Vec::new();
        let released = scheduler
            .next_batch_with_dynamic_admission_observed(
                BatchHint::simple(1),
                AdmissionWakeSnapshot::new(wake, &availability),
                &mut |request| {
                    probes.push(request.id.clone());
                    AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                        request_id: request.id.clone(),
                    })
                },
                &mut |observation| observations.push(observation),
            )
            .unwrap()
            .expect("owner terminal state must release the yielded frontier");
        assert_eq!(probes, vec![owner_id.clone()]);
        assert!(released
            .requests
            .iter()
            .any(|request| request.request.id == owner_id));
        assert!(observations.iter().any(|observation| matches!(
            observation,
            ExecutorAdmissionQueueObservation::PressureHoldReleased {
                request_id,
                progress_owner_id,
                reason: PressureHoldReleaseReason::OwnerTerminal,
                ..
            } if request_id == &owner_id && progress_owner_id == &victim_id
        )));
    }

    #[tokio::test]
    async fn lone_active_decode_capacity_deferral_self_recomputes_to_release_its_source() {
        use ferrum_interfaces::vnext::{
            CapacityAvailabilityEpoch, CapacityAvailabilitySource, CapacityWaitCondition,
            DeferredAction,
        };
        use std::num::NonZeroU64;

        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
        let request = create_test_request(Priority::Normal);
        let request_id = request.id.clone();
        scheduler.submit(request).await.unwrap();

        let source = CapacityAvailabilitySource::ActiveSequenceSlots;
        let availability0 = [CapacityAvailabilityEpoch::new(source, 3).unwrap()];
        let wake0 = AdmissionWakeEpochs::new(NonZeroU64::new(37).unwrap(), 0, 0, 0);
        scheduler
            .next_batch_with_dynamic_admission(
                BatchHint::simple(1),
                AdmissionWakeSnapshot::new(wake0, &availability0),
                &mut |request| {
                    AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                        request_id: request.id.clone(),
                    })
                },
            )
            .unwrap()
            .expect("request must enter prefill before decode");
        scheduler.mark_prefill_complete(&request_id, 1);

        let condition =
            CapacityWaitCondition::from_observation(37, availability0.to_vec()).unwrap();
        let deferral =
            AdmissionDeferral::new(DeferredAction::WaitForRelease, wake0, condition.clone());
        let action = scheduler
            .defer_decode_for_execution_capacity(
                std::slice::from_ref(&request_id),
                deferral,
                &execution_capacity_release_snapshot([&request_id], &condition),
            )
            .unwrap();
        let ExecutionCapacityAction::YieldPlanned { transaction } = action else {
            panic!("a lone releasable decode must plan a typed self recompute");
        };
        assert_eq!(transaction.kind(), PressureYieldKind::SelfRecompute);
        assert_eq!(transaction.victim_request_id(), &request_id);
        assert_eq!(transaction.progress_owner_id(), &request_id);

        scheduler
            .arm_execution_capacity_yield(&transaction)
            .unwrap();
        let completion = scheduler
            .complete_execution_capacity_yield(&transaction, 1, None)
            .unwrap();
        assert!(completion.victim_requeued());
        assert!(!completion.progress_owner_resumable());
        assert_eq!(
            completion.disposition(),
            ExecutionCapacityYieldDisposition::SelfRecomputeQueued
        );
        assert!(completion.closed_transition_ordinal().is_some());
        assert_eq!(completion.closed_reason(), None);

        let snapshot = scheduler.trace_snapshot();
        assert_eq!(snapshot.decode_queue_len, 0);
        assert_eq!(snapshot.waiting_queue_len, 1);
        assert_eq!(snapshot.pressure_active_episodes, 0);
        assert_eq!(snapshot.pressure_pending_release_fences, 0);
    }

    #[tokio::test]
    async fn active_prefill_capacity_deferral_retries_the_exact_chunk_after_source_change() {
        use ferrum_interfaces::vnext::{
            CapacityAvailabilityEpoch, CapacityAvailabilitySource, CapacityWaitCondition,
            DeferredAction,
        };
        use std::num::NonZeroU64;

        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 1,
            prefill_step_chunk: Some(3),
            ..SchedulerConfig::default()
        });
        let request = create_test_request_with_prompt_tokens(Priority::Normal, 8);
        let request_id = request.id.clone();
        scheduler.submit(request).await.unwrap();

        let source = CapacityAvailabilitySource::ActiveSequenceSlots;
        let availability0 = [CapacityAvailabilityEpoch::new(source, 1).unwrap()];
        let wake0 = AdmissionWakeEpochs::new(NonZeroU64::new(31).unwrap(), 0, 0, 0);
        let first = scheduler
            .next_batch_with_dynamic_admission(
                BatchHint::simple(1),
                AdmissionWakeSnapshot::new(wake0, &availability0),
                &mut |request| {
                    AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                        request_id: request.id.clone(),
                    })
                },
            )
            .unwrap()
            .unwrap();
        assert_eq!(first.requests[0].tokens_processed, 0);
        assert_eq!(first.requests[0].tokens_to_process, Some(3));
        assert!(!scheduler.mark_prefill_chunk_processed(&request_id, 8, 3));

        let condition =
            CapacityWaitCondition::from_observation(31, availability0.to_vec()).unwrap();
        let deferral =
            AdmissionDeferral::new(DeferredAction::WaitForRelease, wake0, condition.clone());
        assert_eq!(
            scheduler
                .defer_prefill_for_execution_capacity(
                    &request_id,
                    deferral,
                    &execution_capacity_release_snapshot([&request_id], &condition),
                )
                .unwrap(),
            ExecutionCapacityAction::Deferred { count: 1 }
        );

        let mut observations = Vec::new();
        assert!(scheduler
            .next_batch_with_dynamic_admission_observed(
                BatchHint::simple(1),
                AdmissionWakeSnapshot::new(wake0, &availability0),
                &mut |_| panic!("active prefill must not re-enter admission"),
                &mut |observation| observations.push(observation),
            )
            .unwrap()
            .is_none());
        assert!(observations.iter().any(|observation| matches!(
            observation,
            ExecutorAdmissionQueueObservation::PrefillSkippedUnchanged {
                request_id: observed_id,
                current_wait_sources,
                ..
            } if observed_id == &request_id && current_wait_sources == &availability0
        )));
        assert_eq!(
            scheduler.passive_capacity_wait_condition().unwrap(),
            Some(condition)
        );
        assert_eq!(
            scheduler
                .trace_snapshot()
                .execution_capacity_blocked_prefill_len,
            1
        );

        let availability1 = [CapacityAvailabilityEpoch::new(source, 2).unwrap()];
        let wake1 = AdmissionWakeEpochs::new(NonZeroU64::new(31).unwrap(), 1, 0, 0);
        observations.clear();
        let resumed = scheduler
            .next_batch_with_dynamic_admission_observed(
                BatchHint::simple(1),
                AdmissionWakeSnapshot::new(wake1, &availability1),
                &mut |_| panic!("active prefill resume must not re-enter admission"),
                &mut |observation| observations.push(observation),
            )
            .unwrap()
            .expect("source movement must resume the deferred prefill");
        assert_eq!(resumed.requests[0].request.id, request_id);
        assert_eq!(resumed.requests[0].tokens_processed, 3);
        assert_eq!(resumed.requests[0].tokens_to_process, Some(3));
        assert!(observations.iter().any(|observation| matches!(
            observation,
            ExecutorAdmissionQueueObservation::PrefillResumed {
                exact_source_changed: true,
                policy_epoch_changed: false,
                ..
            }
        )));
        assert_eq!(
            scheduler
                .trace_snapshot()
                .execution_capacity_blocked_prefill_len,
            0
        );
    }

    #[tokio::test]
    async fn partial_prefill_completion_limits_the_next_scheduled_frontier() {
        use ferrum_interfaces::vnext::{CapacityAvailabilityEpoch, CapacityAvailabilitySource};
        use std::num::NonZeroU64;

        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 1,
            prefill_step_chunk: Some(4),
            ..SchedulerConfig::default()
        });
        let request = create_test_request_with_prompt_tokens(Priority::Normal, 8);
        let request_id = request.id.clone();
        scheduler.submit(request).await.unwrap();
        let availability =
            [
                CapacityAvailabilityEpoch::new(CapacityAvailabilitySource::ActiveSequenceSlots, 1)
                    .unwrap(),
            ];
        let wake = AdmissionWakeSnapshot::new(
            AdmissionWakeEpochs::new(NonZeroU64::new(31).unwrap(), 0, 0, 0),
            &availability,
        );

        let first = scheduler
            .next_batch_with_dynamic_admission(BatchHint::simple(1), wake, &mut |request| {
                AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                    request_id: request.id.clone(),
                })
            })
            .unwrap()
            .unwrap();
        assert_eq!(first.requests[0].tokens_to_process, Some(4));
        assert!(!scheduler
            .mark_prefill_chunk_processed_with_capacity_feedback(&request_id, 8, 4, 2)
            .unwrap());

        let second = scheduler
            .next_batch_with_dynamic_admission(BatchHint::simple(1), wake, &mut |_| {
                panic!("active prefill must not re-enter admission")
            })
            .unwrap()
            .unwrap();
        assert_eq!(second.requests[0].tokens_processed, 2);
        assert_eq!(second.requests[0].tokens_to_process, Some(2));
        assert!(!scheduler
            .mark_prefill_chunk_processed_with_capacity_feedback(&request_id, 8, 2, 2)
            .unwrap());

        let third = scheduler
            .next_batch_with_dynamic_admission(BatchHint::simple(1), wake, &mut |_| {
                panic!("active prefill must not re-enter admission")
            })
            .unwrap()
            .unwrap();
        assert_eq!(third.requests[0].tokens_processed, 4);
        assert_eq!(third.requests[0].tokens_to_process, Some(4));
    }

    #[tokio::test]
    async fn typed_wait_for_release_does_not_make_prefill_first_starve_decode() {
        use ferrum_interfaces::vnext::{
            CapacityAvailabilityEpoch, CapacityAvailabilitySource, CapacityWaitCondition,
            DeferredAction,
        };
        use std::num::NonZeroU64;

        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 2,
            prefill_first_until_active: Some(2),
            ..SchedulerConfig::default()
        });
        let blocked = create_test_request(Priority::Normal);
        let blocked_id = blocked.id.clone();
        let runnable = create_test_request(Priority::Normal);
        let runnable_id = runnable.id.clone();
        scheduler.submit(blocked).await.unwrap();
        scheduler.submit(runnable).await.unwrap();

        let wake = AdmissionWakeEpochs::new(NonZeroU64::new(23).unwrap(), 7, 11, 0);
        let availability =
            [
                CapacityAvailabilityEpoch::new(CapacityAvailabilitySource::ActiveSequenceSlots, 19)
                    .unwrap(),
            ];
        let condition = CapacityWaitCondition::from_observation(23, availability.to_vec()).unwrap();
        let first = scheduler
            .next_batch_with_dynamic_admission(
                BatchHint::simple(2),
                AdmissionWakeSnapshot::new(wake, &availability),
                &mut |request| {
                    if request.id == blocked_id {
                        AdmissionProbeOutcome::Deferred(crate::vnext::AdmissionDeferral::new(
                            DeferredAction::WaitForRelease,
                            wake,
                            condition.clone(),
                        ))
                    } else {
                        AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                            request_id: request.id.clone(),
                        })
                    }
                },
            )
            .unwrap()
            .unwrap();
        assert_eq!(first.requests.len(), 1);
        assert_eq!(first.requests[0].request.id, runnable_id);
        scheduler.mark_prefill_complete(&runnable_id, 1);

        let mut probes = 0;
        let mut observations = Vec::new();
        let unchanged = scheduler
            .next_batch_with_dynamic_admission_observed(
                BatchHint::simple(2),
                AdmissionWakeSnapshot::new(wake, &availability),
                &mut |request| {
                    probes += 1;
                    AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                        request_id: request.id.clone(),
                    })
                },
                &mut |observation| observations.push(observation),
            )
            .unwrap()
            .expect("unchanged capacity wait must not suppress runnable decode work");

        assert_eq!(probes, 0);
        assert_eq!(unchanged.requests.len(), 1);
        assert_eq!(unchanged.requests[0].request.id, runnable_id);
        assert_eq!(unchanged.requests[0].tokens_to_process, Some(1));
        assert!(matches!(
            observations.as_slice(),
            [ExecutorAdmissionQueueObservation::SkippedUnchanged {
                request_id,
                current,
                ..
            }] if request_id == &blocked_id && *current == wake
        ));
    }

    #[tokio::test]
    async fn defer_prefill_to_waiting_frees_active_slot_without_cancelling() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
        let request = create_test_request(Priority::Normal);
        let request_id = request.id.clone();
        scheduler.submit(request).await.unwrap();

        let batch = scheduler.next_batch(BatchHint::simple(4)).await.unwrap();
        assert_eq!(batch.size(), 1);
        let active = scheduler.trace_snapshot();
        assert_eq!(active.waiting_queue_len, 0);
        assert_eq!(active.prefill_queue_len, 1);
        assert_eq!(active.active_len, 1);

        assert!(scheduler.defer_prefill_to_waiting(&request_id));
        let deferred = scheduler.trace_snapshot();
        assert_eq!(deferred.waiting_queue_len, 1);
        assert_eq!(deferred.prefill_queue_len, 0);
        assert_eq!(deferred.active_len, 0);
        assert_eq!(deferred.cancelled_total, 0);
        assert_eq!(
            scheduler.trace_phase(&request_id),
            Some(RequestPhase::Waiting)
        );
        assert_eq!(
            scheduler.request_state(&request_id),
            Some(RequestState::Waiting)
        );
    }

    #[test]
    fn defer_prefill_to_waiting_resets_chunk_progress_after_capacity_loss() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 1,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 1,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        let request = create_test_request_with_prompt_tokens(Priority::Normal, 128);
        let request_id = request.id.clone();
        enqueue_waiting(&scheduler, request);

        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests[0].request.id, request_id);
        assert!(!scheduler.mark_prefill_chunk_processed(&request_id, 128, 64));

        assert!(scheduler.defer_prefill_to_waiting(&request_id));
        let waiting = scheduler.waiting_queue.read();
        let deferred = waiting
            .iter()
            .find(|req| req.inner.request.id == request_id)
            .expect("request should be back in waiting queue");
        assert_eq!(deferred.prefill_tokens, 0);
        assert_eq!(deferred.prefill_chunk_offset, 0);
        assert!(!deferred.chunked_prefill);
        drop(waiting);

        let retry_batch = scheduler.create_iteration_batch(hint).unwrap();
        assert_eq!(retry_batch.requests[0].request.id, request_id);
        assert_eq!(
            retry_batch.requests[0].tokens_to_process,
            Some(128),
            "released physical prefill state must be rebuilt from the start"
        );
    }

    #[test]
    fn capacity_defer_halves_next_waiting_admission_width() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 4,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests.len(), 4);
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request_id in &first_ids {
            assert!(scheduler.defer_prefill_to_waiting(request_id));
        }

        let deferred = scheduler.trace_snapshot();
        assert_eq!(deferred.waiting_queue_len, 4);
        assert_eq!(deferred.active_len, 0);
        assert_eq!(deferred.capacity_deferred_total, 4);
        assert_eq!(deferred.capacity_backpressure_admit_limit, Some(2));

        let second_batch = scheduler.create_iteration_batch(hint).unwrap();
        assert_eq!(
            second_batch.requests.len(),
            2,
            "capacity-deferred waiting requests should not be immediately re-admitted at the failed width"
        );
        let after = scheduler.trace_snapshot();
        assert_eq!(after.waiting_queue_len, 2);
        assert_eq!(after.prefill_queue_len, 2);
        assert_eq!(after.active_len, 2);
        assert_eq!(after.admitted_total, 6);
        assert_eq!(after.capacity_backpressure_admit_limit, Some(2));
    }

    #[test]
    fn capacity_deferred_prefill_retries_once_without_release_then_parks() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 1,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 1,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };
        let request = create_test_request_with_prompt_tokens(Priority::Normal, 128);
        let request_id = request.id.clone();
        enqueue_waiting(&scheduler, request);

        let first = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first.requests[0].request.id, request_id);
        assert!(scheduler.defer_prefill_to_waiting(&request_id));

        let reduced_retry = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(reduced_retry.requests[0].request.id, request_id);
        assert!(scheduler.defer_prefill_to_waiting(&request_id));

        assert!(scheduler.create_iteration_batch(hint.clone()).is_none());
        let parked = scheduler.trace_snapshot();
        assert_eq!(parked.waiting_queue_len, 1);
        assert_eq!(parked.active_len, 0);
        assert_eq!(parked.capacity_blocked_waiting_len, 1);
        assert_eq!(parked.admitted_total, 2);
        assert_eq!(parked.capacity_deferred_total, 2);

        scheduler.record_capacity_release_progress();
        let after_release = scheduler.create_iteration_batch(hint).unwrap();
        assert_eq!(after_release.requests[0].request.id, request_id);
    }

    #[test]
    fn decode_capacity_defer_requeues_for_recompute_without_cancelling() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 4,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests.len(), 4);
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert_eq!(first_ids.len(), 4);
        for request_id in &first_ids {
            scheduler.mark_prefill_complete(request_id, 128);
        }
        assert_eq!(scheduler.trace_snapshot().decode_queue_len, 4);

        for request_id in &first_ids {
            assert!(scheduler.defer_decode_to_waiting_for_capacity(request_id, 4));
        }

        let deferred = scheduler.trace_snapshot();
        assert_eq!(deferred.waiting_queue_len, 4);
        assert_eq!(deferred.decode_queue_len, 0);
        assert_eq!(deferred.active_len, 0);
        assert_eq!(deferred.cancelled_total, 0);
        assert_eq!(deferred.capacity_deferred_total, 4);
        assert_eq!(deferred.capacity_backpressure_admit_limit, Some(2));
        for request_id in &first_ids {
            assert_eq!(
                scheduler.trace_phase(request_id),
                Some(RequestPhase::Waiting)
            );
            assert_eq!(
                scheduler.request_state(request_id),
                Some(RequestState::Waiting)
            );
        }

        let second_batch = scheduler.create_iteration_batch(hint).unwrap();
        assert_eq!(
            second_batch.requests.len(),
            2,
            "capacity-deferred decodes should recompute at a lower admission width"
        );
        let after = scheduler.trace_snapshot();
        assert_eq!(after.waiting_queue_len, 2);
        assert_eq!(after.prefill_queue_len, 2);
        assert_eq!(after.active_len, 2);
        assert_eq!(after.capacity_backpressure_admit_limit, Some(2));
    }

    #[test]
    fn capacity_deferred_decode_recomputes_as_bounded_mixed_prefill_under_decode_pressure() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 4,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request_id in &first_ids {
            scheduler.mark_prefill_complete(request_id, 128);
        }

        assert!(scheduler.defer_decode_to_waiting_for_capacity(&first_ids[0], 4));
        let deferred = scheduler.trace_snapshot();
        assert_eq!(deferred.waiting_queue_len, 1);
        assert_eq!(deferred.decode_queue_len, 3);
        assert_eq!(deferred.active_len, 3);
        assert_eq!(deferred.capacity_blocked_waiting_len, 1);
        assert_eq!(deferred.capacity_backpressure_admit_limit, Some(2));

        let decode_only = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let scheduled_ids: HashSet<RequestId> = decode_only
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        let scheduled_decodes = decode_only
            .requests
            .iter()
            .filter(|request| request.tokens_to_process == Some(1))
            .count();
        assert_eq!(
            scheduled_decodes, 3,
            "decode KV pressure should not globally cap decode-ready survivors while recompute runs"
        );
        assert!(
            !scheduler
                .trace_snapshot()
                .decode_execution_pressure_enforced
        );
        assert_eq!(decode_only.requests.len(), 4);
        assert!(
            scheduled_ids.contains(&first_ids[0]),
            "capacity-deferred recompute should use bounded mixed prefill budget under decode pressure"
        );
        let prefill_tokens = decode_only
            .requests
            .iter()
            .find(|request| request.request.id == first_ids[0])
            .and_then(|request| request.tokens_to_process);
        assert_eq!(
            prefill_tokens,
            Some(64),
            "the recompute prefill should still be capped by the mixed-prefill token budget"
        );
        assert_eq!(scheduler.trace_snapshot().capacity_blocked_waiting_len, 0);
    }

    #[test]
    fn execution_capacity_pressure_caps_decode_survivors_while_recompute_runs() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 4,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request_id in &first_ids {
            scheduler.mark_prefill_complete(request_id, 128);
        }

        assert!(scheduler.defer_decode_to_waiting_for_capacity(&first_ids[0], 4));
        scheduler.record_decode_execution_capacity_pressure(3);

        let bounded = scheduler.create_iteration_batch(hint).unwrap();
        let scheduled_decodes = bounded
            .requests
            .iter()
            .filter(|request| request.tokens_to_process == Some(1))
            .count();
        assert_eq!(
            scheduled_decodes, 2,
            "plan-runtime execution pressure must remain effective while a recompute backlog exists"
        );
        assert!(
            scheduler
                .trace_snapshot()
                .decode_execution_pressure_enforced
        );
        assert!(
            bounded.requests.iter().any(|request| {
                request.request.id == first_ids[0] && request.tokens_to_process == Some(64)
            }),
            "execution pressure must not block the bounded mixed recompute"
        );

        scheduler.record_external_capacity_release();
        let still_bounded = scheduler.trace_snapshot();
        assert_eq!(
            still_bounded.decode_capacity_backpressure_admit_limit,
            Some(2)
        );
        assert!(still_bounded.decode_execution_pressure_enforced);

        assert!(!scheduler.record_decode_execution_capacity_success(1));
        assert_eq!(
            scheduler
                .trace_snapshot()
                .decode_capacity_backpressure_admit_limit,
            Some(2)
        );
        assert!(scheduler.record_decode_execution_capacity_success(2));
        assert_eq!(
            scheduler
                .trace_snapshot()
                .decode_capacity_backpressure_admit_limit,
            Some(3)
        );
        assert!(scheduler.record_decode_execution_capacity_success(3));
        let recovered = scheduler.trace_snapshot();
        assert_eq!(recovered.decode_capacity_backpressure_admit_limit, None);
        assert!(!recovered.decode_execution_pressure_enforced);
    }

    #[test]
    fn execution_capacity_pressure_survives_prefill_progress_and_no_backlog_selection() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 16,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 16,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };
        for _ in 0..8 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }
        let prefill = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let request_ids = prefill
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect::<Vec<_>>();
        assert_eq!(request_ids.len(), 8);

        scheduler.record_decode_execution_capacity_pressure(11);
        for request_id in &request_ids {
            scheduler.mark_prefill_complete(request_id, 128);
        }
        let after_prefill = scheduler.trace_snapshot();
        assert_eq!(
            after_prefill.decode_capacity_backpressure_admit_limit,
            Some(6)
        );
        assert!(after_prefill.decode_execution_pressure_enforced);

        let decode = scheduler.create_iteration_batch(hint).unwrap();
        assert_eq!(decode.requests.len(), 6);
        let after_no_backlog_selection = scheduler.trace_snapshot();
        assert_eq!(
            after_no_backlog_selection.decode_capacity_backpressure_admit_limit,
            Some(6)
        );
        assert!(after_no_backlog_selection.decode_execution_pressure_enforced);
    }

    #[test]
    fn capacity_deferred_decode_recompute_spends_available_mixed_slots() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 8,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 8,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..8 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request_id in &first_ids {
            scheduler.mark_prefill_complete(request_id, 128);
        }

        for request_id in first_ids.iter().take(4) {
            assert!(scheduler.defer_decode_to_waiting_for_capacity(request_id, 8));
        }
        let deferred = scheduler.trace_snapshot();
        assert_eq!(deferred.waiting_queue_len, 4);
        assert_eq!(deferred.decode_queue_len, 4);
        assert_eq!(deferred.active_len, 4);
        assert_eq!(deferred.capacity_blocked_waiting_len, 4);

        let mixed_batch = scheduler.create_iteration_batch(hint).unwrap();
        let scheduled_deferred = mixed_batch
            .requests
            .iter()
            .filter(|request| first_ids[..4].contains(&request.request.id))
            .count();
        let scheduled_decodes = mixed_batch
            .requests
            .iter()
            .filter(|request| {
                first_ids[4..].contains(&request.request.id) && request.tokens_to_process == Some(1)
            })
            .count();
        let prefill_tokens: Vec<_> = mixed_batch
            .requests
            .iter()
            .filter(|request| first_ids[..4].contains(&request.request.id))
            .map(|request| request.tokens_to_process)
            .collect();

        assert_eq!(scheduled_decodes, 4);
        assert_eq!(
            scheduled_deferred, 4,
            "bounded mixed recompute should spend available mixed-prefill slots"
        );
        assert_eq!(prefill_tokens, vec![Some(64), Some(64), Some(64), Some(64)]);
        assert_eq!(scheduler.trace_snapshot().capacity_blocked_waiting_len, 0);
    }

    #[test]
    fn capacity_deferred_recompute_waits_after_no_progress_attempt() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 4,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request_id in &first_ids {
            scheduler.mark_prefill_complete(request_id, 128);
        }

        assert!(scheduler.defer_decode_to_waiting_for_capacity(&first_ids[0], 4));

        let first_mixed = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let first_scheduled_ids: HashSet<RequestId> = first_mixed
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert!(
            first_scheduled_ids.contains(&first_ids[0]),
            "the first mixed iteration may spend its bounded recompute slot"
        );
        assert_eq!(scheduler.trace_snapshot().capacity_blocked_waiting_len, 0);
        assert_eq!(scheduler.trace_snapshot().prefill_queue_len, 1);

        let no_progress_retry = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let retry_ids: HashSet<RequestId> = no_progress_retry
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert!(
            !retry_ids.contains(&first_ids[0]),
            "a release-blocked recompute must not be retried in the same release epoch without progress"
        );
        assert_eq!(
            no_progress_retry.requests.len(),
            3,
            "decode-ready survivors should continue at full width while the failed recompute waits"
        );
        assert_eq!(scheduler.trace_snapshot().prefill_queue_len, 1);

        assert!(!scheduler.mark_prefill_chunk_processed(&first_ids[0], 128, 64));
        let after_progress = scheduler.create_iteration_batch(hint).unwrap();
        let progressed_tokens = after_progress
            .requests
            .iter()
            .find(|request| request.request.id == first_ids[0])
            .and_then(|request| request.tokens_to_process);
        assert_eq!(
            progressed_tokens,
            Some(64),
            "recorded prefill progress should make the next recompute chunk eligible again"
        );
    }

    #[test]
    fn capacity_deferred_recompute_skips_marked_requests_without_blocking_later_candidates() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 10,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let init_hint = BatchHint {
            max_batch_size: 10,
            max_tokens: 2048,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };
        let mixed_hint = BatchHint {
            max_batch_size: 8,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..10 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(init_hint.clone()).unwrap();
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request_id in &first_ids {
            scheduler.mark_prefill_complete(request_id, 128);
        }

        for request_id in first_ids.iter().take(4) {
            assert!(scheduler.defer_decode_to_waiting_for_capacity(request_id, 10));
        }

        let first_mixed = scheduler
            .create_iteration_batch(mixed_hint.clone())
            .unwrap();
        let first_mixed_ids: HashSet<RequestId> = first_mixed
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert!(first_mixed_ids.contains(&first_ids[0]));
        assert!(first_mixed_ids.contains(&first_ids[1]));
        assert!(!first_mixed_ids.contains(&first_ids[2]));
        assert!(!first_mixed_ids.contains(&first_ids[3]));
        assert!(scheduler.defer_prefill_to_waiting(&first_ids[0]));
        assert!(scheduler.defer_prefill_to_waiting(&first_ids[1]));

        let second_mixed = scheduler.create_iteration_batch(mixed_hint).unwrap();
        let second_mixed_ids: HashSet<RequestId> = second_mixed
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert!(
            !second_mixed_ids.contains(&first_ids[0]),
            "the first failed recompute must still be skipped in the same release epoch"
        );
        assert!(
            !second_mixed_ids.contains(&first_ids[1]),
            "already failed blocked recomputes must not consume the later candidate's slot"
        );
        assert!(
            second_mixed_ids.contains(&first_ids[2]),
            "marked queue-head requests should not block a later untried recompute candidate"
        );
        assert_eq!(
            second_mixed.requests.len(),
            7,
            "six decode-ready survivors plus one later recompute should be scheduled after same-epoch failures"
        );
    }

    #[tokio::test]
    async fn capacity_deferred_mixed_recompute_waits_after_capacity_feedback_until_release() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 8,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 8,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..8 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request_id in &first_ids {
            scheduler.mark_prefill_complete(request_id, 128);
        }

        assert!(scheduler.defer_decode_to_waiting_for_capacity(&first_ids[0], 4));
        assert!(scheduler.defer_decode_to_waiting_for_capacity(&first_ids[1], 4));

        let first_mixed = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let first_mixed_ids: HashSet<RequestId> = first_mixed
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert!(first_mixed_ids.contains(&first_ids[0]));

        scheduler.defer_capacity_deferred_mixed_recompute_until_release();
        assert!(scheduler.defer_prefill_to_waiting(&first_ids[0]));

        let blocked_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let blocked_ids: HashSet<RequestId> = blocked_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert!(
            !blocked_ids.contains(&first_ids[0]) && !blocked_ids.contains(&first_ids[1]),
            "mixed recompute should wait after capacity feedback until a real release"
        );
        assert_eq!(
            blocked_batch.requests.len(),
            6,
            "decode-ready survivors should continue while blocked recomputes wait for capacity release"
        );

        let response = InferenceResponse {
            request_id: first_ids[2].clone(),
            text: String::new(),
            tokens: Vec::new(),
            finish_reason: ferrum_types::FinishReason::Length,
            usage: ferrum_types::TokenUsage::new(0, 0),
            latency_ms: 0,
            created_at: chrono::Utc::now(),
            metadata: Default::default(),
            api_response: None,
            execution_evidence: None,
        };
        scheduler
            .complete(first_ids[2].clone(), &response)
            .await
            .unwrap();

        let after_release = scheduler.create_iteration_batch(hint).unwrap();
        let after_release_ids: HashSet<RequestId> = after_release
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert!(
            after_release_ids.contains(&first_ids[0]) || after_release_ids.contains(&first_ids[1]),
            "capacity release should reopen bounded mixed recompute scanning"
        );
    }

    #[tokio::test]
    async fn capacity_deferred_mixed_recompute_resumes_after_decode_capacity_release() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 8,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 8,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..8 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request_id in &first_ids {
            scheduler.mark_prefill_complete(request_id, 128);
        }

        assert!(scheduler.defer_decode_to_waiting_for_capacity(&first_ids[0], 4));
        assert!(scheduler.defer_decode_to_waiting_for_capacity(&first_ids[1], 4));

        let first_mixed = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let first_mixed_ids: HashSet<RequestId> = first_mixed
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert!(first_mixed_ids.contains(&first_ids[0]) || first_mixed_ids.contains(&first_ids[1]));

        scheduler.defer_capacity_deferred_mixed_recompute_until_release();
        assert!(scheduler.defer_prefill_to_waiting(&first_ids[0]));

        let blocked_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let blocked_ids: HashSet<RequestId> = blocked_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert!(
            !blocked_ids.contains(&first_ids[0]) && !blocked_ids.contains(&first_ids[1]),
            "mixed recompute should stay blocked until fresh capacity evidence"
        );

        assert!(scheduler.defer_decode_to_waiting_for_capacity(&first_ids[2], 4));
        scheduler.record_capacity_deferred_mixed_recompute_release_evidence();
        let after_decode_defer = scheduler.trace_snapshot();
        assert_eq!(after_decode_defer.capacity_blocked_waiting_len, 2);

        let after_physical_release = scheduler.create_iteration_batch(hint).unwrap();
        let recompute_ids: Vec<_> = after_physical_release
            .requests
            .iter()
            .filter(|request| {
                first_ids[..3].contains(&request.request.id) && request.tokens_to_process != Some(1)
            })
            .map(|request| request.request.id.clone())
            .collect();
        assert_eq!(
            recompute_ids.len(),
            2,
            "physical KV release should reopen older recomputes that fit active capacity"
        );
        assert!(recompute_ids.contains(&first_ids[0]));
        assert!(recompute_ids.contains(&first_ids[1]));
        assert!(
            !recompute_ids.contains(&first_ids[2]),
            "the just-deferred decode should wait behind older blocked recomputes when active capacity is limited"
        );
    }

    #[tokio::test]
    async fn capacity_deferred_mixed_recompute_waits_until_kv_snapshot_has_required_free_blocks() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 8,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 8,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..8 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request_id in &first_ids {
            scheduler.mark_prefill_complete(request_id, 128);
        }

        assert!(scheduler.defer_decode_to_waiting_for_capacity(&first_ids[0], 4));
        assert!(scheduler.defer_decode_to_waiting_for_capacity(&first_ids[1], 4));

        let first_mixed = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let first_mixed_ids: HashSet<RequestId> = first_mixed
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert!(first_mixed_ids.contains(&first_ids[0]) || first_mixed_ids.contains(&first_ids[1]));

        scheduler.defer_capacity_deferred_mixed_recompute_until_kv_capacity(
            Some(4),
            Some(0),
            Some(1),
        );
        let blocked_snapshot = scheduler.trace_snapshot();
        assert_eq!(
            blocked_snapshot.capacity_mixed_recompute_required_blocks_per_slot,
            Some(4)
        );
        assert_eq!(
            blocked_snapshot.capacity_mixed_recompute_observed_free_blocks,
            Some(0)
        );
        assert_eq!(
            blocked_snapshot.capacity_mixed_recompute_blocked_until_epoch,
            blocked_snapshot.capacity_mixed_recompute_epoch + 1
        );
        assert!(scheduler.defer_prefill_to_waiting(&first_ids[0]));

        assert!(scheduler.defer_decode_to_waiting_for_capacity(&first_ids[2], 4));
        scheduler.record_capacity_deferred_mixed_recompute_kv_capacity_snapshot(3);

        let still_blocked = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let still_blocked_ids: HashSet<RequestId> = still_blocked
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert!(
            !still_blocked_ids.contains(&first_ids[0])
                && !still_blocked_ids.contains(&first_ids[1]),
            "insufficient paged-KV free blocks must not reopen failed mixed recompute"
        );

        scheduler.record_capacity_deferred_mixed_recompute_kv_capacity_snapshot(4);
        let exact_without_headroom = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let exact_without_headroom_ids: HashSet<RequestId> = exact_without_headroom
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert!(
            !exact_without_headroom_ids.contains(&first_ids[0])
                && !exact_without_headroom_ids.contains(&first_ids[1]),
            "a KV snapshot must leave allocator headroom before reopening mixed recompute"
        );

        scheduler.record_capacity_deferred_mixed_recompute_kv_capacity_snapshot(5);
        let reopened_snapshot = scheduler.trace_snapshot();
        assert_eq!(
            reopened_snapshot.capacity_mixed_recompute_observed_free_blocks,
            Some(5)
        );
        assert!(
            reopened_snapshot.capacity_mixed_recompute_epoch
                >= reopened_snapshot.capacity_mixed_recompute_blocked_until_epoch
        );
        let after_enough_free = scheduler.create_iteration_batch(hint).unwrap();
        let after_enough_ids: HashSet<RequestId> = after_enough_free
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert!(
            after_enough_ids.contains(&first_ids[0]) || after_enough_ids.contains(&first_ids[1]),
            "mixed recompute should reopen once the model-owned KV snapshot reaches the failed admission need"
        );
    }

    #[tokio::test]
    async fn capacity_deferred_mixed_recompute_reopens_from_capacity_feedback_when_fit() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 8,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 8,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..8 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request_id in &first_ids {
            scheduler.mark_prefill_complete(request_id, 128);
        }

        assert!(scheduler.defer_decode_to_waiting_for_capacity(&first_ids[0], 4));

        let first_mixed = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let first_mixed_ids: HashSet<RequestId> = first_mixed
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert!(first_mixed_ids.contains(&first_ids[0]));

        scheduler.defer_capacity_deferred_mixed_recompute_until_kv_capacity(
            Some(4),
            Some(0),
            Some(1),
        );
        assert!(scheduler.defer_prefill_to_waiting(&first_ids[0]));

        scheduler.defer_capacity_deferred_mixed_recompute_until_kv_capacity(
            Some(4),
            Some(5),
            Some(1),
        );

        let reopened = scheduler.create_iteration_batch(hint).unwrap();
        let reopened_ids: HashSet<RequestId> = reopened
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert!(
            reopened_ids.contains(&first_ids[0]),
            "structured capacity feedback with enough free blocks should reopen a narrower recompute without waiting for a separate snapshot call"
        );
    }

    #[tokio::test]
    async fn capacity_deferred_mixed_recompute_paces_width_by_kv_snapshot() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 8,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 8,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..8 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request_id in &first_ids {
            scheduler.mark_prefill_complete(request_id, 128);
        }

        for request_id in first_ids.iter().take(4) {
            assert!(scheduler.defer_decode_to_waiting_for_capacity(request_id, 8));
        }

        scheduler.defer_capacity_deferred_mixed_recompute_until_kv_capacity(
            Some(16),
            Some(0),
            Some(4),
        );
        scheduler.record_capacity_deferred_mixed_recompute_kv_capacity_snapshot(9);

        let paced_mixed = scheduler.create_iteration_batch(hint).unwrap();
        let scheduled_recomputes = paced_mixed
            .requests
            .iter()
            .filter(|request| {
                first_ids[..4].contains(&request.request.id) && request.tokens_to_process != Some(1)
            })
            .count();
        let scheduled_decodes = paced_mixed
            .requests
            .iter()
            .filter(|request| {
                first_ids[4..].contains(&request.request.id) && request.tokens_to_process == Some(1)
            })
            .count();

        assert_eq!(scheduled_decodes, 4);
        assert_eq!(
            scheduled_recomputes, 2,
            "free KV blocks should pace the number of reopened capacity-blocked recomputes"
        );
        assert_eq!(scheduler.trace_snapshot().capacity_blocked_waiting_len, 2);
    }

    #[tokio::test]
    async fn release_ready_capacity_deferred_recompute_still_uses_kv_budget_under_decode_pressure()
    {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 8,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let init_hint = BatchHint {
            max_batch_size: 8,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..8 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(init_hint.clone()).unwrap();
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request_id in &first_ids {
            scheduler.mark_prefill_complete(request_id, 128);
        }

        for request_id in first_ids.iter().take(4) {
            assert!(scheduler.defer_decode_to_waiting_for_capacity(request_id, 8));
        }

        scheduler.record_capacity_release_progress();

        scheduler.defer_capacity_deferred_mixed_recompute_until_kv_capacity(
            Some(16),
            Some(0),
            Some(4),
        );
        scheduler.record_capacity_deferred_mixed_recompute_kv_capacity_snapshot(9);

        let paced_mixed = scheduler.create_iteration_batch(init_hint).unwrap();
        let scheduled_recomputes = paced_mixed
            .requests
            .iter()
            .filter(|request| {
                first_ids[..4].contains(&request.request.id) && request.tokens_to_process != Some(1)
            })
            .count();
        let scheduled_decodes = paced_mixed
            .requests
            .iter()
            .filter(|request| {
                first_ids[4..].contains(&request.request.id) && request.tokens_to_process == Some(1)
            })
            .count();

        assert_eq!(scheduled_decodes, 4);
        assert_eq!(
            scheduled_recomputes, 2,
            "release-ready recomputes under decode pressure must still obey the KV snapshot budget"
        );
        assert_eq!(scheduler.trace_snapshot().capacity_blocked_waiting_len, 0);
        assert_eq!(
            scheduler.trace_snapshot().waiting_queue_len,
            2,
            "the remaining capacity-deferred recomputes should stay waiting for a later KV window"
        );
    }

    #[tokio::test]
    async fn capacity_deferred_decode_waits_for_release_without_bounded_mixed_budget() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 4,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..2 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request_id in &first_ids {
            scheduler.mark_prefill_complete(request_id, 128);
        }

        assert!(scheduler.defer_decode_to_waiting_for_capacity(&first_ids[0], 4));
        let deferred = scheduler.trace_snapshot();
        assert_eq!(deferred.waiting_queue_len, 1);
        assert_eq!(deferred.decode_queue_len, 1);
        assert_eq!(deferred.active_len, 1);
        assert_eq!(deferred.capacity_blocked_waiting_len, 1);

        let response = InferenceResponse {
            request_id: first_ids[1].clone(),
            text: String::new(),
            tokens: Vec::new(),
            finish_reason: ferrum_types::FinishReason::Length,
            usage: ferrum_types::TokenUsage::new(0, 0),
            latency_ms: 0,
            created_at: chrono::Utc::now(),
            metadata: Default::default(),
            api_response: None,
            execution_evidence: None,
        };
        scheduler
            .complete(first_ids[1].clone(), &response)
            .await
            .unwrap();

        let after_release = scheduler.create_iteration_batch(hint).unwrap();
        let scheduled_ids: HashSet<RequestId> = after_release
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert!(
            scheduled_ids.contains(&first_ids[0]),
            "a real capacity release should make the deferred recompute eligible again"
        );
        assert_eq!(scheduler.trace_snapshot().capacity_blocked_waiting_len, 0);
    }

    #[tokio::test]
    async fn capacity_backpressure_survives_partial_prefill_and_waits_for_release() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 4,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let first_ids: Vec<RequestId> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request in &first_batch.requests {
            assert!(scheduler.defer_prefill_to_waiting(&request.request.id));
        }
        assert_eq!(
            scheduler.trace_snapshot().capacity_backpressure_admit_limit,
            Some(2)
        );

        let second_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(second_batch.requests.len(), 2);
        let second_ids: HashSet<RequestId> = second_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        let progressed_id = second_batch.requests[0].request.id.clone();
        assert!(!scheduler.mark_prefill_chunk_processed(&progressed_id, 128, 1));
        assert_eq!(
            scheduler.trace_snapshot().capacity_backpressure_admit_limit,
            Some(2),
            "partial prefill progress still consumes capacity and should not reopen the failed width"
        );
        assert!(scheduler.mark_prefill_chunk_processed(&progressed_id, 128, 127));
        assert_eq!(
            scheduler.trace_snapshot().capacity_backpressure_admit_limit,
            None,
            "full prefill completion should relax the capacity backpressure window"
        );

        let third_batch = scheduler.create_iteration_batch(hint).unwrap();
        assert_eq!(
            third_batch.requests.len(),
            2,
            "prefill completion may continue existing active work but must not release blocked waiting prefills"
        );
        let after = scheduler.trace_snapshot();
        assert_eq!(after.waiting_queue_len, 2);
        assert_eq!(after.active_len, 2);

        let response = InferenceResponse {
            request_id: progressed_id.clone(),
            text: String::new(),
            tokens: Vec::new(),
            finish_reason: ferrum_types::FinishReason::Length,
            usage: ferrum_types::TokenUsage::new(0, 0),
            latency_ms: 0,
            created_at: chrono::Utc::now(),
            metadata: Default::default(),
            api_response: None,
            execution_evidence: None,
        };
        scheduler.complete(progressed_id, &response).await.unwrap();

        let after_release_batch = scheduler
            .create_iteration_batch(BatchHint {
                max_batch_size: 4,
                max_tokens: 1024,
                target_latency_ms: None,
                available_memory: None,
                resource_constraints: Default::default(),
            })
            .unwrap();
        let after_release_ids: HashSet<RequestId> = after_release_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert!(
            first_ids
                .iter()
                .any(|id| !second_ids.contains(id) && after_release_ids.contains(id)),
            "actual request completion should release capacity and reopen blocked waiting prefills"
        );
    }

    #[tokio::test]
    async fn capacity_backpressure_survives_cancel_without_token_progress() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 4,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint).unwrap();
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request_id in &first_ids {
            assert!(scheduler.defer_prefill_to_waiting(request_id));
        }
        assert_eq!(
            scheduler.trace_snapshot().capacity_backpressure_admit_limit,
            Some(2)
        );

        assert!(scheduler.cancel(first_ids[0].clone()).await.unwrap());
        let after_cancel = scheduler.trace_snapshot();
        assert_eq!(after_cancel.cancelled_total, 1);
        assert_eq!(
            after_cancel.capacity_backpressure_admit_limit,
            Some(2),
            "cancellation frees a slot but is not evidence that the failed admission width now fits"
        );
    }

    #[tokio::test]
    async fn test_batch_creation() {
        let config = SchedulerConfig::default();
        let scheduler = ContinuousBatchScheduler::new(config);

        // Submit some requests
        for _ in 0..5 {
            scheduler
                .submit(create_test_request(Priority::Normal))
                .await
                .unwrap();
        }

        // Get batch
        let batch = scheduler.next_batch(BatchHint::simple(10)).await;
        assert!(batch.is_some());

        // Requests should have been promoted
        assert!(scheduler.prefilling_count() > 0 || scheduler.decoding_count() > 0);
    }

    #[test]
    fn prompt_token_metadata_expands_prefill_admission() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });

        for _ in 0..16 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 256),
            );
        }

        let hint = BatchHint {
            max_batch_size: 32,
            max_tokens: 2048,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };
        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests.len(), 8);

        for request in first_batch.requests {
            scheduler.mark_prefill_complete(&request.request.id, 256);
        }

        let mixed_batch = scheduler.create_iteration_batch(hint).unwrap();
        assert_eq!(mixed_batch.requests.len(), 16);
        assert_eq!(mixed_batch.resource_requirements.gpu_memory, 2048 * 16);
    }

    #[test]
    fn prompt_token_metadata_can_be_disabled_for_prefill_admission() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prompt_token_estimate: false,
            ..SchedulerConfig::default()
        });

        for _ in 0..16 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 256),
            );
        }

        let batch = scheduler
            .create_iteration_batch(BatchHint {
                max_batch_size: 32,
                max_tokens: 2048,
                target_latency_ms: None,
                available_memory: None,
                resource_constraints: Default::default(),
            })
            .unwrap();
        assert_eq!(batch.requests.len(), 4);
    }

    #[test]
    fn scheduler_runtime_config_is_captured_at_construction() {
        let mut config = SchedulerConfig {
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        };
        let scheduler = ContinuousBatchScheduler::new(config.clone());
        config.prompt_token_estimate = false;

        for _ in 0..16 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 256),
            );
        }

        let batch = scheduler
            .create_iteration_batch(BatchHint {
                max_batch_size: 32,
                max_tokens: 2048,
                target_latency_ms: None,
                available_memory: None,
                resource_constraints: Default::default(),
            })
            .unwrap();
        assert_eq!(batch.requests.len(), 8);
    }

    #[test]
    fn max_running_requests_limits_waiting_admission() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 1,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });

        for _ in 0..3 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let hint = BatchHint {
            max_batch_size: 8,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };
        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests.len(), 1);
        assert_eq!(scheduler.prefilling_count(), 1);
        assert_eq!(scheduler.waiting_count(), 2);

        let active_batch = scheduler.create_iteration_batch(hint).unwrap();
        assert_eq!(active_batch.requests.len(), 1);
        assert_eq!(
            active_batch.requests[0].request.id, first_batch.requests[0].request.id,
            "scheduler must not admit another waiting request while the active cap is full"
        );
        assert_eq!(scheduler.prefilling_count(), 1);
        assert_eq!(scheduler.waiting_count(), 2);
    }

    #[test]
    fn newly_admitted_prefill_uses_remaining_budget_with_decode() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 4,
            max_tokens: 4,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        let first = create_test_request_with_prompt_tokens(Priority::Normal, 2);
        let first_id = first.id.clone();
        enqueue_waiting(&scheduler, first);
        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests.len(), 1);
        scheduler.mark_prefill_complete(&first_id, 2);

        let second = create_test_request_with_prompt_tokens(Priority::Normal, 2);
        let second_id = second.id.clone();
        enqueue_waiting(&scheduler, second);

        let mixed_batch = scheduler.create_iteration_batch(hint).unwrap();
        let ids: HashSet<RequestId> = mixed_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert_eq!(mixed_batch.requests.len(), 2);
        assert!(
            ids.contains(&first_id),
            "decode request should remain scheduled"
        );
        assert!(
            ids.contains(&second_id),
            "newly admitted prefill should use remaining same-iteration budget"
        );
        assert_eq!(mixed_batch.resource_requirements.gpu_memory, 3 * 16);
    }

    #[test]
    fn default_scheduler_caps_mixed_prefill_only_under_decode_pressure() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 8,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 8,
            max_tokens: 2048,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        let first = create_test_request_with_prompt_tokens(Priority::Normal, 256);
        let first_id = first.id.clone();
        enqueue_waiting(&scheduler, first);
        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests.len(), 1);
        scheduler.mark_prefill_complete(&first_id, 256);

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 256),
            );
        }

        let low_decode_pressure = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(
            low_decode_pressure.requests.len(),
            5,
            "small decode cohorts should use remaining token budget to build concurrency"
        );
        assert_eq!(
            low_decode_pressure.resource_requirements.gpu_memory,
            (1 + 4 * 256) * 16
        );
        assert_eq!(
            low_decode_pressure
                .requests
                .iter()
                .filter(|request| request.request.id != first_id)
                .map(|request| request.tokens_to_process)
                .collect::<Vec<_>>(),
            vec![Some(256), Some(256), Some(256), Some(256)]
        );

        for request in low_decode_pressure
            .requests
            .iter()
            .filter(|request| request.request.id != first_id)
        {
            scheduler.mark_prefill_complete(&request.request.id, 256);
        }
        assert_eq!(scheduler.decoding_count(), 5);

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 256),
            );
        }

        let high_decode_pressure = scheduler.create_iteration_batch(hint).unwrap();
        let prefill_tokens: Vec<_> = high_decode_pressure
            .requests
            .iter()
            .filter(|request| request.tokens_to_process != Some(1))
            .map(|request| request.tokens_to_process)
            .collect();
        assert_eq!(
            high_decode_pressure.requests.len(),
            8,
            "high decode pressure should admit bounded partial prefills up to available slots"
        );
        assert_eq!(prefill_tokens, vec![Some(64), Some(64), Some(64)]);
        assert_eq!(
            high_decode_pressure.resource_requirements.gpu_memory,
            (5 + 192) * 16
        );
    }

    #[test]
    fn max_batched_tokens_limits_prefill_admission_by_prompt_tokens() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 256),
            );
        }

        let batch = scheduler
            .create_iteration_batch(BatchHint {
                max_batch_size: 8,
                max_tokens: 512,
                target_latency_ms: None,
                available_memory: None,
                resource_constraints: Default::default(),
            })
            .unwrap();

        assert_eq!(batch.requests.len(), 2);
        assert_eq!(batch.resource_requirements.gpu_memory, 512 * 16);
        assert_eq!(
            scheduler.prefilling_count(),
            4,
            "max_tokens limits the emitted iteration batch, not waiting-to-prefill promotion"
        );
        assert_eq!(scheduler.waiting_count(), 0);
    }

    #[test]
    fn long_prefill_uses_remaining_step_budget_instead_of_fixed_chunk() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });

        for _ in 0..2 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 1536),
            );
        }

        let batch = scheduler
            .create_iteration_batch(BatchHint {
                max_batch_size: 8,
                max_tokens: 2048,
                target_latency_ms: None,
                available_memory: None,
                resource_constraints: Default::default(),
            })
            .unwrap();

        assert_eq!(batch.requests.len(), 2);
        assert_eq!(
            batch
                .requests
                .iter()
                .map(|request| request.tokens_to_process)
                .collect::<Vec<_>>(),
            vec![Some(1536), Some(512)]
        );
        assert_eq!(batch.resource_requirements.gpu_memory, 2048 * 16);
    }

    #[test]
    fn prefill_first_until_active_skips_early_decodes() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prefill_first_until_active: Some(4),
            ..SchedulerConfig::default()
        });

        for _ in 0..3 {
            enqueue_waiting(&scheduler, create_test_request(Priority::Normal));
        }

        let hint = BatchHint {
            max_batch_size: 8,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };
        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests.len(), 2);
        let first_ids: Vec<RequestId> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for id in &first_ids {
            scheduler.mark_prefill_complete(id, 512);
        }
        assert_eq!(scheduler.decoding_count(), 2);

        let second_batch = scheduler.create_iteration_batch(hint).unwrap();
        assert_eq!(second_batch.requests.len(), 1);
        assert!(
            second_batch
                .requests
                .iter()
                .all(|request| !first_ids.contains(&request.request.id)),
            "fill-first should schedule more prefills before decoding early requests"
        );
    }

    #[test]
    fn prefill_first_until_active_resumes_decodes_at_active_target() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prefill_first_until_active: Some(4),
            ..SchedulerConfig::default()
        });

        for _ in 0..4 {
            enqueue_waiting(&scheduler, create_test_request(Priority::Normal));
        }

        let hint = BatchHint {
            max_batch_size: 8,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };
        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests.len(), 2);
        let first_ids: Vec<RequestId> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for id in &first_ids {
            scheduler.mark_prefill_complete(id, 512);
        }

        assert_eq!(scheduler.decoding_count(), 2);
        assert_eq!(scheduler.prefilling_count(), 2);
        assert_eq!(scheduler.active_count(), 4);

        let second_batch = scheduler.create_iteration_batch(hint).unwrap();
        let scheduled_decodes = second_batch
            .requests
            .iter()
            .filter(|request| {
                first_ids.contains(&request.request.id) && request.tokens_to_process == Some(1)
            })
            .count();
        assert_eq!(
            scheduled_decodes, 2,
            "fill-first must not starve decode once the active target is reached"
        );
    }

    #[test]
    fn capacity_backpressure_disables_prefill_first_decode_skip() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            prompt_token_estimate: true,
            prefill_first_until_active: Some(4),
            ..SchedulerConfig::default()
        });

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let hint = BatchHint {
            max_batch_size: 4,
            max_tokens: 512,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };
        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests.len(), 4);
        let first_ids: Vec<RequestId> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for id in first_ids.iter().take(3) {
            scheduler.mark_prefill_complete(id, 128);
        }
        assert!(scheduler.defer_prefill_to_waiting(&first_ids[3]));

        let after_defer = scheduler.trace_snapshot();
        assert_eq!(after_defer.decode_queue_len, 3);
        assert_eq!(after_defer.waiting_queue_len, 1);
        assert_eq!(after_defer.active_len, 3);
        assert_eq!(after_defer.capacity_backpressure_admit_limit, Some(1));

        let second_batch = scheduler.create_iteration_batch(hint).unwrap();
        let scheduled_decodes = second_batch
            .requests
            .iter()
            .filter(|request| {
                first_ids[..3].contains(&request.request.id) && request.tokens_to_process == Some(1)
            })
            .count();
        assert_eq!(
            scheduled_decodes, 3,
            "capacity backpressure must let decode-ready requests run instead of repeatedly admitting a capacity-blocked prefill"
        );
        assert_eq!(
            second_batch.requests.len(),
            3,
            "a capacity-blocked prefill must wait for capacity evidence instead of refilling an empty batch slot"
        );
        assert_eq!(scheduler.trace_snapshot().capacity_blocked_waiting_len, 1);
    }

    #[tokio::test]
    async fn prefill_capacity_defer_waits_for_release_while_decode_active() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            prompt_token_estimate: true,
            prefill_first_until_active: Some(4),
            ..SchedulerConfig::default()
        });

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let hint = BatchHint {
            max_batch_size: 4,
            max_tokens: 512,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };
        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests.len(), 4);
        let first_ids: Vec<RequestId> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for id in first_ids.iter().take(3) {
            scheduler.mark_prefill_complete(id, 128);
        }
        assert!(scheduler.defer_prefill_to_waiting(&first_ids[3]));

        let deferred = scheduler.trace_snapshot();
        assert_eq!(deferred.decode_queue_len, 3);
        assert_eq!(deferred.waiting_queue_len, 1);
        assert_eq!(deferred.capacity_blocked_waiting_len, 1);

        let blocked_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let blocked_ids: HashSet<RequestId> = blocked_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert!(
            !blocked_ids.contains(&first_ids[3]),
            "failed prefill must not be retried while only decode progress has happened"
        );
        assert_eq!(blocked_batch.requests.len(), 3);

        let response = InferenceResponse {
            request_id: first_ids[0].clone(),
            text: String::new(),
            tokens: Vec::new(),
            finish_reason: ferrum_types::FinishReason::Length,
            usage: ferrum_types::TokenUsage::new(0, 0),
            latency_ms: 0,
            created_at: chrono::Utc::now(),
            metadata: Default::default(),
            api_response: None,
            execution_evidence: None,
        };
        scheduler
            .complete(first_ids[0].clone(), &response)
            .await
            .unwrap();

        let after_release = scheduler.create_iteration_batch(hint).unwrap();
        let after_release_ids: HashSet<RequestId> = after_release
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert!(
            after_release_ids.contains(&first_ids[3]),
            "real capacity release should make the blocked prefill eligible again"
        );
    }

    #[test]
    fn decode_progress_does_not_relax_capacity_backpressure() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            prompt_token_estimate: true,
            prefill_first_until_active: Some(4),
            ..SchedulerConfig::default()
        });

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let hint = BatchHint {
            max_batch_size: 4,
            max_tokens: 512,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };
        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests.len(), 4);
        let first_ids: Vec<RequestId> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for id in first_ids.iter().take(3) {
            scheduler.mark_prefill_complete(id, 128);
        }
        assert!(scheduler.defer_prefill_to_waiting(&first_ids[3]));
        assert_eq!(
            scheduler.trace_snapshot().capacity_backpressure_admit_limit,
            Some(1)
        );

        for id in first_ids.iter().take(3) {
            scheduler.update_decode_progress(id, 1);
        }
        assert_eq!(
            scheduler.trace_snapshot().capacity_backpressure_admit_limit,
            Some(1),
            "decode progress consumes KV capacity and must not reopen waiting admission"
        );

        let second_batch = scheduler.create_iteration_batch(hint).unwrap();
        let scheduled_decodes = second_batch
            .requests
            .iter()
            .filter(|request| {
                first_ids[..3].contains(&request.request.id) && request.tokens_to_process == Some(1)
            })
            .count();
        assert_eq!(
            scheduled_decodes, 3,
            "capacity backpressure should keep fill-first from skipping decode after decode progress"
        );
        assert_eq!(
            second_batch.requests.len(),
            3,
            "decode progress alone must not make a capacity-limited prefill refill the remaining batch slot"
        );
    }

    #[test]
    fn capacity_backpressure_keeps_decode_survivors_wide_after_decode_defer() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 8,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 8,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..8 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request_id in &first_ids {
            scheduler.mark_prefill_complete(request_id, 128);
        }

        assert!(scheduler.defer_decode_to_waiting_for_capacity(&first_ids[0], 8));
        assert_eq!(
            scheduler.trace_snapshot().capacity_backpressure_admit_limit,
            Some(4)
        );

        let capped = scheduler.create_iteration_batch(hint).unwrap();
        let scheduled_decodes = capped
            .requests
            .iter()
            .filter(|request| request.tokens_to_process == Some(1))
            .count();

        assert_eq!(
            scheduled_decodes, 7,
            "decode scheduling should keep decode-ready survivors wide after a decode KV failure"
        );
    }

    #[test]
    fn decode_capacity_backpressure_uses_structured_free_blocks_when_nearly_fit() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 16,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 16,
            max_tokens: 2048,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..16 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request_id in &first_ids {
            scheduler.mark_prefill_complete(request_id, 128);
        }

        scheduler.record_decode_capacity_pressure(16, Some(12));
        let capped = scheduler.create_iteration_batch(hint).unwrap();
        let scheduled_decodes = capped
            .requests
            .iter()
            .filter(|request| request.tokens_to_process == Some(1))
            .count();

        assert_eq!(
            scheduled_decodes, 11,
            "near-fit KV pressure should cap to usable free blocks instead of blindly halving"
        );
    }

    #[test]
    fn prefill_step_chunk_caps_prefill_first_batches() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prompt_token_estimate: true,
            prefill_first_until_active: Some(4),
            prefill_step_chunk: Some(128),
            ..SchedulerConfig::default()
        });

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 512),
            );
        }

        let batch = scheduler
            .create_iteration_batch(BatchHint {
                max_batch_size: 8,
                max_tokens: 2048,
                target_latency_ms: None,
                available_memory: None,
                resource_constraints: Default::default(),
            })
            .unwrap();

        assert_eq!(batch.requests.len(), 4);
        assert_eq!(
            batch
                .requests
                .iter()
                .map(|request| request.tokens_to_process)
                .collect::<Vec<_>>(),
            vec![Some(128), Some(128), Some(128), Some(128)]
        );
        assert_eq!(batch.resource_requirements.gpu_memory, (4 * 128) * 16);
    }

    #[test]
    fn elastic_prefill_budget_uses_live_capacity_not_configured_concurrency() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 16,
            prompt_token_estimate: true,
            prefill_first_until_active: Some(16),
            prefill_step_chunk: None,
            ..SchedulerConfig::default()
        });

        enqueue_waiting(
            &scheduler,
            create_test_request_with_prompt_tokens(Priority::Normal, 64),
        );

        let batch = scheduler
            .create_iteration_batch(BatchHint {
                max_batch_size: 16,
                max_tokens: 192,
                target_latency_ms: None,
                available_memory: None,
                resource_constraints: Default::default(),
            })
            .unwrap();

        assert_eq!(batch.requests.len(), 1);
        assert_eq!(batch.requests[0].tokens_to_process, Some(64));
        assert_eq!(batch.resource_requirements.gpu_memory, 64 * 16);
    }

    #[test]
    fn active_decode_prefill_chunk_only_caps_when_decode_is_active() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prompt_token_estimate: true,
            active_decode_prefill_chunk: Some(64),
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 2,
            max_tokens: 512,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        let first = create_test_request_with_prompt_tokens(Priority::Normal, 256);
        let first_id = first.id.clone();
        enqueue_waiting(&scheduler, first);
        let initial_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(initial_batch.requests.len(), 1);
        assert_eq!(initial_batch.resource_requirements.gpu_memory, 256 * 16);
        scheduler.mark_prefill_complete(&first_id, 256);

        enqueue_waiting(
            &scheduler,
            create_test_request_with_prompt_tokens(Priority::Normal, 256),
        );
        let mixed_batch = scheduler.create_iteration_batch(hint).unwrap();
        assert_eq!(mixed_batch.requests.len(), 2);
        assert_eq!(mixed_batch.resource_requirements.gpu_memory, (1 + 64) * 16);
    }

    #[test]
    fn active_decode_prefill_chunk_caps_aggregate_mixed_prefill_tokens() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prompt_token_estimate: true,
            active_decode_prefill_chunk: Some(64),
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 8,
            max_tokens: 2048,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        let first = create_test_request_with_prompt_tokens(Priority::Normal, 256);
        let first_id = first.id.clone();
        enqueue_waiting(&scheduler, first);
        let initial_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(initial_batch.requests.len(), 1);
        scheduler.mark_prefill_complete(&first_id, 256);

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 256),
            );
        }

        let mixed_batch = scheduler.create_iteration_batch(hint).unwrap();
        assert_eq!(
            mixed_batch.requests.len(),
            5,
            "low decode pressure should admit more prefill chunks from batch headroom"
        );
        assert_eq!(mixed_batch.resource_requirements.gpu_memory, (1 + 256) * 16);
        assert_eq!(
            scheduler.prefilling_count(),
            4,
            "waiting requests may be promoted, but scheduling must respect the mixed-prefill budget"
        );
    }

    #[test]
    fn active_decode_prefill_budget_scales_down_with_decode_pressure() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prompt_token_estimate: true,
            active_decode_prefill_chunk: Some(64),
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 8,
            max_tokens: 2048,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        let mut decode_ids = Vec::new();
        for _ in 0..6 {
            let request = create_test_request_with_prompt_tokens(Priority::Normal, 128);
            decode_ids.push(request.id.clone());
            enqueue_waiting(&scheduler, request);
        }
        let initial_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(initial_batch.requests.len(), 6);
        for id in &decode_ids {
            scheduler.mark_prefill_complete(id, 128);
        }

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 256),
            );
        }

        let mixed_batch = scheduler.create_iteration_batch(hint).unwrap();
        assert_eq!(
            mixed_batch.requests.len(),
            8,
            "high decode pressure should admit bounded partial prefills up to available slots"
        );
        assert_eq!(mixed_batch.resource_requirements.gpu_memory, (6 + 128) * 16);
    }

    #[test]
    fn active_decode_prefill_budget_caps_small_final_chunks_by_count() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 32,
            prompt_token_estimate: true,
            prefill_step_chunk: Some(6),
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 32,
            max_tokens: 192,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        let mut decode_ids = Vec::new();
        for _ in 0..19 {
            let request = create_test_request_with_prompt_tokens(Priority::Normal, 1);
            decode_ids.push(request.id.clone());
            enqueue_waiting(&scheduler, request);
        }
        let initial_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(initial_batch.requests.len(), 19);
        for id in &decode_ids {
            scheduler.mark_prefill_complete(id, 1);
        }

        for _ in 0..13 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 1),
            );
        }

        let mixed_batch = scheduler.create_iteration_batch(hint).unwrap();
        let prefill_tokens: Vec<_> = mixed_batch
            .requests
            .iter()
            .filter(|request| !decode_ids.contains(&request.request.id))
            .map(|request| request.tokens_to_process)
            .collect();

        assert_eq!(
            mixed_batch.requests.len(),
            23,
            "small final prefill chunks must not bypass the mixed-prefill count budget"
        );
        assert_eq!(prefill_tokens, vec![Some(1), Some(1), Some(1), Some(1)]);
    }

    #[test]
    fn active_decode_prefill_budget_uses_effective_step_chunk_for_aggregate_cap() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 32,
            prompt_token_estimate: true,
            active_decode_prefill_chunk: Some(8192),
            prefill_step_chunk: Some(64),
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 32,
            max_tokens: 8192,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        let mut decode_ids = Vec::new();
        for _ in 0..7 {
            let request = create_test_request_with_prompt_tokens(Priority::Normal, 128);
            decode_ids.push(request.id.clone());
            enqueue_waiting(&scheduler, request);
        }
        let initial_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(initial_batch.requests.len(), 7);
        for id in &decode_ids {
            scheduler.mark_prefill_complete(id, 128);
        }

        for _ in 0..25 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 256),
            );
        }

        let mixed_batch = scheduler.create_iteration_batch(hint).unwrap();
        let prefill_tokens: Vec<_> = mixed_batch
            .requests
            .iter()
            .filter(|request| request.tokens_to_process != Some(1))
            .map(|request| request.tokens_to_process)
            .collect();
        assert_eq!(
            mixed_batch.requests.len(),
            11,
            "large explicit active chunks must not bypass the prefill-step aggregate cap"
        );
        assert_eq!(prefill_tokens, vec![Some(64), Some(64), Some(64), Some(64)]);
        assert_eq!(mixed_batch.resource_requirements.gpu_memory, (7 + 256) * 16);
    }

    #[tokio::test]
    async fn test_cancel_waiting() {
        let config = SchedulerConfig::default();
        let scheduler = ContinuousBatchScheduler::new(config);

        let request = create_test_request(Priority::Normal);
        let id = request.id.clone();
        scheduler.submit(request).await.unwrap();

        assert_eq!(scheduler.waiting_count(), 1);

        let result = scheduler.cancel(id).await.unwrap();
        assert!(result);
        assert_eq!(scheduler.waiting_count(), 0);
    }

    #[tokio::test]
    async fn test_metrics() {
        let config = SchedulerConfig::default();
        let scheduler = ContinuousBatchScheduler::new(config);

        scheduler
            .submit(create_test_request(Priority::Normal))
            .await
            .unwrap();

        let metrics = scheduler.metrics();
        assert_eq!(metrics.waiting_requests, 1);
    }

    #[tokio::test]
    async fn metrics_track_queue_wait_time_on_admission() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
        scheduler
            .submit(create_test_request(Priority::Normal))
            .await
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));

        let batch = scheduler.next_batch(BatchHint::simple(1)).await;
        assert!(batch.is_some());

        let metrics = scheduler.metrics();
        assert_eq!(metrics.waiting_requests, 0);
        assert_eq!(metrics.running_requests, 1);
        assert!(
            metrics.avg_wait_time_ms >= 1.0,
            "expected non-zero wait time, got {}",
            metrics.avg_wait_time_ms
        );
    }

    #[test]
    fn test_cb_request_states() {
        let request = create_test_request(Priority::Normal);
        let cb_req = ContinuousBatchRequest::new(request);

        assert_eq!(cb_req.phase, RequestPhase::Waiting);
        assert!(!cb_req.is_active());
        assert!(!cb_req.is_finished());
    }

    /// Chunked prefill state machine: advance across multiple iterations,
    /// transition Prefilling → Decoding only on the final chunk.
    #[tokio::test]
    async fn chunked_prefill_advances_across_iterations() {
        let cb_cfg = ContinuousBatchConfig {
            enable_chunked_prefill: true,
            prefill_chunk_size: 128,
            ..ContinuousBatchConfig::default()
        };
        let scheduler =
            ContinuousBatchScheduler::with_cb_config(SchedulerConfig::default(), cb_cfg);

        let request = create_test_request(Priority::Normal);
        let req_id = request.id.clone();
        scheduler.submit(request).await.unwrap();

        // Pull a batch to promote waiting → prefilling
        let _ = scheduler.next_batch(BatchHint::simple(1024)).await;
        assert_eq!(scheduler.prefilling_count(), 1);
        assert_eq!(scheduler.decoding_count(), 0);

        // Engine reports: prompt is 400 tokens, first chunk processed 128.
        // 128 < 400 → still prefilling, no phase transition.
        let done = scheduler.mark_prefill_chunk_processed(&req_id, 400, 128);
        assert!(!done, "first chunk should not finish prefill");
        assert_eq!(scheduler.prefilling_count(), 1);
        assert_eq!(scheduler.decoding_count(), 0);

        // Second chunk — 256 of 400.
        let done = scheduler.mark_prefill_chunk_processed(&req_id, 400, 128);
        assert!(!done);
        assert_eq!(scheduler.prefilling_count(), 1);
        assert_eq!(scheduler.decoding_count(), 0);

        // Final chunk — covers remaining 144 (saturates at 400).
        let done = scheduler.mark_prefill_chunk_processed(&req_id, 400, 200);
        assert!(done, "last chunk should complete prefill");
        assert_eq!(scheduler.prefilling_count(), 0);
        assert_eq!(scheduler.decoding_count(), 1);
    }

    /// Legacy one-shot `mark_prefill_complete` still promotes correctly and
    /// sets offset to total (so the request won't be double-scheduled for
    /// more prefill if somehow still in the queue).
    #[tokio::test]
    async fn mark_prefill_complete_sets_offset_to_total() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
        let request = create_test_request(Priority::Normal);
        let req_id = request.id.clone();
        scheduler.submit(request).await.unwrap();
        let _ = scheduler.next_batch(BatchHint::simple(1024)).await;

        scheduler.mark_prefill_complete(&req_id, 256);

        assert_eq!(scheduler.prefilling_count(), 0);
        assert_eq!(scheduler.decoding_count(), 1);
    }

    fn activate_decode_requests(
        scheduler: &ContinuousBatchScheduler,
        count: usize,
    ) -> Vec<RequestId> {
        let mut request_ids = Vec::with_capacity(count);
        for _ in 0..count {
            let request = create_test_request_with_prompt_tokens(Priority::Normal, 1);
            request_ids.push(request.id.clone());
            enqueue_waiting(scheduler, request);
        }
        let batch = scheduler
            .create_iteration_batch(BatchHint::simple(count.max(1)))
            .expect("test requests should enter prefill");
        assert_eq!(batch.requests.len(), count);
        for request_id in &request_ids {
            scheduler.mark_prefill_complete(request_id, 1);
        }
        request_ids
    }

    #[test]
    fn execution_readiness_blocks_only_the_exact_frontier() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
        let request_ids = activate_decode_requests(&scheduler, 2);
        let receipt = scheduler
            .defer_for_execution_readiness(std::slice::from_ref(&request_ids[0]))
            .unwrap();

        let batch = scheduler
            .create_iteration_batch(BatchHint::simple(2))
            .expect("unrelated decode must remain runnable");
        let scheduled = batch
            .requests
            .iter()
            .map(|request| &request.request.id)
            .collect::<Vec<_>>();
        assert!(!scheduled.contains(&&request_ids[0]));
        assert!(scheduled.contains(&&request_ids[1]));
        assert!(!scheduler.all_active_execution_readiness_blocked());

        assert!(receipt.wake().mark_ready());
        let resumed = scheduler
            .create_iteration_batch(BatchHint::simple(2))
            .expect("ready ticket must authorize an exact reprobe");
        assert!(resumed
            .requests
            .iter()
            .any(|request| request.request.id == request_ids[0]));
    }

    #[test]
    fn stale_execution_readiness_wake_cannot_unblock_replacement_ticket() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
        let request_id = activate_decode_requests(&scheduler, 1).remove(0);
        let first = scheduler
            .defer_for_execution_readiness(std::slice::from_ref(&request_id))
            .unwrap();
        let stale = first.wake().clone();
        assert!(stale.mark_ready());
        assert!(scheduler
            .create_iteration_batch(BatchHint::simple(1))
            .is_some());

        let replacement = scheduler
            .defer_for_execution_readiness(std::slice::from_ref(&request_id))
            .unwrap();
        assert_ne!(stale.ticket_id(), replacement.wake().ticket_id());
        assert!(!stale.mark_ready());
        assert!(scheduler
            .create_iteration_batch(BatchHint::simple(1))
            .is_none());
        assert!(scheduler.all_active_execution_readiness_blocked());

        assert!(replacement.wake().mark_ready());
        assert!(scheduler
            .create_iteration_batch(BatchHint::simple(1))
            .is_some());
    }

    #[test]
    fn failed_execution_readiness_ticket_remains_parked_for_terminal_owner() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
        let request_id = activate_decode_requests(&scheduler, 1).remove(0);
        let receipt = scheduler
            .defer_for_execution_readiness(std::slice::from_ref(&request_id))
            .unwrap();
        assert!(receipt.wake().mark_failed());

        assert!(scheduler
            .create_iteration_batch(BatchHint::simple(1))
            .is_none());
        assert!(scheduler.all_active_execution_readiness_blocked());
        let snapshot = scheduler.trace_snapshot();
        assert_eq!(snapshot.execution_readiness_deferred_total, 1);
        assert_eq!(snapshot.execution_readiness_blocked_decode_len, 1);
    }

    #[test]
    fn execution_readiness_install_is_all_or_nothing() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
        let request_id = activate_decode_requests(&scheduler, 1).remove(0);
        let missing = RequestId::new();
        assert!(scheduler
            .defer_for_execution_readiness(&[request_id.clone(), missing])
            .is_err());

        let batch = scheduler
            .create_iteration_batch(BatchHint::simple(1))
            .expect("failed cohort install must not retain a partial block");
        assert_eq!(batch.requests[0].request.id, request_id);
    }
}
