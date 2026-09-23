//! Read-only obligations and compare-exact logical wave publication.
//! Nothing in this module acquires executor resources or output permissions.
use super::*;
use std::num::NonZeroUsize;

mod projection;
mod selection;
#[cfg(test)]
mod tests;

const MAX_PLANNING_REQUESTS: usize = 4096;
const MAX_PLANNING_SOURCES: usize = 4096;
type PlanningResult<T> = std::result::Result<T, PlanningStateUnavailable>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanningStateUnavailable {
    Busy,
    TooManyRequests { actual: usize, maximum: usize },
    TooManyCapacitySources,
    AllocationFailed,
    InvalidWake,
    InvalidState(&'static str),
    Unsupported(&'static str),
    CounterExhausted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanningRequestKey {
    pub request_id: RequestId,
    pub ticket: WaitingAdmissionTicket,
    pub generation: LogicalWorkGeneration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanningQueueKind {
    Waiting,
    Prefill,
    Decode,
    Preempted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanningReadiness {
    pub admitted: bool,
    /// Already committed output reached the request limit. The obligation
    /// remains visible until its actual lifecycle owner removes it.
    pub output_limit_reached: bool,
    pub unfinished_work: bool,
    pub pressure_held: bool,
    pub execution_blocked: bool,
    pub capacity_blocked: bool,
    pub maintenance_blocked: bool,
    pub prefix_blocked: bool,
}
impl PlanningReadiness {
    pub fn ready(self) -> bool {
        self.admitted
            && !self.output_limit_reached
            && !self.unfinished_work
            && !self.pressure_held
            && !self.execution_blocked
            && !self.capacity_blocked
            && !self.maintenance_blocked
            && !self.prefix_blocked
    }
}

/// Scheduler evidence. Engine timing, output readiness and resource authority
/// must still be joined and rechecked by their actual owners.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanningRequestState {
    pub key: PlanningRequestKey,
    pub queue: PlanningQueueKind,
    pub phase: RequestPhase,
    pub priority: Priority,
    pub fairness_rank: usize,
    pub readiness: PlanningReadiness,
    pub computed_tokens: usize,
    pub resident_tokens: usize,
    pub scheduled_tokens: usize,
    pub committed_output_tokens: usize,
    /// Historical KV frontier evicted by capacity yield. It excludes a
    /// sampled token that has not yet entered KV, so it is not a new prefill
    /// authorization boundary.
    pub recompute_target_tokens: Option<usize>,
    /// Exact tokenizer metadata or an already-published prompt boundary.
    /// Absent metadata remains unknown, never an estimated chunk size.
    pub prompt_tokens: Option<usize>,
    /// Complete input for this pending prefill: original tokenized prompt
    /// metadata plus genuinely committed output tokens. Never based on the
    /// mutable last-prefill boundary, which can already contain replayed
    /// output. Missing original prompt evidence remains unknown. Decode rows
    /// have no pending prefill and report None.
    pub prefill_context_tokens: Option<usize>,
    pub prefill_offset: usize,
    pub maximum_output_tokens: usize,
    pub prefill_chunk_ceiling: Option<usize>,
    pub restored_tokens: usize,
    seal: RequestSeal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RequestSeal {
    frontier: LogicalWorkFrontier,
    state: RequestState,
    prefill_tokens: usize,
    decode_tokens: usize,
    readiness: Option<(NonZeroU64, u8)>,
    deferral: Option<AdmissionDeferral>,
    maintenance: Option<ExecutionMaintenanceRetryTicket>,
    last_maintenance_epoch: Option<u64>,
    capacity_deferred_until: u64,
    mixed_attempt: Option<u64>,
    empty_retry: Option<u64>,
    from_decode: bool,
    last_iteration: u64,
    prefix_restore: (bool, usize, bool, bool),
    prefix_rendezvous: (bool, Option<(usize, u64, u64)>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QueueSeal {
    iteration: u64,
    pressure_revision: u64,
    decode_cursor: Option<RequestId>,
    wake: AdmissionWakeEpochs,
    availability: Vec<CapacityAvailabilityEpoch>,
    capacity_release: u64,
    mixed_epoch: u64,
    decode_limit: usize,
    prefill_limit: usize,
}

/// Fields are private: callers cannot prune an obligation or replace a fence.
#[derive(Debug, Clone)]
pub struct PlanningQueueSnapshot {
    owner: Arc<()>,
    maximum_requests: NonZeroUsize,
    requests: Vec<PlanningRequestState>,
    seal: QueueSeal,
}
impl PlanningQueueSnapshot {
    pub fn requests(&self) -> &[PlanningRequestState] {
        &self.requests
    }
    pub fn iteration(&self) -> u64 {
        self.seal.iteration
    }
    pub fn wake_epochs(&self) -> AdmissionWakeEpochs {
        self.seal.wake
    }
    /// Captured logical width ceilings. These are not physical permits.
    pub fn decode_wave_limit(&self) -> usize {
        self.seal.decode_limit
    }
    pub fn prefill_wave_limit(&self) -> usize {
        self.seal.prefill_limit
    }
    fn matches(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner)
            && self.requests == other.requests
            && self.seal == other.seal
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanningWorkAction {
    Decode,
    Prefill { offset: usize, count: NonZeroUsize },
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanningWorkSelection {
    pub key: PlanningRequestKey,
    pub action: PlanningWorkAction,
}

#[derive(Debug)]
pub enum PlanningSelectionOutcome {
    Published {
        batch: BatchPlan,
        receipt: PlanningPublicationReceipt,
    },
    Stale,
    Rejected(&'static str),
}

/// One empty scheduler turn after an unsubmitted maintenance retry. This
/// advances only the logical fairness clock; it grants no work or resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanningMaintenanceFairnessOutcome {
    Advanced {
        previous_iteration: u64,
        next_iteration: u64,
        matured_tickets: usize,
    },
    NoPending,
    Stale,
}

/// Used explicitly only after the executor proved NotSubmitted/Unsupported.
/// Drop does not reopen work: submitted failures must never become retries.
#[derive(Debug)]
pub struct PlanningPublicationReceipt {
    owner: Arc<()>,
    iteration: u64,
    rows: Vec<(PlanningRequestKey, LogicalWorkFrontier)>,
}

/// All rows were examined under the queue locks. Superseded rows belonged to
/// an earlier incarnation/frontier or have left the scheduler; none was changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanningPublicationRelease {
    pub released_rows: usize,
    pub superseded_rows: usize,
}

/// All references are held simultaneously. Callers acquire each lock with
/// try_lock, abandoning the whole operation on contention; no lock-order wait.
struct Queues<'a> {
    waiting: &'a DynamicAdmissionQueue<ContinuousBatchRequest>,
    prefill: &'a VecDeque<ContinuousBatchRequest>,
    decode: &'a DecodeQueueState,
    preempted: &'a HashMap<RequestId, ContinuousBatchRequest>,
    index: &'a HashMap<RequestId, RequestPhase>,
    pressure: &'a PressureCoordinator,
}

impl ContinuousBatchScheduler {
    pub fn planning_state(
        &self,
        maximum_requests: NonZeroUsize,
        wake: AdmissionWakeSnapshot<'_>,
    ) -> PlanningResult<PlanningQueueSnapshot> {
        let waiting = self
            .waiting_queue
            .try_read()
            .ok_or(PlanningStateUnavailable::Busy)?;
        let prefill = self
            .prefill_queue
            .try_read()
            .ok_or(PlanningStateUnavailable::Busy)?;
        let decode = self
            .decode_queue
            .try_read()
            .ok_or(PlanningStateUnavailable::Busy)?;
        let preempted = self
            .preempted_requests
            .try_read()
            .ok_or(PlanningStateUnavailable::Busy)?;
        let index = self
            .request_index
            .try_read()
            .ok_or(PlanningStateUnavailable::Busy)?;
        let pressure = self
            .pressure_coordinator
            .try_lock()
            .ok_or(PlanningStateUnavailable::Busy)?;
        self.project_planning_state(
            maximum_requests,
            wake,
            &Queues {
                waiting: &waiting,
                prefill: &prefill,
                decode: &decode,
                preempted: &preempted,
                index: &index,
                pressure: &pressure,
            },
        )
    }
}
