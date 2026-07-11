use serde::Serialize;
use std::collections::BTreeMap;
use std::fmt;
use std::num::NonZeroU64;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use super::{
    classify_device_error, defer_device_cleanup, DeferredDeviceCleanupDisposition,
    DeferredDeviceCleanupDomainId, DeferredDeviceCleanupTask, DeviceDescriptor, DeviceRuntime,
    DeviceTerminal, ExecutionIdentityEnvelope, FenceQuery, IdentifiedFailure,
    InvocationResourceLease, StreamState, VNextError,
};

fn invalid_completion(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

#[derive(Debug)]
pub enum ExecutionLaneCreationError<E> {
    Contract(VNextError),
    Device(E),
}

impl<E: fmt::Display> fmt::Display for ExecutionLaneCreationError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(error) => write!(formatter, "execution lane contract failed: {error}"),
            Self::Device(error) => write!(formatter, "execution lane creation failed: {error}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for ExecutionLaneCreationError<E> {}

struct ExecutionLaneState<S> {
    stream: S,
    in_flight: u64,
    fail_closed: bool,
}

/// Scheduler-owned stream lane. It is intentionally not bound to any request
/// or sequence and may enqueue multiple mixed-batch commands in stream order.
#[must_use = "scheduler-owned lanes must outlive every fence enqueued on them"]
pub struct ExecutionLane<R: DeviceRuntime> {
    runtime: Arc<R>,
    descriptor: DeviceDescriptor,
    fail_closed: AtomicBool,
    state: Mutex<ExecutionLaneState<R::Stream>>,
}

impl<R: DeviceRuntime> fmt::Debug for ExecutionLane<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExecutionLane")
            .field("device_id", &self.descriptor.id)
            .field(
                "runtime_implementation_fingerprint",
                &self.descriptor.runtime_implementation_fingerprint,
            )
            .finish_non_exhaustive()
    }
}

impl<R: DeviceRuntime> ExecutionLane<R> {
    pub fn create(runtime: Arc<R>) -> Result<Arc<Self>, ExecutionLaneCreationError<R::Error>> {
        runtime
            .descriptor()
            .validate()
            .map_err(ExecutionLaneCreationError::Contract)?;
        let descriptor = runtime.descriptor().clone();
        let stream = runtime
            .create_stream()
            .map_err(ExecutionLaneCreationError::Device)?;
        if runtime.descriptor() != &descriptor
            || runtime.stream_state(&stream) != StreamState::Ready
        {
            return Err(ExecutionLaneCreationError::Contract(invalid_completion(
                "execution lane creation requires a stable runtime descriptor and ready stream",
            )));
        }
        Ok(Arc::new(Self {
            runtime,
            descriptor,
            fail_closed: AtomicBool::new(false),
            state: Mutex::new(ExecutionLaneState {
                stream,
                in_flight: 0,
                fail_closed: false,
            }),
        }))
    }

    pub fn descriptor(&self) -> &DeviceDescriptor {
        &self.descriptor
    }

    pub fn is_reusable(&self) -> bool {
        if self.fail_closed.load(Ordering::Acquire) {
            return false;
        }
        self.state
            .lock()
            .map(|state| !state.fail_closed && !self.fail_closed.load(Ordering::Acquire))
            .unwrap_or(false)
    }

    pub fn is_fail_closed(&self) -> bool {
        !self.is_reusable()
    }

    pub fn in_flight_count(&self) -> u64 {
        self.state
            .lock()
            .map(|state| state.in_flight)
            .unwrap_or(u64::MAX)
    }

    pub(crate) fn runtime(&self) -> &R {
        &self.runtime
    }

    pub(crate) fn runtime_arc(&self) -> &Arc<R> {
        &self.runtime
    }

    pub(crate) fn current_descriptor_matches_snapshot(&self) -> bool {
        self.runtime.descriptor() == &self.descriptor
    }

    /// Locks only the enqueue critical section. Provider encode must complete
    /// before this reservation is acquired.
    pub(crate) fn reserve_enqueue(&self) -> Result<ExecutionLaneEnqueue<'_, R>, VNextError> {
        if self.fail_closed.load(Ordering::Acquire) {
            return Err(invalid_completion("execution lane is fail-closed"));
        }
        if !self.current_descriptor_matches_snapshot() {
            return Err(invalid_completion(
                "execution lane runtime descriptor differs from its creation snapshot",
            ));
        }
        let state = self
            .state
            .lock()
            .map_err(|_| invalid_completion("execution lane state mutex is poisoned"))?;
        if self.fail_closed.load(Ordering::Acquire)
            || state.fail_closed
            || !matches!(
                self.runtime.stream_state(&state.stream),
                StreamState::Ready | StreamState::Submitted
            )
        {
            return Err(invalid_completion(
                "execution lane is failed or not enqueue-capable",
            ));
        }
        Ok(ExecutionLaneEnqueue { lane: self, state })
    }

    fn query_fence(&self, fence: &R::Fence) -> FenceQuery<R::Error> {
        self.runtime.query_fence(fence)
    }

    fn wait_fence(
        &self,
        fence: &R::Fence,
    ) -> Result<DeviceTerminal<R::Error>, super::FenceIndeterminate<R::Error>> {
        self.runtime.wait_fence(fence)
    }

    fn finish_one_terminal(&self) -> Result<(), QuiescentCompletionContractFailure> {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        if state.in_flight == 0 {
            state.fail_closed = true;
            self.fail_closed.store(true, Ordering::Release);
            return Err(QuiescentCompletionContractFailure::new(
                "completion lane in-flight accounting underflowed",
            ));
        }
        state.in_flight -= 1;
        let descriptor_matches = catch_unwind(AssertUnwindSafe(|| {
            self.current_descriptor_matches_snapshot()
        }))
        .unwrap_or(false);
        let stream_not_failed = catch_unwind(AssertUnwindSafe(|| {
            !matches!(
                self.runtime.stream_state(&state.stream),
                StreamState::Failed
            )
        }))
        .unwrap_or(false);
        if !descriptor_matches || !stream_not_failed {
            state.fail_closed = true;
            self.fail_closed.store(true, Ordering::Release);
        }
        if !descriptor_matches {
            Err(QuiescentCompletionContractFailure::new(
                "completion runtime descriptor drifted at terminal accounting",
            ))
        } else if !stream_not_failed {
            Err(QuiescentCompletionContractFailure::new(
                "completion lane stream entered failed state at terminal accounting",
            ))
        } else {
            Ok(())
        }
    }

    fn drain(&self, retires_submitted_fence: bool) -> bool {
        self.fail_closed.store(true, Ordering::Release);
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        state.fail_closed = true;
        let drained = catch_unwind(AssertUnwindSafe(|| {
            self.runtime.synchronize(&mut state.stream).is_ok()
                && self.current_descriptor_matches_snapshot()
                && self.runtime.stream_state(&state.stream) == StreamState::Ready
        }))
        .unwrap_or(false);
        if !drained {
            state.fail_closed = true;
            self.fail_closed.store(true, Ordering::Release);
            return false;
        }
        // Synchronization proves lane-wide quiescence, but this call retires
        // ownership for one exact completion record. Sibling records remain in
        // the reaper and must account for their own terminal observations.
        if retires_submitted_fence {
            let Some(remaining) = state.in_flight.checked_sub(1) else {
                state.fail_closed = true;
                self.fail_closed.store(true, Ordering::Release);
                return false;
            };
            state.in_flight = remaining;
        }
        true
    }

    pub(crate) fn fail_closed(&self) {
        self.fail_closed.store(true, Ordering::Release);
    }
}

pub(crate) enum LaneSubmitOutcome<F, E> {
    Submitted(F),
    DefinitelyNotSubmitted(E),
    PossiblySubmittedPanic,
}

pub(crate) struct ExecutionLaneEnqueue<'a, R: DeviceRuntime> {
    lane: &'a ExecutionLane<R>,
    state: MutexGuard<'a, ExecutionLaneState<R::Stream>>,
}

impl<R: DeviceRuntime> ExecutionLaneEnqueue<'_, R> {
    pub(crate) fn submit(&mut self, command: R::Command) -> LaneSubmitOutcome<R::Fence, R::Error> {
        let submission = catch_unwind(AssertUnwindSafe(|| {
            self.lane.runtime.submit(&mut self.state.stream, command)
        }));
        match submission {
            Ok(Ok(fence)) => {
                if let Some(next) = self.state.in_flight.checked_add(1) {
                    self.state.in_flight = next;
                } else {
                    self.state.fail_closed = true;
                    self.lane.fail_closed.store(true, Ordering::Release);
                }
                LaneSubmitOutcome::Submitted(fence)
            }
            Ok(Err(not_submitted)) => {
                LaneSubmitOutcome::DefinitelyNotSubmitted(not_submitted.into_error())
            }
            Err(_) => {
                self.state.fail_closed = true;
                self.lane.fail_closed.store(true, Ordering::Release);
                LaneSubmitOutcome::PossiblySubmittedPanic
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct CompletionSlotId(NonZeroU64);

impl CompletionSlotId {
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[must_use = "physical submission evidence must be recorded"]
pub struct SubmittedOperationReceipt {
    slot_id: CompletionSlotId,
    identity: ExecutionIdentityEnvelope,
}

impl SubmittedOperationReceipt {
    pub const fn slot_id(&self) -> CompletionSlotId {
        self.slot_id
    }

    pub fn identity(&self) -> &ExecutionIdentityEnvelope {
        &self.identity
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "status", content = "failure")]
pub enum OperationCompletionDisposition {
    Succeeded,
    FailedButQuiescent(IdentifiedFailure),
    ContractFailedButQuiescent(QuiescentCompletionContractFailure),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[must_use = "a quiescent contract failure is a terminal operation outcome"]
pub struct QuiescentCompletionContractFailure {
    reason: String,
}

impl QuiescentCompletionContractFailure {
    fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    pub fn reason(&self) -> &str {
        &self.reason
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[must_use = "a terminal completion receipt releases one exact invocation"]
pub struct OperationCompletionReceipt {
    submission: SubmittedOperationReceipt,
    disposition: OperationCompletionDisposition,
}

impl OperationCompletionReceipt {
    pub fn submission(&self) -> &SubmittedOperationReceipt {
        &self.submission
    }

    pub fn disposition(&self) -> &OperationCompletionDisposition {
        &self.disposition
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletionRecoveryCause {
    SubmissionIndeterminate,
    FenceIndeterminate,
    FenceObservationPanicked,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[must_use = "a lane-drain receipt proves quiescence for one exact completion slot"]
pub struct CompletionDrainReceipt {
    slot_id: CompletionSlotId,
    identity: ExecutionIdentityEnvelope,
    cause: CompletionRecoveryCause,
    had_submission_fence: bool,
}

impl CompletionDrainReceipt {
    pub const fn slot_id(&self) -> CompletionSlotId {
        self.slot_id
    }

    pub fn identity(&self) -> &ExecutionIdentityEnvelope {
        &self.identity
    }

    pub const fn cause(&self) -> CompletionRecoveryCause {
        self.cause
    }

    pub const fn had_submission_fence(&self) -> bool {
        self.had_submission_fence
    }
}

#[derive(Debug, Clone)]
struct CompletionQuarantineFreshness(Arc<AtomicBool>);

impl CompletionQuarantineFreshness {
    fn current() -> Self {
        Self(Arc::new(AtomicBool::new(true)))
    }

    fn is_current(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    fn invalidate(&self) {
        self.0.store(false, Ordering::Release);
    }
}

#[derive(Debug, Clone, Serialize)]
#[must_use = "a quarantine receipt identifies retained device-visible ownership"]
pub struct CompletionQuarantineReceipt {
    slot_id: CompletionSlotId,
    identity: ExecutionIdentityEnvelope,
    cause: CompletionRecoveryCause,
    had_submission_fence: bool,
    device_id: super::DeviceId,
    runtime_implementation_fingerprint: String,
    #[serde(skip)]
    freshness: CompletionQuarantineFreshness,
}

impl PartialEq for CompletionQuarantineReceipt {
    fn eq(&self, other: &Self) -> bool {
        self.slot_id == other.slot_id
            && self.identity == other.identity
            && self.cause == other.cause
            && self.had_submission_fence == other.had_submission_fence
            && self.device_id == other.device_id
            && self.runtime_implementation_fingerprint == other.runtime_implementation_fingerprint
    }
}

impl Eq for CompletionQuarantineReceipt {}

impl CompletionQuarantineReceipt {
    pub const fn slot_id(&self) -> CompletionSlotId {
        self.slot_id
    }

    pub fn identity(&self) -> &ExecutionIdentityEnvelope {
        &self.identity
    }

    pub const fn cause(&self) -> CompletionRecoveryCause {
        self.cause
    }

    pub const fn had_submission_fence(&self) -> bool {
        self.had_submission_fence
    }

    pub fn device_id(&self) -> &super::DeviceId {
        &self.device_id
    }

    pub fn runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime_implementation_fingerprint
    }

    pub fn is_current(&self) -> bool {
        self.freshness.is_current()
    }
}

#[derive(Debug)]
#[must_use = "completion recovery must be recorded as drained or quarantined"]
pub enum CompletionRecoveryOutcome {
    Drained(CompletionDrainReceipt),
    Quarantined(CompletionQuarantineReceipt),
}

#[derive(Debug)]
pub enum CompletionSweepObservation {
    Observed(CompletionObservation),
    Failed(VNextError),
}

#[derive(Debug)]
pub struct CompletionSweepEntry {
    slot_id: CompletionSlotId,
    observation: CompletionSweepObservation,
}

impl CompletionSweepEntry {
    pub const fn slot_id(&self) -> CompletionSlotId {
        self.slot_id
    }

    pub fn observation(&self) -> &CompletionSweepObservation {
        &self.observation
    }

    pub fn into_observation(self) -> CompletionSweepObservation {
        self.observation
    }
}

#[derive(Debug)]
#[must_use = "a bounded completion sweep contains scheduler-owned progress evidence"]
pub struct CompletionSweepReceipt {
    entries: Vec<CompletionSweepEntry>,
    retained_after: usize,
    quarantined_after: usize,
}

impl CompletionSweepReceipt {
    pub fn entries(&self) -> &[CompletionSweepEntry] {
        &self.entries
    }

    pub fn into_entries(self) -> Vec<CompletionSweepEntry> {
        self.entries
    }

    pub const fn retained_after(&self) -> usize {
        self.retained_after
    }

    pub const fn quarantined_after(&self) -> usize {
        self.quarantined_after
    }
}

enum CompletionRecord<R: DeviceRuntime> {
    Reserved,
    InFlight {
        invocation: InvocationResourceLease<R>,
        lane: Arc<ExecutionLane<R>>,
        fence: R::Fence,
        identity: ExecutionIdentityEnvelope,
        receipt: SubmittedOperationReceipt,
        recovery_state: CompletionRecoveryState,
    },
    SubmissionIndeterminate {
        invocation: InvocationResourceLease<R>,
        lane: Arc<ExecutionLane<R>>,
        identity: ExecutionIdentityEnvelope,
    },
    Quarantined {
        ownership: CompletionQuarantineOwnership<R>,
        receipt: CompletionQuarantineReceipt,
    },
    Reaped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompletionRecoveryState {
    Unobserved,
    QueryIndeterminate,
    DrainEligible(CompletionRecoveryCause),
}

enum CompletionQuarantineOwnership<R: DeviceRuntime> {
    InFlight {
        invocation: InvocationResourceLease<R>,
        lane: Arc<ExecutionLane<R>>,
        fence: R::Fence,
        identity: ExecutionIdentityEnvelope,
        submission: SubmittedOperationReceipt,
    },
    SubmissionIndeterminate {
        invocation: InvocationResourceLease<R>,
        lane: Arc<ExecutionLane<R>>,
        identity: ExecutionIdentityEnvelope,
    },
}

impl<R: DeviceRuntime> CompletionQuarantineOwnership<R> {
    fn lane(&self) -> &Arc<ExecutionLane<R>> {
        match self {
            Self::InFlight { lane, .. } | Self::SubmissionIndeterminate { lane, .. } => lane,
        }
    }

    fn deferred_cleanup_domain(&self) -> DeferredDeviceCleanupDomainId {
        match self {
            Self::InFlight { invocation, .. }
            | Self::SubmissionIndeterminate { invocation, .. } => {
                invocation.deferred_cleanup_domain()
            }
        }
    }
}

impl<R: DeviceRuntime> CompletionRecord<R> {
    fn deferred_cleanup_domain(&self) -> Option<DeferredDeviceCleanupDomainId> {
        match self {
            Self::InFlight { invocation, .. }
            | Self::SubmissionIndeterminate { invocation, .. } => {
                Some(invocation.deferred_cleanup_domain())
            }
            Self::Quarantined { ownership, .. } => Some(ownership.deferred_cleanup_domain()),
            Self::Reserved | Self::Reaped => None,
        }
    }
}

type SharedCompletionRecord<R> = Arc<Mutex<CompletionRecord<R>>>;

struct CompletionReaperState<R: DeviceRuntime> {
    next_slot: Option<NonZeroU64>,
    sweep_cursor: Option<CompletionSlotId>,
    slots: BTreeMap<CompletionSlotId, SharedCompletionRecord<R>>,
}

/// Scheduler-owned completion registry. The global map lock only resolves a
/// slot; each fence is queried or waited under its own record lock.
#[must_use = "the scheduler must retain its completion reaper"]
pub struct CompletionReaper<R: DeviceRuntime> {
    state: Mutex<CompletionReaperState<R>>,
}

pub const MAX_COMPLETION_SWEEP_SLOTS: usize = 64;

impl<R: DeviceRuntime> fmt::Debug for CompletionReaper<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompletionReaper")
            .field("retained", &self.retained_count())
            .finish_non_exhaustive()
    }
}

impl<R: DeviceRuntime> CompletionReaper<R> {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(CompletionReaperState {
                next_slot: NonZeroU64::new(1),
                sweep_cursor: None,
                slots: BTreeMap::new(),
            }),
        })
    }

    pub fn retained_count(&self) -> usize {
        self.state
            .lock()
            .map(|state| state.slots.len())
            .unwrap_or(usize::MAX)
    }

    pub fn quarantined_count(&self) -> usize {
        let records = match self.state.lock() {
            Ok(state) => state.slots.values().cloned().collect::<Vec<_>>(),
            Err(_) => return usize::MAX,
        };
        records
            .iter()
            .filter(|record| {
                record
                    .lock()
                    .map(|record| matches!(&*record, CompletionRecord::Quarantined { .. }))
                    .unwrap_or(true)
            })
            .count()
    }

    /// Polls a bounded scheduler-owned snapshot. This path remains available
    /// after every external completion handle has detached.
    pub fn poll_bounded(&self, maximum_slots: usize) -> Result<CompletionSweepReceipt, VNextError> {
        if maximum_slots == 0 || maximum_slots > MAX_COMPLETION_SWEEP_SLOTS {
            return Err(invalid_completion(format!(
                "completion sweep size must be in 1..={MAX_COMPLETION_SWEEP_SLOTS}"
            )));
        }
        let slot_ids = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| invalid_completion("completion reaper state mutex is poisoned"))?;
            let keys = state.slots.keys().copied().collect::<Vec<_>>();
            let start = state
                .sweep_cursor
                .and_then(|cursor| keys.iter().position(|slot_id| *slot_id > cursor))
                .unwrap_or(0);
            let slot_ids = keys
                .iter()
                .cycle()
                .skip(start)
                .take(maximum_slots.min(keys.len()))
                .copied()
                .collect::<Vec<_>>();
            if let Some(last) = slot_ids.last() {
                state.sweep_cursor = Some(*last);
            }
            slot_ids
        };
        let entries = slot_ids
            .into_iter()
            .map(|slot_id| CompletionSweepEntry {
                slot_id,
                observation: match self.poll_bound(slot_id) {
                    Ok(observation) => CompletionSweepObservation::Observed(observation),
                    Err(error) => CompletionSweepObservation::Failed(error),
                },
            })
            .collect();
        let records = self
            .state
            .lock()
            .map_err(|_| invalid_completion("completion reaper state mutex is poisoned"))?
            .slots
            .values()
            .cloned()
            .collect::<Vec<_>>();
        let retained_after = records.len();
        let quarantined_after = records
            .iter()
            .filter(|record| {
                record
                    .lock()
                    .map(|record| matches!(&*record, CompletionRecord::Quarantined { .. }))
                    .unwrap_or(true)
            })
            .count();
        Ok(CompletionSweepReceipt {
            entries,
            retained_after,
            quarantined_after,
        })
    }

    /// Runs the blocking recovery path for an exact slot after an indeterminate
    /// observation. A failed drain moves ownership into an auditable, retryable
    /// quarantine record instead of releasing or forgetting it.
    pub fn recover_slot_by_draining_lane(
        &self,
        slot_id: CompletionSlotId,
    ) -> Result<CompletionRecoveryOutcome, VNextError> {
        self.recover_bound(slot_id)
    }

    /// Performs the blocking exact-fence observation required before lane
    /// recovery. Schedulers must call this from an independent recovery worker,
    /// never from a request or admission thread.
    pub fn wait_slot_for_recovery(
        &self,
        slot_id: CompletionSlotId,
    ) -> Result<CompletionObservation, VNextError> {
        self.wait_bound(slot_id)
    }

    pub(crate) fn reserve(
        reaper: &Arc<Self>,
        invocation: InvocationResourceLease<R>,
        lane: Arc<ExecutionLane<R>>,
        identity: ExecutionIdentityEnvelope,
    ) -> Result<CompletionReservation<R>, VNextError> {
        if !Arc::ptr_eq(invocation.runtime(), lane.runtime_arc()) {
            return Err(invalid_completion(
                "completion lane runtime is not the invocation root runtime instance",
            ));
        }
        let mut state = reaper
            .state
            .lock()
            .map_err(|_| invalid_completion("completion reaper state mutex is poisoned"))?;
        let raw = state
            .next_slot
            .ok_or_else(|| invalid_completion("completion slot identity space is exhausted"))?;
        state.next_slot = raw.get().checked_add(1).and_then(NonZeroU64::new);
        let slot_id = CompletionSlotId(raw);
        let record = Arc::new(Mutex::new(CompletionRecord::Reserved));
        if state.slots.insert(slot_id, Arc::clone(&record)).is_some() {
            return Err(invalid_completion("completion slot identity was reused"));
        }
        drop(state);
        let receipt = SubmittedOperationReceipt {
            slot_id,
            identity: identity.clone(),
        };
        let record_receipt = receipt.clone();
        Ok(CompletionReservation {
            reaper: Arc::clone(reaper),
            record,
            slot_id,
            invocation: Some(invocation),
            lane: Some(lane),
            identity: Some(identity),
            receipt: Some(receipt),
            record_receipt: Some(record_receipt),
            submission_may_have_happened: false,
            finished: false,
        })
    }

    fn lookup(&self, slot_id: CompletionSlotId) -> Result<SharedCompletionRecord<R>, VNextError> {
        self.state
            .lock()
            .map_err(|_| invalid_completion("completion reaper state mutex is poisoned"))?
            .slots
            .get(&slot_id)
            .cloned()
            .ok_or_else(|| invalid_completion("completion slot is unknown or already reaped"))
    }

    fn remove_exact(&self, slot_id: CompletionSlotId, record: &SharedCompletionRecord<R>) {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        if state
            .slots
            .get(&slot_id)
            .is_some_and(|current| Arc::ptr_eq(current, record))
        {
            state.slots.remove(&slot_id);
        }
    }

    pub(crate) fn poll_bound(
        &self,
        slot_id: CompletionSlotId,
    ) -> Result<CompletionObservation, VNextError> {
        self.observe_bound(slot_id, false)
    }

    pub(crate) fn wait_bound(
        &self,
        slot_id: CompletionSlotId,
    ) -> Result<CompletionObservation, VNextError> {
        self.observe_bound(slot_id, true)
    }

    fn observe_bound(
        &self,
        slot_id: CompletionSlotId,
        blocking: bool,
    ) -> Result<CompletionObservation, VNextError> {
        let record = self.lookup(slot_id)?;
        let mut guard = record
            .lock()
            .map_err(|_| invalid_completion("completion slot mutex is poisoned"))?;
        let observation = match &*guard {
            CompletionRecord::Reserved => {
                return Err(invalid_completion(
                    "completion slot has not reached submission",
                ));
            }
            CompletionRecord::SubmissionIndeterminate { .. } => {
                FenceObservation::SubmissionIndeterminate
            }
            CompletionRecord::Quarantined { receipt, .. } => {
                FenceObservation::Quarantined(receipt.clone())
            }
            CompletionRecord::Reaped => {
                return Err(invalid_completion("completion slot is already reaped"));
            }
            CompletionRecord::InFlight {
                lane,
                fence,
                identity,
                ..
            } if blocking => match catch_unwind(AssertUnwindSafe(|| lane.wait_fence(fence))) {
                Ok(Ok(terminal)) => catch_unwind(AssertUnwindSafe(|| {
                    terminal_observation(lane, identity, terminal)
                }))
                .unwrap_or(FenceObservation::ObservationPanicked),
                Ok(Err(indeterminate)) => catch_unwind(AssertUnwindSafe(|| {
                    classify_device_error(lane.runtime(), identity.clone(), indeterminate.error())
                }))
                .map_or(
                    FenceObservation::ObservationPanicked,
                    |classified| match classified {
                        Ok(failure) => FenceObservation::Indeterminate(failure),
                        Err(error) => FenceObservation::ContractIndeterminate(error),
                    },
                ),
                Err(_) => FenceObservation::ObservationPanicked,
            },
            CompletionRecord::InFlight {
                lane,
                fence,
                identity,
                ..
            } => match catch_unwind(AssertUnwindSafe(|| lane.query_fence(fence))) {
                Ok(FenceQuery::Pending) => FenceObservation::Pending,
                Ok(FenceQuery::Terminal(terminal)) => catch_unwind(AssertUnwindSafe(|| {
                    terminal_observation(lane, identity, terminal)
                }))
                .unwrap_or(FenceObservation::ObservationPanicked),
                Ok(FenceQuery::Indeterminate(error)) => catch_unwind(AssertUnwindSafe(|| {
                    classify_device_error(lane.runtime(), identity.clone(), &error)
                }))
                .map_or(
                    FenceObservation::ObservationPanicked,
                    |classified| match classified {
                        Ok(failure) => FenceObservation::Indeterminate(failure),
                        Err(error) => FenceObservation::ContractIndeterminate(error),
                    },
                ),
                Err(_) => FenceObservation::ObservationPanicked,
            },
        };
        let recovery_state = match &observation {
            FenceObservation::Indeterminate(_) | FenceObservation::ContractIndeterminate(_) => {
                Some(if blocking {
                    CompletionRecoveryState::DrainEligible(
                        CompletionRecoveryCause::FenceIndeterminate,
                    )
                } else {
                    CompletionRecoveryState::QueryIndeterminate
                })
            }
            FenceObservation::ObservationPanicked => Some(if blocking {
                CompletionRecoveryState::DrainEligible(
                    CompletionRecoveryCause::FenceObservationPanicked,
                )
            } else {
                CompletionRecoveryState::QueryIndeterminate
            }),
            _ => None,
        };
        if let Some(recovery_state) = recovery_state {
            if let CompletionRecord::InFlight {
                recovery_state: current,
                ..
            } = &mut *guard
            {
                if !matches!(current, CompletionRecoveryState::DrainEligible(_)) {
                    *current = recovery_state;
                }
            }
        }
        if let FenceObservation::Terminal(mut disposition) = observation {
            let old = std::mem::replace(&mut *guard, CompletionRecord::Reaped);
            let CompletionRecord::InFlight { lane, receipt, .. } = old else {
                unreachable!("terminal observation came from an in-flight record")
            };
            if let Err(failure) = lane.finish_one_terminal() {
                disposition = OperationCompletionDisposition::ContractFailedButQuiescent(failure);
            }
            drop(guard);
            self.remove_exact(slot_id, &record);
            return Ok(CompletionObservation::Terminal(
                OperationCompletionReceipt {
                    submission: receipt,
                    disposition,
                },
            ));
        }
        Ok(match observation {
            FenceObservation::Pending => CompletionObservation::Pending,
            FenceObservation::Indeterminate(failure) => {
                CompletionObservation::Indeterminate(failure)
            }
            FenceObservation::SubmissionIndeterminate => {
                CompletionObservation::SubmissionIndeterminate
            }
            FenceObservation::ObservationPanicked => CompletionObservation::ObservationPanicked,
            FenceObservation::ContractIndeterminate(error) => return Err(error),
            FenceObservation::Quarantined(receipt) => CompletionObservation::Quarantined(receipt),
            FenceObservation::Terminal(_) => {
                unreachable!("terminal observation returned through the reaping branch")
            }
        })
    }

    fn recover_bound(
        &self,
        slot_id: CompletionSlotId,
    ) -> Result<CompletionRecoveryOutcome, VNextError> {
        let record = self.lookup(slot_id)?;
        let mut guard = record
            .lock()
            .map_err(|_| invalid_completion("completion slot mutex is poisoned"))?;
        let (lane, identity, cause, had_submission_fence) = match &*guard {
            CompletionRecord::InFlight {
                lane,
                identity,
                recovery_state: CompletionRecoveryState::DrainEligible(cause),
                ..
            } => (Arc::clone(lane), identity.clone(), *cause, true),
            CompletionRecord::SubmissionIndeterminate { lane, identity, .. } => (
                Arc::clone(lane),
                identity.clone(),
                CompletionRecoveryCause::SubmissionIndeterminate,
                false,
            ),
            CompletionRecord::Quarantined { ownership, receipt } => (
                Arc::clone(ownership.lane()),
                receipt.identity.clone(),
                receipt.cause,
                receipt.had_submission_fence,
            ),
            CompletionRecord::InFlight { .. } => {
                return Err(invalid_completion(
                    "completion slot has no blocking indeterminate fence observation",
                ));
            }
            CompletionRecord::Reserved => {
                return Err(invalid_completion(
                    "completion slot has not reached submission",
                ));
            }
            CompletionRecord::Reaped => {
                return Err(invalid_completion("completion slot is already reaped"));
            }
        };
        // A recovery drain proves lane-wide quiescence rather than one fence's
        // terminal state. Retire the lane before draining so older records
        // cannot later decrement accounting for newly submitted work.
        lane.fail_closed();
        if !lane.drain(had_submission_fence) {
            if let CompletionRecord::Quarantined { receipt, .. } = &*guard {
                return Ok(CompletionRecoveryOutcome::Quarantined(receipt.clone()));
            }
            let old = std::mem::replace(&mut *guard, CompletionRecord::Reaped);
            let ownership = match old {
                CompletionRecord::InFlight {
                    invocation,
                    lane,
                    fence,
                    identity,
                    receipt,
                    ..
                } => CompletionQuarantineOwnership::InFlight {
                    invocation,
                    lane,
                    fence,
                    identity,
                    submission: receipt,
                },
                CompletionRecord::SubmissionIndeterminate {
                    invocation,
                    lane,
                    identity,
                } => CompletionQuarantineOwnership::SubmissionIndeterminate {
                    invocation,
                    lane,
                    identity,
                },
                _ => unreachable!("recovery source was validated before lane drain"),
            };
            let receipt = CompletionQuarantineReceipt {
                slot_id,
                identity,
                cause,
                had_submission_fence,
                device_id: lane.descriptor.id.clone(),
                runtime_implementation_fingerprint: lane
                    .descriptor
                    .runtime_implementation_fingerprint
                    .clone(),
                freshness: CompletionQuarantineFreshness::current(),
            };
            *guard = CompletionRecord::Quarantined {
                ownership,
                receipt: receipt.clone(),
            };
            return Ok(CompletionRecoveryOutcome::Quarantined(receipt));
        }
        let old = std::mem::replace(&mut *guard, CompletionRecord::Reaped);
        if let CompletionRecord::Quarantined { receipt, .. } = &old {
            receipt.freshness.invalidate();
        }
        drop(guard);
        self.remove_exact(slot_id, &record);
        drop(old);
        Ok(CompletionRecoveryOutcome::Drained(CompletionDrainReceipt {
            slot_id,
            identity,
            cause,
            had_submission_fence,
        }))
    }
}

enum FenceObservation {
    Pending,
    Terminal(OperationCompletionDisposition),
    Indeterminate(IdentifiedFailure),
    ContractIndeterminate(VNextError),
    SubmissionIndeterminate,
    ObservationPanicked,
    Quarantined(CompletionQuarantineReceipt),
}

fn terminal_observation<R: DeviceRuntime>(
    lane: &ExecutionLane<R>,
    identity: &ExecutionIdentityEnvelope,
    terminal: DeviceTerminal<R::Error>,
) -> FenceObservation {
    let descriptor_is_stable = catch_unwind(AssertUnwindSafe(|| {
        lane.current_descriptor_matches_snapshot()
    }))
    .unwrap_or(false);
    if !descriptor_is_stable {
        return FenceObservation::Terminal(
            OperationCompletionDisposition::ContractFailedButQuiescent(
                QuiescentCompletionContractFailure::new(
                    "completion runtime descriptor differs from its execution lane snapshot",
                ),
            ),
        );
    }
    FenceObservation::Terminal(match terminal {
        DeviceTerminal::Succeeded => OperationCompletionDisposition::Succeeded,
        DeviceTerminal::FailedButQuiescent(error) => {
            match classify_device_error(lane.runtime(), identity.clone(), &error) {
                Ok(failure) => OperationCompletionDisposition::FailedButQuiescent(failure),
                Err(error) => OperationCompletionDisposition::ContractFailedButQuiescent(
                    QuiescentCompletionContractFailure::new(error.to_string()),
                ),
            }
        }
    })
}

struct DeferredCompletionCleanup<R: DeviceRuntime> {
    records: BTreeMap<CompletionSlotId, SharedCompletionRecord<R>>,
}

impl<R: DeviceRuntime> DeferredCompletionCleanup<R> {
    fn new(records: BTreeMap<CompletionSlotId, SharedCompletionRecord<R>>) -> Self {
        Self { records }
    }
}

impl<R: DeviceRuntime> DeferredDeviceCleanupTask for DeferredCompletionCleanup<R> {
    fn try_cleanup(&mut self) -> DeferredDeviceCleanupDisposition {
        let slot_ids = self.records.keys().copied().collect::<Vec<_>>();
        let mut retryable = false;
        let mut quarantined = false;
        for slot_id in slot_ids {
            let Some(record) = self.records.get(&slot_id).cloned() else {
                continue;
            };
            let quiescent = catch_unwind(AssertUnwindSafe(|| {
                cleanup_dropped_completion_record(&record)
            }))
            .unwrap_or(false);
            if quiescent {
                if self
                    .records
                    .get(&slot_id)
                    .is_some_and(|current| Arc::ptr_eq(current, &record))
                {
                    self.records.remove(&slot_id);
                }
                continue;
            }
            retryable = true;
            quarantined |= record
                .lock()
                .map(|record| matches!(&*record, CompletionRecord::Quarantined { .. }))
                .unwrap_or(true);
        }
        if self.records.is_empty() {
            DeferredDeviceCleanupDisposition::Completed
        } else if quarantined {
            DeferredDeviceCleanupDisposition::Quarantined
        } else {
            debug_assert!(retryable);
            DeferredDeviceCleanupDisposition::Retryable
        }
    }
}

fn fail_close_completion_record<R: DeviceRuntime>(record: &SharedCompletionRecord<R>) {
    let guard = match record.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    match &*guard {
        CompletionRecord::InFlight { lane, .. }
        | CompletionRecord::SubmissionIndeterminate { lane, .. } => lane.fail_closed(),
        CompletionRecord::Quarantined { ownership, .. } => ownership.lane().fail_closed(),
        CompletionRecord::Reserved | CompletionRecord::Reaped => {}
    }
}

fn cleanup_dropped_completion_record<R: DeviceRuntime>(record: &SharedCompletionRecord<R>) -> bool {
    let mut guard = match record.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let (quiescent, drained) = match &*guard {
        CompletionRecord::InFlight { lane, fence, .. } => {
            let queried = catch_unwind(AssertUnwindSafe(|| lane.query_fence(fence)))
                .is_ok_and(|query| matches!(query, FenceQuery::Terminal(_)));
            let waited = queried
                || catch_unwind(AssertUnwindSafe(|| lane.wait_fence(fence)))
                    .is_ok_and(|result| result.is_ok());
            if waited {
                (true, false)
            } else {
                let drained = lane.drain(true);
                (drained, drained)
            }
        }
        CompletionRecord::SubmissionIndeterminate { lane, .. } => {
            let drained = lane.drain(false);
            (drained, drained)
        }
        CompletionRecord::Quarantined { ownership, receipt } => {
            let drained = ownership.lane().drain(receipt.had_submission_fence);
            (drained, drained)
        }
        CompletionRecord::Reserved | CompletionRecord::Reaped => (true, false),
    };
    if !quiescent {
        match &*guard {
            CompletionRecord::InFlight { lane, .. }
            | CompletionRecord::SubmissionIndeterminate { lane, .. } => lane.fail_closed(),
            CompletionRecord::Quarantined { ownership, .. } => ownership.lane().fail_closed(),
            CompletionRecord::Reserved | CompletionRecord::Reaped => {}
        }
        return false;
    }

    let old = std::mem::replace(&mut *guard, CompletionRecord::Reaped);
    if let CompletionRecord::Quarantined { receipt, .. } = &old {
        receipt.freshness.invalidate();
    }
    if !drained {
        if let CompletionRecord::InFlight { lane, .. } = &old {
            let _ = lane.finish_one_terminal();
        }
    }
    drop(guard);
    drop(old);
    true
}

impl<R: DeviceRuntime> Drop for CompletionReaper<R> {
    fn drop(&mut self) {
        let records = match self.state.get_mut() {
            Ok(state) => std::mem::take(&mut state.slots),
            Err(poisoned) => std::mem::take(&mut poisoned.into_inner().slots),
        };
        if records.is_empty() {
            return;
        }
        // No active reservation or observer can coexist with the final reaper
        // Arc: both retain or first upgrade that Arc. Record locks are therefore
        // uncontended here, and fail-closing each lane is an atomic operation.
        for record in records.values() {
            fail_close_completion_record(record);
        }
        for (slot_id, record) in records {
            let domain = record
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .deferred_cleanup_domain();
            if let Some(domain) = domain {
                // One slot per task prevents a blocked backend lane from
                // withholding unrelated lanes in the same plan. The global
                // registry mutex is never held during either recovery call.
                defer_device_cleanup(
                    domain,
                    DeferredCompletionCleanup::new(BTreeMap::from([(slot_id, record)])),
                );
            }
        }
    }
}

#[derive(Debug)]
#[must_use = "nonterminal completion observations retain invocation ownership"]
pub enum CompletionObservation {
    Pending,
    Terminal(OperationCompletionReceipt),
    Indeterminate(IdentifiedFailure),
    SubmissionIndeterminate,
    ObservationPanicked,
    Quarantined(CompletionQuarantineReceipt),
}

/// Weak recovery authority for a submit unwind where no fence was returned.
/// Only a successful lane-wide drain can release the retained invocation.
#[must_use = "an indeterminate submission must be drained or retained"]
pub struct IndeterminateSubmissionHandle<R: DeviceRuntime> {
    reaper: Weak<CompletionReaper<R>>,
    slot_id: CompletionSlotId,
}

impl<R: DeviceRuntime> Clone for IndeterminateSubmissionHandle<R> {
    fn clone(&self) -> Self {
        Self {
            reaper: Weak::clone(&self.reaper),
            slot_id: self.slot_id,
        }
    }
}

impl<R: DeviceRuntime> fmt::Debug for IndeterminateSubmissionHandle<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IndeterminateSubmissionHandle")
            .field("slot_id", &self.slot_id)
            .finish_non_exhaustive()
    }
}

impl<R: DeviceRuntime> IndeterminateSubmissionHandle<R> {
    pub const fn slot_id(&self) -> CompletionSlotId {
        self.slot_id
    }

    pub fn recover_by_draining_lane(&self) -> Result<CompletionRecoveryOutcome, VNextError> {
        self.reaper
            .upgrade()
            .ok_or_else(|| invalid_completion("completion reaper owner was dropped"))?
            .recover_bound(self.slot_id)
    }
}

/// Weakly bound observation authority. Dropping a handle cannot drop or reap
/// the scheduler-owned completion registry.
#[must_use = "a submitted operation handle observes its exact completion slot"]
pub struct CompletionHandle<R: DeviceRuntime> {
    reaper: Weak<CompletionReaper<R>>,
    receipt: SubmittedOperationReceipt,
}

impl<R: DeviceRuntime> Clone for CompletionHandle<R> {
    fn clone(&self) -> Self {
        Self {
            reaper: Weak::clone(&self.reaper),
            receipt: self.receipt.clone(),
        }
    }
}

impl<R: DeviceRuntime> fmt::Debug for CompletionHandle<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompletionHandle")
            .field("receipt", &self.receipt)
            .finish_non_exhaustive()
    }
}

impl<R: DeviceRuntime> CompletionHandle<R> {
    pub fn receipt(&self) -> &SubmittedOperationReceipt {
        &self.receipt
    }

    pub fn identity(&self) -> &ExecutionIdentityEnvelope {
        self.receipt.identity()
    }

    pub const fn slot_id(&self) -> CompletionSlotId {
        self.receipt.slot_id()
    }

    pub fn poll(&self) -> Result<CompletionObservation, VNextError> {
        self.reaper
            .upgrade()
            .ok_or_else(|| invalid_completion("completion reaper owner was dropped"))?
            .poll_bound(self.slot_id())
    }

    pub fn wait(&self) -> Result<CompletionObservation, VNextError> {
        self.reaper
            .upgrade()
            .ok_or_else(|| invalid_completion("completion reaper owner was dropped"))?
            .wait_bound(self.slot_id())
    }
}

pub(crate) struct CompletionReservation<R: DeviceRuntime> {
    reaper: Arc<CompletionReaper<R>>,
    record: SharedCompletionRecord<R>,
    slot_id: CompletionSlotId,
    invocation: Option<InvocationResourceLease<R>>,
    lane: Option<Arc<ExecutionLane<R>>>,
    identity: Option<ExecutionIdentityEnvelope>,
    receipt: Option<SubmittedOperationReceipt>,
    record_receipt: Option<SubmittedOperationReceipt>,
    submission_may_have_happened: bool,
    finished: bool,
}

impl<R: DeviceRuntime> CompletionReservation<R> {
    pub(crate) fn invocation(&self) -> &InvocationResourceLease<R> {
        self.invocation
            .as_ref()
            .expect("live completion reservation owns invocation resources")
    }

    pub(crate) fn mark_submission_started(&mut self) {
        self.submission_may_have_happened = true;
    }

    pub(crate) fn definitely_not_submitted(mut self) {
        self.remove_reserved_slot();
        self.submission_may_have_happened = false;
        self.finished = true;
    }

    pub(crate) fn arm(mut self, fence: R::Fence) -> CompletionHandle<R> {
        let invocation = self.invocation.take().expect("reservation owns invocation");
        let lane = self.lane.take().expect("reservation owns lane");
        let identity = self.identity.take().expect("reservation owns identity");
        let receipt = self.receipt.take().expect("reservation owns receipt");
        let record_receipt = self
            .record_receipt
            .take()
            .expect("reservation owns record receipt");
        let mut record = match self.record.lock() {
            Ok(record) => record,
            Err(poisoned) => poisoned.into_inner(),
        };
        if !matches!(&*record, CompletionRecord::Reserved) {
            lane.fail_closed();
            std::mem::forget((invocation, lane, fence));
            self.finished = true;
            panic!("completion reservation changed after submission");
        }
        *record = CompletionRecord::InFlight {
            invocation,
            lane,
            fence,
            identity,
            receipt: record_receipt,
            recovery_state: CompletionRecoveryState::Unobserved,
        };
        drop(record);
        self.finished = true;
        CompletionHandle {
            reaper: Arc::downgrade(&self.reaper),
            receipt,
        }
    }

    pub(crate) fn submission_indeterminate(mut self) -> IndeterminateSubmissionHandle<R> {
        let invocation = self.invocation.take().expect("reservation owns invocation");
        let lane = self.lane.take().expect("reservation owns lane");
        let identity = self.identity.take().expect("reservation owns identity");
        let mut record = match self.record.lock() {
            Ok(record) => record,
            Err(poisoned) => poisoned.into_inner(),
        };
        if matches!(&*record, CompletionRecord::Reserved) {
            *record = CompletionRecord::SubmissionIndeterminate {
                invocation,
                lane,
                identity,
            };
        } else {
            lane.fail_closed();
            std::mem::forget((invocation, lane, identity));
        }
        drop(record);
        self.finished = true;
        IndeterminateSubmissionHandle {
            reaper: Arc::downgrade(&self.reaper),
            slot_id: self.slot_id,
        }
    }

    fn remove_reserved_slot(&mut self) {
        self.reaper.remove_exact(self.slot_id, &self.record);
        let mut record = match self.record.lock() {
            Ok(record) => record,
            Err(poisoned) => poisoned.into_inner(),
        };
        if matches!(&*record, CompletionRecord::Reserved) {
            *record = CompletionRecord::Reaped;
        }
    }
}

impl<R: DeviceRuntime> Drop for CompletionReservation<R> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        if self.submission_may_have_happened {
            let Some(invocation) = self.invocation.take() else {
                return;
            };
            let Some(lane) = self.lane.take() else {
                std::mem::forget(invocation);
                return;
            };
            let Some(identity) = self.identity.take() else {
                std::mem::forget((invocation, lane));
                return;
            };
            lane.fail_closed();
            let mut record = match self.record.lock() {
                Ok(record) => record,
                Err(poisoned) => poisoned.into_inner(),
            };
            if matches!(&*record, CompletionRecord::Reserved) {
                *record = CompletionRecord::SubmissionIndeterminate {
                    invocation,
                    lane,
                    identity,
                };
            } else {
                std::mem::forget((invocation, lane, identity));
            }
        } else {
            self.remove_reserved_slot();
        }
    }
}
