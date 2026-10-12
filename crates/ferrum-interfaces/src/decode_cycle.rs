//! Opt-in, coarse host timing for one physical decode cycle.
//!
//! These witnesses are diagnostic data, never execution authority. They retain
//! no Step, lease, runtime, buffer, or completion receipt. Producers never wait
//! for the recorder: contention and saturation make coverage incomplete.

use crate::vnext::{
    BatchInvocationId, BatchStepId, CompletionSlotId, DeviceId, DeviceRuntime, ExecutionFrameId,
    ExecutionLaneId, PreparedStepSubmissionWave, RequestAuthorityId, RequestIdentity,
    SequenceAuthorityId, SubmittedOperationReceipt,
};
use ferrum_types::RequestId;
use serde::{Serialize, Serializer};
use std::{
    collections::{BTreeSet, HashSet},
    fmt,
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex, TryLockError,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

static NEXT_CLOCK_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DecodeCycleClockId(u64);

/// A timestamp can only be created by its owning diagnostic clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DecodeCycleTimestamp {
    clock_id: DecodeCycleClockId,
    nanos: u64,
}

impl DecodeCycleTimestamp {
    pub const fn clock_id(self) -> DecodeCycleClockId {
        self.clock_id
    }
    pub const fn nanos(self) -> u64 {
        self.nanos
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DecodeCycleClock {
    pub id: DecodeCycleClockId,
    pub wall_anchor_unix_nanos: i64,
    pub wall_anchor_max_error_nanos: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DecodeCycleCallContext {
    clock_id: DecodeCycleClockId,
    pub iteration: u64,
    pub call: u32,
    pub caller_ordinal: u32,
}

impl DecodeCycleCallContext {
    pub const fn clock_id(self) -> DecodeCycleClockId {
        self.clock_id
    }
}

#[derive(Debug, Clone)]
pub struct DecodeCycleCaller {
    request_id: RequestId,
    caller_ordinal: u32,
    context: Option<DecodeCycleCallContext>,
    request_identity: Option<RequestIdentity>,
}

impl DecodeCycleCaller {
    pub fn new(
        request_id: RequestId,
        caller_ordinal: u32,
        context: Option<DecodeCycleCallContext>,
    ) -> Self {
        Self {
            request_id,
            caller_ordinal,
            context,
            request_identity: None,
        }
    }
    /// Copy the actual admitted identity from this caller's resolved sequence.
    /// The witness builder checks it against the prepared canonical participant.
    pub fn with_request_identity(mut self, identity: RequestIdentity) -> Self {
        self.request_identity = Some(identity);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeCycleWaveKind {
    Prefill,
    Decode,
    Mixed,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecodeCycleParticipantOwner {
    pub sequence_authority: SequenceAuthorityId,
    pub request_authority: RequestAuthorityId,
    pub frame_id: ExecutionFrameId,
    pub request_id: RequestId,
    pub request_identity: RequestIdentity,
    pub canonical_ordinal: u32,
    pub caller_ordinal: u32,
    pub context: Option<DecodeCycleCallContext>,
}

#[derive(Debug)]
pub struct DecodeCycleWaveWitness {
    clock_id: DecodeCycleClockId,
    kind: DecodeCycleWaveKind,
    batch_step_id: BatchStepId,
    batch_invocation_id: BatchInvocationId,
    lane_id: ExecutionLaneId,
    device_id: DeviceId,
    runtime_implementation_fingerprint: String,
    prepared_wave_fingerprint: String,
    owners: Vec<DecodeCycleParticipantOwner>,
}

impl DecodeCycleWaveWitness {
    pub const fn clock_id(&self) -> DecodeCycleClockId {
        self.clock_id
    }
    pub const fn kind(&self) -> DecodeCycleWaveKind {
        self.kind
    }
    pub const fn batch_step_id(&self) -> BatchStepId {
        self.batch_step_id
    }
    pub const fn batch_invocation_id(&self) -> BatchInvocationId {
        self.batch_invocation_id
    }
    pub const fn lane_id(&self) -> ExecutionLaneId {
        self.lane_id
    }
    pub fn device_id(&self) -> &DeviceId {
        &self.device_id
    }
    pub fn runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime_implementation_fingerprint
    }
    pub fn prepared_wave_fingerprint(&self) -> &str {
        &self.prepared_wave_fingerprint
    }
    pub fn owners(&self) -> &[DecodeCycleParticipantOwner] {
        &self.owners
    }
}

impl Serialize for DecodeCycleWaveWitness {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Wire<'a> {
            clock_id: DecodeCycleClockId,
            kind: DecodeCycleWaveKind,
            batch_step_id: BatchStepId,
            batch_invocation_id: BatchInvocationId,
            lane_id: ExecutionLaneId,
            device_id: &'a DeviceId,
            runtime_implementation_fingerprint: &'a str,
            prepared_wave_fingerprint: &'a str,
            owners: &'a [DecodeCycleParticipantOwner],
        }
        Wire {
            clock_id: self.clock_id,
            kind: self.kind,
            batch_step_id: self.batch_step_id,
            batch_invocation_id: self.batch_invocation_id,
            lane_id: self.lane_id,
            device_id: &self.device_id,
            runtime_implementation_fingerprint: &self.runtime_implementation_fingerprint,
            prepared_wave_fingerprint: &self.prepared_wave_fingerprint,
            owners: &self.owners,
        }
        .serialize(serializer)
    }
}

#[derive(Debug)]
pub struct DecodeCycleSubmissionWitness {
    wave: Arc<DecodeCycleWaveWitness>,
    slot_id: CompletionSlotId,
    submission_fingerprint: String,
}

impl DecodeCycleSubmissionWitness {
    /// Late binding checks the actual receipt without materializing any nodes.
    pub fn bind(
        wave: Arc<DecodeCycleWaveWitness>,
        receipt: &SubmittedOperationReceipt,
    ) -> Result<Arc<Self>, DecodeCycleTraceError> {
        let batch = receipt.batch_identity();
        validate_submission_metadata(
            &wave,
            batch.batch_step_id(),
            batch.batch_invocation_id(),
            batch.lane_id(),
            batch.claimed_backing_fingerprint(),
        )?;
        if batch.device_id() != &wave.device_id
            || batch.runtime_implementation_fingerprint() != wave.runtime_implementation_fingerprint
        {
            return Err(DecodeCycleTraceError::ForeignSubmission);
        }
        Ok(Arc::new(Self {
            wave,
            slot_id: receipt.slot_id(),
            submission_fingerprint: receipt.fingerprint().to_owned(),
        }))
    }

    pub fn wave(&self) -> &Arc<DecodeCycleWaveWitness> {
        &self.wave
    }
    pub const fn slot_id(&self) -> CompletionSlotId {
        self.slot_id
    }
    pub fn submission_fingerprint(&self) -> &str {
        &self.submission_fingerprint
    }

    pub fn participant(
        self: &Arc<Self>,
        canonical_ordinal: u32,
    ) -> Result<DecodeCycleParticipantWitness, DecodeCycleTraceError> {
        if self.wave.owners.get(canonical_ordinal as usize).is_none() {
            return Err(DecodeCycleTraceError::InvalidCallerMap);
        }
        Ok(DecodeCycleParticipantWitness {
            submission: Arc::clone(self),
            canonical_ordinal,
        })
    }
}

#[derive(Debug, Clone)]
pub struct DecodeCycleParticipantWitness {
    submission: Arc<DecodeCycleSubmissionWitness>,
    canonical_ordinal: u32,
}

impl DecodeCycleParticipantWitness {
    pub fn submission(&self) -> &Arc<DecodeCycleSubmissionWitness> {
        &self.submission
    }
    pub const fn canonical_ordinal(&self) -> u32 {
        self.canonical_ordinal
    }
    pub fn owner(&self) -> &DecodeCycleParticipantOwner {
        &self.submission.wave.owners[self.canonical_ordinal as usize]
    }
    pub fn caller_ordinal(&self) -> u32 {
        self.owner().caller_ordinal
    }
}

/// Owned observer passed across the worker/runtime boundary, holding only data.
#[derive(Debug, Clone)]
pub struct DecodeCycleWaveContext {
    trace: DecodeCycleTrace,
    witness: Arc<DecodeCycleWaveWitness>,
    submission: Option<Arc<DecodeCycleSubmissionWitness>>,
    first_graph_seen: Arc<AtomicBool>,
}

impl DecodeCycleWaveContext {
    pub fn new(
        trace: DecodeCycleTrace,
        witness: Arc<DecodeCycleWaveWitness>,
    ) -> Result<Self, DecodeCycleTraceError> {
        if trace.clock().id != witness.clock_id {
            return Err(DecodeCycleTraceError::ForeignClock);
        }
        Ok(Self {
            trace,
            witness,
            submission: None,
            first_graph_seen: Arc::new(AtomicBool::new(false)),
        })
    }
    pub fn for_submission(
        trace: DecodeCycleTrace,
        submission: Arc<DecodeCycleSubmissionWitness>,
    ) -> Result<Self, DecodeCycleTraceError> {
        if trace.clock().id != submission.wave.clock_id {
            return Err(DecodeCycleTraceError::ForeignClock);
        }
        Ok(Self {
            trace,
            witness: Arc::clone(&submission.wave),
            submission: Some(submission),
            first_graph_seen: Arc::new(AtomicBool::new(false)),
        })
    }
    pub fn trace(&self) -> &DecodeCycleTrace {
        &self.trace
    }
    pub fn witness(&self) -> &Arc<DecodeCycleWaveWitness> {
        &self.witness
    }
    pub fn submission(&self) -> Option<&Arc<DecodeCycleSubmissionWitness>> {
        self.submission.as_ref()
    }
    pub fn timestamp(&self) -> DecodeCycleTimestamp {
        self.trace.timestamp()
    }
    pub fn scope(&self) -> DecodeCycleScope {
        match self.submission.as_ref() {
            Some(submission) => DecodeCycleScope::Submission(Arc::clone(submission)),
            None => DecodeCycleScope::Wave(Arc::clone(&self.witness)),
        }
    }
    pub fn mark(&self, stage: DecodeCycleStage) -> DecodeCycleRecordStatus {
        self.trace.mark(stage, self.scope())
    }
    pub fn mark_at(
        &self,
        at: DecodeCycleTimestamp,
        stage: DecodeCycleStage,
    ) -> DecodeCycleRecordStatus {
        self.trace.mark_at(at, stage, self.scope())
    }
    pub fn mark_with_outcome(
        &self,
        stage: DecodeCycleStage,
        outcome: DecodeCycleOutcome,
    ) -> DecodeCycleRecordStatus {
        self.trace.mark_with_outcome(stage, self.scope(), outcome)
    }
    /// A diagnostic selection flag, never permission to launch device work.
    pub fn claim_first_graph_launch(&self) -> bool {
        !self.first_graph_seen.swap(true, Ordering::Relaxed)
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct DecodeCycleWorkerTiming {
    pub queued_at: DecodeCycleTimestamp,
    pub started_at: DecodeCycleTimestamp,
    pub result_ready_at: DecodeCycleTimestamp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeCycleStage {
    LoopEntered,
    LoopResumed,
    IterationLockStart,
    IterationLockEnd,
    AdmissionStart,
    AdmissionEnd,
    ScheduleStart,
    ScheduleEnd,
    IterationEnd,
    CallStarted,
    CallReturned,
    PrepareStart,
    PrepareEnd,
    WavePrepared,
    EncodeStart,
    EncodeEnd,
    ProgramBindingCoalesceStart,
    ProgramBindingCoalesceEnd,
    ScatterAllocationStart,
    ScatterAllocationEnd,
    ScatterReadinessWaitStart,
    ScatterReadinessWaitEnd,
    CommandEnqueueStart,
    CommandEnqueueEnd,
    /// Host API boundary of the first compute graph; not first kernel/GPU start.
    ComputeGraphLaunchStart,
    ComputeGraphLaunchEnd,
    SubmissionBound,
    CompletionQueued,
    CompletionWorkerStarted,
    TerminalReadbackStart,
    TerminalReadbackEnd,
    CompletionResultReady,
    CompletionAwaitResumed,
    RetirementStart,
    RetirementEnd,
    ProductReturn,
    RequestLockStart,
    RequestLockEnd,
    SampleStart,
    SampleEnd,
    TokenCommit,
    StopCheckStart,
    StopCheckEnd,
    TextDecodeStart,
    TextDecodeEnd,
    TextLockStart,
    TextLockEnd,
    StreamUpdateStart,
    StreamUpdateEnd,
    StreamSendStart,
    StreamSendEnd,
    StreamYieldStart,
    StreamYieldResumed,
    LoopYieldStart,
    LoopYieldResumed,
    RequestCompleteStart,
    RequestCompleteEnd,
    FailureObserved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeCycleOutcome {
    Success,
    Failed,
    Canceled,
    Unavailable,
    Panicked,
}

#[derive(Debug, Clone)]
pub enum DecodeCycleScope {
    Iteration(u64),
    Call(DecodeCycleCallContext),
    Wave(Arc<DecodeCycleWaveWitness>),
    Submission(Arc<DecodeCycleSubmissionWitness>),
    Participant(DecodeCycleParticipantWitness),
}

impl DecodeCycleScope {
    fn clock_id(&self) -> Option<DecodeCycleClockId> {
        match self {
            Self::Iteration(_) => None,
            Self::Call(call) => Some(call.clock_id),
            Self::Wave(wave) => Some(wave.clock_id),
            Self::Submission(submission) => Some(submission.wave.clock_id),
            Self::Participant(participant) => Some(participant.submission.wave.clock_id),
        }
    }
}

// Event scopes serialize small join keys. WavePrepared separately declares the
// full participant table once; it is not repeated at every hot-path boundary.
impl Serialize for DecodeCycleScope {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        #[serde(tag = "scope", rename_all = "snake_case")]
        enum Wire<'a> {
            Iteration {
                iteration: u64,
            },
            Call {
                context: DecodeCycleCallContext,
            },
            Wave {
                clock_id: DecodeCycleClockId,
                batch_step_id: BatchStepId,
                batch_invocation_id: BatchInvocationId,
                prepared_wave_fingerprint: &'a str,
            },
            Submission {
                clock_id: DecodeCycleClockId,
                batch_step_id: BatchStepId,
                batch_invocation_id: BatchInvocationId,
                prepared_wave_fingerprint: &'a str,
                slot_id: CompletionSlotId,
                submission_fingerprint: &'a str,
            },
            Participant {
                clock_id: DecodeCycleClockId,
                slot_id: CompletionSlotId,
                submission_fingerprint: &'a str,
                owner: &'a DecodeCycleParticipantOwner,
            },
        }
        match self {
            Self::Iteration(iteration) => Wire::Iteration {
                iteration: *iteration,
            },
            Self::Call(context) => Wire::Call { context: *context },
            Self::Wave(wave) => Wire::Wave {
                clock_id: wave.clock_id,
                batch_step_id: wave.batch_step_id,
                batch_invocation_id: wave.batch_invocation_id,
                prepared_wave_fingerprint: &wave.prepared_wave_fingerprint,
            },
            Self::Submission(submission) => Wire::Submission {
                clock_id: submission.wave.clock_id,
                batch_step_id: submission.wave.batch_step_id,
                batch_invocation_id: submission.wave.batch_invocation_id,
                prepared_wave_fingerprint: &submission.wave.prepared_wave_fingerprint,
                slot_id: submission.slot_id,
                submission_fingerprint: &submission.submission_fingerprint,
            },
            Self::Participant(participant) => Wire::Participant {
                clock_id: participant.submission.wave.clock_id,
                slot_id: participant.submission.slot_id,
                submission_fingerprint: &participant.submission.submission_fingerprint,
                owner: participant.owner(),
            },
        }
        .serialize(serializer)
    }
}

#[derive(Debug)]
pub struct DecodeCycleEvent {
    pub at: DecodeCycleTimestamp,
    pub stage: DecodeCycleStage,
    pub scope: DecodeCycleScope,
    pub outcome: Option<DecodeCycleOutcome>,
}

impl Serialize for DecodeCycleEvent {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Wire<'a> {
            at: DecodeCycleTimestamp,
            stage: DecodeCycleStage,
            scope: &'a DecodeCycleScope,
            #[serde(skip_serializing_if = "Option::is_none")]
            outcome: Option<DecodeCycleOutcome>,
            #[serde(skip_serializing_if = "Option::is_none")]
            wave_witness: Option<&'a DecodeCycleWaveWitness>,
        }
        let wave_witness = match (&self.stage, &self.scope) {
            (DecodeCycleStage::WavePrepared, DecodeCycleScope::Wave(wave)) => Some(wave.as_ref()),
            _ => None,
        };
        Wire {
            at: self.at,
            stage: self.stage,
            scope: &self.scope,
            outcome: self.outcome,
            wave_witness,
        }
        .serialize(serializer)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeCycleTraceError {
    Allocation,
    ClockUnavailable,
    ClockExhausted,
    EmptyWave,
    InvalidCallerMap,
    ForeignClock,
    ForeignSubmission,
}

impl fmt::Display for DecodeCycleTraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "decode cycle diagnostic: {self:?}")
    }
}
impl std::error::Error for DecodeCycleTraceError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeCycleRecordStatus {
    Accepted,
    Busy,
    Full,
    Closed,
    Poisoned,
    InvalidScope,
    InvalidTimestamp,
}

#[derive(Debug, Default)]
struct Counters {
    attempted: AtomicU64,
    accepted: AtomicU64,
    busy: AtomicU64,
    full: AtomicU64,
    closed: AtomicU64,
    poisoned: AtomicU64,
    invalid_metadata: AtomicU64,
    outstanding_producers: AtomicU64,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct DecodeCycleStats {
    pub attempted: u64,
    pub accepted: u64,
    pub busy: u64,
    pub full: u64,
    pub closed: u64,
    pub poisoned: u64,
    pub invalid_metadata: u64,
    pub outstanding_producers: u64,
}

impl Counters {
    fn snapshot(&self) -> DecodeCycleStats {
        DecodeCycleStats {
            attempted: self.attempted.load(Ordering::Relaxed),
            accepted: self.accepted.load(Ordering::Relaxed),
            busy: self.busy.load(Ordering::Relaxed),
            full: self.full.load(Ordering::Relaxed),
            closed: self.closed.load(Ordering::Relaxed),
            poisoned: self.poisoned.load(Ordering::Relaxed),
            invalid_metadata: self.invalid_metadata.load(Ordering::Relaxed),
            outstanding_producers: self.outstanding_producers.load(Ordering::Acquire),
        }
    }
}

#[derive(Debug)]
struct TraceInner {
    started: Instant,
    clock: DecodeCycleClock,
    capacity: usize,
    closed: AtomicBool,
    events: Mutex<Vec<DecodeCycleEvent>>,
    counters: Counters,
}

#[derive(Debug, Clone)]
pub struct DecodeCycleTrace {
    inner: Arc<TraceInner>,
}

/// Single owner used only after engine and completion-worker natural drain.
#[derive(Debug)]
pub struct DecodeCycleDrainOwner {
    inner: Arc<TraceInner>,
}

impl Drop for DecodeCycleDrainOwner {
    fn drop(&mut self) {
        self.inner.closed.store(true, Ordering::Release);
    }
}

/// Diagnostic producer liveness, not an inference/resource ownership guard.
/// The worker retains this through its last event and result-channel send.
#[derive(Debug)]
pub struct DecodeCycleProducerGuard {
    inner: Arc<TraceInner>,
}

impl Drop for DecodeCycleProducerGuard {
    fn drop(&mut self) {
        self.inner
            .counters
            .outstanding_producers
            .fetch_sub(1, Ordering::Release);
    }
}

#[derive(Debug, Serialize)]
pub struct DecodeCycleCapture {
    pub clock: DecodeCycleClock,
    pub capacity: usize,
    pub active_producers_at_close: u64,
    pub stats: DecodeCycleStats,
    pub events: Vec<DecodeCycleEvent>,
}

impl DecodeCycleCapture {
    /// Collector completeness only; required boundary/identity coverage must
    /// still be checked by the consumer. Events may need timestamp sorting.
    pub fn has_complete_coverage(&self) -> bool {
        self.stats.busy == 0
            && self.stats.full == 0
            && self.stats.closed == 0
            && self.stats.poisoned == 0
            && self.stats.invalid_metadata == 0
            && self.stats.outstanding_producers == 0
            && self.active_producers_at_close == 0
            && self.stats.accepted == self.events.len() as u64
    }
}

impl DecodeCycleTrace {
    pub fn new(
        capacity: NonZeroUsize,
    ) -> Result<(Self, DecodeCycleDrainOwner), DecodeCycleTraceError> {
        let mut events = Vec::new();
        events
            .try_reserve_exact(capacity.get())
            .map_err(|_| DecodeCycleTraceError::Allocation)?;
        let wall_before = unix_nanos()?;
        let started = Instant::now();
        let wall_after = unix_nanos()?;
        let lower = wall_before.min(wall_after);
        let upper = wall_before.max(wall_after);
        let span = upper.saturating_sub(lower);
        let id = NEXT_CLOCK_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| DecodeCycleTraceError::ClockExhausted)?;
        let clock = DecodeCycleClock {
            id: DecodeCycleClockId(id),
            wall_anchor_unix_nanos: lower.saturating_add(span / 2),
            wall_anchor_max_error_nanos: span as u64,
        };
        let inner = Arc::new(TraceInner {
            started,
            clock,
            capacity: capacity.get(),
            closed: AtomicBool::new(false),
            events: Mutex::new(events),
            counters: Counters::default(),
        });
        Ok((
            Self {
                inner: Arc::clone(&inner),
            },
            DecodeCycleDrainOwner { inner },
        ))
    }
    pub const fn clock_source() -> &'static str {
        "rust_std_instant"
    }
    pub fn clock(&self) -> DecodeCycleClock {
        self.inner.clock
    }
    pub fn timestamp(&self) -> DecodeCycleTimestamp {
        let nanos = self.inner.started.elapsed().as_nanos();
        let nanos = match u64::try_from(nanos) {
            Ok(nanos) => nanos,
            Err(_) => {
                self.invalidate(DecodeCycleTraceError::ClockExhausted);
                u64::MAX
            }
        };
        DecodeCycleTimestamp {
            clock_id: self.inner.clock.id,
            nanos,
        }
    }
    pub fn call_context(
        &self,
        iteration: u64,
        call: u32,
        caller_ordinal: u32,
    ) -> DecodeCycleCallContext {
        DecodeCycleCallContext {
            clock_id: self.inner.clock.id,
            iteration,
            call,
            caller_ordinal,
        }
    }
    pub fn producer_guard(&self) -> DecodeCycleProducerGuard {
        self.inner
            .counters
            .outstanding_producers
            .fetch_add(1, Ordering::AcqRel);
        if self.inner.closed.load(Ordering::Acquire) {
            self.inner.counters.closed.fetch_add(1, Ordering::Relaxed);
        }
        DecodeCycleProducerGuard {
            inner: Arc::clone(&self.inner),
        }
    }
    pub fn wave_witness<R: DeviceRuntime>(
        &self,
        wave: &PreparedStepSubmissionWave<R>,
        kind: DecodeCycleWaveKind,
        canonical_callers: Vec<DecodeCycleCaller>,
    ) -> Result<Arc<DecodeCycleWaveWitness>, DecodeCycleTraceError> {
        let first = wave
            .nodes()
            .first()
            .ok_or(DecodeCycleTraceError::EmptyWave)?;
        let owners = validate_callers(
            self.inner.clock.id,
            first
                .participant_frames()
                .iter()
                .zip(first.participants())
                .map(|(frame, sequence)| {
                    (
                        frame.sequence_authority(),
                        frame.request_authority(),
                        frame.frame_id(),
                        sequence.request_id().clone(),
                    )
                }),
            canonical_callers,
        )?;
        let plan = first.plan_evidence();
        Ok(Arc::new(DecodeCycleWaveWitness {
            clock_id: self.inner.clock.id,
            kind,
            batch_step_id: wave.batch_step_id(),
            batch_invocation_id: wave.batch_invocation_id(),
            lane_id: wave.execution_lane_id(),
            device_id: plan.device_id().clone(),
            runtime_implementation_fingerprint: plan
                .runtime_implementation_fingerprint()
                .to_owned(),
            prepared_wave_fingerprint: wave.fingerprint().to_owned(),
            owners,
        }))
    }
    pub fn invalidate(&self, _error: DecodeCycleTraceError) {
        self.inner
            .counters
            .invalid_metadata
            .fetch_add(1, Ordering::Relaxed);
    }
    pub fn stats(&self) -> DecodeCycleStats {
        self.inner.counters.snapshot()
    }
    pub fn mark(
        &self,
        stage: DecodeCycleStage,
        scope: DecodeCycleScope,
    ) -> DecodeCycleRecordStatus {
        self.mark_at(self.timestamp(), stage, scope)
    }
    pub fn mark_at(
        &self,
        at: DecodeCycleTimestamp,
        stage: DecodeCycleStage,
        scope: DecodeCycleScope,
    ) -> DecodeCycleRecordStatus {
        self.mark_with_outcome_at(at, stage, scope, None)
    }
    pub fn mark_with_outcome(
        &self,
        stage: DecodeCycleStage,
        scope: DecodeCycleScope,
        outcome: DecodeCycleOutcome,
    ) -> DecodeCycleRecordStatus {
        self.mark_with_outcome_at(self.timestamp(), stage, scope, Some(outcome))
    }
    pub fn mark_with_outcome_at(
        &self,
        at: DecodeCycleTimestamp,
        stage: DecodeCycleStage,
        scope: DecodeCycleScope,
        outcome: Option<DecodeCycleOutcome>,
    ) -> DecodeCycleRecordStatus {
        let counters = &self.inner.counters;
        counters.attempted.fetch_add(1, Ordering::Relaxed);
        if at.clock_id != self.inner.clock.id {
            self.invalidate(DecodeCycleTraceError::ForeignClock);
            return DecodeCycleRecordStatus::InvalidTimestamp;
        }
        if scope.clock_id().is_some_and(|id| id != self.inner.clock.id) {
            self.invalidate(DecodeCycleTraceError::ForeignClock);
            return DecodeCycleRecordStatus::InvalidScope;
        }
        if self.inner.closed.load(Ordering::Acquire) {
            counters.closed.fetch_add(1, Ordering::Relaxed);
            return DecodeCycleRecordStatus::Closed;
        }
        let mut events = match self.inner.events.try_lock() {
            Ok(events) => events,
            Err(TryLockError::WouldBlock) => {
                counters.busy.fetch_add(1, Ordering::Relaxed);
                return DecodeCycleRecordStatus::Busy;
            }
            Err(TryLockError::Poisoned(_)) => {
                counters.poisoned.fetch_add(1, Ordering::Relaxed);
                return DecodeCycleRecordStatus::Poisoned;
            }
        };
        if self.inner.closed.load(Ordering::Acquire) {
            counters.closed.fetch_add(1, Ordering::Relaxed);
            return DecodeCycleRecordStatus::Closed;
        }
        if events.len() == self.inner.capacity {
            counters.full.fetch_add(1, Ordering::Relaxed);
            return DecodeCycleRecordStatus::Full;
        }
        events.push(DecodeCycleEvent {
            at,
            stage,
            scope,
            outcome,
        });
        counters.accepted.fetch_add(1, Ordering::Relaxed);
        DecodeCycleRecordStatus::Accepted
    }
}

impl DecodeCycleDrainOwner {
    /// Seals and extracts exactly once. Call only after all producers quiesce;
    /// any later call to `mark` is rejected and observable through trace.stats().
    pub fn finish(self) -> DecodeCycleCapture {
        self.inner.closed.store(true, Ordering::Release);
        let active_producers_at_close = self
            .inner
            .counters
            .outstanding_producers
            .load(Ordering::Acquire);
        let mut events = match self.inner.events.lock() {
            Ok(events) => events,
            Err(poisoned) => {
                self.inner.counters.poisoned.fetch_add(1, Ordering::Relaxed);
                poisoned.into_inner()
            }
        };
        let captured = std::mem::take(&mut *events);
        DecodeCycleCapture {
            clock: self.inner.clock,
            capacity: self.inner.capacity,
            active_producers_at_close,
            stats: self.inner.counters.snapshot(),
            events: captured,
        }
    }
}

fn unix_nanos() -> Result<i64, DecodeCycleTraceError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| DecodeCycleTraceError::ClockUnavailable)?;
    i64::try_from(elapsed.as_nanos()).map_err(|_| DecodeCycleTraceError::ClockUnavailable)
}

fn validate_callers(
    clock_id: DecodeCycleClockId,
    frames: impl ExactSizeIterator<
        Item = (
            SequenceAuthorityId,
            RequestAuthorityId,
            ExecutionFrameId,
            RequestIdentity,
        ),
    >,
    callers: Vec<DecodeCycleCaller>,
) -> Result<Vec<DecodeCycleParticipantOwner>, DecodeCycleTraceError> {
    if frames.len() == 0 {
        return Err(DecodeCycleTraceError::EmptyWave);
    }
    if frames.len() != callers.len() {
        return Err(DecodeCycleTraceError::InvalidCallerMap);
    }
    let mut seen_callers = BTreeSet::new();
    let mut seen_requests = HashSet::new();
    let mut seen_owners = BTreeSet::new();
    let mut owners = Vec::new();
    owners
        .try_reserve_exact(frames.len())
        .map_err(|_| DecodeCycleTraceError::Allocation)?;
    for (canonical, (frame, caller)) in frames.zip(callers).enumerate() {
        // A split/adaptive physical wave can preserve gaps in caller ordinals.
        let call = caller
            .context
            .map(|context| (context.iteration, context.call));
        if !seen_callers.insert((call, caller.caller_ordinal))
            || !seen_requests.insert(caller.request_id.clone())
            || !seen_owners.insert((frame.0, frame.1, frame.2))
            || caller.request_identity.as_ref() != Some(&frame.3)
        {
            return Err(DecodeCycleTraceError::InvalidCallerMap);
        }
        if let Some(context) = caller.context {
            if context.clock_id != clock_id {
                return Err(DecodeCycleTraceError::ForeignClock);
            }
            if context.caller_ordinal != caller.caller_ordinal {
                return Err(DecodeCycleTraceError::InvalidCallerMap);
            }
        }
        owners.push(DecodeCycleParticipantOwner {
            sequence_authority: frame.0,
            request_authority: frame.1,
            frame_id: frame.2,
            request_id: caller.request_id,
            request_identity: frame.3,
            canonical_ordinal: u32::try_from(canonical)
                .map_err(|_| DecodeCycleTraceError::InvalidCallerMap)?,
            caller_ordinal: caller.caller_ordinal,
            context: caller.context,
        });
    }
    Ok(owners)
}

fn validate_submission_metadata(
    wave: &DecodeCycleWaveWitness,
    step: BatchStepId,
    invocation: BatchInvocationId,
    lane: ExecutionLaneId,
    fingerprint: &str,
) -> Result<(), DecodeCycleTraceError> {
    if wave.batch_step_id != step
        || wave.batch_invocation_id != invocation
        || wave.lane_id != lane
        || wave.prepared_wave_fingerprint != fingerprint
    {
        return Err(DecodeCycleTraceError::ForeignSubmission);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
