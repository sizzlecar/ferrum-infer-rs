use serde::{ser::SerializeSeq, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::num::NonZeroU64;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::Instant;

use super::{
    classify_device_error, defer_device_cleanup, BackingInitializationEncodeError,
    BatchOperationIdentity, BatchParticipantAuthority, BufferUsage, CopyRegion,
    DeferredDeviceCleanupDisposition, DeferredDeviceCleanupDomainId, DeferredDeviceCleanupTask,
    DefinitelyNotSubmittedRetryAuthority, DefinitelyNotSubmittedWaveRetryAuthority,
    DeviceCommandBatch, DeviceDescriptor, DeviceExecutionTiming, DeviceReusableExecutionPlan,
    DeviceReusableExecutionPreparation, DeviceRuntime, DeviceSubmissionTimingSink, DeviceTerminal,
    DeviceTerminalReceipt, DeviceTimingMeasurement, DeviceTimingMode,
    DeviceTimingUnavailableReason, ExecutionIdentityEnvelope, ExecutionLaneId, FenceQuery,
    HostTransferLayout, IdentifiedFailure, InvocationResourceLease, LogicalBackingBufferView,
    NodeId, PreparedStepSubmissionWave, ResourceId, StreamState, VNextError,
};

mod readback_collection;
pub use readback_collection::*;

fn invalid_completion(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

fn canonical_completion_fingerprint(value: &impl Serialize) -> String {
    format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(value).expect("trusted completion evidence must serialize")
        )
    )
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

enum LaneReadbackError<E> {
    Contract(VNextError),
    Device(E),
}

struct LaneReadback {
    bytes: Vec<u8>,
    timing: DeviceTimingMeasurement<CompletionReadbackTiming>,
}

/// Scheduler-owned stream lane. It is intentionally not bound to any request
/// or sequence and may enqueue multiple mixed-batch commands in stream order.
#[must_use = "scheduler-owned lanes must outlive every fence enqueued on them"]
pub struct ExecutionLane<R: DeviceRuntime> {
    id: ExecutionLaneId,
    runtime: Arc<R>,
    descriptor: DeviceDescriptor,
    fail_closed: AtomicBool,
    state: Mutex<ExecutionLaneState<R::Stream>>,
}

impl<R: DeviceRuntime> fmt::Debug for ExecutionLane<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExecutionLane")
            .field("id", &self.id)
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
            id: ExecutionLaneId::mint().map_err(ExecutionLaneCreationError::Contract)?,
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

    pub const fn id(&self) -> ExecutionLaneId {
        self.id
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

    fn with_quiescent_stream<T>(
        &self,
        operation: &'static str,
        action: impl FnOnce(&R, &mut R::Stream) -> Result<T, R::Error>,
    ) -> Result<T, VNextError> {
        if self.fail_closed.load(Ordering::Acquire) {
            return Err(invalid_completion(format!(
                "fail-closed execution lane cannot {operation}"
            )));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_completion("execution lane state mutex is poisoned"))?;
        if state.fail_closed || self.fail_closed.load(Ordering::Acquire) {
            return Err(invalid_completion(format!(
                "fail-closed execution lane cannot {operation}"
            )));
        }
        if state.in_flight != 0
            || !self.current_descriptor_matches_snapshot()
            || self.runtime.stream_state(&state.stream) != StreamState::Ready
        {
            return Err(invalid_completion(format!(
                "{operation} requires a stable, quiescent execution lane"
            )));
        }
        let result = catch_unwind(AssertUnwindSafe(|| {
            action(self.runtime.as_ref(), &mut state.stream)
        }));
        match result {
            Ok(Ok(value))
                if self.current_descriptor_matches_snapshot()
                    && self.runtime.stream_state(&state.stream) == StreamState::Ready =>
            {
                Ok(value)
            }
            Ok(Ok(_)) => {
                state.fail_closed = true;
                self.fail_closed.store(true, Ordering::Release);
                Err(invalid_completion(format!(
                    "{operation} changed the execution lane state"
                )))
            }
            Ok(Err(error)) => {
                state.fail_closed = true;
                self.fail_closed.store(true, Ordering::Release);
                Err(invalid_completion(format!("{operation} failed: {error}")))
            }
            Err(_) => {
                state.fail_closed = true;
                self.fail_closed.store(true, Ordering::Release);
                Err(invalid_completion(format!(
                    "device runtime panicked while attempting to {operation}"
                )))
            }
        }
    }

    pub fn configure_reusable_executables(
        &self,
        plan: DeviceReusableExecutionPlan,
    ) -> Result<DeviceReusableExecutionPreparation, VNextError> {
        self.with_quiescent_stream("configure reusable executables", |runtime, stream| {
            runtime.configure_reusable_executables(stream, plan)
        })
    }

    pub fn seal_reusable_executables(
        &self,
    ) -> Result<DeviceReusableExecutionPreparation, VNextError> {
        self.with_quiescent_stream("seal reusable executables", |runtime, stream| {
            runtime.seal_reusable_executables(stream)
        })
    }

    pub fn reusable_executable_preparation(
        &self,
    ) -> Result<DeviceReusableExecutionPreparation, VNextError> {
        self.with_quiescent_stream(
            "inspect reusable executable preparation",
            |runtime, stream| runtime.reusable_executable_preparation(stream),
        )
    }

    pub(crate) fn trim_reusable_executables_if_quiescent(&self) -> Result<bool, VNextError> {
        if self.fail_closed.load(Ordering::Acquire) {
            return Err(invalid_completion(
                "fail-closed execution lane cannot trim reusable executables",
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_completion("execution lane state mutex is poisoned"))?;
        if state.fail_closed || self.fail_closed.load(Ordering::Acquire) {
            return Err(invalid_completion(
                "fail-closed execution lane cannot trim reusable executables",
            ));
        }
        if state.in_flight != 0 {
            return Ok(false);
        }
        if !self.current_descriptor_matches_snapshot()
            || self.runtime.stream_state(&state.stream) != StreamState::Ready
        {
            state.fail_closed = true;
            self.fail_closed.store(true, Ordering::Release);
            return Err(invalid_completion(
                "reusable executable trim requires a stable, quiescent execution lane",
            ));
        }
        let trimmed = catch_unwind(AssertUnwindSafe(|| {
            self.runtime.trim_reusable_executables(&mut state.stream)
        }));
        match trimmed {
            Ok(Ok(_))
                if self.current_descriptor_matches_snapshot()
                    && self.runtime.stream_state(&state.stream) == StreamState::Ready =>
            {
                Ok(true)
            }
            Ok(Ok(_)) => {
                state.fail_closed = true;
                self.fail_closed.store(true, Ordering::Release);
                Err(invalid_completion(
                    "reusable executable trim changed the execution lane state",
                ))
            }
            Ok(Err(error)) => {
                state.fail_closed = true;
                self.fail_closed.store(true, Ordering::Release);
                Err(invalid_completion(format!(
                    "device reusable executable trim failed: {error}"
                )))
            }
            Err(_) => {
                state.fail_closed = true;
                self.fail_closed.store(true, Ordering::Release);
                Err(invalid_completion(
                    "device runtime panicked while trimming reusable executables",
                ))
            }
        }
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
    ) -> Result<DeviceTerminalReceipt<R::Error>, super::FenceIndeterminate<R::Error>> {
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

    fn readback_activation(
        &self,
        backing: &LogicalBackingBufferView<'_, R::Buffer>,
        logical_offset_bytes: u64,
        output_layout: HostTransferLayout,
        timing_mode: DeviceTimingMode,
    ) -> Result<LaneReadback, LaneReadbackError<R::Error>> {
        let timing_started = (timing_mode == DeviceTimingMode::Completion).then(Instant::now);
        let output_bytes = output_layout
            .byte_len()
            .map_err(LaneReadbackError::Contract)?;
        let logical_end = logical_offset_bytes
            .checked_add(output_bytes)
            .ok_or_else(|| {
                LaneReadbackError::Contract(invalid_completion(
                    "completion readback logical range overflows u64",
                ))
            })?;
        let element_bytes = output_layout.element_type().size_bytes();
        if backing.usage() != BufferUsage::Activations
            || backing.element_type() != output_layout.element_type()
            || logical_end > backing.size_bytes()
            || logical_offset_bytes % element_bytes != 0
            || output_bytes % element_bytes != 0
        {
            return Err(LaneReadbackError::Contract(invalid_completion(
                "completion readback must select an aligned activation range with matching element type",
            )));
        }
        if self.fail_closed.load(Ordering::Acquire) || !self.current_descriptor_matches_snapshot() {
            return Err(LaneReadbackError::Contract(invalid_completion(
                "completion readback requires its original reusable execution lane",
            )));
        }
        let mut state = self.state.lock().map_err(|_| {
            LaneReadbackError::Contract(invalid_completion(
                "execution lane state mutex is poisoned during completion readback",
            ))
        })?;
        if state.fail_closed
            || !matches!(
                self.runtime.stream_state(&state.stream),
                StreamState::Ready | StreamState::Submitted
            )
        {
            return Err(LaneReadbackError::Contract(invalid_completion(
                "completion readback lane is failed or not readable",
            )));
        }

        let output_capacity = usize::try_from(output_bytes).map_err(|_| {
            LaneReadbackError::Contract(invalid_completion(
                "completion readback output exceeds host address space",
            ))
        })?;
        let mut output = Vec::with_capacity(output_capacity);
        let mut readback_calls = 0_u32;
        let mut logical_cursor = 0_u64;
        for binding in backing.segment_bindings() {
            let segment = binding.segment();
            let segment_logical_end = logical_cursor
                .checked_add(segment.length_bytes())
                .ok_or_else(|| {
                    LaneReadbackError::Contract(invalid_completion(
                        "completion readback backing coverage overflows u64",
                    ))
                })?;
            let overlap_start = logical_cursor.max(logical_offset_bytes);
            let overlap_end = segment_logical_end.min(logical_end);
            if overlap_start < overlap_end {
                let within_segment = overlap_start - logical_cursor;
                let source_offset = segment
                    .offset_bytes()
                    .checked_add(within_segment)
                    .ok_or_else(|| {
                        LaneReadbackError::Contract(invalid_completion(
                            "completion readback physical offset overflows u64",
                        ))
                    })?;
                let length = overlap_end - overlap_start;
                if source_offset % element_bytes != 0 || length % element_bytes != 0 {
                    return Err(LaneReadbackError::Contract(invalid_completion(
                        "completion readback backing segments split an element",
                    )));
                }
                let actual = self.runtime.buffer_descriptor(binding.buffer());
                if &actual != binding.descriptor()
                    || source_offset
                        .checked_add(length)
                        .is_none_or(|end| end > actual.size_bytes)
                {
                    return Err(LaneReadbackError::Contract(invalid_completion(
                        "completion readback backing descriptor differs from its committed extent",
                    )));
                }
                let piece_layout =
                    HostTransferLayout::new(output_layout.element_type(), length / element_bytes)
                        .map_err(LaneReadbackError::Contract)?;
                let region = CopyRegion::new(source_offset, 0, length)
                    .map_err(LaneReadbackError::Contract)?;
                readback_calls = readback_calls.checked_add(1).ok_or_else(|| {
                    LaneReadbackError::Contract(invalid_completion(
                        "completion readback call count exceeds u32",
                    ))
                })?;
                let piece = match self.runtime.readback(
                    &mut state.stream,
                    binding.buffer(),
                    region,
                    piece_layout,
                ) {
                    Ok(piece) => piece,
                    Err(error) => {
                        state.fail_closed = true;
                        self.fail_closed.store(true, Ordering::Release);
                        return Err(LaneReadbackError::Device(error));
                    }
                };
                if piece.len() != usize::try_from(length).unwrap_or(usize::MAX) {
                    state.fail_closed = true;
                    self.fail_closed.store(true, Ordering::Release);
                    return Err(LaneReadbackError::Contract(invalid_completion(
                        "device runtime returned an invalid completion readback byte count",
                    )));
                }
                output.extend_from_slice(&piece);
            }
            logical_cursor = segment_logical_end;
        }
        if output.len() != output_capacity || !self.current_descriptor_matches_snapshot() {
            state.fail_closed = true;
            self.fail_closed.store(true, Ordering::Release);
            return Err(LaneReadbackError::Contract(invalid_completion(
                "completion readback did not cover its exact logical output",
            )));
        }
        let timing = match timing_started {
            Some(started) => {
                let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).map_err(|_| {
                    LaneReadbackError::Contract(invalid_completion(
                        "completion readback host duration exceeds u64 nanoseconds",
                    ))
                })?;
                let bytes = u64::try_from(output.len()).map_err(|_| {
                    LaneReadbackError::Contract(invalid_completion(
                        "completion readback output length exceeds u64",
                    ))
                })?;
                DeviceTimingMeasurement::Measured(CompletionReadbackTiming::new(
                    elapsed_ns,
                    readback_calls,
                    bytes,
                ))
            }
            None => DeviceTimingMeasurement::NotRequested,
        };
        Ok(LaneReadback {
            bytes: output,
            timing,
        })
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
    fn submit_via(
        &mut self,
        submit: impl FnOnce(
            &R,
            &mut R::Stream,
        ) -> Result<R::Fence, super::DefinitelyNotSubmitted<R::Error>>,
    ) -> LaneSubmitOutcome<R::Fence, R::Error> {
        let runtime = self.lane.runtime.as_ref();
        let stream = &mut self.state.stream;
        let submission = catch_unwind(AssertUnwindSafe(|| submit(runtime, stream)));
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

    pub(crate) fn submit(
        &mut self,
        commands: super::DeviceCommandBatch<R::Command>,
    ) -> LaneSubmitOutcome<R::Fence, R::Error> {
        self.submit_via(|runtime, stream| runtime.submit(stream, commands))
    }

    pub(crate) fn submit_with_timing<S>(
        &mut self,
        commands: DeviceCommandBatch<R::Command>,
        timing_sink: &S,
    ) -> LaneSubmitOutcome<R::Fence, R::Error>
    where
        S: DeviceSubmissionTimingSink,
    {
        self.submit_via(|runtime, stream| runtime.submit_with_timing(stream, commands, timing_sink))
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

#[derive(Debug, PartialEq, Eq, Serialize)]
struct SubmittedOperationReceiptData {
    slot_id: CompletionSlotId,
    batch_identity: BatchOperationIdentity,
    participants: Vec<SubmittedOperationParticipantReceipt>,
    fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use = "physical submission evidence must be recorded"]
pub struct SubmittedOperationReceipt {
    data: Arc<SubmittedOperationReceiptData>,
}

impl Serialize for SubmittedOperationReceipt {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.data.as_ref().serialize(serializer)
    }
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct SubmittedOperationParticipantReceiptData {
    slot_id: CompletionSlotId,
    participant_index: u32,
    identity: ExecutionIdentityEnvelope,
    batch_submission_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use = "participant submission projections must remain linked to the physical batch"]
pub struct SubmittedOperationParticipantReceipt {
    data: Arc<SubmittedOperationParticipantReceiptData>,
}

impl SubmittedOperationParticipantReceipt {
    fn new(
        slot_id: CompletionSlotId,
        participant_index: u32,
        identity: ExecutionIdentityEnvelope,
        batch_submission_fingerprint: String,
    ) -> Self {
        Self {
            data: Arc::new(SubmittedOperationParticipantReceiptData {
                slot_id,
                participant_index,
                identity,
                batch_submission_fingerprint,
            }),
        }
    }
}

impl Serialize for SubmittedOperationParticipantReceipt {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.data.as_ref().serialize(serializer)
    }
}

impl SubmittedOperationReceipt {
    fn new(
        slot_id: CompletionSlotId,
        batch_identity: BatchOperationIdentity,
        participants: Vec<SubmittedOperationParticipantReceipt>,
        fingerprint: String,
    ) -> Self {
        Self {
            data: Arc::new(SubmittedOperationReceiptData {
                slot_id,
                batch_identity,
                participants,
                fingerprint,
            }),
        }
    }

    pub fn slot_id(&self) -> CompletionSlotId {
        self.data.slot_id
    }

    pub fn batch_identity(&self) -> &BatchOperationIdentity {
        &self.data.batch_identity
    }

    pub fn participants(&self) -> &[SubmittedOperationParticipantReceipt] {
        &self.data.participants
    }

    pub fn fingerprint(&self) -> &str {
        &self.data.fingerprint
    }
}

impl SubmittedOperationParticipantReceipt {
    pub fn slot_id(&self) -> CompletionSlotId {
        self.data.slot_id
    }

    pub fn participant_index(&self) -> u32 {
        self.data.participant_index
    }

    pub fn identity(&self) -> &ExecutionIdentityEnvelope {
        &self.data.identity
    }

    pub fn batch_submission_fingerprint(&self) -> &str {
        &self.data.batch_submission_fingerprint
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "status", content = "failure")]
pub enum OperationCompletionDisposition {
    Succeeded,
    FailedButQuiescent(Vec<IdentifiedFailure>),
    ContractFailedButQuiescent(QuiescentCompletionContractFailure),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "status", content = "failure")]
pub enum OperationParticipantCompletionDisposition {
    Succeeded,
    FailedButQuiescent(IdentifiedFailure),
    ContractFailedButQuiescent(QuiescentCompletionContractFailure),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[must_use = "participant completion projections must remain linked to the physical batch"]
pub struct OperationParticipantCompletionReceipt {
    submission: SubmittedOperationParticipantReceipt,
    disposition: OperationParticipantCompletionDisposition,
    batch_completion_fingerprint: String,
}

impl OperationParticipantCompletionReceipt {
    pub fn submission(&self) -> &SubmittedOperationParticipantReceipt {
        &self.submission
    }

    pub fn disposition(&self) -> &OperationParticipantCompletionDisposition {
        &self.disposition
    }

    pub fn batch_completion_fingerprint(&self) -> &str {
        &self.batch_completion_fingerprint
    }
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

/// Fence timing for one exact operation completion. Device execution and host
/// wait use different clocks and may overlap; consumers must not add them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CompletionFenceTiming {
    timing_mode: DeviceTimingMode,
    device_execution: DeviceTimingMeasurement<DeviceExecutionTiming>,
    blocking_wait_host_ns: DeviceTimingMeasurement<u64>,
}

impl CompletionFenceTiming {
    fn new(
        timing_mode: DeviceTimingMode,
        device_execution: DeviceTimingMeasurement<DeviceExecutionTiming>,
        blocking_wait_host_ns: DeviceTimingMeasurement<u64>,
    ) -> Self {
        let device_execution = match (timing_mode, device_execution) {
            (DeviceTimingMode::Off, _) => DeviceTimingMeasurement::NotRequested,
            (DeviceTimingMode::Completion, DeviceTimingMeasurement::NotRequested) => {
                DeviceTimingMeasurement::Unavailable(
                    DeviceTimingUnavailableReason::BackendUnsupported,
                )
            }
            (DeviceTimingMode::Completion, measurement) => measurement,
        };
        Self {
            timing_mode,
            device_execution,
            blocking_wait_host_ns,
        }
    }

    pub const fn device_execution(self) -> DeviceTimingMeasurement<DeviceExecutionTiming> {
        self.device_execution
    }

    pub const fn blocking_wait_host_ns(self) -> DeviceTimingMeasurement<u64> {
        self.blocking_wait_host_ns
    }

    pub const fn timing_mode(self) -> DeviceTimingMode {
        self.timing_mode
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CompletionReadbackTiming {
    host_elapsed_ns: u64,
    calls: u32,
    bytes: u64,
}

impl CompletionReadbackTiming {
    const fn new(host_elapsed_ns: u64, calls: u32, bytes: u64) -> Self {
        Self {
            host_elapsed_ns,
            calls,
            bytes,
        }
    }

    pub const fn host_elapsed_ns(self) -> u64 {
        self.host_elapsed_ns
    }

    pub const fn calls(self) -> u32 {
        self.calls
    }

    pub const fn bytes(self) -> u64 {
        self.bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[must_use = "a terminal completion receipt releases one exact invocation"]
pub struct OperationCompletionReceipt {
    submission: SubmittedOperationReceipt,
    disposition: OperationCompletionDisposition,
    fence_timing: CompletionFenceTiming,
    participants: Vec<OperationParticipantCompletionReceipt>,
    fingerprint: String,
}

impl OperationCompletionReceipt {
    fn new(
        submission: SubmittedOperationReceipt,
        disposition: OperationCompletionDisposition,
        fence_timing: CompletionFenceTiming,
    ) -> Result<Self, VNextError> {
        let participant_dispositions = match &disposition {
            OperationCompletionDisposition::Succeeded => submission
                .participants()
                .iter()
                .map(|_| OperationParticipantCompletionDisposition::Succeeded)
                .collect::<Vec<_>>(),
            OperationCompletionDisposition::FailedButQuiescent(failures) => {
                if failures.len() != submission.participants().len()
                    || failures
                        .iter()
                        .zip(submission.participants())
                        .any(|(failure, participant)| failure.identity() != participant.identity())
                {
                    return Err(invalid_completion(
                        "batch completion failures differ from participant submission projections",
                    ));
                }
                failures
                    .iter()
                    .cloned()
                    .map(OperationParticipantCompletionDisposition::FailedButQuiescent)
                    .collect()
            }
            OperationCompletionDisposition::ContractFailedButQuiescent(failure) => submission
                .participants()
                .iter()
                .map(|_| {
                    OperationParticipantCompletionDisposition::ContractFailedButQuiescent(
                        failure.clone(),
                    )
                })
                .collect(),
        };
        #[derive(Serialize)]
        struct CompletionFingerprintInput<'a> {
            domain: &'static str,
            submission_fingerprint: &'a str,
            disposition: &'a OperationCompletionDisposition,
        }
        let fingerprint = canonical_completion_fingerprint(&CompletionFingerprintInput {
            domain: "ferrum.runtime-vnext.batch-operation-completion.v1",
            submission_fingerprint: submission.fingerprint(),
            disposition: &disposition,
        });
        let participants = submission
            .participants()
            .iter()
            .cloned()
            .zip(participant_dispositions)
            .map(
                |(submission, disposition)| OperationParticipantCompletionReceipt {
                    submission,
                    disposition,
                    batch_completion_fingerprint: fingerprint.clone(),
                },
            )
            .collect();
        Ok(Self {
            submission,
            disposition,
            fence_timing,
            participants,
            fingerprint,
        })
    }

    pub fn submission(&self) -> &SubmittedOperationReceipt {
        &self.submission
    }

    pub fn disposition(&self) -> &OperationCompletionDisposition {
        &self.disposition
    }

    pub const fn fence_timing(&self) -> CompletionFenceTiming {
        self.fence_timing
    }

    pub fn participants(&self) -> &[OperationParticipantCompletionReceipt] {
        &self.participants
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
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
    batch_identity: BatchOperationIdentity,
    submission: Option<SubmittedOperationReceipt>,
    cause: CompletionRecoveryCause,
    had_submission_fence: bool,
}

impl CompletionDrainReceipt {
    pub const fn slot_id(&self) -> CompletionSlotId {
        self.slot_id
    }

    pub fn batch_identity(&self) -> &BatchOperationIdentity {
        &self.batch_identity
    }

    pub fn submission(&self) -> Option<&SubmittedOperationReceipt> {
        self.submission.as_ref()
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
    batch_identity: BatchOperationIdentity,
    submission: Option<SubmittedOperationReceipt>,
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
            && self.batch_identity == other.batch_identity
            && self.submission == other.submission
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

    pub fn batch_identity(&self) -> &BatchOperationIdentity {
        &self.batch_identity
    }

    pub fn submission(&self) -> Option<&SubmittedOperationReceipt> {
        self.submission.as_ref()
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

enum CompletionResourceLease<R: DeviceRuntime> {
    Invocation(InvocationResourceLease<R>),
    Wave(PreparedStepSubmissionWave<R>),
}

impl<R: DeviceRuntime> CompletionResourceLease<R> {
    fn runtime(&self) -> &Arc<R> {
        match self {
            Self::Invocation(invocation) => invocation.runtime(),
            Self::Wave(wave) => wave.runtime(),
        }
    }

    fn deferred_cleanup_domain(&self) -> DeferredDeviceCleanupDomainId {
        match self {
            Self::Invocation(invocation) => invocation.deferred_cleanup_domain(),
            Self::Wave(wave) => wave.deferred_cleanup_domain(),
        }
    }

    fn mark_submission_fence_installed(&mut self) -> Result<(), VNextError> {
        match self {
            Self::Invocation(invocation) => invocation.mark_submission_fence_installed(),
            Self::Wave(wave) => wave.mark_submission_fence_installed(),
        }
    }

    fn encode_backing_initializations(
        &self,
        runtime: &R,
        commands: &mut DeviceCommandBatch<R::Command>,
    ) -> Result<usize, BackingInitializationEncodeError<R::Error>> {
        match self {
            Self::Invocation(invocation) => {
                invocation.encode_backing_initializations(runtime, commands)
            }
            Self::Wave(wave) => wave.encode_backing_initializations(runtime, commands),
        }
    }

    fn mark_submission_indeterminate(&mut self) {
        match self {
            Self::Invocation(invocation) => invocation.mark_submission_indeterminate(),
            Self::Wave(wave) => wave.mark_submission_indeterminate(),
        }
    }

    fn finish_backing_initializations(&mut self, succeeded: bool) -> Result<(), VNextError> {
        match self {
            Self::Invocation(invocation) => invocation.finish_backing_initializations(succeeded),
            Self::Wave(wave) => wave.finish_backing_initializations(succeeded),
        }
    }

    fn backing_view(
        &self,
        node_id: &NodeId,
        participant_index: u32,
        resource_id: &ResourceId,
    ) -> Result<LogicalBackingBufferView<'_, R::Buffer>, VNextError> {
        let participant_index = usize::try_from(participant_index).map_err(|_| {
            invalid_completion("completion readback participant index exceeds host address space")
        })?;
        match self {
            Self::Invocation(invocation) => {
                if invocation.node_id() != node_id {
                    return Err(invalid_completion(
                        "completion readback node differs from its invocation",
                    ));
                }
                let is_shared = invocation
                    .backing_slices()
                    .iter()
                    .chain(invocation.step_resources().backing_slices())
                    .any(|authority| authority.resource_id() == resource_id);
                if is_shared {
                    return invocation.backing_view(resource_id);
                }
                let participant = invocation
                    .participants()
                    .nth(participant_index)
                    .ok_or_else(|| {
                        invalid_completion(
                            "completion readback participant is absent from its invocation",
                        )
                    })?;
                invocation.step_resources().participant_backing_view(
                    BatchParticipantAuthority::new(
                        participant.sequence_authority(),
                        participant.request_authority(),
                    ),
                    resource_id,
                )
            }
            Self::Wave(wave) => {
                let node_index = wave
                    .nodes()
                    .iter()
                    .position(|node| node.node_id() == node_id)
                    .ok_or_else(|| {
                        invalid_completion("completion readback node is absent from its wave")
                    })?;
                let is_shared = wave
                    .claimed_backing()
                    .backing_slices()
                    .iter()
                    .chain(wave.step_resources().backing_slices())
                    .any(|authority| authority.resource_id() == resource_id);
                if is_shared {
                    return wave.backing_view(node_index, resource_id);
                }
                let participant = wave.nodes()[node_index]
                    .participants()
                    .nth(participant_index)
                    .ok_or_else(|| {
                        invalid_completion(
                            "completion readback participant is absent from its wave node",
                        )
                    })?;
                wave.step_resources().participant_backing_view(
                    BatchParticipantAuthority::new(
                        participant.sequence_authority(),
                        participant.request_authority(),
                    ),
                    resource_id,
                )
            }
        }
    }
}

enum CompletionRecord<R: DeviceRuntime> {
    Reserved,
    InFlight {
        resources: CompletionResourceLease<R>,
        lane: Arc<ExecutionLane<R>>,
        fence: R::Fence,
        batch_identity: BatchOperationIdentity,
        receipt: SubmittedOperationReceipt,
        timing_mode: DeviceTimingMode,
        recovery_state: CompletionRecoveryState,
    },
    SubmissionIndeterminate {
        resources: CompletionResourceLease<R>,
        lane: Arc<ExecutionLane<R>>,
        batch_identity: BatchOperationIdentity,
    },
    Quarantined {
        ownership: CompletionQuarantineOwnership<R>,
        receipt: CompletionQuarantineReceipt,
    },
    Reaped,
}

enum BoundCompletionObservation<T> {
    Pending,
    Terminal {
        completion: OperationCompletionReceipt,
        terminal: T,
    },
    Indeterminate(Vec<IdentifiedFailure>),
    SubmissionIndeterminate,
    ObservationPanicked,
    Quarantined(CompletionQuarantineReceipt),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompletionRecoveryState {
    Unobserved,
    QueryIndeterminate,
    DrainEligible(CompletionRecoveryCause),
}

enum CompletionQuarantineOwnership<R: DeviceRuntime> {
    InFlight {
        resources: CompletionResourceLease<R>,
        lane: Arc<ExecutionLane<R>>,
        fence: R::Fence,
        batch_identity: BatchOperationIdentity,
        submission: SubmittedOperationReceipt,
    },
    SubmissionIndeterminate {
        resources: CompletionResourceLease<R>,
        lane: Arc<ExecutionLane<R>>,
        batch_identity: BatchOperationIdentity,
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
            Self::InFlight { resources, .. } | Self::SubmissionIndeterminate { resources, .. } => {
                resources.deferred_cleanup_domain()
            }
        }
    }
}

impl<R: DeviceRuntime> CompletionRecord<R> {
    fn deferred_cleanup_domain(&self) -> Option<DeferredDeviceCleanupDomainId> {
        match self {
            Self::InFlight { resources, .. } | Self::SubmissionIndeterminate { resources, .. } => {
                Some(resources.deferred_cleanup_domain())
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
        batch_identity: BatchOperationIdentity,
    ) -> Result<CompletionReservation<R>, VNextError> {
        Self::reserve_resources(
            reaper,
            CompletionResourceLease::Invocation(invocation),
            lane,
            batch_identity,
        )
    }

    pub(crate) fn reserve_wave(
        reaper: &Arc<Self>,
        wave: PreparedStepSubmissionWave<R>,
        lane: Arc<ExecutionLane<R>>,
        batch_identity: BatchOperationIdentity,
    ) -> Result<CompletionReservation<R>, VNextError> {
        Self::reserve_resources(
            reaper,
            CompletionResourceLease::Wave(wave),
            lane,
            batch_identity,
        )
    }

    fn reserve_resources(
        reaper: &Arc<Self>,
        resources: CompletionResourceLease<R>,
        lane: Arc<ExecutionLane<R>>,
        batch_identity: BatchOperationIdentity,
    ) -> Result<CompletionReservation<R>, VNextError> {
        if !Arc::ptr_eq(resources.runtime(), lane.runtime_arc()) {
            return Err(invalid_completion(
                "completion lane runtime is not the submission resource runtime instance",
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
        #[derive(Serialize)]
        struct SubmissionFingerprintInput<'a> {
            domain: &'static str,
            slot_id: CompletionSlotId,
            batch_identity_fingerprint: &'a str,
        }
        let fingerprint = canonical_completion_fingerprint(&SubmissionFingerprintInput {
            domain: "ferrum.runtime-vnext.batch-operation-submission.v1",
            slot_id,
            batch_identity_fingerprint: batch_identity.fingerprint(),
        });
        let participant_receipts = batch_identity
            .participants()
            .iter()
            .map(|participant| {
                SubmittedOperationParticipantReceipt::new(
                    slot_id,
                    participant.participant_index(),
                    participant.identity().clone(),
                    fingerprint.clone(),
                )
            })
            .collect();
        let receipt = SubmittedOperationReceipt::new(
            slot_id,
            batch_identity.clone(),
            participant_receipts,
            fingerprint,
        );
        let record_receipt = receipt.clone();
        Ok(CompletionReservation {
            reaper: Arc::clone(reaper),
            record,
            slot_id,
            resources: Some(resources),
            lane: Some(lane),
            batch_identity: Some(batch_identity),
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
        Self::map_plain_observation(self.observe_bound_with(slot_id, false, |_, _, _, _, _| ())?)
    }

    pub(crate) fn wait_bound(
        &self,
        slot_id: CompletionSlotId,
    ) -> Result<CompletionObservation, VNextError> {
        Self::map_plain_observation(self.observe_bound_with(slot_id, true, |_, _, _, _, _| ())?)
    }

    pub(crate) fn wait_bound_with_readback(
        &self,
        slot_id: CompletionSlotId,
        request: CompletionReadbackRequest,
    ) -> Result<CompletionReadbackObservation, VNextError> {
        let observation = self.observe_bound_with(
            slot_id,
            true,
            |resources, lane, batch_identity, disposition, timing_mode| {
                attempt_completion_readback(
                    resources,
                    lane,
                    batch_identity,
                    disposition,
                    timing_mode,
                    request,
                )
            },
        )?;
        Ok(match observation {
            BoundCompletionObservation::Pending => CompletionReadbackObservation::Pending,
            BoundCompletionObservation::Terminal {
                completion,
                terminal,
            } => CompletionReadbackObservation::Terminal(CompletionReadbackReceipt::new(
                completion, terminal,
            )),
            BoundCompletionObservation::Indeterminate(failures) => {
                CompletionReadbackObservation::Indeterminate(failures)
            }
            BoundCompletionObservation::SubmissionIndeterminate => {
                CompletionReadbackObservation::SubmissionIndeterminate
            }
            BoundCompletionObservation::ObservationPanicked => {
                CompletionReadbackObservation::ObservationPanicked
            }
            BoundCompletionObservation::Quarantined(receipt) => {
                CompletionReadbackObservation::Quarantined(receipt)
            }
        })
    }

    pub(crate) fn wait_bound_with_readbacks(
        &self,
        slot_id: CompletionSlotId,
        request: CompletionReadbackBatchRequest,
    ) -> Result<CompletionReadbackBatchObservation, VNextError> {
        self.validate_bound_readback_batch(slot_id, &request)?;
        let observation = self.observe_bound_with(
            slot_id,
            true,
            |resources, lane, batch_identity, disposition, timing_mode| {
                attempt_completion_readbacks(
                    resources,
                    lane,
                    batch_identity,
                    disposition,
                    timing_mode,
                    request,
                )
            },
        )?;
        Ok(match observation {
            BoundCompletionObservation::Pending => CompletionReadbackBatchObservation::Pending,
            BoundCompletionObservation::Terminal {
                completion,
                terminal,
            } => CompletionReadbackBatchObservation::Terminal(CompletionReadbackBatchReceipt::new(
                completion, terminal,
            )),
            BoundCompletionObservation::Indeterminate(failures) => {
                CompletionReadbackBatchObservation::Indeterminate(failures)
            }
            BoundCompletionObservation::SubmissionIndeterminate => {
                CompletionReadbackBatchObservation::SubmissionIndeterminate
            }
            BoundCompletionObservation::ObservationPanicked => {
                CompletionReadbackBatchObservation::ObservationPanicked
            }
            BoundCompletionObservation::Quarantined(receipt) => {
                CompletionReadbackBatchObservation::Quarantined(receipt)
            }
        })
    }

    pub(crate) fn wait_bound_with_readback_collection(
        &self,
        slot_id: CompletionSlotId,
        request: CompletionReadbackCollectionRequest,
    ) -> Result<CompletionReadbackCollectionObservation, VNextError> {
        self.validate_bound_readback_collection(slot_id, &request)?;
        let observation = self.observe_bound_with(
            slot_id,
            true,
            |resources, lane, batch_identity, disposition, timing_mode| {
                attempt_completion_readback_collection(
                    resources,
                    lane,
                    batch_identity,
                    disposition,
                    timing_mode,
                    request,
                )
            },
        )?;
        Ok(match observation {
            BoundCompletionObservation::Pending => CompletionReadbackBatchObservation::Pending,
            BoundCompletionObservation::Terminal {
                completion,
                terminal,
            } => CompletionReadbackBatchObservation::Terminal(CompletionReadbackBatchReceipt::new(
                completion, terminal,
            )),
            BoundCompletionObservation::Indeterminate(failures) => {
                CompletionReadbackBatchObservation::Indeterminate(failures)
            }
            BoundCompletionObservation::SubmissionIndeterminate => {
                CompletionReadbackBatchObservation::SubmissionIndeterminate
            }
            BoundCompletionObservation::ObservationPanicked => {
                CompletionReadbackBatchObservation::ObservationPanicked
            }
            BoundCompletionObservation::Quarantined(receipt) => {
                CompletionReadbackBatchObservation::Quarantined(receipt)
            }
        })
    }

    fn validate_bound_readback_batch(
        &self,
        slot_id: CompletionSlotId,
        request: &CompletionReadbackBatchRequest,
    ) -> Result<(), VNextError> {
        let record = self.lookup(slot_id)?;
        let guard = record
            .lock()
            .map_err(|_| invalid_completion("completion slot mutex is poisoned"))?;
        match &*guard {
            CompletionRecord::InFlight { batch_identity, .. }
            | CompletionRecord::SubmissionIndeterminate { batch_identity, .. } => {
                request.validate_for(batch_identity)
            }
            CompletionRecord::Reserved => Err(invalid_completion(
                "completion slot has not reached submission",
            )),
            CompletionRecord::Quarantined { .. } => Ok(()),
            CompletionRecord::Reaped => {
                Err(invalid_completion("completion slot is already reaped"))
            }
        }
    }

    fn validate_bound_readback_collection(
        &self,
        slot_id: CompletionSlotId,
        request: &CompletionReadbackCollectionRequest,
    ) -> Result<(), VNextError> {
        let record = self.lookup(slot_id)?;
        let guard = record
            .lock()
            .map_err(|_| invalid_completion("completion slot mutex is poisoned"))?;
        match &*guard {
            CompletionRecord::InFlight { batch_identity, .. }
            | CompletionRecord::SubmissionIndeterminate { batch_identity, .. } => {
                request.validate_for(batch_identity)
            }
            CompletionRecord::Reserved => Err(invalid_completion(
                "completion slot has not reached submission",
            )),
            CompletionRecord::Quarantined { .. } => Ok(()),
            CompletionRecord::Reaped => {
                Err(invalid_completion("completion slot is already reaped"))
            }
        }
    }

    fn map_plain_observation(
        observation: BoundCompletionObservation<()>,
    ) -> Result<CompletionObservation, VNextError> {
        Ok(match observation {
            BoundCompletionObservation::Pending => CompletionObservation::Pending,
            BoundCompletionObservation::Terminal { completion, .. } => {
                CompletionObservation::Terminal(completion)
            }
            BoundCompletionObservation::Indeterminate(failures) => {
                CompletionObservation::Indeterminate(failures)
            }
            BoundCompletionObservation::SubmissionIndeterminate => {
                CompletionObservation::SubmissionIndeterminate
            }
            BoundCompletionObservation::ObservationPanicked => {
                CompletionObservation::ObservationPanicked
            }
            BoundCompletionObservation::Quarantined(receipt) => {
                CompletionObservation::Quarantined(receipt)
            }
        })
    }

    fn observe_bound_with<T, F>(
        &self,
        slot_id: CompletionSlotId,
        blocking: bool,
        terminal_action: F,
    ) -> Result<BoundCompletionObservation<T>, VNextError>
    where
        F: FnOnce(
            &CompletionResourceLease<R>,
            &Arc<ExecutionLane<R>>,
            &BatchOperationIdentity,
            &OperationCompletionDisposition,
            DeviceTimingMode,
        ) -> T,
    {
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
                batch_identity,
                timing_mode,
                ..
            } if blocking => {
                let wait_started =
                    (*timing_mode == DeviceTimingMode::Completion).then(Instant::now);
                match catch_unwind(AssertUnwindSafe(|| lane.wait_fence(fence))) {
                    Ok(Ok(terminal)) => {
                        let wait_timing = match wait_started {
                            Some(started) => u64::try_from(started.elapsed().as_nanos()).map_or(
                                DeviceTimingMeasurement::Unavailable(
                                    DeviceTimingUnavailableReason::DurationOverflow,
                                ),
                                DeviceTimingMeasurement::Measured,
                            ),
                            None => DeviceTimingMeasurement::NotRequested,
                        };
                        catch_unwind(AssertUnwindSafe(|| {
                            terminal_observation(
                                lane,
                                batch_identity,
                                terminal,
                                *timing_mode,
                                wait_timing,
                            )
                        }))
                        .unwrap_or(FenceObservation::ObservationPanicked)
                    }
                    Ok(Err(indeterminate)) => catch_unwind(AssertUnwindSafe(|| {
                        classify_batch_device_error(
                            lane.runtime(),
                            batch_identity,
                            indeterminate.error(),
                        )
                    }))
                    .map_or(
                        FenceObservation::ObservationPanicked,
                        |classified| match classified {
                            Ok(failure) => FenceObservation::Indeterminate(failure),
                            Err(error) => FenceObservation::ContractIndeterminate(error),
                        },
                    ),
                    Err(_) => FenceObservation::ObservationPanicked,
                }
            }
            CompletionRecord::InFlight {
                lane,
                fence,
                batch_identity,
                timing_mode,
                ..
            } => match catch_unwind(AssertUnwindSafe(|| lane.query_fence(fence))) {
                Ok(FenceQuery::Pending) => FenceObservation::Pending,
                Ok(FenceQuery::Terminal(terminal)) => catch_unwind(AssertUnwindSafe(|| {
                    terminal_observation(
                        lane,
                        batch_identity,
                        terminal,
                        *timing_mode,
                        if *timing_mode == DeviceTimingMode::Completion {
                            DeviceTimingMeasurement::Measured(0)
                        } else {
                            DeviceTimingMeasurement::NotRequested
                        },
                    )
                }))
                .unwrap_or(FenceObservation::ObservationPanicked),
                Ok(FenceQuery::Indeterminate(error)) => catch_unwind(AssertUnwindSafe(|| {
                    classify_batch_device_error(lane.runtime(), batch_identity, &error)
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
        if let FenceObservation::Terminal(mut disposition, fence_timing) = observation {
            let old = std::mem::replace(&mut *guard, CompletionRecord::Reaped);
            let CompletionRecord::InFlight {
                resources,
                lane,
                batch_identity,
                receipt,
                timing_mode,
                ..
            } = old
            else {
                unreachable!("terminal observation came from an in-flight record")
            };
            let terminal = terminal_action(
                &resources,
                &lane,
                &batch_identity,
                &disposition,
                timing_mode,
            );
            if let Err(failure) = lane.finish_one_terminal() {
                disposition = OperationCompletionDisposition::ContractFailedButQuiescent(failure);
            }
            let initialization_succeeded =
                matches!(disposition, OperationCompletionDisposition::Succeeded);
            let mut resources = resources;
            if let Err(error) = resources.finish_backing_initializations(initialization_succeeded) {
                disposition = OperationCompletionDisposition::ContractFailedButQuiescent(
                    QuiescentCompletionContractFailure::new(format!(
                        "backing initialization terminal transition failed: {error}"
                    )),
                );
            }
            drop(guard);
            self.remove_exact(slot_id, &record);
            return OperationCompletionReceipt::new(receipt, disposition, fence_timing).map(
                |completion| BoundCompletionObservation::Terminal {
                    completion,
                    terminal,
                },
            );
        }
        Ok(match observation {
            FenceObservation::Pending => BoundCompletionObservation::Pending,
            FenceObservation::Indeterminate(failure) => {
                BoundCompletionObservation::Indeterminate(failure)
            }
            FenceObservation::SubmissionIndeterminate => {
                BoundCompletionObservation::SubmissionIndeterminate
            }
            FenceObservation::ObservationPanicked => {
                BoundCompletionObservation::ObservationPanicked
            }
            FenceObservation::ContractIndeterminate(error) => return Err(error),
            FenceObservation::Quarantined(receipt) => {
                BoundCompletionObservation::Quarantined(receipt)
            }
            FenceObservation::Terminal(_, _) => {
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
        let (lane, batch_identity, submission, cause, had_submission_fence) = match &*guard {
            CompletionRecord::InFlight {
                lane,
                batch_identity,
                receipt,
                recovery_state: CompletionRecoveryState::DrainEligible(cause),
                ..
            } => (
                Arc::clone(lane),
                batch_identity.clone(),
                Some(receipt.clone()),
                *cause,
                true,
            ),
            CompletionRecord::SubmissionIndeterminate {
                lane,
                batch_identity,
                ..
            } => (
                Arc::clone(lane),
                batch_identity.clone(),
                None,
                CompletionRecoveryCause::SubmissionIndeterminate,
                false,
            ),
            CompletionRecord::Quarantined { ownership, receipt } => (
                Arc::clone(ownership.lane()),
                receipt.batch_identity.clone(),
                receipt.submission.clone(),
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
                    resources,
                    lane,
                    fence,
                    batch_identity,
                    receipt,
                    ..
                } => CompletionQuarantineOwnership::InFlight {
                    resources,
                    lane,
                    fence,
                    batch_identity,
                    submission: receipt,
                },
                CompletionRecord::SubmissionIndeterminate {
                    resources,
                    lane,
                    batch_identity,
                } => CompletionQuarantineOwnership::SubmissionIndeterminate {
                    resources,
                    lane,
                    batch_identity,
                },
                _ => unreachable!("recovery source was validated before lane drain"),
            };
            let receipt = CompletionQuarantineReceipt {
                slot_id,
                batch_identity,
                submission,
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
            batch_identity,
            submission,
            cause,
            had_submission_fence,
        }))
    }
}

enum FenceObservation {
    Pending,
    Terminal(OperationCompletionDisposition, CompletionFenceTiming),
    Indeterminate(Vec<IdentifiedFailure>),
    ContractIndeterminate(VNextError),
    SubmissionIndeterminate,
    ObservationPanicked,
    Quarantined(CompletionQuarantineReceipt),
}

fn terminal_observation<R: DeviceRuntime>(
    lane: &ExecutionLane<R>,
    batch_identity: &BatchOperationIdentity,
    terminal: DeviceTerminalReceipt<R::Error>,
    timing_mode: DeviceTimingMode,
    blocking_wait_host_ns: DeviceTimingMeasurement<u64>,
) -> FenceObservation {
    let (terminal, device_execution) = terminal.into_parts();
    let timing = CompletionFenceTiming::new(timing_mode, device_execution, blocking_wait_host_ns);
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
            timing,
        );
    }
    FenceObservation::Terminal(
        match terminal {
            DeviceTerminal::Succeeded => OperationCompletionDisposition::Succeeded,
            DeviceTerminal::FailedButQuiescent(error) => {
                match classify_batch_device_error(lane.runtime(), batch_identity, &error) {
                    Ok(failures) => OperationCompletionDisposition::FailedButQuiescent(failures),
                    Err(error) => OperationCompletionDisposition::ContractFailedButQuiescent(
                        QuiescentCompletionContractFailure::new(error.to_string()),
                    ),
                }
            }
        },
        timing,
    )
}

fn classify_batch_device_error<R: DeviceRuntime>(
    runtime: &R,
    batch_identity: &BatchOperationIdentity,
    error: &R::Error,
) -> Result<Vec<IdentifiedFailure>, VNextError> {
    batch_identity
        .participants()
        .iter()
        .map(|participant| classify_device_error(runtime, participant.identity().clone(), error))
        .collect()
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[must_use = "a completion readback request identifies one exact logical activation range"]
pub struct CompletionReadbackRequest {
    node_id: NodeId,
    participant_index: u32,
    resource_id: ResourceId,
    logical_offset_bytes: u64,
    output_layout: HostTransferLayout,
}

impl CompletionReadbackRequest {
    pub fn new(
        node_id: NodeId,
        participant_index: u32,
        resource_id: ResourceId,
        logical_offset_bytes: u64,
        output_layout: HostTransferLayout,
    ) -> Result<Self, VNextError> {
        let output_bytes = output_layout.byte_len()?;
        if logical_offset_bytes.checked_add(output_bytes).is_none() {
            return Err(invalid_completion(
                "completion readback logical range overflows u64",
            ));
        }
        Ok(Self {
            node_id,
            participant_index,
            resource_id,
            logical_offset_bytes,
            output_layout,
        })
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }

    pub const fn participant_index(&self) -> u32 {
        self.participant_index
    }

    pub const fn logical_offset_bytes(&self) -> u64 {
        self.logical_offset_bytes
    }

    pub const fn output_layout(&self) -> HostTransferLayout {
        self.output_layout
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[must_use = "a completion readback batch must cover one exact plan-node participant set"]
pub struct CompletionReadbackBatchRequest {
    requests: Vec<CompletionReadbackRequest>,
}

impl CompletionReadbackBatchRequest {
    pub fn new(requests: Vec<CompletionReadbackRequest>) -> Result<Self, VNextError> {
        let Some(first) = requests.first() else {
            return Err(invalid_completion(
                "completion readback batch cannot be empty",
            ));
        };
        if requests.iter().enumerate().any(|(index, request)| {
            usize::try_from(request.participant_index()).ok() != Some(index)
                || request.node_id() != first.node_id()
                || request.resource_id() != first.resource_id()
                || request.output_layout() != first.output_layout()
        }) {
            return Err(invalid_completion(
                "completion readback batch must use canonical participant order and one node/resource/layout",
            ));
        }
        Ok(Self { requests })
    }

    pub fn requests(&self) -> &[CompletionReadbackRequest] {
        &self.requests
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    fn validate_for(&self, batch_identity: &BatchOperationIdentity) -> Result<(), VNextError> {
        let node_id = self.requests[0].node_id();
        let node = batch_identity
            .nodes()
            .iter()
            .find(|node| node.node_id() == node_id)
            .ok_or_else(|| {
                invalid_completion("completion readback batch node is absent from its submission")
            })?;
        if node.participants().len() != self.requests.len() {
            return Err(invalid_completion(
                "completion readback batch must cover every submitted node participant exactly once",
            ));
        }
        Ok(())
    }

    fn into_requests(self) -> Vec<CompletionReadbackRequest> {
        self.requests
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[must_use = "successful completion output bytes are exact readback evidence"]
pub struct CompletionReadbackOutput {
    request: CompletionReadbackRequest,
    bytes: Vec<u8>,
    sha256: String,
    #[serde(skip)]
    timing: DeviceTimingMeasurement<CompletionReadbackTiming>,
}

impl CompletionReadbackOutput {
    fn new(request: CompletionReadbackRequest, readback: LaneReadback) -> Result<Self, VNextError> {
        request.output_layout.validate_bytes(readback.bytes.len())?;
        let sha256 = format!("{:x}", Sha256::digest(&readback.bytes));
        Ok(Self {
            request,
            bytes: readback.bytes,
            sha256,
            timing: readback.timing,
        })
    }

    pub fn request(&self) -> &CompletionReadbackRequest {
        &self.request
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub const fn timing(&self) -> DeviceTimingMeasurement<CompletionReadbackTiming> {
        self.timing
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "status", content = "detail")]
pub enum CompletionReadbackDisposition {
    Succeeded(CompletionReadbackOutput),
    NotAttempted(CompletionReadbackRequest),
    FailedButQuiescent {
        request: CompletionReadbackRequest,
        failures: Vec<IdentifiedFailure>,
    },
    ContractFailedButQuiescent {
        request: CompletionReadbackRequest,
        failure: QuiescentCompletionContractFailure,
    },
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", tag = "status", content = "detail")]
enum CompletionReadbackDispositionFingerprint<'a> {
    Succeeded {
        request: &'a CompletionReadbackRequest,
        output_sha256: &'a str,
    },
    NotAttempted {
        request: &'a CompletionReadbackRequest,
    },
    FailedButQuiescent {
        request: &'a CompletionReadbackRequest,
        failures: &'a [IdentifiedFailure],
    },
    ContractFailedButQuiescent {
        request: &'a CompletionReadbackRequest,
        failure: &'a str,
    },
}

impl<'a> From<&'a CompletionReadbackDisposition> for CompletionReadbackDispositionFingerprint<'a> {
    fn from(disposition: &'a CompletionReadbackDisposition) -> Self {
        match disposition {
            CompletionReadbackDisposition::Succeeded(output) => Self::Succeeded {
                request: output.request(),
                output_sha256: output.sha256(),
            },
            CompletionReadbackDisposition::NotAttempted(request) => Self::NotAttempted { request },
            CompletionReadbackDisposition::FailedButQuiescent { request, failures } => {
                Self::FailedButQuiescent { request, failures }
            }
            CompletionReadbackDisposition::ContractFailedButQuiescent { request, failure } => {
                Self::ContractFailedButQuiescent {
                    request,
                    failure: failure.reason(),
                }
            }
        }
    }
}

struct CompletionReadbackDispositionFingerprints<'a>(&'a [CompletionReadbackDisposition]);

impl Serialize for CompletionReadbackDispositionFingerprints<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for disposition in self.0 {
            sequence
                .serialize_element(&CompletionReadbackDispositionFingerprint::from(disposition))?;
        }
        sequence.end()
    }
}

fn completion_readback_batch_fingerprint(
    completion_fingerprint: &str,
    dispositions: &[CompletionReadbackDisposition],
) -> String {
    #[derive(Serialize)]
    struct FingerprintInput<'a> {
        domain: &'static str,
        completion_fingerprint: &'a str,
        dispositions: CompletionReadbackDispositionFingerprints<'a>,
    }
    canonical_completion_fingerprint(&FingerprintInput {
        domain: "ferrum.runtime-vnext.completion-readback-batch.v2",
        completion_fingerprint,
        dispositions: CompletionReadbackDispositionFingerprints(dispositions),
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[must_use = "a terminal readback receipt couples output evidence to its exact completion"]
pub struct CompletionReadbackReceipt {
    completion: OperationCompletionReceipt,
    disposition: CompletionReadbackDisposition,
    readback_timing: Option<DeviceTimingMeasurement<CompletionReadbackTiming>>,
    fingerprint: String,
}

impl CompletionReadbackReceipt {
    fn new(
        completion: OperationCompletionReceipt,
        disposition: CompletionReadbackDisposition,
    ) -> Self {
        let readback_timing = (completion.fence_timing().timing_mode()
            == DeviceTimingMode::Completion)
            .then(|| readback_timing_for_disposition(&disposition));
        #[derive(Serialize)]
        struct FingerprintInput<'a> {
            domain: &'static str,
            completion_fingerprint: &'a str,
            request: &'a CompletionReadbackRequest,
            output_sha256: Option<&'a str>,
            failures: Option<&'a [IdentifiedFailure]>,
            contract_failure: Option<&'a str>,
        }
        let (request, output_sha256, failures, contract_failure) = match &disposition {
            CompletionReadbackDisposition::Succeeded(output) => {
                (output.request(), Some(output.sha256()), None, None)
            }
            CompletionReadbackDisposition::NotAttempted(request) => (request, None, None, None),
            CompletionReadbackDisposition::FailedButQuiescent { request, failures } => {
                (request, None, Some(failures.as_slice()), None)
            }
            CompletionReadbackDisposition::ContractFailedButQuiescent { request, failure } => {
                (request, None, None, Some(failure.reason()))
            }
        };
        let fingerprint = canonical_completion_fingerprint(&FingerprintInput {
            domain: "ferrum.runtime-vnext.completion-readback.v1",
            completion_fingerprint: completion.fingerprint(),
            request,
            output_sha256,
            failures,
            contract_failure,
        });
        Self {
            completion,
            disposition,
            readback_timing,
            fingerprint,
        }
    }

    pub fn completion(&self) -> &OperationCompletionReceipt {
        &self.completion
    }

    pub fn disposition(&self) -> &CompletionReadbackDisposition {
        &self.disposition
    }

    pub const fn readback_timing(
        &self,
    ) -> Option<DeviceTimingMeasurement<CompletionReadbackTiming>> {
        self.readback_timing
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[must_use = "a terminal batch readback receipt owns all participant output evidence"]
pub struct CompletionReadbackBatchReceipt {
    completion: OperationCompletionReceipt,
    dispositions: Vec<CompletionReadbackDisposition>,
    readback_timings: Option<Vec<DeviceTimingMeasurement<CompletionReadbackTiming>>>,
    fingerprint: String,
}

impl CompletionReadbackBatchReceipt {
    fn new(
        completion: OperationCompletionReceipt,
        dispositions: Vec<CompletionReadbackDisposition>,
    ) -> Self {
        let readback_timings =
            (completion.fence_timing().timing_mode() == DeviceTimingMode::Completion).then(|| {
                dispositions
                    .iter()
                    .map(readback_timing_for_disposition)
                    .collect()
            });
        let fingerprint =
            completion_readback_batch_fingerprint(completion.fingerprint(), &dispositions);
        Self {
            completion,
            dispositions,
            readback_timings,
            fingerprint,
        }
    }

    pub fn completion(&self) -> &OperationCompletionReceipt {
        &self.completion
    }

    pub fn dispositions(&self) -> &[CompletionReadbackDisposition] {
        &self.dispositions
    }

    pub fn readback_timings(&self) -> Option<&[DeviceTimingMeasurement<CompletionReadbackTiming>]> {
        self.readback_timings.as_deref()
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
}

fn readback_timing_for_disposition(
    disposition: &CompletionReadbackDisposition,
) -> DeviceTimingMeasurement<CompletionReadbackTiming> {
    match disposition {
        CompletionReadbackDisposition::Succeeded(output) => output.timing(),
        CompletionReadbackDisposition::NotAttempted(_)
        | CompletionReadbackDisposition::FailedButQuiescent { .. }
        | CompletionReadbackDisposition::ContractFailedButQuiescent { .. } => {
            DeviceTimingMeasurement::NotRequested
        }
    }
}

fn attempt_completion_readback<R: DeviceRuntime>(
    resources: &CompletionResourceLease<R>,
    lane: &Arc<ExecutionLane<R>>,
    batch_identity: &BatchOperationIdentity,
    completion_disposition: &OperationCompletionDisposition,
    timing_mode: DeviceTimingMode,
    request: CompletionReadbackRequest,
) -> CompletionReadbackDisposition {
    if !matches!(
        completion_disposition,
        OperationCompletionDisposition::Succeeded
    ) {
        return CompletionReadbackDisposition::NotAttempted(request);
    }
    let backing = match resources.backing_view(
        request.node_id(),
        request.participant_index(),
        request.resource_id(),
    ) {
        Ok(backing) => backing,
        Err(error) => {
            return CompletionReadbackDisposition::ContractFailedButQuiescent {
                request,
                failure: QuiescentCompletionContractFailure::new(error.to_string()),
            };
        }
    };
    let readback = catch_unwind(AssertUnwindSafe(|| {
        lane.readback_activation(
            &backing,
            request.logical_offset_bytes(),
            request.output_layout(),
            timing_mode,
        )
    }));
    match readback {
        Ok(Ok(readback)) => match CompletionReadbackOutput::new(request.clone(), readback) {
            Ok(output) => CompletionReadbackDisposition::Succeeded(output),
            Err(error) => CompletionReadbackDisposition::ContractFailedButQuiescent {
                request,
                failure: QuiescentCompletionContractFailure::new(error.to_string()),
            },
        },
        Ok(Err(LaneReadbackError::Contract(error))) => {
            CompletionReadbackDisposition::ContractFailedButQuiescent {
                request,
                failure: QuiescentCompletionContractFailure::new(error.to_string()),
            }
        }
        Ok(Err(LaneReadbackError::Device(error))) => {
            match classify_batch_device_error(lane.runtime(), batch_identity, &error) {
                Ok(failures) => {
                    CompletionReadbackDisposition::FailedButQuiescent { request, failures }
                }
                Err(error) => CompletionReadbackDisposition::ContractFailedButQuiescent {
                    request,
                    failure: QuiescentCompletionContractFailure::new(error.to_string()),
                },
            }
        }
        Err(_) => {
            lane.fail_closed();
            CompletionReadbackDisposition::ContractFailedButQuiescent {
                request,
                failure: QuiescentCompletionContractFailure::new(
                    "device runtime panicked during completion readback",
                ),
            }
        }
    }
}

fn attempt_completion_readbacks<R: DeviceRuntime>(
    resources: &CompletionResourceLease<R>,
    lane: &Arc<ExecutionLane<R>>,
    batch_identity: &BatchOperationIdentity,
    completion_disposition: &OperationCompletionDisposition,
    timing_mode: DeviceTimingMode,
    request: CompletionReadbackBatchRequest,
) -> Vec<CompletionReadbackDisposition> {
    request
        .into_requests()
        .into_iter()
        .map(|request| {
            attempt_completion_readback(
                resources,
                lane,
                batch_identity,
                completion_disposition,
                timing_mode,
                request,
            )
        })
        .collect()
}

fn attempt_completion_readback_collection<R: DeviceRuntime>(
    resources: &CompletionResourceLease<R>,
    lane: &Arc<ExecutionLane<R>>,
    batch_identity: &BatchOperationIdentity,
    completion_disposition: &OperationCompletionDisposition,
    timing_mode: DeviceTimingMode,
    request: CompletionReadbackCollectionRequest,
) -> Vec<CompletionReadbackDisposition> {
    request
        .into_requests()
        .into_iter()
        .map(|request| {
            attempt_completion_readback(
                resources,
                lane,
                batch_identity,
                completion_disposition,
                timing_mode,
                request,
            )
        })
        .collect()
}

#[derive(Debug)]
#[must_use = "nonterminal completion readback observations retain invocation ownership"]
pub enum CompletionReadbackObservation {
    Pending,
    Terminal(CompletionReadbackReceipt),
    Indeterminate(Vec<IdentifiedFailure>),
    SubmissionIndeterminate,
    ObservationPanicked,
    Quarantined(CompletionQuarantineReceipt),
}

#[derive(Debug)]
#[must_use = "nonterminal batch readback observations retain invocation ownership"]
pub enum CompletionReadbackBatchObservation {
    Pending,
    Terminal(CompletionReadbackBatchReceipt),
    Indeterminate(Vec<IdentifiedFailure>),
    SubmissionIndeterminate,
    ObservationPanicked,
    Quarantined(CompletionQuarantineReceipt),
}

#[derive(Debug)]
#[must_use = "nonterminal completion observations retain invocation ownership"]
pub enum CompletionObservation {
    Pending,
    Terminal(OperationCompletionReceipt),
    Indeterminate(Vec<IdentifiedFailure>),
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

    pub fn batch_identity(&self) -> &BatchOperationIdentity {
        self.receipt.batch_identity()
    }

    pub fn slot_id(&self) -> CompletionSlotId {
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

    pub fn wait_with_readback(
        &self,
        request: CompletionReadbackRequest,
    ) -> Result<CompletionReadbackObservation, VNextError> {
        self.reaper
            .upgrade()
            .ok_or_else(|| invalid_completion("completion reaper owner was dropped"))?
            .wait_bound_with_readback(self.slot_id(), request)
    }

    pub fn wait_with_readbacks(
        &self,
        request: CompletionReadbackBatchRequest,
    ) -> Result<CompletionReadbackBatchObservation, VNextError> {
        self.reaper
            .upgrade()
            .ok_or_else(|| invalid_completion("completion reaper owner was dropped"))?
            .wait_bound_with_readbacks(self.slot_id(), request)
    }

    pub fn wait_with_readback_collection(
        &self,
        request: CompletionReadbackCollectionRequest,
    ) -> Result<CompletionReadbackCollectionObservation, VNextError> {
        self.reaper
            .upgrade()
            .ok_or_else(|| invalid_completion("completion reaper owner was dropped"))?
            .wait_bound_with_readback_collection(self.slot_id(), request)
    }
}

pub(crate) struct CompletionReservation<R: DeviceRuntime> {
    reaper: Arc<CompletionReaper<R>>,
    record: SharedCompletionRecord<R>,
    slot_id: CompletionSlotId,
    resources: Option<CompletionResourceLease<R>>,
    lane: Option<Arc<ExecutionLane<R>>>,
    batch_identity: Option<BatchOperationIdentity>,
    receipt: Option<SubmittedOperationReceipt>,
    record_receipt: Option<SubmittedOperationReceipt>,
    submission_may_have_happened: bool,
    finished: bool,
}

impl<R: DeviceRuntime> CompletionReservation<R> {
    pub(crate) fn invocation(&self) -> &InvocationResourceLease<R> {
        match self
            .resources
            .as_ref()
            .expect("live completion reservation owns submission resources")
        {
            CompletionResourceLease::Invocation(invocation) => invocation,
            CompletionResourceLease::Wave(_) => {
                unreachable!("single-operation reservation cannot own a submission wave")
            }
        }
    }

    pub(crate) fn wave(&self) -> &PreparedStepSubmissionWave<R> {
        match self
            .resources
            .as_ref()
            .expect("live completion reservation owns submission resources")
        {
            CompletionResourceLease::Wave(wave) => wave,
            CompletionResourceLease::Invocation(_) => {
                unreachable!("wave reservation cannot own single-operation resources")
            }
        }
    }

    pub(crate) fn backing_view(
        &self,
        node_id: &NodeId,
        participant_index: u32,
        resource_id: &ResourceId,
    ) -> Result<LogicalBackingBufferView<'_, R::Buffer>, VNextError> {
        self.resources
            .as_ref()
            .expect("live completion reservation owns submission resources")
            .backing_view(node_id, participant_index, resource_id)
    }

    pub(crate) fn encode_backing_initializations(
        &self,
        runtime: &R,
        commands: &mut DeviceCommandBatch<R::Command>,
    ) -> Result<usize, BackingInitializationEncodeError<R::Error>> {
        self.resources
            .as_ref()
            .expect("live completion reservation owns submission resources")
            .encode_backing_initializations(runtime, commands)
    }

    pub(crate) fn mark_submission_started(&mut self) {
        self.submission_may_have_happened = true;
    }

    pub(crate) fn definitely_not_submitted(
        mut self,
    ) -> Result<DefinitelyNotSubmittedRetryAuthority<R>, VNextError> {
        self.remove_reserved_slot();
        self.submission_may_have_happened = false;
        let resources = self
            .resources
            .take()
            .expect("reservation owns definitely-not-submitted resources");
        let CompletionResourceLease::Invocation(invocation) = resources else {
            return Err(invalid_completion(
                "single-operation retry requested from a submission wave reservation",
            ));
        };
        let retry = invocation.definitely_not_submitted()?;
        self.finished = true;
        Ok(retry)
    }

    pub(crate) fn definitely_not_submitted_wave(
        mut self,
    ) -> Result<DefinitelyNotSubmittedWaveRetryAuthority<R>, VNextError> {
        self.remove_reserved_slot();
        self.submission_may_have_happened = false;
        let resources = self
            .resources
            .take()
            .expect("reservation owns definitely-not-submitted resources");
        let CompletionResourceLease::Wave(wave) = resources else {
            return Err(invalid_completion(
                "wave retry requested from a single-operation reservation",
            ));
        };
        let retry = wave.definitely_not_submitted()?;
        self.finished = true;
        Ok(retry)
    }

    pub(crate) fn arm(
        mut self,
        fence: R::Fence,
        timing_mode: DeviceTimingMode,
    ) -> Result<CompletionHandle<R>, (VNextError, CompletionHandle<R>)> {
        let resources = self.resources.take().expect("reservation owns resources");
        let lane = self.lane.take().expect("reservation owns lane");
        let batch_identity = self
            .batch_identity
            .take()
            .expect("reservation owns batch identity");
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
            std::mem::forget((resources, lane, fence));
            self.finished = true;
            panic!("completion reservation changed after submission");
        }
        *record = CompletionRecord::InFlight {
            resources,
            lane,
            fence,
            batch_identity,
            receipt: record_receipt,
            timing_mode,
            recovery_state: CompletionRecoveryState::Unobserved,
        };
        let transition = match &mut *record {
            CompletionRecord::InFlight {
                resources, lane, ..
            } => resources
                .mark_submission_fence_installed()
                .map_err(|error| {
                    lane.fail_closed();
                    error
                }),
            _ => unreachable!("completion record was just armed"),
        };
        drop(record);
        self.finished = true;
        let handle = CompletionHandle {
            reaper: Arc::downgrade(&self.reaper),
            receipt,
        };
        match transition {
            Ok(()) => Ok(handle),
            Err(error) => Err((error, handle)),
        }
    }

    pub(crate) fn submission_indeterminate(mut self) -> IndeterminateSubmissionHandle<R> {
        let mut resources = self.resources.take().expect("reservation owns resources");
        resources.mark_submission_indeterminate();
        let lane = self.lane.take().expect("reservation owns lane");
        let batch_identity = self
            .batch_identity
            .take()
            .expect("reservation owns batch identity");
        let mut record = match self.record.lock() {
            Ok(record) => record,
            Err(poisoned) => poisoned.into_inner(),
        };
        if matches!(&*record, CompletionRecord::Reserved) {
            *record = CompletionRecord::SubmissionIndeterminate {
                resources,
                lane,
                batch_identity,
            };
        } else {
            lane.fail_closed();
            std::mem::forget((resources, lane, batch_identity));
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
            let Some(mut resources) = self.resources.take() else {
                return;
            };
            resources.mark_submission_indeterminate();
            let Some(lane) = self.lane.take() else {
                std::mem::forget(resources);
                return;
            };
            let Some(batch_identity) = self.batch_identity.take() else {
                std::mem::forget((resources, lane));
                return;
            };
            lane.fail_closed();
            let mut record = match self.record.lock() {
                Ok(record) => record,
                Err(poisoned) => poisoned.into_inner(),
            };
            if matches!(&*record, CompletionRecord::Reserved) {
                *record = CompletionRecord::SubmissionIndeterminate {
                    resources,
                    lane,
                    batch_identity,
                };
            } else {
                std::mem::forget((resources, lane, batch_identity));
            }
        } else {
            self.remove_reserved_slot();
        }
    }
}

#[cfg(test)]
mod fingerprint_tests {
    use super::*;
    use crate::vnext::ElementType;

    const LARGE_READBACK_BYTES: usize = 512 * 1024;

    fn request() -> CompletionReadbackRequest {
        CompletionReadbackRequest::new(
            NodeId::try_from("node.fingerprint-test".to_owned()).unwrap(),
            0,
            ResourceId::try_from("resource.fingerprint-test".to_owned()).unwrap(),
            0,
            HostTransferLayout::new(ElementType::U8, LARGE_READBACK_BYTES as u64).unwrap(),
        )
        .unwrap()
    }

    fn successful_output(fill: u8) -> CompletionReadbackOutput {
        let bytes = vec![fill; LARGE_READBACK_BYTES];
        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        CompletionReadbackOutput {
            request: request(),
            bytes,
            sha256,
            timing: DeviceTimingMeasurement::NotRequested,
        }
    }

    #[test]
    fn batch_fingerprint_is_bounded_by_digest_evidence_not_output_bytes() {
        let first = vec![CompletionReadbackDisposition::Succeeded(successful_output(
            0x5a,
        ))];
        let encoded = serde_json::to_vec(&CompletionReadbackDispositionFingerprints(&first))
            .expect("fingerprint evidence serializes");
        assert!(encoded.len() < 1024, "encoded {} bytes", encoded.len());
        let first_sha = match &first[0] {
            CompletionReadbackDisposition::Succeeded(output) => output.sha256(),
            _ => unreachable!(),
        };
        assert!(String::from_utf8(encoded).unwrap().contains(first_sha));

        let completion_fingerprint = "a".repeat(64);
        let first_fingerprint =
            completion_readback_batch_fingerprint(&completion_fingerprint, &first);
        assert_eq!(
            first_fingerprint,
            completion_readback_batch_fingerprint(&completion_fingerprint, &first)
        );

        let second = vec![CompletionReadbackDisposition::Succeeded(successful_output(
            0xa5,
        ))];
        assert_ne!(
            first_fingerprint,
            completion_readback_batch_fingerprint(&completion_fingerprint, &second)
        );

        let not_attempted = vec![CompletionReadbackDisposition::NotAttempted(request())];
        assert_ne!(
            first_fingerprint,
            completion_readback_batch_fingerprint(&completion_fingerprint, &not_attempted)
        );
    }
}
