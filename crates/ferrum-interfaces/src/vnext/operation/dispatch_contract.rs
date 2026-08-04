use serde::{Deserialize, Serialize};
use std::{
    fmt,
    time::{Duration, Instant},
};

use super::super::{
    BatchInvocationId, CompletionHandle, DefinitelyNotSubmittedRetryAuthority,
    DefinitelyNotSubmittedWaveRetryAuthority, DeviceCommandPhase, DeviceComputePathRequirement,
    DeviceRuntime, DeviceSubmissionAttribution, DeviceSubmissionExecutionTiming,
    DeviceSubmissionStage, DeviceSubmissionTimingSink, DeviceTimingMeasurement, HostTransferLayout,
    IdentifiedFailure, IndeterminateSubmissionHandle, NodeId, VNextError,
};
use super::foundation::invalid_operation;
use super::{BatchOperationIdentity, OperationFailure};

pub trait DispatchRetryAuthority: fmt::Debug {
    fn prior_attempt(&self) -> BatchInvocationId;
}

impl<R: DeviceRuntime> DispatchRetryAuthority for DefinitelyNotSubmittedRetryAuthority<R> {
    fn prior_attempt(&self) -> BatchInvocationId {
        self.prior_attempt()
    }
}

impl<R: DeviceRuntime> DispatchRetryAuthority for DefinitelyNotSubmittedWaveRetryAuthority<R> {
    fn prior_attempt(&self) -> BatchInvocationId {
        self.prior_attempt()
    }
}

pub enum OperationDispatchError<R, Retry = DefinitelyNotSubmittedRetryAuthority<R>>
where
    R: DeviceRuntime,
    Retry: DispatchRetryAuthority,
{
    Contract(VNextError),
    Provider(OperationFailure),
    Initialization(IdentifiedFailure),
    InputUpload(IdentifiedFailure),
    DefinitelyNotSubmitted {
        failures: Vec<IdentifiedFailure>,
        retry: Retry,
    },
    SubmissionIndeterminate {
        recovery: IndeterminateSubmissionHandle<R>,
    },
    PostSubmitContract {
        error: VNextError,
        completion: CompletionHandle<R>,
    },
}

pub type SubmissionWaveDispatchError<R> =
    OperationDispatchError<R, DefinitelyNotSubmittedWaveRetryAuthority<R>>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BoundDeviceSubmissionAttribution {
    batch_identity: BatchOperationIdentity,
    submission_fingerprint: String,
    device: DeviceSubmissionAttribution,
    terminal_timing: DeviceTimingMeasurement<DeviceSubmissionExecutionTiming>,
}

impl BoundDeviceSubmissionAttribution {
    pub(super) fn new(
        batch_identity: BatchOperationIdentity,
        submission_fingerprint: String,
        device: DeviceSubmissionAttribution,
    ) -> Result<Self, VNextError> {
        for command in device.commands() {
            let Some(node_index) = command.node_index() else {
                continue;
            };
            let node_index_usize = usize::try_from(node_index).map_err(|_| {
                invalid_operation(format!(
                    "device command {} node index exceeds host address space",
                    command.command_index()
                ))
            })?;
            let node_participant_count = batch_identity
                .node_participant_count(node_index_usize)
                .and_then(|count| u32::try_from(count).ok())
                .ok_or_else(|| {
                    invalid_operation(format!(
                        "device command {} references absent node {}",
                        command.command_index(),
                        node_index
                    ))
                })?;
            let participant_range_is_valid = command.participant_start() < node_participant_count
                && command.participant_end() <= node_participant_count;
            let command_requires_full_node =
                command.command_phase() != DeviceCommandPhase::Initialization;
            if !participant_range_is_valid
                || command_requires_full_node
                    && (command.participant_start() != 0
                        || command.participant_count() != node_participant_count)
            {
                return Err(invalid_operation(format!(
                    "device command {} phase {:?} participant range {}..{} differs from node {} participant count {}",
                    command.command_index(),
                    command.command_phase(),
                    command.participant_start(),
                    command.participant_end(),
                    node_index,
                    node_participant_count
                )));
            }
        }
        for replayed_segment in device.replayed_segments() {
            let program_id = replayed_segment.program_id();
            if program_id.plan_hash() != batch_identity.plan_hash()
                || program_id.runtime_implementation_fingerprint()
                    != batch_identity.runtime_implementation_fingerprint()
                || program_id.lane_id() != batch_identity.lane_id()
                || replayed_segment.logical_commands().iter().any(|command| {
                    let Ok(node_index) = usize::try_from(command.node_index()) else {
                        return true;
                    };
                    batch_identity.node_id_at(node_index).is_none()
                        || u32::try_from(
                            batch_identity
                                .node_participant_count(node_index)
                                .unwrap_or_default(),
                        )
                        .map_or(true, |count| count != command.participant_count())
                })
                || replayed_segment.logical_commands().first().is_none_or(|_| {
                    usize::try_from(replayed_segment.physical_command_index())
                        .ok()
                        .and_then(|index| device.commands().get(index))
                        .is_none_or(|physical| {
                            physical.participant_start() != 0
                                || program_id.immediate_sequences() != physical.participant_count()
                                || program_id.immediate_tokens() != physical.token_count()
                        })
                })
            {
                return Err(invalid_operation(
                    "replayed segment attribution differs from its batch or sealed program identity",
                ));
            }
        }
        Ok(Self {
            batch_identity,
            submission_fingerprint,
            device,
            terminal_timing: DeviceTimingMeasurement::NotRequested,
        })
    }

    pub fn bind_terminal_timing(
        mut self,
        terminal_timing: DeviceTimingMeasurement<DeviceSubmissionExecutionTiming>,
    ) -> Result<Self, VNextError> {
        if let DeviceTimingMeasurement::Measured(timing) = &terminal_timing {
            if usize::try_from(timing.command_count()).ok() != Some(self.device.commands().len())
                || self
                    .device
                    .commands()
                    .iter()
                    .enumerate()
                    .any(|(index, command)| {
                        u32::try_from(index).ok() != Some(command.command_index())
                    })
            {
                return Err(invalid_operation(
                    "terminal device timing coverage differs from submission command attribution",
                ));
            }
        }
        self.terminal_timing = terminal_timing;
        Ok(self)
    }

    pub fn batch_identity(&self) -> &BatchOperationIdentity {
        &self.batch_identity
    }

    pub fn submission_fingerprint(&self) -> &str {
        &self.submission_fingerprint
    }

    pub fn device(&self) -> &DeviceSubmissionAttribution {
        &self.device
    }

    pub const fn terminal_timing(
        &self,
    ) -> &DeviceTimingMeasurement<DeviceSubmissionExecutionTiming> {
        &self.terminal_timing
    }
}

#[must_use = "profiled submission evidence and completion must be consumed together"]
pub struct ProfiledSubmissionHandle<R: DeviceRuntime> {
    completion: CompletionHandle<R>,
    attribution: Option<BoundDeviceSubmissionAttribution>,
}

impl<R: DeviceRuntime> ProfiledSubmissionHandle<R> {
    pub(super) fn new(
        completion: CompletionHandle<R>,
        attribution: Option<BoundDeviceSubmissionAttribution>,
    ) -> Self {
        Self {
            completion,
            attribution,
        }
    }

    pub fn into_parts(
        self,
    ) -> (
        CompletionHandle<R>,
        Option<BoundDeviceSubmissionAttribution>,
    ) {
        (self.completion, self.attribution)
    }
}

/// Typed host boundaries inside one prepared wave dispatch. These intervals
/// are host wall time and must not be combined with backend device timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmissionWaveDispatchStage {
    ContractValidateAndReserve,
    BackingAndInputEncode,
    ProviderNodeEncode,
    LaneReserve,
    DeviceRuntimeSubmit,
    CompletionArm,
    LaneReserveSubmitAndArm,
}

/// Diagnostic-only timing sink for the prepared-wave dispatch hot path.
///
/// The sink receives only a stage and completed host duration; it receives no
/// command, resource, or correctness authority. `ENABLED = false` is the
/// compile-time off path: no clock is read and `record` is never called.
/// Enabled implementations run on the submission thread and must not block,
/// allocate, or panic.
pub trait SubmissionWaveDispatchTimingSink: DeviceSubmissionTimingSink {
    fn record(&self, stage: SubmissionWaveDispatchStage, elapsed: Duration);
}

pub(super) struct DisabledSubmissionWaveDispatchTimingSink;

impl DeviceSubmissionTimingSink for DisabledSubmissionWaveDispatchTimingSink {
    const ENABLED: bool = false;

    fn record_device_submission(&self, _stage: DeviceSubmissionStage, _elapsed: Duration) {
        unreachable!("disabled device submission timing cannot record")
    }
}

impl SubmissionWaveDispatchTimingSink for DisabledSubmissionWaveDispatchTimingSink {
    fn record(&self, _stage: SubmissionWaveDispatchStage, _elapsed: Duration) {
        unreachable!("disabled submission timing cannot record")
    }
}

pub(super) struct SubmissionWaveDispatchStageTimer<'sink, S>
where
    S: SubmissionWaveDispatchTimingSink,
{
    sink: &'sink S,
    stage: SubmissionWaveDispatchStage,
    started: Option<Instant>,
}

impl<'sink, S> SubmissionWaveDispatchStageTimer<'sink, S>
where
    S: SubmissionWaveDispatchTimingSink,
{
    #[inline(always)]
    pub(super) fn start(sink: &'sink S, stage: SubmissionWaveDispatchStage) -> Self {
        Self {
            sink,
            stage,
            started: S::ENABLED.then(Instant::now),
        }
    }
}

impl<S> Drop for SubmissionWaveDispatchStageTimer<'_, S>
where
    S: SubmissionWaveDispatchTimingSink,
{
    fn drop(&mut self) {
        if let Some(started) = self.started.take() {
            if !std::thread::panicking() {
                self.sink.record(self.stage, started.elapsed());
            }
        }
    }
}

#[cfg(test)]
mod submission_wave_dispatch_timing_tests {
    use std::time::Duration;

    use super::{
        DeviceSubmissionStage, DeviceSubmissionTimingSink, SubmissionWaveDispatchStage,
        SubmissionWaveDispatchStageTimer, SubmissionWaveDispatchTimingSink,
    };

    struct DisabledPanicSink;

    impl DeviceSubmissionTimingSink for DisabledPanicSink {
        const ENABLED: bool = false;

        fn record_device_submission(&self, _stage: DeviceSubmissionStage, _elapsed: Duration) {
            panic!("disabled device timing sink was called");
        }
    }

    impl SubmissionWaveDispatchTimingSink for DisabledPanicSink {
        fn record(&self, _stage: SubmissionWaveDispatchStage, _elapsed: Duration) {
            panic!("disabled timing sink was called");
        }
    }

    #[test]
    fn disabled_submission_timing_does_not_record() {
        let timer = SubmissionWaveDispatchStageTimer::start(
            &DisabledPanicSink,
            SubmissionWaveDispatchStage::ProviderNodeEncode,
        );
        drop(timer);

        assert!(!DisabledPanicSink::ENABLED);
    }
}

impl<R, Retry> fmt::Debug for OperationDispatchError<R, Retry>
where
    R: DeviceRuntime,
    Retry: DispatchRetryAuthority,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(error) => formatter.debug_tuple("Contract").field(error).finish(),
            Self::Provider(error) => formatter.debug_tuple("Provider").field(error).finish(),
            Self::Initialization(error) => formatter
                .debug_tuple("Initialization")
                .field(error)
                .finish(),
            Self::InputUpload(error) => formatter.debug_tuple("InputUpload").field(error).finish(),
            Self::DefinitelyNotSubmitted { failures, retry } => formatter
                .debug_struct("DefinitelyNotSubmitted")
                .field("failures", failures)
                .field("retry", retry)
                .finish(),
            Self::SubmissionIndeterminate { recovery } => formatter
                .debug_struct("SubmissionIndeterminate")
                .field("recovery", recovery)
                .finish(),
            Self::PostSubmitContract { error, completion } => formatter
                .debug_struct("PostSubmitContract")
                .field("error", error)
                .field("completion", completion)
                .finish(),
        }
    }
}

impl<R, Retry> fmt::Display for OperationDispatchError<R, Retry>
where
    R: DeviceRuntime,
    Retry: DispatchRetryAuthority,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(error) => {
                write!(formatter, "operation dispatch contract failed: {error}")
            }
            Self::Provider(error) => write!(
                formatter,
                "operation provider failed with {}: {}",
                error.code(),
                error.message()
            ),
            Self::Initialization(error) => write!(
                formatter,
                "operation backing initialization failed with {}: {}",
                error.failure().code(),
                error.failure().message()
            ),
            Self::InputUpload(error) => write!(
                formatter,
                "operation input upload failed with {}: {}",
                error.failure().code(),
                error.failure().message()
            ),
            Self::DefinitelyNotSubmitted { failures, retry } => write!(
                formatter,
                "operation attempt {} with {} participants was definitely not submitted: {}",
                retry.prior_attempt(),
                failures.len(),
                failures
                    .first()
                    .map(|failure| failure.failure().message())
                    .unwrap_or("missing classified participant failure")
            ),
            Self::SubmissionIndeterminate { recovery } => write!(
                formatter,
                "operation submission may have reached the device; completion slot {} retains ownership",
                recovery.slot_id().get()
            ),
            Self::PostSubmitContract { error, completion } => write!(
                formatter,
                "operation submission reached the device but slot {} observed a contract failure: {error}",
                completion.slot_id().get()
            ),
        }
    }
}

/// Scratch bytes presented to every provider invocation in one submission.
/// `ProviderContract` preserves the selected provider's declared reuse policy;
/// explicit fill patterns are diagnostic proof inputs and are encoded before
/// compute outside reusable executable capture.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubmissionScratchInitialization {
    #[default]
    ProviderContract,
    FillByte(u8),
}

/// Core-owned execution controls independent from timing instrumentation.
///
/// Determinism gates use the strict constructors. Product requests use
/// `adaptive`, allowing the runtime to select a compatible eager or resident
/// path without changing provider semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubmissionExecutionPolicy {
    compute_path: DeviceComputePathRequirement,
    scratch_initialization: SubmissionScratchInitialization,
}

impl SubmissionExecutionPolicy {
    pub const fn adaptive() -> Self {
        Self {
            compute_path: DeviceComputePathRequirement::Adaptive,
            scratch_initialization: SubmissionScratchInitialization::ProviderContract,
        }
    }

    pub const fn determinism_eager(scratch_fill: u8) -> Self {
        Self {
            compute_path: DeviceComputePathRequirement::EagerOnly,
            scratch_initialization: SubmissionScratchInitialization::FillByte(scratch_fill),
        }
    }

    pub const fn determinism_replayed(scratch_fill: u8) -> Self {
        Self {
            compute_path: DeviceComputePathRequirement::ReplayedOnly,
            scratch_initialization: SubmissionScratchInitialization::FillByte(scratch_fill),
        }
    }

    pub const fn compute_path(self) -> DeviceComputePathRequirement {
        self.compute_path
    }

    pub const fn scratch_initialization(self) -> SubmissionScratchInitialization {
        self.scratch_initialization
    }
}

impl Default for SubmissionExecutionPolicy {
    fn default() -> Self {
        Self::adaptive()
    }
}

/// One typed host input written into an exact participant's resolved plan
/// input before any provider command executes. The request names semantic
/// plan coordinates rather than exposing backend buffers or allocation ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmissionWaveInputUpload {
    node_id: NodeId,
    participant_index: u32,
    input_ordinal: u32,
    logical_offset_bytes: u64,
    source_layout: HostTransferLayout,
    bytes: Vec<u8>,
}

impl SubmissionWaveInputUpload {
    pub fn new(
        node_id: NodeId,
        participant_index: u32,
        input_ordinal: u32,
        logical_offset_bytes: u64,
        source_layout: HostTransferLayout,
        bytes: Vec<u8>,
    ) -> Result<Self, VNextError> {
        source_layout.validate_bytes(bytes.len())?;
        let byte_len = source_layout.byte_len()?;
        if logical_offset_bytes.checked_add(byte_len).is_none()
            || logical_offset_bytes % source_layout.element_type().size_bytes() != 0
        {
            return Err(invalid_operation(
                "submission input upload has an invalid aligned logical range",
            ));
        }
        Ok(Self {
            node_id,
            participant_index,
            input_ordinal,
            logical_offset_bytes,
            source_layout,
            bytes,
        })
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    pub const fn participant_index(&self) -> u32 {
        self.participant_index
    }

    pub const fn input_ordinal(&self) -> u32 {
        self.input_ordinal
    }

    pub const fn logical_offset_bytes(&self) -> u64 {
        self.logical_offset_bytes
    }

    pub const fn source_layout(&self) -> HostTransferLayout {
        self.source_layout
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl<R, Retry> std::error::Error for OperationDispatchError<R, Retry>
where
    R: DeviceRuntime,
    Retry: DispatchRetryAuthority,
{
}
